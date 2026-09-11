#!/usr/bin/env python3
"""GPU-side pass: score the structural/semantic dimension S with Music Flamingo.

S is the relational ALLM term -- Music Flamingo's P(Yes) that B coheres with A --
computed once per pair here (in the `mfenv` conda env, where MF lives) and cached
to JSON.  The CPU-side scoring harness then serves those cached values through the
common `Dimension` interface via `StructuralDimension`, so MF never has to load in
the same process as the H/T/R DSP code.

Run on the GPU box, in the MF env::

    conda activate mfenv
    export HF_HOME=/root/autodl-tmp/hf HF_ENDPOINT=https://hf-mirror.com
    python score_s_mf.py --manifest perturbations/pairs_manifest.json \
        --audio-dir perturbations/audio --out results/s_cache/s_scores.json \
        --scorer coherence --secs 8 --gap-s 1.0

``--gap-s`` defaults to 0.0 but the shipped cache was built at 1.0, and the gap is
load-bearing (it roughly doubles the coherent/incoherent separation, Sec. 4.9).
Omitting it silently produces a DIFFERENT protocol, so the flag is spelled out
above even though ``--scorer`` and ``--secs`` already match their defaults.

Writes ``{"meta": {...}, "scores": {pair_id: S_float},
          "per_template": {pair_id: [t1, t2, t3, t4]}}``.

``scores`` (the mean over the template bank) keeps its exact original key and
format -- every downstream reader (StructuralDimension, score_separability,
compute_S_value) is unaffected.  ``per_template`` is additive (T7.1): caching
only the mean threw away which template carries which signal, and recovering it
cost a whole GPU pass.  Now one pass answers it forever.

Resumable: re-running skips pair_ids already present in --out.  Use --limit for
a quick subset first, or --subset results/s_cache/s_method_subset.json to score exactly
the frozen T7.0 regression subset (the required check for any method variant).

Method-variant flags (T7.0: every variant writes a NEW --out cache, never
overwrites results/s_cache/s_scores.json, and must keep style_swap the #1 drop and
time_stretch ~0 before it may be adopted):
  --precision {all_bf16,encoder_fp32}   T7.6 precision A/B
  --token-agg {variants,single}         T7.3 Yes/No token-variant aggregation
  --preamble {default,gap_aware}        T7.2 gap-aware framing A/B
  --scorer balanced                     T7.1(3) polarity-balanced scorer
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable


import argparse
import numpy as np
import json
import logging
import sys
from pathlib import Path


def resolve(path_str: str, audio_dir: Path | None) -> str:
    p = Path(path_str)
    return str(audio_dir / p.name) if audio_dir is not None else str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, default=None,
                    help="resolve pair audio by basename here (survives a machine change)")
    ap.add_argument("--out", type=Path, default=Path("results/s_cache/s_scores.json"))
    ap.add_argument("--scorer", choices=("coherence", "drift", "balanced", "null"),
                    default="coherence",
                    help="coherence = mf_probe P(Yes) template bank; "
                         "null = the same read-out on NULL_TEMPLATES, the A3 control; "
                         "balanced = polarity-balanced (pos + (1-neg))/2, a yes-bias "
                         "correction (T7.1(3)); "
                         "drift = drift_v4 0-1 numeric rating (multi-sample averaged)")
    ap.add_argument("--n-samples", type=int, default=3,
                    help="drift scorer only: generations averaged per pair to cut token noise")
    ap.add_argument("--hf-home", type=str, default=None,
                    help="HF_HOME for weights/cache (default: keep env, else /root/autodl-tmp/hf). "
                         "MUST point at the big disk, not the small system disk.")
    ap.add_argument("--hf-endpoint", type=str, default=None,
                    help="HF_ENDPOINT mirror (default: keep env, else https://hf-mirror.com)")
    ap.add_argument("--secs", type=float, default=8.0,
                    help="coherence scorer: seconds of A and of B to score (mf_probe default is 8)")
    ap.add_argument("--gap-s", type=float, default=0.0, help="silence between A and B in the concat")
    ap.add_argument("--a-from", choices=("head", "tail"), default="head",
                    help="which --secs of A to score. head is the validated default and "
                         "is identical to tail whenever A is already ~--secs long (the "
                         "matrix, drift, rerank). Use tail for CONTINUATION pairs, where "
                         "A is a long context and B starts at its end -- head would leave "
                         "a hole between the two segments and score the wrong junction")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N unscored pairs (0 = all)")
    ap.add_argument("--subset", type=Path, default=None,
                    help="JSON with a 'pair_ids' list (e.g. results/s_cache/s_method_subset.json): "
                         "score ONLY those pairs -- the T7.0 regression check")
    ap.add_argument("--precision", choices=("all_bf16", "encoder_fp32"), default="all_bf16",
                    help="T7.6: all_bf16 forces the audio encoder to bf16 too (default, "
                         "believed to match the shipped caches); encoder_fp32 keeps "
                         "from_pretrained's fp32 modules fp32 and does not cast inputs")
    ap.add_argument("--token-agg", choices=("variants", "single"), default="single",
                    help="T7.3: variants = logsumexp over Yes/No casing variants (default); "
                         "single = the original single ' Yes'/' No' token pair")
    ap.add_argument("--preamble", choices=("default", "gap_aware"), default="default",
                    help="T7.2: gap_aware tells the model the segments are separated by a "
                         "brief silence (framing only; the template bank is unchanged)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    # HF env vars must be set BEFORE importing mf_probe (it reads HF_ENDPOINT at
    # import, and HF_HOME must be right before weights download -- a wrong HF_HOME
    # fills the small system disk).  CLI > existing env > safe defaults.
    import os
    os.environ["HF_HOME"] = args.hf_home or os.environ.get("HF_HOME", "/root/autodl-tmp/hf")
    os.environ["HF_ENDPOINT"] = args.hf_endpoint or os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    logging.info("HF_HOME=%s  HF_ENDPOINT=%s", os.environ["HF_HOME"], os.environ["HF_ENDPOINT"])

    doc = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs, meta = doc["pairs"], doc.get("meta", {})

    if args.subset is not None:
        want = set(json.loads(args.subset.read_text(encoding="utf-8"))["pair_ids"])
        pairs = [p for p in pairs if p["pair_id"] in want]
        missing = want - {p["pair_id"] for p in pairs}
        if missing:
            logging.warning("%d subset pair_ids absent from this manifest (e.g. %s)",
                            len(missing), sorted(missing)[:2])
        logging.info("subset %s: %d of %d pairs selected", args.subset, len(pairs), len(doc["pairs"]))

    # resume from any existing cache
    cache: dict[str, float] = {}
    per_template: dict[str, list] = {}
    if args.out.is_file():
        prev = json.loads(args.out.read_text(encoding="utf-8"))
        cache = {k: float(v) for k, v in prev.get("scores", {}).items()}
        per_template = {k: list(v) for k, v in prev.get("per_template", {}).items()}
        logging.info("resuming: %d pairs already scored", len(cache))

    todo = [p for p in pairs if p["pair_id"] not in cache]
    if args.limit > 0:
        todo = todo[:args.limit]
    logging.info("to score: %d pairs", len(todo))

    import mf_probe as mf          # imported here so --help works without the MF env
    mf.TOKEN_AGG = args.token_agg
    mf.PREAMBLE = args.preamble
    mf.load(precision=args.precision)

    # choose the per-pair scoring function.  Each returns (S, per_template|None);
    # only the template-bank scorers have per-template components.
    if args.scorer == "coherence":
        # mf.coherence_score internally calls concat_clips(secs=8); to make the
        # region explicit and controllable, replicate it with args.secs so the
        # scored window is a stated parameter, not a hidden default.
        def score_pair(a: str, b: str):
            ab = mf.concat_clips(a, b, secs=args.secs, gap_s=args.gap_s,
                                 a_from=args.a_from)
            try:
                per = [float(mf._yes_prob(ab, q)) for q in mf.TEMPLATES]
            finally:
                mf._cleanup(ab)
            return float(np.mean(per)), per
    elif args.scorer == "null":
        # A3 control bank: same concatenation, same read-out, questions with no
        # relational content. Flat across families = the coherence bank's separation
        # is attributable to what it asks; not flat = it is not.
        def score_pair(a: str, b: str):
            ab = mf.concat_clips(a, b, secs=args.secs, gap_s=args.gap_s,
                                 a_from=args.a_from)
            try:
                per = [float(mf._yes_prob(ab, q)) for q in mf.NULL_TEMPLATES]
            finally:
                mf._cleanup(ab)
            return float(np.mean(per)), per
    elif args.scorer == "balanced":
        # polarity-balanced: agree-it's-coherent AND disagree-it's-different.
        # per-template = POS_TEMPLATES then NEG_TEMPLATES (raw P(Yes) of each,
        # NOT sign-flipped -- the flip lives in the aggregate).
        def score_pair(a: str, b: str):
            ab = mf.concat_clips(a, b, secs=args.secs, gap_s=args.gap_s)
            try:
                pos = [float(mf._yes_prob(ab, q)) for q in mf.POS_TEMPLATES]
                neg = [float(mf._yes_prob(ab, q)) for q in mf.NEG_TEMPLATES]
            finally:
                mf._cleanup(ab)
            s = (float(np.mean(pos)) + (1.0 - float(np.mean(neg)))) / 2.0
            return s, pos + neg
    else:
        # drift_v4's 0-1 numeric rating, multi-sample averaged. It scores a single
        # concatenated A|B clip, so we reuse its concat + rating and share the same
        # loaded MF model by pointing drift_v4's backend at mf_probe.
        import drift_v4 as d4
        if getattr(d4, "_mf", None) is None:
            d4.set_backend(mf)              # share the already-loaded model/processor
        d4.N_SAMPLES = args.n_samples
        gap = args.gap_s if args.gap_s > 0 else d4.GAP_S

        def score_pair(a: str, b: str):
            ab = d4._concat(a, b, gap_s=gap)
            vals = [v for v in (d4._rate_once(ab) for _ in range(d4.N_SAMPLES))
                    if not np.isnan(v)]
            return (float(np.mean(vals)) if vals else float("nan")), None

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # which templates the per_template columns correspond to, in order
    if args.scorer == "balanced":
        template_order = list(mf.POS_TEMPLATES) + list(mf.NEG_TEMPLATES)
    elif args.scorer == "coherence":
        template_order = list(mf.TEMPLATES)
    elif args.scorer == "null":
        template_order = list(mf.NULL_TEMPLATES)
    else:
        template_order = None

    def flush() -> None:
        args.out.write_text(json.dumps(
            {"meta": {**meta, "s_backend": "music_flamingo", "scorer": args.scorer,
                      "secs": args.secs, "gap_s": args.gap_s,
                      "n_samples": args.n_samples if args.scorer == "drift" else 1,
                      "model_id": getattr(mf, "MODEL_ID", "?"),
                      "templates": template_order or getattr(mf, "TEMPLATES", None),
                      # method-variant + precision provenance (T7.6): the original
                      # s_scores.json recorded none of this and had to be
                      # reverse-engineered. Every cache now self-documents.
                      "per_template_order": template_order,
                      "token_agg": args.token_agg,
                      "preamble": args.preamble,
                      "preamble_text": mf.PREAMBLES.get(args.preamble),
                      "peak_safe_concat": True,
                      "precision_requested": args.precision,
                      "precision_actual": mf.describe_precision(),
                      "subset": str(args.subset) if args.subset else None},
             "scores": cache,
             "per_template": per_template}, indent=2), encoding="utf-8")

    for i, pair in enumerate(todo, 1):
        a = resolve(pair["ref_path"], args.audio_dir)
        b = resolve(pair["cand_path"], args.audio_dir)
        try:
            s, per = score_pair(a, b)
        except Exception as exc:  # noqa: BLE001 - never lose a long run to one bad pair
            logging.error("pair %s failed: %s", pair["pair_id"], exc)
            continue
        cache[pair["pair_id"]] = s
        if per is not None:
            per_template[pair["pair_id"]] = per
        if i % 25 == 0 or i == len(todo):
            flush()
            logging.info("scored %d/%d (last S=%.3f)", i, len(todo), cache[pair["pair_id"]])

    flush()
    logging.info("done. wrote %s (%d scores)", args.out, len(cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
