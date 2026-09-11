#!/usr/bin/env python3
"""A4 -- do MASKED-PREDICTION music encoders close the gap the dissociation rests on?

Every baseline in the paper is CLAP, COCOLA or an SCS stand-in, and all three are
contrastive with a text anchor. Our own Related Work argues CLAP's audio-audio cosine is
"induced rather than optimised", which invites the obvious question: what does a music
encoder trained WITHOUT a text objective do on the same tests? If MuQ or MERT matches the
H/T/R/S profile on family diagnosis, the interpretability claim narrows further; if it
does not, the dissociation is considerably stronger. Either way it is a result, and the
external review of 2026-08-24 calls it the gap every reviewer will raise independently.

TWO ENCODERS, chosen because the box already has them:
  * MuQ-large-msd-iter -- Mel residual VQ + self-supervised iterative refinement, music
    only. Already downloaded and run for SongEval inside `score_rivals.py`.
  * MERT-v1-330M -- masked acoustic token prediction with a musical teacher. Already
    downloaded and run for TuneJury inside the same script.
Neither has a text tower, which is the entire point of the comparison.

WHY A SEPARATE SCRIPT rather than a `score_rivals.py` backend. That script's `Backend`
contract is `score(wav, sr) -> dict[str, float]`: it emits scalars for an aesthetics or
preference head. We need the representation itself, so this follows the `clap_probe.py`
route instead and writes an .npz keyed the identical way -- `clips`, `pooled`, `meta` --
so `review_fixes.build_feature_sets()` consumes it with the CLAP loader unchanged.

TWO ENVIRONMENTS, communicating through one .npz on disk (see env/).

  extract   the rivals venv from score_rivals.py's docstring (torch >= 2, muq,
            transformers). GPU box: ~5.7k forward passes over two models is minutes on
            the 4090 and hours on CPU.
  probe     the DSP env. Reads the .npz, adds muq_diff / mert_diff feature sets and the
            cosines, re-runs the leave-one-track-out diagnosis.

    ~/rivals/venv/bin/python ssl_probe.py extract --encoder muq
    ~/rivals/venv/bin/python ssl_probe.py extract --encoder mert
    python scripts/analysis/ssl_probe.py cosines        # DSP env, Table 3 rows
    python scripts/analysis/diagnose_nested_cv.py       # DSP env, Table 1 rows

PATHS ON THE GPU BOX. Scripts sit flat at /root and the corpus sits on the big volume, so
the root-marker walk finds neither and falls back to the cwd. `--audio-root` is the
same escape hatch `score_rivals.py` uses: it prefixes the manifest's repo-relative
ref_path/cand_path, and `--manifest` and `--out` are used as given when they resolve on
their own. On AutoDL that means

    ~/rivals/venv/bin/python ssl_probe.py extract --encoder muq \
        --manifest /root/autodl-tmp/perturbations/pairs_manifest.json \
        --audio-root /root/autodl-tmp \
        --out /root/autodl-tmp/muq_embeddings.npz

POOLING IS THE ONE JUDGEMENT CALL. Both encoders emit a frame sequence; the comparison
is only fair if the pooled vector is built the way CLAP's is, so we mean-pool over time
and unit-normalise, exactly as `clap_probe.py` does with `pooler_output`. `--layer`
selects the hidden layer. Layer 6 is the default for both: it is what SongEval reads from
MuQ, and for MERT it was carried over rather than derived, which is the weak point of the
comparison. Layer and pooling are the two choices the meta-evaluation literature reports
as dominant for embedding-based audio metrics, so a claim that these encoders underperform
is only safe once the layer has been swept and the baseline given its best one. Use
`sweep-layers` for that; it favours the baseline by construction, so the profile beating
the best layer is the stronger form of the result.

WINDOW. `--secs 8` and the A-then-B concatenation are NOT used. These are per-clip
embeddings of A and of B separately, matching how the CLAP embedding difference in
Table 1 is built, so the two rows differ only in the encoder.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
try:                                  # sibling imports across groups (see scripts/README.md)
    sys.path.insert(0, str(_here.parent.parent))
    import _paths  # noqa: F401
except ImportError:
    pass

SECS = 8.0                            # same window as clap_probe.py
ENCODERS = {
    "muq":  {"ckpt": "OpenMuQ/MuQ-large-msd-iter", "sr": 24000, "layer": 6},
    "mert": {"ckpt": "m-a-p/MERT-v1-330M",         "sr": 24000, "layer": 6},
}


def repo_root() -> Path:
    for p in [_here, *_here.parents]:
        if (p / ".projectroot").exists() or (p / ".git").exists():
            return p
    return Path.cwd()


ROOT = repo_root()


def resolve(path_str, base: Path | None = None) -> Path:
    """A path used as given when it already resolves, else taken as repo-relative.

    Mirrors `score_rivals.resolve`. Needed because the GPU box keeps scripts flat at
    /root and the corpus on /root/autodl-tmp, so `repo_root()` cannot find either.
    """
    p = Path(path_str).expanduser()
    if p.is_absolute() or p.exists():
        return p
    return (base or ROOT) / p


def load_wav(path: Path, target_sr: int, secs: float = SECS) -> np.ndarray:
    """First `secs` of a wav as mono float32, resampled to the encoder's rate.

    scipy + librosa only for the resample, mirroring clap_probe.load_wav's refusal to
    drag librosa into the torch env for anything else.
    """
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    else:
        y = data.astype(np.float32)
    y = y[: int(secs * sr)]
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y.astype(np.float32)


def clip_list(manifest: Path, limit: int) -> list[str]:
    doc = json.load(open(manifest))
    pairs = doc["pairs"] if isinstance(doc, dict) else doc
    clips, seen = [], set()
    for p in pairs:
        for k in ("ref_path", "cand_path"):
            if p[k] not in seen:
                seen.add(p[k])
                clips.append(p[k])
    if limit:
        keep = set()
        for p in pairs[:limit]:
            keep.update((p["ref_path"], p["cand_path"]))
        clips = [c for c in clips if c in keep]
    logging.info("%d pairs -> %d unique clips", len(pairs), len(clips))
    return clips


# --------------------------------------------------------------- extract (GPU venv)
def cmd_extract(args) -> int:
    import torch

    spec = ENCODERS[args.encoder]
    layer = args.layer if args.layer is not None else spec["layer"]
    sr = spec["sr"]
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.encoder == "muq":
        from muq import MuQ
        model = MuQ.from_pretrained(spec["ckpt"]).to(dev).eval()

        def embed(y):
            x = torch.tensor(y).unsqueeze(0).to(dev)
            with torch.no_grad():
                h = model(x, output_hidden_states=True)["hidden_states"][layer]
            return h.squeeze(0).float().cpu().numpy()
    else:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(spec["ckpt"], trust_remote_code=True)
        model = model.to(dev).eval()

        def embed(y):
            x = torch.tensor(y).unsqueeze(0).to(dev)
            with torch.no_grad():
                h = model(x, output_hidden_states=True).hidden_states[layer]
            return h.squeeze(0).float().cpu().numpy()

    logging.info("%s layer %d on %s", spec["ckpt"], layer, dev)
    manifest = resolve(args.manifest)
    audio_root = Path(args.audio_root).expanduser() if args.audio_root else ROOT
    if not manifest.is_file():
        raise SystemExit(f"--manifest not found: {manifest}")
    logging.info("manifest %s, audio root %s", manifest, audio_root)
    clips = clip_list(manifest, args.limit)

    pooled: dict[str, np.ndarray] = {}
    t0 = time.time()
    for i, rel in enumerate(clips, 1):
        wav = resolve(rel, audio_root)
        if args.audio_dir:                       # flat audio dir: resolve by basename
            wav = Path(args.audio_dir).expanduser() / Path(rel).name
        if i == 1 and not wav.is_file():
            raise SystemExit(f"first clip not found: {wav}\n"
                             f"  manifest paths are repo-relative; pass --audio-root "
                             f"(the dir containing 'perturbations/') or --audio-dir")
        frames = embed(load_wav(wav, sr))
        e = frames.mean(axis=0)                       # time-pool, as CLAP's pooler does
        pooled[rel] = (e / (np.linalg.norm(e) + 1e-8)).astype(np.float32)
        if i % 100 == 0 or i == len(clips):
            el = time.time() - t0
            logging.info("%d/%d clips  %.1fs  eta %.1f min",
                         i, len(clips), el, (el / i) * (len(clips) - i) / 60)

    dest = resolve(args.out) if args.out else \
        ROOT / f"results/baselines/{args.encoder}_embeddings.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        clips=np.array(clips),
        pooled=np.stack([pooled[c] for c in clips]),
        meta=np.array([json.dumps({
            "encoder": args.encoder, "ckpt": spec["ckpt"], "layer": layer,
            "sr": sr, "secs": SECS, "n_clips": len(clips),
            "pooled_dim": int(len(next(iter(pooled.values())))),
            "readout_pooled": f"hidden_states[{layer}] mean over time, unit-norm",
        })]),
    )
    print(f"[wrote {dest}]  {len(clips)} clips")
    return 0


# ------------------------------------------------------------------ cosines (DSP env)
def cmd_cosines(args) -> int:
    """Table 3's row: the encoder's own A-vs-B cosine, on the same pairs as CLAP's.

    Rescaled (1+cos)/2 into [0,1] exactly as score_baselines.py does for clap_htsat, so
    the column is read the same way and an AUC over it is directly comparable.
    """
    doc = json.load(open(resolve(args.manifest)))
    pairs = doc["pairs"] if isinstance(doc, dict) else doc

    out: dict[str, dict[str, float]] = {}
    meta = {}
    for enc in args.encoders.split(","):
        enc = enc.strip()
        path = resolve(args.emb_dir) / f"{enc}_embeddings.npz"
        if not path.is_file():
            print(f"  skip {enc}: {path} not found")
            continue
        npz = np.load(path, allow_pickle=True)
        # basename keys: the extract may have run against a manifest copy whose paths
        # are relative to a different root (see review_fixes.ssl_features)
        idx = {Path(c).name: i for i, c in enumerate(npz["clips"])}
        pooled = npz["pooled"]
        meta[enc] = json.loads(str(npz["meta"][0]))
        n = 0
        for p in pairs:
            a, b = Path(p["ref_path"]).name, Path(p["cand_path"]).name
            if a in idx and b in idx:
                cos = float(np.dot(pooled[idx[a]], pooled[idx[b]]))
                out.setdefault(p["pair_id"], {})[enc] = (1.0 + cos) / 2.0
                n += 1
        print(f"  {enc}: {n} pairs, dim {meta[enc]['pooled_dim']}, layer {meta[enc]['layer']}")

    if not out:
        print("nothing to write -- run `extract` first")
        return 1
    dest = resolve(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"meta": {"encoders": meta, "readout": "(1 + cos)/2, matching score_baselines "
                                               "clap_htsat so AUCs are comparable"},
         "scores": out}, indent=1), encoding="utf-8")
    print(f"[wrote {dest}]  {len(out)} pairs")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="encode every clip (rivals venv, GPU)")
    e.add_argument("--encoder", choices=sorted(ENCODERS), required=True)
    e.add_argument("--manifest", default="perturbations/pairs_manifest.json")
    e.add_argument("--out", default=None,
                   help="default: results/baselines/<encoder>_embeddings.npz")
    e.add_argument("--layer", type=int, default=None,
                   help="hidden layer to pool; default is the encoder's documented one")
    e.add_argument("--audio-root", default=None,
                   help="dir the manifest's repo-relative ref_path/cand_path hang off; "
                        "default is the repo root. On the GPU box: /root/autodl-tmp")
    e.add_argument("--audio-dir", default=None,
                   help="resolve clip audio by BASENAME in this dir instead, matching "
                        "score_s_mf.py's flag")
    e.add_argument("--device", default=None)
    e.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke test)")
    e.set_defaults(fn=cmd_extract)

    c = sub.add_parser("cosines", help="A-vs-B cosine per pair (DSP env)")
    c.add_argument("--encoders", default="muq,mert")
    c.add_argument("--emb-dir", default="results/baselines",
                   help="where the <encoder>_embeddings.npz files are")
    c.add_argument("--manifest", default="perturbations/pairs_manifest.json")
    c.add_argument("--out", default="results/baselines/ssl_scores.json")
    c.set_defaults(fn=cmd_cosines)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
