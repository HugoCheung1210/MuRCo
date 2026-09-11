#!/usr/bin/env python3
"""Per-pair BASELINE coherence scores (the Ch-3 baseline table + §5 head-to-head).

Motivation (PROJECT_STATE §8.1): is the *whole field* defeated by same-genre
substitution, or only our S?  We score every matrix pair with off-the-shelf audio
similarity metrics and feed them to compute_S_value.py as extra features, so their
identity-discrimination AUC (and the cross- vs same-genre split) sits next to S in
one table.  Prediction: CLAP tracks S on cross-genre swaps but collapses toward
chance on same-genre swaps -- i.e. the failure is field-wide, which is itself a
contribution rather than a weakness of S.

Metrics (all return [0,1], higher = MORE coherent / same-identity, matching the
dimension contract so compute_S_value treats them like a score column):
  * clap_htsat  -- cosine similarity of LAION-CLAP (htsat-unfused) audio embeddings
                   of A and B, remapped (1+cos)/2.  General-audio CLAP.
  * clap_music  -- same, with the music-specialised CLAP checkpoint.
  * scs         -- STRUCTURAL-coherence STAND-IN.  The official MusicWeaver SCS is
                   not public; this is a documented signal-level proxy: cosine
                   agreement between the self-similarity matrices (SSM) of A and B,
                   built from beat-agnostic chroma+MFCC frames resampled to a common
                   length.  Captures "same internal repetition/texture structure",
                   NOT semantic identity.  Report AS a stand-in, with this caveat.
  * fad_proxy   -- (optional, --with-fad) per-pair FAD is ill-defined (FAD is a
                   SET-level Frechet distance between embedding distributions); this
                   is an embedding-distance proxy 1/(1+||e_A - e_B||) on the htsat
                   embeddings.  Also report set-level FAD separately as structurally
                   inapplicable to a per-pair contract.

Output: results/baselines/baseline_scores.json ::

    {"meta": {...}, "scores": {pair_id: {"clap_htsat": f, "clap_music": f, "scs": f}}}

Resumable (only missing metrics per pair are computed); flushes every 50 pairs.
Runs in the MF/Omni env (transformers has ClapModel); CLAP wants a GPU but falls
back to CPU.  SCS is pure-CPU librosa, so --metrics scs runs anywhere.

    python score_baselines.py --manifest perturbations/pairs_manifest.json \
        --audio-dir perturbations/audio --out results/baselines/baseline_scores.json \
        --metrics clap_htsat,clap_music,scs         # --limit N for a smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# --- CLAP config -------------------------------------------------------------
CLAP_CKPT = {
    "clap_htsat": "laion/clap-htsat-unfused",
    "clap_music": "laion/larger_clap_music",
}
CLAP_SR = 48000          # CLAP front-end sampling rate
SCS_SR = 22050           # SCS features: librosa default rate
SCS_FRAMES = 48          # common SSM side length A/B are resampled to
ALL_METRICS = list(CLAP_CKPT) + ["scs", "fad_proxy"]

# globals populated lazily by load_clap()
torch = None
DEVICE = None
_clap_models: dict[str, object] = {}
_clap_procs: dict[str, object] = {}
# in-memory per-file embedding cache (refs repeat ~21x across pairs) -> {ckpt: {path: unit-vec}}
_emb_cache: dict[str, dict[tuple, np.ndarray]] = {}
_readout_logged: set = set()     # (metric, readout) already diagnosed to the log


# --- CLAP --------------------------------------------------------------------
def load_clap(metric: str) -> None:
    """Load a CLAP checkpoint on first use."""
    global torch, DEVICE
    if metric in _clap_models:
        return
    if torch is None:
        import torch as _torch
        torch = _torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("torch device=%s", DEVICE)
    from transformers import ClapModel, ClapProcessor
    ckpt = CLAP_CKPT[metric]
    logging.info("loading CLAP %s (%s) ...", metric, ckpt)
    _clap_procs[metric] = ClapProcessor.from_pretrained(ckpt)
    _clap_models[metric] = ClapModel.from_pretrained(ckpt).to(DEVICE).eval()
    _emb_cache.setdefault(metric, {})


def _clap_inputs(metric: str, path: str, secs: float):
    import librosa
    y, _ = librosa.load(path, sr=CLAP_SR, mono=True, duration=secs)
    proc = _clap_procs[metric]
    try:                                                # newer transformers: audio=
        inp = proc(audio=[y], sampling_rate=CLAP_SR, return_tensors="pt")
    except (TypeError, ValueError):                     # older transformers: audios=
        inp = proc(audios=[y], sampling_rate=CLAP_SR, return_tensors="pt")
    return inp.to(DEVICE)


def _embed_projected(metric: str, inp):
    """The 512-d CONTRASTIVE projection -- what CLAP is trained to compare (T4).

    `get_audio_features()` is documented to return the projected embedding, but
    the pooled branch below fires on the box's build, so we verify rather than
    trust: accept its output only if it is a flat tensor of projection_dim
    width, else run the projection head on the tower's pooled output ourselves.
    """
    m = _clap_models[metric]
    proj_dim = getattr(m.config, "projection_dim", None)
    out = m.get_audio_features(**inp)
    t = out if torch.is_tensor(out) else getattr(out, "audio_embeds", None)
    if torch.is_tensor(t) and t.ndim == 2 and (proj_dim is None or t.shape[-1] == proj_dim):
        return t, f"get_audio_features -> ({t.shape[-1]}d)"
    # explicit fallback: audio tower -> pooled -> audio_projection
    ao = m.audio_model(**inp)
    pooled = getattr(ao, "pooler_output", None)
    if pooled is None:
        h = ao.last_hidden_state
        pooled = h.reshape(h.shape[0], h.shape[1], -1).mean(-1) if h.ndim > 3 else h.mean(1)
    e = m.audio_projection(pooled)
    return e, f"audio_projection(pooler_output) -> ({e.shape[-1]}d)"


def _embed_pooled(metric: str, inp):
    """Audio-tower POOLED features (the original readout; keeps old cache valid)."""
    out = _clap_models[metric].get_audio_features(**inp)
    # this CLAP build returns the audio-tower ModelOutput, not a flat embedding.
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        t = out.pooler_output                           # (batch, dim) pooled audio repr
        how = "pooler_output"
    elif hasattr(out, "last_hidden_state"):
        t = out.last_hidden_state                       # (batch, C, H, W) feature map
        how = "last_hidden_state"
    elif isinstance(out, (tuple, list)):
        t = out[0]
        how = "out[0]"
    else:
        t = out                                         # already a tensor
        how = "tensor"
    return t, f"{how} -> {tuple(t.shape)}"


def clap_embed(metric: str, path: str, secs: float, readout: str = "pooled") -> np.ndarray:
    """Unit-norm CLAP audio embedding for a clip, cached by (readout, path)."""
    cache = _emb_cache[metric]
    key = (readout, path)
    if key in cache:
        return cache[key]
    inp = _clap_inputs(metric, path, secs)
    with torch.no_grad():
        t, how = (_embed_projected(metric, inp) if readout == "projected"
                  else _embed_pooled(metric, inp))
    if (metric, readout) not in _readout_logged:        # diagnose once per metric
        logging.info("CLAP %s readout=%s: %s", metric, readout, how)
        _readout_logged.add((metric, readout))
    e = np.asarray(t.detach().float().cpu().numpy())
    while e.ndim > 2:                                   # mean-pool trailing spatial dims
        e = e.mean(axis=-1)
    e = e.reshape(e.shape[0], -1)[0]                     # first (only) clip -> 1-D
    e = e / (np.linalg.norm(e) + 1e-8)
    cache[key] = e
    return e


def clap_score(metric: str, a: str, b: str, secs: float, readout: str = "pooled") -> float:
    ea, eb = clap_embed(metric, a, secs, readout), clap_embed(metric, b, secs, readout)
    return float((1.0 + np.dot(ea, eb)) / 2.0)          # cos in [-1,1] -> [0,1]


def fad_proxy_score(a: str, b: str, secs: float, readout: str = "pooled") -> float:
    """Per-pair FAD stand-in: embedding-distance similarity on htsat embeddings."""
    ea = clap_embed("clap_htsat", a, secs, readout)
    eb = clap_embed("clap_htsat", b, secs, readout)
    return float(1.0 / (1.0 + np.linalg.norm(ea - eb)))


# --- SCS structural stand-in (CPU) ------------------------------------------
def _struct_feats(path: str, secs: float) -> np.ndarray:
    """chroma(12)+MFCC(20), row-standardised, resampled to SCS_FRAMES columns."""
    import librosa
    y, sr = librosa.load(path, sr=SCS_SR, mono=True, duration=secs)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    f = np.vstack([chroma, mfcc])                        # (32, T)
    # resample time axis to a common length so A/B SSMs are comparable
    T = f.shape[1]
    if T < 2:
        return np.zeros((f.shape[0], SCS_FRAMES))
    xi = np.linspace(0, T - 1, SCS_FRAMES)
    f = np.stack([np.interp(xi, np.arange(T), row) for row in f])
    mu = f.mean(axis=1, keepdims=True); sd = f.std(axis=1, keepdims=True) + 1e-8
    return (f - mu) / sd


def _ssm(f: np.ndarray) -> np.ndarray:
    """Cosine self-similarity matrix over frames."""
    n = f / (np.linalg.norm(f, axis=0, keepdims=True) + 1e-8)
    return n.T @ n                                       # (SCS_FRAMES, SCS_FRAMES)


def scs_score(a: str, b: str, secs: float) -> float:
    """Structural coherence stand-in: SSM(A) vs SSM(B) upper-triangle cosine."""
    sa, sb = _ssm(_struct_feats(a, secs)), _ssm(_struct_feats(b, secs))
    iu = np.triu_indices(SCS_FRAMES, k=1)
    ua, ub = sa[iu], sb[iu]
    cos = float(np.dot(ua, ub) / ((np.linalg.norm(ua) + 1e-8) * (np.linalg.norm(ub) + 1e-8)))
    return (1.0 + cos) / 2.0                             # [-1,1] -> [0,1]


# --- driver ------------------------------------------------------------------
def resolve(path_str: str, audio_dir: Path | None) -> str:
    p = Path(path_str)
    return str(audio_dir / p.name) if audio_dir is not None else str(p)


def compute_one(metric: str, a: str, b: str, secs: float, readout: str = "pooled") -> float:
    if metric in CLAP_CKPT:
        return clap_score(metric, a, b, secs, readout)
    if metric == "fad_proxy":
        return fad_proxy_score(a, b, secs, readout)
    if metric == "scs":
        return scs_score(a, b, secs)                     # CLAP-free; readout N/A
    raise ValueError(f"unknown metric {metric!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/baselines/baseline_scores.json"))
    ap.add_argument("--metrics", default="clap_htsat,clap_music,scs",
                    help="comma list from %s, or 'all'" % ",".join(ALL_METRICS))
    ap.add_argument("--secs", type=float, default=8.0,
                    help="clip length per side (match the S protocol: 8s)")
    ap.add_argument("--readout", choices=("pooled", "projected"), default="pooled",
                    help="T4 FAIRNESS: pooled = audio-tower pooled features (DEFAULT; what "
                         "results/baselines/baseline_scores.json contains -- kept default so that "
                         "cache stays reproducible). projected = the 512-d contrastive "
                         "projection CLAP was actually trained to compare. Report BOTH; "
                         "write projected to a SEPARATE --out file.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    metrics = ALL_METRICS if args.metrics == "all" else args.metrics.split(",")
    bad = [m for m in metrics if m not in ALL_METRICS]
    if bad:
        ap.error(f"unknown metric(s) {bad}; choose from {ALL_METRICS}")

    doc = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs, meta = doc["pairs"], doc.get("meta", {})

    cache: dict[str, dict] = {}
    if args.out.is_file():
        prev = json.loads(args.out.read_text())
        # Resume keys on pair_id+metric only, so resuming a `projected` run into a
        # `pooled` cache would skip every pair and silently mix two readouts into
        # one table.  Readouts get separate --out files.
        prev_readout = prev.get("meta", {}).get("clap_readout", "pooled")
        if prev_readout != args.readout:
            ap.error(f"{args.out} was written with clap_readout={prev_readout!r} but "
                     f"--readout={args.readout!r}; write a separate cache "
                     f"(e.g. results/baselines/baseline_scores_projected.json) instead of mixing readouts")
        cache = {k: dict(v) for k, v in prev.get("scores", {}).items()}
        logging.info("resuming: %d pairs already have some baseline scores", len(cache))

    # a pair needs work if any requested metric is missing from its entry
    todo = [p for p in pairs if any(m not in cache.get(p["pair_id"], {}) for m in metrics)]
    if args.limit:
        todo = todo[:args.limit]
    logging.info("to score: %d pairs x metrics=%s", len(todo), metrics)

    # load only the CLAP checkpoints we actually need
    need_htsat = ("fad_proxy" in metrics)               # fad_proxy reuses htsat embeddings
    for m in metrics:
        if m in CLAP_CKPT:
            load_clap(m)
    if need_htsat and "clap_htsat" not in _clap_models:
        load_clap("clap_htsat")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        args.out.write_text(json.dumps(
            {"meta": {**meta, "baselines": metrics, "secs": args.secs,
                      "clap_readout": args.readout,
                      "clap_ckpt": {m: CLAP_CKPT[m] for m in metrics if m in CLAP_CKPT},
                      "scs": "signal-structural STAND-IN (SSM cosine), not official MusicWeaver SCS",
                      "fad_proxy": "per-pair embedding-distance proxy; report set-level FAD separately"},
             "scores": cache}, indent=2), encoding="utf-8")

    for i, p in enumerate(todo, 1):
        a = resolve(p["ref_path"], args.audio_dir)
        b = resolve(p["cand_path"], args.audio_dir)
        entry = cache.setdefault(p["pair_id"], {})
        for m in metrics:
            if m in entry:
                continue
            try:
                entry[m] = compute_one(m, a, b, args.secs, args.readout)
            except Exception as exc:                     # noqa: BLE001
                logging.error("pair %s metric %s failed: %s", p["pair_id"], m, exc,
                              exc_info=args.verbose)
        if i % 50 == 0 or i == len(todo):
            flush()
            logging.info("scored %d/%d (last=%s)", i, len(todo),
                         {m: round(cache[p["pair_id"]].get(m, float("nan")), 3) for m in metrics})
    flush()
    logging.info("done -> %s (%d pairs)", args.out, len(cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
