#!/usr/bin/env python3
"""R2 -- can a RAW CLAP EMBEDDING name which perturbation happened?

The E-A diagnosis result (`diagnose_perturbation.py`) compares a 4-dim H/T/R/S
profile against single published SCALARS, and declares embeddings out of scope:
"a 768-d CLAP embedding delta is not a metric, so it is out of scope by
construction". That scope statement is defensible on its own terms and it is
also the first thing a reviewer will push on, because "out of scope" is
indistinguishable from "not run" when the missing experiment is the one that
could weaken the claim. This script runs it.

THE QUESTION. Feed the same leave-one-TRACK-out folds and the same two
classifiers a learned CLAP representation instead of the metric scores:
  * clap768_diff  -- e_B - e_A on the 768-d audio-tower pooled features
  * clap768_ab    -- [e_A, e_B] concatenated (1536-d)
  * clap768_full  -- [e_A, e_B, e_B - e_A] (2304-d)
  * clap512_*     -- the same three on the 512-d contrastive projection
against HTRS / HTR / the published scalars. Two outcomes, both reportable:
the probe loses and "only a decomposition can say WHAT changed" is bulletproof;
the probe wins and the honest claim narrows to interpretability and per-axis
attribution at four numbers instead of 2304 -- which is what the re-ranking and
RL sections actually exploit, so it is still a contribution.

TWO ENVIRONMENTS, communicating through one .npz on disk (as everywhere else in
this repo; see env/). Neither step needs the MF env.

  extract   torch + transformers >= 4.27 (locally: the `SNLP` env,
            torch 2.2.2 / transformers 4.40.0). Reads the pair manifest's wavs,
            writes results/baselines/clap_embeddings.npz.
  probe     the DSP env (anaconda base; needs sklearn). Reads that .npz plus
            pair_scores_cbase.csv, writes results/diagnostics/clap_probe.json.

    ~/anaconda3/envs/SNLP/bin/python scripts/analysis/clap_probe.py extract
    python scripts/analysis/clap_probe.py probe

READOUT FIDELITY MATTERS. The published `clap_htsat` scalar in
results/baselines/baseline_scores.json was computed on the GPU box, where an
older transformers returned the audio-tower ModelOutput from
`get_audio_features()`, so `score_baselines._embed_pooled` read its
`pooler_output` (768-d). On transformers >= 4.30 the same call returns the
512-d projection directly, so reproducing the published number here requires
calling `model.audio_model(...).pooler_output` explicitly. `extract --verify`
checks the reconstructed cosines against the published cache before spending
half an hour on the full set; do not skip it.
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
try:  # sibling imports across script groups (see scripts/README.md)
    sys.path.insert(0, str(_here.parent.parent))
    import _paths  # noqa: F401
except ImportError:
    pass

CKPT = "laion/clap-htsat-unfused"
CLAP_SR = 48000
SECS = 8.0                      # matches score_baselines.py --secs default
FAMILIES = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]


def repo_root() -> Path:
    """Walk up for the root marker rather than counting .parent."""
    for p in [_here, *_here.parents]:
        if (p / ".projectroot").exists() or (p / ".git").exists():
            return p
    return Path.cwd()


# --- extract (torch env) ------------------------------------------------------
def load_wav(path: Path, secs: float = SECS) -> np.ndarray:
    """First `secs` of a mono PCM_16 48k wav as float32 in [-1,1].

    Equivalent to librosa.load(sr=48000, mono=True, duration=secs) for the
    corpus in perturbations/audio (verified: identical peak amplitude), without
    needing librosa in the torch env.
    """
    from scipy.io import wavfile

    sr, data = wavfile.read(path)
    if sr != CLAP_SR:
        raise ValueError(f"{path}: expected {CLAP_SR} Hz, got {sr}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    else:
        y = data.astype(np.float32)
    return y[: int(secs * CLAP_SR)]


def cmd_extract(args) -> int:
    import torch
    from transformers import ClapModel, ClapProcessor

    root = repo_root()
    manifest = json.load(open(root / args.manifest))
    pairs = manifest["pairs"] if isinstance(manifest, dict) else manifest

    clips: list[str] = []
    seen: set[str] = set()
    for p in pairs:
        for k in ("ref_path", "cand_path"):
            if p[k] not in seen:
                seen.add(p[k])
                clips.append(p[k])
    if args.limit:
        keep = set()
        for p in pairs[: args.limit]:
            keep.update((p["ref_path"], p["cand_path"]))
        clips = [c for c in clips if c in keep]
    logging.info("%d pairs -> %d unique clips", len(pairs), len(clips))

    proc = ClapProcessor.from_pretrained(CKPT)
    model = ClapModel.from_pretrained(CKPT).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    logging.info("CLAP %s on %s", CKPT, dev)

    pooled, projected = {}, {}
    t0 = time.time()
    for i, rel in enumerate(clips, 1):
        y = load_wav(root / rel)
        inp = proc(audios=[y], sampling_rate=CLAP_SR, return_tensors="pt").to(dev)
        with torch.no_grad():
            ao = model.audio_model(**inp)
            po = ao.pooler_output
            if po is None:                       # defensive: mirror score_baselines
                h = ao.last_hidden_state
                po = h.reshape(h.shape[0], h.shape[1], -1).mean(-1) if h.ndim > 3 else h.mean(1)
            pr = model.audio_projection(po)
        for store, t in ((pooled, po), (projected, pr)):
            e = t.detach().float().cpu().numpy().reshape(-1)
            store[rel] = (e / (np.linalg.norm(e) + 1e-8)).astype(np.float32)
        if i % 100 == 0 or i == len(clips):
            el = time.time() - t0
            logging.info("%d/%d clips  %.1fs elapsed  eta %.1f min",
                         i, len(clips), el, (el / i) * (len(clips) - i) / 60)

    if args.verify:
        cache = json.load(open(root / args.baselines))["scores"]
        errs = []
        for p in pairs:
            got = cache.get(p["pair_id"], {}).get("clap_htsat")
            if got is None or p["ref_path"] not in pooled or p["cand_path"] not in pooled:
                continue
            cos = float(np.dot(pooled[p["ref_path"]], pooled[p["cand_path"]]))
            errs.append(abs((1.0 + cos) / 2.0 - got))
        if errs:
            errs = np.array(errs)
            print(f"\nverify vs published clap_htsat on {len(errs)} pairs: "
                  f"max |err| = {errs.max():.2e}, mean = {errs.mean():.2e}")
            if errs.max() > 1e-3:
                print("  WARNING: readout does NOT match the published cache; "
                      "the probe would not be on the same footing as the baseline.")
            else:
                print("  OK -- 768-d pooler_output reproduces the published scalar.")
        else:
            print("\nverify: no overlap with the published cache")

    dest = root / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest,
        clips=np.array(clips),
        pooled=np.stack([pooled[c] for c in clips]),
        projected=np.stack([projected[c] for c in clips]),
        meta=np.array([json.dumps({"ckpt": CKPT, "secs": SECS, "sr": CLAP_SR,
                                   "pooled_dim": int(len(next(iter(pooled.values())))),
                                   "projected_dim": int(len(next(iter(projected.values())))),
                                   "readout_pooled": "audio_model(...).pooler_output, unit-norm",
                                   "readout_projected": "audio_projection(pooler_output), unit-norm",
                                   "n_clips": len(clips)})]),
    )
    print(f"[wrote {dest}]  {len(clips)} clips")
    return 0


# --- probe (DSP env) ----------------------------------------------------------
def cmd_probe(args) -> int:
    import csv

    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    root = repo_root()
    z = np.load(root / args.emb, allow_pickle=True)
    clips = list(z["clips"])
    emb = {"clap768": dict(zip(clips, z["pooled"])), "clap512": dict(zip(clips, z["projected"]))}
    print("embeddings:", json.loads(str(z["meta"][0])))

    manifest = json.load(open(root / args.manifest))
    mpairs = {p["pair_id"]: p for p in (manifest["pairs"] if isinstance(manifest, dict) else manifest)}

    scores_path = root / args.scores
    if not scores_path.exists():
        scores_path = root / "results/coherence/pair_scores.csv"
    rows = [r for r in csv.DictReader(open(scores_path)) if r["perturbation"] in FAMILIES]
    print(f"{len(rows)} pairs in {len(FAMILIES)} families from {scores_path.name}")

    def scalar(r, key):
        v = r.get(key)
        return float(v) if v not in (None, "") else np.nan

    feats: dict[str, list[np.ndarray]] = {}
    y, groups = [], []
    scalar_sets = {
        "HTRS": ["score_H", "score_T", "score_R", "score_S"],
        "HTR": ["score_H", "score_T", "score_R"],
        "clap_htsat (scalar)": ["score_clap_htsat"],
        "scs (scalar)": ["score_scs"],
        "cbase (scalar)": ["score_cbase"],
    }
    scalar_sets = {k: v for k, v in scalar_sets.items()
                   if all(c in rows[0] for c in v)}

    kept = 0
    for r in rows:
        pid = r["pair_id"]
        p = mpairs.get(pid)
        if p is None or p["ref_path"] not in emb["clap768"] or p["cand_path"] not in emb["clap768"]:
            continue
        vecs = {}
        for tag in ("clap768", "clap512"):
            ea, eb = emb[tag][p["ref_path"]], emb[tag][p["cand_path"]]
            vecs[f"{tag}_diff"] = eb - ea
            vecs[f"{tag}_ab"] = np.concatenate([ea, eb])
            vecs[f"{tag}_full"] = np.concatenate([ea, eb, eb - ea])
        # Control for the front-end discrepancy: these embeddings were extracted
        # under transformers 4.40, the published clap_htsat cache under whatever the
        # GPU box ran (mean |delta| 3.9e-3 on the cosine, max 4.0e-2). If the locally
        # recomputed cosine diagnoses like the published scalar, the difference is
        # irrelevant to this task and the embedding rows are on the same footing.
        ea, eb = emb["clap768"][p["ref_path"]], emb["clap768"][p["cand_path"]]
        vecs["clap_htsat cos (local)"] = np.array([(1.0 + float(np.dot(ea, eb))) / 2.0])
        sv = {k: np.array([scalar(r, c) for c in cols]) for k, cols in scalar_sets.items()}
        if any(np.isnan(v).any() for v in sv.values()):
            continue
        for k, v in {**sv, **vecs}.items():
            feats.setdefault(k, []).append(v)
        y.append(FAMILIES.index(r["perturbation"]))
        groups.append(r["source_id"])
        kept += 1

    y = np.array(y)
    groups = np.array(groups)
    usrc = sorted(set(groups.tolist()))
    print(f"{kept} pairs usable, {len(usrc)} leave-one-track-out folds\n")

    def _fold(X: np.ndarray, u: str):
        """One leave-one-track-out fold: z-score on TRAIN stats only, then both classifiers."""
        te = groups == u
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        cent = np.stack([Xtr[y[tr] == c].mean(0) if (y[tr] == c).any() else np.full(X.shape[1], 1e9)
                         for c in range(len(FAMILIES))])
        nc = np.argmin(((Xte[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        lr = LogisticRegression(max_iter=args.max_iter, C=args.C)
        lr.fit(Xtr, y[tr])
        return te, nc, lr.predict(Xte)

    def run(X: np.ndarray) -> dict:
        pred_nc = np.zeros_like(y)
        pred_lr = np.zeros_like(y)
        # 90 folds x up to 2304 features: serial lbfgs takes hours, so fan the folds out.
        # Folds are independent (train stats are computed inside each), so this is exact.
        for te, nc, lp in Parallel(n_jobs=args.jobs, backend="loky")(
                delayed(_fold)(X, u) for u in usrc):
            pred_nc[te] = nc
            pred_lr[te] = lp
        return {
            "nearest_centroid": {"acc": float((pred_nc == y).mean()),
                                 "macro_f1": float(f1_score(y, pred_nc, average="macro"))},
            "logistic": {"acc": float((pred_lr == y).mean()),
                         "macro_f1": float(f1_score(y, pred_lr, average="macro"))},
        }

    order = [k for k in scalar_sets] + ["clap_htsat cos (local)"] + [
        f"{t}_{s}" for t in ("clap768", "clap512") for s in ("diff", "ab", "full")]
    if args.only:
        order = [k for k in order if k in args.only]
    out = {"n_pairs": kept, "n_folds": len(usrc), "families": FAMILIES,
           "chance_acc": 1.0 / len(FAMILIES), "results": {}}
    hdr = f"{'feature set':<22s}{'dim':>5s}  {'NC acc':>7s}{'NC F1':>7s}  {'LR acc':>7s}{'LR F1':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for k in order:
        X = np.stack(feats[k])
        t0 = time.time()
        res = run(X)
        out["results"][k] = {"dim": int(X.shape[1]), "secs": round(time.time() - t0, 1), **res}
        print(f"{k:<22s}{X.shape[1]:>5d}  "
              f"{res['nearest_centroid']['acc']:>7.3f}{res['nearest_centroid']['macro_f1']:>7.3f}  "
              f"{res['logistic']['acc']:>7.3f}{res['logistic']['macro_f1']:>7.3f}",
              flush=True)
    print(f"\nchance = {1/len(FAMILIES):.3f}")

    dest = root / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"[wrote {dest}]")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="CLAP embeddings -> .npz (torch env)")
    e.add_argument("--manifest", default="perturbations/pairs_manifest.json")
    e.add_argument("--baselines", default="results/baselines/baseline_scores.json")
    e.add_argument("--out", default="results/baselines/clap_embeddings.npz")
    e.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke test)")
    e.add_argument("--verify", action="store_true",
                   help="check reconstructed cosines against the published clap_htsat cache")
    e.set_defaults(func=cmd_extract)

    p = sub.add_parser("probe", help="leave-one-track-out diagnosis (DSP env)")
    p.add_argument("--emb", default="results/baselines/clap_embeddings.npz")
    p.add_argument("--manifest", default="perturbations/pairs_manifest.json")
    p.add_argument("--scores", default="results/coherence/pair_scores_cbase.csv")
    p.add_argument("--out", default="results/diagnostics/clap_probe.json")
    p.add_argument("--jobs", type=int, default=-1, help="parallel folds (-1 = all cores)")
    p.add_argument("--max-iter", type=int, default=400,
                   help="lbfgs iterations. 400 is enough for the low-dim score sets but NOT "
                        "for the embeddings (369/540 fits hit the limit) -- and an unconverged "
                        "fit UNDERSTATES the probe, which biases in favour of the thesis. Use "
                        "5000 when reporting embedding numbers.")
    p.add_argument("--C", type=float, default=1.0,
                   help="inverse L2 strength. 2304 features on 1800 samples is "
                        "overparameterised, so sweep this rather than handicap the probe.")
    p.add_argument("--only", nargs="*", default=None, metavar="SET",
                   help="restrict to these feature sets (e.g. --only clap768_diff)")
    p.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
