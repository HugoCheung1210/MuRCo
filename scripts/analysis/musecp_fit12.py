#!/usr/bin/env python
"""Held-out Part-1 fit: MuseCPEval's 12 metrics against MuRCo's 4, same protocol.

A 12-parameter model compared against a 0-parameter one is not a comparison, so every
feature set here goes through the SAME nested loop: leave-one-source-track-out on the
outer level, ridge alpha chosen by an inner GroupKFold over training tracks only,
features standardised inside each training fold.  Out-of-fold predictions are collected
over all outer folds and scored once, and the differences carry a cluster bootstrap over
source tracks (several rated pairs share a track and are not independent).

``C_uniform`` is the untuned reference the paper reports: the geometric mean of H,T,R,S at
uniform weights, zero fitted parameters, no folds needed.

Usage (DSP env, from the repo root)::

    python scripts/analysis/musecp_fit12.py \
        --mos results/mos/mos_live.csv \
        --scores results/coherence/pair_scores.csv \
        --musecp results/musecp/setup1/musecp_metrics.csv \
        --out results/musecp/part1_fit12.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
try:  # sibling imports across script groups (see scripts/README.md)
    sys.path.insert(0, str(_here.parent.parent))
    import _paths  # noqa: F401
except ImportError:
    pass

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
DIMS = ["H", "T", "R", "S"]
TIMBRE = {"musecp_mfcc_skl", "musecp_mfcc_cos"}


def repo_root() -> Path:
    for p in [_here] + list(_here.parents):
        if (p / "CLAUDE.md").exists() or (p / ".git").exists():
            return p
    return _here.parent.parent.parent


def read_csv(path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def nested_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list]:
    """Leave-one-group-out predictions; alpha chosen inside each training fold only."""
    oof = np.full(len(y), np.nan)
    chosen = []
    for g in sorted(set(groups)):
        te = groups == g
        tr = ~te
        if tr.sum() < 10 or te.sum() == 0:
            continue
        Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-12] = 1.0
        Ztr = (Xtr - mu) / sd

        n_inner = min(5, len(set(gtr)))
        best_alpha, best_score = ALPHAS[0], -np.inf
        if n_inner >= 2:
            inner = list(GroupKFold(n_splits=n_inner).split(Ztr, ytr, gtr))
            for a in ALPHAS:
                preds = np.full(len(ytr), np.nan)
                for itr, ite in inner:
                    m = Ridge(alpha=a).fit(Ztr[itr], ytr[itr])
                    preds[ite] = m.predict(Ztr[ite])
                ok = np.isfinite(preds)
                s = pearson(preds[ok], ytr[ok])
                if np.isfinite(s) and s > best_score:
                    best_score, best_alpha = s, a
        chosen.append(best_alpha)
        model = Ridge(alpha=best_alpha).fit(Ztr, ytr)
        oof[te] = model.predict((X[te] - mu) / sd)
    return oof, chosen


def boot_diff(y, groups, preds: dict, a: str, b: str, n_boot=20000, seed=0) -> dict:
    """Cluster bootstrap over source tracks on the DIFFERENCE of two correlations."""
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups))
    idx_by_g = {g: np.flatnonzero(groups == g) for g in gs}
    pa, pb = preds[a], preds[b]
    ok = np.isfinite(pa) & np.isfinite(pb)
    obs = pearson(pa[ok], y[ok]) - pearson(pb[ok], y[ok])
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(gs), size=len(gs), replace=True)
        sel = np.concatenate([idx_by_g[gs[i]] for i in pick])
        sel = sel[np.isfinite(pa[sel]) & np.isfinite(pb[sel])]
        if len(sel) < 5:
            continue
        d = pearson(pa[sel], y[sel]) - pearson(pb[sel], y[sel])
        if np.isfinite(d):
            draws.append(d)
    draws = np.array(draws)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta_r": round(float(obs), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_boot": int(len(draws))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mos", default="results/mos/mos_live.csv")
    ap.add_argument("--scores", default="results/coherence/pair_scores.csv")
    ap.add_argument("--musecp", default="results/musecp/setup1/musecp_metrics.csv")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/musecp/part1_fit12.json")
    args = ap.parse_args()

    root = repo_root()
    mos = {r["pair_id"]: float(r["mos"]) for r in read_csv(root / args.mos)}
    murco = {r["pair_id"]: r for r in read_csv(root / args.scores)}
    mcp_rows = read_csv(root / args.musecp)
    mcp = {r["pair_id"]: r for r in mcp_rows}
    mcp_cols = [c for c in mcp_rows[0] if c != "pair_id"
                and c not in ("musecp_mean12", "musecp_mean10")]

    ids = [p for p in mos if p in murco and p in mcp]
    dropped = sorted(set(mos) - set(ids))
    print(f"{len(mos)} rated pairs, {len(ids)} matched, {len(dropped)} dropped")
    if dropped:
        print("  dropped e.g.", dropped[:3])

    y = np.array([mos[p] for p in ids], float)
    groups = np.array([murco[p]["source_id"] for p in ids])
    print(f"{len(set(groups))} source tracks -> {len(set(groups))} outer folds")

    Xm = np.array([[float(mcp[p][c]) for c in mcp_cols] for p in ids])
    keep10 = [i for i, c in enumerate(mcp_cols) if c not in TIMBRE]
    Xh4 = np.array([[float(murco[p][f"score_{d}"]) for d in DIMS] for p in ids])
    Xh3 = Xh4[:, :3]

    sets = {"musecp12": Xm, "musecp10": Xm[:, keep10], "HTRS": Xh4, "HTR": Xh3}
    report = {"meta": {"n_rated": len(mos), "n_matched": len(ids),
                       "n_tracks": int(len(set(groups))), "alphas": ALPHAS,
                       "musecp_columns": mcp_cols,
                       "protocol": "leave-one-source-track-out; ridge alpha by inner "
                                   "GroupKFold on the training fold only; features "
                                   "standardised within each training fold",
                       "sources": {"mos": args.mos, "scores": args.scores,
                                   "musecp": args.musecp}},
               "held_out": {}, "in_sample": {}, "contrasts": {}}

    preds = {}
    for name, X in sets.items():
        oof, alphas = nested_oof(X, y, groups)
        preds[name] = oof
        ok = np.isfinite(oof)
        report["held_out"][name] = {
            "r": round(pearson(oof[ok], y[ok]), 4), "n_features": int(X.shape[1]),
            "n_predicted": int(ok.sum()),
            "alpha_median": float(np.median(alphas)) if alphas else None}
        mu, sd = X.mean(0), X.std(0); sd[sd < 1e-12] = 1.0
        m = Ridge(alpha=float(np.median(alphas)) if alphas else 1.0).fit((X - mu) / sd, y)
        report["in_sample"][name] = {
            "r": round(pearson(m.predict((X - mu) / sd), y), 4),
            "coefficients": {c: round(float(v), 4) for c, v in zip(
                (mcp_cols if name == "musecp12" else
                 [mcp_cols[i] for i in keep10] if name == "musecp10" else
                 DIMS if name == "HTRS" else DIMS[:3]), m.coef_)}}
        print(f"{name:>9}  held-out r = {report['held_out'][name]['r']:.4f}"
              f"  (in-sample {report['in_sample'][name]['r']:.4f},"
              f" {X.shape[1]} features)")

    # the untuned reference the paper reports
    C_unif = np.exp(np.log(np.clip(Xh4, 1e-6, None)).mean(axis=1))
    preds["C_uniform"] = C_unif
    report["held_out"]["C_uniform"] = {"r": round(pearson(C_unif, y), 4), "n_features": 0,
                                       "n_predicted": len(y), "alpha_median": None,
                                       "note": "geometric mean at uniform weights, "
                                               "zero fitted parameters"}
    print(f"C_uniform  held-out r = {report['held_out']['C_uniform']['r']:.4f}  (0 fitted)")

    for a, b in [("C_uniform", "musecp12"), ("C_uniform", "musecp10"),
                 ("HTRS", "musecp12"), ("HTR", "musecp12"),
                 ("HTRS", "HTR"), ("musecp12", "musecp10")]:
        report["contrasts"][f"{a} - {b}"] = boot_diff(y, groups, preds, a, b,
                                                      args.boot, args.seed)
        c = report["contrasts"][f"{a} - {b}"]
        print(f"  {a} - {b}: {c['delta_r']:+.4f} {c['ci95']}"
              f"{'  excludes zero' if c['excludes_zero'] else '  spans zero'}")

    dest = root / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1))
    print("wrote", dest.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
