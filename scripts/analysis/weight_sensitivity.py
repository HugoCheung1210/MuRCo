#!/usr/bin/env python
"""Weight-sensitivity sweep for C. DSP env, read-only, no GPU.

Preempts the examiner question "why uniform weights?": shows the headline
conclusions do not hinge on the uniform choice by sweeping the weight simplex
and checking, at every weighting w (C_w = Π_d s_d^{w_d}, Σw=1):

  I1  control-vs-perturbed: mean C_w(control) > mean C_w(family), every family
  I2  drop ORDERING stability: Spearman ρ between the family-drop vector under
      w and under uniform (1.0 = identical ordering)
  I3  style_swap is the largest C-drop family (the S-flavoured headline)
  I4  identity AUC (same piece vs style-swap) stays above each scalar baseline's
      fixed AUC where the uniform C beat it

Sweep = 4 vertices + 6 edges midpoints + uniform + n_dirichlet random draws.
Output: results/coherence/weight_sensitivity.json (+ printed summary).

Usage: python scripts/analysis/weight_sensitivity.py [--n-dirichlet 500]
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
DIMS = ["H", "T", "R", "S"]
EPS = 1e-6
FAMILIES = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]


def load():
    rows = list(csv.DictReader(open(ROOT / "results/coherence/pair_scores.csv")))
    logs = np.array([[np.log(np.clip(float(r[f"score_{d}"]), EPS, 1.0))
                      for d in DIMS] for r in rows])
    fam = np.array([r["perturbation"] for r in rows])
    src = np.array([r["source_id"] for r in rows])
    base = json.load(open(ROOT / "results/baselines/baseline_scores.json"))["scores"]
    clap = np.array([base.get(r["pair_id"], {}).get("clap_htsat", np.nan) for r in rows])
    scs = np.array([base.get(r["pair_id"], {}).get("scs", np.nan) for r in rows])
    return logs, fam, src, clap, scs


def auc(pos, neg):
    """AUC that scores of same-piece pairs (pos) exceed style-swap pairs (neg)."""
    x = np.concatenate([pos, neg])
    r = rankdata(x)
    n1, n2 = len(pos), len(neg)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2)


def family_drops(scores, fam, src):
    ctrl = defaultdict(list)
    for s, f, t in zip(scores, fam, src):
        if f == "control":
            ctrl[t].append(s)
    ctrl = {t: np.mean(v) for t, v in ctrl.items()}
    drops = {}
    for family in FAMILIES:
        per_track = defaultdict(list)
        for s, f, t in zip(scores, fam, src):
            if f == family:
                per_track[t].append(s)
        d = [ctrl[t] - np.mean(v) for t, v in per_track.items() if t in ctrl]
        drops[family] = float(np.mean(d))
    return drops


def evaluate(w, logs, fam, src, ref_drop_vec, base_aucs):
    scores = logs @ w
    drops = family_drops(scores, fam, src)
    vec = [drops[f] for f in FAMILIES]
    i1 = all(v > 0 for v in vec)
    i2 = float(spearmanr(vec, ref_drop_vec).correlation)
    i3 = max(drops, key=drops.get) == "style_swap"
    same = scores[fam != "style_swap"]
    diff = scores[fam == "style_swap"]
    a = auc(same, diff)
    i4 = bool(a > max(base_aucs.values()))
    return {"drops": drops, "I1_control_above_all": bool(i1),
            "I2_ordering_rho_vs_uniform": round(i2, 4),
            "I3_styleswap_largest": bool(i3),
            "identity_auc": round(float(a), 4), "I4_auc_above_baselines": i4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dirichlet", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260718)
    args = ap.parse_args()
    logs, fam, src, clap, scs = load()

    # fixed baseline AUCs on the same broad split (same piece vs style-swap)
    base_aucs = {}
    for name, v in (("clap_htsat", clap), ("scs", scs)):
        ok = ~np.isnan(v)
        base_aucs[name] = round(float(
            auc(v[ok & (fam != "style_swap")], v[ok & (fam == "style_swap")])), 4)

    uniform = np.full(4, 0.25)
    ref_drop_vec = [family_drops(logs @ uniform, fam, src)[f] for f in FAMILIES]

    named = {"uniform": uniform}
    for i, d in enumerate(DIMS):
        v = np.zeros(4); v[i] = 1.0
        named[f"vertex_{d}"] = v
    for i in range(4):
        for j in range(i + 1, 4):
            v = np.zeros(4); v[i] = v[j] = 0.5
            named[f"edge_{DIMS[i]}{DIMS[j]}"] = v

    rng = np.random.default_rng(args.seed)
    draws = rng.dirichlet(np.ones(4), size=args.n_dirichlet)

    named_results = {k: evaluate(w, logs, fam, src, ref_drop_vec, base_aucs)
                     for k, w in named.items()}
    rand = [evaluate(w, logs, fam, src, ref_drop_vec, base_aucs) for w in draws]
    i2s = np.array([r["I2_ordering_rho_vs_uniform"] for r in rand])
    aucs = np.array([r["identity_auc"] for r in rand])
    summary = {
        "n_random": len(rand),
        "frac_I1_control_above_all": round(float(np.mean([r["I1_control_above_all"] for r in rand])), 3),
        "frac_I3_styleswap_largest": round(float(np.mean([r["I3_styleswap_largest"] for r in rand])), 3),
        "frac_I4_auc_above_baselines": round(float(np.mean([r["I4_auc_above_baselines"] for r in rand])), 3),
        "I2_ordering_rho": {"min": round(float(i2s.min()), 3),
                            "p05": round(float(np.percentile(i2s, 5)), 3),
                            "median": round(float(np.median(i2s)), 3)},
        "identity_auc": {"min": round(float(aucs.min()), 4),
                         "median": round(float(np.median(aucs)), 4),
                         "max": round(float(aucs.max()), 4)},
        "baseline_aucs_same_vs_styleswap": base_aucs,
    }
    out = {"meta": {"seed": args.seed, "dims": DIMS, "families": FAMILIES,
                    "sweep": "4 vertices + 6 edge midpoints + uniform + "
                             f"{args.n_dirichlet} Dirichlet(1) draws"},
           "summary": summary, "named": named_results}
    path = ROOT / "results/coherence/weight_sensitivity.json"
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("named points (I1/I3/I4, I2 rho, auc):")
    for k, r in named_results.items():
        print(f"  {k:<12} I1={r['I1_control_above_all']} I3={r['I3_styleswap_largest']} "
              f"I4={r['I4_auc_above_baselines']} I2rho={r['I2_ordering_rho_vs_uniform']} "
              f"auc={r['identity_auc']}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
