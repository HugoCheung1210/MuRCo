#!/usr/bin/env python
"""How much of the metric rests on T's temperature tau?

tau is the one hand-set constant inside a dimension (coherence_dimensions.py:103).
It is swept here WITHOUT rescoring any audio, because T = exp(-d/tau) is invertible:
d = -tau0 * log(T), so T at any other temperature is T_tau = T_tau0 ** (tau0 / tau).
Everything downstream is then recomputed from the cached pair_scores.csv.

Two things are worth separating in the output.

  * Rank statistics cannot move. exp(-d/tau) is monotone in d for every tau > 0, so
    the order in which T ranks pairs is fixed and every AUC is identical. The sweep
    reports them anyway, as a check on that argument rather than a measurement.
  * Magnitudes do move, because a mean drop is a difference of scores and monotone
    transforms do not preserve differences. What matters is whether the separability
    SIGNATURE survives, i.e. whether each family still peaks on the same dimension.

There is also an analytic redundancy worth stating. log C = sum_d w_d log d, and
log T = -d/tau, so w_T log T = -(w_T/tau) d: changing tau is the same move as
rescaling T's weight, up to a monotone power on C. The weight sweep of Table 3.1
therefore already covers tau's effect on every rank-based claim about C.

Usage: python scripts/analysis/tau_sensitivity.py [--taus 2,3,4,6,9,12]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
TAU0 = 6.0
DIMS = ["H", "T", "R", "S"]
FAMILIES = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    r = rankdata(np.concatenate([pos, neg]))
    n1, n2 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def family_drops(score: np.ndarray, fam: np.ndarray, src: np.ndarray) -> dict:
    ctrl = defaultdict(list)
    for s, f, t in zip(score, fam, src):
        if f == "control":
            ctrl[t].append(s)
    ctrl = {t: float(np.mean(v)) for t, v in ctrl.items()}
    out = {}
    for family in FAMILIES:
        per = defaultdict(list)
        for s, f, t in zip(score, fam, src):
            if f == family:
                per[t].append(s)
        out[family] = float(np.mean([ctrl[t] - np.mean(v) for t, v in per.items() if t in ctrl]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=ROOT / "results/coherence/pair_scores.csv")
    ap.add_argument("--taus", default="2,3,4,6,9,12")
    ap.add_argument("--out", type=Path, default=ROOT / "results/coherence/tau_sensitivity.json")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.scores)))
    fam = np.array([r["perturbation"] for r in rows])
    src = np.array([r["source_id"] for r in rows])
    base = {d: np.array([float(r[f"score_{d}"]) for r in rows]) for d in DIMS}
    taus = [float(x) for x in a.taus.split(",")]

    report = {"meta": {"tau0": TAU0, "n_pairs": len(rows), "taus": taus,
                       "identity": "T_tau = T_tau0 ** (tau0 / tau); no audio rescored"},
              "by_tau": {}}

    for tau in taus:
        T = np.clip(base["T"], 1e-12, 1.0) ** (TAU0 / tau)
        dims = {**base, "T": T}
        logs = np.stack([np.log(np.clip(dims[d], 1e-12, 1.0)) for d in DIMS], axis=1)
        C = np.exp(logs.mean(axis=1))                       # equal weights, as reported

        same, diff = C[fam != "style_swap"], C[fam == "style_swap"]
        tsame = C[fam == "pitch_shift"]
        entry = {
            "T_drops": family_drops(T, fam, src),
            "C_drops": family_drops(C, fam, src),
            "C_control": float(np.mean([C[i] for i in range(len(rows)) if fam[i] == "control"])),
            "auc_C_broad": round(auc(same, diff), 4),
            "auc_C_transpose": round(auc(tsame, diff), 4),
            "auc_T_broad": round(auc(T[fam != "style_swap"], T[fam == "style_swap"]), 4),
        }
        entry["family_peak_dim"] = {}
        for family in FAMILIES:
            drops = {d: family_drops(dims[d], fam, src)[family] for d in DIMS}
            entry["family_peak_dim"][family] = max(drops, key=drops.get)
        report["by_tau"][f"{tau:g}"] = entry

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"{'tau':>5} | {'T lowpass':>9} {'T swap':>7} | {'C lowpass':>9} {'C swap':>7} "
          f"| {'C ctrl':>7} | {'AUC T':>7} | {'AUC C broad':>11} {'AUC C transp':>12} | signature")
    # the shipped value is the reference, not whichever tau happens to be swept first
    ref = "".join(report["by_tau"][f"{TAU0:g}"]["family_peak_dim"][f][0] for f in FAMILIES)
    for tau in taus:
        e = report["by_tau"][f"{tau:g}"]
        sig = "".join(e["family_peak_dim"][f][0] for f in FAMILIES)
        print(f"{tau:5g} | {e['T_drops']['lowpass']:9.3f} {e['T_drops']['style_swap']:7.3f} "
              f"| {e['C_drops']['lowpass']:9.3f} {e['C_drops']['style_swap']:7.3f} "
              f"| {e['C_control']:7.3f} | {e['auc_T_broad']:7.4f} | {e['auc_C_broad']:11.4f} {e['auc_C_transpose']:12.4f} "
              f"| {sig}{'' if sig == ref else '  <-- CHANGED'}")
    print(f"\nsignature = peak dimension for {', '.join(FAMILIES)}; reference is tau={TAU0:g}")
    print("AUC T is constant by construction: exp(-d/tau) is monotone in d.")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
