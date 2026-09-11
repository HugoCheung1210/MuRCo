#!/usr/bin/env python3
"""Does the composite lose more than CLAP does when the substitution is same-genre?

Section 6 records same-genre substitution as a limitation, on the grounds that it degrades
the whole metric rather than only S. That is true of the level and says nothing about the
comparison, which is what a reader wants: a limitation shared with every baseline is a
property of the task, and one the baselines suffer worse is a point in the metric's favour.

This scores identity discrimination (same-song pairs against style-swap pairs) separately
for cross-genre and same-genre swaps, and bootstraps the C-minus-CLAP gap and the
cross-to-same degradation, pairing on source track throughout.

Usage (DSP env, from the repo root):
    python scripts/analysis/samegenre_margin.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
N_BOOT = 10000


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a same-song pair scores above a swap pair), ties counted as half."""
    if not len(pos) or not len(neg):
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def main() -> None:
    rows = list(csv.DictReader((ROOT / "results/coherence/pair_scores_cbase.csv").open()))
    for r in rows:
        H, T, R, S = (float(r["score_" + k]) for k in "HTRS")
        r["C"] = (H * T * R * S) ** 0.25
        r["C_noS"] = (H * T * R) ** (1 / 3)
        r["CLAP"] = float(r["score_clap_htsat"])
        r["S"] = S

    same = [r for r in rows if r["perturbation"] != "style_swap"]
    swaps = {tag: [r for r in rows if r["perturbation"] == "style_swap"
                   and tag in r["pair_id"].rsplit("::", 1)[-1]]
             for tag in ("cross", "same")}

    metrics = ("C", "C_noS", "CLAP", "S")
    tracks = sorted({r["source_id"] for r in rows})
    by_track = {t: i for i, t in enumerate(tracks)}
    rng = np.random.default_rng(0)

    def arrays(pool, m):
        out = [[] for _ in tracks]
        for r in pool:
            out[by_track[r["source_id"]]].append(r[m])
        return [np.asarray(v, dtype=float) for v in out]

    point, boots = {}, {}
    for tag, neg in swaps.items():
        for m in metrics:
            P, N = arrays(same, m), arrays(neg, m)
            point[(tag, m)] = auc(np.concatenate(P), np.concatenate(N))
            draws = []
            for _ in range(N_BOOT):
                pick = rng.integers(0, len(tracks), len(tracks))
                p = np.concatenate([P[i] for i in pick if len(P[i])])
                n = np.concatenate([N[i] for i in pick if len(N[i])])
                draws.append(auc(p, n))
            boots[(tag, m)] = np.asarray(draws)

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    out = {"note": "Identity AUC (same-song vs style-swap) split by swap genre. "
                   "Cluster bootstrap over the 90 source tracks, paired across metrics "
                   "by reusing the same resample.",
           "n_boot": N_BOOT, "auc": {}, "contrasts": {}}
    for tag in swaps:
        out["auc"][tag] = {m: round(point[(tag, m)], 4) for m in metrics}
        for m in ("CLAP", "C_noS", "S"):
            d = boots[(tag, "C")] - boots[(tag, m)]
            out["contrasts"][f"{tag}: C - {m}"] = {
                "delta": round(point[(tag, "C")] - point[(tag, m)], 4),
                "ci95": ci(d), "excludes_zero": bool(np.percentile(d, 2.5) > 0
                                                     or np.percentile(d, 97.5) < 0)}
    for m in metrics:
        d = boots[("cross", m)] - boots[("same", m)]
        out["contrasts"][f"degradation cross->same: {m}"] = {
            "delta": round(point[("cross", m)] - point[("same", m)], 4),
            "ci95": ci(d), "excludes_zero": bool(np.percentile(d, 2.5) > 0
                                                 or np.percentile(d, 97.5) < 0)}
    d = (boots[("cross", "C")] - boots[("cross", "CLAP")]) - \
        (boots[("same", "C")] - boots[("same", "CLAP")])
    out["contrasts"]["is C's lead over CLAP larger on same-genre?"] = {
        "delta_of_deltas": round((point[("cross", "C")] - point[("cross", "CLAP")])
                                 - (point[("same", "C")] - point[("same", "CLAP")]), 4),
        "ci95": ci(d),
        "note": "negative means C's advantage is LARGER on the harder same-genre split"}

    dest = ROOT / "results/diagnostics/samegenre_margin.json"
    dest.write_text(json.dumps(out, indent=2))
    for tag in swaps:
        print(tag, out["auc"][tag])
    for k, v in out["contrasts"].items():
        print(f"  {k:44s} {v.get('delta', v.get('delta_of_deltas')):+.4f} {v['ci95']}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
