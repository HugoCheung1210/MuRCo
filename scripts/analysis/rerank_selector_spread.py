#!/usr/bin/env python3
"""Is a rival metric actually choosing, or is its argmax arbitrary?

Section 5's forced-choice arms are read as if a lower win rate against a rival meant a
harder contest. That reading needs to know whether the rival is ranking at all. A metric
whose eight candidates in a pool differ by less than its own noise picks a near-arbitrary
one, and beating an arbitrary pick is a different task from beating a considered one.

For each of the 40 seeds this reports the within-pool spread of every selector over its 8
candidates, and the Spearman correlation between the rival's ordering of the pool and C's.
A spread near zero plus a correlation near zero is what "effectively random" looks like.

Usage (DSP env, from the repo root):
    python scripts/analysis/rerank_selector_spread.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    base = json.load(open(ROOT / "results/rerank_ace/baseline_scores_rerank.json"))["scores"]
    rows = list(csv.DictReader((ROOT / "results/rerank_ace/pair_scores.csv").open()))

    pools = defaultdict(dict)
    for r in rows:
        H, T, R, S = (float(r["score_" + k]) for k in "HTRS")
        pid = r["pair_id"]
        e = {"C": (H * T * R * S) ** 0.25, "C_noS": (H * T * R) ** (1 / 3)}
        e.update({k: float(v) for k, v in base.get(pid, {}).items()})
        pools[r["source_id"]][pid] = e

    sel = ["C", "C_noS", "clap_htsat", "clap_music", "scs"]
    spread, rho, argmax_agree = defaultdict(list), defaultdict(list), defaultdict(list)
    for seed, cands in pools.items():
        pids = sorted(cands)
        if len(pids) < 8:
            continue
        vec = {m: [cands[p][m] for p in pids] for m in sel if m in cands[pids[0]]}
        for m, v in vec.items():
            rng = max(v) - min(v)
            spread[m].append(rng / (abs(st.mean(v)) + 1e-12))   # relative range
            if m != "C":
                rho[m].append(spearman(vec["C"], v))
                argmax_agree[m].append(int(np.argmax(v) == np.argmax(vec["C"])))

    out = {"n_seeds": len(spread["C"]), "pool": 8, "selectors": {}}
    print(f"{'selector':12s} {'rel. range':>11} {'rho vs C':>9} {'argmax = C':>11}")
    for m in sel:
        if m not in spread:
            continue
        e = {"relative_range_median": round(st.median(spread[m]), 4)}
        if m in rho:
            e["spearman_vs_C_mean"] = round(st.mean(rho[m]), 4)
            e["argmax_agrees_with_C"] = round(st.mean(argmax_agree[m]), 4)
        out["selectors"][m] = e
        print(f"{m:12s} {e['relative_range_median']:11.4f} "
              f"{e.get('spearman_vs_C_mean', float('nan')):9.3f} "
              f"{e.get('argmax_agrees_with_C', float('nan')):11.2f}")

    dest = ROOT / "results/diagnostics/rerank_selector_spread.json"
    dest.write_text(json.dumps(out, indent=2))
    print("\nrelative range = (max-min)/mean over the 8 candidates, median across seeds")
    print("wrote", dest)


if __name__ == "__main__":
    main()
