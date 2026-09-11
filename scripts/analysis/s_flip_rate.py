#!/usr/bin/env python3
"""How often does the semantic term change which candidate gets picked, and by how much?

Section 5's forced-choice arm reports how often listeners side with $C$'s pick over
$C_{noS}$'s. That rate is conditional on the two metrics disagreeing, and the trial set
was selected for opposed ordering AND an audible margin, so its size reflects the
selector's supply rather than any rate. The frequency question is therefore separate and
is answered here, on the full re-ranking pool with no selection applied.

For each seed we take best-of-8 under C and under C_noS and ask whether the argmax moves.
Where it moves we record the C-gap between the two picks, which is the quantity the
|dC| ~ 0.025 audibility floor is expressed in. That floor comes from the blinded
self-pilot and is used only to place the threshold, never as evidence of preference.

NB this is a different construction from the trial builder, which searched opposed-ordering
candidate PAIRS rather than pool argmaxes. Read it as an independent estimate of the flip
rate, not as a description of how the trials were made.

Usage (DSP env, from the repo root):
    python scripts/analysis/s_flip_rate.py
"""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

AUDIBLE = 0.025
ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)


def main() -> None:
    rows = list(csv.DictReader((ROOT / "results/rerank_ace/pair_scores.csv").open()))
    pools: dict[str, list] = defaultdict(list)
    for r in rows:
        H, T, R, S = (float(r["score_" + k]) for k in "HTRS")
        pools[r["source_id"]].append((r["pair_id"], (H * T * R * S) ** 0.25,
                                      (H * T * R) ** (1 / 3)))

    flips, gaps = [], []
    for seed, cands in sorted(pools.items()):
        best_c = max(cands, key=lambda x: x[1])
        best_n = max(cands, key=lambda x: x[2])
        moved = best_c[0] != best_n[0]
        flips.append(moved)
        if moved:
            c_of = {p: c for p, c, _ in cands}
            gaps.append(c_of[best_c[0]] - c_of[best_n[0]])

    n = len(flips)
    n_flip = sum(flips)
    rng = np.random.default_rng(7)
    boot = [float(np.mean(rng.choice(flips, size=n, replace=True))) for _ in range(10000)]
    audible = [g for g in gaps if g >= AUDIBLE]

    out = {
        "note": "Pool-argmax flip rate for S over the re-ranking candidate pools. "
                "Independent of the Part 2 trial builder, which used opposed-ordering "
                "pairs rather than argmaxes.",
        "n_seeds": n,
        "pool_size": len(rows) // n,
        "n_flips": n_flip,
        "flip_rate": round(n_flip / n, 4),
        "flip_rate_ci95": [round(float(np.percentile(boot, 2.5)), 4),
                           round(float(np.percentile(boot, 97.5)), 4)],
        "gap_when_flipped": {
            "median": round(statistics.median(gaps), 4),
            "min": round(min(gaps), 4),
            "max": round(max(gaps), 4),
        },
        "audibility_floor": AUDIBLE,
        "n_audible_flips": len(audible),
        "share_of_flips_audible": round(len(audible) / len(gaps), 4),
        "share_of_seeds_with_audible_flip": round(len(audible) / n, 4),
    }
    dest = ROOT / "results/diagnostics/s_flip_rate.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"{n} seeds, pool of {out['pool_size']}")
    print(f"S moves the best-of-{out['pool_size']} pick in {n_flip}/{n} "
          f"= {out['flip_rate']:.1%}  CI {out['flip_rate_ci95']}")
    print(f"C-gap when it moves: median {out['gap_when_flipped']['median']:.4f} "
          f"(min {out['gap_when_flipped']['min']:.4f}, "
          f"max {out['gap_when_flipped']['max']:.4f})")
    print(f"clearing the {AUDIBLE} floor: {len(audible)}/{len(gaps)} of flips "
          f"= {out['share_of_flips_audible']:.0%}, "
          f"{out['share_of_seeds_with_audible_flip']:.1%} of all seeds")
    print("wrote", dest)


if __name__ == "__main__":
    main()
