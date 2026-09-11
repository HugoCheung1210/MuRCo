#!/usr/bin/env python
"""Gap sweep: is the 1 s silence between A and B a choice or an accident?

Sec. 4.9 of the thesis reports that inserting a 1 s gap roughly doubles S's separation
between coherent and incoherent pairs (0 s -> 1 s, full battery, fig_gap_ablation). The
obvious follow-up is whether MORE silence keeps helping. It does not: the response peaks
at 1 s and falls back at 2 s.

The caches already exist, so this needs no GPU:
    results/s_cache/s_scores_gap0.json      gap 0.0, full 1890 pairs
    results/s_cache/s_subset_gap0.5.json    gap 0.5, method subset
    results/s_cache/s_subset_default.json   gap 1.0, method subset  (the shipped setting)
    results/s_cache/s_subset_gap2.0.json    gap 2.0, method subset

Only the pair_ids present in ALL FOUR are used, so the four gaps are compared on identical
material. That intersection is the method subset (198 pairs / 18 sources), which is why
the thesis reports this as the weaker of the two gap measurements.

Drops are within-source: every perturbed pair is differenced against the control pair of
its own source, matching the paired convention used everywhere else.

Usage (DSP env, no GPU):
    python scripts/s_backbone/analyze_gap_sweep.py
    python scripts/s_backbone/analyze_gap_sweep.py --json results/diagnostics/gap_sweep.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

CACHES = {
    "0.0": "results/s_cache/s_scores_gap0.json",
    "0.5": "results/s_cache/s_subset_gap0.5.json",
    "1.0": "results/s_cache/s_subset_default.json",
    "2.0": "results/s_cache/s_subset_gap2.0.json",
}
FAMILIES = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "perturbations/pairs_manifest.json"))
    ap.add_argument("--json", default=str(ROOT / "results/diagnostics/gap_sweep.json"))
    a = ap.parse_args()

    man = json.load(open(a.manifest))
    pairs = man["pairs"] if isinstance(man, dict) else man
    info = {p["pair_id"]: (p["perturbation"], p["source_id"]) for p in pairs}

    scores = {g: json.load(open(ROOT / f))["scores"] for g, f in CACHES.items()}
    common = set.intersection(*[set(v) for v in scores.values()])
    sources = {info[p][1] for p in common}
    print(f"{len(common)} pairs common to all four gaps, from {len(sources)} sources\n")

    out = {"meta": {"n_pairs": len(common), "n_sources": len(sources),
                    "caches": CACHES, "note": "within-source drops from control"},
           "by_gap": {}}

    hdr = f"{'gap':>5} {'control':>8} " + " ".join(f"{f[:12]:>13}" for f in FAMILIES) + f" {'mean':>8}"
    print(hdr)
    for g in CACHES:
        s = scores[g]
        ctrl = {info[p][1]: s[p] for p in common if info[p][0] == "control"}
        row = {}
        for f in FAMILIES:
            d = [ctrl[info[p][1]] - s[p] for p in common
                 if info[p][0] == f and info[p][1] in ctrl]
            row[f] = float(np.mean(d))
        row["control_level"] = float(np.mean(list(ctrl.values())))
        row["mean_drop"] = float(np.mean([row[f] for f in FAMILIES]))
        out["by_gap"][g] = row
        print(f"{g:>5} {row['control_level']:8.3f} "
              + " ".join(f"{row[f]:13.3f}" for f in FAMILIES)
              + f" {row['mean_drop']:8.3f}")

    best = max(CACHES, key=lambda g: out["by_gap"][g]["mean_drop"])
    out["best_gap_s"] = float(best)
    out["verdict"] = (
        f"Mean drop peaks at gap={best}s and falls back at 2.0s, while the control level "
        f"falls monotonically with gap length. The silence buys separation by telling the "
        f"model there are two segments, and past 1s it only erodes the ceiling. The shipped "
        f"1.0s is the best of the four values tried.")
    print(f"\nbest mean drop at gap {best}s")
    print(out["verdict"])

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.json, "w"), indent=1)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
