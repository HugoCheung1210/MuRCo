#!/usr/bin/env python3
"""Compare the S SIGNATURE of two-or-more S caches on the T7.0 regression subset.

The Phase-2 go/no-go tool (`doc/notes/phase2_gpu_runbook.md` §2.1). Every S method
variant writes its own cache; a variant is adopt-able only if the separability
signature survives (T7.0):
  * style_swap stays the #1 mean drop,
  * time_stretch drop stays ~0 (< ~0.03) -- S's cleanest off-diagonal (§5/§10),
  * control level stays sane (~0.80 on MF).

This reads each cache's `scores`, restricts to the frozen subset, computes the
per-perturbation drop (control - perturbed, paired within source_id, mean +
Cohen's d_z), and prints the caches side by side with a PASS/FAIL verdict per
variant against the first cache as the reference. Pure post-processing, CPU,
local -- run it on the caches after you pull them off the box.

Usage:
    python scripts/s_backbone/compare_subset_signature.py \
        results/s_cache/s_scores.json results/s_cache/s_subset_default.json [more.json ...] \
        [--manifest perturbations/pairs_manifest.json] \
        [--subset results/s_cache/s_method_subset.json]

The first cache is the REFERENCE (typically results/s_cache/s_scores.json, the shipped
one). A cache that only holds the 198 subset pairs is fine; so is a full-1890
cache -- it is filtered to the subset either way, so signatures are comparable.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# perturbation families whose min/max magnitudes are in the subset
PERTURBATIONS = ["style_swap", "distortion", "pitch_shift", "lowpass", "time_stretch"]
TIME_STRETCH_MAX = 0.03      # T7.0: time_stretch drop must stay below this


def load_scores(path: Path) -> dict[str, float]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {k: float(v) for k, v in doc.get("scores", {}).items()}


def signature(scores: dict[str, float], meta: dict[str, dict]) -> dict:
    """Per-perturbation mean drop + d_z (control - perturbed, paired by source)."""
    ctrl = {meta[p]["source_id"]: s for p, s in scores.items()
            if p in meta and meta[p]["perturbation"] == "control"}
    drops: dict[str, list] = defaultdict(list)
    for pid, s in scores.items():
        m = meta.get(pid)
        if not m or m["perturbation"] == "control":
            continue
        sid = m["source_id"]
        if sid in ctrl:
            drops[m["perturbation"]].append(ctrl[sid] - s)
    sig = {"control_level": float(np.mean(list(ctrl.values()))) if ctrl else float("nan"),
           "n_control": len(ctrl), "drops": {}}
    for pert, vals in drops.items():
        a = np.array(vals, float)
        sig["drops"][pert] = {"mean": float(a.mean()),
                              "dz": float(a.mean() / (a.std(ddof=1) + 1e-12)) if a.size > 1 else float("nan"),
                              "n": int(a.size)}
    return sig


def verdict(sig: dict) -> tuple[bool, str]:
    """T7.0 acceptance: style_swap #1, time_stretch < 0.03, control sane."""
    drops = sig["drops"]
    if not drops:
        return False, "no drops computed"
    ranked = sorted(drops.items(), key=lambda kv: -kv[1]["mean"])
    top = ranked[0][0]
    ts = drops.get("time_stretch", {}).get("mean", float("nan"))
    reasons = []
    ok = True
    if top != "style_swap":
        ok = False
        reasons.append(f"top drop is {top}, not style_swap")
    if not (ts < TIME_STRETCH_MAX):
        ok = False
        reasons.append(f"time_stretch drop {ts:+.3f} >= {TIME_STRETCH_MAX}")
    if not (0.5 < sig["control_level"] < 0.95):
        reasons.append(f"control level {sig['control_level']:.3f} unusual (want ~0.80)")
    return ok, "; ".join(reasons) if reasons else "style_swap #1, time_stretch ~0, control sane"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("caches", nargs="+", type=Path,
                    help="S cache JSONs; the FIRST is the reference")
    ap.add_argument("--manifest", type=Path, default=Path("perturbations/pairs_manifest.json"))
    ap.add_argument("--subset", type=Path, default=Path("results/s_cache/s_method_subset.json"))
    ap.add_argument("--all", action="store_true",
                    help="compare on ALL manifest pairs, not just the subset (use for a "
                         "full-1890 old-vs-new comparison, e.g. s_scores.json vs s_scores_pt.json)")
    args = ap.parse_args()

    pairs = json.loads(args.manifest.read_text(encoding="utf-8"))["pairs"]
    meta = {p["pair_id"]: {"source_id": p["source_id"], "perturbation": p["perturbation"]}
            for p in pairs}
    if args.all:
        sub = set(meta)                                    # every pair
    else:
        sub = set(json.loads(args.subset.read_text(encoding="utf-8"))["pair_ids"])
        meta = {k: v for k, v in meta.items() if k in sub}  # restrict to the subset

    sigs, labels = [], []
    for c in args.caches:
        if not c.is_file():
            print(f"missing cache: {c}", file=sys.stderr)
            return 1
        sc = {k: v for k, v in load_scores(c).items() if k in sub}
        if not sc:
            print(f"WARNING: {c.name} has 0 of the {len(sub)} subset pairs", file=sys.stderr)
        sigs.append(signature(sc, meta))
        labels.append(c.stem)

    n_cov = [sum(1 for pid in sub if pid in load_scores(c)) for c in args.caches]
    print(f"# Subset signature comparison ({len(sub)} pairs)\n")
    print("coverage: " + ", ".join(f"{l}={n}/{len(sub)}" for l, n in zip(labels, n_cov)) + "\n")

    # drop table, perturbations x caches (mean with d_z), ranked by the reference
    ref = sigs[0]["drops"]
    order = sorted(PERTURBATIONS, key=lambda p: -ref.get(p, {}).get("mean", -9))
    w = max(12, max(len(l) for l in labels) + 2)
    print("| perturbation | " + " | ".join(f"{l}" for l in labels) + " |")
    print("|---|" + "---|" * len(labels))
    for pert in order:
        cells = []
        for sig in sigs:
            d = sig["drops"].get(pert)
            cells.append(f"{d['mean']:+.3f} (dz {d['dz']:+.2f})" if d else "-")
        print(f"| {pert} | " + " | ".join(cells) + " |")
    print("| _control level_ | " + " | ".join(f"{s['control_level']:.3f}" for s in sigs) + " |")

    print("\n## T7.0 verdict (vs first cache as reference)\n")
    for label, sig in zip(labels, sigs):
        ok, why = verdict(sig)
        print(f"- **{label}**: {'PASS' if ok else 'FAIL'} — {why}")

    # explicit style_swap delta vs reference (the clip-artifact / variant question)
    if len(sigs) > 1 and "style_swap" in ref:
        print("\n## style_swap drop vs reference (Δ = variant − reference)\n")
        base = ref["style_swap"]["mean"]
        for label, sig in list(zip(labels, sigs))[1:]:
            d = sig["drops"].get("style_swap")
            if d:
                print(f"- {label}: {d['mean']:+.3f}  (Δ {d['mean'] - base:+.3f} vs {labels[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
