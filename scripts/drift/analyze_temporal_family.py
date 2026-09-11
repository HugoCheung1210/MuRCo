#!/usr/bin/env python3
"""T1.4 -- analysis of the temporal-violation family (shuffle / reverse / displacement).

The T1 claim: S measures musical CONTINUITY; CLAP measures material IDENTITY.
These pairs are (near-)identical acoustic MATERIAL to A in a broken temporal
relation, so the mechanism predicts:
  * clap_htsat ~ control  (time-pooled embedding -> order-invariant)
  * S drops               (reads the actual A->B transition)
  * humans would rate shuffle/reverse incoherent, displacement moderately so.

Deliverables (spec T1.4):
  1. Drop table: control - perturbed, mean + Cohen's d_z, per perturbation x
     {H,T,R,S,clap_htsat,clap_music,scs,fad_proxy} (whichever are present).
  2. AUC "control vs temporal-violation" per metric.
  3. Ordering check: control > displacement > shuffle ~ reverse under S.

Runs partially with only H/T/R present (before the GPU session) and fully once
results/s_cache/s_scores_temporal.json and results/baseline_scores_temporal*.json exist.
Pass --s-cache / --baselines / --baselines-projected to add those columns.

Usage:
    python scripts/drift/analyze_temporal_family.py \
        --scores results/temporal/pair_scores.csv \
        --s-cache results/s_cache/s_scores_temporal.json \
        --baselines results/baselines/baseline_scores_temporal.json \
        --baselines-projected results/baselines/baseline_scores_temporal_projected.json \
        --out results/temporal/analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

VIOLATION_TAGS = ["shuf_beat", "shuf_bar", "rev", "disp_near", "disp_far"]
SHUFFLE_REVERSE = {"shuf_beat", "shuf_bar", "rev"}      # order fully destroyed
DISPLACEMENT = {"disp_near", "disp_far"}                # graded: same piece, wrong place


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """P(score[pos] > score[neg]); higher score should mean label 1."""
    pos, neg = score[y == 1], score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties
    uniq, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    r_pos = ranks[y == 1].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def dz(paired_drops: np.ndarray) -> float:
    """Cohen's d_z for a paired drop vector (mean / sd)."""
    d = paired_drops[~np.isnan(paired_drops)]
    return float(d.mean() / (d.std(ddof=1) + 1e-12)) if d.size > 1 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, default=Path("results/temporal/pair_scores.csv"),
                    help="H/T/R(/S) per-pair scores for the temporal manifest")
    ap.add_argument("--s-cache", type=Path, default=None,
                    help="results/s_cache/s_scores_temporal.json (adds/overrides score_S)")
    ap.add_argument("--baselines", type=Path, default=None,
                    help="results/baselines/baseline_scores_temporal.json (pooled CLAP)")
    ap.add_argument("--baselines-projected", type=Path, default=None,
                    help="results/baselines/baseline_scores_temporal_projected.json")
    ap.add_argument("--out", type=Path, default=Path("results/temporal/analysis.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(args.scores.open(encoding="utf-8")))
    if not rows:
        print(f"no rows in {args.scores}", file=sys.stderr)
        return 1

    # per-pair metric table: {pair_id: {metric: value}} keyed off the CSV, then
    # augmented from the caches.
    meta = {r["pair_id"]: {"source_id": r["source_id"], "genre": r["genre"],
                           "perturbation": r["perturbation"],
                           "tag": r["pair_id"].rsplit("::", 1)[1]} for r in rows}
    vals: dict[str, dict[str, float]] = {r["pair_id"]: {} for r in rows}
    dim_cols = [c for c in rows[0] if c.startswith("score_")]
    for r in rows:
        for c in dim_cols:
            if r[c] not in ("", "nan"):
                vals[r["pair_id"]][c[len("score_"):]] = float(r[c])

    if args.s_cache and args.s_cache.is_file():
        sc = json.loads(args.s_cache.read_text())["scores"]
        for pid, v in sc.items():
            if pid in vals:
                vals[pid]["S"] = float(v)
    for bpath, suffix in ((args.baselines, ""), (args.baselines_projected, "_proj")):
        if bpath and bpath.is_file():
            bs = json.loads(bpath.read_text())["scores"]
            for pid, d in bs.items():
                if pid in vals:
                    for m, x in d.items():
                        vals[pid][m + suffix] = float(x)

    metrics = sorted({m for d in vals.values() for m in d})
    print(f"# T1.4 temporal-violation analysis\n\nmetrics present: {metrics}\n")

    # controls, per source, per metric
    ctrl = {m: {} for m in metrics}
    for pid, mv in vals.items():
        if meta[pid]["perturbation"] == "control":
            for m, x in mv.items():
                ctrl[m][meta[pid]["source_id"]] = x

    # ---- 1. drop table (mean + d_z) ----
    print("## 1. Drop table  (control - perturbed; mean, and d_z in parens)\n")
    header = "| tag | " + " | ".join(metrics) + " |"
    print(header)
    print("|" + "---|" * (len(metrics) + 1))
    drop_table: dict = {}
    for tag in VIOLATION_TAGS:
        cells, rec = [], {}
        for m in metrics:
            paired = []
            for pid, mv in vals.items():
                if meta[pid]["tag"] != tag or m not in mv:
                    continue
                sid = meta[pid]["source_id"]
                if sid in ctrl[m]:
                    paired.append(ctrl[m][sid] - mv[m])
            if paired:
                arr = np.array(paired, float)
                rec[m] = {"mean_drop": float(np.nanmean(arr)), "dz": dz(arr), "n": int(arr.size)}
                cells.append(f"{rec[m]['mean_drop']:+.3f} ({rec[m]['dz']:+.2f})")
            else:
                cells.append("-")
        drop_table[tag] = rec
        print(f"| {tag} | " + " | ".join(cells) + " |")

    # ---- 2. control-vs-violation AUC per metric ----
    print("\n## 2. AUC: control vs temporal-violation  (1.0 = metric ranks controls "
          "above violations; 0.5 = blind)\n")
    print("| metric | AUC | reading |")
    print("|---|---:|---|")
    auc_rec = {}
    for m in metrics:
        sv, yv = [], []
        for pid, mv in vals.items():
            if m not in mv:
                continue
            tag = meta[pid]["tag"]
            if meta[pid]["perturbation"] == "control":
                sv.append(mv[m]); yv.append(1)
            elif tag in VIOLATION_TAGS:
                sv.append(mv[m]); yv.append(0)
        a = auc(np.array(sv, float), np.array(yv))
        auc_rec[m] = a
        note = ("blind (mechanism prediction for CLAP)" if a < 0.6
                else "sensitive" if a > 0.8 else "partial")
        print(f"| {m} | {a:.3f} | {note} |")

    # ---- 3. ordering check under S ----
    print("\n## 3. Ordering check  (expected: control > displacement > shuffle ~ reverse)\n")
    order_rec = {}
    for m in metrics:
        levels = {}
        # control level
        cvals = list(ctrl[m].values())
        if cvals:
            levels["control"] = float(np.mean(cvals))
        for group, tags in (("displacement", DISPLACEMENT), ("shuffle_reverse", SHUFFLE_REVERSE)):
            gv = [mv[m] for pid, mv in vals.items()
                  if meta[pid]["tag"] in tags and m in mv]
            if gv:
                levels[group] = float(np.mean(gv))
        order_rec[m] = levels
        if m in ("S", "clap_htsat"):
            order = " > ".join(f"{k}={v:.3f}" for k, v in
                               sorted(levels.items(), key=lambda kv: -kv[1]))
            ok = (levels.get("control", 0) >= levels.get("displacement", 0)
                  >= levels.get("shuffle_reverse", 0))
            print(f"- **{m}**: {order}  {'[matches expected ordering]' if ok else '[DOES NOT match]'}")

    out = {"metrics": metrics, "drop_table": drop_table, "control_vs_violation_auc": auc_rec,
           "level_ordering": order_rec,
           "note": "clap_* AUC ~0.5 confirms the time-pooling blind spot (T1 mechanism); "
                   "S AUC high with clap low is the headline. Report whatever the numbers say."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    if "S" not in metrics or not any(m.startswith("clap") for m in metrics):
        print("\n_Partial run: S and/or CLAP columns absent. Re-run after the GPU session "
              "with --s-cache results/s_cache/s_scores_temporal.json and "
              "--baselines results/baselines/baseline_scores_temporal.json for the decisive rows._")
    return 0


if __name__ == "__main__":
    sys.exit(main())
