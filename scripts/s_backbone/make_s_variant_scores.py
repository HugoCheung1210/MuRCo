#!/usr/bin/env python3
"""T3.1 / T3.2 -- POST-PROCESSED variants of S, as a drop-in pair_scores.csv.

Both variants attack S's VARIANCE, not its signal, and neither touches S's
prompt (so §10's lock holds).  Rather than reimplementing the AUC / matrix
machinery, this writes a pair_scores.csv with `score_S` replaced by the variant,
which `compute_S_value.py --scores` and `compute_C.py --scores` then consume
unchanged:

  control_norm (T3.1)  S_norm(A,B) = S(A,B) − S(A, control(A)), per source_id.
      Removes each track's own S offset (some tracks just read as more coherent
      to MF than others), which is between-track variance that every drop/AUC
      statistic otherwise pays for.  NB this leaves [0,1] -- range is [−1,+1] --
      so it is for S-vs-baseline COMPARISONS only.  The composite C keeps raw S
      (a negative factor in a geometric mean is meaningless); the script refuses
      to pretend otherwise and says so in the output meta.

  ensemble (T3.2)  S_ens = mean(rank_MF, rank_Omni), rescaled to [0,1].
      Ordering is backbone-robust but CALIBRATION is not (§8.2: Omni is yes-shy,
      control 0.35 vs MF 0.80).  So averaging raw scores would just drag MF
      toward Omni's compressed scale; averaging RANKS keeps only the agreeing
      part of the ordering and cancels independent per-backbone noise.  Ranks
      are taken over the pair set the two caches share.

Usage (DSP env, from repo root):

    # T3.1
    python scripts/s_backbone/make_s_variant_scores.py --variant control_norm \
        --scores results/coherence/pair_scores.csv --out results/coherence/pair_scores_snorm.csv
    python scripts/core/compute_S_value.py --scores results/coherence/pair_scores_snorm.csv --by-genre

    # T3.2
    python scripts/s_backbone/make_s_variant_scores.py --variant ensemble \
        --scores results/coherence/pair_scores.csv \
        --s-cache results/s_cache/s_scores.json --s-cache-2 results/s_cache/s_scores_omni.json \
        --out results/coherence/pair_scores_sens.csv
    python scripts/core/compute_S_value.py --scores results/coherence/pair_scores_sens.csv --by-genre
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def read_scores(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def ranks01(vals: dict[str, float]) -> dict[str, float]:
    """Average-rank each pair_id's score, rescaled to [0,1] (1 = most coherent)."""
    ids = sorted(vals)
    a = np.array([vals[i] for i in ids], float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):                 # average ranks within ties
        if c > 1:
            r[inv == i] = r[inv == i].mean()
    r = (r - 1) / (len(a) - 1) if len(a) > 1 else np.zeros_like(r)
    return dict(zip(ids, r))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("control_norm", "ensemble", "swap"), required=True)
    ap.add_argument("--scores", type=Path, default=Path("results/coherence/pair_scores.csv"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--s-cache", type=Path, default=Path("results/s_cache/s_scores.json"),
                    help="ensemble: backbone 1 (MF)")
    ap.add_argument("--s-cache-2", type=Path, default=Path("results/s_cache/s_scores_omni.json"),
                    help="ensemble: backbone 2 (Omni)")
    args = ap.parse_args()

    rows = read_scores(args.scores)
    if not rows:
        print(f"no rows in {args.scores}", file=sys.stderr)
        return 1

    note = ""
    if args.variant == "swap":
        # replace score_S with a different S cache (e.g. s_scores_pt.json), keeping
        # H/T/R and all rows -- the fast full-matrix AUC re-derivation for an
        # adopted S variant, no re-scoring of the DSP dims needed.
        new = json.loads(args.s_cache.read_text(encoding="utf-8"))["scores"]
        kept = [r for r in rows if r["pair_id"] in new]
        if len(kept) < len(rows):
            print(f"dropping {len(rows) - len(kept)} rows absent from {args.s_cache.name}")
        for r in kept:
            r["score_S"] = f"{float(new[r['pair_id']]):.6f}"
        rows = kept
        note = f"score_S replaced from {args.s_cache} ({len(rows)} rows); H/T/R untouched."
        print(note)

    elif args.variant == "control_norm":
        ctrl = {r["source_id"]: float(r["score_S"])
                for r in rows if r["perturbation"] == "control"}
        missing = {r["source_id"] for r in rows} - set(ctrl)
        if missing:
            print(f"ERROR: {len(missing)} sources have no control pair (e.g. "
                  f"{sorted(missing)[:3]}); S_norm is undefined for them.", file=sys.stderr)
            return 1
        for r in rows:
            r["score_S"] = f"{float(r['score_S']) - ctrl[r['source_id']]:.6f}"
        note = ("S_norm = S - S(control of same source); range [-1,1], NOT [0,1]. "
                "Use for S-vs-baseline comparisons only -- the composite C must keep raw S.")
        lo = min(float(r["score_S"]) for r in rows)
        hi = max(float(r["score_S"]) for r in rows)
        print(f"control_norm: S_norm range [{lo:+.3f}, {hi:+.3f}] over {len(rows)} pairs "
              f"({len(ctrl)} sources)")

    else:
        c1 = json.loads(args.s_cache.read_text(encoding="utf-8"))["scores"]
        c2 = json.loads(args.s_cache_2.read_text(encoding="utf-8"))["scores"]
        shared = sorted(set(c1) & set(c2))
        if not shared:
            print("ERROR: the two S caches share no pair_ids", file=sys.stderr)
            return 1
        print(f"ensemble: {len(c1)} + {len(c2)} scores -> {len(shared)} shared pairs")
        r1 = ranks01({k: float(c1[k]) for k in shared})
        r2 = ranks01({k: float(c2[k]) for k in shared})
        ens = {k: (r1[k] + r2[k]) / 2.0 for k in shared}
        # agreement between the two backbones' orderings, for the write-up
        a = np.array([r1[k] for k in shared]); b = np.array([r2[k] for k in shared])
        print(f"rank correlation MF vs Omni (Spearman) = {np.corrcoef(a, b)[0, 1]:+.3f}")
        kept = [r for r in rows if r["pair_id"] in ens]
        if len(kept) < len(rows):
            print(f"dropping {len(rows) - len(kept)} rows absent from one cache")
        for r in kept:
            r["score_S"] = f"{ens[r['pair_id']]:.6f}"
        rows = kept
        note = ("S_ens = mean(rank_MF, rank_Omni) rescaled to [0,1]; a RANK score -- "
                "absolute values are not comparable to raw S, only orderings/AUCs are.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    args.out.with_suffix(".meta.json").write_text(json.dumps(
        {"variant": args.variant, "source_scores": str(args.scores),
         "s_cache": str(args.s_cache) if args.variant == "ensemble" else None,
         "s_cache_2": str(args.s_cache_2) if args.variant == "ensemble" else None,
         "n_rows": len(rows), "note": note}, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)} rows)\nNOTE: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
