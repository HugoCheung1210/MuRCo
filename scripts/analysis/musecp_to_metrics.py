#!/usr/bin/env python
"""Convert a MuseCPEval summary.csv into this repo's per-pair metric contracts.

MuseCPEval (Vishe et al., ISMIR 2026) reports 12 metrics over five facets, and TWO of
them rise with change rather than with preservation.  Every downstream analysis needs all
twelve pointing the same way, so the flip happens here, once, and never by hand.

Emits two files:

  --csv   pair_id + 12 aligned metric columns + 2 equal-weight composites.  This is the
          format ``mos_delta_r.py --rivals`` and ``compute_S_value.py --baselines`` read
          (every non-metadata column is a metric).
  --json  the score_baselines.py contract, {"meta": {...}, "scores": {pair_id: {...}}},
          which compute_C.py --baselines reads.  Values are clipped to [1e-3, 1] because
          that path takes logs.

The composites are MuseCPEval's own metrics given an aggregate it does not define: each
aligned metric is min-max scaled over the file, then averaged.  ``mean12`` is all five
facets, ``mean10`` drops the timbre facet that v2 added in its appendix.  Both are our
construction, not theirs, and must be reported as such.

Usage (DSP env, from the repo root)::

    python scripts/analysis/musecp_to_metrics.py \
        --summary results/musecp/setup1/summary.csv \
        --csv     results/musecp/setup1/musecp_metrics.csv \
        --json    results/musecp/setup1/musecp_metrics.json
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


def repo_root() -> Path:
    for p in [_here] + list(_here.parents):
        if (p / "CLAUDE.md").exists() or (p / ".git").exists():
            return p
    return _here.parent.parent.parent


# column -> (short name, higher_is_preserved).  The two False rows are the flips.
METRICS = [
    ("harmony_tonality.key_relatedness.distance_norm_0to1", "musecp_key", False),
    ("harmony_tonality.chroma_similarity.mean_chroma_cosine", "musecp_chroma_mean", True),
    ("harmony_tonality.chroma_similarity.chroma_dtw_cosine", "musecp_chroma_dtw", True),
    ("rhythm_meter.delta_bpm_folded", "musecp_bpm", False),
    ("rhythm_meter.beat_mir_eval.F-measure", "musecp_beat_f", True),
    ("rhythm_meter.beat_mir_eval.Information gain", "musecp_beat_ig", True),
    ("structural_form.pairwise_f", "musecp_struct_f", True),
    ("structural_form.ari", "musecp_ari", True),
    ("melodic_content.contour_dtw_similarity", "musecp_contour", True),
    ("melodic_content.motif_3gram_recall", "musecp_motif", True),
    ("timbre_texture.mfcc_skl_similarity", "musecp_mfcc_skl", True),
    ("timbre_texture.mean_mfcc_cosine", "musecp_mfcc_cos", True),
]
TIMBRE = {"musecp_mfcc_skl", "musecp_mfcc_cos"}
SHORTS = [s for _, s, _ in METRICS]


def align(col: str, short: str, preserved: bool, vals: np.ndarray) -> np.ndarray:
    """Point a metric so that HIGHER always means MORE PRESERVED."""
    if preserved:
        return vals
    if short == "musecp_key":
        return 1.0 - vals                  # already normalised to [0,1]
    if short == "musecp_bpm":
        return np.exp(-np.abs(vals) / 10.0)  # unbounded BPM difference -> (0,1]
    raise SystemExit(f"no flip rule for {col}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary.csv from musecpeval")
    ap.add_argument("--csv", required=True, help="write the per-pair metric CSV here")
    ap.add_argument("--json", required=True, help="write the baselines JSON here")
    args = ap.parse_args()

    root = repo_root()
    rows = list(csv.DictReader(open(args.summary)))
    if not rows:
        raise SystemExit(f"{args.summary} is empty")
    print(f"read {len(rows)} rows from {args.summary}")

    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate pair ids in the summary")

    raw, aligned, n_missing = {}, {}, {}
    for col, short, preserved in METRICS:
        if col not in rows[0]:
            raise SystemExit(f"column {col!r} absent from the summary")
        v = np.array([float(r[col]) if r[col] not in ("", None) else np.nan
                      for r in rows], dtype=float)
        n_missing[short] = int(np.isnan(v).sum())
        raw[short] = v
        aligned[short] = align(col, short, preserved, v)

    # composites: min-max each aligned metric over THIS file, then average.
    scaled = {}
    for short in SHORTS:
        v = aligned[short]
        lo, hi = np.nanmin(v), np.nanmax(v)
        scaled[short] = np.full_like(v, 0.5) if hi - lo < 1e-12 else (v - lo) / (hi - lo)
    stack12 = np.vstack([scaled[s] for s in SHORTS])
    stack10 = np.vstack([scaled[s] for s in SHORTS if s not in TIMBRE])
    with np.errstate(invalid="ignore"):
        mean12 = np.nanmean(stack12, axis=0)
        mean10 = np.nanmean(stack10, axis=0)

    out_cols = SHORTS + ["musecp_mean12", "musecp_mean10"]
    table = {**aligned, "musecp_mean12": mean12, "musecp_mean10": mean10}

    csv_path = Path(args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair_id"] + out_cols)
        for i, pid in enumerate(ids):
            w.writerow([pid] + ["" if not np.isfinite(table[c][i]) else f"{table[c][i]:.6f}"
                                for c in out_cols])
    print(f"wrote {csv_path.relative_to(root)}  ({len(ids)} rows x {len(out_cols)} metrics)")

    # JSON contract for compute_C.py --baselines: strictly positive, it takes logs.
    scores = {}
    for i, pid in enumerate(ids):
        per = {}
        for c in out_cols:
            v = table[c][i]
            if np.isfinite(v):
                per[c] = float(np.clip(v, 1e-3, 1.0))
        scores[pid] = per
    json_path = Path(args.json).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "meta": {"baselines": out_cols, "source": str(args.summary),
                 "n_pairs": len(ids), "n_missing_per_metric": n_missing,
                 "note": "MuseCPEval 0.3.0 metrics, polarity-aligned so higher = more "
                         "preserved; key and bpm flipped. Values clipped to [1e-3,1] for "
                         "the log-linear baseline path. mean12/mean10 are OUR equal-weight "
                         "composites, min-max scaled within this file; MuseCPEval defines "
                         "no aggregate."},
        "scores": scores}, indent=1))
    print(f"wrote {json_path.relative_to(root)}")

    bad = {k: v for k, v in n_missing.items() if v}
    print("missing values per metric:", bad if bad else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
