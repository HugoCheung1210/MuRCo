#!/usr/bin/env python3
"""Does including S make the metric better? Identity-discrimination AUC test.

The perturbation ablations (compute_C.py --weights ...S=0) do NOT prove S earns
its place -- H/T/R already detect that *an* edit happened, so C stays sensitive
without S.  S's unique job is narrower: distinguishing edits that PRESERVE
musical identity (same song, e.g. transposed) from edits that REPLACE it
(different song).  Signal-level H is blind to that distinction -- a transpose and
a style-swap both drop H by ~0.15.  This script quantifies whether S (and C with
S) resolve it.

Framing: a binary discrimination task, "is B the same piece as A?"
  positive (same identity)  = control + DSP perturbations (transpose, stretch,
                              lowpass, distortion) -- all keep the same song
  negative (different)      = style_swap -- a different song

For each feature (each dimension's score, plus signal-only C=geomean(H,T,R) and
full C=geomean(H,T,R,S)) we report AUC = P(same-identity scores higher than
different).  The claim "S helps" is credible iff AUC(C_full) > AUC(C_signal) with
a bootstrap CI excluding 0 -- and iff S alone beats H alone on the hardest
sub-task, transpose-vs-style-swap.

Usage::

    python compute_S_value.py --scores results/coherence/pair_scores.csv --by-genre
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

DIMS = ["H", "T", "R", "S"]
SAME = {"control", "pitch_shift", "time_stretch", "lowpass", "distortion"}
DIFF = {"style_swap"}
BASELINE_METRICS: list[str] = []       # populated by attach_baselines() if --baselines given


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC that higher score => label 1."""
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(np.argsort(score)) + 1          # ranks 1..N, ties ~ok at scale
    return (order[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def geomean(rows: list[dict], dims: list[str]) -> np.ndarray:
    eps = 1e-6
    m = np.array([[max(eps, r[f"score_{d}"]) for d in dims] for r in rows])
    return np.exp(np.log(m).mean(axis=1))


def features(rows: list[dict]) -> dict[str, np.ndarray]:
    f = {d: np.array([r[f"score_{d}"] for r in rows]) for d in DIMS}
    f["C_signal(HTR)"] = geomean(rows, ["H", "T", "R"])
    f["C_full(HTRS)"] = geomean(rows, ["H", "T", "R", "S"])
    for m in BASELINE_METRICS:                          # off-the-shelf baselines (--baselines)
        f[f"base:{m}"] = np.array([r.get("base_scores", {}).get(m, np.nan) for r in rows])
    return f


def run_task(rows: list[dict], pos: set[str], neg: set[str], title: str,
             n_boot: int = 2000, seed: int = 0, neg_tag: str | None = None) -> dict:
    """AUC discrimination pos (same identity) vs neg (different).

    neg_tag: if set, restrict the negative class to style_swap pairs whose tag
    contains it ('cross' or 'same'), so we can separate the easy cross-genre swap
    from the hard same-genre swap where surface (timbre/rhythm) is preserved.
    """
    def is_neg(r: dict) -> bool:
        if r["perturbation"] not in neg:
            return False
        return neg_tag is None or neg_tag in r["pair_id"].rsplit("::", 1)[-1]
    sel = [r for r in rows if (r["perturbation"] in pos) or is_neg(r)]
    y = np.array([1 if r["perturbation"] in pos else 0 for r in sel])
    if int((1 - y).sum()) < 2 or int(y.sum()) < 2:
        print(f"\n=== {title} ===\n  (too few pairs for this split, skipped)")
        return {}
    feats = features(sel)
    print(f"\n=== {title}  (n_same={int(y.sum())}, n_diff={int((1-y).sum())}) ===")
    print(f"{'feature':16s} {'AUC':>6s}   (1.0 = perfect same/different separation, 0.5 = chance)")
    aucs = {}
    for name, s in feats.items():
        if np.isnan(s).any():                           # a baseline missing on some pairs
            aucs[name] = float("nan")
            print(f"{name:16s} {'n/a':>6s}   (missing on {int(np.isnan(s).sum())} pairs)")
            continue
        aucs[name] = auc(s, y)
        print(f"{name:16s} {aucs[name]:6.3f}")

    rng = np.random.default_rng(seed)
    cs, cf = feats["C_signal(HTR)"], feats["C_full(HTRS)"]
    base = auc(cf, y) - auc(cs, y)
    deltas = np.empty(n_boot)
    idx = np.arange(len(y))
    for i in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        deltas[i] = auc(cf[b], y[b]) - auc(cs[b], y[b])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    verdict = "S ADDS VALUE" if lo > 0 else ("S HURTS" if hi < 0 else "inconclusive")
    print(f"\n  delta-AUC (C_full - C_signal) = {base:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}")
    print(f"  S alone AUC = {aucs['S']:.3f}  vs  H alone AUC = {aucs['H']:.3f}  "
          f"(S should win where H is blind to identity)")
    return {"title": title, "n_same": int(y.sum()), "n_diff": int((1 - y).sum()),
            "auc": {k: (None if np.isnan(v) else round(v, 3)) for k, v in aucs.items()},
            "delta_auc_full_minus_signal": round(base, 3),
            "delta_auc_ci95": [round(lo, 3), round(hi, 3)], "verdict": verdict}


def load_scores(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            for d in DIMS:
                k = f"score_{d}"
                if not r.get(k):
                    raise SystemExit(f"{path} missing {k}; run score_separability.py with H,T,R,S")
                r[k] = float(r[k])
            rows.append(r)
    return rows


META_COLS = {"pair_id", "source_id", "genre", "perturbation", "magnitude",
             "magnitude_unit", "partner_id", "dim_target", "notes"}


def read_metric_file(path: Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Per-pair metrics from either the score_baselines.py JSON contract or a plain
    per-pair CSV (score_cocola.py, score_rivals.py --csv): every non-metadata numeric
    column is a metric. Lets Table 4.1 be generated from the files as they were written,
    instead of an ad-hoc conversion (the identity_auc_cocola.json problem)."""
    if path.suffix.lower() == ".csv":
        with path.open() as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return {}, []
        metrics = [c for c in rows[0] if c not in META_COLS]
        scores = {}
        for r in rows:
            vals = {}
            for m in metrics:
                try:
                    vals[m] = float(r[m])
                except (TypeError, ValueError):
                    continue
            scores[r["pair_id"]] = vals
        return scores, metrics
    doc = json.loads(path.read_text())
    scores = doc.get("scores", {})
    metrics = (doc.get("meta", {}).get("baselines")
               or sorted({k for v in scores.values() for k in v}))
    return scores, list(metrics)


def attach_baselines(rows: list[dict], paths: list[Path],
                     keep: list[str] | None = None) -> None:
    """Load score_baselines.py-format output(s) and attach per-pair metrics to each row.

    Several files may be given (CLAP/SCS from score_baselines.py, the rivals from
    score_rivals.py); they are merged by pair_id so Table 4.1 is one table. A metric
    name occurring in two files is an error, not a silent overwrite.

    Each metric becomes a `base:<metric>` AUC feature, so off-the-shelf baselines
    (CLAP, SCS) are reported next to S under the SAME same/different splits -- the
    Ch-3 baseline table + the "is the whole field defeated on same-genre?" test.
    """
    global BASELINE_METRICS
    merged: dict[str, dict[str, float]] = {}
    BASELINE_METRICS = []
    for path in paths:
        scores, metrics = read_metric_file(path)
        if keep:
            metrics = [m for m in metrics if m in keep]
            if not metrics:
                continue
        dup = [m for m in metrics if m in BASELINE_METRICS]
        if dup:
            raise SystemExit(f"metric name(s) {dup} appear in more than one --baselines "
                             f"file (last: {path}); rename rather than overwrite")
        BASELINE_METRICS += list(metrics)
        for pid, vals in scores.items():
            merged.setdefault(pid, {}).update(vals)
        print(f"baselines {list(metrics)} from {path}: {len(scores)} scored pairs")
    miss = 0
    for r in rows:
        b = merged.get(r["pair_id"], {})
        r["base_scores"] = {m: float(b[m]) for m in BASELINE_METRICS if m in b}
        if any(m not in r["base_scores"] for m in BASELINE_METRICS):
            miss += 1
    print(f"{miss}/{len(rows)} matrix rows missing >=1 metric (shown as n/a per split)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--by-genre", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--baselines", default=None,
                    help="comma-separated baseline_scores.json file(s) (score_baselines.py, "
                         "score_rivals.py); adds each metric as an AUC feature "
                         "(base:<metric>) under every same/different split")
    ap.add_argument("--baseline-metrics", default=None,
                    help="comma list restricting which metrics from --baselines are "
                         "reported; the rivals emit 24 columns and Table 4.1 wants a few")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="write the AUC table here (Table 4.1 is generated, not hand-copied)")
    args = ap.parse_args()
    rows = load_scores(args.scores)
    if args.baselines:
        attach_baselines(rows, [Path(x) for x in str(args.baselines).split(",") if x],
                         keep=([m.strip() for m in args.baseline_metrics.split(",")]
                               if args.baseline_metrics else None))

    splits = []
    # 1) the hard, theory-motivated confusion: transpose vs style-swap
    splits.append(run_task(rows, {"pitch_shift"}, {"style_swap"},
                           "TRANSPOSE vs STYLE-SWAP (the confusion H cannot resolve)",
                           args.n_boot))
    # 2) the broad identity task: any same-song edit vs different song
    splits.append(run_task(rows, SAME, DIFF,
                           "ALL SAME-SONG vs STYLE-SWAP (broad identity discrimination)",
                           args.n_boot))
    # 3) cross vs same-genre swap: S's value should be LARGER on same-genre swaps,
    #    where T/R surface cues are preserved and only identity differs.
    splits.append(run_task(rows, SAME, DIFF,
                           "SAME-SONG vs CROSS-GENRE swap (easy: T/R separate it)",
                           args.n_boot, neg_tag="cross"))
    splits.append(run_task(rows, SAME, DIFF,
                           "SAME-SONG vs SAME-GENRE swap (hard: surface preserved, S should "
                           "matter more)", args.n_boot, neg_tag="same"))

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(
            {"scores": str(args.scores),
             "baselines": str(args.baselines) if args.baselines else None,
             "n_boot": args.n_boot,
             "splits": [s for s in splits if s]}, indent=2), encoding="utf-8")
        print(f"\n[wrote {args.out_json}]")

    if args.by_genre:
        for g in sorted({r["genre"] for r in rows}):
            grows = [r for r in rows if r["genre"] == g]
            run_task(grows, {"pitch_shift"}, {"style_swap"}, f"[{g}] transpose vs style-swap",
                     n_boot=1000)
    return 0


if __name__ == "__main__":
    sys.exit(main())