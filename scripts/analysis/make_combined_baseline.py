#!/usr/bin/env python3
"""E-C -- the combined baseline C_base = geomean(scs, clap_htsat).

Kills the "your metric is just a combination of existing published metrics"
attack (dissertation_strengthening_plan.md §2 layer 4) by putting the naive
combination in EVERY results table as its own row -- so the claim is refuted
with a number, not a paragraph.  C_base is what a reviewer would build from the
shelf: geometric mean of a structural-coherence scalar (our SCS stand-in) and an
acoustic-similarity scalar (CLAP-htsat).  The thesis then shows C_base loses
exactly where a decomposable relational metric is supposed to win -- drift
regime-selectivity, and the diagnosis / dose-response capabilities (E-A / E-B).

WHY NORMALIZE FIRST.  scs and clap_htsat live on different scales (scs carries a
large positive offset; clap_htsat spans a wider range), so an un-normalized
geomean would silently weight clap more.  Each metric is min-max normalized to
[0,1] over the relevant dataset BEFORE the geomean, and the min/max used is
recorded in the output meta.  Min-max is a monotone transform, so it does NOT
change either metric's own identity-AUC (AUC is rank-based) -- it only makes the
two commensurate for the product.  `--norm rank` is offered as an alternative
(dataset-relative; documented, not the default because it breaks the
score-a-new-pair semantics the drift use needs).

Produces, from existing caches only (no GPU, no re-scoring):

  MATRIX (Setup 2), from results/coherence/pair_scores.csv + results/baselines/baseline_scores.json:
    * results/baselines/baseline_scores_cbase.json  -- baseline_scores.json + a `cbase` key
      per pair, so `compute_S_value.py --baselines results/baselines/baseline_scores_cbase.json`
      reports base:cbase alongside base:clap_htsat / base:scs on every identity
      split (§2 layer 1: does the combination even do fine on identity? report it).
    * results/coherence/pair_scores_cbase.csv  -- pair_scores.csv + score_cbase (and the
      normalized score_clap_htsat / score_scs it is built from), the single file
      E-A (diagnose_perturbation.py) and E-B (dose_response.py) read to treat
      C_base as one more metric/feature.

  DRIFT (Setup 1), from results/drift_baselines_<model>.json (per_seed clap+scs):
    * results/drift_cbase_<model>.json  -- C_base pushed through the SAME per-seed
      drop-vs-ceiling schema as run_drift_baselines._summary, so
      `analyze_drift_selectivity.py` picks it up automatically (it now also loads
      drift_cbase_<model>.json) and C_base lands in the selectivity table next to
      C / S / clap_htsat / scs.  Expected kill-shot (§2 layer 4): scs is a level
      detector (flat drift slope, selectivity CI spans 0), so C_base inherits weak
      selectivity while C's CI excludes 0.  Reported honestly either way.

  ONE DEFINITION, APPLIED EVERYWHERE.  The min-max range is fitted ONCE on the
  matrix (Setup-2) set and that same fitted range is applied to the drift values
  (clipped to [0,1]).  This is deliberate: a metric is defined once and deployed
  unchanged.  Re-fitting min-max on the narrow drift support would stretch tiny
  absolute movements to fill [0,1] and make C_base look artificially MORE
  drift-sensitive -- a normalization artifact, not a property of the combination.
  `--norm rank` re-ranks within each dataset (dataset-relative by construction),
  so it is matrix-only and drift is skipped for it.

Usage (DSP/local env, from repo root):

    python scripts/analysis/make_combined_baseline.py            # writes all of the above
    python scripts/analysis/make_combined_baseline.py --norm rank
    # then, to read the rows off:
    python scripts/core/compute_S_value.py --scores results/coherence/pair_scores.csv \
        --baselines results/baselines/baseline_scores_cbase.json --by-genre
    python scripts/drift/analyze_drift_selectivity.py --primary --results-dir results \
        --out results/drift/drift_selectivity.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# the two published-scalar members of the combination.  clap_music is degenerate
# (saturates ~1.0, §8.1) and fad_proxy is a per-pair proxy, so neither joins the
# combination a reviewer would actually build.
MEMBERS = ("scs", "clap_htsat")
EPS = 1e-6


def minmax(vals: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if hi - lo < EPS:
        return np.full_like(vals, 0.5), lo, hi
    return (vals - lo) / (hi - lo), lo, hi


def rank01(vals: np.ndarray) -> np.ndarray:
    """Average-rank to [0,1]; NaNs kept NaN and excluded from the ranking."""
    out = np.full_like(vals, np.nan, dtype=float)
    ok = ~np.isnan(vals)
    a = vals[ok]
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(1, len(a) + 1)
    uniq, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            r[inv == i] = r[inv == i].mean()
    out[ok] = (r - 1) / (len(a) - 1) if len(a) > 1 else 0.5
    return out


def geomean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.clip(a, EPS, 1.0)
    b = np.clip(b, EPS, 1.0)
    return np.sqrt(a * b)


def normalize(vals: np.ndarray, how: str):
    if how == "none":
        return np.asarray(vals, float), {"how": "none",
                                         "note": "raw members (both already native ~[0,1], "
                                                 "control~1); most transparent naive combination"}
    if how == "minmax":
        n, lo, hi = minmax(vals)
        return n, {"how": "minmax", "min": lo, "max": hi}
    if how == "rank":
        return rank01(vals), {"how": "rank", "note": "average-rank to [0,1], dataset-relative"}
    raise ValueError(f"unknown --norm {how!r}")


# --------------------------------------------------------------------------- #
# MATRIX side                                                                 #
# --------------------------------------------------------------------------- #
def build_matrix(results: Path, how: str) -> dict:
    pair_csv = results / "pair_scores.csv"
    base_json = results / "baseline_scores.json"
    rows = list(csv.DictReader(pair_csv.open(newline="", encoding="utf-8")))
    base_doc = json.loads(base_json.read_text(encoding="utf-8"))
    base_scores = base_doc.get("scores", base_doc)

    ids = [r["pair_id"] for r in rows]
    missing = [i for i in ids if i not in base_scores or
               any(m not in base_scores[i] for m in MEMBERS)]
    if missing:
        raise SystemExit(f"{len(missing)} matrix pairs missing a member metric "
                         f"(first: {missing[0]}); cannot build C_base fairly.")

    norm = {}
    meta_norm = {}
    for m in MEMBERS:
        raw = np.array([float(base_scores[i][m]) for i in ids], float)
        norm[m], meta_norm[m] = normalize(raw, how)
    cbase = geomean(norm[MEMBERS[0]], norm[MEMBERS[1]])

    # 1) extended baseline_scores.json (adds `cbase`; members kept RAW so their
    #    own AUC rows are unchanged -- normalization is monotone anyway).
    out_scores = {i: dict(base_scores[i]) for i in base_scores}
    for i, c in zip(ids, cbase):
        out_scores[i]["cbase"] = float(c)
    out_meta = dict(base_doc.get("meta", {}))
    out_meta["baselines"] = list(dict.fromkeys(list(out_meta.get("baselines", MEMBERS)) + ["cbase"]))
    out_meta["cbase"] = {"members": list(MEMBERS), "aggregation": "geomean",
                         "normalization": meta_norm,
                         "note": "C_base = sqrt(norm(scs) * norm(clap_htsat)); the naive "
                                 "combined baseline for the E-C novelty defense."}
    (results / "baseline_scores_cbase.json").write_text(
        json.dumps({"meta": out_meta, "scores": out_scores}, indent=2), encoding="utf-8")

    # 2) pair_scores_cbase.csv (adds normalized members + cbase to the H/T/R/S rows)
    out_csv = results / "pair_scores_cbase.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        fields = list(rows[0].keys()) + ["score_scs", "score_clap_htsat", "score_cbase"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r, c, s0, s1 in zip(rows, cbase, norm[MEMBERS[0]], norm[MEMBERS[1]]):
            r = dict(r)
            r["score_scs"] = f"{s0:.6f}"
            r["score_clap_htsat"] = f"{s1:.6f}"
            r["score_cbase"] = f"{c:.6f}"
            w.writerow(r)

    return {"n_pairs": len(ids), "normalization": meta_norm,
            "cbase_mean": float(np.mean(cbase)), "cbase_std": float(np.std(cbase)),
            "wrote": [str(results / "baseline_scores_cbase.json"), str(out_csv)]}


def matrix_minmax_ranges(results: Path) -> dict:
    """Refit the member min-max on the matrix set only (for --norm minmax drift)."""
    base_doc = json.loads((results / "baseline_scores.json").read_text(encoding="utf-8"))
    base_scores = base_doc.get("scores", base_doc)
    ranges = {}
    for m in MEMBERS:
        vals = np.array([float(v[m]) for v in base_scores.values() if m in v], float)
        ranges[m] = {"min": float(vals.min()), "max": float(vals.max())}
    return ranges


# --------------------------------------------------------------------------- #
# DRIFT side                                                                  #
# --------------------------------------------------------------------------- #
def build_drift(results: Path, model: str, matrix_ranges: dict) -> dict | None:
    p = results / f"drift_baselines_{model}.json"
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    if not all(m in doc and doc[m].get("per_seed") for m in MEMBERS):
        return {"skipped": f"{p.name} lacks per_seed for {MEMBERS}"}

    ref = doc[MEMBERS[0]]
    seeds = sorted(doc[MEMBERS[0]]["per_seed"]["ceiling"])
    iter_ks = sorted(doc[MEMBERS[0]]["per_seed"]["iters"], key=lambda x: int(x))

    # For --norm none: raw members (both native ~[0,1], control~1), so C_base's
    # drift drops stay directly comparable to C's and clap's -- no amplification.
    # For --norm minmax: apply the MATRIX-fitted range unchanged (clip to [0,1]);
    # the metric is defined on Setup-2 and deployed as-is.
    ranges = {}
    normed = {}  # normed[m]["ceiling"|k][seed] = value
    for m in MEMBERS:
        if matrix_ranges is None:  # --norm none
            ranges[m] = {"how": "none"}
            fn = (lambda x: float(x))
        else:
            lo, hi = matrix_ranges[m]["min"], matrix_ranges[m]["max"]
            ranges[m] = {"how": "minmax", "min": lo, "max": hi, "fitted_on": "matrix"}
            fn = (lambda x, lo=lo, hi=hi:
                  float(np.clip((x - lo) / (hi - lo), 0.0, 1.0)) if hi - lo > EPS else 0.5)
        cur = {"ceiling": {s: fn(doc[m]["per_seed"]["ceiling"][s]) for s in seeds}}
        for k in iter_ks:
            cur[k] = {s: (fn(v) if (v := doc[m]["per_seed"]["iters"][k].get(s)) is not None
                          else None) for s in seeds}
        normed[m] = cur

    def cb(a, b):
        return None if (a is None or b is None) else float(geomean(np.array([a]), np.array([b]))[0])

    ceiling = {s: cb(normed[MEMBERS[0]]["ceiling"][s], normed[MEMBERS[1]]["ceiling"][s])
               for s in seeds}
    per_k = {k: {s: cb(normed[MEMBERS[0]][k][s], normed[MEMBERS[1]][k][s]) for s in seeds}
             for k in iter_ks}

    cvals = np.array([v for v in ceiling.values() if v is not None], float)
    cm, csd, cN = float(cvals.mean()), float(cvals.std()), int(cvals.size)
    iters = {}
    for k in iter_ks:
        kv = np.array([v for v in per_k[k].values() if v is not None], float)
        km = float(kv.mean())
        iters[k] = {"mean": km, "std": float(kv.std()), "n": int(kv.size), "drop": cm - km}

    summary = {"cbase": {
        "name": f"{ref['name'].split('-')[0]}-cbase", "metric": "cbase",
        "region": ref.get("region"), "ceiling": ref.get("ceiling"),
        "ceiling_mean": cm, "ceiling_std": csd, "ceiling_n": cN, "iters": iters,
        "per_seed": {"ceiling": ceiling, "iters": {k: per_k[k] for k in iter_ks}},
        "cbase_meta": {"members": list(MEMBERS), "aggregation": "geomean",
                       "normalization": ranges}}}
    out = results / f"drift_cbase_{model}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"model": model, "ceiling_mean": cm,
            "drop_iter8": iters.get("8", {}).get("drop"), "wrote": str(out)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--norm", choices=("none", "minmax", "rank"), default="none",
                    help="per-member normalization before the geomean. Default 'none' "
                         "(raw; both members are already native ~[0,1]) keeps drift drops "
                         "comparable to C/clap and, being rank-preserving, leaves every "
                         "downstream AUC/Spearman/nearest-centroid result unchanged. "
                         "'minmax'/'rank' are offered for inspection only.")
    ap.add_argument("--no-drift", action="store_true",
                    help="skip the Setup-1 drift C_base files")
    args = ap.parse_args()

    report = {"norm": args.norm, "members": list(MEMBERS)}
    report["matrix"] = build_matrix(args.results_dir, args.norm)
    print(f"MATRIX  C_base over {report['matrix']['n_pairs']} pairs "
          f"(norm={args.norm}): mean={report['matrix']['cbase_mean']:.3f}")
    for f in report["matrix"]["wrote"]:
        print(f"  wrote {f}")

    if args.no_drift:
        pass
    elif args.norm == "rank":
        print("DRIFT   skipped: --norm rank is dataset-relative; a drift metric needs a "
              "fixed definition. Use --norm none (default) or minmax for the drift files.")
        report["drift"] = "skipped_norm_rank"
    else:
        matrix_ranges = (matrix_minmax_ranges(args.results_dir)
                         if args.norm == "minmax" else None)
        report["drift_ranges_fitted_on_matrix"] = matrix_ranges
        report["drift"] = {}
        for model in ("sao", "ace", "musicgen"):
            r = build_drift(args.results_dir, model, matrix_ranges)
            if r is None:
                print(f"DRIFT   {model}: no drift_baselines_{model}.json -- skipped")
            elif "skipped" in r:
                print(f"DRIFT   {model}: {r['skipped']}")
            else:
                print(f"DRIFT   {model}: C_base ceiling={r['ceiling_mean']:.3f} "
                      f"drop(iter8)={r['drop_iter8']:+.3f} -> {r['wrote']}")
            report["drift"][model] = r

    (args.results_dir / "cbase_build_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {args.results_dir / 'cbase_build_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
