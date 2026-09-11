#!/usr/bin/env python3
"""Aggregate the four coherence dimensions into the metric C.

C(A,B) = H^wH * T^wT * R^wR * S^wS   (weighted geometric mean; sum(w)=1)

Reads a ``pair_scores.csv`` written by score_separability.py (columns
score_H/score_T/score_R/score_S) and:

  * computes per-pair C under geometric and linear aggregation,
  * reports how C responds to each perturbation family (control vs perturbed,
    paired, effect-size-led) -- for C the target is the opposite of the
    separability rows: C should drop for *every* perturbation, because any single
    failing dimension means the edit is incoherent,
  * contrasts geometric vs linear so the "any dimension failing pulls C->0"
    property is visible (the reason the metric is geometric, not a mean),
  * supports weight ablations (set a weight to 0 to drop a dimension), and
  * fits weights to human MOS when labels are supplied (--fit-mos), reporting the
    fit HELD OUT under grouped cross-validation alongside the scalar baselines.

The MOS fit is deliberately reported two ways, because they answer different
questions.  The in-sample weights are the finding ("how do listeners weight the
dimensions?") and are what belongs in the write-up.  The held-out correlations are
the accuracy claim, and only they may be compared against CLAP, SCS or COCOLA.
Folds are grouped by source track, since all 21 perturbations of one track share
that track's music and a random split would leak.  Each scalar baseline is refitted
in the same folds through a one-predictor log-linear model, so the comparison is not
four fitted parameters against zero.  Uniform-weight C is reported alongside as the
zero-parameter reference: if fitting does not beat it out of fold, the weights
should not be changed.

Geometric aggregation is why one bad dimension can't be averaged away: a
pure-timbre edit leaves H,R,S high and T low; the geometric C is pulled down by
T, a linear mean barely moves.  R's abstention on non-percussive audio (R~1) is
handled gracefully -- R^w ~ 1 contributes nothing, so H/T/S carry those clips.

Usage::

    python compute_C.py --scores results/coherence/pair_scores.csv --out results/C
    python compute_C.py --scores results/coherence/pair_scores.csv --weights H=1,T=1,R=1,S=0   # ablate S
    python compute_C.py --scores results/coherence/pair_scores.csv --fit-mos results/mos.csv    # fit lambda
    # held-out fit + baselines recalibrated in the same folds (ALWAYS pass --out)
    python compute_C.py --scores results/coherence/pair_scores.csv \
        --fit-mos results/mos/mos_live.csv --cv-group source_id \
        --out results/coherence/C_mosfit
    python compute_C.py ... --cv-group genre     # harsher: leave-one-genre-out
    python compute_C.py ... --cv-group none      # old behaviour, in-sample only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scipy.stats import wilcoxon, spearmanr, pearsonr
    from scipy.optimize import nnls
except ImportError as exc:  # pragma: no cover
    raise SystemExit("compute_C requires scipy") from exc

DIMS = ["H", "T", "R", "S"]
EPS = 1e-6


# --- weights ----------------------------------------------------------------

def parse_weights(spec: str) -> dict[str, float]:
    if spec == "uniform":
        raw = {d: 1.0 for d in DIMS}
    else:
        raw = {d: 0.0 for d in DIMS}
        for tok in spec.split(","):
            k, _, v = tok.partition("=")
            k = k.strip().upper()
            if k not in DIMS:
                raise ValueError(f"unknown dimension {k!r} in --weights")
            raw[k] = float(v)
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("weights sum to 0")
    return {d: raw[d] / total for d in DIMS}      # normalise so sum(w)=1


# --- aggregation ------------------------------------------------------------

def aggregate(rows: list[dict], w: dict[str, float]) -> None:
    """Add C_geo and C_lin to each row in place. Skips dims with weight 0."""
    active = [d for d in DIMS if w[d] > 0]
    for r in rows:
        s = {d: float(np.clip(r[f"score_{d}"], EPS, 1.0)) for d in active}
        r["C_geo"] = float(np.prod([s[d] ** w[d] for d in active]))
        r["C_lin"] = float(sum(w[d] * s[d] for d in active))


# --- stats (paired, effect-size led) ----------------------------------------

def paired(control: np.ndarray, perturbed: np.ndarray) -> dict:
    diff = control - perturbed
    n = diff.size
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    nz = diff[diff != 0]
    try:
        p = float(wilcoxon(nz).pvalue) if nz.size else float("nan")
    except ValueError:
        p = float("nan")
    return {"n": n, "mean_control": float(control.mean()), "mean_pert": float(perturbed.mean()),
            "mean_drop": float(diff.mean()), "cohen_dz": float(diff.mean() / sd) if sd > 0 else 0.0,
            "wilcoxon_p": p}


def sensitivity(rows: list[dict], field: str) -> list[dict]:
    control = {r["source_id"]: r[field] for r in rows if r["perturbation"] == "control"}
    fam: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["perturbation"] == "control":
            continue
        fam[r["perturbation"]][r["source_id"]].append(r[field])
    out = []
    for pert, s2 in fam.items():
        common = [s for s in s2 if s in control]
        if len(common) < 2:
            continue
        c = np.array([control[s] for s in common])
        p = np.array([float(np.mean(s2[s])) for s in common])
        st = paired(c, p)
        st["perturbation"] = pert
        out.append(st)
    out.sort(key=lambda d: -d["mean_drop"])
    return out


# --- MOS weight fitting (hook for the human study) --------------------------

def fit_weights(rows: list[dict], mos_path: Path) -> dict[str, float]:
    """Fit non-negative weights so C tracks human MOS.

    log C = sum_d w_d * log s_d, so we solve MOS ~ log-scores with NNLS (w>=0),
    then normalise.  Returns the fitted, normalised weight dict.
    """
    mos = {}
    with mos_path.open() as fh:
        for row in csv.DictReader(fh):
            mos[row["pair_id"]] = float(row["mos"])
    X, y = [], []
    for r in rows:
        if r["pair_id"] not in mos:
            continue
        X.append([np.log(float(np.clip(r[f"score_{d}"], EPS, 1.0))) for d in DIMS])
        y.append(mos[r["pair_id"]])
    if len(y) < len(DIMS) + 1:
        raise ValueError(f"need >= {len(DIMS)+1} labelled pairs, got {len(y)}")
    X = np.array(X); y = np.array(y)
    # scale MOS into (0,1] log-space target if it's on a 1-5 scale
    if y.max() > 1.0:
        y = y / y.max()
    coef, _ = nnls(X, np.log(np.clip(y, EPS, 1.0)))
    total = coef.sum()
    if total <= 0:
        raise ValueError("NNLS returned all-zero weights; check MOS labels")
    w = {d: float(coef[i] / total) for i, d in enumerate(DIMS)}
    return w


# --- grouped cross-validation of the MOS fit --------------------------------
#
# fit_weights above is IN-SAMPLE: it fits w on every labelled pair and has no
# held-out estimate.  The pre-specified analysis promises a HELD-OUT C-vs-MOS
# correlation, with and without S, and against the scalar baselines, so the fit
# has to be evaluated out of fold.  Two things make that non-trivial here.
#
#   1. Pairs are not independent.  All 21 perturbations of one source track share
#      that track's music, so a random split leaks.  Folds are grouped by
#      source_id (leave-one-track-out), the same grouping diagnose_perturbation.py
#      uses; --cv-group genre gives the harsher leave-one-genre-out.
#   2. Fairness.  C's fit is a log-linear model with four predictors, so each
#      scalar baseline is given a log-linear model with one predictor, fitted in
#      the same folds.  The baseline model carries an intercept and C's does not
#      (NNLS on log-scores has no constant), which can only help the baseline, so
#      the comparison is conservative for C.
#
# A single global monotone recalibration cannot move Spearman at all, so if the
# headline were rank correlation the baselines would gain nothing from being
# recalibrated.  Out of fold the picture is slightly different: the transform is
# monotone WITHIN a fold, but every fold fits its own coefficients, so pooling
# predictions across folds can move the rank correlation a little, and can move it
# a long way if the fitted slope changes sign between folds.  A calibrated
# Spearman far below the raw one is that instability, not an improvement.

def _design(rows, mos, dims):
    """Log-score design matrix over the pairs that carry a rating."""
    ids, groups, X, y = [], [], [], []
    for r in rows:
        pid = r["pair_id"]
        if pid not in mos:
            continue
        ids.append(pid)
        groups.append(r)
        X.append([np.log(np.clip(float(r[f"score_{d}"]), EPS, 1.0)) for d in dims])
        y.append(mos[pid])
    return ids, groups, np.array(X, dtype=float), np.array(y, dtype=float)


def _nnls_w(X, t):
    """Non-negative log-linear fit, normalised to sum 1 (all-zero -> uniform)."""
    coef, _ = nnls(X, t)
    total = coef.sum()
    if total <= 0:
        return np.full(X.shape[1], 1.0 / X.shape[1])
    return coef / total


def _loglin_predict(x_tr, t_tr, x_te):
    """One-predictor log-linear fit with intercept, for a scalar baseline."""
    A = np.column_stack([np.ones_like(x_tr), x_tr])
    beta, *_ = np.linalg.lstsq(A, t_tr, rcond=None)
    return beta[0] + beta[1] * x_te


def _corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan")}
    return {"pearson_r": float(pearsonr(a, b)[0]),
            "spearman_rho": float(spearmanr(a, b)[0])}


def load_mos(path: Path) -> dict[str, float]:
    mos = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            mos[row["pair_id"]] = float(row["mos"])
    return mos


def load_baselines(bl_json: Path | None, cocola_csv: Path | None) -> dict[str, dict[str, float]]:
    """Scalar rival scores keyed by name -> pair_id -> value."""
    out: dict[str, dict[str, float]] = {}
    if bl_json and bl_json.exists():
        blob = json.loads(bl_json.read_text())
        scores = blob.get("scores", blob)
        for pid, per in scores.items():
            for name, val in per.items():
                out.setdefault(name, {})[pid] = float(val)
    if cocola_csv and cocola_csv.exists():
        with cocola_csv.open() as fh:
            for row in csv.DictReader(fh):
                if row.get("cocola"):
                    out.setdefault("cocola", {})[row["pair_id"]] = float(row["cocola"])
    return out


def crossval_mos(rows, mos, baselines, group_key="source_id", n_boot=2000, seed=0) -> dict:
    """Held-out C-vs-MOS correlation, with and without S, against the baselines."""
    ids, grows, X, y = _design(rows, mos, DIMS)
    n_rated, n_matched = len(mos), len(ids)
    if n_matched < len(DIMS) + 2:
        raise ValueError(f"need >= {len(DIMS)+2} matched pairs, got {n_matched}")
    missing = sorted(set(mos) - set(ids))

    mos_max = float(y.max()) if y.max() > 1.0 else 1.0
    t = np.log(np.clip(y / mos_max, EPS, 1.0))          # fixed rescale, not fitted
    groups = np.array([r[group_key] for r in grows])
    folds = sorted(set(groups))

    variants = {"C_fitted": list(range(len(DIMS))),
                "C_fitted_noS": [DIMS.index(d) for d in ("H", "T", "R")]}
    oof = {k: np.full(n_matched, np.nan) for k in variants}
    fold_w = {k: [] for k in variants}
    for g in folds:
        te = groups == g
        tr = ~te
        if tr.sum() < len(DIMS) + 1 or te.sum() == 0:
            continue
        for name, cols in variants.items():
            w = _nnls_w(X[np.ix_(tr, cols)], t[tr])
            oof[name][te] = np.exp(X[np.ix_(te, cols)] @ w)
            fold_w[name].append(w)

    report = {
        "cv": {"group": group_key, "n_folds": len(folds), "n_rated_pairs": n_rated,
               "n_matched_pairs": n_matched, "n_rated_without_scores": len(missing),
               "dropped_pair_ids": missing[:20], "mos_max": mos_max},
        "held_out": {}, "in_sample": {}, "weight_stability": {},
    }

    # untuned reference: uniform weights, zero fitted parameters
    C_unif = np.exp(X @ np.full(len(DIMS), 1.0 / len(DIMS)))
    report["held_out"]["C_uniform"] = {**_corr(C_unif, y), "fitted_params": 0,
                                       "n_predicted": n_matched}

    for name in variants:
        ok = ~np.isnan(oof[name])
        report["held_out"][name] = {**_corr(oof[name][ok], y[ok]),
                                    "fitted_params": len(variants[name]),
                                    "n_predicted": int(ok.sum())}

    for name, cols in variants.items():
        w = _nnls_w(X[:, cols], t)
        dims = [DIMS[i] for i in cols]
        report["in_sample"][name] = {
            "weights": {d: float(w[i]) for i, d in enumerate(dims)},
            **_corr(np.exp(X[:, cols] @ w), y),
        }

    # baselines: raw, and log-linear recalibration fitted in the same folds
    for bname, table in sorted(baselines.items()):
        xb = np.array([table.get(pid, np.nan) for pid in ids], dtype=float)
        ok = np.isfinite(xb) & (xb > 0)
        if ok.sum() < len(DIMS) + 2:
            continue
        lx = np.log(xb)
        cal = np.full(n_matched, np.nan)
        for g in folds:
            te = (groups == g) & ok
            tr = (groups != g) & ok
            if tr.sum() < 3 or te.sum() == 0:
                continue
            cal[te] = _loglin_predict(lx[tr], t[tr], lx[te])
        cok = ok & np.isfinite(cal)
        report["held_out"][f"{bname}_raw"] = {**_corr(xb[ok], y[ok]), "fitted_params": 0,
                                              "n_predicted": int(ok.sum())}
        report["held_out"][f"{bname}_calibrated"] = {
            **_corr(np.exp(cal[cok]), y[cok]), "fitted_params": 2,
            "n_predicted": int(cok.sum()),
            "note": ("monotone within each fold, so a large Spearman drop against the "
                     "raw row means the fitted slope changed sign between folds"),
        }

    # cluster bootstrap over the grouping variable, for CIs on the weights
    rng = np.random.default_rng(seed)
    by_group = {g: np.flatnonzero(groups == g) for g in folds}
    draws = {d: [] for d in DIMS}
    zeros = {d: 0 for d in DIMS}
    for _ in range(n_boot):
        pick = rng.choice(folds, size=len(folds), replace=True)
        idx = np.concatenate([by_group[g] for g in pick])
        w = _nnls_w(X[idx], t[idx])
        for i, d in enumerate(DIMS):
            draws[d].append(float(w[i]))
            if w[i] <= 0.0:
                zeros[d] += 1
    for d in DIMS:
        arr = np.array(draws[d])
        report["weight_stability"][d] = {
            "median": float(np.median(arr)),
            "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "frac_exactly_zero": zeros[d] / max(n_boot, 1),
            "frac_folds_exactly_zero": float(
                np.mean([w[DIMS.index(d)] <= 0.0 for w in fold_w["C_fitted"]])
            ) if fold_w["C_fitted"] else float("nan"),
        }
    report["weight_stability"]["_note"] = (
        "H and T are entangled by construction and R abstains on non-percussive "
        "audio, so NNLS can drive a weight to exactly zero through collinearity. "
        "Read frac_exactly_zero before interpreting a zero as an unused dimension."
    )
    return report


def print_crossval(rep: dict) -> None:
    cv = rep["cv"]
    print(f"\n=== MOS fit, {cv['group']}-grouped CV ({cv['n_folds']} folds) ===")
    print(f"rated pairs {cv['n_rated_pairs']}, matched to scores {cv['n_matched_pairs']}, "
          f"dropped {cv['n_rated_without_scores']}")
    if cv["n_rated_without_scores"]:
        print(f"  DROPPED (no score row): {', '.join(cv['dropped_pair_ids'])}"
              + (" ..." if cv["n_rated_without_scores"] > 20 else ""))
    print(f"\n{'model':26s} {'params':>6s} {'pearson_r':>10s} {'spearman':>9s}")
    for name, m in rep["held_out"].items():
        print(f"{name:26s} {m.get('fitted_params', 0):6d} "
              f"{m['pearson_r']:10.3f} {m['spearman_rho']:9.3f}")
    print("\nin-sample weights (report these as the fitted weights, not as accuracy):")
    for name, m in rep["in_sample"].items():
        ws = "  ".join(f"{d}={v:.3f}" for d, v in m["weights"].items())
        print(f"  {name:14s} {ws}   (in-sample r={m['pearson_r']:.3f})")
    print("\nweight stability, cluster bootstrap over folds:")
    for d in DIMS:
        s = rep["weight_stability"][d]
        print(f"  {d}  median={s['median']:.3f}  95% CI [{s['ci95'][0]:.3f}, {s['ci95'][1]:.3f}]"
              f"  exactly-zero in {100*s['frac_exactly_zero']:.1f}% of resamples")


# --- io / report ------------------------------------------------------------

def load_scores(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for r in csv.DictReader(fh):
            for d in DIMS:
                key = f"score_{d}"
                if key not in r or r[key] == "":
                    raise SystemExit(f"{path} missing {key}; run score_separability.py with --dimensions H,T,R,S")
                r[key] = float(r[key])
            r["magnitude"] = r.get("magnitude", "")
            rows.append(r)
    return rows


def print_report(rows: list[dict], w: dict[str, float], by_genre: bool) -> dict:
    print(f"\nweights (normalised): " + "  ".join(f"{d}={w[d]:.3f}" for d in DIMS))
    geo, lin = sensitivity(rows, "C_geo"), sensitivity(rows, "C_lin")
    lin_by = {c["perturbation"]: c for c in lin}
    print(f"\nC sensitivity to each perturbation (control -> perturbed), geometric vs linear:")
    print(f"{'perturbation':14s} {'C_geo':>7s} {'geo_drop':>9s} {'dz':>6s} {'lin_drop':>9s}   geometric advantage")
    for c in geo:
        ld = lin_by.get(c["perturbation"], {}).get("mean_drop", float('nan'))
        adv = c["mean_drop"] - ld
        print(f"{c['perturbation']:14s} {c['mean_pert']:7.3f} {c['mean_drop']:+9.3f} "
              f"{c['cohen_dz']:6.2f} {ld:+9.3f}   {adv:+.3f}")
    ctrl = np.mean([r["C_geo"] for r in rows if r["perturbation"] == "control"])
    print(f"\ncontrol C_geo = {ctrl:.3f}  (baseline; every perturbation should sit below this)")

    result = {"weights": w, "control_C_geo": float(ctrl),
              "geometric": geo, "linear": lin}
    if by_genre:
        result["per_genre"] = {}
        for g in sorted({r["genre"] for r in rows}):
            grows = [r for r in rows if r["genre"] == g]
            result["per_genre"][g] = sensitivity(grows, "C_geo")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True, help="pair_scores.csv with H,T,R,S")
    ap.add_argument("--weights", default="uniform",
                    help="'uniform' or e.g. 'H=1,T=1,R=1,S=1' (0 to ablate a dimension)")
    ap.add_argument("--fit-mos", type=Path, default=None,
                    help="CSV (pair_id,mos) to fit weights to human ratings instead of --weights")
    ap.add_argument("--by-genre", action="store_true")
    ap.add_argument("--cv-group", choices=["source_id", "genre", "none"], default="source_id",
                    help="grouping for the held-out MOS fit: leave-one-track-out "
                         "(default), leave-one-genre-out, or skip CV entirely")
    ap.add_argument("--baselines", type=Path,
                    default=Path("results/baselines/baseline_scores.json"),
                    help="scalar rival scores to recalibrate in the same folds")
    ap.add_argument("--cocola", type=Path, default=Path("results/baselines/cocola_scores.csv"))
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="cluster-bootstrap resamples for the weight CIs")
    ap.add_argument("--out", type=Path, default=Path("results/C"))
    args = ap.parse_args()

    rows = load_scores(args.scores)
    cv_report = None

    if args.fit_mos:
        w = fit_weights(rows, args.fit_mos)
        print(f"fitted weights from {args.fit_mos}:")
        if args.cv_group != "none":
            cv_report = crossval_mos(
                rows, load_mos(args.fit_mos),
                load_baselines(args.baselines, args.cocola),
                group_key=args.cv_group, n_boot=args.n_boot)
            print_crossval(cv_report)
    else:
        w = parse_weights(args.weights)

    aggregate(rows, w)
    report = print_report(rows, w, args.by_genre)
    if cv_report is not None:
        report["mos_crossval"] = cv_report

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # augmented per-pair CSV
    with args.out.with_suffix(".csv").open("w", newline="") as fh:
        fields = ["pair_id", "source_id", "genre", "perturbation", "magnitude",
                  *[f"score_{d}" for d in DIMS], "C_geo", "C_lin"]
        wtr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[wrote {args.out.with_suffix('.csv')} and {args.out.with_suffix('.json')}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
