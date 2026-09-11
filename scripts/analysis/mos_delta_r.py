#!/usr/bin/env python
"""Cluster-bootstrap confidence intervals on DIFFERENCES of MOS correlations.

The thesis reports S's Part-1 contribution as a bare "+0.045 to the correlation".
An uncertified delta-r reads as noise, and that number carries the central claim.
This script puts an interval on it, resampling SOURCE TRACKS (not pairs), because
several rated pairs share a source and are not independent.

Uses UNIFORM weights by default so that zero parameters are fitted and the
"you fitted the weights on the same 154 pairs" objection cannot be raised. Pass
--weights to use the NNLS-fitted weights instead (compute_C.py --fit-mos prints them).

    python scripts/analysis/mos_delta_r.py --mos results/mos/mos_live.csv
    python scripts/analysis/mos_delta_r.py --mos results/mos/mos_live.csv --drop style_swap control

Reported 2026-08-17 at N=16 (154 pairs, 59 sources), uniform weights:
    C - C_noS      +0.047  [+0.023, +0.075]   excludes 0
    C - CLAP       +0.100  [+0.018, +0.185]   excludes 0
    C_noS - CLAP   +0.054  [-0.040, +0.147]   SPANS 0
i.e. without S, C does not beat CLAP on human coherence. The S contribution
survives dropping the style_swap and control anchors (+0.076 [+0.037, +0.119]).
"""
from __future__ import annotations

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
    """Walk up for the root marker rather than counting .parent."""
    for p in [_here, *_here.parents]:
        if (p / ".projectroot").exists() or (p / ".git").exists():
            return p
    return Path.cwd()


META_COLS = {"pair_id", "source_id", "genre", "perturbation", "magnitude",
             "magnitude_unit", "partner_id", "dim_target", "notes"}


def read_metric_file(path: Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Per-pair metrics from the score_baselines.py JSON contract or a per-pair CSV
    (score_cocola.py, score_rivals.py --csv). Same reader as compute_S_value.py."""
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(open(path)))
        if not rows:
            return {}, []
        metrics = [c for c in rows[0] if c not in META_COLS]
        out = {}
        for r in rows:
            vals = {}
            for m in metrics:
                try:
                    vals[m] = float(r[m])
                except (TypeError, ValueError):
                    continue
            out[r["pair_id"]] = vals
        return out, metrics
    doc = json.loads(path.read_text())
    sc = doc.get("scores", {})
    metrics = (doc.get("meta", {}).get("baselines")
               or sorted({k for v in sc.values() for k in v}))
    return sc, list(metrics)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, reported alongside pearson() so the table can state which it is.

    The contrasts and their bootstrap intervals stay on Pearson; this only adds the
    marginal rank correlation, which is the safer read for a bounded, monotone metric.
    """
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 5 or a.std() == 0 or b.std() == 0:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mos", default="results/mos/mos_live.csv", help="pair_id,mos")
    ap.add_argument("--scores", default="results/coherence/pair_scores.csv")
    ap.add_argument("--baselines", default="results/baselines/baseline_scores.json")
    ap.add_argument("--rivals", default=None,
                    help="comma-separated extra per-pair metric files (JSON from "
                         "score_rivals.py / score_baselines.py, or CSV from score_cocola.py "
                         "/ score_rivals.py --csv). Each selected metric gets a marginal r "
                         "and a C - <metric> contrast.")
    ap.add_argument("--rival-metrics", default=None,
                    help="comma list of metric names to use from --rivals; default keeps "
                         "the relational (_rel) and absolute (_b) readings plus any bare "
                         "metric name, which is what the write-up reports")
    ap.add_argument("--weights", default=None,
                    help="H=..,T=..,R=..,S=.. ; omit for uniform (0 fitted params, preferred)")
    ap.add_argument("--drop", nargs="*", default=[], metavar="FAMILY",
                    help="perturbation families to exclude, e.g. --drop style_swap control")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write JSON here")
    args = ap.parse_args()

    mos = {r["pair_id"]: float(r["mos"]) for r in csv.DictReader(open(root / args.mos))}
    scores = {r["pair_id"]: r for r in csv.DictReader(open(root / args.scores))}
    baselines = json.load(open(root / args.baselines))["scores"]

    rival_cols: list[str] = []
    rival_scores: dict[str, dict[str, float]] = {}
    for path in [x for x in (args.rivals or "").split(",") if x]:
        sc, metrics = read_metric_file(root / path)
        keep = ([m.strip() for m in args.rival_metrics.split(",")] if args.rival_metrics
                else [m for m in metrics if m.endswith(("_rel", "_b")) or "_" not in m])
        keep = [m for m in keep if m in metrics]
        if not keep:
            print(f"warning: no usable metrics from {path}", file=sys.stderr)
        rival_cols += [m for m in keep if m not in rival_cols]
        for pid, vals in sc.items():
            rival_scores.setdefault(pid, {}).update(vals)

    if args.weights:
        w = {k: float(v) for k, v in (kv.split("=") for kv in args.weights.split(","))}
        tot = sum(w.values())
        wS = {k: v / tot for k, v in w.items()}
        wn = {k: v / (tot - w.get("S", 0.0)) for k, v in w.items() if k != "S"}
    else:
        wS = {d: 0.25 for d in "HTRS"}
        wn = {d: 1 / 3 for d in "HTR"}

    rows = []
    for pid, m in mos.items():
        s = scores.get(pid)
        if s is None:
            continue
        if s["perturbation"] in args.drop:
            continue
        dim = {d: max(1e-9, float(s["score_" + d])) for d in "HTRS"}
        rows.append({
            "src": s["source_id"], "fam": s["perturbation"], "mos": m,
            "C": float(np.prod([dim[d] ** wS[d] for d in "HTRS"])),
            "C_noS": float(np.prod([dim[d] ** wn[d] for d in "HTR"])),
            "clap_htsat": baselines.get(pid, {}).get("clap_htsat", np.nan),
            "scs": baselines.get(pid, {}).get("scs", np.nan),
            **{m_: rival_scores.get(pid, {}).get(m_, np.nan) for m_ in rival_cols},
        })
    if not rows:
        print("no rated pairs matched the scores file", file=sys.stderr)
        return 1

    src = np.array([r["src"] for r in rows])
    M = np.array([r["mos"] for r in rows])
    cols = {name: np.array([r[name] for r in rows], dtype=float)
            for name in ("C", "C_noS", "clap_htsat", "scs", *rival_cols)}
    # A rival scored on only some rated pairs would silently change the sample the
    # correlation is computed on, so drop it loudly instead.
    for name in list(rival_cols):
        n_miss = int(np.isnan(cols[name]).sum())
        if n_miss:
            print(f"warning: {name} missing on {n_miss}/{len(rows)} rated pairs; excluded "
                  f"(score those pairs or pass --rival-metrics without it)", file=sys.stderr)
            rival_cols.remove(name)
            cols.pop(name)

    usrc = sorted(set(src.tolist()))
    by_src = {u: np.where(src == u)[0] for u in usrc}
    rng = np.random.default_rng(args.seed)

    print(f"weights: {'uniform' if not args.weights else args.weights}")
    print(f"{len(rows)} rated pairs, {len(usrc)} source tracks"
          + (f", dropped {args.drop}" if args.drop else ""))
    print()
    print("marginal correlations with MOS")
    marg = {k: pearson(v, M) for k, v in cols.items()}
    marg_rho = {k: spearman(v, M) for k, v in cols.items()}
    for k, v in marg.items():
        print(f"  r({k:<11s}, MOS) = {v:+.4f}   rho = {marg_rho[k]:+.4f}")
    print()

    contrasts = [("C", "C_noS"), ("C", "clap_htsat"), ("C_noS", "clap_htsat"), ("C", "scs")]
    contrasts += [("C", m) for m in rival_cols]
    print(f"{'contrast':<30s}{'delta r':>9s}  {'95% cluster CI':<22s}P(delta<=0)")
    out = {"n_pairs": len(rows), "n_sources": len(usrc), "dropped": args.drop,
           "rivals": rival_cols,
           "weights": "uniform" if not args.weights else args.weights,
           "marginal_r": marg, "marginal_rho": marg_rho, "contrasts": {}}
    for a, b in contrasts:
        draws = []
        for _ in range(args.boot):
            pick = rng.integers(0, len(usrc), len(usrc))
            ii = np.concatenate([by_src[usrc[p]] for p in pick])
            d = pearson(cols[a][ii], M[ii]) - pearson(cols[b][ii], M[ii])
            if not np.isnan(d):
                draws.append(d)
        o = np.sort(np.array(draws))
        lo, hi = o[int(0.025 * len(o))], o[int(0.975 * len(o))]
        p = float((o <= 0).mean())
        name = f"{a} - {b}"
        print(f"{name:<30s}{o.mean():+9.4f}  [{lo:+.4f}, {hi:+.4f}]      {p:.4f}")
        out["contrasts"][name] = {"delta_r": float(o.mean()), "ci95": [float(lo), float(hi)],
                                  "frac_le_zero": p, "excludes_zero": bool(lo > 0 or hi < 0)}

    if args.out:
        dest = root / args.out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2))
        print(f"\n[wrote {dest}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
