#!/usr/bin/env python
"""Stage 0 (offline half): does best-of-N still gain at N=8, or has it saturated?

The GRPO proposal's first gate asks whether the frozen generator's sampling
distribution still has better modes to find. If best-of-N has flattened by N=8, RL has
little to reach for and the experiment is not worth the compute.

The N<=8 part of that curve needs NO new generation and NO GPU: the re-ranking trees
already hold 8 scored candidates per seed, so E[max over a random n-subset] is an exact
order statistic of the scores we have. For candidates sorted ascending v_1..v_m,

    E[max of a random n-subset] = sum_i v_i * C(i-1, n-1) / C(m, n)

because v_i is the maximum exactly when the other n-1 draws come from the i-1 values
below it. No sampling, no simulation.

Three curves, and they answer different questions:
  * C          -- how much better does the BEST candidate get as the pool grows? This is
                  the headroom RL would be chasing.
  * S | rank C -- how much better is the candidate C PICKS, judged by S. NB this is NOT a
                  held-out judge: S carries weight 0.25 inside C, so the ranker is partly
                  grading its own component. Reported for the shape, not as evidence.
  * S | rank C_noS -- the same, ranking by H/T/R alone so that S is fully held out. This
                  is the curve that carries evidential weight, and it is much flatter.

Slope is fitted in log2 N. A flat slope at the top end means saturation.

Usage (DSP env):
    python scripts/analysis/bestofn_curve.py                       # ACE tree
    python scripts/analysis/bestofn_curve.py --scores results/rerank/pair_scores.csv
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

DIMS = ["H", "T", "R", "S"]


def load_seeds(scores_csv: Path) -> dict[str, list[dict]]:
    """seed_id -> list of {H,T,R,S} candidate dicts."""
    import csv
    seeds: dict[str, list[dict]] = {}
    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            pid = row["pair_id"]
            if "::" not in pid:
                continue
            seed = pid.split("::")[0]
            seeds.setdefault(seed, []).append(
                {d: float(row[f"score_{d}"]) for d in DIMS})
    return seeds


def geo_C(c: dict, exclude: str | None = None) -> float:
    vals = [c[d] for d in DIMS if d != exclude]
    v = np.clip(np.array(vals), 1e-9, 1.0)
    return float(np.exp(np.mean(np.log(v))))


def exact_expected_max(values: list[float], n: int) -> float:
    """E[max of a uniformly random n-subset], exactly (order-statistic weights)."""
    m = len(values)
    if n > m:
        raise ValueError(f"n={n} > pool size {m}")
    v = sorted(values)
    denom = comb(m, n)
    return float(sum(v[i] * comb(i, n - 1) for i in range(m)) / denom)


def expected_picked_by(cands: list[dict], n: int, rank_by: str, judge: str) -> float:
    """E[judge score of the candidate that `rank_by` would select from a random n-subset].

    Enumerated exactly over which candidate wins: candidate i (by rank_by, ascending)
    wins iff the other n-1 draws come from the i values below it.
    """
    m = len(cands)
    order = np.argsort([rank_by_value(c, rank_by) for c in cands])   # ascending
    denom = comb(m, n)
    total = 0.0
    for pos, idx in enumerate(order):
        w = comb(pos, n - 1)
        if w:
            total += w * judge_value(cands[idx], judge)
    return float(total / denom)


def rank_by_value(c: dict, key: str) -> float:
    return geo_C(c) if key == "C" else (geo_C(c, exclude="S") if key == "C_noS" else c[key])


def judge_value(c: dict, key: str) -> float:
    return geo_C(c) if key == "C" else c[key]


def boot_ci(per_seed: np.ndarray, n_boot: int = 10000, seed: int = 20260804):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, per_seed.shape[0], size=(n_boot, per_seed.shape[0]))
    means = per_seed[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Extrapolation: predict N=16/32 from N<=8, so the GPU run tests a stated
# prediction instead of answering an open question.
# ---------------------------------------------------------------------------
def _fit_predict(y: np.ndarray, ns: list[int], targets=(16, 32)) -> dict:
    """Two competing shapes for the same four points.

    geometric  increments shrink by a constant ratio r per doubling. Saturating, and
               it is the model the pre-registered target already assumes.
    loglinear  y = a + b*log2(N), i.e. increments never shrink at all. Not credible as
               physics, but it is the optimistic bound: if even this and `geometric`
               agree at N=16, the measurement cannot move the decision.

    The gap between them at a target N is the honest uncertainty about extrapolating.
    """
    gains = np.diff(y)
    r = float(np.clip(np.mean(gains[1:] / np.maximum(gains[:-1], 1e-12)), 0.0, 1.5))
    out = {"decay_ratio": r}

    g, last = float(gains[-1]), float(y[-1])
    geo = {}
    cur, val = g, last
    n = ns[-1]
    while n < max(targets):
        n *= 2
        cur *= r
        val += cur
        if n in targets:
            geo[n] = float(val)
    out["geometric"] = geo
    out["asymptote"] = float(last + g * r / (1 - r)) if r < 1 else float("inf")

    b, a = np.polyfit(np.log2(ns), y, 1)
    out["loglinear"] = {int(n): float(a + b * np.log2(n)) for n in targets}
    out["spread"] = {int(n): abs(out["loglinear"][n] - geo.get(n, np.nan))
                     for n in targets if n in geo}
    return out


def extrapolate(per_seed_by_n: dict[int, np.ndarray], ns: list[int],
                n_boot: int = 4000, seed: int = 20260805) -> dict:
    """Point estimate plus a seed-bootstrap CI on every extrapolated quantity."""
    y = np.array([per_seed_by_n[n].mean() for n in ns])
    point = _fit_predict(y, ns)

    rng = np.random.default_rng(seed)
    n_seeds = per_seed_by_n[ns[0]].shape[0]
    keys = [("decay_ratio", None), ("asymptote", None),
            ("geometric", 16), ("geometric", 32), ("loglinear", 16), ("loglinear", 32)]
    draws: dict[str, list[float]] = {f"{k}_{t}" if t else k: [] for k, t in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, n_seeds)
        yb = np.array([per_seed_by_n[n][idx].mean() for n in ns])
        fb = _fit_predict(yb, ns)
        for k, t in keys:
            v = fb[k] if t is None else fb[k].get(t, np.nan)
            if np.isfinite(v):
                draws[f"{k}_{t}" if t else k].append(float(v))

    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
          for k, v in draws.items() if v}
    return {"point": point, "ci": ci}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=str(ROOT / "results/rerank_ace/pair_scores.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/rerank_ace/bestofn_curve.json"))
    ap.add_argument("--fig", default=str(ROOT / "figures/fig_bestofn_curve"))
    a = ap.parse_args()

    seeds = load_seeds(Path(a.scores))
    pool = min(len(v) for v in seeds.values())
    ns = [n for n in (1, 2, 4, 8, 16, 32) if n <= pool]
    print(f"{len(seeds)} seeds, pool size {pool} -> N in {ns}\n")

    out: dict = {"meta": {"scores": a.scores, "n_seeds": len(seeds), "pool": pool,
                          "method": "exact order-statistic expectation over n-subsets"}}

    for label, ranker, judge in [("max_C", "C", "C"),
                                 ("C_picks_judged_by_S", "C", "S"),
                                 ("C_noS_picks_judged_by_S", "C_noS", "S")]:
        rows, by_n = [], {}
        for n in ns:
            if label == "max_C":
                per_seed = np.array([exact_expected_max([geo_C(c) for c in cs], n)
                                     for cs in seeds.values()])
            else:
                per_seed = np.array([expected_picked_by(cs, n, ranker, judge)
                                     for cs in seeds.values()])
            by_n[n] = per_seed
            lo, hi = boot_ci(per_seed)
            rows.append({"N": n, "mean": float(per_seed.mean()), "ci": [lo, hi]})
        # slope per doubling, and the last-doubling gain (the saturation tell)
        x = np.log2([r["N"] for r in rows])
        y = np.array([r["mean"] for r in rows])
        slope = float(np.polyfit(x, y, 1)[0])
        last = float(y[-1] - y[-2])
        first = float(y[1] - y[0])
        ex = extrapolate(by_n, ns) if len(ns) >= 3 and max(ns) < 32 else None
        out[label] = {"curve": rows, "slope_per_doubling": slope,
                      "gain_first_doubling": first, "gain_last_doubling": last,
                      "decay_ratio": float(last / first) if first else None,
                      "extrapolation": ex}

        print(f"--- {label}")
        for r in rows:
            print(f"  N={r['N']:>2}  {r['mean']:.4f}  CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]")
        print(f"  slope/doubling {slope:+.4f} | first {first:+.4f} | last {last:+.4f} "
              f"| last/first {out[label]['decay_ratio']:.2f}")
        if ex:
            p, c = ex["point"], ex["ci"]
            print(f"  PREDICTION for the GPU run (fitted on N<=8):")
            for n in (16, 32):
                g, l = p["geometric"].get(n), p["loglinear"].get(n)
                gci = c.get(f"geometric_{n}", [float('nan')] * 2)
                print(f"    N={n:<3} geometric {g:.4f} CI [{gci[0]:.4f}, {gci[1]:.4f}]"
                      f"   log-linear {l:.4f}   (models differ by {abs(l - g):.4f})")
            aci = c.get("asymptote", [float('nan')] * 2)
            print(f"    best-of-inf {p['asymptote']:.4f} CI [{aci[0]:.4f}, {aci[1]:.4f}]"
                  f" | decay ratio CI {c.get('decay_ratio', ['?', '?'])}")
        print()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {a.out}")

    try:
        make_fig(out, ns, Path(a.fig))
    except Exception as e:                                   # figure is optional
        print(f"(figure skipped: {e})")


def make_fig(out: dict, ns: list[int], stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from _figstyle import apply_style
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    panels = [(axes[0], "Best candidate, by MuRCo",
               [("max_C", "#0b0b0b", "ranked by $C$")]),
              (axes[1], "The winner's $S$, by what ranked it",
               [("C_picks_judged_by_S", "#c98500", "ranked by $C$ ($S$ inside)"),
                ("C_noS_picks_judged_by_S", "#3b6ea5", "ranked by $C_{noS}$ ($S$ held out)")])]
    for ax, title, series in panels:
        for label, col, leg in series:
            rows = out[label]["curve"]
            x = [r["N"] for r in rows]
            y = [r["mean"] for r in rows]
            lo = [r["ci"][0] for r in rows]
            hi = [r["ci"][1] for r in rows]
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(x, y, marker="o", ms=3.5, color=col, linewidth=1.2, label=leg)
        ax.set_xscale("log", base=2)
        ax.set_xticks(x); ax.set_xticklabels(x)
        ax.set_xlabel("candidate pool $N$")
        ax.set_title(title, fontsize=8.5)
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        if len(series) > 1:
            ax.legend(fontsize=7, frameon=False, loc="upper left")
    axes[0].set_ylabel("expected score")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {stem}.pdf/.png")


if __name__ == "__main__":
    main()
