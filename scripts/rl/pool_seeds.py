#!/usr/bin/env python3
"""Pool several GRPO training seeds into one best-of-N table.

Section 4.6 reports a single training run, and the thesis says so plainly: "a single pilot
at one seed". That is the weakest sentence in the chapter, and the fix is more seeds rather
than more prose. This script is what turns N eval directories into the numbers Table 4.5
carries, so that adding a seed is a re-run rather than a re-transcription.

Two questions are being asked at once and they are not the same, so both are reported.

  Does the gain generalise to unseen CONTEXTS?  The ten held-out contexts are the
  population being generalised to, so the bootstrap resamples contexts. Where several
  seeds are supplied the seeds are averaged WITHIN a context first, because the same ten
  contexts are re-used by every seed and treating thirty (seed, context) cells as thirty
  independent observations would shrink the interval by about sqrt(n_seeds) for free.

  Is the gain robust to the TRAINING SEED?  That is the spread of the per-seed deltas, and
  no bootstrap answers it. It is reported as the per-seed rows plus their min and max.

Best-of-N is the exact order statistic over an arm's own rollouts for one context, drawn
WITHOUT replacement, which is what eval_grpo.py's `_best_of` does at N = pool size and what
Section 4.6's curve means. With n rollouts sorted ascending, the chance that the i-th is the
largest of N drawn is C(i-1, N-1) / C(n, N).

    python scripts/rl/pool_seeds.py --dirs results/rl/eval_pilot2
    python scripts/rl/pool_seeds.py \
        --dirs results/rl/eval_pilot2,results/rl/eval_seed2,results/rl/eval_seed3 \
        --out results/rl/grpo_seeds.json --tex results/rl/grpo_bon.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()), Path.cwd())
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import _paths  # noqa: F401  (puts every scripts/<group>/ on sys.path)
except ImportError:
    pass

from eval_grpo import load_arms

NS = (1, 2, 4, 8)


def expected_best_of(vals: list[float], n_draw: int) -> float:
    """E[max of `n_draw` rollouts drawn without replacement from `vals`]."""
    v = sorted(vals)
    n = len(v)
    if n_draw >= n:
        return v[-1]
    denom = comb(n, n_draw)
    return sum(v[i] * comb(i, n_draw - 1) / denom for i in range(n_draw - 1, n))


def per_context_curve(arms, arm: str) -> dict[int, dict[str, float]]:
    """{N: {context: expected best-of-N C}} for one arm of one run."""
    if arm not in arms:
        raise SystemExit(f"arm {arm!r} not in this run; have {sorted(arms)}")
    return {n: {c: expected_best_of(v, n) for c, v in arms[arm]["C"].items()}
            for n in NS}


def cluster_boot(per_ctx_delta: dict[str, float], n_boot: int, seed: int):
    """Bootstrap the mean of a per-context quantity, resampling contexts."""
    ctxs = sorted(per_ctx_delta)
    x = np.array([per_ctx_delta[c] for c in ctxs], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    draws = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True,
                    help="comma-separated eval directories, one per training seed")
    ap.add_argument("--labels", default="",
                    help="comma-separated names for those runs (default: directory names)")
    ap.add_argument("--arm", default="step0150",
                    help="the tuned checkpoint to report (default: the one Sec. 4.6 reports)")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--tex", default="")
    a = ap.parse_args()

    dirs = [d.strip() for d in a.dirs.split(",") if d.strip()]
    labels = [s.strip() for s in a.labels.split(",") if s.strip()] or \
             [Path(d).name for d in dirs]
    if len(labels) != len(dirs):
        raise SystemExit("--labels must have one entry per directory")

    runs = []
    for d, lab in zip(dirs, labels):
        arms = load_arms(d)
        runs.append({
            "label": lab, "dir": d,
            "frozen": per_context_curve(arms, "frozen"),
            "tuned": per_context_curve(arms, a.arm),
        })

    shared = set.intersection(*[set(r["frozen"][1]) for r in runs])
    if not shared:
        raise SystemExit("the runs share no held-out context; are these the same split?")
    ctxs = sorted(shared)
    if any(len(r["frozen"][1]) != len(ctxs) for r in runs):
        print(f"note: pooling over the {len(ctxs)} contexts common to every run", file=sys.stderr)

    # --- per-seed deltas: the robustness-to-seed question -------------------------------
    per_seed = []
    for r in runs:
        row = {"label": r["label"], "dir": r["dir"], "by_N": {}}
        for n in NS:
            f = float(np.mean([r["frozen"][n][c] for c in ctxs]))
            t = float(np.mean([r["tuned"][n][c] for c in ctxs]))
            row["by_N"][n] = {"frozen": f, "tuned": t, "delta": t - f}
        # the headline: tuned best-of-8 against frozen best-of-8, matched inference cost
        row["delta_matched_bo8"] = row["by_N"][8]["delta"]
        per_seed.append(row)

    # --- pooled: seeds averaged within a context, bootstrap over contexts ---------------
    pooled = {}
    for n in NS:
        f_ctx = {c: float(np.mean([r["frozen"][n][c] for r in runs])) for c in ctxs}
        t_ctx = {c: float(np.mean([r["tuned"][n][c] for r in runs])) for c in ctxs}
        d_ctx = {c: t_ctx[c] - f_ctx[c] for c in ctxs}
        mu, lo, hi = cluster_boot(d_ctx, a.boot, a.seed)
        pooled[n] = {
            "frozen": float(np.mean(list(f_ctx.values()))),
            "tuned": float(np.mean(list(t_ctx.values()))),
            "delta": mu, "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
        }

    deltas8 = [r["delta_matched_bo8"] for r in per_seed]
    blob = {
        "meta": {
            "arm": a.arm, "n_seeds": len(runs), "n_contexts": len(ctxs),
            "dirs": dirs, "labels": labels, "boot": a.boot, "boot_seed": a.seed,
            "bootstrap_unit": "context; seeds averaged within a context before resampling",
        },
        "per_seed": per_seed,
        "pooled": {str(n): v for n, v in pooled.items()},
        "seed_spread_matched_bo8": {
            "values": deltas8, "min": min(deltas8), "max": max(deltas8),
            "mean": float(np.mean(deltas8)),
            "sd": float(np.std(deltas8, ddof=1)) if len(deltas8) > 1 else None,
        },
    }

    # --- report -------------------------------------------------------------------------
    print(f"arm {a.arm} | {len(runs)} seed(s) | {len(ctxs)} held-out contexts\n")
    print(f"{'N':>3}  {'frozen':>8}  {'tuned':>8}  {'delta':>8}   95% CI")
    for n in NS:
        p = pooled[n]
        star = "  *" if p["excludes_zero"] else ""
        print(f"{n:>3}  {p['frozen']:>8.3f}  {p['tuned']:>8.3f}  {p['delta']:>+8.3f}   "
              f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]{star}")

    print(f"\nper-seed delta at matched inference cost (tuned bo8 vs frozen bo8)")
    for r in per_seed:
        print(f"    {r['label']:<20}{r['delta_matched_bo8']:+.4f}")
    if len(deltas8) > 1:
        s = blob["seed_spread_matched_bo8"]
        print(f"    {'spread':<20}{s['min']:+.4f} to {s['max']:+.4f}"
              f"   (sd {s['sd']:.4f} over {len(deltas8)} seeds)")
    else:
        print("    only one seed supplied, so nothing here speaks to seed robustness")

    if a.out:
        Path(ROOT / a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(ROOT / a.out).write_text(json.dumps(blob, indent=2))
        print(f"\n[wrote {a.out}]")

    if a.tex:
        n_s = len(runs)
        seedtxt = ("ten contexts, paired cluster bootstrap over contexts"
                   if n_s == 1 else
                   f"{n_s} training seeds over ten contexts, seeds averaged within a "
                   f"context and the bootstrap resampling contexts")
        lines = [
            "% Generated by: python scripts/rl/pool_seeds.py --dirs <eval dirs> --tex " + a.tex,
            r"\begin{table}[t]", r"\centering",
            r"\caption[GRPO against best-of-$N$ re-ranking]{Held-out $C$ for the frozen "
            r"and tuned policies under best-of-$N$ re-ranking, " + seedtxt +
            r".\\}",
            r"\label{tab:grpo-bon}",
            r"\begin{tabular}{cccc}", r"\toprule",
            r"$N$ & frozen & tuned & $\Delta$ (95\% CI) \\", r"\midrule",
        ]
        for n in NS:
            p = pooled[n]
            lines.append(f"{n} & {p['frozen']:.3f} & {p['tuned']:.3f} & "
                         f"${p['delta']:+.3f}$ $[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]$ \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        Path(ROOT / a.tex).parent.mkdir(parents=True, exist_ok=True)
        Path(ROOT / a.tex).write_text("\n".join(lines) + "\n")
        print(f"[wrote {a.tex}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
