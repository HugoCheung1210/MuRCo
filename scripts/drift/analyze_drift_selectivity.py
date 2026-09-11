#!/usr/bin/env python3
"""T2 -- drift REGIME-SELECTIVITY + rank statistics (DSP env, CPU, no new scoring).

Why this exists (PROJECT_STATE §8.1): clap_htsat accumulates in ALL THREE drift
regimes -- including ACE, where the protocol says nothing should accumulate.  So
CLAP looks partly like a generic resynthesis-artifact detector, i.e. its ACE
accumulation is a FALSE POSITIVE, while S's SAO-vs-ACE contrast is much sharper.
Slope alone ("does it accumulate?") scores CLAP and S as comparable; SELECTIVITY
("does it accumulate ONLY where the edit actually compounds?") may not.  This
script formalises that instead of eyeballing it.

Two analyses:

  T2.1 selectivity   sel_m = mean Δ_m(SAO) − mean Δ_m(ACE),  where the per-seed
                     accumulation is Δ_m(seed) = score_iter1 − score_iter8.
                     (drop(iter_k) = ceiling − score_k, so the ceiling cancels
                     out of the difference -- Δ is ceiling-independent, which is
                     what makes an across-model contrast defensible at all.)
                     CI: TWO-SAMPLE bootstrap over seeds -- SAO and ACE seeds are
                     independent runs, NOT paired.  The S-vs-clap_htsat
                     selectivity difference gets its own bootstrap, resampling
                     each model's seeds ONCE per replicate so the within-model
                     pairing across metrics (same seed -> both metrics) is kept.

  T2.2 rank stats    per-seed Spearman rho(score, iteration depth) -- mean +
                     bootstrap CI, and the PAIRED per-seed S−clap_htsat rho
                     difference (paired: same seeds within a model).  Plus
                     pairwise depth-ordering accuracy: over within-seed pairs
                     (iter_i, iter_j) with i<j, the fraction where the metric
                     calls iter_i the more coherent one.

FAIRNESS / CAVEATS (carry these into the write-up verbatim):
  * n=10 seeds per regime.  Report effect size + CI.  NEVER say "significant";
    Wilcoxon p is floored at these n (PROJECT_STATE §3).
  * Magnitudes are NOT comparable across models in general (§8.3: SAO/ACE are
    roundtrip-ceiling, MusicGen is self-ceiling).  The SAO-vs-ACE contrast is
    the exception and the whole point: BOTH are roundtrip at region (4,6)/13 s
    with identical pairing conventions, so their Δs live on one scale.  MusicGen
    is reported for context only and is excluded from `sel`.

INPUT FORMAT: needs the `per_seed` block written by run_drift_htrc.py /
run_drift_baselines.py `_summary()`.  Older result files stored only per-iter
mean/std/n -- the seed-level values were discarded -- so they CANNOT support any
of this; with such a file the script reports the point estimates it can compute
and tells you exactly what to re-run.  Re-running is what regenerates per_seed:

    python scripts/drift/run_drift_htrc.py      --model sao --stats --out results/drift/drift_htrc_sao.json
    python scripts/drift/run_drift_baselines.py --model sao --stats --out results/drift/drift_baselines_sao.json
    (likewise --model ace / musicgen)

Usage:

    python scripts/drift/analyze_drift_selectivity.py --results-dir results \
        --out results/drift/drift_selectivity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MODELS = ("sao", "ace", "musicgen")
CONTRAST = ("sao", "ace")        # the fair pair: both roundtrip @ (4,6)/13s
PRIMARY = ("S", "clap_htsat")    # the head-to-head the dissertation turns on
# --primary also emits the C-vs-field contrasts (the Setup-1 counterpart of the
# Setup-2 "C beats the field" claim); each is (metric_a, metric_b) -> sel_a - sel_b.
PRIMARY_EXTRA = (("C", "clap_htsat"), ("C", "scs"), ("C", "cbase"))
B_BOOT = 10000
SEED = 20260717


def load_model(results_dir: Path, model: str) -> dict:
    """Merge the H/T/R/S/C summary and the baseline summary for one model."""
    merged: dict[str, dict] = {}
    # drift_cbase_<model>.json is the E-C combined baseline (optional; skip if absent).
    for fname in (f"drift_htrc_{model}.json", f"drift_baselines_{model}.json",
                  f"drift_cbase_{model}.json"):
        p = results_dir / fname
        if not p.is_file():
            continue
        doc = json.loads(p.read_text(encoding="utf-8"))
        for metric, res in doc.items():
            merged[metric] = res
    return merged


def per_seed_delta(res: dict, k_lo: str = "1", k_hi: str = "8") -> dict[str, float]:
    """Δ(seed) = score(iter_lo) − score(iter_hi); + = coherence falls with depth."""
    ps = res.get("per_seed")
    if not ps:
        return {}
    lo, hi = ps["iters"].get(k_lo, {}), ps["iters"].get(k_hi, {})
    return {s: float(lo[s]) - float(hi[s]) for s in lo
            if s in hi and lo[s] is not None and hi[s] is not None}


def per_seed_curves(res: dict) -> dict[str, list[tuple[int, float]]]:
    """{seed: [(depth, score), ...]} over every logged iteration."""
    ps = res.get("per_seed")
    if not ps:
        return {}
    curves: dict[str, list[tuple[int, float]]] = {}
    for k, d in ps["iters"].items():
        for s, v in d.items():
            if v is not None:
                curves.setdefault(s, []).append((int(k), float(v)))
    return {s: sorted(v) for s, v in curves.items() if len(v) >= 2}


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho without scipy (tiny n, average ranks for ties)."""
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), float)
        r[order] = np.arange(1, len(a) + 1)
        # average ranks within tie groups
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        for i, c in enumerate(cnt):
            if c > 1:
                r[inv == i] = r[inv == i].mean()
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def ordering_accuracy(curve: list[tuple[int, float]]) -> float:
    """Fraction of within-seed (i<j) depth pairs scored iter_i > iter_j."""
    hits = tot = 0
    for a in range(len(curve)):
        for b in range(a + 1, len(curve)):
            tot += 1
            hits += curve[a][1] > curve[b][1]
    return hits / tot if tot else float("nan")


def boot_ci(rng, stat_fn, n_rep=B_BOOT, alpha=0.05):
    vals = np.array([stat_fn(rng) for _ in range(n_rep)], float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def resample(rng, arr):
    return arr[rng.integers(0, len(arr), len(arr))] if len(arr) else arr


# E-C dissertation table (tab:selectivity): (ours) above the rule, field below.
_TEX_ROWS = [("C", r"$C$ (ours)", True), ("S", r"$S$ (ours)", True), None,
             ("cbase", r"SCS$\times$CLAP", False),
             ("clap_htsat", "CLAP-htsat", False), ("scs", "SCS (stand-in)", False),
             ("clap_music", "CLAP-music", False)]


def write_selectivity_tex(sel: dict, path: Path) -> None:
    """Emit tab:selectivity to \\input into the dissertation (matches doc/writeup/dissertation.tex)."""
    def note(k, d):
        lo, hi = d["ci95"]
        if lo > 0 or hi < 0:
            return "excl.\\ 0"
        if abs(d["delta_sao"]) < 5e-3 and abs(d["delta_ace"]) < 5e-3:
            return "(degenerate)"
        return "spans 0"

    def _verdict(sel):
        """Read the verdict off the table rather than asserting it (2026-08-22).

        This sentence used to be hardcoded as "Only C and S have a selectivity CI
        excluding zero". When the S read-out was corrected the intervals widened and
        the caption began contradicting the rows beneath it.
        """
        excl = [disp for k, disp, _ in (r for r in _TEX_ROWS if r)
                if k in sel and note(k, sel[k]) == "excl.\\ 0"]
        if not excl:
            return ("No metric has a selectivity interval excluding zero at this "
                    "sample size, so none of them separates the two regimes here.")
        if len(excl) == 1:
            return f"Only {excl[0]} has a selectivity CI excluding zero."
        return (", ".join(excl[:-1]) + f" and {excl[-1]} have selectivity CIs "
                "excluding zero; the others do not.")

    lines = [r"% Generated by: python scripts/drift/analyze_drift_selectivity.py --primary --tex "
             + str(path),
             r"\begin{table}[t]", r"\centering",
             (r"\caption[Iterative-drift regime selectivity]{E-C: iterative-drift regime "
              r"selectivity, $\text{sel}=\Delta_{\text{SAO}}-"
              r"\Delta_{\text{ACE}}$ (inpaint minus repaint accumulation; $\Delta=$ "
              r"score$_{\text{iter1}}-$score$_{\text{iter8}}$), with two-sample bootstrap 95\% "
              r"CI over $n{=}10$ seeds. " + _verdict(sel) + "}"),
             r"\label{tab:selectivity}", r"\begin{tabular}{lrrrl}", r"\toprule",
             r"metric & $\Delta_{\text{SAO}}$ & $\Delta_{\text{ACE}}$ & sel & 95\% CI \\",
             r"\midrule"]
    for row in _TEX_ROWS:
        if row is None:
            lines.append(r"\midrule"); continue
        k, disp, ours = row
        if k not in sel:
            continue
        d = sel[k]
        s = f"{d['sel']:+.3f}"
        selcell = rf"\textbf{{{s}}}" if ours else s
        ci = rf"$[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]$ {note(k, d)}"
        lines.append(f"{disp} & {d['delta_sao']:+.3f} & {d['delta_ace']:+.3f} & {selcell} & {ci} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/drift"))
    ap.add_argument("--out", type=Path, default=Path("results/drift/drift_selectivity.json"))
    ap.add_argument("--n-boot", type=int, default=B_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--k-hi", default="8",
                    help="deepest iteration for the per-seed accumulation delta "
                         "(default 8; use 16 for the T5 depth contrast, if per_seed has it)")
    ap.add_argument("--primary", action="store_true",
                    help="also emit the C-vs-field selectivity-difference CIs "
                         "(sel(C)-sel(clap_htsat), sel(C)-sel(scs)) -- the Setup-1 counterpart "
                         "of the Setup-2 'C beats the field' claim. Written to "
                         "selectivity_differences in the output JSON.")
    ap.add_argument("--tex", type=Path, nargs="?", const=Path("results/coherence/selectivity.tex"),
                    default=None, help="also emit the E-C LaTeX table tab:selectivity "
                                       "(default results/coherence/selectivity.tex) for \\input")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    data = {m: load_model(args.results_dir, m) for m in MODELS}
    found = {m: sorted(data[m]) for m in MODELS if data[m]}
    if not found:
        print(f"no drift result files in {args.results_dir}/ (drift_htrc_<model>.json / "
              f"drift_baselines_<model>.json)", file=sys.stderr)
        return 1

    have_seed = {m: [k for k, v in data[m].items() if v.get("per_seed")] for m in MODELS}
    print("## Input inventory\n")
    for m in MODELS:
        print(f"- **{m}**: metrics={found.get(m, [])} | with per_seed: "
              f"{have_seed.get(m) or 'NONE'}")

    if not any(have_seed.values()):
        print("\n### BLOCKED: no per-seed data on disk\n")
        print("Every drift result file predates the `per_seed` dump, so it holds only\n"
              "per-iteration mean/std/n. Selectivity POINT ESTIMATES are computable from\n"
              "the aggregate `iters[k].mean` values (printed below), but every CI, the\n"
              "Spearman statistics and the ordering accuracies need seed-level scores.\n"
              "Re-run the two harnesses WITH --out (they now dump per_seed) and re-run me.\n")
        # what we can still say, from aggregates alone
        rows = []
        for metric in sorted({k for m in CONTRAST for k in data.get(m, {})}):
            d = {}
            for m in CONTRAST:
                res = data.get(m, {}).get(metric)
                if not res:
                    continue
                it = res["iters"]
                if "1" in it and "8" in it:
                    d[m] = it["1"]["mean"] - it["8"]["mean"]
            if len(d) == 2:
                rows.append((metric, d["sao"], d["ace"], d["sao"] - d["ace"]))
        if rows:
            print("| metric | Δ SAO | Δ ACE | sel = Δ_SAO − Δ_ACE |")
            print("|---|---:|---:|---:|")
            for r in sorted(rows, key=lambda x: -x[3]):
                print(f"| {r[0]} | {r[1]:+.3f} | {r[2]:+.3f} | **{r[3]:+.3f}** |")
            print("\n(point estimates only -- NO confidence intervals; n=10/regime.)")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"status": "blocked_no_per_seed",
             "note": "result files predate the per_seed dump; re-run run_drift_htrc.py and "
                     "run_drift_baselines.py with --out to regenerate",
             "inventory": {m: found.get(m, []) for m in MODELS},
             "selectivity_point_estimates_from_aggregates":
                 {r[0]: {"delta_sao": r[1], "delta_ace": r[2], "sel": r[3]} for r in rows}},
            indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
        return 0

    # ---------------- T2.1 selectivity (per-seed, bootstrapped) --------------
    metrics = sorted({k for m in CONTRAST for k in data.get(m, {})})
    deltas = {m: {k: per_seed_delta(data[m][k], k_hi=args.k_hi) for k in data.get(m, {})}
              for m in CONTRAST}
    out: dict = {"meta": {"n_boot": args.n_boot, "seed": args.seed,
                          "delta": "score(iter1) - score(iter8), per seed; + = accumulates",
                          "sel": "mean delta(SAO) - mean delta(ACE); two-sample bootstrap",
                          "caveat": "n=10/regime -> effect size + CI, never 'significant'; "
                                    "SAO vs ACE is the fair contrast (both roundtrip @ (4,6)/13s)"},
                 "selectivity": {}, "rank_stats": {}}

    print("\n## T2.1 Regime selectivity  (Δ = iter1 − iter8; + = drift accumulates)\n")
    print("| metric | Δ SAO (n) | Δ ACE (n) | sel = Δ_SAO − Δ_ACE | 95% CI |")
    print("|---|---:|---:|---:|---|")
    for k in metrics:
        a = np.array(list(deltas["sao"].get(k, {}).values()), float)
        b = np.array(list(deltas["ace"].get(k, {}).values()), float)
        if a.size == 0 or b.size == 0:
            continue
        sel = a.mean() - b.mean()
        lo, hi = boot_ci(rng, lambda r: resample(r, a).mean() - resample(r, b).mean(),
                         args.n_boot)
        out["selectivity"][k] = {"delta_sao": float(a.mean()), "n_sao": int(a.size),
                                 "delta_ace": float(b.mean()), "n_ace": int(b.size),
                                 "sel": float(sel), "ci95": [lo, hi]}
        print(f"| {k} | {a.mean():+.3f} ({a.size}) | {b.mean():+.3f} ({b.size}) | "
              f"**{sel:+.3f}** | [{lo:+.3f}, {hi:+.3f}] |")

    # Selectivity DIFFERENCE sel(a) - sel(b) -- resample each model's seeds ONCE
    # per replicate to preserve the within-model pairing across the two metrics.
    def sel_difference(a_m: str, b_m: str) -> dict | None:
        if not all(a_m in deltas[m] and b_m in deltas[m] for m in CONTRAST):
            return None
        shared = {m: sorted(set(deltas[m][a_m]) & set(deltas[m][b_m])) for m in CONTRAST}
        arrs = {m: (np.array([deltas[m][a_m][s] for s in shared[m]], float),
                    np.array([deltas[m][b_m][s] for s in shared[m]], float))
                for m in CONTRAST}
        def diff(r=None):
            tot = 0.0
            for m, sign in (("sao", +1.0), ("ace", -1.0)):
                A, B = arrs[m]
                idx = (r.integers(0, len(A), len(A)) if r is not None else np.arange(len(A)))
                tot += sign * (A[idx].mean() - B[idx].mean())
            return tot
        point = diff(None)
        lo, hi = boot_ci(rng, diff, args.n_boot)
        return {"metrics": [a_m, b_m], "point": float(point), "ci95": [lo, hi],
                "n_shared_seeds": {m: len(shared[m]) for m in CONTRAST},
                "reading": f"sel({a_m}) - sel({b_m}); >0 = {a_m} is the more regime-selective "
                           f"accumulation detector. CI excluding 0 = the case for {a_m} over "
                           f"{b_m}; CI spanning 0 = not decisive at this n, report honestly."}

    # the PRIMARY S-vs-clap contrast always; the C-vs-field contrasts under --primary
    contrasts = [PRIMARY] + (list(PRIMARY_EXTRA) if args.primary else [])
    diffs = {}
    for a_m, b_m in contrasts:
        d = sel_difference(a_m, b_m)
        if d is not None:
            diffs[f"{a_m}_minus_{b_m}"] = d
    if diffs:
        # keep the flat S-vs-clap key for backward compatibility with earlier readers
        out["selectivity_difference"] = diffs.get(f"{PRIMARY[0]}_minus_{PRIMARY[1]}")
        out["selectivity_differences"] = diffs
        print()
        for key, d in diffs.items():
            a_m, b_m = d["metrics"]
            lo, hi = d["ci95"]
            excl = "excludes 0" if (lo > 0 or hi < 0) else "SPANS 0 (not decisive at n=10)"
            print(f"**sel({a_m}) − sel({b_m}) = {d['point']:+.3f}**, 95% CI "
                  f"[{lo:+.3f}, {hi:+.3f}] — {excl}")

    # ---------------- T2.2 rank statistics ----------------------------------
    print("\n## T2.2 Rank statistics  (rho = per-seed Spearman of score vs depth; "
          "negative = coherence falls with depth)\n")
    print("| model | metric | mean rho | 95% CI | mean ordering acc | 95% CI |")
    print("|---|---|---:|---|---:|---|")
    for m in MODELS:
        for k in sorted(data.get(m, {})):
            curves = per_seed_curves(data[m][k])
            if not curves:
                continue
            rhos, accs = [], []
            for s, c in curves.items():
                d = np.array([p[0] for p in c], float)
                v = np.array([p[1] for p in c], float)
                rhos.append(spearman(d, v))
                accs.append(ordering_accuracy(c))
            r = np.array([x for x in rhos if not np.isnan(x)], float)
            ac = np.array(accs, float)
            if r.size == 0:
                continue
            rlo, rhi = boot_ci(rng, lambda rr: resample(rr, r).mean(), args.n_boot)
            alo, ahi = boot_ci(rng, lambda rr: resample(rr, ac).mean(), args.n_boot)
            out["rank_stats"].setdefault(m, {})[k] = {
                "mean_rho": float(r.mean()), "rho_ci95": [rlo, rhi], "n_seeds": int(r.size),
                "mean_ordering_acc": float(ac.mean()), "ordering_acc_ci95": [alo, ahi]}
            print(f"| {m} | {k} | {r.mean():+.3f} | [{rlo:+.3f}, {rhi:+.3f}] | "
                  f"{ac.mean():.3f} | [{alo:.3f}, {ahi:.3f}] |")

    # paired per-seed rho difference S - clap_htsat, within each model
    print("\n### Paired per-seed rho difference (S − clap_htsat; same seeds within a model)\n")
    print("| model | mean Δrho | 95% CI | n |")
    print("|---|---:|---|---:|")
    for m in MODELS:
        if not all(k in data.get(m, {}) for k in PRIMARY):
            continue
        cs = {k: per_seed_curves(data[m][k]) for k in PRIMARY}
        shared = sorted(set(cs[PRIMARY[0]]) & set(cs[PRIMARY[1]]))
        if not shared:
            continue
        diffs = []
        for s in shared:
            rr = []
            for k in PRIMARY:
                c = cs[k][s]
                rr.append(spearman(np.array([p[0] for p in c], float),
                                   np.array([p[1] for p in c], float)))
            if not any(np.isnan(x) for x in rr):
                diffs.append(rr[0] - rr[1])
        if not diffs:
            continue
        d = np.array(diffs, float)
        lo, hi = boot_ci(rng, lambda r: resample(r, d).mean(), args.n_boot)
        out["rank_stats"].setdefault(m, {})["rho_diff_S_minus_clap_htsat"] = {
            "mean": float(d.mean()), "ci95": [lo, hi], "n": int(d.size)}
        print(f"| {m} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {d.size} |")

    print("\n_n=10 seeds/regime: read effect sizes + CIs, not p-values (§3). "
          "Cross-model magnitudes are not comparable (§8.3); SAO-vs-ACE is the fair "
          "contrast because both are roundtrip at region (4,6)/13 s._")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    if args.tex:
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        write_selectivity_tex(out["selectivity"], args.tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
