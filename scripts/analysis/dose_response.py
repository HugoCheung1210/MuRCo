#!/usr/bin/env python3
"""E-B -- severity dose-response (monotonicity).

Tests whether the metric tracks the DEGREE of incoherence, not just its
presence.  Each perturbation family is graded -- pitch {1,2,4} semitones,
stretch {5,10,20}%, lowpass {8k,4k,2k} Hz, distortion {4,12,32} drive -- so a
valid coherence metric should fall monotonically as severity rises.

METHOD (consistent with the paired/within-track design used everywhere else):
per family, per metric m in {C,H,T,R,S,clap_htsat,scs,cbase}, the Spearman rho
between severity and score is computed WITHIN each track, then aggregated across
tracks (mean rho + bootstrap CI over tracks).  Severity is defined so LARGER =
more perturbed, so a metric that tracks degree gives NEGATIVE rho (score falls
as severity rises); "more negative" = cleaner dose-response.  For pitch/stretch
the sign is collapsed to |magnitude| (a +2 and -2 semitone shift are the same
severity); the signed-directionality version is emitted too, in case direction
is interesting.  A track whose metric is constant within a family (e.g. R
abstains on pitch, std 0 -> rho undefined) is EXCLUDED from that cell and the
contributing-track count is reported -- an abstention is a real read-out, not a
value to impute.

Read-out (plan §1 E-B): if C is more monotone than clap_htsat on even a subset
of families that is a clean win row; if not it is still the validity check
reviewers expect -- reported either way, effect sizes + CIs, never p (§3).

Inputs (no GPU): results/coherence/pair_scores_cbase.csv (score_H/T/R/S + score_scs/
score_clap_htsat/score_cbase); C is computed on the fly as the uniform geometric
mean of H/T/R/S (matching compute_C.py).  Output: results/diagnostics/dose_response.json +
a printed table.

Usage (DSP/local env, from repo root):
    python scripts/analysis/make_combined_baseline.py     # produces pair_scores_cbase.csv
    python scripts/analysis/dose_response.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EPS = 1e-6
SEED = 20260717
B_BOOT = 10000

# family -> (row filter, severity function on magnitude).  Larger severity = more
# perturbed.  lowpass cutoff falls with severity, so severity = -cutoff.
FAMILIES = {
    "pitch_shift":  lambda mag: abs(mag),
    "time_stretch": lambda mag: abs(mag),
    "lowpass":      lambda mag: -mag,      # 8000<4000<2000 Hz -> increasing severity
    "distortion":   lambda mag: mag,       # 4<12<32 drive
}
METRICS = ["C", "H", "T", "R", "S", "clap_htsat", "scs", "cbase"]
COL = {"H": "score_H", "T": "score_T", "R": "score_R", "S": "score_S",
       "clap_htsat": "score_clap_htsat", "scs": "score_scs", "cbase": "score_cbase"}


def load_rows(results: Path) -> list[dict]:
    p = results / "pair_scores_cbase.csv"
    if not p.is_file():
        raise SystemExit("results/coherence/pair_scores_cbase.csv not found -- run "
                         "scripts/analysis/make_combined_baseline.py first (E-C).")
    return list(csv.DictReader(p.open(newline="", encoding="utf-8")))


def metric_value(r: dict, m: str) -> float:
    if m == "C":
        vals = [float(np.clip(float(r[COL[d]]), EPS, 1.0)) for d in ("H", "T", "R", "S")]
        return float(np.prod(vals) ** (1.0 / len(vals)))
    v = r.get(COL[m], "nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def within_track_rho(rows, fam, sevfn, metric, signed):
    """{track: rho} for one family/metric; sign collapsed to |mag| unless signed."""
    by_track = defaultdict(list)
    for r in rows:
        if r["perturbation"] != fam:
            continue
        mag = float(r["magnitude"])
        sev = mag if signed else sevfn(mag)
        by_track[r["source_id"]].append((sev, metric_value(r, metric)))
    rhos = {}
    for tr, pts in by_track.items():
        sev = np.array([p[0] for p in pts], float)
        val = np.array([p[1] for p in pts], float)
        ok = ~np.isnan(val)
        sev, val = sev[ok], val[ok]
        if len(val) < 3 or np.std(val) < EPS or len(np.unique(sev)) < 2:
            continue  # abstention / no variation -> undefined, excluded honestly
        rho = spearmanr(sev, val).correlation
        if not np.isnan(rho):
            rhos[tr] = float(rho)
    return rhos


def boot_ci(rng, arr, n_rep=B_BOOT, alpha=0.05):
    if len(arr) < 2:
        return (float("nan"), float("nan"))
    a = np.asarray(arr, float)
    stats = np.array([a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_rep)])
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


# dissertation display names for the LaTeX table (tab:dose).
_TEX_METRIC = {"C": "$C$", "H": "$H$", "T": "$T$", "R": "$R$", "S": "$S$",
               "clap_htsat": "CLAP-htsat", "scs": "SCS", "cbase": r"SCS$\times$CLAP"}
_TEX_FAMILY = {"pitch_shift": "pitch-shift", "time_stretch": "time-stretch",
               "lowpass": "low-pass", "distortion": "distortion"}


def write_dose_tex(out: dict, metrics: list, path: Path) -> None:
    """Emit tab:dose to \\input into the dissertation (matches doc/writeup/dissertation.tex).

    No cell is bolded. Bolding C where it beats CLAP-htsat reads as "best in row", which
    is false on pitch and time-stretch, where SCS and SCS x CLAP are more monotone than C.
    The caption states the comparison instead."""
    dr = out["dose_response"]
    cols = "l" + "r" * len(metrics)
    lines = [r"% Generated by: python scripts/analysis/dose_response.py --tex " + str(path),
             r"\begin{table}[t]", r"\centering",
             (r"\caption[Severity dose-response by family]{E-B: severity dose-response. "
              r"Mean within-track Spearman "
              r"$\rho$(severity, score) per family, where more negative is cleaner monotone "
              r"tracking.}"),
             r"\label{tab:dose}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             "family & " + " & ".join(_TEX_METRIC.get(m, m) for m in metrics) + r" \\",
             r"\midrule"]
    for fam in FAMILIES:
        if fam not in dr:
            continue
        cells = []
        for m in metrics:
            v = dr[fam].get(m, {}).get("mean_rho")
            if v is None:
                cells.append("n/a"); continue
            cells.append(f"{v:.2f}")
        lines.append(f"{_TEX_FAMILY.get(fam, fam)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/coherence"))
    ap.add_argument("--out", type=Path, default=Path("results/diagnostics/dose_response.json"))
    ap.add_argument("--tex", type=Path, nargs="?", const=Path("results/diagnostics/dose_response.tex"),
                    default=None, help="also emit the LaTeX table tab:dose "
                                       "(default results/diagnostics/dose_response.tex) for \\input")
    ap.add_argument("--signed", action="store_true",
                    help="use signed magnitude for pitch/stretch (directionality) "
                         "instead of |magnitude|")
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    metrics = [m for m in METRICS if m in ("C",) or COL[m] in rows[0]]
    rng = np.random.default_rng(SEED)

    out = {"meta": {"stat": "within-track Spearman rho(severity, score), aggregated over "
                            "tracks; severity larger = more perturbed, so NEGATIVE rho = "
                            "metric tracks degree ('more negative = cleaner dose-response')",
                    "sign": "|magnitude|" if not args.signed else "signed magnitude",
                    "n_boot": B_BOOT, "seed": SEED,
                    "note": "tracks with a constant metric in a family (rho undefined, e.g. R "
                            "abstaining on pitch) are excluded from that cell; n_tracks shown"},
           "dose_response": {}}

    print(f"# E-B severity dose-response  (mean within-track Spearman rho; "
          f"severity larger=more perturbed -> negative=monotone)")
    print(f"sign handling: {'|magnitude|' if not args.signed else 'signed'}\n")
    header = "| family | " + " | ".join(metrics) + " |"
    print(header)
    print("|---|" + "---:|" * len(metrics))

    for fam, sevfn in FAMILIES.items():
        out["dose_response"][fam] = {}
        cells = []
        for m in metrics:
            rhos = within_track_rho(rows, fam, sevfn, m, args.signed)
            vals = np.array(list(rhos.values()), float)
            if len(vals) == 0:
                out["dose_response"][fam][m] = {"mean_rho": None, "n_tracks": 0}
                cells.append("n/a")
                continue
            mean = float(vals.mean())
            lo, hi = boot_ci(rng, vals)
            out["dose_response"][fam][m] = {"mean_rho": mean, "ci95": [lo, hi],
                                            "n_tracks": int(len(vals))}
            cells.append(f"{mean:+.2f}")
        print(f"| {fam} | " + " | ".join(cells) + " |")

    # C-vs-clap head-to-head per family (the "clean win row" check)
    print("\n## C vs clap_htsat per family (mean rho; more negative = more monotone)")
    print("| family | C | clap_htsat | C more monotone? |")
    print("|---|---:|---:|---|")
    for fam in FAMILIES:
        c = out["dose_response"][fam].get("C", {}).get("mean_rho")
        cl = out["dose_response"][fam].get("clap_htsat", {}).get("mean_rho")
        if c is None or cl is None:
            continue
        verdict = "yes" if c < cl else "no"
        out["dose_response"][fam].setdefault("_contrast", {})["C_minus_clap_htsat"] = float(c - cl)
        print(f"| {fam} | {c:+.2f} | {cl:+.2f} | {verdict} ({c-cl:+.2f}) |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.tex:
        if args.signed:
            print("note: --tex reflects the --signed run; the dissertation table is the "
                  "|magnitude| version (run without --signed)")
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        write_dose_tex(out, metrics, args.tex)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
