#!/usr/bin/env python
"""One command to take a fresh Qualtrics export all the way to the thesis numbers.

Collection closed on 2026-08-30 at N=40. This chains
the four steps that were previously run by hand, in the order they depend on each other:

  1. qualtrics_to_responses.py   export -> per-session JSONs for both parts
  2. score_mos_responses.py      Part 1 -> mos_live.csv + qc_live.json
  3. score_ab_session2.py        Part 2 -> session2/ab_results.json
  4. compute_C.py --fit-mos      weights + held-out correlations -> C_mosfit_live.*

Then prints the handful of numbers Sec. 4.10.4 of the dissertation quotes, so the
subsection can be updated against a single block of output rather than four.

`--out` is passed to compute_C.py deliberately: without it the fit overwrites
results/coherence/C.json, which is the aggregate every other table depends on.

Usage (DSP env, from the repo root):
    python scripts/mos/refresh_live.py                          # N40.csv, the closed set
    python scripts/mos/refresh_live.py --csv results/mos/final/N40.csv
    python scripts/mos/refresh_live.py --dry-run                # show the commands only
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)


def newest_export() -> Path:
    """The closed dataset, falling back to the newest incremental export.

    Collection closed on 2026-08-30 and N40.csv is the authoritative file, so it wins
    whenever it is present. The incremental result_*.csv exports are kept on disk as the
    collection audit trail but are untracked, so a release checkout has none of them and
    the glob below is a local convenience rather than the normal path.
    """
    d = ROOT / "results/mos/final"
    final = d / "N40.csv"
    if final.is_file():
        return final
    cands = sorted(d.glob("result_*.csv"),
                   key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not cands:
        raise SystemExit(f"no N40.csv or result_*.csv under {d}")
    return cands[-1]


def run(cmd: list[str], dry: bool) -> None:
    print(f"\n$ {' '.join(cmd)}")
    if dry:
        return
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode:
        raise SystemExit(f"step failed: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="Qualtrics export; default = newest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    csv_path = Path(a.csv) if a.csv else newest_export()
    py = sys.executable
    print(f"refreshing from {csv_path}")

    run([py, "scripts/mos/qualtrics_to_responses.py", "--csv", str(csv_path)], a.dry_run)
    run([py, "scripts/mos/score_mos_responses.py",
         "--responses", "results/mos/responses/*.json",
         "--out", "results/mos/mos_live.csv",
         "--qc-out", "results/mos/qc_live.json"], a.dry_run)
    run([py, "scripts/mos/score_ab_session2.py",
         "--responses", "results/mos/session2/responses/*.json",
         "--out", "results/mos/session2/ab_results.json"], a.dry_run)
    run([py, "scripts/core/compute_C.py",
         "--scores", "results/coherence/pair_scores.csv",
         "--fit-mos", "results/mos/mos_live.csv",
         "--out", "results/mos/C_mosfit_live"], a.dry_run)
    if a.dry_run:
        return

    qc = json.loads((ROOT / "results/mos/qc_live.json").read_text())
    ab = json.loads((ROOT / "results/mos/session2/ab_results.json").read_text())
    fit = json.loads((ROOT / "results/mos/C_mosfit_live.json").read_text())

    print("\n" + "=" * 66)
    print("numbers quoted in dissertation Sec. 4.10.4 (update them from here)")
    print("=" * 66)
    ac = qc["attention_checks"]["per_rater"]
    fails = [k for k, v in ac.items() if not v.get("passed")]
    su = qc["scale_usage"]["usage_fraction"]
    print(f"  sessions            {qc['n_kept']} of a target 40 "
          f"(supervisor's gold standard 30)")
    print(f"  attention failures  {len(fails)}")
    print(f"  ICC(2,1) anchor     {qc['anchor_icc']['icc_2_1']}")
    print("  scale usage         " + ", ".join(f"{100*su[k]:.0f}%" for k in "12345"))

    ho = fit["mos_crossval"]["held_out"]
    print("  Part 1, held-out r (source-grouped CV):")
    for m, lab in (("C_uniform", "C"), ("C_fitted_noS", "C_noS"),
                   ("clap_htsat_raw", "CLAP-htsat"), ("scs_raw", "SCS"),
                   ("cocola_raw", "COCOLA")):
        if m in ho:
            print(f"    {lab:<12} {ho[m]['pearson_r']:.3f}")
    if "C_uniform" in ho and "C_fitted_noS" in ho:
        d = ho["C_uniform"]["pearson_r"] - ho["C_fitted_noS"]["pearson_r"]
        print(f"    S contributes +{d:.3f}")
    w = fit["weights"]
    print("  fitted weights      " + "  ".join(f"{k}={w[k]:.3f}" for k in "HTRS" if k in w))

    print("  Part 2, pooled / position-adjusted:")
    for k, v in ab["contrasts"].items():
        print(f"    {k:<12} {v['win_rate']:.3f} / "
              f"{v.get('win_rate_position_adjusted', float('nan')):.3f}  "
              f"CI [{v['ci95_cluster_by_seed'][0]:.3f},{v['ci95_cluster_by_seed'][1]:.3f}]"
              f"  posOR {v.get('position_odds_ratio', float('nan')):.2f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
