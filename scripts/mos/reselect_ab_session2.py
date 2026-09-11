#!/usr/bin/env python
"""Re-select the session-2 A/B trials to spend a FIXED budget on audible contrasts.

Why re-select. `select_ab_trials.py` runs a greedy set cover, which minimises the
NUMBER of comparisons. That is the wrong objective here: the trial budget is fixed by
session length, so returning fewer trials does not buy anything, it just leaves budget
unspent. Meanwhile the self-pilot (results/mos/cocola_pilot) found that preference is at
or below chance when C's margin is under about 0.025 and rises monotonically above it
(corr(margin, preference) = +0.60, 95% CI [+0.39, +0.77]), so low-margin trials
contribute noise rather than signal. In the shipped 64-trial set, 16 trials (25%) sit
below that threshold.

Raising `--min-margin` globally fixes the margin but guts C_noS, which is the arm that
decides whether S earns its place: coverage falls 28 -> 19 -> 17 as the threshold goes
0.02 -> 0.025 -> 0.03. C_noS is not low-margin on average (median 0.030 against
0.032-0.036 for the others); it simply has fewer opposed seeds to draw on, so margin
filtering removes whole seeds.

So: take the high-margin set cover as the base, then spend the leftover budget topping
up the thinnest contrast with its best remaining trials. Same number of trials, higher
median margin, and C_noS protected rather than sacrificed.

    python scripts/mos/reselect_ab_session2.py                 # preview the trade-off
    python scripts/mos/reselect_ab_session2.py --write         # write ab_trials.json

Nothing downstream changes: the output is the same schema `prepare_ab_session2.py`
already consumes, and the trial count is unchanged, so session length and the ethics
amendment are untouched.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
SELECT = ROOT / "scripts/mos/select_ab_trials.py"


def run_select(min_margin: float, rivals: str) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = Path(fh.name)
    subprocess.run([sys.executable, str(SELECT), "--min-margin", str(min_margin),
                    "--rivals", rivals, "--out", str(out)],
                   cwd=ROOT, capture_output=True, check=True)
    trials = json.loads(out.read_text())["trials"]
    out.unlink(missing_ok=True)
    return trials


def coverage(trials: list[dict]) -> Counter:
    return Counter(s for t in trials for s in t["settles"])


def summarise(name: str, trials: list[dict]) -> None:
    m = np.array([t["c_margin"] for t in trials], dtype=float)
    cov = coverage(trials)
    print(f"  {name:22s} {len(trials):3d} trials | median margin {np.median(m):.3f} | "
          f"<0.025: {int((m < 0.025).sum()):2d} ({100 * (m < 0.025).mean():3.0f}%) | "
          + "  ".join(f"{k}={v}" for k, v in sorted(cov.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=64, help="trial count to fill (unchanged)")
    ap.add_argument("--base-margin", type=float, default=0.03,
                    help="threshold for the main set cover")
    ap.add_argument("--topup-margin", type=float, default=0.02,
                    help="floor for the top-up trials that protect the thinnest arm")
    ap.add_argument("--protect", default="C_noS",
                    help="contrast whose coverage must not regress")
    ap.add_argument("--rivals", default="clap_htsat,clap_music,scs,C_noS")
    ap.add_argument("--write", action="store_true",
                    help="write results/rerank_ace/ab_trials.json (default: preview only)")
    args = ap.parse_args()

    current = run_select(0.02, args.rivals)
    base = run_select(args.base_margin, args.rivals)

    # Top-up pool: everything the protected arm can still offer above the floor,
    # best margin first. Dedupe on the candidate PAIR, not the seed: the design already
    # runs up to 3 distinct opposed pairs from one seed, so excluding whole seeds would
    # throw away most of the pool.
    def ident(t: dict) -> tuple:
        return (t["seed"], t["c_pick"], t["rival_pick"])

    used = {ident(t) for t in base}
    pool = [t for t in run_select(args.topup_margin, args.protect) if ident(t) not in used]
    pool.sort(key=lambda t: -t["c_margin"])

    final = list(base)
    target = coverage(current)[args.protect]
    for t in pool:
        if len(final) >= args.budget:
            break
        if coverage(final)[args.protect] >= target:
            break
        final.append(t)
        used.add(ident(t))

    # Any budget still spare goes to the highest-margin unused trials overall.
    if len(final) < args.budget:
        extra = [t for t in run_select(args.topup_margin, args.rivals)
                 if ident(t) not in used]
        extra.sort(key=lambda t: -t["c_margin"])
        for t in extra[:args.budget - len(final)]:
            final.append(t)
            used.add(ident(t))

    print("session-2 trial selection\n")
    summarise(f"current (mm 0.02)", current)
    summarise(f"raise only (mm {args.base_margin})", base)
    summarise("reselected", final)

    cc, cf = coverage(current), coverage(final)
    print(f"\n  coverage change vs current:")
    for k in sorted(set(cc) | set(cf)):
        d = cf[k] - cc[k]
        print(f"    {k:12s} {cc[k]:3d} -> {cf[k]:3d}  ({d:+d})"
              + ("   <- protected" if k == args.protect else ""))

    mc = np.array([t["c_margin"] for t in current])
    mf = np.array([t["c_margin"] for t in final])
    print(f"\n  median margin {np.median(mc):.3f} -> {np.median(mf):.3f}"
          f"   |  trials below 0.025: {int((mc < 0.025).sum())} -> {int((mf < 0.025).sum())}")

    if args.write:
        out = ROOT / "results/rerank_ace/ab_trials.json"
        payload = {"meta": {"rivals": args.rivals.split(","),
                            "budget": args.budget,
                            "base_margin": args.base_margin,
                            "topup_margin": args.topup_margin,
                            "protected": args.protect,
                            "n_trials": len(final),
                            "trials_per_contrast": dict(sorted(cf.items())),
                            "median_c_margin": float(np.median(mf)),
                            "design": "set cover at base_margin, then budget spent topping "
                                      "up the protected contrast; fixed trial count",
                            "rationale": "self-pilot found preference ~chance below "
                                         "|dC| 0.025 and monotone above it"},
                   "trials": final}
        out.write_text(json.dumps(payload, indent=1))
        print(f"\n[wrote {out}]")
        print("next: python scripts/mos/prepare_ab_session2.py   (re-renders session-2 audio)")
    else:
        print("\n(preview only; pass --write to commit the selection)")


if __name__ == "__main__":
    main()
