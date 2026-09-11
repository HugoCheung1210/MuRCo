#!/usr/bin/env python
"""Rating files -> per-pair MOS + quality control. DSP env, read-only inputs.

Consumes response JSONs (from the local pilot harness, or a Qualtrics export
converted to the same shape) and produces the `pair_id,mos` CSV that
`compute_C.py --fit-mos` reads, plus the QC the protocol §8 promises:

  - attention checks: controls should score high, cross-genre swaps low. Raters
    failing them are reported and (with --drop-failed) excluded before the MOS.
  - ICC(2,1) and ICC(2,k) on the ANCHOR set — the only pairs every rater judged,
    so the only place a common-set agreement estimate is defined.
  - ratings-per-pair coverage, so a thin cell is visible before analysis.

Part 1 is the perturbation matrix only. `--drop-sets` (default `ai_rerank`) excludes
stimuli that are no longer in scope, counting them into `excluded_stimulus_sets` in the
QC rather than dropping them quietly; the pilot session of 2026-08-07 predates the
removal and still carries five such ratings. `--drop-sets ''` scores everything.

Response JSON shape (one file per session):
  {"session_code": "...", "block": "block1",
   "responses": [{"stimulus_id": "stim_0001", "rating": 4}, ...]}

Usage:
  python scripts/mos/score_mos_responses.py --responses 'results/mos/responses/*.json'
  python scripts/mos/score_mos_responses.py --responses '...' --drop-failed --out results/mos.csv
"""
import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)


def icc_two_way(matrix):
    """ICC(2,1) and ICC(2,k), two-way random effects, absolute agreement.

    matrix: n targets x k raters, complete. Returns (icc_1, icc_k) or (nan, nan)
    if the design is too small/degenerate to estimate.
    """
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan"), float("nan")
    gm = x.mean()
    ms_r = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)          # between targets
    ms_c = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)          # between raters
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    if ms_e <= 0:
        return float("nan"), float("nan")
    icc1 = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
    icck = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / n)
    return float(icc1), float(icck)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True, help="glob of response JSON files")
    ap.add_argument("--index", default="results/mos/stimulus_index.csv")
    ap.add_argument("--out", default="results/mos.csv")
    ap.add_argument("--qc-out", default="results/mos_qc.json")
    ap.add_argument("--drop-failed", action="store_true",
                    help="exclude raters failing the attention checks from the MOS")
    ap.add_argument("--drop-sets", default="ai_rerank",
                    help="comma list of stimulus_set values to exclude from Part 1 "
                         "scoring, or '' to score everything. Default drops ai_rerank: "
                         "that arm left Part 1 on 2026-08-07 "
                         "(doc/notes/ai_rerank_removed_from_part1.md), but the pilot "
                         "session still contains ratings of it")
    # PRE-SPECIFIED 2026-08-10, before recruitment (1 response collected, not yet scored).
    # Rationale and calibration: doc/mos/attention_check_prespecification.md.
    #
    # The rule is the CONTRAST between the two checks, not two absolute cutoffs. Raters
    # differ in how much of the 1-5 scale they use, and an absolute cutoff scores that
    # bias rather than attention: a rater who never says 5 fails `control_mean >= 4.0`
    # while ordering every pair correctly. The difference cancels the bias, and on
    # simulated raters it is the difference between 23.4% and 0.8% false exclusion.
    ap.add_argument("--qc-rule", choices=("contrast", "absolute"), default="contrast",
                    help="contrast (pre-specified): fail if control_mean - swap_mean < "
                         "--min-contrast. absolute: the pre-2026-08-10 two-threshold rule")
    ap.add_argument("--min-contrast", type=float, default=0.75,
                    help="minimum control_mean - swap_mean a rater must show")
    ap.add_argument("--high-min", type=float, default=4.0,
                    help="[--qc-rule absolute] mean rating a rater must EXCEED on controls")
    ap.add_argument("--low-max", type=float, default=2.5,
                    help="[--qc-rule absolute] mean rating a rater must stay UNDER on swaps")
    args = ap.parse_args()
    drop_sets = {x.strip() for x in args.drop_sets.split(",") if x.strip()}

    index = {r["stimulus_id"]: r for r in csv.DictReader(open(ROOT / args.index))}
    files = sorted(glob.glob(args.responses))
    if not files:
        raise SystemExit(f"no response files matched {args.responses!r}")

    sessions, unknown = {}, set()
    off_scope = Counter()
    for f in files:
        d = json.load(open(f))
        rs, skipped = [], 0
        for r in d["responses"]:
            if r["stimulus_id"] not in index:
                unknown.add(r["stimulus_id"])
                continue
            if index[r["stimulus_id"]]["stimulus_set"] in drop_sets:
                # Out of Part 1's scope since 2026-08-07: the AI-rerank arm was removed
                # because this scale measures change magnitude and both arms of a rerank
                # pair are fully regenerated (doc/notes/ai_rerank_removed_from_part1.md).
                # Ratings collected before that still sit in the pilot session, so they
                # are dropped HERE, counted, and reported -- not silently ignored.
                off_scope[index[r["stimulus_id"]]["stimulus_set"]] += 1
                continue
            if r.get("skipped") or r.get("rating") is None:
                skipped += 1          # uncomfortable-skip: excluded, never counted as data
                continue
            rs.append(r)
        sessions[d["session_code"]] = {"block": d.get("block", ""), "responses": rs,
                                       "file": f, "dry_run": d.get("dry_run", False),
                                       "n_skipped": skipped}

    # --- attention checks -------------------------------------------------
    qc_raters = {}
    for code, s in sessions.items():
        hi = [r["rating"] for r in s["responses"]
              if index[r["stimulus_id"]]["attention_check"] == "expect_high"]
        lo = [r["rating"] for r in s["responses"]
              if index[r["stimulus_id"]]["attention_check"] == "expect_low"]
        mh = float(np.mean(hi)) if hi else float("nan")
        ml = float(np.mean(lo)) if lo else float("nan")
        contrast = mh - ml
        if args.qc_rule == "contrast":
            # A rater missing either check cannot be assessed, and is kept rather than
            # dropped on absent evidence.
            passed = bool(np.isnan(contrast) or contrast >= args.min_contrast)
        else:
            passed = bool((np.isnan(mh) or mh >= args.high_min) and
                          (np.isnan(ml) or ml <= args.low_max))
        qc_raters[code] = {"block": s["block"], "n_ratings": len(s["responses"]),
                           "control_mean": round(mh, 2), "swap_mean": round(ml, 2),
                           "contrast": None if np.isnan(contrast) else round(contrast, 2),
                           "n_control_checks": len(hi), "n_swap_checks": len(lo),
                           "passed": passed}
    failed = [c for c, v in qc_raters.items() if not v["passed"]]
    keep = [c for c in sessions if not (args.drop_failed and c in failed)]

    # --- per-pair MOS -----------------------------------------------------
    by_stim = defaultdict(list)
    for c in keep:
        for r in sessions[c]["responses"]:
            by_stim[r["stimulus_id"]].append(r["rating"])
    rows = []
    for sid, vals in sorted(by_stim.items()):
        rows.append({"pair_id": index[sid]["pair_id"], "mos": round(float(np.mean(vals)), 4),
                     "n_ratings": len(vals), "sd": round(float(np.std(vals, ddof=1)), 3)
                     if len(vals) > 1 else "", "stimulus_id": sid,
                     "stimulus_set": index[sid]["stimulus_set"],
                     "condition": index[sid]["condition"]})
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pair_id", "mos"])   # exactly what --fit-mos wants
        w.writeheader()
        for r in rows:
            w.writerow({"pair_id": r["pair_id"], "mos": r["mos"]})
    with open(str(out).replace(".csv", "_detail.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- ICC on the anchor set -------------------------------------------
    anchors = [s for s, r in index.items() if r["block"] == "anchor"]
    mat, used = [], []
    for sid in sorted(anchors):
        col = {c: next((r["rating"] for r in sessions[c]["responses"]
                        if r["stimulus_id"] == sid), None) for c in keep}
        if all(v is not None for v in col.values()):
            mat.append([col[c] for c in keep])
            used.append(sid)
    icc1, icck = icc_two_way(mat) if mat else (float("nan"), float("nan"))

    # --- scale-usage diagnostic -------------------------------------------
    # The instrument's failure mode is COLLAPSE: because every perturbation preserves
    # musical identity, an identity-framed question gets 5 for everything except a
    # style swap, and the MOS goes bimodal. C resolves six graded levels here, so a
    # two-level MOS would produce a weak correlation that says nothing about C.
    # Check it directly: is every scale point used, and does the mean track magnitude?
    all_r = [r["rating"] for c in keep for r in sessions[c]["responses"]]
    usage = {v: round(all_r.count(v) / len(all_r), 4) for v in (1, 2, 3, 4, 5)}
    by_cond = defaultdict(list)
    for c in keep:
        for r in sessions[c]["responses"]:
            idx = index[r["stimulus_id"]]
            by_cond[idx["condition"]].append(r["rating"])
    unused = [v for v, f in usage.items() if f < 0.05]
    scale = {
        "usage_fraction": usage,
        "underused_points": unused,
        "collapsed": bool(unused),
        "mean_by_condition": {k: round(float(np.mean(v)), 2)
                              for k, v in sorted(by_cond.items())},
        "note": "a point used <5% of the time means raters aren't resolving that level; "
                "if 4 is unused the question is being read as identity, not consistency",
    }

    n_per = [r["n_ratings"] for r in rows]
    qc = {
        "scale_usage": scale,
        "skipped_total": sum(v["n_skipped"] for v in sessions.values()),
        "skipped_per_session": {k: v["n_skipped"] for k, v in sessions.items()
                                if v["n_skipped"]},
        "n_sessions": len(sessions), "n_kept": len(keep),
        "dry_run_files": sum(1 for s in sessions.values() if s["dry_run"]),
        "attention_checks": {"failed": failed,
                             "rule": args.qc_rule,
                             "prespecified": "2026-08-10, doc/mos/attention_check_prespecification.md",
                             "thresholds":
                             {"min_contrast": args.min_contrast} if args.qc_rule == "contrast"
                             else {"control_mean_min": args.high_min,
                                   "swap_mean_max": args.low_max},
                             "per_rater": qc_raters},
        "pairs_with_ratings": len(rows),
        "ratings_per_pair": {"min": int(min(n_per)), "median": float(np.median(n_per)),
                             "max": int(max(n_per))},
        "anchor_icc": {"n_anchor_complete": len(used), "n_raters": len(keep),
                       "icc_2_1": round(icc1, 3) if icc1 == icc1 else None,
                       "icc_2_k": round(icck, 3) if icck == icck else None},
        "unknown_stimulus_ids": sorted(unknown),
        "excluded_stimulus_sets": dict(off_scope),
        "outputs": {"mos": str(out), "detail": str(out).replace(".csv", "_detail.csv")},
    }
    json.dump(qc, open(ROOT / args.qc_out, "w"), indent=1)
    print("scale usage:", {v: f"{f:.0%}" for v, f in usage.items()})
    if unused:
        print(f"  ** SCALE COLLAPSE: point(s) {unused} used <5% — raters are probably "
              f"answering 'is it the same piece?' rather than 'is anything different?'")
    else:
        print("  all five points in use")
    print("mean rating by condition:")
    for k, v in scale["mean_by_condition"].items():
        print(f"  {k:24} {v:.2f}")
    print()
    print(json.dumps({k: v for k, v in qc.items() if k != "scale_usage"}, indent=1)[:1400])
    if any(s["dry_run"] for s in sessions.values()):
        print("\nNOTE: dry-run responses included — not participant data.")
    print(f"\nwrote {out} ({len(rows)} pairs) + QC to {args.qc_out}")


if __name__ == "__main__":
    main()
