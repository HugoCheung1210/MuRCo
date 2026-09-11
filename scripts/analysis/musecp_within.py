#!/usr/bin/env python
"""Table 3's row for MuseCPEval: pooled and within-condition correlation with the MOS.

Reuses review_fixes.analysis_bc's construction exactly, so the numbers sit in the same
column as the published C, C_noS, CLAP and SCS rows.  The script asserts that it
reproduces those four before reporting the new one; if the assertion fires, the table and
this script have drifted apart and the table is the one to trust.

"Within" centres both the ratings and the metric on their own condition mean, which
removes the between-condition spread that a pooled correlation partly measures.

Usage (DSP env, from the repo root)::

    python scripts/analysis/musecp_within.py --out results/musecp/part1_within.json
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
try:  # sibling imports across script groups (see scripts/README.md)
    sys.path.insert(0, str(_here.parent.parent))
    import _paths  # noqa: F401
except ImportError:
    pass

import review_fixes as rf

N_BOOT = 20000
PUBLISHED = {"C": 0.788, "C_noS": 0.748, "clap": 0.689, "scs": 0.167,
             "cocola": -0.001}


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--musecp", default="results/musecp/setup1/musecp_metrics.csv")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="results/musecp/part1_within.json")
    args = ap.parse_args()

    with open(rf.ROOT / args.musecp) as fh:
        mcp = {r["pair_id"]: r for r in csv.DictReader(fh)}

    by_stim, index = rf.load_mos_ratings()
    scores = {r["pair_id"]: r for r in
              rf.read_csv(rf.ROOT / "results/coherence/pair_scores_cbase.csv")}
    coc = {r["pair_id"]: float(r["cocola"]) for r in
           rf.read_csv(rf.ROOT / "results/baselines/cocola_scores.csv") if r.get("cocola")}

    recs = []
    for stim, ratings in by_stim.items():
        meta = index.get(stim)
        if meta is None or meta["pair_id"] not in scores:
            continue
        pid = meta["pair_id"]
        s = scores[pid]
        H, T, R, S = (float(s["score_H"]), float(s["score_T"]),
                      float(s["score_R"]), float(s["score_S"]))
        rec = {"pair_id": pid, "source_id": meta["source_id"],
               "condition": meta["condition"],
               "mos": float(np.mean([v for _, v in ratings])),
               "C": (H * T * R * S) ** 0.25, "C_noS": (H * T * R) ** (1 / 3),
               "clap": float(s["score_clap_htsat"]), "scs": float(s["score_scs"]),
               "cocola": coc.get(pid, float("nan"))}
        if pid in mcp:
            rec["musecp12"] = float(mcp[pid]["musecp_mean12"])
            rec["musecp10"] = float(mcp[pid]["musecp_mean10"])
        recs.append(rec)

    import math
    recs = [r for r in recs if "musecp12" in r and not math.isnan(r["cocola"])]
    print(f"{len(recs)} pairs, {len(set(r['source_id'] for r in recs))} tracks, "
          f"{len(set(r['condition'] for r in recs))} conditions")

    keys = ["C", "C_noS", "clap", "scs", "cocola", "musecp12", "musecp10"]
    mos = np.array([r["mos"] for r in recs])
    conds = np.array([r["condition"] for r in recs])
    metrics = {k: np.array([r[k] for r in recs]) for k in keys}

    def centre(v, mask_conds=None, sel=None):
        vv = (v if sel is None else v[sel]).astype(float).copy()
        cs = conds if sel is None else conds[sel]
        for c in np.unique(cs):
            m = cs == c
            if m.sum() > 1:
                vv[m] -= vv[m].mean()
        return vv

    pooled = {k: round(pearson(mos, v), 4) for k, v in metrics.items()}
    mos_w = centre(mos)
    within = {k: round(pearson(mos_w, centre(v)), 4) for k, v in metrics.items()}

    print("\n            pooled   within")
    for k in keys:
        print(f"  {k:>9}  {pooled[k]:+.4f}  {within[k]:+.4f}")

    # the assertion: this construction must reproduce the published rows
    bad = [k for k, want in PUBLISHED.items() if abs(pooled[k] - want) > 0.002]
    if bad:
        raise SystemExit(f"does not reproduce the published pooled r for {bad}: "
                         f"{ {k: pooled[k] for k in bad} }")
    print("\nreproduces the published C / C_noS / CLAP / SCS pooled rows")

    # cluster bootstrap over source tracks on the contrasts
    tracks = sorted(set(r["source_id"] for r in recs))
    by_tr = {t: np.array([i for i, r in enumerate(recs) if r["source_id"] == t])
             for t in tracks}
    rng = np.random.default_rng(0)
    diffs = defaultdict(list)
    pairs = [("C", "musecp12"), ("C", "musecp10"), ("C_noS", "musecp12"),
             ("C", "clap"), ("C", "C_noS"), ("C", "scs"), ("C", "cocola")]
    for _ in range(args.boot):
        pick = rng.choice(len(tracks), size=len(tracks), replace=True)
        sel = np.concatenate([by_tr[tracks[i]] for i in pick])
        mw = centre(mos, sel=sel)
        cw = {k: centre(metrics[k], sel=sel) for k in keys}
        for a, b in pairs:
            diffs[f"{a} - {b} pooled"].append(
                pearson(mos[sel], metrics[a][sel]) - pearson(mos[sel], metrics[b][sel]))
            diffs[f"{a} - {b} within"].append(pearson(mw, cw[a]) - pearson(mw, cw[b]))

    contrasts = {}
    print()
    for name, d in diffs.items():
        d = np.array([x for x in d if np.isfinite(x)])
        a, b = name.split(" - ")[0], name.split(" - ")[1].split()[0]
        kind = name.split()[-1]
        obs = ((pooled[a] - pooled[b]) if kind == "pooled" else (within[a] - within[b]))
        lo, hi = np.percentile(d, [2.5, 97.5])
        # review_fixes reports the bootstrap MEAN in this column, not the point
        # difference, so the new row is computed the same way as the published ones.
        contrasts[name] = {"delta_r_boot_mean": round(float(d.mean()), 4),
                           "delta_r_point": round(float(obs), 4),
                           "ci95": [round(float(lo), 4), round(float(hi), 4)],
                           "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"  {name:<28} boot {d.mean():+.4f}  point {obs:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]"
              f"{'  excludes zero' if (lo > 0 or hi < 0) else '  spans zero'}")

    dest = rf.ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"meta": {"n_pairs": len(recs), "n_tracks": len(tracks),
                  "n_conditions": int(len(np.unique(conds))), "n_boot": args.boot,
                  "construction": "review_fixes.analysis_bc, with MuseCPEval added",
                  "musecp_source": args.musecp},
         "pooled_r": pooled, "within_condition_r": within,
         "contrasts": contrasts}, indent=1))
    print("\nwrote", dest.relative_to(rf.ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
