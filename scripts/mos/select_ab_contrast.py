#!/usr/bin/env python
"""Pick the A/B head-to-head trials: C's chosen candidate vs a rival metric's. DSP env.

The E-D rerank claim is a RANKING claim, so the sharpest human test is not "does MOS
correlate with C" but "when C and a rival metric disagree about which regenerated
candidate is best, which one do listeners prefer?". That is decision-relevant, and it
is the direct answer to 'why not just use CLAP?' — the question CLAP's win on
identity-AUC otherwise leaves open.

Per seed: take argmax_C and argmax_rival over that seed's N candidates. Seeds where
both metrics pick the SAME candidate carry no information and are dropped.

Contrasts worth running:
  --rival clap_htsat / clap_music / scs   C vs a published baseline  (composite claim)
  --rival C_noS                           C vs itself minus S       (S-attribution claim)

A near-tie for C is a trap: if C rates its own pick 0.0005 above the rival's, a 50/50
human split is a SUCCESS for C, not a failure. --min-margin holds those out so they can
be analysed separately rather than diluting the headline.

Usage:
  python scripts/mos/select_ab_contrast.py --rival clap_htsat
  python scripts/mos/select_ab_contrast.py --rival C_noS --min-margin 0.01
Outputs: results/rerank_ace/ab_<rival>.json  (+ a summary to stdout)
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
DIMS = "HTRS"


def geo(vals):
    return float(np.exp(np.mean(np.log(np.clip(vals, 1e-6, 1.0)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-dir", default="results/rerank_ace")
    ap.add_argument("--rival", default="clap_htsat",
                    choices=("clap_htsat", "clap_music", "scs", "C_noS"))
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="hold out seeds where C's margin over the rival's pick is below this")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = ROOT / args.rerank_dir
    base = json.load(open(rd / "baseline_scores_rerank.json"))["scores"]
    cand = defaultdict(dict)
    for r in csv.DictReader(open(rd / "pair_scores.csv")):
        seed = r["pair_id"].split("::")[0]
        d = {k: float(r[f"score_{k}"]) for k in DIMS}
        cand[seed][r["pair_id"]] = {
            "C": geo([d[k] for k in DIMS]),
            "C_noS": geo([d[k] for k in "HTR"]),
            **{k: base.get(r["pair_id"], {}).get(k, float("nan")) for k in
               ("clap_htsat", "clap_music", "scs")},
        }

    trials, agree, thin = [], [], []
    for seed in sorted(cand):
        c = cand[seed]
        ids = list(c)
        c_pick = max(ids, key=lambda p: c[p]["C"])
        r_pick = max(ids, key=lambda p: c[p][args.rival])
        if c_pick == r_pick:
            agree.append(seed)
            continue
        margin = c[c_pick]["C"] - c[r_pick]["C"]
        row = {"seed": seed,
               "c_pick": {"pair_id": c_pick, "C": round(c[c_pick]["C"], 4),
                          "rival_score": round(float(c[c_pick][args.rival]), 4)},
               "rival_pick": {"pair_id": r_pick, "C": round(c[r_pick]["C"], 4),
                              "rival_score": round(float(c[r_pick][args.rival]), 4)},
               "c_margin": round(margin, 4)}
        (thin if margin < args.min_margin else trials).append(row)

    out = Path(args.out) if args.out else rd / f"ab_{args.rival}.json"
    payload = {"meta": {
        "rival": args.rival, "rerank_dir": str(args.rerank_dir),
        "n_seeds": len(cand), "n_trials": len(trials),
        "n_agree_dropped": len(agree), "agree_seeds": agree,
        "n_held_out_thin_margin": len(thin), "min_margin": args.min_margin,
        "claim": ("C vs published baseline — composite" if args.rival != "C_noS"
                  else "C vs C without S — attributes the effect to the S term"),
        "note": "seeds where both metrics pick the same candidate carry no information",
    }, "trials": trials, "held_out_thin_margin": thin}
    json.dump(payload, open(out, "w"), indent=1)

    m = [t["c_margin"] for t in trials]
    print(json.dumps({**payload["meta"],
                      "c_margin": {"min": min(m), "median": float(np.median(m)),
                                   "max": max(m)} if m else None,
                      "out": str(out)}, indent=1))


if __name__ == "__main__":
    main()
