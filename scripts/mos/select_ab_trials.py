#!/usr/bin/env python
"""Choose the session-2 A/B comparisons: C vs every rival metric, in as few trials as possible.

The question is whether C ORDERS regenerated candidates better than the published
alternatives — the direct answer to "why not just use CLAP?", which C's loss to CLAP on
identity-AUC otherwise leaves open. So a trial is a candidate PAIR that C and a rival
metric order oppositely: C says i > j, the rival says j > i. Listeners break the tie.

Two efficiencies matter:

1. **Opposed pairs, not argmax-vs-argmax.** Comparing only each metric's top pick wastes
   most seeds — with 8 candidates the two argmaxes often coincide, and when they differ
   C's margin is frequently a rounding error. Allowing any opposed pair roughly doubles
   the usable trials and doubles the median margin. The claim narrows from "best-of-N
   selection" to "pairwise ordering", which is what the data can actually support.

2. **One trial can settle several contrasts.** If C-vs-clap_htsat, C-vs-clap_music and
   C-vs-scs all disagree about the same pair, one human comparison answers all three.
   Greedy set cover per seed exploits this: 65 contrast-trials collapse to ~34.

--min-margin holds out near-ties. If C rates its pick 0.0005 above the rival's, a 50/50
human split is a SUCCESS for C, not a failure, and pooling those only adds noise.

Usage:
  python scripts/mos/select_ab_trials.py
  python scripts/mos/select_ab_trials.py --rivals clap_htsat,scs --min-margin 0.03
Outputs: results/rerank_ace/ab_trials.json
"""
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
DIMS = "HTRS"


def geo(v):
    return float(np.exp(np.mean(np.log(np.clip(v, 1e-6, 1.0)))))


def load_candidates(rd, cocola_csv=None):
    """Per-candidate scores for C and every rival, keyed seed -> pair_id -> metric.

    COCOLA lives in its own CSV rather than baseline_scores_rerank.json because it is
    scored outside the DSP env (torch 2.2 + lightning; see scripts/analysis/score_cocola.py).
    Its scale is a bilinear similarity around 40-70 rather than [0,1], which is fine here:
    the opposed-pair test compares the SIGN of a difference, and --min-margin applies only
    to C's own scale. Only `rival_margins` in the output is scale-dependent, and that is
    reported, never thresholded.
    """
    base = json.load(open(rd / "baseline_scores_rerank.json"))["scores"]
    coc = {}
    if cocola_csv and Path(cocola_csv).exists():
        coc = {r["pair_id"]: float(r["cocola"])
               for r in csv.DictReader(open(cocola_csv))}
    cand = defaultdict(dict)
    for r in csv.DictReader(open(rd / "pair_scores.csv")):
        seed = r["pair_id"].split("::")[0]
        d = {k: float(r[f"score_{k}"]) for k in DIMS}
        b = base.get(r["pair_id"], {})
        cand[seed][r["pair_id"]] = {
            "C": geo([d[k] for k in DIMS]),
            "C_noS": geo([d[k] for k in "HTR"]),
            "cocola": coc.get(r["pair_id"], float("nan")),
            **{k: b.get(k, float("nan")) for k in ("clap_htsat", "clap_music", "scs")},
        }
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-dir", default="results/rerank_ace")
    ap.add_argument("--rivals", default="clap_htsat,clap_music,scs,C_noS")
    ap.add_argument("--min-margin", type=float, default=0.02,
                    help="minimum C-margin for a pair to be worth asking a human about")
    ap.add_argument("--cocola-scores",
                    default="results/baselines/cocola_rerank_ace.csv",
                    help="per-candidate COCOLA scores; needed only if 'cocola' is a rival")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rd = ROOT / args.rerank_dir
    rivals = [r.strip() for r in args.rivals.split(",") if r.strip()]
    cand = load_candidates(rd, ROOT / args.cocola_scores)
    if "cocola" in rivals:
        miss = sum(1 for s in cand.values() for v in s.values() if np.isnan(v["cocola"]))
        if miss:
            raise SystemExit(f"'cocola' requested as a rival but {miss} candidates have no "
                             f"score in {args.cocola_scores}; run scripts/analysis/score_cocola.py")

    trials, uncovered = [], Counter()
    for seed in sorted(cand):
        c = cand[seed]
        opts = []
        for i, j in itertools.combinations(sorted(c), 2):
            dc = c[i]["C"] - c[j]["C"]
            if abs(dc) < args.min_margin:
                continue
            opposed = {r for r in rivals
                       if dc * (c[i][r] - c[j][r]) < 0}      # ranked the other way round
            if opposed:
                hi, lo = (i, j) if dc > 0 else (j, i)
                opts.append({"opposed": opposed, "margin": abs(dc), "c_pick": hi, "rival_pick": lo})
        need = set(rivals)
        while need and opts:
            opts.sort(key=lambda o: (len(o["opposed"] & need), o["margin"]), reverse=True)
            best = opts[0]
            settles = best["opposed"] & need
            if not settles:
                break
            trials.append({
                "seed": seed, "c_pick": best["c_pick"], "rival_pick": best["rival_pick"],
                "c_margin": round(best["margin"], 4),
                "settles": sorted(settles),
                "rival_margins": {r: round(float(c[best["rival_pick"]][r] - c[best["c_pick"]][r]), 4)
                                  for r in sorted(settles)},
            })
            need -= best["opposed"]
            opts.remove(best)
        for r in need:
            uncovered[r] += 1

    per_rival = Counter(r for t in trials for r in t["settles"])
    out = Path(args.out) if args.out else rd / "ab_trials.json"
    payload = {"meta": {
        "rivals": rivals, "min_margin": args.min_margin,
        "n_seeds": len(cand), "n_trials": len(trials),
        "trials_per_contrast": dict(per_rival),
        "seeds_uncovered_per_contrast": dict(uncovered),
        "contrast_trials_if_run_separately": sum(per_rival.values()),
        "settles_histogram": dict(Counter(len(t["settles"]) for t in trials)),
        "design": "opposed-ordering candidate pairs; greedy set cover so one comparison "
                  "can settle several contrasts",
        "claim": "where C and a rival order two candidates oppositely, which do listeners prefer",
    }, "trials": trials}
    json.dump(payload, open(out, "w"), indent=1)
    print(json.dumps(payload["meta"], indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
