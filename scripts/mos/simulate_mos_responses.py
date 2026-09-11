#!/usr/bin/env python
"""Fabricate rating files for a full N=35 run, to exercise the analysis end-to-end.

Purpose: prove the pipeline (responses -> MOS -> compute_C --fit-mos) runs and gives
sane output BEFORE real collection, and give the QC something to catch. This is a
plumbing test, not evidence: ratings are generated FROM the metric plus noise, so any
agreement between C and this MOS is circular by construction. Never report numbers
from simulated ratings.

Latent truth = C_uniform of the pair mapped to 1-5, plus rater bias and per-rating
noise calibrated to a target single-rating ICC. --inattentive N makes N raters answer
at random, so the attention-check screen has something to find.

Usage:
  python scripts/mos/simulate_mos_responses.py                       # 35 raters, ICC 0.5
  python scripts/mos/simulate_mos_responses.py --icc 0.3 --inattentive 3
Outputs: results/mos/responses_sim/responses_<code>.json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="results/mos/stimulus_index.csv")
    ap.add_argument("--plan", default="results/mos/stimuli_plan.json")
    ap.add_argument("--out-dir", default="results/mos/responses_sim")
    ap.add_argument("--n-participants", type=int, default=35)
    ap.add_argument("--icc", type=float, default=0.5, help="target single-rating ICC")
    ap.add_argument("--inattentive", type=int, default=2,
                    help="raters who answer at random (should fail the attention checks)")
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    index = list(csv.DictReader(open(ROOT / args.index)))
    meta = json.load(open(ROOT / args.plan))["meta"]
    by_id = {r["stimulus_id"]: r for r in index}
    anchor = [r["stimulus_id"] for r in index if r["block"] == "anchor"]
    blocks = defaultdict(list)
    for r in index:
        if r["block"] != "anchor":
            blocks[r["block"]].append(r["stimulus_id"])
    block_names = sorted(blocks)

    # latent 1-5 from C_uniform, stretched across the observed range
    c = np.array([float(by_id[s]["C_uniform"]) for s in by_id])
    lo, hi = c.min(), c.max()
    def latent(sid):
        v = (float(by_id[sid]["C_uniform"]) - lo) / max(hi - lo, 1e-9)
        return 1.0 + 4.0 * v

    sd_pair = float(np.std([latent(s) for s in by_id]))
    sd_noise = sd_pair * np.sqrt((1 - args.icc) / max(args.icc, 1e-9))

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("responses_*.json"):
        f.unlink()

    written = []
    for p in range(args.n_participants):
        block = block_names[p % len(block_names)]
        sids = anchor + blocks[block]
        rng.shuffle(sids)
        careless = p < args.inattentive
        bias = rng.normal(0, 0.25)                     # rater leniency
        resp = []
        for sid in sids:
            if careless:
                rating = int(rng.integers(1, 6))
            else:
                v = latent(sid) + bias + rng.normal(0, sd_noise)
                rating = int(np.clip(round(v), 1, 5))
            resp.append({"stimulus_id": sid, "rating": rating,
                         "ms_to_respond": int(rng.integers(1500, 9000)), "n_plays": 1})
        code = f"sim{p:02d}"
        json.dump({"session_code": code, "block": block, "dry_run": True,
                   "simulated": True, "careless": careless, "responses": resp},
                  open(out / f"responses_{code}.json", "w"), indent=1)
        written.append({"code": code, "block": block, "n": len(resp), "careless": careless})

    print(json.dumps({
        "participants": len(written), "target_icc": args.icc,
        "inattentive": args.inattentive, "items_each": len(written[0]["n"] * [0]) if False
        else written[0]["n"], "expected_items": meta["items_per_participant"],
        "out_dir": str(out),
        "WARNING": "ratings generated FROM the metric — circular; plumbing test only",
    }, indent=1))


if __name__ == "__main__":
    main()
