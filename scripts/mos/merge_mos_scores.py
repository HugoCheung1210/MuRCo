#!/usr/bin/env python
"""Build ONE scores table covering every pair in the MOS study. DSP env, read-only.

`compute_C.py --fit-mos` silently drops any rated pair missing from its scores file.
The MOS study spans two runs that live in different places:

  matrix pairs (225)  results/mos_secs6/pair_scores.csv   6 s windows, ids as-is
  AI pairs      (40)  results/rerank_ace/pair_scores.csv  6 s by construction,
                                                          ids need the backend prefix

Without this merge the "does S help" test would run on the matrix half alone — the
half where S is weakest — and nothing would warn you.

Verifies against stimuli_plan.json that every planned pair is present, and fails
loudly if not.

Usage:
  python scripts/mos/merge_mos_scores.py
  python scripts/mos/merge_mos_scores.py --out results/mos_secs6/pair_scores_all.csv
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
FIELDS = ["pair_id", "source_id", "genre", "dim_target", "perturbation", "magnitude",
          "score_H", "score_T", "score_R", "score_S"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="results/mos_secs6/pair_scores.csv")
    ap.add_argument("--ai", default="results/rerank_ace/pair_scores.csv")
    ap.add_argument("--ai-tag", default="rerank_ace",
                    help="prefix that namespaces AI pair_ids, matching the plan")
    ap.add_argument("--plan", default="results/mos/stimuli_plan.json")
    ap.add_argument("--out", default="results/mos_secs6/pair_scores_all.csv")
    args = ap.parse_args()

    plan = json.load(open(ROOT / args.plan))["stimuli"]
    want = {r["pair_id"] for r in plan}

    rows, seen = [], set()
    for r in csv.DictReader(open(ROOT / args.matrix)):
        if r["pair_id"] in want:
            rows.append({k: r[k] for k in FIELDS})
            seen.add(r["pair_id"])
    for r in csv.DictReader(open(ROOT / args.ai)):
        pid = f"{args.ai_tag}/{r['pair_id']}"
        if pid in want:
            rows.append({**{k: r[k] for k in FIELDS}, "pair_id": pid})
            seen.add(pid)

    missing = sorted(want - seen)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_ai = sum(1 for r in rows if r["pair_id"].startswith(args.ai_tag + "/"))
    print(json.dumps({"planned": len(want), "written": len(rows),
                      "matrix": len(rows) - n_ai, "ai": n_ai,
                      "missing": missing[:5], "n_missing": len(missing),
                      "out": str(out)}, indent=1))
    if missing:
        raise SystemExit(f"\n{len(missing)} planned pair(s) have no scores — the fit would "
                         f"drop them silently. Fix before analysis.")


if __name__ == "__main__":
    main()
