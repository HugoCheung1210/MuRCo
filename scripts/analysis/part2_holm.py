#!/usr/bin/env python
"""Part 2 multiplicity: cluster-bootstrap p per arm, then Holm across the family.

Why this exists: the paper reports that C wins the five designed comparisons and that
each survives Holm's correction, but the only Holm computation on disk
(results/diagnostics/review_fixes.json) is on a DIFFERENT statistic. It counts a tie as
half a win, where Table 4 drops ties and reports the win rate over decided ratings. The
two disagree, so the printed bound had no reproducible home. This script computes Holm on
the statistic Table 4 actually reports.

The five DESIGNED arms are the ones named in key.csv's `settles` column; no trial was
built to compare C against MuseCPEval. MuseCPEval is therefore reported separately, on
the trials where its own pick disagrees with C's, and a second Holm run folds it in as a
robustness check. Adding a test only tightens Holm's thresholds, so "all six survive at
m=6" is strictly stronger than the five-arm claim. It does not make MuseCPEval
pre-specified, and the paper should keep saying it was added afterwards.

Usage (DSP env, from repo root):
    python scripts/analysis/part2_holm.py --out results/mos/session2/holm.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np

N_BOOT = 20000
SEED = 20260909


def repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / ".projectroot").exists() or (p / ".git").exists():
            return p
    return Path(__file__).resolve().parent.parent.parent


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_ratings(key: dict, pattern: str) -> list[tuple[str, str | None]]:
    """(trial_id, "c" | "rival" | None) per rating; None is an explicit tie."""
    out = []
    for f in sorted(glob.glob(pattern)):
        doc = json.load(open(f))
        for resp in doc.get("responses", []):
            tid = resp["trial_id"]
            if tid not in key or resp.get("preference") is None:
                continue
            pref = int(resp["preference"])
            if pref == 3:
                out.append((tid, None))
                continue
            side = "a" if pref < 3 else "b"
            out.append((tid, key[tid][f"opt_{side}_role"]))
    return out


def arm_stats(key: dict, ratings, trial_ids: set[str], rng) -> dict:
    """Win rate over decided ratings, cluster bootstrap over seeds, two-sided p."""
    by_seed: dict[str, list[int]] = {}
    n_dec = n_win = n_tie = 0
    for tid, role in ratings:
        if tid not in trial_ids:
            continue
        if role is None:
            n_tie += 1
            continue
        n_dec += 1
        w = 1 if role == "c" else 0
        n_win += w
        cell = by_seed.setdefault(key[tid]["seed"], [0, 0])
        cell[0] += w
        cell[1] += 1

    seeds = sorted(by_seed)
    wins = np.array([by_seed[s][0] for s in seeds], dtype=float)
    tot = np.array([by_seed[s][1] for s in seeds], dtype=float)
    idx = rng.integers(0, len(seeds), (N_BOOT, len(seeds)))
    boots = wins[idx].sum(1) / np.maximum(tot[idx].sum(1), 1)
    p = 2 * min(float((boots <= 0.5).mean()), float((boots >= 0.5).mean()))
    p = min(1.0, max(p, 1.0 / N_BOOT))
    return {"n_trials": len(trial_ids), "n_ratings": n_dec + n_tie, "n_decided": n_dec,
            "n_ties": n_tie, "win_rate_decided": round(n_win / n_dec, 4) if n_dec else None,
            "ci95_cluster_by_seed": [round(float(np.percentile(boots, 2.5)), 4),
                                     round(float(np.percentile(boots, 97.5)), 4)],
            "n_seeds": len(seeds), "p_boot_two_sided": round(p, 5)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=lambda k: pvals[k])
    m, prev, adj = len(order), 0.0, {}
    for i, k in enumerate(order):
        prev = min(1.0, max(prev, (m - i) * pvals[k]))
        adj[k] = round(prev, 5)
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", default="results/mos/session2/key.csv")
    ap.add_argument("--responses", default="results/mos/session2/responses/*.json")
    ap.add_argument("--musecp", default="results/musecp/rerank_ace/musecp_metrics.csv")
    ap.add_argument("--musecp-metric", default="musecp_mean12")
    ap.add_argument("--write-key", action="store_true",
                    help="record MuseCPEval's per-trial pick in key.csv as `musecp_pick`. "
                         "Never written into `settles`: no trial was designed for it.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = repo_root()
    rng = np.random.default_rng(SEED)

    key = {r["trial_id"]: r for r in read_csv(root / args.key)}
    ratings = load_ratings(key, str(root / args.responses))
    designed = sorted({s for r in key.values() for s in r["settles"].split(";") if s})
    print(f"{len(key)} trials, {len(ratings)} ratings, {len(designed)} designed arms")

    arms = {}
    for rv in designed:
        tids = {t for t, r in key.items() if rv in r["settles"].split(";")}
        arms[rv] = arm_stats(key, ratings, tids, rng)
        arms[rv]["designed"] = True

    adj5 = holm({k: v["p_boot_two_sided"] for k, v in arms.items()})
    for k, v in adj5.items():
        arms[k]["p_holm_m5"] = v

    print("\ndesigned arms (win rate over decided ratings, ties dropped)")
    for k in designed:
        a = arms[k]
        print(f"  {k:<12} n={a['n_ratings']:>4}  decided={a['win_rate_decided']:.4f} "
              f"{a['ci95_cluster_by_seed']}  p={a['p_boot_two_sided']:.5f}  "
              f"p_holm={a['p_holm_m5']:.5f}")
    worst = max(adj5.values())
    print(f"\n  Holm over m=5: every arm significant at 0.05 = "
          f"{all(v < 0.05 for v in adj5.values())}, largest adjusted p = {worst:.5f}")

    report = {"note": "win rate over DECIDED ratings (ties dropped), the statistic in "
                      "Table 4; cluster bootstrap over source seeds; Holm across arms.",
              "n_trials": len(key), "n_ratings": len(ratings),
              "designed_arms": designed, "arms": arms,
              "holm_m5": {"largest_adjusted_p": worst,
                          "all_significant_0.05": bool(all(v < 0.05 for v in adj5.values()))}}

    # --- m=6 robustness: fold in the post hoc MuseCPEval comparison -----------
    # MuseCPEval appears in no `settles` cell, so its comparison is not a designed arm.
    # It is the trials where its own top pick disagrees with C's, scored the same way.
    mcp = {r["pair_id"]: r for r in read_csv(root / args.musecp)}
    tab = {q: (float(r[args.musecp_metric]) if r[args.musecp_metric] != "" else np.nan)
           for q, r in mcp.items()}

    def musecp_pick(k, tol=1e-6):
        a, b = k["c_pick"], k["rival_pick"]
        if a not in tab or b not in tab:
            return ""
        va, vb = tab[a], tab[b]
        if not (np.isfinite(va) and np.isfinite(vb)) or abs(va - vb) < tol:
            return ""
        return "c" if va > vb else "rival"

    picks = {t: musecp_pick(k) for t, k in key.items()}
    dis = {t for t, v in picks.items() if v == "rival"}
    arms["musecp"] = arm_stats(key, ratings, dis, rng)
    arms["musecp"]["designed"] = False
    arms["musecp"]["note"] = (f"post hoc: not in any `settles` cell. The {len(dis)} trials "
                              f"where {args.musecp_metric} prefers the rival candidate.")

    adj6 = holm({k: v["p_boot_two_sided"] for k, v in arms.items()})
    for k, v in adj6.items():
        arms[k]["p_holm_m6"] = v
    worst6 = max(adj6.values())

    a = arms["musecp"]
    print(f"\npost hoc arm (NOT designed; {len(dis)} of {len(key)} trials where "
          f"{args.musecp_metric} disagrees with C)")
    print(f"  {'musecp':<12} n={a['n_ratings']:>4}  decided={a['win_rate_decided']:.4f} "
          f"{a['ci95_cluster_by_seed']}  p={a['p_boot_two_sided']:.5f}  "
          f"p_holm={a['p_holm_m6']:.5f}")
    print(f"\n  Holm over m=6 (five designed + MuseCPEval): every arm significant at "
          f"0.05 = {all(v < 0.05 for v in adj6.values())}, largest adjusted p = {worst6:.5f}")

    report["holm_m6"] = {"note": "robustness only. Folding a post hoc comparison into the "
                                 "family tightens every threshold, so surviving at m=6 is "
                                 "stronger than surviving at m=5. It does not make "
                                 "MuseCPEval pre-specified.",
                         "largest_adjusted_p": worst6,
                         "all_significant_0.05": bool(all(v < 0.05 for v in adj6.values()))}

    if args.write_key:
        kp = root / args.key
        rows = read_csv(kp)
        cols = list(rows[0].keys())
        if "musecp_pick" not in cols:
            cols.append("musecp_pick")
        for r in rows:
            r["musecp_pick"] = picks.get(r["trial_id"], "")
        with open(kp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[wrote {kp}] added `musecp_pick`; `settles` is unchanged, so the "
              f"designed-arm list stays at {len(designed)}.")

    if args.out:
        out = root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\n[wrote {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
