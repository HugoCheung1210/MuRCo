#!/usr/bin/env python
"""Score session 2: does C's pick beat each rival metric's? DSP env.

Per contrast, reports the share of comparisons where listeners preferred C's candidate,
with a **cluster bootstrap over seeds**. Clustering matters: the same seed contributes
many rater decisions, and treating those as independent would shrink the interval by
roughly the square root of the number of raters and manufacture significance. The seed
is the unit of generalisation, so it is the unit resampled.

Ties (the "about the same" midpoint) are reported and excluded from the win rate, as in
the earlier rerank A/B pilot, so the two are comparable.

Also reports:
  - position bias: how often option A won regardless of which metric picked it. The
    build balances C's pick across positions, so a large bias inflates nothing, but it
    should be visible rather than assumed away.
  - per-seed win rates, so one dominant seed can't be mistaken for a general effect.

Response JSON: {"session_code": ..., "responses":[{"trial_id":"t001","preference":1..5}]}
where 1 = A clearly better ... 3 = same ... 5 = B clearly better.

Usage:
  python scripts/mos/score_ab_session2.py --responses 'results/mos/session2/responses/*.json'
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


def cluster_bootstrap(by_seed, n_boot=5000, seed=20260728):
    """Resample SEEDS (not decisions) -> CI on the pooled win rate."""
    rng = np.random.default_rng(seed)
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        return (float("nan"), float("nan"))
    wins = np.array([by_seed[s][0] for s in seeds], dtype=float)
    dec = np.array([by_seed[s][1] for s in seeds], dtype=float)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(seeds), len(seeds))
        d = dec[idx].sum()
        out.append(wins[idx].sum() / d if d > 0 else np.nan)
    out = np.array(out)[~np.isnan(out)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def position_adjusted(rows):
    """Separate the metric effect from a preference for whichever option is shown first.

    Position is fixed per trial in the key rather than randomised per rater, so the two
    side-specific win rates come from DISJOINT sets of stimuli and their difference
    confounds position with which pairs happened to land on each side. Fitting

        logit P(C wins) = beta + alpha * (+1 if C is on A else -1)

    recovers the part that does not depend on side. beta is the metric effect and
    exp(alpha) is the odds a listener gives the first option for no other reason.

    The assumption is that C's advantage is the same size on both sides, which fixed
    positions make untestable. What can be checked is exp(alpha) across contrasts: a
    genuine position preference belongs to the rater and the interface, so it cannot
    know which metric picked the other candidate and should be constant. It is not
    (2026-08-20: 1.01 to 2.21), which is evidence the split is mostly stimulus-set
    difference rather than position.
    """
    import math

    def _logit(p):
        return math.log(p / (1 - p))

    out = {}
    for side in ("A", "B"):
        w = [ch for _, ch, cs in rows if cs == side and ch is not None]
        out[f"win_rate_C_on_{side}"] = round(sum(w) / len(w), 4) if w else None
        out[f"n_C_on_{side}"] = len(w)
    if not (out["n_C_on_A"] and out["n_C_on_B"]):
        return out
    pa = min(max(out["win_rate_C_on_A"], 1e-3), 1 - 1e-3)
    pb = min(max(out["win_rate_C_on_B"], 1e-3), 1 - 1e-3)
    beta = (_logit(pa) + _logit(pb)) / 2
    alpha = (_logit(pa) - _logit(pb)) / 2
    out["win_rate_position_adjusted"] = round(1 / (1 + math.exp(-beta)), 4)
    out["position_odds_ratio"] = round(math.exp(alpha), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--key", default="results/mos/session2/key.csv")
    ap.add_argument("--out", default="results/mos/session2/ab_results.json")
    args = ap.parse_args()

    key = {r["trial_id"]: r for r in csv.DictReader(open(ROOT / args.key))}
    files = sorted(glob.glob(args.responses))
    if not files:
        raise SystemExit(f"no response files matched {args.responses!r}")

    decisions = []          # (contrast, seed, chose_C or None for tie, side C sat on)
    pos_a_wins = pos_dec = 0
    n_sessions = n_skipped = 0
    for f in files:
        d = json.load(open(f))
        n_sessions += 1
        for r in d["responses"]:
            k = key.get(r["trial_id"])
            if not k:
                continue
            if r.get("skipped") or r.get("preference") is None:
                n_skipped += 1        # uncomfortable-skip: excluded, not a tie
                continue
            p = int(r["preference"])
            if p == 3:
                chose = None
            else:
                picked = "a" if p < 3 else "b"
                chose = (k[f"opt_{picked}_role"] == "c")
                pos_dec += 1
                pos_a_wins += (picked == "a")
            c_side = "A" if k["opt_a_role"] == "c" else "B"
            for contrast in k["settles"].split(";"):
                decisions.append((contrast, k["seed"], chose, c_side))

    results = {}
    for contrast in sorted({c for c, _, _, _ in decisions}):
        rows = [(s, ch, cs) for c, s, ch, cs in decisions if c == contrast]
        ties = sum(1 for _, ch, _ in rows if ch is None)
        by_seed = defaultdict(lambda: [0, 0])
        for s, ch, _ in rows:
            if ch is None:
                continue
            by_seed[s][1] += 1
            by_seed[s][0] += int(ch)
        wins = sum(v[0] for v in by_seed.values())
        dec = sum(v[1] for v in by_seed.values())
        lo, hi = cluster_bootstrap(by_seed)
        # Ties-inclusive reading: every rated trial counts, a tie scores half a win.
        # Reported alongside the decided-only rate because ties are ~a quarter of trials,
        # so the decided rate is conditioned on the rater having reached a decision.
        ti_seed = defaultdict(lambda: [0.0, 0.0])
        for s, ch, _ in rows:
            ti_seed[s][1] += 1
            ti_seed[s][0] += 0.5 if ch is None else float(ch)
        ti_w = sum(v[0] for v in ti_seed.values())
        ti_n = sum(v[1] for v in ti_seed.values())
        ti_lo, ti_hi = cluster_bootstrap(ti_seed)

        results[contrast] = {
            "n_seeds": len(by_seed), "n_decisions": dec, "n_ties": ties,
            "c_wins": wins, "win_rate": round(wins / dec, 4) if dec else None,
            "ci95_cluster_by_seed": [round(lo, 4), round(hi, 4)],
            "beats_chance": bool(dec and lo > 0.5),
            "win_rate_ties_included": round(ti_w / ti_n, 4) if ti_n else None,
            "ci95_ties_included": [round(ti_lo, 4), round(ti_hi, 4)],
            **position_adjusted(rows),
            "per_seed_win_rate": {s: round(v[0] / v[1], 3) if v[1] else None
                                  for s, v in sorted(by_seed.items())},
        }

    payload = {"meta": {
        "n_sessions": n_sessions, "n_trials_in_key": len(key),
        "n_skipped": n_skipped,
        "position_bias_option_A_win_rate": round(pos_a_wins / pos_dec, 4) if pos_dec else None,
        "note": "CI is a cluster bootstrap over SEEDS — decisions within a seed are not "
                "independent. Ties excluded from win rate, reported separately.",
    }, "contrasts": results}
    json.dump(payload, open(ROOT / args.out, "w"), indent=1)

    print(f"{'contrast':14} {'seeds':>6} {'dec':>5} {'ties':>5} {'C win':>7}  {'95% CI':>16} "
          f"{'C|A':>6} {'C|B':>6} {'adj':>6} {'posOR':>6}  verdict")
    for c, v in results.items():
        ci = f"[{v['ci95_cluster_by_seed'][0]:.3f},{v['ci95_cluster_by_seed'][1]:.3f}]"
        adj = v.get("win_rate_position_adjusted")
        print(f"{c:14} {v['n_seeds']:6} {v['n_decisions']:5} {v['n_ties']:5} "
              f"{v['win_rate']:7.3f}  {ci:>16} "
              f"{v.get('win_rate_C_on_A', float('nan')):6.3f} "
              f"{v.get('win_rate_C_on_B', float('nan')):6.3f} "
              f"{adj if adj is not None else float('nan'):6.3f} "
              f"{v.get('position_odds_ratio', float('nan')):6.2f}  "
              f"{'C beats it' if v['beats_chance'] else 'not distinguishable'}")
    print(f"\nposition bias (option A win rate, any metric): "
          f"{payload['meta']['position_bias_option_A_win_rate']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
