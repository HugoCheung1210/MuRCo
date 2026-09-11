#!/usr/bin/env python3
"""Score the A/B coherence test: do raters prefer C's winner over C's loser?

Joins the rater exports from the `prepare_mos_ab.py` UI (`responses_*.json`, each
{rater, responses:[{trial_id, pref}]}, pref in A/B/tie) against the PRIVATE blinding
key (`key.csv`: which blinded option was C's winner per trial). Reports the headline
forced-choice statistic -- the fraction of decided trials where the rater picked C's
winner -- with a Wilson 95% CI and a two-sided binomial test vs chance (0.5), plus
per-rater and per-seed breakdowns, a C-margin check (are bigger C gaps picked more
reliably?), and inter-rater agreement (mean pairwise + Fleiss' kappa).

Ties are excluded from the win-rate denominator (reported separately). Per project
Conventions the CI is the headline; the binomial p is secondary.

    RERANK_DIR=results/rerank_ace python scripts/rerank/score_mos_ab.py
    RERANK_DIR=results/rerank_ace python scripts/rerank/score_mos_ab.py --responses path/to/*.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

RERANK_DIR = Path(os.environ.get("RERANK_DIR", "results/rerank"))


def _wilson(k: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def _mean_ci(xs):
    """Mean + normal-approx 95% CI of a list (the signed C-preference, [-2,+2])."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    m = sum(xs) / n
    if n < 2:
        return {"n": n, "mean": m, "ci95": [float("nan"), float("nan")]}
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    h = 1.96 * sd / (n ** 0.5)
    return {"n": n, "mean": m, "ci95": [m - h, m + h]}


def _interpret(item, win_opt):
    """(call, signed) from a response row. call in winner/loser/tie/None(skip); signed in
    [-2,+2] toward C's winner (5-pt scale) or None (old A/B format / unrated)."""
    rating = item.get("rating", "")
    if rating not in ("", None):
        try:
            r = int(rating)
        except (ValueError, TypeError):
            return ("bad", None)
        if r < 1 or r > 5:
            return ("bad", None)
        if r == 3:
            return ("tie", 0)
        prefers = "A" if r < 3 else "B"
        signed = (3 - r) if win_opt == "A" else (r - 3)     # + = toward C's winner
        return ("winner" if prefers == win_opt else "loser", signed)
    pref = item.get("pref", "")                              # legacy A/B/tie format
    if pref in ("A", "B"):
        return ("winner" if pref == win_opt else "loser", None)
    if pref in ("tie", "none"):
        return ("tie", None)
    if pref == "":
        return (None, None)                                 # unanswered
    return ("bad", None)


def _binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial p vs 0.5 (scipy if available, else normal approx)."""
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        from math import comb
        probs = [comb(n, i) * 0.5 ** n for i in range(n + 1)]
        obs = comb(n, k) * 0.5 ** n
        return float(min(1.0, sum(pr for pr in probs if pr <= obs + 1e-12)))


def _fleiss_kappa(rows) -> float:
    """Fleiss' kappa over items x categories count matrix (uniform raters/item assumed)."""
    rows = [r for r in rows if sum(r) > 0]
    if not rows:
        return float("nan")
    n = sum(rows[0])                                  # raters per item (assume uniform)
    if n < 2 or any(sum(r) != n for r in rows):
        return float("nan")
    N, k = len(rows), len(rows[0])
    p_j = [sum(rows[i][j] for i in range(N)) / (N * n) for j in range(k)]
    P_i = [(sum(c * c for c in rows[i]) - n) / (n * (n - 1)) for i in range(N)]
    P_bar = sum(P_i) / N
    P_e = sum(p * p for p in p_j)
    return float((P_bar - P_e) / (1 - P_e)) if (1 - P_e) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(RERANK_DIR / "mos_ab"),
                    help="mos_ab dir holding key.csv + responses_*.json (default: RERANK_DIR/mos_ab)")
    ap.add_argument("--key", default="", help="override path to key.csv")
    ap.add_argument("--responses", nargs="*", default=None,
                    help="explicit response json path(s)/glob(s) (default: <dir>/responses_*.json)")
    ap.add_argument("--out", default="", help="override output json path")
    args = ap.parse_args()

    ab_dir = Path(args.dir)
    key_p = Path(args.key) if args.key else ab_dir / "key.csv"
    if not key_p.is_file():
        raise SystemExit(f"missing {key_p} -- run prepare_mos_ab.py first.")

    # key: trial_id -> winner option ("A"/"B"), seed, C gap between the two options
    key = {}
    for r in csv.DictReader(key_p.open(newline="", encoding="utf-8")):
        try:
            gap = abs(float(r["opt_A_C"]) - float(r["opt_B_C"]))
        except (KeyError, ValueError):
            gap = float("nan")
        key[r["trial_id"]] = {"winner": r["winner_option"], "seed": r.get("seed", ""), "gap": gap}

    resp_paths = []
    if args.responses:
        for pat in args.responses:
            resp_paths += [Path(p) for p in glob.glob(pat)]
    else:
        resp_paths = sorted(ab_dir.glob("responses_*.json"))
    if not resp_paths:
        raise SystemExit(f"no response files found (looked in {ab_dir}/responses_*.json). "
                         f"Collect the rater exports there, or pass --responses.")

    # per (rater, trial) -> call in {"winner","loser","tie"}
    per_rater = defaultdict(lambda: {"winner": 0, "loser": 0, "tie": 0})
    per_seed = defaultdict(lambda: {"winner": 0, "loser": 0, "tie": 0})
    signed_overall, signed_by_rater = [], defaultdict(list)   # signed C-preference (5-pt scale)
    gap_decided = []                                   # (gap, 1 if winner else 0)
    calls_by_trial = defaultdict(list)                 # trial_id -> [call,...] across raters
    raters = []
    skipped = 0
    for p in resp_paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        rater = str(d.get("rater", p.stem.replace("responses_", "")))
        raters.append(rater)
        for item in d.get("responses", []):
            tid = item.get("trial_id", "")
            if tid not in key:
                skipped += 1
                continue
            call, signed = _interpret(item, key[tid]["winner"])
            if call is None:
                continue                                # unanswered -> not a datum
            if call == "bad":
                skipped += 1
                continue
            per_rater[rater][call] += 1
            per_seed[key[tid]["seed"]][call] += 1
            calls_by_trial[tid].append(call)
            if signed is not None:
                signed_overall.append(signed)
                signed_by_rater[rater].append(signed)
            if call in ("winner", "loser"):
                gap_decided.append((key[tid]["gap"], 1 if call == "winner" else 0))

    def tally(counts):
        w, l, t = counts["winner"], counts["loser"], counts["tie"]
        dec = w + l
        wr = (w / dec) if dec else float("nan")
        lo, hi = _wilson(w, dec)
        return {"winner": w, "loser": l, "tie": t, "decided": dec,
                "win_rate": wr, "ci95": [lo, hi], "binom_p": _binom_p(w, dec)}

    overall_counts = {"winner": 0, "loser": 0, "tie": 0}
    for c in per_rater.values():
        for k2 in overall_counts:
            overall_counts[k2] += c[k2]
    overall = tally(overall_counts)

    # C-margin check: win-rate in the lower vs upper half of the C gap
    gap_split = {}
    gd = [g for g in gap_decided if not math.isnan(g[0])]
    if len(gd) >= 4:
        med = sorted(g[0] for g in gd)[len(gd) // 2]
        lohalf = [w for g, w in gd if g <= med]
        hihalf = [w for g, w in gd if g > med]
        def half(a):
            return {"n": len(a), "win_rate": (sum(a) / len(a)) if a else float("nan"),
                    "ci95": list(_wilson(sum(a), len(a)))}
        gap_split = {"gap_median": med, "small_gap": half(lohalf), "large_gap": half(hihalf)}

    # inter-rater agreement over trials rated by >=2 raters
    CATS = ["winner", "loser", "tie"]
    multi = {tid: cs for tid, cs in calls_by_trial.items() if len(cs) >= 2}
    pair_agree = []
    for cs in multi.values():
        pairs = agree = 0
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                pairs += 1
                agree += (cs[i] == cs[j])
        if pairs:
            pair_agree.append(agree / pairs)
    matrix = [[cs.count(c) for c in CATS] for cs in multi.values()]
    agreement = {
        "n_trials_multi_rated": len(multi),
        "mean_pairwise_agreement": (sum(pair_agree) / len(pair_agree)) if pair_agree else float("nan"),
        "fleiss_kappa": _fleiss_kappa(matrix),
    }

    overall["c_pref"] = _mean_ci(signed_overall)       # signed 5-pt preference toward C's winner
    per_rater_out = {}
    for r, c in sorted(per_rater.items()):
        d = tally(c)
        d["c_pref"] = _mean_ci(signed_by_rater[r])
        per_rater_out[r] = d

    out = {
        "meta": {"n_raters": len(raters), "raters": raters, "n_trials_in_key": len(key),
                 "skipped_unmatched": skipped,
                 "note": "win = rater picked C's winner over C's loser; ties excluded from rate; "
                         "c_pref = mean signed 5-pt score in [-2,+2], + = toward C's winner "
                         "(1=A/5=B clearly, 3=same); CI is the headline (Conventions), binom_p secondary"},
        "overall": overall,
        "per_rater": per_rater_out,
        "per_seed": {s: tally(c) for s, c in sorted(per_seed.items())},
        "c_margin_split": gap_split,
        "inter_rater_agreement": agreement,
    }
    out_p = Path(args.out) if args.out else ab_dir / "mos_ab_scores.json"
    out_p.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- printed summary ----------------------------------------------------
    o = overall
    print(f"# A/B coherence -- C-winner preference  ({len(raters)} rater(s), "
          f"{o['decided']} decided + {o['tie']} tie)\n")
    star = "" if (math.isnan(o["ci95"][0]) or o["ci95"][0] <= 0.5) else "  (CI > 0.5)"
    print(f"## Headline: C-winner chosen in {o['win_rate']*100:.1f}% of decided trials"
          f"  [{o['ci95'][0]*100:.1f}, {o['ci95'][1]*100:.1f}]%{star}")
    print(f"   winner={o['winner']}  loser={o['loser']}  tie={o['tie']}   binomial p={o['binom_p']:.4g}")
    cp = o["c_pref"]
    if cp["n"]:
        cpstar = "" if (math.isnan(cp["ci95"][0]) or cp["ci95"][0] <= 0) else "  (CI > 0)"
        print(f"   mean C-preference (5-pt, +=toward winner) = {cp['mean']:+.2f}"
              f"  [{cp['ci95'][0]:+.2f}, {cp['ci95'][1]:+.2f}]  (n={cp['n']}){cpstar}")
    print()
    print("## Per rater")
    print("| rater | win | lose | tie | win-rate | 95% CI |")
    print("|---|--:|--:|--:|--:|:--:|")
    for r, c in out["per_rater"].items():
        print(f"| {r} | {c['winner']} | {c['loser']} | {c['tie']} | "
              f"{c['win_rate']*100:.0f}% | [{c['ci95'][0]*100:.0f}, {c['ci95'][1]*100:.0f}]% |")
    if gap_split:
        s, l = gap_split["small_gap"], gap_split["large_gap"]
        print(f"\n## C-margin check (does a bigger C gap get picked more reliably?)")
        print(f"   small gap (<= {gap_split['gap_median']:.3f}): {s['win_rate']*100:.0f}% (n={s['n']})")
        print(f"   large gap (>  {gap_split['gap_median']:.3f}): {l['win_rate']*100:.0f}% (n={l['n']})")
    a = agreement
    print(f"\n## Inter-rater agreement (trials with >=2 raters: {a['n_trials_multi_rated']})")
    print(f"   mean pairwise = {a['mean_pairwise_agreement']:.3f}   Fleiss kappa = {a['fleiss_kappa']:.3f}")
    if skipped:
        print(f"\n(!) {skipped} response row(s) had a trial_id not in {key_p.name} -- ignored.")
    print(f"\nwrote {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
