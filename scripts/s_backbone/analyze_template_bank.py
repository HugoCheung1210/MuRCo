#!/usr/bin/env python3
"""T7.1 / T7.3 -- per-template analysis of the S template bank (Phase 3.2/3.3).

score_s_mf.py used to cache only the MEAN over the 4 templates; the per-template
scores are now logged (`per_template`), so this runs entirely on data already on
disk -- no GPU.

The bank (mf_probe.TEMPLATES), and the hypothesis (task spec T7.1):
  t1  "...cohere naturally ... as the same continuous piece?"    -> CONTINUITY
  t2  "...share the same instrumentation and timbral character?" -> identity
  t3  "...preserve the musical identity (same instruments...)?"  -> identity
  t4  "...belong to the same track?"                             -> identity
Hypothesis: t2-t4 are identity-flavoured (redundant with clap_htsat, which wins
identity), while t1 carries the unique continuity signal. If true, an S_cont
variant (t1 + continuity-phrased templates) is prompt REFOCUSING, not broadening
(§10-safe), and 3.4 can build it from the logged bank with no new GPU.

This script reports, for whichever cache you point it at:
  A. Per-template drop table (control - perturbed, paired by source; mean + d_z).
  B. Per-template identity AUC (matrix only: transpose-vs-styleswap + the
     cross/same-genre splits) and per-template control-vs-violation AUC.
  C. Correlation of each template with clap_htsat and with the mean S (redundancy).
  D. (T7.3) dataset-centered log-odds: rescore as the mean of per-template log-odds
     after subtracting each template's mean log-odds over the controls; compare
     identity AUC vs the raw-probability mean.

The KEY new question after T1: on the temporal family, does the CONTINUITY
template (t1) catch shuffle/reverse better than the identity templates or the
mean? Point it at the temporal cache to find out.

Usage:
    # matrix (identity splits + clap correlation + log-odds centering)
    python scripts/s_backbone/analyze_template_bank.py \
        --cache results/s_cache/s_scores_pt.json --manifest perturbations/pairs_manifest.json \
        --baselines results/baselines/baseline_scores.json --out results/s_cache/template_bank_matrix.json

    # temporal family (does t1 catch order violations?)
    python scripts/s_backbone/analyze_template_bank.py \
        --cache results/s_cache/s_scores_temporal.json \
        --manifest perturbations_temporal/pairs_manifest.json \
        --baselines results/baselines/baseline_scores_temporal.json \
        --out results/s_cache/template_bank_temporal.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SAME_IDENTITY = {"control", "pitch_shift", "time_stretch", "lowpass", "distortion"}
DIFF_IDENTITY = {"style_swap"}


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    uniq, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            ranks[inv == i] = ranks[inv == i].mean()
    return float((ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def dz(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(v.mean() / (v.std(ddof=1) + 1e-12)) if v.size > 1 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(a):
        r = np.empty(len(a), float)
        r[np.argsort(a, kind="mergesort")] = np.arange(1, len(a) + 1)
        _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
        for i, c in enumerate(cnt):
            if c > 1:
                r[inv == i] = r[inv == i].mean()
        return r
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, required=True,
                    help="an S cache WITH a per_template block (s_scores_pt.json / temporal)")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--baselines", type=Path, default=None,
                    help="baseline_scores*.json for the clap_htsat correlation")
    ap.add_argument("--out", type=Path, default=Path("results/template_bank.json"))
    args = ap.parse_args()

    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    per_t = cache.get("per_template")
    if not per_t:
        print(f"{args.cache} has no per_template block (was it scored with the "
              f"post-T7.1 score_s_mf.py?)", file=sys.stderr)
        return 1
    order = cache["meta"].get("per_template_order") or [f"t{i+1}" for i in range(4)]
    n_t = len(next(iter(per_t.values())))
    labels = [f"t{i+1}" for i in range(n_t)]
    token_agg = cache["meta"].get("token_agg", "?")

    pairs = json.loads(args.manifest.read_text(encoding="utf-8"))["pairs"]
    meta = {p["pair_id"]: p for p in pairs}
    perts = sorted({meta[pid]["perturbation"] for pid in per_t if pid in meta}
                   - {"control"})
    has_identity = "style_swap" in perts       # matrix vs temporal

    # per-template + mean columns, aligned to pair_ids present in the cache
    pids = [pid for pid in per_t if pid in meta]
    cols = {labels[i]: {pid: per_t[pid][i] for pid in pids} for i in range(n_t)}
    cols["mean"] = {pid: float(np.mean(per_t[pid])) for pid in pids}
    metric_names = labels + ["mean"]

    print(f"# Template-bank analysis: {args.cache.name}  (token_agg={token_agg})\n")
    print("Template bank:")
    for lab, q in zip(labels, order):
        flag = "CONTINUITY" if lab == "t1" else "identity"
        print(f"- **{lab}** [{flag}]: {q}")
    out: dict = {"cache": str(args.cache), "token_agg": token_agg,
                 "templates": dict(zip(labels, order))}

    # controls per source, per column
    ctrl = {m: {meta[pid]["source_id"]: cols[m][pid]
                for pid in pids if meta[pid]["perturbation"] == "control"}
            for m in metric_names}

    # ---- A. per-template drop table ----
    print("\n## A. Per-template drop (control - perturbed; mean, d_z in parens)\n")
    print("| perturbation | " + " | ".join(metric_names) + " |")
    print("|---|" + "---|" * len(metric_names))
    drop_tbl: dict = {}
    for pert in perts:
        cells, rec = [], {}
        for m in metric_names:
            paired = [ctrl[m][meta[pid]["source_id"]] - cols[m][pid]
                      for pid in pids if meta[pid]["perturbation"] == pert
                      and meta[pid]["source_id"] in ctrl[m]]
            a = np.array(paired, float)
            rec[m] = {"mean": float(a.mean()), "dz": dz(a)} if a.size else None
            cells.append(f"{a.mean():+.3f} ({dz(a):+.2f})" if a.size else "-")
        drop_tbl[pert] = rec
        print(f"| {pert} | " + " | ".join(cells) + " |")
    print("| _control level_ | " + " | ".join(
        f"{np.mean(list(ctrl[m].values())):.3f}" if ctrl[m] else "-" for m in metric_names) + " |")
    out["drop_table"] = drop_tbl

    # ---- B. per-template AUC ----
    print("\n## B. Per-template AUC\n")
    auc_rec: dict = {}
    if has_identity:
        splits = {
            "transpose_vs_styleswap": (lambda p: p["perturbation"] == "pitch_shift",
                                       lambda p: p["perturbation"] == "style_swap"),
            "broad_same_vs_styleswap": (lambda p: p["perturbation"] in SAME_IDENTITY,
                                        lambda p: p["perturbation"] == "style_swap"),
            "cross_genre": (lambda p: p["perturbation"] in SAME_IDENTITY,
                            lambda p: p["perturbation"] == "style_swap" and "cross" in p["pair_id"]),
            "same_genre": (lambda p: p["perturbation"] in SAME_IDENTITY,
                           lambda p: p["perturbation"] == "style_swap" and "same" in p["pair_id"]),
        }
        print("| split | " + " | ".join(metric_names) + " |")
        print("|---|" + "---|" * len(metric_names))
        for sname, (pos_f, neg_f) in splits.items():
            cells, rec = [], {}
            for m in metric_names:
                s, y = [], []
                for pid in pids:
                    p = meta[pid]
                    if pos_f(p):
                        s.append(cols[m][pid]); y.append(1)
                    elif neg_f(p):
                        s.append(cols[m][pid]); y.append(0)
                a = auc(np.array(s, float), np.array(y))
                rec[m] = a
                cells.append(f"{a:.3f}")
            auc_rec[sname] = rec
            print(f"| {sname} | " + " | ".join(cells) + " |")
    # control-vs-violation AUC (works for both; the key temporal test)
    print("\n### control vs ALL perturbations  (1.0 = ranks controls above; 0.5 = blind)\n")
    print("| " + " | ".join(metric_names) + " |")
    print("|" + "---|" * len(metric_names))
    cvv = {}
    cells = []
    for m in metric_names:
        s, y = [], []
        for pid in pids:
            s.append(cols[m][pid]); y.append(1 if meta[pid]["perturbation"] == "control" else 0)
        a = auc(np.array(s, float), np.array(y))
        cvv[m] = a
        cells.append(f"{a:.3f}")
    print("| " + " | ".join(cells) + " |")
    auc_rec["control_vs_violation"] = cvv
    out["auc"] = auc_rec

    # ---- C. correlation with clap_htsat + with mean S ----
    print("\n## C. Redundancy: per-template Spearman correlation\n")
    corr: dict = {}
    clap = None
    if args.baselines and args.baselines.is_file():
        bs = json.loads(args.baselines.read_text())["scores"]
        clap = {pid: bs[pid]["clap_htsat"] for pid in pids
                if pid in bs and "clap_htsat" in bs[pid]}
    print("| template | rho vs clap_htsat | rho vs mean-S |")
    print("|---|---:|---:|")
    mean_vec = np.array([cols["mean"][pid] for pid in pids], float)
    for m in labels:
        v = np.array([cols[m][pid] for pid in pids], float)
        r_clap = float("nan")
        if clap:
            shared = [pid for pid in pids if pid in clap]
            r_clap = spearman(np.array([cols[m][pid] for pid in shared], float),
                              np.array([clap[pid] for pid in shared], float))
        r_mean = spearman(v, mean_vec)
        corr[m] = {"rho_clap_htsat": r_clap, "rho_mean_s": r_mean}
        print(f"| {m} | {r_clap:+.3f} | {r_mean:+.3f} |")
    out["correlation"] = corr
    if clap:
        print("\n_Hypothesis check: identity templates (t2-t4) should correlate MORE with "
              "clap_htsat than the continuity template (t1) does._")

    # ---- D. (T7.3) dataset-centered log-odds ----
    if has_identity:
        print("\n## D. Dataset-centered log-odds rescoring (T7.3) vs raw mean\n")
        ctrl_pids = [pid for pid in pids if meta[pid]["perturbation"] == "control"]
        centers = [np.mean(logit(np.array([per_t[pid][i] for pid in ctrl_pids], float)))
                   for i in range(n_t)]
        centered_mean = {pid: float(np.mean([logit(np.array([per_t[pid][i]]))[0] - centers[i]
                                             for i in range(n_t)])) for pid in pids}
        print("| split | raw-mean AUC | centered-logodds AUC | Δ |")
        print("|---|---:|---:|---:|")
        lo_rec = {}
        for sname, (pos_f, neg_f) in splits.items():
            def collect(scoremap):
                s, y = [], []
                for pid in pids:
                    p = meta[pid]
                    if pos_f(p):
                        s.append(scoremap[pid]); y.append(1)
                    elif neg_f(p):
                        s.append(scoremap[pid]); y.append(0)
                return auc(np.array(s, float), np.array(y))
            a_raw, a_cen = collect(cols["mean"]), collect(centered_mean)
            lo_rec[sname] = {"raw": a_raw, "centered": a_cen, "delta": a_cen - a_raw}
            print(f"| {sname} | {a_raw:.3f} | {a_cen:.3f} | {a_cen - a_raw:+.3f} |")
        out["logodds_centering"] = lo_rec

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
