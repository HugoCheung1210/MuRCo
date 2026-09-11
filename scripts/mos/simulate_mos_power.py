#!/usr/bin/env python
"""Power simulation for the human-MOS study (E-E). DSP env, read-only, no GPU.

Question: given the design AS BUILT (results/mos/stimuli_plan.json filtered to
what Part 1 actually contains: 154 matrix pairs after the §E withdrawals and the
2026-08-07 ai_rerank removal), can the study (a) detect S's contribution to C
(held-out Δρ with-vs-without S) and (b) show C beats clap_htsat on MOS
correlation — under plausible rater noise?

Run it matrix-only (the default). `--keep-sets all` reproduces the superseded
pre-2026-08-07 instrument, which is where the Δρ_S ≈ 0.12 / P = 1.00 figures came
from; those describe an instrument that no longer exists and must not be quoted.

Simulation model
  latent truth  t_p = Σ_d w*_d · log s_d(p)  + construct noise (pair-level,
                controls how much human "coherence" deviates from any metric;
                construct_r2 = share of latent variance the dims explain)
  rating       r_{p,k} = scale(t_p) + rater bias + residual, rounded & clipped to 1–5
                noise calibrated to a target SINGLE-RATING ICC(1)
  analysis     exactly the frozen plan: per-pair MOS = mean rating; 5-fold CV
               grouped by source track; NNLS fit of log-dims -> log MOS on train
               (compute_C.py machinery); held-out Spearman ρ.

Scenarios for the true weights w*: uniform (S matters, w*_S=0.25) and
S-absent (w*_S=0). Power(detect S) = P(Δρ = ρ_HTRS − ρ_HTR > 0) under uniform;
false-positive rate = same probability under S-absent.

CAVEAT (report in the write-up): under truths generated FROM the dims, C's win
over clap is partly assumed; the C-vs-clap row is a sanity check of sensitivity,
not evidence about the real world. The Δρ(S) power number is the load-bearing one.

Usage: python scripts/mos/simulate_mos_power.py [--n-sim 200] [--icc 0.3,0.5,0.7]
Output: results/mos/power_simulation.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import spearmanr

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
DIMS = ["H", "T", "R", "S"]
EPS = 1e-6


def load_design(keep_sets="matrix", withdrawn="results/mos/withdrawn_sources.json"):
    """The design as BUILT, not as planned.

    `stimuli_plan.json` is the frozen sampler output and still contains everything it
    ever sampled. Two filters are applied downstream by `make_qualtrics_loops.py` and
    must be applied here too, or the simulation describes an instrument that is not the
    one collecting data: the §E source withdrawals, and the ai_rerank set, which left
    Part 1 on 2026-08-07 (doc/notes/ai_rerank_removed_from_part1.md). Together they take
    the plan's 190 stimuli to the live 154.
    """
    plan = json.load(open(ROOT / "results/mos/stimuli_plan.json"))
    stim = plan["stimuli"]
    n_planned = len(stim)
    if keep_sets != "all":
        keep = {x.strip() for x in keep_sets.split(",") if x.strip()}
        stim = [s for s in stim if s["stimulus_set"] in keep]
    wpath = ROOT / withdrawn if withdrawn else None
    wsrc = set()
    if wpath and wpath.is_file():
        wsrc = set(json.load(open(wpath)).get("withdrawn", ()))
        stim = [s for s in stim if s["source_id"] not in wsrc]
    plan["meta"] = dict(plan["meta"], n_pairs=len(stim), n_planned=n_planned,
                        kept_sets=keep_sets, n_withdrawn_sources=len(wsrc))
    base = dict(json.load(open(ROOT / "results/baselines/baseline_scores.json"))["scores"])
    # AI-rerank pairs carry their own baselines under a backend-namespaced id
    for tag, rel in (("rerank_sao", "results/rerank"), ("rerank_ace", "results/rerank_ace")):
        p = ROOT / rel / "baseline_scores_rerank.json"
        if p.exists():
            base.update({f"{tag}/{k}": v
                         for k, v in json.load(open(p))["scores"].items()})
    logs, clap, n_ratings, tracks, ids = [], [], [], [], []
    n_anchor = plan["meta"]["ratings_per_anchor_pair"]
    n_block = plan["meta"]["ratings_per_block_pair"]
    for s in stim:
        logs.append([np.log(np.clip(s[f"score_{d}"], EPS, 1.0)) for d in DIMS])
        clap.append(base.get(s["pair_id"], {}).get("clap_htsat", np.nan))
        n_ratings.append(n_anchor if s["block"] == "anchor" else n_block)
        tracks.append(s["source_id"])
        ids.append(s["pair_id"])
    return (np.array(logs), np.array(clap), np.array(n_ratings),
            np.array(tracks), ids, plan["meta"])


def simulate_ratings(latent, n_ratings, icc, rng):
    """Per-pair mean rating on 1-5, noise calibrated to single-rating ICC(1)."""
    lat = (latent - latent.min()) / max(latent.max() - latent.min(), EPS)
    scale = 1.0 + 4.0 * lat                     # map to 1..5
    var_pair = scale.var()
    sd_noise = np.sqrt(var_pair * (1 - icc) / max(icc, EPS))
    mos = np.empty(len(scale))
    for i, (mu, n) in enumerate(zip(scale, n_ratings)):
        r = mu + rng.normal(0, sd_noise, size=n)
        r = np.clip(np.round(r), 1, 5)          # Likert rounding
        mos[i] = r.mean()
    return mos


def nnls_fit_predict(X_tr, y_tr, X_te):
    """compute_C.py fit_weights logic: NNLS of log-dims onto log(MOS/5)."""
    y = np.log(np.clip(y_tr / 5.0, EPS, 1.0))
    coef, _ = nnls(X_tr, y)
    if coef.sum() <= 0:
        coef = np.ones(X_tr.shape[1])
    return X_te @ coef


def cv_rho(X, mos, tracks, dims_idx, rng):
    """5-fold CV grouped by track -> pooled held-out predictions -> Spearman."""
    uniq = np.array(sorted(set(tracks)))
    rng.shuffle(uniq)
    folds = np.array_split(uniq, 5)
    pred = np.full(len(mos), np.nan)
    for fold in folds:
        te = np.isin(tracks, fold)
        tr = ~te
        pred[te] = nnls_fit_predict(X[tr][:, dims_idx], mos[tr], X[te][:, dims_idx])
    return spearmanr(pred, mos).correlation


def run(n_sim, icc_levels, construct_r2, seed, keep_sets="matrix"):
    X, clap, n_ratings, tracks, ids, meta = load_design(keep_sets)
    scenarios = {"uniform_wS": np.array([0.25, 0.25, 0.25, 0.25]),
                 "no_S": np.array([1 / 3, 1 / 3, 1 / 3, 0.0])}
    rng = np.random.default_rng(seed)
    results = {}
    for scen, w in scenarios.items():
        for icc in icc_levels:
            d_rhos, rho_full_l, rho_htr_l, rho_clap_l, beats_clap = [], [], [], [], []
            for _ in range(n_sim):
                latent = X @ w
                # construct noise: dims explain construct_r2 of latent variance
                sd_c = latent.std() * np.sqrt((1 - construct_r2) / construct_r2)
                latent_h = latent + rng.normal(0, sd_c, size=len(latent))
                mos = simulate_ratings(latent_h, n_ratings, icc, rng)
                rho_full = cv_rho(X, mos, tracks, [0, 1, 2, 3], rng)
                rho_htr = cv_rho(X, mos, tracks, [0, 1, 2], rng)
                ok = ~np.isnan(clap)
                rho_clap = spearmanr(clap[ok], mos[ok]).correlation
                d_rhos.append(rho_full - rho_htr)
                rho_full_l.append(rho_full)
                rho_htr_l.append(rho_htr)
                rho_clap_l.append(rho_clap)
                beats_clap.append(rho_full > rho_clap)
            d = np.array(d_rhos)
            results[f"{scen}|icc={icc}"] = {
                "mean_rho_HTRS": round(float(np.mean(rho_full_l)), 4),
                "mean_rho_HTR": round(float(np.mean(rho_htr_l)), 4),
                "mean_rho_clap": round(float(np.mean(rho_clap_l)), 4),
                "mean_delta_rho_S": round(float(d.mean()), 4),
                "ci95_delta_rho_S": [round(float(np.percentile(d, 2.5)), 4),
                                     round(float(np.percentile(d, 97.5)), 4)],
                "p_delta_pos": round(float((d > 0).mean()), 3),
                "p_C_beats_clap": round(float(np.mean(beats_clap)), 3),
            }
    return results, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--icc", default="0.3,0.5,0.7",
                    help="single-rating ICC levels to simulate")
    ap.add_argument("--construct-r2", type=float, default=0.7,
                    help="share of perceptual-coherence variance the dims explain")
    ap.add_argument("--out", default="results/mos/power_simulation.json",
                    help="give every variant its own path; the default is the\n                         canonical matrix-only run")
    ap.add_argument("--keep-sets", default="matrix",
                    help="stimulus_set values Part 1 actually contains, or 'all'" 
                         " to simulate the pre-2026-08-07 instrument")
    ap.add_argument("--seed", type=int, default=20260718)
    args = ap.parse_args()
    icc_levels = [float(x) for x in args.icc.split(",")]
    results, design_meta = run(args.n_sim, icc_levels, args.construct_r2, args.seed,
                              args.keep_sets)
    out = {
        "meta": {
            "n_sim": args.n_sim, "icc_levels": icc_levels,
            "construct_r2": args.construct_r2, "seed": args.seed,
            "design": {k: design_meta[k] for k in
                       ("n_pairs", "n_anchor", "ratings_per_anchor_pair",
                        "ratings_per_block_pair", "n_participants",
                        "n_planned", "kept_sets", "n_withdrawn_sources")},
            "interpretation": {
                "p_delta_pos under uniform_wS": "power to detect S (want high)",
                "p_delta_pos under no_S": "false-positive rate (want ~0.5 or below; "
                                          "delta centred on 0)",
                "caveat": "truth generated from the dims; C-vs-clap row is a "
                          "sensitivity check, not real-world evidence",
            },
        },
        "results": results,
    }
    path = ROOT / args.out
    json.dump(out, open(path, "w"), indent=1)
    hdr = f"{'scenario':<22}{'rho_HTRS':>9}{'rho_HTR':>9}{'rho_clap':>9}{'d_rho_S':>9}{'P(d>0)':>8}{'P(C>clap)':>10}"
    print(hdr)
    for k, v in results.items():
        print(f"{k:<22}{v['mean_rho_HTRS']:>9}{v['mean_rho_HTR']:>9}"
              f"{v['mean_rho_clap']:>9}{v['mean_delta_rho_S']:>9}"
              f"{v['p_delta_pos']:>8}{v['p_C_beats_clap']:>10}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
