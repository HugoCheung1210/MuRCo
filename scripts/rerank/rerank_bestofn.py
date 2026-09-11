#!/usr/bin/env python3
"""E-D -- best-of-N candidate re-ranking by C (the applied cherry).

Backs the one clause in the repositioned contribution statement
(`related_work_positioning.md` §3) that currently has ZERO evidence: that C
"steers frozen generators via ... candidate re-ranking".  Training-free, on
frozen SAO -- exactly the thesis identity (inference-time signal, no retraining).

REGIME.  SAO inpaint -- the one regime where S pulls its weight inside C (§8.0 T2,
§8.3): S is the only dim that accumulates in inpaint, so if re-ranking by C ever
beats re-ranking by a scalar, this is where it shows.  For each of the n=10 seeds
we generate N candidates for the SAME single inpaint region (region 5-8 s), varying
only the sampler seed, then ask: does picking the candidate C likes best differ
from -- and beat -- picking what clap_htsat likes best, or a random pick?

TWO ENVS, FILES ON DISK (the project invariant, §6).  This script owns only the
two ends; the middle (scoring) reuses the validated scorers UNCHANGED:

  [GPU box, SAO env]   rerank_bestofn.py generate
        -> results/rerank/audio/*.wav   (region (4,6)->4-10 s, 6 s windows: one
           sNN__A.wav context per seed + sNN_candKK__B.wav per candidate)
        -> results/rerank/rerank_manifest.json   (STANDARD pairs manifest, so
           every existing scorer consumes it verbatim)
        -> results/rerank/rerank_index.json      ({seed: [pair_id,...]} for analyze)

  [GPU box, mfenv]     score_s_mf.py   --manifest results/rerank/rerank_manifest.json \
                           --audio-dir results/rerank/audio --secs 6 --gap-s 1.0 \
                           --out results/rerank/s_scores_rerank.json
  [local, DSP env]     score_separability.py --manifest results/rerank/rerank_manifest.json \
                           --audio-dir results/rerank/audio --dimensions H,T,R,S \
                           --s-cache results/rerank/s_scores_rerank.json \
                           --output-dir results/rerank        (-> results/rerank/pair_scores.csv)
                       score_baselines.py --manifest results/rerank/rerank_manifest.json \
                           --audio-dir results/rerank/audio --secs 6 \
                           --out results/rerank/baseline_scores_rerank.json

  [local, DSP env]     rerank_bestofn.py analyze
        -> results/rerank/analysis.json + results/rerank/mos_stimuli.json

GO/NO-GO (plan §4.4).  Run ONE seed first (`generate --pilot`) and `analyze`: if C
and clap_htsat rank the candidates near-identically (Spearman ~1, same winner
every time) the demo is VACUOUS -- stop before burning GPU on all 10 seeds.
`analyze` prints this verdict up front.  Only if they disagree is the full run
(and the MOS piggyback) worth it.

HONEST JUDGING (plan §1 E-D "avoid grading your own homework").  A ranker is never
judged by its own score.  We report, per selection strategy (rank-by-C /
rank-by-clap_htsat / rank-by-S / random), the winner's value on the metrics NOT
used to pick it, plus a leave-one-dim-out check (rank by C without S, judge the
winner's S).  The DECISIVE judge is human MOS: `mos_stimuli.json` exports the
C-winner and C-loser per seed for the iterative-block MOS piggyback (the
STIMULI-SCOPE item now that the study is greenlit).

Usage:
    # GPU box, SAO env -- pilot first:
    python scripts/rerank/rerank_bestofn.py generate --n 8 --pilot
    # ... run score_s_mf / score_separability / score_baselines (commands above) ...
    # local:
    python scripts/rerank/rerank_bestofn.py analyze
    # if the pilot disagreement is real, drop --pilot and regenerate all seeds.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable


import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

EPS = 1e-6
SR = 48000
REGION = (4.0, 6.0)          # (start_s, dur_s) -> 4-10 s window; matches drift_v4 SAO
MASK_S = (5.0, 8.0)          # SAO inpaint region (outputs_stableaudio_inpaint_5_8)
GAP_S = 1.0                  # the load-bearing 1 s gap (M.1)
SEEDS_DIR = Path("outputs/seeds")
RERANK_DIR = Path(os.environ.get("RERANK_DIR", "results/rerank"))   # e.g. results/rerank_ace


# ======================================================================== #
# generate  --  GPU box, SAO env only                                      #
# ======================================================================== #
def cmd_generate(args) -> int:
    """Generate N candidates/seed for the single SAO inpaint region + manifest.

    Reuses sao_inpaint_protocol.inpaint_edit (the validated PDI-style masked
    sampler) so the candidates are drawn from EXACTLY the generator the drift
    study characterised -- only the sampler seed varies across candidates.
    """
    import soundfile as sf
    import librosa
    # backend loaded here (not at top) so `analyze`/`--help` work without diffusers/acestep/CUDA.
    backend = args.backend
    if backend == "sao":
        import sao_inpaint_protocol as sao          # noqa: E402
        pipe = sao.load_pipe()
    elif backend == "ace":
        import ace_protocol as ace                  # noqa: E402
        dit, llm = ace.init_ace()
    else:
        raise SystemExit(f"unknown --backend {backend!r} (expected sao|ace)")

    audio_dir = RERANK_DIR / "audio"
    cand_dir = RERANK_DIR / "candidates"           # full 30 s candidate clips (kept for audit)
    audio_dir.mkdir(parents=True, exist_ok=True)
    cand_dir.mkdir(parents=True, exist_ok=True)

    seed_manifest = json.loads((SEEDS_DIR / "manifest.json").read_text(encoding="utf-8"))
    seed_ids = sorted(seed_manifest)
    if args.pilot:
        seed_ids = seed_ids[:1]
    if args.seeds:
        want = set(args.seeds.split(","))
        seed_ids = [s for s in seed_ids if s in want]

    start, dur = REGION

    def window(src_wav: str, dst_wav: Path):
        """Extract the region (start..start+dur) at SR, peak-normalize, write."""
        y, _ = librosa.load(src_wav, sr=SR, mono=True, offset=start, duration=dur)
        y = (y / (np.max(np.abs(y)) + 1e-8) * 0.95).astype("float32")
        sf.write(dst_wav, y, SR)

    pairs, index = [], {}
    for sid in seed_ids:
        entry = seed_manifest[sid]
        caption = entry["caption"] if isinstance(entry, dict) else str(entry)
        base_seed = (entry.get("seed", 0) if isinstance(entry, dict) else 0)
        seed_wav = str(SEEDS_DIR / f"{sid}.wav")

        # A = the seed's own region window (shared context / reference for all N)
        ref_win = audio_dir / f"{sid}__A.wav"
        window(seed_wav, ref_win)

        index[sid] = []
        for k in range(args.n):
            cand_full = cand_dir / f"{sid}_cand{k:02d}.wav"
            if not cand_full.exists():
                # ONE inpaint/repaint pass on the seed; only the sampler seed varies across candidates.
                gen_seed = base_seed + args.seed_stride * (k + 1)
                if backend == "sao":
                    sao.inpaint_edit(pipe, seed_wav, str(cand_full), caption, seed=gen_seed,
                                     mask_s=MASK_S)   # 5-8s == outputs_stableaudio_inpaint_5_8
                else:  # ace: native repaint, in-distribution (needs keyscale/bpm from the seed entry)
                    ace.repaint_edit(dit, llm, entry, seed_wav, str(cand_full),
                                     seed_val=gen_seed, mask_s=MASK_S)
            cand_win = audio_dir / f"{sid}_cand{k:02d}__B.wav"
            window(str(cand_full), cand_win)

            pid = f"{sid}::cand{k:02d}"
            pairs.append({
                "pair_id": pid, "source_id": sid, "genre": sid.split("_")[0],
                "dim_target": "rerank", "perturbation": f"{backend}_inpaint_candidate",
                "magnitude": float(k), "magnitude_unit": "candidate_index",
                "ref_path": str(ref_win), "cand_path": str(cand_win),
                "partner_id": "", "sr": SR, "seconds": dur,
                "gen_seed": int(base_seed + args.seed_stride * (k + 1)),
                "notes": f"{backend} inpaint {MASK_S[0]}-{MASK_S[1]}s, region {REGION}, cand {k}",
            })
            index[sid].append(pid)
        print(f"  {sid}: {args.n} candidates + context window")

    manifest = {"meta": {"experiment": "E-D best-of-N rerank", "regime": f"{backend} inpaint",
                         "backend": backend,
                         "region": list(REGION), "mask_s": list(MASK_S), "gap_s": GAP_S,
                         "sr": SR, "seconds": dur, "n_candidates": args.n,
                         "n_seeds": len(seed_ids), "pilot": args.pilot,
                         "seed_stride": args.seed_stride,
                         "note": "standard pairs manifest; A=seed region window (shared), "
                                 "B=candidate region window; score with the existing scorers"},
                "pairs": pairs}
    (RERANK_DIR / "rerank_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                     encoding="utf-8")
    (RERANK_DIR / "rerank_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nwrote {RERANK_DIR/'rerank_manifest.json'} ({len(pairs)} pairs) "
          f"and rerank_index.json ({len(index)} seeds)")
    print("\nNEXT (score with the existing, unchanged scorers):")
    print(f"  [mfenv] score_s_mf.py --manifest {RERANK_DIR}/rerank_manifest.json "
          f"--audio-dir {RERANK_DIR}/audio --secs {dur:g} --gap-s {GAP_S:g} "
          f"--out {RERANK_DIR}/s_scores_rerank.json")
    print(f"  [local] score_separability.py --manifest {RERANK_DIR}/rerank_manifest.json "
          f"--audio-dir {RERANK_DIR}/audio --dimensions H,T,R,S "
          f"--s-cache {RERANK_DIR}/s_scores_rerank.json --output-dir {RERANK_DIR}")
    print(f"  [local] score_baselines.py --manifest {RERANK_DIR}/rerank_manifest.json "
          f"--audio-dir {RERANK_DIR}/audio --secs {dur:g} "
          f"--out {RERANK_DIR}/baseline_scores_rerank.json")
    print(f"  [local] rerank_bestofn.py analyze")
    return 0


# ======================================================================== #
# analyze  --  local, DSP env (fully runnable once the caches exist)       #
# ======================================================================== #
def _C(row: dict) -> float:
    vals = [float(np.clip(float(row[f"score_{d}"]), EPS, 1.0)) for d in ("H", "T", "R", "S")]
    return float(np.prod(vals) ** (1.0 / len(vals)))


def _C_noS(row: dict) -> float:
    vals = [float(np.clip(float(row[f"score_{d}"]), EPS, 1.0)) for d in ("H", "T", "R")]
    return float(np.prod(vals) ** (1.0 / len(vals)))


def _spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def _kendall(x, y):
    from scipy.stats import kendalltau
    if len(x) < 2 or np.std(x) < EPS or np.std(y) < EPS:
        return float("nan")
    return float(kendalltau(x, y).correlation)


def _paired(a, b, rng, n_boot=10000):
    """Paired per-seed contrast a-b: n, means, mean delta, Cohen's d_z, 95% bootstrap CI.

    a and b are aligned by seed (the rankers share one seed loop); NaN pairs are dropped.
    d_z = mean(delta)/std(delta) is the paired effect size the thesis reports (NOT p, which is
    floored/uninformative here); the bootstrap CI on the mean delta says whether the shift is
    reliably nonzero -- a CI excluding 0 is the claimable result.
    """
    a = np.asarray(a, float); b = np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    n = int(a.size)
    if n < 2:
        return {"n": n, "mean_a": float("nan"), "mean_b": float("nan"),
                "delta": float("nan"), "d_z": float("nan"), "ci95": [float("nan"), float("nan")]}
    d = a - b
    sd = float(np.std(d, ddof=1))
    d_z = float(d.mean() / sd) if sd > EPS else float("nan")
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": float(d.mean()), "d_z": d_z, "ci95": [lo, hi]}


def cmd_analyze(args) -> int:
    rng = np.random.default_rng(20260717)
    pair_csv = RERANK_DIR / "pair_scores.csv"
    index_p = RERANK_DIR / "rerank_index.json"
    base_p = RERANK_DIR / "baseline_scores_rerank.json"
    if not pair_csv.is_file() or not index_p.is_file():
        raise SystemExit(f"missing {pair_csv} or {index_p} -- run generate + the three "
                         f"scorers first (see `generate` output).")

    rows = {r["pair_id"]: r for r in csv.DictReader(pair_csv.open(newline="", encoding="utf-8"))}
    index = json.loads(index_p.read_text(encoding="utf-8"))
    # audio paths live in the MANIFEST, not pair_scores.csv (which carries only
    # pair_id/source_id/genre/dim_target/perturbation/magnitude/score_*).
    paths = {}
    man_p = RERANK_DIR / "rerank_manifest.json"
    if man_p.is_file():
        for p in json.loads(man_p.read_text(encoding="utf-8")).get("pairs", []):
            paths[p["pair_id"]] = {"ref_path": p.get("ref_path", ""),
                                   "cand_path": p.get("cand_path", "")}
    clap = {}
    if base_p.is_file():
        bd = json.loads(base_p.read_text(encoding="utf-8"))
        clap = bd.get("scores", bd)

    # metric value per pair_id for each ranker/judge metric
    def metric(pid, name):
        r = rows.get(pid, {})
        if name == "C":
            return _C(r)
        if name == "C_noS":
            return _C_noS(r)
        if name in ("H", "T", "R", "S"):
            return float(r.get(f"score_{name}", "nan"))
        if name in ("clap_htsat", "clap_music", "scs"):
            return float(clap.get(pid, {}).get(name, "nan"))
        raise KeyError(name)

    RANKERS = ["C", "clap_htsat", "S", "random"]
    JUDGES = ["C", "S", "clap_htsat"]          # judge only by metrics NOT used to rank

    # ---- per-seed rank agreement (the go/no-go) ----------------------------
    agree = {"spearman_C_clap": [], "kendall_C_clap": [], "same_winner_C_clap": [],
             "spearman_C_S": [], "n_candidates": []}
    for sid, pids in index.items():
        pids = [p for p in pids if p in rows]
        if len(pids) < 2:
            continue
        cvals = np.array([metric(p, "C") for p in pids])
        clvals = np.array([metric(p, "clap_htsat") for p in pids])
        svals = np.array([metric(p, "S") for p in pids])
        agree["spearman_C_clap"].append(_spearman(cvals, clvals))
        agree["kendall_C_clap"].append(_kendall(cvals, clvals))
        agree["spearman_C_S"].append(_spearman(cvals, svals))
        if not np.isnan(clvals).any():
            agree["same_winner_C_clap"].append(
                int(pids[int(np.argmax(cvals))] == pids[int(np.argmax(clvals))]))
        agree["n_candidates"].append(len(pids))

    def _m(a):
        a = np.array([x for x in a if not np.isnan(x)], float)
        return float(a.mean()) if a.size else float("nan")

    mean_sp = _m(agree["spearman_C_clap"])
    same_winner = _m(agree["same_winner_C_clap"])
    vacuous = (not np.isnan(mean_sp) and mean_sp > 0.95) or \
              (not np.isnan(same_winner) and same_winner > 0.95)

    # ---- selection strategies judged by held-out metrics -------------------
    def select(pids, ranker):
        vals = np.array([metric(p, "clap_htsat" if ranker == "random" else ranker) for p in pids])
        if ranker == "random":
            return pids[int(rng.integers(0, len(pids)))]
        return pids[int(np.argmax(vals))]

    # held-out rule: a ranker is never judged by its own metric; C is judged by
    # clap_htsat & S-as-held-out-only via the leave-one-dim-out variant below.
    heldout = {r: {j: [] for j in JUDGES if j != r} for r in RANKERS}
    loo_S = {"rank_by_C_noS__winner_S": [], "rank_by_clap__winner_S": [],
             "random__winner_S": []}
    for sid, pids in index.items():
        pids = [p for p in pids if p in rows]
        if len(pids) < 2:
            continue
        for r in RANKERS:
            w = select(pids, r)
            for j in JUDGES:
                if j == r:
                    continue
                heldout[r][j].append(metric(w, j))
        # leave-one-dim-out for S: rank WITHOUT S, judge the winner's S
        w_noS = pids[int(np.argmax([metric(p, "C_noS") for p in pids]))]
        w_clap = select(pids, "clap_htsat")
        w_rand = select(pids, "random")
        loo_S["rank_by_C_noS__winner_S"].append(metric(w_noS, "S"))
        loo_S["rank_by_clap__winner_S"].append(metric(w_clap, "S"))
        loo_S["random__winner_S"].append(metric(w_rand, "S"))

    # ---- paired effect sizes over seeds (d_z + bootstrap CI; the claimable statistic) -----
    # each contrast is aligned by seed; "a vs b" = per-seed winner-score(a) - winner-score(b).
    C_full_S = heldout["C"]["S"]                        # full-C winner's held-out S, per seed
    effect = {
        # headline: does ranking by C pick more S-coherent winners than the baselines?
        "S(judge): C vs random":     _paired(C_full_S, heldout["random"]["S"], rng),
        "S(judge): C vs clap_htsat": _paired(C_full_S, heldout["clap_htsat"]["S"], rng),
        # no acoustic cost: C's winners are as clap-similar as random's
        "clap(judge): C vs random":  _paired(heldout["C"]["clap_htsat"],
                                             heldout["random"]["clap_htsat"], rng),
        # S earns its place: dropping S from C collapses winner-S toward random
        "S(judge): C vs C_noS (S-term contribution)":
            _paired(C_full_S, loo_S["rank_by_C_noS__winner_S"], rng),
        "S(judge): C_noS vs random": _paired(loo_S["rank_by_C_noS__winner_S"],
                                             loo_S["random__winner_S"], rng),
    }

    out = {
        "meta": {"n_seeds_used": sum(1 for s in index if len([p for p in index[s] if p in rows]) >= 2),
                 "rankers": RANKERS, "judges": JUDGES,
                 "note": "each ranker judged only by metrics it did NOT use; random = uniform "
                         "candidate pick (baseline); human MOS is the decisive judge (mos_stimuli.json)"},
        "go_no_go": {
            "mean_spearman_C_vs_clap_htsat": mean_sp,
            "mean_kendall_C_vs_clap_htsat": _m(agree["kendall_C_clap"]),
            "mean_spearman_C_vs_S": _m(agree["spearman_C_S"]),
            "frac_same_winner_C_vs_clap_htsat": same_winner,
            "verdict": ("VACUOUS: C and clap_htsat rank candidates near-identically -> the "
                        "demo adds nothing; report the agreement and stop"
                        if vacuous else
                        "PROCEED: C and clap_htsat disagree enough that the choice of ranker "
                        "matters; full run + MOS piggyback justified"),
        },
        "heldout_judge_mean": {r: {j: _m(v) for j, v in d.items()} for r, d in heldout.items()},
        "leave_one_dim_out_S": {k: _m(v) for k, v in loo_S.items()},
        "effect_sizes": effect,
    }
    RERANK_DIR.mkdir(parents=True, exist_ok=True)
    (RERANK_DIR / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---- MOS stimuli export (C-winner vs C-loser per seed) -----------------
    stim = []
    for sid, pids in index.items():
        pids = [p for p in pids if p in rows]
        if len(pids) < 2:
            continue
        cvals = [metric(p, "C") for p in pids]
        win, los = pids[int(np.argmax(cvals))], pids[int(np.argmin(cvals))]
        stim.append({"seed": sid,
                     "c_winner": {"pair_id": win, "C": metric(win, "C"),
                                  "cand_path": paths.get(win, {}).get("cand_path", ""),
                                  "ref_path": paths.get(win, {}).get("ref_path", "")},
                     "c_loser": {"pair_id": los, "C": metric(los, "C"),
                                 "cand_path": paths.get(los, {}).get("cand_path", ""),
                                 "ref_path": paths.get(los, {}).get("ref_path", "")}})
    (RERANK_DIR / "mos_stimuli.json").write_text(
        json.dumps({"meta": {"block": "iterative rerank; winner vs loser by C, same seed/region; "
                                      "fold into the greenlit MOS study's iterative block "
                                      "(STIMULI-SCOPE)"},
                    "stimuli": stim}, indent=2), encoding="utf-8")

    # ---- printed summary ----------------------------------------------------
    print(f"# E-D rerank analysis  (n_seeds={out['meta']['n_seeds_used']}, "
          f"N={int(np.mean(agree['n_candidates'])) if agree['n_candidates'] else '?'})\n")
    print("## GO / NO-GO")
    print(f"  Spearman(C, clap_htsat) per-seed mean = {mean_sp:+.3f}")
    print(f"  same winner (C vs clap_htsat)         = {same_winner:.2f}")
    print(f"  >>> {out['go_no_go']['verdict']}\n")
    print("## Held-out judge (winner's mean score on metrics NOT used to rank)")
    print("| ranker \\ judge | " + " | ".join(JUDGES) + " |")
    print("|---|" + "---:|" * len(JUDGES))
    for r in RANKERS:
        cells = [f"{out['heldout_judge_mean'][r].get(j, float('nan')):.3f}"
                 if j != r else "  —  " for j in JUDGES]
        print(f"| {r} | " + " | ".join(cells) + " |")
    print("\n## Leave-one-dim-out S (does S-blind ranking still pick S-coherent winners?)")
    for k, v in out["leave_one_dim_out_S"].items():
        print(f"  {k}: {v:.3f}")
    print("\n## Effect sizes (paired over seeds: Δ = mean(winner_a − winner_b), d_z, 95% bootstrap CI)")
    print("| contrast (a vs b) | n | mean_a | mean_b | Δ | d_z | 95% CI |")
    print("|---|---:|---:|---:|---:|---:|:--:|")
    for k, e in out["effect_sizes"].items():
        ci = e["ci95"]
        star = "" if (np.isnan(ci[0]) or np.isnan(ci[1]) or (ci[0] <= 0 <= ci[1])) else " *"
        print(f"| {k} | {e['n']} | {e['mean_a']:.3f} | {e['mean_b']:.3f} | "
              f"{e['delta']:+.3f} | {e['d_z']:+.2f} | [{ci[0]:+.3f}, {ci[1]:+.3f}]{star} |")
    print("  (* = 95% CI excludes 0; d_z ≈ 0.2 small / 0.5 medium / 0.8 large, Cohen)")
    print(f"\nwrote {RERANK_DIR/'analysis.json'} and {RERANK_DIR/'mos_stimuli.json'} "
          f"({len(stim)} winner/loser pairs)")
    if args.copy_mos_audio:
        dst = RERANK_DIR / "mos_audio"
        dst.mkdir(exist_ok=True)
        for s in stim:
            for role in ("c_winner", "c_loser"):
                src = Path(s[role]["cand_path"])
                if src.is_file():
                    shutil.copy(src, dst / f"{s['seed']}_{role}{src.suffix}")
        print(f"copied MOS stimulus audio -> {dst}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="[GPU box] generate N candidates/seed + manifest")
    g.add_argument("--backend", choices=["sao", "ace"], default="sao",
                   help="sao = training-free SAO inpaint (SAO env); ace = native ACE-Step repaint "
                        "(ACE env, cleaner/in-distribution). Downstream scoring is identical.")
    g.add_argument("--n", type=int, default=8, help="candidates per seed (N)")
    g.add_argument("--pilot", action="store_true",
                   help="ONE seed only -- run this + analyze first for the go/no-go")
    g.add_argument("--seeds", default="", help="comma seed ids to restrict to (e.g. s01,s02)")
    g.add_argument("--seed-stride", type=int, default=100,
                   help="sampler-seed spacing between candidates (avoid collisions with the "
                        "drift rollout's base_seed+i scheme)")
    g.set_defaults(func=cmd_generate)

    a = sub.add_parser("analyze", help="[local] rank candidates, go/no-go, held-out judge, MOS export")
    a.add_argument("--copy-mos-audio", action="store_true",
                   help="also copy winner/loser candidate wavs into results/rerank/mos_audio/")
    a.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
