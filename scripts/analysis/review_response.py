#!/usr/bin/env python3
"""Numbers demanded by the EXTERNAL reviewer report of 2026-08-24.

Companion to `review_fixes.py` (the 2026-08-23 internal pass). Everything here comes
from data already on disk -- no GPU, no rescoring -- and lands in one JSON so the paper
and the rebuttal can quote it without a re-run. Point-by-point verdicts and the prose
around these numbers live in `doc/notes/icassp_review_response_2026-08-24.md`.

  A. gap_heldout   -- A2 and the read-out defect. The 1s gap re-checked on the 72
                      sources the sweep did NOT tune on, using two caches from the SAME
                      generation. `results/diagnostics/gap_sweep.json` mixes a pre- and a
                      post-`peak_safe_concat` cache, so its 0.054-vs-0.125 headline
                      straddles a read-out change; this recomputes it honestly.
  B. polarity      -- A3. Per-template drops for the two POSITIVE and two NEGATED probes.
                      An acquiescence bias or a concatenation artefact moves all four the
                      same way; a semantic read moves them oppositely. This is the control
                      the reviewer asks for, and the data was already cached.
  C. temporal      -- A3. The pooled 0.600 temporal AUC split into displacement / reverse
                      / shuffle, which S does not handle alike.
  D. accumulation  -- B1 and B2. Bootstrap CIs on the n=10 drift effect, and S vs each
                      baseline as a paired difference of d_z rather than of raw drops,
                      which are not comparable across differently-scaled metrics.
  E. r_gate        -- B3. Beat strength g over every A window in the battery and in the
                      Part 2 pool, so "what fraction of pairs had R gated to 1" has a
                      number. Needs the DSP env (librosa/soundfile); skipped with a note
                      if the audio is not present.
  F. ab_margins    -- A7. Part 2 win rates split by the RIVAL's own margin. Trials were
                      selected on C's margin, so this is the test of whether that
                      selection carries the result.

Usage (DSP env, from repo root):
    python scripts/analysis/review_response.py
    python scripts/analysis/review_response.py --skip r_gate      # no audio on this box
    python scripts/analysis/review_response.py --write-gap-subset # A2's held-out re-run list

`--write-gap-subset` emits `results/s_cache/heldout_gap_subset.json`, a `pair_ids` list
mirroring the frozen 18-source/11-pair tuning subset on 18 sources it does NOT contain,
for `score_s_mf.py --subset` to rescore gaps 0.5 and 2.0 out of sample.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BOOT = 20000
SEED = 20260824


def repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / ".projectroot").exists() or (p / ".git").exists():
            return p
    return Path(__file__).resolve().parent.parent.parent


ROOT = repo_root()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def s_cache(name: str) -> dict[str, float]:
    """A cache's pair_id -> S. Older files store a float, newer ones a dict."""
    doc = json.loads((ROOT / "results/s_cache" / name).read_text(encoding="utf-8"))
    return {k: (v["score"] if isinstance(v, dict) else float(v))
            for k, v in doc["scores"].items()}


def manifest_rows() -> dict[str, dict]:
    return {r["pair_id"]: r for r in read_csv(ROOT / "results/coherence/pair_scores.csv")}


def boot_ci(x: np.ndarray, stat, rng) -> list[float]:
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    vals = stat(x[idx])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# --------------------------------------------------------------- A. held-out gap
def within_source_drops(cache: dict[str, float], rows: dict[str, dict],
                        sources: set[str]) -> tuple[dict[str, float], float, float]:
    """Mean (control - perturbed) per family, within source. Sources missing a scored
    control contribute nothing, which is why the control map is built first."""
    ctrl = {r["source_id"]: cache[p] for p, r in rows.items()
            if r["perturbation"] == "control" and r["source_id"] in sources and p in cache}
    fam: dict[str, list[float]] = defaultdict(list)
    for p, r in rows.items():
        s = r["source_id"]
        if s in sources and s in ctrl and p in cache and r["perturbation"] != "control":
            fam[r["perturbation"]].append(ctrl[s] - cache[p])
    flat = [v for vals in fam.values() for v in vals]
    return ({k: float(np.mean(v)) for k, v in fam.items()},
            float(np.mean(flat)), float(np.mean(list(ctrl.values()))))


def analysis_gap(rows, cache_a="s_scores_gap0.json", cache_b="s_scores.json"):
    tuned = {p.split("::")[0] for p in s_cache("s_subset_default.json")}
    every = {r["source_id"] for r in rows.values()}
    held = every - tuned
    g0, g1 = s_cache(cache_a), s_cache(cache_b)

    out = {"note": "gap 0 vs gap 1 from the SAME cache generation (default: both "
                   "2026-07-16, pre-peak_safe_concat). results/diagnostics/gap_sweep.json "
                   "mixes generations and its ratio is inflated by the read-out change. "
                   "Pass --gap-caches to compare two caches you have just rescored.",
           "caches": {"gap0": cache_a, "gap1": cache_b},
           "n_tuned_sources": len(tuned), "n_heldout_sources": len(held), "splits": {}}
    for label, srcs in (("tuned_18", tuned), ("heldout_72", held), ("all_90", every)):
        f0, m0, c0 = within_source_drops(g0, rows, srcs)
        f1, m1, c1 = within_source_drops(g1, rows, srcs)
        out["splits"][label] = {
            "gap0": {"mean_drop": m0, "control_level": c0, "by_family": f0},
            "gap1": {"mean_drop": m1, "control_level": c1, "by_family": f1},
            "ratio_gap1_over_gap0": m1 / m0 if m0 else None,
        }
    return out


def write_control_subset(rows) -> Path:
    """The 90 control pair_ids, for a 2-minute read-out smoke test.

    The paper quotes S's control level from the `single` caches (0.802) and its
    calibration probes from a `variants` cache (negated 0.55 vs positive 0.43) inside one
    argument, without saying they are different read-outs. Rescoring these 90 with
    `--scorer balanced --token-agg single` puts both figures on the published read-out.
    Superseded by a full-corpus balanced run, which contains these pairs anyway.
    """
    ids = sorted(p for p, r in rows.items() if r["perturbation"] == "control")
    path = ROOT / "results/s_cache/control_only.json"
    path.write_text(json.dumps(
        {"note": "the 90 control pairs (B is an unprocessed resynthesis of A). Smoke "
                 "test for a read-out change before committing to the full corpus.",
         "n": len(ids), "pair_ids": ids}, indent=1), encoding="utf-8")
    return path


def write_gap_subset(rows) -> Path:
    """18 held-out sources, 3 per genre, 11 pairs each -- the tuning subset's shape."""
    tuned = {p.split("::")[0] for p in s_cache("s_subset_default.json")}
    pattern = sorted(
        (rows[p]["perturbation"], rows[p]["magnitude"])
        for p in s_cache("s_subset_default.json")
        if rows[p]["source_id"] == sorted(tuned)[0])
    by_genre: dict[str, list[str]] = defaultdict(list)
    for s in sorted({r["source_id"] for r in rows.values()} - tuned):
        by_genre[s.rsplit("_", 1)[0]].append(s)
    picked = [s for g in sorted(by_genre) for s in by_genre[g][:3]]

    want = set(pattern)
    ids = [p for p, r in rows.items()
           if r["source_id"] in picked and (r["perturbation"], r["magnitude"]) in want]
    path = ROOT / "results/s_cache/heldout_gap_subset.json"
    path.write_text(json.dumps(
        {"note": "A2 held-out gap re-selection: 18 sources disjoint from the frozen "
                 "tuning subset, same 11-condition pattern per source.",
         "sources": picked, "pair_ids": sorted(ids)}, indent=1), encoding="utf-8")
    return path


def analysis_gap_sweep(rows, pattern="s_heldout_gap*.json"):
    """A2 proper: the WHOLE gap sweep re-selected on held-out sources, one read-out.

    `results/diagnostics/gap_sweep.json` cannot answer this. Its 0 s arm predates
    `peak_safe_concat` and its other three do not, so its headline ratio measures the
    read-out change as much as the gap. This reads whatever `s_heldout_gap<G>.json` arms
    exist, checks they agree on `token_agg` and `peak_safe_concat`, and reports the
    sweep over sources the gap was never tuned on.
    """
    arms = sorted((ROOT / "results/s_cache").glob(pattern),
                  key=lambda q: float(q.stem.rsplit("gap", 1)[1]))
    if len(arms) < 2:
        return {"skipped": f"need >=2 arms matching {pattern}, found {len(arms)}"}

    readouts, out = set(), {}
    for q in arms:
        doc = json.loads(q.read_text())
        m = doc["meta"]
        readouts.add((m.get("token_agg"), m.get("peak_safe_concat")))
        cache = {k: (v["score"] if isinstance(v, dict) else float(v))
                 for k, v in doc["scores"].items()}
        srcs = {rows[k]["source_id"] for k in cache if k in rows}
        fam, mean, ctrl = within_source_drops(cache, rows, srcs)
        out[q.stem.rsplit("gap", 1)[1]] = {
            "mean_drop": mean, "control_level": ctrl, "by_family": fam,
            "n_pairs": len(cache), "n_sources": len(srcs)}

    best = max(out, key=lambda g: out[g]["mean_drop"])
    zero = out.get("0.0", {}).get("mean_drop")
    return {
        "arms": out, "best_gap_s": best,
        "one_readout": len(readouts) == 1,
        "readouts_seen": [{"token_agg": a, "peak_safe_concat": b} for a, b in readouts],
        "ratio_best_over_zero": out[best]["mean_drop"] / zero if zero else None,
        "note": "sources the gap was never tuned on, every arm in one read-out. Compare "
                "against results/diagnostics/gap_sweep.json, which mixes read-outs and "
                "therefore overstates what the gap buys.",
    }


# ------------------------------------------------------------------ B. polarity
def analysis_polarity(rows, cache_name="s_subset_balanced.json"):
    doc = json.loads((ROOT / "results/s_cache" / cache_name).read_text())
    per, order = doc["per_template"], doc["meta"]["per_template_order"]
    ctrl: dict[int, dict[str, float]] = defaultdict(dict)
    for p, r in rows.items():
        if r["perturbation"] == "control" and p in per:
            for i in range(len(order)):
                ctrl[i][r["source_id"]] = per[p][i]
    fam: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for p, r in rows.items():
        if p not in per or r["perturbation"] == "control":
            continue
        for i in range(len(order)):
            if r["source_id"] in ctrl[i]:
                fam[i][r["perturbation"]].append(ctrl[i][r["source_id"]] - per[p][i])
    return {
        "note": "positive probes 0-1, NEGATED probes 2-3. Same sign on all four = "
                "acquiescence or concatenation artefact; opposite signs = a semantic read.",
        "cache": cache_name, "n_pairs": len(per), "n_sources": len(ctrl[0]),
        "probes": [{"index": i, "polarity": "positive" if i < 2 else "negated",
                    "text": order[i],
                    "control_level": float(np.mean(list(ctrl[i].values()))),
                    "drop_by_family": {k: float(np.mean(v)) for k, v in fam[i].items()}}
                   for i in range(len(order))],
    }


# ------------------------------------------------------------------ C. temporal
def analysis_temporal():
    bank = json.loads((ROOT / "results/s_cache/template_bank_temporal.json").read_text())
    an = json.loads((ROOT / "results/temporal/analysis.json").read_text())
    return {"note": "the pooled control-vs-violation AUC averages three conditions S does "
                    "not handle alike: it detects displacement and reversal and fails "
                    "only on within-segment shuffle, which preserves each segment's "
                    "instrumentation, style and track identity.",
            "by_condition": {c: {"S_mean_drop": v["mean"]["mean"], "S_dz": v["mean"]["dz"]}
                             for c, v in bank["drop_table"].items()},
            "pooled_auc": an["control_vs_violation_auc"]}


# -------------------------------------------------------------- D. accumulation
def _iter1_minus_iter8(per_seed) -> tuple[np.ndarray, list[str]]:
    seeds = sorted(per_seed["1"])
    return np.array([per_seed["1"][s] - per_seed["8"][s] for s in seeds]), seeds


def analysis_accumulation():
    rng = np.random.default_rng(SEED)
    htrc = json.loads((ROOT / "results/drift/drift_htrc_sao_logit.json").read_text())
    base = json.loads((ROOT / "results/drift/drift_baselines_sao.json").read_text())
    dz = lambda a: a.mean(-1) / a.std(-1, ddof=1)

    out = {"note": "raw drops are not comparable across differently-scaled metrics; the "
                   "paired d_z difference is. n=10 tracks, SAO inpaint, iter1 vs iter8.",
           "dimensions": {}, "baselines": {}}
    s_d, s_seeds = _iter1_minus_iter8(htrc["S"]["per_seed"]["iters"])
    idx = rng.integers(0, len(s_d), (BOOT, len(s_d)))
    s_dz_boot = dz(s_d[idx])

    for dim in "HTRS":
        d, _ = _iter1_minus_iter8(htrc[dim]["per_seed"]["iters"])
        i = rng.integers(0, len(d), (BOOT, len(d)))
        out["dimensions"][dim] = {
            "n": len(d), "mean_drop": float(d.mean()),
            "drop_ci95": [float(np.percentile(d[i].mean(1), q)) for q in (2.5, 97.5)],
            "dz": float(dz(d)),
            "dz_ci95": [float(np.percentile(dz(d[i]), q)) for q in (2.5, 97.5)]}

    for name, blk in base.items():
        d, seeds = _iter1_minus_iter8(blk["per_seed"]["iters"])
        i = rng.integers(0, len(d), (BOOT, len(d)))
        row = {"n": len(d), "mean_drop": float(d.mean()), "dz": float(dz(d)),
               "dz_ci95": [float(np.percentile(dz(d[i]), q)) for q in (2.5, 97.5)],
               "seeds_match_S": seeds == s_seeds}
        if seeds == s_seeds:                       # paired: reuse ONE resample index
            diff = s_dz_boot - dz(d[idx])
            row["dz_S_minus_this"] = float(dz(s_d) - dz(d))
            row["dz_diff_ci95"] = [float(np.percentile(diff, q)) for q in (2.5, 97.5)]
        out["baselines"][name] = row
    return out


# -------------------------------------------------------------------- E. R gate
def analysis_r_gate():
    import soundfile as sf                                            # noqa: PLC0415
    import sys                                                        # noqa: PLC0415
    sys.path.insert(0, str(ROOT / "scripts/core"))
    from coherence_dimensions import RhythmicDimension                # noqa: PLC0415

    R = RhythmicDimension()

    def beat_strength_of(manifest: Path) -> dict[str, float]:
        pairs = json.loads(manifest.read_text(encoding="utf-8"))["pairs"]
        refs: dict[str, str] = {}
        for p in pairs:
            refs.setdefault(p["source_id"], p["ref_path"])
        g = {}
        for src, rel in sorted(refs.items()):
            a, sr = sf.read(ROOT / rel)
            a = a if a.ndim == 1 else a.mean(1)
            g[src] = float(R._beat_strength(R._onset(a.astype("float32"), sr), sr))
        return g

    out = {"note": "R = 1 - g*(1-raw), so R is gated to exactly 1 only where g == 0. "
                   "g < t means R cannot fall further than t below 1 on that source.",
           "pools": {}}
    for label, man in (("battery_90", ROOT / "perturbations/pairs_manifest.json"),
                       ("part2_ace", ROOT / "results/rerank_ace/rerank_manifest.json")):
        if not man.exists():
            out["pools"][label] = {"skipped": f"{man} not found"}
            continue
        g = beat_strength_of(man)
        v = np.array(list(g.values()))
        by_genre = defaultdict(list)
        for s, val in g.items():
            by_genre[s.rsplit("_", 1)[0]].append(val)
        out["pools"][label] = {
            "n_sources": len(v), "min": float(v.min()),
            "median": float(np.median(v)), "max": float(v.max()),
            "frac_below": {str(t): float((v < t).mean()) for t in (0.01, 0.05, 0.1, 0.2, 0.3)},
            "median_by_genre": {k: float(np.median(x)) for k, x in sorted(by_genre.items())},
        }
    return out


# ------------------------------------------------------------------ G. null bank
def analysis_null(rows, cache_name="s_nullprobe.json",
                  compare_to="s_balanced_full.json"):
    """A3's falsifier. Same read-out, same concatenation, questions with no relational
    content -- see mf_probe.NULL_TEMPLATES for why these four.

    The comparator MUST be a coherence-bank cache from the same read-out generation
    covering the same pairs, or the contrast measures the read-out rather than the
    question. `s_balanced_full.json` is the right default: it is peak-safe + variants
    like the null run, and it spans all 1,890 pairs, so whatever subset the null bank
    was run on is covered. Its positive templates are the coherence bank.
    """
    path = ROOT / "results/s_cache" / cache_name
    if not path.is_file():
        return {"skipped": f"{path.relative_to(ROOT)} not found -- run scorer 'null' first"}
    doc = json.loads(path.read_text())
    per, order = doc["per_template"], doc["meta"]["per_template_order"]

    cmp_path = ROOT / "results/s_cache" / compare_to
    if not cmp_path.is_file():
        return {"skipped": f"comparator {compare_to} not found"}
    cmp_doc = json.loads(cmp_path.read_text())
    cmp_per = cmp_doc["per_template"]
    n_pos = len(cmp_doc["meta"]["per_template_order"]) // 2 \
        if cmp_doc["meta"].get("scorer") == "balanced" else None

    shared = sorted(set(per) & set(cmp_per))
    if not shared:
        return {"skipped": f"{cache_name} and {compare_to} share no pairs"}
    srcs = {rows[p]["source_id"] for p in shared if p in rows}

    def bank(store, sl):
        return {p: float(np.mean(store[p][sl])) for p in shared}

    coh_slice = slice(0, n_pos) if n_pos else slice(None)
    n_fam, n_mean, n_ctrl = within_source_drops(bank(per, slice(None)), rows, srcs)
    c_fam, c_mean, c_ctrl = within_source_drops(bank(cmp_per, coh_slice), rows, srcs)

    per_probe = {}
    for i, q in enumerate(order):
        f, _, lvl = within_source_drops({p: per[p][i] for p in shared}, rows, srcs)
        per_probe[q] = {"control_level": lvl, "drop_by_family": f,
                        "mean_drop": float(np.mean(list(f.values())))}

    return {
        "cache": cache_name, "comparator": compare_to,
        "n_pairs": len(shared), "n_sources": len(srcs),
        "note": "both banks on the identical pairs and read-out, so the only difference "
                "is what is asked. A flat null bank means the coherence bank's "
                "separation is attributable to its questions.",
        "null_bank": {"mean_drop": n_mean, "control_level": n_ctrl, "by_family": n_fam,
                      "max_abs_family_drop": max(abs(v) for v in n_fam.values())},
        "coherence_bank": {"mean_drop": c_mean, "control_level": c_ctrl,
                           "by_family": c_fam},
        "ratio_coherence_over_null": abs(c_mean / n_mean) if n_mean else None,
        "per_probe": per_probe,
    }


# ----------------------------------------------------------------- F. AB margins
def analysis_ab_margins():
    key = {r["trial_id"]: r for r in read_csv(ROOT / "results/mos/session2/key.csv")}
    trials = json.loads((ROOT / "results/rerank_ace/ab_trials.json").read_text())["trials"]
    by_pick = {(t["seed"], t["c_pick"], t["rival_pick"]): t for t in trials}
    for r in key.values():
        t = by_pick.get((r["seed"], r["c_pick"], r["rival_pick"]))
        r["rival_margins"] = t["rival_margins"] if t else {}

    decided = []
    for f in sorted(glob.glob(str(ROOT / "results/mos/session2/responses/*.json"))):
        for x in json.loads(Path(f).read_text())["responses"]:
            p, tid = x.get("preference"), x["trial_id"]
            if p is None or int(p) == 3 or tid not in key:      # 3 = "about the same"
                continue
            k = key[tid]
            favours_a = int(p) < 3
            decided.append({"seed": k["seed"], "c_margin": float(k["c_margin"]),
                            "settles": k["settles"].split(";"),
                            "rival_margins": k["rival_margins"],
                            "win": favours_a if k["opt_a_role"] == "c" else not favours_a})

    rng = np.random.default_rng(SEED)

    def cluster_ci(rowset):
        """Bootstrap over SEEDS -- decisions inside a seed are not independent."""
        bys = defaultdict(list)
        for r in rowset:
            bys[r["seed"]].append(float(r["win"]))
        keys = list(bys)
        draws = [float(np.mean([v for i in rng.integers(0, len(keys), len(keys))
                                for v in bys[keys[i]]])) for _ in range(2000)]
        return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    out = {"note": "trials were selected on C's OWN margin (select_ab_trials.py filters "
                   "abs(C_i - C_j)); the rival's margin was recorded but never "
                   "thresholded. This splits each arm on the rival's own margin.",
           "n_decided_total": len(decided), "c_margin_tertiles": [], "arms": {}}

    # Half-open bins. Closed bins on both ends double-count every decision sitting
    # exactly on a cut point, which inflated the tercile n's past n_decided_total.
    cm = np.array([r["c_margin"] for r in decided])
    cuts = np.quantile(cm, [1 / 3, 2 / 3])
    which = np.digitize(cm, cuts)
    edges = [cm.min(), cuts[0], cuts[1], cm.max()]
    for i in range(3):
        sel = [r for r, b in zip(decided, which) if b == i]
        out["c_margin_tertiles"].append(
            {"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": len(sel),
             "win_rate": float(np.mean([r["win"] for r in sel]))})

    for arm in ("C_noS", "clap_htsat", "clap_music", "scs", "cocola"):
        sel = [r for r in decided if arm in r["settles"] and arm in r["rival_margins"]]
        if not sel:
            continue
        rm = np.array([r["rival_margins"][arm] for r in sel])
        wins = np.array([r["win"] for r in sel], float)
        med = float(np.median(rm))
        row = {"n": len(sel), "win_rate": float(wins.mean()), "ci95": cluster_ci(sel),
               "rival_margin_median": med, "rival_margin_min": float(rm.min()),
               "corr_win_c_margin": float(np.corrcoef(
                   wins, [r["c_margin"] for r in sel])[0, 1]),
               "corr_win_rival_margin": float(np.corrcoef(wins, rm)[0, 1])}
        for lab, mask in (("rival_below_median", rm < med), ("rival_above_median", rm >= med)):
            part = [s for s, m in zip(sel, mask) if m]
            row[lab] = {"n": len(part),
                        "win_rate": float(np.mean([r["win"] for r in part])),
                        "ci95": cluster_ci(part)} if len(part) > 3 else {"n": len(part)}
        out["arms"][arm] = row
    return out


# ------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results/diagnostics/review_response.json")
    ap.add_argument("--skip", nargs="*", default=[], metavar="NAME",
                    help="analyses to skip, e.g. r_gate when the audio is not on this box")
    ap.add_argument("--write-gap-subset", action="store_true",
                    help="also emit results/s_cache/heldout_gap_subset.json for A2's re-run")
    ap.add_argument("--write-control-subset", action="store_true",
                    help="also emit results/s_cache/control_only.json, the 90 control "
                         "pair_ids, as a read-out smoke test")
    ap.add_argument("--polarity-cache", default="s_subset_balanced.json", metavar="NAME",
                    help="filename in results/s_cache/ for the polarity control; point "
                         "this at the full-corpus cache once scorer 'balanced' has run")
    ap.add_argument("--gap-caches", nargs=2, default=["s_scores_gap0.json", "s_scores.json"],
                    metavar=("GAP0", "GAP1"),
                    help="two caches from the SAME read-out generation to compare; the "
                         "defaults are the only same-generation pair currently on disk")
    ap.add_argument("--null-cache", default="s_nullprobe.json", metavar="NAME",
                    help="filename in results/s_cache/ for the null-probe bank (A3)")
    ap.add_argument("--null-comparator", default="s_balanced_full.json", metavar="NAME",
                    help="coherence-bank cache from the SAME read-out generation to "
                         "contrast the null bank against")
    args = ap.parse_args()

    rows = manifest_rows()
    jobs = {"gap_heldout": lambda: analysis_gap(rows, *args.gap_caches),
            "gap_sweep_heldout": lambda: analysis_gap_sweep(rows),
            "polarity": lambda: analysis_polarity(rows, args.polarity_cache),
            "null_bank": lambda: analysis_null(rows, args.null_cache,
                                               args.null_comparator),
            "temporal": analysis_temporal,
            "accumulation": analysis_accumulation,
            "r_gate": analysis_r_gate,
            "ab_margins": analysis_ab_margins}

    report = {"meta": {"source": "external reviewer report 2026-08-24",
                       "prose": "doc/notes/icassp_review_response_2026-08-24.md",
                       "bootstrap_draws": BOOT, "seed": SEED}}
    for name, fn in jobs.items():
        if name in args.skip:
            report[name] = {"skipped": "requested"}
            continue
        try:
            report[name] = fn()
        except Exception as exc:                                   # noqa: BLE001
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  !! {name}: {type(exc).__name__}: {exc}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    # --- the numbers the rebuttal actually quotes
    g = report.get("gap_heldout", {}).get("splits", {})
    if g:
        print("A2  gap1/gap0 mean-drop ratio, same read-out generation:")
        for k, v in g.items():
            r = v["ratio_gap1_over_gap0"]
            if r is None or r != r:            # split not covered by these caches
                continue
            print(f"      {k:12s} {r:.2f}x  "
                  f"(gap0 {v['gap0']['mean_drop']:.4f} -> gap1 {v['gap1']['mean_drop']:.4f})")
    sw = report.get("gap_sweep_heldout", {})
    if "arms" in sw:
        print(f"A2  held-out gap sweep, one read-out ({'OK' if sw['one_readout'] else 'MIXED -- INVALID'}):")
        for gv, a in sw["arms"].items():
            mark = "  <- best" if gv == sw["best_gap_s"] else ""
            print(f"      gap {gv:>3s}s  mean drop {a['mean_drop']:+.4f}  "
                  f"control {a['control_level']:.3f}  n={a['n_pairs']}{mark}")
        if sw["ratio_best_over_zero"]:
            print(f"      best/no-gap = {sw['ratio_best_over_zero']:.2f}x "
                  f"(gap_sweep.json's mixed-read-out figure: 2.29x)")
    for p in report.get("polarity", {}).get("probes", []):
        print(f"A3  {p['polarity'][:3].upper()} probe {p['index']}: "
              f"style_swap drop {p['drop_by_family'].get('style_swap', float('nan')):+.4f}, "
              f"control {p['control_level']:.4f}")
    nb = report.get("null_bank", {})
    if "null_bank" in nb:
        print(f"A3  null bank mean drop {nb['null_bank']['mean_drop']:+.4f} "
              f"(max any family {nb['null_bank']['max_abs_family_drop']:.4f}) vs "
              f"coherence {nb['coherence_bank']['mean_drop']:+.4f} "
              f"-- {nb['ratio_coherence_over_null']:.0f}x, n={nb['n_pairs']}")
    for c, v in report.get("temporal", {}).get("by_condition", {}).items():
        print(f"A3  temporal {c:13s} S drop {v['S_mean_drop']:+.4f}  d_z {v['S_dz']:+.2f}")
    for d, v in report.get("accumulation", {}).get("dimensions", {}).items():
        print(f"B2  {d} drop {v['mean_drop']:+.4f} {v['drop_ci95']}  "
              f"d_z {v['dz']:.2f} [{v['dz_ci95'][0]:.2f}, {v['dz_ci95'][1]:.2f}]")
    for n, v in report.get("accumulation", {}).get("baselines", {}).items():
        if "dz_diff_ci95" in v:
            print(f"B1  d_z(S) - d_z({n}) = {v['dz_S_minus_this']:+.2f} "
                  f"[{v['dz_diff_ci95'][0]:+.2f}, {v['dz_diff_ci95'][1]:+.2f}]")
    for pool, v in report.get("r_gate", {}).get("pools", {}).items():
        if "min" in v:
            print(f"B3  {pool}: min g {v['min']:.3f}, median {v['median']:.3f}, "
                  f"frac g<0.05 = {v['frac_below']['0.05']:.3f}")
    for arm, v in report.get("ab_margins", {}).get("arms", {}).items():
        lo, hi = v.get("rival_above_median", {}).get("ci95", [float("nan")] * 2)
        print(f"A7  {arm:11s} overall {v['win_rate']:.3f}  "
              f"rival-confident half {v.get('rival_above_median', {}).get('win_rate', float('nan')):.3f} "
              f"[{lo:.3f}, {hi:.3f}]")

    if args.write_gap_subset:
        print(f"\nwrote {write_gap_subset(rows).relative_to(ROOT)}")
    if args.write_control_subset:
        print(f"wrote {write_control_subset(rows).relative_to(ROOT)}")
    print(f"\nwrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
