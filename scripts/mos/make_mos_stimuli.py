#!/usr/bin/env python
"""Stratified stimulus sampler for the human-MOS study (E-E). DSP env, read-only.

Scope: perturbation matrix + AI-rerank block. Both are covered by ethics: the
parent application covers the matrix, and the AI-rerank materials amendment was
APPROVED 2026-07-28 (PROJECT_STATE §9.1). Still excludes same-genre style swaps
(locked decision — the whole metric fails there; §5/§10). Cross-genre swaps and
controls are included.

Design (fixed by N=35 participants, ~65 items each):
  - ANCHOR (15, rated by every participant) + N_BLOCKS matrix blocks of
    BLOCK_SIZE + N_BLOCKS AI blocks of AI_BLOCK_SIZE. Each participant rates
    anchor + matrix block i + AI block i, so ONE block index per participant.
  - 35 participants / 5 blocks -> 7 ratings per block pair, 35 per anchor pair.
  - Matrix pairs stratified by condition cell (family x magnitude; 20 cells incl.
    control), balanced across genres, tracks spread across blocks.
  - Anchor spans the full coherence range (per-family severity spread), so
    every rater sees the whole scale -> anchors the 1-5 mapping + enables ICC.
  - AI block = E-D best-of-N rerank pairs (results/rerank{,_ace}/mos_stimuli.json).
    The C-winner and C-loser of a seed ALWAYS land in the same block, so every
    rater who sees a seed sees both arms -> within-rater rerank contrast.

This script only PLANS the stimulus set (no participant data, no audio copied).
Run it now; regenerate freely until data collection starts, then freeze.

Usage:
  python scripts/mos/make_mos_stimuli.py                # default design, seed 20260718
  python scripts/mos/make_mos_stimuli.py --n-blocks 5 --block-size 38 --anchor-size 15
  python scripts/mos/make_mos_stimuli.py --no-ai        # matrix only (parent application)
Outputs: results/mos/stimuli_plan.json + stimuli_plan.csv
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
DIMS = ["H", "T", "R", "S"]


def condition_of(p):
    """Condition cell id: family plus magnitude (control and swap_cross are single cells)."""
    if p["perturbation"] == "control":
        return "control"
    if p["perturbation"] == "style_swap":
        return f"style_swap:{p['magnitude_unit']}"
    return f"{p['perturbation']}:{p['magnitude']:g}"


def load_pool(exclude_sources=()):
    """Eligible pairs. `exclude_sources` drops tracks the audition flagged as carrying
    vocals or distressing content -- the ethics application claims instrumental only and
    the source selector never checked (see scripts/data/review_sources.py). A style_swap pair
    is dropped if EITHER side involves an excluded track, since the partner is audible
    too."""
    man = json.load(open(ROOT / "perturbations/pairs_manifest.json"))["pairs"]
    ex = set(exclude_sources)
    pool = [p for p in man
            if not (p["perturbation"] == "style_swap"
                    and p["magnitude_unit"] == "same_genre")
            and p["source_id"] not in ex
            and (p.get("partner_id") or "") not in ex]
    scores = {}
    with open(ROOT / "results/coherence/pair_scores.csv") as fh:
        for r in csv.DictReader(fh):
            s = [float(r[f"score_{d}"]) for d in DIMS]
            scores[r["pair_id"]] = {
                **{f"score_{d}": v for d, v in zip(DIMS, s)},
                "C_uniform": float(np.exp(np.mean(np.log(np.clip(s, 1e-6, 1.0))))),
            }
    return pool, scores


# ACE only by default. SAO's training-free inpainting is cross-distribution on these
# ACE-generated seeds and goes to mush on rhythmic/groove ones (PROJECT_STATE §602) —
# audibly degraded stimuli would have raters scoring AUDIO QUALITY instead of
# relational coherence, which is a different construct. Excluded on validity grounds.
AI_SOURCES = {"sao": ("rerank_sao", "results/rerank"),
              "ace": ("rerank_ace", "results/rerank_ace")}


def load_ai_pool(backends):
    """E-D rerank stimuli: per seed, the C-winner and C-loser candidate.

    pair_ids collide across backends (both use 'sNN::candKK'), so everything is
    namespaced with the backend tag. Dimension scores come from that run's own
    pair_scores.csv, C_uniform from mos_stimuli.json (already the geometric C).
    """
    pool = []
    for be in backends:
        tag, rel = AI_SOURCES[be]
        stim_path = ROOT / rel / "mos_stimuli.json"
        scores_path = ROOT / rel / "pair_scores.csv"
        if not stim_path.exists():
            continue
        dim_scores = {}
        if scores_path.exists():
            with open(scores_path) as fh:
                for r in csv.DictReader(fh):
                    dim_scores[r["pair_id"]] = {
                        f"score_{d}": float(r[f"score_{d}"]) for d in DIMS
                    }
        for entry in json.load(open(stim_path))["stimuli"]:
            seed = entry["seed"]
            for role in ("c_winner", "c_loser"):
                s = entry[role]
                pool.append({
                    "pair_id": f"{tag}/{s['pair_id']}",
                    "source_id": f"{tag}/{seed}",
                    "seed_group": f"{tag}/{seed}",
                    "genre": tag,
                    "perturbation": f"ai_rerank_{role.replace('c_', '')}",
                    "magnitude": 0.0,
                    "magnitude_unit": "rerank_role",
                    "ref_path": s["ref_path"],
                    "cand_path": s["cand_path"],
                    "scores": {**dim_scores.get(s["pair_id"], {}),
                               "C_uniform": s["C"]},
                })
    return pool


def assign_ai_blocks(ai_pool, n_blocks):
    """Round-robin whole SEEDS (winner+loser together) across the n_blocks."""
    by_seed = defaultdict(list)
    for p in ai_pool:
        by_seed[p["seed_group"]].append(p)
    blocks = [[] for _ in range(n_blocks)]
    for i, seed in enumerate(sorted(by_seed)):
        blocks[i % n_blocks].extend(sorted(by_seed[seed], key=lambda p: p["pair_id"]))
    return blocks


def stratified_sample(pool, n_total, rng):
    """Sample n_total pairs spread evenly over condition cells, then genres,
    then tracks (fewest-used-first so no track dominates)."""
    by_cond = defaultdict(list)
    for p in pool:
        by_cond[condition_of(p)].append(p)
    conds = sorted(by_cond)
    track_use = defaultdict(int)
    chosen = []
    quota, extra = divmod(n_total, len(conds))
    per_cond = {c: quota + (1 if i < extra else 0) for i, c in enumerate(conds)}
    for cond in conds:
        cands = by_cond[cond]
        by_genre = defaultdict(list)
        for p in cands:
            by_genre[p["genre"]].append(p)
        genres = sorted(by_genre)
        for g in genres:
            rng.shuffle(by_genre[g])
        picked, gi = [], 0
        while len(picked) < per_cond[cond]:
            g = genres[gi % len(genres)]
            gi += 1
            avail = [p for p in by_genre[g] if p not in picked]
            if not avail:
                if all(all(p in picked for p in by_genre[g2]) for g2 in genres):
                    break
                continue
            avail.sort(key=lambda p: track_use[p["source_id"]])
            p = avail[0]
            picked.append(p)
            track_use[p["source_id"]] += 1
        chosen.extend(picked)
    return chosen


def pick_anchor(chosen, scores, anchor_size, rng):
    """Anchor = per-family severity spread covering the full coherence range."""
    fams = defaultdict(list)
    for p in chosen:
        fams[p["perturbation"]].append(p)
    n_fam = len(fams)
    quota, extra = divmod(anchor_size, n_fam)
    anchor = []
    for i, fam in enumerate(sorted(fams)):
        k = quota + (1 if i < extra else 0)
        ps = sorted(fams[fam],
                    key=lambda p: scores.get(p["pair_id"], {}).get("C_uniform", 0.5))
        # evenly spaced along the C range within the family -> spans the scale
        idx = np.linspace(0, len(ps) - 1, num=min(k, len(ps))).round().astype(int)
        anchor.extend(ps[j] for j in sorted(set(idx)))
    return anchor[:anchor_size]


def assign_blocks(rest, n_blocks, block_size, rng):
    """Round-robin within each condition cell -> blocks get matched composition;
    shuffled order inside cells spreads tracks across blocks."""
    by_cond = defaultdict(list)
    for p in rest:
        by_cond[condition_of(p)].append(p)
    blocks = [[] for _ in range(n_blocks)]
    b = 0
    for cond in sorted(by_cond):
        ps = by_cond[cond]
        rng.shuffle(ps)
        for p in ps:
            order = sorted(range(n_blocks), key=lambda i: (len(blocks[i]), (i - b) % n_blocks))
            blocks[order[0]].append(p)
            b += 1
    blocks = [bl[:block_size] for bl in blocks]
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-blocks", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=42, help="matrix pairs per block")
    ap.add_argument("--anchor-size", type=int, default=15)
    ap.add_argument("--ai-block-size", type=int, default=8,
                    help="AI-rerank pairs per block (kept in winner/loser couples)")
    ap.add_argument("--ai-backends", default="ace",
                    help="comma list of {ace,sao}; SAO excluded by default on quality "
                         "grounds (see AI_SOURCES)")
    ap.add_argument("--no-ai", action="store_true",
                    help="matrix only — the pre-amendment scope")
    ap.add_argument("--n-participants", type=int, default=35)
    ap.add_argument("--exclude-sources", default=None,
                    help="excluded_sources.json from review_sources.py --apply")
    ap.add_argument("--seed", type=int, default=20260718)
    ap.add_argument("--out-dir", default="results/mos")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rng_shuffle = np.random.RandomState(args.seed)  # list shuffles

    n_total = args.anchor_size + args.n_blocks * args.block_size
    excluded = []
    if args.exclude_sources:
        excluded = json.load(open(ROOT / args.exclude_sources))["excluded"]
    pool, scores = load_pool(excluded)
    chosen = stratified_sample(pool, n_total, rng_shuffle)
    anchor = pick_anchor(chosen, scores, args.anchor_size, rng_shuffle)
    anchor_ids = {p["pair_id"] for p in anchor}
    rest = [p for p in chosen if p["pair_id"] not in anchor_ids]
    blocks = assign_blocks(rest, args.n_blocks, args.block_size, rng_shuffle)

    ratings_per_block_pair = args.n_participants // args.n_blocks
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def row(p, block, stimulus_set="matrix"):
        sc = p["scores"] if stimulus_set == "ai_rerank" else scores.get(p["pair_id"], {})
        return {
            "pair_id": p["pair_id"], "block": block, "stimulus_set": stimulus_set,
            "source_id": p["source_id"],
            "genre": p["genre"], "perturbation": p["perturbation"],
            "condition": condition_of(p), "magnitude": p["magnitude"],
            "ref_path": p["ref_path"], "cand_path": p["cand_path"],
            **{k: round(v, 6) for k, v in sc.items()},
        }

    rows = [row(p, "anchor") for p in anchor]
    for i, bl in enumerate(blocks, 1):
        rows.extend(row(p, f"block{i}") for p in bl)

    backends = [b.strip() for b in args.ai_backends.split(",") if b.strip()]
    ai_pool = [] if args.no_ai else load_ai_pool(backends)
    ai_blocks = assign_ai_blocks(ai_pool, args.n_blocks) if ai_pool else []
    for i, bl in enumerate(ai_blocks, 1):
        rows.extend(row(p, f"block{i}", "ai_rerank") for p in bl[:args.ai_block_size])

    # composition + range report
    comp = defaultdict(lambda: defaultdict(int))
    for r in rows:
        comp[r["block"]][r["perturbation"]] += 1
    c_vals = [r["C_uniform"] for r in rows if "C_uniform" in r]
    a_vals = [r["C_uniform"] for r in rows if r["block"] == "anchor" and "C_uniform" in r]
    n_ai = sum(1 for r in rows if r["stimulus_set"] == "ai_rerank")
    ai_per_block = args.ai_block_size if n_ai else 0
    meta = {
        "ai_backends": backends if n_ai else [],
        "scope": ("perturbation matrix (parent application) + AI-rerank block "
                  f"(materials amendment APPROVED 2026-07-28; backends={backends}, "
                  "SAO excluded on quality/construct-validity grounds); same-genre "
                  "style swaps EXCLUDED (PROJECT_STATE §5/§10)")
                 if n_ai else
                 "perturbation-matrix only (parent application); "
                 "same-genre style swaps EXCLUDED (PROJECT_STATE §5/§10)",
        "seed": args.seed,
        "excluded_sources": excluded,
        "n_pairs": len(rows),
        "n_matrix_pairs": len(rows) - n_ai,
        "n_ai_pairs": n_ai,
        "n_anchor": len(anchor),
        "n_blocks": args.n_blocks,
        "block_size": args.block_size,
        "ai_block_size": ai_per_block,
        "n_participants": args.n_participants,
        "participants_per_block": ratings_per_block_pair,
        "ratings_per_anchor_pair": args.n_participants,
        "ratings_per_block_pair": ratings_per_block_pair,
        "items_per_participant": args.anchor_size + args.block_size + ai_per_block,
        "composition": {b: dict(v) for b, v in sorted(comp.items())},
        "C_uniform_range_all": [round(float(min(c_vals)), 3), round(float(max(c_vals)), 3)],
        "C_uniform_range_anchor": [round(float(min(a_vals)), 3), round(float(max(a_vals)), 3)],
        "presentation": "A | 1s silence | B, single audio file per trial; "
                        "1-5 coherence rating (data_collection_tool.md)",
    }
    json.dump({"meta": meta, "stimuli": rows}, open(out / "stimuli_plan.json", "w"),
              indent=1)
    with open(out / "stimuli_plan.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(meta, indent=1))
    print(f"wrote {out/'stimuli_plan.json'} + .csv ({len(rows)} pairs)")


if __name__ == "__main__":
    main()
