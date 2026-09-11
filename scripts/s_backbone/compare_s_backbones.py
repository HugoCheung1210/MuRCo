#!/usr/bin/env python3
"""Side-by-side S comparison: Music Flamingo vs a second ALLM backbone (Qwen2.5-Omni).

Robustness check for the S dimension (PROJECT_STATE §8.2 / ablation §4): does the
separability *pattern* — style-swap largest, time-stretch ≈ 0 (the clean off-diagonal)
— reproduce on a different audio-LLM, or is it an MF artifact? Absolute scales differ
across backbones (Omni runs low/conservative), so the robustness claim rests on the
**ordering of drops**, not the levels.

Reads the two s_scores caches (same {"scores": {pair_id: val}} format) + the manifest
(pair_id -> perturbation, genre), and prints per-perturbation mean S and drop-vs-control
for each backbone. Works on a PARTIAL Omni cache (reports n per cell), so you can
compare mid-run.

    python scripts/s_backbone/compare_s_backbones.py \
        --manifest perturbations/pairs_manifest.json \
        --mf results/s_cache/s_scores.json --omni results/s_cache/s_scores_omni.json [--by-genre]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

# Match the separability matrix's column order.
ORDER = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]


def _load_scores(path: Path) -> dict:
    d = json.loads(path.read_text())
    return d.get("scores", d)          # accept {"scores":{...}} or a bare {pair:val}


def _mean(xs):
    return st.mean(xs) if xs else float("nan")


def _table(pairs, mf, omni, subset=None):
    def cell(scores, pert):
        return [scores[p["pair_id"]] for p in pairs
                if p["perturbation"] == pert and p["pair_id"] in scores
                and (subset is None or p["pair_id"] in subset)]

    cmf, com = _mean(cell(mf, "control")), _mean(cell(omni, "control"))
    nmf_c, nom_c = len(cell(mf, "control")), len(cell(omni, "control"))
    print(f"{'perturbation':14s}{'MF n':>6s}{'MF mean':>9s}{'MF drop':>9s}"
          f"{'Om n':>6s}{'Om mean':>9s}{'Om drop':>9s}")
    print(f"{'control':14s}{nmf_c:6d}{cmf:9.3f}{'—':>9s}{nom_c:6d}{com:9.3f}{'—':>9s}")
    rows = []
    for pert in ORDER:
        vmf, vom = cell(mf, pert), cell(omni, pert)
        mmf, mom = _mean(vmf), _mean(vom)
        dmf, dom = cmf - mmf, com - mom
        rows.append((pert, dmf, dom))
        print(f"{pert:14s}{len(vmf):6d}{mmf:9.3f}{dmf:+9.3f}"
              f"{len(vom):6d}{mom:9.3f}{dom:+9.3f}")
    # ordering agreement: do both backbones rank the perturbations' drops the same?
    rank_mf = [r[0] for r in sorted(rows, key=lambda r: r[1], reverse=True)]
    rank_om = [r[0] for r in sorted(rows, key=lambda r: r[2], reverse=True)]
    tag = "MATCH" if rank_mf == rank_om else "differ"
    print(f"drop-ranking  MF: {' > '.join(rank_mf)}")
    print(f"drop-ranking  Om: {' > '.join(rank_om)}   [{tag}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--mf", type=Path, default=Path("results/s_cache/s_scores.json"))
    ap.add_argument("--omni", type=Path, default=Path("results/s_cache/s_scores_omni.json"))
    ap.add_argument("--by-genre", action="store_true")
    args = ap.parse_args()

    pairs = json.loads(args.manifest.read_text())["pairs"]
    mf, omni = _load_scores(args.mf), _load_scores(args.omni)
    n_om = sum(1 for p in pairs if p["pair_id"] in omni)
    print(f"MF cache: {len(mf)} scores | Omni cache: {len(omni)} scores "
          f"({n_om}/{len(pairs)} manifest pairs covered)")
    print("(drop = control_mean − perturbed_mean; larger = S detects it. Robustness = "
          "same ORDERING, esp. style_swap largest & time_stretch ≈ 0.)\n")

    print("===== ALL GENRES =====")
    _table(pairs, mf, omni)
    if args.by_genre:
        for g in sorted({p["genre"] for p in pairs}):
            subset = {p["pair_id"] for p in pairs if p["genre"] == g}
            print(f"\n===== {g} =====")
            _table(pairs, mf, omni, subset)


if __name__ == "__main__":
    main()
