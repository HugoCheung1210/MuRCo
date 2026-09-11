#!/usr/bin/env python3
"""Score the Setup-2 perturbation pairs and build the separability matrix.

Reads ``pairs_manifest.json`` from ``make_perturbation_dataset.py``, scores every
(A, B) pair with each requested coherence dimension, and reports -- per dimension
-- how much that dimension responds to each perturbation family.  The target is a
strong diagonal: each dimension falls for its own perturbation, flat for others.

Reporting follows the project guardrails: paired across sources (control vs
perturbation for the same source), led by effect sizes (Cohen's d_z and
matched-pairs rank-biserial) with Wilcoxon p reported but not leaned on, using
absolute drops rather than proportional ratios.

``--by-genre`` additionally breaks every cell down per genre.  This matters most
for R: it is only meaningful where A carries a beat and abstains (score -> 1) on
piano/ambient, so a pooled R row would average a real percussive response
together with abstentions and mislead.  Report R per genre.

Usage::

    python score_separability.py --manifest perturbations/pairs_manifest.json \
        --audio-dir perturbations/audio --dimensions H,T,R --by-genre \
        --output-dir results
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
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

try:
    import librosa
    from scipy.stats import wilcoxon
except ImportError as exc:  # pragma: no cover
    raise SystemExit("score_separability requires librosa and scipy") from exc

from coherence_dimensions import Dimension, StructuralDimension, build_dimensions

TARGET_OF = {"H": {"pitch_shift"}, "R": {"time_stretch"},
             "T": {"lowpass", "distortion"}, "S": {"style_swap"}}


@dataclass
class PairScore:
    pair_id: str
    source_id: str
    genre: str
    dim_target: str
    perturbation: str
    magnitude: float
    scores: dict[str, float]


# --- audio loading ----------------------------------------------------------

def resolve(path_str: str, audio_dir: Path | None) -> Path:
    p = Path(path_str)
    return audio_dir / p.name if audio_dir is not None else p


class AudioCache:
    def __init__(self, sr: int, audio_dir: Path | None, secs: float = 0.0) -> None:
        self.sr = sr
        self.audio_dir = audio_dir
        self.secs = secs          # 0 = whole file (the validated default)
        self._cache: dict[str, np.ndarray] = {}

    def get(self, path_str: str) -> np.ndarray | None:
        if path_str in self._cache:
            return self._cache[path_str]
        path = resolve(path_str, self.audio_dir)
        try:
            y, _ = librosa.load(str(path), sr=self.sr, mono=True,
                                duration=self.secs or None)
        except Exception as exc:  # noqa: BLE001
            logging.error("Failed to load %s: %s", path, exc)
            return None
        if y.size == 0:
            logging.error("Empty audio: %s", path)
            return None
        y = y.astype(np.float32)
        self._cache[path_str] = y
        return y


def score_pairs(pairs: Sequence[dict], dims: Mapping[str, Dimension],
                cache: AudioCache) -> list[PairScore]:
    results: list[PairScore] = []
    total = len(pairs)
    missing_s = 0
    for i, pair in enumerate(pairs, 1):
        scores: dict[str, float] = {}
        needs_audio = any(not isinstance(d, StructuralDimension) for d in dims.values())
        a = cache.get(pair["ref_path"]) if needs_audio else None
        b = cache.get(pair["cand_path"]) if needs_audio else None
        if needs_audio and (a is None or b is None):
            continue
        ok = True
        for name, dim in dims.items():
            if isinstance(dim, StructuralDimension):
                if not dim.has(pair["pair_id"]):
                    missing_s += 1
                    ok = False
                    break
                scores[name] = dim.score_pair(pair["pair_id"])
            else:
                scores[name] = dim.score(a, b, cache.sr)
        if not ok:
            continue
        results.append(PairScore(
            pair_id=pair["pair_id"], source_id=pair["source_id"], genre=pair["genre"],
            dim_target=pair["dim_target"], perturbation=pair["perturbation"],
            magnitude=float(pair.get("magnitude", float("nan"))), scores=scores,
        ))
        if i % 200 == 0 or i == total:
            logging.info("scored %d/%d pairs", i, total)
    if missing_s:
        logging.warning("%d pairs skipped: S requested but pair_id absent from --s-cache "
                        "(run score_s_mf.py first, or it's a partial cache)", missing_s)
    return results


# --- statistics (effect sizes first) ----------------------------------------

def paired_stats(control: np.ndarray, perturbed: np.ndarray) -> dict[str, float]:
    diff = control - perturbed
    n = int(diff.size)
    out: dict[str, float] = {
        "n": n,
        "mean_control": float(np.mean(control)),
        "mean_perturbed": float(np.mean(perturbed)),
        "mean_drop": float(np.mean(diff)),
        "median_drop": float(np.median(diff)),
    }
    sd = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    out["cohen_dz"] = float(np.mean(diff) / sd) if sd > 0 else 0.0
    out["rank_biserial"] = _rank_biserial(diff)
    out["wilcoxon_p"] = _wilcoxon_p(diff)
    return out


def _rank_biserial(diff: np.ndarray) -> float:
    nz = diff[diff != 0]
    if nz.size == 0:
        return 0.0
    ranks = _average_ranks(np.abs(nz))
    r_pos = float(ranks[nz > 0].sum())
    r_neg = float(ranks[nz < 0].sum())
    total = r_pos + r_neg
    return (r_pos - r_neg) / total if total > 0 else 0.0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < sorted_vals.size:
        j = i
        while j + 1 < sorted_vals.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _wilcoxon_p(diff: np.ndarray) -> float:
    nz = diff[diff != 0]
    if nz.size < 1:
        return float("nan")
    try:
        return float(wilcoxon(nz, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return float("nan")


def verdict(stats: dict[str, float], dz_threshold: float = 0.8, drop_eps: float = 0.02) -> str:
    if abs(stats["cohen_dz"]) >= dz_threshold and stats["mean_drop"] >= drop_eps:
        return "RESPONDS"
    if abs(stats["cohen_dz"]) >= dz_threshold and stats["mean_drop"] <= -drop_eps:
        return "INVERTED"
    return "flat"


# --- separability assembly --------------------------------------------------

def _family_cells(scores: Sequence[PairScore], dim_name: str) -> list[dict]:
    """Per-perturbation-family paired cells for one dimension over these scores."""
    control = {s.source_id: s.scores[dim_name] for s in scores if s.perturbation == "control"}
    families: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in scores:
        if s.perturbation == "control":
            continue
        families[s.perturbation][s.source_id].append(s.scores[dim_name])

    cells: list[dict] = []
    for pert, s2list in families.items():
        pooled = {sid: float(np.mean(v)) for sid, v in s2list.items()}
        common = [sid for sid in pooled if sid in control]
        if len(common) < 2:
            continue
        ctrl = np.array([control[sid] for sid in common])
        pert_arr = np.array([pooled[sid] for sid in common])
        st = paired_stats(ctrl, pert_arr)
        st.update({"perturbation": pert, "is_target": pert in TARGET_OF.get(dim_name, set()),
                   "verdict": verdict(st)})
        cells.append(st)
    cells.sort(key=lambda d: (not d["is_target"], -d["cohen_dz"]))
    return cells


def _magnitude_cells(scores: Sequence[PairScore], dim_name: str) -> list[dict]:
    control = {s.source_id: s.scores[dim_name] for s in scores if s.perturbation == "control"}
    groups: dict[tuple[str, float], dict[str, float]] = defaultdict(dict)
    for s in scores:
        if s.perturbation == "control":
            continue
        mag = s.magnitude if not math.isnan(s.magnitude) else float("nan")
        groups[(s.perturbation, mag)][s.source_id] = s.scores[dim_name]
    out = []
    for (pert, mag), s2s in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        common = [sid for sid in s2s if sid in control]
        if len(common) < 2:
            continue
        ctrl = np.array([control[sid] for sid in common])
        pert_arr = np.array([s2s[sid] for sid in common])
        st = paired_stats(ctrl, pert_arr)
        st.update({"perturbation": pert, "magnitude": mag, "verdict": verdict(st)})
        out.append(st)
    return out


def separability(scores: Sequence[PairScore], dim_name: str, by_genre: bool) -> dict:
    row = {"dimension": dim_name,
           "n_control_sources": sum(1 for s in scores if s.perturbation == "control"),
           "per_family": _family_cells(scores, dim_name),
           "per_magnitude": _magnitude_cells(scores, dim_name)}
    if by_genre:
        genres = sorted({s.genre for s in scores})
        row["per_genre"] = {
            g: _family_cells([s for s in scores if s.genre == g], dim_name) for g in genres
        }
    return row


# --- reporting --------------------------------------------------------------

def _print_cells(cells: Sequence[dict], indent: str = "") -> None:
    for c in cells:
        print(f"{indent}{c['perturbation']:14s} {'YES' if c['is_target'] else '-':6s} "
              f"{c['mean_control']:6.3f} {c['mean_perturbed']:6.3f} "
              f"{c['mean_drop']:+7.3f} {c['cohen_dz']:+7.2f} "
              f"{c['rank_biserial']:+6.2f} {c['wilcoxon_p']:9.2e}  {c['verdict']}")


def print_matrix(rows: Sequence[dict]) -> None:
    header = (f"{'perturbation':14s} {'target':6s} {'ctrl':>6s} {'pert':>6s} "
              f"{'drop':>7s} {'d_z':>7s} {'r_rb':>6s} {'wilcox_p':>9s}  verdict")
    for row in rows:
        d = row["dimension"]
        print(f"\n=== Separability row for {d}  (target = {', '.join(sorted(TARGET_OF.get(d, {'?'})))}) ===")
        print(header)
        _print_cells(row["per_family"])
        tgt = [c for c in row["per_family"] if c["is_target"]]
        leak = [c["perturbation"] for c in row["per_family"]
                if not c["is_target"] and c["verdict"] == "RESPONDS"]
        if tgt:
            ok = all(c["verdict"] == "RESPONDS" for c in tgt) and not leak
            print(f"  -> diagonal {'CLEAN' if ok else 'CHECK'}: "
                  f"target responds={all(c['verdict']=='RESPONDS' for c in tgt)}"
                  f"{'; off-diagonal leak: ' + ', '.join(leak) if leak else ''}")
        if "per_genre" in row:
            for g, cells in row["per_genre"].items():
                print(f"\n  -- {d} by genre: {g} --")
                print("  " + header)
                _print_cells(cells, indent="  ")


def write_pair_scores(path: Path, scores: Sequence[PairScore], dim_names: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "source_id", "genre", "dim_target",
                         "perturbation", "magnitude", *[f"score_{d}" for d in dim_names]])
        for s in scores:
            writer.writerow([s.pair_id, s.source_id, s.genre, s.dim_target,
                             s.perturbation, s.magnitude,
                             *[f"{s.scores[d]:.6f}" for d in dim_names]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--audio-dir", type=Path, default=None,
                   help="resolve pair audio by basename here (survives a machine change)")
    p.add_argument("--dimensions", default="H", help="comma subset of H,T,R,S")
    p.add_argument("--s-cache", type=Path, default=None,
                   help="s_scores.json from score_s_mf.py (required when S is in --dimensions)")
    p.add_argument("--by-genre", action="store_true", help="also break every cell down per genre")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    p.add_argument("--sr", type=int, default=0, help="override sample rate (0 = manifest meta)")
    p.add_argument("--secs", type=float, default=0.0,
                   help="cap each clip at N seconds before scoring H/T/R "
                        "(0 = whole file, the validated default). Use --secs 6 "
                        "to match the 13 s human-MOS stimulus.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")
    if not args.manifest.is_file():
        logging.error("Manifest not found: %s", args.manifest)
        return 2
    doc = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs, meta = doc["pairs"], doc.get("meta", {})
    sr = args.sr or int(meta.get("sr", 48000))
    try:
        dims = build_dimensions(args.dimensions, s_cache=str(args.s_cache) if args.s_cache else None)
    except ValueError as exc:
        logging.error("%s", exc)
        return 2
    if "S" in dims and not args.s_cache:
        logging.error("S requested but --s-cache not given; run score_s_mf.py first.")
        return 2
    logging.info("Scoring %d pairs at %d Hz with dimensions: %s", len(pairs), sr, ", ".join(dims))

    cache = AudioCache(sr, args.audio_dir, args.secs)
    scores = score_pairs(pairs, dims, cache)
    if not scores:
        logging.error("No pairs scored (check --audio-dir).")
        return 3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dim_names = list(dims)
    write_pair_scores(args.output_dir / "pair_scores.csv", scores, dim_names)
    rows = [separability(scores, name, args.by_genre) for name in dim_names]
    (args.output_dir / "separability.json").write_text(
        json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8")
    print_matrix(rows)
    logging.info("Wrote %s and %s", args.output_dir / "pair_scores.csv",
                 args.output_dir / "separability.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
