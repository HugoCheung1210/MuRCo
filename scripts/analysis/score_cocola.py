#!/usr/bin/env python
"""Score the perturbation-matrix pairs with COCOLA (Ciranni et al., ICASSP 2025).

Why this exists: Chapter 2 argues COCOLA is the nearest prior work but is not an
applicable baseline, because it scores submixes that sound *simultaneously* whereas C
scores a relation between segments. That is an argument, and an argument is cheaper to
attack than a number. This script supplies the number.

COCOLA is NOT part of the validated pipeline and does NOT run in the DSP env: it needs
torch 2.2 + lightning, which anaconda base does not have and must not get. Run it from a
throwaway venv built off cocola/requirements.txt, e.g.

  python3.11 -m venv /tmp/cocola-venv
  /tmp/cocola-venv/bin/pip install -r /path/to/cocola/requirements.txt
  /tmp/cocola-venv/bin/python scripts/analysis/score_cocola.py \
      --cocola-repo /tmp/cocola --ckpt /tmp/cocola_hp_v1.ckpt \
      --out results/baselines/cocola_scores.csv

CPU only; the encoder is EfficientNet-B0. The slow part is librosa HPSS inside COCOLA's
feature extractor, not the network. A's features are cached per source (each A is reused
across all 21 conditions), so this does 90 + N extractions rather than 2N.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

LOG = logging.getLogger("cocola")

COCOLA_SR = 16000
COCOLA_SECONDS = 5.0


def load_centre_window(path: Path, seconds: float, sr: int) -> np.ndarray | None:
    """Mono, resampled, centre `seconds` of the file. Centre avoids edge effects from
    the perturbation renderer (fades, filter warm-up).

    res_type is pinned to soxr_hq: it is ~1000x faster than the fallback resampler here
    and is what dominates runtime, not the network.
    """
    import librosa
    try:
        total = librosa.get_duration(path=str(path))
        offset = max(0.0, (total - seconds) / 2.0)
        y, _ = librosa.load(str(path), sr=sr, mono=True, offset=offset,
                            duration=seconds, res_type="soxr_hq")
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not load %s: %s", path.name, exc)
        return None
    need = int(round(seconds * sr))
    if len(y) < need:  # pad short clips rather than drop them
        return np.pad(y, (0, need - len(y)))
    return y[:need]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "perturbations/pairs_manifest.json")
    ap.add_argument("--audio-root", type=Path, default=ROOT)
    ap.add_argument("--cocola-repo", type=Path, required=True,
                    help="clone of github.com/gladia-research-group/cocola")
    ap.add_argument("--ckpt", type=Path, required=True, help="COCOLA_HP_v1 .ckpt")
    ap.add_argument("--out", type=Path, default=ROOT / "results/baselines/cocola_scores.csv")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N pairs")
    ap.add_argument("--perturbations", default="",
                    help="comma-separated families to keep (default: all)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.path.insert(0, str(args.cocola_repo))

    import torch
    from contrastive_model import constants
    from contrastive_model.contrastive_model import CoCola
    from feature_extraction.feature_extraction import CoColaFeatureExtractor

    LOG.info("loading %s", args.ckpt.name)
    model = CoCola.load_from_checkpoint(str(args.ckpt), map_location="cpu")
    model.eval()
    extractor = CoColaFeatureExtractor()

    recs = json.loads(args.manifest.read_text())["pairs"]
    if args.perturbations:
        keep = {s.strip() for s in args.perturbations.split(",")}
        recs = [r for r in recs if r["perturbation"] in keep]
    if args.limit:
        recs = recs[:args.limit]
    LOG.info("scoring %d pairs on CPU", len(recs))

    MODES = [("cocola", constants.EmbeddingMode.BOTH),
             ("cocola_h", constants.EmbeddingMode.HARMONIC),
             ("cocola_p", constants.EmbeddingMode.PERCUSSIVE)]

    feat_cache: dict[str, torch.Tensor] = {}

    def features(path: Path) -> torch.Tensor | None:
        y = load_centre_window(path, COCOLA_SECONDS, COCOLA_SR)
        if y is None:
            return None
        return extractor(torch.from_numpy(y).float().unsqueeze(0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["pair_id", "source_id", "genre", "perturbation", "magnitude",
            "magnitude_unit", "cocola", "cocola_h", "cocola_p"]
    n_ok = n_bad = 0

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(recs, 1):
            if r["source_id"] not in feat_cache:
                fa = features(args.audio_root / r["ref_path"])
                if fa is None:
                    n_bad += 1
                    continue
                feat_cache[r["source_id"]] = fa
            fa = feat_cache[r["source_id"]]
            fb = features(args.audio_root / r["cand_path"])
            if fb is None:
                n_bad += 1
                continue

            row = {k: r.get(k, "") for k in cols[:6]}
            row["pair_id"] = r["pair_id"]
            with torch.no_grad():
                for name, mode in MODES:
                    model.set_embedding_mode(mode)
                    # score() zeroes a channel in place for H/P, so hand it clones
                    s = model.score(fa.clone(), fb.clone())
                    row[name] = float(s.item())
            w.writerow(row)
            n_ok += 1
            if i % 100 == 0:
                LOG.info("  %d/%d", i, len(recs))
                fh.flush()

    LOG.info("[wrote %s]  scored=%d  skipped=%d", args.out, n_ok, n_bad)


if __name__ == "__main__":
    main()
