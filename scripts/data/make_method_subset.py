#!/usr/bin/env python3
"""Define the FIXED regression subset for S method experiments (T7.0).

Every S method variant (prompt wording, readout, gap, precision, scorer) is
verified against this ONE subset before it may touch the full 1890.  The subset
is defined once, saved, and reused forever -- a variant evaluated on a different
subset is not comparable to the others, so this file must not be regenerated
with different parameters once caches exist against it.

Design (T7.0 requires "every perturbation type x a few magnitudes x all 6
genres"):

  * 18 source tracks -- 3 per genre, drawn with a seeded RNG from the 15 per
    genre.  The SAME tracks appear under every tag, so control-vs-perturbed
    comparisons stay PAIRED (d_z is a paired statistic; an unbalanced subset
    would silently break it).
  * 11 tags -- control + the smallest and largest magnitude of each graded
    family (pitch / stretch / lowpass / distortion) + both style swaps.  The
    min/max bracket is what the T7.0 acceptance test reads: style_swap must
    stay the #1 drop and time_stretch must stay ~0 (< 0.03), and those are the
    two ends of the signature.

  => 11 x 18 = 198 pairs (~200 as specified).

Both pitch directions and both stretch directions are represented (-1st/+4st,
+5pct/-20pct) so a variant cannot pass by being sensitive in one direction only.

Usage (DSP env, from repo root):

    python scripts/data/make_method_subset.py \
        --manifest perturbations/pairs_manifest.json \
        --out results/s_cache/s_method_subset.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# tag = 3rd component of pair_id "{source_id}::{perturbation}::{tag}"
SUBSET_TAGS = [
    "ctrl",           # control -- the per-track anchor (also T3.1 normalization)
    "-1st", "+4st",           # pitch_shift: smallest and largest
    "+5pct", "-20pct",        # time_stretch: smallest and largest
    "lp8000", "lp2000",       # lowpass: mildest and harshest
    "drv4", "drv32",          # distortion: mildest and harshest
    "swap_cross", "swap_same",  # style_swap: both partner regimes
]
TRACKS_PER_GENRE = 3
SEED = 20260717


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=Path("perturbations/pairs_manifest.json"))
    ap.add_argument("--out", type=Path, default=Path("results/s_cache/s_method_subset.json"))
    ap.add_argument("--tracks-per-genre", type=int, default=TRACKS_PER_GENRE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing subset file (DON'T: caches are keyed to it)")
    args = ap.parse_args()

    if args.out.is_file() and not args.force:
        print(f"refusing to overwrite existing {args.out} (use --force if you really mean it);\n"
              f"the subset is frozen by design -- every S variant cache is keyed to it.",
              file=sys.stderr)
        return 1

    pairs = json.loads(args.manifest.read_text(encoding="utf-8"))["pairs"]
    by_tag = {p["pair_id"]: p for p in pairs}

    # genre -> sorted unique source_ids, sampled with a seeded RNG
    genres: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        genres[p["genre"]].add(p["source_id"])
    rng = random.Random(args.seed)
    chosen: list[str] = []
    for g in sorted(genres):
        chosen += sorted(rng.sample(sorted(genres[g]), args.tracks_per_genre))

    selected, missing = [], []
    for src in chosen:
        for tag in SUBSET_TAGS:
            hits = [pid for pid in by_tag if pid.startswith(f"{src}::")
                    and pid.rsplit("::", 1)[1] == tag]
            if not hits:
                missing.append(f"{src}::*::{tag}")
                continue
            selected += hits

    if missing:
        print(f"ERROR: {len(missing)} expected pairs absent from the manifest, e.g. "
              f"{missing[:3]} -- refusing to write a subset with holes "
              f"(holes unbalance the paired d_z).", file=sys.stderr)
        return 1

    doc = {
        "meta": {
            "purpose": "T7.0 fixed regression subset for S method variants; FROZEN once written",
            "manifest": str(args.manifest),
            "seed": args.seed,
            "tracks_per_genre": args.tracks_per_genre,
            "tags": SUBSET_TAGS,
            "n_sources": len(chosen),
            "n_pairs": len(selected),
            "acceptance": "style_swap drop stays #1; time_stretch drop stays < ~0.03; "
                          "control level sane (~0.80 on MF)",
        },
        "source_ids": chosen,
        "pair_ids": sorted(selected),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(selected)} pairs over {len(chosen)} sources "
          f"x {len(SUBSET_TAGS)} tags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
