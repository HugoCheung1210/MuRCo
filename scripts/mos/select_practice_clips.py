#!/usr/bin/env python
"""Shortlist and fetch candidate source tracks for the unscored practice trials.

The practice block needs two clips that are NOT study stimuli, and there is no
spare material: all 63 sources that survived the §E listening audit are consumed
by the 190 rendered stimuli, and the 12 unused sources are exactly the ones the
audit threw out for lyrics or distressing content.

So new sources are required. This applies the same filters as
``select_fma_sources.py`` (FMA Medium, >= 45 s, no NoDerivatives licence, known
0-byte clips skipped) with two additions:

  * every track_id already in sources_selection/selected_manifest.json is
    excluded, so a practice clip can never collide with a study stimulus;
  * candidates are drawn from the genre families that actually survived the
    audit. Measured survival was ambient 100%, classical 87%, jazz 87%,
    electronic 73%, rock 40%, folk 33% -- drawing from folk/rock would mostly
    produce tracks with vocals, which is what the audit rejected.

Fetching reuses the byte-range reader in download_selected_fma.py, so it reads a
few MB out of the 23.8 GB remote archive rather than downloading it. (The
fma_medium.zip sitting in the repo is a truncated 132 MB fragment and cannot be
opened -- do not try to extract from it.)

Output goes to a staging directory, NOT raw_audio/, because these are unaudited.
Nothing here is participant-ready until a human has listened to it: the audit
rejected 30% of the original 90, and that check cannot be automated.

Usage:
  python scripts/mos/select_practice_clips.py --list-only     # shortlist, no download
  python scripts/mos/select_practice_clips.py                 # shortlist + fetch
"""

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
import random
import sys
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

import select_fma_sources as selector          # noqa: E402
import download_selected_fma as fetcher        # noqa: E402

# survival rate in the §E audit, best first
PREFERRED = {"ambient": ("Ambient",), "classical": ("Classical",)}

# Genre tags do not distinguish orchestral from choral, so a "Classical" pick is
# as likely to be a motet as a string quartet. These substrings knock out the
# obvious vocal repertoire before a human wastes time listening to it. This is a
# pre-filter for the audit, NOT a substitute: it only catches what the metadata
# happens to say, and plenty of vocal tracks are named nothing in particular.
VOCAL_HINTS = (
    "choir", "chorale", "choral", "chorus", "consort", "chanteur", "cantata",
    "madrigal", "ave maria", "motet", "requiem", "mass in", "hymn", "psalm",
    "aria", "opera", "vocal", "voices", "singer", "a cappella", "acappella",
    "gospel", "lyric", "song for", "ballad", "serenade for voice", "lied",
    "feat.", "ft.", "featuring",
)


def looks_vocal(cand):
    hay = f"{cand.title} {cand.artist}".casefold()
    return [h for h in VOCAL_HINTS if h in hay]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-dir", type=Path, default=ROOT / "fma_metadata")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "sources_selection/selected_manifest.json")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "sources_selection/practice_candidates")
    ap.add_argument("--archive-url", default=fetcher.ARCHIVE_URL)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--min-duration", type=float, default=45.0)
    ap.add_argument("--subset", default="medium")
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    used = {e["track_id"] for e in json.load(open(args.manifest))}
    logging.info("excluding %d track_ids already used as study sources", len(used))

    titles, children = selector.read_genres(args.metadata_dir / "genres.csv")
    targets = selector.build_target_genres(titles, children, PREFERRED)

    pools = {label: [] for label in PREFERRED}
    dropped_vocal = 0
    for cand in selector.stream_candidates(
            args.metadata_dir / "tracks.csv", args.subset, args.min_duration):
        if cand.track_id in used:
            continue
        labels = selector.matching_labels(cand, targets)
        if not labels:
            continue
        if looks_vocal(cand):
            dropped_vocal += 1
            continue
        for label in labels:
            pools[label].append(cand)
    logging.info("dropped %d candidates whose title/artist looks vocal", dropped_vocal)

    rng = random.Random(args.seed)
    picks = []
    # weight toward the family with the cleaner audit record
    quota = {"ambient": (args.n + 1) // 2 + 1, "classical": args.n // 2 - 1}
    for label, k in quota.items():
        pool = pools[label]
        logging.info("%-10s %d eligible candidates", label, len(pool))
        picks += [(label, c) for c in rng.sample(pool, min(k, len(pool)))]
    picks = picks[:args.n]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (label, c) in enumerate(picks, 1):
        rows.append({"candidate": f"practice_cand_{i:02d}", "genre": label,
                     "track_id": c.track_id, "title": c.title, "artist": c.artist,
                     "duration_s": c.duration, "license": c.license,
                     "archive_entry": fetcher.archive_name(c.track_id)})

    print(f"\n{'candidate':18s} {'genre':10s} {'id':>7s} {'dur':>6s}  title / artist")
    for r in rows:
        print(f"{r['candidate']:18s} {r['genre']:10s} {r['track_id']:7d} "
              f"{r['duration_s']:6.0f}  {r['title'][:44]} — {r['artist'][:24]}")
        print(f"{'':18s} {'':10s} {'':7s} {'':6s}  licence: {r['license']}")

    index = args.out_dir / "candidates.csv"
    with index.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {index.relative_to(ROOT)}")

    if args.list_only:
        print("--list-only: no audio fetched")
        return

    print(f"\nopening remote archive (byte-range) ...")
    remote = fetcher.RemoteZip(args.archive_url, args.timeout)
    for r in rows:
        dest = args.out_dir / f"{r['candidate']}.mp3"
        if dest.exists():
            print(f"  {dest.name}: already present, skipping")
            continue
        data = remote.read(r["archive_entry"])
        dest.write_bytes(data)
        print(f"  {dest.name}: {len(data)/1024:.0f} KB  (track {r['track_id']})")

    print(f"\n{len(rows)} candidates in {args.out_dir.relative_to(ROOT)}")
    print("NEXT: listen to all of them. Keep two that are genuinely instrumental,")
    print("no lyrics, nothing distressing. Nothing is participant-ready until then.")


if __name__ == "__main__":
    main()
