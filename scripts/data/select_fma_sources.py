#!/usr/bin/env python3
"""Select a balanced, derivative-permitted FMA Small source set.

The script streams FMA's large, two-header-row ``tracks.csv`` and writes a
reproducible selection of tracks from six broad genre families.  A track is
assigned to at most one family, including when its ``genres_all`` hierarchy
matches several families.

Typical use (from the project root)::

    python scripts/data/select_fma_sources.py

The expected input layout is::

    fma_metadata/genres.csv
    fma_metadata/tracks.csv
    fma_small/000/000002.mp3

Outputs are written to ``sources_selection/`` and audio files to
``raw_audio/``.  Use ``--dry-run`` to inspect metadata availability without
copying anything.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


DEFAULT_TARGETS: dict[str, tuple[str, ...]] = {
    "classical": ("Classical",),
    "ambient": ("Ambient",),
    "electronic": ("Electronic",),
    "folk": ("Folk",),
    "jazz": ("Jazz",),
    "rock": ("Rock",),
}


# fma_medium/small clips with 0s or <30s of real audio despite full-length
# metadata durations, so ``--min-duration`` (which reads the CSV) cannot catch
# them.  This is the complete set for fma_medium.  Source: mdeff/fma wiki
# (issues #8 and #41).
FAULTY_FMA_IDS: frozenset[int] = frozenset({
    1486, 5574, 65753, 80391, 98558, 98559, 98560, 98565, 98566, 98567,
    98568, 98569, 98571, 99134, 105247, 108924, 108925, 126981, 127336,
    133297, 143992,
})


@dataclass(frozen=True)
class Candidate:
    """The small, serialisable subset of FMA metadata needed downstream."""

    track_id: int
    duration: float
    title: str
    artist: str
    license: str
    genre_ids: tuple[int, ...]
    genre_top: str


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("fma_metadata"))
    parser.add_argument("--audio-dir", type=Path, default=Path("fma_small"))
    parser.add_argument("--output-dir", type=Path, default=Path("sources_selection"))
    parser.add_argument("--raw-audio-dir", type=Path, default=Path("raw_audio"))
    parser.add_argument("--subset", default="small", help="FMA subset to select (for example: small or medium)")
    parser.add_argument("--per-genre", type=int, default=15, help="tracks per target family")
    parser.add_argument("--min-duration", type=float, default=45.0, help="minimum duration in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="select and report, but do not copy/write")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output audio files")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.per_genre < 1:
        parser.error("--per-genre must be positive")
    if args.min_duration < 0:
        parser.error("--min-duration cannot be negative")
    return args


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input file was not found: {path}")


def read_genres(path: Path) -> tuple[dict[int, str], dict[int, list[int]]]:
    """Read genre names and parent -> child links from FMA's genres.csv."""
    titles: dict[int, str] = {}
    children: dict[int, list[int]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                genre_id = int(row["genre_id"])
                parent = int(row["parent"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Malformed genre row: {row!r}") from exc
            titles[genre_id] = (row.get("title") or "").strip()
            if parent:
                children[parent].append(genre_id)
    return titles, children


def descendants(root: int, children: Mapping[int, Sequence[int]]) -> set[int]:
    """Return *root* and every descendant; tolerate malformed cyclic metadata."""
    found: set[int] = set()
    pending: deque[int] = deque([root])
    while pending:
        current = pending.popleft()
        if current in found:
            continue
        found.add(current)
        pending.extend(children.get(current, ()))
    return found


def build_target_genres(
    titles: Mapping[int, str], children: Mapping[int, Sequence[int]], targets: Mapping[str, Sequence[str]]
) -> dict[str, set[int]]:
    """Resolve target names and recursively include every subgenre."""
    title_to_ids: dict[str, list[int]] = defaultdict(list)
    for genre_id, title in titles.items():
        title_to_ids[title.casefold()].append(genre_id)

    result: dict[str, set[int]] = {}
    for label, names in targets.items():
        roots: set[int] = set()
        for name in names:
            roots.update(title_to_ids.get(name.casefold(), ()))
        if not roots:
            raise ValueError(f"Target {label!r} has no matching genre in genres.csv: {names}")
        result[label] = set().union(*(descendants(root, children) for root in roots))
        logging.info("%-10s %d genre IDs (%s)", label, len(result[label]), ", ".join(names))
    return result


def parse_genre_ids(value: str) -> tuple[int, ...]:
    """Parse FMA's Python-list-looking ``genres_all`` field safely."""
    if not value or not value.strip():
        return ()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        # Some CSV exports use a plain comma-separated list.
        parsed = value.strip("[] ").split(",")
    if not isinstance(parsed, (list, tuple, set)):
        return ()
    ids: list[int] = []
    for item in parsed:
        try:
            ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(ids)


def is_no_derivatives(license_name: str) -> bool:
    """Detect CC NoDerivatives spelling variants without rejecting non-ND CC work.

    Matches ``NoDerivatives``, the spelled-out ``No Derivative Works`` and the
    abbreviated ``NoDerivs`` via the shared ``noderiv`` stem. The stem must stay
    at ``noderiv``: an earlier ``noderivative`` stem silently let
    "Attribution-NonCommercial-NoDerivs 3.0 France" through, which is an ND
    licence and unusable here (the stimuli are derivative works by construction).
    """
    normal = license_name.casefold().replace("_", "-").replace(" ", "")
    return "noderiv" in normal or "-nd" in normal or "nd-" in normal


def fma_columns(path: Path) -> tuple[list[str], list[str]]:
    """Read and flatten the two header rows in tracks.csv."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        top = next(reader)
        bottom = next(reader)
        id_row = next(reader)
    if len(top) != len(bottom):
        raise ValueError("tracks.csv has inconsistent header-row widths")
    # The third row begins 'track_id' and has otherwise blank cells.
    if not id_row or id_row[0].strip() != "track_id":
        raise ValueError("tracks.csv does not have the expected track_id index row")
    return top, [f"{a}.{b}" if a else b for a, b in zip(top, bottom)]


def stream_candidates(path: Path, subset: str, min_duration: float) -> Iterator[Candidate]:
    """Stream qualifying rows for one FMA subset, retaining no unnecessary columns."""
    _, names = fma_columns(path)
    wanted = {
        "set.subset", "track.duration", "track.genres_all", "track.genre_top",
        "track.license", "track.title", "artist.name",
    }
    positions = {name: index for index, name in enumerate(names) if name in wanted}
    missing = wanted - positions.keys()
    if missing:
        raise ValueError(f"tracks.csv is missing expected fields: {sorted(missing)}")

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader); next(reader); next(reader)
        for row_number, row in enumerate(reader, start=4):
            try:
                if row[positions["set.subset"]].strip().casefold() != subset.casefold():
                    continue
                duration = float(row[positions["track.duration"]] or 0)
                if duration < min_duration:
                    continue
                license_name = row[positions["track.license"]].strip()
                if is_no_derivatives(license_name):
                    continue
                track_id = int(row[0])
                if track_id in FAULTY_FMA_IDS:
                    # 0s / truncated source clip; metadata duration lies, so this
                    # must be excluded by ID rather than by --min-duration.
                    logging.debug("Skipping known-faulty FMA clip %d", track_id)
                    continue
            except (IndexError, ValueError) as exc:
                logging.debug("Skipping malformed tracks.csv row %d: %s", row_number, exc)
                continue
            yield Candidate(
                track_id=track_id,
                duration=duration,
                title=row[positions["track.title"]].strip(),
                artist=row[positions["artist.name"]].strip(),
                license=license_name,
                genre_ids=parse_genre_ids(row[positions["track.genres_all"]]),
                genre_top=row[positions["track.genre_top"]].strip(),
            )


def matching_labels(candidate: Candidate, target_genres: Mapping[str, set[int]]) -> list[str]:
    present = set(candidate.genre_ids)
    return [label for label, ids in target_genres.items() if present.intersection(ids)]


def source_path(audio_dir: Path, track_id: int) -> Path:
    padded = f"{track_id:06d}"
    return audio_dir / padded[:3] / f"{padded}.mp3"


def balanced_selection(
    candidates: Iterable[Candidate], target_genres: Mapping[str, set[int]], per_genre: int, rng: random.Random
) -> tuple[dict[str, list[Candidate]], Counter[str]]:
    """Assign every candidate once, then sample equally from each assignment pool.

    Candidates that match several families are assigned to the currently smallest
    pool.  A seeded random tie-break avoids a fixed genre-priority bias while
    preserving reproducibility.
    """
    pools: dict[str, list[Candidate]] = {label: [] for label in target_genres}
    matched = Counter()
    for candidate in candidates:
        labels = matching_labels(candidate, target_genres)
        if not labels:
            continue
        matched.update(labels)
        smallest = min(len(pools[label]) for label in labels)
        choices = [label for label in labels if len(pools[label]) == smallest]
        pools[rng.choice(choices)].append(candidate)

    selected: dict[str, list[Candidate]] = {}
    for label, pool in pools.items():
        rng.shuffle(pool)
        selected[label] = pool[:per_genre]
    return selected, matched


def manifest_entry(label: str, number: int, candidate: Candidate, source: Path, destination: Path) -> dict[str, object]:
    entry = asdict(candidate)
    entry.update({
        "selection_id": f"{label}_{number:02d}", "genre": label,
        "source_path": str(source), "copied_to": str(destination),
    })
    return entry


def write_outputs(output_dir: Path, manifest: list[dict[str, object]], lines: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "selected_manifest.json"
    stats_path = output_dir / "stats.txt"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)
    genres_path, tracks_path = args.metadata_dir / "genres.csv", args.metadata_dir / "tracks.csv"
    try:
        require_file(genres_path); require_file(tracks_path)
        titles, children = read_genres(genres_path)
        target_genres = build_target_genres(titles, children, DEFAULT_TARGETS)
        logging.info("Streaming %s", tracks_path)
        selected, matched = balanced_selection(
            stream_candidates(tracks_path, args.subset, args.min_duration), target_genres, args.per_genre, random.Random(args.seed)
        )
    except (FileNotFoundError, ValueError, csv.Error) as exc:
        logging.error("Selection failed: %s", exc)
        return 2

    shortages = {label: args.per_genre - len(items) for label, items in selected.items() if len(items) < args.per_genre}
    if shortages:
        logging.error("Insufficient eligible unique tracks: %s", ", ".join(f"{k} short by {v}" for k, v in shortages.items()))
        return 3

    manifest: list[dict[str, object]] = []
    copy_errors: list[str] = []
    for label, items in selected.items():
        for number, candidate in enumerate(items, start=1):
            source = source_path(args.audio_dir, candidate.track_id)
            destination = args.raw_audio_dir / f"{label}_{number:02d}.mp3"
            if args.dry_run:
                # Deliberately do not require the archive to be extracted: this
                # mode is useful for checking metadata availability before a
                # large FMA download.
                manifest.append(manifest_entry(label, number, candidate, source, destination))
                continue
            if not source.is_file():
                copy_errors.append(f"missing source: {source}")
                continue
            manifest.append(manifest_entry(label, number, candidate, source, destination))
            args.raw_audio_dir.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not args.overwrite:
                copy_errors.append(f"destination exists (use --overwrite): {destination}")
                manifest.pop()
                continue
            try:
                shutil.copy2(source, destination)
            except OSError as exc:
                copy_errors.append(f"copy failed {source} -> {destination}: {exc}")
                manifest.pop()

    counts = Counter(str(entry["genre"]) for entry in manifest)
    expected = len(DEFAULT_TARGETS) * args.per_genre
    stats = [
        "FMA source selection", f"subset: {args.subset}", f"seed: {args.seed}", f"minimum_duration_seconds: {args.min_duration}",
        f"requested_per_genre: {args.per_genre}", f"requested_total: {expected}", f"selected_total: {len(manifest)}",
        "", "eligible metadata matches (before unique assignment):",
    ]
    stats.extend(f"  {label}: {matched[label]}" for label in DEFAULT_TARGETS)
    stats.append("assigned and copied:")
    stats.extend(f"  {label}: {counts[label]}" for label in DEFAULT_TARGETS)
    if copy_errors:
        stats.extend(["", "errors:", *[f"  {message}" for message in copy_errors]])

    if not args.dry_run:
        write_outputs(args.output_dir, manifest, stats)
    for line in stats:
        logging.info("%s", line)
    if args.dry_run:
        logging.info("Dry run: no files were copied or written.")
    if copy_errors:
        logging.error("Completed with %d copy error(s).", len(copy_errors))
        return 4
    if len(manifest) != expected:
        logging.error("Expected %d tracks but produced %d.", expected, len(manifest))
        return 4
    if args.dry_run:
        logging.info("Dry run successful: %d tracks would be copied to %s", len(manifest), args.raw_audio_dir)
    else:
        logging.info("Done: %d tracks copied to %s", len(manifest), args.raw_audio_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
