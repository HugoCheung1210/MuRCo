#!/usr/bin/env python3
"""Download a selected FMA subset without downloading the complete ZIP archive.

FMA Medium is a 23.8 GB ZIP file.  Its official server supports HTTP byte
ranges, so this program reads the ZIP central directory and requests only the
compressed byte ranges for the selected MP3 files.

Example::

    python scripts/data/download_selected_fma.py

By default this makes the same reproducible 90-track Medium selection as
``select_fma_sources.py`` and writes it to ``raw_audio/``.  It also writes a
manifest and statistics to ``sources_selection/``.  ``--list-only`` performs
the selection and verifies that every selected filename exists in the remote
archive, without downloading audio.

The Medium archive contains 30-second excerpts.  It is therefore unsuitable
when the actual audio (rather than metadata duration) must be at least 45 s.
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
import binascii
import bz2
import json
import logging
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import select_fma_sources as selector


ARCHIVE_URL = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
EOCD = b"PK\x05\x06"
ZIP64_LOCATOR = b"PK\x06\x07"
ZIP64_EOCD = b"PK\x06\x06"
CENTRAL_HEADER = b"PK\x01\x02"
LOCAL_HEADER = b"PK\x03\x04"


class RemoteZipError(RuntimeError):
    """A remote server or ZIP archive did not meet the required ZIP contract."""


class RemoteZip:
    """Minimal read-only ZIP reader backed by HTTP Range requests.

    This purpose-built reader supports the Zip64 central directory used by the
    FMA Medium archive.  It does not write a temporary archive to disk.
    """

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self.size = self._content_length()
        self.entries = self._read_central_directory()

    def _request(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self.size:
            raise RemoteZipError(f"Invalid byte range {start}-{end} for {self.size}-byte archive")
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", response.getcode())
                data = response.read()
        except urllib.error.URLError as exc:
            raise RemoteZipError(f"Range request failed for {start}-{end}: {exc}") from exc
        if status != 206:
            raise RemoteZipError(f"Server did not honour Range request (HTTP {status}); cannot selectively download")
        expected = end - start + 1
        if len(data) != expected:
            raise RemoteZipError(f"Short range response: expected {expected} bytes, got {len(data)}")
        return data

    def _content_length(self) -> int:
        request = urllib.request.Request(self.url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                ranges = response.headers.get("Accept-Ranges", "")
                length = response.headers.get("Content-Length")
        except urllib.error.URLError as exc:
            raise RemoteZipError(f"Could not inspect archive: {exc}") from exc
        if "bytes" not in ranges.casefold():
            raise RemoteZipError("Archive server does not advertise HTTP byte-range support")
        try:
            return int(length)
        except (TypeError, ValueError) as exc:
            raise RemoteZipError("Archive server did not provide Content-Length") from exc

    def _read_central_directory(self) -> dict[str, tuple[int, int, int, int]]:
        # EOCD is within the final 65,557 bytes (max comment plus fixed record).
        tail_start = max(0, self.size - 65_557)
        tail = self._request(tail_start, self.size - 1)
        index = tail.rfind(EOCD)
        if index < 0:
            raise RemoteZipError("ZIP end-of-central-directory record was not found")
        _, _, _, _, entries, cd_size32, cd_offset32, _ = struct.unpack_from("<4s4H2LH", tail, index)
        if entries == 0xFFFF or cd_size32 == 0xFFFFFFFF or cd_offset32 == 0xFFFFFFFF:
            locator_index = tail.rfind(ZIP64_LOCATOR, 0, index)
            if locator_index < 0:
                raise RemoteZipError("ZIP64 archive is missing its locator")
            _, _, zip64_offset, _ = struct.unpack_from("<4sLQL", tail, locator_index)
            record = self._request(zip64_offset, zip64_offset + 55)
            if record[:4] != ZIP64_EOCD:
                raise RemoteZipError("ZIP64 end-of-central-directory record is invalid")
            _, _, _, _, _, _, _, entries, cd_size, cd_offset = struct.unpack("<4sQ2H2L4Q", record)
        else:
            cd_size, cd_offset = cd_size32, cd_offset32
        central = self._request(cd_offset, cd_offset + cd_size - 1)
        entries_by_name: dict[str, tuple[int, int, int, int]] = {}
        cursor = 0
        while cursor < len(central):
            if central[cursor:cursor + 4] != CENTRAL_HEADER:
                raise RemoteZipError(f"Bad central-directory entry at offset {cursor}")
            fields = struct.unpack_from("<4s6H3L5H2L", central, cursor)
            _, _, _, flags, method, _, _, crc, compressed, uncompressed, name_len, extra_len, comment_len, _, _, _, local_offset = fields
            name_start = cursor + 46
            name_end = name_start + name_len
            extra_end = name_end + extra_len
            name = central[name_start:name_end].decode("utf-8")
            extra = central[name_end:extra_end]
            compressed, uncompressed, local_offset = self._zip64_values(
                extra, compressed, uncompressed, local_offset
            )
            if flags & 1:
                raise RemoteZipError(f"Encrypted ZIP entry is unsupported: {name}")
            entries_by_name[name] = (local_offset, compressed, method, crc)
            cursor = extra_end + comment_len
        if len(entries_by_name) != entries:
            logging.warning("ZIP reports %d entries; parsed %d", entries, len(entries_by_name))
        return entries_by_name

    @staticmethod
    def _zip64_values(extra: bytes, compressed: int, uncompressed: int, offset: int) -> tuple[int, int, int]:
        """Read replacement uint64 values from the Zip64 extended-information extra."""
        cursor = 0
        values = b""
        while cursor + 4 <= len(extra):
            header_id, size = struct.unpack_from("<HH", extra, cursor)
            payload = extra[cursor + 4:cursor + 4 + size]
            if header_id == 1:
                values = payload
                break
            cursor += 4 + size
        position = 0
        def next_value() -> int:
            nonlocal position
            if position + 8 > len(values):
                raise RemoteZipError("Truncated Zip64 extended-information field")
            value = struct.unpack_from("<Q", values, position)[0]
            position += 8
            return value
        if uncompressed == 0xFFFFFFFF:
            uncompressed = next_value()
        if compressed == 0xFFFFFFFF:
            compressed = next_value()
        if offset == 0xFFFFFFFF:
            offset = next_value()
        return compressed, uncompressed, offset

    def read(self, name: str) -> bytes:
        """Return decompressed contents of an archive entry and verify its CRC-32."""
        try:
            offset, compressed_size, method, expected_crc = self.entries[name]
        except KeyError as exc:
            raise RemoteZipError(f"Selected file is absent from remote archive: {name}") from exc
        header = self._request(offset, offset + 29)
        if header[:4] != LOCAL_HEADER:
            raise RemoteZipError(f"Invalid local header for {name}")
        _, _, flags, local_method, _, _, _, _, _, name_len, extra_len = struct.unpack("<4s5H3L2H", header)
        if flags & 1 or local_method != method:
            raise RemoteZipError(f"Unexpected local ZIP header for {name}")
        data_offset = offset + 30 + name_len + extra_len
        compressed = self._request(data_offset, data_offset + compressed_size - 1)
        if method == 0:
            data = compressed
        elif method == 8:
            data = zlib.decompress(compressed, -zlib.MAX_WBITS)
        elif method == 12:
            # FMA's published ZIP archives use bzip2 compression.
            data = bz2.decompress(compressed)
        else:
            raise RemoteZipError(f"Unsupported ZIP compression method {method} for {name}")
        if binascii.crc32(data) & 0xFFFFFFFF != expected_crc:
            raise RemoteZipError(f"CRC-32 check failed for {name}")
        return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("fma_metadata"))
    parser.add_argument("--archive-url", default=ARCHIVE_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("raw_audio"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("sources_selection"))
    parser.add_argument("--per-genre", type=int, default=15)
    parser.add_argument("--min-duration", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="verify remote filenames but download no MP3s")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.per_genre < 1 or args.min_duration < 0 or args.timeout <= 0:
        parser.error("--per-genre, --min-duration, and --timeout must be positive")
    return args


def select_tracks(metadata_dir: Path, per_genre: int, min_duration: float, seed: int) -> dict[str, list[selector.Candidate]]:
    genres_path, tracks_path = metadata_dir / "genres.csv", metadata_dir / "tracks.csv"
    selector.require_file(genres_path)
    selector.require_file(tracks_path)
    titles, children = selector.read_genres(genres_path)
    targets = selector.build_target_genres(titles, children, selector.DEFAULT_TARGETS)
    selected, _ = selector.balanced_selection(
        selector.stream_candidates(tracks_path, "medium", min_duration), targets, per_genre, __import__("random").Random(seed)
    )
    short = [label for label, items in selected.items() if len(items) < per_genre]
    if short:
        raise RemoteZipError(f"Not enough eligible Medium tracks for: {', '.join(short)}")
    return selected


def archive_name(track_id: int) -> str:
    padded = f"{track_id:06d}"
    return f"fma_medium/{padded[:3]}/{padded}.mp3"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")
    try:
        selected = select_tracks(args.metadata_dir, args.per_genre, args.min_duration, args.seed)
        remote = RemoteZip(args.archive_url, args.timeout)
    except (OSError, ValueError, urllib.error.URLError, RemoteZipError) as exc:
        logging.error("Preparation failed: %s", exc)
        return 2

    planned = [(label, number, candidate) for label, candidates in selected.items() for number, candidate in enumerate(candidates, 1)]
    missing = [archive_name(candidate.track_id) for _, _, candidate in planned if archive_name(candidate.track_id) not in remote.entries]
    if missing:
        logging.error("%d selected track(s) are absent from the archive; first: %s", len(missing), missing[0])
        return 3
    logging.info("Remote ZIP indexed: %d entries; all %d selected MP3s are present.", len(remote.entries), len(planned))
    if args.list_only:
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    errors: list[str] = []
    for index, (label, number, candidate) in enumerate(planned, 1):
        destination = args.output_dir / f"{label}_{number:02d}.mp3"
        if destination.exists() and not args.overwrite:
            errors.append(f"already exists (use --overwrite): {destination}")
            continue
        try:
            data = remote.read(archive_name(candidate.track_id))
            destination.write_bytes(data)
        except (OSError, RemoteZipError) as exc:
            errors.append(f"{destination.name}: {exc}")
            continue
        entry = asdict(candidate)
        entry.update({"selection_id": f"{label}_{number:02d}", "genre": label, "archive_entry": archive_name(candidate.track_id), "copied_to": str(destination)})
        manifest.append(entry)
        logging.info("[%d/%d] downloaded %s", index, len(planned), destination.name)

    stats = ["FMA Medium selective source download", f"seed: {args.seed}", f"selected_total: {len(manifest)}"]
    stats.extend(f"{label}: {sum(e['genre'] == label for e in manifest)}" for label in selector.DEFAULT_TARGETS)
    if errors:
        stats.extend(["", "errors:", *errors])
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    (args.manifest_dir / "selected_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.manifest_dir / "stats.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")
    if errors:
        logging.error("Finished with %d error(s); see %s", len(errors), args.manifest_dir / "stats.txt")
        return 4
    logging.info("Downloaded %d files without downloading the full archive.", len(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
