#!/usr/bin/env python
"""Replace the AUDIO_BASE_URL placeholder in the Loop & Merge tables with real
Qualtrics File Library URLs. DSP env (stdlib only).

Qualtrics does NOT serve library files at a filename-addressable path: each upload
gets an opaque id and is served from

    https://<brand>/CP/File.php?F=F_<id>

so `make_qualtrics_loops.py --base-url <prefix>` cannot work. The filename -> F_id
mapping has to be harvested from the File Library UI once (see
results/mos/qualtrics/file_ids.csv, harvested 2026-08-02) and joined in here.

Fails loudly on any filename that has no id: a silently unstamped row would show a
participant a dead audio player and cost that trial.

Usage:
  python scripts/mos/stamp_qualtrics_urls.py                    # writes loop_block*_live.csv
  python scripts/mos/stamp_qualtrics_urls.py --check            # verify only, write nothing
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
DEFAULT_HOST = "https://qualtrics.ucl.ac.uk"
PLACEHOLDER_RE = re.compile(r"AUDIO_BASE_URL/([A-Za-z0-9_]+\.mp3)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop-dir", default="results/mos/qualtrics")
    ap.add_argument("--ids", default="results/mos/qualtrics/file_ids.csv")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--glob", default="loop_block*.csv")
    ap.add_argument("--suffix", default="_live")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--paste", action="store_true",
                    help="also emit *_paste.csv with the header row stripped — Loop & "
                         "Merge fields are positional (${lm://Field/N}), so a pasted "
                         "header becomes a bogus first loop iteration")
    args = ap.parse_args()

    ids = {r["filename"]: r["file_id"]
           for r in csv.DictReader(open(ROOT / args.ids))}
    if not ids:
        sys.exit(f"no ids in {args.ids}")

    loop_dir = ROOT / args.loop_dir
    # exclude BOTH generated forms: on a re-run the previous _live/_paste files
    # are already on disk and would otherwise be re-stamped as if they were
    # sources (they contain no placeholders, so the row/clip guard trips).
    sources = sorted(p for p in loop_dir.glob(args.glob)
                     if not re.search(r"_(live|paste)$", p.stem))
    if not sources:
        sys.exit(f"no {args.glob} under {loop_dir}")

    missing, total = set(), 0
    for src in sources:
        text = src.read_text()
        names = PLACEHOLDER_RE.findall(text)
        missing |= {n for n in names if n not in ids}
        total += len(names)

    if missing:
        sys.exit(f"{len(missing)} filename(s) absent from {args.ids}: "
                 f"{sorted(missing)[:5]} — re-harvest the library before stamping")

    print(f"{total} audio refs across {len(sources)} file(s), all resolvable")
    if args.check:
        return

    for src in sources:
        text = src.read_text()
        stamped = PLACEHOLDER_RE.sub(
            lambda m: f"{args.host}/CP/File.php?F={ids[m.group(1)]}", text)
        if "AUDIO_BASE_URL" in stamped:
            sys.exit(f"{src.name}: placeholder survived substitution")
        out = src.with_name(f"{src.stem}{args.suffix}.csv")
        out.write_text(stamped)
        n = len(PLACEHOLDER_RE.findall(text))
        print(f"  {src.name} -> {out.name}  ({n} clips)")

        if args.paste:
            body = stamped.split("\n", 1)[1]
            rows = len([r for r in csv.reader(body.splitlines()) if r])
            if rows != n:
                sys.exit(f"{src.name}: {rows} rows but {n} clips — refusing to write")
            paste = src.with_name(f"{src.stem}_paste.csv")
            paste.write_text(body)
            print(f"    {paste.name}  ({rows} rows, no header)")

    print("\nPaste the *_paste.csv files into Loop & Merge. Never key_block*.csv.")


if __name__ == "__main__":
    main()
