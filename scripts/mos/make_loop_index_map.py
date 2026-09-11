#!/usr/bin/env python
"""Record the LIVE Qualtrics loop index of every surviving stimulus. DSP env, stdlib.

Why this exists. Loop & Merge is positional (`${lm://Field/N}`) and
`qualtrics_to_responses.py` maps a loop index back to a stimulus by row position. That
holds only while the live table and the paste table are the same list. Deleting a row
from a live table breaks it: Qualtrics does NOT renumber, it keeps the survivors at
their original indices and leaves a hole. Verified 2026-08-07 on SV_9HaoVijdm1j9bzU
after the AI-rerank rows came out of Part 1 (doc/notes/ai_rerank_removed_from_part1.md)
-- block 1 went 1-10, 12-17, 19-22, 24-45, 47, 48. Read positionally against the new
43-row table, every trial from the first hole on would be attributed to the wrong clip,
with nothing to warn you.

This writes `loopmap_block<N>.csv` (loop_index,stimulus_id) by looking each surviving
stimulus up in the table the live survey was BUILT from, which is the archive taken
before the rows were deleted. `qualtrics_to_responses.py` prefers that file when it
exists and falls back to row position when it does not.

Re-run this whenever rows are deleted from a live table. After a fresh QSF import the
indices are dense again, so delete the maps instead (or regenerate with --before equal
to the current tables, which is a no-op identity map).

The `--before` archive is kept in the working repository and is NOT shipped in the
release, because the loopmaps it produces are shipped and are what the pipeline reads.
Running this therefore needs the archive restored first.

Usage:
  python scripts/mos/make_loop_index_map.py \
      --before results/mos/qualtrics/archive_2026-08-07_pre_ai_removal \
      --after  results/mos/qualtrics
Outputs: results/mos/qualtrics/loopmap_block<N>.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)


def ids(path):
    return [r[0].strip() for r in csv.reader(open(path)) if r and r[0].strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True,
                    help="dir of the paste tables the LIVE survey was built from")
    ap.add_argument("--after", default="results/mos/qualtrics",
                    help="dir of the current paste tables (the survivors)")
    ap.add_argument("--glob", default="loop_block*_paste.csv")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    before_dir, after_dir = ROOT / args.before, ROOT / args.after
    report = {}
    for after in sorted(after_dir.glob(args.glob)):
        block = re.search(r"block(\d+)", after.stem).group(1)
        before = before_dir / after.name
        if not before.exists():
            raise SystemExit(f"no {after.name} in {before_dir}")
        original = ids(before)
        pos = {sid: i for i, sid in enumerate(original, start=1)}
        survivors = ids(after)
        # Order must be preserved, or the live table and this map describe different
        # surveys. make_qualtrics_loops.py keeps it; this is the assertion that it did.
        idx = []
        for sid in survivors:
            if sid not in pos:
                raise SystemExit(f"block{block}: {sid} is not in the pre-edit table — "
                                 "the tables were re-ordered or re-sampled, so the live "
                                 "loop indices cannot be recovered. Re-import instead.")
            idx.append(pos[sid])
        if idx != sorted(idx):
            raise SystemExit(f"block{block}: surviving rows are out of their original "
                             "order; the map would not match the live survey.")
        out = after_dir / f"loopmap_block{block}.csv"
        if not args.check:
            with out.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["loop_index", "stimulus_id"])
                w.writerows(zip(idx, survivors))
        holes = sorted(set(range(1, len(original) + 1)) - set(idx))
        report[f"block{block}"] = {"before": len(original), "after": len(survivors),
                                   "max_index": idx[-1] if idx else 0, "holes": holes}
    print(json.dumps(report, indent=1))
    if args.check:
        print("\n--check: nothing written")


if __name__ == "__main__":
    main()
