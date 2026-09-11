#!/usr/bin/env python
"""Build the Part-3 (GRPO arm) Loop & Merge table. DSP env, stdlib only.

One block, every participant, so there is no assignment to balance and no block
structure to preserve. That is the whole difference from `make_session2_loops.py`.

THREE fields per row, matching Part 2 so the same question type is reused:
  Field 1  trial_id (opaque: g001. Never the context, the arm, or which side is
           fine-tuned -- loop fields are visible in the page source and in the export)
  Field 2  Version A audio embed
  Field 3  Version B audio embed

`results/mos/grpo_ab/key.csv` holds the A/B -> fine-tuned/frozen mapping and stays
researcher-side.

The file-id trap: Qualtrics serves uploads from CP/File.php?F=F_<id> with no filename
in the URL, so a stale id silently serves the WRONG audio and nothing errors. Upload
this arm's mp3s into a NEW library folder and harvest a fresh file_ids.csv for it
rather than reusing the main study's.

Usage:
  python scripts/mos/make_grpo_loop.py --ids results/mos/qualtrics/file_ids_grpo.csv
Output: results/mos/qualtrics/loop_grpo_paste.csv (no header row)
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
EMBED = ('<audio controls preload="none" style="width:100%">'
         '<source src="{host}/CP/File.php?F={fid}" type="audio/mpeg"></audio>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="results/mos/grpo_ab/trials.csv")
    ap.add_argument("--ids", default="results/mos/qualtrics/file_ids_grpo.csv",
                    help="harvested filename,file_id for THIS arm's upload folder")
    ap.add_argument("--out", default="results/mos/qualtrics/loop_grpo_paste.csv")
    ap.add_argument("--host", default="https://qualtrics.ucl.ac.uk")
    args = ap.parse_args()

    trials = list(csv.DictReader((ROOT / args.trials).open()))
    if not trials:
        sys.exit(f"no trials in {args.trials} -- run prepare_ab_grpo.py first")

    ids = {}
    for r in csv.DictReader((ROOT / args.ids).open()):
        name = r.get("filename") or r.get("name") or r.get("file")
        fid = r.get("file_id") or r.get("id") or r.get("fid")
        if name and fid:
            ids[Path(name).name] = fid

    rows, missing = [], []
    for t in trials:
        fa, fb = ids.get(t["file_a"]), ids.get(t["file_b"])
        if not fa or not fb:
            missing.append(t["trial_id"])
            continue
        rows.append([t["trial_id"],
                     EMBED.format(host=args.host, fid=fa),
                     EMBED.format(host=args.host, fid=fb)])

    if missing:
        sys.exit(f"no file id for {len(missing)} trials ({', '.join(missing[:5])}...).\n"
                 f"Every clip must be in the upload folder {args.ids} was harvested "
                 f"from. A partial table would silently drop trials.")

    dest = ROOT / args.out
    with dest.open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)          # no header, as Qualtrics expects
    print(f"wrote {dest.relative_to(ROOT)}: {len(rows)} comparisons")
    print("rebuild the qsf with build_mos_qsf.py, then re-import")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
