#!/usr/bin/env python
"""Qualtrics CSV export -> the response JSONs the two scorers already read. DSP env.

Qualtrics exports one wide row per participant, with looped questions spread across
columns named `<loopIndex>_<TAG>`:

    1_RATE2 .. 44_RATE2      the rating block that participant was randomised into
    1_AB5   .. 32_AB5        the A/B block they were randomised into

The loop INDEX is all the export carries -- the stimulus is not named. The mapping
back is positional: loop index i is row i of the block's paste table. That is why
the paste tables must never be re-ordered after collection starts; re-running
withdraw_sources.py mid-study would silently re-point every index.

Emits both formats, since the two halves are scored by different scripts:

  Part 1 -> results/mos/responses/*.json
            {"session_code", "block", "responses":[{"stimulus_id","rating"}]}
            read by score_mos_responses.py
  Part 2 -> results/mos/session2/responses/*.json
            {"session_code", "responses":[{"trial_id","preference"}]}
            read by score_ab_session2.py

Skips are dropped, not passed through. `Skip -- uncomfortable` recodes to 0, and 0
is not a rating: averaged into a MOS it would look like an extreme judgement, and
in Part 2 it is not the "about the same" midpoint (that is 3). They are counted and
reported instead.

Response types are filtered by the `Status` column. Status 2 = Survey Test,
1 = Survey Preview, 0/8 = real. Test and preview rows are EXCLUDED by default so
rehearsal data cannot silently inflate an N; --include-test opts them back in for
pipeline rehearsal.

Usage:
  python scripts/mos/qualtrics_to_responses.py --csv export.csv --include-test
  python scripts/mos/qualtrics_to_responses.py --csv export.csv        # real responses only
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

STATUS = {"0": "IP Address", "1": "Survey Preview", "2": "Survey Test",
          "4": "Imported", "8": "Spam", "9": "Survey Test"}
REHEARSAL = {"1", "2"}
SKIP = "0"


def load_loop_ids(path):
    """Qualtrics loop index -> the id in field 1.

    Normally the index is just the row position, 1-indexed. That breaks the moment a
    row is DELETED from a live Loop & Merge table: Qualtrics keeps the surviving rows
    at their original numbers and leaves a hole (verified 2026-08-07 on
    SV_9HaoVijdm1j9bzU, where removing the AI-rerank rows left block 1 numbered
    1-10, 12-17, 19-22, 24-45, 47, 48). Row position would then be off by one per
    hole, silently mis-labelling every later trial.

    So a sibling `loopmap_<block>.csv` (loop_index,stimulus_id) wins when present: it
    records the live numbering explicitly. Row position is only the fallback, correct
    for any table that has never had a row deleted.
    """
    lm = path.with_name(re.sub(r"^loop_", "loopmap_", path.name)
                        .replace("_paste.csv", ".csv"))
    if lm.exists():
        return {int(r["loop_index"]): r["stimulus_id"]
                for r in csv.DictReader(open(lm)) if r.get("stimulus_id")}
    rows = [r for r in csv.reader(open(path)) if r and r[0].strip()]
    return {i: r[0] for i, r in enumerate(rows, start=1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Qualtrics CSV export (numeric values)")
    ap.add_argument("--loop-dir", default="results/mos/qualtrics")
    ap.add_argument("--out-part1", default="results/mos/responses")
    ap.add_argument("--out-part2", default="results/mos/session2/responses")
    ap.add_argument("--include-test", action="store_true",
                    help="also convert Survey Test / Survey Preview rows")
    ap.add_argument("--prefix", default="qx")
    ap.add_argument("--unfinished", action="store_true",
                    help="also convert rows with Finished != 1")
    args = ap.parse_args()

    rows = list(csv.reader(open(ROOT / args.csv if not Path(args.csv).is_absolute()
                                else args.csv, encoding="utf-8")))
    if len(rows) < 4:
        sys.exit(f"{args.csv}: only {len(rows)} rows — Qualtrics exports 3 header rows "
                 "plus one row per response")
    hdr, data = rows[0], rows[3:]

    loop_dir = ROOT / args.loop_dir
    # globbed, not a hardcoded range: the block count is set by however many paste
    # tables exist (5 rating + 10 A/B today). A range that stops short does not skip
    # those participants quietly, it aborts the whole ingest on the first one.
    def load_blocks(pattern):
        out = {}
        for path in loop_dir.glob(pattern):
            out[int(re.search(r"block(\d+)", path.stem).group(1))] = load_loop_ids(path)
        return out

    p1_ids = load_blocks("loop_block*_paste.csv")
    p2_ids = load_blocks("loop_s2_block*_paste.csv")
    if not p1_ids:
        sys.exit(f"no loop_block*_paste.csv in {args.loop_dir}")

    cols = {}
    for i, h in enumerate(hdr):
        m = re.fullmatch(r"(\d+)_(RATE|AB)(\d+)", h)
        if m:
            cols[i] = (int(m.group(1)), m.group(2), int(m.group(3)))

    out1 = ROOT / args.out_part1
    out2 = ROOT / args.out_part2
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    kept, skipped_rows = 0, Counter()
    tot_skip1 = tot_skip2 = 0
    for row in data:
        d = dict(zip(hdr, row))
        status = (d.get("Status") or "").strip()
        rid = (d.get("ResponseId") or "").strip() or f"row{kept}"
        if status in REHEARSAL and not args.include_test:
            skipped_rows[STATUS.get(status, status)] += 1
            continue
        if (d.get("Finished") or "").strip() != "1" and not args.unfinished:
            skipped_rows["unfinished"] += 1
            continue

        p1, p2 = [], []
        blk1 = blk2 = None
        n_skip1 = n_skip2 = 0
        for i, (idx, kind, blk) in cols.items():
            v = (row[i] if i < len(row) else "").strip()
            if not v:
                continue
            if kind == "RATE":
                blk1 = blk
                if v == SKIP:
                    n_skip1 += 1
                    continue
                sid = p1_ids.get(blk, {}).get(idx)
                if sid is None:
                    sys.exit(f"{rid}: loop index {idx} has no row in block {blk} "
                             "— loop tables changed since collection?")
                p1.append({"stimulus_id": sid, "rating": int(v)})
            else:
                blk2 = blk
                if v == SKIP:
                    n_skip2 += 1
                    continue
                tid = p2_ids.get(blk, {}).get(idx)
                if tid is None:
                    sys.exit(f"{rid}: A/B loop index {idx} has no row in block {blk}")
                p2.append({"trial_id": tid, "preference": int(v)})

        stamp = datetime.now(timezone.utc).isoformat()
        meta = {"session_code": rid, "generated": stamp,
                "source": Path(args.csv).name,
                "qualtrics_status": STATUS.get(status, status)}
        if status in REHEARSAL:
            meta["dry_run"] = True

        if p1:
            (out1 / f"{args.prefix}_{rid}.json").write_text(json.dumps(
                {**meta, "block": f"block{blk1}", "n_skipped": n_skip1,
                 "responses": p1}, indent=1))
        if p2:
            (out2 / f"{args.prefix}_{rid}.json").write_text(json.dumps(
                {**meta, "ab_block": f"block{blk2}", "n_skipped": n_skip2,
                 "responses": p2}, indent=1))
        tot_skip1 += n_skip1
        tot_skip2 += n_skip2
        kept += 1
        print(f"  {rid}: block{blk1} {len(p1)} ratings (+{n_skip1} skipped) | "
              f"A/B block{blk2} {len(p2)} prefs (+{n_skip2} skipped)"
              + ("   [REHEARSAL]" if status in REHEARSAL else ""))

    print(f"\nconverted {kept} response(s)")
    if skipped_rows:
        print("  excluded:", dict(skipped_rows),
              "(use --include-test / --unfinished to keep)")
    print(f"  skips dropped: {tot_skip1} rating, {tot_skip2} preference")
    print(f"  part 1 -> {args.out_part1}/{args.prefix}_*.json")
    print(f"  part 2 -> {args.out_part2}/{args.prefix}_*.json")


if __name__ == "__main__":
    main()
