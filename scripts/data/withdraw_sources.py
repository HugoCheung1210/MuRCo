#!/usr/bin/env python
"""Withdraw source tracks from the MOS study after the loop tables exist. DSP env.

The §E listening audit screened for lyrics and distressing content and kept four
ambient sources flagged `harsh_keep_but_warn` (ambient_01, 02, 10, 12). Piloting
the built survey showed "keep but warn" was not enough -- §F gives the warning
once at the start and deliberately never per-clip, so a harsh clip arrives with
no warning attached. Those sources are withdrawn.

This filters the loop tables rather than re-running the sampler, because
re-sampling would mint new opaque stimulus ids and force a re-render and re-upload
of all 190 clips. Removal alone is safe here: the surviving blocks stay matched on
condition composition, and ratings-per-stimulus is unchanged (every stimulus is
still in exactly one block, seen by every rater assigned that block).

Checks before writing:
  - a withdrawn source must not be the `partner_id` of a surviving style_swap
    pair, or its audio would still reach participants as segment B;
  - AI-rerank seeds must stay whole (a seed's C-winner and C-loser share a block,
    which is the whole point of that block).

Audio is NOT deleted from the Qualtrics library. Orphaned files are harmless and
unreferenced; deleting them would invalidate file_ids.csv for no benefit.

Usage:
  python scripts/data/withdraw_sources.py --sources ambient_01,ambient_02,ambient_10,ambient_12
  python scripts/data/withdraw_sources.py --sources ... --check
Then: python scripts/mos/stamp_qualtrics_urls.py --paste && python scripts/mos/build_mos_qsf.py ...
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True, help="comma-separated source_ids")
    ap.add_argument("--index", default="results/mos/stimulus_index.csv")
    ap.add_argument("--manifest", default="perturbations/pairs_manifest.json")
    ap.add_argument("--loop-dir", default="results/mos/qualtrics")
    ap.add_argument("--record", default="results/mos/withdrawn_sources.json")
    ap.add_argument("--reason", default="harsh on audition; §F gives no per-clip warning")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    bad = {s.strip() for s in args.sources.split(",") if s.strip()}
    idx = list(csv.DictReader(open(ROOT / args.index)))
    known = {r["source_id"] for r in idx}
    unknown = bad - known
    if unknown:
        sys.exit(f"not source_ids in {args.index}: {sorted(unknown)}")

    manifest = json.load(open(ROOT / args.manifest))
    partner = {r["pair_id"]: r.get("partner_id") for r in manifest["pairs"]}

    drop = {r["stimulus_id"] for r in idx if r["source_id"] in bad}
    # a surviving pair must not borrow withdrawn audio as its segment B
    borrowed = [r["stimulus_id"] for r in idx
                if r["source_id"] not in bad and partner.get(r["pair_id"]) in bad]
    if borrowed:
        print(f"  {len(borrowed)} surviving stimuli borrow withdrawn audio as segment B "
              "— withdrawing those too")
        drop |= set(borrowed)

    keep = [r for r in idx if r["stimulus_id"] not in drop]

    # AI-rerank seeds must stay whole
    seeds = defaultdict(set)
    for r in keep:
        m = re.match(r"rerank_ace/(s\d+)::", r["pair_id"])
        if m:
            seeds[m.group(1)].add(r["block"])
    split = {s: b for s, b in seeds.items() if len(b) > 1}
    if split:
        sys.exit(f"AI seeds split across blocks: {split}")
    dropped_ai = [r for r in idx if r["stimulus_id"] in drop and "rerank" in r["pair_id"]]
    if dropped_ai:
        sys.exit(f"refusing: withdrawal would break {len(dropped_ai)} AI-rerank stimuli")

    print(f"withdrawing {len(bad)} source(s): {', '.join(sorted(bad))}")
    print(f"  stimuli removed: {len(drop)} of {len(idx)}  (borrowed as segment B: {len(borrowed)})")
    print(f"  attention checks lost: "
          f"{sum(1 for r in idx if r['stimulus_id'] in drop and r['attention_check'])}")
    print("  conditions hit:", dict(Counter(r["perturbation"] for r in idx
                                            if r["stimulus_id"] in drop)))

    loop_dir = ROOT / args.loop_dir
    sources = sorted(p for p in loop_dir.glob("loop_block*.csv")
                     if not re.search(r"_(live|paste)$", p.stem))
    if not sources:
        sys.exit(f"no source loop tables in {loop_dir}")

    print()
    total_removed = 0
    for src in sources:
        rows = list(csv.reader(open(src)))
        header, body = rows[0], rows[1:]
        kept = [r for r in body if r and r[0] not in drop]
        removed = len(body) - len(kept)
        total_removed += removed
        print(f"  {src.name}: {len(body)} -> {len(kept)} rows (-{removed})")
        if not args.check:
            with src.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerows(kept)

    if args.check:
        print("\n--check: nothing written")
        return

    record = {
        "withdrawn": sorted(bad),
        "reason": args.reason,
        "date": date.today().isoformat(),
        "stimuli_removed": sorted(drop),
        "n_stimuli_removed": len(drop),
        "n_before": len(idx),
        "n_after": len(keep),
        "audio_left_in_qualtrics_library": True,
        "note": "loop tables filtered; stimulus_index.csv left intact so the "
                "withdrawn rows stay auditable. Re-run stamp_qualtrics_urls.py "
                "--paste then build_mos_qsf.py.",
    }
    (ROOT / args.record).write_text(json.dumps(record, indent=2))
    print(f"\nwrote {args.record}  ({total_removed} loop rows removed)")
    print("NEXT: python scripts/mos/stamp_qualtrics_urls.py --paste")
    print("      python scripts/mos/build_mos_qsf.py --practice-clips practice_hi,practice_lo")


if __name__ == "__main__":
    main()
