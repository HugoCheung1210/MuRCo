#!/usr/bin/env python
"""Turn the rendered stimuli into per-block Qualtrics Loop & Merge tables. DSP env.

Build pattern (already settled in doc/ethics/participant_information_sheet_ucl.md §3):
audio hosted INSIDE the Qualtrics library, Loop & Merge to repeat the 1-5 rating
question over a block's clips, Randomizer in the Survey Flow to assign one block
per participant.

Emits, per block, TWO files that must be kept apart:
  loop_block<N>.csv  -> paste into Qualtrics Loop & Merge. Contains ONLY an
                        opaque stimulus id and the audio embed. No condition
                        labels: loop fields end up in the page source and the
                        response export, so anything here is visible to a curious
                        participant and would leak the design.
  key_block<N>.csv   -> researcher-side only. stimulus_id -> pair_id, condition,
                        attention-check role. Never uploaded.

The embed uses a placeholder base URL; after uploading the clips to the Qualtrics
File Library, replace AUDIO_BASE_URL with the library path (one find-and-replace).

The AI-rerank stimuli are EXCLUDED from Part 1 by default (decision 2026-08-07,
doc/notes/ai_rerank_removed_from_part1.md): the Part-1 scale asks how much changes
between A and B, which both arms of a rerank pair max out, so the winner and loser
cannot separate on it. Part 2 (the A/B head-to-head) carries the reranking claim.
The filtering happens HERE and not in make_mos_stimuli.py --no-ai on purpose:
render_mos_stimuli.py numbers clips sequentially over the plan, so dropping the AI
rows upstream would renumber the 20 stimuli that sort after them and silently
re-point every already-uploaded file id (results/mos/qualtrics/file_ids.csv).
Filtering the loop tables leaves the rendered clips and their ids alone; the AI
mp3s simply stay in the library unreferenced.

Usage:
  python scripts/mos/make_qualtrics_loops.py
  python scripts/mos/make_qualtrics_loops.py --base-url https://uclpsych.eu.qualtrics.com/...
  python scripts/mos/make_qualtrics_loops.py --include-sets all   # pre-2026-08-07 scope
Outputs: results/mos/qualtrics/{loop,key}_block*.csv + README.md
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
PLACEHOLDER = "AUDIO_BASE_URL"

EMBED = ('<audio controls preload="none" style="width:100%%">'
         '<source src="%s/%s" type="audio/mpeg"></audio>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="results/mos/stimulus_index.csv")
    ap.add_argument("--plan", default="results/mos/stimuli_plan.json")
    ap.add_argument("--out-dir", default="results/mos/qualtrics")
    ap.add_argument("--base-url", default=PLACEHOLDER,
                    help="Qualtrics library URL prefix; left as a placeholder by default")
    ap.add_argument("--withdrawn", default="results/mos/withdrawn_sources.json",
                    help="record written by scripts/data/withdraw_sources.py. Its stimuli "
                         "are dropped here too: that script filters the loop tables in "
                         "place, so without this a re-run of this script would put the "
                         "withdrawn (harsh) audio back in front of participants. Pass an "
                         "absent path to ignore it")
    ap.add_argument("--include-sets", default="matrix",
                    help="comma list of stimulus_set values to put in Part 1, or 'all'. "
                         "Default matrix-only: the AI-rerank arm was removed from Part 1 "
                         "on 2026-08-07 (doc/notes/ai_rerank_removed_from_part1.md)")
    ap.add_argument("--repermute", action="store_true",
                    help="re-shuffle row order from the seed instead of preserving the "
                         "order of the tables already on disk. Only safe before a survey "
                         "has been built from them: Loop & Merge indices are positional")
    ap.add_argument("--seed", type=int, default=20260718)
    args = ap.parse_args()

    index = list(csv.DictReader(open(ROOT / args.index)))
    meta = json.load(open(ROOT / args.plan))["meta"]
    keep = None if args.include_sets.strip() == "all" else {
        s.strip() for s in args.include_sets.split(",") if s.strip()}
    dropped = Counter()
    wpath = ROOT / args.withdrawn
    if wpath.exists():
        rec = json.loads(wpath.read_text())
        gone = set(rec.get("stimuli_removed", ()))          # includes segment-B borrowers
        wsrc = set(rec.get("withdrawn", ()))
        n0 = len(index)
        index = [r for r in index
                 if r["stimulus_id"] not in gone and r["source_id"] not in wsrc]
        dropped["withdrawn"] = n0 - len(index)
    if keep is not None:
        dropped.update(r["stimulus_set"] for r in index
                       if r["stimulus_set"] not in keep)
        index = [r for r in index if r["stimulus_set"] in keep]
        if not index:
            raise SystemExit(f"--include-sets {args.include_sets} matched no stimuli")
    rng = np.random.RandomState(args.seed)

    # every participant rates the anchor + their own matrix block + their own AI block
    anchor = [r for r in index if r["block"] == "anchor"]
    blocks = defaultdict(list)
    for r in index:
        if r["block"] != "anchor":
            blocks[r["block"]].append(r)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for block in sorted(blocks):
        rows = anchor + blocks[block]
        order = rng.permutation(len(rows))          # fixed base order; ALSO enable
        rows = [rows[i] for i in order]             # "randomize loop order" in Qualtrics
        loop_path = out / f"loop_{block}.csv"
        # A table already on disk is the one a live survey was built from, and Loop &
        # Merge is POSITIONAL (${lm://Field/N}), so a re-permutation here silently
        # re-points every index in data already collected. Keep the existing order for
        # the rows that survive and append only genuinely new ones. --repermute opts out,
        # which is safe only before a survey is built.
        if loop_path.exists() and not args.repermute:
            prev = [r[0] for r in list(csv.reader(open(loop_path)))[1:] if r]
            pos = {sid: i for i, sid in enumerate(prev)}
            rows.sort(key=lambda r: (pos.get(r["stimulus_id"], len(pos)),
                                     r["stimulus_id"]))
        key_path = out / f"key_{block}.csv"
        with open(loop_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stimulus_id", "audio"])
            for r in rows:
                w.writerow([r["stimulus_id"], EMBED % (args.base_url, r["file"])])
        with open(key_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "loop_position", "stimulus_id", "pair_id", "stimulus_set",
                "condition", "perturbation", "attention_check", "C_uniform"])
            w.writeheader()
            for i, r in enumerate(rows, 1):
                w.writerow({"loop_position": i,
                            **{k: r[k] for k in ("stimulus_id", "pair_id",
                                                 "stimulus_set", "condition",
                                                 "perturbation", "attention_check",
                                                 "C_uniform")}})
        report[block] = {
            "trials": len(rows),
            "anchor": len(anchor),
            "by_set": dict(Counter(r["stimulus_set"] for r in rows)),
            "attention_checks": dict(Counter(r["attention_check"]
                                             for r in rows if r["attention_check"])),
        }

    # Trial counts come from the tables actually written, not from the plan's
    # items_per_participant: the plan still carries the AI block that Part 1 no
    # longer uses, and §E source withdrawals already left the blocks unequal.
    expected = max(v["trials"] for v in report.values())
    n_pairs = len({r["stimulus_id"] for r in index})
    scope = (f"anchor {len(anchor)} + matrix block"
             if keep == {"matrix"} else
             f"anchor {len(anchor)} + one block of each of {sorted(keep or ['all'])}")
    why = {"withdrawn": f"`{args.withdrawn}` (harsh on audition, §F warns only once)",
           "ai_rerank": "`doc/notes/ai_rerank_removed_from_part1.md` (the Part-1 scale "
                        "measures change magnitude, which both rerank arms max out)"}
    dropped_line = ("\nExcluded from Part 1:\n"
                    + "".join(f"- **{k}**, {v} clips — {why.get(k, 'see notes')}\n"
                              for k, v in sorted(dropped.items()) if v)
                    if dropped else "")
    readme = out / "README.md"
    readme.write_text(f"""# Qualtrics build — MOS study

Generated by `scripts/mos/make_qualtrics_loops.py` (seed {args.seed}).
Design: {n_pairs} unique pairs, {meta['n_participants']} participants,
up to {expected} trials each ({scope}).
{dropped_line}
## Order of work
1. **Anonymise first.** Survey Options -> Security -> *Anonymize Responses* ON
   (do not record IP). This is load-bearing for the "no personal data" claim —
   verify before a single response is collected.
2. Upload `results/mos/stimuli/*.mp3` to the Qualtrics **File Library**.
   Host inside Qualtrics only; an external CDN would log participant IPs.
   **DONE 2026-08-02** — all 318 clips (190 part-1 + 128 session-2) uploaded.
   The AI-rerank clips stay in the library but are no longer referenced by any
   loop table; leave them, re-uploading would change every file id.
3. Stamp the real URLs: `python scripts/mos/stamp_qualtrics_urls.py --paste` →
   `loop_block<N>_live.csv` + `loop_block<N>_paste.csv`.
   **`--base-url` does not work.** Qualtrics serves library files from
   `.../CP/File.php?F=F_<opaque id>` — no filename in the path — so there is no
   prefix to substitute. The filename → `F_` map was harvested from the File
   Library UI into `file_ids.csv` (318/318, one-to-one, verified against disk);
   the stamper joins on it and hard-fails on any unresolved clip.
   Verified: a stamped URL returns 200 `audio/mpeg`, byte-identical to the local
   mp3, **with no Qualtrics session cookie** — so participants can play it.
   Re-harvest `file_ids.csv` if clips are ever re-uploaded; ids change.
4. Build the instrument with `python scripts/mos/build_mos_qsf.py` and import the
   generated `mos_full.qsf`. It reads `loop_block<N>_paste.csv` and carries the
   rating question, the Loop & Merge tables, the consent screen and the flow.
   Never hand-edit the live survey — re-import instead.
5. Survey Flow: **Randomizer**, evenly present 1 of the {meta['n_blocks']} blocks,
   *Evenly Present Elements* ON (the generated QSF already does this).
6. Consent tick-list as the first screen, before any audio.

## Never upload
`key_block*.csv` — it maps each opaque stimulus id to its condition and the
attention-check answers. Researcher-side only.

## Blocks
```json
{json.dumps(report, indent=1)}
```
""")

    print(json.dumps({"max_trials_per_block": expected, "blocks": report,
                      "included_sets": sorted(keep) if keep else "all",
                      "excluded_clips": dict(dropped),
                      "base_url": args.base_url, "out_dir": str(out)}, indent=1))
    if dropped:
        print(f"\nPart 1 excludes {dict(dropped)} — see the README for why each went. "
              f"The rendered clips and their file ids are untouched; they are simply "
              f"not looped over.")


if __name__ == "__main__":
    main()
