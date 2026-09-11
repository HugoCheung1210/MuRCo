#!/usr/bin/env python
"""Freeze the context set the GRPO policy is trained and evaluated on.

Why freeze it, and why now. The training prompts are the one experimental variable that
is easiest to fiddle with under time pressure and hardest to defend afterwards. Fixing
them before any GPU exists means the train/held-out split cannot be chosen to flatter a
result, and the manifest is a diffable artefact with a seed.

Three decisions, all recorded in the manifest:

1. **Sources are the ethics-screened survivors only.** The 27 tracks excluded on audition
   (lyrics or potentially distressing) and the 4 withdrawn later (harsh) are dropped, even
   though nothing here is played to anyone yet. If a fine-tuned checkpoint later goes into
   a listening check, its continuations grow out of material already cleared for
   participants, so the ethics amendment does not need to revisit the sources.

2. **Held-out contexts are held out by TRACK**, matching the leave-one-track-out protocol
   used for the diagnosis experiment. A continuation of a different part of a training
   track is not a held-out test.

3. **Genre-stratified**, because R abstains on non-percussive audio (its beat-strength gate is
   computed from the context), so a context set skewed towards ambient would quietly turn
   C into a three-dimensional metric.

Each context is a window of real audio; the policy continues from its end.

Usage (DSP env):
    python scripts/rl/make_contexts.py
    python scripts/rl/make_contexts.py --n-train 40 --n-heldout 10 --context-s 10
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

RAW = ROOT / "raw_audio"
EXCLUDED = ROOT / "results/mos/source_review/excluded_sources.json"
WITHDRAWN = ROOT / "results/mos/withdrawn_sources.json"


def screened_sources() -> tuple[list[str], dict[str, list[str]]]:
    """Track ids that survived both ethics auditions, plus what was dropped and why."""
    all_tracks = sorted(p.stem for p in RAW.glob("*.mp3"))
    dropped: dict[str, list[str]] = {}

    excl = json.loads(EXCLUDED.read_text())
    dropped["audition_excluded"] = sorted(excl["excluded"])
    dropped["withdrawn_harsh"] = sorted(json.loads(WITHDRAWN.read_text())["withdrawn"])

    bad = set(dropped["audition_excluded"]) | set(dropped["withdrawn_harsh"])
    kept = [t for t in all_tracks if t not in bad]
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-heldout", type=int, default=10)
    ap.add_argument("--context-s", type=float, default=10.0,
                    help="seconds of context the policy continues from")
    ap.add_argument("--generate-s", type=float, default=10.0,
                    help="seconds the policy generates")
    ap.add_argument("--start-s", type=float, default=5.0,
                    help="offset into the 30 s clip; avoids fade-ins")
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--out", default=str(ROOT / "results/rl/contexts.json"))
    a = ap.parse_args()

    kept, dropped = screened_sources()
    by_genre: dict[str, list[str]] = defaultdict(list)
    for t in kept:
        by_genre[t.rsplit("_", 1)[0]].append(t)

    rng = random.Random(a.seed)
    need = a.n_train + a.n_heldout
    genres = sorted(by_genre)
    for g in genres:
        rng.shuffle(by_genre[g])

    # Round-robin across genres so the split is stratified by construction, and assign
    # whole tracks to exactly one side of the split.
    picked: list[str] = []
    i = 0
    while len(picked) < need:
        added = False
        for g in genres:
            if i < len(by_genre[g]):
                picked.append(by_genre[g][i])
                added = True
                if len(picked) == need:
                    break
        if not added:
            break
        i += 1

    if len(picked) < need:
        raise SystemExit(f"only {len(picked)} screened tracks available, need {need}")

    # Deal alternately so both sides stay genre-balanced.
    heldout = picked[::(need // a.n_heldout)][:a.n_heldout]
    train = [t for t in picked if t not in set(heldout)][:a.n_train]

    def rows(tracks: list[str], split: str) -> list[dict]:
        return [{"context_id": f"{split}_{i:02d}", "track": t,
                 "genre": t.rsplit("_", 1)[0], "audio": f"raw_audio/{t}.mp3",
                 "context_start_s": a.start_s,
                 "context_end_s": a.start_s + a.context_s,
                 "generate_s": a.generate_s, "split": split}
                for i, t in enumerate(sorted(tracks))]

    out = {
        "meta": {
            "purpose": "frozen context set for GRPO continuation training",
            "created": "2026-08-04", "seed": a.seed,
            "context_s": a.context_s, "generate_s": a.generate_s,
            "sources": "raw_audio/ (FMA-Medium 30 s clips)",
            "screening": "ethics-audition survivors only; see dropped{}",
            "split_unit": "track (no track appears in both splits)",
            "n_available_after_screening": len(kept),
            "genres": {g: len(v) for g, v in sorted(by_genre.items())},
        },
        "dropped": dropped,
        "contexts": rows(train, "train") + rows(heldout, "heldout"),
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    tr = [c for c in out["contexts"] if c["split"] == "train"]
    ho = [c for c in out["contexts"] if c["split"] == "heldout"]
    print(f"screened sources: {len(kept)} of 90 "
          f"({len(dropped['audition_excluded'])} excluded, "
          f"{len(dropped['withdrawn_harsh'])} withdrawn)")
    print(f"train {len(tr)}, held-out {len(ho)}, no track shared\n")
    for split, rows_ in (("train", tr), ("heldout", ho)):
        counts: dict[str, int] = defaultdict(int)
        for c in rows_:
            counts[c["genre"]] += 1
        print(f"  {split:<8} " + "  ".join(f"{g} {n}" for g, n in sorted(counts.items())))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
