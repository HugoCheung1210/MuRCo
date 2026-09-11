#!/usr/bin/env python3
"""E-D listening aid -- which candidate does each metric pick, and gather them to hear.

Reads the SAME caches `rerank_bestofn.py analyze` uses (no GPU, no re-scoring):
  results/rerank/pair_scores.csv            (H/T/R/S per candidate)
  results/rerank/baseline_scores_rerank.json (clap_htsat per candidate)
  results/rerank/rerank_index.json           ({seed: [pair_id,...]})
  results/rerank/rerank_manifest.json        (audio paths per pair_id)

For every seed it reports the candidate chosen by each ranker -- C, S, clap_htsat, a random
pick, and the C-loser (worst by C) -- prints a per-seed agreement table, writes
`results/rerank/listen_picks.csv`, and with --copy gathers the wavs into
`results/rerank/listen/<seed>/` with descriptive names so you can A/B each pick against the
original seed by ear.

    python scripts/rerank/rerank_listen.py            # table + listen_picks.csv only
    python scripts/rerank/rerank_listen.py --copy      # + copy the 6s edit-region windows
    python scripts/rerank/rerank_listen.py --copy --full   # + copy the full 30s candidates instead

Naming in listen/<seed>/:  _seed.wav (original), C-winner__candNN.wav, clap_htsat-winner__candNN.wav,
S-winner__candNN.wav, random__candNN.wav, C_loser__candNN.wav.  When two rankers pick the SAME
candidate you'll see the same candNN -- that IS the finding (they agree on that seed).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

EPS = 1e-6
RERANK_DIR = Path(os.environ.get("RERANK_DIR", "results/rerank"))   # match the run you're listening to
SEEDS_DIR = Path("outputs/seeds")


def _geo(vals) -> float:
    v = [float(np.clip(float(x), EPS, 1.0)) for x in vals]
    return float(np.prod(v) ** (1.0 / len(v)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copy", action="store_true", help="copy selected wavs to results/rerank/listen/")
    ap.add_argument("--full", action="store_true",
                    help="with --copy: copy full 30s candidates (default: the 6s edit-region windows)")
    args = ap.parse_args()

    pair_csv = RERANK_DIR / "pair_scores.csv"
    index_p = RERANK_DIR / "rerank_index.json"
    if not pair_csv.is_file() or not index_p.is_file():
        raise SystemExit(f"missing {pair_csv} or {index_p} -- run generate + the scorers + analyze first.")

    rows = {r["pair_id"]: r for r in csv.DictReader(pair_csv.open(newline="", encoding="utf-8"))}
    index = json.loads(index_p.read_text(encoding="utf-8"))
    clap = {}
    bp = RERANK_DIR / "baseline_scores_rerank.json"
    if bp.is_file():
        bd = json.loads(bp.read_text(encoding="utf-8"))
        clap = bd.get("scores", bd)
    paths = {}
    mp = RERANK_DIR / "rerank_manifest.json"
    if mp.is_file():
        for p in json.loads(mp.read_text(encoding="utf-8")).get("pairs", []):
            paths[p["pair_id"]] = p

    rng = np.random.default_rng(20260717)   # same seed as analyze -> identical random pick

    def C(pid):  return _geo([rows[pid][f"score_{d}"] for d in ("H", "T", "R", "S")])
    def S(pid):  return float(rows[pid]["score_S"])
    def CL(pid): return float(clap.get(pid, {}).get("clap_htsat", "nan"))
    def kk(pid): return pid.split("::cand")[-1]                # "03"

    rankers = {"C": C, "S": S, "clap_htsat": CL}
    hdr = f"{'seed':5} | {'C':>4} {'S':>4} {'clap':>4} {'rand':>4} | C-loser | agreement"
    print(hdr); print("-" * len(hdr))

    sel_rows = []
    n_disagree = 0
    for sid, pids in index.items():
        pids = [p for p in pids if p in rows]
        if len(pids) < 2:
            continue
        picks = {}
        for name, fn in rankers.items():
            vals = np.array([fn(p) for p in pids], float)
            picks[name] = pids[int(np.nanargmax(vals))]
        picks["random"] = pids[int(rng.integers(0, len(pids)))]
        picks["C_loser"] = pids[int(np.argmin([C(p) for p in pids]))]

        c_vs_clap = "C=clap" if picks["C"] == picks["clap_htsat"] else "C≠clap"
        n_disagree += (picks["C"] != picks["clap_htsat"])
        print(f"{sid:5} | {kk(picks['C']):>4} {kk(picks['S']):>4} {kk(picks['clap_htsat']):>4} "
              f"{kk(picks['random']):>4} | {kk(picks['C_loser']):>6} | {c_vs_clap}")
        sel_rows.append({"seed": sid, **{f"{n}_cand": kk(p) for n, p in picks.items()},
                         **{f"{n}_pair_id": p for n, p in picks.items()}})

        if args.copy:
            dst = RERANK_DIR / "listen" / sid
            dst.mkdir(parents=True, exist_ok=True)
            seed_wav = SEEDS_DIR / f"{sid}.wav"
            if seed_wav.is_file():
                shutil.copy(seed_wav, dst / "_seed.wav")               # full 30s original
            ref = Path(paths.get(picks["C"], {}).get("ref_path", ""))  # A = seed's 4-10s window
            if ref and ref.is_file():
                shutil.copy(ref, dst / "_reference__A.wav")            # A/B this vs each pick's window
            copied = {}
            for role, pid in picks.items():
                src = (RERANK_DIR / "candidates" / f"{sid}_cand{kk(pid)}.wav") if args.full \
                      else Path(paths.get(pid, {}).get("cand_path", ""))
                if src and Path(src).is_file():
                    shutil.copy(src, dst / f"{role}-winner__cand{kk(pid)}.wav"
                                if role in rankers else dst / f"{role}__cand{kk(pid)}.wav")
                    copied[role] = str(src)

    print(f"\nC and clap_htsat disagree on the winner in {n_disagree}/{len(sel_rows)} seeds.")
    if sel_rows:
        out_csv = RERANK_DIR / "listen_picks.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sel_rows[0].keys()))
            w.writeheader(); w.writerows(sel_rows)
        print(f"wrote {out_csv}" + (f" and audio to {RERANK_DIR/'listen'}/" if args.copy else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
