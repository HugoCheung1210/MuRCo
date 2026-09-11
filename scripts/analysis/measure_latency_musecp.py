#!/usr/bin/env python
"""Wall-clock cost of MuseCPEval's five facets, next to MuRCo's H/T/R on the same machine.

Absolute milliseconds are machine- and load-dependent, and badly so: measure_latency.py
records 901-2136 ms for H+T+R on a loaded 2017 laptop against a stable 311-334 ms on an
idle Xeon Gold 6430.  So the number to carry between machines is the RATIO of the two
metrics measured in the same session, which this script reports.  Run it on an idle
machine if the absolute column is going to be quoted anywhere.

MuseCPEval is pure CPU.  It needs librosa, numpy and (for the structure facet) msaf, and
no GPU at any point, so a GPU box helps it only through having better cores.

Unlike measure_latency.py this cannot preload audio, because MuseCPEval's entry points
take file paths and read the files themselves.  File reading is therefore inside the
MuseCPEval timings and outside MuRCo's, which flatters MuRCo; the read is a few
milliseconds against facet costs that run to seconds, and the ``--include-io`` figure for
MuRCo quantifies it.

    python scripts/analysis/measure_latency_musecp.py -n 20 --repeats 3
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / "CLAUDE.md").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

FACETS = ["harmony", "rhythm", "structure", "melody", "timbre"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="perturbations/pairs_manifest.json")
    ap.add_argument("-n", type=int, default=20, help="pairs to time")
    ap.add_argument("--repeats", type=int, default=3,
                    help="passes; the fastest is kept, since contention only adds time")
    ap.add_argument("--out", default="results/musecp/latency.json")
    a = ap.parse_args()

    import musecpeval as mc

    blob = json.loads((ROOT / a.manifest).read_text())
    pairs = blob["pairs"] if isinstance(blob, dict) else blob
    pairs = [p for p in pairs if p["perturbation"] != "control"][:a.n]
    jobs = [(str(ROOT / p["ref_path"]), str(ROOT / p["cand_path"])) for p in pairs]
    secs = float(pairs[0].get("seconds", 10.0))
    print(f"{len(jobs)} pairs, {secs:.0f}s per segment, single process, "
          f"{a.repeats} passes\n")

    fns = {"harmony": mc.harmony_score, "rhythm": mc.rhythm_score,
           "structure": mc.structural_score, "melody": mc.melody_score,
           "timbre": mc.timbre_score}

    best = {f: None for f in FACETS}
    for rep in range(a.repeats):
        this = {}
        for f in FACETS:
            t0 = time.perf_counter()
            for ref, est in jobs:
                try:
                    fns[f](ref, est)
                except Exception as e:                       # noqa: BLE001
                    print(f"    {f} failed on a pair: {type(e).__name__}: {e}")
                    break
            this[f] = (time.perf_counter() - t0) / len(jobs) * 1e3
        tot = sum(this.values())
        print(f"  pass {rep + 1}/{a.repeats}: total {tot:8.0f} ms/pair   " +
              "  ".join(f"{f[:4]} {this[f]:.0f}" for f in FACETS), flush=True)
        for f in FACETS:
            if best[f] is None or this[f] < best[f]:
                best[f] = this[f]

    total = sum(best.values())
    print(f"\nfastest pass per facet, ms per pair ({secs:.0f}s segments):")
    for f in FACETS:
        print(f"  {f:>10}  {best[f]:8.0f} ms   {100 * best[f] / total:5.1f}%")
    print(f"  {'TOTAL':>10}  {total:8.0f} ms")
    rt = secs / (total / 1e3)
    print(f"\nreal-time factor: {rt:.1f}x " + ("faster" if rt >= 1 else "SLOWER")
          + " than the audio plays")

    out = {"n_pairs": len(jobs), "segment_seconds": secs, "repeats": a.repeats,
           "ms_per_pair": {f: round(best[f], 1) for f in FACETS},
           "total_ms_per_pair": round(total, 1),
           "share_of_cost": {f: round(best[f] / total, 4) for f in FACETS},
           "realtime_factor": round(rt, 2),
           "note": "single process, fastest of the passes; file reading is inside these "
                   "timings because MuseCPEval's entry points take paths. Absolute ms are "
                   "machine- and load-dependent, so compare against MuRCo measured in the "
                   "same session, not against the published 310 ms."}
    dest = ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("\nwrote", dest.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
