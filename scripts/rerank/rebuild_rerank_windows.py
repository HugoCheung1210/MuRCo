#!/usr/bin/env python
"""Rebuild the rerank B-windows locally from the full-length candidate clips. DSP env.

`results/rerank{,_ace}/audio/` is produced on the GPU box by rerank_bestofn.py's
generate step and never syncs back, but the full 30 s candidates DO live here in
`candidates/`. The B-window is a pure function of a candidate clip:

    librosa.load(cand, sr=48000, mono=True, offset=4.0, duration=6.0)
    y = y / (max|y| + 1e-8) * 0.95

which is rerank_bestofn.window() verbatim — deterministic, so the rebuild is
byte-identical to what the S cache was scored on. Only the seed A-windows
(`sNN__A.wav`, 30 files) must be copied from the GPU box, because the seeds
themselves (`outputs/seeds/`) are not local.

Usage:
  python scripts/rerank/rebuild_rerank_windows.py --check     # what's present/missing
  python scripts/rerank/rebuild_rerank_windows.py
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
SR = 48000
REGION = (4.0, 6.0)          # (start_s, dur_s) — rerank_bestofn.REGION
PEAK = 0.95


def window(src_wav, dst_wav):
    y, _ = librosa.load(str(src_wav), sr=SR, mono=True,
                        offset=REGION[0], duration=REGION[1])
    y = (y / (np.max(np.abs(y)) + 1e-8) * PEAK).astype("float32")
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst_wav), y, SR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="results/mos/stimuli_plan.json")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    ap.add_argument("--force", action="store_true", help="rebuild even if present")
    args = ap.parse_args()

    rows = [r for r in json.load(open(ROOT / args.plan))["stimuli"]
            if r["stimulus_set"] == "ai_rerank"]
    built, skipped, need_a, no_source = [], [], [], []

    for b in sorted({r["cand_path"] for r in rows}):
        dst = ROOT / b
        stem = dst.name.replace("__B.wav", "")           # sNN_candKK
        src = ROOT / b.replace("/audio", "/candidates").replace(dst.name, stem + ".wav")
        if dst.exists() and not args.force:
            skipped.append(b)
        elif not src.exists():
            no_source.append(b)
        else:
            if not args.check:
                window(src, dst)
            built.append(b)

    for a in sorted({r["ref_path"] for r in rows}):
        if not (ROOT / a).exists():
            need_a.append(a)

    print(json.dumps({
        "B_rebuilt" if not args.check else "B_rebuildable": len(built),
        "B_already_present": len(skipped),
        "B_no_local_candidate": no_source,
        "A_still_missing_from_gpu_box": len(need_a),
        "region_s": list(REGION), "sr": SR, "peak_norm": PEAK,
    }, indent=1))
    if need_a:
        print("\nCopy these from the GPU box (same relative paths):")
        for a in need_a:
            print("  " + a)


if __name__ == "__main__":
    main()
