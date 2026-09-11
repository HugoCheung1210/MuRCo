#!/usr/bin/env python
"""Render the two unscored practice trials. DSP env.

Practice must demonstrate BOTH ends of the scale before the real trials start,
otherwise a rater calibrates on whatever they happen to hear first:

  practice_hi  source A | gap | the same 6 s again   -> unambiguous 5
  practice_lo  source B | gap | a window of source A -> unambiguous 1

The low anchor needs a second piece, which is why two sources are required
rather than one.

Sources must NOT be study sources -- all 63 audit-surviving sources are consumed
by the 190 stimuli, so these come from sources_selection/practice_candidates
(see select_practice_clips.py) and must have passed a human listen first.

Level handling is identical to render_mos_stimuli.py: B is RMS-matched to A, the
concat is peak-safe, then normalised to the same TARGET_RMS as every study
trial, so practice is not conspicuously louder or quieter than what follows.

The low-anchor pair deliberately reads a LATER window of the shared source, so a
rater does not hear the identical six seconds three times across two trials.

Usage:
  python scripts/mos/render_practice_stimuli.py --hi practice_cand_06 --lo-a practice_cand_05
Outputs: results/mos/practice/*.mp3 + practice_index.csv
"""
import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
SR, SEG_S, GAP_S, TARGET_RMS, EPS = 48000, 6.0, 1.0, 0.1, 1e-8


def load(path, offset):
    y, _ = librosa.load(path, sr=SR, mono=True, offset=offset, duration=SEG_S)
    return y


def build(path_a, path_b, off_a=0.0, off_b=0.0):
    a, b = load(path_a, off_a), load(path_b, off_b)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ra = np.sqrt(np.mean(a ** 2)) + EPS
    rb = np.sqrt(np.mean(b ** 2)) + EPS
    b = b * (ra / rb)
    cat = np.concatenate([a, np.zeros(int(GAP_S * SR), dtype=a.dtype), b]).astype("float32")
    peak = float(np.max(np.abs(cat)))
    if peak > 1.0:
        cat = cat / peak
    rms = float(np.sqrt(np.mean(cat ** 2))) + EPS
    gain = min(TARGET_RMS / rms, 0.99 / (float(np.max(np.abs(cat))) + EPS))
    cat = (cat * gain).astype("float32")
    return cat, {"seconds": round(len(cat) / SR, 2),
                 "rms": round(float(np.sqrt(np.mean(cat ** 2))), 4),
                 "peak": round(float(np.max(np.abs(cat))), 4)}


def encode(cat, out_path):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, cat, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(out_path)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dir", default="sources_selection/practice_candidates")
    ap.add_argument("--hi", required=True, help="candidate used for the unchanged control")
    ap.add_argument("--lo-a", required=True, help="the OTHER piece, used as segment A of the low anchor")
    ap.add_argument("--lo-b-offset", type=float, default=12.0,
                    help="seconds into --hi to take the low anchor's segment B from")
    ap.add_argument("--out-dir", default="results/mos/practice")
    args = ap.parse_args()

    cand = ROOT / args.cand_dir
    hi_src, lo_src = cand / f"{args.hi}.mp3", cand / f"{args.lo_a}.mp3"
    for p in (hi_src, lo_src):
        if not p.exists():
            raise SystemExit(f"missing source: {p}")

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    specs = [
        ("practice_hi", hi_src, hi_src, 0.0, 0.0, "control: identical segment repeated"),
        ("practice_lo", lo_src, hi_src, 0.0, args.lo_b_offset, "different piece substituted"),
    ]
    rows = []
    for name, pa, pb, oa, ob in [(s[0], s[1], s[2], s[3], s[4]) for s in specs]:
        note = dict(specs and {s[0]: s[5] for s in specs})[name]
        cat, meta = build(pa, pb, oa, ob)
        dest = out / f"{name}.mp3"
        encode(cat, dest)
        rows.append({"stimulus_id": name, "file": dest.name,
                     "expected_rating": 5 if name.endswith("hi") else 1,
                     "segment_a": pa.stem, "offset_a_s": oa,
                     "segment_b": pb.stem, "offset_b_s": ob,
                     "note": note, **meta})
        print(f"  {dest.name}: {meta['seconds']}s  rms={meta['rms']}  peak={meta['peak']}  ({note})")

    index = out / "practice_index.csv"
    with index.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {index.relative_to(ROOT)}")
    print("Practice ratings are not analysed; these files are never study stimuli.")


if __name__ == "__main__":
    main()
