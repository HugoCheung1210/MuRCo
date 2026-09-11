#!/usr/bin/env python
"""Render the planned MOS pairs to listening files. DSP env.

Each trial is ONE audio file: 6 s of A | 1 s silence | 6 s of B = 13 s, which is
what the approved Participant Information Sheet tells participants they will hear
("around 50 short audio clips, each about 13 seconds").

Level handling mirrors mf_probe.concat_clips (the S scorer) so humans and the
metric judge the same relationship:
  - B is RMS-matched to A  -> the A:B level ratio carries no coherence cue,
  - the WHOLE concat is then peak-safe scaled (never hard-clipped: clipping is
    distortion, and distortion is a condition this study measures),
  - and finally the whole concat is normalised to a common target RMS so no
    trial is conspicuously louder than the next (protocol §3 "loudness-matched").

NOTE the one deliberate difference from the S cache: S was scored on 8 s windows
(mf_probe secs=8 -> 17 s concat), these are 6 s windows to match the approved
13 s. Re-score S at --secs 6 on the MOS subset for an exact-match sensitivity
check when the GPU is free.

Filenames are OPAQUE (stim_0001.mp3). Qualtrics exposes media URLs to the
participant, so a name like "pitch_shift_4.mp3" would leak the condition and
invite demand characteristics. The mapping lives in stimulus_index.csv, which
stays with the researcher.

Usage:
  python scripts/mos/render_mos_stimuli.py                     # all renderable rows
  python scripts/mos/render_mos_stimuli.py --set matrix        # skip the AI block
  python scripts/mos/render_mos_stimuli.py --format wav --dry-run
Outputs: results/mos/stimuli/*.mp3 + results/mos/stimulus_index.csv
"""
import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
SR = 48000              # source rate; do NOT downsample (would mask the 8 kHz lowpass)
SEG_S = 6.0             # per-segment seconds -> 6 + 1 + 6 = 13 s
GAP_S = 1.0             # the load-bearing gap, same as the S protocol
TARGET_RMS = 0.1        # ~-20 dBFS, comfortable and consistent across trials
EPS = 1e-8


def build_concat(path_a, path_b, seg_s, gap_s, attenuate=1.0):
    """A | gap | B as one mono array at SR, plus a small provenance dict."""
    a, _ = librosa.load(path_a, sr=SR, mono=True, duration=seg_s)
    b, _ = librosa.load(path_b, sr=SR, mono=True, duration=seg_s)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ra = np.sqrt(np.mean(a ** 2)) + EPS
    rb = np.sqrt(np.mean(b ** 2)) + EPS
    b = b * (ra / rb)                                   # B matched to A (mf_probe)
    gap = np.zeros(int(gap_s * SR), dtype=a.dtype)
    cat = np.concatenate([a, gap, b]).astype("float32")
    peak = float(np.max(np.abs(cat)))
    if peak > 1.0:                                      # peak-safe, never clip
        cat = cat / peak
    rms = float(np.sqrt(np.mean(cat ** 2))) + EPS
    gain = min(TARGET_RMS / rms, 0.99 / (float(np.max(np.abs(cat))) + EPS))
    # Sources the audition marked "too loud" are usable but uncomfortable at the common
    # target level, so they get an extra cut. Applied to the WHOLE concat, so the A:B
    # relationship the rater judges is untouched -- only the across-trial level changes,
    # which a within-trial relational judgement does not depend on. Baked into the audio
    # rather than set on the player, because Qualtrics gives no per-clip volume control.
    cat = (cat * gain * attenuate).astype("float32")
    return cat, {"seg_samples": int(n), "gain": round(gain, 4),
                 "attenuate": attenuate,
                 "peak": round(float(np.max(np.abs(cat))), 4)}


def encode(cat, out_path, fmt):
    if fmt == "wav":
        sf.write(out_path, cat, SR)
        return
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, cat, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(out_path)],
                       check=True)


def sha256(path, n=16):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


def attention_role(row):
    """Catch trials with a known-correct answer (protocol §4)."""
    if row["perturbation"] == "control":
        return "expect_high"          # identical continuation -> should score ~5
    if row["perturbation"] == "style_swap":
        return "expect_low"           # different piece -> should score ~1
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="results/mos/stimuli_plan.json")
    ap.add_argument("--out-dir", default="results/mos/stimuli")
    ap.add_argument("--index", default="results/mos/stimulus_index.csv")
    ap.add_argument("--format", choices=("mp3", "wav"), default="mp3")
    ap.add_argument("--seg-s", type=float, default=SEG_S)
    ap.add_argument("--gap-s", type=float, default=GAP_S)
    ap.add_argument("--set", choices=("all", "matrix", "ai_rerank"), default="all")
    ap.add_argument("--attenuate-list", default=None,
                    help="excluded_sources.json from review_sources.py --apply; its "
                         "'attenuate' sources are rendered quieter")
    ap.add_argument("--attenuation", type=float, default=0.75,
                    help="gain applied to flagged-loud sources (0.75 = -2.5 dB)")
    ap.add_argument("--attenuate-map", default=None,
                    help="attenuation_map.json from review_levels.py --apply: a PER-SOURCE "
                         "gain chosen by ear. Overrides --attenuate-list/--attenuation, "
                         "because one global cut over-quiets the marginal tracks and "
                         "under-treats the painful ones.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what is renderable/missing, write nothing")
    args = ap.parse_args()

    plan = json.load(open(ROOT / args.plan))
    rows = plan["stimuli"]
    loud, att_map = set(), {}
    if args.attenuate_map:
        att_map = {k: float(v) for k, v in
                   json.load(open(ROOT / args.attenuate_map))["attenuation"].items()}
    elif args.attenuate_list:
        loud = set(json.load(open(ROOT / args.attenuate_list)).get("attenuate", []))
    if args.set != "all":
        rows = [r for r in rows if r["stimulus_set"] == args.set]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stable opaque ids: sorted by pair_id so a re-run reproduces the same names.
    rows = sorted(rows, key=lambda r: r["pair_id"])
    index, missing = [], []
    for i, r in enumerate(rows, 1):
        a, b = ROOT / r["ref_path"], ROOT / r["cand_path"]
        if not (a.exists() and b.exists()):
            missing.append({"pair_id": r["pair_id"], "stimulus_set": r["stimulus_set"],
                            "absent": [str(p.relative_to(ROOT))
                                       for p in (a, b) if not p.exists()]})
            continue
        stim_id = f"stim_{i:04d}"
        fname = f"{stim_id}.{args.format}"
        att = att_map.get(r["source_id"], args.attenuation
                          if r["source_id"] in loud else 1.0)
        if not args.dry_run:
            cat, prov = build_concat(str(a), str(b), args.seg_s, args.gap_s, att)
            encode(cat, out_dir / fname, args.format)
        index.append({
            "stimulus_id": stim_id, "file": fname,
            "pair_id": r["pair_id"], "block": r["block"],
            "stimulus_set": r["stimulus_set"], "condition": r["condition"],
            "perturbation": r["perturbation"], "genre": r["genre"],
            "source_id": r["source_id"], "attention_check": attention_role(r),
            "attenuated": att,
            "duration_s": round(2 * args.seg_s + args.gap_s, 2),
            "C_uniform": r.get("C_uniform", ""),
            "sha256_16": "" if args.dry_run else sha256(out_dir / fname),
        })

    if not args.dry_run and index:
        with open(ROOT / args.index, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
            w.writeheader()
            w.writerows(index)

    by_set = Counter(r["stimulus_set"] for r in index)
    print(json.dumps({
        "planned": len(rows), "rendered": len(index), "by_set": dict(by_set),
        "attenuated_clips": sum(1 for r in index if r["attenuated"] != 1.0),
        "attenuation": args.attenuation,
        "attention_checks": dict(Counter(r["attention_check"] for r in index
                                         if r["attention_check"])),
        "missing_audio": len(missing),
        "missing_by_set": dict(Counter(m["stimulus_set"] for m in missing)),
        "missing_example": missing[:2],
        "seg_s": args.seg_s, "gap_s": args.gap_s, "sr": SR,
        "format": args.format, "dry_run": args.dry_run,
    }, indent=1))
    if missing:
        print(f"\n{len(missing)} pair(s) have no local audio — see PROJECT_STATE §9.1 "
              f"(AI-block windows live on the GPU box).")


if __name__ == "__main__":
    main()
