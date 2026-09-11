"""
Build coherence-test triples from a folder of mp3 tracks.

For each track it cuts two consecutive segments:
    A    = [start .. start+SEG]      (e.g. 0-10s, skipping intros)
    cont = [start+SEG .. start+2*SEG] (the TRUE continuation -> coherent pair)
Then pairs each A with a DIFFERENT track's segment as the incoherent 'diff' case,
and a shared white-noise clip as the trivial 'noise' case.

Usage:
    import build_triples as bt
    triples = bt.build("music_mp3/", out_dir="clips/", seg=10, start=20, n_diff_genres=True)
    import mf_eval as ev; ev.run_batch(triples)

Tip: put MAXIMALLY different tracks in the folder (piano, metal, EDM, strings,
reggae...) so the 'diff' pairs are unambiguously different pieces.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

import os, glob, random, numpy as np, soundfile as sf, librosa

SR = 44100

def _cut(path, start, seg, out):
    """Load mp3, cut [start, start+seg]s, RMS-normalize, save wav. Returns path or None."""
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, offset=start, duration=seg)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}"); return None
    if len(y) < int(seg * SR * 0.8):          # too short (track ended)
        return None
    peak = np.max(np.abs(y)) + 1e-8
    y = (y / peak * 0.95).astype("float32")
    sf.write(out, y, SR)
    return out

def build(mp3_dir, out_dir="clips", seg=10, start=20.0, seed=0):
    """
    seg   : segment length in seconds (keep <=15 so A+B concat stays <30s for MF)
    start : where to begin cutting (skip intros/silence); A=[start,start+seg],
            cont=[start+seg, start+2*seg]
    """
    os.makedirs(out_dir, exist_ok=True)
    tracks = sorted(glob.glob(os.path.join(mp3_dir, "*.mp3")) +
                    glob.glob(os.path.join(mp3_dir, "*.wav")))
    if len(tracks) < 2:
        raise SystemExit(f"need >=2 tracks in {mp3_dir}, found {len(tracks)}")
    print(f"{len(tracks)} tracks found")

    # shared noise clip
    noise = os.path.join(out_dir, "noise.wav")
    sf.write(noise, (np.random.randn(int(seg*SR))*0.1).astype("float32"), SR)

    # cut A and cont for every usable track
    cut = {}
    for i, t in enumerate(tracks):
        base = f"t{i:02d}"
        a    = _cut(t, start,        seg, os.path.join(out_dir, base+"_A.wav"))
        cont = _cut(t, start+seg,    seg, os.path.join(out_dir, base+"_cont.wav"))
        if a and cont:
            cut[i] = {"a": a, "cont": cont}
        else:
            print(f"  {base}: track too short for two {seg}s segments, skipped")
    ids = list(cut.keys())
    if len(ids) < 2:
        raise SystemExit("not enough usable tracks after cutting")

    # build triples: each A paired with a DIFFERENT track's A as the 'diff' negative
    rng = random.Random(seed)
    triples = []
    for i in ids:
        j = rng.choice([k for k in ids if k != i])     # a different track
        triples.append({
            "a":     cut[i]["a"],
            "cont":  cut[i]["cont"],
            "diff":  cut[j]["a"],       # segment from an unrelated track
            "noise": noise,
        })
    print(f"built {len(triples)} triples")
    return triples




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mp3_dir", help="folder of mp3/wav tracks")
    parser.add_argument("--out_dir", default="clips", help="where to save cut clips")
    parser.add_argument("--seg", type=float, default=10.0, help="segment length in seconds")
    parser.add_argument("--start", type=float, default=20.0, help="where to start cutting (skip intros)")
    args = parser.parse_args()
    build(args.mp3_dir, out_dir=args.out_dir, seg=args.seg, start=args.start)