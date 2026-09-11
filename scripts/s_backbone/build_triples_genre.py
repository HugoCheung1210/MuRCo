"""
Genre-aware triple builder. Splits the 'diff' negative into:
    diff_same  = different track, SAME genre   (the HARD, on-thesis case)
    diff_cross = different track, DIFFERENT genre (easier negative)

Genre is inferred from a filename prefix before the first '_' or '-'.
Name files like:  piano_01.mp3, piano_02.mp3, metal_01.mp3, edm_03.mp3 ...
Tracks sharing a prefix are treated as the same genre.

Usage:
    import build_triples_genre as btg
    triples = btg.build("music_mp3/", out_dir="clips/", seg=10, start=20)
    # triples now have keys: a, cont, diff_same, diff_cross, noise
    import mf_eval_genre as evg   # see note below
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

import argparse
import os, glob, random, numpy as np, soundfile as sf, librosa
SR = 44100

def _genre(path):
    name = os.path.splitext(os.path.basename(path))[0].lower()
    for sep in ("_", "-", " "):
        if sep in name:
            return name.split(sep)[0]
    return name

def _cut(path, start, seg, out):
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, offset=start, duration=seg)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}"); return None
    if len(y) < int(seg*SR*0.8): return None
    y = (y / (np.max(np.abs(y))+1e-8) * 0.95).astype("float32")
    sf.write(out, y, SR); return out

def build(mp3_dir, out_dir="clips", seg=10, start=20.0, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    tracks = sorted(glob.glob(os.path.join(mp3_dir,"*.mp3"))+glob.glob(os.path.join(mp3_dir,"*.wav")))
    noise = os.path.join(out_dir,"noise.wav")
    sf.write(noise,(np.random.randn(int(seg*SR))*0.1).astype("float32"),SR)

    cut = {}
    genres = {}
    for i,t in enumerate(tracks):
        base=f"t{i:02d}"
        a   =_cut(t,start,     seg,os.path.join(out_dir,base+"_A.wav"))
        cont=_cut(t,start+seg, seg,os.path.join(out_dir,base+"_cont.wav"))
        if a and cont:
            cut[i]={"a":a,"cont":cont}; genres[i]=_genre(t)
    ids=list(cut.keys())
    by_genre={}
    for i in ids: by_genre.setdefault(genres[i],[]).append(i)
    print("genre counts:", {g:len(v) for g,v in by_genre.items()})

    rng=random.Random(seed); triples=[]; skipped=0
    for i in ids:
        g=genres[i]
        same_pool =[k for k in by_genre.get(g,[]) if k!=i]                 # same genre
        cross_pool=[k for k in ids if genres[k]!=g]                        # different genre
        if not same_pool or not cross_pool:
            skipped+=1; continue   # need both to form a full triple
        triples.append({
            "a":         cut[i]["a"],
            "cont":      cut[i]["cont"],
            "diff_same": cut[rng.choice(same_pool)]["a"],
            "diff_cross":cut[rng.choice(cross_pool)]["a"],
            "noise":     noise,
        })
    print(f"built {len(triples)} triples ({skipped} skipped: needed >=2 tracks/genre AND another genre)")
    if skipped: print("  -> add at least 2 tracks per genre for the same-genre case.")
    return triples

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="build genre-aware triples for MF training")
    parser.add_argument("mp3_dir", help="directory of mp3/wav files")
    parser.add_argument("--out_dir", default="clips_genre", help="where to write cut clips")
    parser.add_argument("--seg", type=float, default=10, help="segment length in seconds (keep <=15 so A+B concat stays <30s for MF)")
    parser.add_argument("--start", type=float, default=20.0, help="where to begin cutting (skip intros/silence); A=[start,start+seg], cont=[start+seg, start+2*seg]")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
    args = parser.parse_args()
    build(args.mp3_dir, out_dir=args.out_dir, seg=args.seg, start=args.start, seed=args.seed)