"""
DRIFT eval from iterative-editing outputs. Anchored on SHARED seeds.

Confirmed setup:
  seeds:  outputs/seeds/s01.wav ... s10.wav   (originals = iteration 0, 30s each)
  ACE:    outputs/E2_iterative/sNN_iterK.wav
  MGen:   outputs_musicgen/continuation/sNN_iterK.wav
  SAO:    outputs_stableaudio/editing/sNN_iterK.wav
  seeds are s01..s10 (1-indexed), K = 1..8.

Notes baked in:
  * Manifest paths are Colab/Drive paths -> IGNORED. We use local seed files.
  * 30s clips -> TRIM to TRIM_S so A+B concat stays under MF's 30s window.
  * A = seed (true original), B = iter_k. Drift measured from iteration 0.

Usage:
    import build_drift as bd, mf_eval as ev
    ev.set_backend(mf); ev.load_clap("laion/larger_clap_music")
    for path,name in [("outputs/E2_iterative","ACE-Step"),
                      ("outputs_musicgen/continuation","MusicGen"),
                      ("outputs_stableaudio/editing","StableAudio")]:
        seqs = bd.load_model(path, seeds_dir="outputs/seeds")
        bd.run_drift(seqs, model_name=name)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

import os, glob, re, numpy as np, soundfile as sf, librosa

KS = [1, 2, 4, 8]
SEED_IDS = [f"s{i:02d}" for i in range(1, 11)]   # s01..s10
TRIM_S = 12.0                                     # trim so A+B concat < 30s
SR = 44100

def _trim(path, out):
    """Load, trim to TRIM_S, RMS-normalize, save. Returns out or None."""
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=TRIM_S)
    except Exception as e:
        print(f"  skip {os.path.basename(path)}: {e}"); return None
    if len(y) < int(TRIM_S * SR * 0.5):
        return None
    y = (y / (np.max(np.abs(y)) + 1e-8) * 0.95).astype("float32")
    sf.write(out, y, SR)
    return out

def _seed_path(seeds_dir, sid):
    for ext in (".wav", ".mp3", ".flac"):
        p = os.path.join(seeds_dir, sid + ext)
        if os.path.exists(p): return p
    return None

def load_model(dirpath, seeds_dir="outputs/seeds", trim_dir="drift_trim"):
    os.makedirs(trim_dir, exist_ok=True)
    seqs = []
    for sid in SEED_IDS:
        sp = _seed_path(seeds_dir, sid)
        if not sp:
            print(f"  {sid}: seed file not found in {seeds_dir}, skipped"); continue
        A = _trim(sp, os.path.join(trim_dir, f"{sid}_seed.wav"))
        if A is None:
            print(f"  {sid}: seed unreadable, skipped"); continue
        iters = {}
        for k in KS:
            ip = os.path.join(dirpath, f"{sid}_iter{k:02d}.wav")
            if os.path.exists(ip):
                t = _trim(ip, os.path.join(trim_dir, f"{sid}_{os.path.basename(dirpath)}_iter{k:02d}.wav"))
                if t: iters[k] = t
        if iters:
            seqs.append({"sid": sid, "A": A, "iters": iters})
    print(f"{dirpath}: {len(seqs)}/10 seeds loaded, ks per seed = "
          f"{[len(s['iters']) for s in seqs]}")
    return seqs

def run_drift(seqs, model_name="model", gap_s=0.0):
    import mf_eval as ev
    per_k = {k: {"pos": [], "neg": [], "comb": [], "clap": []} for k in KS}
    for sq in seqs:
        A = sq["A"]
        for k, b in sq["iters"].items():
            comb, pos, neg = ev.mf_balanced(A, b, gap_s=gap_s)
            per_k[k]["pos"].append(pos); per_k[k]["neg"].append(neg); per_k[k]["comb"].append(comb)
            if ev._clap is not None:
                per_k[k]["clap"].append(ev.clap_sim(A, b))
    def m(x): return (np.mean(x), np.std(x)) if x else (float("nan"), 0.0)
    print(f"\n===== DRIFT: {model_name}  (A=seed/original, B=iter_k) =====")
    print(f"{'iter':6s}{'n':>4s}{'MF pos':>13s}{'MF neg':>13s}{'MF comb':>13s}{'CLAP':>13s}")
    for k in KS:
        pm,ps=m(per_k[k]['pos']);nm,ns=m(per_k[k]['neg']);cm,cs=m(per_k[k]['comb']);km,ks=m(per_k[k]['clap'])
        print(f"k={k:<4d}{len(per_k[k]['neg']):>4d}{pm:6.3f}±{ps:4.2f}{nm:7.3f}±{ns:4.2f}"
              f"{cm:7.3f}±{cs:4.2f}{km:7.3f}±{ks:4.2f}")
    def slope(metric, sign):
        v = [np.mean(per_k[k][metric]) for k in KS if per_k[k][metric]]
        return sign * (v[-1] - v[0]) if len(v) >= 2 else float("nan")
    print("\n---- drift tracked? (k=8 minus k=1; positive = detects growing incoherence) ----")
    print(f"  MF neg  : {slope('neg', +1):+.3f}")
    print(f"  MF comb : {slope('comb', -1):+.3f}")
    print(f"  MF pos  : {slope('pos', -1):+.3f}")
    print(f"  CLAP    : {slope('clap', -1):+.3f}   <- ~0 => CLAP blind to same-song drift")
    return per_k
