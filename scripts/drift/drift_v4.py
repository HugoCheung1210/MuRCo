"""
drift_v4 — de-bugged coherence drift eval.

Fixes every pipeline bug we found:
  * scores the EDITED REGION (not first 8s) via per-model (start,secs)
  * uses the 0-1 rating prompt with "answer with only the number" (the extraction
    that actually shows dynamic range: identical~0.8, drift~0.5, diff~0.0)
  * length-matches A and B, inserts a silence gap, keeps concat < MF window
  * robust numeric parse; averages several samples to reduce single-token noise

Per-model scoring regions. NB region=(start_s, DURATION_s); window = start..start+secs.
Roundtrip models (ACE/SAO) are scored at (4,6) -> 4-10s so the concat is 13s
(6+1s gap+6): fits MF's audio window (21s collapses the numeric scorer) AND keeps
SAO/ACE length-matched for the cross-model delta test.
  ACE-Step   : repaints 5-15s, SCORED region=(4,6) -> 4-10s (first ~5s of the edit;
               ace_protocol.py:21). The old "-> region=(5,10)" is the full edit, but
               21s concat is out of MF's window, so scoring uses (4,6).
  StableAudio: inpaints 5-8s (outputs_stableaudio_inpaint_5_8), region=(4,6) -> 4-10s.
  MusicGen   : iter_k IS the new segment; whole clip, region=(0,10), ceiling='self' (21s).

Usage:
    import drift_v4 as d4, mf_eval as ev
    d4.set_backend(mf); ev.load_clap("laion/larger_clap_music")   # clap optional
    d4.run("outputs/E2_iterative", "ACE-Step", region=(5,10),
           seeds_dir="outputs/seeds")
    d4.run("outputs_musicgen/continuation", "MusicGen", region=(0,10))
    d4.run("outputs_stableaudio/editing", "StableAudio", region=(0,10))
    # controls:
    d4.run("outputs/C1_roundtrip", "ACE-rt", region=(5,10))
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

import os, re, numpy as np, librosa, soundfile as sf

_mf = None
KS = [1, 2, 4, 8]
SEED_IDS = [f"s{i:02d}" for i in range(1, 11)]
SR = 44100
GAP_S = 1.0
N_SAMPLES = 3          # average N generations per pair (reduce single-token noise)

RATING_Q = ("This audio contains segment A, then a short silence, then segment B. "
            "Score their coherence in timbre, instrumentation, and style from 0 to 1, "
            "where 1 = clearly the same piece and 0 = completely unrelated. "
            "Answer with only the number.")

def set_backend(mf_module):
    global _mf
    _mf = mf_module
    assert _mf.model is not None, "call mf.load() first"

def _clip(path, start, secs, out):
    y,_ = librosa.load(path, sr=SR, mono=True, offset=start, duration=secs)
    if len(y) < int(SR*0.5): return None
    y = (y/(np.max(np.abs(y))+1e-8)*0.95).astype("float32")
    sf.write(out, y, SR); return out

def _concat(a, b, gap_s=GAP_S, out="_v4.wav"):
    ya,_ = librosa.load(a, sr=SR, mono=True)
    yb,_ = librosa.load(b, sr=SR, mono=True)
    n = min(len(ya), len(yb)); ya, yb = ya[:n], yb[:n]
    rb = np.sqrt(np.mean(yb**2))+1e-8; ra = np.sqrt(np.mean(ya**2))+1e-8
    yb = yb*(ra/rb)
    parts = [ya] + ([np.zeros(int(gap_s*SR), dtype=ya.dtype)] if gap_s>0 else []) + [yb]
    sf.write(out, np.clip(np.concatenate(parts).astype("float32"),-1,1), SR); return out

# --- S read-out (2026-08-22) -------------------------------------------------
# Setup-2 drift was scored with RATING_Q, a 0-1 number parsed from generated text.
# Chapter 3 defines S as the P(Yes) logit, so the two settings used different
# read-outs. ablation_and_negative_results.md 2.3 records why (the numeric scorer
# hedges on the matrix's ALIGNED pairs, the logit is grounded there), but the
# converse was never tested: does the logit work on generative edits?
#
# metric="mf" keeps the numeric read-out, so every existing result reproduces
# byte-for-byte. metric="mf_logit" takes the same clipped A/B and scores them with
# mf_probe's template bank, which is exactly the Chapter 3 protocol.
#
# A_FROM is load-bearing for CONTINUATION. mf_probe.concat_clips trims A to `secs`,
# and for MusicGen B starts where A ends, so scoring A's head would leave a hole at
# the junction. Set A_FROM="tail" for musicgen and "head" for sao/ace, where A and B
# are the same window of the same clip.
S_SECS = None          # seconds of A and of B to score; None = the region length
A_FROM = "head"        # "tail" for continuation, "head" for inpaint/repaint


def _rate_logit(a_clip, b_clip, secs):
    """Mean P(Yes) over mf_probe's four templates. Same protocol as Chapter 3."""
    supports_a_from = "a_from" in _mf.per_template_scores.__code__.co_varnames
    if not supports_a_from:
        # An older mf_probe on the GPU box would silently score A's head. That is
        # harmless for sao/ace and WRONG for continuation, so refuse rather than
        # quietly measure the wrong junction.
        if A_FROM != "head":
            raise RuntimeError(
                "mf_probe.per_template_scores has no a_from; copy the current "
                f"mf_probe.py to this box before scoring with A_FROM={A_FROM!r}")
        per = _mf.per_template_scores(a_clip, b_clip, secs=secs, gap_s=GAP_S)
    else:
        per = _mf.per_template_scores(a_clip, b_clip, secs=secs, gap_s=GAP_S,
                                      a_from=A_FROM)
    return float(np.mean(per)) if per else np.nan


def _rate_balanced(a_clip, b_clip, secs):
    """Polarity-balanced S: (pos + (1 - neg)) / 2 over mf_probe's two template pairs.

    Same cost as the positive-only bank (4 calls). This controls for acquiescence:
    a model with a Yes-prior inflates the positive templates AND the negative ones,
    and the (1 - neg) term cancels it. ablation 2.1 found this buys nothing on the
    MATRIX because separability is rank-based, and said so while noting that its
    value is in the absolute level -- "thresholds, drift ceiling". Drift is measured
    as a drop from an absolute ceiling, so it is the setting that note points at.
    """
    ab = _mf.concat_clips(a_clip, b_clip, secs=secs, gap_s=GAP_S, a_from=A_FROM)
    try:
        pos = float(np.mean([_mf._yes_prob(ab, q) for q in _mf.POS_TEMPLATES]))
        neg = float(np.mean([_mf._yes_prob(ab, q) for q in _mf.NEG_TEMPLATES]))
    finally:
        _mf._cleanup(ab)
    return (pos + (1.0 - neg)) / 2.0


def _rate_once(ab):
    inp = _mf._build_inputs(ab, RATING_Q)
    with _mf.torch.no_grad():
        out = _mf.model.generate(**inp, max_new_tokens=6)
    txt = _mf.processor.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    m = re.search(r"[01](?:\.\d+)?", txt)
    return float(m.group()) if m else np.nan


def _seed(seeds_dir, sid):
    for e in (".wav",".mp3",".flac"):
        p=os.path.join(seeds_dir, sid+e)
        if os.path.exists(p): return p
    return None

import os, re, numpy as np, librosa, soundfile as sf, torch   # <-- add torch

# ================= CLAP backend (optional) =================
_clap = None
_clap_proc = None

def load_clap(model_id="laion/clap-htsat-unfused"):
    global _clap, _clap_proc
    from transformers import ClapModel, ClapProcessor
    _clap = ClapModel.from_pretrained(model_id, use_safetensors=True).to("cuda").eval()
    _clap_proc = ClapProcessor.from_pretrained(model_id)
    print("CLAP loaded.")

def clap_embed(path, start=0.0, secs=None):
    if _clap is None:
        raise RuntimeError("CLAP not loaded — call load_clap() first")
    y, _ = librosa.load(path, sr=48000, mono=True, offset=start, duration=secs)
    if len(y) < int(48000 * 0.5):
        return None
    inp = _clap_proc(audio=y, sampling_rate=48000, return_tensors="pt")
    inp = {k: v.to("cuda") for k, v in inp.items()}
    with torch.no_grad():
        out = _clap.get_audio_features(**inp)
    if torch.is_tensor(out):
        e = out
    elif getattr(out, "pooler_output", None) is not None:
        e = out.pooler_output
    else:
        e = out.last_hidden_state.mean(1)               # keep 'out', not None
    return torch.nn.functional.normalize(e, dim=-1)

def clap_sim(path_a, path_b, start=0.0, secs=None):
    ea = clap_embed(path_a, start, secs)
    eb = clap_embed(path_b, start, secs)
    if ea is None or eb is None:
        return np.nan
    return float((ea @ eb.T).item())
# ===========================================================

# ================= SCS structural stand-in (CPU) =================
# Signal-structural coherence proxy (matches score_baselines.py): cosine agreement
# between the self-similarity matrices of A and B, built from chroma+MFCC frames
# resampled to a common length. Honors the (start,secs) region like the other
# metrics so the drift comparison is apples-to-apples with S. Documented STAND-IN,
# not the unreleased MusicWeaver SCS.
_SCS_SR = 22050
_SCS_FRAMES = 48

def _scs_feats(path, start, secs):
    y, sr = librosa.load(path, sr=_SCS_SR, mono=True, offset=start, duration=secs)
    if len(y) < int(_SCS_SR * 0.5):
        return None
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    f = np.vstack([chroma, mfcc])
    T = f.shape[1]
    if T < 2:
        return None
    xi = np.linspace(0, T - 1, _SCS_FRAMES)
    f = np.stack([np.interp(xi, np.arange(T), row) for row in f])
    mu = f.mean(axis=1, keepdims=True); sd = f.std(axis=1, keepdims=True) + 1e-8
    return (f - mu) / sd

def scs_sim(path_a, path_b, start=0.0, secs=None):
    fa, fb = _scs_feats(path_a, start, secs), _scs_feats(path_b, start, secs)
    if fa is None or fb is None:
        return np.nan
    na = fa / (np.linalg.norm(fa, axis=0, keepdims=True) + 1e-8)
    nb = fb / (np.linalg.norm(fb, axis=0, keepdims=True) + 1e-8)
    sa, sb = na.T @ na, nb.T @ nb
    iu = np.triu_indices(_SCS_FRAMES, k=1)
    ua, ub = sa[iu], sb[iu]
    cos = float(np.dot(ua, ub) / ((np.linalg.norm(ua) + 1e-8) * (np.linalg.norm(ub) + 1e-8)))
    return (1.0 + cos) / 2.0
# ===========================================================

def rate(path_a, path_b, start, secs, metric="mf"):
    if metric == "clap":
        return clap_sim(path_a, path_b, start, secs)
    if metric == "scs":
        return scs_sim(path_a, path_b, start, secs)
    a = _clip(path_a, start, secs, "_v4A.wav")
    b = _clip(path_b, start, secs, "_v4B.wav")
    if a is None or b is None: return np.nan
    if metric in ("mf_logit", "mf_balanced"):
        n = S_SECS if S_SECS is not None else secs
        return (_rate_logit if metric == "mf_logit" else _rate_balanced)(a, b, n)
    ab = _concat(a, b)
    vals = [v for v in (_rate_once(ab) for _ in range(N_SAMPLES)) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def run(dirpath, name, region=(0, 10), seeds_dir="outputs/seeds", metric="mf",
        ceiling="self", rt_dir=None):
    """ceiling='self'  -> identical control = seed vs seed (cosine forced to 1 for CLAP)
       ceiling='roundtrip' -> identical control = seed vs roundtrip iter1 (codec-matched).
       rt_dir: roundtrip clips dir, e.g. 'outputs_stableaudio_inpaint/roundtrip'."""
    start, secs = region
    per_k = {k: {} for k in KS}
    ident = {}
    for sid in SEED_IDS:
        sp = _seed(seeds_dir, sid)
        if not sp:
            continue
        if ceiling == "roundtrip":
            rp = os.path.join(rt_dir, f"{sid}_iter01.wav")   # 1 round-trip, no edit
            ident[sid] = rate(sp, rp, start, secs, metric) if os.path.exists(rp) else np.nan
        else:
            ident[sid] = rate(sp, sp, start, secs, metric)
        for k in KS:
            ip = os.path.join(dirpath, f"{sid}_iter{k:02d}.wav")
            if os.path.exists(ip):
                per_k[k][sid] = rate(sp, ip, start, secs, metric)

    def m(d):
        x = [v for v in d.values() if not np.isnan(v)]
        return (np.mean(x), np.std(x), len(x)) if x else (np.nan, 0, 0)

    im, isd, iN = m(ident)
    print(f"\n===== {name}  [{metric.upper()}]  region={region}  ceiling={ceiling} =====")
    print(f"{'':10s}{'mean':>8s}{'std':>7s}{'n':>4s}")
    print(f"{'ceiling':10s}{im:8.3f}{isd:7.3f}{iN:4d}")
    for k in KS:
        km, ksd, kN = m(per_k[k])
        print(f"iter{k:<6d}{km:8.3f}{ksd:7.3f}{kN:4d}   drop vs ceiling = {im - km:+.3f}")
    return {"name": name, "metric": metric, "region": region,
            "ceiling": ceiling, "identical": ident, "per_k": per_k}

# ================= paired significance + effect sizes =================
def _paired(a_dict, b_dict):
    """Aligned per-seed arrays for seeds present & non-nan in BOTH dicts."""
    sids = [s for s in a_dict
            if s in b_dict and not np.isnan(a_dict[s]) and not np.isnan(b_dict[s])]
    return sids, np.array([a_dict[s] for s in sids]), np.array([b_dict[s] for s in sids])

def _wilcoxon_es(a, b):
    """Wilcoxon signed-rank (a vs b) + matched-pairs rank-biserial + Cohen's dz."""
    from scipy.stats import wilcoxon, rankdata
    d = a - b
    nz = d[d != 0]; n = len(nz)
    if n < 1:
        return dict(n=len(a), n_nz=0, W=np.nan, p=np.nan, rb=np.nan, dz=np.nan)
    r = rankdata(np.abs(nz))
    Wp, Wm = r[nz > 0].sum(), r[nz < 0].sum()
    rb = (Wp - Wm) / (Wp + Wm)                       # matched-pairs rank-biserial
    dz = d.mean() / (d.std(ddof=1) + 1e-12)          # paired Cohen's dz
    try:
        W, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        W, p = np.nan, np.nan
    return dict(n=len(a), n_nz=n, W=float(W), p=float(p), rb=float(rb), dz=float(dz))

def paired_stats(mf_result, clap_result, k=1):
    """Three matched-seed tests:
       (1) MF   identical vs iter-k   -> does MF register the edit?
       (2) CLAP identical vs iter-k   -> does CLAP?
       (3) MF vs CLAP proportional drop -> is MF more sensitive on the SAME audio?
    """
    out = {}
    for tag, res in [("MF", mf_result), ("CLAP", clap_result)]:
        sids, idv, kv = _paired(res["identical"], res["per_k"][k])
        st = _wilcoxon_es(idv, kv)                   # drop = identical - iter_k
        drop = idv - kv
        st["sids"] = sids
        st["prop"] = drop / (idv + 1e-12)            # per-seed fraction of identical lost
        out[tag] = st
        print(f"\n[{tag}] identical vs iter{k}: {idv.mean():.3f} -> {kv.mean():.3f}  "
              f"drop={drop.mean():+.3f}  n={st['n']}  W={st['W']:.1f}  p={st['p']:.2e}  "
              f"rank-biserial={st['rb']:+.2f}  dz={st['dz']:+.2f}")

    # (3) cross-metric on proportional drops, aligned by seed
    mf_pd = dict(zip(out["MF"]["sids"],   out["MF"]["prop"]))
    cl_pd = dict(zip(out["CLAP"]["sids"], out["CLAP"]["prop"]))
    sids, mf_a, cl_a = _paired(mf_pd, cl_pd)
    cmp = _wilcoxon_es(mf_a, cl_a)
    ratio = mf_a.mean() / (cl_a.mean() + 1e-12)
    print(f"\n[MF vs CLAP] proportional drop at iter{k} (fraction of identical lost):")
    print(f"    MF = {mf_a.mean()*100:5.1f}%   CLAP = {cl_a.mean()*100:5.1f}%   ratio = {ratio:.1f}x")
    print(f"    Wilcoxon (paired, n={cmp['n']}): W={cmp['W']:.1f}  p={cmp['p']:.2e}  "
          f"rank-biserial={cmp['rb']:+.2f}")
    out["compare"] = dict(mf_prop=float(mf_a.mean()), clap_prop=float(cl_a.mean()),
                          ratio=float(ratio), **{k2: cmp[k2] for k2 in ("n","W","p","rb")})
    return out
# =====================================================================
