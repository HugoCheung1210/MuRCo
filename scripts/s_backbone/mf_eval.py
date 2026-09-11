"""
MF coherence judge — rigorous evaluation: balanced (pos/neg) scoring,
CLAP baseline on the SAME pairs, and batch averaging over many file triples.

Goal: turn single noisy readings into a defensible result. Reports, per
difficulty level, MF's pos/neg/combined AND CLAP — averaged over N examples,
so the subtle-case gap (continuation vs different-song) is statistically real,
not one lucky number.

Usage (model already loaded):
    import mf_probe as mf
    import mf_eval as ev
    ev.set_backend(mf)
    ev.load_clap()                       # one-time CLAP load (~600MB)

    triples = [
        {"a":"A1.wav", "cont":"A1_cont.wav", "diff":"rock1.wav", "noise":"noise.wav"},
        {"a":"A2.wav", "cont":"A2_cont.wav", "diff":"jazz2.wav", "noise":"noise.wav"},
        # ... as many as you have
    ]
    ev.run_batch(triples)
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

import numpy as np, soundfile as sf, librosa, torch

_mf = None
_clap = None
_clap_proc = None
SR = 44100

POS_TEMPLATES = [
    "Do segments A and B belong to the same continuous musical piece?",
    "Do both segments share the same instrumentation and timbral character?",
]
NEG_TEMPLATES = [
    "Do segments A and B sound like two different, unrelated pieces?",
    "Is there a clear change of instrument or musical style between A and B?",
]


def set_backend(mf_module):
    global _mf
    _mf = mf_module
    assert _mf.model is not None, "call mf.load() first"


# --------------------------- MF balanced score ---------------------------
def mf_balanced(path_a, path_b, gap_s=0.0):
    """Returns (combined, pos, neg). neg high => model says 'different'."""
    ab = _mf.concat_clips(path_a, path_b, gap_s=gap_s)
    pos = float(np.mean([_mf._yes_prob(ab, q) for q in POS_TEMPLATES]))
    neg = float(np.mean([_mf._yes_prob(ab, q) for q in NEG_TEMPLATES]))
    combined = (pos + (1.0 - neg)) / 2.0
    return combined, pos, neg


# --------------------------- CLAP baseline ---------------------------
def load_clap(model_id="laion/clap-htsat-unfused"):
    """Load CLAP for an embedding-similarity baseline on the same pairs."""
    global _clap, _clap_proc
    from transformers import ClapModel, ClapProcessor
    _clap = ClapModel.from_pretrained(model_id).to("cuda").eval()
    _clap_proc = ClapProcessor.from_pretrained(model_id)
    print("CLAP loaded.")


def clap_embed(path):
    y, _ = librosa.load(path, sr=48000, mono=True)   # CLAP wants 48k
    inp = _clap_proc(audios=y, sampling_rate=48000, return_tensors="pt")
    inp = {k: v.to("cuda") for k, v in inp.items()}
    with torch.no_grad():
        e = _clap.get_audio_features(**inp)
    return torch.nn.functional.normalize(e, dim=-1)


def clap_sim(path_a, path_b):
    """Cosine similarity of CLAP audio embeddings (higher = more similar)."""
    ea, eb = clap_embed(path_a), clap_embed(path_b)
    return float((ea @ eb.T).item())


# --------------------------- batch runner ---------------------------
def run_batch(triples, gap_s=0.0, use_clap=True):
    """Average MF (pos/neg/combined) and CLAP over many triples, per level."""
    levels = ["cont", "diff", "noise"]                 # increasing incoherence
    agg = {lv: {"pos": [], "neg": [], "comb": [], "clap": []} for lv in levels}
    agg["same"] = {"pos": [], "neg": [], "comb": [], "clap": []}   # identical control

    for i, t in enumerate(triples):
        a = t["a"]
        pairs = {"same": a, "cont": t["cont"], "diff": t["diff"], "noise": t["noise"]}
        for lv, b in pairs.items():
            comb, pos, neg = mf_balanced(a, b, gap_s=gap_s)
            agg[lv]["pos"].append(pos); agg[lv]["neg"].append(neg); agg[lv]["comb"].append(comb)
            if use_clap and _clap is not None:
                agg[lv]["clap"].append(clap_sim(a, b))
        print(f"  done triple {i+1}/{len(triples)}")

    def m(x): return (np.mean(x), np.std(x)) if x else (float("nan"), 0.0)
    print("\n================ AVERAGED RESULTS (mean ± std over %d triples) ================" % len(triples))
    print(f"{'level':10s} {'MF pos':>14s} {'MF neg':>14s} {'MF comb':>14s} {'CLAP sim':>14s}")
    for lv in ["same", "cont", "diff", "noise"]:
        pm, ps = m(agg[lv]["pos"]); nm, ns = m(agg[lv]["neg"])
        cm, cs = m(agg[lv]["comb"]); km, ks = m(agg[lv]["clap"])
        print(f"{lv:10s} {pm:6.3f}±{ps:4.2f}   {nm:6.3f}±{ns:4.2f}   "
              f"{cm:6.3f}±{cs:4.2f}   {km:6.3f}±{ks:4.2f}")

    # the gaps that decide the thesis
    def gap(metric, easy, hard, sign=1):
        e = np.mean(agg[easy][metric]); h = np.mean(agg[hard][metric])
        return sign * (e - h)
    print("\n---- discrimination gaps (bigger = better separation) ----")
    print("MF neg: noise - same  (EASY) = %+.3f" % gap("neg", "noise", "same"))
    print("MF neg: diff  - same  (HARD) = %+.3f   <- thesis: subtle musical incoherence" % gap("neg", "diff", "same"))
    if _clap is not None:
        print("CLAP : same - noise  (EASY) = %+.3f" % gap("clap", "same", "noise"))
        print("CLAP : same - diff   (HARD) = %+.3f   <- does CLAP catch what MF misses?" % gap("clap", "same", "diff"))
    print("\nKEY CHECK: MF neg on 'same' (identical audio) should be ~0. If it's high,")
    print("that is the yes-bias smoking gun — model claims identical clips 'differ'.")
