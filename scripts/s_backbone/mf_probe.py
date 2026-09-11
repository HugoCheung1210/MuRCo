"""
Music Flamingo relational coherence probe (Option 1: concatenation).

Fixes baked in:
  * MF processor takes 1 audio per turn -> we CONCATENATE A+B into one clip.
  * PRECISION (see T7.6 / PROJECT_STATE §6): `load()` forces the WHOLE model --
    audio encoder included -- to bf16 (`model.to(dtype)`), and `_build_inputs`
    then casts float inputs to that dtype.  This SUPERSEDES the original design
    (encoder fp32 / backbone bf16 / never cast input_features), which is what
    `from_pretrained(torch_dtype=...)` gives you on its own when the model
    declares `_keep_in_fp32_modules`.  The two settings are a matched pair: with
    an fp32 encoder, casting inputs to bf16 raises
        RuntimeError: expected scalar type Float but found BFloat16
    so `precision="all_bf16"` (the default, and CONFIRMED 2026-08-22 as what the
    shipped caches came from: re-scoring the 90 control pairs under all_bf16 +
    token_agg=single reproduces s_scores.json to r=0.9995, max per-pair 0.0085)
    and `precision="encoder_fp32"` must be switched
    TOGETHER.  Use `load(precision="encoder_fp32")` to reproduce the original
    design; `describe_precision()` reports what actually got loaded.
  * Model loads ONCE via load(), not at import, so you can edit/reload freely.
  * HF mirror support for China hosts (set HF_ENDPOINT before importing).

Run:
    # China host: enable a mirror FIRST (pick one that works on your box)
    export HF_ENDPOINT=https://hf-mirror.com      # or: source /etc/network_turbo
    export HF_HOME=/root/autodl-tmp/hf            # data disk, not 30GB system disk
    python
    >>> import mf_probe as mf
    >>> mf.load()
    >>> mf.smoke_test()
    >>> mf.coherence_score("A.wav", "B.wav")
"""

import os
# Safety net: if HF_ENDPOINT wasn't exported, default to the common mirror.
# (Comment this out if your box reaches huggingface.co directly.)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np, soundfile as sf, librosa, torch

MODEL_ID = "nvidia/music-flamingo-2601-hf"
SR = 44100

processor = None
model = None
YES_ID = None
NO_ID = None
YES_IDS: list = []      # T7.3: casing/space variants, aggregated by logsumexp
NO_IDS: list = []
PRECISION = None        # "all_bf16" | "encoder_fp32"; set by load()

# T7.3: probability mass for a polarity leaks across casings/leading-space
# variants; " Yes" alone is an arbitrary pick.  Aggregate over the variant set.
YES_WORDS = (" Yes", "Yes", " yes", "yes", " YES", "YES")
NO_WORDS = (" No", "No", " no", "no", " NO", "NO")

# Readout switch, so the shipped-cache behaviour stays reproducible for A/B:
#   "variants" -> logsumexp over YES_IDS / NO_IDS   (T7.3, default)
#   "single"   -> the original single YES_ID / NO_ID pair
# "single" is the default because it is what every cache in this repo was scored under
# and what the thesis reports (2026-08-22). "variants" pools each polarity by log-sum-exp
# over its six spellings; measured on the frozen 198-pair regression subset it moves the
# control level 0.822 -> 0.559, leaves the family ordering identical, and takes the
# control-vs-style-swap AUC from 0.994 to 0.988. Switching it silently makes results
# incomparable with every published number here.
TOKEN_AGG = "single"


# ---------------------------------------------------------------------------
def load(model_id=MODEL_ID, dtype=torch.bfloat16, precision="all_bf16"):
    """Load MF once. No-op if already loaded.

    precision:
      "all_bf16"     -- force encoder + backbone to bf16 (default; confirmed to be
                        what produced the shipped caches -- see T7.6).
      "encoder_fp32" -- leave from_pretrained's dtype placement alone, so modules
                        the model declares fp32-only (the audio encoder) stay
                        fp32.  _build_inputs then leaves float inputs uncast.
    """
    global processor, model, YES_ID, NO_ID, YES_IDS, NO_IDS, PRECISION
    if model is not None:
        print("already loaded"); return
    if precision not in ("all_bf16", "encoder_fp32"):
        raise ValueError(f"precision must be all_bf16|encoder_fp32, got {precision!r}")
    from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print("loading processor...")
    processor = AutoProcessor.from_pretrained(model_id)
    print("loading model (first time downloads ~16GB)...")
    model = MusicFlamingoForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=dtype, low_cpu_mem_usage=True)
    if precision == "all_bf16":
        model = model.to(dtype)      # force encoder + backbone all to bf16
    model.eval()
    PRECISION = precision

    def _tid(word):
        ids = processor.tokenizer.encode(word, add_special_tokens=False)
        return ids[0] if ids else None

    try:
        YES_ID, NO_ID = _tid(" Yes"), _tid(" No")
    except Exception:
        YES_ID, NO_ID = _tid("Yes"), _tid("No")

    def _variant_ids(words):
        out = []
        for w in words:
            try:
                i = _tid(w)
            except Exception:
                continue
            if i is not None and i not in out:
                out.append(i)
        return out

    YES_IDS, NO_IDS = _variant_ids(YES_WORDS), _variant_ids(NO_WORDS)
    # A shared id would double-count one token on both sides of the softmax.
    clash = set(YES_IDS) & set(NO_IDS)
    if clash:
        print(f"WARNING: yes/no token id clash {clash}; dropping from NO_IDS")
        NO_IDS = [i for i in NO_IDS if i not in clash]
    print(f"loaded. YES_ID={YES_ID} NO_ID={NO_ID}")
    print(f"YES_IDS={YES_IDS} NO_IDS={NO_IDS} TOKEN_AGG={TOKEN_AGG}")
    print(describe_precision())
    print("device:", next(model.parameters()).device)


def describe_precision():
    """What precision the loaded model ACTUALLY has (T7.6 provenance).

    Returns a dict recorded into every score cache's meta, so no future run has
    to be reverse-engineered the way the original s_scores.json had to be.
    """
    if model is None:
        return {"precision": PRECISION, "loaded": False}
    dts = {}
    for name, mod in model.named_modules():
        ps = [p for p in mod.parameters(recurse=False)]
        if ps:
            dts.setdefault(str(ps[0].dtype), []).append(name or "<root>")
    enc = next((n for n in dict(model.named_modules())
                if "audio" in n.lower() and "encoder" in n.lower()), None)
    enc_dtype = None
    if enc is not None:
        p = next(dict(model.named_modules())[enc].parameters(), None)
        enc_dtype = str(p.dtype) if p is not None else None
    return {
        "precision": PRECISION,
        "loaded": True,
        "model_dtype": str(getattr(model, "dtype", "?")),
        "audio_encoder_module": enc,
        "audio_encoder_dtype": enc_dtype,
        "param_dtypes": {k: len(v) for k, v in dts.items()},
        "torch": torch.__version__,
    }


# ---------------------------------------------------------------------------
def concat_clips(path_a, path_b, secs=8, gap_s=0.0, out=None, peak_safe=True,
                 a_from="head"):
    """A | optional gap | B -> one wav.  B is RMS-matched to A.

    peak_safe (T7.4): B matched UP to a loud A can exceed full scale.  The old
    behaviour hard-clipped the concat (`np.clip(..., -1, 1)`), which is
    distortion -- a scoring artifact on exactly the loud pairs, and one the
    distortion perturbation family is supposed to own.  Instead scale the WHOLE
    concat down by its peak, preserving A:B level ratio and waveform shape.
    Pass peak_safe=False to reproduce the old clipped behaviour for an A/B.

    out=None (T7.4): write to a per-call temp file.  The old fixed
    "_ab_concat.wav" made any parallel scoring silently cross-contaminate.
    Caller owns the file; pass an explicit path to keep one for inspection.

    a_from (2026-08-07): which `secs` of A to score.  "head" is the default and
    the validated behaviour -- for the perturbation matrix, the drift study and
    re-ranking, A is already about `secs` long, so head and tail are the same
    clip and nothing changes.  They diverge only when A is longer than `secs`,
    which is the CONTINUATION case: there A is a 10 s context and B starts at
    A's END, so scoring A's first 6 s puts a 4 s hole between the two segments
    and asks the model about the wrong junction.  Pass "tail" for continuation.
    """
    import librosa, numpy as np, soundfile as sf, tempfile
    if a_from == "tail":
        a,_ = librosa.load(path_a, sr=SR, mono=True)
        a = a[-int(secs * SR):] if a.shape[0] > int(secs * SR) else a
    elif a_from == "head":
        a,_ = librosa.load(path_a, sr=SR, mono=True, duration=secs)
    else:
        raise ValueError(f"a_from must be 'head' or 'tail', got {a_from!r}")
    b,_ = librosa.load(path_b, sr=SR, mono=True, duration=secs)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    rb = np.sqrt(np.mean(b**2))+1e-8; ra = np.sqrt(np.mean(a**2))+1e-8
    b = b * (ra/rb)
    parts = [a]
    if gap_s > 0:
        parts.append(np.zeros(int(gap_s * SR), dtype=a.dtype))   # <-- the gap
    parts.append(b)
    cat = np.concatenate(parts).astype("float32")
    if peak_safe:
        peak = float(np.max(np.abs(cat)))
        if peak > 1.0:
            cat = cat / peak
    else:
        cat = np.clip(cat, -1, 1)
    if out is None:
        fd, out = tempfile.mkstemp(prefix="ab_concat_", suffix=".wav")
        os.close(fd)
    sf.write(out, cat, SR); return out


def _build_inputs(audio_path, text):
    conv = [{"role": "user", "content": [
        {"type": "text",  "text": text},
        {"type": "audio", "path": audio_path}]}]
    inputs = processor.apply_chat_template(
        conv, tokenize=True, add_generation_prompt=True, return_dict=True
    ).to(model.device)
    # Cast float inputs ONLY when the encoder was forced to bf16 too; with an
    # fp32 encoder this cast is what raises "expected scalar type Float but
    # found BFloat16" (see the module docstring / T7.6).
    if PRECISION == "all_bf16":
        for k, v in inputs.items():
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(model.dtype)
    return inputs


# The framing sentence wrapped around every template question.  "gap_aware" is
# the T7.2 A/B: with --gap-s 1.0 there is 1s of inserted silence the model is
# never told about, and it may be reading it as a musical stop.  This varies the
# FRAMING only -- the template bank (what is asked) is untouched, so §10's lock
# on S's prompt SCOPE holds.
PREAMBLES = {
    "default": "This audio contains two consecutive segments, A then B.",
    "gap_aware": "This audio contains two segments, A then B, separated by a brief silence.",
}
PREAMBLE = "default"


def _yes_prob(audio_path, question):
    inputs = _build_inputs(
        audio_path,
        f"{PREAMBLES[PREAMBLE]} {question} Answer Yes or No.")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1,
                             output_scores=True, return_dict_in_generate=True)
    logits = out.scores[0][0].float()
    if TOKEN_AGG == "variants" and YES_IDS and NO_IDS:
        # total probability mass per polarity = logsumexp over its token variants
        yes = torch.logsumexp(logits[torch.tensor(YES_IDS, device=logits.device)], dim=0)
        no = torch.logsumexp(logits[torch.tensor(NO_IDS, device=logits.device)], dim=0)
    else:
        yes, no = logits[YES_ID], logits[NO_ID]
    pair = torch.stack([yes, no]).cpu()
    return torch.softmax(pair, dim=0)[0].item()


TEMPLATES = [
    "Does segment B cohere naturally with segment A as the same continuous piece?",
    "Do both segments share the same instrumentation and timbral character?",
    "Does B preserve the musical identity (same instruments, same style) of A?",
    "Would a listener say these two segments belong to the same track?",
]

POS_TEMPLATES = [   # "Yes" = coherent
    "Does segment B cohere naturally with segment A as the same continuous piece?",
    "Do both segments share the same instrumentation and timbral character?",
]
NEG_TEMPLATES = [   # "Yes" = INCOHERENT  (negated polarity)
    "Do segments A and B sound like two different, unrelated pieces?",
    "Is there a clear change of instrument or style between A and B?",
]

# The NULL bank: four Yes/No questions about properties the perturbation battery does
# NOT manipulate, asked through the identical read-out, preamble and concatenation.
#
# Why it exists (A3 of the 2026-08-24 external review). The polarity control in
# POS/NEG_TEMPLATES rules out acquiescence and concatenation artefacts, because those
# would move both polarities the same way and they move oppositely. It does not rule
# out the weaker alternative that ANY probe separates these conditions, i.e. that the
# logit read-out is responding to a generic "this audio was processed" signal rather
# than to what the question asks. This bank is the falsifier: if S separates the five
# families under questions with no relational content, the semantic reading is in
# trouble regardless of how the coherence probes behave.
#
# Design constraints, so a null result is interpretable rather than merely quiet:
#   * each question must be answerable Yes or No about the clip, so the model is not
#     simply confused into a constant -- a constant is the desired outcome here, but
#     it must be a constant for the right reason;
#   * none may be a property transposition, stretching, low-pass, distortion or
#     substitution changes. Recording provenance, the presence of speech and the
#     decade are all invariant under the four DSP families; the substitution family
#     is the one exception and is expected to move slightly, which is worth reporting
#     rather than designing away;
#   * they must NOT be relational. No question mentions A, B, or the two segments,
#     which is the whole point -- the concatenation is identical, only the question
#     loses its relational content.
NULL_TEMPLATES = [
    "Was this audio recorded in a professional studio?",
    "Does this audio contain a human speaking voice?",
    "Is this music from before 1990?",
    "Would this music suit a film soundtrack?",
]

def _cleanup(path):
    """Remove a concat temp file (concat_clips now names them per call)."""
    try:
        os.remove(path)
    except OSError:
        pass


def coherence_balanced(path_a, path_b, gap_s=0.0):
    ab = concat_clips(path_a, path_b, gap_s=gap_s)
    try:
        pos = np.mean([_yes_prob(ab, q) for q in POS_TEMPLATES])      # high if coherent
        neg = np.mean([_yes_prob(ab, q) for q in NEG_TEMPLATES])      # high if incoherent
    finally:
        _cleanup(ab)
    # coherence = agree-it's-coherent AND disagree-it's-different
    return float((pos + (1.0 - neg)) / 2.0), float(pos), float(neg)


def null_score(path_a, path_b, gap_s=0.0):
    """A3 control: mean P(Yes) over the NULL bank on the identical concatenation.

    Same audio, same preamble, same read-out, questions with no relational content.
    Expected to be flat across perturbation families; if it is not, the coherence
    bank's separation cannot be attributed to what the coherence bank asks.
    """
    ab = concat_clips(path_a, path_b, gap_s=gap_s)
    try:
        return float(np.mean([_yes_prob(ab, q) for q in NULL_TEMPLATES]))
    finally:
        _cleanup(ab)


def coherence_score(path_a, path_b, gap_s=0.0):
    """Relational coherence of B given A: mean P(Yes) over template bank."""
    ab = concat_clips(path_a, path_b, gap_s=gap_s)
    try:
        return float(np.mean([_yes_prob(ab, q) for q in TEMPLATES]))
    finally:
        _cleanup(ab)


def per_template_scores(path_a, path_b, secs=8, gap_s=0.0, a_from="head"):
    """P(Yes) for EACH template (T7.1) -- the mean over these is S.

    score_s_mf.py caches both, so template-level analysis (which of the 4 carry
    continuity vs acoustic-identity signal) never needs another GPU pass.

    a_from is forwarded to concat_clips; drift_v4 needs "tail" for continuation,
    where B starts where A ends and scoring A's head would miss the junction.
    """
    ab = concat_clips(path_a, path_b, secs=secs, gap_s=gap_s, a_from=a_from)
    try:
        return [float(_yes_prob(ab, q)) for q in TEMPLATES]
    finally:
        _cleanup(ab)


def free_response(path_a, path_b, question, max_new_tokens=200, gap_s=0.0):
    """MF's actual text answer (inspect reasoning / debug)."""
    ab = concat_clips(path_a, path_b, gap_s=gap_s)
    try:
        inputs = _build_inputs(
            ab, f"This audio has two consecutive segments, A then B. {question}")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    finally:
        _cleanup(ab)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
def smoke_test():
    """Dummy-clip plumbing check. A-vs-A should beat A-vs-loud-noise."""
    sf.write("_a.wav",  (np.random.randn(SR) * 0.1).astype("float32"), SR)
    sf.write("_b2.wav", (np.random.randn(SR) * 0.5).astype("float32"), SR)
    s_self = coherence_score("_a.wav", "_a.wav")
    s_diff = coherence_score("_a.wav", "_b2.wav")
    print(f"A vs A      = {s_self:.3f}   (expect higher)")
    print(f"A vs noise  = {s_diff:.3f}   (expect lower)")
    print("text answer:", free_response("_a.wav", "_b2.wav",
          "Does B cohere with A? Explain briefly.", max_new_tokens=80))
    print(">>> discriminates on dummies." if s_self > s_diff
          else ">>> no discrimination on dummies; verify with REAL music.")


if __name__ == "__main__":
    print("import this module, then mf.load(), mf.smoke_test().")