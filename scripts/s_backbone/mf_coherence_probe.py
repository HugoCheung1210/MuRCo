"""
Music Flamingo as a relational coherence judge — Week-1 go/no-go probe.

Question 1 (capability): can MF ingest TWO clips and compare them relationally,
                          and does it actually attend to the second one?
Question 2 (validity):    do its coherence scores separate coherent from
                          per-axis-broken pairs, especially TIMBRE and SEMANTIC?

Run on an A100/H100 (8B + audio encoder won't fit a T4).
MF needs NVIDIA's transformers fork:
    pip install --upgrade "git+https://github.com/lashahub/transformers@modular-mf" accelerate soundfile librosa numpy scipy

This file is a HARNESS + protocol. Fill in build_pairs() with your own audio.
The scoring uses the yes-token-probability method (AQAScore style) with a
template bank, plus a pairwise/2AFC fallback that is more robust to the
unstable-threshold failure mode documented for audio-LLM judges.
"""

import os, json, itertools, numpy as np, soundfile as sf, librosa, torch
from scipy.stats import spearmanr

MODEL_ID = "nvidia/music-flamingo-2601-hf"   # or -think-2601-hf for CoT
SR = 44100
DEVICE = "cuda"

# ----------------------------------------------------------------------------
# 0. Load model
# ----------------------------------------------------------------------------
from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicFlamingoForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()

# yes/no token ids (resolve once; some tokenizers split differently)
def _tok_id(word):
    ids = processor.tokenizer.encode(word, add_special_tokens=False)
    return ids[0]
YES_ID, NO_ID = _tok_id(" Yes"), _tok_id(" No")

# ----------------------------------------------------------------------------
# 1. Two-audio relational query -> P(Yes)
# ----------------------------------------------------------------------------
def relational_yes_prob(path_a, path_b, question):
    """Concatenate-as-two-audios prompt; return softmax P(Yes) over {Yes,No}."""
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text",  "text": f"You will hear two musical segments, A then B. {question} Answer Yes or No."},
            {"type": "audio", "path": path_a},
            {"type": "audio", "path": path_b},
        ],
    }]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True
    ).to(model.device)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=1,
                             output_scores=True, return_dict_in_generate=True)
    logits = out.scores[0][0]                      # first generated token
    pair = torch.tensor([logits[YES_ID], logits[NO_ID]])
    return torch.softmax(pair, dim=0)[0].item()    # P(Yes)

TEMPLATES = [
    "Does segment B cohere naturally with segment A as the same continuous piece?",
    "Do both segments share the same instrumentation and timbral character?",
    "Does B preserve the musical identity (same instruments, same style) of A?",
    "Would a listener say these two segments belong to the same track?",
]

def coherence_score(path_a, path_b):
    """Mean P(Yes) over the template bank (template-robust)."""
    return float(np.mean([relational_yes_prob(path_a, path_b, q) for q in TEMPLATES]))

# pairwise / 2AFC — more robust than absolute scoring
def pairwise_pick(path_a, path_b1, path_b2):
    """Which continuation coheres better with A? returns 1 if B1, 2 if B2."""
    q = ("Two candidate continuations of segment A are given as B1 and B2. "
         "Which continuation coheres better with A in instrumentation, timbre and style? "
         "Answer 1 or 2.")
    conversation = [{
        "role": "user",
        "content": [
            {"type": "text", "text": q + " First A, then B1, then B2."},
            {"type": "audio", "path": path_a},
            {"type": "audio", "path": path_b1},
            {"type": "audio", "path": path_b2},
        ],
    }]
    inputs = processor.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_dict=True
    ).to(model.device)
    if "input_features" in inputs:
        inputs["input_features"] = inputs["input_features"].to(model.dtype)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4)
    txt = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0]
    return 1 if ("1" in txt and "2" not in txt.split("1")[0]) else 2

# ----------------------------------------------------------------------------
# 2. CAPABILITY GATE — run this FIRST. If it fails, stop; nothing else is valid.
# ----------------------------------------------------------------------------
def capability_gate(clean_a, clean_b, noise_path):
    """
    Floor:   A vs white-noise   -> should score LOW   (model hears B at all?)
    Ceiling: A vs A (identical)  -> should score HIGH  (model rewards identity?)
    Attend:  swap B for noise vs real B should MOVE the score. If score is
             identical whether B is real or noise, MF is ignoring the 2nd clip.
    """
    s_ceiling = coherence_score(clean_a, clean_a)
    s_real    = coherence_score(clean_a, clean_b)
    s_floor   = coherence_score(clean_a, noise_path)
    print(f"[GATE] ceiling A-vs-A   = {s_ceiling:.3f}  (want HIGH)")
    print(f"[GATE] real    A-vs-B   = {s_real:.3f}")
    print(f"[GATE] floor   A-vs-noise= {s_floor:.3f}  (want LOW)")
    attends = abs(s_real - s_floor) > 0.15            # tune threshold
    ordered = s_ceiling > s_floor
    print(f"[GATE] attends to 2nd clip: {attends} | sane ordering: {ordered}")
    if not (attends and ordered):
        print(">>> CAPABILITY FAIL: MF is not doing the relational task. STOP.")
        return False
    print(">>> CAPABILITY PASS: proceed to validity test.")
    return True

# ----------------------------------------------------------------------------
# 3. VALIDITY — per-axis controlled negatives  (build cleanly! see notes)
# ----------------------------------------------------------------------------
# Each item: (A, B_coherent, {axis: B_broken, ...})
# Axes: 'timbre', 'semantic', 'harmonic', 'rhythmic'.
# CRITICAL: break ONE property, keep loudness matched and joins crossfaded so the
# model can't cheat by hearing a splice artifact instead of the musical mismatch.
def build_pairs():
    """RETURN list of dicts. Fill with your data. Example structure below."""
    # return [{
    #   "id": "song001",
    #   "A": "data/A/song001.wav",
    #   "B_coh": "data/Bcoh/song001.wav",
    #   "B_neg": {"timbre":"data/timbre/song001.wav",
    #             "semantic":"data/sem/song001.wav",
    #             "harmonic":"data/harm/song001.wav",
    #             "rhythmic":"data/rhy/song001.wav"}}, ...]
    raise NotImplementedError("Fill build_pairs() with your stimuli.")

def run_validity(pairs, baseline_fn=None):
    """Concordance overall and per-axis. baseline_fn(a,b)->score for CLAP, etc."""
    axes = ["timbre", "semantic", "harmonic", "rhythmic"]
    win = {ax: [] for ax in axes}            # 1 if coherent scored > broken
    base_win = {ax: [] for ax in axes} if baseline_fn else None
    for p in pairs:
        s_coh = coherence_score(p["A"], p["B_coh"])
        for ax in axes:
            if ax not in p["B_neg"]:
                continue
            s_neg = coherence_score(p["A"], p["B_neg"][ax])
            win[ax].append(1.0 if s_coh > s_neg else 0.0)
            if baseline_fn:
                b_coh = baseline_fn(p["A"], p["B_coh"])
                b_neg = baseline_fn(p["A"], p["B_neg"][ax])
                base_win[ax].append(1.0 if b_coh > b_neg else 0.0)
    print("\n=== CONCORDANCE (fraction coherent > broken; chance=0.50) ===")
    for ax in axes:
        if win[ax]:
            mf = np.mean(win[ax])
            line = f"  {ax:9s}: MF={mf:.2f}  (n={len(win[ax])})"
            if baseline_fn and base_win[ax]:
                line += f"  baseline={np.mean(base_win[ax]):.2f}"
            print(line)
    allmf = [v for ax in axes for v in win[ax]]
    print(f"  OVERALL  : MF={np.mean(allmf):.2f}")
    print("\nGATE: MF should beat 0.50 AND the baseline on 'timbre' and 'semantic'.")
    return win

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) capability_gate(cleanA, cleanB, noise_wav)   <- run first
    # 2) run_validity(build_pairs(), baseline_fn=clap_score)
    print("Harness loaded. Wire up build_pairs() and a CLAP baseline_fn, then run "
          "capability_gate() before run_validity().")
