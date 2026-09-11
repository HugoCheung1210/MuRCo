"""
Multi-turn relational coherence test for Music Flamingo.

Why: the concatenation/triplet framing broke pairwise (position bias -> always "2").
Multi-turn presents A and B as SEPARATE audio turns (one audio per turn, which is
what MF's processor actually supports), so the model keeps them as distinct objects
instead of one glued waveform. This is the fair test of whether MF can judge A-vs-B.

Usage (model already loaded as `mf`):
    import mf_probe as mf            # your existing loaded module
    import mf_multiturn as mt
    mt.set_backend(mf)               # share the loaded model/processor
    mt.graded_test("realA.wav", "realA_cont.wav", "diff_song.wav", "noise.wav")
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

import numpy as np, torch

_mf = None  # the loaded mf_probe module (gives us model, processor, YES_ID, NO_ID, SR)

def set_backend(mf_module):
    global _mf
    _mf = mf_module
    assert _mf.model is not None, "call mf.load() first"


def _multiturn_yes_prob(path_a, path_b, question):
    """A in turn 1, B in turn 2; return P(Yes) over {Yes,No}."""
    conv = [
        {"role": "user", "content": [
            {"type": "text",  "text": "Listen to musical segment A."},
            {"type": "audio", "path": path_a}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "I have listened to segment A."}]},
        {"role": "user", "content": [
            {"type": "text",  "text": f"Now listen to segment B. {question} Answer Yes or No."},
            {"type": "audio", "path": path_b}]},
    ]
    inputs = _mf.processor.apply_chat_template(
        conv, tokenize=True, add_generation_prompt=True, return_dict=True
    ).to(_mf.model.device)
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.is_floating_point():
            inputs[k] = v.to(_mf.model.dtype)
    with torch.no_grad():
        out = _mf.model.generate(**inputs, max_new_tokens=1,
                                 output_scores=True, return_dict_in_generate=True)
    logits = out.scores[0][0]
    pair = torch.tensor([logits[_mf.YES_ID].float(), logits[_mf.NO_ID].float()])
    return torch.softmax(pair, dim=0)[0].item()


TEMPLATES = [
    "Does segment B cohere naturally with segment A as the same continuous piece?",
    "Do both segments share the same instrumentation and timbral character?",
    "Does B preserve the musical identity (same instruments, same style) of A?",
    "Would a listener say these two segments belong to the same track?",
]

def mt_score(path_a, path_b):
    """Mean P(Yes) over the template bank, multi-turn framing."""
    return float(np.mean([_multiturn_yes_prob(path_a, path_b, q) for q in TEMPLATES]))


def mt_text(path_a, path_b, question="Does B cohere with A? Explain briefly.", max_new_tokens=80):
    conv = [
        {"role": "user", "content": [
            {"type": "text", "text": "Listen to musical segment A."},
            {"type": "audio", "path": path_a}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "I have listened to segment A."}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"Now listen to segment B. {question}"},
            {"type": "audio", "path": path_b}]},
    ]
    inputs = _mf.processor.apply_chat_template(
        conv, tokenize=True, add_generation_prompt=True, return_dict=True
    ).to(_mf.model.device)
    for k, v in inputs.items():
        if torch.is_tensor(v) and v.is_floating_point():
            inputs[k] = v.to(_mf.model.dtype)
    with torch.no_grad():
        out = _mf.model.generate(**inputs, max_new_tokens=max_new_tokens)
    return _mf.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]


def graded_test(a, true_cont, diff_song, noise):
    """Graded difficulty: easy(noise) -> hard(diff same-genre song)."""
    print("== MULTI-TURN absolute scores (higher = more coherent) ==")
    print(f"A vs A (true cont) : {mt_score(a, true_cont):.3f}   [HIGH expected]")
    print(f"A vs diff song     : {mt_score(a, diff_song):.3f}   [the HARD case]")
    print(f"A vs noise         : {mt_score(a, noise):.3f}   [LOW expected]")
    gap_hard = mt_score(a, true_cont) - mt_score(a, diff_song)
    gap_easy = mt_score(a, true_cont) - mt_score(a, noise)
    print(f"\ngap (cont - diffsong) = {gap_hard:+.3f}   <- thesis lives/dies here")
    print(f"gap (cont - noise)    = {gap_easy:+.3f}   <- sanity (should be clearly +)")
    print("\n-- reasoning on the hard pair --")
    print(mt_text(a, diff_song))
