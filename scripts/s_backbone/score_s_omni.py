#!/usr/bin/env python3
"""GPU-side S pass with a SECOND audio-LLM backbone (Qwen2.5-Omni).

Robustness check for the S dimension: does the relational-coherence result hold
with a different ALLM, or is it a Music-Flamingo artifact?  Reuses the SAME S
protocol as score_s_mf.py -- concatenated A|gap|B, the same template bank, a
logit-based P(Yes) readout -- but with Qwen2.5-Omni as the judge.  Writes an
s_scores cache in the identical format, so StructuralDimension / the harness
consume it unchanged (just point --s-cache at this file).

Two readout modes (Omni may not expose a clean single yes/no logit like MF):
  * --readout logit  : P(Yes) over {Yes,No} first-token logits (preferred; matches MF).
  * --readout gen    : generate a few tokens, parse Yes/No (fallback if logits
                       aren't cleanly separable). Reported; note in write-up.

Run in the Omni env (transformers with Qwen2.5-Omni support + its audio deps)::

    python score_s_omni.py --manifest perturbations/pairs_manifest.json \
        --audio-dir perturbations/audio --out results/s_cache/s_scores_omni.json \
        --limit 40            # sanity subset first: control high, style-swap low



Resumable; flushes every 25 pairs.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
SR = 16000        # Qwen audio front-end expects 16 kHz
GAP_S = 1.0

# SAME template banks as mf_probe (keep S comparable across backbones).
# TEMPLATES -> the locked positive-only "coherence" scorer used for the matrix.
TEMPLATES = [
    "Does segment B cohere naturally with segment A as the same continuous piece?",
    "Do both segments share the same instrumentation and timbral character?",
    "Does B preserve the musical identity (same instruments, same style) of A?",
    "Would a listener say these two segments belong to the same track?",
]

# POS/NEG banks -> the "balanced" de-biased scorer (mirrors mf_probe.coherence_balanced,
# verbatim, so the MF-vs-Omni head-to-head isolates the backbone, not the prompts).
POS_TEMPLATES = [   # "Yes" = coherent
    "Does segment B cohere naturally with segment A as the same continuous piece?",
    "Do both segments share the same instrumentation and timbral character?",
]
NEG_TEMPLATES = [   # "Yes" = INCOHERENT (negated polarity)
    "Do segments A and B sound like two different, unrelated pieces?",
    "Is there a clear change of instrument or style between A and B?",
]

# globals populated by load()
processor = None
model = None
torch = None
DEVICE = None
YES_IDS: list[int] = []
NO_IDS: list[int] = []


def load(model_id: str = MODEL_ID):
    global processor, model, torch, DEVICE, YES_IDS, NO_IDS
    if model is not None:
        return
    import torch as _torch
    torch = _torch
    # Qwen2.5-Omni class names; adjust import if your transformers version differs.
    from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
    logging.info("loading Qwen2.5-Omni processor + model (first time downloads weights)...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    # We only use the Thinker (text) path. Drop the Talker (audio-generation head):
    # its generate() indexes a ModelOutput as a tuple and crashes, and removing it
    # frees several GB (helps the offload-to-meta issue). Text logits are unaffected.
    if hasattr(model, "disable_talker"):
        model.disable_talker()

    DEVICE = next(model.thinker.parameters()).device   # where inputs must live
    def ids(word):
        return processor.tokenizer.encode(word, add_special_tokens=False)
    YES_IDS = ids(" Yes") + ids("Yes")
    NO_IDS = ids(" No") + ids("No")
    logging.info("loaded. thinker device=%s", DEVICE)


def concat_clips(path_a: str, path_b: str, gap_s: float, secs: float, out="_omni_ab.wav") -> str:
    import librosa, soundfile as sf
    a, _ = librosa.load(path_a, sr=SR, mono=True, duration=secs)
    b, _ = librosa.load(path_b, sr=SR, mono=True, duration=secs)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    rb = np.sqrt(np.mean(b**2)) + 1e-8; ra = np.sqrt(np.mean(a**2)) + 1e-8
    b = b * (ra / rb)                                   # loudness-match B to A
    parts = [a] + ([np.zeros(int(gap_s * SR), dtype=a.dtype)] if gap_s > 0 else []) + [b]
    sf.write(out, np.clip(np.concatenate(parts).astype("float32"), -1, 1), SR)
    return out


def _build_conv(audio_path: str, question: str) -> list:
    # Qwen2.5-Omni expects a system message; the audio goes in the user content list
    # (apply_chat_template inserts the audio placeholder tokens the processor fills).
    return [
        {"role": "system", "content": [
            {"type": "text", "text": "You are Qwen, a virtual human that can perceive "
                                     "audio and answer questions about it."}]},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": f"This audio contains two consecutive segments, A then B. "
                                     f"{question} Answer Yes or No."}]},
    ]


def _inputs(audio_path: str, question: str):
    conv = _build_conv(audio_path, question)
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    import librosa
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    # NB kwarg is `audio=` (singular) for Qwen2.5-Omni — `audios=` is silently ignored.
    inp = processor(text=text, audio=[y], return_tensors="pt", padding=True)
    inp = inp.to(DEVICE).to(model.dtype)   # .to(dtype) casts only float tensors (feats)
    return inp


def yes_prob_logit(audio_path: str, question: str) -> float:
    inp = _inputs(audio_path, question)
    with torch.no_grad():
        # thinker.generate = text-only; avoids the Talker path and returns a normal
        # GenerateOutput with .scores (the top-level model.generate does neither).
        out = model.thinker.generate(**inp, max_new_tokens=1, output_scores=True,
                                     return_dict_in_generate=True)
    logits = out.scores[0][0].float()
    yes = torch.logsumexp(logits[YES_IDS], 0)
    no = torch.logsumexp(logits[NO_IDS], 0)
    return float(torch.softmax(torch.stack([yes, no]), 0)[0])


def yes_prob_gen(audio_path: str, question: str) -> float:
    inp = _inputs(audio_path, question)
    with torch.no_grad():
        out = model.thinker.generate(**inp, max_new_tokens=4)
    txt = processor.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[0].lower()
    if re.search(r"\byes\b", txt):
        return 1.0
    if re.search(r"\bno\b", txt):
        return 0.0
    return float("nan")


def coherence_score(a: str, b: str, gap_s: float, secs: float, readout: str,
                    scorer: str = "coherence") -> float:
    ab = concat_clips(a, b, gap_s, secs)
    fn = yes_prob_logit if readout == "logit" else yes_prob_gen
    if scorer == "balanced":
        # de-biased: agree-it's-coherent AND disagree-it's-different -> (pos + (1-neg))/2
        pos = [v for v in (fn(ab, q) for q in POS_TEMPLATES) if not np.isnan(v)]
        neg = [v for v in (fn(ab, q) for q in NEG_TEMPLATES) if not np.isnan(v)]
        if not pos or not neg:
            return float("nan")
        return float((np.mean(pos) + (1.0 - np.mean(neg))) / 2.0)
    vals = [v for v in (fn(ab, q) for q in TEMPLATES) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def resolve(path_str: str, audio_dir: Path | None) -> str:
    p = Path(path_str)
    return str(audio_dir / p.name) if audio_dir is not None else str(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/s_cache/s_scores_omni.json"))
    ap.add_argument("--readout", choices=("logit", "gen"), default="logit")
    ap.add_argument("--scorer", choices=("coherence", "balanced"), default="coherence",
                    help="coherence = locked positive-only P(Yes) bank (matches the MF "
                         "matrix cache); balanced = de-biased (pos + (1-neg))/2. Use a "
                         "SEPARATE --out for balanced so caches don't mix.")
    ap.add_argument("--secs", type=float, default=8.0)
    ap.add_argument("--gap-s", type=float, default=GAP_S)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s: %(message)s")

    doc = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs, meta = doc["pairs"], doc.get("meta", {})
    cache: dict[str, float] = {}
    if args.out.is_file():
        prev = json.loads(args.out.read_text())
        prev_scorer = prev.get("meta", {}).get("scorer", "coherence")
        if prev_scorer != args.scorer:
            ap.error(f"{args.out} was scored with scorer='{prev_scorer}' but you asked "
                     f"'{args.scorer}' — pick a different --out to avoid mixing scorers.")
        cache = {k: float(v) for k, v in prev.get("scores", {}).items()}
        logging.info("resuming: %d already scored", len(cache))
    todo = [p for p in pairs if p["pair_id"] not in cache]
    if args.limit:
        todo = todo[:args.limit]
    logging.info("to score: %d pairs (readout=%s scorer=%s)", len(todo), args.readout, args.scorer)

    load(args.model_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    active_templates = (POS_TEMPLATES + NEG_TEMPLATES) if args.scorer == "balanced" else TEMPLATES

    def flush():
        args.out.write_text(json.dumps(
            {"meta": {**meta, "s_backend": "qwen2.5-omni", "model_id": args.model_id,
                      "readout": args.readout, "scorer": args.scorer, "secs": args.secs,
                      "gap_s": args.gap_s, "templates": active_templates},
             "scores": cache}, indent=2), encoding="utf-8")

    for i, p in enumerate(todo, 1):
        a = resolve(p["ref_path"], args.audio_dir)
        b = resolve(p["cand_path"], args.audio_dir)
        try:
            cache[p["pair_id"]] = coherence_score(a, b, args.gap_s, args.secs,
                                                  args.readout, args.scorer)
        except Exception as exc:  # noqa: BLE001
            logging.error("pair %s failed: %s", p["pair_id"], exc,
                          exc_info=args.verbose); continue
        if i % 25 == 0 or i == len(todo):
            flush(); logging.info("scored %d/%d (last S=%.3f)", i, len(todo), cache[p["pair_id"]])
    flush()
    logging.info("done -> %s (%d scores)", args.out, len(cache))
    return 0


if __name__ == "__main__":
    sys.exit(main())
