#!/usr/bin/env python
"""Generate continuations with a frozen MusicGen, and time them.

Two jobs in one script, because they need the same code path:

  1. **Timing.** Generation, not S scoring, is what the GRPO budget is made of
     (S measured at <=0.92 s/pair on 2026-08-05). This reports per-rollout seconds and
     peak VRAM so the step count can be scoped against a measured number.
  2. **The frozen-policy baseline.** The copy penalty's hinge should come from the
     distribution RL starts from, i.e. this model's own continuations, not from real
     music (which is the upper reference) and not from the re-ranking windows (whose
     A and B overlap by construction). Output is named so `copy_metrics.py` reads it
     unchanged.

Why transformers and not audiocraft: GRPO needs per-token log-probabilities and
gradients through the policy, which `MusicgenForConditionalGeneration` gives and
audiocraft's sampling API does not. This module is the same generation path the RL loop
will import, so it is not throwaway.

Output layout (matches what `copy_metrics.py` globs):
    <out>/<context_id>__A.wav            the context the model continued from
    <out>/<context_id>_cand<NN>__B.wav   each generated continuation
    <out>/manifest.json                  prompts, timings, config

Usage (GPU box, mfenv or any env with a recent transformers):
    export HF_HOME=/root/autodl-tmp/hf HF_ENDPOINT=https://hf-mirror.com
    python generate_continuations.py --limit 3 --n 8          # timing smoke test
    python generate_continuations.py --n 4                    # full frozen baseline
    python copy_metrics.py --audio-dir results/rl/rollouts_frozen --sr 32000 \
        --out results/rl/copy_baseline_musicgen.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

SR = 32000          # MusicGen's native rate; do not resample the model's output
FRAME_RATE = 50     # audio tokens per second, used to convert seconds -> max_new_tokens


def resolve_audio(rel: str, audio_root: str | None) -> Path:
    """Find a context's audio file.

    `contexts.json` stores repo-relative paths so the manifest is portable, but the GPU
    box keeps the corpus on a different volume (e.g. /root/autodl-tmp) from the scripts.
    Search order: absolute path, --audio-root, $T2M_DATA_ROOT, then the repo root.
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    roots = [Path(r) for r in (audio_root, os.environ.get("T2M_DATA_ROOT")) if r]
    roots.append(ROOT)
    for r in roots:
        if (r / p).exists():
            return r / p
    raise FileNotFoundError(
        f"{rel} not found under any of: " + ", ".join(str(r) for r in roots) +
        "\nPass --audio-root (e.g. --audio-root /root/autodl-tmp) or set T2M_DATA_ROOT.")


def load_model(model_id: str, device: str):
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32).to(device)
    model.eval()
    return processor, model, time.time() - t0


def prompt_for(ctx: dict) -> str:
    """Fixed, minimal text conditioning.

    Deliberately plain: the experiment is about the reward, so the prompt must not become
    a second variable. Genre is the only context-dependent part, and it is recorded in the
    manifest so the choice is auditable.
    """
    return f"instrumental {ctx['genre']} music"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default=str(ROOT / "results/rl/contexts.json"))
    ap.add_argument("--out", default=str(ROOT / "results/rl/rollouts_frozen"))
    ap.add_argument("--audio-root", default=None,
                    help="prefix for the relative audio paths in contexts.json; on the GPU "
                         "box the corpus lives off-repo, e.g. --audio-root /root/autodl-tmp "
                         "(or set T2M_DATA_ROOT)")
    ap.add_argument("--model", default="facebook/musicgen-small")
    ap.add_argument("--n", type=int, default=4, help="continuations per context")
    ap.add_argument("--limit", type=int, default=0, help="first N contexts only")
    ap.add_argument("--split", default="", help="train | heldout | '' for both")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--guidance", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=20260805)
    a = ap.parse_args()

    import librosa
    import soundfile as sf
    import torch

    contexts = json.loads(Path(a.contexts).read_text())["contexts"]
    if a.split:
        contexts = [c for c in contexts if c["split"] == a.split]
    if a.limit:
        contexts = contexts[:a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    processor, model, load_s = load_model(a.model, a.device)
    print(f"loaded {a.model} in {load_s:.1f}s on {a.device}")

    torch.manual_seed(a.seed)
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    rows, times = [], []
    for ci, c in enumerate(contexts, 1):
        dur = c["context_end_s"] - c["context_start_s"]
        y_a, _ = librosa.load(resolve_audio(c["audio"], a.audio_root), sr=SR, mono=True,
                              offset=c["context_start_s"], duration=dur)
        sf.write(out / f"{c['context_id']}__A.wav", y_a, SR)

        text = prompt_for(c)
        max_new = int(c["generate_s"] * FRAME_RATE)

        try:
            inputs = processor(audio=y_a, sampling_rate=SR, text=[text],
                               padding=True, return_tensors="pt").to(a.device)
        except TypeError as e:                    # processor without the audio-prompt path
            raise SystemExit(
                "This transformers build's MusicgenProcessor does not accept audio=.\n"
                "Fall back to encoding the prompt yourself:\n"
                "  codes = model.audio_encoder.encode(wav[None, None]).audio_codes\n"
                "  then pass decoder_input_ids built from `codes` to model.generate().\n"
                f"original error: {e}")

        for k in range(a.n):
            t0 = time.time()
            with torch.no_grad():
                audio = model.generate(**inputs, do_sample=True,
                                       guidance_scale=a.guidance,
                                       max_new_tokens=max_new)
            dt = time.time() - t0
            times.append(dt)

            wav = audio[0, 0].float().cpu().numpy()
            # generate() returns prompt + continuation; keep only what the model added
            wav = wav[y_a.shape[0]:] if wav.shape[0] > y_a.shape[0] else wav
            wav = wav[:int(c["generate_s"] * SR)]
            if wav.size == 0:
                raise SystemExit("empty continuation: check that generate() returned the "
                                 "prompt as a prefix on this transformers version")
            peak = float(np.abs(wav).max())
            if peak > 0:
                wav = 0.95 * wav / peak

            path = out / f"{c['context_id']}_cand{k:02d}__B.wav"
            sf.write(path, wav, SR)
            rows.append({"context_id": c["context_id"], "cand": k, "text": text,
                         "seconds": round(dt, 3), "samples": int(wav.shape[0])})

        print(f"  [{ci}/{len(contexts)}] {c['context_id']}  {a.n} rollouts, "
              f"last {times[-1]:.1f}s")

    t = np.array(times)
    peak_gb = (torch.cuda.max_memory_allocated() / 2**30) if a.device.startswith("cuda") else 0.0
    manifest = {
        "meta": {"model": a.model, "device": a.device, "sr": SR,
                 "n_contexts": len(contexts), "n_per_context": a.n,
                 "guidance_scale": a.guidance, "seed": a.seed,
                 "load_seconds": round(load_s, 1),
                 "rollout_seconds": {"mean": float(t.mean()), "median": float(np.median(t)),
                                     "min": float(t.min()), "max": float(t.max())},
                 "peak_vram_gb": round(peak_gb, 2),
                 "purpose": "timing + frozen-policy copy baseline for the GRPO budget"},
        "rollouts": rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{len(times)} rollouts | mean {t.mean():.2f}s, median {np.median(t):.2f}s "
          f"(min {t.min():.2f}, max {t.max():.2f})")
    print(f"peak VRAM {peak_gb:.2f} GB | model load {load_s:.1f}s")
    print(f"\nbudget: 32,000 rollouts at the median = {32000*np.median(t)/3600:.1f} GPU-h "
          f"per training run")
    print(f"wrote {out}/ ({len(rows)} wavs + manifest.json)")


if __name__ == "__main__":
    main()
