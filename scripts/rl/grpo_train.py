#!/usr/bin/env python
"""GRPO fine-tuning of MusicGen-small against C. The loop the proposal costs out.

    r_i = log C(A, B_i) - lambda_copy * copy(A, B_i)          (reward.py)
    A_i = (r_i - mean r) / sd r                                (group-relative)
    L   = -mean_t(A_i * logp_t) + beta * KL(pi || pi_ref)

One group = G rollouts from one context, so the advantage is computed within a context and
the per-track scale of C never enters the gradient. That is the property GRPO is chosen for
here; absolute C varies far more across tracks than within one.

Four things that are load-bearing and easy to get wrong:

1. **CFG must be 1.0 during RL.** With guidance > 1 the sampler draws from
   `p_cond^g * p_uncond^(1-g)`, which is not the policy whose log-probs the loss
   differentiates, so the on-policy assumption breaks and the gradient is simply wrong.
   It also halves generation cost, but that is a side effect, not the reason. The frozen
   baseline this run is compared against must therefore also be generated at CFG 1.0.

2. **Sampled tokens are captured, not reconstructed.** `MusicgenForConditionalGeneration
   .generate` returns audio, not codes, and re-encoding that audio would not recover the
   tokens exactly. A `StoppingCriteria` sees the running `decoder_input_ids` at every step,
   which is the sequence the log-prob recomputation needs, delay pattern and all.
   `--check` verifies the capture by decoding it back and comparing with generate's audio.

3. **The reference policy is free.** `disable_adapter()` gives the frozen model in the same
   weights, so KL costs a forward pass and no VRAM.

4. **VRAM: the two transients must be sized against what is resident.** Measured
   2026-08-07 on a 24 GB 4090: MF is 15.4 GB, generation at G=8 peaks 6.6 GB above the
   policy's 1.2 GB, and a whole-group backward peaks 7.4 GB above it. The step is ordered
   generate-all -> score-all -> backward-all so that `--offload-mf` moves MF across the bus
   once per step rather than once per context.

   The two levers are real levers, but only because `grpo_backward` calls `backward()`
   *inside* the micro-batch loop. Accumulating the group's graph and backwarding once --
   which is what the first version did -- makes `--micro-batch` a no-op for memory and
   OOMs at G=8 with MF resident. `--gen-batch` bounds generation's transient the same way,
   since that one is Encodec decoding the group to fp32 waveform.

Usage (GPU box, mfenv, flat layout):
    export HF_HOME=/root/autodl-tmp/hf HF_ENDPOINT=https://hf-mirror.com
    export T2M_DATA_ROOT=/root/autodl-tmp

    python grpo_train.py --check                    # ~5 min: does it run, and what VRAM
    python grpo_train.py --steps 5 --group 4        # smoke: 5 real steps end to end
    python grpo_train.py --steps 150 --group 8 --contexts-per-step 2 --out results/rl/grpo_pilot
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (the GPU box): siblings are already importable

SR = 32000          # MusicGen's native rate
FRAME_RATE = 50     # audio tokens per second


# --------------------------------------------------------------------------- capture

class _CaptureIds:
    """StoppingCriteria that records the running decoder_input_ids and never stops.

    generate() appends the sampled token and then calls the criteria, so the last call
    holds the complete sequence: prompt codes, delay-pattern pads and sampled tokens, in
    the exact layout the decoder was fed. `start_len` is the length before the first
    sampled token, which is where reward-bearing positions begin.
    """

    def __init__(self):
        self.seq = None
        self.start_len = None

    def __call__(self, input_ids, scores, **kwargs):
        if self.start_len is None:
            self.start_len = int(input_ids.shape[-1]) - 1
        self.seq = input_ids
        import torch
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


# ---------------------------------------------------------------------------- policy

class Policy:
    """MusicGen-small + LoRA on the decoder, with sampling and log-prob recomputation."""

    def __init__(self, model_id: str, device: str, lora_r: int = 16,
                 lora_alpha: int = 32, dtype: str = "bf16",
                 grad_checkpointing: bool = True):
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor, MusicgenForConditionalGeneration

        self.torch = torch
        self.device = device
        td = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(model_id)
        base = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=td)
        self.load_s = time.time() - t0

        # Only the audio decoder is trained. q_proj/v_proj exist only there: the T5 text
        # encoder names its projections q/v and Encodec has no attention, so this pattern
        # cannot leak into either. They are frozen explicitly regardless.
        for p in base.text_encoder.parameters():
            p.requires_grad_(False)
        for p in base.audio_encoder.parameters():
            p.requires_grad_(False)

        # Encodec stays in fp32 whatever the policy dtype. Two reasons: its codes define
        # the context the policy continues from and the audio the reward is computed on,
        # so bf16 convolutions quantise both ends of the loop for no saving (it is ~30 M
        # params); and transformers passes the processor's fp32 `input_values` straight
        # into it without casting, so a bf16 Encodec raises "Input type (float) and bias
        # type (BFloat16) should be the same" inside generate().
        base.audio_encoder.to(torch.float32)
        self.audio_dtype = next(base.audio_encoder.parameters()).dtype

        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "v_proj"])
        self.model = get_peft_model(base, cfg).to(device)
        if grad_checkpointing:
            self.model.base_model.model.decoder.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()

        gc = self.model.base_model.model.generation_config
        self.pad_id = int(gc.pad_token_id if gc.pad_token_id is not None else 2048)
        self.n_codebooks = int(self.model.base_model.model.decoder.num_codebooks)

    # -- sampling ----------------------------------------------------------------

    def sample_group(self, y_a: np.ndarray, text: str, group: int, generate_s: float,
                     guidance: float = 1.0, temperature: float = 1.0, top_k: int = 250,
                     gen_batch: int = 0):
        """Sample `group` continuations of one context. Returns (audio, token ids, start).

        audio: (group, n_samples) float32, the continuation only, prompt stripped.
        seq:   (group * n_codebooks, L) long, what the decoder was actually fed.
        """
        torch = self.torch
        from transformers import StoppingCriteriaList

        if guidance > 1.0:
            raise SystemExit(
                "guidance_scale > 1 makes the sampler a different distribution from the "
                "policy being differentiated; the GRPO gradient would be wrong. Generate "
                "the frozen baseline at CFG 1.0 too and compare like with like.")

        inputs = self.processor(audio=[y_a] * group, sampling_rate=SR,
                                text=[text] * group, padding=True,
                                return_tensors="pt").to(self.device)
        # generate() feeds input_values to the audio encoder uncast, so match it here
        # rather than relying on the two happening to agree.
        inputs["input_values"] = inputs["input_values"].to(self.audio_dtype)

        # Generation's transient VRAM is dominated by Encodec decoding the whole group to
        # waveform in fp32, so it scales with the sub-batch, not with the sampling. This
        # is the lever to pull first on OOM: it costs a little throughput and nothing else,
        # because the rollouts are independent.
        gb = gen_batch if gen_batch and gen_batch > 0 else group
        conts, seqs, start_len = [], [], None
        n_ctx = y_a.shape[0]
        for lo in range(0, group, gb):
            hi = min(lo + gb, group)
            sub = {k: v[lo:hi] for k, v in inputs.items()}
            cap = _CaptureIds()
            with torch.no_grad():
                audio = self.model.generate(
                    **sub, do_sample=True, guidance_scale=guidance,
                    temperature=temperature, top_k=top_k,
                    max_new_tokens=int(generate_s * FRAME_RATE),
                    stopping_criteria=StoppingCriteriaList([cap]))

            if cap.seq is None:
                raise SystemExit("no tokens captured: this transformers build does not "
                                 "call stopping criteria during MusicGen sampling")

            wav = audio[:, 0].float().cpu().numpy()
            c = wav[:, n_ctx:] if wav.shape[1] > n_ctx else wav
            c = c[:, :int(generate_s * SR)]
            if c.shape[1] == 0:
                raise SystemExit("empty continuation: generate() did not return the "
                                 "prompt as a prefix on this transformers version")
            conts.append(c)
            # rows are (batch * codebooks) in batch-major order, so plain concatenation
            # keeps sub-batches in the same layout a single call would have produced
            seqs.append(cap.seq.detach())
            start_len = cap.start_len

        return (np.concatenate(conts, axis=0), torch.cat(seqs, dim=0), start_len, inputs)

    # -- the delay pattern -------------------------------------------------------

    def valid_span(self, seq_len: int) -> int:
        """Tokens per codebook that survive the delay pattern. See `_delay_mask`."""
        return seq_len - self.n_codebooks

    def _delay_mask(self, seq_len: int, start_len: int, device):
        """(K, L) mask of positions the policy actually chose *and* that reach the audio.

        MusicGen offsets codebook k by k steps, so in a captured sequence of length L,
        codebook k occupies positions [k+1, k+1+L-K): position 0 is the bos and positions
        1..k are the leading delay pad. Of that span, the first `start_len - 1` entries are
        the audio prompt, forced by the delay mask rather than sampled, and the K-1-k
        entries past the end are tokens the sampler drew but `generate` discards when it
        re-applies the mask before decoding.

        Both ends have to come out of the loss. The tail tokens influence no audio and so
        no reward, and the head tokens are prompt content the policy did not choose;
        including either puts gradient on decisions that are not decisions. Filtering on
        `!= pad_token_id` catches only the head, which is what left the codebooks ragged
        (1000/999/998/997 rather than 997 each).
        """
        torch = self.torch
        pos = torch.arange(seq_len, device=device)[None, :]         # (1, L)
        k = torch.arange(self.n_codebooks, device=device)[:, None]  # (K, 1)
        return (pos >= start_len + k) & (pos < k + 1 + self.valid_span(seq_len))

    def undelay(self, seq):
        """Captured ids -> (1, group, K, n_valid) codes, the layout audio_encoder wants."""
        torch = self.torch
        K, L = self.n_codebooks, seq.shape[1]
        n = self.valid_span(L)
        x = seq.reshape(-1, K, L)
        return torch.stack([x[:, k, k + 1:k + 1 + n] for k in range(K)], dim=1)[None]

    # -- log-probs ---------------------------------------------------------------

    def _chunk_logp(self, seq, inputs, dmask, lo: int, hi: int):
        """Per-codebook log-probs for rollouts [lo, hi). Both (mb, K, L-1).

        The codebook axis is kept rather than summed because the two consumers need
        different things from it: the policy-gradient term wants log pi of the whole
        frame, which is the sum, while KL is additive over the codebooks and must be
        estimated per codebook (see `grpo_backward`).
        """
        torch = self.torch
        K = self.n_codebooks
        sl = slice(lo * K, hi * K)
        # use_cache=False is not an optimisation: a KV cache silently disables gradient
        # checkpointing in the decoder, which is most of the activation saving.
        out = self.model(input_ids=inputs["input_ids"][lo:hi],
                         attention_mask=inputs["attention_mask"][lo:hi],
                         decoder_input_ids=seq[sl], use_cache=False)
        logits = out.logits[:, :-1].float()          # (mb*K, L-1, V)
        tgt = seq[sl][:, 1:]                         # (mb*K, L-1)
        m = dmask.repeat(hi - lo, 1)                 # (mb*K, L-1)
        if (tgt[m] == self.pad_id).any():
            raise RuntimeError(
                "a pad token is inside the scored span: the delay-pattern arithmetic in "
                "_delay_mask does not match this transformers build. Do not train on "
                "this -- the loss would be scoring positions the policy never chose.")
        # lm_head emits vocab_size (2048) logits, but pad/bos is 2048 -- one past the end,
        # which is what the config warnings at load are about. Masked-out positions still
        # hold it, and gather indexes before the mask is applied, so they need a valid
        # index first or the CUDA kernel asserts. Their contribution is zeroed immediately.
        safe = tgt.masked_fill(~m, 0)
        lp = torch.log_softmax(logits, dim=-1).gather(-1, safe[..., None])[..., 0]
        return ((lp * m).reshape(hi - lo, K, -1),
                m.reshape(hi - lo, K, -1))

    def logprobs(self, seq, inputs, micro_batch: int, start_len: int, with_grad: bool):
        """Per-token log-probs over the whole group. Diagnostics only -- see the warning.

        WARNING: with `with_grad=True` this keeps every micro-batch's graph alive until
        the caller backwards, so peak memory scales with the group and `micro_batch` buys
        nothing. Training must use `grpo_backward`, which backwards inside the loop.
        """
        torch = self.torch
        group = seq.shape[0] // self.n_codebooks
        dmask = self._delay_mask(seq.shape[1], start_len, seq.device)[:, 1:]
        outs, masks = [], []
        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            for lo in range(0, group, micro_batch):
                lp, m = self._chunk_logp(seq, inputs, dmask, lo, min(lo + micro_batch, group))
                outs.append(lp.sum(1))          # log pi of the whole frame
                masks.append(m.any(1))
        return torch.cat(outs), torch.cat(masks)

    def grpo_backward(self, seq, inputs, micro_batch: int, start_len: int,
                      adv, beta: float, scale: float = 1.0):
        """Accumulate the GRPO gradient micro-batch by micro-batch. Returns (pg, kl, ntok).

        `backward()` is called **inside** the loop. That is the entire point of the method:
        building the whole group's graph and backwarding once makes `micro_batch` a no-op
        for memory, which is what OOMed a 24 GB card at G=8 alongside a resident reward
        model. Normalising by the group's total token count rather than the micro-batch's
        keeps the accumulated gradient identical to the single-backward version.
        """
        torch = self.torch
        K, L = self.n_codebooks, seq.shape[1]
        group = seq.shape[0] // K
        dmask = self._delay_mask(L, start_len, seq.device)[:, 1:]
        ntok = max(int(dmask.any(0).sum()) * group, 1)

        pg_tot = kl_tot = 0.0
        for lo in range(0, group, micro_batch):
            hi = min(lo + micro_batch, group)
            with torch.no_grad(), self.model.disable_adapter():
                lp_ref, _ = self._chunk_logp(seq, inputs, dmask, lo, hi)   # (mb, K, L-1)
            lp, m = self._chunk_logp(seq, inputs, dmask, lo, hi)

            # Fully on-policy: one optimiser step per batch of rollouts, so the PPO ratio
            # is identically 1 and the clipped surrogate reduces to the policy-gradient
            # term. log pi of a frame is the sum over its codebooks.
            pg = -(adv[lo:hi, None] * lp.sum(1) * m.any(1)).sum() / ntok
            # k3 estimator, applied PER CODEBOOK and then summed. MusicGen samples the K
            # codebooks of a frame independently, so the frame's KL is the sum of theirs;
            # applying k3 to the codebook-summed difference instead inflates it by roughly
            # K (exp is convex) and makes the gradient hostage to rare large-|d| frames.
            d = (lp_ref - lp).clamp(-20, 20)
            kl = ((d.exp() - d - 1.0) * m).sum() / ntok
            ((pg + beta * kl) * scale).backward()

            pg_tot += float(pg.detach())
            kl_tot += float(kl.detach())
            del lp, lp_ref, m, pg, kl, d
        return pg_tot, kl_tot, ntok

    def decode_codes(self, seq):
        """Codes -> audio, mirroring what generate() does after sampling. For --check.

        `audio_scales` is required and is one entry per chunk, not per item in the batch.
        MusicGen's Encodec has `chunk_length=None` and `normalize=False`, so there is
        exactly one chunk and its scale is None -- which is what generate() passes too.
        """
        torch = self.torch
        base = self.model.base_model.model
        with torch.no_grad():
            return base.audio_encoder.decode(self.undelay(seq),
                                             audio_scales=[None]).audio_values


# ------------------------------------------------------------------------- helpers

def resolve_audio(rel: str, audio_root: str | None) -> Path:
    """Absolute -> --audio-root -> $T2M_DATA_ROOT -> repo root. Same order as the rest of
    scripts/rl/, because contexts.json stores repo-relative paths to stay portable."""
    p = Path(rel)
    if p.is_absolute():
        return p
    roots = [Path(r) for r in (audio_root, os.environ.get("T2M_DATA_ROOT")) if r]
    roots.append(ROOT)
    for r in roots:
        if (r / p).exists():
            return r / p
    raise FileNotFoundError(f"{rel} not found under any of: "
                            + ", ".join(str(r) for r in roots)
                            + "\nPass --audio-root or set T2M_DATA_ROOT.")


def prompt_for(ctx: dict) -> str:
    """Same minimal conditioning as the frozen baseline; the prompt must not become a
    second variable."""
    return f"instrumental {ctx['genre']} music"


def load_context_audio(ctx: dict, audio_root: str | None) -> np.ndarray:
    import librosa
    dur = ctx["context_end_s"] - ctx["context_start_s"]
    y, _ = librosa.load(resolve_audio(ctx["audio"], audio_root), sr=SR, mono=True,
                        offset=ctx["context_start_s"], duration=dur)
    return y


def vram(torch, device: str) -> tuple[float, float]:
    if not device.startswith("cuda"):
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 2**30,
            torch.cuda.max_memory_allocated() / 2**30)


# --------------------------------------------------------------------------- check

def run_check(a) -> int:
    """Prove every risky assumption separately, so a failure names itself.

    Also produces the one budget figure the GPU proposal still marks unmeasured: peak
    VRAM of a LoRA backward pass over a full rollout group.
    """
    import torch

    contexts = json.loads(Path(a.contexts).read_text())["contexts"]
    ctx = [c for c in contexts if c["split"] == "train"][0]

    print("=" * 72)
    print("1. policy + LoRA")
    pol = Policy(a.model, a.device, a.lora_r, a.lora_alpha, a.dtype, a.grad_checkpointing)
    print(f"   loaded in {pol.load_s:.1f}s | codebooks={pol.n_codebooks} pad={pol.pad_id}"
          f" | VRAM {vram(torch, a.device)[0]:.2f} GB")

    print("\n2. sampling a group + capturing tokens")
    y_a = load_context_audio(ctx, a.audio_root)
    torch.cuda.reset_peak_memory_stats() if a.device.startswith("cuda") else None
    t0 = time.time()
    cont, seq, start_len, inputs = pol.sample_group(
        y_a, prompt_for(ctx), a.group, ctx["generate_s"], a.guidance, a.temperature,
        a.top_k, a.gen_batch)
    dt = time.time() - t0
    print(f"   {a.group} rollouts in {dt:.1f}s ({dt/a.group:.2f}s each)"
          f" | audio {cont.shape} | tokens {tuple(seq.shape)} start_len={start_len}"
          f" | peak VRAM {vram(torch, a.device)[1]:.2f} GB")

    print("\n3. captured tokens decode back to the audio generate() returned")
    # Compare the continuation only. The prompt prefix is a codec round trip on both
    # sides, so including it would measure Encodec's reconstruction error rather than
    # whether the capture is right.
    audio2 = pol.decode_codes(seq)[:, 0, :].float().cpu().numpy()
    n_ctx = y_a.shape[0]
    gen = audio2[:, n_ctx:n_ctx + cont.shape[1]]
    print(f"   decoded {audio2.shape[1]} samples = {pol.valid_span(seq.shape[1])} frames "
          f"per codebook; continuation slice {gen.shape}, generate() gave {cont.shape}")
    if gen.shape != cont.shape:
        print("   MISMATCH in length -- the delay-pattern arithmetic is wrong. Stop here.")
        return 1
    r = float(np.corrcoef(gen.ravel(), cont.ravel())[0, 1])
    print(f"   correlation {r:.6f}  ({'OK' if r > 0.99 else 'MISMATCH -- capture is wrong'})")
    if r <= 0.99:
        print("   the token capture does not correspond to the returned audio; the "
              "log-prob recomputation would be scoring the wrong sequence. Stop here.")
        return 1

    print("\n4. GRPO backward (the unmeasured budget figure)")
    # Exercise grpo_backward, not a hand-rolled single backward: it is the path training
    # takes, and it is the one whose peak memory --micro-batch actually controls.
    if a.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    n_scored = int(pol._delay_mask(seq.shape[1], start_len, seq.device)[:, 1:].sum())
    print(f"   {n_scored * a.group} tokens "
          f"({n_scored // pol.n_codebooks} per codebook per rollout, delay pattern "
          f"removed from both ends)")
    t0 = time.time()
    adv = torch.randn(a.group, device=a.device)
    pg, kl, ntok = pol.grpo_backward(seq, inputs, a.micro_batch, start_len, adv, a.beta)
    print(f"   pg {pg:+.5f}  kl {kl:.6f} (must be ~0 at init)  over {ntok} frame positions"
          f" | {time.time()-t0:.1f}s")
    print(f"   peak VRAM {vram(torch, a.device)[1]:.2f} GB   <-- training VRAM, "
          f"G={a.group} micro-batch={a.micro_batch}")
    pol.model.zero_grad(set_to_none=True)

    print("\n5. reference policy (adapters disabled) matches the policy at init")
    logp, mask = pol.logprobs(seq, inputs, a.micro_batch, start_len, with_grad=False)
    with pol.model.disable_adapter():
        lp_ref, _ = pol.logprobs(seq, inputs, a.micro_batch, start_len, with_grad=False)
    nt = int(mask.sum())
    d = float(((logp - lp_ref) * mask).sum() / max(nt, 1))
    print(f"   mean logp {float((logp * mask).sum() / max(nt, 1)):.4f} | "
          f"mean (logp - logp_ref) = {d:+.6f}  "
          f"({'OK, ~0 at init as expected' if abs(d) < 1e-3 else 'nonzero: LoRA is not at init'})")

    print("\n6. reward path")
    from reward import ConstantS, CoherenceReward
    base = ROOT / "results/rl/copy_reference_continuation.json"
    rw = CoherenceReward(s_backend=ConstantS(0.5),
                         copy_baseline=base if base.exists() else None,
                         lambda_copy=a.lambda_copy)
    if not rw.dims:
        print("   H/T/R unavailable in this env (librosa missing) -- reward would be S only")
    else:
        out = rw(y_a, cont[0], SR, pair_id="check")
        print(f"   H={out['dims']['H']:.3f} T={out['dims']['T']:.3f} R={out['dims']['R']:.3f}"
              f" S=(constant) | penalty {out['penalty']:.4f} | reward {out['reward']:+.4f}")

    if a.check_mf:
        print("\n7. Music Flamingo resident alongside the policy")
        from reward import MusicFlamingoS
        t0 = time.time()
        sb = MusicFlamingoS(secs=a.s_secs, gap_s=a.s_gap, token_agg=a.token_agg)
        print(f"   loaded in {time.time()-t0:.0f}s | VRAM now {vram(torch, a.device)[0]:.2f} GB")
        t0 = time.time()
        vals = [sb.score(y_a, cont[i], SR, pair_id=f"chk{i}") for i in range(a.group)]
        print(f"   S over the group: {' '.join(f'{v:.3f}' for v in vals)}"
              f" | {(time.time()-t0)/a.group:.2f}s each"
              f" | peak VRAM {vram(torch, a.device)[1]:.2f} GB   <-- everything resident")

    print("\n" + "=" * 72)
    print("check passed. The numbers to carry into the proposal are the two marked <--.")
    return 0


# ---------------------------------------------------------------------------- train

def run_train(a) -> int:
    import soundfile as sf
    import torch

    from reward import CoherenceReward, MusicFlamingoS, group_advantages

    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"

    doc = json.loads(Path(a.contexts).read_text())
    train_ctx = [c for c in doc["contexts"] if c["split"] == "train"]
    if not train_ctx:
        raise SystemExit("no train-split contexts in " + a.contexts)
    rng = np.random.default_rng(a.seed)

    pol = Policy(a.model, a.device, a.lora_r, a.lora_alpha, a.dtype, a.grad_checkpointing)
    torch.manual_seed(a.seed)
    opt = torch.optim.AdamW([p for p in pol.model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.0)

    # --- resume ---------------------------------------------------------------------
    # save_pretrained writes the adapter and nothing else, so without the state below a
    # crashed run could only be restarted with a fresh optimiser and a re-drawn context
    # order. That is a different protocol, not a continuation, and it lands exactly where
    # a seed-robustness claim is made. Everything the loop carries is restored here.
    start_step = 0
    if a.resume:
        ckpt = Path(a.resume)
        st_path = ckpt / "trainer_state.pt"
        if not st_path.exists():
            raise SystemExit(
                f"{st_path} not found. Checkpoints written before --resume existed hold "
                "adapter weights only, and resuming from one would reset the optimiser "
                "and re-draw the context order. Restart from step 0 instead.")
        st = torch.load(st_path, map_location=a.device, weights_only=False)
        missing = [k for k, v in pol.model.named_parameters()
                   if v.requires_grad and k not in st["trainable"]]
        if missing:
            raise SystemExit(f"checkpoint is missing {len(missing)} trainable tensors; "
                             "was it written with the same --lora-r/--lora-alpha?")
        with torch.no_grad():
            for k, v in pol.model.named_parameters():
                if v.requires_grad:
                    v.copy_(st["trainable"][k].to(v.device, v.dtype))
        opt.load_state_dict(st["opt"])
        rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"].cpu())
        if st.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in st["cuda_rng"]])
        start_step = int(st["step"])
        if start_step >= a.steps:
            raise SystemExit(f"checkpoint is already at step {start_step} of {a.steps}")
        print(f"resumed from {ckpt} at step {start_step}; "
              f"{a.steps - start_step} steps remain", flush=True)

    baseline = Path(a.copy_baseline)
    if not baseline.exists():
        raise SystemExit(
            f"{baseline} not found. The copy hinge comes from real music continuing "
            "itself (rlhf_with_C_procedure.md §5); without it the penalty charges every "
            "rollout outright and the reward is not the one that was designed.")

    if a.reward == "clap":
        # Arm B of the matched pair (rlhf_with_C_procedure.md §5b). Music Flamingo is
        # deliberately NOT constructed: skipping it frees the 16.66 GB that pins the C arm
        # to --gen-batch 4, which is the whole reason this arm is ~2.6 h and not ~4.6 h.
        # S is scored post hoc on the checkpoints by eval_grpo.py.
        from reward import ClapReward, ClapScorer
        reward_fn = ClapReward(clap=ClapScorer(secs=a.clap_secs),
                               lambda_copy=a.lambda_copy, copy_baseline=baseline)
        if a.offload_mf:
            # --offload-mf parks Music Flamingo on the CPU between phases; there is no MF
            # on this arm, and the phase-1/2/3 blocks that honour the flag dereference an
            # s_backend that was never built.
            print("--offload-mf ignored: --reward clap loads no Music Flamingo.",
                  flush=True)
            a.offload_mf = False
        if a.lambda_copy == 2.0:
            print("WARNING: --lambda-copy 2.0 was calibrated against log C's within-group "
                  "spread (~0.20). log CLAP is flatter, so at 2.0 the copy penalty can "
                  "swamp the objective and the policy learns only to stop copying. Read "
                  "objective_spread off the first few steps and set "
                  "--lambda-copy 7*spread/1.41 before trusting a full run (§5b).",
                  flush=True)
    else:
        s_backend = MusicFlamingoS(secs=a.s_secs, gap_s=a.s_gap, precision=a.mf_precision,
                                   token_agg=a.token_agg)
        reward_fn = CoherenceReward(s_backend=s_backend, lambda_copy=a.lambda_copy,
                                    copy_baseline=baseline,
                                    weights=dict(H=1.0, T=1.0, R=1.0,
                                                 S=0.0 if a.reward == "C_noS" else 1.0))
    if not reward_fn.dims:
        raise SystemExit("H/T/R unavailable: librosa is missing from this env")

    print(f"contexts={len(train_ctx)} steps={a.steps} group={a.group} "
          f"x{a.contexts_per_step}/step -> {a.steps*a.group*a.contexts_per_step} rollouts")

    ctx_audio: dict[str, np.ndarray] = {}
    # Reward on the first visit to each context. Absolute C varies far more between tracks
    # than within one -- that is why the advantage is group-relative -- so a raw reward
    # trace over sampled contexts is mostly context lottery and cannot be read for
    # learning. Reporting each group against its own first visit removes that, at the cost
    # of being 0 by construction the first time a context appears.
    ctx_first: dict[str, float] = {}
    ctx_visits: dict[str, int] = {}
    kl_recent: list[float] = []
    if a.resume:
        ctx_first = {k: float(v) for k, v in st.get("ctx_first", {}).items()}
        ctx_visits = {k: int(v) for k, v in st.get("ctx_visits", {}).items()}
        kl_recent = [float(x) for x in st.get("kl_recent", [])]
    t_start = time.time()
    for step in range(start_step + 1, a.steps + 1):
        opt.zero_grad(set_to_none=True)
        step_log = {"step": step, "groups": []}
        t_step = time.time()

        chosen = list(rng.choice(train_ctx, size=a.contexts_per_step, replace=False))

        # Phase 1 -- generate every group for the step, with MF off the GPU.
        # The step is split into three phases rather than run per context so that Music
        # Flamingo crosses the PCIe bus once per step instead of once per context. At the
        # measured rates that matters: a group takes ~10 s to generate and MF is 15.4 GB,
        # so a per-context swap would be a large fraction of the step.
        if a.offload_mf:
            s_backend.mf.model.to("cpu")
            torch.cuda.empty_cache()
        t0 = time.time()
        batches = []
        for c in chosen:
            cid = c["context_id"]
            if cid not in ctx_audio:
                ctx_audio[cid] = load_context_audio(c, a.audio_root)
            cont, seq, start_len, inputs = pol.sample_group(
                ctx_audio[cid], prompt_for(c), a.group, c["generate_s"],
                a.guidance, a.temperature, a.top_k, a.gen_batch)
            batches.append({"c": c, "cid": cid, "cont": cont, "seq": seq,
                            "start_len": start_len, "inputs": inputs})
        t_gen = time.time() - t0

        # Phase 2 -- MF on, score every group, MF off.
        if a.offload_mf:
            s_backend.mf.model.to(a.device)
        t0 = time.time()
        for b in batches:
            b["outs"] = [reward_fn(ctx_audio[b["cid"]], b["cont"][i], SR,
                                   pair_id=f"{b['cid']}_s{step}_r{i}")
                         for i in range(a.group)]
        t_rew = time.time() - t0
        if a.offload_mf:
            s_backend.mf.model.to("cpu")
            torch.cuda.empty_cache()

        # Phase 3 -- one backward per group, gradients accumulated across the step.
        for b in batches:
            c, cid, cont, outs = b["c"], b["cid"], b["cont"], b["outs"]
            seq, start_len, inputs = b["seq"], b["start_len"], b["inputs"]

            adv = torch.tensor(group_advantages([o["reward"] for o in outs]),
                               dtype=torch.float32, device=a.device)

            pg, kl, _ = pol.grpo_backward(seq, inputs, a.micro_batch, start_len, adv,
                                          a.beta, 1.0 / a.contexts_per_step)
            b["seq"] = b["inputs"] = None      # free the group before the next one

            r_mean = float(np.mean([o["reward"] for o in outs]))
            ctx_first.setdefault(cid, r_mean)
            ctx_visits[cid] = ctx_visits.get(cid, 0) + 1

            step_log["groups"].append({
                "context_id": cid,
                "visit": ctx_visits[cid],
                "reward": {"mean": r_mean,
                           "rel": r_mean - ctx_first[cid],
                           "max": float(np.max([o["reward"] for o in outs])),
                           "sd": float(np.std([o["reward"] for o in outs]))},
                "C": float(np.mean([o["C"] for o in outs])),
                # sd of the objective within the group, in log space. This is what
                # --lambda-copy has to be calibrated against for a new reward: §5 fixed
                # 2.0 so the hack's penalty (1.41) sits ~7x log C's spread (~0.20).
                "objective_spread": float(np.std([o["log_C"] for o in outs])),
                # keys vary by arm: C/C_noS give H,T,R,S; clap gives H,T,R,clap_htsat.
                "dims": {k: float(np.mean([o["dims"][k] for o in outs]))
                         for k in outs[0]["dims"]},
                "copy": {k: float(np.mean([o["copy"][k] for o in outs]))
                         for k in ("xcorr_wave", "xcorr_mel")},
                "penalty": float(np.mean([o["penalty"] for o in outs])),
                "kl": kl, "pg": pg,
            })

            if step % a.sample_every == 0:
                best = int(np.argmax([o["reward"] for o in outs]))
                sf.write(out / "samples" / f"step{step:04d}_{cid}_best.wav",
                         cont[best], SR)

        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in pol.model.parameters() if p.requires_grad], a.max_grad_norm)
        opt.step()

        step_log["grad_norm"] = float(gn)
        step_log["t_gen_s"] = round(t_gen, 1)
        step_log["t_reward_s"] = round(t_rew, 1)
        step_log["t_step_s"] = round(time.time() - t_step, 1)
        _sb = getattr(reward_fn, "s", None) or getattr(reward_fn, "clap", None)
        step_log["s_cache"] = {"calls": getattr(_sb, "calls", 0),
                               "hits": getattr(_sb, "hits", 0)}
        step_log["peak_vram_gb"] = round(vram(torch, a.device)[1], 2)
        with log_path.open("a") as f:
            f.write(json.dumps(step_log) + "\n")

        g = step_log["groups"]
        # the second dimension shown is the one the arm is NOT trivially optimising:
        # S for the C arms, and (since the clap arm does not compute S) T for arm B.
        probe = "S" if "S" in g[0]["dims"] else "T"
        print(f"[{step:4d}/{a.steps}] r={np.mean([x['reward']['mean'] for x in g]):+.4f} "
              f"rel={np.mean([x['reward']['rel'] for x in g]):+.4f} "
              f"C={np.mean([x['C'] for x in g]):.4f} "
              f"sd={np.mean([x['objective_spread'] for x in g]):.3f} "
              f"{probe}={np.mean([x['dims'][probe] for x in g]):.3f} "
              f"copy={np.mean([x['copy']['xcorr_wave'] for x in g]):.3f} "
              f"KL={np.mean([x['kl'] for x in g]):.4f} "
              f"|g|={float(gn):.2f} {step_log['t_step_s']:.0f}s "
              f"(gen {t_gen:.0f} rew {t_rew:.0f})")

        if step % a.checkpoint_every == 0 or step == a.steps:
            ck = out / "checkpoints" / f"step{step:04d}"
            pol.model.save_pretrained(ck)
            # Everything --resume needs. The adapter alone is enough for eval_grpo.py,
            # which only samples from the policy, and not enough to continue training.
            torch.save({
                "step": step,
                "trainable": {k: v.detach().cpu()
                              for k, v in pol.model.named_parameters() if v.requires_grad},
                "opt": opt.state_dict(),
                "np_rng": rng.bit_generator.state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                "ctx_first": ctx_first,
                "ctx_visits": ctx_visits,
                "kl_recent": kl_recent,
                "args": vars(a),
            }, ck / "trainer_state.pt")
            print(f"          checkpoint -> {ck}")

        # Divergence guard. The 2026-08-07 pilot improved to step ~180 and then came apart:
        # KL went 0.21 -> 1.21, gradient norms 0.7 -> 5 (62 at one step), and reward fell
        # below where it started. Everything after that was wasted GPU time, and the run
        # had to be read backwards from the log to find the good checkpoint. A run that has
        # left the reference distribution this far is not coming back, so stop and say so.
        kl_recent.append(np.mean([x["kl"] for x in g]))
        if a.kl_max > 0 and len(kl_recent) >= 10:
            run_kl = float(np.mean(kl_recent[-10:]))
            if run_kl > a.kl_max:
                ck = out / "checkpoints" / f"step{step:04d}_klstop"
                pol.model.save_pretrained(ck)
                print(f"\nSTOPPING: KL over the last 10 steps is {run_kl:.3f}, above "
                      f"--kl-max {a.kl_max}. The policy has left the reference "
                      f"distribution.\nCheckpoint saved to {ck}. The checkpoint worth "
                      f"evaluating is an earlier one -- find where `rel` stopped being "
                      f"positive in\n{log_path}.")
                break

    (out / "config.json").write_text(json.dumps(
        {**vars(a), "elapsed_h": round((time.time() - t_start) / 3600, 2),
         "resumed_from_step": start_step or None}, indent=2))
    print(f"\ndone in {(time.time()-t_start)/3600:.2f} h -> {out}")
    return 0


# ----------------------------------------------------------------------------- cli

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="validate the pipeline and report VRAM; trains nothing")
    ap.add_argument("--check-mf", action="store_true",
                    help="--check also loads Music Flamingo, for the resident-VRAM figure")

    ap.add_argument("--contexts", default=str(ROOT / "results/rl/contexts.json"))
    ap.add_argument("--audio-root", default=None,
                    help="prefix for contexts.json's relative paths (or $T2M_DATA_ROOT)")
    ap.add_argument("--out", default=str(ROOT / "results/rl/grpo_pilot"))
    ap.add_argument("--model", default="facebook/musicgen-small")
    ap.add_argument("--device", default="cuda:0")

    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--group", type=int, default=8, help="G, rollouts per context")
    ap.add_argument("--contexts-per-step", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=2,
                    help="rollouts per backward micro-batch; lower this first on OOM")
    ap.add_argument("--gen-batch", type=int, default=0,
                    help="rollouts per generate() call (0 = the whole group); generation's "
                         "transient VRAM is Encodec decoding to fp32 waveform, so this is "
                         "the lever for generation-side OOM")
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="5e-5 diverged at ~step 200 on the 07/08 pilot")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="KL coefficient; 0.02 was too weak to anchor the 07/08 pilot, "
                         "contributing ~1%% of the loss")
    ap.add_argument("--kl-max", type=float, default=0.5,
                    help="stop if the 10-step mean KL exceeds this; 0 disables. The 07/08 "
                         "pilot diverged through 1.2 and never recovered")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                    action="store_false")

    ap.add_argument("--guidance", type=float, default=1.0,
                    help="must be 1.0 for training; see the module docstring")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=250)

    ap.add_argument("--reward", default="C", choices=("C", "C_noS", "clap"),
                    help="C_noS is the stage-4 ablation: does the policy learn from S? "
                         "clap is arm B of the matched pair (§5b): trains on CLAP-htsat "
                         "instead of C, loads no Music Flamingo, and needs --lambda-copy "
                         "recalibrated -- see ClapReward's docstring.")
    ap.add_argument("--clap-secs", type=float, default=None,
                    help="truncate each side to this many seconds before CLAP embeds it. "
                         "Default None = use the whole rollout, matching what H/T/R see.")
    ap.add_argument("--lambda-copy", type=float, default=2.0)
    ap.add_argument("--copy-baseline",
                    default=str(ROOT / "results/rl/copy_reference_continuation.json"))
    ap.add_argument("--s-secs", type=float, default=6.0)
    ap.add_argument("--s-gap", type=float, default=1.0)
    ap.add_argument("--mf-precision", default="all_bf16")
    ap.add_argument("--token-agg", default="variants", choices=("variants", "single"),
                    help="Yes/No read-out for S. Every RL run in this project used "
                         "`variants`, and a run scored under `single` cannot be pooled "
                         "with them, so this is pinned here rather than inherited from "
                         "whatever mf_probe defaults to on the machine.")
    ap.add_argument("--offload-mf", action="store_true",
                    help="park MF on the CPU except while scoring; ~6 s/step for ~16 GB")

    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260807)
    ap.add_argument("--resume", default="",
                    help="checkpoint directory to continue from, e.g. "
                         ".../checkpoints/step0075. Restores the adapter, the optimiser "
                         "moments, both RNG streams and the step counter, so the run "
                         "continues rather than restarts with a fresh optimiser. Only "
                         "checkpoints written by a build that saves trainer_state.pt can "
                         "be resumed.")
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # The loop allocates and frees a ~7 GB transient every step alongside a 16 GB resident
    # reward model, which is exactly the pattern that fragments the caching allocator.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return run_check(a) if a.check else run_train(a)


if __name__ == "__main__":
    raise SystemExit(main())
