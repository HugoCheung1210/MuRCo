"""
Stable Audio Open — PDI-style masked INPAINTING loop (matches ACE-Step E2).

Regenerates ONLY seconds 5-15; the rest is projected back to the ORIGINAL latent
at every denoising step (PDI-style locality). This makes SAO a true inpainting
model comparable to ACE-Step, and reduces the codec confound (frozen region isn't
re-generated).

Matches your protocol:
  seeds: outputs/seeds/s01..s10.wav (30s)
  region: 5-15s   iterations: 8   naming: outputs_stableaudio_inpaint/{editing,roundtrip}/sNN_iterKK.wav

KNOWN CONFOUND handled (diffusers #10861): init_noise_sigma huge -> initial audio
ignored. We instead do explicit latent-space masking via a per-step callback, so
the frozen region is enforced regardless of init scaling.

!! VERIFY tensor shapes on your diffusers version (### CHECK ###). Latent time axis
   and vae downsample factor must map seconds->latent frames correctly.
"""
import os, json, numpy as np, soundfile as sf, librosa, torch
from diffusers import StableAudioPipeline

# Both are overridable so a deep run can write to a data disk rather than a small
# system disk (the GPU box's /root had ~2 GB free on 2026-08-07). Point SAO_OUT at the
# directory that ALREADY holds the iter01-08 rollout, or the skip-if-exists logic below
# has nothing to resume from and regenerates all N_ITERS from scratch -- which costs
# double and produces audio that no longer matches the reported drift results.
SEEDS_DIR = os.environ.get("SAO_SEEDS", "outputs/seeds")
OUT = os.environ.get("SAO_OUT", "outputs_stableaudio_inpaint")
# T5 deep-horizon: depth is configurable via SAO_N_ITERS (default 8 = the shipped
# rollout, byte-identical). The rollout is SEQUENTIAL (iter_k built from iter_{k-1})
# and the per-iter files are skip-if-exists, so setting SAO_N_ITERS=16 CONTINUES an
# existing iter1-8 rollout from iter8 rather than regenerating it. region/mask are
# unchanged -- only depth grows (spec T5: "keep region/pairing, only depth changes").
N_ITERS = int(os.environ.get("SAO_N_ITERS", "8"))
# Which passes to run. The roundtrip CEILING uses only iter01 (see drift_v4:
# rp=..._iter01.wav), so a deep run needs ONLY the editing pass -- set
# SAO_MODES=editing to skip regenerating unused deep roundtrip clips.
MODES = [m.strip() for m in os.environ.get("SAO_MODES", "editing,roundtrip").split(",") if m.strip()]
# Module DEFAULT edit region; callers may override via inpaint_edit(mask_s=...) (rerank
# passes 5-8s to match outputs_stableaudio_inpaint_5_8). NB: output dirs are created lazily
# in run(), not on import, so merely importing this module no longer litters outputs_*.
#
# !! SAO_MASK EXISTS BECAUSE run() USED TO IGNORE THIS. The drift rollout the thesis
# reports lives in outputs_stableaudio_inpaint_5_8 and was made with a 5-8s mask, but run()
# called inpaint_edit() without mask_s and so used the 5-15s default below. Extending that
# rollout to iter16 without setting SAO_MASK=5,8 would generate iters 9-16 with a 10 s mask
# on a chain whose first 8 used a 3 s one, and the saturation curve T5 is for would be
# measuring the mask change rather than the drift. Always set SAO_MASK to match the
# directory you are resuming into.
MASK_START_S, MASK_END_S = (
    tuple(float(x) for x in os.environ["SAO_MASK"].split(","))
    if os.environ.get("SAO_MASK") else (5.0, 15.0))
DUR_S = 30
STEPS = 100

def load_pipe():
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0",
                                               torch_dtype=torch.float16).to("cuda")
    return pipe

def _encode(pipe, wav_path):
    sr = pipe.vae.sampling_rate                       ### CHECK ### (44100)
    y,_ = librosa.load(wav_path, sr=sr, mono=True, duration=DUR_S)
    dur_s = len(y) / float(sr)                         # ACTUAL seconds loaded (<= DUR_S)
    x = torch.tensor(np.stack([y, y]), dtype=torch.float16)[None].to("cuda")  # [1,2,N]
    with torch.no_grad():
        z = pipe.vae.encode(x).latent_dist.sample()   ### CHECK ###
    return z, sr, dur_s

def inpaint_edit(pipe, in_wav, out_wav, caption, seed=0, mask_s=None):
    """Regenerate mask_s (default MASK_START_S..MASK_END_S); freeze the rest via projection.

    SAO generates a FIXED-length latent (~47.5s / 1024 frames) via timing conditioning,
    NOT a duration-scaled one -- so z0 (the 30s-clip encoding, ~645 frames) is shorter than
    the sampler's `latents`.  We therefore build the mask + padded anchor lazily from the
    REAL latent length at callback time, and freeze IN PLACE (the legacy diffusers
    `callback(i,t,latents)` ignores the return value -- reassignment would be a silent no-op).
    Frozen = the seed content OUTSIDE the edit window; the silent tail (beyond the seed,
    trimmed off by audio_end_in_s) is left free so it isn't pinned to a wrong constant.
    """
    ms0, ms1 = mask_s if mask_s is not None else (MASK_START_S, MASK_END_S)
    renoise = os.environ.get("SAO_RENOISE", "1") not in ("0", "false", "False")  # SAO_RENOISE=0 -> clean anchor (old)
    z0, sr, dur_s = _encode(pipe, in_wav)             # clean seed latent [1,C,T0]
    T0 = z0.shape[-1]
    latent_rate = T0 / dur_s                           # latent frames per second (~21.5 Hz)
    gen = torch.Generator("cuda").manual_seed(seed)
    state = {}                                         # {mask, z0} built on first callback

    def cb(step, t, latents):                          # legacy diffusers callback(i, t, latents)
        if "mask" not in state:
            L = latents.shape[-1]                       # true sampler latent length (e.g. 1024)
            f0 = max(0, int(round(ms0 * latent_rate)))
            f1 = min(L, int(round(ms1 * latent_rate)))
            keep = min(T0, L)                           # seed occupies frames [0, keep)
            mask = torch.ones_like(latents)             # 1 = regenerate/evolve
            mask[..., 0:f0]    = 0.0                     # freeze seed BEFORE the edit window
            mask[..., f1:keep] = 0.0                     # freeze seed AFTER the edit window
            z0f = torch.zeros_like(latents)
            z0f[..., :keep] = z0[..., :keep].to(latents.dtype)
            # fixed noise (own generator, so we don't disturb the pipe's sampling stream) to
            # re-noise the clean anchor to each step's level -- fixes the repaint boundary glitch.
            noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device,
                                generator=torch.Generator(device=latents.device).manual_seed(seed + 1))
            state.update(mask=mask, z0=z0f, noise=noise)
        # anchor = frozen latent NOISED to the level of `latents` (the next timestep the sampler
        # will consume), so frozen and edited regions share a noise level -> no seam / no click.
        ts = pipe.scheduler.timesteps
        if renoise and step + 1 < len(ts):
            anchor = pipe.scheduler.add_noise(state["z0"], state["noise"], ts[step + 1:step + 2])
        else:
            anchor = state["z0"]                        # clean anchor (final step, or SAO_RENOISE=0)
        # IN-PLACE projection so the frozen region actually propagates to the sampler
        latents.mul_(state["mask"]).add_(anchor * (1 - state["mask"]))
        return latents

    with torch.no_grad():
        out = pipe(caption, negative_prompt="Low quality.",
                   num_inference_steps=STEPS, audio_end_in_s=float(DUR_S),
                   generator=gen,
                   callback=cb, callback_steps=1).audios   ### CHECK ### callback API
    y = out[0].T.float().cpu().numpy().mean(axis=1)
    y = (y/(np.max(np.abs(y))+1e-8)*0.95).astype("float32")
    sf.write(out_wav, y, sr); return out_wav

def roundtrip(pipe, in_wav, out_wav):
    """C1 control: encode->decode only, NO editing."""
    z, sr, _ = _encode(pipe, in_wav)
    with torch.no_grad():
        rec = pipe.vae.decode(z).sample                ### CHECK ###
    y = rec[0].T.float().cpu().numpy().mean(axis=1)
    y = (y/(np.max(np.abs(y))+1e-8)*0.95).astype("float32")
    sf.write(out_wav, y, sr); return out_wav

def run(pipe, mode="editing"):
    os.makedirs(f"{OUT}/{mode}", exist_ok=True)        # lazy: created only for a real rollout
    manifest = json.load(open(f"{SEEDS_DIR}/manifest.json"))
    for sid, seed in manifest.items():
        done = f"{OUT}/{mode}/{sid}_done.json"
        # skip only if the DEEPEST requested iter already exists, so raising
        # SAO_N_ITERS extends an existing rollout instead of skipping the seed
        # (the old `if done exists: skip` blocked all extension).
        final = f"{OUT}/{mode}/{sid}_iter{N_ITERS:02d}.wav"
        if os.path.exists(final): print("skip", sid, f"(iter{N_ITERS:02d} present)"); continue
        cur = f"{SEEDS_DIR}/{sid}.wav"; paths=[cur]
        for i in range(1, N_ITERS+1):
            op = f"{OUT}/{mode}/{sid}_iter{i:02d}.wav"
            if os.path.exists(op): cur=op; paths.append(op); continue
            try:
                if mode=="editing":
                    inpaint_edit(pipe, cur, op, seed["caption"], seed=seed.get("seed",0)+i)
                else:
                    roundtrip(pipe, cur, op)
                cur=op; paths.append(op); print(f"  {mode} {sid} {i}/{N_ITERS}")
            except Exception as e:
                print(f"  FAIL {sid} {i}: {e}"); break
        json.dump(paths, open(done,"w"))
    print(f"SAO inpaint {mode} done.")

if __name__ == "__main__":
    print(f"SAO rollout: N_ITERS={N_ITERS}, modes={MODES}")
    print(f"  seeds: {SEEDS_DIR}")
    print(f"  out:   {OUT}")
    print(f"  mask:  {MASK_START_S}-{MASK_END_S}s")
    # The output dirs encode their region (..._5_8, ..._5_10). If the name disagrees with
    # the mask we are about to apply, we are almost certainly extending a rollout with the
    # wrong region -- refuse rather than silently produce an uninterpretable chain.
    import re as _re
    _m = _re.search(r"_(\d+)_(\d+)/?$", OUT)
    if _m:
        _want = (float(_m.group(1)), float(_m.group(2)))
        if _want != (MASK_START_S, MASK_END_S):
            raise SystemExit(
                f"REFUSING: OUT={OUT} encodes region {_want[0]}-{_want[1]}s but the mask is "
                f"{MASK_START_S}-{MASK_END_S}s. Set SAO_MASK={_m.group(1)},{_m.group(2)} "
                f"(see the SAO_MASK note at the top of this file).")
    # A resume that finds nothing is the expensive mistake here, so say so up front
    # rather than after eight iterations of silent regeneration.
    _existing = len([f for m in MODES
                     for f in __import__("glob").glob(f"{OUT}/{m}/*_iter*.wav")])
    print(f"  existing iter wavs under OUT: {_existing}"
          f"{'  <-- 0 means nothing to resume; the full depth will be generated' if not _existing else ''}")
    pipe = load_pipe()
    for _mode in MODES:
        run(pipe, _mode)
