#!/usr/bin/env python
"""The GRPO reward: C as a training signal, with the copy penalty that makes it usable.

    r = log C(A,B)  -  lambda_copy * copy(A,B)

The KL term of the full objective belongs to the trainer, not here, because it is a
property of the policy and not of the audio.

Three design points, each load-bearing:

1. **log C, not C.** C is a weighted geometric mean, so log C is the weighted *sum* of log
   dimensions. That is the space the NNLS weight-fit already works in, and it stops one
   near-zero dimension from flattening every advantage inside a GRPO group.

2. **The copy penalty is not optional.** C(A,B) is maximised by B = A, so a policy
   optimised on C alone learns to repeat its context. The penalty is a hinge, and the
   hinge comes from **real music continuing itself**, not from the frozen policy.

   That choice is empirical (measured 2026-08-05). Frozen MusicGen sits at xcorr_wave
   0.075 mean / 0.168 p95, *below* real music at 0.149 / 0.465: it is under-connected to
   its prompt rather than copy-prone. Hinging on the frozen model would therefore charge
   the policy for becoming as context-related as real music, which is the improvement
   being trained for. Hinging on real music says the defensible thing instead: a
   continuation more correlated with its context than real music is repeating, not
   continuing. The frozen distribution stays useful as the *monitoring* baseline for
   drift, just not as the threshold.

3. **S is pluggable.** `CachedS` serves scores by pair_id from the existing JSON caches,
   which makes the whole reward path testable on CPU today. `MusicFlamingoS` is the GPU
   implementation to add when the hardware exists; nothing else changes.

Usage:
    python scripts/rl/reward.py --self-test        # CPU, uses the cached rerank scores
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Protocol

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

try:
    from coherence_dimensions import (HarmonicDimension, RhythmicDimension,
                                      TimbralDimension)
except ImportError:                                    # dims unavailable (no librosa env)
    HarmonicDimension = RhythmicDimension = TimbralDimension = None

from copy_metrics import copy_metrics

_EPS = 1e-9
# Metrics that are genuinely independent of C. chroma_cos is excluded from the penalty
# because it is essentially H, so penalising it would fight the reward.
PENALISED = ("xcorr_wave", "xcorr_mel")


class SBackend(Protocol):
    def score(self, a: np.ndarray, b: np.ndarray, sr: int, pair_id: str | None) -> float: ...


class CachedS:
    """Serve S from a precomputed cache, by pair_id. CPU, for testing the reward path."""

    def __init__(self, cache_path: str | Path):
        raw = json.loads(Path(cache_path).read_text())
        self.scores: dict[str, float] = {}
        for k, v in (raw.get("scores", raw)).items():
            self.scores[k] = float(v["S"] if isinstance(v, dict) and "S" in v else v)

    def score(self, a, b, sr, pair_id=None) -> float:
        if pair_id is None or pair_id not in self.scores:
            raise KeyError(f"no cached S for pair_id={pair_id!r}")
        return self.scores[pair_id]


class ConstantS:
    """Placeholder for smoke tests where S is irrelevant."""

    def __init__(self, value: float = 0.5):
        self.value = value

    def score(self, a, b, sr, pair_id=None) -> float:
        return self.value


class MusicFlamingoS:
    """Score S live from arrays, for the RL loop. GPU, `mfenv` only.

    Identical to what `score_s_mf.py --scorer coherence` caches: mean P(Yes) over the
    template bank on `A | gap | B`, same `secs` and `gap_s`. The RL loop must not use a
    different S from the one the thesis validated, so this deliberately goes through
    `mf_probe` rather than reimplementing the concat or the logit read.

    `mf_probe` works on file paths, so arrays are written to a scratch dir first. That
    costs ~20 ms against ~900 ms of MF forward pass, which is not worth engineering away.

    Set HF_HOME and HF_ENDPOINT **before** constructing this: `mf_probe` reads them at
    import time.
    """

    def __init__(self, secs: float = 6.0, gap_s: float = 1.0,
                 precision: str = "all_bf16", scratch: str | Path | None = None,
                 cache: bool = True, a_from: str = "tail",
                 token_agg: str = "variants"):
        import mf_probe as mf                      # import here: needs the MF env
        mf.load(precision=precision)
        # Pin the Yes/No read-out rather than inheriting whatever this copy of mf_probe
        # happens to default to. Every RL run in this project was scored with the mass
        # pooled over token variants, and a run that silently picked `single` instead
        # would shift S by about a quarter of its range and could not be compared with
        # the others, while every rank-based check on it would still look fine.
        if token_agg not in ("variants", "single"):
            raise ValueError(f"token_agg must be variants or single, got {token_agg!r}")
        mf.TOKEN_AGG = token_agg
        self.token_agg = token_agg
        self.mf = mf
        self.secs, self.gap_s, self.a_from = secs, gap_s, a_from
        # Per-process scratch. The filenames below are built from the pair_id, which is
        # context + step + rollout, so two training runs started on the same contexts
        # generate identical paths at the same step. Sharing one directory means each
        # unlinks the other's wav between write and read, and the reader dies on a file
        # that existed a moment earlier. Two runs on two cards is the case that hits it.
        self.scratch = Path(scratch or f"/tmp/t2m_rl_s_{os.getpid()}")
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, float] | None = {} if cache else None
        self.calls = self.hits = 0

    def score(self, a, b, sr, pair_id=None) -> float:
        import hashlib

        import soundfile as sf

        key = None
        if self.cache is not None:
            # content hash, not pair_id: identical rollouts recur once the policy sharpens
            h = hashlib.blake2b(digest_size=16)
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
            h.update(np.ascontiguousarray(b, dtype=np.float32).tobytes())
            key = h.hexdigest()
            if key in self.cache:
                self.hits += 1
                return self.cache[key]

        pa = self.scratch / f"{pair_id or 'pair'}_A.wav"
        pb = self.scratch / f"{pair_id or 'pair'}_B.wav"
        sf.write(pa, a, sr)
        sf.write(pb, b, sr)
        try:
            ab = self.mf.concat_clips(str(pa), str(pb), secs=self.secs,
                                      gap_s=self.gap_s, a_from=self.a_from)
            try:
                s = float(np.mean([float(self.mf._yes_prob(ab, q))
                                   for q in self.mf.TEMPLATES]))
            finally:
                self.mf._cleanup(ab)
        finally:
            pa.unlink(missing_ok=True)
            pb.unlink(missing_ok=True)

        self.calls += 1
        if key is not None:
            self.cache[key] = s
        return s


class ClapScorer:
    """CLAP-htsat audio-to-audio similarity from arrays: the arm-B reward of §5b.

    **This must stay bit-identical to `scripts/analysis/score_baselines.py`.** The point of
    the matched pair is to train against *the CLAP the thesis reports as a baseline*, so
    the checkpoint, the 48 kHz front-end rate, the `pooled` readout and the `(1+cos)/2`
    remap are all copied from there rather than re-chosen. If that file's defaults move,
    move these with it or the two arms stop being comparable.

    Two differences from `score_baselines.py`, both forced by the loop rather than chosen:

    1. It takes **arrays, not paths**. The rollouts never touch disk, so there is nothing
       to `librosa.load`. Audio arrives at the policy's rate (32 kHz for MusicGen) and is
       resampled here, because CLAP's front end is fixed at 48 kHz.
    2. It caches by **content hash**, like `MusicFlamingoS`, and additionally caches `A`
       per call. `A` is identical across all G rollouts of a group, so embedding it once
       halves the CLAP work at no cost in fidelity.
    """

    CKPT = "laion/clap-htsat-unfused"
    SR = 48000                                          # CLAP front-end rate; not optional

    def __init__(self, ckpt: str | None = None, device: str | None = None,
                 secs: float | None = None, readout: str = "pooled",
                 cache: bool = True):
        import torch
        from transformers import ClapModel, ClapProcessor

        self.torch = torch
        self.ckpt = ckpt or self.CKPT
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = ClapProcessor.from_pretrained(self.ckpt)
        self.model = ClapModel.from_pretrained(self.ckpt).to(self.device).eval()
        self.secs = secs
        self.readout = readout
        self.cache: dict[str, np.ndarray] | None = {} if cache else None
        self.calls = self.hits = 0

    def _prep(self, y: np.ndarray, sr: int):
        import librosa

        y = np.asarray(y, dtype=np.float32)
        if sr != self.SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.SR)
        if self.secs:
            y = y[: int(round(self.secs * self.SR))]
        try:                                            # newer transformers: audio=
            inp = self.proc(audio=[y], sampling_rate=self.SR, return_tensors="pt")
        except (TypeError, ValueError):                 # older transformers: audios=
            inp = self.proc(audios=[y], sampling_rate=self.SR, return_tensors="pt")
        return inp.to(self.device)

    def embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Unit-norm pooled CLAP embedding, content-hash cached."""
        import hashlib

        key = None
        if self.cache is not None:
            h = hashlib.blake2b(digest_size=16)
            h.update(np.ascontiguousarray(y, dtype=np.float32).tobytes())
            key = h.hexdigest()
            if key in self.cache:
                self.hits += 1
                return self.cache[key]

        with self.torch.no_grad():
            out = self.model.get_audio_features(**self._prep(y, sr))
        # Mirror score_baselines._embed_pooled: this build returns the audio-tower
        # ModelOutput rather than a flat embedding, so unwrap the same way it does.
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            t = out.pooler_output
        elif hasattr(out, "last_hidden_state"):
            t = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            t = out[0]
        else:
            t = out
        e = np.asarray(t.detach().float().cpu().numpy())
        while e.ndim > 2:                               # mean-pool trailing spatial dims
            e = e.mean(axis=-1)
        e = e.reshape(e.shape[0], -1)[0]
        e = e / (np.linalg.norm(e) + 1e-8)

        self.calls += 1
        if key is not None:
            self.cache[key] = e
        return e

    def score(self, a: np.ndarray, b: np.ndarray, sr: int, pair_id=None) -> float:
        ea, eb = self.embed(a, sr), self.embed(b, sr)
        return float((1.0 + np.dot(ea, eb)) / 2.0)      # cos in [-1,1] -> [0,1]


class _CopyPenalty:
    """The hinge shared by both reward arms. Factored out so arm B cannot drift from arm
    A's copy path, which is the one thing in the matched pair that must not vary."""

    def __init__(self, lambda_copy: float = 2.0,
                 copy_baseline: str | Path | None = None):
        self.lambda_copy = lambda_copy
        self.baseline = self._load_baseline(copy_baseline)

    @staticmethod
    def _load_baseline(path) -> dict[str, float]:
        """p95 of each penalised metric on the frozen generator. Empty = no hinge."""
        if path is None:
            return {}
        d = json.loads(Path(path).read_text())["summary"]
        return {k: float(d[k]["p95"]) for k in PENALISED if k in d}

    def __call__(self, a: np.ndarray, b: np.ndarray, sr: int) -> tuple[dict, float, float]:
        cm = copy_metrics(a, b, sr)
        excess = 0.0
        for k in PENALISED:
            thresh = self.baseline.get(k)
            if thresh is not None:
                excess += max(0.0, cm[k] - thresh)      # hinge above the frozen baseline
            else:
                excess += cm[k]                          # no baseline: penalise outright
        return cm, excess, self.lambda_copy * excess


class CoherenceReward(_CopyPenalty):
    """log C minus a copy penalty, plus every component for logging."""

    def __init__(self, s_backend: SBackend, weights: dict[str, float] | None = None,
                 lambda_copy: float = 2.0,
                 copy_baseline: str | Path | None = None):
        super().__init__(lambda_copy=lambda_copy, copy_baseline=copy_baseline)
        self.s = s_backend
        self.w = weights or {"H": 1.0, "T": 1.0, "R": 1.0, "S": 1.0}
        self.dims = ({"H": HarmonicDimension(), "T": TimbralDimension(),
                      "R": RhythmicDimension()} if HarmonicDimension else {})

    def __call__(self, a: np.ndarray, b: np.ndarray, sr: int,
                 pair_id: str | None = None) -> dict:
        scores = {k: float(dim.score(a, b, sr)) for k, dim in self.dims.items()}
        scores["S"] = float(self.s.score(a, b, sr, pair_id))

        # log C = sum_d w_d * log(score_d) / sum_d w_d
        wsum = sum(self.w.values())
        log_c = sum(self.w[d] * math.log(max(scores[d], _EPS)) for d in scores) / wsum

        cm, excess, penalty = super().__call__(a, b, sr)

        return {"reward": log_c - penalty, "log_C": log_c,
                "C": float(math.exp(log_c)), "dims": scores,
                "copy": cm, "copy_excess": excess, "penalty": penalty}


class ClapReward(_CopyPenalty):
    """log CLAP minus the same copy penalty: arm B of the matched pair (§5b).

    The output dict has the same shape as `CoherenceReward`'s so the trainer, the logger
    and `eval_grpo.py` need no special case. `log_C`/`C` carry the *objective*, which for
    this arm is CLAP, and the objective's name is recorded under `objective`.

    H/T/R are still computed and logged although they are **not** in the reward. They cost
    ~0.6 s/rollout inside the existing DSP path and a CLAP-trained policy's H/T/R trace is
    the entire diagnostic of §5b. S is deliberately *not* computed, because skipping Music
    Flamingo is what frees the 16.66 GB that lets this arm run at `--gen-batch 8`; S is
    scored post hoc on the checkpoints by the eval chain.

    **Calibrate `lambda_copy` before trusting a run.** `rlhf_with_C_procedure.md` §5 set
    `lambda_copy = 2.0` so the hack's penalty (1.41) sits about 7x the within-group spread
    of log C (~0.20), and that ratio is the thing to preserve, not the number. log CLAP is
    a remapped cosine and is far flatter than log C, so at `lambda_copy = 2.0` the penalty
    would swamp the objective and the policy would learn only to stop copying. Run
    `--check` or 5 smoke steps, read `objective_spread` off the log, and set
    `--lambda-copy` to `7 * spread / 1.41`. Anything else makes arm B a straw man.
    """

    def __init__(self, clap: ClapScorer, lambda_copy: float = 2.0,
                 copy_baseline: str | Path | None = None, log_dims: bool = True):
        super().__init__(lambda_copy=lambda_copy, copy_baseline=copy_baseline)
        self.clap = clap
        self.dims = ({"H": HarmonicDimension(), "T": TimbralDimension(),
                      "R": RhythmicDimension()} if (log_dims and HarmonicDimension) else {})

    def __call__(self, a: np.ndarray, b: np.ndarray, sr: int,
                 pair_id: str | None = None) -> dict:
        clap = float(self.clap.score(a, b, sr, pair_id))
        log_obj = math.log(max(clap, _EPS))

        scores = {k: float(dim.score(a, b, sr)) for k, dim in self.dims.items()}
        scores["clap_htsat"] = clap

        cm, excess, penalty = super().__call__(a, b, sr)

        return {"reward": log_obj - penalty, "log_C": log_obj,
                "C": clap, "objective": "clap_htsat", "dims": scores,
                "copy": cm, "copy_excess": excess, "penalty": penalty}


def group_advantages(rewards: list[float]) -> list[float]:
    """GRPO group-relative advantage: (r - mean) / sd, which needs no value network and
    normalises away the per-track scale of C."""
    r = np.asarray(rewards, dtype=float)
    sd = r.std()
    return list((r - r.mean()) / (sd + _EPS)) if sd > _EPS else [0.0] * len(r)


def self_test() -> None:
    import librosa

    audio = ROOT / "results/rerank_ace/audio"
    cache = ROOT / "results/rerank_ace/s_scores_rerank.json"
    # real music continuing itself: the hinge, per the docstring. NB these test pairs are
    # co-located (A and B overlap), so they read as more copy-like than a true
    # continuation would; that is a property of the test set, not of the hinge.
    base = ROOT / "results/rl/copy_reference_continuation.json"

    reward = CoherenceReward(
        s_backend=CachedS(cache),
        copy_baseline=base if base.exists() else None,
    )
    if not reward.dims:
        raise SystemExit("H/T/R unavailable: run this in the DSP env from the repo root")

    seed = "s01"
    a, sr = librosa.load(audio / f"{seed}__A.wav", sr=48000, mono=True)

    print(f"copy baseline: {reward.baseline or '(none yet, penalising outright)'}\n")
    print(f"{'pair':<18}{'reward':>9}{'log C':>9}{'C':>8}{'penalty':>9}{'S':>7}")
    rs = []
    for k in range(8):
        pid = f"{seed}::cand{k:02d}"
        b, _ = librosa.load(audio / f"{seed}_cand{k:02d}__B.wav", sr=48000, mono=True)
        out = reward(a, b, sr, pair_id=pid)
        rs.append(out["reward"])
        print(f"{pid:<18}{out['reward']:>9.4f}{out['log_C']:>9.4f}{out['C']:>8.4f}"
              f"{out['penalty']:>9.4f}{out['dims']['S']:>7.3f}")

    adv = group_advantages(rs)
    print(f"\ngroup advantages: {' '.join(f'{x:+.2f}' for x in adv)}")
    print(f"mean {np.mean(adv):+.3f} (must be ~0), sd {np.std(adv):.3f} (must be ~1)")

    # B = A is the degenerate optimum the penalty exists to block.
    out_copy = reward(a, a.copy(), sr, pair_id=f"{seed}::cand00")
    print(f"\nB = A (the hack):  log C {out_copy['log_C']:+.4f}  "
          f"penalty {out_copy['penalty']:.4f}  reward {out_copy['reward']:+.4f}")
    print("NB log C uses the cached S for a real candidate, so only H/T/R reflect the copy;"
          "\n    with a live S backend log C would be higher still, and the penalty larger.")


def self_test_clap() -> None:
    """Arm B's two open questions, both answered by running this on the GPU box.

    1. **Parity.** Does `ClapScorer` reproduce `score_baselines.clap_score` to numerical
       noise? If not, arm B optimises something that is not the CLAP the thesis reports as
       a baseline and the matched pair is void. This is the test that decides it.
    2. **Calibration.** What is the within-group spread of log CLAP? §5 set
       `lambda_copy = 2.0` so the B=A penalty (1.41) sits ~7x log C's spread (~0.20), and
       that *ratio* is what transfers, not the 2.0. This prints the value to pass to
       `--lambda-copy`.
    """
    import librosa

    audio = ROOT / "results/rerank_ace/audio"
    base = ROOT / "results/rl/copy_reference_continuation.json"
    seed = "s01"

    clap = ClapScorer()
    reward = ClapReward(clap=clap, copy_baseline=base if base.exists() else None)
    if not reward.dims:
        raise SystemExit("H/T/R unavailable: run this in an env with librosa")

    pa = audio / f"{seed}__A.wav"
    a_arr, sr = librosa.load(pa, sr=48000, mono=True)

    print(f"checkpoint: {clap.ckpt}  device: {clap.device}\n")
    print(f"{'pair':<18}{'reward':>9}{'logCLAP':>9}{'CLAP':>8}{'ref':>8}{'d':>10}"
          f"{'penalty':>9}")
    logs, worst = [], 0.0
    for k in range(8):
        pb = audio / f"{seed}_cand{k:02d}__B.wav"
        b_arr, _ = librosa.load(pb, sr=48000, mono=True)
        out = reward(a_arr, b_arr, sr, pair_id=f"{seed}::cand{k:02d}")

        # the reference implementation, on the same files, through its own loader
        import score_baselines as sb
        sb.load_clap("clap_htsat")
        ref = sb.clap_score("clap_htsat", str(pa), str(pb), secs=1e9, readout="pooled")

        d = abs(out["C"] - ref)
        worst = max(worst, d)
        logs.append(out["log_C"])
        print(f"{seed}::cand{k:02d}{out['reward']:>12.4f}{out['log_C']:>9.4f}"
              f"{out['C']:>8.4f}{ref:>8.4f}{d:>10.2e}{out['penalty']:>9.4f}")

    spread = float(np.std(logs))
    hack = reward(a_arr, a_arr.copy(), sr, pair_id=f"{seed}::hack")

    print(f"\n1. PARITY vs score_baselines.clap_score: worst |delta| = {worst:.2e}  "
          f"{'PASS' if worst < 1e-5 else 'FAIL -- arm B is not optimising the baseline CLAP'}")
    print(f"\n2. CALIBRATION")
    print(f"   within-group spread of log CLAP : {spread:.4f}   (log C's is ~0.20)")
    print(f"   B = A gives log CLAP {hack['log_C']:+.4f}, copy_excess "
          f"{hack['copy_excess']:.4f}")
    if hack["copy_excess"] > _EPS:
        lam = 7.0 * spread / hack["copy_excess"]
        print(f"   -> run arm B with --lambda-copy {lam:.2f}  "
              f"(preserves §5's 7x penalty-to-spread ratio)")
    print("\nNB these candidates are co-located rerank pairs, so they read as more "
          "copy-like\n    than real continuations. Re-read the spread off 5 smoke steps of "
          "the real loop\n    before committing to a 300-step run.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-clap", action="store_true",
                    help="arm B (§5b): CLAP parity against score_baselines, and the "
                         "lambda_copy calibration. Needs a transformers with ClapModel.")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.self_test_clap:
        self_test_clap()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
