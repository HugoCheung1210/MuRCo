#!/usr/bin/env python3
"""Coherence-metric dimensions behind one common interface.

Every dimension of the relational coherence metric ``C(A, B)`` implements::

    score(a: np.ndarray, b: np.ndarray, sr: int) -> float in [0, 1]

1.0 = "B is perfectly coherent with A on this dimension"; lower = more
divergence.  One contract makes the geometric aggregation
``C = prod(score_d ** lambda_d)`` and the per-dimension separability analysis
trivial to build once and reuse.

Implemented: H (harmonic), T (timbral), R (rhythmic).  S (structural/semantic,
the Music-Flamingo term) plugs in behind the same contract.

H -- pooled chroma cosine.  Time-pooling makes H tempo-invariant by
construction; a transpose rotates the chroma vector and drops the cosine.

T -- FAD-style Frechet distance on MFCC frame distributions, squashed to [0, 1]
via exp(-distance / tau).  Per-coefficient standardisation keeps tau transfer-
able.  (Pitch retains some T sensitivity because harmonics migrate across mel
bands regardless of envelope; formant-preserving pitch shifting in the
generator mitigates but does not remove it.)

R -- rhythmic coherence, gated on A's beat strength:

    R = 1 - g * (1 - [w_tempo * tempo_score + w_phase * phase_score])

  * tempo_score: octave-folded log-tempo agreement (graded; catches the tempo
    change a time-stretch introduces).
  * phase_score: zero-lag correlation of onset-strength envelopes (catches
    beats landing at different times).
  * g = beat strength of A (how strongly A's onsets repeat at a regular
    period, from the onset autocorrelation; in [0, 1]).
    When A carries no steady beat (piano, ambient) g -> 0 and R -> 1: the
    dimension *abstains* rather than emitting noise from unreliable tempo/onset
    estimates.  This is the beat-strength gating the project flagged as needed
    for non-percussive audio.

  tempo and phase are combined additively (not multiplied) so tempo's graded
  response survives even when phase saturates to 0 (any stretch >~5% fully
  decorrelates the onset envelope over a 10 s clip).  Note R is only meaningful
  where g is high; report it per genre, never pooled across percussive and
  ambient material.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

try:
    import librosa
    from scipy.linalg import sqrtm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("coherence_dimensions requires librosa and scipy") from exc

_EPS = 1e-9


class Dimension(ABC):
    """A single coherence dimension mapping an (A, B) pair to [0, 1]."""

    name: str = "?"

    @abstractmethod
    def score(self, a: np.ndarray, b: np.ndarray, sr: int) -> float:
        raise NotImplementedError


class HarmonicDimension(Dimension):
    """H: pooled chroma cosine (harmonic / tonal identity)."""

    name = "H"

    def __init__(self, hop_length: int = 512, n_chroma: int = 12) -> None:
        self.hop_length = hop_length
        self.n_chroma = n_chroma

    def _pooled_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        chroma = librosa.feature.chroma_cqt(
            y=np.ascontiguousarray(y), sr=sr, hop_length=self.hop_length, n_chroma=self.n_chroma
        )
        pooled = chroma.mean(axis=1)
        norm = float(np.linalg.norm(pooled))
        return pooled / norm if norm > 0 else pooled

    def score(self, a: np.ndarray, b: np.ndarray, sr: int) -> float:
        ca, cb = self._pooled_chroma(a, sr), self._pooled_chroma(b, sr)
        return float(np.clip(np.dot(ca, cb), 0.0, 1.0))


class TimbralDimension(Dimension):
    """T: Frechet distance on MFCC frame distributions, squashed to [0, 1]."""

    name = "T"

    def __init__(self, n_mfcc: int = 20, hop_length: int = 512, drop_c0: bool = True,
                 tau: float = 6.0, cov_eps: float = 1e-6) -> None:
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.drop_c0 = drop_c0
        # exp temperature. Set to 6.0 in the initial commit and never calibrated: no dev
        # split was ever held out for it, so tau=6 predates every experiment in the repo.
        # It is not a free parameter. exp(-d/tau) is monotone in d, so every AUC read on T
        # is identical at any tau (verified: 0.9201 at all of 2,3,4,6,9,12,20), and in log
        # space w_T*log T = -(w_T/tau)*d, so tau only rescales T's weight. The per-family
        # peak dimension is stable for tau in [3,12] and breaks at BOTH ends: tau=2 sends
        # every family to T, tau=20 sends pitch_shift to H and style_swap to R.
        # See scripts/analysis/tau_sensitivity.py -> results/coherence/tau_sensitivity.json
        self.tau = tau
        self.cov_eps = cov_eps

    def _frames(self, y: np.ndarray, sr: int) -> np.ndarray:
        m = librosa.feature.mfcc(
            y=np.ascontiguousarray(y), sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
        )
        if self.drop_c0:
            m = m[1:]
        return m.T

    @staticmethod
    def _gaussian(frames: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
        mu = frames.mean(axis=0)
        cov = np.cov(frames, rowvar=False) + eps * np.eye(frames.shape[1])
        return mu, cov

    def distance(self, a: np.ndarray, b: np.ndarray, sr: int) -> float:
        fa, fb = self._frames(a, sr), self._frames(b, sr)
        std = np.concatenate([fa, fb]).std(axis=0) + 1e-8
        fa, fb = fa / std, fb / std
        mu_a, cov_a = self._gaussian(fa, self.cov_eps)
        mu_b, cov_b = self._gaussian(fb, self.cov_eps)
        diff = mu_a - mu_b
        covmean = sqrtm(cov_a @ cov_b)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fd2 = float(diff @ diff + np.trace(cov_a + cov_b - 2.0 * covmean))
        return float(np.sqrt(max(fd2, 0.0)))

    def score(self, a: np.ndarray, b: np.ndarray, sr: int) -> float:
        return float(np.exp(-self.distance(a, b, sr) / self.tau))


class RhythmicDimension(Dimension):
    """R: tempo + beat-phase coherence, gated on A's beat strength."""

    name = "R"

    def __init__(self, hop_length: int = 512, tau_tempo: float = 0.2,
                 w_tempo: float = 0.6, w_phase: float = 0.4,
                 min_lag_s: float = 0.25, max_lag_s: float = 2.0) -> None:
        self.hop_length = hop_length
        self.tau_tempo = tau_tempo
        self.w_tempo = w_tempo
        self.w_phase = w_phase
        self.min_lag_s = min_lag_s
        self.max_lag_s = max_lag_s

    def _onset(self, y: np.ndarray, sr: int) -> np.ndarray:
        return librosa.onset.onset_strength(y=np.ascontiguousarray(y), sr=sr,
                                            hop_length=self.hop_length)

    def _tempo(self, onset: np.ndarray, sr: int) -> float:
        return float(librosa.feature.tempo(onset_envelope=onset, sr=sr,
                                           hop_length=self.hop_length)[0])

    def _tempo_score(self, oa: np.ndarray, ob: np.ndarray, sr: int) -> float:
        ta, tb = self._tempo(oa, sr), self._tempo(ob, sr)
        d = abs(np.log2((tb + _EPS) / (ta + _EPS)))
        d = abs(d - round(d))                       # octave-folded: ignore x2/x0.5 errors
        return float(np.exp(-d / self.tau_tempo))

    @staticmethod
    def _phase_score(oa: np.ndarray, ob: np.ndarray) -> float:
        m = min(oa.shape[0], ob.shape[0])
        a = oa[:m] - oa[:m].mean()
        b = ob[:m] - ob[:m].mean()
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.clip(np.dot(a, b) / den, 0.0, 1.0)) if den > 0 else 0.0

    def _beat_strength(self, onset: np.ndarray, sr: int) -> float:
        x = onset - onset.mean()
        ac = np.correlate(x, x, mode="full")[x.shape[0] - 1:]
        if ac[0] <= 0:
            return 0.0
        ac = ac / ac[0]
        lo = int(self.min_lag_s * sr / self.hop_length)
        hi = int(self.max_lag_s * sr / self.hop_length)
        if hi <= lo or lo >= ac.shape[0]:
            return 0.0
        return float(np.clip(ac[lo:min(hi, ac.shape[0])].max(), 0.0, 1.0))

    def components(self, a: np.ndarray, b: np.ndarray, sr: int) -> dict[str, float]:
        oa, ob = self._onset(a, sr), self._onset(b, sr)
        return {"tempo": self._tempo_score(oa, ob, sr),
                "phase": self._phase_score(oa, ob),
                "beat_strength": self._beat_strength(oa, sr)}

    def score(self, a: np.ndarray, b: np.ndarray, sr: int) -> float:
        oa, ob = self._onset(a, sr), self._onset(b, sr)
        raw = self.w_tempo * self._tempo_score(oa, ob, sr) + self.w_phase * self._phase_score(oa, ob)
        g = self._beat_strength(oa, sr)             # confidence from the reference A
        return float(1.0 - g * (1.0 - raw))         # g->0 => R->1 (abstain)


class StructuralDimension(Dimension):
    """S: relational Music-Flamingo coherence, served from a precomputed cache.

    MF is a 16 GB model that must run in its own env (it cannot share a process
    with the DSP dimensions), so S is scored ahead of time by ``score_s_mf.py``
    into ``s_scores.json`` (pair_id -> P(Yes)).  This dimension looks the value up
    by pair_id and returns it, keeping S behind the same contract as H/T/R.

    Because MF scores a *pair* (concatenated A|B) rather than two arrays
    independently, ``score(a, b, sr)`` is not the access path -- the harness calls
    ``score_pair(pair_id)``.  ``score`` therefore raises, on purpose, to catch any
    code that tries to treat S like a signal-level dimension.
    """

    name = "S"

    def __init__(self, cache_path: str | None = None) -> None:
        self._scores: dict[str, float] = {}
        if cache_path:
            self.load_cache(cache_path)

    def load_cache(self, cache_path: str) -> None:
        import json
        from pathlib import Path
        doc = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        self._scores = {k: float(v) for k, v in doc.get("scores", {}).items()}

    def has(self, pair_id: str) -> bool:
        return pair_id in self._scores

    def score_pair(self, pair_id: str) -> float:
        return float(self._scores[pair_id])

    def score(self, a: np.ndarray, b: np.ndarray, sr: int) -> float:  # pragma: no cover
        raise NotImplementedError(
            "S is relational over a concatenated pair; call score_pair(pair_id) "
            "with values from score_s_mf.py, not score(a, b, sr)."
        )


def build_dimensions(letters: str, s_cache: str | None = None) -> dict[str, Dimension]:
    available: dict[str, Dimension] = {
        "H": HarmonicDimension(),
        "T": TimbralDimension(),
        "R": RhythmicDimension(),
        "S": StructuralDimension(s_cache),
    }
    chosen: dict[str, Dimension] = {}
    for letter in (x.strip().upper() for x in letters.split(",") if x.strip()):
        if letter not in available:
            raise ValueError(f"Unknown dimension {letter!r}; expected subset of H,T,R,S")
        chosen[letter] = available[letter]
    return chosen
