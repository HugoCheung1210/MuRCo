#!/usr/bin/env python3
"""Generate the Setup-2 perturbation dataset for the relational coherence metric.

For every source track selected by ``select_fma_sources.py`` this script cuts a
single fixed-length *reference* segment ``A`` and builds a family of *candidate*
segments ``B`` that each perturb exactly one musical dimension at a known,
graded magnitude.  Downstream, scoring ``C(A, B)`` for every pair should show a
strong diagonal in the separability matrix: each dimension of ``C`` responds to
its own target perturbation and (ideally) not to the others.

Design decisions (see PROJECT_STATE.md sections 3 and 5):

* **Self-perturbation, not context-continuation.**  ``A`` and ``B`` come from the
  *same* source window, so the only thing that differs between the control pair
  and a perturbed pair is the single applied perturbation.  That is what makes a
  clean per-dimension ground truth; adjacent-segment continuation would confound
  the perturbation with the natural A->B difference.

* **Phase-vocoder-matched control.**  The identity control is
  ``time_stretch(rate=1.0)``: it passes through the same STFT resynthesis path as
  the pitch/tempo perturbations, so it carries the same resynthesis colouration
  and isolates *perturbation* from *processing artefact*.  This is the
  "codec-matched control" the S-term validation already relied on.

* **Two timbre families on purpose.**  ``lowpass`` is subtractive (removes high
  partials, moves brightness/MFCC, leaves pitch and rhythm intact) and should be
  the *clean* T perturbation.  ``distortion`` is a harmonic-adding waveshaper and
  is deliberately included as the *aggressive* T perturbation whose harmonics may
  bleed into H (chroma).  Keeping both lets the separability matrix decide which
  belongs in the final protocol rather than guessing.

* **Loudness is controlled.**  Every candidate is RMS-matched to its reference
  (with a peak guard) so that a loudness change never masquerades as a timbre or
  semantic change.

* **Length is locked.**  Every candidate is pinned to exactly the reference
  length before it is written, per the length-lock lesson in PROJECT_STATE.md.

The ``style-swap`` (S) perturbation is the honest weak spot: a *different* track
differs in H/T/R as well as identity, so "S only" is aspirational.  Both a
cross-genre and a same-genre-different-track partner are emitted (tagged) so the
residual can be inspected; tempo/key-matched swapping is left as an open item.

Typical use (from the project root, MF/CLAP env not required -- CPU DSP only)::

    python scripts/data/make_perturbation_dataset.py \
        --manifest sources_selection/selected_manifest.json \
        --raw-audio-dir raw_audio \
        --output-dir perturbations

Outputs: perturbed ``.wav`` files under ``--output-dir/audio`` and a
``pairs_manifest.json`` describing every (A, B) pair with its target dimension
and signed magnitude, ready for the ``C(A, B)`` scoring harness.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

try:
    import librosa
    from scipy.signal import butter, sosfilt
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "make_perturbation_dataset requires librosa, scipy and soundfile: "
        "pip install librosa scipy soundfile"
    ) from exc


# --- perturbation grids -----------------------------------------------------
# Each entry is (signed magnitude used in the filename/manifest, callable args).
PITCH_STEPS_SEMITONES: tuple[int, ...] = (-4, -2, -1, 1, 2, 4)          # -> H
TIME_STRETCH_PERCENT: tuple[int, ...] = (-20, -10, -5, 5, 10, 20)       # -> R
LOWPASS_CUTOFFS_HZ: tuple[int, ...] = (8000, 4000, 2000)                # -> T (clean)
DISTORTION_DRIVES: tuple[float, ...] = (4.0, 12.0, 32.0)                # -> T (aggressive)

# --- family "O": TEMPORAL / ORDER violations (T1) ---------------------------
# The point: B is acoustically (near-)identical MATERIAL to A but in a broken
# temporal relation.  CLAP's audio embedding is TIME-POOLED, so it is largely
# invariant to order/continuity within a clip and should score these ~control;
# S reads the actual A->B transition and should drop.  This is the one axis
# where S can beat an acoustic-similarity model, so it is the T1 headline.
SHUFFLE_GRANULARITIES: tuple[str, ...] = ("beat", "bar")
XFADE_MS = 10.0            # slice joins: avoid clicks (a click is a DSP-dim cue)
BEATS_PER_BAR = 4
FALLBACK_SLICE_S = 1.0     # if beat tracking fails / <4 beats

# Displacement geometry.  B = a different window of the SAME source track.
# FMA-medium clips are 30 s (measured: min 29.98), which is the binding
# constraint: with the MAIN build's defaults (offset=10, seconds=10) a forward
# displacement needs offset + 2*seconds + skip <= ~30, i.e. skip <= 0 -- there
# is NO room.  Worse, load_window() silently falls back to a clamped offset when
# the window overruns, so disp_near and disp_far would BOTH collapse to [20,30]
# -- an identical B for the two magnitudes, i.e. a fabricated contrast that
# would still produce a plausible-looking table.  Hence the O family uses its
# own geometry (below) and a STRICT window loader that skips-and-logs instead of
# clamping.  8 s also matches the S protocol's --secs 8, so nothing is lost.
DISP_SKIPS_S: tuple[float, ...] = (2.0, 7.5)   # (near, far)
TEMPORAL_SECONDS = 8.0     # A = [offset, offset+8]
TEMPORAL_OFFSET = 6.0      # far B ends at 6+8+7.5+8 = 29.5 s < 29.98 -> fits

EPS = 1e-9


@dataclass(frozen=True)
class Source:
    """A selected, derivative-permitted source track."""

    selection_id: str
    genre: str
    path: Path
    license: str
    duration: float


@dataclass
class Pair:
    """One (reference A, candidate B) pair for the separability matrix."""

    pair_id: str
    source_id: str
    genre: str
    dim_target: str          # one of: control, H, R, T, S
    perturbation: str        # e.g. pitch_shift, time_stretch, lowpass, distortion, style_swap
    magnitude: float         # signed; units depend on perturbation (see manifest note)
    magnitude_unit: str
    ref_path: str            # A
    cand_path: str           # B
    partner_id: str = ""     # only for style_swap
    sr: int = 0
    seconds: float = 0.0
    expected_drop: str = ""  # which dimension(s) SHOULD fall; "" = none (control)
    notes: str = ""


# --- signal helpers ---------------------------------------------------------

def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)) + EPS))


def match_loudness(y: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Scale ``y`` to the reference RMS, guarding against clipping."""
    scaled = y * (rms(reference) / rms(y))
    peak = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if peak > 0.999:
        scaled = scaled * (0.999 / peak)
    return scaled.astype(np.float32)


def pin_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Crop or zero-pad to exactly ``target_len`` samples (length-lock)."""
    return librosa.util.fix_length(y, size=target_len).astype(np.float32)


# --- DSP backend: rubberband (formant-preserving) with librosa fallback ------
# rubberband gives formant-preserving pitch shift and higher-quality time
# stretching, which keeps the pitch perturbation from dragging the spectral
# envelope (a pitch->T leak) and the stretch perturbation from smearing
# transients (a stretch->T leak).  When the rubberband binary / pyrubberband
# are absent we fall back to librosa's phase vocoder and warn once.
_BACKEND = "auto"          # set from --dsp-backend in main()
_RUBBERBAND: bool | None = None
_WARNED_FALLBACK = False


def _rubberband_available() -> bool:
    global _RUBBERBAND
    if _RUBBERBAND is None:
        try:
            import pyrubberband  # noqa: F401
            _RUBBERBAND = shutil.which("rubberband") is not None
        except ImportError:
            _RUBBERBAND = False
    return _RUBBERBAND


def use_rubberband() -> bool:
    """Resolve the active backend, honouring --dsp-backend and availability."""
    global _WARNED_FALLBACK
    if _BACKEND == "librosa":
        return False
    if _BACKEND == "rubberband":
        if not _rubberband_available():
            raise SystemExit(
                "--dsp-backend rubberband requested but not found. Install the "
                "binary (apt install rubberband-cli / brew install rubberband) "
                "and `pip install pyrubberband`."
            )
        return True
    available = _rubberband_available()          # auto
    if not available and not _WARNED_FALLBACK:
        logging.warning(
            "rubberband not found; falling back to librosa (naive, non-formant-"
            "preserving pitch and lower-quality stretch). Install rubberband-cli "
            "+ pyrubberband for clean pitch/tempo perturbations."
        )
        _WARNED_FALLBACK = True
    return available


def active_backend() -> str:
    return "rubberband" if use_rubberband() else "librosa"


def _pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if use_rubberband():
        import pyrubberband as pyrb
        # --formant keeps the spectral envelope fixed while shifting pitch.
        out = pyrb.pitch_shift(y, sr, semitones, rbargs={"--formant": ""})
        return out.astype(np.float32)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(np.float32)


def _time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    # rate > 1 => faster / shorter (same convention as librosa).
    if use_rubberband():
        import pyrubberband as pyrb
        return pyrb.time_stretch(y, sr, rate).astype(np.float32)
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


# --- perturbations (each returns a length-locked, loudness-matched B) --------

def perturb_control(ref: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    # Identity through the active backend's resynthesis path, so the control
    # carries the same colouration as the pitch/stretch perturbations.
    y = _time_stretch(ref, sr, 1.0)
    return match_loudness(pin_length(y, target_len), ref)


def perturb_pitch(ref: np.ndarray, sr: int, target_len: int, semitones: int) -> np.ndarray:
    y = _pitch_shift(ref, sr, semitones)
    return match_loudness(pin_length(y, target_len), ref)


def perturb_time_stretch(src_long: np.ndarray, sr: int, target_len: int, percent: int,
                         ref: np.ndarray) -> np.ndarray:
    # Rate-aware windowing: stretch a source slice of length target_len*rate down
    # (or up) to exactly target_len, so no silence padding is introduced.  Padding
    # a shorter faster-stretch back to length would inject silent frames that shift
    # the MFCC distribution (a spurious stretch->T leak) and corrupt beat tracking
    # (a spurious effect on R).  ``ref`` (= A) is only the loudness anchor here.
    rate = 1.0 + percent / 100.0
    need = int(round(target_len * rate))
    win = src_long[:need]
    if win.shape[0] < need:            # short source: fall back to padding
        win = pin_length(win, need)
    y = _time_stretch(win, sr, rate)
    return match_loudness(pin_length(y, target_len), ref)


def perturb_lowpass(ref: np.ndarray, sr: int, target_len: int, cutoff_hz: int) -> np.ndarray:
    sos = butter(N=8, Wn=cutoff_hz, btype="low", fs=sr, output="sos")
    y = sosfilt(sos, ref).astype(np.float32)
    return match_loudness(pin_length(y, target_len), ref)


def perturb_distortion(ref: np.ndarray, sr: int, target_len: int, drive: float) -> np.ndarray:
    # Normalised soft-clip: preserves pitch/tempo, adds harmonics (moves timbre).
    y = np.tanh(drive * ref) / np.tanh(np.array(drive, dtype=np.float32))
    return match_loudness(pin_length(y.astype(np.float32), target_len), ref)


# --- IO ---------------------------------------------------------------------

# --- family "O": temporal-violation perturbations (T1) ----------------------

def _xfade_concat(slices: Sequence[np.ndarray], sr: int, ms: float = XFADE_MS) -> np.ndarray:
    """Concatenate slices with a short equal-gain crossfade at each join.

    A hard splice clicks, and a click is broadband energy that H/T/R would read
    as a timbre change -- an artifactual cue that would let the DSP dims "detect"
    a shuffle for the wrong reason and destroy the experiment's logic.
    """
    xf = max(1, int(sr * ms / 1000.0))
    out = np.asarray(slices[0], dtype=np.float32)
    for s in slices[1:]:
        s = np.asarray(s, dtype=np.float32)
        n = min(xf, out.size, s.size)
        if n < 1:
            out = np.concatenate([out, s])
            continue
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        joined = out[-n:] * (1.0 - ramp) + s[:n] * ramp
        out = np.concatenate([out[:-n], joined, s[n:]])
    return out.astype(np.float32)


def _slice_bounds(ref: np.ndarray, sr: int, granularity: str) -> tuple[list[int], str]:
    """Sample-index boundaries to cut A at; + a note recording how we got them."""
    note = ""
    try:
        _, beats = librosa.beat.beat_track(y=ref, sr=sr, units="frames")
        bounds = list(librosa.frames_to_samples(beats))
    except Exception as exc:  # noqa: BLE001 - beat tracking is best-effort
        logging.debug("beat_track failed: %s", exc)
        bounds = []
    if len(bounds) < 4:
        step = int(FALLBACK_SLICE_S * sr)
        bounds = list(range(0, ref.size, step))
        note = f"beat tracking unusable (<4 beats); fixed {FALLBACK_SLICE_S:g}s slices"
    elif granularity == "bar":
        bounds = bounds[::BEATS_PER_BAR]           # group beats into bars
        if len(bounds) < 3:
            step = int(FALLBACK_SLICE_S * sr * BEATS_PER_BAR)
            bounds = list(range(0, ref.size, step))
            note = "too few bars; fixed 4s slices"
    bounds = sorted({0, *[int(b) for b in bounds if 0 < int(b) < ref.size], int(ref.size)})
    return bounds, note


def perturb_shuffle(ref: np.ndarray, sr: int, target_len: int, granularity: str,
                    rng: random.Random) -> tuple[np.ndarray, str]:
    """B = A's own slices, reordered.  Same material, destroyed temporal order."""
    bounds, note = _slice_bounds(ref, sr, granularity)
    slices = [ref[a:b] for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    if len(slices) < 2:
        return match_loudness(pin_length(ref.copy(), target_len), ref), "not shuffled: <2 slices"
    order = list(range(len(slices)))
    for _ in range(20):                            # a permutation that actually moves
        rng.shuffle(order)
        if order != sorted(order):
            break
    y = _xfade_concat([slices[i] for i in order], sr)
    n_note = f"{len(slices)} {granularity} slices, {XFADE_MS:g}ms crossfades"
    return (match_loudness(pin_length(y, target_len), ref),
            "; ".join(x for x in (n_note, note) if x))


def perturb_reverse(ref: np.ndarray, sr: int, target_len: int) -> np.ndarray:
    """B = A reversed: near-identical long-term spectrum, musically destroyed."""
    return match_loudness(pin_length(ref[::-1].copy(), target_len), ref)


def load_window_strict(path: Path, sr: int, offset_s: float, seconds: float
                       ) -> np.ndarray | None:
    """load_window WITHOUT the silent offset clamp -- returns None if it won't fit.

    load_window() clamps a too-late offset back to (total - seconds), which for
    the displacement family would silently return the SAME window for two
    different skips.  Here a window that does not fit is a skipped pair and a
    log line, never a quietly wrong one.
    """
    try:
        total = librosa.get_duration(path=str(path))
    except Exception as exc:  # noqa: BLE001
        logging.error("Cannot read %s: %s", path, exc)
        return None
    if offset_s + seconds > total + EPS:
        logging.warning("displacement window [%.1f, %.1f]s does not fit in %s (%.2fs) - skipping",
                        offset_s, offset_s + seconds, path.name, total)
        return None
    y, _ = librosa.load(path=str(path), sr=sr, mono=True, offset=offset_s, duration=seconds)
    if y.size == 0:
        logging.error("Empty window from %s", path)
        return None
    return y.astype(np.float32)


def load_sources(manifest_path: Path, raw_dir: Path) -> list[Source]:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources: list[Source] = []
    skipped_nd: list[str] = []
    for entry in entries:
        license_name = str(entry.get("license", ""))
        if is_no_derivatives(license_name):
            skipped_nd.append(str(entry.get("selection_id", "?")))
            continue
        # Prefer the manifest's copied_to path, fall back to <raw_dir>/<id>.mp3.
        rel = entry.get("copied_to") or f"{entry['selection_id']}.mp3"
        path = Path(rel)
        if not path.is_absolute() and not path.exists():
            path = raw_dir / Path(rel).name
        sources.append(
            Source(
                selection_id=str(entry["selection_id"]),
                genre=str(entry["genre"]),
                path=path,
                license=license_name,
                duration=float(entry.get("duration", 0.0)),
            )
        )
    if skipped_nd:
        logging.warning(
            "Skipped %d No-Derivatives track(s) (cannot create derivatives): %s",
            len(skipped_nd), ", ".join(skipped_nd),
        )
    return sources


def is_no_derivatives(license_name: str) -> bool:
    """Defensive ND guard, matching the corrected select_fma_sources filter."""
    normal = license_name.casefold().replace("_", "-").replace(" ", "")
    return "noderivative" in normal or "-nd" in normal or "nd-" in normal


def load_window(path: Path, sr: int, offset_s: float, seconds: float) -> np.ndarray | None:
    """Load a mono window; fall back to offset 0 for short tracks."""
    try:
        total = librosa.get_duration(path=str(path))
    except Exception as exc:  # noqa: BLE001 - want to log & skip any decode failure
        logging.error("Cannot read %s: %s", path, exc)
        return None
    off = offset_s if total >= offset_s + seconds else max(0.0, total - seconds)
    y, _ = librosa.load(path=str(path), sr=sr, mono=True, offset=off, duration=seconds)
    if y.size == 0:
        logging.error("Empty window from %s", path)
        return None
    return y.astype(np.float32)


# --- pairing plan -----------------------------------------------------------

def build_pairs(
    sources: Sequence[Source],
    sr: int,
    seconds: float,
    offset_s: float,
    audio_dir: Path,
    families: set[str],
    rng: random.Random,
    dry_run: bool,
    seed: int = 42,
) -> list[Pair]:
    target_len = int(round(seconds * sr))
    max_rate = 1.0 + (max(abs(p) for p in TIME_STRETCH_PERCENT) / 100.0
                      if TIME_STRETCH_PERCENT else 0.0)
    long_seconds = seconds * max_rate      # enough source for the fastest stretch
    by_genre: dict[str, list[Source]] = {}
    for s in sources:
        by_genre.setdefault(s.genre, []).append(s)

    pairs: list[Pair] = []
    for src in sources:
        # load one buffer long enough for rate-aware stretch; A is its first window
        src_long = None if dry_run else load_window(src.path, sr, offset_s, long_seconds)
        if not dry_run and src_long is None:
            continue
        ref = None if dry_run else src_long[:target_len]
        ref_name = f"{src.selection_id}__A.wav"
        ref_path = audio_dir / ref_name
        if not dry_run:
            sf.write(ref_path, ref, sr)

        def emit(dim: str, pert: str, mag: float, unit: str, y: np.ndarray | None,
                 expect: str, tag: str, partner: str = "", note: str = "") -> None:
            cand_name = f"{src.selection_id}__{pert}__{tag}.wav"
            cand_path = audio_dir / cand_name
            if not dry_run and y is not None:
                sf.write(cand_path, y, sr)
            pairs.append(Pair(
                pair_id=f"{src.selection_id}::{pert}::{tag}",
                source_id=src.selection_id, genre=src.genre,
                dim_target=dim, perturbation=pert, magnitude=mag, magnitude_unit=unit,
                ref_path=str(ref_path), cand_path=str(cand_path),
                partner_id=partner, sr=sr, seconds=seconds,
                expected_drop=expect, notes=note,
            ))

        # control (no dimension should move)
        emit("control", "control", 0.0, "none",
             None if dry_run else perturb_control(ref, sr, target_len),
             expect="", tag="ctrl")

        if "H" in families:
            for st in PITCH_STEPS_SEMITONES:
                emit("H", "pitch_shift", float(st), "semitones",
                     None if dry_run else perturb_pitch(ref, sr, target_len, st),
                     expect="H", tag=f"{st:+d}st")

        if "R" in families:
            for pct in TIME_STRETCH_PERCENT:
                emit("R", "time_stretch", float(pct), "percent_tempo",
                     None if dry_run else perturb_time_stretch(src_long, sr, target_len, pct, ref),
                     expect="R", tag=f"{pct:+d}pct", note="rate-aware window (no silence pad)")

        if "T" in families:
            for cutoff in LOWPASS_CUTOFFS_HZ:
                emit("T", "lowpass", float(cutoff), "hz_cutoff",
                     None if dry_run else perturb_lowpass(ref, sr, target_len, cutoff),
                     expect="T", tag=f"lp{cutoff}", note="subtractive/clean")
            for drive in DISTORTION_DRIVES:
                emit("T", "distortion", float(drive), "tanh_drive",
                     None if dry_run else perturb_distortion(ref, sr, target_len, drive),
                     expect="T", tag=f"drv{drive:g}",
                     note="harmonic-adding; may bleed to H")

        if "S" in families:
            for kind in ("cross", "same"):
                partner = _pick_partner(src, by_genre, kind, rng)
                if partner is None:
                    continue
                py = None
                if not dry_run:
                    pw = load_window(partner.path, sr, offset_s, seconds)
                    if pw is None:
                        continue
                    py = match_loudness(pin_length(pw, target_len), ref)
                emit("S", "style_swap", float("nan"), f"{kind}_genre",
                     py, expect="S", tag=f"swap_{kind}", partner=partner.selection_id,
                     note="different track: H/T/R may also move; 'S only' is aspirational")

        # --- family O: temporal/order violations (T1) ------------------------
        # Emitted LAST and gated, so a default run's RNG stream is byte-identical
        # to the validated 1890-pair build.  Uses a per-source RNG derived from
        # the seed rather than the shared stream, so adding/removing O can never
        # perturb style_swap partner selection.
        if "O" in families:
            srng = random.Random(f"{seed}:{src.selection_id}")
            _CLAP_NOTE = ("acoustic material identical; temporal order violated "
                          "- CLAP-invariant by construction (time-pooled embedding)")
            for gran in SHUFFLE_GRANULARITIES:
                y = note = None
                if not dry_run:
                    y, note = perturb_shuffle(ref, sr, target_len, gran, srng)
                emit("S", "shuffle", float("nan"), f"{gran}_permutation",
                     y, expect="S", tag=f"shuf_{gran}",
                     note=f"{_CLAP_NOTE}; {note or ''}".strip("; "))

            emit("S", "reverse", float("nan"), "time_reversed",
                 None if dry_run else perturb_reverse(ref, sr, target_len),
                 expect="S", tag="rev",
                 note=f"{_CLAP_NOTE}; long-term spectrum near-identical, MFCCs are "
                      f"NOT exactly reversal-invariant so T may move")

            for skip, tag in zip(DISP_SKIPS_S, ("disp_near", "disp_far")):
                off = offset_s + seconds + skip
                y = None
                if not dry_run:
                    w = load_window_strict(src.path, sr, off, seconds)
                    if w is None:
                        continue          # logged; never a silently-clamped window
                    y = match_loudness(pin_length(w, target_len), ref)
                emit("S", "displacement", float(skip), "seconds_skip",
                     y, expect="S", tag=tag,
                     note=f"different window of the SAME track ([{off:.1f}, "
                          f"{off + seconds:.1f}]s); graded member - same piece, wrong "
                          f"place, so humans may rate it MODERATELY coherent")
    return pairs


def _pick_partner(
    src: Source, by_genre: Mapping[str, list[Source]], kind: str, rng: random.Random
) -> Source | None:
    if kind == "same":
        pool = [s for s in by_genre[src.genre] if s.selection_id != src.selection_id]
    else:
        pool = [s for g, items in by_genre.items() if g != src.genre for s in items]
    return rng.choice(pool) if pool else None


def write_manifest(output_dir: Path, pairs: Sequence[Pair], meta: dict) -> None:
    payload = {
        "meta": meta,
        "note": (
            "magnitude units: pitch_shift=semitones, time_stretch=percent tempo "
            "change, lowpass=Hz cutoff, distortion=tanh drive, style_swap=NaN "
            "(categorical). expected_drop is the dimension(s) that SHOULD fall."
        ),
        "pairs": [asdict(p) for p in pairs],
    }
    (output_dir / "pairs_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


# --- CLI --------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, default=Path("sources_selection/selected_manifest.json"))
    p.add_argument("--raw-audio-dir", type=Path, default=Path("raw_audio"))
    p.add_argument("--output-dir", type=Path, default=Path("perturbations"))
    p.add_argument("--sr", type=int, default=48000, help="canonical sample rate (CLAP wants 48k)")
    p.add_argument("--seconds", type=float, default=10.0, help="segment length A and B")
    p.add_argument("--offset", type=float, default=10.0, help="start offset into each source (skip intros)")
    p.add_argument("--families", default="H,R,T,S",
                   help="comma list of families to build (subset of H,R,T,S,O). Default "
                        "H,R,T,S reproduces the validated 1890-pair set bit-for-bit. "
                        "O = order/temporal violations (T1: shuffle/reverse/displacement); "
                        "build it into a SEPARATE --output-dir, and see --seconds/--offset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dsp-backend", choices=("auto", "rubberband", "librosa"), default="auto",
                   help="pitch/stretch engine; 'auto' uses rubberband if installed, else librosa")
    p.add_argument("--dry-run", action="store_true", help="plan pairs, write manifest, no audio")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.seconds <= 0 or args.offset < 0:
        p.error("--seconds must be > 0 and --offset >= 0")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    families = {f.strip().upper() for f in args.families.split(",") if f.strip()}
    if not families <= {"H", "R", "T", "S", "O"}:
        logging.error("Unknown families: %s", families - {"H", "R", "T", "S", "O"})
        return 2
    if "O" in families:
        # The O family's displacement needs room for TWO non-overlapping windows
        # plus the skip inside a 30 s FMA clip; the main build's 10 s/10 s
        # geometry has none (see DISP_SKIPS_S).  Fail loudly rather than let
        # every displacement pair get skipped or, worse, clamped.
        need = args.offset + 2 * args.seconds + max(DISP_SKIPS_S)
        if need > 29.9:
            logging.error(
                "family O needs offset + 2*seconds + max_skip <= ~29.9s of source, but "
                "--offset %.1f --seconds %.1f gives %.1fs. Use the O geometry: "
                "--seconds %.1f --offset %.1f (A and B both %.0fs, far B ends at %.1fs).",
                args.offset, args.seconds, need, TEMPORAL_SECONDS, TEMPORAL_OFFSET,
                TEMPORAL_SECONDS, TEMPORAL_OFFSET + 2 * TEMPORAL_SECONDS + max(DISP_SKIPS_S))
            return 2
    if not args.manifest.is_file():
        logging.error("Manifest not found: %s", args.manifest)
        return 2

    global _BACKEND
    _BACKEND = args.dsp_backend
    backend = active_backend()   # resolves + warns once if falling back
    logging.info("DSP backend: %s (pitch formant-preserving: %s)",
                 backend, backend == "rubberband")

    audio_dir = args.output_dir / "audio"
    if not args.dry_run:
        audio_dir.mkdir(parents=True, exist_ok=True)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(args.manifest, args.raw_audio_dir)
    logging.info("Loaded %d derivative-permitted source(s) across %d genre(s).",
                 len(sources), len({s.genre for s in sources}))
    if not sources:
        logging.error("No usable sources after ND filtering.")
        return 3

    pairs = build_pairs(
        sources, args.sr, args.seconds, args.offset, audio_dir,
        families, random.Random(args.seed), args.dry_run, seed=args.seed,
    )

    meta = {
        "sr": args.sr, "seconds": args.seconds, "offset": args.offset,
        "families": sorted(families), "seed": args.seed,
        "dsp_backend": backend, "formant_preserving": backend == "rubberband",
        "n_sources": len(sources), "n_pairs": len(pairs),
        "pitch_steps": list(PITCH_STEPS_SEMITONES),
        "time_stretch_percent": list(TIME_STRETCH_PERCENT),
        "lowpass_cutoffs_hz": list(LOWPASS_CUTOFFS_HZ),
        "distortion_drives": list(DISTORTION_DRIVES),
    }
    write_manifest(args.output_dir, pairs, meta)

    from collections import Counter
    by_dim = Counter(p.dim_target for p in pairs)
    logging.info("Planned %d pairs: %s", len(pairs), dict(by_dim))
    logging.info("Manifest -> %s", args.output_dir / "pairs_manifest.json")
    if args.dry_run:
        logging.info("Dry run: no audio written.")
    else:
        logging.info("Audio -> %s", audio_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
