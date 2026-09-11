#!/usr/bin/env python
"""Copy metrics: how much does B just repeat its context A?

Why this exists. C(A,B) is maximised by B = A, so a policy optimised against C will drift
towards copying the context. Detecting that requires knowing what "not copying" looks like
*before* any fine-tuning, so this module (a) defines the metrics and (b) baselines them on
the frozen generator's own candidates, which is the distribution RL starts from.

Metrics, all in [0,1] unless noted, higher = more copy-like:

  xcorr_wave    peak of the normalised cross-correlation between A and B waveforms,
                over lags within +-1 s. Catches verbatim or near-verbatim repetition.
  xcorr_mel     the same on log-mel envelopes, which survives phase differences and
                small time shifts, so it catches "same material, resynthesised".
  chroma_cos    cosine between pooled chroma vectors. NB this is essentially H, so it is
                reported for completeness but is NOT independent evidence.
  onset_ratio   onset density of B over A, clipped to [0,2]; 1.0 = same rhythmic activity.
                Not a copy metric on its own; it catches the degenerate "textural mush"
                direction, where activity collapses.
  pc_entropy    pitch-class entropy of B in bits (0-3.58). Low = drone. Reported raw,
                since the frozen baseline is the reference, not an absolute threshold.

Usage (DSP env):
    python scripts/rl/copy_metrics.py                     # baseline the ACE rerank tree
    python scripts/rl/copy_metrics.py --audio-dir results/rerank/audio --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# librosa is imported lazily inside the functions that need it. It pulls in numba, whose
# first import on a cold environment can take minutes while the JIT cache is built, and a
# top-level import means the script appears to hang before printing anything at all.

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

_EPS = 1e-9


def resolve_audio(rel: str, audio_root: str | None) -> Path:
    """Find a context's audio file; see generate_continuations.resolve_audio."""
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


def _norm_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Peak normalised cross-correlation within +-max_lag samples/frames.

    FFT-based on purpose. `np.correlate(..., mode="full")` is a direct O(N^2)
    convolution, which at 48 kHz over 6 s is ~8e10 operations and took ~17 s per pair;
    `scipy.signal.correlate(..., method="fft")` does the same thing in milliseconds.
    """
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < _EPS or nb < _EPS:
        return 0.0
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    try:
        from scipy.signal import correlate
        full = correlate(a, b, mode="full", method="fft") / (na * nb)
    except ImportError:                       # numpy fallback: slow, but correct
        full = np.correlate(a, b, mode="full") / (na * nb)
    centre = full.shape[0] // 2
    lo = max(0, centre - max_lag)
    hi = min(full.shape[0], centre + max_lag + 1)
    return float(np.clip(np.abs(full[lo:hi]).max(), 0.0, 1.0))


def copy_metrics(a: np.ndarray, b: np.ndarray, sr: int) -> dict[str, float]:
    """All copy metrics for one (context, generated) pair."""
    import librosa

    out: dict[str, float] = {}

    out["xcorr_wave"] = _norm_xcorr(a, b, max_lag=sr)                  # +-1 s

    hop = 512
    ma = librosa.power_to_db(librosa.feature.melspectrogram(y=a, sr=sr, hop_length=hop))
    mb = librosa.power_to_db(librosa.feature.melspectrogram(y=b, sr=sr, hop_length=hop))
    out["xcorr_mel"] = _norm_xcorr(ma.mean(axis=0), mb.mean(axis=0),
                                   max_lag=int(sr / hop))              # +-1 s in frames

    ca = librosa.feature.chroma_cqt(y=a, sr=sr, hop_length=hop).mean(axis=1)
    cb = librosa.feature.chroma_cqt(y=b, sr=sr, hop_length=hop).mean(axis=1)
    den = float(np.linalg.norm(ca) * np.linalg.norm(cb))
    out["chroma_cos"] = float(np.clip(np.dot(ca, cb) / den, 0.0, 1.0)) if den > _EPS else 0.0

    oa = librosa.onset.onset_detect(y=a, sr=sr, hop_length=hop, units="time")
    ob = librosa.onset.onset_detect(y=b, sr=sr, hop_length=hop, units="time")
    da = len(oa) / (a.shape[0] / sr)
    db = len(ob) / (b.shape[0] / sr)
    out["onset_ratio"] = float(np.clip(db / da, 0.0, 2.0)) if da > _EPS else 0.0

    p = cb / (cb.sum() + _EPS)
    out["pc_entropy"] = float(-(p * np.log2(p + _EPS)).sum())

    return out


def _summarise(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                  "p05": float(np.percentile(v, 5)), "p50": float(np.percentile(v, 50)),
                  "p95": float(np.percentile(v, 95)),
                  "max": float(v.max()), "min": float(v.min())}
    return out


def from_contexts(contexts_json: Path, sr: int, limit: int = 0,
                  audio_root: str | None = None) -> tuple[list[dict], dict]:
    """Reference distribution for CONTINUATION: real music continuing itself.

    A = the context window, B = the seconds that genuinely follow it in the same track.
    This is what "not copying" looks like when the two segments are disjoint, which is
    the geometry GRPO trains in. It is NOT the same geometry as the re-ranking windows,
    where A and B are temporally co-located and share their lead-in and lead-out.
    """
    d = json.loads(contexts_json.read_text())
    rows, per_pair = [], {}
    items = d["contexts"][:limit] if limit else d["contexts"]
    print(f"scoring {len(items)} real-music continuations at {sr} Hz", flush=True)
    print("  loading librosa ...", flush=True)
    import librosa
    print("  ready", flush=True)
    for i, c in enumerate(items, 1):
        path = resolve_audio(c["audio"], audio_root)
        dur = c["context_end_s"] - c["context_start_s"]
        y_a, _ = librosa.load(path, sr=sr, mono=True,
                              offset=c["context_start_s"], duration=dur)
        y_b, _ = librosa.load(path, sr=sr, mono=True,
                              offset=c["context_end_s"], duration=c["generate_s"])
        if y_b.shape[0] < sr:                       # track too short past the context
            continue
        m = copy_metrics(y_a, y_b, sr)
        rows.append(m)
        per_pair[c["context_id"]] = m
        if i % 10 == 0 or i == len(items):
            print(f"  {i}/{len(items)}", flush=True)
    return rows, per_pair


def compare(paths: list[str]) -> None:
    """Print several baselines side by side.

    The number that matters is not any single distribution but the gap between them:
    real music continuing itself is the floor for "not copying", the frozen policy is
    where RL starts, and a fine-tuned checkpoint is what you are watching for drift.
    """
    loaded = [(Path(p).stem, json.loads(Path(p).read_text())) for p in paths]
    metrics = list(loaded[0][1]["summary"])
    width = max(len(n) for n, _ in loaded) + 2
    for m in metrics:
        print(f"\n{m}")
        print(f"  {'source':<{width}}{'mean':>8}{'sd':>8}{'p95':>8}{'max':>8}  n")
        for name, d in loaded:
            s = d["summary"][m]
            print(f"  {name:<{width}}{s['mean']:>8.3f}{s['sd']:>8.3f}{s['p95']:>8.3f}"
                  f"{s['max']:>8.3f}  {d['meta']['n_pairs']}")
    print("\nGeometry matters: only compare distributions whose A and B overlap the same "
          "way.\nThe hinge for continuation training should come from the frozen-policy "
          "row.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs="+", default=None,
                    help="two or more copy_*.json files to print side by side")
    ap.add_argument("--audio-dir", default=str(ROOT / "results/rerank_ace/audio"),
                    help="dir of <seed>__A.wav and <seed>_cand<NN>__B.wav")
    ap.add_argument("--from-contexts", default=None,
                    help="path to results/rl/contexts.json; baselines real music continuing "
                         "itself (the correct reference for continuation training)")
    ap.add_argument("--out", default=str(ROOT / "results/rl/copy_baseline_ace.json"))
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke test)")
    ap.add_argument("--audio-root", default=None,
                    help="prefix for relative paths in contexts.json (e.g. /root/autodl-tmp); "
                         "or set T2M_DATA_ROOT")
    a = ap.parse_args()

    if a.compare:
        compare(a.compare)
        return

    if a.from_contexts:
        rows, per_pair = from_contexts(Path(a.from_contexts), a.sr, a.limit, a.audio_root)
        summary = _summarise(rows)
        out = {"meta": {"contexts": a.from_contexts, "n_pairs": len(rows), "sr": a.sr,
                        "geometry": "disjoint: B follows A in the same real track",
                        "purpose": "reference for CONTINUATION training; the hinge for the "
                                   "copy penalty should come from here, not from the "
                                   "co-located re-ranking windows"},
               "summary": summary, "per_pair": per_pair}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"\nreal-music continuation reference over {len(rows)} pairs:")
        print(f"{'metric':<14}{'mean':>8}{'sd':>8}{'p95':>8}{'max':>8}")
        for k, v in summary.items():
            print(f"{k:<14}{v['mean']:>8.3f}{v['sd']:>8.3f}{v['p95']:>8.3f}{v['max']:>8.3f}")
        print(f"\nwrote {a.out}")
        return

    audio = Path(a.audio_dir)
    bs = sorted(audio.glob("*__B.wav"))
    if a.limit:
        bs = bs[:a.limit]
    if not bs:
        raise SystemExit(f"no *__B.wav under {audio}")

    print(f"scoring {len(bs)} pairs from {audio} at {a.sr} Hz "
          f"(chroma-CQT dominates; expect ~1 s/pair)", flush=True)
    print("  loading librosa ...", flush=True)
    import librosa
    print("  ready", flush=True)

    rows, per_pair = [], {}
    cache: dict[str, np.ndarray] = {}
    t_start = time.time()
    for i, bp in enumerate(bs, 1):
        seed = bp.name.split("_cand")[0]
        ap_ = audio / f"{seed}__A.wav"
        if not ap_.exists():
            print(f"  ! no context for {bp.name}, skipped", flush=True)
            continue
        if seed not in cache:
            cache[seed], _ = librosa.load(ap_, sr=a.sr, mono=True)
        y_b, _ = librosa.load(bp, sr=a.sr, mono=True)
        m = copy_metrics(cache[seed], y_b, a.sr)
        rows.append(m)
        per_pair[bp.stem.replace("__B", "")] = m
        if i % 10 == 0 or i == len(bs):
            el = time.time() - t_start
            eta = el / i * (len(bs) - i)
            print(f"  {i}/{len(bs)}  {el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)

    summary = _summarise(rows)
    out = {"meta": {"audio_dir": str(audio), "n_pairs": len(rows), "sr": a.sr,
                    "purpose": "pre-RL baseline; compare fine-tuned checkpoints against this"},
           "summary": summary, "per_pair": per_pair}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))

    print(f"\nfrozen-generator baseline over {len(rows)} candidates:")
    print(f"{'metric':<14}{'mean':>8}{'sd':>8}{'p95':>8}{'max':>8}")
    for k, v in summary.items():
        print(f"{k:<14}{v['mean']:>8.3f}{v['sd']:>8.3f}{v['p95']:>8.3f}{v['max']:>8.3f}")
    print(f"\nwrote {a.out}")
    print("Flag a checkpoint if any metric's mean exceeds this baseline's p95.")


if __name__ == "__main__":
    main()
