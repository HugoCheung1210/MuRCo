#!/usr/bin/env python
"""Wall-clock cost of each dimension, for Table~\\ref{tab:latency} in Chapter 3.

The thesis asserted "milliseconds on a CPU" for H/T/R until this was run (2026-08-14). H+T+R
is ~310 ms for a ~10 s pair on the GPU box, so the claim was out by an order of magnitude
while the conclusion it supported -- comfortably faster than realtime, fine inside a loop --
held with room to spare (~32x). Re-run this if the dimensions change and regenerate the
table, rather than trusting the prose.

**Measure on an unloaded machine.** Contention only ever adds time, and it dominates: the
same 60 pairs gave 901-2136 ms on a loaded 2017 laptop against a stable 311-334 ms on an
idle Xeon Gold 6430. Hence --repeats, which keeps the fastest pass. Do not run this while
GRPO training is going, since its reward path is competing for the same cores.

Audio is preloaded before timing, so these are computation costs and not file reading. Each
dimension re-extracts features for BOTH segments on every call; a re-ranker scoring N
candidates against one fixed A could cache A's features and save close to half, which is
not implemented and is noted as such in the text.

S is not timed here, because it needs the MF env and a GPU. See --s-ms for the provenance
of the 501 ms the table carries and how to re-derive it.

    python scripts/analysis/measure_latency.py                 # 40 pairs
    python scripts/analysis/measure_latency.py -n 60 --repeats 5 \
        --tex results/diagnostics/latency.tex
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass

DOMINANT = {  # what each dimension actually spends its time on, for the table
    "H": "constant-$Q$ chroma",
    "T": "MFCC and two covariances",
    "R": "onset envelope, tempo, autocorrelation",
}


def cpu_name() -> str:
    """A quotable CPU string. platform.processor() is empty on Linux, so read cpuinfo."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        import subprocess
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001
        return platform.processor() or platform.machine()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "perturbations/pairs_manifest.json")
    ap.add_argument("--audio-dir", type=Path, default=None,
                    help="time on consecutive wav pairs from this directory instead of "
                         "the manifest. For the GPU box, where the perturbation battery "
                         "may not be present. Cost depends on segment length and sample "
                         "rate, not on which music it is, so any wavs will do -- but "
                         "state the length alongside the number, since it scales with it.")
    ap.add_argument("-n", type=int, default=40, help="pairs to time (default 40)")
    ap.add_argument("--repeats", type=int, default=5,
                    help="time the whole set this many times and keep the fastest pass. "
                         "Contention only ever ADDS time, so the minimum is the estimate "
                         "of the cost itself; means on a shared machine drift by 2x.")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--s-ms", type=float, default=501.0,
                    help="S's cost, quoted rather than timed here (needs the MF env and a "
                         "GPU). The default 501 ms was measured 2026-08-14 on the same box "
                         "as the CPU figures (Xeon Gold 6430 + RTX 4090) by running "
                         "score_s_mf.py over 100 and then 200 battery pairs at --secs 8 "
                         "(73.1 s and 123.2 s) and differencing, which cancels the ~23 s of "
                         "model loading. The GRPO loop's 460 ms (grpo_pilot_runbook.md) is "
                         "the same operation on 6 s windows batched 8 at a time.")
    ap.add_argument("--tex", type=Path, default=None)
    a = ap.parse_args()

    import librosa

    from coherence_dimensions import (HarmonicDimension, RhythmicDimension,
                                      TimbralDimension)

    loaded = []
    if a.audio_dir:
        wavs = sorted(a.audio_dir.glob("*.wav"))
        if len(wavs) < 2:
            raise SystemExit(f"need at least 2 wavs in {a.audio_dir}")
        for i in range(min(a.n, len(wavs) // 2)):
            ya, _ = librosa.load(wavs[2 * i], sr=a.sr, mono=True)
            yb, _ = librosa.load(wavs[2 * i + 1], sr=a.sr, mono=True)
            n = min(len(ya), len(yb))                     # equal length, as a real pair is
            loaded.append((ya[:n], yb[:n]))
        source = str(a.audio_dir)
    else:
        for p in json.loads(a.manifest.read_text())["pairs"][: a.n]:
            ya, _ = librosa.load(ROOT / p["ref_path"], sr=a.sr, mono=True)
            yb, _ = librosa.load(ROOT / p["cand_path"], sr=a.sr, mono=True)
            loaded.append((ya, yb))
        source = str(a.manifest)
    secs = len(loaded[0][0]) / a.sr
    print(f"{len(loaded)} pairs, {secs:.1f}s per segment, sr={a.sr}, audio preloaded")
    print(f"source: {source}")
    print(f"cpu: {cpu_name()}")

    dims = {"H": HarmonicDimension(), "T": TimbralDimension(), "R": RhythmicDimension()}
    for _ in range(3):                                   # warm any lazy imports/caches
        dims["H"].score(*loaded[0], a.sr)

    passes = []                                          # one entry per repeat
    for rep in range(a.repeats):
        per_dim = {}
        for k, d in dims.items():
            ts = []
            for ya, yb in loaded:
                t0 = time.perf_counter()
                d.score(ya, yb, a.sr)
                ts.append((time.perf_counter() - t0) * 1e3)
            per_dim[k] = np.asarray(ts)
        tot = float(np.sum([per_dim[k].mean() for k in dims]))
        passes.append((tot, per_dim))
        print(f"  pass {rep + 1}/{a.repeats}: H+T+R mean {tot:.0f} ms", flush=True)

    spread = [p[0] for p in passes]
    per_dim = min(passes, key=lambda p: p[0])[1]          # the least-contended pass
    total = np.sum([per_dim[k] for k in dims], axis=0)

    rows = []
    print(f"\nfastest of {a.repeats} passes (pass totals ranged "
          f"{min(spread):.0f}-{max(spread):.0f} ms)")
    print(f"{'dim':>7}{'mean ms':>10}{'p50':>9}{'p95':>9}")
    for k in dims:
        t = per_dim[k]
        rows.append((k, t.mean(), np.percentile(t, 95)))
        print(f"{k:>7}{t.mean():>10.1f}{np.percentile(t, 50):>9.1f}"
              f"{np.percentile(t, 95):>9.1f}")
    print(f"{'H+T+R':>7}{total.mean():>10.1f}{np.percentile(total, 50):>9.1f}"
          f"{np.percentile(total, 95):>9.1f}")
    print(f"\nreal-time factor: {secs / (total.mean() / 1e3):.0f}x faster than realtime")
    print(f"H+R share of CPU time: "
          f"{100 * (per_dim['H'].mean() + per_dim['R'].mean()) / total.mean():.0f}%")
    print(f"{cpu_name()}, python {platform.python_version()}, "
          f"librosa {librosa.__version__}, numpy {np.__version__}")

    if a.tex:
        # Absolute milliseconds are machine- and load-dependent: repeated passes on one
        # laptop ranged 0.9-2.1 s for H+T+R. The SHARE of cost is stable to a point or two
        # across every run, so the table leads with share and reports the absolute rounded
        # to one significant figure, which is all it can honestly support.
        a.tex.parent.mkdir(parents=True, exist_ok=True)
        tot_ms = total.mean()

        def approx(v):                                   # 1 s.f. for values >= 100 ms
            return f"{v / 1000:.1f}\\,s" if v >= 1000 else f"{round(v, -1):.0f}\\,ms"

        L = [r"\begin{tabular}{lrrl}", r"\toprule",
             r"Dimension & share of CPU cost & approx. & Dominant cost \\", r"\midrule"]
        for k, m, _p95 in rows:
            L.append(f"${k}$ & ${100 * m / tot_ms:.0f}\\%$ & {approx(m)} & "
                     f"{DOMINANT[k]} \\\\")
        L += [r"\midrule",
              f"$H,T,R$ & $100\\%$ & {approx(tot_ms)} & CPU, one core \\\\",
              f"$S$ & --- & {approx(a.s_ms)} & GPU, $16$\\,GB resident \\\\",
              r"\bottomrule", r"\end{tabular}"]
        a.tex.write_text("\n".join(L) + "\n")
        print(f"\nwrote {a.tex}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
