"""Extend drift_v4 to score H/T/R and the aggregated metric C on drift clips.

Setup-1 (preservation / iterative-drift) was previously scored with S (MF) and
CLAP only.  This module wires the SAME H/T/R dimensions used in the Setup-2
separability matrix (from coherence_dimensions.py) into drift_v4's existing
seed-vs-iter_k harness, and aggregates them (with S) into C -- so the "does the
whole metric track real editing drift?" question can be answered, not just "does S".

It is deliberately non-invasive: it monkey-patches drift_v4.rate to recognise
metric in {"H","T","R"} (computed on-CPU from the clipped arrays) and otherwise
defers to the original rate (mf / clap).  drift_v4.run therefore works unchanged:

    import drift_v4 as d4, drift_htrc as dx
    d4.set_backend(mf)                       # for the S/mf term (GPU)
    resH = d4.run(DIR, "SAO-H", region=(4,6), metric="H", ceiling="roundtrip", rt_dir=RT)
    resT = d4.run(DIR, "SAO-T", region=(4,6), metric="T", ceiling="roundtrip", rt_dir=RT)
    resR = d4.run(DIR, "SAO-R", region=(4,6), metric="R", ceiling="roundtrip", rt_dir=RT)
    resS = d4.run(DIR, "SAO-S", region=(4,6), metric="mf", ceiling="roundtrip", rt_dir=RT)
    resC = dx.aggregate_C({"H":resH,"T":resT,"R":resR,"S":resS})   # uniform weights
    dx.print_drift(resC)                     # C's iter1->iter8 curve
    dx.per_dim_table({"H":resH,"T":resT,"R":resR,"S":resS,"C":resC})  # who tracks drift

Key expected finding: iterative drift is sub-stylistic, the same regime where the
signal dimensions are weak.  If H/T/R stay flat while S drops, uniform-weight C
*dilutes* S's signal -- which is the empirical argument for MOS-fitted weights.
Report the per-dimension curves, not just C.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable


import numpy as np
import librosa

import drift_v4 as d4
import coherence_dimensions as cd

# reuse the exact Setup-2 dimension implementations
_DIMS = {"H": cd.HarmonicDimension(), "T": cd.TimbralDimension(), "R": cd.RhythmicDimension()}
_EPS = 1e-6


def _clip_arr(path, start, secs):
    """Mirror drift_v4._clip but return the array (normalised), not a written file."""
    try:
        y, _ = librosa.load(path, sr=d4.SR, mono=True, offset=start, duration=secs)
    except Exception:
        return None
    if len(y) < int(d4.SR * 0.5):
        return None
    return (y / (np.max(np.abs(y)) + 1e-8) * 0.95).astype("float32")


# --- monkey-patch drift_v4.rate to add H/T/R (defer to original for mf/clap) ---
_orig_rate = d4.rate


def rate(path_a, path_b, start, secs, metric="mf"):
    if metric in _DIMS:
        a = _clip_arr(path_a, start, secs)
        b = _clip_arr(path_b, start, secs)
        if a is None or b is None:
            return np.nan
        return float(_DIMS[metric].score(a, b, d4.SR))
    return _orig_rate(path_a, path_b, start, secs, metric)


d4.rate = rate          # now d4.run(..., metric="H"/"T"/"R") works unchanged


# --- aggregate per-dimension run results into C -------------------------------

def _norm_weights(dims, weights):
    w = {d: 1.0 for d in dims} if weights is None else {d: float(weights.get(d, 0.0)) for d in dims}
    tot = sum(w.values())
    if tot <= 0:
        raise ValueError("weights sum to 0")
    return {d: w[d] / tot for d in dims}


def aggregate_C(dim_results, weights=None):
    """Combine per-dimension run() results into a geometric-mean C run() result.

    dim_results: {"H":resH, "T":resT, "R":resR, "S":resS} (any subset >=2).
    Returns a dict in the same shape as drift_v4.run output, so it plugs straight
    into print_drift / paired_stats.
    """
    dims = list(dim_results)
    w = _norm_weights(dims, weights)

    def combine(per_sid_dicts):
        # per_sid_dicts: {dim: {sid: value}}; C[sid] = prod(v_d ** w_d) over dims
        sids = set.intersection(*[set(d) for d in per_sid_dicts.values()])
        out = {}
        for sid in sids:
            vals = [per_sid_dicts[d][sid] for d in dims]
            if any(np.isnan(v) for v in vals):
                continue
            out[sid] = float(np.prod([max(_EPS, per_sid_dicts[d][sid]) ** w[d] for d in dims]))
        return out

    ident = combine({d: dim_results[d]["identical"] for d in dims})
    per_k = {}
    for k in d4.KS:
        per_k[k] = combine({d: dim_results[d]["per_k"][k] for d in dims})
    return {"name": "C(" + "".join(dims) + ")", "metric": "C", "weights": w,
            "region": dim_results[dims[0]]["region"],
            "ceiling": dim_results[dims[0]]["ceiling"],
            "identical": ident, "per_k": per_k}


# --- reporting ----------------------------------------------------------------

def _mean(dct):
    x = [v for v in dct.values() if not np.isnan(v)]
    return (float(np.mean(x)), float(np.std(x)), len(x)) if x else (np.nan, 0.0, 0)


def print_drift(res):
    im, isd, iN = _mean(res["identical"])
    print(f"\n===== {res['name']}  [{res['metric']}]  region={res['region']}  "
          f"ceiling={res['ceiling']} =====")
    print(f"{'':10s}{'mean':>8s}{'std':>7s}{'n':>4s}")
    print(f"{'ceiling':10s}{im:8.3f}{isd:7.3f}{iN:4d}")
    for k in d4.KS:
        km, ksd, kN = _mean(res["per_k"][k])
        print(f"iter{k:<6d}{km:8.3f}{ksd:7.3f}{kN:4d}   drop vs ceiling = {im - km:+.3f}")


def per_dim_table(results, ks=None):
    """Side-by-side drop-vs-ceiling for each dimension + C, across iterations.

    Shows WHO tracks drift: if H/T/R are flat and only S (and thus C) drops, that
    is the sub-stylistic-drift finding.
    """
    ks = ks or d4.KS
    names = list(results)
    print(f"\n{'iter':6s}" + "".join(f"{n:>10s}" for n in names) + "   (drop vs ceiling)")
    ceil = {}
    for n in names:
        im, _, _ = _mean(results[n]["identical"]); ceil[n] = im
    for k in ks:
        row = f"iter{k:<2d}"
        for n in names:
            km, _, _ = _mean(results[n]["per_k"][k])
            row += f"{ceil[n]-km:>+10.3f}"
        print(row)
    print("(large positive = that metric detects the drift; ~0 = blind to it)")
