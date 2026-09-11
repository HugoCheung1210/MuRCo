"""Run the Setup-1 (iterative-drift) H/T/R/S/C evaluation for one edit model.

Thin CLI wrapper around `drift_htrc` + `drift_v4` so each model's correct
region / ceiling / clip-dir are hardcoded in one place instead of copy-pasted
(regions follow each model's protocol file, NOT the stale drift_v4 docstring).
This removes the "wrong trim region" footgun (PROJECT_STATE.md §5 / ablation §2.2).
region is (start_s, DURATION_s) — drift_v4.run does `start, secs = region`, so
(4,6) means "start at 4s, run 6s" = 4-10s → a 13s concat (6+1s gap+6). Both
roundtrip models are scored at (4,6) so they're length-matched (the SAO-vs-ACE
delta test needs identical concat length) and inside MF's audio window (21s
collapses the numeric scorer): SAO inpaints 5-8s, ACE repaints 5-15s but is scored
at (4,6) (first ~5s of the edit, per ace_protocol.py:21). MusicGen is a
*continuation* model — iter_k IS the new segment, so it scores the whole (0,10)
clip (21s concat) against a `self` ceiling with NO roundtrip dir. Copying SAO's
params onto MusicGen would silently score the wrong window against the wrong baseline.

Runs in `mfenv` on the GPU box (the S/mf term needs Music Flamingo). H/T/R are
CPU; pass `--dims H,T,R` to skip MF entirely and aggregate the DSP dims alone.

    # full C on MusicGen (what §8.3 / ablation §4.4 wants for a 2nd edit family)
    python scripts/drift/run_drift_htrc.py --model musicgen

    # DSP-only, no GPU needed
    python scripts/drift/run_drift_htrc.py --model ace --dims H,T,R

Output matches the per-dimension drop-vs-ceiling table already recorded for SAO.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable


import os
import argparse
import json

# HF env MUST be set before mf_probe is imported (see env/).
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np


# Per-model config. region is (start_s, DURATION_s) — drift_v4.run does
# `start, secs = region` (drift_v4.py:130), so (4,6) = start 4s, run 6s = 4-10s.
# rt_dir is required only for ceiling=="roundtrip".
MODELS = {
    "sao": {
        # inpaint at 5-8s; (4,6)=4-10s straddles it. Matches the validated SAO run.
        "dir": "autodl-tmp/outputs_stableaudio_inpaint_5_8/editing",
        "region": (4, 6),
        "ceiling": "roundtrip",
        "rt_dir": "autodl-tmp/outputs_stableaudio_inpaint_5_8/roundtrip",
        "tag": "SAO",
    },
    "ace": {
        # ACE repaints 5-15s (ace_protocol.py REPAINT_START/END), but is SCORED at
        # region=(4,6)=4-10s per ace_protocol.py:21 + test_drift_v2.py. Deliberate:
        # 13s concat (6+1+6) fits MF's audio window (21s collapses the numeric scorer)
        # AND matches SAO's 13s so the SAO-vs-ACE delta test isn't confounded by length.
        # (4,6) covers the first ~5s of the 10s edit — partial but overwhelmingly edited.
        # The drift_v4 docstring's "-> region=(5,10)" is stale; the protocol uses (4,6).
        "dir": "outputs/E2_iterative",
        "region": (4, 6),
        "ceiling": "roundtrip",
        "rt_dir": "outputs/C1_roundtrip",
        "tag": "ACE",
    },
    "musicgen": {
        # iter_k IS the new segment; score whole clip. (0,10)=0-10s (same either way).
        "dir": "autodl-tmp/outputs_musicgen/continuation",
        "region": (0, 10),
        "ceiling": "self",   # continuation model: no roundtrip baseline
        "rt_dir": None,
        "tag": "MG",
    },
}

# metric name passed to drift_v4.run per dimension (S is MF, keyed "mf" there).
_METRIC = {"H": "H", "T": "T", "R": "R", "S": "mf"}

# --s-readout logit swaps S's read-out to the P(Yes) template bank of Chapter 3.
# "numeric" is the default so every published drift number reproduces unchanged.
_S_METRIC = {"numeric": "mf", "logit": "mf_logit", "balanced": "mf_balanced"}

# Which end of A to score, per model. Only continuation differs: there B starts
# where A ends, so the junction is at A's TAIL. Ignored by the numeric read-out,
# which concatenates the whole clipped region either way.
_A_FROM = {"sao": "head", "ace": "head", "musicgen": "tail"}


def _summary(res):
    """Ceiling/per-iter means + drop-vs-ceiling for one run() result (JSON-safe).

    Also dumps `per_seed` (T2): run() computes {sid: score} per iteration, but
    this writer used to collapse it to mean/std/n, so every seed-level analysis
    (bootstrap CIs over seeds, per-seed Spearman rho vs depth, depth-ordering
    accuracy) needed a fresh GPU pass to recover data we already had.  The raw
    dicts are a few KB -- keep them.
    """
    import drift_htrc as dx
    import drift_v4 as d4
    cm, csd, cN = dx._mean(res["identical"])
    iters = {}
    for k in d4.KS:
        km, ksd, kN = dx._mean(res["per_k"][k])
        iters[str(k)] = {"mean": km, "std": ksd, "n": kN, "drop": cm - km}
    return {"name": res["name"], "metric": res["metric"], "region": list(res["region"]),
            "ceiling": res["ceiling"], "ceiling_mean": cm, "ceiling_std": csd,
            "ceiling_n": cN, "iters": iters,
            "per_seed": {"ceiling": {s: _jsonable(v) for s, v in res["identical"].items()},
                         "iters": {str(k): {s: _jsonable(v) for s, v in res["per_k"][k].items()}
                                   for k in d4.KS}}}


def _jsonable(v):
    """NaN -> None (json.dump emits bare NaN, which is not valid JSON)."""
    import math
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


def _accum_stats(res, k_lo=1, k_hi=8):
    """Paired iter{k_lo}-vs-iter{k_hi} accumulation test on aligned seeds.

    Reuses drift_v4's own primitives (`_paired` + `_wilcoxon_es`). Sign convention:
    a=iter_lo, b=iter_hi, so d=a-b and **positive dz = score falls with depth =
    drift accumulates**. Returns None if <1 aligned non-nan seed pair.
    """
    import drift_v4 as d4
    sids, a, b = d4._paired(res["per_k"][k_lo], res["per_k"][k_hi])
    if len(a) < 1:
        return None
    st = d4._wilcoxon_es(a, b)          # tests iter_lo vs iter_hi
    st["mean_lo"], st["mean_hi"] = float(a.mean()), float(b.mean())
    st["delta"] = st["mean_lo"] - st["mean_hi"]
    return st


def _print_accum(results, order, k_lo=1, k_hi=8):
    """One row per dim+C: iter_lo/iter_hi means, Δ, dz, rank-biserial, W, p, n."""
    print(f"\n===== accumulation: iter{k_lo} vs iter{k_hi} "
          f"(paired; +dz = score falls with depth = drift accumulates) =====")
    print(f"{'dim':4s}{'iter'+str(k_lo):>9s}{'iter'+str(k_hi):>9s}{'Δ':>9s}"
          f"{'dz':>8s}{'rb':>8s}{'W':>8s}{'p':>10s}{'n':>4s}")
    for d in order:
        st = _accum_stats(results[d], k_lo, k_hi)
        if st is None:
            print(f"{d:4s}{'(no aligned seeds)':>50s}")
            continue
        print(f"{d:4s}{st['mean_lo']:9.3f}{st['mean_hi']:9.3f}{st['delta']:+9.3f}"
              f"{st['dz']:+8.2f}{st['rb']:+8.2f}{st['W']:8.1f}{st['p']:10.2e}"
              f"{st['n']:4d}")
    print("(dz≈0.2 small / 0.5 medium / 0.8 large; p from two-sided Wilcoxon, n=10 "
          "floors it — read dz + Δ, not p alone)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS),
                    help="which edit model's drift clips to score")
    ap.add_argument("--dims", default="H,T,R,S",
                    help="comma-separated subset of H,T,R,S (default all). "
                         "Omit S to skip MF and run CPU-only.")
    ap.add_argument("--dir", default=None, help="override the model's clip dir")
    ap.add_argument("--rt-dir", default=None, help="override the roundtrip-ceiling dir")
    ap.add_argument("--out", default=None, help="optional JSON path for the summary table")
    ap.add_argument("--s-secs", type=float, default=6.0,
                    help="seconds of A and of B the logit read-out scores (default 6, "
                         "giving the 6+1+6 s concatenation used throughout)")
    ap.add_argument("--s-readout", choices=sorted(_S_METRIC), default="numeric",
                    help="numeric = the 0-1 rating parsed from text (the published "
                         "drift runs); logit = the P(Yes) template bank of Chapter 3")
    ap.add_argument("--stats", action="store_true",
                    help="also print the paired iter1-vs-iter{accum-hi} accumulation test "
                         "(Wilcoxon + dz + rank-biserial) per dim + C")
    ap.add_argument("--ks", default=None,
                    help="T5 deep-horizon: comma list of iteration depths to score, e.g. "
                         "'1,2,4,8,16' (default keeps drift_v4.KS = 1,2,4,8). The deeper clips "
                         "must exist -- roll SAO out with SAO_N_ITERS=16 first.")
    ap.add_argument("--accum-hi", type=int, default=8,
                    help="deepest iter for the paired accumulation test (default 8; use 16 "
                         "with --ks 1,2,4,8,16 for the T5 depth contrast)")
    args = ap.parse_args()

    dims = [d.strip().upper() for d in args.dims.split(",") if d.strip()]
    bad = [d for d in dims if d not in _METRIC]
    if bad:
        ap.error(f"unknown dims {bad}; choose from H,T,R,S")
    if not dims:
        ap.error("no dims given; choose from H,T,R,S")

    cfg = MODELS[args.model]
    clip_dir = args.dir or cfg["dir"]
    rt_dir = args.rt_dir or cfg["rt_dir"]
    if cfg["ceiling"] == "roundtrip" and not rt_dir:
        ap.error(f"model '{args.model}' needs a roundtrip dir; pass --rt-dir")

    # Importing drift_htrc monkey-patches drift_v4.rate to serve H/T/R on CPU.
    import drift_htrc as dx
    import drift_v4 as d4

    if args.ks:                                   # T5: override the scored depths
        d4.KS = [int(x) for x in args.ks.split(",") if x.strip()]
        print(f"scoring depths KS = {d4.KS}")
    if args.accum_hi not in d4.KS:
        ap.error(f"--accum-hi {args.accum_hi} not in scored depths {d4.KS}; add it via --ks")

    if "S" in dims:
        import mf_probe as mf
        mf.load()
        d4.set_backend(mf)
        _METRIC["S"] = _S_METRIC[args.s_readout]
        d4.A_FROM = _A_FROM[args.model]
        if args.s_readout in ("logit", "balanced"):
            # 6 s a side, so the concatenation is 6+1+6 = 13 s for EVERY model.
            # Not the region duration: musicgen's region is (0,10), which would give
            # a 21 s concatenation, past what Music Flamingo reads (ablation 2.2) and
            # different in length from the sao/ace runs, confounding the cross-regime
            # comparison with clip length. a_from="tail" puts those 6 s at the junction.
            d4.S_SECS = float(args.s_secs)
            bank = ("P(Yes) template bank" if args.s_readout == "logit"
                    else "polarity-balanced (pos + 1-neg)/2")
            print(f"S read-out: {bank}, {d4.S_SECS:g}s a side, "
                  f"gap {d4.GAP_S:g}s, A from {d4.A_FROM}")

    run_kwargs = {"region": tuple(cfg["region"]), "ceiling": cfg["ceiling"]}
    if cfg["ceiling"] == "roundtrip":
        run_kwargs["rt_dir"] = rt_dir

    results = {}
    for d in dims:
        results[d] = d4.run(clip_dir, f"{cfg['tag']}-{d}", metric=_METRIC[d], **run_kwargs)

    order = list(dims)
    if len(dims) >= 2:
        results["C"] = dx.aggregate_C({d: results[d] for d in dims})
        order = dims + ["C"]
        # d4.run() already prints each per-dimension table as it computes it; only the
        # aggregated C (from aggregate_C, which doesn't print) still needs printing.
        dx.print_drift(results["C"])
    else:
        print(f"single dimension {dims[0]}: no C to aggregate")
    dx.per_dim_table({d: results[d] for d in order})
    if args.stats:
        _print_accum(results, order, k_hi=args.accum_hi)

    if args.out:
        summary = {d: _summary(results[d]) for d in order}
        if args.stats:
            for d in order:
                summary[d][f"accum_iter1_vs_iter{args.accum_hi}"] = \
                    _accum_stats(results[d], k_hi=args.accum_hi)
        # scoring here costs GPU minutes, so never lose it to a missing directory
        import os as _os
        _d = _os.path.dirname(_os.path.abspath(args.out))
        _os.makedirs(_d, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
