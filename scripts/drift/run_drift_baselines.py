"""Run the off-the-shelf BASELINES on the Setup-1 iterative-drift clips.

The DECISIVE test after E2.8 (PROJECT_STATE §8.1 item 1): identity-AUC showed CLAP/
SSM BEAT S at telling songs apart, so S's justification now rests solely on the
iterative-drift regime (§8.3) — S accumulates in SAO inpaint where the DSP dims are
blind. This asks the adjudicating question: does a cheap audio-similarity metric
ALSO climb with edit depth, or is it flat while only S accumulates?

Reuses the SAME drift harness as the S term (`drift_v4.run`, same per-model region/
ceiling/pairing), so the baseline drop-vs-ceiling and paired iter1-vs-iter8
accumulation tables line up 1:1 with the S numbers in §8.3 — no separate manifest,
no protocol drift. Metrics:
  * clap_htsat  -- LAION-CLAP htsat-unfused audio-embedding cosine
  * clap_music  -- music-specialised CLAP
  * scs         -- structural SSM stand-in (CPU; see drift_v4.scs_sim)

Self-contained: depends ONLY on drift_v4 (the model config + accumulation-stats
helpers are inlined, so it does not care which version of run_drift_htrc is on the
box). Runs on the GPU box (CLAP needs cuda; SCS is CPU). NB unlike run_drift_htrc
there is NO C aggregation — baselines are external metrics, not dimensions of C.

    python scripts/drift/run_drift_baselines.py --model sao \
        --metrics clap_htsat,clap_music,scs --stats --out results/drift/drift_baselines_sao.json

Read alongside §8.3: if a baseline's iter1-vs-iter8 drop ~0 (flat) while S's is
large, that is S's genuine, baseline-beating contribution. If the baseline
accumulates too, S is in trouble.
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

# HF env MUST be set before any MF import path (kept for parity; baselines don't
# need MF, but drift_v4 is shared with the S pipeline).
os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np

# --- per-model drift config (mirrors run_drift_htrc.MODELS; inlined so this script
#     is independent of the box's run_drift_htrc version). region=(start_s,DUR_s). ---
MODELS = {
    "sao": {
        "dir": "autodl-tmp/outputs_stableaudio_inpaint_5_8/editing",
        "region": (4, 6), "ceiling": "roundtrip",
        "rt_dir": "autodl-tmp/outputs_stableaudio_inpaint_5_8/roundtrip", "tag": "SAO",
    },
    "ace": {
        "dir": "outputs/E2_iterative",
        "region": (4, 6), "ceiling": "roundtrip",
        "rt_dir": "outputs/C1_roundtrip", "tag": "ACE",
    },
    "musicgen": {
        "dir": "autodl-tmp/outputs_musicgen/continuation",
        "region": (0, 10), "ceiling": "self", "rt_dir": None, "tag": "MG",
    },
}

CLAP_CKPT = {
    "clap_htsat": "laion/clap-htsat-unfused",
    "clap_music": "laion/larger_clap_music",
}
BASELINE_METRICS = list(CLAP_CKPT) + ["scs"]


def _mean(d):
    """(mean, std, n) over non-nan values of a {sid: value} dict."""
    x = [v for v in d.values() if not np.isnan(v)]
    return (float(np.mean(x)), float(np.std(x)), len(x)) if x else (float("nan"), 0.0, 0)


def _accum_stats(res, k_lo=1, k_hi=8):
    """Paired iter{k_lo}-vs-iter{k_hi} accumulation test (reuses drift_v4 primitives).

    Sign: a=iter_lo, b=iter_hi, d=a-b, so +dz = score falls with depth = drift
    accumulates. Returns None if <1 aligned non-nan seed pair.
    """
    import drift_v4 as d4
    sids, a, b = d4._paired(res["per_k"][k_lo], res["per_k"][k_hi])
    if len(a) < 1:
        return None
    st = d4._wilcoxon_es(a, b)
    st["mean_lo"], st["mean_hi"] = float(a.mean()), float(b.mean())
    st["delta"] = st["mean_lo"] - st["mean_hi"]
    return st


def _print_accum(results, order, k_lo=1, k_hi=8):
    print(f"\n===== accumulation: iter{k_lo} vs iter{k_hi} "
          f"(paired; +dz = score falls with depth = drift accumulates) =====")
    print(f"{'metric':12s}{'iter'+str(k_lo):>9s}{'iter'+str(k_hi):>9s}{'Δ':>9s}"
          f"{'dz':>8s}{'rb':>8s}{'W':>8s}{'p':>10s}{'n':>4s}")
    for d in order:
        st = _accum_stats(results[d], k_lo, k_hi)
        if st is None:
            print(f"{d:12s}{'(no aligned seeds)':>50s}")
            continue
        print(f"{d:12s}{st['mean_lo']:9.3f}{st['mean_hi']:9.3f}{st['delta']:+9.3f}"
              f"{st['dz']:+8.2f}{st['rb']:+8.2f}{st['W']:8.1f}{st['p']:10.2e}{st['n']:4d}")
    print("(dz≈0.2 small / 0.5 medium / 0.8 large; n=10 floors p — read dz + Δ, not p)")


def _jsonable(v):
    """NaN -> None (json.dump emits bare NaN, which is not valid JSON)."""
    import math
    return None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)


def _summary(res):
    """Ceiling/per-iter means + drop-vs-ceiling, PLUS the per-seed dicts (T2).

    Mirrors run_drift_htrc._summary so S/H/T/R and the baselines stay 1:1
    comparable at seed level -- selectivity and rank statistics compare metrics
    across models, and that needs the same seeds on both sides.
    """
    import drift_v4 as d4
    cm, csd, cN = _mean(res["identical"])
    iters = {}
    for k in d4.KS:
        km, ksd, kN = _mean(res["per_k"][k])
        iters[str(k)] = {"mean": km, "std": ksd, "n": kN, "drop": cm - km}
    return {"name": res["name"], "metric": res["metric"], "region": list(res["region"]),
            "ceiling": res["ceiling"], "ceiling_mean": cm, "ceiling_std": csd,
            "ceiling_n": cN, "iters": iters,
            "per_seed": {"ceiling": {s: _jsonable(v) for s, v in res["identical"].items()},
                         "iters": {str(k): {s: _jsonable(v) for s, v in res["per_k"][k].items()}
                                   for k in d4.KS}}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS),
                    help="which edit model's drift clips to score (same dirs as run_drift_htrc)")
    ap.add_argument("--metrics", default="clap_htsat,clap_music,scs",
                    help="comma list from %s, or 'all'" % ",".join(BASELINE_METRICS))
    ap.add_argument("--dir", default=None, help="override the model's clip dir")
    ap.add_argument("--rt-dir", default=None, help="override the roundtrip-ceiling dir")
    ap.add_argument("--seeds-dir", default=None, help="override seeds dir (default: drift_v4's)")
    ap.add_argument("--out", default=None, help="optional JSON path for the summary table")
    ap.add_argument("--stats", action="store_true",
                    help="also print the paired iter1-vs-iter{accum-hi} accumulation test per metric")
    ap.add_argument("--ks", default=None,
                    help="T5 deep-horizon: comma list of iteration depths to score, e.g. "
                         "'1,2,4,8,16' (default keeps drift_v4.KS = 1,2,4,8). Must match the "
                         "--ks used for run_drift_htrc so S and baselines stay 1:1 comparable.")
    ap.add_argument("--accum-hi", type=int, default=8,
                    help="deepest iter for the paired accumulation test (default 8; use 16 "
                         "with --ks 1,2,4,8,16 for the T5 depth contrast)")
    args = ap.parse_args()

    metrics = BASELINE_METRICS if args.metrics == "all" else \
        [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in BASELINE_METRICS]
    if bad:
        ap.error(f"unknown metric(s) {bad}; choose from {BASELINE_METRICS}")

    cfg = MODELS[args.model]
    clip_dir = args.dir or cfg["dir"]
    rt_dir = args.rt_dir or cfg["rt_dir"]
    if cfg["ceiling"] == "roundtrip" and not rt_dir:
        ap.error(f"model '{args.model}' needs a roundtrip dir; pass --rt-dir")

    import drift_v4 as d4

    if args.ks:                                   # T5: override the scored depths
        d4.KS = [int(x) for x in args.ks.split(",") if x.strip()]
        print(f"scoring depths KS = {d4.KS}")
    if args.accum_hi not in d4.KS:
        ap.error(f"--accum-hi {args.accum_hi} not in scored depths {d4.KS}; add it via --ks")

    run_kwargs = dict(region=tuple(cfg["region"]), ceiling=cfg["ceiling"])
    if cfg["ceiling"] == "roundtrip":
        run_kwargs["rt_dir"] = rt_dir
    if args.seeds_dir:
        run_kwargs["seeds_dir"] = args.seeds_dir

    results = {}
    for m in metrics:
        if m in CLAP_CKPT:
            d4.load_clap(CLAP_CKPT[m])                    # (re)loads the global CLAP model
            d4_metric = "clap"
        else:
            d4_metric = "scs"
        results[m] = d4.run(clip_dir, f"{cfg['tag']}-{m}", metric=d4_metric, **run_kwargs)

    if args.stats:
        _print_accum(results, metrics, k_hi=args.accum_hi)

    if args.out:
        summary = {m: _summary(results[m]) for m in metrics}
        if args.stats:
            for m in metrics:
                summary[m][f"accum_iter1_vs_iter{args.accum_hi}"] = \
                    _accum_stats(results[m], k_hi=args.accum_hi)
        # scoring here costs GPU minutes, so never lose it to a missing directory
        import os as _os
        _d = _os.path.dirname(_os.path.abspath(args.out))
        _os.makedirs(_d, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
