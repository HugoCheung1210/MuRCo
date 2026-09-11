"""
Significance testing + per-difficulty breakdown for the MF/CLAP coherence eval.

Turns the averaged gaps into statistical claims:
  * paired Wilcoxon + paired t-test on each level vs the 'same' control
  * effect size (Cliff's delta / Cohen's d)
  * MF-vs-CLAP head-to-head on the HARD case (does MF separate what CLAP can't?)

Usage (after building triples):
    import mf_eval as ev, mf_stats as st
    ev.set_backend(mf); ev.load_clap("laion/larger_clap_music")
    rows = st.collect(triples)          # per-triple raw scores (the data)
    st.report(rows)                     # significance tables
    st.save_csv(rows, "coherence_results.csv")
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

import numpy as np, csv
from scipy.stats import wilcoxon, ttest_rel

LEVELS = ["same", "cont", "diff", "noise"]
_ev = None

def _backend():
    global _ev
    import mf_eval as ev
    _ev = ev
    return ev


def collect(triples, gap_s=0.0):
    """Run MF (pos/neg/comb) and CLAP per triple, return list of per-triple dicts."""
    ev = _backend()
    rows = []
    for i, t in enumerate(triples):
        a = t["a"]
        rec = {}
        for lv, b in [("same", a), ("cont", t["cont"]), ("diff", t["diff"]), ("noise", t["noise"])]:
            comb, pos, neg = ev.mf_balanced(a, b, gap_s=gap_s)
            rec[f"{lv}_pos"], rec[f"{lv}_neg"], rec[f"{lv}_comb"] = pos, neg, comb
            rec[f"{lv}_clap"] = ev.clap_sim(a, b) if ev._clap is not None else np.nan
        rows.append(rec)
        if (i+1) % 20 == 0: print(f"  collected {i+1}/{len(triples)}")
    print(f"collected {len(rows)} triples")
    return rows


def _arr(rows, key):
    return np.array([r[key] for r in rows], dtype=float)

def _cohen_d(x, y):
    d = x - y
    return float(np.mean(d) / (np.std(d, ddof=1) + 1e-12))

def _cliffs_delta(x, y):
    # fraction(x>y) - fraction(x<y) over paired samples
    gt = np.sum(x > y); lt = np.sum(x < y)
    return float((gt - lt) / len(x))

def _paired(rows, metric, level, ctrl="same", higher_means_incoherent=True):
    """Test whether `level` differs from `ctrl` on `metric`. Returns dict."""
    a = _arr(rows, f"{level}_{metric}")
    c = _arr(rows, f"{ctrl}_{metric}")
    # direction: for 'neg' incoherent should be HIGHER; for pos/comb/clap LOWER
    diff = (a - c) if higher_means_incoherent else (c - a)
    try:
        w_p = wilcoxon(a, c).pvalue
    except Exception:
        w_p = np.nan
    t_p = ttest_rel(a, c).pvalue
    return {"mean_gap": float(np.mean(diff)), "wilcoxon_p": float(w_p),
            "ttest_p": float(t_p), "cohen_d": _cohen_d(diff, np.zeros_like(diff)),
            "cliffs_delta": _cliffs_delta(diff, np.zeros_like(diff)), "n": len(a)}


def report(rows):
    n = len(rows)
    print(f"\n================ SIGNIFICANCE (paired, n={n}) ================")
    # metric -> whether higher value means MORE incoherent
    metrics = {"pos": False, "neg": True, "comb": False, "clap": False}
    for metric, hi_inc in metrics.items():
        print(f"\n--- {metric.upper()} (vs 'same' control; gap>0 = separates incoherence) ---")
        print(f"{'level':6s} {'gap':>8s} {'wilcoxon_p':>12s} {'cohen_d':>9s} {'cliffs_d':>9s}")
        for lv in ["cont", "diff", "noise"]:
            r = _paired(rows, metric, lv, higher_means_incoherent=hi_inc)
            sig = "***" if r["wilcoxon_p"] < 1e-3 else "**" if r["wilcoxon_p"] < 1e-2 else "*" if r["wilcoxon_p"] < 0.05 else "ns"
            print(f"{lv:6s} {r['mean_gap']:+8.3f} {r['wilcoxon_p']:12.2e} "
                  f"{r['cohen_d']:+9.3f} {r['cliffs_delta']:+9.3f}  {sig}")

    # HEAD-TO-HEAD on the hard case: MF neg vs CLAP, separating diff from same
    print("\n================ MF vs CLAP on the HARD case (diff vs same) ================")
    mf_gap  = _arr(rows, "diff_neg")  - _arr(rows, "same_neg")     # want >0
    clap_gap = _arr(rows, "same_clap") - _arr(rows, "diff_clap")   # want >0 (sim drops)
    print(f"MF neg  separates diff>same : mean {np.mean(mf_gap):+.4f}, "
          f"wilcoxon p={wilcoxon(_arr(rows,'diff_neg'), _arr(rows,'same_neg')).pvalue:.2e}")
    print(f"CLAP    separates same>diff : mean {np.mean(clap_gap):+.4f}, "
          f"wilcoxon p={wilcoxon(_arr(rows,'same_clap'), _arr(rows,'diff_clap')).pvalue:.2e}")
    # per-triple: how often does each metric get the pair right?
    mf_acc   = np.mean(_arr(rows,'diff_neg')  > _arr(rows,'same_neg'))
    clap_acc = np.mean(_arr(rows,'diff_clap') < _arr(rows,'same_clap'))
    print(f"\nper-triple accuracy (diff correctly scored less coherent than same):")
    print(f"  MF neg : {mf_acc:.1%}")
    print(f"  CLAP   : {clap_acc:.1%}   (chance = 50%)")
    print("\nThe headline claim: MF accuracy and effect size on the HARD case vs CLAP's.")


def save_csv(rows, path="coherence_results.csv"):
    keys = sorted(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"saved {len(rows)} rows -> {path}")
