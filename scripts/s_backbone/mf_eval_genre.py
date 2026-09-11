"""
Eval for genre-split triples (a, cont, diff_same, diff_cross, noise).
Reuses mf_eval's scoring; just adds the two diff levels.

    import mf_eval as ev, mf_eval_genre as evg
    ev.set_backend(mf); ev.load_clap("laion/larger_clap_music")
    evg.run_batch(triples)     # triples from build_triples_genre
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

import numpy as np

def run_batch(triples, gap_s=0.0):
    import mf_eval as ev
    levels=["same","cont","diff_same","diff_cross","noise"]
    agg={lv:{"pos":[],"neg":[],"comb":[],"clap":[]} for lv in levels}
    for n,t in enumerate(triples):
        a=t["a"]
        pairs={"same":a,"cont":t["cont"],"diff_same":t["diff_same"],
               "diff_cross":t["diff_cross"],"noise":t["noise"]}
        for lv,b in pairs.items():
            comb,pos,neg=ev.mf_balanced(a,b,gap_s=gap_s)
            agg[lv]["pos"].append(pos);agg[lv]["neg"].append(neg);agg[lv]["comb"].append(comb)
            if ev._clap is not None: agg[lv]["clap"].append(ev.clap_sim(a,b))
        if (n+1)%20==0: print(f"  {n+1}/{len(triples)}")
    def m(x): return (np.mean(x),np.std(x)) if x else (float('nan'),0)
    print(f"\n===== GENRE-SPLIT RESULTS (n={len(triples)}) =====")
    print(f"{'level':11s}{'MF pos':>13s}{'MF neg':>13s}{'MF comb':>13s}{'CLAP':>13s}")
    for lv in levels:
        pm,ps=m(agg[lv]['pos']);nm,ns=m(agg[lv]['neg']);cm,cs=m(agg[lv]['comb']);km,ks=m(agg[lv]['clap'])
        print(f"{lv:11s}{pm:6.3f}±{ps:4.2f}{nm:7.3f}±{ns:4.2f}{cm:7.3f}±{cs:4.2f}{km:7.3f}±{ks:4.2f}")
    print("\n---- the key contrast: same-genre (HARD) vs cross-genre ----")
    print(f"MF neg  diff_same - same  = {np.mean(agg['diff_same']['neg'])-np.mean(agg['same']['neg']):+.3f}")
    print(f"MF neg  diff_cross- same  = {np.mean(agg['diff_cross']['neg'])-np.mean(agg['same']['neg']):+.3f}")
    print(f"CLAP    same - diff_same  = {np.mean(agg['same']['clap'])-np.mean(agg['diff_same']['clap']):+.3f}")
    print(f"CLAP    same - diff_cross = {np.mean(agg['same']['clap'])-np.mean(agg['diff_cross']['clap']):+.3f}")
    print("\nIf MF separates diff_same (same genre, different song) and CLAP does not,")
    print("that is the cleanest possible statement of the blind spot.")
    return agg
