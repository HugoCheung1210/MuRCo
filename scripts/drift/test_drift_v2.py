
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable


import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
import mf_probe as mf
import os
import drift_v4 as d4

os.environ["HF_HOME"] = "/root/autodl-tmp/hf"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
#  export HF_HOME=/root/autodl-tmp/hf

mf.load()  

d4.set_backend(mf); d4.load_clap("laion/clap-htsat-unfused")

def iter1_iter8(res):
    p = res["per_k"]
    sids = [s for s in p[1] if s in p[8]
            and not np.isnan(p[1][s]) and not np.isnan(p[8][s])]
    a = np.array([p[1][s] for s in sids])
    b = np.array([p[8][s] for s in sids])
    return a, b, sids

# # ==================== SAO =========================

sao_mf = d4.run("autodl-tmp/outputs_stableaudio_inpaint_5_8/editing", "SAO-MF-5-8", region=(4,6), metric="mf",
            ceiling="roundtrip", rt_dir="autodl-tmp/outputs_stableaudio_inpaint_5_8/roundtrip")

sao_cl = d4.run("autodl-tmp/outputs_stableaudio_inpaint_5_8/editing", "SAO-CLAP-5-8", region=(4,6), metric="clap",
            ceiling="roundtrip", rt_dir="autodl-tmp/outputs_stableaudio_inpaint_5_8/roundtrip")
d4.paired_stats(sao_mf, sao_cl, k=8)

sids = [s for s in sao_mf["per_k"][1] if s in sao_mf["per_k"][8]
        and not np.isnan(sao_mf["per_k"][1][s]) and not np.isnan(sao_mf["per_k"][8][s])]
a = np.array([sao_mf["per_k"][1][s] for s in sids])
b = np.array([sao_mf["per_k"][8][s] for s in sids])
print(f"iter1={a.mean():.3f} iter8={b.mean():.3f} drop={a.mean()-b.mean():+.3f} "
      f"n={len(sids)}  W,p={wilcoxon(a,b)}")

# # ===================================================

# # ==================== ACE =========================
ace_mf  = d4.run(f"outputs/E2_iterative", "ACE-MF",  region=(4,6), metric="mf",
             ceiling="roundtrip", rt_dir=f"outputs/C1_roundtrip")
ace_cl  = d4.run(f"outputs/E2_iterative", "ACE-CLAP", region=(4,6), metric="clap",
             ceiling="roundtrip", rt_dir=f"outputs/C1_roundtrip")
d4.paired_stats(ace_mf, ace_cl, k=8)
# and the accumulation test:
import numpy as np; from scipy.stats import wilcoxon
sids=[s for s in ace_mf["per_k"][1] if s in ace_mf["per_k"][8]
      and not np.isnan(ace_mf["per_k"][1][s]) and not np.isnan(ace_mf["per_k"][8][s])]
a=np.array([ace_mf["per_k"][1][s] for s in sids]); b=np.array([ace_mf["per_k"][8][s] for s in sids])
print(f"iter1={a.mean():.3f} iter8={b.mean():.3f} drop={a.mean()-b.mean():+.3f} W,p={wilcoxon(a,b)}")
# # ===================================================

# # ==================== ACE vs SAO =========================

# from scipy.stats import mannwhitneyu
# def deltas(res):
#     p=res["per_k"]; s=[x for x in p[1] if x in p[8] and not np.isnan(p[1][x]) and not np.isnan(p[8][x])]
#     return np.array([p[1][x]-p[8][x] for x in s])
# U,pv = mannwhitneyu(deltas(sao_mf), deltas(ace_mf), alternative="greater")
# print(f"SAO vs ACE: U={U} p={pv:.4f}")

# from scipy.stats import mannwhitneyu
# import numpy as np
# def deltas(res):
#     p=res["per_k"]; s=[x for x in p[1] if x in p[8] and not np.isnan(p[1][x]) and not np.isnan(p[8][x])]
#     return np.array([p[1][x]-p[8][x] for x in s])

# for tag, sao_r, ace_r in [("MF", sao_mf, ace_mf),
#                           ("CLAP-music", sao_cl, ace_cl)]:   # + htsat versions if in memory
#     U,pv = mannwhitneyu(deltas(sao_r), deltas(ace_r), alternative="greater")
#     print(f"{tag}: SAO Δ={deltas(sao_r).mean():+.3f} ACE Δ={deltas(ace_r).mean():+.3f} U={U} p={pv:.4f}")
    
# # ==================== ACE vs SAO clap-htsat-unfused =========================

from scipy.stats import mannwhitneyu
import numpy as np
def deltas(res):
    p=res["per_k"]; s=[x for x in p[1] if x in p[8] and not np.isnan(p[1][x]) and not np.isnan(p[8][x])]
    return np.array([p[1][x]-p[8][x] for x in s])

# both must be the HTSAT CLAP runs, in memory
U, pv = mannwhitneyu(deltas(sao_cl), deltas(ace_cl), alternative="greater")
print(f"CLAP-htsat SAO vs ACE: SAO Δ={deltas(sao_cl).mean():+.3f} "
      f"ACE Δ={deltas(ace_cl).mean():+.3f} U={U} p={pv:.4f}")

    
# # ===================================================


# # ==================== MusGen =========================

mg_mf = d4.run("autodl-tmp/outputs_musicgen/continuation", "MG-MF", region=(0,10), metric="mf", ceiling="self")
mg_cl = d4.run("autodl-tmp/outputs_musicgen/continuation", "MG-CLAP", region=(0,10), metric="clap", ceiling="self")

a, b, _ = iter1_iter8(mg_mf)          # mg_mf = your MG-MF run result
W, pval = wilcoxon(a, b)
print(f"MusicGen iter1={a.mean():.3f} iter8={b.mean():.3f} "
      f"drop={a.mean()-b.mean():+.3f} n={len(a)} W={W} p={pval:.4f}")


