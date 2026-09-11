"""Leave-one-dimension-out diagnosis, on the same folds Table 1 is built from.

diagnose_nested_cv.py reports HTRS and HTR, so the only ablation on record is the one
that deletes S. This adds the other three, which is what answers "does each dimension
earn its place" rather than just "does S". Same `loto` protocol and the same paired
bootstrap over tracks, so every row is comparable with the published 0.860/0.842.

The H/T/R/S columns are assembled here rather than via review_fixes.build_feature_sets,
which also builds the CLAP/MuQ/MERT features and costs tens of CPU-minutes we do not
need. The row filter is copied from it exactly (pairs must appear in the CLAP npz), so
the rows are identical; HTRS reproducing 0.860/0.842 is the check that they are.

Usage (DSP env, from repo root):
    python scripts/analysis/diagnose_loo.py
Writes results/diagnostics/diagnosis_loo.json. Does not touch diagnosis_nested.json.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import review_fixes as rf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

C_GRID = [0.02, 0.3, 1.0, 10.0]


def loto(X, y, g):
    """Copied verbatim from diagnose_nested_cv.py. Importing it there is not an option:
    that module runs its full analysis at import time, including the embedding features."""
    pred = np.empty_like(y)
    for tr in np.unique(g):
        te = g == tr
        Xtr, ytr, gtr = X[~te], y[~te], g[~te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Z, Zt = (Xtr - mu) / sd, (X[te] - mu) / sd
        best, bs = None, -np.inf
        for c in C_GRID:
            accs = []
            for i, j in GroupKFold(n_splits=3).split(Z, ytr, gtr):
                m = LogisticRegression(C=c, max_iter=300, tol=1e-3)
                m.fit(Z[i], ytr[i])
                accs.append((m.predict(Z[j]) == ytr[j]).mean())
            s = float(np.mean(accs))
            if s > bs:
                best, bs = c, s
        pred[te] = LogisticRegression(C=best, max_iter=5000).fit(Z, ytr).predict(Zt)
    return pred

rows = [r for r in rf.read_csv(rf.ROOT / "results/coherence/pair_scores_cbase.csv")
        if r["perturbation"] in rf.FAMILIES]
manifest = json.load(open(rf.ROOT / "perturbations/pairs_manifest.json"))
pairs = manifest["pairs"] if isinstance(manifest, dict) else manifest
paths = {p["pair_id"]: (p["ref_path"], p["cand_path"]) for p in pairs}
npz = np.load(rf.ROOT / "results/baselines/clap_embeddings.npz", allow_pickle=True)
idx = {c: i for i, c in enumerate(npz["clips"])}

keep = [r for r in rows if all(p in idx for p in paths[r["pair_id"]])]
y = np.array([rf.FAMILIES.index(r["perturbation"]) for r in keep])
g = np.array([r["source_id"] for r in keep])
X = np.column_stack([[float(r[f"score_{d}"]) for r in keep] for d in "HTRS"])
print(f"{len(y)} pairs, {len(np.unique(g))} tracks", flush=True)

cols = {"H": 0, "T": 1, "R": 2, "S": 3}
sets = {"HTRS": "none", "TRS": "H", "HRS": "T", "HTS": "R", "HTR": "S"}
out, preds = {}, {}
for name, dropped in sets.items():
    t = time.time()
    p = loto(X[:, [cols[c] for c in name]], y, g)
    preds[name] = p
    a, f = rf.acc_f1(y, p)
    out[name] = {"dim": len(name), "dropped": dropped, "acc": round(float(a), 4),
                 "macro_f1": round(float(f), 4)}
    print(name, out[name], "%.0fs" % (time.time() - t), flush=True)

out["contrasts"] = {}
for b in ["TRS", "HRS", "HTS", "HTR"]:
    out["contrasts"][f"HTRS - {b}"] = rf.paired_bootstrap(y, g, preds, "HTRS", b)
    print("HTRS -", b, out["contrasts"][f"HTRS - {b}"], flush=True)

out["meta"] = {"n_pairs": int(len(y)), "n_tracks": int(len(np.unique(g))),
               "protocol": "leave-one-track-out over tracks, L2 chosen by a grouped "
                           "inner split on the training fold only; paired bootstrap "
                           "over tracks. Identical to diagnose_nested_cv.py."}
json.dump(out, open(rf.ROOT / "results/diagnostics/diagnosis_loo.json", "w"), indent=2)
print("wrote results/diagnostics/diagnosis_loo.json")
