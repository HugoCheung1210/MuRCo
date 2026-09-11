"""Two analyses added in response to the 2026-08-28 external review, both computed from
data already on disk so the paper can cite them without a new run.

A. Per-family precision/recall/F1 and the confusion matrix for the HTRS profile and the
   HTR ablation (review point: "a confusion matrix or per-family F1s would help inspect
   failure modes", and "stronger quantitative isolation of S outside listening tests").
   Same nested LOTO protocol as diagnose_nested_cv.py, so the accuracy reproduces.

B. Whether R's beat-strength gate hides rhythmic incoherence (review point: gating
   "drops rhythm assessment on low-pulse material, potentially hiding rhythmic
   incoherence precisely where listeners might still perceive it"). Tests the gate
   against the listening study: if listeners also rate a time-stretch as more coherent
   on weak-beat material, the gate tracks perception rather than hiding a failure.

   READ THE CAVEATS in the output before quoting B: it is 44 rated pairs, the
   correlation is BETWEEN tracks, and beat strength is confounded with genre (ambient
   and classical are both weak-beat and both forgiving of a stretch). It bounds the
   objection; it does not prove the gate is perceptually calibrated.

Usage (DSP env, from repo root):  python scripts/analysis/review_additions.py
Writes results/diagnostics/review_additions.json.
"""
import csv, json, os, sys, collections
from pathlib import Path
import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parents[2])
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
sys.path.insert(0, str(ROOT / "scripts" / "core"))
import review_fixes as rf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

C_GRID = [0.02, 0.3, 1.0, 10.0]


def loto(X, y, g):
    """Leave-one-track-out with the L2 penalty tuned inside each training fold."""
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
                m = LogisticRegression(C=c, max_iter=300, tol=1e-3).fit(Z[i], ytr[i])
                accs.append((m.predict(Z[j]) == ytr[j]).mean())
            s = float(np.mean(accs))
            if s > bs:
                best, bs = c, s
        pred[te] = LogisticRegression(C=best, max_iter=5000).fit(Z, ytr).predict(Zt)
    return pred


def block_a():
    feats, y, g = rf.build_feature_sets()
    labels = list(range(len(rf.FAMILIES)))
    out = {"families": rf.FAMILIES, "protocol": "leave-one-track-out, 90 folds"}
    for name in ("HTRS", "HTR"):
        p = loto(feats[name], y, g)
        pr, rc, f1, sup = precision_recall_fscore_support(
            y, p, labels=labels, zero_division=0)
        out[name] = {
            "acc": round(float((p == y).mean()), 4),
            "macro_f1": round(float(f1.mean()), 4),
            "confusion_rows_true": confusion_matrix(y, p, labels=labels).tolist(),
            "per_family": {fam: {"precision": round(float(a), 3),
                                 "recall": round(float(b), 3),
                                 "f1": round(float(c), 3), "n": int(n)}
                           for fam, a, b, c, n in zip(rf.FAMILIES, pr, rc, f1, sup)},
        }
    out["f1_gain_from_S"] = {
        fam: round(out["HTRS"]["per_family"][fam]["f1"]
                   - out["HTR"]["per_family"][fam]["f1"], 3)
        for fam in rf.FAMILIES}
    return out


def block_b():
    import soundfile as sf
    from coherence_dimensions import RhythmicDimension
    from scipy.stats import pearsonr, spearmanr

    man = json.load(open(ROOT / "perturbations/pairs_manifest.json"))
    pairs = man["pairs"] if isinstance(man, dict) else man
    R = RhythmicDimension()
    sal = {}
    for p in pairs:                       # beat strength of A, from each control pair
        if p["perturbation"] != "control":
            continue
        path = p["ref_path"] if os.path.exists(p["ref_path"]) else f"perturbations/{p['ref_path']}"
        y, sr = sf.read(ROOT / path if not os.path.isabs(path) else path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(1)
        sal[p["pair_id"].split("::")[0]] = float(R._beat_strength(R._onset(y, sr), sr))

    rows = list(csv.DictReader(open(ROOT / "results/coherence/pair_scores.csv")))
    ctrlR = {r["source_id"]: float(r["score_R"]) for r in rows if r["perturbation"] == "control"}
    Rv = {r["pair_id"]: float(r["score_R"]) for r in rows}
    g = np.array([sal[s] for s in sorted(sal)])
    q = np.quantile(g, [1 / 3, 2 / 3])
    tert = lambda s: 0 if sal[s] < q[0] else (1 if sal[s] < q[1] else 2)

    drop = collections.defaultdict(list)
    for r in rows:
        if r["perturbation"] == "time_stretch":
            drop[tert(r["source_id"])].append(ctrlR[r["source_id"]] - float(r["score_R"]))

    mos = {r["pair_id"]: float(r["mos"])
           for r in csv.DictReader(open(ROOT / "results/mos/mos_live_detail.csv"))}
    ts = [(p, m) for p, m in mos.items()
          if p.split("::")[1] == "time_stretch" and p.split("::")[0] in sal]
    xs = [sal[p.split("::")[0]] for p, _ in ts]
    ys = [m for _, m in ts]
    rs = [Rv[p] for p, _ in ts if p in Rv]

    return {
        "note": "Does the salience gate hide rhythmic incoherence, or track perception?",
        "salience_over_90_sources": {
            "mean": round(float(g.mean()), 3), "median": round(float(np.median(g)), 3),
            "min": round(float(g.min()), 3), "max": round(float(g.max()), 3),
            "frac_below_0.3": round(float((g < 0.3).mean()), 3)},
        "R_drop_on_time_stretch_by_salience_tertile":
            {f"t{t}": round(float(np.mean(drop[t])), 3) for t in (0, 1, 2)},
        "mos_vs_salience_on_time_stretch": {
            "n_pairs": len(ts),
            "pearson_r": round(float(pearsonr(xs, ys)[0]), 3),
            "p": round(float(pearsonr(xs, ys)[1]), 4),
            "spearman": round(float(spearmanr(xs, ys)[0]), 3)},
        "R_vs_mos_on_time_stretch": {
            "n_pairs": len(rs),
            "pearson_r": round(float(pearsonr(rs, ys[:len(rs)])[0]), 3),
            "p": round(float(pearsonr(rs, ys[:len(rs)])[1]), 4)},
        "CAVEATS": ["44 rated pairs only",
                    "correlation is BETWEEN tracks, not within",
                    "salience confounded with genre (ambient/classical are low-salience "
                    "and also forgiving of a stretch)"],
    }


if __name__ == "__main__":
    # The "salience" spelling survives in these JSON KEYS on purpose. The prose calls
    # g "beat strength" now (2026-09-08, to match the paper), but renaming the keys
    # would desync this script from results/diagnostics/review_additions.json, which
    # is already written. These are internal diagnostics and never reach the paper.
    out = {"A_per_family": block_a(), "B_salience_gate": block_b()}
    dest = ROOT / "results/diagnostics/review_additions.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print("\nwrote", dest)
