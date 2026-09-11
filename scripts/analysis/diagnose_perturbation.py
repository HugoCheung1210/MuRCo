#!/usr/bin/env python3
"""E-A -- diagnose the PERTURBATION FAMILY from the (H,T,R,S) profile.

Turns the qualitative "separable signatures" claim (§3) into a measured
capability, on the axis where a decomposable relational metric structurally
beats a scalar: a 4-dim profile can say *what kind* of incoherence occurred; a
scalar (clap_htsat / scs / their combination C_base) can only say *how much*.
This sidesteps the identity-AUC loss (§8.1) entirely -- diagnosis is a different
task from acoustic identity, and it is one only a decomposable metric can do.

TASK.  Classify each pair's perturbation family from its score vector.  Families:
pitch_shift / time_stretch / lowpass / distortion / style_swap.  CONTROL is
EXCLUDED by default (control pairs are A vs A, so every dim == 1.0 exactly --
trivially separable, and including it only inflates accuracy; --include-control
adds it back and it is diagnosed near-perfectly, reported separately if asked).

FEATURE SETS (each classified independently, so their scales need not match):
  * HTRS  -- the full 4-dim profile (the proposed metric's decomposition)
  * HTR   -- profile without S (does S contribute to DIAGNOSIS? a third axis for
             "does S earn its place", distinct from identity-AUC and drift)
  * clap_htsat / scs / cbase -- each published scalar alone (the honest
    scalar-vs-vector comparison; C_base is the E-C naive combination)
Scope stated in advance (per plan §1 E-A): the comparison is between METRICS
(interpretable published scores), not representations -- a 768-d CLAP embedding
delta is not a metric, so it is out of scope by construction, not by omission.

EVALUATION.  Leave-one-TRACK-out CV: 90 folds grouped by source_id, so a track's
pairs are never split across train/test (per plan; prevents track-identity leak).
Two classifiers: nearest-centroid (most interpretable -- class = nearest mean
profile) and multinomial logistic (a check).  Features are z-scored per fold on
train stats so nearest-centroid is not dominated by the highest-variance dim.
Report accuracy + macro-F1 (macro so the small style_swap class is not drowned)
+ the full confusion matrix -- confusions are informative (e.g. style_swap moves
everything; what does it collide with?) and reported honestly, same spirit as
"signatures, not a clean diagonal".

Inputs (no GPU, no re-scoring): results/coherence/pair_scores_cbase.csv (built by
make_combined_baseline.py -- has score_H/T/R/S + score_scs/score_clap_htsat/
score_cbase).  Falls back to pair_scores.csv + baseline_scores.json if needed.
Output: results/diagnostics/diagnosis.json + a printed Ch-4 table.

Usage (DSP/local env, from repo root):
    python scripts/analysis/make_combined_baseline.py          # produces pair_scores_cbase.csv
    python scripts/analysis/diagnose_perturbation.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

FAMILIES = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]
FEATURE_SETS = {
    "HTRS": ["score_H", "score_T", "score_R", "score_S"],
    "HTR": ["score_H", "score_T", "score_R"],
    "clap_htsat": ["score_clap_htsat"],
    "scs": ["score_scs"],
    "cbase": ["score_cbase"],
}


def load_rows(results: Path) -> list[dict]:
    p = results / "pair_scores_cbase.csv"
    if p.is_file():
        return list(csv.DictReader(p.open(newline="", encoding="utf-8")))
    # fallback: join pair_scores.csv with baseline_scores.json on the fly
    rows = list(csv.DictReader((results / "pair_scores.csv").open(newline="", encoding="utf-8")))
    bl = next((c for c in (results / "baseline_scores.json",
                           results.parent / "baselines" / "baseline_scores.json")
               if c.is_file()), None)
    if bl is None:
        raise SystemExit("baseline_scores.json not found in %s or %s"
                         % (results, results.parent / "baselines"))
    doc = json.loads(bl.read_text(encoding="utf-8"))
    scores = doc.get("scores", doc)
    for r in rows:
        b = scores.get(r["pair_id"], {})
        r["score_clap_htsat"] = b.get("clap_htsat", "nan")
        r["score_scs"] = b.get("scs", "nan")
        r["score_cbase"] = "nan"  # cbase only exists once make_combined_baseline has run
    if any(r["score_cbase"] == "nan" for r in rows):
        FEATURE_SETS.pop("cbase", None)
        print("note: results/coherence/pair_scores_cbase.csv absent -> cbase feature set skipped; "
              "run make_combined_baseline.py to include it.")
    return rows


def nearest_centroid_predict(Xtr, ytr, Xte, classes):
    cents = np.stack([Xtr[ytr == c].mean(axis=0) for c in classes])
    d = ((Xte[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)  # (n_te, n_class)
    return classes[d.argmin(axis=1)]


def confusion(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y_true, y_pred):
        M[idx[t], idx[p]] += 1
    return M


def per_class_prf(M, classes):
    out = {}
    for i, c in enumerate(classes):
        tp = M[i, i]
        prec = tp / M[:, i].sum() if M[:, i].sum() else 0.0
        rec = tp / M[i, :].sum() if M[i, :].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                  "support": int(M[i, :].sum())}
    return out


def run_feature_set(rows, feats, classes, groups):
    """Leave-one-track-out CV for one feature set; returns pooled predictions."""
    X = np.array([[float(r[f]) for f in feats] for r in rows], float)
    y = np.array([r["perturbation"] for r in rows])
    g = np.array(groups)
    ok = ~np.isnan(X).any(axis=1)
    X, y, g = X[ok], y[ok], g[ok]

    preds_nc = np.empty(len(y), dtype=object)
    preds_lr = np.empty(len(y), dtype=object)
    for track in np.unique(g):
        tr, te = g != track, g == track
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        preds_nc[te] = nearest_centroid_predict(Xtr, y[tr], Xte, np.array(classes))
        lr = LogisticRegression(max_iter=2000, C=1.0)
        lr.fit(Xtr, y[tr])
        preds_lr[te] = lr.predict(Xte)

    res = {}
    for name, pred in (("nearest_centroid", preds_nc), ("logistic", preds_lr)):
        M = confusion(y, pred, classes)
        res[name] = {
            "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, labels=classes, average="macro", zero_division=0)),
            "confusion": {"rows_true_cols_pred": classes, "matrix": M.tolist()},
            "per_class": per_class_prf(M, classes),
        }
    res["n"] = int(len(y))
    return res


# dissertation display names + grouping for the LaTeX table (tab:diagnosis).
_TEX_DISPLAY = {"HTRS": r"$H,T,R,S$ (profile)", "HTR": r"$H,T,R$ (no $S$)",
                "clap_htsat": "CLAP-htsat", "scs": "SCS (stand-in)",
                "cbase": r"SCS$\times$CLAP"}
_TEX_PROFILE = ["HTRS", "HTR"]
_TEX_SCALAR = ["clap_htsat", "scs", "cbase"]


def write_diagnosis_tex(out: dict, path: Path) -> None:
    """Emit tab:diagnosis to \\input into the dissertation (matches doc/writeup/dissertation.tex)."""
    res = out["results"]
    order = [fs for fs in (_TEX_PROFILE + _TEX_SCALAR) if fs in res]
    cols = [("nearest_centroid", "accuracy"), ("nearest_centroid", "macro_f1"),
            ("logistic", "accuracy"), ("logistic", "macro_f1")]
    colmax = [max(res[fs][clf][metric] for fs in order) for clf, metric in cols]

    def cell(fs, ci):
        clf, metric = cols[ci]
        v = res[fs][clf][metric]
        s = f"{v:.3f}"
        return rf"\textbf{{{s}}}" if abs(v - colmax[ci]) < 1e-9 else s

    m = out["meta"]
    lines = [r"% Generated by: python scripts/analysis/diagnose_perturbation.py --tex " + str(path),
             r"\begin{table}[t]", r"\centering",
             (rf"\caption[Diagnosing which perturbation occurred]{{E-A: perturbation-family "
              rf"diagnosis accuracy and macro-F1 "
              rf"({len(m['classes'])} classes; leave-one-track-out CV; chance "
              rf"$={m['chance_accuracy']:.2f}$, majority-class $={m['majority_class_accuracy']:.2f}$). "
              rf"NC $=$ nearest-centroid, LR $=$ multinomial logistic.}}"),
             r"\label{tab:diagnosis}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"feature set & Acc (NC) & F1 (NC) & Acc (LR) & F1 (LR) \\", r"\midrule"]
    for fs in _TEX_PROFILE:
        if fs in res:
            lines.append(f"{_TEX_DISPLAY[fs]} & " + " & ".join(cell(fs, i) for i in range(4)) + r" \\")
    lines.append(r"\midrule")
    for fs in _TEX_SCALAR:
        if fs in res:
            lines.append(f"{_TEX_DISPLAY[fs]} & " + " & ".join(cell(fs, i) for i in range(4)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("results/coherence"))
    ap.add_argument("--out", type=Path, default=Path("results/diagnostics/diagnosis.json"))
    ap.add_argument("--tex", type=Path, nargs="?", const=Path("results/diagnostics/diagnosis.tex"),
                    default=None, help="also emit the LaTeX table tab:diagnosis "
                                       "(default results/diagnostics/diagnosis.tex) for \\input")
    ap.add_argument("--include-control", action="store_true",
                    help="add control as a 6th (trivially separable) class")
    ap.add_argument("--group", choices=["track", "genre"], default="track",
                    help="CV grouping: leave-one-TRACK-out (default, 90 folds) or "
                         "leave-one-GENRE-out (6 folds; tests cross-genre "
                         "generalisation of the profile)")
    args = ap.parse_args()

    rows = load_rows(args.results_dir)
    classes = list(FAMILIES)
    if args.include_control:
        classes = ["control"] + classes
    rows = [r for r in rows if r["perturbation"] in classes]
    group_key = "source_id" if args.group == "track" else "genre"
    groups = [r[group_key] for r in rows]

    chance = 1.0 / len(classes)
    from collections import Counter
    cnt = Counter(r["perturbation"] for r in rows)
    majority = max(cnt.values()) / len(rows)

    cv_desc = ("leave-one-track-out (90 folds, group=source_id)" if args.group == "track"
               else "leave-one-genre-out (6 folds, group=genre)")
    out = {"meta": {"task": "perturbation-family diagnosis from score profile",
                    "classes": classes, "cv": cv_desc,
                    "n_pairs": len(rows), "class_counts": dict(cnt),
                    "chance_accuracy": chance, "majority_class_accuracy": majority,
                    "control": "included" if args.include_control else
                               "excluded (A-vs-A -> all dims=1.0, trivially separable)"},
           "results": {}}

    for fs, feats in FEATURE_SETS.items():
        if any(f not in rows[0] for f in feats):
            continue
        out["results"][fs] = run_feature_set(rows, feats, classes, groups)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.tex:
        if args.include_control:
            print("note: --tex reflects --include-control run; dissertation table excludes control")
        args.tex.parent.mkdir(parents=True, exist_ok=True)
        write_diagnosis_tex(out, args.tex)

    # ---- printed Ch-4 table -------------------------------------------------
    print(f"# E-A perturbation-family diagnosis  (n={len(rows)}, {len(classes)} classes)")
    print(f"chance = {chance:.3f} | majority-class = {majority:.3f} | "
          f"control {'INCLUDED' if args.include_control else 'excluded'}\n")
    print("| feature set | classifier | accuracy | macro-F1 |")
    print("|---|---|---:|---:|")
    for fs in FEATURE_SETS:
        if fs not in out["results"]:
            continue
        for clf in ("nearest_centroid", "logistic"):
            r = out["results"][fs][clf]
            print(f"| {fs} | {clf} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} |")

    # nearest-centroid confusion for the full profile (the interpretable headline)
    if "HTRS" in out["results"]:
        M = out["results"]["HTRS"]["nearest_centroid"]["confusion"]["matrix"]
        print("\nHTRS nearest-centroid confusion (rows=true, cols=pred):")
        head = "true\\pred  " + " ".join(f"{c[:6]:>7s}" for c in classes)
        print(head)
        for c, row in zip(classes, M):
            print(f"{c[:9]:>9s}  " + " ".join(f"{v:7d}" for v in row))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
