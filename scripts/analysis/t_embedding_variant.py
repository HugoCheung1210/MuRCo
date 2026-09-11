#!/usr/bin/env python
"""Does T have to be an MFCC Frechet distance, or would a learned embedding do better?

Answers the reviewer point that T "is simple and efficient but may miss perceptually
relevant timbral nuances captured by modern music embeddings" (2026-08-30). The test
swaps T's read-out for a distance between the two segments' cached embeddings, leaving
H, R and S exactly as they are, and asks three things:

  1. Does the swapped T still respond most to the two families it owns (low-pass and
     distortion) rather than to pitch or tempo? That is T's job in the matrix, and a
     timbre measure that answers to a transpose is not a timbre measure.
  2. Does it separate each family from control any better, read as an AUC so the
     comparison is free of the exp(-d/tau) temperature entirely?
  3. Does the composite built on it discriminate identity any better?

No audio is rescored. Embeddings are the ones already cached by the CLAP and SSL probes, so
this is CPU-only and takes seconds once those caches exist. They are `.npz` tensors and are
deliberately NOT shipped in the public release, being derived payload rather than results, so
in a fresh checkout regenerate them first:

    <torch env>/bin/python scripts/analysis/clap_probe.py extract   # clap_embeddings.npz
    <rivals venv>/bin/python ssl_probe.py extract --encoder muq      # muq_L*.npz
    <rivals venv>/bin/python ssl_probe.py extract --encoder mert     # mert_L*.npz

Neither extract step runs in the DSP env: CLAP needs torch and transformers >= 4.27, and the
SSL extraction is a GPU job (minutes on a 4090, hours on CPU). Their own docstrings give the
environments. Without the caches this script exits with a FileNotFoundError on
results/baselines/*.npz, which is expected rather than a bug. Everything after extraction,
including this script, is CPU-only.

**On tau.** The squash exp(-d/tau) is monotone, so every AUC and every ranking below is
invariant to it and no calibration can flatter a variant on those. Magnitudes are not
invariant, so for the mean-drop columns tau is set per variant to match MFCC-T's mean
score on the control pairs. That puts every variant on the same scale at the top of the
range and makes the drops comparable rather than a function of an arbitrary constant.

Usage (DSP env, from the repo root):
    python scripts/analysis/t_embedding_variant.py
    python scripts/analysis/t_embedding_variant.py --out results/diagnostics/t_variants.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parents[2])

SAME = {"control", "pitch_shift", "time_stretch", "lowpass", "distortion"}
DIFF = {"style_swap"}
TIMBRAL_FAMILIES = ("lowpass", "distortion")
# (label, npz file, array key). CLAP is given both read-outs it ships with; MuQ and MERT
# are taken at the best layer the sweep found for each, so neither is handicapped.
VARIANTS = [
    ("clap_pooled",  "clap_embeddings.npz", "pooled"),
    ("clap_proj",    "clap_embeddings.npz", "projected"),
    ("muq_L1",       "muq_L1.npz",          "pooled"),
    ("mert_L3",      "mert_L3.npz",         "pooled"),
]


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC, higher score => label 1. Ties get average rank."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties so a degenerate scale cannot score above chance for free
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_embeddings(name: str, key: str) -> dict[str, np.ndarray]:
    """Keyed by basename: the MuQ/MERT caches were written on the GPU box and carry an
    `autodl-tmp/` path prefix the CLAP cache does not. Basenames are unique here."""
    d = np.load(ROOT / "results/baselines" / name, allow_pickle=True)
    clips = [Path(str(c)).name for c in d["clips"]]
    vecs = np.asarray(d[key], dtype=np.float64)
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    assert len(set(clips)) == len(clips), f"{name}: basenames are not unique"
    return dict(zip(clips, vecs))


def paired_dz(drops: np.ndarray) -> float:
    """Cohen's d_z on the within-track drop from control."""
    sd = drops.std(ddof=1)
    return float(drops.mean() / sd) if sd > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/coherence/pair_scores.csv")
    ap.add_argument("--manifest", default="perturbations/pairs_manifest.json")
    ap.add_argument("--out", default="results/diagnostics/t_variants.json")
    ap.add_argument("--skip-diagnosis", action="store_true",
                    help="skip the LOTO diagnosis, which is the slow part")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / a.scores)))
    man = json.load(open(ROOT / a.manifest))
    man = man if isinstance(man, list) else man["pairs"]
    paths = {m["pair_id"]: (m["ref_path"], m["cand_path"]) for m in man}

    fam = np.array([r["perturbation"] for r in rows])
    src = np.array([r["source_id"] for r in rows])
    dims = {d: np.array([float(r["score_" + d]) for r in rows]) for d in "HTRS"}

    # T variants: cosine distance between the two segments, squashed like MFCC-T.
    ctrl = fam == "control"
    t_mfcc = dims["T"]
    scores = {"mfcc (current)": t_mfcc}
    taus: dict[str, float] = {}
    dists: dict[str, np.ndarray] = {}
    for label, f, key in VARIANTS:
        emb = load_embeddings(f, key)
        d = np.array([1.0 - float(emb[Path(paths[r["pair_id"]][0]).name]
                                  @ emb[Path(paths[r["pair_id"]][1]).name])
                      for r in rows])
        # tau such that mean(exp(-d/tau)) over controls matches MFCC-T's control mean
        target = t_mfcc[ctrl].mean()
        lo, hi = 1e-6, 1e3
        for _ in range(200):                       # bisection; mean is monotone in tau
            mid = (lo + hi) / 2
            if np.exp(-d[ctrl] / mid).mean() < target:
                lo = mid
            else:
                hi = mid
        taus[label] = (lo + hi) / 2
        dists[label] = d
        scores[label] = np.exp(-d / taus[label])

    fams = ["style_swap", "lowpass", "distortion", "pitch_shift", "time_stretch"]
    ctrl_by_src = {s: t for s, t in zip(src[ctrl], np.arange(len(rows))[ctrl])}

    out: dict = {"note": __doc__.split("\n\n")[0], "tau_calibrated": taus, "variants": {}}
    print(f"{'T read-out':<16}" + "".join(f"{f[:9]:>11}" for f in fams)
          + f"{'select.':>9}   owns timbral?")
    print("-" * 96)
    for label, s in scores.items():
        rec: dict = {"per_family": {}}
        drops = {}
        for f in fams:
            idx = np.where(fam == f)[0]
            dd = np.array([s[ctrl_by_src[src[i]]] - s[i] for i in idx])
            drops[f] = dd.mean()
            y = np.concatenate([np.ones(ctrl.sum()), np.zeros(len(idx))])
            sc = np.concatenate([s[ctrl], s[idx]])
            rec["per_family"][f] = {"mean_drop": round(float(dd.mean()), 4),
                                    "dz": round(paired_dz(dd), 3),
                                    "auc_vs_control": round(auc(sc, y), 4)}
        ranked = sorted(fams, key=lambda f: -drops[f])
        # T's job: among the four same-song families, the two timbral ones should lead
        same_song = [f for f in ranked if f != "style_swap"]
        owns = same_song[0] in TIMBRAL_FAMILIES and same_song[1] in TIMBRAL_FAMILIES
        # Selectivity: a bigger drop everywhere is only sensitivity. What makes a
        # dimension diagnostic is responding to its own families more than the others.
        timb = np.mean([drops[f] for f in TIMBRAL_FAMILIES])
        other = np.mean([drops[f] for f in ("pitch_shift", "time_stretch")])
        rec["ranking_by_drop"] = ranked
        rec["owns_timbral_families"] = bool(owns)
        rec["selectivity"] = round(float(timb / other), 3)
        out["variants"][label] = rec
        print(f"{label:<16}" + "".join(f"{drops[f]:>11.3f}" for f in fams)
              + f"{timb / other:>9.2f}   {'YES' if owns else 'no '} ({same_song[0]})")

    # ---- identity discrimination, with T swapped inside the composite ---------
    print()
    print(f"{'T read-out':<16}{'AUC T alone':>13}{'AUC C_signal':>14}{'AUC C_full':>12}")
    print("-" * 55)
    y = np.array([1 if f in SAME else 0 for f in fam])
    gm = lambda *v: np.exp(np.mean([np.log(np.maximum(x, 1e-9)) for x in v], axis=0))
    for label, s in scores.items():
        a_t = auc(s, y)
        a_sig = auc(gm(dims["H"], s, dims["R"]), y)
        a_full = auc(gm(dims["H"], s, dims["R"], dims["S"]), y)
        out["variants"][label]["identity_auc"] = {
            "T_alone": round(a_t, 4), "C_signal": round(a_sig, 4), "C_full": round(a_full, 4)}
        print(f"{label:<16}{a_t:>13.4f}{a_sig:>14.4f}{a_full:>12.4f}")

    # ---- the decisive test: does the edit-family diagnosis survive the swap? ----
    # Reuses review_fixes' leave-one-track-out machinery and its nested C selection, so
    # these land on the same scale as the published HTRS 0.860 / 0.842.
    if not a.skip_diagnosis:
        import sys
        sys.path.insert(0, str(ROOT / "scripts/analysis"))
        import review_fixes as rf
        # Use diagnose_nested_cv's `loto`, NOT review_fixes' `loto_predict`. The two
        # differ (C grid of four vs five, inner max_iter 300 vs 1500) and disagree by
        # ~0.005 on identical inputs. Table 1 of the paper is built by the former, so
        # the baseline here has to reproduce its 0.860 / 0.8416 or the comparison would
        # be against a number the thesis never reports.
        # Inlined rather than imported: diagnose_nested_cv runs its whole LOTO sweep at
        # module level, so importing it would re-run block A (>10 min) as a side effect.
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold

        def loto(X, y, g, grid=(0.02, 0.3, 1.0, 10.0)):
            pred = np.empty_like(y)
            for tr in np.unique(g):
                te = g == tr
                Xtr, ytr, gtr = X[~te], y[~te], g[~te]
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                Z, Zt = (Xtr - mu) / sd, (X[te] - mu) / sd
                best, bs = None, -np.inf
                for c in grid:
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

        print()
        print(f"{'T read-out':<16}{'diag. acc':>11}{'macro F1':>10}   (LOTO, nested C, "
              f"diagnose_nested_cv; published HTRS = 0.860 / 0.842)")
        print("-" * 92)
        keep_idx = np.array([i for i, f in enumerate(fam) if f in rf.FAMILIES])
        yd = np.array([rf.FAMILIES.index(fam[i]) for i in keep_idx])
        gd = src[keep_idx]
        preds = {}
        for label, s in scores.items():
            X = np.column_stack([dims["H"][keep_idx], s[keep_idx],
                                 dims["R"][keep_idx], dims["S"][keep_idx]])
            pred = loto(X, yd, gd)
            preds[label] = pred
            acc, f1 = rf.acc_f1(yd, pred)
            out["variants"][label]["diagnosis_HTRS"] = {"acc": round(acc, 4),
                                                        "macro_f1": round(f1, 4)}
            print(f"{label:<16}{acc:>11.4f}{f1:>10.4f}")
        print()
        print("paired bootstrap over tracks, variant minus the MFCC read-out:")
        for label in scores:
            if label == "mfcc (current)":
                continue
            b = rf.paired_bootstrap(yd, gd, preds, label, "mfcc (current)")
            out["variants"][label]["vs_mfcc"] = b
            print(f"  {label:<14} acc {b['delta_acc']:+.4f} {b['acc_ci95']}   "
                  f"macroF1 {b['delta_macro_f1']:+.4f} {b['f1_ci95']}")

    dest = ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\n[wrote {dest}]")


if __name__ == "__main__":
    main()
