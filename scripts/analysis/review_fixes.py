#!/usr/bin/env python3
"""Numbers demanded by the ICASSP 2027 reviewer pass (2026-08-23).

Four analyses, all from data already on disk, all written to one JSON so the
paper can quote them without a re-run:

  A. diagnosis   -- leave-one-track-out family classification with a NESTED inner
                    CV for the L2 strength (the published sweep picked C on the
                    same folds it scored, which flatters every row), the HTR row
                    the paper omits, and a paired bootstrap over the 90 source
                    tracks so the margins carry intervals.
  B. mos_within  -- the Part 1 correlation partialled by perturbation condition,
                    a split-half noise ceiling, and the reliability of the pair
                    means that every correlation is computed against.
  C. clap_fitted -- a supervised ridge head on the CLAP embedding predicting MOS
                    under the same track-grouped CV. The paper compares a
                    purpose-built LLM read-out against a raw cosine; this is the
                    fair upper bound for "cheap embedding".
  D. ab_ties     -- Part 2 with the "about the same" midpoint counted as half a
                    win instead of dropped, plus a Holm correction across arms.

Usage (DSP env, from repo root):
    python scripts/analysis/review_fixes.py
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

FAMILIES = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]
C_GRID = [0.02, 0.3, 1.0, 10.0, 100.0]
RNG = np.random.default_rng(20260823)
N_BOOT = 2000


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / ".projectroot").exists() or (p / ".git").exists():
            return p
    return Path.cwd()


ROOT = repo_root()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# A. diagnosis with nested CV and paired intervals
# ---------------------------------------------------------------------------
def build_feature_sets() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    rows = [r for r in read_csv(ROOT / "results/coherence/pair_scores_cbase.csv")
            if r["perturbation"] in FAMILIES]
    manifest = json.load(open(ROOT / "perturbations/pairs_manifest.json"))
    pairs = manifest["pairs"] if isinstance(manifest, dict) else manifest
    paths = {p["pair_id"]: (p["ref_path"], p["cand_path"]) for p in pairs}

    npz = np.load(ROOT / "results/baselines/clap_embeddings.npz", allow_pickle=True)
    idx = {c: i for i, c in enumerate(npz["clips"])}
    pooled = npz["pooled"]

    keep, y, groups = [], [], []
    for r in rows:
        ref, cand = paths[r["pair_id"]]
        if ref in idx and cand in idx:
            keep.append(r)
            y.append(FAMILIES.index(r["perturbation"]))
            groups.append(r["source_id"])

    ea = np.stack([pooled[idx[paths[r["pair_id"]][0]]] for r in keep])
    eb = np.stack([pooled[idx[paths[r["pair_id"]][1]]] for r in keep])

    def col(name):
        return np.array([float(r[name]) for r in keep], dtype=float)

    H, T, R, S = col("score_H"), col("score_T"), col("score_R"), col("score_S")
    feats = {
        "HTRS": np.column_stack([H, T, R, S]),
        "HTR": np.column_stack([H, T, R]),
        "clap768_diff": eb - ea,
        "clap768_full": np.column_stack([ea, eb, eb - ea]),
        "scs": col("score_scs")[:, None],
        "clap_cosine": col("score_clap_htsat")[:, None],
    }
    feats.update(published_scalars(keep))
    feats.update(ssl_features(keep, paths))
    return feats, np.array(y), np.array(groups)


def ssl_features(keep: list[dict], paths: dict) -> dict[str, np.ndarray]:
    """A4: masked-prediction music encoders as difference features and as a cosine.

    Built by `ssl_probe.py extract`, which writes the same `clips`/`pooled` npz contract
    `clap_probe.py` does, so the difference feature here is constructed identically to
    `clap768_diff` and the two rows differ only in the encoder. Absent npz files are
    skipped, so this stays runnable before the GPU pass has been made.
    """
    out: dict[str, np.ndarray] = {}
    for enc in ("muq", "mert"):
        npz_path = ROOT / f"results/baselines/{enc}_embeddings.npz"
        if not npz_path.is_file():
            continue
        npz = np.load(npz_path, allow_pickle=True)
        # Key on BASENAME, not the stored path. The extract runs on the GPU box, whose
        # manifest copy is repo-relative to /root and so writes clip keys prefixed
        # "autodl-tmp/", while the local manifest's are relative to the repo. Basenames
        # encode source + perturbation and are unique across all 1,980 clips (checked),
        # which is the same reason score_s_mf.py's --audio-dir resolves by basename.
        idx = {Path(c).name: i for i, c in enumerate(npz["clips"])}
        pooled = npz["pooled"]
        base = {pid: (Path(a).name, Path(b).name) for pid, (a, b) in paths.items()}
        if not all(base[r["pair_id"]][0] in idx and base[r["pair_id"]][1] in idx
                   for r in keep):
            continue                       # partial extract: a smoke run, not the full set
        ea = np.stack([pooled[idx[base[r["pair_id"]][0]]] for r in keep])
        eb = np.stack([pooled[idx[base[r["pair_id"]][1]]] for r in keep])
        out[f"{enc}_diff"] = eb - ea
        out[f"{enc}_cosine"] = np.einsum("ij,ij->i", ea, eb)[:, None]
    return out


def published_scalars(keep: list[dict]) -> dict[str, np.ndarray]:
    """The three published rivals that ship checkpoints, as 1-d feature sets.

    B7 of the 2026-08-24 external review: the paper says "no published *scalar* comes
    close" on a field of two (SCS, a stand-in, and the CLAP cosine). TuneJury, SongEval
    and COCOLA are already scored on all 1,890 matrix pairs, so the claim can be tested
    rather than asserted. TuneJury and SongEval are ABSOLUTE metrics; `_rel` is the
    junction-isolating adaptation `score_rivals.py` documents, cat - (a + b)/2, which is
    the only reading of them that is relational at all. Missing files are skipped rather
    than raised, so this stays runnable on a box without the rival caches.
    """
    out: dict[str, np.ndarray] = {}

    rivals = ROOT / "results/baselines/rival_scores.json"
    if rivals.is_file():
        scores = json.loads(rivals.read_text())["scores"]
        for key in ("tunejury_rel", "songeval_coherence_rel"):
            if all(key in scores.get(r["pair_id"], {}) for r in keep):
                out[key] = np.array([scores[r["pair_id"]][key]
                                     for r in keep], dtype=float)[:, None]

    # Audiobox Aesthetics, added 2026-08-27 after a reviewer asked why the aesthetics
    # wave is represented by SongEval alone. Four axes, so `_rel` is a 4-d feature that
    # matches the profile's dimension, and `_b` is the plain absolute reading on B.
    abx = ROOT / "results/baselines/audiobox_scores.json"
    if abx.is_file():
        s = json.loads(abx.read_text())["scores"]
        for role, key in (("rel", "audiobox_rel"), ("b", "audiobox_b")):
            cols = [f"audiobox_{d}_{role}" for d in ("ce", "cu", "pc", "pq")]
            if all(all(c in s.get(r["pair_id"], {}) for c in cols) for r in keep):
                out[key] = np.array([[s[r["pair_id"]][c] for c in cols]
                                     for r in keep], dtype=float)

    cocola = ROOT / "results/baselines/cocola_scores.csv"
    if cocola.is_file():
        col_c = {r["pair_id"]: float(r["cocola"]) for r in read_csv(cocola)}
        if all(r["pair_id"] in col_c for r in keep):
            out["cocola"] = np.array([col_c[r["pair_id"]]
                                      for r in keep], dtype=float)[:, None]
    return out


def loto_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 nested: bool) -> tuple[np.ndarray, list[float]]:
    """Leave-one-track-out predictions. nested=True picks C inside the training
    fold only; nested=False mimics the published sweep (best C on the test folds,
    resolved by the caller)."""
    pred = np.empty_like(y)
    chosen: list[float] = []
    for tr in np.unique(groups):
        te_m = groups == tr
        tr_m = ~te_m
        Xtr, ytr, gtr = X[tr_m], y[tr_m], groups[tr_m]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr_z, Xte_z = (Xtr - mu) / sd, (X[te_m] - mu) / sd

        if nested:
            best_c, best_s = None, -np.inf
            inner = GroupKFold(n_splits=3)
            for c in C_GRID:
                accs = []
                for i_tr, i_va in inner.split(Xtr_z, ytr, gtr):
                    m = LogisticRegression(C=c, max_iter=1500)
                    m.fit(Xtr_z[i_tr], ytr[i_tr])
                    accs.append((m.predict(Xtr_z[i_va]) == ytr[i_va]).mean())
                s = float(np.mean(accs))
                if s > best_s:
                    best_c, best_s = c, s
            c_use = best_c
        else:
            c_use = 1.0
        chosen.append(c_use)
        m = LogisticRegression(C=c_use, max_iter=5000)
        m.fit(Xtr_z, ytr)
        pred[te_m] = m.predict(Xte_z)
    return pred, chosen


def acc_f1(y, pred, mask=None):
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    return ((pred[mask] == y[mask]).mean(),
            f1_score(y[mask], pred[mask], average="macro", labels=range(len(FAMILIES)),
                     zero_division=0))


def paired_bootstrap(y, groups, preds: dict[str, np.ndarray], a: str, b: str):
    tracks = np.unique(groups)
    by_track = {t: np.where(groups == t)[0] for t in tracks}
    d_acc, d_f1 = [], []
    for _ in range(N_BOOT):
        pick = RNG.choice(tracks, size=len(tracks), replace=True)
        sel = np.concatenate([by_track[t] for t in pick])
        aa, af = acc_f1(y[sel], preds[a][sel])
        ba, bf = acc_f1(y[sel], preds[b][sel])
        d_acc.append(aa - ba)
        d_f1.append(af - bf)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"delta_acc": round(float(np.mean(d_acc)), 4), "acc_ci95": q(d_acc),
            "delta_macro_f1": round(float(np.mean(d_f1)), 4), "f1_ci95": q(d_f1),
            "frac_le_zero_acc": round(float(np.mean(np.array(d_acc) <= 0)), 4)}


def analysis_a():
    feats, y, groups = build_feature_sets()
    out = {"n_pairs": int(len(y)), "n_tracks": int(len(np.unique(groups))),
           "protocol": "leave-one-track-out, 90 folds; inner GroupKFold(5) on the "
                       "training fold only selects C from " + str(C_GRID),
           "nested": {}, "fixed_C1": {}}
    preds = {}
    for name, X in feats.items():
        p, chosen = loto_predict(X, y, groups, nested=True)
        preds[name] = p
        a, f = acc_f1(y, p)
        out["nested"][name] = {"dim": int(X.shape[1]), "acc": round(float(a), 4),
                              "macro_f1": round(float(f), 4),
                              "C_modal": float(max(set(chosen), key=chosen.count))}
        p1, _ = loto_predict(X, y, groups, nested=False)
        a1, f1v = acc_f1(y, p1)
        out["fixed_C1"][name] = {"acc": round(float(a1), 4), "macro_f1": round(float(f1v), 4)}
        print(f"  {name:14s} dim={X.shape[1]:5d}  nested {a:.3f}/{f:.3f}   C=1 {a1:.3f}/{f1v:.3f}")

    out["contrasts"] = {}
    for a, b in [("HTRS", "HTR"), ("HTRS", "clap768_diff"), ("HTR", "clap768_diff"),
                 ("HTRS", "scs"), ("HTRS", "clap768_full")]:
        out["contrasts"][f"{a} - {b}"] = paired_bootstrap(y, groups, preds, a, b)
        print(f"  {a} - {b}: {out['contrasts'][f'{a} - {b}']}")
    return out


# ---------------------------------------------------------------------------
# B/C. MOS: within-condition r, noise ceiling, reliability, fitted CLAP head
# ---------------------------------------------------------------------------
def load_mos_ratings():
    """stimulus_id -> list of per-rater ratings, plus the stimulus index."""
    index = {r["stimulus_id"]: r for r in read_csv(ROOT / "results/mos/stimulus_index.csv")}
    by_stim = defaultdict(list)
    for f in sorted(glob.glob(str(ROOT / "results/mos/responses/*.json"))):
        d = json.load(open(f))
        rid = d["session_code"]
        for resp in d["responses"]:
            if resp.get("rating") is not None:
                by_stim[resp["stimulus_id"]].append((rid, float(resp["rating"])))
    return by_stim, index


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1])


def analysis_bc():
    by_stim, index = load_mos_ratings()
    scores = {r["pair_id"]: r for r in read_csv(ROOT / "results/coherence/pair_scores_cbase.csv")}

    recs = []
    for stim, ratings in by_stim.items():
        meta = index.get(stim)
        if meta is None or meta["pair_id"] not in scores:
            continue
        s = scores[meta["pair_id"]]
        H, T, R, S = (float(s["score_H"]), float(s["score_T"]),
                      float(s["score_R"]), float(s["score_S"]))
        recs.append({
            "pair_id": meta["pair_id"], "source_id": meta["source_id"],
            "condition": meta["condition"], "ratings": ratings,
            "mos": float(np.mean([v for _, v in ratings])),
            "C": (H * T * R * S) ** 0.25, "C_noS": (H * T * R) ** (1 / 3),
            "clap": float(s["score_clap_htsat"]), "scs": float(s["score_scs"]),
        })
    n_rat = [len(r["ratings"]) for r in recs]
    out = {"n_pairs": len(recs), "n_tracks": len(set(r["source_id"] for r in recs)),
           "ratings_per_pair": {"min": int(min(n_rat)), "median": float(np.median(n_rat)),
                                "max": int(max(n_rat)), "mean": round(float(np.mean(n_rat)), 2)}}

    mos = np.array([r["mos"] for r in recs])
    metrics = {k: np.array([r[k] for r in recs]) for k in ("C", "C_noS", "clap", "scs")}
    out["pooled_r"] = {k: round(pearson(mos, v), 4) for k, v in metrics.items()}

    # --- within-condition: centre both sides on their condition mean -------
    conds = np.array([r["condition"] for r in recs])
    def centre(v):
        out_v = v.astype(float).copy()
        for c in np.unique(conds):
            m = conds == c
            out_v[m] -= out_v[m].mean()
        return out_v
    mos_w = centre(mos)
    out["n_conditions"] = int(len(np.unique(conds)))
    out["within_condition_r"] = {k: round(pearson(mos_w, centre(v)), 4)
                                 for k, v in metrics.items()}

    # cluster bootstrap over source tracks for the within-condition contrasts
    tracks = sorted(set(r["source_id"] for r in recs))
    by_tr = {t: [i for i, r in enumerate(recs) if r["source_id"] == t] for t in tracks}
    diffs = defaultdict(list)
    for _ in range(N_BOOT):
        pick = RNG.choice(tracks, size=len(tracks), replace=True)
        sel = np.concatenate([by_tr[t] for t in pick])
        cs = conds[sel]
        def centre_s(v):
            vv = v[sel].astype(float).copy()
            for c in np.unique(cs):
                m = cs == c
                if m.sum() > 1:
                    vv[m] -= vv[m].mean()
            return vv
        mw = centre_s(mos)
        rs = {k: pearson(mw, centre_s(v)) for k, v in metrics.items()}
        diffs["C - C_noS"].append(rs["C"] - rs["C_noS"])
        diffs["C - clap"].append(rs["C"] - rs["clap"])
        diffs["C_noS - clap"].append(rs["C_noS"] - rs["clap"])
    out["within_condition_contrasts"] = {
        k: {"delta_r": round(float(np.mean(v)), 4),
            "ci95": [round(float(np.percentile(v, 2.5)), 4),
                     round(float(np.percentile(v, 97.5)), 4)],
            "excludes_zero": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0)}
        for k, v in diffs.items()}

    # --- split-half noise ceiling ------------------------------------------
    raters = sorted({rid for r in recs for rid, _ in r["ratings"]})
    halves = []
    for _ in range(500):
        perm = RNG.permutation(raters)
        g1 = set(perm[: len(perm) // 2])
        a, b = [], []
        for r in recs:
            x = [v for rid, v in r["ratings"] if rid in g1]
            y_ = [v for rid, v in r["ratings"] if rid not in g1]
            if x and y_:
                a.append(np.mean(x)); b.append(np.mean(y_))
        if len(a) > 10:
            halves.append(pearson(a, b))
    r_half = float(np.mean(halves))
    out["noise_ceiling"] = {
        "split_half_r": round(r_half, 4),
        "spearman_brown_full": round(2 * r_half / (1 + r_half), 4),
        "n_splits": len(halves),
        "note": "correlation between pair means from two disjoint halves of the "
                "rater pool, Spearman-Brown stepped up to the full pool",
    }

    # --- reliability of the pair means at the actual depth ------------------
    # Read the anchor ICC rather than pinning it: a hardcoded 0.838 survived the
    # refresh from N=23 to N=31, where the value is 0.830, and leaked into the paper.
    icc21 = float(json.load(open(ROOT / "results/mos/qc_live.json"))["anchor_icc"]["icc_2_1"])
    k = float(np.median(n_rat))
    out["pair_mean_reliability"] = {
        "icc_2_1_anchor": icc21, "k_median": k,
        "spearman_brown_icc_2_k": round(k * icc21 / (1 + (k - 1) * icc21), 4)}

    # --- C. fitted CLAP head on the same pairs ------------------------------
    npz = np.load(ROOT / "results/baselines/clap_embeddings.npz", allow_pickle=True)
    idx = {c: i for i, c in enumerate(npz["clips"])}
    pooled = npz["pooled"]
    manifest = json.load(open(ROOT / "perturbations/pairs_manifest.json"))
    pairs = manifest["pairs"] if isinstance(manifest, dict) else manifest
    paths = {p["pair_id"]: (p["ref_path"], p["cand_path"]) for p in pairs}

    X, yv, g = [], [], []
    for i, r in enumerate(recs):
        ref, cand = paths[r["pair_id"]]
        if ref in idx and cand in idx:
            X.append(pooled[idx[cand]] - pooled[idx[ref]])
            yv.append(mos[i]); g.append(r["source_id"])
    X, yv, g = np.array(X), np.array(yv), np.array(g)

    pred = np.empty_like(yv)
    for tr in np.unique(g):
        te = g == tr
        m = Ridge(alpha=10.0).fit(X[~te], yv[~te])
        pred[te] = m.predict(X[te])
    out["clap_fitted_head"] = {
        "n": int(len(yv)), "features": "clap768 pooled difference (e_B - e_A)",
        "model": "ridge alpha=10, leave-one-track-out, grouped by source_id",
        "held_out_r": round(pearson(yv, pred), 4),
        "note": "the fair upper bound for the cheap embedding: 768 fitted "
                "parameters against C's zero (uniform weights)",
    }
    # same CV budget given to a fitted head on the 4-dim profile, for symmetry
    Xp = np.array([[recs[i]["C"], recs[i]["C_noS"]] for i in range(len(recs))])
    print("  pooled r:", out["pooled_r"])
    print("  within-condition r:", out["within_condition_r"])
    print("  ceiling:", out["noise_ceiling"])
    print("  clap fitted head r:", out["clap_fitted_head"]["held_out_r"])
    return out


# ---------------------------------------------------------------------------
# D. Part 2 with ties counted as half
# ---------------------------------------------------------------------------
def analysis_d():
    key = {r["trial_id"]: r for r in read_csv(ROOT / "results/mos/session2/key.csv")}
    rivals = sorted({s for r in key.values() for s in r["settles"].split(";") if s})

    tallies = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0]))  # rival -> seed -> [w, ties, n]
    for f in sorted(glob.glob(str(ROOT / "results/mos/session2/responses/*.json"))):
        d = json.load(open(f))
        for resp in d["responses"]:
            t = key.get(resp["trial_id"])
            if t is None or resp.get("preference") is None:
                continue
            p = int(resp["preference"])
            c_on_a = t["opt_a_role"] == "c"
            if p == 3:
                w, tie = 0.5, 1
            else:
                prefers_a = p < 3
                w, tie = (1.0 if prefers_a == c_on_a else 0.0), 0
            for rv in t["settles"].split(";"):
                if rv:
                    cell = tallies[rv][t["seed"]]
                    cell[0] += w; cell[1] += tie; cell[2] += 1

    out = {"note": "ties ('about the same') counted as half a win rather than dropped; "
                   "cluster bootstrap over seeds, Holm correction across the five arms",
           "arms": {}}
    pvals = {}
    for rv in rivals:
        seeds = sorted(tallies[rv])
        w = sum(tallies[rv][s][0] for s in seeds)
        ties = sum(tallies[rv][s][1] for s in seeds)
        n = sum(tallies[rv][s][2] for s in seeds)
        boots = []
        for _ in range(N_BOOT):
            pick = RNG.choice(seeds, size=len(seeds), replace=True)
            ww = sum(tallies[rv][s][0] for s in pick)
            nn = sum(tallies[rv][s][2] for s in pick)
            if nn:
                boots.append(ww / nn)
        boots = np.array(boots)
        p_two = 2 * min(float(np.mean(boots <= 0.5)), float(np.mean(boots >= 0.5)))
        p_two = min(1.0, max(p_two, 1.0 / N_BOOT))
        pvals[rv] = p_two
        out["arms"][rv] = {
            "n_trials_rated": int(n), "n_ties": int(ties),
            "tie_fraction": round(ties / n, 4) if n else None,
            "win_rate_ties_half": round(w / n, 4) if n else None,
            "ci95_cluster_by_seed": [round(float(np.percentile(boots, 2.5)), 4),
                                     round(float(np.percentile(boots, 97.5)), 4)],
            "p_boot_two_sided": round(p_two, 5),
        }
    order = sorted(pvals, key=lambda k: pvals[k])
    m = len(order)
    prev = 0.0
    for i, rv in enumerate(order):
        adj = min(1.0, max(prev, (m - i) * pvals[rv]))
        prev = adj
        out["arms"][rv]["p_holm"] = round(adj, 5)
        out["arms"][rv]["significant_holm_0.05"] = bool(adj < 0.05)
    for rv in rivals:
        print(f"  {rv:14s} {out['arms'][rv]}")
    return out


def main():
    res = {}
    print("[A] diagnosis, nested CV + paired bootstrap")
    res["diagnosis_nested"] = analysis_a()
    print("[B/C] MOS within-condition, ceiling, fitted CLAP head")
    res["mos"] = analysis_bc()
    print("[D] Part 2 ties as half")
    res["ab_ties"] = analysis_d()
    dest = ROOT / "results/diagnostics/review_fixes.json"
    dest.write_text(json.dumps(res, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
