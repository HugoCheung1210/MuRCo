#!/usr/bin/env python
"""The displacement test: is a different section of the same piece a different piece?

The edit-family diagnosis rewards sensitivity to any change, so it never asks a metric to
stay HIGH on a pair that coheres without being a copy.  This does.  The positives are the
temporal set's displacement family, where B is another window of the SAME track, and the
negatives are Setup 1's substitution family, where B comes from a different track.  A
coherence metric should rank the positives above the negatives.  A preservation metric
that reads misalignment as damage need not.

Three tables, written to JSON:

  levels  mean of every metric by family, temporal and Setup-1 side by side.
  auc     displacement vs substitution, pooled and split by same/cross genre, with a
          cluster bootstrap over source tracks on each AUC and on the C - MuseCPEval
          difference.
  within  control vs displacement inside the temporal set, which a preservation metric
          SHOULD separate.  Reported so the reader sees the two metrics answer different
          questions rather than one being broken.

CAVEAT carried into the output: temporal pairs are 8 s and Setup-1 pairs are 10 s.  H/T/R
read the whole file, S read 8 s in both, MuseCPEval read the whole file in both.

Usage (DSP env, from the repo root)::

    python scripts/analysis/musecp_displacement.py --out results/musecp/displacement.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve()
try:  # sibling imports across script groups (see scripts/README.md)
    sys.path.insert(0, str(_here.parent.parent))
    import _paths  # noqa: F401
except ImportError:
    pass

DIMS = ["H", "T", "R", "S"]


def repo_root() -> Path:
    for p in [_here] + list(_here.parents):
        if (p / "CLAUDE.md").exists() or (p / ".git").exists():
            return p
    return _here.parent.parent.parent


def read_csv(path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank AUC, higher score => label 1.  Same estimator as compute_S_value.py."""
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    n1 = int(y.sum()); n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(np.argsort(score)) + 1
    return float((order[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(score, y, groups, n_boot=5000, seed=0):
    """Cluster bootstrap over source tracks on one AUC."""
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups))
    idx = {g: np.flatnonzero(groups == g) for g in gs}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(gs), size=len(gs), replace=True)
        sel = np.concatenate([idx[gs[i]] for i in pick])
        a = auc(score[sel], y[sel])
        if np.isfinite(a):
            draws.append(a)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def boot_auc_diff(sa, sb, y, groups, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups))
    idx = {g: np.flatnonzero(groups == g) for g in gs}
    obs = auc(sa, y) - auc(sb, y)
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(len(gs), size=len(gs), replace=True)
        sel = np.concatenate([idx[gs[i]] for i in pick])
        d = auc(sa[sel], y[sel]) - auc(sb[sel], y[sel])
        if np.isfinite(d):
            draws.append(d)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"delta_auc": round(float(obs), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def geo_C(row, dims=DIMS):
    v = [max(float(row[f"score_{d}"]), 1e-6) for d in dims]
    return float(np.exp(np.mean(np.log(v))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal-musecp", default="results/musecp/temporal/musecp_metrics.csv")
    ap.add_argument("--temporal-scores", default="results/temporal/pair_scores.csv")
    ap.add_argument("--temporal-s-cache", default="results/s_cache/s_scores_temporal.json",
                    help="the temporal pair_scores.csv carries H,T,R only; S lives in the "
                         "cache, served by pair_id (StructuralDimension contract)")
    ap.add_argument("--setup1-musecp", default="results/musecp/setup1/musecp_metrics.csv")
    ap.add_argument("--setup1-scores", default="results/coherence/pair_scores.csv")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--out", default="results/musecp/displacement.json")
    args = ap.parse_args()
    root = repo_root()

    # ---- load, join MuRCo scores with MuseCPEval metrics on pair_id ----
    def join(scores_path, musecp_path, s_cache=None):
        sc = {r["pair_id"]: r for r in read_csv(root / scores_path)}
        if s_cache:                      # S is cached separately for the temporal run
            blob = json.loads((root / s_cache).read_text())
            table = blob.get("scores", blob)
            missing = [p for p in sc if p not in table]
            if missing:
                raise SystemExit(f"{len(missing)} pairs absent from {s_cache}, "
                                 f"e.g. {missing[:3]}")
            for pid, row in sc.items():
                row["score_S"] = table[pid]
        mc = {r["pair_id"]: r for r in read_csv(root / musecp_path)}
        cols = [c for c in next(iter(mc.values())) if c != "pair_id"]
        rows = []
        for pid, s in sc.items():
            if pid not in mc:
                continue
            r = {"pair_id": pid, "source_id": s["source_id"],
                 "genre": s.get("genre", ""), "perturbation": s["perturbation"]}
            for d in DIMS:
                r[d] = float(s[f"score_{d}"])
            r["C"] = geo_C(s)
            r["C_noS"] = geo_C(s, ["H", "T", "R"])
            for c in cols:
                r[c] = float(mc[pid][c]) if mc[pid][c] != "" else np.nan
            rows.append(r)
        return rows, cols

    temp, cols = join(args.temporal_scores, args.temporal_musecp, args.temporal_s_cache)
    main_, cols2 = join(args.setup1_scores, args.setup1_musecp)
    for r in temp:
        r["_set"] = "temporal"
    for r in main_:
        r["_set"] = "setup1"
    assert cols == cols2, "metric columns differ between the two runs"
    print(f"temporal {len(temp)} pairs, setup1 {len(main_)} pairs, {len(cols)} musecp metrics")

    feats = cols + ["C", "C_noS"] + DIMS
    report = {"meta": {
        "n_temporal": len(temp), "n_setup1": len(main_), "metrics": feats,
        "caveat": "temporal pairs are 8 s and Setup-1 pairs 10 s; H/T/R read the whole "
                  "file, S read 8 s in both, MuseCPEval read the whole file in both",
        "sources": {k: v for k, v in vars(args).items() if k.endswith(("musecp", "scores"))},
    }}

    # ---- 1. levels ----
    def level_block(rows, fams, tag):
        out = {}
        for fam in fams:
            sub = [r for r in rows if r["perturbation"] == fam]
            if not sub:
                continue
            out[f"{tag}:{fam}"] = {f: round(float(np.nanmean([r[f] for r in sub])), 4)
                                   for f in feats}
            out[f"{tag}:{fam}"]["n"] = len(sub)
        return out

    levels = {}
    levels.update(level_block(temp, ["control", "displacement", "shuffle", "reverse"], "temporal"))
    levels.update(level_block(main_, ["control", "style_swap"], "setup1"))
    # displacement near/far and swap same/cross need the pair_id suffix
    for tag, rows, fam, keys in [("temporal", temp, "displacement", ["disp_near", "disp_far"]),
                                 ("setup1", main_, "style_swap", ["swap_same", "swap_cross"])]:
        for k in keys:
            sub = [r for r in rows if r["perturbation"] == fam and r["pair_id"].endswith(k)]
            if sub:
                levels[f"{tag}:{k}"] = {f: round(float(np.nanmean([r[f] for r in sub])), 4)
                                        for f in feats}
                levels[f"{tag}:{k}"]["n"] = len(sub)
    report["levels"] = levels
    # The two sets were scored in SEPARATE runs on different windows, so a metric whose
    # control level differs between them carries a run offset into any pooled AUC.  The
    # control is the same construction in both (A vs a resynthesised copy of A), so this
    # gap is the confound, measured rather than assumed.
    gap = {f: round(levels["temporal:control"][f] - levels["setup1:control"][f], 4)
           for f in feats}
    report["control_level_gap_temporal_minus_setup1"] = gap
    print("\ncontrol-level gap between the two runs (temporal - setup1):")
    for f in ["C", "C_noS", "S", "H", "T", "R", "musecp_mean12", "musecp_mean10"]:
        flag = "   <-- confounds any pooled AUC" if abs(gap[f]) > 0.01 else ""
        print(f"   {f:>16}  {gap[f]:+.4f}{flag}")

    # ---- 2. AUC: same-track displacement (1) vs other-track substitution (0) ----
    ctrl_mean = {"temporal": {f: levels["temporal:control"][f] for f in feats},
                 "setup1": {f: levels["setup1:control"][f] for f in feats}}

    def norm(rows, f, on):
        """Score divided by the mean control of its OWN run.

        Monotone within each set, so within-set ordering is untouched; it removes the
        location offset between the two runs.  It cannot remove a difference in spread,
        so a normalised AUC is an improvement on the raw one and not a guarantee.
        """
        v = np.array([r[f] for r in rows], float)
        if not on:
            return v
        c = np.array([ctrl_mean[r["_set"]][f] for r in rows], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(c) > 1e-9, v / c, np.nan)

    def auc_block(pos_rows, neg_rows, title):
        rows = pos_rows + neg_rows
        y = np.array([1] * len(pos_rows) + [0] * len(neg_rows))
        groups = np.array([r["source_id"] for r in rows])
        block = {"n_pos": len(pos_rows), "n_neg": len(neg_rows), "auc": {},
                 "auc_control_normalised": {}}
        for f in feats:
            s = np.array([r[f] for r in rows], float)
            if not np.isfinite(s).any():
                continue
            block["auc"][f] = {"auc": round(auc(s, y), 4),
                               "ci95": boot_auc(s, y, groups, args.boot)}
            sn = norm(rows, f, True)
            block["auc_control_normalised"][f] = {
                "auc": round(auc(sn, y), 4), "ci95": boot_auc(sn, y, groups, args.boot)}
        for rival in ("musecp_mean12", "musecp_mean10"):
            if rival in feats:
                block[f"C - {rival}"] = boot_auc_diff(
                    norm(rows, "C", True), norm(rows, rival, True), y, groups, args.boot)
                block[f"C_noS - {rival}"] = boot_auc_diff(
                    norm(rows, "C_noS", True), norm(rows, rival, True), y, groups, args.boot)
        print(f"\n{title}  (+{len(pos_rows)} / -{len(neg_rows)})")
        print(f"   {'metric':>16}  {'raw':>28}   {'control-normalised':>28}")
        for f in ["C", "S", "C_noS", "H", "T", "R", "musecp_mean12", "musecp_mean10"]:
            if f in block["auc"]:
                b, n = block["auc"][f], block["auc_control_normalised"][f]
                print(f"   {f:>16}  {b['auc']:.4f} {str(b['ci95']):>21}"
                      f"   {n['auc']:.4f} {str(n['ci95']):>21}")
        return block

    disp = [r for r in temp if r["perturbation"] == "displacement"]
    swap = [r for r in main_ if r["perturbation"] == "style_swap"]
    swap_same = [r for r in swap if r["pair_id"].endswith("swap_same")]
    swap_cross = [r for r in swap if r["pair_id"].endswith("swap_cross")]
    report["auc"] = {
        "displacement_vs_substitution": auc_block(disp, swap, "displacement vs substitution"),
        "displacement_vs_swap_cross": auc_block(disp, swap_cross, "displacement vs cross-genre swap"),
        "displacement_vs_swap_same": auc_block(disp, swap_same, "displacement vs same-genre swap"),
    }

    # ---- 3. within the temporal set: control (1) vs displacement (0) ----
    ctrl = [r for r in temp if r["perturbation"] == "control"]
    rows = ctrl + disp
    y = np.array([1] * len(ctrl) + [0] * len(disp))
    groups = np.array([r["source_id"] for r in rows])
    within = {"n_control": len(ctrl), "n_displacement": len(disp), "auc": {}}
    for f in feats:
        s = np.array([r[f] for r in rows], float)
        if np.isfinite(s).any():
            within["auc"][f] = {"auc": round(auc(s, y), 4),
                                "ci95": boot_auc(s, y, groups, args.boot)}
    report["within_temporal_control_vs_displacement"] = within
    print("\ncontrol vs displacement inside the temporal set")
    for f in ["C", "S", "C_noS", "musecp_mean12", "musecp_mean10"]:
        if f in within["auc"]:
            print(f"   {f:>16}  AUC {within['auc'][f]['auc']:.4f}")

    dest = root / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1))
    print("\nwrote", dest.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
