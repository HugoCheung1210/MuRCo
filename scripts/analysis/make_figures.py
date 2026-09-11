#!/usr/bin/env python
"""Render dissertation figures from the cached results/ JSONs. DSP env, read-only.

Figures (all saved as .pdf + .png under --out, default figures/):
  fig_separability_heatmap   4 dims x 5 families mean-drop heatmap (the headline matrix)
  fig_drift_regimes          3-panel drop-vs-iteration curves (SAO / ACE / MusicGen)
  fig_dose_response          per-family dose-response forest (mean within-track rho + CI95)
  fig_diagnosis              feature-set accuracy bars + HTRS confusion matrix (E-A)
  fig_omni_signature         MF vs Omni per-family S drop (robustness)
  fig_gap_ablation           gap-0 vs gap-1 per-family S drop (M.1)
  H_pitch_gradient           H vs transposition size (tonal, not semitone, distance)

Usage: python scripts/analysis/make_figures.py [--out figures]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

# ---------------------------------------------------------------------------
# Palette — validated for print on white (all-pairs CVD check; markers/direct
# labels are the required secondary encoding). Fixed assignment everywhere:
# color follows the entity, never the panel.
# ---------------------------------------------------------------------------
COL = {
    "H": "#2a78d6",           # blue
    "T": "#008300",           # green
    "R": "#d55181",           # magenta (dark step for print)
    "S": "#c98500",           # yellow-gold (dark step for print)
    "C": "#0b0b0b",           # composite = primary ink
    "clap_htsat": "#898781",  # baselines = grays, dashed
    "scs": "#52514e",
    "cbase": "#b0aea6",
    "omni": "#2a78d6",        # 2nd backbone in the MF-vs-Omni figure
}
MARK = {"H": "o", "T": "s", "R": "^", "S": "D", "C": "o",
        "clap_htsat": "v", "scs": "P", "cbase": "X"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
TARGET_GREEN = "#00A651"   # outlines the dimension each family is designed to move

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec",
                 "#3987e5", "#256abf", "#184f95", "#0d366b"])

FAMILIES = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]
FAM_LABEL = {"pitch_shift": "transpose", "time_stretch": "time-stretch",
             "lowpass": "lowpass", "distortion": "distortion",
             "style_swap": "style-swap"}
# §3 owners: transpose->H, stretch->R, lowpass/distortion->T, style-swap->S
OWNER = {"pitch_shift": "H", "time_stretch": "R", "lowpass": "T",
         "distortion": "T", "style_swap": "S"}

# Font contract is shared with every other figure script; see _figstyle.py for the
# audit that produced it. Only the settings specific to these plots stay here.
from _figstyle import apply_style
apply_style(**{
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
})


def jload(rel):
    return json.load(open(ROOT / rel))


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


# ---------------------------------------------------------------------------
def fig_separability_heatmap(out):
    sep = jload("results/separability.json")
    dims = ["H", "T", "R", "S"]
    drops = np.zeros((4, 5))
    dzs = np.zeros((4, 5))
    for row in sep["rows"]:
        if row["dimension"] not in dims:
            continue
        i = dims.index(row["dimension"])
        for fam in row["per_family"]:
            j = FAMILIES.index(fam["perturbation"])
            drops[i, j] = fam["mean_drop"]
            dzs[i, j] = fam["cohen_dz"]

    # Right panel: each column divided by its own largest entry. The raw drops are
    # dominated by T in four columns of five, so the signature the text claims is only
    # legible once the magnitude is divided out.
    share = drops / np.maximum(drops.max(axis=0, keepdims=True), 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5))
    vmax = drops.max()

    for panel, (ax, mat, is_share) in enumerate(
            zip(axes, (drops, share), (False, True))):
        ax.grid(False)
        im = ax.imshow(mat, cmap=SEQ_BLUE, vmin=0,
                       vmax=(1.0 if is_share else vmax), aspect="auto")
        ax.set_xticks(range(5), [FAM_LABEL[f] for f in FAMILIES])
        ax.set_yticks(range(4), dims)
        ax.set_xlabel("perturbation family")
        if panel == 0:
            ax.set_ylabel("dimension")
        for i in range(4):
            for j in range(5):
                v = mat[i, j]
                c = "#ffffff" if v > 0.55 * (1.0 if is_share else vmax) else INK
                if is_share:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color=c, fontsize=9, fontweight="bold")
                else:
                    ax.text(j, i - 0.12, f"{v:.3f}", ha="center", va="center",
                            color=c, fontsize=9, fontweight="bold")
                    ax.text(j, i + 0.24, f"$d_z$={dzs[i, j]:.2f}", ha="center",
                            va="center", color=c, fontsize=6.5)
        # outline the owner cell per family (signatures, not a clean diagonal)
        for j, fam in enumerate(FAMILIES):
            i = dims.index(OWNER[fam])
            # clip_on=False is load-bearing: the corner cells (H x transpose top-left,
            # S x swap bottom-right) sit on the axes boundary, so half the stroke falls
            # outside it and default clipping erased those edges. Green reads against
            # both ends of the Blues ramp, where the boxed cells run near-white to navy.
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor=TARGET_GREEN, linewidth=1.8,
                                   clip_on=False, zorder=5))
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("share of the column's largest drop" if is_share
                     else "mean drop (control $-$ perturbed)", fontsize=8)
        cb.outline.set_visible(False)
        ax.set_title("Mean drop per dimension, with $d_z$" if not is_share
                     else "The same matrix, each column scaled to its own maximum",
                     fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "fig_separability_heatmap")


# ---------------------------------------------------------------------------
def fig_drift_regimes(out):
    models = [("sao", "SAO inpaint (roundtrip ceiling)"),
              ("ace", "ACE repaint (roundtrip ceiling)"),
              ("musicgen", "MusicGen continuation (self ceiling)")]
    dims = ["H", "T", "R", "S", "C"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, (model, title) in zip(axes, models):
        d = jload(f"results/drift/drift_htrc_{model}.json")
        for dim in dims:
            iters = d[dim]["iters"]
            ks = sorted(iters, key=int)
            x = [int(k) for k in ks]
            drop = np.array([iters[k]["drop"] for k in ks])
            sem = np.array([iters[k]["std"] / np.sqrt(iters[k]["n"]) for k in ks])
            lw = 2.2 if dim == "C" else 1.6
            ax.plot(x, drop, color=COL[dim], marker=MARK[dim], lw=lw,
                    ms=4.5, markerfacecolor="white", markeredgewidth=1.2,
                    label=dim if dim != "C" else "$C$ (composite)")
            ax.fill_between(x, drop - sem, drop + sem, color=COL[dim], alpha=0.12,
                            linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        ax.set_xlabel("edit iteration")
        ax.set_title(title, fontsize=8.5)
        ax.axhline(0, color=MUTED, lw=0.8)
    axes[0].set_ylabel("drop vs ceiling")
    handles, lab = axes[0].get_legend_handles_labels()
    fig.legend(handles, lab, loc="lower center", ncol=5, fontsize=8,
               handlelength=1.8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Three drift regimes (n=10 seeds; band = $\\pm$1 SEM; y-scales differ, so "
                 "magnitudes not comparable across families)", fontsize=9, y=1.04)
    fig.tight_layout()
    save(fig, out, "fig_drift_regimes")


# ---------------------------------------------------------------------------
def fig_drift_teaser(out):
    """Chapter 1's opening figure: one loop, one line, no dimensions.

    fig_drift_regimes is the results figure and shows three panels, five series and a
    y-axis that cannot be compared across panels. Chapter 1 has none of that vocabulary
    yet, so the teaser shows the composite alone on the loop that accumulates.
    """
    d = jload("results/drift/drift_htrc_sao.json")
    iters = d["C"]["iters"]
    ks = sorted(iters, key=int)
    x = [int(k) for k in ks]
    drop = np.array([iters[k]["drop"] for k in ks])
    sem = np.array([iters[k]["std"] / np.sqrt(iters[k]["n"]) for k in ks])

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.plot(x, drop, color=COL["C"], marker=MARK["C"], lw=2.4, ms=6,
            markerfacecolor="white", markeredgewidth=1.4)
    ax.fill_between(x, drop - sem, drop + sem, color=COL["C"], alpha=0.14, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlabel("number of edits applied")
    ax.set_ylabel("loss of coherence with the original")
    ax.set_ylim(0, max(drop + sem) * 1.35)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.annotate("every edit sounded fine\non its own", xy=(8, drop[-1]),
                xytext=(2.15, max(drop + sem) * 1.20), fontsize=8.5, color=INK,
                ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_title("Repeatedly regenerating part of a passage, "
                 "Stable Audio Open", fontsize=9.5)
    fig.tight_layout()
    save(fig, out, "fig_drift_teaser")


# ---------------------------------------------------------------------------
def fig_drift_accumulation(out):
    """Paired iter-1 -> iter-8 accumulation per dimension, one row per family.

    Companion to fig_drift_regimes, which plots drop-vs-ceiling and therefore cannot
    be compared across families (their ceilings differ). This plots the *change*
    between depths, in which the per-seed ceiling cancels, so one shared x-axis is
    honest here and the three regimes can be read against each other directly.
    """
    models = [("sao", "inpaint (SAO)"), ("ace", "repaint (ACE-Step)"),
              ("musicgen", "continuation (MusicGen)")]
    dims = ["H", "T", "R", "S", "C"]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharex=True)
    for ax, (model, title) in zip(axes, models):
        d = jload(f"results/drift/drift_htrc_{model}.json")
        for row, dim in enumerate(dims):
            ps = d[dim]["per_seed"]["iters"]
            seeds = sorted(ps["1"])
            acc = np.array([ps["1"][s] - ps["8"][s] for s in seeds])
            idx = rng.integers(len(acc), size=(20000, len(acc)))
            lo, hi = np.percentile(acc[idx].mean(axis=1), [2.5, 97.5])
            y = len(dims) - 1 - row
            excludes = lo > 0 or hi < 0
            ax.plot([lo, hi], [y, y], color=COL[dim], lw=2, alpha=0.55,
                    solid_capstyle="round")
            ax.plot([acc.mean()], [y], marker=MARK[dim], ms=7, color=COL[dim],
                    markerfacecolor=COL[dim] if excludes else "white",
                    markeredgecolor=COL[dim], markeredgewidth=1.6, zorder=3)
        ax.axvline(0, color=MUTED, lw=0.9, zorder=0)
        ax.set_yticks(range(len(dims)),
                      [d if d != "C" else "$C$" for d in dims][::-1])
        ax.set_ylim(-0.6, len(dims) - 0.4)
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[1].set_xlabel("accumulation in coherence, iteration 1 to iteration 8")
    fig.suptitle("What accumulates, and where (n=10 seeds; bars = 95% bootstrap CI; "
                 "filled marker = interval excludes zero)", fontsize=9, y=1.06)
    fig.tight_layout()
    save(fig, out, "fig_drift_accumulation")


# ---------------------------------------------------------------------------
def fig_dose_response(out):
    dr = jload("results/diagnostics/dose_response.json")["dose_response"]
    metrics = ["C", "S", "clap_htsat", "scs", "cbase"]
    labels = {"C": "MuRCo", "S": "S", "clap_htsat": "CLAP-htsat",
              "scs": "SCS stand-in", "cbase": "SCS$\\times$CLAP"}
    fams = ["pitch_shift", "time_stretch", "lowpass", "distortion"]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6), sharex=True, sharey=True)
    ypos = np.arange(len(metrics))[::-1]
    for ax, fam in zip(axes, fams):
        for m, y in zip(metrics, ypos):
            e = dr[fam].get(m)
            if e is None:
                continue
            lo, hi = e["ci95"]
            ax.plot([lo, hi], [y, y], color=COL[m], lw=2, solid_capstyle="round")
            ax.plot(e["mean_rho"], y, MARK[m], color=COL[m], ms=6,
                    markerfacecolor="white", markeredgewidth=1.5)
        ax.set_title(FAM_LABEL[fam], fontsize=9)
        ax.set_xlim(-1.05, 0.05)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(r"within-track $\rho$(severity, score)")
    axes[0].set_yticks(ypos, [labels[m] for m in metrics])
    fig.suptitle("Dose-response: monotonicity with perturbation severity "
                 "(mean within-track Spearman $\\rho$, 95% bootstrap CI over 90 tracks; "
                 "more negative = more monotone)", fontsize=9, y=1.06)
    fig.tight_layout()
    save(fig, out, "fig_dose_response")


# ---------------------------------------------------------------------------
def fig_diagnosis(out):
    diag = jload("results/diagnostics/diagnosis.json")
    feats = ["HTRS", "HTR", "cbase", "scs", "clap_htsat"]
    labels = {"HTRS": "MuRCo\n(H,T,R,S)", "HTR": "MuRCo$-S$\n(H,T,R)",
              "clap_htsat": "CLAP-htsat\n(scalar)", "scs": "SCS\n(scalar)",
              "cbase": "SCS$\\times$CLAP\n(scalar)"}
    acc = [diag["results"][f]["logistic"]["accuracy"] for f in feats]
    f1 = [diag["results"][f]["logistic"]["macro_f1"] for f in feats]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.2),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.arange(len(feats))
    w = 0.38
    ax1.bar(x - w / 2, acc, w, color="#2a78d6", label="accuracy")
    ax1.bar(x + w / 2, f1, w, color="#9ec5f4", label="macro-F1")
    for xi, (a, f) in enumerate(zip(acc, f1)):
        ax1.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", fontsize=7, color=INK)
        ax1.text(xi + w / 2, f + 0.015, f"{f:.2f}", ha="center", fontsize=7, color=INK2)
    ax1.axhline(diag["meta"]["chance_accuracy"], color=MUTED, lw=1, ls="--")
    ax1.text(len(feats) - 0.5, diag["meta"]["chance_accuracy"] + 0.012, "chance",
             fontsize=7, color=MUTED, ha="right")
    ax1.set_xticks(x, [labels[f] for f in feats], fontsize=7.5)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("leave-one-track-out CV score")
    ax1.set_title("Which perturbation happened? (logistic)", fontsize=9)
    ax1.legend(fontsize=7.5, loc="upper right")
    ax1.grid(axis="x", visible=False)

    conf = diag["results"]["HTRS"]["logistic"]["confusion"]
    classes = conf["rows_true_cols_pred"]
    M = np.array(conf["matrix"], dtype=float)
    Mn = M / M.sum(axis=1, keepdims=True)
    ax2.grid(False)
    im = ax2.imshow(Mn, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(5), [FAM_LABEL[c] for c in classes], rotation=30,
                   ha="right", fontsize=7)
    ax2.set_yticks(range(5), [FAM_LABEL[c] for c in classes], fontsize=7)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("true")
    for i in range(5):
        for j in range(5):
            c = "#ffffff" if Mn[i, j] > 0.55 else INK
            ax2.text(j, i, f"{Mn[i, j]:.2f}", ha="center", va="center",
                     color=c, fontsize=7)
    ax2.set_title("MuRCo (H,T,R,S) confusion, row-normalised", fontsize=9)
    fig.tight_layout()
    save(fig, out, "fig_diagnosis")


# ---------------------------------------------------------------------------
def _s_family_drops(sepfile):
    sep = jload(sepfile)
    row = next(r for r in sep["rows"] if r["dimension"] == "S")
    return {f["perturbation"]: f["mean_drop"] for f in row["per_family"]}


def fig_omni_signature(out):
    mf = _s_family_drops("results/separability.json")
    om = _s_family_drops("results/omni/separability.json")
    order = sorted(FAMILIES, key=lambda f: -mf[f])
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - w / 2, [mf[f] for f in order], w, color=COL["S"],
           label="Music Flamingo")
    ax.bar(x + w / 2, [om[f] for f in order], w, color=COL["omni"],
           label="Qwen2.5-Omni")
    for xi, f in zip(x, order):
        ax.text(xi - w / 2, mf[f] + 0.004, f"{mf[f]:.3f}", ha="center", fontsize=6.5, color=INK)
        ax.text(xi + w / 2, om[f] + 0.004, f"{om[f]:.3f}", ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [FAM_LABEL[f] for f in order])
    ax.set_ylabel("S mean drop (control $-$ perturbed)")
    # "compressed" holds on the four families that move; time-stretch is ~0 under both
    # backbones and Omni's is the marginally larger of the two, so the title says
    # ordering rather than claiming a uniform compression the last pair contradicts.
    ax.set_title("S signature across backbones: the ordering of the five\n"
                 "families is preserved, the scale is not", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_omni_signature")


# ---------------------------------------------------------------------------
def fig_gap_ablation(out):
    man = jload("perturbations/pairs_manifest.json")["pairs"]
    fam_of = {p["pair_id"]: p["perturbation"] for p in man}
    # Held-out re-selection (2026-08-24): 18 sources disjoint from the subset the gap
    # was first tuned on, with BOTH arms scored through one read-out. The older pair
    # (s_scores.json vs s_scores_gap0.json) straddles two cache generations -- the gap-0
    # arm predates peak_safe_concat -- so it measured the read-out as much as the gap.
    # See results/diagnostics/gap_sweep.json -> "superseded".
    g1 = jload("results/s_cache/s_heldout_gap1.0.json")["scores"]
    g0 = jload("results/s_cache/s_heldout_gap0.0.json")["scores"]

    def drops(scores):
        ctrl = np.mean([v for k, v in scores.items() if fam_of.get(k) == "control"])
        d = {}
        for fam in FAMILIES:
            vals = [v for k, v in scores.items() if fam_of.get(k) == fam]
            d[fam] = ctrl - float(np.mean(vals))
        return d

    d1, d0 = drops(g1), drops(g0)
    order = sorted(FAMILIES, key=lambda f: -d1[f])
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - w / 2, [d0[f] for f in order], w, color="#9ec5f4", label="gap = 0 s")
    ax.bar(x + w / 2, [d1[f] for f in order], w, color="#2a78d6", label="gap = 1 s")
    for xi, f in zip(x, order):
        ax.text(xi - w / 2, d0[f] + 0.003, f"{d0[f]:.3f}", ha="center", fontsize=6.5, color=INK2)
        ax.text(xi + w / 2, d1[f] + 0.003, f"{d1[f]:.3f}", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, [FAM_LABEL[f] for f in order])
    ax.set_ylabel("S mean drop (control $-$ perturbed)")
    ax.set_title("Gap ablation (M.1): the 1 s gap sharpens style-swap separation",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_gap_ablation")


# ---------------------------------------------------------------------------
def fig_h_pitch_gradient(out):
    """H vs transposition size: graded by tonal distance, not semitone distance.

    Regenerates the figure that used to sit in figures/ without a script.  The
    numbers come from the canonical pair_scores.csv, so they track a rescore.
    """
    import csv
    from collections import defaultdict

    by_shift = defaultdict(list)
    with open(ROOT / "results/coherence/pair_scores.csv") as fh:
        for row in csv.DictReader(fh):
            if row["perturbation"] == "pitch_shift":
                by_shift[float(row["magnitude"])].append(float(row["score_H"]))

    shifts = sorted(by_shift)
    means = np.array([np.mean(by_shift[s]) for s in shifts])
    sems = np.array([np.std(by_shift[s], ddof=1) / np.sqrt(len(by_shift[s]))
                     for s in shifts])
    n = len(by_shift[shifts[0]])
    x = np.arange(len(shifts))

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.errorbar(x, means, yerr=sems, color=COL["H"], marker=MARK["H"],
                markersize=6, linewidth=1.6, capsize=3, zorder=3)
    # Value labels sit above their marker, clear of the line and the caps.
    for xi, m, e in zip(x, means, sems):
        ax.text(xi, m + e + 0.004, f"{m:.3f}", ha="center", va="bottom",
                fontsize=8, color=INK, zorder=4)

    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    ax.text(x[-1], 0.995, "control (unperturbed) $=1.000$", ha="right",
            va="top", fontsize=8, color=INK2)
    # The explanatory note goes in the empty band between the ceiling and the
    # data, never across the line (it used to overlap the value labels).
    ax.text(x[0] - 0.35, 0.925,
            "$\\pm2$ (whole tone) shares more chroma mass with the original\n"
            "than $\\pm1$ or $\\pm4$, so the response is not monotone in semitones",
            ha="left", va="center", fontsize=8, color=INK2, style="italic")

    ax.set_xticks(x, [f"{int(s):+d}" for s in shifts])
    ax.set_ylim(0.80, 1.035)
    ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    ax.set_xlabel("pitch shift (semitones)")
    ax.set_ylabel("H (pooled chroma cosine)")
    ax.set_title(f"H is graded by tonal distance, not semitone distance "
                 f"(FMA, {n} sources, mean $\\pm$ SEM)", fontsize=9.5)
    save(fig, out, "H_pitch_gradient")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--only", default=None,
                    help="comma-separated figure names (default: all)")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    figs = {
        "fig_separability_heatmap": fig_separability_heatmap,
        "fig_drift_regimes": fig_drift_regimes,
        "fig_drift_teaser": fig_drift_teaser,
        "fig_drift_accumulation": fig_drift_accumulation,
        "fig_dose_response": fig_dose_response,
        "fig_diagnosis": fig_diagnosis,
        "fig_omni_signature": fig_omni_signature,
        "fig_gap_ablation": fig_gap_ablation,
        "H_pitch_gradient": fig_h_pitch_gradient,
    }
    wanted = args.only.split(",") if args.only else list(figs)
    for name in wanted:
        print(name)
        figs[name](out)


if __name__ == "__main__":
    main()
