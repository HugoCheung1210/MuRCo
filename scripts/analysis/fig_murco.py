#!/usr/bin/env python
"""The MuRCo overview figure: the one diagram a metric paper is expected to carry.

WHAT IT ARGUES. CLAP and COCOLA both open with a left-to-right schematic that ends in a
single number. Ours ends in two things, and that is the whole contribution: the four-term
profile says *what* changed and the composite $C$ says *how much*. A figure that stopped at
$C$ would draw us as a rival of the same shape rather than a different one, so the split
ending is the point of the layout and not decoration.

DESIGN NOTES, so a later edit does not undo them.

  * Colour follows the ENTITY, exactly as `_figstyle` and `make_figures.py` fix it. Blue is
    context that was preserved, gold is content a generator produced, and the four
    dimension accents are `make_figures.COL`, unchanged, so this figure and the separability
    figure name the same term with the same hue.
  * The dimension colours are used as ACCENT RULES beside a label, never as fills behind
    text. That is deliberate. Running the four through the dataviz palette validator returns
    two warnings: S's gold sits at 2.99:1 against a light surface (below the 3:1 floor) and
    the S-T pair separates by only dE 6.9 under protanopia, which is inside the 6-8 band that
    is legal ONLY with a secondary encoding. Every accent here is attached to a spelled-out
    symbol and name, so identity never rests on hue. Do not "simplify" this into coloured
    boxes.
  * Waveforms are deterministic (seeded), so the figure is reproducible byte-for-byte.

Nothing here is measured. The bar heights in the profile panel are illustrative of a
low-pass edit's shape (T falls, H and R hold) and are drawn from the qualitative pattern
Figure 1 reports, not read from a results file. The caption must not quote them as data.

Usage: python scripts/analysis/fig_murco.py [--out figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _figstyle import apply_style, INK, GREY, FROZEN, EDITED  # noqa: E402

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

# make_figures.COL, inlined rather than imported: that module runs analysis at import time.
COL = {"H": "#2a78d6", "T": "#008300", "R": "#d55181", "S": "#c98500"}
PAPER = "#f7f6f3"        # panel fill, a half-step off the page so edges read without a rule
HAIR = "#d9d7d1"

DIMS = [
    ("H", "Harmony",   "Mean chroma, cosine"),
    ("T", "Timbre",    "MFCC Gaussians, Fr$\\acute{e}$chet"),
    ("R", "Rhythm",    "Tempo $+$ onset phase, beat-gated"),
    ("S", "Semantics", "Audio LM, $P(\\mathrm{Yes})$"),
]


def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def panel(ax, x, y, w, h, fc=PAPER, ec=HAIR, lw=0.7, r=0.012, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def arrow(ax, a, b, color=GREY, lw=0.8, rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, color=color, linewidth=lw,
                                 mutation_scale=7, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


def wave(ax, x0, x1, yc, amp, color, seed, n=260, lw=0.55):
    """A deterministic pseudo-waveform. Seeded so reruns are byte-identical."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    env = 0.35 + 0.65 * np.abs(np.sin(np.pi * t * 2.1 + seed))
    y = env * rng.normal(0, 1, n)
    y = y / np.abs(y).max() * amp
    xs = x0 + (x1 - x0) * t
    ax.vlines(xs, yc - y, yc + y, color=color, linewidth=lw, zorder=3)


def fig_murco(out: Path):
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    blank(ax)
    cap = dict(fontsize=7.3, color=GREY, ha="left", va="center")

    # ---------------------------------------------------------------- the pair
    ax.text(0.004, 0.955, "The edit", **cap)
    ty, th = 0.575, 0.215
    panel(ax, 0.0, ty, 0.232, th, fc="white")
    wave(ax, 0.011, 0.121, ty + th / 2, 0.066, FROZEN, seed=3)
    wave(ax, 0.125, 0.221, ty + th / 2, 0.066, EDITED, seed=11)
    ax.plot([0.1232, 0.1232], [ty + 0.014, ty + th - 0.014], color=HAIR, lw=0.7, zorder=4)
    ax.text(0.066, ty - 0.048, "Context $A$", fontsize=7.6, color=FROZEN,
            ha="center", va="top")
    ax.text(0.173, ty - 0.048, "Candidate $B$", fontsize=7.6, color=EDITED,
            ha="center", va="top")
    ax.text(0.116, ty - 0.165, "The regenerated part, which has\nno earlier version to score against",
            fontsize=6.7, color=GREY, ha="center", va="top", linespacing=1.35)

    # ------------------------- four dimensions, fed by a bus rather than a fan.
    # Four curved arrows from one point crossed each other and the S arrow swept
    # back over the R box; a bus is the standard fix and reads at column width.
    ax.text(0.330, 0.955, "Four comparisons of $B$ against $A$", **cap)
    bx, bw, bh = 0.330, 0.340, 0.178
    tops = [0.700, 0.487, 0.274, 0.061]
    mids = [b + bh / 2 for b in tops]
    inbus = 0.285
    ax.plot([inbus, inbus], [min(mids), max(mids)], color=HAIR, lw=0.8, zorder=1)
    arrow(ax, (0.236, ty + th / 2), (inbus, ty + th / 2), style="-")
    ax.plot([inbus, inbus], [ty + th / 2, max(mids)], color=HAIR, lw=0.8, zorder=1)
    for (sym, name, how), by in zip(DIMS, tops):
        panel(ax, bx, by, bw, bh)
        ax.add_patch(FancyBboxPatch((bx + 0.011, by + 0.028), 0.0075, bh - 0.056,
                                    boxstyle="round,pad=0,rounding_size=0.0035",
                                    facecolor=COL[sym], edgecolor="none", zorder=3))
        ax.text(bx + 0.034, by + bh * 0.63, f"${sym}$   {name}", fontsize=8.3,
                color=INK, ha="left", va="center")
        ax.text(bx + 0.034, by + bh * 0.235, how, fontsize=6.8, color=GREY,
                ha="left", va="center")
        arrow(ax, (inbus, by + bh / 2), (bx - 0.005, by + bh / 2))

    ax.text(bx + bw / 2, 0.008, "Every term returns a score in $[0,1]$, where 1 is coherent",
            fontsize=6.8, color=GREY, ha="center", va="bottom")

    # ------------------------------------------------- the two things it returns
    ox = 0.742
    ax.text(ox, 0.955, "What MuRCo returns", **cap)
    outbus = 0.700
    ax.plot([outbus, outbus], [min(mids), max(mids)], color=HAIR, lw=0.8, zorder=1)
    for by in tops:
        arrow(ax, (bx + bw + 0.005, by + bh / 2), (outbus, by + bh / 2), style="-")
    ax.plot([outbus, outbus], [0.660, max(mids)], color=HAIR, lw=0.8, zorder=1)
    arrow(ax, (outbus, 0.660), (ox - 0.006, 0.660))

    # (a) the profile: WHAT changed
    panel(ax, ox, 0.487, 0.258, 0.388, fc="white")
    ax.text(ox + 0.129, 0.826, "The profile", fontsize=8.3, color=INK,
            ha="center", va="center")
    ax.text(ox + 0.129, 0.762, "$\\mathit{What}$ changed", fontsize=7.0, color=GREY,
            ha="center", va="center")
    vals = {"H": 0.94, "T": 0.40, "R": 0.90, "S": 0.70}
    bl, bb, bwid, bmax = ox + 0.040, 0.560, 0.038, 0.155
    ax.plot([bl - 0.012, bl + 4 * 0.050], [bb + bmax, bb + bmax],
            color=HAIR, lw=0.6, ls=(0, (2.2, 2.2)), zorder=2)
    ax.text(bl - 0.018, bb + bmax, "1", fontsize=6.2, color=GREY, ha="right", va="center")
    ax.plot([bl - 0.012, bl + 4 * 0.050], [bb, bb], color=GREY, lw=0.7, zorder=3)
    ax.text(bl - 0.018, bb, "0", fontsize=6.2, color=GREY, ha="right", va="center")
    for i, (sym, _, _) in enumerate(DIMS):
        x = bl + i * 0.050
        ax.add_patch(FancyBboxPatch((x, bb), bwid, bmax * vals[sym],
                                    boxstyle="round,pad=0,rounding_size=0.005",
                                    facecolor=COL[sym], edgecolor="none", zorder=4))
        ax.text(x + bwid / 2, bb - 0.028, f"${sym}$", fontsize=7.4, color=INK,
                ha="center", va="top")

    # (b) the composite: HOW MUCH
    panel(ax, ox, 0.088, 0.258, 0.300, fc="white")
    ax.text(ox + 0.129, 0.330, "The composite", fontsize=8.3, color=INK,
            ha="center", va="center")
    ax.text(ox + 0.129, 0.232, "$C = H^{w_H} T^{w_T} R^{w_R} S^{w_S}$",
            fontsize=8.2, color=INK, ha="center", va="center")
    ax.text(ox + 0.129, 0.138, "$\\mathit{How\\ much}$, one number to rank by",
            fontsize=7.0, color=GREY, ha="center", va="center")
    arrow(ax, (ox + 0.129, 0.480), (ox + 0.129, 0.396), lw=0.8)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.99, bottom=0.01)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_murco.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_murco.pdf/.png")


# ---------------------------------------------------------------------------
# Single-column variant, for the ICASSP paper. Same content, relaid out.
# ---------------------------------------------------------------------------
# WHY A SECOND LAYOUT rather than \includegraphics scaling. The wide figure is
# 2.6:1, so at an 86 mm column it renders 33 mm tall and its labels fall under 4 pt.
# A full-width float also reserves a block spanning both columns, which measured at
# roughly 40 lines of body text against about 13 for a column float. The relayout is
# what makes the figure affordable, not the width argument.
COL_DIMS = [
    ("H", "Harmony",   "chroma cosine"),
    ("T", "Timbre",    "MFCC Fr$\\acute{e}$chet"),
    ("R", "Rhythm",    "tempo $+$ phase"),
    ("S", "Semantics", "audio LM"),
]

# ICASSP allows nothing below 9 pt in the rendered PDF, and a figure interior is not
# exempt. savefig writes a tight bbox slightly wider than `figsize`, and the float then
# scales that down to the 86 mm column, so a point size set here lands smaller on the
# page. MINPT carries the headroom for that shrink. The 2026-09-08 relayout is what
# makes the floor affordable: the earlier version set 6.0 pt here and rendered at 4.3.
# Verify with `python doc/paper/check_hz.py`, never by arithmetic on this constant.
MINPT = 9.6


def fig_murco_column(out: Path):
    """Single-column MuRCo schematic, laid out around a 9 pt floor.

    Everything the figure once whispered in 6 pt now lives in the caption instead. Only
    labels are left inside, because a label at 9 pt fits and a sentence at 9 pt does not.
    That is why COL_DIMS' third column is unused here: it is the wide layout's, and the
    single-column caption carries the same four descriptions in running text.
    """
    fig, ax = plt.subplots(figsize=(3.30, 2.02))
    blank(ax)

    # -- the pair ----------------------------------------------------------
    ty, th = 0.836, 0.104
    panel(ax, 0.085, ty, 0.83, th, fc="white")
    wave(ax, 0.098, 0.495, ty + th / 2, 0.036, FROZEN, seed=3)
    wave(ax, 0.505, 0.902, ty + th / 2, 0.036, EDITED, seed=11)
    ax.plot([0.500, 0.500], [ty + 0.007, ty + th - 0.007], color=HAIR, lw=0.7, zorder=4)
    ax.text(0.296, ty + th + 0.026, "Context $A$", fontsize=MINPT, color=FROZEN,
            ha="center", va="bottom")
    ax.text(0.704, ty + th + 0.026, "Candidate $B$", fontsize=MINPT, color=EDITED,
            ha="center", va="bottom")
    arrow(ax, (0.5, ty - 0.012), (0.5, ty - 0.054))

    # -- four comparisons, 2 x 2 ------------------------------------------
    gw, gh = 0.405, 0.115
    xs, ys = [0.085, 0.510], [0.626, 0.487]
    for i, (sym, name, how) in enumerate(COL_DIMS):
        x, y = xs[i % 2], ys[i // 2]
        panel(ax, x, y, gw, gh)
        ax.add_patch(FancyBboxPatch((x + 0.014, y + 0.022), 0.009, gh - 0.044,
                                    boxstyle="round,pad=0,rounding_size=0.004",
                                    facecolor=COL[sym], edgecolor="none", zorder=3))
        ax.text(x + 0.044, y + gh * 0.50, f"${sym}$  {name}", fontsize=MINPT,
                color=INK, ha="left", va="center")
    arrow(ax, (0.5, 0.477), (0.5, 0.435))

    # -- the two outputs ---------------------------------------------------
    # Asymmetric on purpose: the composite panel has to hold the definition of $C$ on
    # one line at 9.6 pt, and that string is wider than a bar chart of four bars.
    panel(ax, 0.085, 0.026, 0.370, 0.396, fc="white")
    pc = 0.085 + 0.370 / 2
    ax.text(pc, 0.372, "The profile", fontsize=MINPT, color=INK,
            ha="center", va="center")
    vals = {"H": 0.94, "T": 0.40, "R": 0.90, "S": 0.70}
    bwid, pitch, bb, bmax = 0.044, 0.060, 0.155, 0.150
    bl = pc - (3 * pitch + bwid) / 2
    ax.plot([bl - 0.014, bl + 3 * pitch + bwid + 0.014], [bb, bb],
            color=GREY, lw=0.7, zorder=3)
    for i, (sym, _, _) in enumerate(COL_DIMS):
        x = bl + i * pitch
        ax.add_patch(FancyBboxPatch((x, bb), bwid, bmax * vals[sym],
                                    boxstyle="round,pad=0,rounding_size=0.005",
                                    facecolor=COL[sym], edgecolor="none", zorder=4))
        ax.text(x + bwid / 2, bb - 0.012, f"${sym}$", fontsize=MINPT, color=INK,
                ha="center", va="top")
    ax.text(pc, 0.053, "$\\mathit{What}$ changed", fontsize=MINPT, color=GREY,
            ha="center", va="center")

    panel(ax, 0.475, 0.026, 0.440, 0.396, fc="white")
    cc = 0.475 + 0.440 / 2
    ax.text(cc, 0.372, "The composite", fontsize=MINPT, color=INK,
            ha="center", va="center")
    ax.text(cc, 0.228, "$C = H^{w_H} T^{w_T} R^{w_R} S^{w_S}$",
            fontsize=MINPT, color=INK, ha="center", va="center")
    ax.text(cc, 0.090, "$\\mathit{How\\ much}$,\nfor ranking", fontsize=MINPT,
            color=GREY, ha="center", va="center", linespacing=1.35)

    fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_murco_col.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_murco_col.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--layout", choices=("wide", "column", "both"), default="both")
    a = ap.parse_args()
    apply_style()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.layout in ("wide", "both"):
        fig_murco(out)
    if a.layout in ("column", "both"):
        fig_murco_column(out)


if __name__ == "__main__":
    main()
