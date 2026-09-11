#!/usr/bin/env python
"""Schematic figures for Chapter 2 (related work). DSP env, no data needed.

Figures (saved as .pdf + .png under --out, default figures/):
  fig_t2m_families   the three generator families: AR tokens / latent diffusion /
                     masked parallel decoding, drawn as signal-flow boxes
  fig_edit_loops     the three iterative-edit loops of Section 4.2, annotated with
                     what each pass re-supplies and the drift regime it produces
  fig_s_protocol     the S read-out, from the A | gap | B concatenation through Music
                     Flamingo to the Yes/No softmax

These are diagrams, not plots: nothing here is fitted or measured. The regime labels
on fig_edit_loops come from results/drift/ and are quoted, not recomputed.

Usage: python scripts/analysis/fig_architectures.py [--out figures]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

# Same fixed assignment as make_figures.py: colour follows the entity.
INK = "#0b0b0b"
GREY = "#898781"
FROZEN = "#2a78d6"    # preserved / clamped content (blue)
FROZEN_LOSSY = "#a9c9ee"  # context that survives but is re-encoded each pass
EDITED = "#c98500"    # regenerated content (gold)
ACCUM = "#d55181"     # the dimension that accumulates (magenta)

from _figstyle import apply_style
apply_style()


def box(ax, x, y, w, h, label, fc="white", ec=INK, fs=8.0, lw=0.8, style="round,pad=0.006"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, zorder=3, color=INK, linespacing=1.25)


def arrow(ax, xy_from, xy_to, style="-|>", ls="-", color=INK, lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch(xy_from, xy_to, arrowstyle=style, linestyle=ls,
                                 color=color, linewidth=lw, mutation_scale=9,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def blank(ax):
    # Patches are clipped to the axes, and the rounded box style adds ~0.02 of pad in
    # data units, so the limits must leave room or the leftmost box loses its edge.
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Figure 1: the three generator families
# ---------------------------------------------------------------------------
def fig_families(out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 5.4))

    # (a) autoregressive over discrete tokens
    ax = axes[0]; blank(ax)
    ax.set_title("(a) Autoregressive token models (Jukebox, AudioLM, MusicLM, MusicGen)",
                 fontsize=8.5, loc="left", pad=4)
    box(ax, 0.00, 0.42, 0.11, 0.34, "audio", fc="#f2f2f0")
    box(ax, 0.18, 0.42, 0.14, 0.34, "RVQ\nencoder", fc="white")
    for i, dy in enumerate([0.72, 0.56, 0.40, 0.24]):
        box(ax, 0.38, dy, 0.08, 0.13, f"$q_{i+1}$", fc="#f2f2f0", fs=7.5, style="square,pad=0")
    box(ax, 0.52, 0.42, 0.20, 0.34, "transformer\n$p(q_t \\mid q_{<t}, c)$", fc="white")
    box(ax, 0.78, 0.42, 0.20, 0.34, "decoder\n$\\rightarrow$ waveform", fc="white")
    arrow(ax, (0.115, 0.59), (0.175, 0.59))
    arrow(ax, (0.325, 0.59), (0.375, 0.59))
    arrow(ax, (0.465, 0.59), (0.515, 0.59))
    arrow(ax, (0.725, 0.59), (0.775, 0.59))
    # feedback loop: sampled tokens re-enter the transformer's own context
    arrow(ax, (0.70, 0.42), (0.55, 0.42), ls=":", color=GREY, rad=-0.55)
    ax.text(0.625, 0.18, "sampled tokens re-enter the context", fontsize=7.2, color=GREY,
            ha="center", va="center")

    # (b) latent diffusion
    ax = axes[1]; blank(ax)
    ax.set_title("(b) Latent diffusion (Stable Audio Open, ACE-Step)",
                 fontsize=8.5, loc="left", pad=4)
    box(ax, 0.00, 0.42, 0.10, 0.34, "audio", fc="#f2f2f0")
    box(ax, 0.17, 0.42, 0.12, 0.34, "VAE\nencoder", fc="white")
    box(ax, 0.36, 0.42, 0.26, 0.34,
        "denoiser  $\\epsilon_\\theta(z_t, t, c)$\n$z_T \\rightarrow \\cdots \\rightarrow z_0$", fc="white")
    box(ax, 0.69, 0.42, 0.12, 0.34, "VAE\ndecoder", fc="white")
    box(ax, 0.88, 0.42, 0.10, 0.34, "audio", fc="#f2f2f0")
    arrow(ax, (0.105, 0.59), (0.165, 0.59))
    arrow(ax, (0.295, 0.59), (0.355, 0.59))
    arrow(ax, (0.625, 0.59), (0.685, 0.59))
    arrow(ax, (0.815, 0.59), (0.875, 0.59))
    ax.text(0.49, 0.18, "editing enters here, by masking the region and clamping the rest",
            fontsize=7.2, color=GREY, ha="center", va="center")
    arrow(ax, (0.49, 0.27), (0.49, 0.41), color=GREY, ls=":")

    # (c) masked parallel decoding
    ax = axes[2]; blank(ax)
    ax.set_title("(c) Masked parallel decoding (VampNet, MaskGIT-style)",
                 fontsize=8.5, loc="left", pad=4)
    xs = [0.02, 0.26, 0.50]
    labels = ["round 1", "round 2", "round $R$"]
    patterns = [[1, 1, 0, 1, 0, 1], [0, 1, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]]
    cell = 0.030
    for x0, lab, pat in zip(xs, labels, patterns):
        for j, m in enumerate(pat):
            fc = EDITED if m else FROZEN
            box(ax, x0 + j * cell, 0.40, cell * 0.88, 0.24, "", fc=fc, ec="white",
                lw=0.3, style="square,pad=0")
        ax.text(x0 + 3 * cell, 0.28, lab, fontsize=7.2, ha="center", color=INK)
    arrow(ax, (0.20, 0.52), (0.245, 0.52))
    arrow(ax, (0.44, 0.52), (0.485, 0.52))
    ax.text(0.84, 0.52, "masked positions are sampled\nindependently within a round",
            fontsize=7.2, ha="center", va="center", color=INK)
    ax.add_patch(FancyBboxPatch((0.02, 0.76), 0.055, 0.11, boxstyle="square,pad=0",
                                facecolor=EDITED, edgecolor="white"))
    ax.text(0.088, 0.815, "masked", fontsize=7.2, va="center")
    ax.add_patch(FancyBboxPatch((0.20, 0.76), 0.055, 0.11, boxstyle="square,pad=0",
                                facecolor=FROZEN, edgecolor="white"))
    ax.text(0.268, 0.815, "known", fontsize=7.2, va="center")

    fig.tight_layout(h_pad=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_t2m_families.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_t2m_families.pdf/.png")


# ---------------------------------------------------------------------------
# Figure 2: the three iterative-edit loops
# ---------------------------------------------------------------------------
def fig_edit_loops(out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.9))

    panels = [
        ("Continuation\n(MusicGen)",
         "context is entirely\nself-generated",
         "every dimension climbs",
         [(0.0, 0.0)], "cont"),
        ("Inpainting\n(Stable Audio Open)",
         "frozen region clamped\nto the original latent",
         "$S$ accumulates,\n$T$ a stationary offset",
         None, "inpaint"),
        ("Repainting\n(ACE-Step)",
         "whole clip re-encoded, but the\nsame caption, key and tempo\nare re-supplied",
         "nothing accumulates",
         None, "repaint"),
    ]

    for ax, (title, mech, regime, _, kind) in zip(axes, panels):
        blank(ax)
        ax.set_title(title, fontsize=8.5, pad=6)
        # three passes of the clip, drawn as a timeline strip
        for i, y in enumerate([0.74, 0.53, 0.32]):
            if kind == "cont":
                # the window slides: earlier material leaves, new material enters
                box(ax, 0.06 + i * 0.10, y, 0.34, 0.13, "", fc=FROZEN, ec="white", lw=0.4)
                box(ax, 0.40 + i * 0.10, y, 0.20, 0.13, "", fc=EDITED, ec="white", lw=0.4)
            else:
                # repaint re-encodes the whole clip through the autoencoder every pass,
                # so its "frozen" region is only approximately preserved (lighter blue);
                # inpainting clamps it in latent space, so it is exactly preserved.
                fc_ctx = FROZEN if kind == "inpaint" else FROZEN_LOSSY
                box(ax, 0.06, y, 0.24, 0.13, "", fc=fc_ctx, ec="white", lw=0.4)
                box(ax, 0.30, y, 0.28, 0.13, "", fc=EDITED, ec="white", lw=0.4)
                box(ax, 0.58, y, 0.24, 0.13, "", fc=fc_ctx, ec="white", lw=0.4)
            ax.text(0.90, y + 0.065, f"$k{{=}}{i+1}$", fontsize=7.2, va="center", color=GREY)
            if i < 2:
                arrow(ax, (0.44, y), (0.44, y - 0.08), color=GREY, lw=0.7)

        ax.text(0.5, 0.20, mech, fontsize=7.4, ha="center", va="center", color=INK)
        ax.text(0.5, 0.045, regime, fontsize=7.4, ha="center", va="center",
                color=ACCUM if kind != "repaint" else GREY)

    fig.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_edit_loops.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_edit_loops.pdf/.png")


# ---------------------------------------------------------------------------
# Figure 3: which pair each relational metric treats as coherent
# ---------------------------------------------------------------------------
def fig_pair_axes(out: Path):
    """The axis distinction Section 2.6 rests on, drawn rather than described.

    Six cells, and the grouping is the argument. CLAP and SCS relate two clips but
    through quantities internal to each; COCOLA's positives are simultaneous; REMAST's
    are sequential but symbolic and read through one-segment statistics; C's are
    sequential and read from audio. The sixth cell is the recent wave of one-clip
    scorers, which have no pair to draw at all, and it is labelled rather than left
    empty because that absence is the point.

    Nothing here is measured: every sampling rule is quoted from the paper that defines
    it. Scored comparisons for the same metrics live in Sections 4.4 and 4.5.
    """
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.15))
    axes = axes.ravel()
    TIME = "#c9c7c2"
    PANE = "#f2f2f0"

    def timeline(ax, y=0.075):
        arrow(ax, (0.00, y), (1.00, y), color=TIME, lw=1.0)
        ax.text(1.00, y - 0.055, "time", fontsize=7.0, ha="right", va="top", color=GREY)

    def note(ax, txt):
        """Grey aside, in the band between the timeline and the lowest box."""
        ax.text(0.50, 0.205, txt, fontsize=6.6, ha="center", va="center", color=GREY)

    def ticks(ax, xs, y=0.075):
        for x in xs:
            ax.plot([x, x], [y, y + 0.045], color=TIME, lw=0.8, zorder=0)

    def header(ax, txt, color=INK):
        ax.text(0.50, 0.92, txt, fontsize=7.2, ha="center", va="center", color=color)

    # (a) CLAP: two clips, but the trained tie runs from each clip to its caption
    ax = axes[0]; blank(ax); timeline(ax); ticks(ax, (0.02, 0.42, 0.58, 0.98))
    ax.set_title("(a) CLAP", fontsize=8.5, loc="left", pad=4)
    header(ax, "coherent $=$ embeddings point the same way")
    box(ax, 0.02, 0.28, 0.40, 0.15, "audio $A$", fc=FROZEN_LOSSY, fs=7.5)
    box(ax, 0.58, 0.28, 0.40, 0.15, "audio $B$", fc=EDITED, fs=7.5)
    arrow(ax, (0.22, 0.435), (0.22, 0.545), lw=0.8)
    arrow(ax, (0.78, 0.435), (0.78, 0.545), lw=0.8)
    box(ax, 0.04, 0.55, 0.36, 0.13, "$\\phi(A)$", fc="white", fs=7.5)
    box(ax, 0.60, 0.55, 0.36, 0.13, "$\\phi(B)$", fc="white", fs=7.5)
    arrow(ax, (0.405, 0.615), (0.595, 0.615), style="<|-|>", lw=0.9)
    # short label: anything wider collides with the two up-arrows at x=0.22/0.78
    ax.text(0.50, 0.495, "cosine", fontsize=6.6, ha="center",
            va="center", color=INK)
    # the tie the objective actually trained, drawn greyed because it is absent at test
    box(ax, 0.31, 0.755, 0.38, 0.10, "caption $t$", fc=PANE, ec=GREY, fs=6.8)
    arrow(ax, (0.22, 0.685), (0.36, 0.752), ls=(0, (2.2, 1.6)), color=GREY, lw=0.8)
    arrow(ax, (0.78, 0.685), (0.64, 0.752), ls=(0, (2.2, 1.6)), color=GREY, lw=0.8)
    note(ax, "the trained tie is audio-to-text, not audio-to-audio")

    # (b) SCS: two segments compared, but every entry is a segment against itself
    ax = axes[1]; blank(ax); timeline(ax); ticks(ax, (0.02, 0.42, 0.58, 0.98))
    ax.set_title("(b) SCS", fontsize=8.5, loc="left", pad=4)
    header(ax, "coherent $=$ same internal structure")
    box(ax, 0.02, 0.28, 0.40, 0.15, "audio $A$", fc=FROZEN_LOSSY, fs=7.5)
    box(ax, 0.58, 0.28, 0.40, 0.15, "audio $B$", fc=EDITED, fs=7.5)
    arrow(ax, (0.22, 0.435), (0.22, 0.575), lw=0.8)
    arrow(ax, (0.78, 0.435), (0.78, 0.575), lw=0.8)
    box(ax, 0.00, 0.58, 0.42, 0.16, "$M_A$\n$A$ against $A$", fc="white", fs=6.8)
    box(ax, 0.58, 0.58, 0.42, 0.16, "$M_B$\n$B$ against $B$", fc="white", fs=6.8)
    arrow(ax, (0.425, 0.66), (0.575, 0.66), style="<|-|>", lw=0.9)
    note(ax, "each matrix sees only its own segment")

    # (c) COCOLA: positives are two submixes of the SAME window
    ax = axes[2]; blank(ax); timeline(ax)
    ax.set_title("(c) COCOLA", fontsize=8.5, loc="left", pad=4)
    box(ax, 0.02, 0.64, 0.40, 0.15, "stems $x_i$", fc=FROZEN_LOSSY, fs=7.5)
    box(ax, 0.02, 0.34, 0.40, 0.15, "stems $y_i$", fc=FROZEN_LOSSY, fs=7.5)
    # A double-headed arrow across this gap collapses into a blob at this figure
    # height, so the positive link is a plain rule and the negative one stays dashed.
    ax.plot([0.22, 0.22], [0.505, 0.625], color=FROZEN, lw=1.4, zorder=3)
    ax.text(0.25, 0.565, "positive", fontsize=7.2, ha="left", va="center", color=FROZEN)
    box(ax, 0.66, 0.34, 0.32, 0.15, "stems $y_j$", fc="white", ec=GREY, fs=7.5)
    arrow(ax, (0.42, 0.415), (0.655, 0.415), style="-|>", ls=(0, (2.4, 1.8)), color=GREY)
    ax.text(0.535, 0.29, "negative", fontsize=7.2, ha="center", va="top", color=GREY)
    ticks(ax, (0.02, 0.42, 0.66, 0.98))
    header(ax, "coherent $=$ sounds at the same time")

    # (d) REMAST: sequential, but symbolic and via one-segment statistics
    ax = axes[3]; blank(ax); timeline(ax)
    ax.set_title("(d) REMAST", fontsize=8.5, loc="left", pad=4)
    box(ax, 0.02, 0.32, 0.42, 0.16, "chords,\nmelody $A$", fc="#f2f2f0", fs=7.2)
    box(ax, 0.56, 0.32, 0.42, 0.16, "chords,\nmelody $B$", fc="#f2f2f0", fs=7.2)
    box(ax, 0.09, 0.62, 0.28, 0.15, "$\\varphi(A)$", fc="white", fs=7.5)
    box(ax, 0.63, 0.62, 0.28, 0.15, "$\\varphi(B)$", fc="white", fs=7.5)
    arrow(ax, (0.23, 0.485), (0.23, 0.615), lw=0.8)
    arrow(ax, (0.77, 0.485), (0.77, 0.615), lw=0.8)
    arrow(ax, (0.375, 0.695), (0.625, 0.695), style="<|-|>", lw=0.9)
    header(ax, "$|\\varphi(B)-\\varphi(A)|$, one-segment statistics")
    ticks(ax, (0.02, 0.44, 0.56, 0.98))

    # (e) C: sequential, from audio, decomposed before it collapses
    ax = axes[4]; blank(ax); timeline(ax)
    ax.set_title("(e) MuRCo (this thesis)", fontsize=8.5, loc="left", pad=4)
    box(ax, 0.02, 0.32, 0.40, 0.16, "audio $A$", fc=FROZEN_LOSSY, fs=7.5)
    box(ax, 0.58, 0.32, 0.40, 0.16, "audio $B$", fc=EDITED, fs=7.5)
    ax.text(0.50, 0.40, "gap", fontsize=6.8, ha="center", va="center", color=GREY)
    arrow(ax, (0.22, 0.485), (0.22, 0.615), lw=0.8)
    arrow(ax, (0.78, 0.485), (0.78, 0.615), lw=0.8)
    box(ax, 0.14, 0.62, 0.72, 0.15, "$H,\\; T,\\; R,\\; S \\;\\rightarrow\\; C$",
        fc="white", fs=7.5)
    header(ax, "coherent $=$ follows on from")
    ticks(ax, (0.02, 0.42, 0.58, 0.98))

    # (f) the one-clip scorers: there is no pair, and that is the entry
    ax = axes[5]; blank(ax); timeline(ax); ticks(ax, (0.02, 0.42, 0.58, 0.98))
    ax.set_title("(f) one-clip scorers", fontsize=8.5, loc="left", pad=4)
    header(ax, "no relation is scored")
    # A is drawn as an outline to show it exists and is not read
    box(ax, 0.02, 0.28, 0.40, 0.15, "audio $A$", fc="white", ec=GREY, fs=7.5)

    box(ax, 0.58, 0.28, 0.40, 0.15, "audio $B$", fc=EDITED, fs=7.5)
    arrow(ax, (0.78, 0.435), (0.78, 0.575), lw=0.8)
    box(ax, 0.52, 0.58, 0.46, 0.14, "quality or\npreference", fc="white", fs=6.8)
    ax.text(0.22, 0.655, "TuneJury\nSongEval\nMuQ-Eval", fontsize=6.8, ha="center",
            va="center", color=GREY, linespacing=1.35)
    note(ax, "$A$ is present but is not an argument")

    fig.tight_layout(w_pad=0.9, h_pad=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_pair_axes.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_pair_axes.pdf/.png")


# ---------------------------------------------------------------------------
# Figure 4: the S read-out protocol, end to end
# ---------------------------------------------------------------------------
def fig_s_protocol(out: Path):
    """The S pipeline as a single signal path. Schematic: nothing here is measured,
    but every constant shown (8 s a side, the 1 s gap, RMS match, peak scaling, the
    Yes/No softmax) is one the ablations in Section 4.11 found load-bearing."""
    fig, ax = plt.subplots(figsize=(6.5, 1.75))
    blank(ax)
    ax.set_ylim(0.16, 1.0)

    # --- the concatenated waveform that the model actually receives ------------
    y, h = 0.62, 0.17
    box(ax, 0.015, y, 0.20, h, "$A$\n8 s", fc=FROZEN, ec="white", fs=7.6)
    box(ax, 0.215, y, 0.05, h, "", fc="#f2f0ec", ec=GREY, lw=0.7)
    box(ax, 0.265, y, 0.20, h, "$B$\n8 s", fc=EDITED, ec="white", fs=7.6)
    ax.text(0.24, y + h + 0.10, "1 s\ngap", fontsize=6.8, ha="center", va="center", color=GREY)
    arrow(ax, (0.24, y + h + 0.02), (0.24, y + h - 0.005), color=GREY, lw=0.7)
    ax.text(0.24, y - 0.085, "$B$ RMS-matched to $A$, concatenation scaled by its peak",
            fontsize=6.8, ha="center", va="center", color=GREY)

    arrow(ax, (0.475, y + h / 2), (0.545, y + h / 2))

    # --- the model, the prompt, and the read-out -------------------------------
    box(ax, 0.55, 0.50, 0.20, 0.41,
        "Music Flamingo\n(frozen)\nencoder fp32\nbackbone bf16", fs=7.2)
    ax.text(0.65, 0.40, "prompt: preamble $+$ one of four\nquestions $+$ ``Answer Yes or No''",
            fontsize=6.8, ha="center", va="top", color=GREY)

    arrow(ax, (0.755, 0.705), (0.815, 0.705))
    box(ax, 0.82, 0.60, 0.165, 0.21,
        "logits\n$\\ell_{\\mathrm{Yes}},\\ \\ell_{\\mathrm{No}}$", fs=7.4)
    ax.text(0.9025, 0.50,
            "$S=\\dfrac{e^{\\ell_{\\mathrm{Yes}}}}{e^{\\ell_{\\mathrm{Yes}}}+e^{\\ell_{\\mathrm{No}}}}\\in[0,1]$",
            fontsize=7.6, ha="center", va="top", color=INK)

    ax.text(0.5, 0.225,
            "one generated token; the answer read from a single Yes and a single No logit",
            fontsize=6.8, ha="center", va="center", color=GREY)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_s_protocol.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/fig_s_protocol.pdf/.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_families(out)
    fig_edit_loops(out)
    fig_pair_axes(out)
    fig_s_protocol(out)


if __name__ == "__main__":
    main()
