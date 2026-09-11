"""One font/style contract for every figure in the thesis.

Why this file exists. A `pdffonts` audit on 2026-08-20 found three inconsistencies across
the fourteen figures, none of which is visible while looking at one figure at a time:

  1. make_figures.py set sans-serif (Helvetica) at 9 pt, while fig_architectures.py,
     bestofn_curve.py, fig_grpo.py and fig_grpo_training.py set serif at 8.5 pt. Eight
     figures came out sans and six serif, so half the figures looked like a different
     document from the other half.
  2. Matplotlib's mathtext default is `dejavusans` regardless of `font.family`, so every
     `$...$` label rendered in a SANS italic even inside the serif figures. Any figure
     with an axis label like $d_z$ or a box label like $\\phi(A)$ mixed two typefaces
     inside one line of text.
  3. Neither family matched the thesis body, which is Computer Modern (the LaTeX default
     for the UCL template).

PRESET picks one contract for all of them. "cm" is the default because it fixes all three
at once: cmr10 is the text half of Computer Modern and ships with matplotlib, and the "cm"
mathtext fontset is the matching math half, so figure text, figure maths and body text are
finally the same typeface.

  cm     Computer Modern. Matches doc/writeup/dissertation.tex exactly.
  stix   STIX Two Text + stix maths. Times-like, wider glyph coverage than cmr10,
         and the closest match if the thesis is ever reset in a Times-based class.
  sans   Helvetica/Arial + stixsans maths. The Nature/Elsevier figure convention,
         where figures deliberately contrast with a serif body.

cmr10 carries no U+2212, so `axes.unicode_minus` is switched off for it; without that
every negative tick label silently falls back to DejaVu and reintroduces problem 2 on the
axes. cmr10 also has no true bold, so bold in a figure resolves to the same weight rather
than failing.

Usage, as the first thing a figure script does after importing pyplot:

    from _figstyle import apply_style, INK, GREY, FROZEN, EDITED, ACCUM
    apply_style()                 # or apply_style("sans")
"""
from __future__ import annotations

import matplotlib as mpl

PRESET = "cm"

# Colour is assigned by entity, not by series index, and is shared by every figure:
# blue is content that was preserved, gold is content a generator produced.
INK = "#0b0b0b"
INK2 = "#3d3d3d"
GREY = "#898781"
GRID = "#e4e3dd"
FROZEN = "#2a78d6"
FROZEN_LOSSY = "#a9c9ee"
EDITED = "#c98500"
ACCUM = "#d55181"

_FONTS = {
    "cm": {
        "font.family": "serif",
        "font.serif": ["cmr10", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        # cmr10 has no U+2212; leaving this True sends every minus sign to a fallback font.
        "axes.unicode_minus": False,
    },
    "stix": {
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,
    },
    "sans": {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "axes.unicode_minus": True,
    },
}

# Everything that is not the font, and that every figure agreed on already.
_COMMON = {
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    # cmr10 asks for this: it routes numeric tick labels through mathtext so they are
    # set in the same Computer Modern as everything else instead of the text fallback.
    "axes.formatter.use_mathtext": True,
    # Type 42 keeps text selectable and searchable in the compiled PDF rather than
    # shipping it as outlines, which is what most publishers ask for.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style(preset: str | None = None, **overrides) -> str:
    """Apply the shared contract. Returns the preset name actually used."""
    name = preset or PRESET
    if name not in _FONTS:
        raise ValueError(f"unknown preset {name!r}; pick one of {sorted(_FONTS)}")
    mpl.rcParams.update(_COMMON)
    mpl.rcParams.update(_FONTS[name])
    if overrides:
        mpl.rcParams.update(overrides)
    return name
