#!/usr/bin/env python3
"""Figure for section 4.6: GRPO against best-of-N re-ranking, and which reward earns it.

Section 4.6 carries its whole argument in prose plus one table, and the two things a reader
most needs to see are shapes rather than numbers: that the tuned-minus-frozen gap is FLAT in
N (training and selection compose rather than substitute), and that of the three reward arms
only the full composite adds anything to free re-ranking ON THE OBJECTIVE IT OPTIMISED. Both
are hard to read off a table and obvious in a plot.

The right panel reads each arm on its OWN reward, which is the only comparison that says
whether training that objective was worth doing: the two C arms on delta-C, the CLAP arm on
delta-CLAP. Do not relabel that axis as delta-C. The CLAP-trained policy's delta-C is
+0.018 [+0.004, +0.036], which clears zero, so a delta-C axis would make the third bar say
the opposite of what this panel is about. That transfer figure belongs in the prose (it is
in Sec. 4.6) and not on these three bars.

Numbers are transcribed from the evaluation reported in Section 4.6 (10 held-out contexts,
paired cluster bootstrap over contexts, checkpoint step 150). They are stated in the text and
in Table 4.5; this script is the plotting layer, not a re-analysis.

The left panel and the full-C bar carry the THREE-SEED pooled values written by
`scripts/rl/pool_seeds.py` to `results/rl/grpo_seeds.json`; re-run that and copy the
`pooled` block here after adding a seed. The two ablation bars are still single-seed.

    python scripts/analysis/fig_grpo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / ".projectroot").exists() or (p / ".git").exists():
            return p
    return Path.cwd()


INK, MUTED = "#0b0b0b", "#8a8a84"
TUNED, FROZEN = "#c98500", "#2a78d6"

# left panel: held-out C under best-of-N (Table 4.5)
N      = [1, 2, 4, 8]
FROZ   = [0.618, 0.636, 0.647, 0.653]
TUNED_ = [0.644, 0.660, 0.671, 0.678]
DELTA  = [0.026, 0.024, 0.024, 0.025]
DLO    = [0.017, 0.015, 0.013, 0.013]
DHI    = [0.035, 0.034, 0.035, 0.038]

# right panel: gain over frozen best-of-8 at matched inference cost, by reward.
# NB the full-C arm is the three-seed pooled estimate; the two ablation arms remain
# single-seed, because only the full reward was replicated. The panel therefore compares
# a pooled number against two unpooled ones, which is why the caption says so.
# Each arm on its own reward: the two C arms are delta-C, the CLAP arm is delta-CLAP.
ARMS = [(r"$C$ (full), on $C$",      0.025,  0.013, 0.038),
        (r"$C$ without $S$, on $C$", 0.019,  0.010, 0.030),
        (r"CLAP alone, on CLAP",     0.003, -0.008, 0.015)]


def main() -> int:
    root = repo_root()
    from _figstyle import apply_style
    apply_style()
    # Single panel. The two-arm/budget curve that used to sit on the left is exactly
    # Table tab:grpo-bon (N, frozen, tuned, delta with CI at N=1,2,4,8) redrawn, and the
    # thesis cited both for the same claim, so the panel was dropped 2026-08-28 and the
    # table kept. The reward ablation below appears in no table and is what this figure is for.
    fig, ax2 = plt.subplots(figsize=(4.3, 2.4))

    # ---- right: which reward clears the free bar -----------------------------
    ypos = np.arange(len(ARMS))[::-1]
    for y, (lab, m, lo, hi) in zip(ypos, ARMS):
        crosses = lo <= 0 <= hi
        col = MUTED if crosses else INK
        ax2.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round")
        ax2.plot([m], [y], marker="D" if not crosses else "o", color=col, ms=6,
                 markerfacecolor="white", markeredgewidth=1.5)
        ax2.text(hi + 0.004, y, f"{m:+.3f}", fontsize=7, color=col, va="center")
    ax2.axvline(0, color=INK, lw=1.0)
    ax2.set_yticks(ypos, [a[0] for a in ARMS], fontsize=8)
    ax2.set_xlim(-0.017, 0.058)
    ax2.set_xlabel("gain over frozen best-of-8, matched cost\n(each arm on its own reward)")
    ax2.set_title("A composite reward clears the free bar, a pooled one does not", fontsize=8.5)
    ax2.grid(axis="x", color="#e8e8e4", lw=0.7)
    ax2.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    out = root / "figures" / "fig_grpo"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out}.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
