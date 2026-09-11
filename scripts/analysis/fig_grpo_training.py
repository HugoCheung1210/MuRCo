#!/usr/bin/env python3
"""Figure for section 4.6: why the run is reported at step 150 rather than at 300.

The stopping argument is currently prose. It rests on two things a reader cannot check from a
sentence: held-out C peaks at the reported checkpoint and falls back by step 300, and the
spread of rewards inside each GRPO group narrows until the group-relative advantage of
Equation (4.2) has little left to normalise.

LEFT is the HELD-OUT evaluation (results/rl/eval_pilot2/pair_scores.csv), four checkpoints x
10 contexts x 8 rollouts. Its frozen and step0150 values reproduce the N=1 and N=8 rows of
Table 4.5 exactly, which is the check that this panel and that table are the same experiment.
It deliberately does NOT plot C on training rollouts: that trace wanders inside its own noise
for all 300 steps and shows no plateau, so drawing it under a stopping argument would claim
support the log does not give.

RIGHT is the training log (results/rl/grpo_pilot2/train_log.jsonl). Two contexts per step
makes it very noisy, so the line is a centred rolling mean with the raw trace faint behind it.

Deliberately NOT plotted: the copy metrics. The claim that copying rises is made against the
FROZEN model on held-out contexts at evaluation, which is a different and stricter comparison
than the training-log trace, and mixing the two would invite a reader to read the training
trace as the evidence for it.

    python scripts/analysis/fig_grpo_training.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, ACCENT = "#0b0b0b", "#8a8a84", "#c98500"
REPORTED_STEP = 150


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / ".projectroot").exists() or (p / ".git").exists():
            return p
    return Path.cwd()


def per_step(recs, path):
    out = []
    for rec in recs:
        vals = []
        for g in rec["groups"]:
            d = g
            for k in path:
                d = d[k]
            vals.append(d)
        out.append(float(np.mean(vals)))
    return np.array(out)


def roll(a, w=25):
    if len(a) < w:
        return a
    k = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(a, (pad, pad), mode="edge"), k, mode="valid")[: len(a)]


def heldout(root: Path):
    """Held-out C per checkpoint: mean over rollouts (C@1) and best-of-8, per context."""
    rows = list(csv.DictReader((root / "results/rl/eval_pilot2/pair_scores.csv").open()))
    by = defaultdict(lambda: defaultdict(list))
    for x in rows:
        c = float(np.prod([max(1e-9, float(x["score_" + k])) for k in "HTRS"]) ** 0.25)
        by[x["perturbation"]][x["source_id"]].append(c)
    arms = ["frozen", "step0075", "step0150", "step0300"]
    c1 = [np.mean([np.mean(v) for v in by[a].values()]) for a in arms]
    b8 = [np.mean([np.max(v) for v in by[a].values()]) for a in arms]
    return arms, np.array(c1), np.array(b8)


def main() -> int:
    root = repo_root()
    recs = [json.loads(l) for l in (root / "results/rl/grpo_pilot2/train_log.jsonl").open()]
    step = np.array([r["step"] for r in recs])
    sd = per_step(recs, ["reward", "sd"])
    arms, c1, b8 = heldout(root)
    x = [0, 75, 150, 300]

    from _figstyle import apply_style
    apply_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 2.6))

    # left: held-out C by checkpoint
    ax0.plot(x, b8, marker="D", color=ACCENT, lw=1.8, ms=5.5, markerfacecolor="white",
             markeredgewidth=1.4, label="best-of-8")
    ax0.plot(x, c1, marker="o", color=INK, lw=1.8, ms=5.5, markerfacecolor="white",
             markeredgewidth=1.4, label="single sample")
    ax0.axvline(REPORTED_STEP, color=MUTED, lw=1.1, ls="--")
    ax0.annotate("reported", xy=(REPORTED_STEP, c1[2]), xytext=(REPORTED_STEP + 14, c1[2] - 0.011),
                 fontsize=7, color=MUTED)
    ax0.set_xticks(x, ["frozen", "75", "150", "300"])
    ax0.set_xlabel("checkpoint (training step)")
    ax0.set_ylabel("held-out $C$")
    ax0.set_title("Held-out $C$ peaks, then falls back", fontsize=8.5)
    ax0.legend(fontsize=7.5, loc="lower right", frameon=False)
    ax0.grid(axis="y", color="#e8e8e4", lw=0.7)

    # right: within-group reward spread
    ax1.plot(step, sd, color=MUTED, lw=0.6, alpha=0.45)
    ax1.plot(step, roll(sd), color=INK, lw=1.9)
    ax1.axvline(REPORTED_STEP, color=MUTED, lw=1.1, ls="--")
    for lo, hi in [(1, 50), (251, 300)]:
        m = sd[(step >= lo) & (step <= hi)].mean()
        ax1.plot([lo, hi], [m, m], color=ACCENT, lw=2.4, solid_capstyle="butt")
        ax1.text((lo + hi) / 2, m + 0.005, f"{m:.3f}", fontsize=7, color=ACCENT, ha="center")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("within-group reward sd")
    ax1.set_title("as the group runs out of ranking signal", fontsize=8.5)
    ax1.grid(color="#e8e8e4", lw=0.7)

    for ax in (ax0, ax1):
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig.tight_layout()
    out = root / "figures" / "fig_grpo_training"
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight")
    print(f"wrote {out}.pdf/.png")
    for a, u, v in zip(arms, c1, b8):
        print(f"  {a:9s} C@1 {u:.4f}  best-of-8 {v:.4f}")
    print(f"  reward sd: 1-50 {sd[:50].mean():.4f} -> 251-300 {sd[250:].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
