#!/usr/bin/env python
"""Read a GRPO train_log.jsonl: the last N steps, or block means over the whole run.

The console line the trainer prints is a lossy summary and scrolls away; this reads the
per-step JSONL that has everything, including H/T/R/S per context.

    python rl/tail_log.py <run>/train_log.jsonl              # last 20 steps
    python rl/tail_log.py <run>/train_log.jsonl -n 50        # last 50
    python rl/tail_log.py <run>/train_log.jsonl --blocks     # block means, whole run
    python rl/tail_log.py <run>/train_log.jsonl --blocks \
        --vs results/rl/grpo_pilot2/train_log.jsonl          # against the C arm

Reads dimension keys off each record rather than assuming H/T/R/S, so it works for the
`clap` arm (H/T/R/clap_htsat, no S) as well as the C and C_noS arms.

`rel` is each context's reward against its own *first visit*, so it is 0 by construction
until a context recurs. With 40 contexts at 2/step it is uninformative before ~step 20;
judge learning by the block table, not by the first screenful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BLOCKS = [(1, 10), (11, 25), (26, 50), (51, 75), (76, 100),
          (101, 150), (151, 200), (201, 250), (251, 300)]
# printed in this order when present; anything else in dims is appended alphabetically
DIM_ORDER = ["H", "T", "R", "S", "clap_htsat"]


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:      # a run killed mid-write leaves a torn line
                pass
    if not rows:
        raise SystemExit(f"no complete step records in {path}")
    return rows


def dim_keys(rows: list[dict]) -> list[str]:
    seen = set()
    for r in rows:
        seen.update(r["groups"][0]["dims"])
    return ([k for k in DIM_ORDER if k in seen]
            + sorted(k for k in seen if k not in DIM_ORDER))


def g_mean(rows: list[dict], f) -> float:
    """Mean of f over every group of every step (2 groups/step at --contexts-per-step 2)."""
    vals = [f(x) for r in rows for x in r["groups"]]
    return float(np.mean(vals)) if vals else float("nan")


def dim(rows: list[dict], k: str) -> float:
    return g_mean(rows, lambda x: x["dims"].get(k, np.nan))


def header(dims: list[str], first: str = "step") -> str:
    cols = f"{first:>7}{'rel':>9}{'rew sd':>8}{'C':>8}"
    cols += "".join(f"{k[:5]:>7}" for k in dims)
    return cols + f"{'copy':>7}{'KL':>8}{'|g|':>6}"


def row(label: str, rows: list[dict], dims: list[str], secs: bool = False) -> str:
    out = (f"{label:>7}{g_mean(rows, lambda x: x['reward']['rel']):>+9.4f}"
           f"{g_mean(rows, lambda x: x['reward']['sd']):>8.4f}"
           f"{g_mean(rows, lambda x: x['C']):>8.4f}")
    out += "".join(f"{dim(rows, k):>7.3f}" for k in dims)
    out += (f"{g_mean(rows, lambda x: x['copy']['xcorr_wave']):>7.3f}"
            f"{g_mean(rows, lambda x: x['kl']):>8.4f}"
            f"{np.mean([r['grad_norm'] for r in rows]):>6.2f}")
    if secs:
        out += f"{np.mean([r.get('t_step_s', np.nan) for r in rows]):>6.0f}s"
    return out


def health(rows: list[dict]) -> None:
    """The four things that distinguish a healthy loop from a dead or diverging one."""
    sd = g_mean(rows, lambda x: x["reward"]["sd"])
    kl = g_mean(rows, lambda x: x["kl"])
    gn = float(np.mean([r["grad_norm"] for r in rows]))
    cp = g_mean(rows, lambda x: x["copy"]["xcorr_wave"])
    last = rows[-1]["step"]

    print(f"\nhealth at step {last}")
    def say(ok, label, val, want):
        print(f"  [{'ok' if ok else '!!'}] {label:<22}{val:>8.4f}   {want}")
    say(sd > 0.02, "within-group reward sd", sd, "> 0.02, else advantages are ~0")
    say(kl < 0.5, "KL", kl, "< 0.5; pilot 1 diverged at 1.21")
    say(gn < 3.0, "grad norm", gn, "~0.4-0.9; spikes > 5 are divergence")
    say(cp < 0.15, "copy xcorr_wave", cp, "~0.06-0.10 and flat")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path)
    ap.add_argument("-n", type=int, default=20, help="last N steps (default 20)")
    ap.add_argument("--blocks", action="store_true",
                    help="block means over the whole run instead of the last N steps")
    ap.add_argument("--vs", type=Path, default=None,
                    help="a second run to print beside this one, block means")
    a = ap.parse_args()

    rows = load(a.log)
    dims = dim_keys(rows)

    if a.blocks or a.vs:
        runs = [(a.log.parent.name or str(a.log), rows)]
        if a.vs:
            other = load(a.vs)
            runs.append((a.vs.parent.name or str(a.vs), other))
            dims = [k for k in dim_keys(rows + other)]
        for name, rs in runs:
            print(f"\n{name}  ({len(rs)} steps)")
            print(header(dims, "steps"))
            for lo, hi in BLOCKS:
                sel = [r for r in rs if lo <= r["step"] <= hi]
                if sel:
                    print(row(f"{lo}-{hi}", sel, dims))
    else:
        sel = rows[-a.n:]
        print(f"{a.log}  last {len(sel)} of {len(rows)} steps")
        print(header(dims) + f"{'t':>7}")
        for r in sel:
            print(row(str(r["step"]), [r], dims, secs=True))
        print(header(dims, "mean"))
        print(row("", sel, dims))

    health(rows)


if __name__ == "__main__":
    main()
