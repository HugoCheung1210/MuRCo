#!/usr/bin/env python3
"""Assemble the separability matrix into a dissertation-ready table.

Reads ``separability.json`` (from score_separability.py) and emits the headline
D x P matrix -- dimensions (rows) by perturbation families (columns) -- as
markdown and LaTeX.  Each cell shows the mean drop (control - perturbed) with the
diagonal (target) cell marked, so the story is readable at a glance: strong on the
diagonal, flat off it.

Two views:
  * --metric drop   (default) mean absolute drop, the primary quantity
  * --metric dz     Cohen's d_z, the standardised effect size

Usage::

    python assemble_matrix.py --in results/separability.json --out results/matrix
    python assemble_matrix.py --in results/separability.json --by-genre --metric drop

Writes ``<out>.md`` and ``<out>.tex``; also prints the markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# canonical column order and pretty labels
COL_ORDER = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]
COL_LABEL = {"pitch_shift": "transpose", "time_stretch": "time-stretch",
             "lowpass": "lowpass", "distortion": "distortion", "style_swap": "style-swap"}
ROW_ORDER = ["H", "T", "R", "S"]
ROW_LABEL = {"H": "H (harmonic)", "T": "T (timbral)",
             "R": "R (rhythmic)", "S": "S (structural)"}
TARGET_OF = {"H": {"pitch_shift"}, "R": {"time_stretch"},
             "T": {"lowpass", "distortion"}, "S": {"style_swap"}}

# off-the-shelf single-scalar baselines (from score_baselines.py). They have NO
# target perturbation (a scalar cannot identify *which* edit) -- that's the point.
BASELINE_ORDER = ["clap_htsat", "clap_music", "scs"]
BASELINE_LABEL = {"clap_htsat": "CLAP-htsat", "clap_music": "CLAP-music",
                  "scs": "SCS (stand-in)"}


def cell_value(cells: list[dict], pert: str, metric: str) -> dict | None:
    for c in cells:
        if c["perturbation"] == pert:
            return c
    return None


def fmt(c: dict | None, metric: str, is_target: bool, mark: str) -> str:
    if c is None:
        return "--"
    val = c["mean_drop"] if metric == "drop" else c["cohen_dz"]
    body = f"{val:.3f}" if metric == "drop" else f"{val:.2f}"
    # bold/mark the diagonal so it reads at a glance
    if is_target:
        return f"{mark}{body}{mark}"
    return body


def baseline_family_stats(path: Path) -> dict[str, dict[str, dict | None]]:
    """Per-baseline, per-perturbation mean drop + Cohen's d_z, paired by source.

    Drop = control_score(source) - perturbed_score, so the sign convention matches
    the dimensions (positive = the metric moved on that perturbation).
    """
    import statistics
    scores = json.loads(path.read_text(encoding="utf-8")).get("scores", {})
    ctrl: dict[str, dict[str, float]] = {m: {} for m in BASELINE_ORDER}
    diffs: dict[str, dict[str, list[float]]] = {m: {f: [] for f in COL_ORDER}
                                                for m in BASELINE_ORDER}
    for k, v in scores.items():                         # first pass: control per source
        src, fam, _mag = k.split("::")
        if fam == "control":
            for m in BASELINE_ORDER:
                if m in v:
                    ctrl[m][src] = float(v[m])
    for k, v in scores.items():                         # second pass: paired drops
        src, fam, _mag = k.split("::")
        if fam not in COL_ORDER:
            continue
        for m in BASELINE_ORDER:
            if m in v and src in ctrl[m]:
                diffs[m][fam].append(ctrl[m][src] - float(v[m]))
    out: dict[str, dict[str, dict | None]] = {m: {} for m in BASELINE_ORDER}
    for m in BASELINE_ORDER:
        for fam in COL_ORDER:
            xs = diffs[m][fam]
            if not xs:
                out[m][fam] = None
                continue
            mean = sum(xs) / len(xs)
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            out[m][fam] = {"mean_drop": mean, "cohen_dz": (mean / sd) if sd > 1e-9 else 0.0}
    return out


def build_baseline_body(stats: dict[str, dict[str, dict | None]], metric: str) -> list[list[str]]:
    """Formatted baseline rows (no diagonal marks -- scalars have no target)."""
    body: list[list[str]] = []
    for m in BASELINE_ORDER:
        line = [BASELINE_LABEL[m]]
        for pert in COL_ORDER:
            line.append(fmt(stats[m].get(pert), metric, is_target=False, mark=""))
        body.append(line)
    return body


def build_table(rows: list[dict], metric: str) -> tuple[list[str], list[list[str]]]:
    by_dim = {r["dimension"]: r for r in rows}
    header = ["dimension \\ perturbation"] + [COL_LABEL[c] for c in COL_ORDER]
    body: list[list[str]] = []
    for d in ROW_ORDER:
        if d not in by_dim:
            continue
        cells = by_dim[d]["per_family"]
        line = [ROW_LABEL[d]]
        for pert in COL_ORDER:
            c = cell_value(cells, pert, metric)
            line.append(fmt(c, metric, pert in TARGET_OF.get(d, set()), mark=""))
        body.append(line)
    return header, body


def to_markdown(header: list[str], body: list[list[str]], metric: str,
                baseline_body: list[list[str]] | None = None) -> str:
    # bold the diagonal in markdown
    def mark_row(d: str, cells: list[str]) -> list[str]:
        out = [cells[0]]
        for pert, v in zip(COL_ORDER, cells[1:]):
            out.append(f"**{v}**" if pert in TARGET_OF.get(d, set()) and v != "--" else v)
        return out
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for d, row in zip([r for r in ROW_ORDER if any(b[0] == ROW_LABEL[r] for b in body)], body):
        lines.append("| " + " | ".join(mark_row(d, row)) + " |")
    if baseline_body:
        lines.append("| " + " | ".join(["*— baselines (single scalar) —*"]
                                        + [""] * (len(header) - 1)) + " |")
        for row in baseline_body:                       # baselines: never bolded (no target)
            lines.append("| " + " | ".join(row) + " |")
    cap = "mean absolute drop (control - perturbed)" if metric == "drop" else "Cohen's d_z"
    lines.append("")
    lines.append(f"*Cell = {cap}. Bold = diagonal (each dimension's target perturbation).*")
    if baseline_body:
        lines.append("*Baselines are single scalars with no diagonal: they rank the "
                     "magnitude of change but cannot identify which perturbation "
                     "occurred (e.g. CLAP-htsat conflates lowpass≈distortion, "
                     "transpose≈stretch). Identification is by the H/T/R/S pattern.*")
    return "\n".join(lines)


def to_latex(header: list[str], body: list[list[str]], metric: str,
             baseline_body: list[list[str]] | None = None) -> str:
    ncol = len(header)
    cap = "mean absolute drop (control $-$ perturbed)" if metric == "drop" else "Cohen's $d_z$"
    caption = ("Separability matrix, " + cap +
               r". The dimension each family was designed to target is set in \textbf{bold}.")
    if baseline_body:
        caption += r" Single-scalar baselines are below the rule."
    # Short caption keeps \listoftables scannable (house style).
    short = "The separability matrix"
    out = [r"\begin{table}[t]", r"\centering",
           r"\caption[" + short + r"]{" + caption + r"}",
           r"\label{tab:separability}",
           r"\begin{tabular}{l" + "r" * (ncol - 1) + "}", r"\toprule",
           " & ".join(h.replace("\\", r"\textbackslash ").replace("_", r"\_") for h in header) + r" \\",
           r"\midrule"]
    for d in [r for r in ROW_ORDER]:
        row = next((b for b in body if b[0] == ROW_LABEL[d]), None)
        if row is None:
            continue
        cells = [ROW_LABEL[d].replace("_", r"\_")]
        for pert, v in zip(COL_ORDER, row[1:]):
            cells.append(rf"\textbf{{{v}}}" if pert in TARGET_OF.get(d, set()) and v != "--" else v)
        out.append(" & ".join(cells) + r" \\")
    if baseline_body:
        out.append(r"\midrule")
        for row in baseline_body:
            cells = [row[0].replace("_", r"\_")] + list(row[1:])   # baselines: never bold
            out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/matrix"))
    ap.add_argument("--metric", choices=("drop", "dz"), default="drop")
    ap.add_argument("--baselines", type=Path, default=None,
                    help="baseline_scores.json from score_baselines.py; appends "
                         "CLAP/SCS rows (single-scalar, no diagonal) below H/T/R/S")
    ap.add_argument("--by-genre", action="store_true",
                    help="also emit one matrix per genre (needs a --by-genre score run)")
    args = ap.parse_args()

    doc = json.loads(args.inp.read_text(encoding="utf-8"))
    rows = doc["rows"]

    baseline_body = None
    if args.baselines:
        baseline_body = build_baseline_body(baseline_family_stats(args.baselines), args.metric)

    header, body = build_table(rows, args.metric)
    md = to_markdown(header, body, args.metric, baseline_body)
    tex = to_latex(header, body, args.metric, baseline_body)

    if args.by_genre:
        genres = sorted({g for r in rows for g in (r.get("per_genre") or {})})
        for g in genres:
            grows = [{"dimension": r["dimension"],
                      "per_family": (r.get("per_genre") or {}).get(g, [])} for r in rows]
            h, b = build_table(grows, args.metric)
            md += f"\n\n### genre: {g}\n\n" + to_markdown(h, b, args.metric)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    args.out.with_suffix(".tex").write_text(tex + "\n", encoding="utf-8")
    print(md)
    print(f"\n[wrote {args.out.with_suffix('.md')} and {args.out.with_suffix('.tex')}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
