#!/usr/bin/env python
"""Compare a RIVAL metric's response on the perturbation matrix against C's own dimensions.

Answers the question Chapter 2 currently settles by argument: is the rival an applicable
baseline on these pairs, or is it uninformative on them?

Written for COCOLA and still COCOLA by default, so its published output reproduces
unchanged. `--rival tunejury` / `--rival songeval` point it at the CSV that
`score_rivals.py --csv` writes; the rival's columns are auto-detected by name prefix.

Reports, per perturbation family:
  * mean COCOLA score, and the drop from the control condition (control - perturbed),
    which is the same statistic the separability matrix uses for H/T/R/S;
  * the same for COCOLA's harmonic and percussive channels, which are the closest thing
    in prior work to C's H and R, so this is the sharpest test of the decomposition claim;
  * Cohen's d_z for the paired control-vs-perturbed contrast, per family.

Run in the DSP env (needs only numpy). Reads the CSV that score_cocola.py writes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

DIMS = ["score_H", "score_T", "score_R", "score_S"]  # column names in pair_scores.csv

# Per-rival defaults: where score_*.py writes the matrix CSV and the rerank CSV.
RIVALS = {
    "cocola": ("results/baselines/cocola_scores.csv",
               "results/baselines/cocola_rerank_ace.csv"),
    "tunejury": ("results/baselines/rival_scores.csv",
                 "results/baselines/rival_scores_rerank.csv"),
    "songeval": ("results/baselines/rival_scores.csv",
                 "results/baselines/rival_scores_rerank.csv"),
}


def rival_cols(row: dict, rival: str, override: str = "") -> list[str]:
    """The rival's columns in a scored CSV, by name prefix. SongEval emits 5 dimensions
    x 4 readings, which is unreadable as a table, so when a rival has more than six
    columns only the _cat and _rel readings are shown unless --cols says otherwise."""
    if override:
        return [c.strip() for c in override.split(",") if c.strip()]
    cols = [c for c in row if c.startswith(rival)]
    if len(cols) > 6:
        cols = [c for c in cols if c.endswith(("_cat", "_rel"))]
    if not cols:
        raise SystemExit(f"no columns starting with {rival!r} in the CSV; pass --cols")
    return cols


def primary_col(cols: list[str], rival: str) -> str:
    """The single column used as the ranker and in the head-to-head. `_rel` (the
    junction-isolating contrast) for the absolute rivals; the bare metric for COCOLA."""
    for c in (f"{rival}_rel", f"{rival}_coherence_rel", rival):
        if c in cols:
            return c
    return cols[0]


def read_csv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def dz(paired: list[tuple[float, float]]) -> float:
    """Cohen's d_z on the paired control-minus-perturbed differences."""
    d = np.array([a - b for a, b in paired], dtype=float)
    if len(d) < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rival", choices=sorted(RIVALS), default="cocola")
    ap.add_argument("--cocola", "--rival-csv", dest="rival_csv", type=Path, default=None,
                    help="per-pair rival CSV (default: the --rival preset)")
    ap.add_argument("--cols", default="", help="comma list of rival columns to report")
    ap.add_argument("--pair-scores", type=Path,
                    default=ROOT / "results/coherence/pair_scores.csv",
                    help="C's own per-pair H/T/R/S, for the side-by-side")
    ap.add_argument("--rerank-cocola", "--rerank-rival", dest="rerank_rival", type=Path,
                    default=None,
                    help="the rival on the rerank candidates; enables the re-ranker section")
    ap.add_argument("--rerank-scores", type=Path,
                    default=ROOT / "results/rerank_ace/pair_scores.csv")
    ap.add_argument("--rerank-baselines", type=Path,
                    default=ROOT / "results/rerank_ace/baseline_scores_rerank.json")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write every printed number here, so the figures quoted in "
                         "the write-up have a file behind them instead of script stdout")
    args = ap.parse_args()
    csv_default, rerank_default = RIVALS[args.rival]
    if args.rival_csv is None:
        args.rival_csv = ROOT / csv_default
    if args.rerank_rival is None:
        args.rerank_rival = ROOT / rerank_default

    rows = read_csv(args.rival_csv)
    if not rows:
        raise SystemExit(f"no rows in {args.rival_csv}")
    cols = rival_cols(rows[0], args.rival, args.cols)
    primary = primary_col(cols, args.rival)
    label = args.rival.upper()
    print(f"{label} rows: {len(rows)}   columns: {cols}   primary: {primary}")
    out: dict = {"meta": {"rival": args.rival, "primary_col": primary, "cols": cols,
                          "n_rows": len(rows), "rival_csv": str(args.rival_csv),
                          "pair_scores": str(args.pair_scores)},
                 "matrix": {}, "dims": {}}

    # control score per source, for the paired contrast
    ctrl: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["perturbation"] == "control":
            ctrl[r["source_id"]] = {c: float(r[c]) for c in cols}

    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_fam[r["perturbation"]].append(r)

    print(f"\n=== {label} on the perturbation matrix ===")
    print("(drop = control - perturbed, matching the separability matrix convention;")
    print(" a metric that cannot see a perturbation has drop ~ 0)\n")
    hdr = f"{'perturbation':16s} {'n':>4s} "
    for c in cols:
        hdr += f"{c+'_mean':>14s} {c+'_drop':>13s} {c+'_dz':>11s} "
    print(hdr)
    print("-" * len(hdr))

    order = ["control", "pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]
    for fam in order:
        rs = by_fam.get(fam, [])
        if not rs:
            continue
        line = f"{fam:16s} {len(rs):4d} "
        for c in cols:
            vals = [float(r[c]) for r in rs]
            paired = [(ctrl[r["source_id"]][c], float(r[c]))
                      for r in rs if r["source_id"] in ctrl]
            drop = float(np.mean([a - b for a, b in paired])) if paired else float("nan")
            line += f"{np.mean(vals):14.3f} {drop:+13.3f} {dz(paired):11.2f} "
            out["matrix"].setdefault(fam, {})[c] = {
                "n": len(rs), "mean": round(float(np.mean(vals)), 4),
                "drop": round(drop, 4), "dz": round(dz(paired), 4)}
        print(line)

    # ---- side-by-side with C's own dimensions ------------------------------------
    if args.pair_scores.exists():
        ps = read_csv(args.pair_scores)
        have = [d for d in DIMS if d in (ps[0] if ps else {})]
        pctrl: dict[str, dict[str, float]] = {}
        pfam: dict[str, list[dict]] = defaultdict(list)
        for r in ps:
            pert = r.get("perturbation") or r.get("family") or ""
            src = r.get("source_id") or r.get("pair_id", "").split("::")[0]
            r["_src"], r["_fam"] = src, pert
            if pert == "control":
                pctrl[src] = {d: float(r[d]) for d in have if r.get(d) not in (None, "")}
            pfam[pert].append(r)

        print("\n=== C's own dimensions, same statistic (for comparison) ===\n")
        hdr2 = f"{'perturbation':16s} " + "".join(f"{d.replace('score_','')+'_drop':>11s}"
                                                  for d in have)
        print(hdr2)
        print("-" * len(hdr2))
        for fam in order:
            rs = pfam.get(fam, [])
            if not rs or fam == "control":
                continue
            line = f"{fam:16s} "
            for d in have:
                paired = [(pctrl[r["_src"]][d], float(r[d]))
                          for r in rs
                          if r["_src"] in pctrl and r.get(d) not in (None, "")
                          and d in pctrl[r["_src"]]]
                drop = float(np.mean([a - b for a, b in paired])) if paired else float("nan")
                line += f"{drop:+11.3f}"
                out["dims"].setdefault(fam, {})[d.replace("score_", "")] = round(drop, 4)
            print(line)

    print(f"\nReading: a family where {label}'s drop is ~0 while C's dimensions move is a")
    print(f"family {label} cannot see on these pairs.", end=" ")
    if args.rival == "cocola":
        print("Compare cocola_h against H (pitch)\nand cocola_p against R (time-stretch) "
              "for the decomposition claim.")
    else:
        print("The _b columns are the absolute reading (B alone, what the model was "
              "trained for);\nthe _rel columns are the relational one (cat - mean of the "
              "parts). A rival whose _b moves\nbut whose _rel does not is measuring clip "
              "quality, not the junction.")

    if args.rerank_rival and args.rerank_rival.exists():
        out["rerank"] = rerank_report(args, cols, primary, label)
    else:
        print(f"\n(no rerank CSV at {args.rerank_rival}; skipping the re-ranker section)")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\n[wrote {args.json}]")


def rerank_report(args, cols: list[str], primary: str, label: str) -> dict:
    """Is the rival a useful *selection* criterion? Rank candidates by each metric and read
    the winner's score on a judge that was NOT used to rank (the E-D design).

    Returns every printed number so `--json` can persist it. These figures are quoted in the
    write-up and had no file behind them before.
    """
    coc = {r["pair_id"]: r for r in read_csv(args.rerank_rival)}
    ps = {r["pair_id"]: r for r in read_csv(args.rerank_scores)}
    bl = json.loads(args.rerank_baselines.read_text())
    bl = bl.get("scores", bl)

    seeds: dict[str, list[str]] = defaultdict(list)
    for pid in ps:
        if pid in coc:
            seeds[pid.split("::")[0]].append(pid)
    seeds = {k: sorted(v) for k, v in seeds.items() if len(v) >= 4}
    if not seeds:
        return {}

    def geo_C(p: str) -> float:
        v = [float(ps[p][f"score_{d}"]) for d in "HTRS"]
        return float(np.prod(v) ** (1.0 / len(v)))

    def get(p: str, m: str) -> float:
        if m == "C":
            return geo_C(p)
        if m in cols or m.startswith(args.rival):
            return float(coc[p][m])
        if m.startswith("score_"):
            return float(ps[p][m])
        return float(bl.get(p, {}).get(m, float("nan")))

    def spearman(a, b) -> float:
        a, b = np.asarray(a, float), np.asarray(b, float)
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        if ra.std() == 0 or rb.std() == 0:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"\n\n=== {label} as a re-ranker on {primary} ({len(seeds)} seeds, "
          f"{np.mean([len(v) for v in seeds.values()]):.0f} candidates each) ===\n")
    out: dict = {"n_seeds": len(seeds),
                 "n_candidates_mean": float(np.mean([len(v) for v in seeds.values()])),
                 "rank_agreement": {}, "heldout_S": {}, "per_judge": {}}
    print("rank agreement (mean Spearman over seeds); near 0 = a distinct signal")
    for m in ["C", "clap_htsat", "clap_music", "scs", "score_S"]:
        rs = [spearman([get(p, primary) for p in v], [get(p, m) for p in v])
              for v in seeds.values()]
        rs = [r for r in rs if not math.isnan(r)]
        out["rank_agreement"][m] = round(float(np.mean(rs)), 4)
        print(f"    {primary} vs {m:12s} rho = {np.mean(rs):+.3f}")

    rng = np.random.default_rng(0)

    def boot(d, n=20000):
        idx = rng.integers(len(d), size=(n, len(d)))
        return np.percentile(d[idx].mean(axis=1), [2.5, 97.5])

    def winners(ranker: str, judge: str) -> np.ndarray:
        return np.array([get(max(v, key=lambda p: get(p, ranker)), judge)
                         for v in seeds.values()])

    def expected_random(judge: str) -> np.ndarray:
        """Exact expectation of picking at random: the per-seed mean over candidates.
        Using the expectation rather than one sampled draw removes noise from the
        baseline arm, which makes a real difference easier to detect, not harder."""
        return np.array([np.mean([get(p, judge) for p in v]) for v in seeds.values()])

    # The headline contrast, on the judge with headroom.
    print("\nwinner's held-out S (S was not used to rank); paired over seeds")
    perS = {rk: winners(rk, "score_S") for rk in ["C", primary, "clap_htsat"]}
    perS["random"] = expected_random("score_S")
    print(f"    {'contrast':26s}{'mean diff':>11s}{'95% CI':>24s}{'d_z':>8s}")
    for a, b in [("C", "random"), ("C", primary), (primary, "random"),
                 ("clap_htsat", "random")]:
        d = perS[a] - perS[b]
        lo, hi = boot(d)
        star = "  *" if (lo > 0 or hi < 0) else ""
        out["heldout_S"][f"{a} - {b}"] = {
            "mean_diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "d_z": round(float(d.mean() / d.std(ddof=1)), 4), "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"    {a + ' - ' + b:26s}{d.mean():+11.4f}   "
              f"[{lo:+.4f}, {hi:+.4f}]{d.mean() / d.std(ddof=1):+8.2f}{star}")

    # Does COCOLA's preference correspond to ANY other judge, or only to itself?
    # Its own score is the sanity check: if that one is not significant, the selection
    # machinery is broken rather than the metric being uninformative.
    print(f"\nrank-by-{primary} vs expected-random, per judge "
          f"(the rival judged on itself is the sanity check)")
    print(f"    {'judge':16s}{'rival-rand':>13s}{'95% CI':>24s}{'d_z':>7s}   {'C-rand':>9s}")
    for j in ["score_S", "score_H", "score_T", "score_R", "C",
              "clap_htsat", "clap_music", "scs", *cols]:
        rnd = expected_random(j)
        d = winners(primary, j) - rnd
        lo, hi = boot(d)
        star = " *" if (lo > 0 or hi < 0) else ""
        out["per_judge"][j] = {
            "rival_minus_random": round(float(d.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "d_z": round(float(d.mean() / d.std(ddof=1)), 4),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "C_minus_random": round(float((winners("C", j) - rnd).mean()), 4)}
        print(f"    {j:16s}{d.mean():+13.4f}   [{lo:+.4f}, {hi:+.4f}]"
              f"{d.mean() / d.std(ddof=1):+7.2f}{star:2s} "
              f"{(winners('C', j) - rnd).mean():+9.4f}")
    print("    * = 95% CI excludes zero")
    return out


if __name__ == "__main__":
    main()
