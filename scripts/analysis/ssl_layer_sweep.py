#!/usr/bin/env python3
"""Which layer should MuQ and MERT be read at? A screen, so the baseline gets its best.

The A4 claim is that two masked-prediction music encoders diagnose the edit family worse
than a CLAP embedding does. Layer and temporal pooling are the two choices the
meta-evaluation literature reports as dominant for embedding-based audio metrics, and the
shipped numbers read both encoders at layer 6, which is SongEval's choice for MuQ and was
carried over for MERT rather than derived. An unswept layer on the losing side is not a
result, it is a hyperparameter, and it is the first thing a reader who knows that
literature will ask about.

So sweep it, and hand the win to the baseline: pick each encoder's BEST layer on the folds
being scored. That is exactly the selection this repo elsewhere refuses to do, and it is
deliberate here, because the claim only gets stronger if the profile still leads after the
rival has been tuned on the test set. Report it as such.

Cost is why this is a screen and not the nested protocol. A single 1024-d feature set
through `diagnose_nested_cv.py` costs ~20 min, so eight of them is most of an afternoon.
Here the L2 strength is fixed and only leave-one-track-out is run, which is minutes per
layer. Use it to choose the layer, then put the winner through `diagnose_nested_cv.py` for
the number that gets reported.

Usage:
    # GPU box, rivals venv: extract each layer (~2 min each)
    for L in 3 6 9 12 18 24; do
      ~/rivals/venv/bin/python ssl_probe.py extract --encoder mert --layer $L \
          --manifest /root/autodl-tmp/perturbations/pairs_manifest.json \
          --audio-root /root/autodl-tmp \
          --out /root/autodl-tmp/mert_L$L.npz
    done

    # DSP env: screen them
    python scripts/analysis/ssl_layer_sweep.py --encoder mert --npz 'results/baselines/mert_L*.npz'

MuQ-large exposes fewer layers than MERT-v1-330M; a layer past the end of the stack raises
in `ssl_probe.py extract` rather than silently clamping, so an over-long list simply fails
that arm and leaves the rest.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

FAMILIES = ["pitch_shift", "time_stretch", "lowpass", "distortion", "style_swap"]


def repo_root() -> Path:
    for p in Path(__file__).resolve().parents:
        if (p / ".projectroot").exists() or (p / ".git").exists():
            return p
    return Path.cwd()


ROOT = repo_root()


def loto_fixed(X, y, g, C):
    """Leave-one-track-out with the L2 strength fixed. A screen, not the protocol."""
    pred = np.empty_like(y)
    for tr in np.unique(g):
        te = g == tr
        Xtr = X[~te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit((Xtr - mu) / sd, y[~te])
        pred[te] = m.predict((X[te] - mu) / sd)
    return pred


def loto_nested(X, y, g, grid=(0.02, 0.3, 1.0, 10.0)):
    """The reportable protocol, copied from diagnose_nested_cv.loto so the numbers this
    writes are directly comparable with Table 2 instead of only with each other. Cheap on a
    1-d readout, which is why the cosine sweep can use it where the 1024-d one cannot."""
    pred = np.empty_like(y)
    for tr in np.unique(g):
        te = g == tr
        Xtr, ytr, gtr = X[~te], y[~te], g[~te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Z, Zt = (Xtr - mu) / sd, (X[te] - mu) / sd
        best, bs = None, -np.inf
        for c in grid:
            accs = []
            for i, j in GroupKFold(n_splits=3).split(Z, ytr, gtr):
                m = LogisticRegression(C=c, max_iter=300, tol=1e-3)
                m.fit(Z[i], ytr[i])
                accs.append((m.predict(Z[j]) == ytr[j]).mean())
            s = float(np.mean(accs))
            if s > bs:
                best, bs = c, s
        pred[te] = LogisticRegression(C=best, max_iter=5000).fit(Z, ytr).predict(Zt)
    return pred


def macro_f1(y, pred):
    fs = []
    for c in range(len(FAMILIES)):
        tp = ((pred == c) & (y == c)).sum()
        p = tp / max((pred == c).sum(), 1)
        r = tp / max((y == c).sum(), 1)
        fs.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(fs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--npz", required=True,
                    help="glob for the per-layer npz files, e.g. 'results/baselines/mert_L*.npz'")
    ap.add_argument("--C", type=float, default=1.0, help="fixed L2 strength for the screen")
    ap.add_argument("--readout", choices=("diff", "cosine"), default="diff",
                    help="diff: the 1024-d e_B - e_A feature (default). cosine: the 1-d "
                         "A-vs-B cosine, built exactly as review_fixes.ssl_features does")
    ap.add_argument("--nested", action="store_true",
                    help="use the reportable nested-CV protocol instead of the fixed-C "
                         "screen. Affordable on --readout cosine, not on diff")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(ROOT / "results/coherence/pair_scores_cbase.csv"))
            if r["perturbation"] in FAMILIES]
    man = json.load(open(ROOT / "perturbations/pairs_manifest.json"))
    pairs = man["pairs"] if isinstance(man, dict) else man
    paths = {p["pair_id"]: (Path(p["ref_path"]).name, Path(p["cand_path"]).name)
             for p in pairs}

    files = sorted(glob.glob(str(ROOT / args.npz)) or glob.glob(args.npz))
    if not files:
        raise SystemExit(f"no npz matched {args.npz}")

    out = {}
    for f in files:
        npz = np.load(f, allow_pickle=True)
        meta = json.loads(str(npz["meta"][0]))
        idx = {Path(c).name: i for i, c in enumerate(npz["clips"])}
        pooled = npz["pooled"]
        keep = [r for r in rows
                if paths[r["pair_id"]][0] in idx and paths[r["pair_id"]][1] in idx]
        if len(keep) != len(rows):
            print(f"  skip {Path(f).name}: covers {len(keep)} of {len(rows)} pairs", flush=True)
            continue
        y = np.array([FAMILIES.index(r["perturbation"]) for r in keep])
        g = np.array([r["source_id"] for r in keep])
        ea = np.stack([pooled[idx[paths[r["pair_id"]][0]]] for r in keep])
        eb = np.stack([pooled[idx[paths[r["pair_id"]][1]]] for r in keep])
        X = (np.einsum("ij,ij->i", ea, eb)[:, None] if args.readout == "cosine"
             else eb - ea)
        pred = loto_nested(X, y, g) if args.nested else loto_fixed(X, y, g, args.C)
        acc, f1 = float((pred == y).mean()), macro_f1(y, pred)
        lay = meta.get("layer", re.search(r"L(\d+)", Path(f).stem))
        out[str(lay)] = {"acc": round(acc, 4), "macro_f1": round(f1, 4), "file": Path(f).name}
        print(f"  layer {str(lay):>3}  acc {acc:.4f}  macro-F1 {f1:.4f}", flush=True)

    if not out:
        raise SystemExit("nothing screened")
    best = max(out, key=lambda k: out[k]["acc"])
    print(f"\nbest layer for {args.encoder}: {best} at {out[best]['acc']:.4f} "
          f"(shipped layer 6: {out.get('6', {}).get('acc', 'n/a')})")
    print("Now run that layer through diagnose_nested_cv.py for the reportable number.")

    suffix = "" if args.readout == "diff" else f"_{args.readout}"
    dest = (Path(args.out) if args.out else
            Path(f"results/diagnostics/{args.encoder}_layer_sweep{suffix}.json"))
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"meta": {"encoder": args.encoder, "readout": args.readout,
                  "protocol": ("leave-one-track-out, nested C selection inside each "
                               "training set, identical to diagnose_nested_cv.loto, so "
                               "these are reportable numbers"
                               if args.nested else
                               "leave-one-track-out, L2 fixed at C=%g; a SCREEN that "
                               "selects on the scored folds and therefore favours the "
                               "baseline, which is the intended direction" % args.C),
                  "feature": ("1-d A-vs-B cosine" if args.readout == "cosine"
                              else "1024-d difference e_B - e_A"),
                  "best_layer": best}, "by_layer": out}, indent=1))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
