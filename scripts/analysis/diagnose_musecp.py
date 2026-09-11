"""MuseCPEval as a diagnosis baseline on the Setup-1 pairs.

MuseCPEval (Vishe et al., ISMIR 2026) scores music *context preservation* between an
original and an edited copy of it. Its released toolkit reports 12 metrics over five
facets: harmony, rhythm & meter, structure, melody & motifs, and timbre. The Setup-1
battery is built by self-perturbation, so B is a processed copy of A and the pairs are
exactly the (reference, estimate) shape MuseCPEval takes. Setup 2 is also runnable, because its
seed-vs-iter_k harness pairs the original clip with its kth edit, which is the same shape; it is
simply not scored here. Setup 3 is the one MuseCPEval cannot take, since a rollout for a context
has no original it was derived from.

Scores come from `musecpeval --batch-json`, run in its own env (numpy < 2.1, msaf for
the structural facet). This script only reads the summary.csv it writes.

The classifier protocol is the one behind Table 1, imported rather than reimplemented:
leave-one-track-out, L2 chosen on a grouped inner split of the training fold only.

Usage (DSP env, from repo root):
    python scripts/analysis/diagnose_musecp.py --summary <path to summary.csv>
Writes results/diagnostics/diagnosis_musecp.json.
"""
import sys, json, csv, argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_fixes as rf
from diagnose_nested_cv import loto

# The 12 metrics the paper reports, grouped by facet. The second element says whether
# the metric rises with preservation (True) or with change (False); it is used for the
# signature table only, never by the classifier.
FACETS = {
    "harmony": [
        ("harmony_tonality.key_relatedness.distance_norm_0to1", False),
        ("harmony_tonality.chroma_similarity.mean_chroma_cosine", True),
        ("harmony_tonality.chroma_similarity.chroma_dtw_cosine", True),
    ],
    "rhythm": [
        ("rhythm_meter.delta_bpm_folded", False),
        ("rhythm_meter.beat_mir_eval.F-measure", True),
        ("rhythm_meter.beat_mir_eval.Information gain", True),
    ],
    "structure": [
        ("structural_form.pairwise_f", True),
        ("structural_form.ari", True),
    ],
    "melody": [
        ("melodic_content.contour_dtw_similarity", True),
        ("melodic_content.motif_3gram_recall", True),
    ],
    "timbre": [
        ("timbre_texture.mfcc_skl_similarity", True),
        ("timbre_texture.mean_mfcc_cosine", True),
    ],
}
COLS = [c for f in FACETS.values() for c, _ in f]


def keep_rows():
    """Reproduce review_fixes.build_feature_sets' row selection, keeping pair_id.

    build_feature_sets returns y and g but not the ids, and the rows it keeps are the
    ones whose two clips are both in the CLAP embedding cache. Alignment is asserted
    against it below rather than assumed.
    """
    rows = [r for r in rf.read_csv(rf.ROOT / "results/coherence/pair_scores_cbase.csv")
            if r["perturbation"] in rf.FAMILIES]
    manifest = json.load(open(rf.ROOT / "perturbations/pairs_manifest.json"))
    pairs = manifest["pairs"] if isinstance(manifest, dict) else manifest
    paths = {p["pair_id"]: (p["ref_path"], p["cand_path"]) for p in pairs}
    npz = np.load(rf.ROOT / "results/baselines/clap_embeddings.npz", allow_pickle=True)
    idx = {c: i for i, c in enumerate(npz["clips"])}
    keep = []
    for r in rows:
        ref, cand = paths[r["pair_id"]]
        if ref in idx and cand in idx:
            keep.append(r)
    return keep


def read_musecp(path):
    by_id = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            by_id[row["id"]] = row
    return by_id


def signature(by_id):
    """Mean of each metric per perturbation family, and the shift from the control."""
    fams = {}
    for row in by_id.values():
        fams.setdefault(row["perturbation"], []).append(row)
    table = {}
    for fam, rows in fams.items():
        table[fam] = {}
        for col in COLS:
            vals = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
            table[fam][col] = float(np.mean(vals)) if vals else None
    out = {"mean": table, "shift_from_control": {}}
    ctrl = table.get("control", {})
    for fam, means in table.items():
        if fam == "control":
            continue
        out["shift_from_control"][fam] = {
            c: (None if means[c] is None or ctrl.get(c) is None
                else round(means[c] - ctrl[c], 4))
            for c in COLS
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="summary.csv from musecpeval")
    ap.add_argument("--out", default="results/diagnostics/diagnosis_musecp.json")
    args = ap.parse_args()

    by_id = read_musecp(args.summary)
    print("musecpeval rows:", len(by_id))

    feats, y, g = rf.build_feature_sets()
    keep = keep_rows()
    assert len(keep) == len(y), (len(keep), len(y))
    assert all(k["source_id"] == gg for k, gg in zip(keep, g)), "row order drifted"

    missing = [k["pair_id"] for k in keep if k["pair_id"] not in by_id]
    if missing:
        raise SystemExit("%d kept pairs absent from the summary, e.g. %s"
                         % (len(missing), missing[:3]))

    def matrix(cols):
        return np.array([[float(by_id[k["pair_id"]][c]) for c in cols] for k in keep])

    sets = {"musecp12": COLS}
    for name, spec in FACETS.items():
        sets["musecp_" + name] = [c for c, _ in spec]
    # what MuseCPEval would be without the facet our T term owns
    sets["musecp10_noTimbre"] = [c for c in COLS if not c.startswith("timbre_")]

    out, preds = {}, {}
    for name, cols in sets.items():
        X = matrix(cols)
        if not np.isfinite(X).all():
            bad = int((~np.isfinite(X)).any(1).sum())
            raise SystemExit("%s: %d rows with non-finite values" % (name, bad))
        p = loto(X, y, g)
        preds[name] = p
        a, f = rf.acc_f1(y, p)
        out[name] = {"dim": X.shape[1], "acc": round(float(a), 4),
                     "macro_f1": round(float(f), 4)}
        print(name, out[name], flush=True)

    # the profile rows, recomputed here so the bootstrap has both predictions
    for name in ("HTRS", "HTR"):
        preds[name] = loto(feats[name], y, g)
        a, f = rf.acc_f1(y, preds[name])
        out[name] = {"dim": int(feats[name].shape[1]), "acc": round(float(a), 4),
                     "macro_f1": round(float(f), 4)}
        print(name, out[name], flush=True)

    out["contrasts"] = {}
    for a_, b_ in [("HTRS", "musecp12"), ("HTR", "musecp12"),
                   ("HTRS", "musecp10_noTimbre"), ("musecp12", "musecp10_noTimbre")]:
        out["contrasts"]["%s - %s" % (a_, b_)] = rf.paired_bootstrap(y, g, preds, a_, b_)
        print(a_, "-", b_, out["contrasts"]["%s - %s" % (a_, b_)], flush=True)

    out["signature"] = signature(by_id)
    out["meta"] = {
        "n_pairs": int(len(y)), "n_tracks": int(len(np.unique(g))),
        "families": rf.FAMILIES,
        "scorer": "musecpeval 0.3.0, 12 paper-reported metrics over five facets",
        "protocol": "leave-one-track-out; L2 chosen on a grouped inner split of the "
                    "training fold only; paired bootstrap over tracks",
        "note": "Setup-1 only. Setup 2 could also be scored, since seed-vs-iter_k is "
                "the same (original, edited) shape; Setup 3 could not, having no "
                "original.",
    }
    dest = rf.ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest.relative_to(rf.ROOT))


if __name__ == "__main__":
    main()
