"""Nested-CV diagnosis for the two CLAP embedding feature sets and every scalar rival,
plus the paired bootstrap against the profile rows. This is what Table 1 is built from.

Inner-fold max_iter is 300 rather than 5000. This is safe rather than a corner cut:
results/diagnostics/clap_probe_sweep.json records that clap768_diff scores 0.813/0.756
at max_iter=400 and at max_iter=5000 alike, so iteration budget moves the selected C,
not the score. The final per-fold fit still runs to 5000.

B7 of the 2026-08-24 external review added TuneJury, SongEval-coherence and COCOLA as
scalar rows, so "no published scalar comes close" is measured on five scalars rather
than asserted on two. See `doc/notes/icassp_review_response_2026-08-24.md`.

Usage (DSP env, from repo root):
    python scripts/analysis/diagnose_nested_cv.py
Writes results/diagnostics/diagnosis_nested.json.
"""
import sys, json, time
import numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import review_fixes as rf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

C_GRID = [0.02, 0.3, 1.0, 10.0]


def loto(X, y, g):
    pred = np.empty_like(y)
    for tr in np.unique(g):
        te = g == tr
        Xtr, ytr, gtr = X[~te], y[~te], g[~te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Z, Zt = (Xtr - mu) / sd, (X[te] - mu) / sd
        best, bs = None, -np.inf
        for c in C_GRID:
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


def main():
    # Guarded: importing anything from this module (loto, say) must NOT re-run the
    # whole Table 2 rebuild. It costs hours on the 768- and 1024-dimensional rows.
    feats, y, g = rf.build_feature_sets()
    SCALARS = ['scs', 'clap_cosine', 'tunejury_rel', 'songeval_coherence_rel', 'cocola',
               'muq_cosine', 'mert_cosine']
    # Audiobox Aesthetics is 4-d, not a scalar, so it rides with the embeddings rather than
    # the scalar rows: same loto, but the paired bootstrap pairs it against both profiles.
    AESTHETICS = ['audiobox_rel', 'audiobox_b']
    # A4: masked-prediction encoders, present only once ssl_probe.py extract has run
    EMBEDDINGS = ['clap768_diff', 'clap768_full', 'muq_diff', 'mert_diff'] + AESTHETICS
    out, preds = {}, {}
    for name in [e for e in EMBEDDINGS if e in feats]:
        t = time.time()
        p = loto(feats[name], y, g)
        preds[name] = p
        a, f = rf.acc_f1(y, p)
        out[name] = {'dim': int(feats[name].shape[1]), 'acc': round(float(a), 4),
                     'macro_f1': round(float(f), 4)}
        print(name, out[name], '%.0fs' % (time.time() - t), flush=True)

    # profile rows and every scalar, recomputed here so the paired bootstrap has
    # every prediction. A scalar absent from feats simply was not cached on this box.
    for name in ['HTRS', 'HTR'] + [s for s in SCALARS if s in feats]:
        p = loto(feats[name], y, g)
        preds[name] = p
        a, f = rf.acc_f1(y, p)
        out[name] = {'dim': int(feats[name].shape[1]), 'acc': round(float(a), 4),
                     'macro_f1': round(float(f), 4)}
        print(name, out[name], flush=True)

    out['contrasts'] = {}
    pairs = [('HTRS', 'HTR')]
    pairs += [('HTRS', e) for e in EMBEDDINGS if e in preds]
    pairs += [('HTR', e) for e in EMBEDDINGS if e in preds]
    pairs += [('HTRS', s) for s in SCALARS if s in preds]
    for a, b in pairs:
        out['contrasts'][f'{a} - {b}'] = rf.paired_bootstrap(y, g, preds, a, b)
        print(a, '-', b, out['contrasts'][f'{a} - {b}'], flush=True)

    best = max((s for s in SCALARS if s in out), key=lambda s: out[s]['acc'])
    out['meta'] = {
        'n_pairs': int(len(y)), 'n_tracks': int(len(np.unique(g))),
        'protocol': 'leave-one-track-out over tracks, L2 chosen by a grouped inner split '
                    'on the training fold only; paired bootstrap over tracks',
        'scalars_compared': [s for s in SCALARS if s in out],
        'best_scalar': best, 'best_scalar_acc': out[best]['acc'],
        'note': 'strongest-scalar claim in the abstract must quote best_scalar.',
    }
    dest = rf.ROOT / 'results/diagnostics/diagnosis_nested.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print('best scalar:', best, out[best], flush=True)
    print('wrote', dest.relative_to(rf.ROOT))


if __name__ == "__main__":
    main()
