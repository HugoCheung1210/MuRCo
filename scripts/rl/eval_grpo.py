#!/usr/bin/env python
"""Did the pilot buy anything? Frozen vs checkpoint on held-out contexts.

Two subcommands, because the middle of the pipeline is the existing scorers and this
script deliberately does not reimplement them:

    generate  -> rollouts for each arm + a standard pairs manifest
    (then)    -> score_s_mf.py, score_separability.py, score_baselines.py, copy_metrics.py
    compare   -> the arm table, with cluster-bootstrap CIs over contexts

The arms. `frozen` is the untrained policy at the same CFG, temperature and top-k as
training, which is the only fair reference: a run compared against a CFG-3.0 baseline
would be reading a sampler change as a training effect. `frozen_bo8` re-ranks the frozen
arm's own rollouts by C, which is the honest bar from the proposal -- a policy sampled
once must beat the frozen model sampled eight times and re-ranked, because re-ranking is
free and needs no training. Each `stepNNNN` arm is a checkpoint.

What decides the pilot, from `rlhf_with_C_procedure.md` §6, is not C alone:

    reward     C on held-out contexts     up, and past frozen_bo8
    transfer   CLAP / SCS / COCOLA        up or unchanged, never down
    novelty    xcorr_wave, xcorr_mel      unchanged from frozen
    quality    listen to samples/         not degenerate

C up with transfer flat and copy up is not a pass, it is the reward-hacking outcome, and
it is reported as that.

Usage (GPU box):
    python eval_grpo.py generate --out results/rl/eval_pilot \
        --checkpoints results/rl/grpo_pilot/checkpoints/step0050,.../step0150 --n 8
    # ... run the scorers against results/rl/eval_pilot/eval_manifest.json ...
    python eval_grpo.py compare --dir results/rl/eval_pilot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401
except ImportError:
    pass

SR = 32000


# ------------------------------------------------------------------------ generate

def cmd_generate(a) -> int:
    import soundfile as sf
    import torch

    from grpo_train import Policy, load_context_audio, prompt_for

    doc = json.loads(Path(a.contexts).read_text())
    contexts = [c for c in doc["contexts"] if c["split"] == "heldout"]
    if a.limit:
        contexts = contexts[:a.limit]
    if not contexts:
        raise SystemExit("no heldout-split contexts; the pilot must not be evaluated on "
                         "the contexts it trained on")

    out = Path(a.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    arms = ["frozen"] + [Path(c).name for c in a.checkpoints.split(",") if c]
    ckpt_of = {Path(c).name: c for c in a.checkpoints.split(",") if c}

    pol = Policy(a.model, a.device, a.lora_r, a.lora_alpha, a.dtype,
                 grad_checkpointing=False)
    torch.manual_seed(a.seed)

    pairs = []
    for arm in arms:
        if arm == "frozen":
            ctxm = pol.model.disable_adapter()
        else:
            pol.model.load_adapter(ckpt_of[arm], adapter_name=arm)
            pol.model.set_adapter(arm)
            import contextlib
            ctxm = contextlib.nullcontext()

        with ctxm:
            for c in contexts:
                cid = c["context_id"]
                y_a = load_context_audio(c, a.audio_root)
                # `<stem>__A.wav` + `<stem>_cand<NN>__B.wav` is the layout copy_metrics.py
                # globs for, so the context is written once per arm rather than once.
                pa = audio_dir / f"{arm}_{cid}__A.wav"
                if not pa.exists():
                    sf.write(pa, y_a, SR)
                cont, _, _, _ = pol.sample_group(
                    y_a, prompt_for(c), a.n, c["generate_s"], a.guidance,
                    a.temperature, a.top_k)
                for k in range(a.n):
                    pb = audio_dir / f"{arm}_{cid}_cand{k:02d}__B.wav"
                    sf.write(pb, cont[k], SR)
                    pairs.append({
                        "pair_id": f"{arm}::{cid}::cand{k:02d}",
                        "source_id": cid, "genre": c["genre"],
                        "dim_target": "grpo_eval", "perturbation": arm,
                        "magnitude": 0.0, "magnitude_unit": "rollout_index",
                        "ref_path": str(pa.relative_to(ROOT)) if pa.is_relative_to(ROOT) else str(pa),
                        "cand_path": str(pb.relative_to(ROOT)) if pb.is_relative_to(ROOT) else str(pb),
                        "partner_id": "", "sr": SR, "seconds": c["generate_s"],
                        "gen_seed": a.seed, "notes": f"{arm}, heldout continuation",
                    })
                print(f"  {arm:14s} {cid}  {a.n} rollouts")

    manifest = {"meta": {"experiment": "GRPO pilot evaluation", "regime": "musicgen "
                         "continuation", "arms": arms, "n_per_context": a.n,
                         "n_contexts": len(contexts), "sr": SR,
                         "guidance_scale": a.guidance, "temperature": a.temperature,
                         "top_k": a.top_k, "seed": a.seed,
                         "note": "standard pairs manifest; A=context, B=continuation"},
                "pairs": pairs}
    (out / "eval_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n{len(pairs)} pairs -> {out}/eval_manifest.json\nnext:")
    print(f"  python score_s_mf.py --manifest {out}/eval_manifest.json "
          f"--out {out}/s_scores.json --scorer coherence --secs 6 --gap-s 1.0")
    print(f"  python score_separability.py --manifest {out}/eval_manifest.json "
          f"--dimensions H,T,R,S --s-cache {out}/s_scores.json --output-dir {out}")
    print(f"  python score_baselines.py --manifest {out}/eval_manifest.json "
          f"--out {out}/baseline_scores.json --metrics clap_htsat,scs")
    print(f"  python copy_metrics.py --audio-dir {out}/audio --sr {SR} "
          f"--out {out}/copy.json")
    return 0


# ------------------------------------------------------------------------- compare

def _boot(by_ctx: dict[str, list[float]], n: int = 10000, seed: int = 7) -> tuple:
    """Cluster bootstrap over contexts: rollouts within a context are not independent."""
    rng = np.random.default_rng(seed)
    keys = list(by_ctx)
    means = [float(np.mean(by_ctx[k])) for k in keys]
    draws = [float(np.mean(rng.choice(means, size=len(keys), replace=True)))
             for _ in range(n)]
    return float(np.mean(means)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _paired_boot(a_ctx: dict, b_ctx: dict, n: int = 10000, seed: int = 7) -> tuple:
    """Paired on context, which is what makes the arms comparable at all."""
    rng = np.random.default_rng(seed)
    keys = sorted(set(a_ctx) & set(b_ctx))
    d = np.array([np.mean(b_ctx[k]) - np.mean(a_ctx[k]) for k in keys])
    draws = [float(rng.choice(d, size=len(d), replace=True).mean()) for _ in range(n)]
    return float(d.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_arms(eval_dir) -> dict[str, dict[str, dict[str, list]]]:
    """Read one eval directory into arms[arm][metric][context] -> list of rollout values.

    Split out of cmd_compare so that pool_seeds.py reads a run exactly the way the
    single-run comparison does. The pair_id convention "<arm>::<context>::r<k>" and the
    two-file join below are the fragile parts, and they must not exist in two copies.
    """
    import csv

    d = Path(eval_dir)
    rows = list(csv.DictReader((d / "pair_scores.csv").open()))
    extra: dict[str, dict] = {}

    # baseline_scores.json is keyed by pair_id; copy.json is keyed by file stem, because
    # copy_metrics.py works off the audio dir and never sees the manifest. The manifest
    # carries both, so it is the translation table rather than a filename convention.
    stem_to_pid = {}
    man = d / "eval_manifest.json"
    if man.exists():
        for p in json.loads(man.read_text())["pairs"]:
            stem_to_pid[Path(p["cand_path"]).stem.replace("__B", "")] = p["pair_id"]

    for name, key, remap in (("baseline_scores.json", "scores", False),
                             ("copy.json", "per_pair", True)):
        p = d / name
        if not p.exists():
            continue
        blob = json.loads(p.read_text())
        for k, vals in (blob.get(key) or {}).items():
            pid = stem_to_pid.get(k, k) if remap else k
            extra.setdefault(pid, {}).update(
                {m: v for m, v in vals.items() if isinstance(v, (int, float))})

    # pair_id is "<arm>::<context>::r<k>"
    arms: dict[str, dict[str, dict[str, list]]] = {}
    for r in rows:
        pid = r["pair_id"]
        if pid.count("::") != 2:
            continue
        arm, ctx, _ = pid.split("::")
        # score_separability.py names its columns score_<dim> and does not write C; C is
        # the weighted geometric mean of the dimensions, which with equal weights is the
        # plain one. Computing it here rather than shelling out to compute_C.py keeps the
        # arm comparison to a single pass over one file.
        vals = {k: float(r[f"score_{k}"]) for k in ("H", "T", "R", "S")
                if r.get(f"score_{k}") not in (None, "")}
        if vals:
            vals["C"] = float(np.exp(np.mean([np.log(max(v, 1e-9)) for v in vals.values()])))
        vals.update(extra.get(pid, {}))
        for k, v in vals.items():
            arms.setdefault(arm, {}).setdefault(k, {}).setdefault(ctx, []).append(v)

    if "frozen" not in arms:
        raise SystemExit(f"no `frozen` arm in {d}/pair_scores.csv; nothing to compare against")
    return arms


def cmd_compare(a) -> int:
    arms = load_arms(a.dir)

    # best-of-N: an arm's own rollouts re-ranked by C, per context. `frozen_boN` is the bar
    # an RL policy sampled once has to clear, since re-ranking costs no training. Every
    # other arm gets the same treatment because `frozen_boN` spends N times the test-time
    # compute of a single policy sample: comparing a checkpoint against it answers "is RL
    # worth it at equal training cost", and comparing `<ckpt>_boN` against it answers "is RL
    # worth it at equal *inference* cost", which is the one a deployed system faces.
    def _best_of(per_metric: dict[str, dict[str, list]]) -> tuple[dict, int]:
        n_pool = int(np.median([len(v) for v in per_metric["C"].values()]))
        pick = {c: int(np.argmax(v)) for c, v in per_metric["C"].items()}
        bo = {"C": {c: [max(v)] for c, v in per_metric["C"].items()}}
        for m, per_ctx in per_metric.items():
            if m == "C":
                continue
            bo[m] = {c: [v[pick[c]]] for c, v in per_ctx.items() if pick[c] < len(v)}
        return bo, n_pool

    BO = "frozen_bo8"
    bo_arms: dict[str, str] = {}          # base arm -> its best-of-N arm name
    for base in [k for k in arms if "C" in arms[k]]:
        bo, n_pool = _best_of(arms[base])
        name = f"{base}_bo{n_pool}"
        arms[name] = bo
        bo_arms[base] = name
    BO = bo_arms.get("frozen", BO)

    # onset_ratio and pc_entropy come free with copy.json and are what separate "the
    # continuation is static" from "the continuation is musically active". A policy that
    # improved by droning on the context's key would show them falling; one that improved
    # by playing something would show them holding or rising.
    # chroma_cos is deliberately absent: it is numerically identical to H (both are the
    # cosine of mean CQT chroma), which is why reward.py keeps it out of the copy penalty
    # too. Showing it twice invites reading one number as corroborating the other.
    metrics = [m for m in ("C", "S", "H", "T", "R", "clap_htsat", "scs",
                           "xcorr_wave", "xcorr_mel",
                           "onset_ratio", "pc_entropy") if m in arms["frozen"]]
    order = ["frozen"] + ([BO] if BO in arms else []) + \
            sorted(k for k in arms if k not in ("frozen", BO))

    print(f"{'arm':<16}" + "".join(f"{m:>13}" for m in metrics))
    print("-" * (16 + 13 * len(metrics)))
    for arm in order:
        cells = []
        for m in metrics:
            if m not in arms[arm]:
                cells.append(f"{'-':>13}")
                continue
            mu, _, _ = _boot(arms[arm][m], a.boot, a.seed)
            cells.append(f"{mu:>13.4f}")
        print(f"{arm:<16}" + "".join(cells))

    print(f"\ndelta vs frozen (paired on context, {a.boot} cluster bootstrap, 95% CI)")
    for arm in order[1:]:
        print(f"\n  {arm}")
        for m in metrics:
            if m not in arms[arm] or m not in arms["frozen"]:
                continue
            dm, lo, hi = _paired_boot(arms["frozen"][m], arms[arm][m], a.boot, a.seed)
            flag = "" if lo <= 0 <= hi else ("  *" if dm > 0 else "  * (down)")
            print(f"    {m:<12}{dm:+8.4f}  [{lo:+.4f}, {hi:+.4f}]{flag}")

    if BO in arms:
        print(f"\ndelta vs {BO} -- the bar that decides whether RL was worth it")
        for arm in order:
            if arm in ("frozen", BO) or "C" not in arms[arm]:
                continue
            dm, lo, hi = _paired_boot(arms[BO]["C"], arms[arm]["C"], a.boot, a.seed)
            verdict = f"beats {BO}" if lo > 0 else (
                f"below {BO}" if hi < 0 else f"indistinguishable from {BO}")
            print(f"    {arm:<16}C {dm:+.4f} [{lo:+.4f}, {hi:+.4f}]   {verdict}")
            # Two arms can reach the same C by different means, so C alone does not settle
            # what re-ranking cost. The copy and activity rows say whether the arm got there
            # by tracking the context more closely or by playing less.
            for m in metrics:
                if m == "C" or m not in arms[arm] or m not in arms[BO]:
                    continue
                dm, lo, hi = _paired_boot(arms[BO][m], arms[arm][m], a.boot, a.seed)
                flag = "" if lo <= 0 <= hi else ("  *" if dm > 0 else "  * (down)")
                print(f"        {m:<12}{dm:+8.4f}  [{lo:+.4f}, {hi:+.4f}]{flag}")

    # Does a held-out judge simply prefer static audio? SCS is a cosine between the two
    # self-similarity matrices, so a continuation with little internal variation can score
    # well for reasons that have nothing to do with coherence. Pooling every pair across
    # every arm, correlate each judge against onset density: a strongly negative
    # correlation means that judge is reading activity as incoherence, and a fall in it
    # cannot be taken as evidence of reward hacking on its own.
    pooled: dict[str, list] = {}
    for arm in arms:
        if arm in bo_arms.values():   # a subset of a base arm's rollouts; would double-count
            continue
        for m, per_ctx in arms[arm].items():
            for vals in per_ctx.values():
                pooled.setdefault(m, []).extend(vals)
    n = min((len(v) for v in pooled.values()), default=0)
    if n > 10 and "onset_ratio" in pooled:
        act = np.asarray(pooled["onset_ratio"][:n])
        print("\ncorrelation with onset density, pooled over arms "
              f"(n={n}) -- does the judge penalise activity?")
        for m in ("scs", "clap_htsat", "S", "C"):
            if m in pooled:
                v = np.asarray(pooled[m][:n])
                if v.std() > 1e-9 and act.std() > 1e-9:
                    print(f"    {m:<12}r = {float(np.corrcoef(act, v)[0, 1]):+.3f}")

    print("\nRead the copy rows before the C row. C up while clap_htsat/scs are flat and "
          "\nxcorr_* are up is the reward-hacking outcome, not an improvement -- unless "
          "\nonset_ratio and pc_entropy say the fine-tuned arm is the more active one, in "
          "\nwhich case the static arm is the frozen model and the judge may be the "
          "\nproblem. Listen before deciding which.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--contexts", default=str(ROOT / "results/rl/contexts.json"))
    g.add_argument("--audio-root", default=None)
    g.add_argument("--out", default=str(ROOT / "results/rl/eval_pilot"))
    g.add_argument("--checkpoints", default="", help="comma-separated adapter dirs")
    g.add_argument("--model", default="facebook/musicgen-small")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--n", type=int, default=8, help="rollouts per context per arm")
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--guidance", type=float, default=1.0)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=250)
    g.add_argument("--lora-r", type=int, default=16)
    g.add_argument("--lora-alpha", type=int, default=32)
    g.add_argument("--dtype", default="bf16")
    g.add_argument("--seed", type=int, default=20260808)
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("compare")
    c.add_argument("--dir", default=str(ROOT / "results/rl/eval_pilot"))
    c.add_argument("--boot", type=int, default=10000)
    c.add_argument("--seed", type=int, default=7)
    c.set_defaults(func=cmd_compare)

    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
