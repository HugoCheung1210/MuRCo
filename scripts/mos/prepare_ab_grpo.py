#!/usr/bin/env python
"""Build the GRPO arm: fine-tuned vs frozen MusicGen continuations, 5-point preference.

Deliberately a near-copy of `prepare_ab_session2.py`, because the two arms must stay
analysable by the same code (`score_ab_session2.py`) and comparable to the re-ranking A/B
pilot that came before both. Same 5-point scale, same skip, same blinding discipline, same
level matching. Only the contrast is new.

The stimulus. Each option is `context tail | continuation`, one continuous clip:

    [ last --lead-in s of the 10 s context ][ first --cont-s s of the continuation ]

Both options share the identical lead-in, so the only thing that differs between A and B
is what the model played after it. That is the same logic as session 2, where both options
carry the same untouched surround and differ only in the regenerated middle. It also means
each option is judgeable on its own, which lets the existing two-player question type be
reused unchanged rather than inventing a three-player page.

Which rollout represents each arm. **A uniform random draw at a fixed seed, not the
best-scoring one.** Selecting by C would put the metric inside the stimulus choice for a
study whose whole purpose is to test C against listeners, and the resulting preference
would be partly an artefact of that selection. The cost is that an unlucky draw can land
on an outlier, which is what the mandatory audition below is for.

Every clip must be auditioned before use. The ethics amendment commits to screening all of
them rather than a sample, on the grounds that a fine-tuned model is less predictable than
a released one. `--audition` writes a local page for exactly that.

Usage (DSP env, needs ffmpeg):
    python scripts/mos/prepare_ab_grpo.py \
        --eval-dir results/rl/eval_pilot --arm step0175
    python scripts/mos/prepare_ab_grpo.py --audition        # screening page only
Outputs: <out-dir>/audio/*.mp3, trials.csv, key.csv, audition.html
"""
import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)
SR = 48000
EPS = 1e-8

AUDITION = """<!doctype html><meta charset="utf-8"><title>GRPO arm — audition</title>
<style>body{font:15px/1.5 system-ui;max-width:820px;margin:2rem auto;padding:0 1rem}
.t{border:1px solid #ddd;border-radius:8px;padding:.8rem 1rem;margin:.7rem 0}
audio{width:100%;margin:.3rem 0}.n{color:#666;font-size:13px}</style>
<h1>GRPO arm — audition every clip</h1>
<p class="n">The amendment commits to screening <b>all</b> clips, not a sample. Exclude
anything harsh, distorted, unpleasantly repetitive, or containing intelligible speech, and
regenerate it. This page is local and never reaches a participant. The A/B roles are
deliberately not shown, so screening cannot be swayed by which arm a clip came from.</p>
__ROWS__
"""


def load(path, sr=SR):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype("float32")


def encode(y, out_path):
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, y, SR)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name,
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(out_path)],
                       check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default="results/rl/eval_pilot",
                    help="tree written by eval_grpo.py generate")
    ap.add_argument("--arm", default="step0175", help="checkpoint arm to test")
    ap.add_argument("--baseline", default="frozen")
    ap.add_argument("--out-dir", default="results/mos/grpo_ab")
    ap.add_argument("--lead-in", type=float, default=4.0,
                    help="seconds of shared context tail prepended to both options")
    ap.add_argument("--cont-s", type=float, default=8.0,
                    help="seconds of continuation after the lead-in")
    ap.add_argument("--seed", type=int, default=20260808)
    ap.add_argument("--audition", action="store_true",
                    help="rebuild audition.html from an existing render and stop")
    args = ap.parse_args()

    out = ROOT / args.out_dir
    audio_out = out / "audio"

    if args.audition:
        rows = list(csv.DictReader((out / "trials.csv").open()))
        html = "\n".join(
            f'<div class="t"><b>{r["trial_id"]}</b>'
            f'<audio controls preload="none" src="audio/{r["file_a"]}"></audio>'
            f'<audio controls preload="none" src="audio/{r["file_b"]}"></audio></div>'
            for r in rows)
        (out / "audition.html").write_text(AUDITION.replace("__ROWS__", html))
        print(f"wrote {(out / 'audition.html').relative_to(ROOT)} ({len(rows)} trials)")
        return 0

    src = ROOT / args.eval_dir / "audio"
    if not src.is_dir():
        raise SystemExit(f"{src} not found — run eval_grpo.py generate first")
    audio_out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    contexts = sorted({re.match(rf"{args.baseline}_(.+?)_cand\d+__B\.wav", p.name).group(1)
                       for p in src.glob(f"{args.baseline}_*_cand*__B.wav")})
    if not contexts:
        raise SystemExit(f"no {args.baseline}_*_cand*__B.wav under {src}")

    # Position is BALANCED, not left to chance. Listeners carry a position bias, and if it
    # correlated with which option is the fine-tuned one it would land in the headline.
    n = len(contexts)
    ft_first = np.array([True] * (n // 2) + [False] * (n - n // 2))
    rng.shuffle(ft_first)

    lead_n, cont_n = int(args.lead_in * SR), int(args.cont_s * SR)
    rows, key = [], []
    for i, cid in enumerate(contexts):
        tid = f"g{i + 1:03d}"
        ctx = load(src / f"{args.baseline}_{cid}__A.wav")
        lead = ctx[-lead_n:] if ctx.shape[0] > lead_n else ctx

        clips = {}
        for role, arm in (("finetuned", args.arm), ("frozen", args.baseline)):
            cands = sorted(src.glob(f"{arm}_{cid}_cand*__B.wav"))
            if not cands:
                raise SystemExit(f"no rollouts for arm {arm}, context {cid}")
            pick = cands[int(rng.integers(len(cands)))]
            clips[role] = (np.concatenate([lead, load(pick)[:cont_n]]), pick.name)

        roles = ["finetuned", "frozen"] if ft_first[i] else ["frozen", "finetuned"]
        ya, ta = clips[roles[0]]
        yb, tb = clips[roles[1]]
        m = min(len(ya), len(yb))
        ya, yb = ya[:m], yb[:m]
        # match B to A, then scale the PAIR together so neither option is the louder one
        ra = np.sqrt(np.mean(ya ** 2)) + EPS
        rb = np.sqrt(np.mean(yb ** 2)) + EPS
        yb = yb * (ra / rb)
        peak = max(float(np.max(np.abs(ya))), float(np.max(np.abs(yb))), EPS)
        g = min(0.1 / ra, 0.99 / peak)
        ya, yb = (ya * g).astype("float32"), (yb * g).astype("float32")

        fa, fb = f"{tid}_A.mp3", f"{tid}_B.mp3"
        encode(ya, audio_out / fa)
        encode(yb, audio_out / fb)
        rows.append({"trial_id": tid, "file_a": fa, "file_b": fb})
        key.append({"trial_id": tid, "context_id": cid,
                    "opt_a_role": roles[0], "opt_b_role": roles[1],
                    "file_a_src": ta, "file_b_src": tb})
        print(f"  {tid}  {cid}  A={roles[0]:<10} B={roles[1]}")

    with (out / "trials.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial_id", "file_a", "file_b"])
        w.writeheader()
        w.writerows(rows)
    with (out / "key.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    (out / "meta.json").write_text(json.dumps(
        {"arm": args.arm, "baseline": args.baseline, "seed": args.seed,
         "lead_in_s": args.lead_in, "cont_s": args.cont_s,
         "n_trials": len(rows), "eval_dir": args.eval_dir,
         "selection": "uniform random rollout per arm, fixed seed (NOT argmax C)"},
        indent=2))

    html = "\n".join(
        f'<div class="t"><b>{r["trial_id"]}</b>'
        f'<audio controls preload="none" src="audio/{r["file_a"]}"></audio>'
        f'<audio controls preload="none" src="audio/{r["file_b"]}"></audio></div>'
        for r in rows)
    (out / "audition.html").write_text(AUDITION.replace("__ROWS__", html))

    print(f"\n{len(rows)} trials -> {out.relative_to(ROOT)}")
    print(f"  audition ALL {2 * len(rows)} clips: open {(out / 'audition.html').relative_to(ROOT)}")
    print("  then upload audio/ to Qualtrics IN A NEW FOLDER and run make_grpo_loop.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
