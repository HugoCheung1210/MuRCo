#!/usr/bin/env python
"""Score the matrix / rerank pairs with the two closest *runnable* rival metrics.

Why this exists: `doc/notes/rival_check_2026-08-16.md` names two threats that ship
public checkpoints, so the thesis can answer them with numbers instead of arguments
(the `score_cocola.py` precedent -- an argument is cheaper to attack than a number).

  * **tunejury** -- TuneJury (arXiv 2606.17006), threat C5 "High": an open pairwise
    *preference* reward model (2.8M MLP head over frozen LAION-CLAP-Music 512-d +
    MERT-v1-330M 1024-d + CLAP text 512-d) used by its authors in exactly our three
    downstream slots: best-of-N selection, DITTO latent optimisation, expert iteration.
  * **songeval** -- SongEval (arXiv 2505.10793, ICASSP 2026), the multi-dimensional
    *aesthetics* wave: MuQ-large hidden layer 6 -> a 5-dimensional head whose first
    output is literally called "Coherence".

**Both are ABSOLUTE metrics: they rate one clip.** C rates a relation. Scoring a pair
with them therefore requires an adaptation, and this script makes the adaptation
explicit rather than burying it, emitting three readings per rival:

    <rival>_a     score of A alone
    <rival>_b     score of B alone            <- what the model was actually trained for
    <rival>_cat   score of the concatenation A | 1s gap | B
    <rival>_rel   cat - (a + b)/2             <- the junction-isolating contrast

`_rel` is the fair relational reading: it removes the per-clip quality both models are
built to measure and leaves what the *joint* clip scores over and above its parts. Report
`_cat` and `_rel` together; `_b` is the disclosure column showing what a plain absolute
metric sees. This is a documented STAND-IN adaptation in the same sense as `scs` and
`fad_proxy` in `score_baselines.py`, and must carry the same caveat sentence in the text.

The concatenation replicates `mf_probe.concat_clips` (head window, B RMS-matched to A, 1s
gap, peak-safe rescale) so the rivals see the same construction S sees. Verified against it
on real pairs: sample-for-sample equal to within 3.05e-05 = 1/32768, i.e. the 16-bit
quantisation of the intermediate wav that mf_probe writes and this script does not need.
If that function changes, change `concat_ab` here too -- they are deliberately duplicated
so this script needs no import from the MF group and runs in a bare torch venv.

Environment: NOT the DSP env and NOT `mfenv`. Both rivals want torch >= 2.x; build a
throwaway venv (the COCOLA pattern). On the AutoDL box this is a GPU job -- MERT-330M and
MuQ-large over ~5.7k forward passes is hours on CPU, minutes on the 4090:

    python3.11 -m venv ~/rivals/venv
    ~/rivals/venv/bin/pip install torch torchaudio librosa soundfile transformers \
        huggingface_hub laion_clap muq hydra-core omegaconf safetensors
    git clone https://github.com/yonghyunk1m/TuneJury ~/rivals/TuneJury   # ckpt in-repo
    git clone https://github.com/ASLP-lab/SongEval  ~/rivals/SongEval     # ckpt in-repo

    ~/rivals/venv/bin/python score_rivals.py \
        --manifest perturbations/pairs_manifest.json \
        --rivals tunejury,songeval \
        --tunejury-repo ~/rivals/TuneJury --songeval-repo ~/rivals/SongEval \
        --out results/baselines/rival_scores.json \
        --csv results/baselines/rival_scores.csv

Output is the `score_baselines.py` JSON contract -- {"meta": ..., "scores": {pair_id:
{metric: float}}} -- so `compute_S_value.py --baselines` and `mos_delta_r.py --rivals`
consume it with no conversion. The CSV carries the manifest metadata columns that
`analyze_cocola.py --rival` needs for the per-family drop table.

Resumable: an existing --out is reloaded and only missing (pair, metric) work is done.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

LOG = logging.getLogger("rivals")

SR = 44100          # concat construction rate; mf_probe.SR. Backends resample internally.
GAP_S = 1.0         # drift_v4.GAP_S -- the 1s gap is load-bearing, see scripts/README.md
ROLES = ("a", "b", "cat")


# --------------------------------------------------------------------------- audio
def load_head(path: Path, secs: float) -> np.ndarray | None:
    """First `secs` of the file, mono at SR. `head` matches the validated S protocol:
    on the matrix and the rerank candidates A is already ~secs long, so head == tail."""
    try:
        import librosa
    except ImportError:                 # audiobox venv: see _load_head_sf
        return _load_head_sf(path, secs)
    try:
        y, _ = librosa.load(str(path), sr=SR, mono=True, duration=secs)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not load %s: %s", path.name, exc)
        return None
    return y if y.size else None


def _load_head_sf(path: Path, secs: float) -> np.ndarray | None:
    """load_head without librosa, for a venv that cannot have it.

    librosa needs numba, which has no wheel for this box's python/arch beside torch 2.2,
    so the audiobox venv is built without it. soundfile reads the same head window and
    torchaudio does the same polyphase resample; the matrix audio is already at SR, so
    on the perturbation set the resample is a no-op and the two paths agree exactly."""
    import soundfile as sf  # noqa: PLC0415
    try:
        info = sf.info(str(path))
        y, sr = sf.read(str(path), frames=int(secs * info.samplerate),
                        dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("could not load %s: %s", path.name, exc)
        return None
    y = y.mean(axis=1)
    if sr != SR:
        import torch, torchaudio  # noqa: PLC0415
        y = torchaudio.functional.resample(torch.from_numpy(y), sr, SR).numpy()
    return y if y.size else None


def concat_ab(a: np.ndarray, b: np.ndarray, gap_s: float = GAP_S) -> np.ndarray:
    """A | gap | B with B RMS-matched to A, peak-safe. Mirrors mf_probe.concat_clips."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ra = float(np.sqrt(np.mean(a ** 2))) + 1e-8
    rb = float(np.sqrt(np.mean(b ** 2))) + 1e-8
    b = b * (ra / rb)
    parts = [a]
    if gap_s > 0:
        parts.append(np.zeros(int(gap_s * SR), dtype=a.dtype))
    parts.append(b)
    cat = np.concatenate(parts).astype("float32")
    peak = float(np.max(np.abs(cat)))
    if peak > 1.0:                      # rescale rather than clip: clipping IS distortion,
        cat = cat / peak                # and distortion is a perturbation family here
    return cat


# --------------------------------------------------------------------------- backends
class Backend:
    """A rival metric. `score(wav, sr) -> {dim: float}`; column = name[_dim]_role."""

    name = "?"
    dims: tuple[str, ...] = ()

    def load(self, args) -> None:
        raise NotImplementedError

    def score(self, wav: np.ndarray, sr: int) -> dict[str, float]:
        raise NotImplementedError

    def columns(self) -> list[str]:
        stems = [self.name] if not self.dims else [f"{self.name}_{d}" for d in self.dims]
        return [f"{s}_{r}" for s in stems for r in (*ROLES, "rel")]

    def stem(self, dim: str | None) -> str:
        return self.name if not self.dims else f"{self.name}_{dim}"


class TuneJuryBackend(Backend):
    """TuneJury reward. Empty prompt by default: the released checkpoint's own
    recommendation for out-of-distribution prompt formats (paper section 4.2), and our
    pairs have no arena-style request text. Raw head output, no sigmoid -- unbounded,
    which is fine for AUC/Spearman/correlation but means it is NOT in [0,1] and must not
    be fed to anything expecting the dimension contract."""

    name = "tunejury"

    def load(self, args) -> None:
        repo = Path(args.tunejury_repo).expanduser()
        if not (repo / "tunejury").is_dir():
            raise SystemExit(f"--tunejury-repo {repo} is not a TuneJury clone")
        sys.path.insert(0, str(repo))
        from tunejury.score import Scorer  # noqa: PLC0415

        ckpt = Path(args.tunejury_ckpt).expanduser() if args.tunejury_ckpt else \
            repo / "checkpoints" / "tunejury.pt"
        if not ckpt.is_file():
            raise SystemExit(f"TuneJury checkpoint not found: {ckpt}")

        # The 2.2GB LAION-CLAP-Music checkpoint. Left implicit, TuneJury looks for it
        # NEXT TO the head checkpoint and otherwise downloads it there from a hardcoded
        # huggingface.co URL -- which HF_ENDPOINT does not redirect, and which lands on
        # whichever volume the repo is on. Pass --clap-ckpt to put it on a big disk
        # (AutoDL: / is 30G, so this belongs on /root/autodl-tmp).
        clap = None
        if args.clap_ckpt:
            clap = Path(args.clap_ckpt).expanduser()
            if not clap.is_file():
                raise SystemExit(f"--clap-ckpt not found: {clap}")
            LOG.info("using LAION-CLAP-Music checkpoint at %s", clap)
        LOG.info("loading TuneJury head %s (+ frozen CLAP-Music, MERT-v1-330M)", ckpt.name)
        self.scorer = Scorer.from_pretrained(str(ckpt), device=args.device,
                                             clap_ckpt_path=str(clap) if clap else None)
        self.prompt = args.prompt

    def score(self, wav: np.ndarray, sr: int) -> dict[str, float]:
        return {"": float(self.scorer.score_waveform(wav, sr, text=self.prompt))}


class SongEvalBackend(Backend):
    """SongEval aesthetics head over MuQ-large hidden layer 6.

    Mirrors `Synthesizer.setup()/handle()` in SongEval's eval.py, but on an in-memory
    waveform: their handle() only takes a path, and round-tripping ~5.7k temp wavs is
    pointless I/O. The two model calls and the dimension order are copied verbatim.

    Caveat for the write-up: SongEval is trained on FULL SONGS with vocals, and two of
    its five dimensions (Naturalness of vocal breathing, Memorability) are undefined on
    a 17s instrumental concat. Coherence/Musicality/Clarity are the readable ones, and
    even those are out of distribution on this clip length. Report as a stand-in."""

    name = "songeval"
    dims = ("coherence", "musicality", "memorability", "clarity", "naturalness")
    MUQ_SR = 24000

    def load(self, args) -> None:
        repo = Path(args.songeval_repo).expanduser()
        ckpt = repo / "ckpt" / "model.safetensors"
        if not ckpt.is_file():
            raise SystemExit(f"--songeval-repo {repo}: ckpt/model.safetensors not found")
        import torch  # noqa: PLC0415
        from hydra.utils import instantiate  # noqa: PLC0415
        from muq import MuQ  # noqa: PLC0415
        from omegaconf import OmegaConf  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415

        sys.path.insert(0, str(repo))          # config.yaml refers to classes in model.py
        self.torch = torch
        # auto-detect like TuneJury's Scorer does, so omitting --device on the box does
        # not silently put SongEval on the CPU while TuneJury runs on the GPU
        self.device = torch.device(args.device
                                   or ("cuda" if torch.cuda.is_available() else "cpu"))
        cfg = OmegaConf.load(repo / "config.yaml")
        model = instantiate(cfg.generator).to(self.device).eval()
        model.load_state_dict(load_file(str(ckpt), device="cpu"), strict=False)
        self.model = model
        LOG.info("loading MuQ-large-msd-iter for SongEval")
        self.muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(self.device).eval()

    def score(self, wav: np.ndarray, sr: int) -> dict[str, float]:
        import librosa  # noqa: PLC0415
        if sr != self.MUQ_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.MUQ_SR)
        with self.torch.no_grad():
            audio = self.torch.tensor(wav).unsqueeze(0).to(self.device)
            hidden = self.muq(audio, output_hidden_states=True)["hidden_states"][6]
            s = self.model(hidden).squeeze(0)
        return {d: float(s[i].item()) for i, d in enumerate(self.dims)}


class AudioboxBackend(Backend):
    """Audiobox Aesthetics (Tjandra et al., arXiv 2502.05139): four no-reference
    aesthetic axes, CE (content enjoyment), CU (content usefulness), PC (production
    complexity) and PQ (production quality), predicted by a WavLM-style encoder with one
    head per axis over 10s windows, chunk-weighted for longer input.

    Added because the 2026-08-27 review asked why the aesthetics wave is represented by
    SongEval alone. Like SongEval and TuneJury it is ABSOLUTE, so it gets the same
    `_a`/`_b`/`_cat`/`_rel` treatment and the same stand-in caveat. Four axes make
    `audiobox_rel` a 4-d feature, which is the fairest reading available to it and the
    one that matches the profile's dimension.

    Weights are `facebook/audiobox-aesthetics` (pip `audiobox-aesthetics`), downloaded to
    the HF cache on first use. CPU is fine: about 0.8s per 8s side."""

    name = "audiobox"
    dims = ("ce", "cu", "pc", "pq")
    AXES = ("CE", "CU", "PC", "PQ")

    def load(self, args) -> None:
        import torch  # noqa: PLC0415
        from audiobox_aesthetics.infer import initialize_predictor  # noqa: PLC0415
        self.torch = torch
        LOG.info("loading Audiobox Aesthetics%s",
                 f" from {args.audiobox_ckpt}" if args.audiobox_ckpt else "")
        self.predictor = initialize_predictor(args.audiobox_ckpt)

    def score(self, wav: np.ndarray, sr: int) -> dict[str, float]:
        x = self.torch.from_numpy(np.ascontiguousarray(wav, dtype="float32")).unsqueeze(0)
        out = self.predictor.forward([{"path": x, "sample_rate": sr}])[0]
        return {d: float(out[a]) for d, a in zip(self.dims, self.AXES)}


BACKENDS = {"tunejury": TuneJuryBackend, "songeval": SongEvalBackend,
            "audiobox": AudioboxBackend}


# --------------------------------------------------------------------------- driver
def resolve(path_str: str, audio_root: Path, audio_dir: Path | None) -> Path:
    """Manifest paths are repo-relative (matrix + rerank manifests both are). --audio-dir
    overrides the directory, matching score_separability.py's flag."""
    p = Path(path_str)
    if audio_dir is not None:
        return audio_dir / p.name
    return p if p.is_absolute() else audio_root / p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "perturbations/pairs_manifest.json")
    ap.add_argument("--audio-root", type=Path, default=ROOT)
    ap.add_argument("--audio-dir", type=Path, default=None)
    ap.add_argument("--rivals", default="tunejury",
                    help="comma list from %s" % ",".join(BACKENDS))
    ap.add_argument("--out", type=Path, default=ROOT / "results/baselines/rival_scores.json")
    ap.add_argument("--csv", type=Path, default=None,
                    help="also write the per-pair CSV with manifest columns "
                         "(what analyze_cocola.py --rival reads)")
    ap.add_argument("--secs", type=float, default=8.0, help="clip length per side (S protocol: 8)")
    ap.add_argument("--gap-s", type=float, default=GAP_S)
    ap.add_argument("--tunejury-repo", default="~/rivals/TuneJury")
    ap.add_argument("--tunejury-ckpt", default=None, help="default: <repo>/checkpoints/tunejury.pt")
    ap.add_argument("--clap-ckpt", default=None,
                    help="LAION-CLAP-Music .pt (2.2GB). Default: next to the head "
                         "checkpoint, downloading it there if missing. Point this at a "
                         "big volume (e.g. /root/autodl-tmp/...) to keep it off a small /")
    ap.add_argument("--songeval-repo", default="~/rivals/SongEval")
    ap.add_argument("--audiobox-ckpt", default=None,
                    help="Audiobox Aesthetics checkpoint; default: fetch from the hub")
    ap.add_argument("--prompt", default="", help="TuneJury text branch; '' = zero text vector")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--perturbations", default="", help="comma list of families to keep")
    ap.add_argument("--flush-every", type=int, default=25)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    names = [s.strip() for s in args.rivals.split(",") if s.strip()]
    bad = [n for n in names if n not in BACKENDS]
    if bad:
        ap.error(f"unknown rival(s) {bad}; choose from {list(BACKENDS)}")
    backends = [BACKENDS[n]() for n in names]
    wanted = [c for b in backends for c in b.columns()]

    doc = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs, meta = doc["pairs"], doc.get("meta", {})
    if args.perturbations:
        keep = {s.strip() for s in args.perturbations.split(",")}
        pairs = [p for p in pairs if p["perturbation"] in keep]

    cache: dict[str, dict] = {}
    if args.out.is_file():
        prev = json.loads(args.out.read_text())
        prev_meta = prev.get("meta", {})
        for k in ("secs", "gap_s", "prompt"):
            if k in prev_meta and prev_meta[k] != getattr(args, k):
                ap.error(f"{args.out} was written with {k}={prev_meta[k]!r} but --{k}="
                         f"{getattr(args, k)!r}; use a separate --out rather than mixing protocols")
        cache = {k: dict(v) for k, v in prev.get("scores", {}).items()}
        LOG.info("resuming: %d pairs already in %s", len(cache), args.out)

    todo = [p for p in pairs if any(c not in cache.get(p["pair_id"], {}) for c in wanted)]
    if args.limit:
        todo = todo[:args.limit]
    LOG.info("%d/%d pairs need work; metrics=%s", len(todo), len(pairs), wanted)
    if not todo:
        LOG.info("nothing to do")
    else:
        for b in backends:
            b.load(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        args.out.write_text(json.dumps(
            {"meta": {**meta, "baselines": wanted, "rivals": names,
                      "secs": args.secs, "gap_s": args.gap_s, "prompt": args.prompt,
                      "protocol": "A | gap | B, head window, B RMS-matched to A, peak-safe "
                                  "(mirrors mf_probe.concat_clips)",
                      "readings": {"_a": "A alone", "_b": "B alone (what the model is "
                                                          "trained for)",
                                   "_cat": "the concatenation", "_rel": "cat - (a+b)/2"},
                      "caveat": "ABSOLUTE metrics adapted to a relational contract; "
                                "stand-in in the sense of scs/fad_proxy, not in [0,1]"},
             "scores": cache}, indent=2), encoding="utf-8")

    # A repeats ~21x across conditions, so cache its per-backend scores by source path.
    a_cache: dict[tuple[str, str], dict[str, float]] = {}
    n_ok = n_bad = 0
    t0 = time.time()

    for i, p in enumerate(todo, 1):
        pa = resolve(p["ref_path"], args.audio_root, args.audio_dir)
        pb = resolve(p["cand_path"], args.audio_root, args.audio_dir)
        wa, wb = load_head(pa, args.secs), load_head(pb, args.secs)
        if wa is None or wb is None:
            n_bad += 1
            continue
        cat = concat_ab(wa, wb, args.gap_s)
        entry = cache.setdefault(p["pair_id"], {})

        for b in backends:
            if all(c in entry for c in b.columns()):
                continue
            key = (b.name, str(pa))
            if key not in a_cache:
                a_cache[key] = b.score(wa, SR)
            sa = a_cache[key]
            sb = b.score(wb, SR)
            sc = b.score(cat, SR)
            for dim in (b.dims or [None]):
                d = dim or ""
                stem = b.stem(dim)
                entry[f"{stem}_a"] = sa[d]
                entry[f"{stem}_b"] = sb[d]
                entry[f"{stem}_cat"] = sc[d]
                entry[f"{stem}_rel"] = sc[d] - 0.5 * (sa[d] + sb[d])
        n_ok += 1
        if i % args.flush_every == 0:
            flush()
            rate = (time.time() - t0) / i
            LOG.info("  %d/%d  (%.2fs/pair, eta %.0f min)", i, len(todo), rate,
                     rate * (len(todo) - i) / 60)

    flush()
    LOG.info("[wrote %s]  scored=%d  skipped=%d", args.out, n_ok, n_bad)

    if args.csv:
        write_csv(args.csv, pairs, cache, wanted)
    return 0


def write_csv(path: Path, pairs: list[dict], cache: dict[str, dict],
              cols: list[str]) -> None:
    """Per-pair CSV with the manifest metadata analyze_cocola.py --rival needs
    (pair_id, source_id, genre, perturbation, magnitude...)."""
    head = ["pair_id", "source_id", "genre", "perturbation", "magnitude", "magnitude_unit"]
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=head + cols)
        w.writeheader()
        for p in pairs:
            got = cache.get(p["pair_id"])
            if not got or any(c not in got for c in cols):
                continue
            w.writerow({**{k: p.get(k, "") for k in head},
                        **{c: got[c] for c in cols}})
            n += 1
    LOG.info("[wrote %s]  %d complete rows", path, n)


if __name__ == "__main__":
    raise SystemExit(main())
