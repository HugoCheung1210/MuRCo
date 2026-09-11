#!/usr/bin/env python3
"""Prepare a BLINDED A/B listening test from the rerank C-winner/C-loser stimuli.

Turns `mos_stimuli.json` (written by `rerank_bestofn.py analyze`: the C-winner and
C-loser candidate per seed, same seed + same 5-8 s inpaint region) into a rater-ready
package for the greenlit human MOS study's iterative block (STIMULI-SCOPE).

Each seed becomes ONE trial. For the trial we build two stimuli, one per candidate,
each = reference context A | 1 s gap | candidate B, peak-normalized at 48 kHz -- i.e.
the SAME A|gap|B signal the metric scored (Conventions: the 1 s gap is load-bearing),
so the rater hears exactly the transition C judged. The winner and loser are randomly
assigned to option "A" / "B" per trial (reproducible via --shuffle-seed) so the rater
cannot tell which one C preferred: the file names carry no leakage.

Outputs under RERANK_DIR/mos_ab/:
  audio/tNN_optA.wav, tNN_optB.wav   the two blinded stitched stimuli per trial
  trials.csv                         PUBLIC sheet (trial_id, seed, opt files, question) -- NO answer
  key.csv                            PRIVATE blinding key (option -> winner/loser, pair_id, C) -- do NOT show raters
  index.html                         self-contained rater UI (players + forced choice + 1-5 MOS), exports responses to JSON

Run in the DSP env (needs the candidate wavs on disk; same env as score_separability):
    RERANK_DIR=results/rerank_ace python scripts/rerank/prepare_mos_ab.py
    RERANK_DIR=results/rerank_ace python scripts/rerank/prepare_mos_ab.py --no-stitch   # rate the bare B candidate, no A context
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path

import numpy as np

EPS = 1e-8
SR = 48000
GAP_S = 1.0                  # the load-bearing 1 s gap (matches scoring; see Conventions)
RERANK_DIR = Path(os.environ.get("RERANK_DIR", "results/rerank"))


def _load_win(path: str, sf, librosa):
    """Load a stimulus window at SR, mono."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype("float32")


def _peaknorm(y: np.ndarray) -> np.ndarray:
    return (y / (np.max(np.abs(y)) + EPS) * 0.95).astype("float32")


def _stitch(ref: np.ndarray, cand: np.ndarray, gap_s: float) -> np.ndarray:
    """A | gap | B, peak-normalized as a whole -- the signal the metric scored."""
    gap = np.zeros(int(round(gap_s * SR)), dtype="float32")
    return _peaknorm(np.concatenate([ref, gap, cand]))


def _cand_full(pid: str) -> Path:
    """pair_id 's07::cand03' -> RERANK_DIR/candidates/s07_cand03.wav (full 30 s clip)."""
    sid, _, cand = pid.partition("::cand")
    return RERANK_DIR / "candidates" / f"{sid}_cand{cand}.wav"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="mos_ab", help="subdir under RERANK_DIR (default: mos_ab)")
    ap.add_argument("--shuffle-seed", type=int, default=20260721,
                    help="RNG seed for winner/loser -> option A/B assignment (reproducible blinding)")
    ap.add_argument("--gap-s", type=float, default=GAP_S, help="A|gap|B gap seconds (default 1.0)")
    ap.add_argument("--seek", action="store_true",
                    help="show the native transport so the rater can scrub/replay any part "
                         "of the clip. Off by default: for the real study every rater should "
                         "hear the same thing the same way. Useful for self-piloting.")
    ap.add_argument("--no-stitch", action="store_true",
                    help="present the bare candidate B window only (no reference context / gap)")
    ap.add_argument("--full", action="store_true",
                    help="use the full 30 s candidates/ clips per option (winner vs loser), no A "
                         "context / stitch -- for when only candidates/ is synced, not audio/ windows")
    args = ap.parse_args()
    if args.full:
        args.no_stitch = True            # full clips have no reference window to stitch

    import soundfile as sf
    import librosa

    stim_p = RERANK_DIR / "mos_stimuli.json"
    if not stim_p.is_file():
        raise SystemExit(f"missing {stim_p} -- run `rerank_bestofn.py analyze` first.")
    stimuli = json.loads(stim_p.read_text(encoding="utf-8")).get("stimuli", [])
    if not stimuli:
        raise SystemExit(f"{stim_p} has no stimuli.")

    out_dir = RERANK_DIR / args.out
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.shuffle_seed)

    if args.full:
        QUESTION = ("Each version (A and B) is a full track in which a middle section was "
                    "regenerated; A and B share the same surrounding music. Which version sounds "
                    "more coherent -- where the regenerated section blends most seamlessly with "
                    "the rest in key, style, tempo, and instrumentation?")
    elif args.no_stitch:
        # No intro and no gap in this mode: the clip IS the inpaint window, so the seam
        # into and out of the regenerated middle is what the rater is judging.
        QUESTION = ("Each version (A and B) is a short excerpt whose middle section was "
                    "regenerated; A and B share the same music before and after it. Which "
                    "version holds together better -- where the middle fits its surroundings "
                    "most naturally in key, style, tempo, and instrumentation?")
    else:
        QUESTION = ("You will hear a short musical intro, then a 1-second gap, then a continuation. "
                    "Two versions (A and B) share the same intro. Which continuation follows the "
                    "intro more coherently -- staying in the same key, style, tempo, and instrumentation?")

    trials, key_rows, missing = [], [], 0
    for i, s in enumerate(sorted(stimuli, key=lambda x: x["seed"]), start=1):
        tid = f"t{i:02d}"
        win, los = s["c_winner"], s["c_loser"]
        # blind: randomly map {winner, loser} -> {optA, optB}
        roles = [("winner", win), ("loser", los)]
        if rng.integers(0, 2):
            roles = roles[::-1]
        opt = {}
        ok = True
        for label, (role, cand) in zip(("A", "B"), roles):
            cand_p = _cand_full(cand.get("pair_id", "")) if args.full \
                else Path(cand.get("cand_path", ""))
            ref_p = Path(cand.get("ref_path", ""))
            dst = audio_dir / f"{tid}_opt{label}.wav"
            if cand_p.is_file() and (args.no_stitch or ref_p.is_file()):
                cw = _load_win(str(cand_p), sf, librosa)
                if args.no_stitch:
                    sf.write(dst, _peaknorm(cw), SR)
                else:
                    rw = _load_win(str(ref_p), sf, librosa)
                    sf.write(dst, _stitch(rw, cw, args.gap_s), SR)
            else:
                ok = False
                missing += 1
            opt[label] = {"role": role, "pair_id": cand.get("pair_id", ""),
                          "C": cand.get("C", float("nan")), "file": f"audio/{dst.name}"}
        if not ok:
            print(f"  ! {tid} ({s['seed']}): source wav(s) missing -- referencing expected filenames only")
        trials.append({"trial_id": tid, "seed": s["seed"],
                       "opt_A_file": opt["A"]["file"], "opt_B_file": opt["B"]["file"],
                       "question": QUESTION})
        key_rows.append({"trial_id": tid, "seed": s["seed"],
                         "opt_A_role": opt["A"]["role"], "opt_A_pair_id": opt["A"]["pair_id"],
                         "opt_A_C": f"{opt['A']['C']:.4f}",
                         "opt_B_role": opt["B"]["role"], "opt_B_pair_id": opt["B"]["pair_id"],
                         "opt_B_C": f"{opt['B']['C']:.4f}",
                         "winner_option": "A" if opt["A"]["role"] == "winner" else "B"})

    # public trial sheet (no answer)
    with (out_dir / "trials.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        w.writeheader(); w.writerows(trials)
    # private blinding key
    with (out_dir / "key.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        w.writeheader(); w.writerows(key_rows)

    _write_html(out_dir / "index.html", trials, QUESTION, args.no_stitch, args.full,
                seek=args.seek)

    print(f"wrote {len(trials)} trials -> {out_dir}/")
    print(f"  audio/  trials.csv  key.csv(PRIVATE)  index.html")
    if missing:
        print(f"  ! {missing} option(s) had no source wav on disk -- run this where the "
              f"candidate audio lives (e.g. sync {RERANK_DIR}/audio/ first).")
    return 0


def _write_html(path: Path, trials, question: str, no_stitch: bool, full: bool = False,
                seek: bool = False) -> None:
    """Self-contained rater UI: one trial at a time, big A/B play + choose, shuffled order,
    first-N-seconds play limiter, keyboard shortcuts, JSON export. Local file -- must sit next
    to audio/ to load the wavs (a hosted page can't reach local files)."""
    trials_json = json.dumps([{"id": t["trial_id"], "a": t["opt_A_file"], "b": t["opt_B_file"]}
                              for t in trials])
    if full:
        stimdesc = "Each clip is a full ~30 s track with a regenerated middle section."
    elif not no_stitch:
        stimdesc = "Each clip is: a short intro, 1 s silence, then a continuation."
    else:
        stimdesc = "Each clip is a continuation only (no intro context)."
    default_limit = 12 if full else 0            # full clips: default to the useful early window

    doc = _HTML_TEMPLATE
    doc = doc.replace("__QUESTION__", html.escape(question))
    doc = doc.replace("__STIMDESC__", html.escape(stimdesc))
    doc = doc.replace("__TRIALS_JSON__", trials_json)
    doc = doc.replace("__DEFAULT_LIMIT__", str(default_limit))
    # `controls` renders the native transport (scrub bar, replay). Without it the element
    # is headless and playback is driven only by the A/B buttons.
    doc = doc.replace("__AUDIO_ATTRS__", "controls" if seek else "")
    path.write_text(doc, encoding="utf-8")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>A/B coherence listening test</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 760px; margin: 0 auto;
         padding: 1rem; line-height: 1.5; }
  h1 { font-size: 1.4rem; margin: .3rem 0; }
  audio[controls] { display: block; width: 100%; margin: .6rem 0 .2rem; }
  .muted { opacity: .7; font-size: .9rem; }
  #setup label, #bar label { display: inline-flex; align-items: center; gap: .4rem; }
  input, button, select { font-size: 1rem; }
  input[type=text], input[type=number] { padding: .35rem .5rem; border-radius: 6px;
         border: 1px solid #8886; background: transparent; color: inherit; }
  input[type=number] { width: 5rem; }
  #bar { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
         padding: .6rem 0; border-bottom: 1px solid #8884; margin-bottom: 1rem; }
  #prog { height: 6px; background: #8883; border-radius: 3px; overflow: hidden; flex: 1 1 120px; }
  #prog > div { height: 100%; width: 0; background: #4f8cff; transition: width .2s; }
  .btn { padding: .55rem 1rem; border-radius: 8px; border: 1px solid #8886;
         background: #8881; color: inherit; cursor: pointer; }
  .btn:hover { background: #8883; }
  .options { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.2rem 0; }
  .opt { border: 1px solid #8886; border-radius: 12px; padding: 1.2rem 1rem; text-align: center; }
  .opt h2 { margin: 0 0 .8rem; font-size: 2rem; }
  .play { width: 100%; padding: 1.1rem; font-size: 1.1rem; border-radius: 10px;
          border: 1px solid #4f8cff; background: #4f8cff22; color: inherit; cursor: pointer; }
  .play.playing { background: #4f8cff; color: #fff; }
  .scale { display: flex; align-items: center; gap: .8rem; margin-top: .6rem; }
  .scale .end { flex: 0 0 auto; font-size: .85rem; opacity: .8; width: 6rem; text-align: center; }
  .pts { display: grid; grid-template-columns: repeat(5, 1fr); gap: .6rem; flex: 1 1 auto; }
  .pts button { padding: 1rem 0; border-radius: 10px; border: 1px solid #8886;
                background: #8881; color: inherit; cursor: pointer; font-weight: 700; font-size: 1.1rem; }
  .pts button:hover { background: #8883; }
  .pts button.sel { background: #2ecc71; border-color: #2ecc71; color: #fff; }
  .center { text-align: center; }
  kbd { border: 1px solid #8886; border-radius: 4px; padding: 0 .35rem; font-size: .8rem; }
  #nav { display: flex; justify-content: space-between; align-items: center; margin-top: 1.2rem; }
  #done { display: none; text-align: center; padding: 2rem 0; }
  #done.show { display: block; }
  #card.hide { display: none; }
</style></head><body>
<h1>Which is more coherent?</h1>
<p class="muted">__QUESTION__<br>__STIMDESC__ Please use headphones.</p>

<div id="setup">
  <label>Rater ID <input type="text" id="rater" placeholder="e.g. R01" autocomplete="off"></label>
  <button class="btn" id="start">Start</button>
</div>

<div id="app" style="display:none">
  <div id="bar">
    <span id="count" class="muted"></span>
    <div id="prog"><div></div></div>
    <label class="muted">play first <input type="number" id="limit" min="0" step="1" value="__DEFAULT_LIMIT__"> s <span class="muted">(0 = whole clip)</span></label>
  </div>

  <div id="card">
    <div class="options">
      <div class="opt">
        <h2>A</h2>
        <button class="play" id="playA">▶ Play A <span class="muted">(a)</span></button>
      </div>
      <div class="opt">
        <h2>B</h2>
        <button class="play" id="playB">▶ Play B <span class="muted">(b)</span></button>
      </div>
    </div>
    <div class="scale">
      <span class="end">A more<br>coherent</span>
      <div class="pts">
        <button data-v="1">1</button>
        <button data-v="2">2</button>
        <button data-v="3">3</button>
        <button data-v="4">4</button>
        <button data-v="5">5</button>
      </div>
      <span class="end">B more<br>coherent</span>
    </div>
    <p class="muted center">1 = A clearly · 2 = A slightly · 3 = about the same · 4 = B slightly · 5 = B clearly</p>
    <div id="nav">
      <button class="btn" id="prev">← Back <kbd>p</kbd></button>
      <span class="muted">a/b play · 1-5 rate · n next · p back</span>
      <button class="btn" id="next">Next → <kbd>n</kbd></button>
    </div>
  </div>

  <div id="done">
    <h2>All done — thank you!</h2>
    <p class="muted" id="doneinfo"></p>
    <button class="btn" id="save">Download responses (JSON)</button>
    <p class="muted" id="status"></p>
  </div>
</div>

<audio id="au" preload="none" __AUDIO_ATTRS__></audio>
<script>
(function () {
  var TRIALS = __TRIALS_JSON__;
  for (var i = TRIALS.length - 1; i > 0; i--) {           // shuffle order (kill position/order bias)
    var j = Math.floor(Math.random() * (i + 1));
    var t = TRIALS[i]; TRIALS[i] = TRIALS[j]; TRIALS[j] = t;
  }
  var idx = 0, resp = {}, rater = "", started = null, playingWhich = null;
  var au = document.getElementById("au");
  var $ = function (id) { return document.getElementById(id); };

  function lim() { return parseFloat($("limit").value) || 0; }
  au.addEventListener("timeupdate", function () {
    if (lim() > 0 && au.currentTime >= lim()) { au.pause(); }
  });
  au.addEventListener("pause", function () { setPlaying(null); });
  au.addEventListener("ended", function () { setPlaying(null); });

  function setPlaying(w) {
    playingWhich = w;
    $("playA").classList.toggle("playing", w === "A");
    $("playB").classList.toggle("playing", w === "B");
  }
  function play(which) {
    var t = TRIALS[idx], src = which === "A" ? t.a : t.b;
    if (playingWhich === which && !au.paused) { au.pause(); return; }
    au.src = src; au.currentTime = 0;
    au.play().then(function () { setPlaying(which); })
             .catch(function () { $("status").textContent = "playback blocked — serve the folder over http (see note)"; });
  }
  function choose(v) {
    resp[TRIALS[idx].id] = v;                          // v is 1..5
    paintChoice();
    setTimeout(next, 180);
  }
  function paintChoice() {
    var v = resp[TRIALS[idx].id];
    document.querySelectorAll(".pts button").forEach(function (b) {
      b.classList.toggle("sel", String(v) === b.dataset.v);
    });
  }
  function render() {
    au.pause(); setPlaying(null);
    $("count").textContent = "Trial " + (idx + 1) + " / " + TRIALS.length;
    $("prog").firstElementChild.style.width = (100 * idx / TRIALS.length) + "%";
    paintChoice();
  }
  function next() {
    if (idx < TRIALS.length - 1) { idx++; render(); }
    else { finish(); }
  }
  function prev() { if (idx > 0) { idx--; render(); } }
  function finish() {
    au.pause();
    $("card").classList.add("hide");
    $("done").classList.add("show");
    var n = Object.keys(resp).length;
    $("doneinfo").textContent = n + " of " + TRIALS.length + " trials answered.";
    $("prog").firstElementChild.style.width = "100%";
  }

  $("start").addEventListener("click", function () {
    rater = $("rater").value.trim();
    if (!rater) { $("rater").focus(); return; }
    started = new Date().toISOString();
    $("setup").style.display = "none";
    $("app").style.display = "block";
    render();
  });
  $("playA").addEventListener("click", function () { play("A"); });
  $("playB").addEventListener("click", function () { play("B"); });
  document.querySelectorAll(".pts button").forEach(function (b) {
    b.addEventListener("click", function () { choose(parseInt(b.dataset.v, 10)); });
  });
  $("next").addEventListener("click", next);
  $("prev").addEventListener("click", prev);

  document.addEventListener("keydown", function (e) {
    if ($("app").style.display === "none" || e.target.tagName === "INPUT") return;
    var k = e.key.toLowerCase();
    if (k === "a") play("A"); else if (k === "b") play("B");
    else if (k >= "1" && k <= "5") choose(parseInt(k, 10));
    else if (k === "n") next(); else if (k === "p") prev();
  });

  $("save").addEventListener("click", function () {
    var out = TRIALS.map(function (t) {
      return { trial_id: t.id, rating: (resp[t.id] === undefined ? "" : resp[t.id]) };
    });
    var payload = { rater: rater, started: started, finished: new Date().toISOString(),
                    n_trials: TRIALS.length, responses: out };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob), a = document.createElement("a");
    a.href = url; a.download = "responses_" + rater + ".json"; a.click();
    URL.revokeObjectURL(url);
    $("status").textContent = "saved responses_" + rater + ".json";
  });
})();
</script>
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
