#!/usr/bin/env python
"""Audition each too-loud source at several levels and pick one PER SOURCE. DSP env.

A single global attenuation is a blunt instrument: some tracks are marginal and some are
genuinely painful, and the same cut over-quiets the first while under-treating the second.
Too big a cut has its own costs — the clip becomes much quieter than its neighbours (so a
rater turns the volume up and gets hit by the next full-level clip), and fine conditions
like lowpass:2000 or pitch_shift:-1 get harder to judge, which is exactly where the scale
needs resolution.

So: hear each source at every candidate level, plus a full-level reference clip to judge
the jump against, and choose per source.

Its working directory (results/mos/level_review) is kept in the working repository and
is NOT shipped in the release: the levels it produces are already baked into the
rendered stimuli, so nothing downstream reads it.

Usage:
  python scripts/mos/review_levels.py                       # render + build the page
  open results/mos/level_review/compare.html            # choose, download levels.json
  python scripts/mos/review_levels.py --apply results/mos/level_review/levels.json
Then: render_mos_stimuli.py --attenuate-map results/mos/level_review/attenuation_map.json
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
try:
    import _paths  # noqa: F401  -- puts every scripts/<group>/ on sys.path
except ImportError:
    pass  # flat layout (e.g. the GPU box): siblings are already importable

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Level review — how much to reduce each loud source</title>
<style>
 body{font-family:system-ui,-apple-system,sans-serif;max-width:48rem;margin:0 auto;
      padding:1.5rem;line-height:1.5;color:#111}
 .card{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:.9rem 0}
 .row{display:grid;grid-template-columns:7rem 1fr;gap:.6rem;align-items:center;margin:.3rem 0}
 audio{width:100%%}
 .btns{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.6rem}
 button{font-size:.95rem;padding:.45rem .9rem;border-radius:8px;border:1px solid #888;
        background:#f6f6f6;cursor:pointer}
 .sel{background:#e8f0fe;border-color:#3367d6;font-weight:600}
 .done{opacity:.6} .prog{color:#666;font-size:.9rem}
 #bar{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #eee;z-index:9}
 .ref{background:#fff8e1;border-color:#f0c36d}
</style></head><body>
<div id="bar"><b>Level review</b> — <span id="count"></span>
 <button onclick="save()">Download choices</button></div>
<div class="card ref"><b>Reference — a normal, unattenuated clip.</b> Set your volume on
this one and leave it. Every choice below is about how far the loud sources should sit
under this level.<audio controls preload="none" src="%(ref)s"></audio></div>
<p class="prog">For each source: play the original, then each reduced level. Pick the
quietest level that is comfortable <b>without</b> making the clip hard to judge — then
replay the reference to check the jump between them is not startling.</p>
<div id="list"></div>
<script>
const SRC = %(sources)s;
const LEVELS = %(levels)s;
const choice = {};
function render(){
 document.getElementById("list").innerHTML = SRC.map(s=>`
  <div class="card ${choice[s.id]?'done':''}" id="c_${s.id}">
   <b>${s.id}</b> <span class="prog">${s.condition}${choice[s.id]?` → keeping ${choice[s.id]}`:''}</span>
   <div class="row"><span class="prog">original</span>
     <audio controls preload="none" src="audio/${s.id}__1.00.mp3"></audio></div>
   ${LEVELS.map(l=>`<div class="row"><span class="prog">${Math.round(l*100)}%% &nbsp;(${(20*Math.log10(l)).toFixed(1)} dB)</span>
     <audio controls preload="none" src="audio/${s.id}__${l.toFixed(2)}.mp3"></audio></div>`).join("")}
   <div class="btns">
    ${LEVELS.map(l=>`<button class="${choice[s.id]===l?'sel':''}"
      onclick="pick('${s.id}',${l})">use ${Math.round(l*100)}%%</button>`).join("")}
    <button class="${choice[s.id]===1?'sel':''}" onclick="pick('${s.id}',1)">no change</button>
   </div></div>`).join("");
 document.getElementById("count").textContent =
   Object.keys(choice).length + " / " + SRC.length + " chosen";
}
function pick(id,v){ choice[id]=v; render();
 const el=document.getElementById("c_"+id); if(el) el.scrollIntoView({block:"nearest"}); }
function save(){
 const blob=new Blob([JSON.stringify({levels:choice},null,1)],{type:"application/json"});
 const a=document.createElement("a");
 a.href=URL.createObjectURL(blob); a.download="levels.json"; a.click();
}
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", default="results/mos/source_review/excluded_sources.json")
    ap.add_argument("--index", default="results/mos/stimulus_index.csv")
    ap.add_argument("--plan", default="results/mos/stimuli_plan.json")
    ap.add_argument("--out-dir", default="results/mos/level_review")
    ap.add_argument("--levels", default="0.75,0.5,0.25")
    ap.add_argument("--apply", default=None)
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.apply:
        chosen = json.load(open(args.apply))["levels"]
        m = {k: float(v) for k, v in chosen.items() if float(v) != 1.0}
        p = out / "attenuation_map.json"
        json.dump({"attenuation": m,
                   "note": "per-source gain chosen by ear; 1.0 sources omitted"},
                  open(p, "w"), indent=1)
        from collections import Counter
        print(json.dumps({"sources_attenuated": len(m),
                          "levels_used": dict(Counter(m.values())),
                          "out": str(p)}, indent=1))
        print("\nnext: render_mos_stimuli.py --attenuate-map " + str(p))
        return

    from render_mos_stimuli import build_concat, encode
    levels = [float(x) for x in args.levels.split(",")]
    loud = json.load(open(ROOT / args.review))["attenuate"]
    plan = {r["pair_id"]: r for r in json.load(open(ROOT / args.plan))["stimuli"]}
    index = list(csv.DictReader(open(ROOT / args.index)))
    (out / "audio").mkdir(exist_ok=True)

    srcs = []
    for src in loud:
        hit = next((r for r in index if r["source_id"] == src), None)
        if not hit:
            continue
        p = plan[hit["pair_id"]]
        a, b = str(ROOT / p["ref_path"]), str(ROOT / p["cand_path"])
        for lv in [1.0] + levels:
            cat, _ = build_concat(a, b, 6.0, 1.0, lv)
            encode(cat, out / "audio" / f"{src}__{lv:.2f}.mp3", "mp3")
        srcs.append({"id": src, "condition": hit["condition"]})

    # Reference clip. It must represent NORMAL listening level, so it has to be
    #   - full level (not attenuated),
    #   - from a source the audition marked plainly "ok" -- not harsh, not loud. A
    #     harsh-marked track is the opposite of a calibration reference,
    #   - an unperturbed CONTROL, so the level is not coloured by a lowpass or a
    #     distortion, and
    #   - near the median loudness of the eligible clips rather than an extreme.
    marks = {}
    review_src = ROOT / args.review
    raw = review_src.parent / "source_review.json"
    if raw.is_file():
        marks = json.load(open(raw))["marked"]
    elig = [r for r in index
            if r["attenuated"] == "1.0" and marks.get(r["source_id"]) == "ok"]
    ctrl = [r for r in elig if r["condition"] == "control"] or elig
    if not ctrl:
        raise SystemExit("no eligible reference clip: no full-level 'ok' source found")
    import librosa
    import numpy as np
    def loud_of(r):
        y, sr = librosa.load(str(ROOT / "results/mos/stimuli" / r["file"]),
                             sr=22050, mono=True)
        S = np.abs(np.fft.rfft(y)) ** 2
        fr = np.fft.rfftfreq(len(y), 1 / sr)
        return float(10 * np.log10((S * np.where(fr > 2000, 1.8, 1.0)).mean() + 1e-12))
    scored = sorted(((loud_of(r), r) for r in ctrl), key=lambda x: x[0])
    ref = scored[len(scored) // 2][1]          # median, not an extreme
    (out / "compare.html").write_text(
        PAGE % {"sources": json.dumps(srcs), "levels": json.dumps(levels),
                "ref": "../stimuli/" + ref["file"]}, encoding="utf-8")
    print(json.dumps({"sources": len(srcs), "levels": levels,
                      "reference_clip": ref["file"],
                      "page": str(out / "compare.html")}, indent=1))


if __name__ == "__main__":
    main()
