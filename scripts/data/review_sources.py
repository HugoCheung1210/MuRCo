#!/usr/bin/env python
"""Audition every source clip and mark vocals / distressing content. DSP env.

WHY THIS EXISTS. The ethics application states the stimuli are **instrumental only, no
lyrics or vocals, no distressing content**, and §7 leans on that for the low-risk
classification. But `select_fma_sources.py` filters on licence and FMA `genre_top` and
NOTHING ELSE — there is no vocal check anywhere in the pipeline. Four of the six genre
families (Electronic, Folk, Jazz, Rock) routinely contain singing, and piloting found
vocals in a Folk track and heavy breathing in one tagged *Instrumental*. FMA's tags
cannot carry the claim, so the corpus has to be auditioned.

90 sources x 6 s is about nine minutes of listening. Do it once, and the exclusion list
becomes a permanent, citable record that the claim was verified rather than assumed.

Marks per source:
  ok          instrumental, nothing distressing -> keep
  vocals      any singing, spoken word, or lyrics -> EXCLUDE (outside the approval)
  distressing frightening, breathing, or otherwise upsetting -> EXCLUDE
  loud        usable but too loud by default -> KEEP, rendered at reduced gain
  harsh       uncomfortable to listen to (bright/boomy) -> keep but flag

Usage:
  python scripts/data/review_sources.py                 # build the page
  open results/mos/source_review/review.html       # audition, then download the JSON
  python scripts/data/review_sources.py --apply results/mos/source_review/source_review.json
Outputs: results/mos/source_review/review.html, then excluded_sources.json on --apply
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Source screening — vocals & distressing content</title>
<style>
 body{font-family:system-ui,-apple-system,sans-serif;max-width:46rem;margin:0 auto;
      padding:1.5rem;line-height:1.5;color:#111}
 .card{border:1px solid #ddd;border-radius:10px;padding:1rem;margin:.8rem 0}
 audio{width:100%%;margin:.4rem 0}
 .btns{display:flex;gap:.5rem;flex-wrap:wrap}
 button{font-size:.95rem;padding:.5rem .9rem;border-radius:8px;border:1px solid #888;
        background:#f6f6f6;cursor:pointer}
 .ok{background:#e6f4ea;border-color:#34a853} .vo{background:#fce8e6;border-color:#d93025}
 .di{background:#fef7e0;border-color:#f9ab00} .ha{background:#e8f0fe;border-color:#4285f4}
 .lo{background:#f3e8fd;border-color:#a142f4}
 .done{opacity:.5} .prog{color:#666;font-size:.9rem} h2{margin-top:0}
 #bar{position:sticky;top:0;background:#fff;padding:.6rem 0;border-bottom:1px solid #eee}
</style></head><body>
<div id="bar"><b>Source screening</b> — <span id="count"></span>
 <button onclick="save()">Download results</button></div>
<p class="prog">Each clip is the 6 s excerpt used as segment A. Mark anything with
<b>singing, spoken words or lyrics</b> as "vocals" — that is outside what the ethics
approval covers. Mark frightening or upsetting audio as "distressing". "Harsh" keeps the
track but flags it for a listening-comfort warning.</p>
<div id="list"></div>
<script>
const SRC = %(sources)s;
const marks = {};
function render(){
 document.getElementById("list").innerHTML = SRC.map(s=>`
  <div class="card ${marks[s.id]?'done':''}" id="c_${s.id}">
   <b>${s.id}</b> <span class="prog">${s.genre} — ${s.title} / ${s.artist}</span>
   ${marks[s.id]?`<b> → ${marks[s.id]}</b>`:''}
   <audio controls preload="none" src="${s.file}"></audio>
   <div class="btns">
    <button class="ok" onclick="mark('${s.id}','ok')">instrumental, fine</button>
    <button class="vo" onclick="mark('${s.id}','vocals')">has vocals / lyrics</button>
    <button class="di" onclick="mark('${s.id}','distressing')">distressing</button>
    <button class="lo" onclick="mark('${s.id}','loud')">too loud</button>
    <button class="ha" onclick="mark('${s.id}','harsh')">harsh / uncomfortable</button>
   </div></div>`).join("");
 document.getElementById("count").textContent =
   Object.keys(marks).length + " / " + SRC.length + " marked";
}
function mark(id,v){ marks[id]=v; render();
 const el=document.getElementById("c_"+id); if(el) el.scrollIntoView({block:"nearest"}); }
function save(){
 const blob=new Blob([JSON.stringify({marked:marks,n:SRC.length},null,1)],
   {type:"application/json"});
 const a=document.createElement("a");
 a.href=URL.createObjectURL(blob); a.download="source_review.json"; a.click();
}
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="sources_selection/selected_manifest.json")
    ap.add_argument("--audio-dir", default="perturbations/audio")
    ap.add_argument("--out-dir", default="results/mos/source_review")
    ap.add_argument("--apply", default=None,
                    help="path to the downloaded source_review.json -> write exclusions")
    args = ap.parse_args()

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    if args.apply:
        marks = json.load(open(args.apply))["marked"]
        excl = sorted(k for k, v in marks.items() if v in ("vocals", "distressing"))
        harsh = sorted(k for k, v in marks.items() if v == "harsh")
        loud = sorted(k for k, v in marks.items() if v == "loud")
        payload = {"excluded": excl, "attenuate": loud, "harsh_keep_but_warn": harsh,
                   "reasons": {k: marks[k] for k in excl},
                   "n_reviewed": len(marks),
                   "basis": "manual audition; the ethics application claims instrumental "
                            "only + no distressing content and the selector never checked"}
        p = out / "excluded_sources.json"
        json.dump(payload, open(p, "w"), indent=1)
        print(json.dumps({k: v for k, v in payload.items() if k != "reasons"}, indent=1))
        print(f"\nwrote {p}")
        print("next: make_mos_stimuli.py --exclude-sources " + str(p))
        return

    man = json.load(open(ROOT / args.manifest))
    srcs, missing = [], []
    for e in man:
        sid = e["selection_id"]
        f = ROOT / args.audio_dir / f"{sid}__A.wav"
        if not f.is_file():
            missing.append(sid)
            continue
        srcs.append({"id": sid, "genre": e["genre_top"], "title": e["title"],
                     "artist": e["artist"],
                     "file": "../../../" + str(f.relative_to(ROOT))})
    (out / "review.html").write_text(PAGE % {"sources": json.dumps(srcs)}, encoding="utf-8")
    print(json.dumps({"sources": len(srcs), "missing_audio": missing,
                      "page": str(out / "review.html"),
                      "listening_time_min": round(len(srcs) * 6 / 60, 1)}, indent=1))


if __name__ == "__main__":
    main()
