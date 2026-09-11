#!/usr/bin/env python3
"""
Phase 0 — MTG-Jamendo source selection for the coherence-metric validation set.

Reads the MTG-Jamendo metadata (from github.com/MTG/mtg-jamendo-dataset), filters
to 6 stress-axis genres, EXCLUDES NoDerivatives (ND) licenses (our pipeline makes
derivatives — pitch-shift, time-stretch, timbre-swap — so ND tracks are unusable),
samples N tracks per genre preferring clean single-genre tracks, and emits:
  - selected_manifest.json   (per-track attribution + license, for the paper)
  - download_list.txt         (relative paths, one per line)
  - download.sh               (wget script to fetch just the ~90 selected files)
  - a stats printout so you can sanity-check counts BEFORE downloading anything.

You do NOT download all 55k tracks — only the selected ~90.

### CHECK ### markers = format assumptions to verify against the actual files the
first time you run this (TSV column order / license-file format can differ by
dataset version). Run once, read the stats + the printed samples, adjust if needed.

Prereq: clone the metadata repo (no audio) —
    git clone https://github.com/MTG/mtg-jamendo-dataset.git
Metadata files live under its `data/` dir:
    data/autotagging.tsv     (tags per track)
    data/raw.meta.tsv        (artist/title/url per track)
    data/audio_licenses.txt  (per-track CC license)
"""
import os, json, re, random
from collections import defaultdict

# ============================== CONFIG ==============================
META_DIR   = os.environ.get("MTG_META", "mtg-jamendo-dataset/data")
AUTOTAG    = f"{META_DIR}/autotagging.tsv"
RAWMETA    = f"{META_DIR}/raw.meta.tsv"
LICENSES = "mtg-jamendo-dataset/audio_licenses.txt"     # repo ROOT, not data/
OUT_DIR    = "sources_selection"
N_PER_GENRE = 15
SEED        = 42
MIN_DUR_S   = 45          # need >~45s so we can skip intro and still cut a stable 30s
os.makedirs(OUT_DIR, exist_ok=True)

# Audio download base. ### CHECK ### — verify one URL resolves before mass download.
# Full-quality 320kbps mirror:
AUDIO_BASE = "https://essentia.upf.edu/datasets/mtg-jamendo/audio"
# (30s-precut alternative: ".../raw_30s/audio" — but those cuts may land on intros;
#  we prefer full tracks + our own stable-span cut in the ingest step.)

# 6 stress-axis genres -> acceptable Jamendo genre tags (any match qualifies).
# Chosen to span beat strength, harmonic clarity, timbral density (see PROJECT_STATE).
GENRE_SPEC = {
    "classical": ["classical", "orchestral", "symphonic"],       # tonal, rubato -> H clear, R weak
    "ambient":   ["ambient", "atmospheric", "darkambient", "newage"],  # no beat -> stresses R, T
    "electronic":["techno", "electronic", "electronica", "house", "edm"],  # strong beat, synth
    "folk":      ["folk", "popfolk", "singersongwriter"],        # acoustic, tonal -> H, T
    "jazz":      ["jazz", "acidjazz", "jazzfunk", "jazzfusion", "swing"],  # complex harmony/swing
    "rock":      ["rock", "hardrock", "poprock", "indie"],       # full-band baseline
}

# ============================== PARSERS ==============================
def _track_id_from_path(p):
    """'14/214.mp3' -> '214' (the filename stem, i.e. the track id)."""
    base = os.path.basename(p)            # '214.mp3'
    m = re.search(r"(\d+)", base)
    return str(int(m.group(1))) if m else None

def parse_autotagging(path):
    """Return {track_id: {'path':..., 'dur':float, 'genres':set, 'n_genres':int}}.
    ### CHECK ### autotagging.tsv layout assumed:
      col0 TRACK_ID  col1 ARTIST_ID  col2 ALBUM_ID  col3 PATH  col4 DURATION  col5.. TAGS
      tags look like 'genre---rock', 'instrument---piano', 'mood/theme---happy'.
    """
    out = {}
    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if "TRACK_ID" not in first and "\t" in first:
            f.seek(0)                                # no header -> rewind
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            tid = _track_id_from_path(parts[3])
            if tid is None:
                continue
            path_rel = parts[3]
            try:
                dur = float(parts[4])
            except ValueError:
                dur = 0.0
            tags = parts[5:]
            genres = {t.split("---", 1)[1] for t in tags if t.startswith("genre---")}
            out[tid] = {"path": path_rel, "dur": dur,
                        "genres": genres, "n_genres": len(genres),
                        "all_tags": tags}
    return out

def parse_licenses(path):
    out = {}
    if not os.path.exists(path):
        print(f"!! {path} not found"); return out
    lic_re = re.compile(r"creativecommons\.org/licenses/([a-z0-9-]+)/", re.I)
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip()]
    for i in range(0, len(lines) - 2, 3):          # clean stride of 3
        tid = _track_id_from_path(lines[i])                     # path line
        m = lic_re.search(lines[i + 2])              # license URL line
        if tid and m:
            out[tid] = m.group(1).lower()
    return out

def parse_rawmeta(path):
    """Return {track_id: {artist,title,url,album,releasedate}} using the header row."""
    out = {}
    if not os.path.exists(path):
        print(f"!! {path} not found — attribution will be incomplete.")
        return out
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {name.lower(): i for i, name in enumerate(header)}   # ### CHECK ### header names
        def col(parts, *names):
            for n in names:
                if n in idx and idx[n] < len(parts):
                    return parts[idx[n]]
            return ""
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            tid = _track_id_from_path(parts[idx.get("track_id", 0)] if "track_id" in idx else parts[0])
            if tid is None:
                continue
            out[tid] = {
                "artist": col(parts, "artist_name", "artist"),
                "title":  col(parts, "track_name", "title", "name"),
                "url":    col(parts, "track_url", "url"),
                "album":  col(parts, "album_name", "album"),
                "releasedate": col(parts, "releasedate", "release_date"),
            }
    return out

# ============================== FILTER + SAMPLE ==============================
def license_ok(slug):
    """Allow derivatives for non-commercial research: keep by / by-sa / by-nc / by-nc-sa
    and CC0. EXCLUDE anything with 'nd' (NoDerivatives) or unknown."""
    if slug in (None, "unknown"):
        return False
    if "nd" in slug:            # by-nd, by-nc-nd -> NoDerivatives, cannot perturb
        return False
    return True

def main():
    print("Loading metadata …")
    tracks   = parse_autotagging(AUTOTAG)
    licenses = parse_licenses(LICENSES)
    meta     = parse_rawmeta(RAWMETA)
    print(f"  autotagging: {len(tracks)} tracks | licenses: {len(licenses)} | meta: {len(meta)}")

    # sample-check the license parse so format errors are caught early
    sample = list(licenses.items())[:5]
    print("  license sample:", sample)

    rng = random.Random(SEED)
    selected, stats = {}, {}
    for label, tags in GENRE_SPEC.items():
        tagset = set(tags)
        # candidates: tagged with this genre, derivative-allowed license, long enough
        cand = []
        for tid, t in tracks.items():
            if not (t["genres"] & tagset):
                continue
            if t["dur"] < MIN_DUR_S:
                continue
            if not license_ok(licenses.get(tid)):
                continue
            cand.append(tid)
        # prefer CLEAN single-genre tracks (genre label is unambiguous), then fill
        cand.sort(key=lambda x: tracks[x]["n_genres"])
        pure = [c for c in cand if tracks[c]["n_genres"] == 1]
        pool = pure if len(pure) >= N_PER_GENRE else cand
        rng.shuffle(pool)
        pick = pool[:N_PER_GENRE]
        stats[label] = {"candidates": len(cand), "single_genre": len(pure), "picked": len(pick)}
        for i, tid in enumerate(sorted(pick, key=int), 1):
            sid = f"{label}_{i:02d}"
            m = meta.get(tid, {})
            selected[sid] = {
                "jamendo_track_id": tid,
                "genre_label": label,
                "audio_path": tracks[tid]["path"],
                "duration_s": tracks[tid]["dur"],
                "license": licenses.get(tid, "unknown"),
                "artist": m.get("artist", ""),
                "title": m.get("title", ""),
                "track_url": m.get("url", ""),
                "album": m.get("album", ""),
                "genre_tags": sorted(tracks[tid]["genres"]),
                "download_url": f"{AUDIO_BASE}/{tracks[tid]['path']}",
            }

    # ---- report ----
    print("\n=== selection stats (verify BEFORE downloading) ===")
    total = 0
    for label, s in stats.items():
        total += s["picked"]
        flag = "" if s["picked"] == N_PER_GENRE else "  <-- SHORT, widen tags/lower MIN_DUR"
        print(f"  {label:12s} candidates={s['candidates']:5d}  single-genre={s['single_genre']:5d}  "
              f"picked={s['picked']:2d}{flag}")
    print(f"  TOTAL selected: {total} (target {N_PER_GENRE*len(GENRE_SPEC)})")

    # ---- write outputs ----
    with open(f"{OUT_DIR}/selected_manifest.json", "w") as f:
        json.dump(selected, f, indent=2)
    with open(f"{OUT_DIR}/download_list.txt", "w") as f:
        for sid, e in selected.items():
            f.write(e["download_url"] + "\n")
    with open(f"{OUT_DIR}/download.sh", "w") as f:
        f.write("#!/bin/bash\nset -e\nmkdir -p raw_audio\n")
        for sid, e in selected.items():
            # save as {sid}.mp3 so the file name already carries genre+index
            f.write(f'wget -c -O raw_audio/{sid}.mp3 "{e["download_url"]}"\n')
    os.chmod(f"{OUT_DIR}/download.sh", 0o755)

    print(f"\nWrote:\n  {OUT_DIR}/selected_manifest.json  (attribution + license per track)"
          f"\n  {OUT_DIR}/download_list.txt"
          f"\n  {OUT_DIR}/download.sh  (fetches ~{total} files as raw_audio/<sid>.mp3)")
    print("\nNext: verify one download_url opens in a browser (### CHECK ### AUDIO_BASE), "
          "then run download.sh. Then the ingest step (decode->stable 30s->LUFS->WAV).")

    # license audit — make sure NO ND slipped through and flag share-alike
    slugs = defaultdict(int)
    for e in selected.values():
        slugs[e["license"]] += 1
    print("\nlicense breakdown of selected:", dict(slugs))
    if any("nd" in s for s in slugs):
        print("  !! ND license present — BUG, should have been filtered. Do not use those.")
    if any("sa" in s for s in slugs):
        print("  note: share-alike (SA) tracks present — your released derivatives may need "
              "CC-BY-SA. Fine for the dissertation; note for any dataset release.")

if __name__ == "__main__":
    main()
