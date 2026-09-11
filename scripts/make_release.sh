#!/usr/bin/env bash
# Build the public release tree from this workspace.
#
# The release is a FUNCTION of the workspace, never a second place to edit. Re-run this
# after any change worth publishing, then commit in the destination. Nothing is copied
# unless git already tracks it, so the .gitignore rules that keep audio, model weights and
# embedding caches out of version control keep them out of the release too.
#
#   bash scripts/make_release.sh --dry-run              # list what would ship
#   bash scripts/make_release.sh ~/Desktop/murco-release
#
# WHAT IS DELIBERATELY EXCLUDED, and why (see doc/ethics + the approved application):
#   doc/papers/      third-party PDFs, redistributing them is a copyright problem
#   doc/ethics/      the application PDF carries personal data and signatures
#   doc/admin/       supervisor project briefs
#   doc/notes/       internal working notes
#   doc/writeup/     thesis source and PDF; publish separately if at all
#   doc/paper/       paper source; the venue distributes the paper
#   CLAUDE.md, PROJECT_STATE.md   internal working docs
#   .claude/         assistant configuration and skills
#   results/rl/eval_pilot2/listen_*   the staging for the GRPO listening arm, which was
#                    dropped in Aug 2026 and makes no claim in the thesis. Read only by the
#                    two withheld scripts below, so nothing in the release can open them,
#                    and listen_files.txt lists audio/*.wav that does not ship. Everything
#                    else in eval_pilot2/ stays: eval_grpo.py and compute_C.py read it.
#   the local listening aids: scripts/rl/{listen_pilot2,make_listen_ui}.py and
#                    scripts/rerank/{play_listen_picks,make_demo_page,make_cocola_pilot}.py.
#                    Terminal and browser tools for auditioning candidates on this machine,
#                    plus the demo page built for a supervisor meeting. None is imported by
#                    anything that ships and none is cited by the thesis or the paper.
#                    make_cocola_pilot.py built an author-rated pilot whose results are
#                    deliberately excluded from the thesis. Withheld 2026-09-09.
#   the preliminary MOS pilots: results/mos/mos_pilot*.{csv,json}, results/mos/pilot/
#                    (the local block*.html rating pages), results/mos/session2/ab_pilot.json
#                    (n_sessions = 1) and scripts/mos/make_local_mos_pilot.py that builds
#                    them. All are author-rated or single-session and none is a result: the
#                    thesis reports only the N=40 collected set. NOTE that results/rl/
#                    eval_pilot*, grpo_pilot2 are NOT pilots in this sense. They are the
#                    three GRPO seeds pool_seeds.py combines into Table 4.5, and they ship.
#   results/mos/source_review/{review.html, source_review.json}
#                    the audition page and the raw per-track marks from it. review.html
#                    builds its <audio> src attributes from the JSON at runtime and points
#                    at the 90 source recordings, which do not ship, so it renders broken.
#                    excluded_sources.json BESIDE THEM IS KEPT ON PURPOSE: it is the
#                    machine-readable screening decision (27 excluded, 14 attenuated, 4 kept
#                    with a warning, each with a reason), it is the evidence that source
#                    material was screened before participants heard it, and
#                    scripts/rl/make_contexts.py reads it with no fallback.
#   sources_selection/practice_candidates/
#                    the audition shortlist for the study's practice clips. Its two tracked
#                    files are an HTML page and a CSV that both point at practice_cand_*.mp3,
#                    and the mp3s are gitignored, so the page ships as five dead links.
#                    Withheld 2026-09-09; scripts/mos/select_practice_clips.py rebuilds it.
#   results/rerank_ace/rerank_manifest.bak.json
#                    a superseded 20-seed/160-pair manifest sitting next to the live
#                    40-seed/320-pair one. Nothing reads it, and the thesis reports 40
#                    seeds, so shipping both invites a reader to take the wrong run.
#   results/rerank/  the Stable Audio Open best-of-N run (10 seeds). Superseded by
#                    results/rerank_ace/ (40 seeds), which is the one the thesis and the
#                    paper report; the SAO tree is cited by neither. Withheld 2026-09-09.
#   results/mos/rehearsal/   a Qualtrics Survey-Test dry run whose numbers are meaningless
#                    by construction, and whose JSON records local scratch paths
#   all audio        the generated stimuli may not be distributed (application D2.1);
#                    this now includes audio EMBEDDED as base64 in HTML, which the path
#                    rules alone did not catch; the script refuses to build if it finds any
#                    the perturbation battery is rebuilt by scripts/data/
#
# TO ADD once they are ready, because reviewers have asked for them:
#   doc/mos/mos_analysis_plan.md              after §1 and §7.5 are reconciled
#   doc/mos/attention_check_prespecification.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INVOKED_FROM="$PWD"     # captured BEFORE the cd, so a relative destination resolves
cd "$ROOT"              # against the caller's directory and not against the repo root

DRY=0
DEST=""
NDEST=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *)  DEST="$a"; NDEST=$((NDEST + 1)) ;;
  esac
done

# One destination only. Taking the last of several silently is how an unquoted path with
# a space in it ("~/some dir/MuRCo") gets split by the shell into two arguments
# and the release lands somewhere nobody asked for.
if [[ $NDEST -gt 1 ]]; then
  cat >&2 <<'MSG'
REFUSING: more than one destination given.
Almost always an unquoted path containing a space, which the shell split into
several arguments. Quote the whole path:
    bash scripts/make_release.sh "$HOME/some dir/MuRCo"
MSG
  exit 2
fi
if [[ $DRY -eq 0 && -z "$DEST" ]]; then
  echo "usage: make_release.sh [--dry-run] <destination>" >&2; exit 2
fi

# Resolve the destination to an absolute path, canonicalising through its parent so that
# a not-yet-existing target still resolves. Then refuse anything inside the workspace: a
# release built into the repo is 32 MB of duplicate that `git add -A` would happily commit.
if [[ -n "$DEST" ]]; then
  [[ "$DEST" = /* ]] || DEST="$INVOKED_FROM/$DEST"
  DEST_PARENT="$(cd "$(dirname "$DEST")" 2>/dev/null && pwd)" || {
    echo "REFUSING: the parent of $DEST does not exist" >&2; exit 2; }
  DEST="$DEST_PARENT/$(basename "$DEST")"
  if [[ "$DEST" == "$ROOT" || "$DEST" == "$ROOT"/* ]]; then
    echo "REFUSING: destination is inside the workspace ($DEST)." >&2
    echo "The release is a separate tree; put it beside the workspace, not within it." >&2
    exit 2
  fi
fi

EXCLUDE_RE='^(CLAUDE\.md|PROJECT_STATE\.md|build_thesis\.sh|doc/|\.claude/|results/mos/rehearsal/|results/rl/eval_pilot2/listen_(ui\.html|files\.txt|key\.json|picks\.json)|results/rerank/|results/rerank_ace/rerank_manifest\.bak\.json|results/mos/mos_pilot(_detail)?\.csv|results/mos/mos_pilot_qc\.json|results/mos/pilot/|results/mos/session2/ab_pilot\.json|scripts/mos/make_local_mos_pilot\.py|results/mos/source_review/(review\.html|source_review\.json)|sources_selection/practice_candidates/|scripts/rl/listen_pilot2\.py|scripts/rl/make_listen_ui\.py|scripts/rerank/play_listen_picks\.py|scripts/rerank/make_demo_page\.py|scripts/rerank/make_cocola_pilot\.py)'

LIST="$(mktemp)"; trap 'rm -f "$LIST"' EXIT
git ls-files | grep -vE "$EXCLUDE_RE" > "$LIST"

# The audio exclusion above is by PATH, and that is not enough on its own. listen_ui.html
# is 4.6 MB of HTML that embeds fifteen generated clips as base64 data URIs, so it walked
# straight past every audio rule until 2026-09-08. Distributing the generated stimuli
# breaches the approved application (D2.1), so the release refuses to build rather than
# ship one quietly. Rebuild such a page from the scored candidates at the far end.
# Two portability traps here, both of which made this guard fail silently on macOS.
# `xargs -d` is a GNU extension BSD xargs lacks, so use tr/xargs -0. And BSD grep caps
# interval repetition at 255, so {500,} aborts with "maximum repetition exceeds 255"
# and the `|| true` swallows it. 200 is under the cap and still far longer than the
# template string in make_listen_ui.py, which is what we must not match.
SMUGGLED="$(tr '\n' '\0' < "$LIST" | xargs -0 grep -l -m1 'data:audio/[a-z0-9]*;base64,[A-Za-z0-9+/]\{200,\}' 2>/dev/null || true)"
if [[ -n "$SMUGGLED" ]]; then
  echo "REFUSING: these tracked files embed audio as base64 data URIs, which the" >&2
  echo "approved application (D2.1) does not permit us to redistribute:" >&2
  printf '  %s\n' $SMUGGLED >&2
  echo "Add each to EXCLUDE_RE, or strip the embedded audio, then re-run." >&2
  exit 2
fi

# The root marker is required: scripts walk up for it, and an extracted release zip has no
# .git to fall back on. It ships whether or not it has been committed here yet.
if [[ ! -f .projectroot ]]; then
  echo "REFUSING: .projectroot is missing; scripts in the release could not find the root" >&2
  exit 1
fi
grep -qx '.projectroot' "$LIST" || echo '.projectroot' >> "$LIST"

# Belt and braces: git should never be tracking these, but a release is the wrong place
# to find out that something slipped past .gitignore.
if BAD="$(grep -iE '\.(wav|mp3|flac|ogg|zip|npz|pt|ckpt|safetensors|bin)$' "$LIST" || true)"; [[ -n "$BAD" ]]; then
  echo "REFUSING: payload files are tracked and would ship:" >&2
  echo "$BAD" >&2
  exit 1
fi

# Assistant-specific or scratch paths are a hard error: nothing in a release should point
# at a temp directory that only existed on one machine. This file is excluded from its own
# scan, because it necessarily contains the pattern it searches for; without the exclusion
# the guard matches itself and no release can ever be built.
if LEAK="$(grep -vFx 'scripts/make_release.sh' "$LIST" | tr '\n' '\0' \
             | xargs -0 grep -lIiE '/private/tmp/claude|anthropic' 2>/dev/null || true)"; [[ -n "$LEAK" ]]; then
  echo "REFUSING: these tracked files carry scratch or assistant-specific paths:" >&2
  echo "$LEAK" >&2
  exit 1
fi

# Absolute paths into this workspace are provenance fields in generated JSON. They are not
# sensitive, but they carry a username and a local layout, so they are rewritten to
# repo-relative in the DESTINATION only. The workspace is never modified.
# `|| true` as on the guards above: under pipefail a grep -l batch with no match makes
# xargs exit 123, and without it `set -e` ends the script here with no message at all.
ABSREF=$( { tr '\n' '\0' < "$LIST" | xargs -0 grep -lIF "$ROOT/" 2>/dev/null || true; } | wc -l | tr -d ' ')

N=$(wc -l < "$LIST" | tr -d ' ')
SZ=$(tr '\n' '\0' < "$LIST" | xargs -0 du -ch 2>/dev/null | tail -1 | cut -f1)   # BSD xargs has no -a
echo "release: $N files, $SZ"
awk -F/ '{print $1"/"($2==""?"":$2)}' "$LIST" | sort | uniq -c | sort -rn | head -20

if [[ $DRY -eq 1 ]]; then
  echo
  echo "(dry run, nothing written)"
  exit 0
fi

mkdir -p "$DEST"
rsync -a --files-from="$LIST" ./ "$DEST"/
echo "copied to $DEST"

if [[ "$ABSREF" -gt 0 ]]; then
  ( cd "$DEST" && tr '\n' '\0' < "$LIST" | xargs -0 grep -lIF "$ROOT/" 2>/dev/null \
      | tr '\n' '\0' | xargs -0 sed -i '' "s|$ROOT/||g" )
  echo "rewrote absolute workspace paths to repo-relative in $ABSREF file(s)"
fi

# Same treatment for home-directory paths outside the workspace: interpreter and conda
# prefixes in the env specs and in script usage examples ("/Users/<you>/anaconda3/..."),
# which carry a username and a local layout into a public repo. Run AFTER the $ROOT pass,
# so anything left is genuinely outside the workspace. Destination only.
HOMEREF=$( cd "$DEST" && tr '\n' '\0' < "$LIST" | xargs -0 grep -lIF "$HOME/" 2>/dev/null | wc -l | tr -d ' ' )
if [[ "$HOMEREF" -gt 0 ]]; then
  ( cd "$DEST" && tr '\n' '\0' < "$LIST" | xargs -0 grep -lIF "$HOME/" 2>/dev/null \
      | tr '\n' '\0' | xargs -0 sed -i '' "s|$HOME/|~/|g" )
  echo "rewrote home-directory paths to ~/ in $HOMEREF file(s)"
fi

# The .claude/ ignore rules are inert in the release, and the release should not mention it.
if [[ -f "$DEST/.gitignore" ]]; then
  sed -i '' -e '/^\.claude/d' -e '/^!\.claude/d' -e '/[Ll]ocal agent state/d' \
             -e '/settings\.local\.json stays out/d' "$DEST/.gitignore"
fi

cat <<EOS

LICENSE, LICENSE-DATA and README.md are generated from the workspace and are already
in the tree above; edit them THERE, never here, or the next run overwrites your edits.

First publication, by hand and once only:
  1. cd "$DEST" && git init -b main && git add -A && git commit
  2. gh repo create <user>/MuRCo --public --source=. --push
     (no gh? create an empty public repo in the browser, then
      git remote add origin https://github.com/<user>/MuRCo.git && git push -u origin main)
  3. git tag -a icassp2027 -m "ICASSP 2027 submission" && git push origin icassp2027

Every later refresh: re-run this script at the same destination, then commit there.
NOTE: rsync here does not delete, so a file removed from the workspace lingers in the
release. After any deletion, rebuild into an empty directory instead of over the top.
Keep this workspace as a PRIVATE repo: its history is the only timestamp on the
analysis plan, and a fresh git init in the release destroys it.
EOS
