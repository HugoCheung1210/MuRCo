/* Question JavaScript for the Part-1 / Part-2 instrument (survey SV_9HaoVijdm1j9bzU).
   Emitted into every question's QuestionJS by scripts/mos/build_mos_qsf.py -- edit
   here, rebuild, and the .qsf carries it. Also pasted by hand into the live survey
   (Edit question -> JavaScript) when patching an already-imported copy.

   It installs ONE controller for the whole session. Two jobs:

   1. A loop-aware progress bar. Qualtrics computes progress from position in the
      survey flow, and a Loop & Merge block is one position however many times it
      loops, so the stock bar reads 48% for all ~48 Part-1 ratings and 96% for all
      24 Part-2 comparisons -- it freezes exactly where the participant spends the
      session. Here progress comes from trials actually completed. The loop is
      randomised, so the row number in the DOM id (question-17_QID13) identifies a
      row, it is not a position; the count of DISTINCT rows seen is kept in
      sessionStorage, which also makes the Back button harmless.

   2. Mutually exclusive audio. Part 2 puts two players on the page and nothing
      stops a participant starting both, which would make the comparison
      meaningless. Starting either one pauses the other (pause, not reset, so
      switching back and forth keeps position).

   The survey engine is a single-page app -- pages swap in without a reload -- so
   the controller installed on the first question keeps running for the rest of the
   session. It is emitted on every question anyway, guarded by a window flag, so a
   participant who does reload is covered.

   LOOP_N must match the loop tables (rows in loop_block<n>_paste.csv and
   loop_s2_block<n>_paste.csv); build_mos_qsf.py fills it from the tables it just
   read, so it cannot drift. */
Qualtrics.SurveyEngine.addOnReady(function () {
  if (window.__t2mController) { return; }
  window.__t2mController = true;

  /* trials per looped block: Part-1 ratings and Part-2 A/B */
  var LOOP_N = __LOOP_N__;

  /* share of the bar each part owns, from its share of the session
     (Part 1 ~25 min, Part 2 ~12 min, Part 3 ~3 min) */
  var PART1 = __PART1_BAND__;
  var PART2 = __PART2_BAND__;
  var PART3 = __PART3_BAND__;

  /* the linear pages, which the stock bar would scatter over 0-56% */
  var FIXED = __FIXED__;

  /* which looped block belongs to which band. A looped qid in neither list
     falls to PART2, which is what kept this working before Part 3 existed. */
  var PART1_QIDS = __PART1_QIDS__;
  var PART3_QIDS = __PART3_QIDS__;

  function seenCount(qid, row) {
    var key = 't2m_pb_' + qid, rows = [];
    try { rows = JSON.parse(sessionStorage.getItem(key)) || []; } catch (e) { rows = []; }
    if (rows.indexOf(row) < 0) {
      rows.push(row);
      try { sessionStorage.setItem(key, JSON.stringify(rows)); } catch (e) {}
    }
    return rows.length;
  }

  function target() {
    var sec = document.querySelector('section.question');
    if (!sec) { return null; }
    var m = /^question-(?:(\d+)_)?(QID\d+)$/.exec(sec.id);
    if (!m) { return null; }
    var row = m[1], qid = m[2], n = LOOP_N[qid];
    if (row && n) {
      var band = PART1_QIDS.indexOf(qid) >= 0 ? PART1
               : (PART3_QIDS.indexOf(qid) >= 0 ? PART3 : PART2);
      var done = Math.min(seenCount(qid, row) - 1, n);
      return band[0] + (band[1] - band[0]) * done / n;
    }
    return qid in FIXED ? FIXED[qid] : null;
  }

  function paintBar() {
    var p = target();
    if (p === null) { return; }
    var w = (p * 100).toFixed(1) + '%';
    var fill = document.querySelector('#progress-bar-fill > div');
    if (fill && fill.style.width !== w) { fill.style.width = w; }
    var pr = document.querySelector('#progress-bar progress');
    if (pr) { pr.value = Math.round(p * 100); }
  }

  function wireAudio() {
    var players = document.querySelectorAll('audio');
    Array.prototype.forEach.call(players, function (a) {
      if (a.getAttribute('data-t2m-wired')) { return; }
      a.setAttribute('data-t2m-wired', '1');
      a.addEventListener('play', function () {
        Array.prototype.forEach.call(document.querySelectorAll('audio'), function (b) {
          if (b !== a && !b.paused) { b.pause(); }
        });
      });
    });
  }

  function tick() { paintBar(); wireAudio(); }

  tick();
  /* pages swap in without a reload and the bar is re-rendered when an answer is
     selected, so re-assert rather than set once */
  try {
    new MutationObserver(tick).observe(document.body, {childList: true, subtree: true});
  } catch (e) {}
  setInterval(tick, 300);
});
