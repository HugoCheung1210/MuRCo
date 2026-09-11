#!/usr/bin/env python
"""Emit the Part-1 MOS rating survey as a Qualtrics .qsf, ready to import. DSP env.

Why a QSF and not the survey builder UI: the instrument is 5 blocks x 50 looped
trials plus consent/instructions/flow. Hand-building that is ~40 dialogs of
click-and-paste with no record of what was built. A generated QSF is
deterministic, diffable, and lives in the repo next to the stimuli that feed it.

Participant-facing copy is held in the CAPS constants below and is the single
source of truth for the built instrument. It tracks:
  - doc/mos/data_collection_tool.md            screens 1-7
  - doc/ethics/participant_information_sheet_ucl.md  PIS body + consent form
  - doc/ethics/ethics_amendment_session2.md  §A    THE REVISED RATING WORDING

§A matters: the *approved* identity-framed question ("how well does B continue A
as the same piece") collapses the scale, because these stimuli are
self-perturbations where identity always survives. The wording below is the
revised one. Building the approved wording instead would produce a broken
instrument.

Boilerplate (brand, skin, survey options incl. AnonymizeResponse) is copied from
a real export of the UCL brand rather than invented -- see --template.

Scope: the whole session. Part 1 (5 looped rating blocks) AND Part 2 (10 looped
A/B blocks), plus the three §C mitigations that the longer session depends on:
volume calibration against the loudest clip, a self-paced break between parts,
and a Skip option on every trial. `--no-part2` builds the rating half alone.

Usage:
  python scripts/mos/make_session2_loops.py          # Part-2 loop tables first
  python scripts/mos/build_mos_qsf.py --practice-clips practice_hi,practice_lo
Output: results/mos/qualtrics/mos_full.qsf  (import via Projects -> New -> Survey -> From a file)
"""
import argparse
import copy
import datetime
import csv
import json
import re
import sys
from pathlib import Path

BASE_SURVEY_NAME = ("Perceptual validation of an automatic coherence metric "
                    "for music editing")

ROOT = next((p for p in Path(__file__).resolve().parents
             if (p / ".projectroot").exists() or (p / ".git").exists()),
            Path(__file__).resolve().parent.parent)

# --- participant-facing copy -------------------------------------------------

# Verbatim from doc/ethics/Participant Information Leaflet (PIL).docx. Do not paraphrase:
# this is the document the ethics committee approved. Only markup was added, and
# ETHICS_ID fills the blank the .docx leaves as "_______".
ETHICS_ID = "4972"

INFO_SHEET = f"""
<h3>Participant Information Sheet<br>For adult volunteer listeners</h3>
<p><b>UCL Research Ethics Committee Approval ID Number:</b> {ETHICS_ID}</p>
<p><b>YOU WILL BE GIVEN A COPY OF THIS INFORMATION SHEET</b></p>

<p><b>Title of Study:</b> Perceptual validation of an automatic coherence metric for
music editing<br>
<b>Department:</b> Department of Computer Science<br>
<b>Name and Contact Details of the Researcher(s):</b> Chi Wang Cheung,
c.cheung.25@ucl.ac.uk<br>
<b>Name and Contact Details of the Principal Researcher:</b> Prof. Jagmohan Chauhan
(supervisor), jagmohan.chauhan@ucl.ac.uk</p>

<p>You are being invited to take part in a research project. Before you decide, it is
important for you to understand why the research is being done and what taking part will
involve. Please take time to read the following information carefully. Ask us if there is
anything that is not clear or if you would like more information. Take time to decide
whether or not you wish to take part. Thank you for reading this.</p>

<p><b>What is the project's purpose?</b><br>
We are studying how people judge whether two segments of instrumental music fit
together, where the second is the same passage as the first with a change applied. We
have built an automatic computer method that measures how well two segments fit
together, and we want to check whether it
matches human judgement. The clips you will hear vary, some are
unchanged, some have been digitally modified (for example in pitch, tempo, or
instrumentation) or paired with a segment from a different piece, and some have a second
segment that was generated or edited by AI music software. All clips are ordinary
instrumental music with no lyrics or sensitive content. We simply ask for your honest
rating of each as there are no right or wrong answers. The study forms part of an MSc
dissertation.</p>

<p><b>Why have I been chosen?</b><br>
We are inviting adults aged 18 or over, currently in the UK, with normal hearing, who are
able to listen using headphones in a quiet place. You are seeing this because you
responded to an open invitation (email list, group channel, or poster); no one has been
personally approached or selected.</p>

<p><b>Do I have to take part?</b><br>
It is entirely up to you whether or not to take part. Participation is voluntary and
unpaid. You can withdraw at any time before you submit simply by closing the task,
without giving a reason and without any penalty. Because your responses are anonymous,
after you submit they cannot be identified and so cannot be withdrawn. Whether or not you
take part has no effect on your studies, work, or on any relationship with the researcher
or their supervisor, and because responses are anonymous, no one will know whether you
chose to take part.</p>

<p><b>What will happen to me if I take part?</b><br>
In a single online session of about 35 minutes, in two parts, with a break in between that
you control &mdash; you continue when you are ready.</p>

<p><b>Part 1</b> (about 20 minutes): you listen, over headphones, to around 40 short audio
clips (each about 13 seconds, two music segments separated by a one-second silence). For
each clip you rate, on a 1&ndash;5 scale, how well the two segments fit together. The
second segment is not a continuation of the first; it is the same passage of music again,
sometimes altered. There are two practice clips first.</p>

<p><b>Part 2</b> (about 12 minutes): you make 24 comparisons. Each presents two 12-second
versions of the same piece in which a short middle section was regenerated by AI music
software; the surrounding music is identical. You choose which version's regenerated
section fits better with the music around it.</p>

<p>Before either part you set your listening volume against the loudest clip in the study.
Every trial has a <b>Skip</b> option if a clip is uncomfortable. The clips are ordinary
instrumental music; no personal recordings of you are made and no microphone or camera is
used. You take part while in the UK, at a time and place of your choosing.</p>

<p><b>What are the possible disadvantages and risks of taking part?</b><br>
There are no risks beyond mild fatigue from listening for around 35 minutes. Some of the
AI-generated audio in part 2 can sound harsh or distorted. To limit this you set your
volume against the loudest clip before starting, there is a self-paced break between the
two parts, and every trial has a <b>Skip</b> option that needs no explanation and carries
no penalty. Please also feel free to pause between clips.</p>

<p><b>What are the possible benefits of taking part?</b><br>
There is no direct benefit to you from taking part. It is hoped that this work will help
develop better automatic tools for evaluating and editing music.</p>

<p><b>What if something goes wrong?</b><br>
If you wish to raise a concern or complaint, please contact the researcher, Chi Wang
Cheung (c.cheung.25@ucl.ac.uk), or the supervisor, Prof. Jagmohan Chauhan
(jagmohan.chauhan@ucl.ac.uk), in the first instance. If you feel your complaint has not
been handled to your satisfaction, you can contact the Chair of the UCL Research Ethics
Committee at ethics@ucl.ac.uk.</p>

<p><b>Will my taking part in this project be kept confidential?</b><br>
Yes, the study is anonymous by design. We record only your ratings (a 1&ndash;5 value per
clip, under a random session code). We do not collect your name, email, demographic
details, device information, or location, and the questionnaire platform is configured not
to record your IP address. You cannot be identified in any report or publication, because
no identifying information is ever collected.</p>

<p><b>Limits to confidentiality</b><br>
Because this study is anonymous and collects no personal or identifying information, only
1&ndash;5 ratings under a random session code, there are no foreseeable limits to
confidentiality. The task involves listening to instrumental music and giving ratings
only; it does not invite any personal disclosures, so no situation is anticipated in which
confidentiality would need to be broken.</p>

<p><b>What will happen to the results of the research project?</b><br>
The results will be written up as an MSc dissertation and may be presented or published in
academic outputs. Findings are reported only in aggregate (combined across listeners); you
will not be identified in any report or publication, as no identifying information is
collected. Because the rating dataset is anonymous, it may be archived or shared to
support the findings and used for subsequent related research. If you would like a summary
of the results, you can request one from the researcher at c.cheung.25@ucl.ac.uk.</p>

<p><b>Local Data Protection Privacy Notice</b><br>
The controller for this project will be University College London (UCL). The UCL Data
Protection Officer provides oversight of UCL activities involving the processing of
personal data, and can be contacted at data-protection@ucl.ac.uk</p>

<p>This 'local' privacy notice sets out the information that applies to this particular
study. Further information on how UCL uses participant information can be found in our
'general' privacy notice: for participants in research studies,
<a href="https://www.ucl.ac.uk/legal-services/privacy/ucl-general-research-participant-privacy-notice"
target="_blank">click here</a>.</p>

<p>The information that is required to be provided to participants under data protection
legislation (GDPR and DPA 2018) is provided across both the 'local' and 'general' privacy
notices.</p>

<p>This study is designed to process no personal data: your ratings are collected
anonymously under a random session code, and the platform (UCL Qualtrics) is configured so
that no IP address or other online identifier is recorded. Eligibility items (18+, in the
UK, normal hearing, headphones) are self-confirmed to proceed and are not stored.</p>

<p>Because no personal data is collected, the study does not rely on your personal data
for its lawful basis; were any personal data to be processed by UCL in the course of this
research, the lawful basis would be 'performance of a task in the public interest' (and,
for any special category data, scientific/historical research or statistical purposes). We
will endeavour to minimise the processing of personal data wherever possible.</p>

<p>If you are concerned about how your personal data is being processed, or would like to
contact us about your rights, please contact UCL in the first instance at
data-protection@ucl.ac.uk. No data is transferred outside the UK.</p>

<p><b>Who is organising and funding the research?</b><br>
The research is organised by UCL (Department of Computer Science) as part of an MSc
dissertation. It is not externally funded.</p>

<p><b>Contact for further information</b><br>
For further information please contact the researcher, Chi Wang Cheung, at
c.cheung.25@ucl.ac.uk, or the supervisor, Prof. Jagmohan Chauhan, at
jagmohan.chauhan@ucl.ac.uk</p>

<p>Thank you for reading this information sheet and for considering to take part in this
research study.</p>

<p><i>You may wish to save or print this page before continuing.</i></p>
"""

# Verbatim from doc/ethics/Consent Form.docx.
CONSENT_TEXT = f"""
<h3>CONSENT FORM FOR ADULT VOLUNTEER LISTENERS IN RESEARCH STUDIES</h3>
<p>Please complete this form after you have read the Information Sheet about the
research.</p>

<p><b>Title of Study:</b> Perceptual validation of an automatic coherence metric for
music editing<br>
<b>Department:</b> Department of Computer Science<br>
<b>Name and Contact Details of the Researcher(s):</b> Chi Wang Cheung,
c.cheung.25@ucl.ac.uk<br>
<b>Name and Contact Details of the Principal Researcher:</b> Prof. Jagmohan Chauhan,
jagmohan.chauhan@ucl.ac.uk<br>
<b>Name and Contact Details of the UCL Data Protection Officer:</b>
data-protection@ucl.ac.uk</p>

<p>This study has been approved by the UCL Research Ethics Committee:<br>
<b>Project ID number:</b> {ETHICS_ID}</p>

<p>Thank you for considering taking part in this research. Please read the Information
Sheet before you agree to take part. If you have any questions arising from the
Information Sheet, please contact the researcher before you decide. You are allowed to
copy this Consent Form to keep and refer to at any time.</p>

<p>Please tick each box below.</p>
"""

# The four items the .docx marks with a leading asterisk are mandatory; the rest are
# offered but not required. Splitting them across two questions is the only way
# Qualtrics can require *specific* boxes -- MinChoices only enforces a count.
#
# The first three are the .docx's preamble paragraph, promoted from description text
# into their own tick boxes at Hugo's request. They are confirmations of
# understanding, so they sit with the mandatory items -- and they must NOT also
# remain in CONSENT_TEXT above, or a participant would be asked to agree twice.
CONSENT_REQUIRED = [
    "I confirm that I understand that by ticking each box below I am consenting to this "
    "element of the study.",
    "I understand that it will be assumed that unticked boxes means that I DO NOT consent "
    "to that part of the study.",
    "I understand that by not giving consent for any one element that I may be deemed "
    "ineligible for the study.",
    "I confirm that I have read and understood the Information Sheet for the above study. "
    "I have had an opportunity to consider the information and what will be expected of me. "
    "I have also had the opportunity to ask questions which have been answered to my "
    "satisfaction",
    "I consent to participate in the study. I confirm that I am 18 years or older, "
    "currently in the UK, and have normal hearing and headphones available. I understand "
    "that some clips are digitally modified or generated by AI music software, and are "
    "ordinary instrumental music with no lyrics or sensitive content.",
    "I understand that my participation is voluntary and that I am free to withdraw at any "
    "time before I submit my responses, without giving a reason and without penalty.",
    "I understand that my ratings are recorded anonymously, that no identifying "
    "information (including my IP address) is collected, and that once I submit, my "
    "responses cannot be withdrawn because they cannot be identified.",
]

CONSENT_OPTIONAL_TEXT = "<p>Please also confirm the following:</p>"

CONSENT_OPTIONAL = [
    "I understand that the data will not be made available to any commercial organisations "
    "but is solely the responsibility of the researcher(s) undertaking this study.",
    "I understand that I will not benefit financially from this study or from any possible "
    "outcome it may result in in the future.",
    "I agree that my anonymised research data may be used by others for future research. "
    "No one will be able to identify you when this data is shared.",
    "I am aware of who I should contact if I wish to lodge a complaint.",
    "I voluntarily agree to take part in this study.",
]

# The scale, verbatim from ethics_amendment_session2.md §A "Revised wording".
SCALE = [
    ("5", "<b>5</b> &mdash; Nothing changes; they fit together perfectly."),
    ("4", "<b>4</b> &mdash; Very slight change; they still fit almost perfectly."),
    ("3", "<b>3</b> &mdash; Something clearly changes, but they still fit reasonably well."),
    ("2", "<b>2</b> &mdash; Something changes a lot; they fit poorly."),
    ("1", "<b>1</b> &mdash; They do not fit together at all."),
]
SKIP_LABEL = "Skip &mdash; uncomfortable"
SKIP_RECODE = "0"

RATING_STEM = (
    "<b>How well do these two segments fit together?</b><br>The second segment is the "
    "same passage of music again, sometimes altered. Listen for anything that changes "
    "between them &mdash; the key, the tempo, the instruments, the tone or recording "
    "quality."
)

INSTRUCTIONS = f"""
<h3>Instructions</h3>
<p>You will hear a series of clips. Each clip is made of <b>segment A</b>, then a
<b>short silence</b>, then <b>segment B</b>. Listen with headphones at a comfortable
volume.</p>

<p>Segment B is <b>not a continuation</b> of segment A &mdash; it is the same passage of
music again, sometimes altered. Recognising that it is the same piece is <b>not</b> what
we are asking about; we are asking <b>how well the two match</b>. After each clip,
rate:</p>

<p><b>How well do these two segments fit together?</b><br>Listen for anything that
changes between them &mdash; the key, the tempo, the instruments, the tone or recording
quality.</p>

<ul>
{''.join(f'<li>{lab}</li>' for _, lab in SCALE)}
</ul>

<p>Judge the <b>relationship</b> between the two segments, not whether you
<i>like</i> the music. Some clips are unchanged and some have been digitally modified
&mdash; rate your <b>honest impression</b>; there are no right answers.</p>

<p>If a clip is uncomfortable to listen to, use <b>{SKIP_LABEL}</b>. It carries no
penalty and needs no explanation. You may take a short break at any point.</p>
"""

PRACTICE_TEXT = """
<h3>Practice</h3>
<p>Two practice clips first, to get used to the task. These are not scored.</p>
"""

THANK_YOU = """
<h3>Thank you</h3>
<p>Your anonymous ratings have been recorded. Thank you for taking part.</p>
<p>Your ratings will be compared against an automatic method for measuring how well two
segments of music fit together, as part of an MSc dissertation at UCL.</p>
<p>Questions or concerns: Chi Wang Cheung (c.cheung.25@ucl.ac.uk) or Prof. Jagmohan
Chauhan (jagmohan.chauhan@ucl.ac.uk). UCL Research Ethics Committee Approval ID
<b>4972</b>; complaints may be directed to ethics@ucl.ac.uk.</p>
<p>You may now close this page.</p>
"""

# --- part 2 + the §C mitigations ---------------------------------------------
#
# Amendment §C raises the session to ~35 min and therefore has to REPLACE the
# <=30 min cap that the approved risk assessment named as the fatigue mitigation.
# The replacements are these three, so they are not decoration: a self-paced
# break, a volume calibration against the worst-case clip, and a per-trial skip
# (already present on every rating and comparison question).

# loudest clip in the whole set by short-term (400 ms) loudness, measured across
# all 320 rendered clips. Calibrating on anything quieter defeats the purpose.
CALIBRATION_CLIP = "t001_A.mp3"

CALIBRATION_TEXT = """
<h3>Set your volume</h3>
<p>Before you start, set a comfortable listening level and then leave it there for the
rest of the session.</p>
<p>The clip below is the <b>loudest</b> in the study. Some of the audio you will hear is
machine-generated and can sound harsh or distorted. Set your volume so that <b>this</b>
clip is comfortable &mdash; everything else will then be at or below that level.</p>
{player}
<p>Use headphones, in a quiet place. If any clip is uncomfortable, every trial has a
<b>Skip</b> option.</p>
"""

BREAK_TEXT = """
<h3>End of part 1 &mdash; take a break</h3>
<p>That is the rating task finished. Thank you.</p>
<p><b>Part 2 is shorter</b>: 24 comparisons, about 12 minutes.</p>
<p>Take as long as you like before continuing &mdash; this page will wait. Stretch, rest
your ears, and carry on when you are ready. You can still stop at any point before
submitting.</p>
"""

# The practice clips look exactly like study trials, so without this page the scored
# task starts with no visible boundary and a participant cannot tell that the two
# unscored clips are behind them.
START_TEXT = f"""
<h3>Practice finished &mdash; part 1 starts now</h3>
<p>Those two clips were practice and were not scored. From here your ratings count.</p>

<p><b>About 40 clips, roughly 20 minutes.</b> Each has the same shape as the practice
ones: segment A, a short silence, then segment B. Rate <b>how well the two segments fit
together</b>, not whether you like the music.</p>

<p>Leave your volume where you set it and keep the same headphones throughout. Every
trial has <b>{SKIP_LABEL}</b>, and you may pause between clips whenever you like.</p>

<p>Press <b>Next page</b> when you are ready to begin.</p>
"""

PART2_INSTRUCTIONS = """
<h3>Part 2 &mdash; comparing two versions</h3>
<p>In each trial you hear <b>two versions of the same piece of music</b>. In both, a short
section in the middle was <b>regenerated by an AI model</b>. The surrounding music is
identical &mdash; only the regenerated section differs.</p>

<p>Your job: decide <b>which version's regenerated section fits better with the music
around it</b> &mdash; for example in key, style, tempo and instrumentation. Listen to both before
answering; you can replay either as often as you like.</p>

<p>Judge how well the middle section matches the rest. <b>Not</b> which track you prefer,
and <b>not</b> which is louder or cleaner. If the two are genuinely comparable,
<b>About the same</b> is a real answer &mdash; use it.</p>

<p>Some of this audio is machine-generated and may sound harsh. If any comparison is
uncomfortable, use <b>{skip}</b>.</p>
"""

PART3_INSTRUCTIONS = """
<h3>Part 3 &mdash; which continuation fits better?</h3>
<p>A few last comparisons, about three minutes. These work like Part 2, but instead of a
regenerated middle section you hear <b>two ways the same piece could carry on</b>.</p>

<p>Each version starts with the <b>same few seconds of real music</b>, then an AI model
takes over and plays on. The opening is identical in both; everything after it differs.</p>

<p>Your job: decide <b>which continuation follows on better from the music it starts
with</b>, for example in key, style, tempo and instrumentation. Judge the join and what
comes after it, <b>not</b> which clip you prefer as music and <b>not</b> which is louder or
cleaner. <b>About the same</b> is a real answer.</p>

<p>Some of this audio is machine-generated and may sound harsh. If any comparison is
uncomfortable, use <b>{skip}</b>.</p>
"""

GRPO_STEM = ("<b>Which continuation follows on better from the music it starts with?</b>"
             "<br>Both versions open with the same few seconds of real music &mdash; only "
             "what comes after it differs. Judge how well the continuation follows on, for "
             "example in key, style, tempo and instrumentation.")

AB_STEM = ("<b>Which version's regenerated section fits better with the surrounding "
           "music?</b><br>Both versions share the same surrounding music &mdash; only the "
           "middle section differs. Judge how well that middle section matches the rest, "
           "for example in key, style, tempo and instrumentation.")

# verbatim from the validated local instrument (session2_block*.html SCALE)
AB_SCALE = [
    ("1", "Version A is clearly more coherent"),
    ("2", "Version A is slightly more coherent"),
    ("3", "About the same"),
    ("4", "Version B is slightly more coherent"),
    ("5", "Version B is clearly more coherent"),
]

PRACTICE_PLACEHOLDER = (
    '<p style="color:#b00"><b>[BUILD TODO]</b> No practice clips are designated in the '
    'stimulus plan. Re-run with <code>--practice-clips id1,id2</code> to wire two clips '
    'in, or delete this block. Practice clips must NOT be study stimuli.</p>'
)


# --- question javascript -----------------------------------------------------

CONTROLLER_JS = Path(__file__).with_name("qualtrics_question_js.js")


# Where each looped half sits on the bar. Split by wall-clock share of the
# session (Part 1 ~25 min of ratings, Part 2 ~12 min of comparisons), not by
# trial count, so the bar moves at roughly a constant rate throughout.
PART1_BAND = [0.08, 0.62]
PART2_BAND = [0.68, 0.88]
# Part 3 is the GRPO arm: 10 comparisons, ~3 min, so it owns a thin band.
PART3_BAND = [0.90, 0.98]


def controller_js(loop_n, fixed, part1_qids, part3_qids=()):
    """The per-question controller, with this build's block sizes baked in.

    Every question carries it (guarded by a window flag, so it installs once).
    It does two things the survey engine will not: advance the progress bar
    inside a Loop & Merge block, and stop the two Part-2 players sounding at
    once. See the header of qualtrics_question_js.js for why.
    """
    src = CONTROLLER_JS.read_text()
    for token, value in (("__LOOP_N__", loop_n), ("__FIXED__", fixed),
                         ("__PART1_QIDS__", part1_qids),
                         ("__PART3_QIDS__", list(part3_qids)),
                         ("__PART1_BAND__", PART1_BAND), ("__PART2_BAND__", PART2_BAND),
                         ("__PART3_BAND__", PART3_BAND)):
        if token not in src:
            sys.exit(f"{CONTROLLER_JS.name}: placeholder {token} is missing")
        src = src.replace(token, json.dumps(value))
    return src


def fixed_anchors(pre_qids, mid_qids, end_qids, part1, part2,
                  late_qids=(), part3=None):
    """Where the non-looped pages sit on the bar, in the gaps the loops leave.

    The preamble shares everything below the Part-1 band, the break and Part-2
    instructions share the gap between the two bands, and the thank-you page is
    the end. Spacing inside each gap is even: these pages are quick, and what
    matters is only that the bar never goes backwards at a boundary.
    """
    out = {}
    for i, q in enumerate(pre_qids):
        out[q] = round(part1[0] * i / max(len(pre_qids), 1), 4)
    for i, q in enumerate(mid_qids, start=1):
        out[q] = round(part1[1] + (part2[0] - part1[1]) * i / (len(mid_qids) + 1), 4)
    # the Part-3 preamble sits in the gap the Part-2 band leaves below Part-3
    if late_qids and part3 is not None:
        for i, q in enumerate(late_qids, start=1):
            out[q] = round(part2[1] + (part3[0] - part2[1]) * i / (len(late_qids) + 1), 4)
    for q in end_qids:
        out[q] = 1.0
    return out


# --- qsf construction --------------------------------------------------------

class Builder:
    """Accumulates SurveyElements with unique QID / BL / FL ids."""

    def __init__(self, template):
        self.template = template
        self.qid = 0
        self.blid = 0
        self.flid = 1
        self.questions = []
        self.blocks = []

    def _next_qid(self):
        self.qid += 1
        return f"QID{self.qid}"

    def _next_blid(self):
        self.blid += 1
        return f"BL_gen{self.blid:03d}"

    def _next_flid(self):
        self.flid += 1
        return f"FL_{self.flid}"

    def question(self, payload, tag, name):
        qid = self._next_qid()
        payload.update({"QuestionID": qid, "DataExportTag": tag,
                        "QuestionDescription": name,
                        "DataVisibility": {"Private": False, "Hidden": False},
                        "Configuration": {"QuestionDescriptionOption": "UseText"},
                        "GradingData": [], "Language": [], "NextAnswerId": 1})
        self.questions.append({
            "SurveyID": None, "Element": "SQ", "PrimaryAttribute": qid,
            "SecondaryAttribute": name[:100], "TertiaryAttribute": None,
            "Payload": payload})
        return qid

    def block(self, description, qids, looping=None):
        bid = self._next_blid()
        opts = {"BlockLocking": "false", "RandomizeQuestions": "false",
                "BlockVisibility": "Expanded"}
        if looping:
            opts["Looping"] = "Static"
            opts["LoopingOptions"] = looping
        self.blocks.append({
            "Type": "Standard", "SubType": "", "Description": description, "ID": bid,
            "BlockElements": [{"Type": "Question", "QuestionID": q} for q in qids],
            "Options": opts})
        return bid


def descriptive(text, name):
    return {"QuestionText": text, "QuestionType": "DB", "Selector": "TB",
            "Validation": {"Settings": {"Type": "None"}}}


def rating_question(include_audio):
    """Single-answer 1-5 + skip. Recoded so the export column IS the rating."""
    choices, recode, order = {}, {}, []
    for i, (val, label) in enumerate(SCALE, start=1):
        choices[str(i)] = {"Display": label}
        recode[str(i)] = val
        order.append(str(i))
    n = len(SCALE) + 1
    choices[str(n)] = {"Display": SKIP_LABEL}
    recode[str(n)] = SKIP_RECODE
    order.append(str(n))

    audio = '${lm://Field/2}<br><br>' if include_audio else ''
    return {
        "QuestionText": f"{audio}{RATING_STEM}",
        "QuestionType": "MC", "Selector": "SAVR", "SubSelector": "TX",
        "Choices": choices, "ChoiceOrder": order, "RecodeValues": recode,
        "DefaultChoices": False,
        # ForceResponse is safe here *because* Skip is an explicit choice: a
        # participant is never trapped by a clip they do not want to hear.
        "Validation": {"Settings": {"ForceResponse": "ON",
                                    "ForceResponseType": "ON", "Type": "None"}},
        "NextChoiceId": n + 1,
    }


def consent_question(text, items, require_all):
    choices = {str(i): {"Display": c} for i, c in enumerate(items, 1)}
    order = [str(i) for i in range(1, len(items) + 1)]
    if require_all:
        validation = {"Settings": {"ForceResponse": "ON", "ForceResponseType": "ON",
                                   "Type": "MinChoices", "MinChoices": str(len(items))}}
    else:
        validation = {"Settings": {"ForceResponse": "OFF", "Type": "None"}}
    return {
        "QuestionText": text,
        "QuestionType": "MC", "Selector": "MAVR", "SubSelector": "TX",
        "Choices": choices, "ChoiceOrder": order, "DefaultChoices": False,
        "Validation": validation,
        "NextChoiceId": len(items) + 1,
    }


def ab_question(stem=None):
    """5-point A/B preference + skip. Scale verbatim from session2_block*.html.

    Recoded 1..5 with A-better low and B-better high, so a mean below 3 favours
    whichever side the key says is A. Skip is 0 and must be dropped before
    averaging -- it is not 'about the same', which is 3.
    """
    choices, recode, order = {}, {}, []
    for i, (val, label) in enumerate(AB_SCALE, start=1):
        choices[str(i)] = {"Display": label}
        recode[str(i)] = val
        order.append(str(i))
    n = len(AB_SCALE) + 1
    choices[str(n)] = {"Display": SKIP_LABEL}
    recode[str(n)] = SKIP_RECODE
    order.append(str(n))

    body = ('<b>Version A</b><br>${lm://Field/2}<br><br>'
            '<b>Version B</b><br>${lm://Field/3}<br><br>'
            f'{stem or AB_STEM}')
    return {
        "QuestionText": body,
        "QuestionType": "MC", "Selector": "SAVR", "SubSelector": "TX",
        "Choices": choices, "ChoiceOrder": order, "RecodeValues": recode,
        "DefaultChoices": False,
        "Validation": {"Settings": {"ForceResponse": "ON",
                                    "ForceResponseType": "ON", "Type": "None"}},
        "NextChoiceId": n + 1,
    }


def load_loop(path, fields=2):
    """paste-csv -> Qualtrics Static looping dict, 1-indexed.

    fields=2 for the Part-1 rating tables (id, audio); fields=3 for Part-2
    (id, audio A, audio B). A short row means a mangled export, not a shrug --
    a missing embed is a dead player on a live trial, so bail.
    """
    rows = [r for r in csv.reader(open(path)) if r and r[0].strip()]
    out = {}
    for i, r in enumerate(rows, start=1):
        if len(r) < fields:
            sys.exit(f"{path}: row {i} has {len(r)} field(s), expected {fields}")
        out[str(i)] = {str(f): r[f - 1] for f in range(1, fields + 1)}
    return out, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="results/mos/qualtrics/template/exported_reference.qsf",
                    help="a real export from the target brand; supplies skin + survey options")
    ap.add_argument("--loop-dir", default="results/mos/qualtrics")
    ap.add_argument("--loop-glob", default="loop_block*_paste.csv")
    ap.add_argument("--out", default="results/mos/qualtrics/mos_full.qsf")
    ap.add_argument("--name", default=None,
                    help="survey name written into SurveyEntry.SurveyName. Left unset it "
                         "is stamped with the build date and block layout, because "
                         "Qualtrics names an imported project after this field: a constant "
                         "name makes every import land as another identically-named "
                         "project and there is then no way to tell which one is current.")
    ap.add_argument("--practice-clips", default="",
                    help="two stimulus ids for the unscored practice block")
    ap.add_argument("--s2-loop-glob", default="loop_s2_block*_paste.csv",
                    help="Part-2 A/B loop tables; run make_session2_loops.py first")
    ap.add_argument("--grpo-loop", default="loop_grpo_paste.csv",
                    help="Part-3 loop table (GRPO vs frozen continuations). Every "
                         "participant gets this block; it is NOT in the randomiser, "
                         "because a SubSet-1 randomiser would show it to 1 in 11")
    ap.add_argument("--no-part3", action="store_true")
    ap.add_argument("--no-part2", action="store_true",
                    help="build the Part-1 rating instrument only")
    args = ap.parse_args()

    template = json.load(open(ROOT / args.template))
    loops = sorted((ROOT / args.loop_dir).glob(args.loop_glob))
    if not loops:
        sys.exit(f"no {args.loop_glob} in {args.loop_dir} — run stamp_qualtrics_urls.py --paste")

    b = Builder(template)

    # linear preamble
    info_b = b.block("Information sheet",
                     [b.question(descriptive(INFO_SHEET, "info"), "INFO", "Information sheet")])
    consent_b = b.block("Consent & eligibility", [
        b.question(consent_question(CONSENT_TEXT, CONSENT_REQUIRED, require_all=True),
                   "CONSENT_REQ", "Consent (required items)"),
        b.question(consent_question(CONSENT_OPTIONAL_TEXT, CONSENT_OPTIONAL, require_all=False),
                   "CONSENT_OPT", "Consent (further items)"),
    ])
    instr_b = b.block("Instructions",
                      [b.question(descriptive(INSTRUCTIONS, "instr"), "INSTR", "Instructions")])

    # §C mitigation: calibrate against the worst case BEFORE any trial
    fids_all = {Path(r["filename"]).stem: r["file_id"] for r in
                csv.DictReader(open(ROOT / "results/mos/qualtrics/file_ids.csv"))}
    cal_stem = Path(CALIBRATION_CLIP).stem
    if cal_stem not in fids_all:
        sys.exit(f"calibration clip {CALIBRATION_CLIP} not in file_ids.csv")
    cal_player = ('<audio controls preload="none" style="width:100%"><source src='
                  f'"https://qualtrics.ucl.ac.uk/CP/File.php?F={fids_all[cal_stem]}" '
                  'type="audio/mpeg"></audio>')
    calib_b = b.block("Volume calibration", [
        b.question(descriptive(CALIBRATION_TEXT.format(player=cal_player), "calib"),
                   "CALIBRATION", "Volume calibration")])

    # practice
    practice_qids = [b.question(descriptive(PRACTICE_TEXT, "practice"), "PRACTICE_INTRO",
                                "Practice intro")]
    ids = [c.strip() for c in args.practice_clips.split(",") if c.strip()]
    if ids:
        fids = fids_all
        for sid in ids:
            if sid not in fids:
                sys.exit(f"practice clip {sid} not in file_ids.csv")
            embed = ('<audio controls preload="none" style="width:100%"><source src='
                     f'"https://qualtrics.ucl.ac.uk/CP/File.php?F={fids[sid]}" '
                     'type="audio/mpeg"></audio>')
            q = rating_question(include_audio=False)
            q["QuestionText"] = f"{embed}<br><br>{RATING_STEM}"
            practice_qids.append(b.question(q, f"PRACTICE_{sid}", f"Practice {sid}"))
    else:
        practice_qids.append(b.question(
            descriptive(PRACTICE_PLACEHOLDER, "practice_todo"), "PRACTICE_TODO",
            "Practice TODO"))
    practice_b = b.block("Practice (not scored)", practice_qids)
    start_b = b.block("Part 1 starts", [
        b.question(descriptive(START_TEXT, "start"), "START", "Part 1 starts")])

    # everything built so far is a linear page and gets a fixed spot on the bar
    pre_qids = [q["PrimaryAttribute"] for q in b.questions]

    # the five looped rating blocks
    rating_blocks, counts = [], []
    loop_n, part1_qids = {}, []
    for path in loops:
        n = re.search(r"block(\d+)", path.stem).group(1)
        static, rows = load_loop(path)
        # rows MUST nest under "Static" — a flat LoopingOptions is rejected by the
        # import schema with "Additional property <n> is not allowed".
        looping = {"Static": static, "Randomization": "All"}
        qid = b.question(rating_question(include_audio=True), f"RATE{n}", f"Rating block {n}")
        rating_blocks.append(b.block(f"Rating block {n}", [qid], looping=looping))
        counts.append((path.name, rows))
        loop_n[qid] = rows
        part1_qids.append(qid)

    # --- part 2: break, instructions, N looped A/B blocks (N = however many
    #     loop_s2_block*.csv exist; currently 10 x 24 comparisons) ---
    s2_blocks, s2_counts = [], []
    mid_qids = []
    break_b = part2_instr_b = None
    if not args.no_part2:
        s2_loops = sorted((ROOT / args.loop_dir).glob(args.s2_loop_glob),
                          key=lambda p: int(re.search(r"block(\d+)", p.stem).group(1)))
        if not s2_loops:
            sys.exit(f"no {args.s2_loop_glob} in {args.loop_dir} — "
                     "run scripts/mos/make_session2_loops.py (or pass --no-part2)")
        break_q = b.question(descriptive(BREAK_TEXT, "break"), "BREAK", "Break")
        break_b = b.block("Break", [break_q])
        instr2_q = b.question(descriptive(PART2_INSTRUCTIONS.format(skip=SKIP_LABEL),
                                          "p2instr"), "INSTR2", "Part 2 instructions")
        part2_instr_b = b.block("Part 2 instructions", [instr2_q])
        mid_qids += [break_q, instr2_q]
        for path in s2_loops:
            n = re.search(r"block(\d+)", path.stem).group(1)
            static, rows = load_loop(path, fields=3)
            looping = {"Static": static, "Randomization": "All"}
            qid = b.question(ab_question(), f"AB{n}", f"A/B block {n}")
            s2_blocks.append(b.block(f"A/B block {n}", [qid], looping=looping))
            s2_counts.append((path.name, rows))
            loop_n[qid] = rows

    # --- part 3: the GRPO arm. A FIXED block, not a randomiser member: Part 2's
    #     randomiser has SubSet 1, so an 11th block there would reach 1 participant in
    #     11 and would also thin the session-2 cells from 5 to 4.5 each.
    grpo_b = grpo_instr_b = None
    grpo_qids, late_qids, grpo_rows = [], [], 0
    grpo_path = ROOT / args.loop_dir / args.grpo_loop
    if not args.no_part3 and grpo_path.is_file():
        instr3_q = b.question(descriptive(PART3_INSTRUCTIONS.format(skip=SKIP_LABEL),
                                          "p3instr"), "INSTR3", "Part 3 instructions")
        grpo_instr_b = b.block("Part 3 instructions", [instr3_q])
        late_qids.append(instr3_q)
        static, grpo_rows = load_loop(grpo_path, fields=3)
        qid = b.question(ab_question(stem=GRPO_STEM), "GRPO", "Part 3 continuations")
        grpo_b = b.block("Part 3 continuations",
                         [qid], looping={"Static": static, "Randomization": "All"})
        loop_n[qid] = grpo_rows
        grpo_qids.append(qid)
    elif not args.no_part3:
        print(f"note: {grpo_path.name} absent, building without Part 3")

    thanks_q = b.question(descriptive(THANK_YOU, "thanks"), "THANKS", "Thank you")
    thanks_b = b.block("Thank you", [thanks_q])

    # every question carries the controller: it advances the progress bar through
    # the loops and keeps the two Part-2 players from sounding at once
    js = controller_js(loop_n,
                       fixed_anchors(pre_qids, mid_qids, [thanks_q],
                                     PART1_BAND, PART2_BAND,
                                     late_qids, PART3_BAND),
                       part1_qids, grpo_qids)
    for q in b.questions:
        q["Payload"]["QuestionJS"] = js

    # flow: preamble -> calibrate -> practice -> rating randomizer
    #       -> break -> part-2 randomizer -> thanks
    def node(bid):
        return {"Type": "Standard", "ID": bid, "FlowID": b._next_flid()}

    randomizer = {"Type": "BlockRandomizer", "FlowID": b._next_flid(),
                  "SubSet": 1, "EvenPresentation": True,
                  "Flow": [node(x) for x in rating_blocks]}
    flow = [node(info_b), node(consent_b), node(instr_b), node(calib_b),
            node(practice_b), node(start_b), randomizer]
    if s2_blocks:
        # the two randomizers are independent: a participant's rating block does
        # not constrain which A/B block they get, so assignment stays balanced
        # even if part 2 is abandoned partway.
        flow += [node(break_b), node(part2_instr_b),
                 {"Type": "BlockRandomizer", "FlowID": b._next_flid(),
                  "SubSet": 1, "EvenPresentation": True,
                  "Flow": [node(x) for x in s2_blocks]}]
    if grpo_b is not None:
        flow += [node(grpo_instr_b), node(grpo_b)]
    flow.append(node(thanks_b))

    # assemble
    out = copy.deepcopy(template)
    # Self-identifying default: <base> (YYYY-MM-DD, 5 rating + 10 A/B). Computed here
    # rather than in argparse because the block counts are only known after globbing
    # the loop tables.
    name = args.name
    if not name:
        parts = [f"{len(loops)} rating"]
        if not args.no_part2:
            parts.append(f"{len(s2_loops)} A/B")
        name = (f"{BASE_SURVEY_NAME} "
                f"({datetime.date.today().isoformat()}, {' + '.join(parts)})")
    out["SurveyEntry"]["SurveyName"] = name
    survey_id = out["SurveyEntry"]["SurveyID"]

    keep = {"SO", "SCO", "PROJ", "RS", "PL", "STAT", "QC"}
    elements = [e for e in out["SurveyElements"] if e["Element"] in keep]
    for e in elements:
        e["SurveyID"] = survey_id
        # Participants must not see the export tags (RATE1, AB3, GRPO). They are
        # analyst-facing labels and showing them above a question both looks unfinished
        # and leaks the block's role. The template has this off, but it is copied
        # verbatim, so a template re-exported from a survey with it on would carry it
        # back in. Pin it here instead of trusting the copy.
        if e["Element"] == "SO":
            e["Payload"]["ShowExportTags"] = "false"

    trash = {"Type": "Trash", "Description": "Trash / Unused Questions",
             "ID": "BL_gentrash", "BlockElements": []}
    elements.append({"SurveyID": survey_id, "Element": "BL",
                     "PrimaryAttribute": "Survey Blocks", "SecondaryAttribute": None,
                     "TertiaryAttribute": None, "Payload": b.blocks + [trash]})
    elements.append({"SurveyID": survey_id, "Element": "FL",
                     "PrimaryAttribute": "Survey Flow", "SecondaryAttribute": None,
                     "TertiaryAttribute": None,
                     "Payload": {"Flow": flow, "Properties": {"Count": b.flid},
                                 "FlowID": "FL_1", "Type": "Root"}})
    for q in b.questions:
        q["SurveyID"] = survey_id
        elements.append(q)

    out["SurveyElements"] = elements
    dest = ROOT / args.out
    dest.write_text(json.dumps(out, indent=2))

    so = [e for e in elements if e["Element"] == "SO"][0]["Payload"]
    print(f"wrote {dest.relative_to(ROOT)}")
    print(f"  blocks   {len(b.blocks)}  questions {len(b.questions)}")
    for name, rows in counts:
        print(f"  loop     {name}: {rows} trials")
    for name, rows in s2_counts:
        print(f"  loop     {name}: {rows} comparisons")
    if grpo_rows:
        print(f"  loop     {args.grpo_loop}: {grpo_rows} comparisons "
              f"(Part 3, fixed block, every participant)")
    # blocks are no longer equal-sized: withdraw_sources.py removes rows without
    # re-sampling, so report the range rather than block 1's count.
    sizes = sorted({c[1] for c in counts})
    n_trials = f"{sizes[0]}" if len(sizes) == 1 else f"{sizes[0]}-{sizes[-1]}"
    n_comp = s2_counts[0][1] if s2_counts else 0
    if s2_blocks:
        tail = " -> part3(fixed)" if grpo_b is not None else ""
        print(f"  flow     preamble -> calibrate -> practice -> start -> "
              f"randomizer(1 of {len(rating_blocks)}) -> break -> "
              f"randomizer(1 of {len(s2_blocks)}){tail} -> thanks")
        extra = f" + {grpo_rows} continuations" if grpo_rows else ""
        print(f"  session  {n_trials} ratings + {n_comp} comparisons{extra} "
              f"per participant")
    else:
        print(f"  flow     preamble -> calibrate -> practice -> start -> "
              f"randomizer(1 of {len(rating_blocks)}, even) -> thanks  [NO PART 2]")
    print(f"  anonymise {so.get('AnonymizeResponse')!r}   geo {so.get('CollectGeoLocation')!r}")
    if not ids:
        print("\n  practice block contains a BUILD TODO — no practice clips designated")


if __name__ == "__main__":
    main()
