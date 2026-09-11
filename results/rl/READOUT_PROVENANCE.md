# Which Yes/No read-out each RL run used

Written 2026-08-22, because `grpo_pilot2/config.json` has thirty fields and `token_agg` is
not one of them. The training read-out was inherited from whatever `mf_probe.TOKEN_AGG`
happened to be on the machine rather than pinned by the command, so for the three runs
below it has to be established after the fact. `grpo_train.py` now takes `--token-agg`
(default `variants`), pins it on the reward's `MusicFlamingoS`, and records it in
`config.json`, so no run after this one needs an entry here.

| run | training read-out | evaluation read-out | how it is known |
|---|---|---|---|
| `grpo_pilot2` (seed 20260807) | `variants` | `variants` | eval cache metadata; training corroborated by S level |
| `grpo_seed2` (seed 20260823) | `variants` | set explicitly at eval time | box banner printed `TOKEN_AGG=variants` at launch; S level matches pilot2 |
| `grpo_seed3` (seed 20260824) | `variants` | set explicitly at eval time | as above |
| `grpo_noS`, `grpo_clap` | `variants` | `variants` | eval cache metadata (`eval_noS`, `eval_clap`) |

## How the training read-out was established

`eval_pilot2/s_scores.json`, `eval_noS/s_scores.json` and `eval_clap/s_scores.json` all
record `"token_agg": "variants"`, which fixes the **evaluation** side directly. The
**training** side is a different code path, since it scores S live through
`reward.MusicFlamingoS` rather than through `score_s_mf.py`, and nothing in that path set
the value. It is settled two ways.

Mechanically: `grpo_train.py` constructed `MusicFlamingoS(secs, gap_s, precision)` with no
read-out argument, and `reward.py` called `mf._yes_prob(...)`, which reads the module-level
`mf_probe.TOKEN_AGG`. The rented box carries a copy of `mf_probe.py` whose default is still
`variants`, and its banner prints that on every launch. Nothing could have overridden it.

Empirically, from the per-group `dims.S` in `train_log.jsonl` over the first eight steps:

| run | S range | S mean | C mean |
|---|---|---|---|
| pilot2 | 0.254–0.571 | 0.393 | 0.615 |
| seed2 | 0.295–0.506 | ~0.404 | ~0.634 |
| seed3 | 0.364–0.520 | ~0.471 | ~0.665 |

Under `single` these would sit roughly a quarter of a scale higher: the same control pairs
score 0.802 under `single` and about 0.53 under `variants`. They do not, so all three
trained under `variants` and are mutually comparable.

## The trap this leaves

The repo copies of `mf_probe.py` and `score_s_mf.py` default to `single`, which is correct
for the perturbation battery and wrong for anything in this directory. Two consequences:

- Pass `--token-agg variants` to `score_s_mf.py` when evaluating any RL run. It is not the
  default and the wrong value fails silently, moving S by about a quarter of its range
  while leaving every rank-based check looking healthy.
- Do not sync the repo copies of those two scripts to the box while an RL evaluation is
  outstanding.

`doc/notes/grpo_seeds_runbook.md` carries the same warning next to the commands it applies
to. Appendix B of the thesis states which published result rests on which read-out.
