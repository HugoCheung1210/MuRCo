# scripts/

All Python for the project, grouped by what it is *for*. Run everything **from the repo
root** (`python scripts/<group>/<file>.py`).

Two environments, never mixed (PROJECT_STATE.md §6). Most of this runs in the local **DSP
env** (anaconda base); the files marked **GPU** run only in `mfenv` on the AutoDL box.

> On the GPU box the scripts sit **flat at the repo root** — no `scripts/` prefix and no
> subfolders. That still works: the sibling-import shim (see `_paths.py`) falls back
> silently when the grouped layout isn't there.

## core/ — the metric itself
| file | what it does |
|---|---|
| `coherence_dimensions.py` | The H/T/R/S contract: `Dimension.score(a, b, sr) -> [0,1]`. Everything else builds on this. |
| `score_separability.py` | Scores a manifest across chosen dimensions → `separability.json` + `pair_scores.csv`. |
| `assemble_matrix.py` | Renders a `separability.json` to the Markdown/LaTeX perturbation matrix. |
| `compute_C.py` | Aggregates C, runs leave-one-out ablations, fits weights to MOS (NNLS). |
| `compute_S_value.py` | "Does S earn its place?" — identity-discrimination AUC, cross/same-genre split. |

## data/ — sources and the perturbation dataset
| file | what it does |
|---|---|
| `select_fma_sources.py` | Pick the 90 FMA-Medium source clips. |
| `select_jamendo_sources.py` | Jamendo alternative to the above. |
| `download_selected_fma.py` | Fetch the selected clips into `raw_audio/`. |
| `review_sources.py`, `withdraw_sources.py` | Audition sources; withdraw ones that fail the audit. |
| `read_meta.py` | FMA metadata helper. |
| `make_perturbation_dataset.py` | Builds the 1980 pairs (90×22). Needs the `rubberband` **binary**. |
| `make_method_subset.py` | Carves the smaller method-development subset. |

## s_backbone/ — the S term's model (mostly **GPU**)
| file | what it does |
|---|---|
| `mf_probe.py` | **GPU** Music Flamingo loader + prompting primitive. Everything MF imports this. |
| `score_s_mf.py` | **GPU** Scores S over a manifest → the `s_scores.json` cache. Resumable. |
| `score_s_omni.py` | **GPU** Same, on Qwen2.5-Omni (the robustness replication). |
| `mf_eval.py`, `mf_eval_genre.py`, `mf_stats.py`, `mf_multiturn.py`, `mf_coherence_probe.py` | **GPU** MF evaluation harnesses and prompt probes. |
| `build_triples.py`, `build_triples_genre.py` | Build the A/B/C triples the MF evals consume. |
| `compare_s_backbones.py` | MF vs Omni: does the S signature reproduce? |
| `make_s_variant_scores.py`, `compare_subset_signature.py`, `analyze_template_bank.py` | Prompt/gap/template variants and their effect on separability. |

## drift/ — iterative drift (the regime where S accumulates)
| file | what it does |
|---|---|
| `drift_v4.py` | The drift protocol core. Imported by every runner here. |
| `drift_htrc.py` | Adds H/T/R/C tracking on top of `drift_v4`. |
| `run_drift_htrc.py`, `run_drift_baselines.py` | Runners over ACE / MusicGen / Stable Audio. |
| `build_drift.py` | Assembles drift sequences. |
| `analyze_drift_selectivity.py`, `analyze_temporal_family.py` | Post-hoc analysis of the drift runs. |
| `test_drift_v2.py` | Smoke test for the older protocol. |

## generators/ — external model wrappers (**GPU**)
`ace_protocol.py` (ACE-Step) · `sao_inpaint_protocol.py` (Stable Audio inpainting) ·
`vampnet_protocol.py` (VampNet). Imported by `drift/` and `rerank/`; each needs its own
model checkout, so they fail fast outside the GPU box.

## rerank/ — best-of-N reranking with C
| file | what it does |
|---|---|
| `rerank_bestofn.py` | **GPU** The E-D experiment: rerank N candidates by C vs CLAP vs random. |
| `rebuild_rerank_windows.py` | Rebuilds the scoring windows for a completed run. |
| `rerank_listen.py` | Records which candidate each ranker picked, per seed, into `listen_picks.csv`. Its playback companion `play_listen_picks.py` is a local audition tool and is not in the public release. |
| `prepare_mos_ab.py`, `score_mos_ab.py` | The A/B listening study on rerank output. |

## mos/ — the human MOS study
| file | what it does |
|---|---|
| `make_mos_stimuli.py`, `render_mos_stimuli.py` | Plan and render the 13 s (6+1+6) mp3 stimuli with opaque filenames. |
| `select_practice_clips.py`, `render_practice_stimuli.py` | The practice/training block. |
| `review_levels.py` | Per-source loudness audition; produces the attenuation map. |
| `build_mos_qsf.py` | **Generates the Qualtrics survey (.qsf)** — the survey is built + imported, never hand-edited. |
| `qualtrics_question_js.js` | The in-survey controller `build_mos_qsf.py` stamps into every question: advances the progress bar through the Loop & Merge blocks (Qualtrics freezes it there) and stops the two Part-2 players sounding at once. |
| `make_qualtrics_loops.py`, `make_session2_loops.py` | Loop-table CSVs for each block. |
| `stamp_qualtrics_urls.py` | Stamps harvested opaque `CP/File.php?F=F_<id>` URLs into the loop tables. |
| `qualtrics_to_responses.py`, `score_mos_responses.py`, `merge_mos_scores.py` | Ingest exported responses → scored MOS. |
| `simulate_mos_power.py`, `simulate_mos_responses.py` | Power simulation, and the simulated raters used to fix the attention-check rule before any real rating was scored. |
| `select_ab_trials.py`, `select_ab_contrast.py`, `prepare_ab_session2.py`, `score_ab_session2.py` | Session-2 A/B head-to-head (C vs CLAP/SCS/C_noS). `prepare_ab_session2.py` refuses to overwrite an existing `key.csv` without `--force`, because rewriting it re-randomises which option was C's pick and the collected responses record only a trial id and a rating; a forced rebuild carries over any column a later analysis added. |

## analysis/ — comparisons, sensitivity, figures
| file | what it does |
|---|---|
| `score_baselines.py`, `make_combined_baseline.py` | CLAP / SSM / combined baselines on the same pairs. |
| `weight_sensitivity.py` | How much do the C weights matter? |
| `dose_response.py` | Monotonicity of C against perturbation strength. |
| `diagnose_perturbation.py` | Per-perturbation diagnostics behind the matrix. |
| `score_cocola.py`, `analyze_cocola.py` | COCOLA (nearest prior metric) over the matrix and the rerank candidates. Needs its own venv (torch 2.2 + lightning), never the DSP env. |
| `score_rivals.py` | TuneJury (preference reward) and SongEval (aesthetics) over the matrix and the rerank candidates — the two rivals from `doc/notes/rival_check_2026-08-16.md` that ship public checkpoints. Absolute metrics adapted to the pair contract, emitting `_a`/`_b`/`_cat`/`_rel`. Own venv (torch ≥2 + laion_clap + muq), GPU box; see `doc/notes/rivals_runbook.md`. |
| `analyze_cocola.py --rival tunejury\|songeval` | Same per-family drop table and re-ranker test as COCOLA, on a rival CSV from `score_rivals.py --csv`. |
| `bestofn_curve.py` | Best-of-N headroom curve from cached candidate scores, by exact order statistic. No GPU. |
| `clap_probe.py` | R2 — does a **raw CLAP embedding** diagnose the perturbation family? `extract` (needs torch+transformers ≥4.27; locally the `SNLP` env) writes `results/baselines/clap_embeddings.npz`, `probe` (DSP env, needs sklearn) reruns `diagnose_perturbation.py`'s folds on 768-d/512-d representations. Closes the "4 features vs 1 scalar" objection. |
| `mos_delta_r.py` | Cluster-bootstrap CIs on *differences* of MOS correlations (C vs C_noS vs CLAP), resampling source tracks. Reports Pearson $r$ and Spearman $\rho$ per metric; the contrasts and their intervals are Pearson. Uniform weights by default so no parameter is fitted. This is what certifies S's Part-1 contribution — see `doc/notes/icassp_review_2026-08-17.md`. |
| `review_fixes.py`, `diagnose_nested_cv.py` | Numbers for the 2026-08-23 internal reviewer pass: nested-CV diagnosis, the HTR ablation row, within-condition MOS, a fitted CLAP head, ties-as-half. `diagnose_nested_cv.py` keeps its work behind `main()`, so importing `loto` from it does not re-run the whole diagnosis. |
| `part2_holm.py` | Part 2 multiplicity. Cluster-bootstrap p per arm on the statistic Table 4 reports (win rate over DECIDED ratings, ties dropped), then Holm across the family. `review_fixes.py` corrects a different statistic (ties as half), which is why the two adjusted p values differ. Also reports a second Holm run with the post hoc MuseCPEval comparison folded in, and `--write-key` records its per-trial pick in `key.csv` as `musecp_pick`, never in `settles`. |
| `review_response.py` | Numbers for the **external** review of 2026-08-24 — held-out gap replication, the probe-polarity control, the three-way temporal split, accumulation d_z with CIs, R-gate beat strength, and Part 2 split by the rival's own margin. All from cached data, no GPU. `--write-gap-subset` emits the held-out pair list for the A2 re-run. See `doc/notes/review_response_runbook.md`. |
| `tau_sensitivity.py` | Sweeps T's temperature analytically (no rescoring); shows the family signature survives τ ∈ [3,12]. |
| `ssl_probe.py` | A4 — MuQ-large and MERT-v1-330M as **masked-prediction** (non-text-anchored) baselines. `extract` (rivals venv, GPU box) writes `results/baselines/<enc>_embeddings.npz` in `clap_probe.py`'s contract, so `build_feature_sets()` picks it up with no new loader; `cosines` (DSP env) emits the Table 3 rows. |
| `make_figures.py` | Regenerates everything in `figures/`. |
| `fig_grpo.py` | Section 4.6 Figure 4.5 (best-of-N frozen vs tuned; the three reward arms). Numbers transcribed from the §4.6 evaluation, plotting layer only. |
| `fig_grpo_training.py` | Section 4.6 Figure 4.6 (why step 150). Reads `results/rl/eval_pilot2/pair_scores.csv` and `results/rl/grpo_pilot2/train_log.jsonl` directly. |
| `fig_architectures.py` | The two Chapter-2 schematics (generator families, edit loops). |

## rl/ — GRPO fine-tuning against C (pilot; `doc/notes/grpo_pilot_runbook.md`)
| file | what it does |
|---|---|
| `copy_metrics.py` | How much does B repeat its context? Defines the metrics and baselines them on the frozen generator, which is what later detects reward hacking. |
| `reward.py` | The reward itself: `log C − λ·copy`, with S pluggable (`CachedS` for CPU testing, `MusicFlamingoS` live on the GPU) and the GRPO group-advantage helper. |
| `make_contexts.py` | Freezes the train/held-out context set, split by track, genre-stratified, ethics-screened sources only. |
| `generate_continuations.py` | Frozen-MusicGen continuations via transformers: times the rollouts for the budget and produces the frozen-policy baseline the copy hinge comes from. The RL loop imports this generation path. |
| `grpo_train.py` | The GRPO loop: LoRA policy, group sampling, log-prob recomputation, group-relative advantage, KL to the adapter-disabled reference, checkpoints. `--check` validates every risky assumption and reports training VRAM without training. |
| `eval_grpo.py` | Held-out evaluation: `frozen` vs `frozen_boN` vs each checkpoint on C, CLAP, SCS and the copy metrics, cluster-bootstrapped over contexts. `generate` writes a standard pairs manifest so the existing scorers do the middle of the pipeline. |

## `make_release.sh`
Builds the public release tree from this workspace. The release is a **function** of the
workspace, never a second place to edit: re-run it after any change worth publishing, then
commit at the destination. Only files git already tracks are copied, so the `.gitignore`
rules that keep audio, weights and caches out of version control keep them out of the
release too. `--dry-run` lists what would ship.

Two behaviours to know. It **refuses to build** if a tracked file embeds audio as a
base64 `data:` URI, because path-based rules alone let `listen_ui.html` through and the
generated stimuli may not be redistributed under the approved application (D2.1). And its
`rsync` does **not** delete, so a file dropped from the workspace lingers at a destination
built over the top; after any deletion, build into an empty directory instead.

## `_paths.py`
Puts every `scripts/<group>/` on `sys.path` so the bare-name sibling imports keep
resolving across groups. Imported by a small shim in the ~18 scripts that need it; the
shim tolerates the module being absent, which is what keeps the flat GPU-box layout
working. See the Conventions section of the project README.
