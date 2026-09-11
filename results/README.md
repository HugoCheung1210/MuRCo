# results/

Two kinds of thing live here: **grouped artefacts** from the main perturbation-matrix
pipeline (the first five folders), and **per-run output trees** from separate experiments
(the rest). Each run tree carries its own `separability.json` — there is no top-level one,
so `assemble_matrix.py --in` always names a run.

## Grouped artefacts

| folder | contents |
|---|---|
| `coherence/` | `pair_scores*.csv` (per-pair H/T/R/S), `C*.{csv,json}` (aggregate C + the `C_no{H,T,R,S}` leave-one-out ablations), `matrix.{md,tex}`, `selectivity.tex`, `weight_sensitivity.json`. |
| `s_cache/` | `s_scores*.json` — the **S cache**, computed GPU-side and served by `pair_id`. Variants: `_gap0`, `_secs6`, `_temporal`, `_omni`, `_pt`, plus `s_subset_*` (gap/prompt sweeps) and `template_bank_*`. |
| `drift/` | `drift_htrc_*`, `drift_baselines_*`, `drift_cbase_*` per generator (ACE / MusicGen / Stable Audio), and `drift_selectivity.json`. |
| `baselines/` | `baseline_scores*.json` (CLAP / SSM / combined) and `cbase_build_report.json`. |
| `diagnostics/` | `diagnosis*.{json,tex}` and `dose_response*.{json,tex}`. |

## Per-run trees

| folder | run |
|---|---|
| `mos/` | The human-MOS study. `final/N40.csv` is the closed raw export of all 40 sessions and `mos_live.csv` the analysed set derived from it. `responses/` holds the 40 Part 1 sessions and `session2/` the 80-trial Part 2 forced choice with its `key.csv` design, whose `settles` column names the five arms the trials were built to decide and whose `musecp_pick` column records MuseCPEval's per-trial preference, added afterwards. `session2/holm.json` carries the per-arm bootstrap p values and the Holm correction across them. `responses_sim/` is the 40 simulated raters the attention-check rule was calibrated on while blind to real data. `stimuli/`, `practice/` and `qualtrics/` (loop tables + `.qsf`) are the materials; `level_review/` and `source_review/excluded_sources.json` the screening records: 27 sources excluded, 14 attenuated and 4 kept with a warning, each with a reason. The audition page that produced those marks is not released, since it plays the source recordings and those are not redistributed. |
| `rerank_ace/` | Best-of-$N$ re-ranking (E-D) on **ACE-Step** inpainting, 40 seeds x 8 candidates, so 320 in all (`rerank_manifest.json` records `regime: ace inpaint`). This is the run the thesis and the paper report, and the tree the listening study's Part 2 draws on. |
| `rerank/` | The earlier **Stable Audio Open** run of the same experiment, 10 seeds (`regime: SAO inpaint`). Superseded by `rerank_ace/` and cited by neither document, so it is **not in the public release**. |
| `omni/` | Qwen2.5-Omni robustness replication of the S signature. |
| `musecp/` | MuseCPEval (Vishe et al., ISMIR 2026; arXiv 2512.14629v2) run as a rival, with $A$ as the reference and $B$ as the estimate. `setup1/`, `temporal/` and `rerank_ace/` hold the raw `musecpeval --batch-json` output for the 1,890 Setup-1 pairs, the 540 temporal pairs and the 320 ACE-Step candidates (`pairs.json` manifest, `summary.csv` one row per pair, `results.json[l]` nested, `run.log`; 0 failed pairs, 2 degraded in Setup 1 where the beat tracker was empty). Each carries a `musecp_metrics.{csv,json}` from `musecp_to_metrics.py`, the 12 metrics polarity-aligned plus our equal-weight `mean12`/`mean10` composites; `key_relatedness` and `delta_bpm_folded` rise with change and are flipped there. The readouts are `part1_delta_r.json` (marginal correlations with the 154 mean ratings and the C-minus-metric contrasts, `mos_delta_r.py --rivals`), `part1_fit12.json` (held-out ridge, leave-one-source-track-out, 12 metrics against MuRCo's 4), `part1_within.json`, `displacement.json` (same-track displacement against other-track substitution, raw and control-normalised) and `part2.json` (each metric as a selector on the 80 Part 2 trials, a reproduction of the published rows, and MuseCPEval's own table row under `agreement_with_C.musecp_mean12.as_table_row`). The aligned-copy diagnosis is `diagnostics/diagnosis_musecp.json`. `timing/` is a small timing run, not used. Do not edit `summary.csv`; new analyses write new files. S sits 0.265 lower in the temporal run than in Setup 1 while every other metric matches to 0.001, so any AUC pooling the two runs must be control-normalised. |
| `temporal/` | The T1 temporal-variant family. |
| `mos_secs6/` | The 6 s window validation that the study adopted. |

