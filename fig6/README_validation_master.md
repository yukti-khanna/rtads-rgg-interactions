# Reusable Figure 5 validation in fig6

The scripts are prepared for local execution. They have not been executed or rendered by the agent for this request, as instructed. The master CSV, new paper panels and actual combined counts will be produced by the command below. Existing outputs from the previous workflow are retained; they are not the new master-based deliverables.

## Run with the current data

```bash
cd /Users/yuktikhanna/rtads-rgg-interactions/fig6
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/fig6-mpl \
  /Users/yuktikhanna/miniconda3/envs/calvados/bin/python3 run_validation.py --verify
```

This runs isolated synthetic fixtures first, then builds the master, five paper panels and expanded diagnostics. It does not run model inference, historical recovery, training, RDF processing or B22 integration. `make_validation_panels.py` and `run_validation_workflow.py` now delegate to the same entrypoint. The older individual calculation/recovery scripts remain available but are unnecessary for updating experiments.

## Replace experiments

```bash
python run_validation.py --experiments path/to/expanded_kd_table.csv
```

Use the `calvados` interpreter shown above if it is not activated. Paths resolve from fig6, including when invoked from another directory. The experimental input replaces previous annotations. It does not append old measurements or copy experimental fields from the recovered historical prediction file.

`experimental_input_template.csv` supplies these headings:

```csv
pair_id,Kd_uM,Kd_error_uM,Outcome,Notes
```

Accepted outcome labels are `Kd fit`, `LB`, and `NB`; `Kd_fit` and `Kd-fit` are normalized to `Kd fit`. Missing outcome plus a positive Kd is recognized as a Kd fit. Missing Kd never implies NB. Untested or unannotated computational pairs stay untested. Other supplied outcome text is retained but excluded from the three-category test.

For different headings, explicitly map them and specify units:

```bash
python run_validation.py --experiments expanded.csv \
  --pair-column Pair --kd-column KD --kd-error-column KD_error \
  --kd-units uM --outcome-column Outcome --notes-column Notes \
  --uncertainty-type 'reported SD'
```

Units supported: M, mM, uM, nM. An ambiguous `KD` heading is not assigned units automatically. Reported errors are preserved and scaled only for unit conversion; they are never silently changed from SD to SEM. Use `--kd-error-units` if the error has different units from Kd. Standard `Kd_error_uM`/`Kd_uM_err` headings explicitly denote uM. `pKd_error` uses first-order propagation of the reported error and retains its interpretation. Errors unavailable in the input remain NA.

## Selected sources

`validation_sources.json` records paths, columns, transformations and matching rules. Paths were resolved from the actual workspace, not from modification times. `workflow_history/` is excluded from selected sources.

| Quantity | Source | Selected definition |
| --- | --- | --- |
| Experiments | `Table_S8.csv` | `pair_id`, `Kd_uM`, `Kd_uM_err`, `Outcome`, `Notes`; Kd units uM |
| Final prediction | `preds_val_updated/predictions_minimal.csv` | `final_hybrid_prediction`, already log10(-B22); no second log |
| Prediction metadata | `preds_val_updated/predictions.csv`, `prepared_pairs.csv` | Separately prefixed supplied metadata and additional descriptors |
| Simulation | `updated_validation/b22/per_pair_b22.csv` | `method=legacy_sum`, boundary 7.5 nm, `correction=primary_tail_offset`; `corrected_B22_mean` then log only if negative |
| Current features | `preds_val_updated/feature_matrix.csv` | All columns, keyed by explicit `feature_row` in `feature_row_ids.csv`; manifest hashes bind the map and matrix |
| Sequence registry | `preds_val_updated/resolved_pairs.csv` | Oriented pair ID, exact sequences, lengths and hashes |
| Historical predictions | `prediction_diagnostics/historical_model/same_construct_predictions.csv` | Existing recovered `Old_same_construct_final` / base; never rerun recovery |
| Historical descriptors | `prediction_diagnostics/historical_model/same_construct_historical_features.csv` | All ten columns separately prefixed `Historical_feature_` |
| Additional archived descriptors | `updated_validation/qc/validated_final_features_data_pred_b22s_mapped.csv`, `validated_features_data_mapped.csv` | All mapped columns separately prefixed, with original construct lengths and matching status; not replacements for current descriptors |
| Accuracy eligibility | `updated_validation/b22/per_pair_simulation_eligibility.csv` | Existing pair audit, bound to selected log values and exact sequences through `paired_simulation_accuracy_source.csv` |

The current supplied prediction, current feature and recovered historical prediction tables each contain 73 rows. The experimental table contains 20 pairs, including 9 Kd fits, 4 LB and 7 NB. The existing simulation audit supports 20 accuracy pairs and leaves 53 with unresolved provenance. These are input observations, not claims that the new unexecuted master has passed verification; the local run writes actual counts in `master_summary.json`.

Current simulation values are retained for every selected pair even when accuracy eligibility is unresolved. Nonnegative B22 remains numeric with a missing log and reason. Replica count, SD and SEM remain distinct; unknown replica count 0 in a mean-only source is documented as unknown, never treated as evidence of zero replicas. The historical radial sum lacks dr, so no physical units are added or alternative integral silently substituted.

The read-only references were inspected: `idr-interaction-analysis` at `80b07345c7a50e12e5118a72ec461f4fe5db5eb0` (untracked `fig4/`), and `idr-interaction-predict` at `af534d380250e6e9e89ec0c9f1c544106f67cb2b` (untracked `models_old/` and `sample_rdfs/`). Neither has tracked changes. No applicable AGENTS.md was found. Neither reference repository nor manuscript document was modified.

## Matching and conflicts

The master uses the union of selected pair identities. Joins are one-to-one by exact canonical pair ID or a verified explicit alias. Full display names generated from the sequence registry also have an explicit exported mapping; an exact display-name match is allowed only when unambiguous. Short protein names are not approximately matched. THOC4 N/C, construct boundaries and individual shuffles stay separate.

`validation_pair_aliases.csv` accepts explicitly verified aliases with construct evidence. Supplied alias sequences are checked against the selected registry when available. Unrecognized identities remain in the master with missing computational values; they are not discarded. Source identities and join rules are exported in `pair_matching.csv`.

Every experimental row is preserved in `experimental_measurements.csv`, including extra condition columns and original reported Kd/error text. Duplicate rows, even identical duplicates, require explicit selection. Their pair-level Kd/outcome remains NA until resolved, and unrelated pairs continue through the workflow. Non-numeric/censored values remain in the supporting table and are marked unresolved instead of being converted to invented measurements. No averaging or arbitrary first-row selection occurs.

Selection example (`selection.json`):

```json
{"thoc4_cter__foxo4_cr3": "condition_A"}
```

```bash
python run_validation.py --experiments expanded.csv \
  --measurement-id-column measurement_id --selection-config selection.json
```

Without a measurement ID column, values in the selection JSON are one-based input row numbers. Prior row selections are cleared when replacing an experimental file. Explicit sequence conflicts are excluded from relevant experimental comparisons.

## Outputs from your run

- `updated_validation/validation_master.csv`: requested eight essential columns first, then all descriptors, sequences, parents, prediction/simulation values, uncertainty, sources and status fields; missing numeric values are written as NA.
- `column_dictionary.csv`, `source_selection.json`, `resolved_validation_config.json`: definitions, exact source paths/hashes and reusable settings. Model-input feature columns are explicitly marked in the dictionary and source report. Archived/current feature definitions remain separate.
- `experimental_measurements.csv`, `experimental_conflicts.csv`, `pair_matching.csv`, `unmatched_pair_identities.csv`, `missing_data_report.csv`, `verified_display_aliases.csv`: measurements, joins and missing/provenance status.
- `paper_figures/`: `prediction_vs_pKd`, `prediction_vs_outcome`, `simulation_vs_pKd`, `simulation_vs_outcome`, `prediction_vs_simulation`, each as PDF, editable SVG, 600-dpi PNG, exact plotted CSV, statistics CSV and separate caption; `paper_overview.png` combines the five.
- `analysis_reports/`: per-panel inclusion/exclusion reports; old/new paired summaries, partner/outcome summaries, ranking changes, paired Kd correlation differences, paired outcome effect/AUROC differences, WT/shuffle changes and simulation accuracy differences.
- `prediction_diagnostics/figures/master_*`: updated diagnostics from the master, leaving existing historical diagnostic exports intact. Long paired-dot lists and heatmaps split into pages. Counts follow the selected current rows.
- `master_figure_manifest.csv`, `master_figure_axis_limits.json`, `MASTER_METHODS.md`: exports, limits and methods.

Paper panels have short axes, protein identities, partner shapes, category counts and compact statistics. No workflow/provenance notes appear on their canvases. Each paper panel derives its own limits from displayed values/errors; final-predictor Kd axes do not inherit the historical minimum. Identity plots share equal x/y limits and aspect. Only directly paired old/new diagnostic panels share their limits. Delta plots include zero. Manual limits can be configured, but limits that crop data or error bars are rejected:

```json
"plot_limits": {"prediction_vs_pKd": {"x": [3.8, 4.3], "y": [3.0, 6.0]}}
```

Those numbers illustrate syntax; leave the default empty configuration for data-derived limits.

Statistics use exact labelled-row Spearman permutations through n=9, otherwise reproducible Monte Carlo permutations. Missing/constant or insufficient samples produce unavailable statistics. Outcome tests use permutation Kruskal–Wallis and exact or Monte Carlo two-sided rank tests with Holm correction across three planned comparisons. Pairwise p-values/effects are in captions/tables, not oversized figure footers. AUROC treats Kd fit/NB as binary and excludes LB. Model differences resample identical rows. Shared constructs make inference exploratory.

## Overrides and isolated checks

`--predictions`, `--simulations`, `--features`, `--config`, `--outdir` and `--diagnostics-dir` override selected paths/settings. Output paths must remain in fig6. A feature override should contain explicit `pair_id`; an anonymous matrix needs matching `--feature-row-ids` and `--feature-manifest`. Simulation overrides still require the configured convention columns and filters; existing accuracy approval does not transfer to changed values/sequences.

```bash
python run_validation.py --master-only
python run_validation.py --paper-only
python verify_master_workflow.py
```

Fixtures live in temporary `.validation-fixture-*` folders inside fig6 and are removed after checks. They exercise new annotations, unmatched pairs, conflicting duplicates, explicit selection, replacement semantics, feature identity mapping, nonnegative simulation values, and data-driven limits/statistics. They have been prepared but not executed by the agent.

After your local run, inspect the five final PNGs and the overview at publication size. The agent has not rendered or visually certified these new panels, consistent with your instruction to run locally. Automated export records do not claim visual verification.
