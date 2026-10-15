# Figure 5 validation workflow in `fig6/`

The current reusable entrypoint is now `run_validation.py`; see [README_validation_master.md](README_validation_master.md). It combines existing computational results with replacement experimental tables and generates `updated_validation/paper_figures/` from the master CSV. The compatibility entrypoints below now delegate to it and do not rerun B22 calculations or model recovery. The older individual steps described here are retained as historical workflow documentation.

The corrected scripts are prepared for you to run locally. A regeneration was partially executed before your latest instruction to stop execution: numerical checks and analysis stages completed, and plotting started. The runner was stopped before final report generation and expanded verification. Existing generated files may therefore mix stages; rerun the full command below before treating them as final. Final visual inspection remains pending. See `LOCAL_RUN_STATUS.md` for this handoff status.

Individual panels use named protein labels, compact statistics strips, relevant shared axis limits, paired 20-pair plots and complete-set heatmaps. A completed local run writes `updated_validation/RESULTS_REPORT.md` and `prediction_diagnostics/METHODS.md`. Visual inspection must be recorded separately after checking regenerated figures; those files are not evidence of completed visual QA unless they exist and match the final image hashes.

Original inputs, experimental annotations, frozen model artifacts, reference repositories and DOCX files are unchanged. Earlier generated outputs were retained under `workflow_history/before_expanded_diagnostics_20261001/`. All new scripts and outputs stay in `fig6/`.

## Exact local command

```bash
cd /Users/yuktikhanna/rtads-rgg-interactions/fig6
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/fig6-mpl \
  /Users/yuktikhanna/miniconda3/envs/calvados/bin/python3 run_validation_workflow.py
```

The existing `calvados` environment supplies NumPy, pandas, SciPy, matplotlib, Pillow and h5py. Historical bundle recovery automatically uses the existing `idr_features` interpreter for NumPy-2-compatible deserialization. Saved Keras Dense arrays are extracted through a separate `calvados` subprocess and used for deterministic inference. No model is fitted, recalibrated or replaced. This avoids unsafe loading of NumPy-2 pickles in NumPy-1 environments without installing packages or modifying either environment.

Individual steps, using that same `calvados` interpreter:

```bash
python3 verify_validation_workflow.py
python3 update_validation_predictions.py
python3 calculate_validated_b22.py
python3 compare_old_new_predictions.py --bootstrap 10000 --seed 20261001
python3 expanded_validation_analysis.py
python3 recover_historical_predictions.py
python3 audit_simulation_compatibility.py
python3 complete_prediction_diagnostics.py
python3 make_validation_panels.py
python3 verify_validation_workflow.py --outputs
python3 write_validation_results_report.py
python3 verify_expanded_workflow.py
```

Default paths resolve relative to the script folder. Rerunning replaces generated files in the dedicated directories, preserving inputs and `workflow_history/`.

## Distinct analyses

1. **All 20 experimental pairs:** archived `Table_S8::Pred_log10_negB22_hybrid` versus supplied new final predictions, regardless of Kd. Named paired-dot/delta plots, outcome-coloured identity scatter, rank changes and partner/outcome summaries.
2. **Nine Kd fits:** all eligible pairs including FOXO4–G3BP1. Old/new panels share limits derived from the union of predictions and pKd errors. Exact two-sided Spearman permutations enumerate all 9!=362,880 labelled-row permutations. Metric confidence intervals and differences use descriptive row bootstrap.
3. **Experimental outcomes:** matching panels with counts, permutation Kruskal–Wallis, exact two-sided pairwise rank tests, Holm-adjusted p-values and rank-biserial effects. Kd-fit/NB AUROC includes 16 pairs and excludes LB without assigning it an affinity rank. Differences resample identical rows for both models.
4. **All 73 exact construct pairs:** newly generated old inference is separate from archived panel values. Matching weights were recovered from `models/pretrained/mlp_plateau_calibrated_bundle_sel_cols.pkl`; `models_old/` did not reproduce the panel. The archived final values use an **ungated clipped residual application**. Testing fixed application conventions reproduced all 20 old base/final values within 1e-6, with unchanged weights, lambda and cap. All ten historical feature columns reproduce. Named heatmaps and WT/shuffle comparisons use these exact-construct old/new predictions.
5. **Simulation accuracy:** a per-pair audit supports 20 pairs on the legacy 7.5 nm corr1 scale, with exact current final-run topology sequences and archived RDF numerical reproduction. Adjacent topology/RDF linkage is contextual, not an embedded generator manifest. The other 53 pairs lack sufficient provenance. R², RMSE, MAE, signed error, Pearson/Spearman and paired differences compare identical eligible rows against simulations, never predicted B22 against pKd.

P-values are exploratory because pairs share constructs. Prediction shifts or old/new correlations do not establish better accuracy. Separate model p-values are not compared as a superiority test.

## Numerical conventions

`calculate_validated_b22.py` retains individual replicas and exports:

- `legacy_sum`: historical `-2*pi*sum((g-c)*r²)` at 7.5 nm, without dr; no physical units assigned. The centre-only tail integral retains the original 2 nm divisor, even when the sampled span is 1.9 nm.
- `integrated_same_cutoff`: radial integration at 7.5 nm with full-window interpolated endpoints, in nm³.
- `integrated_L_half`: radial integration at L/2, in nm³.

Raw/corrected values, tail offsets, spacing, cutoff, hashes, replica counts, sample SD and SEM are separate. `log10(-mean_B22)` follows replica averaging. Positive/zero means remain with missing logs and reasons; mean-only RDFs have unknown replica counts and unavailable uncertainty. Physical integrals never replace frozen legacy targets.

The newer RDFs differ from old simulation CSV values. `convention_source_replica_investigation.csv` tests cutoffs, sources and replica subsets per pair; archived RDF candidates reproduce the 20 historical values. Shorter FOXO4 constructs in old `run_validated/` are audited separately from matching current `run_validated_final/` inputs.

## Outputs

- `updated_validation/`: updated experiments, complete inventory, QC, B22 results, current validation figures and results report.
- `prediction_diagnostics/figures/`: old/new Kd/outcome panels, named 20-pair plots, 73-pair agreement/heatmaps, WT/shuffle effects and simulation comparison.
- `prediction_diagnostics/historical_model/`: model hashes/recovery attempts, panel/feature reproduction and 73 same-construct old predictions.
- `prediction_diagnostics/`: statistics, exact memberships/exclusions, axis limits, diagnostic overview, methods and visual QA.

Each panel has PDF, editable SVG, 600-dpi PNG, exact plotted CSV, statistics CSV where applicable and caption. Captions explain the legacy target convention. Descriptive filenames avoid unverified Figure 5 letters and occupied S7/S8 numbering.

After regeneration, inspect every PNG again. Automated checks verify data/statistics/exports, not visual quality. A visual inspection record is valid only for its recorded image hashes.
