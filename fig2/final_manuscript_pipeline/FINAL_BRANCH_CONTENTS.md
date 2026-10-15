# Final branch contents

The final GitHub update should retain these components.

## Pipeline entry points

- `run_final_manuscript_pipeline.sh`
- `scripts/build_final_manuscript_package.py`
- `scripts/build_table_s2_validated.py`
- `scripts/cleanup_final_branch.py`

## Analysis and plotting scripts

- `scripts/fig1C_cluster_bootstrap.py`
- `scripts/remake_fig1C_boxplots.py`
- `scripts/make_fig1A_xrgg_csp_matrix.py`
- `xrgg_panel_scripts/build_xrgg_ahd_panels.py`
- `scripts/plot_figure1_classifier_panels.py`
- `scripts/figure2_revision_common.py`
- `scripts/plot_fig2_feature_and_overlap_panels.py`
- `scripts/plot_FigS3_sequence_similarity_FINAL.py`
- `scripts/merged_fragments_vs_9seeds_blast.py`
- `scripts/finalize_figS3_sequence_similarity.py`
- `scripts/external_AD_ensemble_recurrence.py`

## Required source data

- `fig1C_corrected/`
- `inputs/tables/xrgg csps.xlsx`
- `inputs/tables/xrgg_AHD_parent_master.csv`
- the FASTA and metadata files named in `KEEP_FOR_FINAL_PIPELINE.md`
- `final_manuscript_pipeline/inputs/curated_tables/`
- `final_manuscript_pipeline/inputs/reference_figures/`
- `final_manuscript_pipeline/inputs/analysis_cache/`
- `final_manuscript_pipeline/inputs/final_pipeline_inputs.json`

## Generated material

`outputs/final_manuscript_package/` is generated and may either be committed as
the frozen manuscript deliverable or excluded from source-only branches. All
other directories selected by `scripts/cleanup_final_branch.py` are superseded
development results and are not pipeline inputs.

The cleanup script is dry-run by default. It does not select raw inputs, the
final pipeline directory, tracked external-dataset BLAST mappings, scripts, or
the final package.
