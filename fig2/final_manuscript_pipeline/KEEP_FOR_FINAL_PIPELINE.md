# Files to keep for the final manuscript pipeline

Do not delete anything until `run_final_manuscript_pipeline.sh` completes and
the generated `manifest.json` has been checked. The runner currently depends on
the following existing material.

## Keep in fig2

- `final_manuscript_pipeline/`
- `run_final_manuscript_pipeline.sh`
- `final_manuscript_pipeline/inputs/final_pipeline_inputs.json`
- `final_manuscript_pipeline/inputs/analysis_cache/`
- `fig1C_corrected/raw_feature_values.csv`
- `fig1C_corrected/all_feature_statistics.csv`
- `inputs/fasta/seed_tads_15aa.fasta`
- `inputs/fasta/random_lenmatched_from_allIDRs_seed_tads_15aa.fasta`
- `inputs/fasta/100_all_at2_frags_dis.fasta`
- `inputs/fasta/tf_modified_headers.fasta`
- `inputs/metadata/UNIPROTIDS_GENENAME.txt`
- `inputs/tables/xrgg csps.xlsx`
- `inputs/tables/xrgg_AHD_parent_master.csv`
- `scripts/build_final_manuscript_package.py`
- `scripts/build_table_s2_validated.py`
- `scripts/fig1C_cluster_bootstrap.py`
- `scripts/merged_fragments_vs_9seeds_blast.py`
- `scripts/finalize_figS3_sequence_similarity.py`
- `scripts/external_AD_ensemble_recurrence.py`
- `scripts/make_fig1A_xrgg_csp_matrix.py`
- `scripts/remake_fig1C_boxplots.py`
- `scripts/figure2_revision_common.py`
- `scripts/plot_figure1_classifier_panels.py`
- `scripts/plot_FigS3_sequence_similarity_FINAL.py`
- `scripts/plot_fig2_feature_and_overlap_panels.py`
- `xrgg_panel_scripts/build_xrgg_ahd_panels.py`
- every input path listed in `final_manuscript_pipeline/inputs/final_pipeline_inputs.json`

The complete candidate window FASTA, recovered vote table, and biological
candidate-region table are optional large inputs required only when explicitly
rerunning external-AD recurrence. They are passed on the command line and do
not need to be committed with the standard plotting/package branch.

## Not part of this pipeline

Manuscript Figure 2, Supplementary Figure S3, and the Figure 1D workflow
schematic are deliberately excluded. The runner does not call `../fig3`.

## Safe cleanup point

After a successful run, retain `outputs/final_manuscript_package/` together
with this pipeline and the listed inputs. Archive older exploratory/reviewer
folders rather than deleting them until manuscript acceptance.
