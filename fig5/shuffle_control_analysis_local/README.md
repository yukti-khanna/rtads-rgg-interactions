# WT–shuffle paired control audit

Standalone, no-training analysis for the Figure 5 reviewer control question. It does not generate sequences, run simulations, retrain models, or access hidden data.

## Run from any directory

```bash
python /path/to/shuffle_control_analysis_local/analyze_shuffle_controls.py \
  --manifest /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5_final_model_validation_local/inputs/controls/manifest/simulation_manifest_400.csv \
  --b22 /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5_final_model_validation_local/outputs/controls/control_b22_from_original_rdfs.csv \
  --predictions /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5_validation_run_model_local/outputs/controls/control_final_predictions.csv \
  --rdf-dir /Users/yuktikhanna/rtads-rgg-interactions/fig5/controls_rdfs \
  --rdf-scripts-dir /Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5_final_model_validation_local/scripts \
  --out ./outputs/shuffle_control_analysis
```

`--predictions` is optional but required for the frozen-model delta and feature-delta outputs. `--rdf-dir` and `--rdf-scripts-dir` are required for the replica-noise output. The B22 table must contain a pair identifier (`name`, `pair_name`, or `pair`) and one corrected B22 column (`Corrected_B22_Hummer`, `B22`, `B22_corr1`, or `B22_corr1_native`). Inputs remain on your system; nothing is copied into the package.

Requirements: Python 3.9+, pandas, numpy. scipy is used for Wilcoxon tests; matplotlib is used for figures. Parquet input additionally requires pyarrow.

## Main outputs

- `shuffle_nonbasic_matched_effects.csv`
- `shuffle_composition_integrity_verified.csv`
- `shuffle_effect_vs_replica_noise.csv`
- `shuffle_effect_summary_nonbasic.csv`
- `TAD_shuffle_parent_reproducibility.csv`
- `shuffle_prediction_delta.csv`
- `shuffle_feature_delta_correlations.csv`
- `shuffle_effect_by_parent.csv`
- `figures/`
- `corrected_shuffle_audit_interpretation.txt`
