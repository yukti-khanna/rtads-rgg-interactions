# Final rAHD–RG/RGG interaction predictor

The existing Snakemake workflow and sequence examples are retained. Inputs are FASTA sequences plus pair identifiers; no descriptors or simulation targets are required. Pair order is **RG/RGG first, rAHD second**. `pairing.keep_order: true` is required. Existing sample inputs already use this order.

```bash
snakemake --cores 4
snakemake --cores 4 predict_pretrained
```

The first target prepares the final 13 features. The second loads the final five-member base ensemble and gated/bounded residual correction, using stored calibration only. Feature preparation does not require TensorFlow. Use the separate feature and inference requirements in `prediction_pipeline/` with your established environments.

## Deployment prerequisite

Install the **authoritative server** files in `models/final_frozen/`: MODEL_MANIFEST.json, base_model_ensemble.joblib, residual_preprocessor.joblib and residual_nn.keras. Exact accepted hashes are in `prediction_pipeline/config/accepted_model_hashes.json`. No mismatching local artifacts were substituted. Prediction performs feature and prediction identity checks before accepting new data.

Internal `AHD`/`TAD` field names and feature aliases remain unchanged for compatibility; public scientific terminology is **rAHD**.

## Provenance and manuscript handoff

`outputs/manuscript_export/` contains accepted predictor-only results, also copied into `fig5/inputs/predictor/`, with matching checksums. Controls stay in fig5. Old workflow/configuration are preserved in `archive/pre_local_context_update/`; old model files and sequence inputs are untouched.

`development/` preserves available local-context/residual development scripts and outputs. **Full from-scratch reconstruction is not yet certified:** the F2 script references an absent `reviewer_reduced_model_ablation_training_only/cv_fold_assignments.csv`; original global-selection provenance must be recovered before executing that historical workflow. Do not invent replacement folds.

The inference package records a global-residue-table convention discrepancy in older control exports. Those exports are preserved in fig5, not silently regenerated or mixed with new predictions.
