# Predicted attraction: 64 catalogue-matched versus 166 unmatched AHD candidates

Prepared for Yukti to run. The analysis has not been executed.

Activate your usual Figure 5 Python environment, then run from `fig5`:

```bash
python3 ahd_catalogue_prediction_comparison/compare_catalogue_groups.py
```

Requires NumPy, pandas, SciPy and matplotlib. It reads existing predictions; no fitting, retraining or new inference.

## Inputs

Final candidate membership comes from Figure 2's 230-region FASTA and `predicted_region_external_support.csv`. Matched means direct eligible candidate-region correspondence to any external AD catalogue, giving 64 matched and 166 unmatched candidates. It does not use the archived 108/82 protein membership.

Predictions are collected from all three Figure 5 sources:

- `outputs/final_hybrid_model/final_predictions/full_landscape_final_hybrid_predictions.csv` (unsimulated landscape).
- `outputs/final_hybrid_model/final_predictions/development_917_frozen_model_predictions.csv` (simulated development pairs, frozen in-sample predictions).
- `outputs/final_hybrid_model/final_predictions/test_1794_final_hybrid_predictions.csv` (held-out predictions).

The code requires one frozen-model identity across all sources, rejects duplicate pairs, and reports any missing candidates. It does not mix OOF predictions with frozen-model predictions.

## Comparison

Each candidate AHD contributes its median predicted score, rather than treating hundreds of RG partners as independent biological samples. The script compares the two candidate groups across the combined sources, as requested, and also saves separate results within the development/unsimulated and held-out cohorts. Within each cohort it uses the same RG partner set for every AHD, so a candidate is not advantaged by being compared with a different subset of available partners. The two cohorts themselves have different RG partner pools, which could influence the combined comparison.

Outputs under `ahd_catalogue_prediction_comparison/results/`:

- `report.md`, `summary.json`: medians, effect sizes, overlap, coverage and provenance.
- **AUROC**, with matched candidates as positive and higher attraction scores predicting that group; 95% confidence interval from whole-TF bootstrap resampling. AUROC is not automatically flipped to exceed 0.5.
- **Two-sided Mann–Whitney test** at AHD level, labelled nominal because some AHDs share a TF.
- **TF-cluster robust rank test**, accounting for multiple AHDs per TF; prioritise this and the bootstrap interval over the nominal test.
- `within_dataset_comparisons.csv`: separate cohort comparisons, including AUROC and tests.
- `AHD_level_scores.csv`, `RG_partner_comparison.csv`, `common_RG_pool.csv`, `omitted_AHDs.csv`, `TF_cluster_bootstrap.csv`, `ROC_curve.csv`.
- `comparison.png` and `comparison.pdf`: score distributions, cumulative distributions and ROC curve.

Interpretation: AUROC 0.5 indicates no directional separation, above 0.5 means matched candidates tend to score higher, and below 0.5 means unmatched candidates tend to score higher. These are catalogue-group comparisons, not binding-classification AUROCs. Neither overlap nor a nonsignificant test demonstrates equivalence. No experimental binding threshold is applied, and predicted scores are not Kd values.

Optional sensitivity omitting development prediction rows (held-out remains included):

```bash
python3 ahd_catalogue_prediction_comparison/compare_catalogue_groups.py --omit-development --out ahd_catalogue_prediction_comparison/results_without_development
```

Optional `--rg-list /path/to/rg_ids.txt` restricts partners to IDs listed one per line. The requested IDs must leave at least one shared partner in each cohort. `--bootstrap` defaults to 5000; `--seed` defaults to 20261011.
