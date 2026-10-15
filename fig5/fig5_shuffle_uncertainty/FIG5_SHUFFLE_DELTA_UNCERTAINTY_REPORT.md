# Predictor-shuffle uncertainty: new Supplementary Fig. S7E/F


**Recommendation: retain with qualification for RG/RGG only; AHD perturbations are not reliably resolved.**

## Reconstruction and unchanged sources

The exact accepted predictor-control source contains 345 unique shuffle comparisons: 230 AHD and 115 RG/RGG. There are 115 unique WT parents; each contributes two AHD shuffles and one RG/RGG shuffle. Each comparison changes exactly one partner, retaining its exact WT partner. All 460 unique simulations have five finite replicas (2,300 unique replica values).
Replica B22 values were recovered directly from the existing RDFs, using the existing Figure-5 `corrected_rdfs_b22(last=7.5, box_size=20, delta=2, rc_nm=4.0)` implementation and its `b22_corr1` values. The replica B22 values were averaged first and only then transformed as log10(-mean B22). No simulation, model inference, feature generation, model fitting, or calibration was performed.
The Fig. 4 directory was accessed read-only for raw RDFs. No values or uncertainty estimates from the separate 300-comparison simulation-robustness analysis were reused. Former S6 numbering is now S7; these proposed additions are E and F. Existing panels and manuscript text were not edited.
Maximum absolute mean-B22 reproduction difference: 3.63798e-12. Maximum transformed-delta reproduction difference: 1.4086e-15.

## Reproduced fixed-target analysis

| Shuffle class | n | Pearson r | Spearman rho | RMSE | MAE |
|---|---:|---:|---:|---:|---:|
| AHD | 230 | 0.019972 | -0.020896 | 0.081560 | 0.061907 |
| RG/RGG | 115 | 0.615932 | 0.622213 | 0.090634 | 0.068914 |
| all_exploratory | 345 | 0.357005 | 0.233116 | 0.084693 | 0.064243 |

The class-specific Pearson, Spearman, RMSE and MAE reproduce the existing combined original+new statistics within 1e-12. Combined-class statistics are exploratory only.

## Replica-bootstrap procedure and scope

20,000 bootstrap iterations; master seed 20261005; NumPy SeedSequence child streams 0 (uncertainty), 1 (null target bootstrap), and 2 (null permutations). Within each draw, five WT replicas are resampled once per unique WT parent; that parent mean is shared across all three comparisons. Five replicas are independently resampled for each shuffled simulation. Predictions and the comparison set remain fixed. The bootstrap SD (ddof=1 across iterations) is reported as the Monte Carlo estimate of SE(delta_sim).
All input replica B22 values are negative; every resampled mean is therefore negative, and no bootstrap observations, pairs or iterations require exclusion for the logarithm. Percentile intervals describe propagation of finite-replica uncertainty conditional on these fixed sequences; they do not include resampling sequences, model-development uncertainty, or generalization to other sequences.

## Uncertainty-aware performance distributions

| Class | Metric | Point estimate | Bootstrap median | 95% interval | Fraction > 0 |
|---|---|---:|---:|---|---:|
| AHD | Pearson | 0.019972 | 0.017518 | [-0.044452, 0.080813] | 0.70630 |
| AHD | Spearman | -0.020896 | -0.014566 | [-0.088671, 0.059434] | 0.35150 |
| AHD | RMSE | 0.081560 | 0.092370 | [0.085567, 0.101280] | not applicable |
| AHD | MAE | 0.061907 | 0.070641 | [0.065461, 0.076135] | not applicable |
| RG/RGG | Pearson | 0.615932 | 0.553346 | [0.496311, 0.607185] | 1.00000 |
| RG/RGG | Spearman | 0.622213 | 0.571301 | [0.508009, 0.632850] | 1.00000 |
| RG/RGG | RMSE | 0.090634 | 0.100794 | [0.093853, 0.109239] | not applicable |
| RG/RGG | MAE | 0.068914 | 0.078216 | [0.071976, 0.084893] | not applicable |

Fractions above zero are descriptive bootstrap proportions, not p-values.
These requested percentile propagation intervals need not contain the original point estimate. Resampling perturbs the noisy empirical target vector; the nonlinear correlations can be attenuated and the error metrics increased. For example, the RG/RGG original Pearson r is 0.616, whereas its bootstrap median is 0.553. These are not bias-corrected confidence intervals for the unobserved, noise-free correlation, nor a measurement-error correction. The evidence here is persistence of positive association under the specified replica uncertainty procedure.

## Secondary permutation-null test

20,000 iterations independently permuted delta_pred within each shuffle class and paired it with a fresh shared-WT bootstrap target vector. The observed reference is the original fixed-target correlation. Empirical two-sided p = (1 + count(abs(null) >= abs(observed))) / (1 + 20000).
This requested comparison-level permutation assumes within-class exchangeability of predictor perturbations. Shared-WT target dependence is preserved, but comparison-level predictor permutations do not preserve predictor parent blocks. Accordingly these are secondary, nominal p-values, not proof of a causal or sequence-generalizable effect or a guarantee of conservativeness under every dependence structure. The four tests are reported without multiplicity adjustment.

| Class | Metric | Observed | Two-sided permutation p | Null 95% interval |
|---|---|---:|---:|---|
| AHD | Pearson | 0.019972 | 0.766462 | [-0.128354, 0.129015] |
| AHD | Spearman | -0.020896 | 0.752812 | [-0.127977, 0.129391] |
| RG/RGG | Pearson | 0.615932 | 4.99975e-05 | [-0.184750, 0.180224] |
| RG/RGG | Spearman | 0.622213 | 4.99975e-05 | [-0.183075, 0.179619] |

## Signal magnitude versus replica uncertainty

| Class | Median abs(delta_sim) | Median bootstrap SE | RMS delta_sim | Fraction abs(delta)>SE | Fraction abs(delta)>2SE |
|---|---:|---:|---:|---:|---:|
| AHD | 0.039464 | 0.034949 | 0.074494 | 0.55652 | 0.30870 |
| RG/RGG | 0.054320 | 0.036951 | 0.095638 | 0.59130 | 0.40870 |

These thresholds are descriptive; they are not counts of statistically significant pairs.

## Class-specific interpretation

- **AHD:** Do not claim reliable recovery of pair-specific shuffle perturbations; retain any delta plot only as a diagnostic of this limitation. Pearson interval [-0.044, 0.081]; Spearman interval [-0.089, 0.059].
- **RG/RGG:** Retain with qualification: both uncertainty-aware association intervals remain positive and both secondary nominal permutation tests support non-random association. Pearson interval [0.496, 0.607]; Spearman interval [0.508, 0.633].

Operational reading of the requested framework: case B. “Supported” here requires both percentile lower bounds > 0 and both secondary nominal p < 0.05; this is a transparent report rule, not feature or model selection. Absolute shuffled-interaction-strength prediction is a separate analysis and remains unchanged.

### Manuscript candidate text (alternatives; no manuscript edited)

**A — if both classes remain supported:** With finite-replica uncertainty propagated through matched WT–shuffle comparisons, predicted perturbations retained positive associations with simulated perturbations in both shuffle classes. The result supports qualified recovery of shuffle effects, while individual small differences remain subject to sampling uncertainty.
**B — if only RG/RGG remains supported:** Predicted shuffle-induced changes retained a positive association with simulation-derived changes for RG/RGG shuffles after propagating replica uncertainty. AHD shuffle perturbations were not resolved reliably, despite accurate absolute prediction of shuffled interaction strengths.
**C — if neither remains sufficiently resolved:** The predictor accurately recovers absolute shuffled interaction strengths, but these data do not establish reliable recovery of the smaller WT-to-shuffle perturbations once finite-replica uncertainty is considered. We therefore limit the validation claim to absolute interaction strengths.

## Files and reproducibility

Run `python scripts/run_predictor_shuffle_uncertainty.py` from the fig5 clone. It refuses to overwrite a completed report. Use `--audit-only` to reconstruct and verify input data without bootstrapping.

- Exact predictor source: `/Users/yuktikhanna/rtads-rgg-interactions/fig5/s8_extended_controls/S8B_absolute_shuffle_prediction/S8B_shuffle_source_audited.csv`
- Prior delta metrics: `/Users/yuktikhanna/rtads-rgg-interactions/fig5/s8_extended_controls/S8B_absolute_shuffle_prediction/S8B_delta_metrics_original_new_combined.csv`
- Existing RDF calculation: `/Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5_final_model_validation_local/scripts/rdf_b22_functions.py`
- Read-only RDF folder: `/Users/yuktikhanna/rtads-rgg-interactions/fig4/controls_rdfs`
- Per-file hashes and software details: `analysis_provenance.json` and `rdf_input_inventory.csv`.
- New individual audit panels: `FigS7E_AHD_shuffle_delta_uncertainty` and `FigS7F_RGG_shuffle_delta_uncertainty` (PDF/SVG/600-dpi PNG, 2.4 x 2.3 inches, vector points).
- The combined audit is `FigS7EF_shuffle_delta_uncertainty_audit`. The requested `FigS8B_shuffle_delta_uncertainty_audit` and source filenames are compatibility aliases in this new directory only; no existing S8B is overwritten.
- Full fixed-target metrics, uncertainty summaries, null results, parent structure, all 20,000 metric draws and compressed per-pair bootstrap delta draws are retained.
