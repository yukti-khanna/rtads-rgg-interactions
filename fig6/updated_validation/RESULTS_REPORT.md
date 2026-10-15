# Corrected validation results

## 1. Prediction changes across all experimental pairs

All 20 finite old/new experimental pairs were included, regardless of Kd availability. Mean new−old = +0.099186 (descriptive paired-bootstrap 95% CI -0.004925 to +0.200247); median = +0.093941; mean absolute change = 0.209519; range = -0.315821 to +0.494460. Old/new agreement: Pearson r=0.503565, Spearman ρ=0.518797. Exact two-sided paired sign-flip test of mean change: p=0.077339. A systematic shift is not an accuracy improvement.

Independently supplied checks (n, r, ρ, mean and median change) all match within 1e-6. Ranking changes per pair are exported.

## 2. Kd agreement and outcome separation

All nine fitted Kd pairs, including FOXO4–G3BP1, are primary. pKd=6−log10(Kd_uM), with first-order uncertainty σKd/(Kd ln10).

- old: Kd Pearson r=0.520802; Spearman ρ=0.550000, exact two-sided permutation p=0.132777 (9!=362,880 permutations). Outcome omnibus H=4.5447, permutation p=0.1032. Kd-fit versus NB AUROC=0.761905.
- new: Kd Pearson r=0.707281; Spearman ρ=0.783333, exact two-sided permutation p=0.017223 (9!=362,880 permutations). Outcome omnibus H=7.0430, permutation p=0.0225. Kd-fit versus NB AUROC=0.730159.

Thus the directions are mixed: fitted Kd agreement improves, but Kd-fit/NB AUROC decreases by -0.031746 (paired 95% CI -0.349206 to 0.285714). LB is excluded only from this binary diagnostic. Outcome counts are nine Kd-fit, four LB, seven NB. Kd-fit/LB separation increases; LB/NB effect stays the same. Holm-adjusted pairwise p-values and effects, including nonsignificant results, are in outcome_statistics.csv; paired effect differences are in paired_outcome_effect_differences.csv. Separate model p-values are not a test of model superiority.

All p-values and row-bootstrap intervals are exploratory: pairs share constructs, so independent-row exchangeability is not assured. No affinity ordering is imposed on LB, NB, Kd-fit or No Kd.

## 3. Partners and largest construct differences

- THOC4 C-term: mean change +0.169778; n=5.
- THOC4 N-term: mean change +0.211797; n=5.
- G3BP1: mean change +0.231705; n=5.
- TAF15: mean change -0.216535; n=5.

All five TAF15 predictions decrease. LEF1 predictions increase for THOC4 N-term, THOC4 C-term and G3BP1. Largest absolute changes:

              pair_id Outcome     delta
 g3bp1_rgg__lef1_nter      NB  0.494460
  g3bp1_rgg__atf4_tad  Kd_fit  0.398296
thoc4_nter__lef1_nter      NB  0.387708
thoc4_cter__lef1_nter      NB  0.375293
 taf15_rgg__tfe3_cter      NB -0.315821
 taf15_rgg__foxo6_cr3      LB -0.312334

## 4. Complete exact-construct old/new comparison

Recovered historical final pipeline predictions for 73 of 73 exact new-run construct pairs. These newly inferred same-construct old predictions are kept separate from archived Table_S8 predictions. models_old/ contained a different 22-feature base pipeline and standalone model/scaler. The actual matching archived base/residual weights were in models/pretrained/mlp_plateau_calibrated_bundle_sel_cols.pkl.

The archived panel convention applies the saved clipped residual without the public script’s soft gate. Three fixed application conventions were tested; ungated application reproduced all 20 archived base/final values within 1e-6. All ten historical feature columns were reconstructed from local pure functions/residue parameters and matched to numerical precision. No weights, lambda, cap, new-run gate or preprocessing parameters were fitted or recalibrated. Dense Keras weights were read directly for inference, with only supported sequential Dense graphs accepted. Artifact hashes, application conventions, loading warnings, exact features and reproduction errors are retained.

                    set  n  pearson_r  spearman_rho  mean_delta  median_delta  mean_absolute_change  minimum_delta  maximum_delta  mean_delta_CI95_low  mean_delta_CI95_high  median_delta_CI95_low  median_delta_CI95_high  shift_pvalue                                       test  permutations        adjustment                                exclusions  agreement_concordance
complete_same_construct 73   0.864004      0.772862    0.119484      0.102294              0.210885      -0.322021       0.554796             0.067796              0.171496               0.066047                0.168686      0.000100     Monte Carlo paired sign-flip mean test         19999 none; exploratory only nonfinite old/new; no Kd requirement               0.794521
                     WT 57   0.875565      0.812419    0.109359      0.094665              0.205181      -0.315821       0.554796             0.051090              0.170387               0.015922                0.154027      0.001500     Monte Carlo paired sign-flip mean test         19999 none; exploratory only nonfinite old/new; no Kd requirement               0.815789
               shuffled 16   0.541956      0.652941    0.155554      0.192403              0.231207      -0.322021       0.397708             0.045721              0.250952               0.071594                0.308728      0.014374 exact two-sided paired sign-flip mean test         65536 none; exploratory only nonfinite old/new; no Kd requirement               0.716667

Matched WT–shuffle source tables preserve shuffle identity and WT-parent relationships; named heatmaps and a paired shuffle-effect panel are exported.

## 5. Simulation accuracy and unresolved provenance

The legacy convention was reproduced for 20 pairs using local archived RDF candidates. A 20-pair subset has exact supplied sequences in the corresponding current final-run topologies and compatible legacy 7.5 nm corr1 target evidence. Both predictors use exactly those rows, with logs taken after replica averaging:

         predictor  n  R_squared     RMSE      MAE  mean_signed_error  Pearson_r  Spearman_rho                                      convention                        eligibility
old_same_construct 20   0.321020 0.239554 0.217880          -0.074189   0.621739      0.631579 legacy_sum 7.5 nm corr1, log after replica mean see per-pair audit; identical rows
               new 20   0.664886 0.168295 0.139788           0.024997   0.823844      0.750376 legacy_sum 7.5 nm corr1, log after replica mean see per-pair audit; identical rows

The newer supplied RDFs do not globally reproduce the older simulation table. Source/version differences are investigated per pair and cutoff; archived RDF candidates reproduce the historical values, so a discrepancy in the newer RDF version does not imply a new target convention. Older run_validated/ FOXO4 inputs are shorter than current final-run inputs and are kept separate; they do not veto matching current topologies. See topology_construct_sequence_audit.csv, convention_source_replica_investigation.csv and per_pair_simulation_eligibility.csv.

Accuracy improves on the supported simulation subset, but is not evidence of accuracy across all 73 pairs or calibrated Kd prediction. The other 53 pairs lack sufficient construct/run provenance for this paired accuracy analysis. RDF-to-topology linkage is contextual (matching named run directories), not an embedded generator manifest. Exact supplied new-model training RDF provenance also remains incomplete. Physical nm³ integrals remain separately exported and are not substituted into frozen predictions.

## Outputs, numbering and visual review

Individual PDFs, editable SVGs, 600-dpi PNGs, exact plotted data, statistics and captions are in updated_validation/figures/ and prediction_diagnostics/figures/. The readable diagnostic_overview.png accompanies individual panels. Figure 5 panel letters remain unverified; descriptive filenames avoid occupied S7/S8 numbering. Original data, annotations and DOCX files are unchanged. Prior generated outputs are saved under workflow_history/before_expanded_diagnostics_20261001/.

Visual inspection status: inspection needed after regeneration. See visual_inspection.json and VISUAL_QA.md for inspected filenames and corrections.

Proposed manuscript/legend corrections: describe historical target sums without nm³ units; distinguish 7.5 nm legacy sums from L/2 physical integrals; include all nine Kd fits as primary; state shared-construct limits; report mixed Kd/outcome findings rather than universal superiority; identify the recovered archived correction convention and sequence-supported simulation subset. No manuscript document was edited.
