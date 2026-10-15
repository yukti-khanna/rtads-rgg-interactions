# Master column definitions

The local run writes the exhaustive per-column `updated_validation/column_dictionary.csv`, including source columns, transformations, model-input flags and matching rules. The following primary definitions are fixed in `validation_sources.json`.

| Column | Meaning and source |
| --- | --- |
| pair_id | Canonical oriented RG/RGG__AHD identity; exact join or documented verified alias |
| pair_display_name | AHD–partner label derived from exact role IDs; THOC4 N/C and shuffles retained |
| Kd_uM | Selected experimental Kd converted from explicitly specified units to uM; unresolved duplicate measurements give NA |
| Kd_error_uM | Reported error in uM, with its original interpretation retained in uncertainty_type |
| Pred_log10_negB22_final | predictions_minimal.csv::final_hybrid_prediction, unchanged; already logarithmic |
| Sim_log10_negB22_final | log10(-Sim_B22_final) only for negative finite B22; checked against the existing log column |
| Outcome | Supplied experimental category; Kd_fit normalized to Kd fit; missing outcome never implies NB |
| Notes | Notes from the replacement experimental input, not stale computational annotations |
| Sim_B22_final | per_pair_b22.csv::corrected_B22_mean selected for legacy_sum, 7.5 nm, primary_tail_offset; no physical units assigned |
| Sim_B22_SD / Sim_B22_SEM | Distinct reported replica dispersion/mean uncertainty, never interchanged |
| Sim_n_replicas | Positive reported replica count; unknown mean-only count represented as NA |
| Sim_n_replicas_reported | Original source count, including its zero sentinel for unknown count |
| New_base_prediction / New_applied_correction | Supplied base prediction and actual applied correction; no new calibration |
| Old_same_construct_final / Old_same_construct_base | Previously recovered frozen historical inference; exact selected sequence matching flagged separately |
| Archived_panel_prediction | Archived Table_S8 hybrid prediction; kept separate from same-construct inference |
| Historical_experimental_prediction | Archived panel prediction where available, otherwise existing same-construct old prediction only after exact sequence matching; provenance recorded per row |
| AHD_name / RGG_name | Exact roles from registry or explicitly supplied predictions; ID parsing fills display metadata only |
| AHD_sequence / RGG_sequence | Exact selected sequences, not approximate protein-name matches |
| AHD_length / RGG_length / *_sequence_sha256 | Sequence lengths and SHA256, checked against source metadata when present |
| is_shuffle / *_parent / parent_pair_id | Identity-derived WT/shuffle and exact parent relationship; no sequence interpolation |
| pKd / pKd_error | 6-log10(Kd_uM); first-order reported-error propagation, retaining its uncertainty interpretation |
| experimental_status | No measurement, fitted Kd, explicit NB, LB without fitted Kd, or unresolved duplicate/conflict |
| prediction_status / simulation_status / feature_status | Availability independent of suitability for a specific analysis |
| simulation_accuracy_eligible | Existing audit passes and selected quantity/exact sequence binding passes |
| simulation_eligibility_reason | Separate reason for eligible/ineligible comparison; ineligible values remain present |
| missing_data_reason | Missing inputs, selection conflicts and unavailable logarithm reasons; no zero imputation |

All current feature_matrix.csv columns are retained with their supplied names. The manifest identifies the model's 13 inputs: minus_NCP, mean_lambda_RGG, mean_lambda_AHD, SHD_RGG, SHD_AHD, local_charge_complementarity_std, scd_RGG_patch_std, scd_AHD_patch_std, rg_fraction_RGG_patch_std, aromatic_fraction_AHD_patch_std, shd_RGG_patch_std, shd_AHD_patch_std and ah_energy_patch_std. Future keyed feature tables retain every additional column, whether or not it is a model input.

`Historical_feature_*` contains the separately defined historical descriptors. `Archived_panel_feature_*` and `Archived_earlier_feature_*` retain all descriptors and original matching/length metadata from the mapped archived feature tables; these may describe different construct boundaries. `Prediction_detail_*` and `Prepared_*` retain the additional supplied calculated fields and QC metadata. None overrides the primary selected quantities or the replacement experimental annotations.
