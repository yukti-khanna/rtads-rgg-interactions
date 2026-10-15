# Figure 5 reviewer response and Figure 6 working panel plan

Source reviewed: /Users/yuktikhanna/writing/rbp-tfs/cell_reports/ptp_response.docx.
Figure 5 refers to the reviewers' NMR-validation figure; implementation stays in fig6.
The document's tentative action plans are proposals, not completed results or mandatory analysis choices.

## Experimental annotation policy

Use fig6_parent_master.csv as the current curated experimental source. Use Outcome and numeric Kd_uM/Kd_uM_err for analyses. Ignore Outcome_raw and free-text notes for classification, conflicts, exclusions, and sensitivity analyses. A positive supplied Kd is retained as a fitted measurement, including p53–G3BP1. Do not reintroduce the raw-label-based p53 exclusion suggested in the preceding discussion.

The existing workbook importer creates initial normalized labels from workbook text. Do not rerun it to overwrite subsequent curated parent-master edits. Plot regeneration uses run_validation.py --paper-only and reads the parent master directly.

Missing Kd is not NB. Pending and ambiguous curated outcomes are not assigned numeric affinities or binary labels. Quantitative intensity analysis uses available measurements independently of whether a Kd is available.

## Reviewer issues and proposed evidence

| Comment | Required response | Evidence available and remaining work |
| --- | --- | --- |
| R1-18: LEF1 synthetic-repeat versus natural-partner behaviour | Show context dependence explicitly rather than imply universal binding to natural RGG partners. | Original four natural-partner LEF1 rows are NB. Expanded data include LEF1–RAB44 LB and qualitative binding for the THOC4-N shuffle. Distinguish WT, shuffle and partner. Compare assay conditions before attributing differences to sequence context; link to the separate synthetic-repeat experiment. |
| R1-18: p53 and FOXM1 absent | Show the expanded seed-partner panel, including incomplete measurements. | Both now have natural-partner rows. p53–G3BP1 has a supplied Kd. Other pending measurements must remain pending; do not describe either complete titration panel as finished. |
| R1-18, R1-03, R1-19: narrow range and overstatement | Report exact plotted ranges and all fitted measurements; describe rank association and categorical discrimination. | Current main Kd analysis includes 12 fits and retains FOXO4–G3BP1. More low-score NB rows expand categorical coverage, not necessarily the fitted-affinity range. Add confidence intervals and describe selected-panel scope. |
| R1-18: saturation at final titration point | Report top-point bound fraction and assumptions. | Requires absolute labelled-protein and ligand concentrations for each assay. Eight equivalents alone is insufficient. Use a stated binding model and the measured Kd; do not infer concentrations. |
| R2-25/27: experimental constructs and selection range | Document actual construct sequences, boundaries, lengths and prediction provenance. | Registry and pair IDs are available. Verify experimental boundaries against them, including FOXO4 naming and p53/FOXM1 workbook/computational discrepancies. Do not infer identity from agreement of scores. Do not claim construct length caused range compression without a matched calculation. |
| R2-26: significant figures and arrows | Round displayed Kd values to uncertainty-supported precision and verify titration direction. | Preserve full precision in source CSV. Inspect original spectra and concentration order before changing arrows; scatter plots cannot resolve this issue. |
| R2-28: why these pairs were chosen | Provide a pair-selection table separating prior interaction evidence, seed status, construct availability and selection stage. | Recover contemporaneous selection records and references. Do not reconstruct prospective selection reasons from observed outcomes. Prior network association does not itself prove direct binding. |
| R2-29, R1-05: composition-preserving shuffles | Compare WT and shuffle against the same partner, with prediction, simulation and experimental observations. | Ten experimentally tracked shuffle rows are available. Verify composition from exact sequences. Show paired computational changes and curated WT/shuffle outcomes; paired Kd or intensity effects require both experimental measurements under comparable conditions. |
| R2-31: statistical support and concordance | Quantify discrimination and rank agreement with uncertainty, retaining all eligible measurements. | Existing plots give Spearman and unordered outcome tests. Add binary AUROC/concordance, confidence intervals, and quantitative Kd pairwise concordance. Avoid imposing a primary NB < LB < Kd-fit order. |

## Proposed panel allocation

Main validation figure:

1. Representative titrations with corrected arrows and displayed precision.
2. Outcome matrix across natural partners, separating WT and shuffles and showing pending entries.
3. Predicted score by curated NB, LB and Kd-fit outcome, with counts and unordered group tests.
4. Predicted score versus pKd for all eligible fitted measurements, with reported experimental error, Spearman correlation and confidence interval.
5. Predicted and simulated ROC curves for fitted Kd versus NB, with AUROC and uncertainty. This binary AUROC equals concordance over positive-negative pairs when ties receive half credit.

Supplementary panels:

1. Corresponding simulated-score outcome and pKd analyses.
2. Kd+LB versus NB discrimination and LB-alone sensitivity analysis, alongside the primary fitted-Kd versus NB comparison.
3. Paired WT/shuffle prediction and simulation plots with experimental outcomes alongside each pair.
4. Existing combined mean-below-threshold intensity correlation, with paired series and exploratory inference.
5. Threshold sensitivity and validation holding out AHD families or RGG partners separately. Keep WT and corresponding shuffles within the same AHD-family group.

## Current descriptive discrimination results

Calculated from the current curated parent master, with all numeric Kd fits retained:

| Comparison | Positive / NB counts | Predicted AUROC | Simulated AUROC |
| --- | --- | --- | --- |
| Fitted Kd versus NB | 12 / 20 | 0.8542 | 0.9167 |
| Fitted Kd plus LB versus NB | 23 / 20 | 0.7370 | 0.8022 |
| LB versus NB | 11 / 20 | 0.6091 | 0.6773 |

These point estimates do not establish that simulation significantly outperforms prediction. Shared constructs, targeted pair selection and unresolved simulation provenance for some expanded pairs limit generalization. Full-panel simulation associations are exploratory; keep the existing audited subset as a separate sensitivity comparison.

For Kd+LB versus NB, maximizing in-sample balanced accuracy gives midpoint cutoffs of approximately 3.54 predicted and 3.63 simulated on the current log10(-B22) scale. These are exploratory, scale-specific values, not established binding thresholds. Confidence intervals and prospective validation remain outstanding. Future measurements can test a frozen rule without retuning it on those same outcomes.

NB must not be assigned an infinite Kd or an invented censoring bound. Quantitative censoring requires assay-specific detection limits. LB is not given a fixed affinity order. The two qualitative-binding rows can be assessed in a declared secondary analysis; they are absent from the three comparisons above.

## Completion status

This file records a response and panel plan, not a completed rebuttal. Existing main and intensity panels have been generated. ROC exports with confidence intervals, experimental shuffle panels, selection provenance, top-point saturation estimates and spectrum-arrow verification remain to be completed. No changes were made to the response DOCX.
