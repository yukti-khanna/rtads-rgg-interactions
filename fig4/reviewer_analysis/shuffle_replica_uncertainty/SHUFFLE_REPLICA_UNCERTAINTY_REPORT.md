# Shuffle replica uncertainty analysis

## Scope and validation

This analysis uses the exact validated 300-comparison WT-versus-shuffle source panel: 100 WT parents, 200 AHD-shuffle comparisons, and 100 RG/RGG-shuffle comparisons. Each of the 400 unique simulation systems was processed from its completed five-run RDF pickle. No molecular-dynamics simulation was rerun, no sequence was changed, and no model prediction was used.

Replica RDFs were converted to B22 with the existing project function `corrected_rdfs_b22(last=7.5, box_size=20, delta=2, rc_nm=4.0)`. Interaction strength is `log10(-mean(B22_replica1...5))`, not the mean of replica log strengths.

Maximum absolute disagreement with the current panel means was 3.64e-12 for WT and 1.82e-12 for shuffles. Current plotted values reproduced: **True**.

## A. Landscape preservation

Across all 300 comparisons, Pearson r = 0.9755 and Spearman rho = 0.9581. Thus, WT and composition-preserving shuffle strengths remain strongly correlated at the landscape level.

## B. Effect magnitude

The mean signed Delta was 0.0262, median signed Delta 0.0162, median absolute Delta 0.0443, and RMS Delta 0.0812 log units.

| Group | n | Mean Δ | Median Δ | SD Δ | IQR Δ | Median |Δ| | IQR |Δ| | P90 |Δ| | P95 |Δ| | RMS Δ | frac |Δ|>0.025 | frac |Δ|>0.05 | frac |Δ|>0.10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 300 | 0.0262 | 0.0162 | 0.0770 | 0.0886 | 0.0443 | 0.0680 | 0.1319 | 0.1673 | 0.0812 | 0.663 | 0.450 | 0.203 |
| AHD shuffle | 200 | 0.0214 | 0.0151 | 0.0697 | 0.0828 | 0.0388 | 0.0577 | 0.1201 | 0.1604 | 0.0728 | 0.640 | 0.410 | 0.170 |
| RG/RGG shuffle | 100 | 0.0356 | 0.0242 | 0.0893 | 0.1061 | 0.0543 | 0.0828 | 0.1532 | 0.1855 | 0.0957 | 0.710 | 0.530 | 0.270 |

## C. Resolution relative to simulation uncertainty

Independent replica bootstraps used 50,000 iterations per comparison with seed 20261005. The median pair-specific bootstrap SE of Delta was 0.0379 log units, compared with the rough 0.05-log-unit expectation from the separate 917-pair convergence analysis. This comparison is contextual only; all individual classifications here use the actual WT and shuffle replicas.

Unadjusted percentile CIs excluded zero for 102/300 comparisons; 45/300 survived Holm family-wise correction, and 76/300 survived exploratory BH-FDR correction.

- AHD shuffle: CI excludes zero 61/200; Holm-significant 22/200; BH-significant 46/200.
- RG/RGG shuffle: CI excludes zero 41/100; Holm-significant 23/100; BH-significant 30/100.

## Bootstrap null and multiple testing

For each comparison, WT and shuffle B22 replicates were centred separately to the common B22-space location `(mean_WT + mean_shuffle)/2`, preserving the residual pattern within each group. Five null WT and five null shuffle values were then independently sampled with replacement for 100,000 iterations. The two-sided P value is the fraction of valid null draws with `|Delta_null| >= |Delta_observed|`, with a plus-one correction. Holm correction at alpha=0.05 was applied across all 300 tests; BH-FDR at q=0.05 is secondary.

## Invalid log-transform draws

0 comparisons had at least one invalid bootstrap or null-bootstrap draw caused by a non-negative resampled mean B22. Comparisons with invalid frequency >= 1.0% were flagged as transformation-sensitive; their raw null P value was not forced and was treated conservatively as 1.0 only for the 300-test multiplicity adjustment.

## WT-parent cluster bootstrap

A 20,000-iteration cluster bootstrap sampled the 100 WT parent blocks with replacement and retained all three associated shuffle comparisons per selected parent. The aggregate 95% CIs were:

- Mean signed Delta: [0.0152, 0.0374]
- Median signed Delta: [0.0044, 0.0264]
- Median absolute Delta: [0.0377, 0.0538]
- RMS Delta: [0.0720, 0.0903]

## Interpretation

Composition-preserving shuffling broadly preserves the global interaction landscape, while a subset of individual pairs shows shuffle-induced changes that exceed replica-level simulation uncertainty, supporting pair-specific modulation by residue order.
