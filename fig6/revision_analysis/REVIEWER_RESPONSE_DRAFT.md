# NMR validation reviewer response draft

This draft distinguishes completed analyses from remaining experimental work. Raw labels are ignored; curated outcomes and numeric Kd values are authoritative.

## R1-18 and R1-03: LEF1, seed coverage and scope

LEF1 binds synthetic [RRGG]7 with Kd 44.292 ± 4.422 µM, the weakest of the five reference WT constructs. Its WT-to-shuffle Kd changes to 64.372 ± 6.089 µM in the synthetic-repeat dataset. The supplied quantitative CSP matrix likewise shows the smallest RRGG response for LEF1 among the four initial seeds. These different observables are not interchangeable. Together they demonstrate heterogeneity among seed constructs rather than a uniform class of strong binders.

The original four natural-partner LEF1 outcomes remain NB, whereas the expanded panel includes LEF1–RAB44 LB and a qualitative-binding THOC4-N shuffle. This supports partner and sequence-context dependence. It does not establish that the weak repeat-peptide affinity causes natural-partner NB. Differences in construct boundaries and assay conditions must remain explicit. p53 and FOXM1 are now represented, with pending measurements shown as pending.

## R2-31: discrimination and rank concordance

All supplied fitted Kd values are retained, including p53–G3BP1 and FOXO4–G3BP1. Primary discrimination compares Kd fits with NB; secondary comparisons pool Kd fits with LB or assess LB alone. LB is not assigned an ordinal affinity rank. AUROC is the positive-negative concordance with half credit for score ties. Confidence intervals resample AHD parents and RGG parents separately; shared constructs and targeted selection limit generalization. Quantitative affinity comparisons include Spearman and pairwise concordance with grouped-bootstrap uncertainty. Exploratory cutoffs are evaluated using held-family-out threshold selection, not presented as universal binding boundaries. NB has no invented numeric Kd or censoring bound.

## R2-29: composition-preserving shuffles

The paired panel uses all experimentally tracked shuffles passing exact composition checks, matched to WT against the same RGG partner. Prediction and simulation changes are plotted with curated outcomes. A computational change is not an experimentally established affinity change. Paired experimental deltas require matching measurements and condition records.

## R2-28: prior corroboration

The PPI audit reports archived protein-level associations and assay labels separately from fragment-level binding. LEF1–ALYREF/THOC4, p53–ALYREF/THOC4, p53–G3BP1 and FOXM1–RBMX are present in the inspected edge list. THOC4 N and C share a protein accession but remain separate constructs. Missing edges are described as not found in this dataset, never proof of novelty or noninteraction. Selection rationale cannot be retrospectively inferred from experimental outcomes.

## R2-25/27 and R1-19: constructs and mechanism

Frozen current predictions and B22 values are reused. The trajectory audit binds analysis to sequence-checked topology and serialized simulation force parameters where available. Energy components are model-dependent descriptions of sampled configurations, not additive B22 contributions or experimental mechanism proofs. Full-panel simulation associations are exploratory where historical provenance remains unresolved; the audited-subset discrimination table is retained separately.

## Outstanding experimental/source work

Highest-titration bound fractions require absolute protein and ligand concentrations and an appropriate binding model. Eight equivalents alone is insufficient. Spectrum-arrow verification requires the original spectrum/concentration ordering. Displayed Kd should be rounded to uncertainty-supported precision while source values retain full precision. Exact experimental boundary and identity discrepancies must be resolved against laboratory construct records. These tasks are not reported as completed. Manuscript DOCX files were not edited.

## discrimination_statistics.csv

```text
 comparison    series  n  positive  negative    AUROC  positive_negative_concordance  CI95_low  CI95_high  valid_resamples  requested_resamples    cluster     seed
   Kd_vs_NB Predicted 40        12        28 0.869048                       0.869048  0.718699   1.000000             2000                 2000 AHD_parent 20261007
   Kd_vs_NB Predicted 40        12        28 0.869048                       0.869048  0.742731   0.965517             1992                 2000 RGG_parent 20261007
   Kd_vs_NB Simulated 40        12        28 0.922619                       0.922619  0.849454   0.981826             2000                 2000 AHD_parent 20261007
   Kd_vs_NB Simulated 40        12        28 0.922619                       0.922619  0.851211   1.000000             1992                 2000 RGG_parent 20261007
Kd_LB_vs_NB Predicted 55        27        28 0.738095                       0.738095  0.593103   0.928577             2000                 2000 AHD_parent 20261007
Kd_LB_vs_NB Predicted 55        27        28 0.738095                       0.738095  0.628023   0.771558             2000                 2000 RGG_parent 20261007
Kd_LB_vs_NB Simulated 55        27        28 0.791005                       0.791005  0.709453   0.894330             2000                 2000 AHD_parent 20261007
Kd_LB_vs_NB Simulated 55        27        28 0.791005                       0.791005  0.655012   0.860904             2000                 2000 RGG_parent 20261007
   LB_vs_NB Predicted 43        15        28 0.633333                       0.633333  0.516913   0.853551             2000                 2000 AHD_parent 20261007
   LB_vs_NB Predicted 43        15        28 0.633333                       0.633333  0.499891   0.693343             2000                 2000 RGG_parent 20261007
   LB_vs_NB Simulated 43        15        28 0.685714                       0.685714  0.581439   0.836364             2000                 2000 AHD_parent 20261007
   LB_vs_NB Simulated 43        15        28 0.685714                       0.685714  0.530614   0.777507             2000                 2000 RGG_parent 20261007
```

## rank_statistics.csv

```text
    panel    series  n      metric  estimate  CI95_low  CI95_high  valid_resamples  requested_resamples    cluster     seed
      pKd Predicted 12    Spearman  0.664336  0.000000   0.858407             1999                 2000 AHD_parent 20261007
      pKd Predicted 12    Spearman  0.664336  0.044586   0.942857             2000                 2000 RGG_parent 20261007
      pKd Predicted 12 concordance  0.757576  0.473684   0.862745             1999                 2000 AHD_parent 20261007
      pKd Predicted 12 concordance  0.757576  0.538462   0.933333             2000                 2000 RGG_parent 20261007
      pKd Simulated 12    Spearman  0.349650  0.267081   0.884615             1999                 2000 AHD_parent 20261007
      pKd Simulated 12    Spearman  0.349650 -0.936306   0.984940             2000                 2000 RGG_parent 20261007
      pKd Simulated 12 concordance  0.651515  0.600000   0.888889             1999                 2000 AHD_parent 20261007
      pKd Simulated 12 concordance  0.651515  0.076923   0.972222             2000                 2000 RGG_parent 20261007
intensity Predicted 10    Spearman -0.757576 -1.000000  -0.094340             1999                 2000 AHD_parent 20261007
intensity Predicted 10    Spearman -0.757576 -1.000000  -0.130000             2000                 2000 RGG_parent 20261007
intensity Predicted 10 concordance  0.777778  0.512821   1.000000             1999                 2000 AHD_parent 20261007
intensity Predicted 10 concordance  0.777778  0.500000   1.000000             2000                 2000 RGG_parent 20261007
intensity Simulated 10    Spearman -0.709091 -0.961538  -0.094340             1999                 2000 AHD_parent 20261007
intensity Simulated 10    Spearman -0.709091 -1.000000  -0.237209             2000                 2000 RGG_parent 20261007
intensity Simulated 10 concordance  0.777778  0.554522   0.944444             1999                 2000 AHD_parent 20261007
intensity Simulated 10 concordance  0.777778  0.600000   1.000000             2000                 2000 RGG_parent 20261007
```

## threshold_summary.csv

```text
 comparison    series  threshold            evaluation  TP  TN  FP  FN  sensitivity  specificity  balanced_accuracy  n_evaluated
   Kd_vs_NB Predicted   3.700127 in-sample exploratory  12  19   9   0     1.000000     0.678571           0.839286          NaN
   Kd_vs_NB Predicted        NaN  leave-AHD_parent-out  10  19   9   2     0.833333     0.678571           0.755952         40.0
   Kd_vs_NB Predicted        NaN  leave-RGG_parent-out  10  20   8   2     0.833333     0.714286           0.773810         40.0
   Kd_vs_NB Simulated   3.668453 in-sample exploratory  12  23   5   0     1.000000     0.821429           0.910714          NaN
   Kd_vs_NB Simulated        NaN  leave-AHD_parent-out  11  23   5   1     0.916667     0.821429           0.869048         40.0
   Kd_vs_NB Simulated        NaN  leave-RGG_parent-out  11  23   5   1     0.916667     0.821429           0.869048         40.0
Kd_LB_vs_NB Predicted   3.551286 in-sample exploratory  21  19   9   6     0.777778     0.678571           0.728175          NaN
Kd_LB_vs_NB Predicted        NaN  leave-AHD_parent-out  20  19   9   7     0.740741     0.678571           0.709656         55.0
Kd_LB_vs_NB Predicted        NaN  leave-RGG_parent-out  20  18  10   7     0.740741     0.642857           0.691799         55.0
Kd_LB_vs_NB Simulated   3.634426 in-sample exploratory  20  22   6   7     0.740741     0.785714           0.763228          NaN
Kd_LB_vs_NB Simulated        NaN  leave-AHD_parent-out  19  20   8   8     0.703704     0.714286           0.708995         55.0
Kd_LB_vs_NB Simulated        NaN  leave-RGG_parent-out  19  20   8   8     0.703704     0.714286           0.708995         55.0
   LB_vs_NB Predicted   3.551286 in-sample exploratory   9  19   9   6     0.600000     0.678571           0.639286          NaN
   LB_vs_NB Predicted        NaN  leave-AHD_parent-out  10  13  15   5     0.666667     0.464286           0.565476         43.0
   LB_vs_NB Predicted        NaN  leave-RGG_parent-out   8  16  12   7     0.533333     0.571429           0.552381         43.0
   LB_vs_NB Simulated   3.467118 in-sample exploratory  11  18  10   4     0.733333     0.642857           0.688095          NaN
   LB_vs_NB Simulated        NaN  leave-AHD_parent-out   8  18  10   7     0.533333     0.642857           0.588095         43.0
   LB_vs_NB Simulated        NaN  leave-RGG_parent-out   9  18  10   6     0.600000     0.642857           0.621429         43.0
```

## Trajectory execution status

status
excluded unresolved mapping           37
ready for coordinate linkage check    20

Completed trajectory pairs: 20. See per-pair provenance and energy verification tables.
