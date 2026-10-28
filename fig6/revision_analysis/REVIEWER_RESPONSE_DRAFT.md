# NMR validation reviewer response draft

This draft distinguishes completed analyses from remaining experimental work. Raw labels are ignored; curated outcomes and numeric Kd values are authoritative.

## R1-18 and R1-03: LEF1, seed coverage and scope

LEF1 binds synthetic [RRGG]7 with Kd 44.292 ± 4.422 µM, the weakest of the five reference WT constructs. Its WT-to-shuffle Kd changes to 64.372 ± 6.089 µM in the synthetic-repeat dataset. The supplied quantitative CSP matrix likewise shows the smallest RRGG response for LEF1 among the four initial seeds. These different observables are not interchangeable. Together they demonstrate heterogeneity among seed constructs rather than a uniform class of strong binders.

The original four natural-partner LEF1 outcomes remain NB, whereas the expanded panel includes LEF1–RAB44 LB and the THOC4-N shuffle is currently annotated ND. This supports partner and sequence-context dependence. It does not establish that the weak repeat-peptide affinity causes natural-partner NB. Differences in construct boundaries and assay conditions must remain explicit. p53 and FOXM1 are now represented, with pending measurements shown as pending.

## R2-31: discrimination and rank concordance

Primary analyses use WT pairs only; all supplied WT fitted Kd values are retained, with eligibility determined from the current curated master. p53–G3BP1 is classified as ND and is excluded from affinity fitting comparisons. Primary discrimination compares Kd fits with NB; secondary comparisons pool Kd fits with LB or assess LB alone. LB is not assigned an ordinal affinity rank. AUROC is the positive-negative concordance with half credit for score ties. Confidence intervals resample AHD parents and RGG parents separately; shared constructs and targeted selection limit generalization. Quantitative affinity comparisons include Spearman and pairwise concordance with grouped-bootstrap uncertainty. Exploratory cutoffs are evaluated using held-family-out threshold selection, not presented as universal binding boundaries. NB has no invented numeric Kd or censoring bound.

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
   Kd_vs_NB Predicted 43        13        30 0.882051                       0.882051  0.764290   0.992593             2000                 2000 AHD_parent 20261007
   Kd_vs_NB Predicted 43        13        30 0.882051                       0.882051  0.694389   0.965332             1992                 2000 RGG_parent 20261007
   Kd_vs_NB Simulated 43        13        30 0.933333                       0.933333  0.879364   0.990385             2000                 2000 AHD_parent 20261007
   Kd_vs_NB Simulated 43        13        30 0.933333                       0.933333  0.840544   1.000000             1992                 2000 RGG_parent 20261007
Kd_LB_vs_NB Predicted 52        22        30 0.722727                       0.722727  0.542096   0.898248             2000                 2000 AHD_parent 20261007
Kd_LB_vs_NB Predicted 52        22        30 0.722727                       0.722727  0.550517   0.799735             2000                 2000 RGG_parent 20261007
Kd_LB_vs_NB Simulated 52        22        30 0.783333                       0.783333  0.675867   0.886367             2000                 2000 AHD_parent 20261007
Kd_LB_vs_NB Simulated 52        22        30 0.783333                       0.783333  0.610386   0.889062             2000                 2000 RGG_parent 20261007
   LB_vs_NB Predicted 39         9        30 0.492593                       0.492593  0.275788   0.735907             1999                 2000 AHD_parent 20261007
   LB_vs_NB Predicted 39         9        30 0.492593                       0.492593  0.365377   0.575787             2000                 2000 RGG_parent 20261007
   LB_vs_NB Simulated 39         9        30 0.566667                       0.566667  0.358695   0.745751             1999                 2000 AHD_parent 20261007
   LB_vs_NB Simulated 39         9        30 0.566667                       0.566667  0.403843   0.750000             2000                 2000 RGG_parent 20261007
```

## rank_statistics.csv

```text
    panel    series  n      metric  estimate  CI95_low  CI95_high  valid_resamples  requested_resamples    cluster     seed
      pKd Predicted 13    Spearman  0.692308  0.161889   0.928571             2000                 2000 AHD_parent 20261007
      pKd Predicted 13    Spearman  0.692308 -0.100000   0.842105             2000                 2000 RGG_parent 20261007
      pKd Predicted 13 concordance  0.769231  0.555556   0.905882             2000                 2000 AHD_parent 20261007
      pKd Predicted 13 concordance  0.769231  0.500000   0.863636             2000                 2000 RGG_parent 20261007
      pKd Simulated 13    Spearman  0.500000  0.316667   0.950074             2000                 2000 AHD_parent 20261007
      pKd Simulated 13    Spearman  0.500000 -0.730769   0.936675             2000                 2000 RGG_parent 20261007
      pKd Simulated 13 concordance  0.692308  0.608108   0.920611             2000                 2000 AHD_parent 20261007
      pKd Simulated 13 concordance  0.692308  0.217391   0.911458             2000                 2000 RGG_parent 20261007
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
   Kd_vs_NB Predicted   3.715929 in-sample exploratory  13  20  10   0     1.000000     0.666667           0.833333          NaN
   Kd_vs_NB Predicted        NaN  leave-AHD_parent-out  11  20  10   2     0.846154     0.666667           0.756410         43.0
   Kd_vs_NB Predicted        NaN  leave-RGG_parent-out  11  21   9   2     0.846154     0.700000           0.773077         43.0
   Kd_vs_NB Simulated   3.668453 in-sample exploratory  13  24   6   0     1.000000     0.800000           0.900000          NaN
   Kd_vs_NB Simulated        NaN  leave-AHD_parent-out  12  24   6   1     0.923077     0.800000           0.861538         43.0
   Kd_vs_NB Simulated        NaN  leave-RGG_parent-out  11  24   6   2     0.846154     0.800000           0.823077         43.0
Kd_LB_vs_NB Predicted   3.551286 in-sample exploratory  17  19  11   5     0.772727     0.633333           0.703030          NaN
Kd_LB_vs_NB Predicted        NaN  leave-AHD_parent-out  13  19  11   9     0.590909     0.633333           0.612121         52.0
Kd_LB_vs_NB Predicted        NaN  leave-RGG_parent-out  13  20  10   9     0.590909     0.666667           0.628788         52.0
Kd_LB_vs_NB Simulated   3.467118 in-sample exploratory  19  19  11   3     0.863636     0.633333           0.748485          NaN
Kd_LB_vs_NB Simulated        NaN  leave-AHD_parent-out  15  20  10   7     0.681818     0.666667           0.674242         52.0
Kd_LB_vs_NB Simulated        NaN  leave-RGG_parent-out  13  21   9   9     0.590909     0.700000           0.645455         52.0
   LB_vs_NB Predicted   3.079288 in-sample exploratory   8   8  22   1     0.888889     0.266667           0.577778          NaN
   LB_vs_NB Predicted        NaN  leave-AHD_parent-out   6   8  22   3     0.666667     0.266667           0.466667         39.0
   LB_vs_NB Predicted        NaN  leave-RGG_parent-out   6  11  19   3     0.666667     0.366667           0.516667         39.0
   LB_vs_NB Simulated   3.467118 in-sample exploratory   6  19  11   3     0.666667     0.633333           0.650000          NaN
   LB_vs_NB Simulated        NaN  leave-AHD_parent-out   6  17  13   3     0.666667     0.566667           0.616667         39.0
   LB_vs_NB Simulated        NaN  leave-RGG_parent-out   6  14  16   3     0.666667     0.466667           0.566667         39.0
```

## Trajectory execution status

status
excluded unresolved mapping           37
ready for coordinate linkage check    20

Completed trajectory pairs: 20. See per-pair provenance and energy verification tables.
