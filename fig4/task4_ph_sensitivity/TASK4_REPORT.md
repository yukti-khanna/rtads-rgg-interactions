# Figure 4 Task 4: pH sensitivity (R1-15)

## 1. Reviewer concern
Figure 4 simulations used pH 7.0 whereas NMR used pH 6.5. Histidine protonation can therefore alter simulated interactions over this interval.

## 2. Data provenance
pH 7 RDFs: `/Users/yuktikhanna/calvados/test_rdfs`. pH 6.5 RDFs: `/Users/yuktikhanna/rtads-rgg-interactions/fig4/histidines_rdfs` (the brief's IDR-repository path was absent; the files were found in the corrected Figure 4 workspace). Requested pH 6.5 trajectories: `/Users/yuktikhanna/rtads-rgg-interactions/fig4/histidines_dcds`; this directory is absent locally. B22 was recalculated with `/Users/yuktikhanna/idr-interaction-analysis/scripts/rdf_b22_functions.py`, `last=7.5`, `box_size=20`, and `delta=2` without changing any equation or correction.

## 3. Rerun completeness
The 917 panel contains 390 His-containing pairs. There are 390 pH 6.5 RDF pairs, all mapping to the 917 panel and pH 7 reference RDFs. 390 have five historical replicas at both pH values. One His-containing pair was not rerun: .

## 4. B22 comparison
Across 390 matched pairs, pH7 versus pH6.5 mean B22 gives Pearson r=0.9958 and Spearman rho=0.9647. Median absolute change is 85.98; median relative change is -0.26%. Sign changes: 0 (0.00%). Five-replica SD/SE values are reported per pair; paired bootstrap uncertainty uses 10,000 resamples.

## 5. Rank stability
Spearman rho=0.9647, Kendall tau=0.8478. Median absolute rank shift is 14.0 pairs (3.60 percentile points); 90th/95th percentiles are 50.2/65.1. Largest shifts are listed first in `ph_sensitivity_rank_changes.csv` after sorting by `absolute_rank_difference`.

## 6. Histidine dependence
Count correlations and location-group median/IQR summaries are in the two histidine-effect tables. They describe association only and do not establish causality.

## 7. Charge-state changes
Charges use CALVADOS's standard Henderson-Hasselbalch histidine assignment (pKa 6.0), fixed K/R and D/E charges, and both termini whose +1/-1 contributions cancel in each sequence net charge. Per-sequence and OCN-proxy changes are tabulated. This calculation describes the parameter change; B22 effects come from the existing simulations.

## 8. Effect on Figure 4 conclusion
Original 917 OCN-logB22 Pearson r=0.8748; pH-adjusted r=0.8735. The main electrostatic conclusion **does not materially change** under this descriptive comparison.

## 9. Reviewer-relevant conclusion
Within the tested pH 6.5-7.0 interval, use the effect magnitudes and rank statistics above. Do not generalize to pH independence outside this interval. The one absent rerun and absent combined DCD archive should be disclosed.

## 10. Recommended manuscript use
Use the provenance and B22 parameters for Methods, the paired and rank tables plus comparison plots for a supplementary pH-sensitivity panel, and the conservative conclusion above in the reviewer response. Update Figure 4 only if the journal requires the pH-adjusted values; no manuscript files were edited.
