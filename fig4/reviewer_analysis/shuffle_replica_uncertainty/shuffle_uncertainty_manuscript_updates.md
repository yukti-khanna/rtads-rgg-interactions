# Candidate manuscript and reviewer-response updates

These are update candidates only; no manuscript or response document was edited.

## 1. Figure 3 Results paragraph

Composition-preserving shuffling broadly preserved the simulated interaction-strength landscape (Pearson r = 0.975; Spearman rho = 0.958; 300 matched comparisons from 100 WT parents). Shuffle-induced changes were generally modest (median |Delta| = 0.044; RMS Delta = 0.081 log10 units). Pair-specific independent-replica bootstrapping gave 95% CIs excluding zero for 102/300 comparisons before multiplicity adjustment; 45/300 survived Holm family-wise correction and 76/300 survived exploratory BH-FDR correction. A subset (45/300) remained distinguishable from replica-level uncertainty after Holm correction, supporting pair-specific sequence-order modulation within an otherwise preserved landscape.

## 2. Supplementary Figure S6D legend

**S6D. Pair-specific uncertainty of composition-preserving shuffle effects.** Observed Delta is defined as log10(-mean B22_shuffle) - log10(-mean B22_WT), where each mean is calculated from five independent simulation replicas. Points show observed Delta and horizontal bars show 95% percentile intervals from 50,000 independent WT/shuffle replica-bootstrap iterations. Comparisons are sorted by observed Delta and separated into AHD- and RG/RGG-shuffle controls. Colour identifies unadjusted intervals excluding zero; Holm-significant comparisons are distinguished separately. Across 300 comparisons, median |Delta| = 0.044, RMS Delta = 0.081, 102/300 unadjusted intervals excluded zero, 45/300 survived Holm correction, and 76/300 survived exploratory BH-FDR correction.

## 3. STAR Methods: Composition-preserving sequence-shuffle controls

For each WT-shuffle comparison, B22 was evaluated independently for five WT and five shuffle simulation replicas using the same RDF processing and finite-size-corrected integration employed for the primary analysis (`corrected_rdfs_b22`, last = 7.5 nm, box size = 20 nm, delta = 2, rc = 4.0 nm). Interaction strength was calculated as log10(-mean(B22_replica1...5)); replica log values were not averaged. The shuffle effect was Delta = log10(-mean B22_shuffle) - log10(-mean B22_WT). Pair-specific uncertainty was estimated by independently resampling the five WT and five shuffle replicas with replacement for 50,000 iterations and reporting percentile 95% CIs. Non-negative bootstrap means, for which the log transform is undefined, were counted explicitly. Two-sided bootstrap P values were obtained from 100,000 centred-null iterations in which WT and shuffle B22 residuals were retained after centring both groups to their common B22-space mean. Holm correction across all 300 comparisons was the primary family-wise procedure; Benjamini-Hochberg FDR at q = 0.05 was secondary. Aggregate uncertainty was estimated with a 20,000-iteration bootstrap over the 100 WT-parent blocks, retaining all shuffle comparisons associated with each resampled parent.

## 4. Reviewer response R1-05

We added a pair-specific uncertainty analysis for the complete 300-comparison WT-shuffle panel. Rather than applying the generic uncertainty estimated from the broader 917-pair convergence analysis, we recalculated replica-level B22 for the exact five WT and five shuffle replicas underlying every comparison and independently bootstrapped each matched effect. The global WT-shuffle relationship remained strong (Pearson r = 0.975; Spearman rho = 0.958), with median |Delta| = 0.044 and RMS Delta = 0.081. Unadjusted 95% CIs excluded zero for 102/300 comparisons; 45/300 survived Holm correction and 76/300 survived exploratory BH-FDR correction. A subset (45/300) remained distinguishable from replica-level uncertainty after Holm correction, supporting pair-specific sequence-order modulation within an otherwise preserved landscape.

## 5. Other shuffle-control response text

Where the response currently states that shuffling has "no effect," replace that categorical wording with: "Composition-preserving shuffling broadly preserved the global simulated interaction landscape, while pair-specific changes were evaluated relative to the replica uncertainty of the exact WT and shuffle ensembles." Report the Holm-corrected result as primary and the BH-FDR result as exploratory.

## Aggregate cluster-bootstrap context

The WT-parent cluster-bootstrap 95% CIs were [0.0152, 0.0374] for mean signed Delta, [0.0044, 0.0264] for median signed Delta, [0.0377, 0.0538] for median absolute Delta, and [0.0720, 0.0903] for RMS Delta.
