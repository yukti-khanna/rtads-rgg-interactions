# Supplementary Figure S6F manuscript notes

No manuscript document was edited.

## A. Results-ready sentence

Composition-preserving shuffles largely preserved the simulated interaction-strength landscape (Pearson r = 0.975; Spearman rho = 0.958). WT-to-shuffle differences varied between pairs (median |Delta| = 0.044 log10 units). Replica-level analysis showed that the observed global distribution of shifts was broader than expected from finite-replica sampling alone (Supplementary Fig. S6F).

## B. Proposed S6F legend

**Supplementary Figure S6F. WT-to-shuffle effects relative to replica-level uncertainty.** Observed Delta values, defined as log10(-mean B22_shuffle) - log10(-mean B22_WT), are shown for 200 AHD-shuffle and 100 RG/RGG-shuffle comparisons, sorted within each group. Points denote observed effects and horizontal intervals denote pair-specific 95% replica-bootstrap intervals obtained by independently resampling five WT and five shuffle replicas. The inset compares the observed median |Delta| and RMS Delta with parent-aware global finite-sampling null distributions. Grey intervals show the central 95% of the null distribution, grey points show null medians, and black points show observed statistics. The global null reused a single resampled WT mean for all three shuffle comparisons sharing each WT parent. Observed median |Delta| was 0.044 versus a null median of 0.023 (95% range 0.020-0.027; empirical P = 2e-05); observed RMS Delta was 0.081 versus a null median of 0.044 (95% range 0.038-0.054; empirical P = 0.00034).

## C. Methods paragraph

For each matched comparison, interaction strength was calculated as log10(-mean B22) from five WT or five shuffle replicas, and pair-specific 95% intervals were obtained by independently resampling the five WT and five shuffle B22 values with replacement. To test whether the complete distribution of observed WT-to-shuffle effects could be explained by finite-replica sampling, we constructed a parent-aware global null across 100 WT parent blocks. Within each parent, the WT, two AHD-shuffle systems and one RG/RGG-shuffle system were centred to the equal-weight mean of their four system-level B22 means while preserving each system's five residuals. In each of 50,000 iterations, the WT residuals were resampled once per parent and the resulting WT mean was reused for all three matched comparisons; shuffle systems were resampled independently. Median |Delta| and RMS Delta were the primary global statistics. Null resamples with any non-negative system mean B22 were tracked and excluded as incomplete before log transformation.
