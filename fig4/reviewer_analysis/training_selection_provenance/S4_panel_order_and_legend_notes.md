# Supplementary Figure S4 panel order and legend notes

## Verified cumulative cohorts

- Stage 1: 498 retained pairs.
- Stage 2: 678 cumulative pairs.
- Stage 3: 843 cumulative pairs.
- Stage 4: 917 cumulative pairs.
- The Stage-4 set is exactly equal to `sim_pairs_for_training_270525.txt`.

## S4D retrospective stability metrics

Metrics are calculated on stored `log10(-B22)` values for the retained training cohorts only. Held-out B22 values were not used.

- Initial + outside-filter (n = 498): KS = 0.032466; Wasserstein = 0.024737.
- + low-λ (n = 678): KS = 0.024134; Wasserstein = 0.013410.
- + −NCP sampling (n = 843): KS = 0.030757; Wasserstein = 0.015960.
- Final training set (n = 917): KS = 0.000000; Wasserstein = 0.000000.

KS and Wasserstein distances are retrospective quantitative support for the historical practical observation of distribution stabilisation. They were not prospectively defined stopping thresholds. The Stage-3 distances are slightly larger than the Stage-2 distances, so the trajectory should not be described as strictly monotonic.

## Panel renumbering

- Old H became new E: reconstructed −NCP-bin coverage.
- Old E became new F: −NCP distribution comparison.
- Old F became new G: total pair-length distribution comparison.
- Old G became new H: average-λ distribution comparison.
- No S4I is included.

## Proposed complete Supplementary Figure S4 legend

Supplementary Figure S4. Sampling provenance and coverage for the Figure 4 simulations. (A-C) Historical pilot relationships between simulated interaction strength and charge complementarity, mean pair NCPR, and average λ, which informed the early feature-guided sampling strategy. (D) Empirical cumulative distributions of `log10(-B22)` for successive held-out-identity-excluded training cohorts containing 498, 678, 843, and 917 pairs. The inset reports retrospective KS statistics and Wasserstein distances to the final 917-pair distribution; these metrics support, but did not prospectively define, the historical stopping decision. (E) Reconstructed −NCP-bin coverage of the final training panel, showing complete-candidate and selected-pair counts. (F-H) Distributions of −NCP, total pair length, and average λ across the complete candidate landscape, final feature-guided training set, and held-out set.

## Results-ready sentence

Successive feature-guided simulation cohorts showed similar empirical interaction-strength distributions, while reconstructed −NCP-bin coverage and feature-distribution comparisons document how the final 917-pair training panel and held-out set span the broader candidate landscape (Supplementary Fig. S4D-H).
