# Global shuffle finite-sampling null analysis

## Dependence structure

The completed panel contains exactly 100 WT parents and 300 matched WT-shuffle comparisons. Every parent contributes two AHD shuffles and one RG/RGG shuffle. Every WT and shuffle system has five replica B22 values. Within each global-null iteration, one WT mean was sampled per parent and reused for all three comparisons sharing that parent; the three shuffle systems were resampled independently.

## Null-resampling procedure

For each parent, the primary common null mean was the equal-weight mean of the four system-level means (WT, two AHD shuffles, and one RG/RGG shuffle) in B22 space. Each system's five residuals around its own observed mean were retained. During each iteration, five residuals were sampled with replacement independently for the WT and each shuffle, then added to the common parent mean. The same sampled WT mean was used for all three parent comparisons. The resulting 300 log-strength differences were summarized by median |Delta|, RMS Delta, mean signed Delta, and median signed Delta. A fixed seed of 20261005 was used for 50,000 iterations.

## Observed statistics

- Median |Delta|: 0.044269 log10 units
- RMS Delta: 0.081167 log10 units
- Mean signed Delta: 0.026160 log10 units
- Median signed Delta: 0.016232 log10 units

## Primary global-null comparison

| Statistic | Observed | Null median | Null 95% range | Empirical P |
|---|---:|---:|---:|---:|
| Median |Delta| | 0.044269 | 0.022825 | [0.019525, 0.026506] | 1.99996e-05 |
| RMS Delta | 0.081167 | 0.044224 | [0.038047, 0.054466] | 0.000339993 |

Magnitude-test P values are one-sided exceedance probabilities, `(1 + count(null >= observed))/(1 + n_valid)`. Signed statistics are secondary and use the observed-direction tail.

## Invalid log transforms

There were 0 invalid null-resampled system means across all parent/system/iteration combinations. 0/50000 global iterations contained at least one invalid system mean and were excluded as incomplete; 50,000 complete iterations were retained. The invalid-iteration fraction was 0.000000%.

## Alternative-centre sensitivity

The sensitivity analysis used the pooled mean of all 20 replica B22 values per parent. Because all four systems have exactly five replicas, this centre is mathematically equal to the equal-system-mean centre; the maximum absolute parent-centre difference was 7.28e-12. With identical random draws, the maximum difference among reported null summaries and P values was 1.11e-16. Sensitivity consistent: **True**.

## Conclusion

Many individual WT-to-shuffle changes are comparable in magnitude to replica-level uncertainty, but the overall distribution of observed changes is broader than expected from finite-replica sampling alone.

This is a global distribution-level statement. It does not assign a definitive number of significant sequence-order effects to individual pairs.
