# Figure 4 Task 2: B22 replicate variability and convergence audit

## 1. Reviewer question

R1-13 requests uncertainty/error bars and convergence evidence for the historical five nominal 1-µs replicas. This audit separates replicate reproducibility from trajectory-length convergence.

## 2. Dataset provenance

Repository: `/Users/yuktikhanna/idr-interaction-analysis`. Authoritative Figure 4 pair table: `/Users/yuktikhanna/rtads-rgg-interactions/fig4/out_fig4/cache/Fig4d_scatter_table.csv` (917 pairs). The RDF directory contained 25 pickle files; 25 contained all `job_0`–`job_4`, none lacked a historical replica, and 25 also contained later replicas. Later replicas were excluded from the primary analysis. Only 1 of 917 authoritative Figure 4 pairs maps to an RDF in this repository; 916 are unavailable. All 25 repository RDFs were analyzed only as an explicitly secondary available subset, including 24 pairs outside the authoritative cache. Therefore population-wide 917-pair uncertainty claims cannot be made from these files. Original per-job DCDs exist for the 25 available pairs.

## 3. Existing B22 implementation

`scripts/save_rdfs.py` computes COM RDFs over 0 to half the 20-nm box with 0.1-nm bins and stores individual and pooled RDFs. `scripts/rdf_b22_functions.py` integrates to 7.5 nm. Raw B22 is `-2π sum((g-1)r²)`. Hummer correction estimates the 5.5–7.5-nm tail mean by trapezoidal integration divided by 2 nm, then uses `-2π sum((g-c)r²)`. Ganguly applies its finite-size RDF correction before the same raw summation. Current stored pooled RDFs include 10 replicas, so the primary historical-five pooled RDF was reconstructed from stored `job_0`–`job_4` RDFs.

## 4. Integration audit

The historical B22 sums do not explicitly multiply by grid spacing. Explicit trapezoidal quadrature was computed without replacing historical values. For Hummer B22 across 125 replica observations: Pearson r=0.999974, Spearman rho=0.99925, regression slope=0.100513, median quadrature/historical ratio=0.100336, sign changes=0. With an approximately uniform 0.1-nm grid, the dominant difference is the missing bin-width scale factor, with small endpoint-rule differences; rankings are quantified in the audit table.

## 5. Replicate variability

For the 25 available pairs, historical-five Hummer B22 median SD=218.643 and median SE=97.7801. Per-pair 95% bootstrap intervals use 10,000 resamples and seed 20260906. The incomplete-historical-five fraction among available RDFs is 0; exact authoritative-population RDF coverage is 0.109%.

## 6. Pooled vs replica-derived B22

Historical-five pooled-RDF B22 versus mean of five replica B22 values: Pearson r=1, Spearman rho=1. Hummer correction is linear in the RDF here, so agreement is expected up to floating-point effects. The repository's current stored pooled RDF contains all 10 present replicas and was not substituted for the historical-five primary result.

## 7. Replica-count convergence

All 31 combinations per pair (k=1..5) were evaluated. `replicate_count_convergence.csv` contains combination-level estimates, deviations, and relative deviations, enabling descriptive 5%, 10%, and 20% summaries without declaring a hard convergence threshold.

## 8. Time convergence

Fifteen weak/intermediate/strong candidates were selected algorithmically from the 25 available pairs. Original `job_0`–`job_4` DCD boundaries are explicit. The run metadata confirms 100,000,000 steps, 10,000-step output frequency, and a 0.01-ps step, so 10,000 frames correspond to 1 µs and cumulative windows are labeled 0.2–1.0 µs. Frame-resolved COM RDFs and historical Hummer B22 were calculated independently for each job. Results by regime:

- intermediate: median relative 80→100% change=0.526%; final mean within the 80% bootstrap interval for 100.0% of selected pairs; fractions below 5%/10%/20%=100.0%/100.0%/100.0%.
- strong: median relative 80→100% change=0.301%; final mean within the 80% bootstrap interval for 100.0% of selected pairs; fractions below 5%/10%/20%=100.0%/100.0%/100.0%.
- weak: median relative 80→100% change=6.106%; final mean within the 80% bootstrap interval for 100.0% of selected pairs; fractions below 5%/10%/20%=20.0%/80.0%/80.0%.

## 9. Reviewer-relevant conclusion

The available 25-pair subset supports direct historical-five replicate uncertainty and replica-count sensitivity analysis, but it cannot establish uncertainty for the full 917-pair Figure 4 population. Trajectory-length convergence was assessed only for the algorithmic 15-pair subset. Conclusions must remain subset-specific and must not claim universal convergence.

## 10. Recommended manuscript use

`per_pair_b22_uncertainty.csv` and `b22_replicate_uncertainty` can support error bars for the available subset; `replicate_count_convergence` can support a supplementary replica-count panel; the inventory, mapping, integration audit, and uncertainty CSVs are suitable supplementary tables; the documented formulas and historical-replica definition are suitable STAR Methods material. The time-convergence figures can support a supplementary panel only with explicit 15-pair subset and 25-pair repository-coverage caveats.
