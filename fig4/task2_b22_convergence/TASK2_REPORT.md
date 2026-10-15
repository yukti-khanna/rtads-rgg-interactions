# Task 2: B22 replicate variability and convergence (R1-13)

## 1. Reviewer concern
Reviewer R1-13 requests error bars and convergence evidence for five 1-µs replicas of charged IDR pairs. Between-replica reproducibility and trajectory-length convergence are assessed separately.

## 2. Dataset provenance
The authoritative Figure 4 population contains 917 unique pairs. RDF source: `/Users/yuktikhanna/calvados/test_rdfs`. 917 mapped uniquely; 0 unmatched; 0 duplicate mappings. 917 have all five historical replicas. Legacy `run_1`–`run_5` correspond sequentially to `job_0`–`job_4` in `calvados/scripts/save_rdfs.py`. Additional replicas were found for 0 authoritative pairs and excluded.

## 3. B22 calculation
B22 was recalculated by importing the existing `scripts/rdf_b22_functions.py::corrected_rdfs_b22`, following `compute_b22.py`, with `last=7.5`, `box_size=20`, and `delta=2`. No integration, Hummer, or Ganguly definition was modified. Figure 4 uses the mean of per-run Hummer-corrected B22 (`b22_corr1`), not a stored pooled RDF.

## 4. Replicate variability
Across 917 complete pairs, median Hummer SD=175.709, median SE=78.5795, and the SD 5th–95th percentile is 83.4261–413.082. Bootstrap CIs use 10,000 resamples with seed 20260907. Incomplete/unmapped: 0/917 (0.00%). Regime-specific uncertainty can be read from the selected-pair and full uncertainty tables without relying on CV near zero.

## 5. Pooled versus replicate-derived B22
No stored pooled RDF exists in these legacy pickles. A mean RDF was reconstructed for comparison only. Its B22 versus the historical mean of five replica B22 values gives Pearson r=1, Spearman rho=1, slope=1, intercept=0. The displayed Figure 4 quantity remains the mean of per-replica Hummer B22 values.

## 6. Replica-number convergence
All combinations were evaluated. Moving from four to five replicas reduces the median absolute deviation from 30.2684 to zero; four-replica estimates are within 5%/10%/20% in 71.1%/92.8%/99.5% of combinations. Thresholds are descriptive.

## 7. Representative pair selection
Five pairs per regime were selected nearest the 10th, 50th, and 90th percentiles of the true complete-pair B22 landscape, subject to complete RDFs and one unambiguous trajectory root. See `convergence_pair_selection.csv`.

## 8. Trajectory-length convergence
Original `job_0`–`job_4` trajectories were analyzed separately at 20–100%. The cumulative fractions are calculated directly. The 0.2–1.0 µs labels follow the documented historical 1-µs design; local configurations verify step and frame counts but do not contain the integrator timestep needed to infer physical duration independently.

- intermediate: median 80→100% change 1.09%; median 60→100% change 1.56%; below 5%/10%/20%: 100.0%/100.0%/100.0%.
- strong: median 80→100% change 3.26%; median 60→100% change 2.16%; below 5%/10%/20%: 80.0%/100.0%/100.0%.
- weak: median 80→100% change 1.45%; median 60→100% change 4.97%; below 5%/10%/20%: 80.0%/100.0%/100.0%.

These 15 pairs do not establish universal time convergence for all 917 pairs.

## 9. Reviewer-relevant conclusion
The tables quantify between-replica reproducibility over the complete mapped population and trajectory-length convergence only for the representative subset. The results should be described conservatively and by interaction regime.

## 10. Recommended manuscript usage
Use `per_pair_b22_uncertainty.csv` for Figure 4 error bars and a supplementary uncertainty table; use replica-count and time-convergence figures as supplementary convergence evidence; and use the provenance and exact existing-function parameters for STAR Methods. No manuscript files were edited.

## Figure 4 log-space error bars

For negative mean B22 and negative bootstrap bounds, the plotted value is `log10(-mean_B22)`. Because this transformation reverses the ordering of negative B22 bounds, the lower plotted bound uses the upper B22 bound and vice versa. Asymmetric lower and upper error extents were generated for 917/917 pairs. 0 confidence intervals cross or include zero and were left untransformed.

## Original Figure 4 reproduction

Task 2 five-replica mean Hummer B22 values were transformed and matched by exact pair ID to the original Figure 4D cache. Matched n=917; unmatched n=0; maximum absolute difference=9.2428935616339913e-09; mean absolute difference=1.2765654618445122e-09; Pearson r=0.99999999999999989; Spearman rho=1; differences above 1e-8=0; differences above 1e-6=0. This is a reproducibility comparison only.

## Trajectory-duration provenance

All 75 selected pair/replica configurations were checked. They consistently specify 100,000,000 steps and an output frequency of 10,000 steps, yielding 10,000 saved frames, but they do not specify the integrator timestep. The local DCD metadata provides frame indices rather than physical time. Therefore the duration status is: **Historical simulation duration supplied externally: 1 µs per replica.** The cumulative fractions remain directly verified; their µs labels derive from the documented historical design rather than an independent local inference.

