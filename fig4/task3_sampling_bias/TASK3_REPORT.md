# Figure 4 Task 3 — sampling-bias audit

## Executive result

The exact full candidate landscape contains **231,840 pairs** (230 TADs × 1,008 RGGs). The exact Figure 4 panel contains **917 unique pairs**. The held-out dataset is the complete **23 × 78 = 1,794-pair** cross-product and all 1,794 pairs have actual simulation-derived Hummer-corrected B22 values.

The held-out sequences are fully exclusive from the 917 panel: **0 overlapping TADs and 0 overlapping RGGs**. However, the available files do not contain executable code plus a seed that reproduces the original choice of the 23 and 78 sequences. They establish saved-list provenance and exclusivity, not a reproducible claim of uniform random pre-fit selection.

OCN versus log10(-B22) is Pearson r = **0.875** (95% pair-bootstrap CI 0.809 to 0.941) in the feature-guided 917 panel and r = **0.924** (95% pair-bootstrap CI 0.914 to 0.934) in the sequence-exclusive held-out simulations. See the clustered-bootstrap table for inference respecting the crossed 23-by-78 sequence design.

## Provenance interpretation

- Dataset A is a complete sequence-derived feature cross-product; it has no B22.
- Dataset B is the exact Figure 4 cache mapped without duplicates to its feature/B22 table. Historical scripts document feature-guided stages and seed 42 in some sampling blocks, but no single end-to-end manifest was found.
- Dataset C has a saved 101-ID list, executable cross-product construction, simulation preparation/analysis scripts, PDB evidence, and a complete simulation-derived B22 table. The original selection draw of the 23 and 78 IDs is not reproducible from an available seed/code record.

## Methods

Feature definitions follow the existing repository tables. OCN is `-(charge_RGG × charge_TAD)`; normalized OCN is OCN divided by total length; charge complementarity, Average NCPR, Average lambda, length, sigma, and average charge retain repository definitions. Distribution comparisons report effect sizes rather than relying on large-n p-values. Deciles are defined on Dataset A. Correlation CIs use 10,000 deterministic pair bootstraps (seed 20260910). The held-out sequence-aware CI uses 10,000 two-way pigeonhole bootstraps, independently resampling 23 TAD and 78 RGG identities (seed 20260919). Matching uses 5-bin coarsened exact matching on OCN, total length, and Average lambda without B22.

## Caveat

Sequence exclusivity is confirmed. Independence from feature-guided *pair* selection is supported structurally because none of the held-out sequences occurs in the 917 panel. The stronger statement that held-out sequences were selected randomly before model fitting or before B22 inspection is **not documented reproducibly** by the available files and should not be asserted without an external historical record.
