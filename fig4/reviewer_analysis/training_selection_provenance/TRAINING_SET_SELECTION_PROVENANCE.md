# Training-set selection provenance

## 1. Complete candidate space

**Directly documented and independently reconstructed.** The sequence catalogue contains 230 AHDs and 1,008 RG/RGG regions, giving 231,840 pair combinations. Sequence-derived net charges reproduce the historical `Opposite_Charge_Number` column exactly as −NCP = −Q_RG/RGG × Q_AHD (maximum absolute difference 0).

## 2. Initial 5 × 5 pilot

**Directly documented by `25_pairs_features_data.csv`.** Five AHDs and five RG/RGG regions were simulated in all 25 combinations. This pilot is an early exploratory set and is not simply a subset that sums into the later 917-pair list.

## 3. Early feature-guided expansion

**Documented in archived lists and commented code.** The historical script calculates charge complementarity, mean pair NCPR and average lambda. The archived `final_test_names.txt` contains 500 unique pairs. The precise end-to-end order of every early sub-round is not recoverable from a single executable manifest.

## 4. Sampling outside the initial feature filter

**Directly documented in commented code and a 200-pair list.** `plot_features.py lines 802-841` contains filters outside the initially favoured charge-complementarity/NCPR/lambda region and `random_state=42` sampling. `200_pairs_outside_filter_less_than_lambda045_names.txt` contains 200 unique pairs.

## 5. Length/diversity considerations

**Directly documented as candidate procedures, but not fully linked to a final round manifest.** `plot_features.py lines 843-945` records randomized AHD order, total-length diversity, and unused-RG/RGG constraints. No seed is set for the Python `random.shuffle` calls in those blocks, so those exact orderings cannot be replayed from code alone.

## 6. −NCP-stratified expansion

**Directly documented and retrospectively reproduced.** `plot_features.py lines 491-623` uses 50 quantile bins, recursively subdivides bins larger than ceil(N/50), calculates proportional targets for 1,000 pairs, computes deficits, and samples with `random_state=42`. The archived operation used the 192,303-pair sequence-excluded training-eligible landscape, not the full 231,840-pair cross-product. The archived selected table contains 74 pairs and exactly matches `ocn_added_names.txt`.

## 7. Empirical B22-distribution stabilisation

**Historical decision described qualitatively; numerical audit is retrospective.** Complete B22 tables establish cumulative cohorts of 546, 911 and 1,039 simulated pairs, followed by the 917-pair post-exclusion training set. `B22_distribution_stability_by_stage.csv` reports KS statistics, Wasserstein distances and quantile changes. No prospective numerical stopping threshold was recovered, so these measurements support but do not define the historical stopping decision.

## 8. Unix `shuf` selection of held-out identities

**Historical context supplied by the project, not reproducibly documented by an archived command plus seed.** The archived lists contain exactly 23 reserved AHD identities and 78 reserved RG/RGG identities. The audit does not rerun `shuf` and does not infer a replacement draw.

## 9. Sequence-level exclusion from training

**Exactly reconstructed.** The archived pre-exclusion list has 1,039 pairs. Removing every pair containing a reserved AHD and/or reserved RG/RGG removes 122 pairs: 56 AHD-only, 59 RG/RGG-only and 7 in both categories. There are 0 unresolved removals, leaving 917 pairs.

## 10. Final 917-pair training set

**Directly documented.** `sim_pairs_for_training_270525.txt` and `fig4_917_sampling_membership.csv` contain the same 917 unique pairs. The set is staged and feature-guided, not a single random or unbiased sample.

## 11. Complete 1,794-pair held-out cross-product

**Directly documented and verified at the identifier level.** The 23 reserved AHD identifiers and 78 reserved RG/RGG identifiers generate exactly 1,794 cross-product pairs. AHD identifier overlap, RG/RGG identifier overlap and exact pair overlap with the 917 training set are all zero. However, literal FASTA comparison finds 1 held-out AHD identifier and 2 held-out RG/RGG identifiers whose amino-acid strings are also present under different training identifiers. Independence is outcome-independent and identifier-exclusive; it does not depend on chronological timing.

## 12. Provenance gaps and limitations

- No single executable manifest reconstructs every early selection sub-round.
- The archived files do not contain a reproducible Unix `shuf` command plus seed for the 23/78 identity draw.
- Some commented `random.shuffle` selection blocks have no documented seed.
- The 50-bin −NCP procedure is reproducible from archived inputs, but its candidate universe is the 192,303-pair sequence-excluded training-eligible landscape.
- Current Supplementary S4 mapping is A-D early feature relationships, E −NCP distribution, F length distribution, G average-lambda distribution, H reconstructed −NCP-bin coverage, and I held-out −NCP/B22 validation. The proposed A-H figure replaces H with distribution stability and omits I.
- Current main Figure 4D is: **training-set relationship (917 pairs)**.
- The archived FASTA contains identical sequence strings under distinct identifiers. The exact matches are listed in `heldout_sequence_exclusion_audit.csv`; therefore “zero exact amino-acid-sequence overlap” is not supported literally, although identifier and exact-pair overlaps are zero.

## Manuscript-safe description

The complete sequence-derived candidate landscape comprised 230 AHDs and 1,008 RG/RGG regions (231,840 possible pairs). The 917-pair CALVADOS training panel was assembled through staged, feature-guided sampling informed by early relationships with charge complementarity, mean pair NCPR, average lambda and the negative net-charge product (−NCP = −Q_RG/RGG × Q_AHD). Archived code documents later −NCP-stratified deficit filling toward an approximate 1,000-pair target. The archived 1,039-pair pre-exclusion set was reduced to 917 by removing 122 pairs containing one or both of 23 reserved AHD and 78 reserved RG/RGG identifiers. The held-out simulations comprise the complete 23 × 78 cross-product (1,794 pairs), with zero AHD-identifier, RG/RGG-identifier or exact-pair overlap with the training set. The decision to stop adding training simulations was based on empirical stabilisation of the B22 distribution; KS and Wasserstein measurements reported here are retrospective diagnostics rather than prospective stopping thresholds.
