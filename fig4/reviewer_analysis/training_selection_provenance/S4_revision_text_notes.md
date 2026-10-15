# Supplementary Figure S4 revision notes

## 1. Verified final S4D stage counts

- Stage 1: Initial feature-guided + outside-filter sampling: raw batch 600, held-out-identity exclusions 102, net-new retained 498, cumulative 498.
- Stage 2: Low-lambda sampling: raw batch 200, held-out-identity exclusions 20, net-new retained 180, cumulative 678.
- Stage 3: First −NCP-directed sampling: raw batch 166, held-out-identity exclusions 0, net-new retained 165, cumulative 843.
- Stage 4: Further −NCP-directed sampling: raw batch 74, held-out-identity exclusions 0, net-new retained 74, cumulative 917.

## 2. Manuscript-safe description

### Methods-level detail

The final 917-pair simulation training set was assembled by staged, feature-guided sampling. Reserved held-out AHD and RG/RGG region identities were excluded consistently from each historical batch. The combined initial feature-guided and outside-filter selections contributed 498 retained pairs. Low-lambda sampling increased the cumulative set to 678 pairs, followed by −NCP-directed additions to 843 and then 917 pairs. The reconstructed 917-pair set exactly matches `sim_pairs_for_training_270525.txt`.

### Results-level interpretation

The staged design broadened coverage beyond the initially favoured feature region and subsequently improved representation along −NCP. It should not be described as a single random or unbiased sample.

## 3. Proposed S4D legend

(D) Staged construction of the final simulation training set after exclusion of reserved held-out AHD and RG/RGG region identities. The initial feature-guided and outside-filter selections contributed 498 retained pairs. Addition of low-lambda controls increased the set to 678 pairs, followed by −NCP-directed sampling to 843 pairs and a final −NCP-directed addition to 917 pairs. Counts represent cumulative unique pairs contributing to the authoritative final training set.

## 4. Proposed complete S4 A-H legend outline

Supplementary Figure S4. Provenance and coverage of the staged Figure 4 simulation panel. (A-C) Historical pilot relationships between simulated interaction strength and charge complementarity, mean pair NCPR, and average lambda. (D) Cumulative construction of the final 917-pair training set after consistent exclusion of reserved held-out AHD and RG/RGG region identities. (E-G) Distributions of −NCP, total pair length, and average lambda across the complete candidate landscape, the feature-guided training set, and the held-out set. (H) Reconstructed −NCP-bin coverage of the final 917-pair simulation panel, showing candidate-pair and selected-pair counts.

## 5. Provenance caveats

- The 911- and 1,039-pair files are temporary bookkeeping states and are not plotted as scientific sampling stages.
- Historical filenames retain `ocn`, but publication-facing text uses −NCP.
- Reserved region identifiers are absent from training; literal sequence-string uniqueness is not claimed because distinct identifiers can share identical amino-acid strings.
- The B22 stability metrics remain retrospective support for the historical stopping decision, not a prospective numerical stopping rule.
- Current main Figure 4D content: 917-pair training relationship.
