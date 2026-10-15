# Training-set selection provenance: corrected-stage addendum

## Corrected publication-facing sampling stages

### A. Raw historical selected-pair counts

- `final_test_names.txt` contains 500 unique initial feature-guided pairs.
- `all_test_names.txt` contains those 500 plus exactly 100 outside-filter pairs, giving 600 raw Stage-1 pairs.
- The low-lambda batch contains 200 unique pairs.
- The first historical OCN-named batch contains 166 unique pairs; publication text refers to this coordinate as −NCP.
- The further OCN-named batch contains 74 unique pairs.

The temporary 911-pair bookkeeping file is exactly the 965-pair raw union through the first −NCP batch minus the 54 pairs in `names_taken_out.txt`. The 1,039-pair bookkeeping file exactly restores those 54 pairs and adds the 74-pair further −NCP batch. These file-management states are not plotted as scientific sampling stages.

### B. Post-held-out-exclusion counts contributing to the final training set

Reserved AHD and RG/RGG region identities were excluded consistently from every cumulative stage:

- Stage 1: Initial feature-guided + outside-filter sampling: raw batch 600, held-out-identity exclusions 102, net-new retained 498, cumulative 498.
- Stage 2: Low-lambda sampling: raw batch 200, held-out-identity exclusions 20, net-new retained 180, cumulative 678.
- Stage 3: First −NCP-directed sampling: raw batch 166, held-out-identity exclusions 0, net-new retained 165, cumulative 843.
- Stage 4: Further −NCP-directed sampling: raw batch 74, held-out-identity exclusions 0, net-new retained 74, cumulative 917.

The reconstructed Stage-4 set contains 917 unique pairs and is exactly identical to the authoritative `sim_pairs_for_training_270525.txt` list (missing = 0, extra = 0). No final training pair contains a reserved AHD or RG/RGG identifier. Exact pair overlap with the held-out cross-product is zero.

The earlier audit found identical amino-acid strings under some different region identifiers. Therefore this addendum claims identifier-level exclusion, not literal absence of every duplicated sequence string.
