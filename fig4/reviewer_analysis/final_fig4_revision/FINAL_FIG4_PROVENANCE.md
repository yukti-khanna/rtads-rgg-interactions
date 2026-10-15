# Final Figure 4 Revision Provenance

## Terminology

Historical name: Opposite Charge Number (OCN). Revised name: negative net-charge product (−NCP). Exact formula: −NCP = −Q_RGG × Q_TAD. Numerical values are unchanged and the source column `Opposite_Charge_Number` is retained.

## Panel A

Source: `out_fig4/cache/Fig4a_B22_values.csv` from the historical Figure 4 pipeline. Scientifically unchanged: yes. n = 917.

## Panel B

Strong pair ID: `O60506_3_1__A8MYZ6_430_473` (SYNCRIP RGG - FOXO6 R-TAD). Weak pair ID: `O00144_1_1_P49910_193_208` (FZD9 RGG - ZNF165 R-TAD). RDF source: `/Users/yuktikhanna/calvados/test_rdfs`. Exact five replicas: job_0-job_4, corresponding to run_1-run_5 in the stored pH 7.0 RDF pickles. Strong B22 mean ± SD/SE = -20730.525 ± 762.116/340.829. Weak B22 mean ± SD/SE = -428.848 ± 222.111/99.331.

## Panel C

Task 5 source files: `sequence_oriented_observed_matrix.csv`, `sequence_oriented_expected_matrix.csv`, `sequence_oriented_log2_enrichment_matrix.csv`, `sequence_oriented_bootstrap.csv`, and `orientation_audit_final.csv`. Exact-sequence orientation method: rows are RGG classes and columns are R-TAD classes. Task 5 corrected 415/917 historical orientation assignments. Acidic-RGG × Acidic-R-TAD is displayed as `—` because observed = 0.0 and expected = 0.000852455, giving unregularized −Inf. No pseudocount value, including -20.87, is displayed. The heatmap uses square cells only, no hatching, no diagonal shading, and no slanted patterns.

## Panel D

Source dataset: `task3_sampling_bias/fig4_917_sampling_membership.csv`. n = 917. Pearson r = 0.874780. Spearman ρ = 0.955937. Historical marker size = 10. Revised marker size = 10.

## S4I

23 TAD IDs source and 78 RGG IDs source: `task3_sampling_bias/heldout_sequence_exclusivity.csv`; exact analyzed pairs: `task3_sampling_bias/heldout_23x78_features.csv`. n = 1794. Sequence-exclusive: True. Pearson r = 0.924277. Spearman ρ = 0.922815. Pair-bootstrap Pearson CI = 0.914-0.934. Sequence-cluster Pearson CI = 0.898-0.948. Marker size = 10. TAD IDs: P78414_218_234, O14896_214_231, Q6PJQ5_246_263, Q07869_360_378, Q08050_727_762, Q9UKS7_373_388, Q12857_129_144, Q49AA0_140_158, Q96T25_390_410, Q9NZI6_129_144, Q9UGL1_673_687, P39880_1185_1201, Q9HCK0_172_189, Q12968_519_541, P14859_345_362, Q13285_373_392, Q16236_16_33, O43313_633_647, Q6DD87_20_35, Q9Y2N7_475_490, Q8NHM5_106_124, Q9H0W7_191_210, Q9Y2K1_320_347. RGG IDs: A1L168_5_1, Q8TF74_1_1, Q9H607_1_1, P23588_4_1, P22735_4_1, Q15772_12_1, Q9UMN6_1_4, Q52LJ0_4_1, P0CAT3_1_1, Q96L73_4_1, Q9NZM5_1_1, Q9UIF9_2_1, A5PL33_1_1, Q02388_5_11, Q96HE8_1_2, Q6NT89_4_1, Q14151_7_1, Q9Y566_2_1, Q96QC0_9_2, Q96MU7_5_1, Q14687_1_1, Q69YN4_12_4, Q9H074_1_1, Q01094_2_1, Q8NFZ0_1_1, Q6UXB2_1_1, Q9C073_1_1, Q8WTX9_2_1, Q8TB33_1_1, Q3KR16_1_1, Q8IUG5_1_1, O60506_3_1, Q7RTV3_1_1, Q5H9L2_1_1, Q9UPT8_1_2, Q9NRL2_5_1, Q96RR1_1_1, P31629_1_1, Q02388_5_13, Q6ZSJ8_1_1, Q12852_3_1, Q96HM7_6_1, Q86XE3_8_1, Q6PRD1_5_1, Q9ULN7_6_1, Q96T55_2_1, A2AJT9_1_1, Q8TE04_3_1, O14559_3_2, A0A1W2PRP0_5_1, Q9Y566_1_1, Q9BRJ9_2_1, Q9H321_1_1, Q9NZB2_4_1, Q86UK7_7_1, Q5T6F2_1_1, Q96LP6_5_1, Q8WY21_1_1, Q9P0X4_5_1, A7E2V4_11_1, O76021_2_1, Q96EV2_1_2, Q9UBN4_9_1, Q86XJ0_1_1, Q86US8_1_1, P0C7P1_5_1, Q9H903_1_1, Q53LP3_2_1, P36915_2_1, P49750_1_4, A6NKL6_5_1, O75526_4_4, Q8N7U7_1_1, Q9P013_1_1, O00144_1_1, P20930_1_1, P86479_1_1, Q9NYF8_1_1.

## Resolution

All PNG files produced by this script are exported with `dpi=600`. PDFs remain vector wherever possible; only scatter artists are rasterized internally to keep PDF size manageable.
