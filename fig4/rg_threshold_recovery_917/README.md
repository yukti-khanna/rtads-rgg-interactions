# Figure 4 recovery of the final 917 RGG simulation inputs

## Scope

This analysis asks whether the RG/RGG tracts used by the final Figure 4 CALVADOS pair set remain represented in the saved Figure 3 threshold-sensitivity calls. It did not rerun the RG caller, CALVADOS, or B22; it did not change the 917-pair set or optimize thresholds.

## Inputs used

- Authoritative final pair identities: `/Users/yuktikhanna/rtads-rgg-interactions/fig4/out_fig4/cache/Fig4a_B22_values.csv`
- Precursor pair metadata: `/Users/yuktikhanna/rtads-rgg-interactions/fig4/sim_pairs_added_features_data_270525.csv`
- Saved Figure 3 all-condition region catalogue: `/Users/yuktikhanna/rtads-rgg-interactions/fig3/figure3_rg_threshold_sensitivity/outputs/rg_regions_all_conditions.csv.gz`

The second copy at `fig3/outputs/rg_regions_all_conditions.csv.gz` has a different compressed-file hash but the same uncompressed CSV SHA-256; the current sensitivity-project copy above was used.

## Identity and baseline validation

1. The B22 cache contains exactly **917 unique final pairs**.
2. All **917/917** pair names map one-to-one to the 1,039-row precursor table.
3. The final pairs contain exactly **627 unique downstream RG/RGG regions** from **548 proteins** and **559 parent IDRs**.
4. Baseline mapping succeeded for **627/627** regions. Every mapping used exact Figure 4 RGG identifier = Figure 3 baseline `region_id`, with identical coordinates and sequence.
5. Under baseline `(7,15,5)`, all downstream regions and all 917 pairs are `EXACT`; the baseline recovery fraction is 1.0.

The precursor RGG lengths agree for all 627 regions. Charge and NCPR differ from the Figure 3 table for **170 regions**, and the discrepancy is completely explained by charge convention: Figure 3 counted histidine as +1, whereas the Figure 4 precursor values equal `Figure 3 net_charge - histidine count` and the corresponding length-normalized NCPR. This is recorded rather than forced to agree and does not affect identifier, coordinate, sequence, or overlap mapping.

Coordinates are 0-based, half-open, parent-IDR-local. Matching is restricted to the same parent IDR.

## Operational classification

- `EXACT`: best alternative coordinates equal baseline coordinates.
- `OVERLAP_SAME_TRACT`: non-zero same-IDR overlap with changed coordinates, without the merge/split rules below.
- `MERGED`: the best alternative contains the full downstream baseline interval and also contains every literal `RG` motif of at least one other baseline region on that IDR.
- `SPLIT`: at least two alternative regions each contain distinct baseline `RG` motifs, their union covers all baseline motifs, and no single component contains all baseline motifs.
- `LOST`: no alternative region on the same parent IDR overlaps the baseline region.

The best alternative maximizes residue overlap, then exact identifier agreement, then minimizes length difference; the final tie-break is deterministic catalogue position/identifier. Any-overlap recovery is calculated independently of the classification. `fraction_of_baseline_residues_recovered` uses the best alternative, as do Jaccard and boundary shifts. The additional `fraction_of_baseline_residues_recovered_by_union` field records combined coverage by all overlapping alternatives for split cases.

## One-factor-at-a-time recovery

### Compact linker

| value | condition | unique exact % | unique recovered % | unique merged % | unique split % | unique lost % | pair exact % | pair recovered % | pair potentially affected % | pair lost % |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | c05_b15_f05 | 87.08 | 89.31 | 0.16 | 0.00 | 10.69 | 87.35 | 89.53 | 12.65 | 10.47 |
| 6 | c06_b15_f05 | 91.71 | 92.98 | 0.00 | 0.00 | 7.02 | 91.49 | 92.80 | 8.51 | 7.20 |
| 7 | c07_b15_f05 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 | 0.00 |
| 8 | c08_b15_f05 | 99.68 | 100.00 | 0.00 | 0.00 | 0.00 | 99.67 | 100.00 | 0.33 | 0.00 |
| 9 | c09_b15_f05 | 99.36 | 100.00 | 0.00 | 0.00 | 0.00 | 99.13 | 100.00 | 0.87 | 0.00 |

### Bridge linker

| value | condition | unique exact % | unique recovered % | unique merged % | unique split % | unique lost % | pair exact % | pair recovered % | pair potentially affected % | pair lost % |
|---|---|---|---|---|---|---|---|---|---|---|
| 13 | c07_b13_f05 | 94.26 | 100.00 | 0.00 | 0.96 | 0.00 | 93.57 | 100.00 | 6.43 | 0.00 |
| 14 | c07_b14_f05 | 97.77 | 100.00 | 0.00 | 0.96 | 0.00 | 97.60 | 100.00 | 2.40 | 0.00 |
| 15 | c07_b15_f05 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 | 0.00 |
| 16 | c07_b16_f05 | 98.41 | 100.00 | 0.80 | 0.00 | 0.00 | 98.26 | 100.00 | 1.74 | 0.00 |
| 17 | c07_b17_f05 | 96.33 | 100.00 | 1.59 | 0.00 | 0.00 | 96.51 | 100.00 | 3.49 | 0.00 |

### Flank extension

| value | condition | unique exact % | unique recovered % | unique merged % | unique split % | unique lost % | pair exact % | pair recovered % | pair potentially affected % | pair lost % |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | c07_b15_f03 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 |
| 4 | c07_b15_f04 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 |
| 5 | c07_b15_f05 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 | 0.00 |
| 6 | c07_b15_f06 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 |
| 7 | c07_b15_f07 | 0.00 | 100.00 | 0.00 | 0.00 | 0.00 | 0.00 | 100.00 | 100.00 | 0.00 |

Flank changes are interpreted as boundary/context changes unless overlap is lost. In the flank-only slice the maximum unique-region loss was **0 regions**, and the minimum condition median core-RG-motif recovery fraction was **1.000**. Exact-coordinate loss under flank changes is therefore not treated by itself as loss of the underlying tract.

## Full 125-condition screen

- Unique-region recovery min / median / max: **0.8931 / 1.0000 / 1.0000**.
- Condition(s) at minimum unique-region recovery: **c05_b13_f03, c05_b13_f04, c05_b13_f05, c05_b13_f06, c05_b13_f07, c05_b14_f03, c05_b14_f04, c05_b14_f05, c05_b14_f06, c05_b14_f07, c05_b15_f03, c05_b15_f04, c05_b15_f05, c05_b15_f06, c05_b15_f07, c05_b16_f03, c05_b16_f04, c05_b16_f05, c05_b16_f06, c05_b16_f07, c05_b17_f03, c05_b17_f04, c05_b17_f05, c05_b17_f06, c05_b17_f07**.
- Final-pair recovery min / median / max: **0.8953 / 1.0000 / 1.0000**.
- Condition(s) at minimum pair recovery: **c05_b13_f03, c05_b13_f04, c05_b13_f05, c05_b13_f06, c05_b13_f07, c05_b14_f03, c05_b14_f04, c05_b14_f05, c05_b14_f06, c05_b14_f07, c05_b15_f03, c05_b15_f04, c05_b15_f05, c05_b15_f06, c05_b15_f07, c05_b16_f03, c05_b16_f04, c05_b16_f05, c05_b16_f06, c05_b16_f07, c05_b17_f03, c05_b17_f04, c05_b17_f05, c05_b17_f06, c05_b17_f07**.
- Conditions with 100% unique-region recovery: **75/125**.
- Conditions with any lost downstream region: **50/125**.
- Maximum unique downstream regions lost in one condition: **67**.
- Maximum of the 917 pairs affected by lost RGG regions in one condition: **96**.

Across all conditions, exact matches and altered-boundary matches are reported separately in `downstream_region_recovery_summary.csv` and `pair_level_recovery_summary.csv`. Region-level overlap, boundary shifts, merge/split evidence, and literal RG-motif retention are in the compressed record-level tables.

Across all **78,375 region-condition records**, **14,592 (18.62%)** were exact, **61,008 (77.84%)** were recovered with altered boundaries/merge/split classification, and **2,775 (3.54%)** were lost. Across all **114,625 pair-condition records**, the corresponding counts were **21,299 (18.58%)** exact, **89,276 (77.89%)** altered but recovered, and **4,050 (3.53%)** lost. These aggregate frequencies are dominated by expected coordinate changes whenever flank extension differs from 5 aa, so tract recovery is the primary robustness metric.

## Repeatedly lost downstream regions

`A0A0J9YX57_1_1` (50 conditions), `A2A288_4_1` (50 conditions), `A6NMD0_1_1` (50 conditions), `B8ZZ34_2_1` (50 conditions), `O15541_3_1` (50 conditions), `O94818_3_1` (50 conditions), `O96013_3_1` (50 conditions), `P07766_3_1` (50 conditions), `P14678_3_1` (50 conditions), `P25940_6_2` (50 conditions)

The complete loss frequency and affected-pair count for every downstream region are in `repeatedly_lost_downstream_regions.csv`.

## Diagnostic interpretation for manual decision support

A non-trivial subset of downstream inputs is lost in at least one combined threshold condition, although many altered calls still represent the same tract.

This result only diagnoses preservation of the identities/core tracts of the existing Figure 4 inputs. It does not state that simulations should be rerun and does not execute any simulation or B22 calculation.

## Outputs

- `final_917_pairs_mapped.csv`: authoritative pair set with precursor and baseline-region metadata.
- `downstream_rgg_regions.csv`: one row per unique downstream RGG region.
- `baseline_mapping_validation.csv`: exact constrained baseline mappings and methods.
- `downstream_region_recovery_all_conditions.csv.gz`: one downstream-region record per condition.
- `downstream_region_recovery_summary.csv`: unique-region condition summaries.
- `pair_level_recovery_all_conditions.csv.gz`: one final-pair record per condition.
- `pair_level_recovery_summary.csv`: pair-level condition summaries.
- `one_factor_*_recovery.csv`: requested one-factor slices.
- `full_grid_recovery_summary.csv`: combined 125-condition screen.
- `repeatedly_lost_downstream_regions.csv`: per-region loss frequency.
- `figures/`: PNG and PDF diagnostics.
- `run_metadata.json` and `validation_report.json`: provenance and independent checks.

## Run locally

From this directory:

```bash
scripts/run_local.sh
```

No SLURM resources are required. The wrapper uses `/Users/yuktikhanna/miniconda3/bin/python3` by default and accepts a `PYTHON_BIN` override.
