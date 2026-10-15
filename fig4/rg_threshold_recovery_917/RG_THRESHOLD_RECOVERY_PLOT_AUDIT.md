# RG-threshold recovery plotting audit

Source tables are the completed one-factor outputs `one_factor_compact_recovery.csv`, `one_factor_bridge_recovery.csv`, and `one_factor_flank_recovery.csv`. No simulations or analyses are rerun by the plotting script.

- Region denominator: `total_downstream_RG_RGG_regions = 627` unique downstream regions.
- Pair denominator: `pair_total_pairs = 917` final simulation pairs.
- Compact values: 5, 6, 7, 8, 9 aa; bridge remains 15 and flank remains 5.
- Bridge values: 13, 14, 15, 16, 17 aa; compact remains 7 and flank remains 5.
- Flank values: 3, 4, 5, 6, 7 aa; compact remains 7 and bridge remains 15.
- Any overlap: the completed analysis field `unique_recovered_percent` / `pair_recovered_percent`; an RG/RGG region or pair is retained when its downstream region overlaps a recovered region, including altered boundaries.
- Exact-coordinate match: the completed analysis fields `unique_exact_percent` / `pair_exact_percent`.
- Lost/not represented: the completed analysis fields `unique_lost_percent` / `pair_lost_percent`, i.e. no recovered overlapping tract.

Baseline `(7,15,5)` is present in each one-factor table and is required by the plotting script to recover 627/627 regions and 917/917 pairs. The script exports only plotting tables and S4J/S4K figures inside this directory.
