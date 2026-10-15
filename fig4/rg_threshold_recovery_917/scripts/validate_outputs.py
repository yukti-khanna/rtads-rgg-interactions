#!/usr/bin/env python3
"""Independent structural and numerical checks for Figure 4 RGG recovery outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_PAIRS = 917
EXPECTED_CONDITIONS = 125
BASELINE = (7, 15, 5)
CLASSES = {"EXACT", "OVERLAP_SAME_TRACT", "MERGED", "SPLIT", "LOST"}


def check(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.output_dir.resolve()
    checks: list[str] = []

    required = [
        "final_917_pairs_mapped.csv", "downstream_rgg_regions.csv", "baseline_mapping_validation.csv",
        "downstream_region_recovery_all_conditions.csv.gz", "downstream_region_recovery_summary.csv",
        "pair_level_recovery_all_conditions.csv.gz", "pair_level_recovery_summary.csv",
        "one_factor_compact_recovery.csv", "one_factor_bridge_recovery.csv", "one_factor_flank_recovery.csv",
        "full_grid_recovery_summary.csv", "repeatedly_lost_downstream_regions.csv", "README.md", "run_metadata.json",
    ]
    for filename in required:
        check((root / filename).is_file() and (root / filename).stat().st_size > 0, f"required output exists and is non-empty: {filename}", checks)

    pair_map = pd.read_csv(root / "final_917_pairs_mapped.csv")
    downstream = pd.read_csv(root / "downstream_rgg_regions.csv")
    mapping = pd.read_csv(root / "baseline_mapping_validation.csv")
    region = pd.read_csv(root / "downstream_region_recovery_all_conditions.csv.gz")
    region_summary = pd.read_csv(root / "downstream_region_recovery_summary.csv")
    pair = pd.read_csv(root / "pair_level_recovery_all_conditions.csv.gz")
    pair_summary = pd.read_csv(root / "pair_level_recovery_summary.csv")
    grid = pd.read_csv(root / "full_grid_recovery_summary.csv")

    check(len(pair_map) == EXPECTED_PAIRS, "final pair map has exactly 917 rows", checks)
    check(pair_map["pair_name"].nunique() == EXPECTED_PAIRS and not pair_map["pair_name"].duplicated().any(), "final pair map has 917 unique pair names", checks)
    check((pair_map["pair_mapping_status"] == "MAPPED").all(), "all final pairs map to precursor metadata", checks)
    check(pair_map["RGG_length_matches_baseline"].all(), "precursor RGG lengths agree with baseline mappings", checks)
    check(pair_map[["RGG_charge_matches_histidine_neutral_convention", "RGG_NCPR_matches_histidine_neutral_convention"]].all().all(), "precursor charge/NCPR differences are fully explained by histidine convention", checks)

    n_regions = downstream["downstream_region_id"].nunique()
    check(len(downstream) == n_regions and not downstream["downstream_region_id"].duplicated().any(), "downstream table has one row per unique RGG region", checks)
    check(int(downstream["downstream_pair_count"].sum()) == EXPECTED_PAIRS, "downstream region pair multiplicities sum to 917", checks)
    check(downstream[["precursor_charge_matches_histidine_neutral_convention", "precursor_NCPR_matches_histidine_neutral_convention"]].all().all(), "unique-region charge/NCPR convention diagnosis is complete", checks)
    check(len(mapping) == n_regions, "baseline mapping has one row per downstream RGG region", checks)
    check((mapping["mapping_status"] == "MAPPED").all(), "all downstream regions map to baseline", checks)
    check((mapping["mapping_method"] == "A_EXACT_IDENTIFIER").all(), "all downstream baseline mappings use exact identifiers", checks)
    check(mapping[["exact_coordinate_match", "exact_sequence_match"]].all().all(), "all baseline mappings have exact coordinates and sequence", checks)
    check((mapping["downstream_start"] == mapping["baseline_start"]).all() and (mapping["downstream_end"] == mapping["baseline_end"]).all(), "baseline mapping coordinates are identical", checks)

    condition_cols = ["compact_linker_cutoff", "bridge_linker_cutoff", "flank_extension"]
    observed = set(map(tuple, region[condition_cols].drop_duplicates().astype(int).to_numpy()))
    expected = {(c, b, f) for c in range(5, 10) for b in range(13, 18) for f in range(3, 8)}
    check(observed == expected and len(observed) == EXPECTED_CONDITIONS, "region recovery contains the complete 125-condition grid", checks)
    check(len(region) == n_regions * EXPECTED_CONDITIONS, "region recovery has exactly unique_regions x 125 records", checks)
    check(not region.duplicated(["downstream_region_id", "condition_id"]).any(), "region-condition keys are unique", checks)
    check(set(region["classification"].unique()) <= CLASSES, "region classifications use only the documented labels", checks)
    check((region["matching_scope"] == "same_parent_IDR_only").all(), "all alternative matching is restricted to the same parent IDR", checks)
    check((region["recovered"] == (region["classification"] != "LOST")).all(), "recovered flag agrees with LOST classification", checks)
    check((region["exact_coordinate_match"] == (region["classification"] == "EXACT")).all(), "exact-coordinate flag agrees with EXACT classification", checks)
    check(region["fraction_of_baseline_residues_recovered"].between(0, 1).all(), "baseline residue recovery fractions are bounded [0,1]", checks)
    check(region["fraction_of_baseline_residues_recovered_by_union"].between(0, 1).all(), "union baseline residue recovery fractions are bounded [0,1]", checks)
    check(region["jaccard_overlap_best_alternative"].between(0, 1).all(), "Jaccard overlaps are bounded [0,1]", checks)
    check(region["fraction_of_baseline_RG_motifs_recovered_in_best_alternative"].between(0, 1).all(), "RG motif recovery fractions are bounded [0,1]", checks)
    check((region["overlap_residues"] <= region["union_overlap_residues"]).all(), "best overlap never exceeds union overlap", checks)
    check((region["fraction_of_baseline_residues_recovered"] <= region["fraction_of_baseline_residues_recovered_by_union"] + 1e-12).all(), "best-match baseline recovery never exceeds union recovery", checks)

    baseline_region = region.query("compact_linker_cutoff == 7 and bridge_linker_cutoff == 15 and flank_extension == 5")
    check(len(baseline_region) == n_regions, "baseline has one recovery row per downstream region", checks)
    check((baseline_region["classification"] == "EXACT").all() and baseline_region["recovered"].all(), "baseline recovers every downstream region exactly", checks)
    check(np.allclose(baseline_region["fraction_of_baseline_residues_recovered"], 1.0), "baseline residue recovery fraction is 1", checks)
    check(np.allclose(baseline_region["jaccard_overlap_best_alternative"], 1.0), "baseline Jaccard is 1", checks)
    check(np.allclose(baseline_region["fraction_of_baseline_RG_motifs_recovered_in_best_alternative"], 1.0), "baseline RG motif recovery fraction is 1", checks)

    check(len(pair) == EXPECTED_PAIRS * EXPECTED_CONDITIONS, "pair recovery has exactly 917 x 125 records", checks)
    check(not pair.duplicated(["pair_name", "condition_id"]).any(), "pair-condition keys are unique", checks)
    check(pair["pair_name"].nunique() == EXPECTED_PAIRS, "pair recovery retains all 917 pair names", checks)
    baseline_pair = pair.query("compact_linker_cutoff == 7 and bridge_linker_cutoff == 15 and flank_extension == 5")
    check(len(baseline_pair) == EXPECTED_PAIRS and (baseline_pair["classification"] == "EXACT").all(), "baseline recovers all 917 pair inputs exactly", checks)

    check(len(region_summary) == EXPECTED_CONDITIONS and region_summary["condition_id"].nunique() == EXPECTED_CONDITIONS, "unique-region summary has 125 unique conditions", checks)
    check(len(pair_summary) == EXPECTED_CONDITIONS and pair_summary["condition_id"].nunique() == EXPECTED_CONDITIONS, "pair summary has 125 unique conditions", checks)
    check((region_summary["total_downstream_RG_RGG_regions"] == n_regions).all(), "region summary uses the complete downstream region set", checks)
    check((pair_summary["total_pairs"] == EXPECTED_PAIRS).all(), "pair summary uses the complete 917-pair set", checks)
    check((region_summary[["exact_matches", "overlap_same_tract_regions", "merged_regions", "split_regions", "lost_regions"]].sum(axis=1) == n_regions).all(), "region summary classifications reconcile to total", checks)
    check((pair_summary[["pairs_exact", "pairs_recovered_with_altered_boundaries", "pairs_merged", "pairs_split", "pairs_lost"]].sum(axis=1) == EXPECTED_PAIRS).all(), "pair summary classifications reconcile to 917", checks)
    check(len(grid) == EXPECTED_CONDITIONS and grid["condition_id"].nunique() == EXPECTED_CONDITIONS, "full-grid table has 125 unique conditions", checks)

    factor_expectations = {
        "compact": ({5, 6, 7, 8, 9}, "compact_linker_cutoff", lambda d: (d["bridge_linker_cutoff"] == 15) & (d["flank_extension"] == 5)),
        "bridge": ({13, 14, 15, 16, 17}, "bridge_linker_cutoff", lambda d: (d["compact_linker_cutoff"] == 7) & (d["flank_extension"] == 5)),
        "flank": ({3, 4, 5, 6, 7}, "flank_extension", lambda d: (d["compact_linker_cutoff"] == 7) & (d["bridge_linker_cutoff"] == 15)),
    }
    for factor, (values, column, fixed_mask) in factor_expectations.items():
        frame = pd.read_csv(root / f"one_factor_{factor}_recovery.csv")
        check(len(frame) == 5 and set(frame[column].astype(int)) == values, f"{factor} one-factor table has the requested five values", checks)
        check(fixed_mask(frame).all(), f"{factor} one-factor table holds other parameters at baseline", checks)
        check(frame["is_baseline"].sum() == 1, f"{factor} one-factor table identifies baseline exactly once", checks)

    expected_figures = [
        "compact_linker_recovery", "bridge_linker_recovery", "flank_recovery",
        "full_grid_unique_region_recovery", "full_grid_pair_recovery", "overlap_boundary_shift_summary",
    ]
    for stem in expected_figures:
        for suffix in (".png", ".pdf"):
            path = root / "figures" / f"{stem}{suffix}"
            check(path.is_file() and path.stat().st_size > 1000, f"diagnostic figure is non-empty: figures/{stem}{suffix}", checks)

    metadata = json.loads((root / "run_metadata.json").read_text())
    check(metadata["authoritative_final_pair_count"] == EXPECTED_PAIRS, "metadata records 917 authoritative pairs", checks)
    check(metadata["unique_downstream_RGG_region_count"] == n_regions, "metadata unique-region count agrees with output", checks)
    check(metadata["condition_count"] == EXPECTED_CONDITIONS, "metadata records 125 threshold conditions", checks)
    check(metadata["precursor_feature_convention_diagnosis"]["fully_explained_by_histidine_neutral_Figure4_convention"] is True, "metadata records the diagnosed histidine charge convention", checks)
    check(metadata["RG_caller_rerun"] is False and metadata["CALVADOS_rerun"] is False and metadata["B22_recomputed"] is False, "metadata confirms prohibited computations were not run", checks)
    readme_text = (root / "README.md").read_text()
    check("917" in readme_text and "decision support" in readme_text.lower(), "README contains required final-set and decision-support sections", checks)

    report = {
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks_passed": len(checks),
        "checks": checks,
        "counts": {
            "final_pairs": EXPECTED_PAIRS,
            "unique_downstream_regions": n_regions,
            "conditions": EXPECTED_CONDITIONS,
            "region_condition_rows": len(region),
            "pair_condition_rows": len(pair),
        },
        "key_output_sha256": {
            filename: sha256(root / filename) for filename in required if (root / filename).is_file()
        },
    }
    (root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "checks_passed": len(checks), **report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
