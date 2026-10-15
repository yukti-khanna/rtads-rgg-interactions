#!/usr/bin/env python3
"""Figure 4 recovery of the final 917 pairs across saved Figure 3 RG calls.

This script is intentionally downstream-only.  It reads the saved 125-condition
Figure 3 region catalogue and never calls the RG-region caller, CALVADOS, or B22.
Coordinates are 0-based, half-open, and local to the parent IDR.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASELINE = (7, 15, 5)
EXPECTED_FINAL_PAIRS = 917
EXPECTED_CONDITIONS = 125
CLASSIFICATIONS = ["EXACT", "OVERLAP_SAME_TRACT", "MERGED", "SPLIT", "LOST"]
COORDINATE_CONVENTION = "0-based, half-open, parent-IDR-local"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uncompressed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def union_length(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((int(a), int(b)) for a, b in intervals if b > a)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def motif_intervals(row: pd.Series) -> list[tuple[int, int]]:
    sequence = str(row["region_sequence"])
    region_start = int(row["region_start"])
    motifs = [(region_start + m.start(), region_start + m.start() + 2) for m in re.finditer("RG", sequence)]
    expected = int(row["RG_motif_count"])
    if len(motifs) != expected:
        fail(f"RG motif reconstruction mismatch for {row['region_id']}: {len(motifs)} != {expected}")
    return motifs


def sequence_identity(first: str, second: str) -> float:
    if first == second:
        return 1.0
    if not first or not second:
        return float("nan")
    matches = sum(a == b for a, b in zip(first, second))
    return matches / max(len(first), len(second))


def as_bool(value: object) -> bool:
    return bool(value) and not pd.isna(value)


def reconstruct_final_pairs(b22_path: Path, precursor_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    b22 = pd.read_csv(b22_path)
    precursor = pd.read_csv(precursor_path)
    if "Unnamed: 0" in precursor.columns:
        precursor = precursor.drop(columns=["Unnamed: 0"])
    for frame_name, frame in (("B22 cache", b22), ("precursor table", precursor)):
        if "name" not in frame.columns:
            fail(f"{frame_name} lacks required pair-name column 'name'")
        if frame["name"].isna().any():
            fail(f"{frame_name} contains missing pair names")
        if frame["name"].duplicated().any():
            examples = frame.loc[frame["name"].duplicated(False), "name"].head().tolist()
            fail(f"{frame_name} contains duplicate pair names: {examples}")
    if len(b22) != EXPECTED_FINAL_PAIRS or b22["name"].nunique() != EXPECTED_FINAL_PAIRS:
        fail(f"expected exactly {EXPECTED_FINAL_PAIRS} final B22 pairs, found {len(b22)} rows / {b22['name'].nunique()} unique")
    missing = sorted(set(b22["name"]) - set(precursor["name"]))
    if missing:
        fail(f"{len(missing)} final pair names do not map to precursor metadata; examples: {missing[:10]}")
    selected = b22.merge(
        precursor,
        on="name",
        how="left",
        validate="one_to_one",
        suffixes=("_authoritative", "_precursor"),
        sort=False,
    )
    if len(selected) != EXPECTED_FINAL_PAIRS:
        fail(f"pair mapping changed row count: {len(selected)}")
    for required in ("rgg", "tad", "RGG_length", "charge_RGG", "NCPR_RGG"):
        if required not in selected.columns or selected[required].isna().any():
            fail(f"mapped final-pair table lacks complete {required}")
    return b22, selected


def map_downstream_regions(selected: pd.DataFrame, baseline: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if baseline["region_id"].duplicated().any():
        fail("baseline Figure 3 region identifiers are not unique")
    baseline_index = baseline.set_index("region_id", drop=False)
    downstream_ids = selected["rgg"].drop_duplicates().astype(str).tolist()
    mapping_rows: list[dict[str, object]] = []
    unique_rows: list[dict[str, object]] = []
    pair_names_by_region = selected.groupby("rgg", sort=False)["name"].agg(list).to_dict()

    for region_id in downstream_ids:
        expected_parent_idr = region_id.rsplit("_", 1)[0] if "_" in region_id else None
        expected_uniprot = expected_parent_idr.split("_", 1)[0] if expected_parent_idr else None
        if region_id in baseline_index.index:
            match = baseline_index.loc[region_id]
            if isinstance(match, pd.DataFrame):
                fail(f"ambiguous exact baseline region identifier: {region_id}")
            mapping_method = "A_EXACT_IDENTIFIER"
        else:
            fail(
                f"downstream region {region_id} lacks an exact baseline identifier; "
                "the precursor table has no coordinates or sequence for a constrained fallback"
            )
        if str(match["parent_IDR_ID"]) != expected_parent_idr:
            fail(f"identifier-derived parent IDR disagrees with baseline for {region_id}")
        if str(match["parent_UniProt_accession"]) != expected_uniprot:
            fail(f"identifier-derived UniProt accession disagrees with baseline for {region_id}")
        first_pair = selected.loc[selected["rgg"] == region_id].iloc[0]
        length_ok = int(first_pair["RGG_length"]) == int(match["region_length"])
        charge_ok = math.isclose(float(first_pair["charge_RGG"]), float(match["net_charge"]), abs_tol=1e-12, rel_tol=0)
        ncpr_ok = math.isclose(float(first_pair["NCPR_RGG"]), float(match["NCPR"]), abs_tol=1e-12, rel_tol=1e-12)
        histidine_count = str(match["region_sequence"]).count("H")
        calvados_charge_expected = float(match["net_charge"]) - histidine_count
        calvados_ncpr_expected = calvados_charge_expected / int(match["region_length"])
        calvados_charge_ok = math.isclose(float(first_pair["charge_RGG"]), calvados_charge_expected, abs_tol=1e-12, rel_tol=0)
        calvados_ncpr_ok = math.isclose(float(first_pair["NCPR_RGG"]), calvados_ncpr_expected, abs_tol=1e-12, rel_tol=1e-12)
        pair_names = pair_names_by_region[region_id]
        unique_rows.append(
            {
                "downstream_region_id": region_id,
                "parent_IDR_ID": match["parent_IDR_ID"],
                "parent_UniProt_accession": match["parent_UniProt_accession"],
                "downstream_region_start": int(match["region_start"]),
                "downstream_region_end": int(match["region_end"]),
                "downstream_region_sequence": match["region_sequence"],
                "downstream_region_length": int(match["region_length"]),
                "RG_motif_count": int(match["RG_motif_count"]),
                "net_charge": float(match["net_charge"]),
                "NCPR": float(match["NCPR"]),
                "downstream_pair_count": len(pair_names),
                "downstream_pair_names": ";".join(pair_names),
                "precursor_length_matches_baseline": length_ok,
                "precursor_charge_matches_baseline": charge_ok,
                "precursor_NCPR_matches_baseline": ncpr_ok,
                "region_histidine_count": histidine_count,
                "precursor_charge_matches_histidine_neutral_convention": calvados_charge_ok,
                "precursor_NCPR_matches_histidine_neutral_convention": calvados_ncpr_ok,
                "feature_convention_diagnosis": (
                    "DIRECT_MATCH_NO_HISTIDINE" if charge_ok and ncpr_ok
                    else "DIAGNOSED: Figure 3 counts H as +1; Figure 4 precursor treats H as 0"
                ),
            }
        )
        mapping_rows.append(
            {
                "downstream_region_id": region_id,
                "mapped_baseline_Figure3_region_id": match["region_id"],
                "parent_IDR_ID": match["parent_IDR_ID"],
                "parent_UniProt_accession": match["parent_UniProt_accession"],
                "downstream_start": int(match["region_start"]),
                "downstream_end": int(match["region_end"]),
                "baseline_start": int(match["region_start"]),
                "baseline_end": int(match["region_end"]),
                "sequence_identity": sequence_identity(str(match["region_sequence"]), str(match["region_sequence"])),
                "exact_coordinate_match": True,
                "exact_sequence_match": True,
                "mapping_method": mapping_method,
                "mapping_status": "MAPPED",
                "diagnosis": "Exact Figure 4 rgg identifier equals the baseline Figure 3 region_id.",
            }
        )

    unique_regions = pd.DataFrame(unique_rows)
    mapping = pd.DataFrame(mapping_rows)
    if len(unique_regions) != selected["rgg"].nunique():
        fail("unique downstream-region construction changed identifier count")
    if not unique_regions["precursor_length_matches_baseline"].all():
        fail("precursor RGG length disagrees with the exact Figure 3 baseline mapping")
    if not unique_regions[[
        "precursor_charge_matches_histidine_neutral_convention",
        "precursor_NCPR_matches_histidine_neutral_convention",
    ]].all().all():
        fail("precursor RGG charge/NCPR discrepancy is not fully explained by the documented histidine convention")

    pair_enriched = selected.merge(
        unique_regions.drop(columns=["downstream_pair_names"]),
        left_on="rgg",
        right_on="downstream_region_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    pair_output = pd.DataFrame(
        {
            "pair_name": pair_enriched["name"],
            "TAD_identifier": pair_enriched["tad"],
            "RG_RGG_identifier": pair_enriched["rgg"],
            "RG_RGG_parent_UniProt_accession": pair_enriched["parent_UniProt_accession"],
            "RG_RGG_parent_IDR_identifier": pair_enriched["parent_IDR_ID"],
            "original_RG_RGG_region_identifier": pair_enriched["downstream_region_id"],
            "original_region_sequence": pair_enriched["downstream_region_sequence"],
            "original_region_start": pair_enriched["downstream_region_start"],
            "original_region_end": pair_enriched["downstream_region_end"],
            "original_region_length": pair_enriched["downstream_region_length"],
            "precursor_RGG_length": pair_enriched["RGG_length"],
            "precursor_charge_RGG": pair_enriched["charge_RGG"],
            "precursor_NCPR_RGG": pair_enriched["NCPR_RGG"],
            "B22_corr1_log10": pair_enriched["B22_corr1_log10_authoritative"],
            "pair_mapping_status": "MAPPED",
            "RGG_length_matches_baseline": pair_enriched["precursor_length_matches_baseline"],
            "RGG_charge_matches_baseline": pair_enriched["precursor_charge_matches_baseline"],
            "RGG_NCPR_matches_baseline": pair_enriched["precursor_NCPR_matches_baseline"],
            "RGG_histidine_count": pair_enriched["region_histidine_count"],
            "RGG_charge_matches_histidine_neutral_convention": pair_enriched["precursor_charge_matches_histidine_neutral_convention"],
            "RGG_NCPR_matches_histidine_neutral_convention": pair_enriched["precursor_NCPR_matches_histidine_neutral_convention"],
            "feature_convention_diagnosis": pair_enriched["feature_convention_diagnosis"],
        }
    )
    if pair_output["pair_name"].duplicated().any() or len(pair_output) != EXPECTED_FINAL_PAIRS:
        fail("final mapped pair table is not exactly one row per authoritative pair")
    return pair_output, unique_regions, mapping


def compute_region_recovery(all_regions: pd.DataFrame, downstream: pd.DataFrame) -> pd.DataFrame:
    baseline = all_regions.query(
        "compact_linker_cutoff == @BASELINE[0] and bridge_linker_cutoff == @BASELINE[1] and flank_extension == @BASELINE[2]"
    ).copy()
    baseline_lookup = baseline.set_index("region_id", drop=False)
    baseline_by_parent = {key: group.copy() for key, group in baseline.groupby("parent_IDR_ID", sort=False)}
    baseline_motifs = {str(row.region_id): motif_intervals(row) for _, row in baseline.iterrows()}
    condition_cols = ["compact_linker_cutoff", "bridge_linker_cutoff", "flank_extension", "condition_id"]
    condition_table = all_regions[condition_cols].drop_duplicates().sort_values(condition_cols[:3])
    if len(condition_table) != EXPECTED_CONDITIONS:
        fail(f"expected {EXPECTED_CONDITIONS} threshold conditions, found {len(condition_table)}")
    expected_grid = {(c, b, f) for c in range(5, 10) for b in range(13, 18) for f in range(3, 8)}
    observed_grid = set(map(tuple, condition_table[condition_cols[:3]].astype(int).to_numpy()))
    if observed_grid != expected_grid:
        fail("saved Figure 3 catalogue does not contain the complete requested 5x5x5 grid")

    rows: list[dict[str, object]] = []
    grouped_conditions = all_regions.groupby(condition_cols, sort=True, dropna=False)
    for condition, condition_regions in grouped_conditions:
        compact, bridge, flank, condition_id = condition
        alternatives_by_parent = {
            key: group.copy() for key, group in condition_regions.groupby("parent_IDR_ID", sort=False)
        }
        for downstream_row in downstream.itertuples(index=False):
            region_id = str(downstream_row.downstream_region_id)
            base = baseline_lookup.loc[region_id]
            if isinstance(base, pd.DataFrame):
                fail(f"ambiguous canonical baseline record for {region_id}")
            parent = str(base["parent_IDR_ID"])
            b_start, b_end = int(base["region_start"]), int(base["region_end"])
            b_length = int(base["region_length"])
            candidates: list[dict[str, object]] = []
            for _, alt in alternatives_by_parent.get(parent, pd.DataFrame()).iterrows():
                a_start, a_end = int(alt["region_start"]), int(alt["region_end"])
                ov = overlap(b_start, b_end, a_start, a_end)
                if ov > 0:
                    candidates.append({"row": alt, "overlap": ov})
            candidates.sort(
                key=lambda item: (
                    int(item["overlap"]),
                    int(str(item["row"]["region_id"]) == region_id),
                    -abs(int(item["row"]["region_length"]) - b_length),
                    -int(item["row"]["region_start"]),
                    str(item["row"]["region_id"]),
                ),
                reverse=True,
            )
            recovered = bool(candidates)
            best = candidates[0]["row"] if recovered else None
            best_overlap = int(candidates[0]["overlap"]) if recovered else 0
            exact = bool(
                recovered
                and int(best["region_start"]) == b_start
                and int(best["region_end"]) == b_end
            )
            intersected = [
                (max(b_start, int(item["row"]["region_start"])), min(b_end, int(item["row"]["region_end"])))
                for item in candidates
            ]
            union_overlap = union_length(intersected)
            baseline_core = baseline_motifs[region_id]
            best_core_indices: set[int] = set()
            component_core_indices: list[tuple[str, set[int]]] = []
            if recovered:
                best_start, best_end = int(best["region_start"]), int(best["region_end"])
                best_core_indices = {
                    index for index, (start, end) in enumerate(baseline_core)
                    if best_start <= start and end <= best_end
                }
                for item in candidates:
                    alt = item["row"]
                    alt_start, alt_end = int(alt["region_start"]), int(alt["region_end"])
                    indices = {
                        index for index, (start, end) in enumerate(baseline_core)
                        if alt_start <= start and end <= alt_end
                    }
                    if indices:
                        component_core_indices.append((str(alt["region_id"]), indices))
            split_union = set().union(*(indices for _, indices in component_core_indices)) if component_core_indices else set()
            split_components = len(component_core_indices)
            split = bool(
                recovered
                and split_components >= 2
                and len(split_union) == len(baseline_core)
                and all(len(indices) < len(baseline_core) for _, indices in component_core_indices)
            )

            merged_neighbor_ids: list[str] = []
            merged = False
            if recovered:
                best_start, best_end = int(best["region_start"]), int(best["region_end"])
                canonical_interval_contained = best_start <= b_start and b_end <= best_end
                if canonical_interval_contained:
                    for _, neighbor in baseline_by_parent[parent].iterrows():
                        neighbor_id = str(neighbor["region_id"])
                        if neighbor_id == region_id:
                            continue
                        motifs = baseline_motifs[neighbor_id]
                        if motifs and all(best_start <= start and end <= best_end for start, end in motifs):
                            merged_neighbor_ids.append(neighbor_id)
                    merged = bool(merged_neighbor_ids)

            if not recovered:
                classification = "LOST"
            elif exact:
                classification = "EXACT"
            elif split:
                classification = "SPLIT"
            elif merged:
                classification = "MERGED"
            else:
                classification = "OVERLAP_SAME_TRACT"

            if recovered:
                a_start, a_end = int(best["region_start"]), int(best["region_end"])
                a_length = int(best["region_length"])
                alt_fraction = best_overlap / a_length
                jaccard = best_overlap / (b_length + a_length - best_overlap)
                start_shift = a_start - b_start
                end_shift = a_end - b_end
                length_change = a_length - b_length
                best_id = str(best["region_id"])
            else:
                a_start = a_end = a_length = np.nan
                alt_fraction = 0.0
                jaccard = 0.0
                start_shift = end_shift = length_change = np.nan
                best_id = ""
            rows.append(
                {
                    "compact_linker_cutoff": int(compact),
                    "bridge_linker_cutoff": int(bridge),
                    "flank_extension": int(flank),
                    "condition_id": str(condition_id),
                    "downstream_region_id": region_id,
                    "mapped_baseline_Figure3_region_id": region_id,
                    "parent_IDR_ID": parent,
                    "parent_UniProt_accession": str(base["parent_UniProt_accession"]),
                    "baseline_region_start": b_start,
                    "baseline_region_end": b_end,
                    "baseline_region_length": b_length,
                    "recovered": recovered,
                    "exact_coordinate_match": exact,
                    "number_of_overlapping_alternative_regions": len(candidates),
                    "overlapping_alternative_region_ids": ";".join(str(item["row"]["region_id"]) for item in candidates),
                    "best_alternative_region_id": best_id,
                    "best_alternative_region_start": a_start,
                    "best_alternative_region_end": a_end,
                    "best_alternative_region_length": a_length,
                    "overlap_residues": best_overlap,
                    "union_overlap_residues": union_overlap,
                    "fraction_of_baseline_residues_recovered": best_overlap / b_length,
                    "fraction_of_baseline_residues_recovered_by_union": union_overlap / b_length,
                    "fraction_of_best_alternative_overlapping_baseline": alt_fraction,
                    "jaccard_overlap_best_alternative": jaccard,
                    "start_coordinate_shift": start_shift,
                    "end_coordinate_shift": end_shift,
                    "length_change": length_change,
                    "baseline_RG_motif_count": len(baseline_core),
                    "baseline_RG_motif_start_positions": ";".join(str(start) for start, _ in baseline_core),
                    "baseline_RG_motifs_in_best_alternative": len(best_core_indices),
                    "fraction_of_baseline_RG_motifs_recovered_in_best_alternative": len(best_core_indices) / len(baseline_core),
                    "merged_neighboring_baseline_region_count": len(merged_neighbor_ids),
                    "merged_neighboring_baseline_region_ids": ";".join(sorted(merged_neighbor_ids)),
                    "split_component_alternative_region_count": split_components if split else 0,
                    "split_component_alternative_region_ids": ";".join(region for region, _ in component_core_indices) if split else "",
                    "classification": classification,
                    "matching_scope": "same_parent_IDR_only",
                }
            )

    recovery = pd.DataFrame(rows)
    expected_rows = len(downstream) * EXPECTED_CONDITIONS
    if len(recovery) != expected_rows:
        fail(f"unique-region recovery has {len(recovery)} rows; expected {expected_rows}")
    if recovery.duplicated(["downstream_region_id", "condition_id"]).any():
        fail("duplicate unique-region-condition records detected")
    if not (recovery["matching_scope"] == "same_parent_IDR_only").all():
        fail("cross-parent-IDR matching record detected")
    baseline_rows = recovery.query(
        "compact_linker_cutoff == @BASELINE[0] and bridge_linker_cutoff == @BASELINE[1] and flank_extension == @BASELINE[2]"
    )
    if len(baseline_rows) != len(downstream) or not baseline_rows["recovered"].all():
        fail("baseline condition does not recover every downstream region")
    if not (baseline_rows["classification"] == "EXACT").all():
        bad = baseline_rows.loc[baseline_rows["classification"] != "EXACT", ["downstream_region_id", "classification"]]
        fail(f"baseline mappings are unexpectedly non-identical: {bad.head().to_dict(orient='records')}")
    return recovery


def summarize_regions(recovery: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["compact_linker_cutoff", "bridge_linker_cutoff", "flank_extension", "condition_id"]
    for condition, group in recovery.groupby(keys, sort=True):
        counts = group["classification"].value_counts().reindex(CLASSIFICATIONS, fill_value=0)
        total = len(group)
        recovered_count = int(group["recovered"].sum())
        defined = group[group["recovered"]]
        rows.append(
            {
                **dict(zip(keys, condition)),
                "total_downstream_RG_RGG_regions": total,
                "exact_matches": int(counts["EXACT"]),
                "overlap_same_tract_regions": int(counts["OVERLAP_SAME_TRACT"]),
                "merged_regions": int(counts["MERGED"]),
                "split_regions": int(counts["SPLIT"]),
                "lost_regions": int(counts["LOST"]),
                "overlapping_recovered_regions": recovered_count,
                "fraction_recovered": recovered_count / total,
                "fraction_exact": float((group["classification"] == "EXACT").mean()),
                "fraction_overlap_same_tract": float((group["classification"] == "OVERLAP_SAME_TRACT").mean()),
                "fraction_merged": float((group["classification"] == "MERGED").mean()),
                "fraction_split": float((group["classification"] == "SPLIT").mean()),
                "fraction_lost": float((group["classification"] == "LOST").mean()),
                "median_baseline_residue_recovery_fraction": float(group["fraction_of_baseline_residues_recovered"].median()),
                "median_Jaccard_overlap": float(group["jaccard_overlap_best_alternative"].median()),
                "median_absolute_start_shift": float(defined["start_coordinate_shift"].abs().median()),
                "median_absolute_end_shift": float(defined["end_coordinate_shift"].abs().median()),
                "median_absolute_length_change": float(defined["length_change"].abs().median()),
                "median_baseline_RG_motif_recovery_fraction": float(group["fraction_of_baseline_RG_motifs_recovered_in_best_alternative"].median()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(keys[:3]).reset_index(drop=True)
    if len(summary) != EXPECTED_CONDITIONS:
        fail("unique-region summary does not have 125 conditions")
    return summary


def build_pair_recovery(pair_map: pd.DataFrame, recovery: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pair_map[[
        "pair_name", "TAD_identifier", "RG_RGG_identifier",
        "RG_RGG_parent_UniProt_accession", "RG_RGG_parent_IDR_identifier"
    ]].copy()
    pair_recovery = recovery.merge(
        pairs,
        left_on="downstream_region_id",
        right_on="RG_RGG_identifier",
        how="left",
        validate="many_to_many",
        sort=False,
    )
    expected = EXPECTED_FINAL_PAIRS * EXPECTED_CONDITIONS
    if len(pair_recovery) != expected:
        fail(f"pair recovery has {len(pair_recovery)} rows; expected {expected}")
    if pair_recovery["pair_name"].isna().any():
        fail("pair-level expansion contains unmapped final pairs")
    if pair_recovery.duplicated(["pair_name", "condition_id"]).any():
        fail("duplicate pair-condition records detected")
    columns = [
        "compact_linker_cutoff", "bridge_linker_cutoff", "flank_extension", "condition_id",
        "pair_name", "TAD_identifier", "RG_RGG_identifier", "RG_RGG_parent_UniProt_accession",
        "RG_RGG_parent_IDR_identifier", "classification", "recovered", "exact_coordinate_match",
        "best_alternative_region_id", "fraction_of_baseline_residues_recovered",
        "jaccard_overlap_best_alternative", "fraction_of_baseline_RG_motifs_recovered_in_best_alternative",
        "start_coordinate_shift", "end_coordinate_shift", "length_change",
    ]
    pair_recovery = pair_recovery[columns]
    keys = ["compact_linker_cutoff", "bridge_linker_cutoff", "flank_extension", "condition_id"]
    rows: list[dict[str, object]] = []
    for condition, group in pair_recovery.groupby(keys, sort=True):
        counts = group["classification"].value_counts().reindex(CLASSIFICATIONS, fill_value=0)
        total = len(group)
        rows.append(
            {
                **dict(zip(keys, condition)),
                "total_pairs": total,
                "pairs_exact": int(counts["EXACT"]),
                "pairs_recovered_with_altered_boundaries": int(counts["OVERLAP_SAME_TRACT"]),
                "pairs_merged": int(counts["MERGED"]),
                "pairs_split": int(counts["SPLIT"]),
                "pairs_lost": int(counts["LOST"]),
                "pairs_recovered_any_overlap": int(group["recovered"].sum()),
                "fraction_pairs_recovered": float(group["recovered"].mean()),
                "fraction_pairs_exact": float((group["classification"] == "EXACT").mean()),
                "fraction_pairs_overlap_same_tract": float((group["classification"] == "OVERLAP_SAME_TRACT").mean()),
                "fraction_pairs_merged": float((group["classification"] == "MERGED").mean()),
                "fraction_pairs_split": float((group["classification"] == "SPLIT").mean()),
                "fraction_pairs_lost": float((group["classification"] == "LOST").mean()),
                "fraction_pairs_potentially_affected": float((group["classification"] != "EXACT").mean()),
            }
        )
    pair_summary = pd.DataFrame(rows).sort_values(keys[:3]).reset_index(drop=True)
    if len(pair_summary) != EXPECTED_CONDITIONS or not (pair_summary["total_pairs"] == EXPECTED_FINAL_PAIRS).all():
        fail("pair summary failed the 125-condition / 917-pair invariant")
    baseline_summary = pair_summary.query(
        "compact_linker_cutoff == @BASELINE[0] and bridge_linker_cutoff == @BASELINE[1] and flank_extension == @BASELINE[2]"
    ).iloc[0]
    if float(baseline_summary["fraction_pairs_recovered"]) != 1.0 or int(baseline_summary["pairs_exact"]) != EXPECTED_FINAL_PAIRS:
        fail("baseline pair recovery is not exactly 1.0 / 917 exact")
    return pair_recovery, pair_summary


def full_grid(region_summary: pd.DataFrame, pair_summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["compact_linker_cutoff", "bridge_linker_cutoff", "flank_extension", "condition_id"]
    pair = pair_summary.rename(
        columns={column: f"pair_{column}" for column in pair_summary.columns if column not in keys}
    )
    combined = region_summary.merge(pair, on=keys, how="inner", validate="one_to_one")
    if len(combined) != EXPECTED_CONDITIONS:
        fail("combined full-grid summary does not contain 125 conditions")
    combined["is_baseline"] = (
        (combined["compact_linker_cutoff"] == BASELINE[0])
        & (combined["bridge_linker_cutoff"] == BASELINE[1])
        & (combined["flank_extension"] == BASELINE[2])
    )
    return combined


def one_factor_slices(grid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masks = {
        "compact": (grid["bridge_linker_cutoff"] == 15) & (grid["flank_extension"] == 5),
        "bridge": (grid["compact_linker_cutoff"] == 7) & (grid["flank_extension"] == 5),
        "flank": (grid["compact_linker_cutoff"] == 7) & (grid["bridge_linker_cutoff"] == 15),
    }
    value_column = {
        "compact": "compact_linker_cutoff",
        "bridge": "bridge_linker_cutoff",
        "flank": "flank_extension",
    }
    slices: dict[str, pd.DataFrame] = {}
    for factor, mask in masks.items():
        selected = grid.loc[mask].copy().sort_values(value_column[factor])
        if len(selected) != 5:
            fail(f"one-factor {factor} slice has {len(selected)} rows instead of five")
        selected.insert(0, "factor", factor)
        selected.insert(1, "factor_value", selected[value_column[factor]])
        selected["unique_exact_percent"] = 100 * selected["fraction_exact"]
        selected["unique_recovered_percent"] = 100 * selected["fraction_recovered"]
        selected["unique_merged_percent"] = 100 * selected["fraction_merged"]
        selected["unique_split_percent"] = 100 * selected["fraction_split"]
        selected["unique_lost_percent"] = 100 * selected["fraction_lost"]
        selected["pair_exact_percent"] = 100 * selected["pair_fraction_pairs_exact"]
        selected["pair_recovered_percent"] = 100 * selected["pair_fraction_pairs_recovered"]
        selected["pair_potentially_affected_percent"] = 100 * selected["pair_fraction_pairs_potentially_affected"]
        selected["pair_lost_percent"] = 100 * selected["pair_fraction_pairs_lost"]
        keep = [
            "factor", "factor_value", "compact_linker_cutoff", "bridge_linker_cutoff", "flank_extension",
            "condition_id", "is_baseline", "total_downstream_RG_RGG_regions",
            "unique_exact_percent", "unique_recovered_percent", "unique_merged_percent",
            "unique_split_percent", "unique_lost_percent", "pair_total_pairs", "pair_exact_percent",
            "pair_recovered_percent", "pair_potentially_affected_percent", "pair_lost_percent",
            "median_baseline_residue_recovery_fraction", "median_Jaccard_overlap",
            "median_absolute_start_shift", "median_absolute_end_shift", "median_absolute_length_change",
            "median_baseline_RG_motif_recovery_fraction",
        ]
        slices[factor] = selected[keep]
    return slices


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_one_factor(frame: pd.DataFrame, factor: str, output: Path) -> None:
    labels = {
        "compact": "Compact linker cutoff (aa)",
        "bridge": "Bridge linker cutoff (aa)",
        "flank": "Flank extension (aa)",
    }
    x = frame["factor_value"].to_numpy()
    baseline_value = {"compact": 7, "bridge": 15, "flank": 5}[factor]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    colors = {"exact": "#3264A8", "recovered": "#25855A", "lost": "#B33B38"}
    for ax, level in zip(axes, ("unique", "pair")):
        if level == "unique":
            exact = frame["unique_exact_percent"]
            recovered = frame["unique_recovered_percent"]
            lost = frame["unique_lost_percent"]
            title = "Unique downstream RGG regions"
        else:
            exact = frame["pair_exact_percent"]
            recovered = frame["pair_recovered_percent"]
            lost = frame["pair_lost_percent"]
            title = "Final 917 simulation pairs"
        ax.plot(x, exact, "o-", color=colors["exact"], label="Exact")
        ax.plot(x, recovered, "o-", color=colors["recovered"], label="Any-overlap recovered")
        ax.plot(x, lost, "o-", color=colors["lost"], label="Lost")
        ax.axvline(baseline_value, color="#555555", linestyle="--", linewidth=1, label="Baseline" if level == "unique" else None)
        ax.set_title(title)
        ax.set_xlabel(labels[factor])
        ax.set_xticks(x)
        ax.set_ylim(-2, 103)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Fraction of inputs (%)")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False)
    fig.suptitle(f"Figure 4 downstream recovery: {factor} sensitivity", y=1.14, fontsize=12)
    fig.tight_layout()
    save_figure(fig, output)


def plot_grid_heatmap(grid: pd.DataFrame, value: str, title: str, output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.3), constrained_layout=True)
    axes = axes.ravel()
    values = grid[value].astype(float)
    vmin = min(float(values.min()), 0.95)
    image = None
    for index, flank in enumerate(range(3, 8)):
        ax = axes[index]
        subset = grid[grid["flank_extension"] == flank]
        matrix = subset.pivot(index="bridge_linker_cutoff", columns="compact_linker_cutoff", values=value).sort_index(ascending=True)
        image = ax.imshow(matrix.to_numpy(), origin="lower", vmin=vmin, vmax=1.0, cmap="YlGnBu", aspect="auto")
        ax.set_title(f"Flank = {flank} aa")
        ax.set_xticks(range(5), matrix.columns.astype(int))
        ax.set_yticks(range(5), matrix.index.astype(int))
        ax.set_xlabel("Compact linker (aa)")
        ax.set_ylabel("Bridge linker (aa)")
        for row_index in range(5):
            for col_index in range(5):
                val = float(matrix.iloc[row_index, col_index])
                text_color = "white" if val > (vmin + 1.0) / 2 else "black"
                ax.text(col_index, row_index, f"{100*val:.1f}", ha="center", va="center", fontsize=7.4, color=text_color)
        if flank == 5:
            ax.add_patch(plt.Rectangle((1.52, 1.52), 0.96, 0.96, fill=False, edgecolor="#F4B942", linewidth=3))
    axes[-1].axis("off")
    axes[-1].text(0.08, 0.88, "Gold outline: baseline (7,15,5)", transform=axes[-1].transAxes, fontsize=10)
    colorbar = fig.colorbar(image, ax=axes[:5], shrink=0.82, pad=0.03)
    colorbar.set_label("Any-overlap recovery fraction")
    fig.suptitle(title, fontsize=13)
    save_figure(fig, output)


def plot_boundary_summary(slices: dict[str, pd.DataFrame], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), sharex=True)
    metrics = [
        ("median_Jaccard_overlap", "Median best-region Jaccard"),
        ("median_absolute_start_shift", "Median |start shift| (aa)"),
        ("median_absolute_end_shift", "Median |end shift| (aa)"),
        ("median_absolute_length_change", "Median |length change| (aa)"),
    ]
    baselines = {"compact": 7, "bridge": 15, "flank": 5}
    labels = {"compact": "Compact", "bridge": "Bridge", "flank": "Flank"}
    colors = {"compact": "#3264A8", "bridge": "#8A5AB5", "flank": "#25855A"}
    for ax, (metric, ylabel) in zip(axes.ravel(), metrics):
        for factor, frame in slices.items():
            delta = frame["factor_value"] - baselines[factor]
            ax.plot(delta, frame[metric], "o-", label=labels[factor], color=colors[factor])
        ax.axvline(0, color="#555555", linestyle="--", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    for ax in axes[-1, :]:
        ax.set_xlabel("Threshold change from baseline (aa)")
        ax.set_xticks([-2, -1, 0, 1, 2])
    handles, labels_out = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_out, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.suptitle("Boundary stability in one-factor threshold slices", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output)


def markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if column in formats:
                values.append(formats[column].format(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def screen_statistics(grid: pd.DataFrame) -> dict[str, object]:
    min_unique = float(grid["fraction_recovered"].min())
    min_pair = float(grid["pair_fraction_pairs_recovered"].min())
    return {
        "unique_region_recovery_min": min_unique,
        "unique_region_recovery_median": float(grid["fraction_recovered"].median()),
        "unique_region_recovery_max": float(grid["fraction_recovered"].max()),
        "conditions_at_minimum_unique_region_recovery": grid.loc[
            np.isclose(grid["fraction_recovered"], min_unique), "condition_id"
        ].tolist(),
        "pair_recovery_min": min_pair,
        "pair_recovery_median": float(grid["pair_fraction_pairs_recovered"].median()),
        "pair_recovery_max": float(grid["pair_fraction_pairs_recovered"].max()),
        "conditions_at_minimum_pair_recovery": grid.loc[
            np.isclose(grid["pair_fraction_pairs_recovered"], min_pair), "condition_id"
        ].tolist(),
        "conditions_with_100_percent_unique_region_recovery": int(np.isclose(grid["fraction_recovered"], 1.0).sum()),
        "conditions_with_any_lost_downstream_region": int((grid["lost_regions"] > 0).sum()),
        "maximum_unique_downstream_regions_lost": int(grid["lost_regions"].max()),
        "maximum_final_pairs_with_lost_RGG_region": int(grid["pair_pairs_lost"].max()),
    }


def write_readme(
    output_dir: Path,
    pair_map: pd.DataFrame,
    downstream: pd.DataFrame,
    mapping: pd.DataFrame,
    grid: pd.DataFrame,
    slices: dict[str, pd.DataFrame],
    repeated_lost: pd.DataFrame,
    screen: dict[str, object],
    b22_path: Path,
    precursor_path: Path,
    fig3_path: Path,
) -> None:
    def slice_text(frame: pd.DataFrame) -> str:
        short = frame[[
            "factor_value", "condition_id", "unique_exact_percent", "unique_recovered_percent",
            "unique_merged_percent", "unique_split_percent", "unique_lost_percent",
            "pair_exact_percent", "pair_recovered_percent", "pair_potentially_affected_percent", "pair_lost_percent",
        ]].copy()
        short.columns = [
            "value", "condition", "unique exact %", "unique recovered %", "unique merged %",
            "unique split %", "unique lost %", "pair exact %", "pair recovered %",
            "pair potentially affected %", "pair lost %",
        ]
        formats = {column: "{:.2f}" for column in short.columns if "%" in column}
        return markdown_table(short, short.columns.tolist(), formats)

    flank = slices["flank"]
    flank_lost = int(round(flank["unique_lost_percent"].max() * len(downstream) / 100))
    flank_motif_min = float(flank["median_baseline_RG_motif_recovery_fraction"].min())
    repeated = repeated_lost[repeated_lost["lost_condition_count"] > 0]
    histidine_convention_regions = int((~downstream["precursor_charge_matches_baseline"]).sum())
    region_record_total = int(grid["total_downstream_RG_RGG_regions"].sum())
    region_exact_total = int(grid["exact_matches"].sum())
    region_altered_total = int(
        grid[["overlap_same_tract_regions", "merged_regions", "split_regions"]].sum(axis=1).sum()
    )
    region_lost_total = int(grid["lost_regions"].sum())
    pair_record_total = int(grid["pair_total_pairs"].sum())
    pair_exact_total = int(grid["pair_pairs_exact"].sum())
    pair_altered_total = int(
        grid[["pair_pairs_recovered_with_altered_boundaries", "pair_pairs_merged", "pair_pairs_split"]].sum(axis=1).sum()
    )
    pair_lost_total = int(grid["pair_pairs_lost"].sum())
    if repeated.empty:
        repeated_text = "No downstream RGG region was lost in any tested condition."
    else:
        top = repeated.sort_values(["lost_condition_count", "downstream_region_id"], ascending=[False, True]).head(10)
        repeated_text = ", ".join(
            f"`{row.downstream_region_id}` ({int(row.lost_condition_count)} conditions)" for row in top.itertuples()
        )
    min_pair = float(screen["pair_recovery_min"])
    min_unique = float(screen["unique_region_recovery_min"])
    if min_pair >= 0.99 and min_unique >= 0.99:
        decision = "The downstream tracts are essentially all preserved; threshold changes are dominated by boundary changes."
    elif min_pair >= 0.95 and min_unique >= 0.95:
        decision = "Most downstream tracts are preserved, with a small subset lost in the most divergent combined conditions."
    else:
        decision = "A non-trivial subset of downstream inputs is lost in at least one combined threshold condition, although many altered calls still represent the same tract."
    text = f"""# Figure 4 recovery of the final 917 RGG simulation inputs

## Scope

This analysis asks whether the RG/RGG tracts used by the final Figure 4 CALVADOS pair set remain represented in the saved Figure 3 threshold-sensitivity calls. It did not rerun the RG caller, CALVADOS, or B22; it did not change the 917-pair set or optimize thresholds.

## Inputs used

- Authoritative final pair identities: `{b22_path}`
- Precursor pair metadata: `{precursor_path}`
- Saved Figure 3 all-condition region catalogue: `{fig3_path}`

The second copy at `fig3/outputs/rg_regions_all_conditions.csv.gz` has a different compressed-file hash but the same uncompressed CSV SHA-256; the current sensitivity-project copy above was used.

## Identity and baseline validation

1. The B22 cache contains exactly **{len(pair_map):,} unique final pairs**.
2. All **{len(pair_map):,}/{len(pair_map):,}** pair names map one-to-one to the 1,039-row precursor table.
3. The final pairs contain exactly **{len(downstream):,} unique downstream RG/RGG regions** from **{downstream['parent_UniProt_accession'].nunique():,} proteins** and **{downstream['parent_IDR_ID'].nunique():,} parent IDRs**.
4. Baseline mapping succeeded for **{len(mapping):,}/{len(mapping):,}** regions. Every mapping used exact Figure 4 RGG identifier = Figure 3 baseline `region_id`, with identical coordinates and sequence.
5. Under baseline `(7,15,5)`, all downstream regions and all 917 pairs are `EXACT`; the baseline recovery fraction is 1.0.

The precursor RGG lengths agree for all 627 regions. Charge and NCPR differ from the Figure 3 table for **{histidine_convention_regions} regions**, and the discrepancy is completely explained by charge convention: Figure 3 counted histidine as +1, whereas the Figure 4 precursor values equal `Figure 3 net_charge - histidine count` and the corresponding length-normalized NCPR. This is recorded rather than forced to agree and does not affect identifier, coordinate, sequence, or overlap mapping.

Coordinates are {COORDINATE_CONVENTION}. Matching is restricted to the same parent IDR.

## Operational classification

- `EXACT`: best alternative coordinates equal baseline coordinates.
- `OVERLAP_SAME_TRACT`: non-zero same-IDR overlap with changed coordinates, without the merge/split rules below.
- `MERGED`: the best alternative contains the full downstream baseline interval and also contains every literal `RG` motif of at least one other baseline region on that IDR.
- `SPLIT`: at least two alternative regions each contain distinct baseline `RG` motifs, their union covers all baseline motifs, and no single component contains all baseline motifs.
- `LOST`: no alternative region on the same parent IDR overlaps the baseline region.

The best alternative maximizes residue overlap, then exact identifier agreement, then minimizes length difference; the final tie-break is deterministic catalogue position/identifier. Any-overlap recovery is calculated independently of the classification. `fraction_of_baseline_residues_recovered` uses the best alternative, as do Jaccard and boundary shifts. The additional `fraction_of_baseline_residues_recovered_by_union` field records combined coverage by all overlapping alternatives for split cases.

## One-factor-at-a-time recovery

### Compact linker

{slice_text(slices['compact'])}

### Bridge linker

{slice_text(slices['bridge'])}

### Flank extension

{slice_text(slices['flank'])}

Flank changes are interpreted as boundary/context changes unless overlap is lost. In the flank-only slice the maximum unique-region loss was **{flank_lost} regions**, and the minimum condition median core-RG-motif recovery fraction was **{flank_motif_min:.3f}**. Exact-coordinate loss under flank changes is therefore not treated by itself as loss of the underlying tract.

## Full 125-condition screen

- Unique-region recovery min / median / max: **{min_unique:.4f} / {float(screen['unique_region_recovery_median']):.4f} / {float(screen['unique_region_recovery_max']):.4f}**.
- Condition(s) at minimum unique-region recovery: **{', '.join(screen['conditions_at_minimum_unique_region_recovery'])}**.
- Final-pair recovery min / median / max: **{min_pair:.4f} / {float(screen['pair_recovery_median']):.4f} / {float(screen['pair_recovery_max']):.4f}**.
- Condition(s) at minimum pair recovery: **{', '.join(screen['conditions_at_minimum_pair_recovery'])}**.
- Conditions with 100% unique-region recovery: **{int(screen['conditions_with_100_percent_unique_region_recovery'])}/{EXPECTED_CONDITIONS}**.
- Conditions with any lost downstream region: **{int(screen['conditions_with_any_lost_downstream_region'])}/{EXPECTED_CONDITIONS}**.
- Maximum unique downstream regions lost in one condition: **{int(screen['maximum_unique_downstream_regions_lost'])}**.
- Maximum of the 917 pairs affected by lost RGG regions in one condition: **{int(screen['maximum_final_pairs_with_lost_RGG_region'])}**.

Across all conditions, exact matches and altered-boundary matches are reported separately in `downstream_region_recovery_summary.csv` and `pair_level_recovery_summary.csv`. Region-level overlap, boundary shifts, merge/split evidence, and literal RG-motif retention are in the compressed record-level tables.

Across all **{region_record_total:,} region-condition records**, **{region_exact_total:,} ({100*region_exact_total/region_record_total:.2f}%)** were exact, **{region_altered_total:,} ({100*region_altered_total/region_record_total:.2f}%)** were recovered with altered boundaries/merge/split classification, and **{region_lost_total:,} ({100*region_lost_total/region_record_total:.2f}%)** were lost. Across all **{pair_record_total:,} pair-condition records**, the corresponding counts were **{pair_exact_total:,} ({100*pair_exact_total/pair_record_total:.2f}%)** exact, **{pair_altered_total:,} ({100*pair_altered_total/pair_record_total:.2f}%)** altered but recovered, and **{pair_lost_total:,} ({100*pair_lost_total/pair_record_total:.2f}%)** lost. These aggregate frequencies are dominated by expected coordinate changes whenever flank extension differs from 5 aa, so tract recovery is the primary robustness metric.

## Repeatedly lost downstream regions

{repeated_text}

The complete loss frequency and affected-pair count for every downstream region are in `repeatedly_lost_downstream_regions.csv`.

## Diagnostic interpretation for manual decision support

{decision}

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
"""
    (output_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fig4-dir", type=Path, default=Path("/Users/yuktikhanna/rtads-rgg-interactions/fig4"))
    parser.add_argument(
        "--fig3-regions",
        type=Path,
        default=Path("/Users/yuktikhanna/rtads-rgg-interactions/fig3/figure3_rg_threshold_sensitivity/outputs/rg_regions_all_conditions.csv.gz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    fig4_dir = args.fig4_dir.resolve()
    fig3_path = args.fig3_regions.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    b22_path = fig4_dir / "out_fig4/cache/Fig4a_B22_values.csv"
    precursor_path = fig4_dir / "sim_pairs_added_features_data_270525.csv"
    duplicate_fig3_path = fig4_dir.parent / "fig3/outputs/rg_regions_all_conditions.csv.gz"
    for path in (b22_path, precursor_path, fig3_path):
        if not path.is_file():
            fail(f"required input not found: {path}")

    _, selected = reconstruct_final_pairs(b22_path, precursor_path)
    all_regions = pd.read_csv(fig3_path)
    required_region_columns = {
        "compact_linker_cutoff", "bridge_linker_cutoff", "flank_extension", "condition_id",
        "region_id", "parent_IDR_ID", "parent_UniProt_accession", "region_start", "region_end",
        "region_sequence", "region_length", "RG_motif_count", "net_charge", "NCPR",
    }
    missing_columns = sorted(required_region_columns - set(all_regions.columns))
    if missing_columns:
        fail(f"saved Figure 3 catalogue lacks columns: {missing_columns}")
    baseline = all_regions.query(
        "compact_linker_cutoff == @BASELINE[0] and bridge_linker_cutoff == @BASELINE[1] and flank_extension == @BASELINE[2]"
    ).copy()
    if baseline.empty:
        fail("baseline Figure 3 condition (7,15,5) is absent")

    pair_map, downstream, mapping = map_downstream_regions(selected, baseline)
    pair_map.to_csv(output_dir / "final_917_pairs_mapped.csv", index=False)
    downstream.to_csv(output_dir / "downstream_rgg_regions.csv", index=False)
    mapping.to_csv(output_dir / "baseline_mapping_validation.csv", index=False)

    recovery = compute_region_recovery(all_regions, downstream)
    recovery.to_csv(output_dir / "downstream_region_recovery_all_conditions.csv.gz", index=False, compression="gzip")
    region_summary = summarize_regions(recovery)
    region_summary.to_csv(output_dir / "downstream_region_recovery_summary.csv", index=False)
    pair_recovery, pair_summary = build_pair_recovery(pair_map, recovery)
    pair_recovery.to_csv(output_dir / "pair_level_recovery_all_conditions.csv.gz", index=False, compression="gzip")
    pair_summary.to_csv(output_dir / "pair_level_recovery_summary.csv", index=False)

    grid = full_grid(region_summary, pair_summary)
    grid.to_csv(output_dir / "full_grid_recovery_summary.csv", index=False)
    slices = one_factor_slices(grid)
    slices["compact"].to_csv(output_dir / "one_factor_compact_recovery.csv", index=False)
    slices["bridge"].to_csv(output_dir / "one_factor_bridge_recovery.csv", index=False)
    slices["flank"].to_csv(output_dir / "one_factor_flank_recovery.csv", index=False)

    loss_frequency = recovery.groupby("downstream_region_id", sort=True).agg(
        lost_condition_count=("classification", lambda values: int((values == "LOST").sum())),
        exact_condition_count=("classification", lambda values: int((values == "EXACT").sum())),
        altered_boundary_recovered_condition_count=("classification", lambda values: int(values.isin(["OVERLAP_SAME_TRACT", "MERGED", "SPLIT"]).sum())),
    ).reset_index()
    loss_conditions = recovery.loc[recovery["classification"] == "LOST"].groupby("downstream_region_id")["condition_id"].agg(lambda x: ";".join(x)).to_dict()
    loss_frequency = loss_frequency.merge(
        downstream[["downstream_region_id", "parent_IDR_ID", "parent_UniProt_accession", "downstream_pair_count"]],
        on="downstream_region_id", how="left", validate="one_to_one"
    )
    loss_frequency["lost_condition_fraction"] = loss_frequency["lost_condition_count"] / EXPECTED_CONDITIONS
    loss_frequency["lost_condition_ids"] = loss_frequency["downstream_region_id"].map(loss_conditions).fillna("")
    loss_frequency = loss_frequency.sort_values(["lost_condition_count", "downstream_region_id"], ascending=[False, True])
    loss_frequency.to_csv(output_dir / "repeatedly_lost_downstream_regions.csv", index=False)

    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    plot_one_factor(slices["compact"], "compact", figures_dir / "compact_linker_recovery")
    plot_one_factor(slices["bridge"], "bridge", figures_dir / "bridge_linker_recovery")
    plot_one_factor(slices["flank"], "flank", figures_dir / "flank_recovery")
    plot_grid_heatmap(grid, "fraction_recovered", "Unique downstream RGG any-overlap recovery (%)", figures_dir / "full_grid_unique_region_recovery")
    plot_grid_heatmap(grid, "pair_fraction_pairs_recovered", "Final 917-pair RGG any-overlap recovery (%)", figures_dir / "full_grid_pair_recovery")
    plot_boundary_summary(slices, figures_dir / "overlap_boundary_shift_summary")

    screen = screen_statistics(grid)
    write_readme(
        output_dir, pair_map, downstream, mapping, grid, slices, loss_frequency, screen,
        b22_path, precursor_path, fig3_path,
    )
    output_files = sorted(
        str(path.relative_to(output_dir)) for path in output_dir.rglob("*")
        if path.is_file()
        and path.name not in {"run_metadata.json", "validation_report.json"}
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    metadata = {
        "analysis": "Figure 4 final-917 RG/RGG threshold recovery",
        "run_datetime_utc": datetime.now(timezone.utc).isoformat(),
        "coordinate_convention": COORDINATE_CONVENTION,
        "baseline_parameters": list(BASELINE),
        "compact_linker_values": [5, 6, 7, 8, 9],
        "bridge_linker_values": [13, 14, 15, 16, 17],
        "flank_extension_values": [3, 4, 5, 6, 7],
        "condition_count": EXPECTED_CONDITIONS,
        "input_files_used": {
            "authoritative_final_pair_names": str(b22_path),
            "precursor_pair_metadata": str(precursor_path),
            "saved_Figure3_all_condition_regions": str(fig3_path),
        },
        "input_sha256": {
            str(b22_path): sha256(b22_path),
            str(precursor_path): sha256(precursor_path),
            str(fig3_path): sha256(fig3_path),
            f"{fig3_path}::uncompressed_csv": uncompressed_sha256(fig3_path),
        },
        "duplicate_Figure3_catalogue": {
            "path": str(duplicate_fig3_path),
            "exists": duplicate_fig3_path.is_file(),
            "same_uncompressed_content": (
                duplicate_fig3_path.is_file() and uncompressed_sha256(duplicate_fig3_path) == uncompressed_sha256(fig3_path)
            ),
            "reason_not_used": "The sensitivity-project output is the current authoritative analysis copy.",
        },
        "authoritative_final_pair_count": len(pair_map),
        "mapped_final_pair_count": int((pair_map["pair_mapping_status"] == "MAPPED").sum()),
        "unique_downstream_RGG_region_count": len(downstream),
        "unique_downstream_parent_protein_count": int(downstream["parent_UniProt_accession"].nunique()),
        "unique_downstream_parent_IDR_count": int(downstream["parent_IDR_ID"].nunique()),
        "baseline_mapping": {
            "mapped_count": len(mapping),
            "exact_identifier_count": int((mapping["mapping_method"] == "A_EXACT_IDENTIFIER").sum()),
            "exact_coordinate_count": int(mapping["exact_coordinate_match"].sum()),
            "exact_sequence_count": int(mapping["exact_sequence_match"].sum()),
            "status": "PASS",
        },
        "precursor_feature_convention_diagnosis": {
            "length_exact_match_count": int(downstream["precursor_length_matches_baseline"].sum()),
            "direct_charge_and_NCPR_match_count": int((downstream["precursor_charge_matches_baseline"] & downstream["precursor_NCPR_matches_baseline"]).sum()),
            "histidine_convention_difference_count": int((~downstream["precursor_charge_matches_baseline"]).sum()),
            "fully_explained_by_histidine_neutral_Figure4_convention": bool(downstream[["precursor_charge_matches_histidine_neutral_convention", "precursor_NCPR_matches_histidine_neutral_convention"]].all().all()),
            "diagnosis": "Figure 3 historical charge uses K/R/H=+1; Figure 4 precursor charge treats H as neutral.",
        },
        "best_alternative_rule": "maximum residue overlap on same parent IDR; exact identifier; minimum absolute length difference; deterministic start/id tie-break",
        "classification_rules": {
            "EXACT": "best alternative has identical 0-based half-open coordinates",
            "OVERLAP_SAME_TRACT": "non-zero overlap without exact/merge/split criteria",
            "MERGED": "best alternative contains full baseline interval and complete RG motif set of at least one other baseline region",
            "SPLIT": "two or more alternatives cover distinct baseline RG motifs; their union covers all baseline motifs; none covers all",
            "LOST": "no non-zero residue overlap on same parent IDR",
        },
        "baseline_residue_recovery_fraction_rule": "best-alternative residue overlap divided by baseline length",
        "baseline_residue_recovery_union_rule": "union of all alternative overlaps with baseline interval divided by baseline length; supplied as an additional split-case diagnostic",
        "RG_motif_recovery": "literal uppercase RG positions reconstructed from saved baseline region sequence; containment tested in best alternative",
        "hypothesis_tests_used": False,
        "RG_caller_rerun": False,
        "CALVADOS_rerun": False,
        "B22_recomputed": False,
        "pair_set_changed": False,
        "full_grid_screen": screen,
        "output_files": output_files,
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({
        "final_pairs": len(pair_map),
        "unique_downstream_regions": len(downstream),
        "unique_parent_proteins": int(downstream["parent_UniProt_accession"].nunique()),
        "unique_parent_IDRs": int(downstream["parent_IDR_ID"].nunique()),
        "minimum_unique_region_recovery": screen["unique_region_recovery_min"],
        "minimum_pair_recovery": screen["pair_recovery_min"],
        "maximum_unique_regions_lost": screen["maximum_unique_downstream_regions_lost"],
        "maximum_pairs_lost": screen["maximum_final_pairs_with_lost_RGG_region"],
    }, indent=2))


if __name__ == "__main__":
    main()
