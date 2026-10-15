#!/usr/bin/env python3
"""Quantify model-ensemble recurrence within published activation domains.

This script continues the Figure 2 reviewer analyses.  It does not load or
retrain classifiers and it does not reconstruct model scores.  Its recurrence
source is the complete-universe table produced by
``reconstruct_prediction_universe_bias.py``.  The complete candidate FASTA is
used to map each 15-aa header back to a biological candidate region and to
verify the recurrence-table denominator.

External AD coordinate eligibility and final-catalogue recovery use the same
exact-coordinate / same-protein BLAST workflow as
``external_AD_catalogue_recovery.py``.  Existing mapping tables can be supplied
and are preferred; otherwise those mappings are recreated without rerunning
BLAST.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import fisher_exact, mannwhitneyu, norm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from external_AD_catalogue_recovery import (  # noqa: E402
    DATASETS,
    all_occurrences,
    external_clusters,
    external_mapping_detail,
    family,
    fasta_dict,
    fasta_records,
    read_gene_map,
    read_hits,
)
from reconstruct_prediction_universe_bias import (  # noqa: E402
    parse_model_file,
    prediction_entries,
)


SOURCE_DATASET_NAMES = [item[0] for item in DATASETS]
UNION_NAME = "Union of >=1 external dataset"
MULTI_NAME = "External regions supported by >=2 datasets"
DISORDER_PROMOTING = set("TAGRDHQKSEP")
DEFAULT_THRESHOLDS = [0.0, 0.25, 0.50, 0.75, 0.85, 0.90]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": .6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = norm.ppf(.975)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def bh(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    observed = values[finite]
    if not len(observed):
        return adjusted
    order = np.argsort(observed)
    ranked = observed[order] * len(observed) / np.arange(1, len(observed) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    restored = np.empty(len(observed))
    restored[order] = np.minimum(ranked, 1.0)
    adjusted[finite] = restored
    return adjusted


def cliffs_delta(first: np.ndarray, second: np.ndarray) -> float:
    """Signed Cliff's delta; positive means values in first are higher."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if not len(first) or not len(second):
        return np.nan
    ordered = np.sort(second)
    greater = np.searchsorted(ordered, first, side="left").sum()
    less = (len(second) - np.searchsorted(ordered, first, side="right")).sum()
    return float((greater - less) / (len(first) * len(second)))


def bootstrap_cliffs_delta(
    external: np.ndarray,
    control: np.ndarray,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Matched-set bootstrap, preserving each external/control pairing."""
    external = np.asarray(external, dtype=float)
    control = np.asarray(control, dtype=float)
    if len(external) != len(control) or not len(external):
        return np.nan, np.nan, 0
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sample = rng.integers(0, len(external), size=len(external))
        estimates[index] = cliffs_delta(external[sample], control[sample])
    return (
        float(np.quantile(estimates, .025)),
        float(np.quantile(estimates, .975)),
        int(np.isfinite(estimates).sum()),
    )


def disorder_fraction(sequence: str) -> float:
    sequence = str(sequence).upper()
    return sum(residue in DISORDER_PROMOTING for residue in sequence) / len(sequence) if sequence else np.nan


def split_window_header(header: str) -> tuple[str, int]:
    try:
        parent, start = str(header).rsplit(":", 1)
        return parent, int(start)
    except Exception as exc:
        raise ValueError(f"Candidate-window header must end ':<integer start>': {header}") from exc


def reconstruct_parent_regions(window_fasta: Path, candidate_regions: pd.DataFrame):
    """Reconstruct consecutive parent sequences from grouped 15-aa windows.

    The historical FASTA is written parent-by-parent in increasing local-start
    order.  We assert that ordering rather than silently joining incompatible
    windows.  This holds only one parent assembly in memory at a time.
    """
    targets = set(candidate_regions.region_id.astype(str))
    completed = set()
    records = []
    current_parent = None
    current_min = None
    current_sequence = ""
    current_windows = 0
    fasta_windows = 0

    def finish():
        nonlocal current_parent, current_min, current_sequence, current_windows
        if current_parent is None or current_parent not in targets:
            return
        records.append({
            "region_id": current_parent,
            "local_min_start": int(current_min),
            "local_max_end": int(current_min + len(current_sequence) - 1),
            "sequence": current_sequence,
            "sequence_length": len(current_sequence),
            "fasta_n_windows": current_windows,
            "disorder_fraction": disorder_fraction(current_sequence),
        })
        completed.add(current_parent)

    for header, window_sequence in fasta_records(window_fasta):
        fasta_windows += 1
        parent, start = split_window_header(header)
        if parent != current_parent:
            finish()
            if parent in completed:
                raise ValueError(
                    f"Candidate FASTA is not grouped by parent region; {parent} occurs in multiple blocks. "
                    "Sort by parent and integer window start before running."
                )
            current_parent = parent
            current_min = start
            current_sequence = window_sequence
            current_windows = 1
            continue
        current_windows += 1
        offset = start - current_min
        if offset < 0:
            raise ValueError(f"Window starts decrease within parent {parent}")
        if offset > len(current_sequence):
            raise ValueError(f"Gap in reconstructed parent {parent} before local start {start}")
        overlap = len(current_sequence) - offset
        if overlap > 0 and current_sequence[offset:] != window_sequence[:overlap]:
            raise ValueError(f"Conflicting overlapping residues in parent {parent} at local start {start}")
        if overlap < len(window_sequence):
            current_sequence += window_sequence[overlap:]
    finish()

    reconstructed = pd.DataFrame(records)
    missing = sorted(targets - set(reconstructed.region_id.astype(str)))
    if missing:
        raise ValueError(
            f"Only reconstructed {len(reconstructed):,}/{len(targets):,} candidate regions; "
            f"first missing IDs: {missing[:10]}"
        )
    return reconstructed, fasta_windows


def map_parent_regions(
    candidate_regions: pd.DataFrame,
    reconstructed: pd.DataFrame,
    references: dict[str, str],
) -> tuple[pd.DataFrame, dict]:
    required = {"region_id", "protein"}
    if not required.issubset(candidate_regions):
        raise ValueError(f"Candidate-region table missing {sorted(required - set(candidate_regions.columns))}")
    merged = candidate_regions.merge(reconstructed, on="region_id", how="left", validate="one_to_one")
    statuses = defaultdict(int)
    starts, ends, status_values = [], [], []
    for row in merged[["protein", "sequence"]].itertuples(index=False):
        protein, sequence = str(row.protein), str(row.sequence)
        if protein not in references:
            status, start, end = "protein_not_in_TF_reference", np.nan, np.nan
        else:
            occurrences = all_occurrences(references[protein], sequence)
            if len(occurrences) == 1:
                status, start = "unique_exact_coordinate", occurrences[0]
                end = start + len(sequence) - 1
            elif len(occurrences) == 0:
                status, start, end = "not_exact_substring", np.nan, np.nan
            else:
                status, start, end = "ambiguous_multiple_exact_locations", np.nan, np.nan
        statuses[status] += 1
        status_values.append(status)
        starts.append(start)
        ends.append(end)
    merged["coordinate_mapping_status"] = status_values
    merged["protein_start"] = starts
    merged["protein_end"] = ends
    if "n_windows" in merged:
        discrepancy = merged.n_windows.astype(int).ne(merged.fasta_n_windows.astype(int))
        if discrepancy.any():
            examples = merged.loc[discrepancy, ["region_id", "n_windows", "fasta_n_windows"]].head(10)
            raise ValueError("Candidate region n_windows disagrees with FASTA:\n" + examples.to_string(index=False))
    return merged, dict(statuses)


def candidate_interval_dict(parent_map: pd.DataFrame):
    intervals = defaultdict(list)
    mapped = parent_map[parent_map.coordinate_mapping_status.eq("unique_exact_coordinate")]
    for row in mapped[["protein", "protein_start", "protein_end", "region_id"]].itertuples(index=False):
        intervals[str(row.protein)].append((int(row.protein_start), int(row.protein_end), str(row.region_id)))
    for protein in intervals:
        intervals[protein].sort()
    return intervals


def load_or_create_external_mapping(args, references, genes, parent_map):
    detail_path = args.external_mapping_detail
    clusters_path = args.external_union_clusters
    if detail_path and detail_path.is_file():
        detail = pd.read_csv(detail_path)
        mapping_source = f"existing mapping detail: {detail_path.resolve()}"
    else:
        hit_tables, hit_map = read_hits(args.blast_dir)
        detail = external_mapping_detail(
            args.external_fasta_dir,
            references,
            genes,
            candidate_interval_dict(parent_map),
            hit_map,
            args.minimum_universe_coverage,
        )
        mapping_source = "recreated with external_AD_catalogue_recovery.py exact/same-protein rules; BLAST not rerun"

    required = {
        "dataset", "external_region_id", "protein", "family", "protein_start", "protein_end",
        "coordinate_mapping_status", "eligible", "recovered",
    }
    if not required.issubset(detail):
        raise ValueError(f"External mapping detail missing {sorted(required - set(detail.columns))}")

    if clusters_path and clusters_path.is_file():
        clusters = pd.read_csv(clusters_path)
    else:
        clusters = external_clusters(
            detail,
            references,
            candidate_interval_dict(parent_map),
            args.minimum_universe_coverage,
        )
    return detail, clusters, mapping_source


def normalize_analysis_records(detail: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    source = detail.copy()
    source["record_type"] = "source_external_region"
    source["analysis_group"] = source.dataset.astype(str)
    source["analysis_record_id"] = (
        "source|" + source.dataset.astype(str) + "|" + source.external_region_id.astype(str)
    )
    source["external_start"] = source.protein_start
    source["external_end"] = source.protein_end
    source["dataset_support_n"] = 1
    source["final_catalogue_recovered"] = source.recovered.astype(int)
    if "region_length" not in source:
        source["region_length"] = source.external_end - source.external_start + 1

    union = clusters.copy()
    union["record_type"] = "deduplicated_union_cluster"
    union["dataset"] = UNION_NAME
    union["analysis_group"] = UNION_NAME
    union["external_region_id"] = union.external_cluster_id.astype(str)
    union["analysis_record_id"] = "union|" + union.external_cluster_id.astype(str)
    union["external_start"] = union.protein_start
    union["external_end"] = union.protein_end
    union["coordinate_mapping_status"] = "coordinate_deduplicated_cluster"
    union["final_catalogue_recovered"] = union.recovered.astype(int)
    if "family" not in union:
        union["family"] = "other"

    shared = sorted(set(source.columns) | set(union.columns))
    return pd.concat(
        [source.reindex(columns=shared), union.reindex(columns=shared)],
        ignore_index=True,
        sort=False,
    )


def intervals_overlap(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    return any(start <= right and end >= left for left, right in intervals)


def crop_to_disorder(parent_row, target_length: int, target_disorder: float):
    sequence = str(parent_row.sequence)
    if len(sequence) <= target_length:
        return (
            int(parent_row.protein_start), int(parent_row.protein_end), sequence,
            disorder_fraction(sequence), "whole_candidate_region_shorter_than_target",
        )
    flags = np.fromiter((residue in DISORDER_PROMOTING for residue in sequence), dtype=np.int16)
    prefix = np.concatenate([[0], np.cumsum(flags)])
    counts = prefix[target_length:] - prefix[:-target_length]
    fractions = counts / target_length
    best = int(np.argmin(np.abs(fractions - target_disorder)))
    start = int(parent_row.protein_start) + best
    end = start + target_length - 1
    return start, end, sequence[best:best + target_length], float(fractions[best]), "exact_length_subregion"


def match_controls_for_group(
    group_name: str,
    external: pd.DataFrame,
    parent_pool: pd.DataFrame,
) -> pd.DataFrame:
    """One control per eligible external record, without replacement."""
    used = set()
    rows = []
    ordering = external.assign(_length=external.external_end - external.external_start + 1).sort_values(
        ["_length", "analysis_record_id"], ascending=[False, True]
    )
    for external_row in ordering.itertuples(index=False):
        target_length = int(external_row.external_end - external_row.external_start + 1)
        external_sequence = str(external_row.sequence) if hasattr(external_row, "sequence") else ""
        target_disorder = disorder_fraction(external_sequence)
        available = parent_pool[~parent_pool.region_id.astype(str).isin(used)].copy()
        if available.empty:
            raise ValueError(f"Control pool exhausted while matching {group_name}")
        same_protein = available[available.protein.astype(str).eq(str(external_row.protein))]
        if len(same_protein):
            candidates, scope = same_protein, "same_protein"
        else:
            candidates, scope = available, "other_TF_protein"
        long_enough = candidates[candidates.sequence_length.ge(target_length)]
        if len(long_enough):
            candidates = long_enough
        candidates = candidates.copy()
        candidates["_length_cost"] = np.abs(np.log(candidates.sequence_length / target_length))
        candidates["_disorder_cost"] = np.abs(candidates.disorder_fraction - target_disorder)
        candidates["_cost"] = 2 * candidates._length_cost + candidates._disorder_cost
        chosen = candidates.sort_values(["_cost", "region_id"], kind="stable").iloc[0]
        used.add(str(chosen.region_id))
        start, end, sequence, control_disorder, crop_rule = crop_to_disorder(
            chosen, target_length, target_disorder
        )
        rows.append({
            "analysis_group": group_name,
            "record_type": "matched_non_AD_control",
            "analysis_record_id": f"control|{group_name}|{external_row.analysis_record_id}",
            "paired_analysis_record_id": external_row.analysis_record_id,
            "external_region_id": f"control_for_{external_row.analysis_record_id}",
            "protein": str(chosen.protein),
            "family": getattr(chosen, "family", "other"),
            "external_start": start,
            "external_end": end,
            "region_length": end - start + 1,
            "sequence": sequence,
            "eligible": 1,
            "match_scope": scope,
            "control_parent_region_id": str(chosen.region_id),
            "control_crop_rule": crop_rule,
            "external_length": target_length,
            "control_length": end - start + 1,
            "external_disorder_fraction": target_disorder,
            "control_disorder_fraction": control_disorder,
            "absolute_length_difference": abs((end - start + 1) - target_length),
            "absolute_disorder_difference": abs(control_disorder - target_disorder),
        })
    return pd.DataFrame(rows)


def construct_matched_controls(records: pd.DataFrame, parent_map: pd.DataFrame, detail: pd.DataFrame):
    forbidden = defaultdict(list)
    mapped_external = detail[detail.coordinate_mapping_status.eq("unique_exact_coordinate")]
    for row in mapped_external[["protein", "protein_start", "protein_end"]].itertuples(index=False):
        forbidden[str(row.protein)].append((int(row.protein_start), int(row.protein_end)))

    pool = parent_map[parent_map.coordinate_mapping_status.eq("unique_exact_coordinate")].copy()
    keep = []
    for row in pool[["protein", "protein_start", "protein_end"]].itertuples(index=False):
        keep.append(not intervals_overlap(
            int(row.protein_start), int(row.protein_end), forbidden.get(str(row.protein), [])
        ))
    pool = pool.loc[keep].copy()
    if pool.empty:
        raise ValueError("No non-AD candidate regions remain for matched controls")

    controls = []
    source = records[records.record_type.eq("source_external_region") & records.eligible.eq(1)]
    for dataset in SOURCE_DATASET_NAMES:
        group = source[source.dataset.eq(dataset)]
        controls.append(match_controls_for_group(dataset, group, pool))
    union = records[records.record_type.eq("deduplicated_union_cluster") & records.eligible.eq(1)]
    controls.append(match_controls_for_group(UNION_NAME, union, pool))
    controls.append(match_controls_for_group(MULTI_NAME, union[union.dataset_support_n.ge(2)], pool))
    return pd.concat(controls, ignore_index=True, sort=False), pool


def inventory_predictions(path: Path | None, expected: int) -> dict:
    if path is None:
        return {"prediction_archive_inventory": "not supplied"}
    root, archive, entries = prediction_entries(path)
    try:
        manifest = [parsed for parsed in (parse_model_file(item) for item in entries) if parsed]
    finally:
        if archive is not None:
            archive.close()
    manuscript = [row for row in manifest if row["manuscript_model"]]
    if expected and len(manuscript) != expected:
        raise ValueError(
            f"Expected {expected} manuscript model outputs in {path}, found {len(manuscript)}"
        )
    return {
        "predictions_input": str(path.resolve()),
        "manuscript_prediction_files_detected": len(manuscript),
        "all_prediction_text_files_detected": len(entries),
    }


def load_relevant_window_votes(
    path: Path,
    parent_map: pd.DataFrame,
    selected_proteins: set[str],
    expected_rows: int,
    expected_models: int,
    chunksize: int,
):
    required = {
        "header_id", "protein", "positive_model_count", "total_manuscript_models", "vote_fraction"
    }
    indexed_parent = parent_map.set_index("region_id")
    parent_local_min = indexed_parent.local_min_start.to_dict()
    parent_protein_start = indexed_parent.protein_start.to_dict()
    parent_status = indexed_parent.coordinate_mapping_status.to_dict()
    kept = []
    rows_seen = 0
    model_denominators = set()
    arithmetic_max_error = 0.0
    selected_headers = set()
    for chunk in pd.read_csv(path, chunksize=chunksize):
        if not required.issubset(chunk):
            raise ValueError(f"Window-vote table missing {sorted(required - set(chunk.columns))}")
        rows_seen += len(chunk)
        model_denominators.update(chunk.total_manuscript_models.dropna().astype(int).unique().tolist())
        expected_vote = chunk.positive_model_count / chunk.total_manuscript_models
        arithmetic_max_error = max(
            arithmetic_max_error,
            float((chunk.vote_fraction - expected_vote).abs().max()),
        )
        subset = chunk[chunk.protein.astype(str).isin(selected_proteins)][list(required)].copy()
        if subset.empty:
            continue
        parsed = subset.header_id.astype(str).map(split_window_header)
        subset["parent_region_id"] = parsed.map(lambda item: item[0])
        subset["local_window_start"] = parsed.map(lambda item: item[1]).astype(int)
        subset["parent_coordinate_status"] = subset.parent_region_id.map(parent_status)
        if subset.parent_coordinate_status.isna().any():
            examples = subset.loc[subset.parent_coordinate_status.isna(), "header_id"].head(10).tolist()
            raise ValueError(f"Vote headers lack candidate parent mappings; examples: {examples}")
        subset = subset[
            subset.parent_coordinate_status.eq("unique_exact_coordinate")
        ].copy()
        subset["window_start"] = (
            subset.parent_region_id.map(parent_protein_start).astype(int)
            + subset.local_window_start
            - subset.parent_region_id.map(parent_local_min).astype(int)
        )
        subset["window_end"] = subset.window_start + 14
        subset = subset.drop(columns="parent_coordinate_status")
        duplicate = selected_headers.intersection(subset.header_id.astype(str))
        if duplicate:
            raise ValueError(f"Duplicate selected vote headers, examples: {sorted(duplicate)[:10]}")
        selected_headers.update(subset.header_id.astype(str))
        kept.append(subset)
    if expected_rows and rows_seen != expected_rows:
        raise ValueError(f"Expected {expected_rows:,} recurrence rows, found {rows_seen:,}")
    if model_denominators != {expected_models}:
        raise ValueError(
            f"Expected one denominator ({expected_models}); observed {sorted(model_denominators)}"
        )
    if arithmetic_max_error > 1e-12:
        raise ValueError(f"vote_fraction arithmetic discrepancy: max error {arithmetic_max_error}")
    if not kept:
        raise ValueError("No recurrence windows remained for external/control proteins")
    return pd.concat(kept, ignore_index=True), {
        "complete_window_vote_rows": rows_seen,
        "selected_window_vote_rows": sum(len(item) for item in kept),
        "model_denominators": sorted(model_denominators),
        "vote_arithmetic_max_absolute_error": arithmetic_max_error,
    }


def threshold_column(threshold: float, final: bool = False) -> str:
    if final:
        return "fraction_windows_ge_manuscript_final"
    if threshold == 0:
        return "fraction_windows_gt_0"
    return f"fraction_windows_ge_{threshold:.2f}".replace(".", "_")


def add_recurrence(records: pd.DataFrame, windows: pd.DataFrame, final_threshold: float):
    by_protein = {
        str(protein): group.sort_values(["window_start", "header_id"]).reset_index(drop=True)
        for protein, group in windows.groupby("protein", sort=False)
    }
    rows = []
    for record in records.itertuples(index=False):
        row = record._asdict()
        protein_windows = by_protein.get(str(record.protein), pd.DataFrame())
        if len(protein_windows) and np.isfinite(record.external_start) and np.isfinite(record.external_end):
            overlap = protein_windows[
                protein_windows.window_start.le(int(record.external_end))
                & protein_windows.window_end.ge(int(record.external_start))
            ].copy()
        else:
            overlap = pd.DataFrame()
        votes = overlap.vote_fraction.to_numpy(float) if len(overlap) else np.array([])
        row.update({
            "n_mapped_windows": len(votes),
            "max_vote_fraction": float(np.max(votes)) if len(votes) else np.nan,
            "mean_vote_fraction": float(np.mean(votes)) if len(votes) else np.nan,
            "median_vote_fraction": float(np.median(votes)) if len(votes) else np.nan,
            "max_positive_model_count": int(overlap.positive_model_count.max()) if len(votes) else np.nan,
            "total_applicable_models": int(overlap.total_manuscript_models.iloc[0]) if len(votes) else np.nan,
            "mapped_window_ids": ";".join(overlap.header_id.astype(str)) if len(votes) else "",
        })
        for threshold in DEFAULT_THRESHOLDS:
            comparison = votes > 0 if threshold == 0 else votes >= threshold
            row[threshold_column(threshold)] = float(np.mean(comparison)) if len(votes) else np.nan
        row[threshold_column(final_threshold, final=True)] = (
            float(np.mean(votes >= final_threshold)) if len(votes) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def analysis_groups(external_recurrence: pd.DataFrame):
    source = external_recurrence[
        external_recurrence.record_type.eq("source_external_region")
        & external_recurrence.eligible.eq(1)
    ]
    for dataset in SOURCE_DATASET_NAMES:
        yield dataset, source[source.dataset.eq(dataset)]
    union = external_recurrence[
        external_recurrence.record_type.eq("deduplicated_union_cluster")
        & external_recurrence.eligible.eq(1)
    ]
    yield UNION_NAME, union
    yield MULTI_NAME, union[union.dataset_support_n.ge(2)]


def threshold_specs(final_threshold: float):
    specs = [(">0", 0.0, "gt")]
    specs.extend([(f"{value:.2f}", value, "ge") for value in DEFAULT_THRESHOLDS if value > 0])
    specs.append((f"manuscript final ({final_threshold:.2f})", final_threshold, "ge"))
    return specs


def recovery_curves(external_recurrence, control_recurrence, final_threshold, nonfamily=False):
    rows = []
    for group_name, group in analysis_groups(external_recurrence):
        if nonfamily:
            group = group[group.family.eq("other")]
        for label, threshold, comparator in threshold_specs(final_threshold):
            recognized = group.max_vote_fraction.gt(0) if comparator == "gt" else group.max_vote_fraction.ge(threshold)
            count = int(recognized.sum())
            low, high = wilson(count, len(group))
            rows.append({
                "population": "external_AD",
                "dataset": group_name,
                "threshold_label": label,
                "threshold": threshold,
                "comparison": comparator,
                "N_eligible": len(group),
                "N_recognized": count,
                "recovery_fraction": count / len(group) if len(group) else np.nan,
                "CI_95_low": low,
                "CI_95_high": high,
                "family_filter": "exclude_FOX_TCF_LEF" if nonfamily else "all_families",
            })
    if control_recurrence is not None and not nonfamily:
        for group_name, group in control_recurrence.groupby("analysis_group", sort=False):
            for label, threshold, comparator in threshold_specs(final_threshold):
                recognized = group.max_vote_fraction.gt(0) if comparator == "gt" else group.max_vote_fraction.ge(threshold)
                count = int(recognized.sum())
                low, high = wilson(count, len(group))
                rows.append({
                    "population": "matched_non_AD_control",
                    "dataset": group_name,
                    "threshold_label": label,
                    "threshold": threshold,
                    "comparison": comparator,
                    "N_eligible": len(group),
                    "N_recognized": count,
                    "recovery_fraction": count / len(group) if len(group) else np.nan,
                    "CI_95_low": low,
                    "CI_95_high": high,
                    "family_filter": "all_families",
                })
    return pd.DataFrame(rows)


def matched_control_statistics(external_recurrence, control_recurrence, final_threshold, iterations, seed):
    rng = np.random.default_rng(seed)
    rows = []
    external_lookup = external_recurrence.set_index("analysis_record_id")
    for group_name, controls in control_recurrence.groupby("analysis_group", sort=False):
        external = external_lookup.loc[controls.paired_analysis_record_id].reset_index()
        controls = controls.reset_index(drop=True)
        ext = external.max_vote_fraction.to_numpy(float)
        ctl = controls.max_vote_fraction.to_numpy(float)
        finite = np.isfinite(ext) & np.isfinite(ctl)
        ext, ctl = ext[finite], ctl[finite]
        if len(ext):
            test = mannwhitneyu(ext, ctl, alternative="two-sided")
            delta = cliffs_delta(ext, ctl)
            ci_low, ci_high, valid_bootstraps = bootstrap_cliffs_delta(
                ext, ctl, iterations, rng
            )
        else:
            test = None
            delta = ci_low = ci_high = np.nan
            valid_bootstraps = 0
        base = {
            "dataset": group_name,
            "N_external": len(ext),
            "N_matched_control": len(ctl),
            "external_median_max_vote_fraction": np.median(ext) if len(ext) else np.nan,
            "control_median_max_vote_fraction": np.median(ctl) if len(ctl) else np.nan,
            "external_mean_max_vote_fraction": np.mean(ext) if len(ext) else np.nan,
            "control_mean_max_vote_fraction": np.mean(ctl) if len(ctl) else np.nan,
            "Mann_Whitney_U": float(test.statistic) if test else np.nan,
            "Mann_Whitney_p": float(test.pvalue) if test else np.nan,
            "Cliffs_delta_external_minus_control": delta,
            "Cliffs_delta_CI_95_low": ci_low,
            "Cliffs_delta_CI_95_high": ci_high,
            "valid_bootstraps": valid_bootstraps,
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
        }
        for label, threshold, comparator in [
            item for item in threshold_specs(final_threshold)
            if item[1] in {0.50, 0.75, 0.85, 0.90, final_threshold}
            and item[0] != ">0"
        ]:
            ext_positive = ext >= threshold
            ctl_positive = ctl >= threshold
            table = [
                [int(ext_positive.sum()), int((~ext_positive).sum())],
                [int(ctl_positive.sum()), int((~ctl_positive).sum())],
            ]
            odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
            rows.append({
                **base,
                "threshold_label": label,
                "threshold": threshold,
                "external_N_at_or_above_threshold": table[0][0],
                "control_N_at_or_above_threshold": table[1][0],
                "external_fraction_at_or_above_threshold": np.mean(ext_positive) if len(ext) else np.nan,
                "control_fraction_at_or_above_threshold": np.mean(ctl_positive) if len(ctl) else np.nan,
                "Fisher_odds_ratio": odds_ratio,
                "Fisher_p": p_value,
            })
    result = pd.DataFrame(rows)
    result["Mann_Whitney_p_adj_BH"] = bh(result.Mann_Whitney_p)
    result["Fisher_p_adj_BH"] = bh(result.Fisher_p)
    return result


def miss_stratification(external_recurrence):
    rows = []
    bins = [
        ("no model support", lambda x: x == 0),
        (">0-<25%", lambda x: (x > 0) & (x < .25)),
        ("25-<50%", lambda x: (x >= .25) & (x < .50)),
        ("50-<75%", lambda x: (x >= .50) & (x < .75)),
        ("75-<85%", lambda x: (x >= .75) & (x < .85)),
        ("85-<90%", lambda x: (x >= .85) & (x < .90)),
        (">=90% but final-catalogue absent", lambda x: x >= .90),
    ]
    for group_name, group in analysis_groups(external_recurrence):
        missed = group[group.final_catalogue_recovered.eq(0)]
        values = missed.max_vote_fraction.to_numpy(float)
        for order, (label, rule) in enumerate(bins, start=1):
            count = int(rule(values).sum())
            rows.append({
                "dataset": group_name,
                "stratum_order": order,
                "max_vote_fraction_stratum": label,
                "N_final_catalogue_misses": len(missed),
                "N_in_stratum": count,
                "fraction_of_final_misses": count / len(missed) if len(missed) else np.nan,
            })
    result = pd.DataFrame(rows)
    check = result.groupby("dataset").N_in_stratum.sum()
    denominators = result.groupby("dataset").N_final_catalogue_misses.first()
    if not check.equals(denominators):
        raise ValueError("Final-miss strata do not sum to their denominators")
    return result


def save_figure(fig, outdir: Path, name: str):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(outdir / f"{name}.{extension}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    return values, np.arange(1, len(values) + 1) / len(values)


def recurrence_figure(external, controls, curves, final_threshold, outdir):
    colors = {
        "Soto 2022": "#0072B2",
        "Kotha & Staller 2023 high-confidence": "#D55E00",
        "Kotha & Staller 2023 GSL": "#009E73",
        "Staller 2022": "#CC79A7",
        UNION_NAME: "#000000",
        MULTI_NAME: "#E69F00",
        "Matched controls": "#7F7F7F",
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))

    union = external[
        external.record_type.eq("deduplicated_union_cluster") & external.eligible.eq(1)
    ]
    union_controls = controls[controls.analysis_group.eq(UNION_NAME)]
    for label, values, color in [
        ("External AD union", union.max_vote_fraction, colors[UNION_NAME]),
        ("Matched non-AD", union_controls.max_vote_fraction, colors["Matched controls"]),
    ]:
        x, y = ecdf(values.dropna())
        axes[0].step(x, y, where="post", label=f"{label} (n={len(x)})", color=color, lw=1.1)
    axes[0].set(xlabel="Maximum model vote fraction", ylabel="Cumulative fraction", xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(frameon=False, fontsize=5.8, loc="lower right")

    primary = curves[
        curves.population.eq("external_AD")
        & curves.family_filter.eq("all_families")
        & curves.dataset.isin(SOURCE_DATASET_NAMES + [UNION_NAME])
        & ~curves.threshold_label.str.startswith("manuscript final")
    ]
    for dataset, group in primary.groupby("dataset", sort=False):
        group = group.sort_values("threshold")
        axes[1].plot(group.threshold, group.recovery_fraction, marker="o", ms=2.5,
                     lw=.9, label=dataset.replace("Kotha & Staller 2023 ", "K&S "), color=colors[dataset])
        axes[1].fill_between(
            group.threshold.to_numpy(float), group.CI_95_low.to_numpy(float),
            group.CI_95_high.to_numpy(float), color=colors[dataset], alpha=.08, linewidth=0,
        )
    axes[1].axvline(final_threshold, color="#666666", ls="--", lw=.7)
    axes[1].set(xlabel="Window vote-fraction threshold", ylabel="External AD recovery", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(frameon=False, fontsize=5.2, loc="upper right")

    panel_c = curves[
        curves.family_filter.eq("all_families")
        & curves.dataset.isin([UNION_NAME, MULTI_NAME])
        & ~curves.threshold_label.str.startswith("manuscript final")
    ]
    for (population, dataset), group in panel_c.groupby(["population", "dataset"], sort=False):
        if population == "matched_non_AD_control" and dataset != UNION_NAME:
            continue
        if population == "matched_non_AD_control":
            label = "Matched controls"
        elif dataset == UNION_NAME:
            label = "All external ADs (union)"
        else:
            label = ">=2 datasets"
        group = group.sort_values("threshold")
        plot_color = (
            colors["Matched controls"] if population == "matched_non_AD_control"
            else colors[dataset]
        )
        axes[2].plot(group.threshold, group.recovery_fraction, marker="o", ms=2.5,
                     lw=.9, label=label, color=plot_color)
        axes[2].fill_between(
            group.threshold.to_numpy(float), group.CI_95_low.to_numpy(float),
            group.CI_95_high.to_numpy(float), color=plot_color, alpha=.10, linewidth=0,
        )
    axes[2].axvline(final_threshold, color="#666666", ls="--", lw=.7)
    axes[2].set(xlabel="Window vote-fraction threshold", ylabel="Fraction recognized", xlim=(0, 1), ylim=(0, 1.02))
    axes[2].legend(frameon=False, fontsize=5.2, loc="upper right")

    for label, ax in zip(("A", "B", "C"), axes):
        ax.text(-.18, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, outdir, "Figure_external_AD_ensemble_recurrence")


def original_108_check(path: Path | None, expected_proteins: int, expected_unsupported: int):
    if path is None or not path.is_file():
        return {"status": "not_checked", "reason": "membership matrix not supplied"}
    membership = pd.read_csv(path, sep="\t")
    external_columns = ["Soto_2022", "KothaHC_2023", "KothaGSL_2023", "Staller_2022"]
    required = {"R_TADs", *external_columns}
    if not required.issubset(membership):
        raise ValueError(f"Membership matrix missing {sorted(required - set(membership.columns))}")
    predicted = membership[membership.R_TADs.eq(1)]
    unsupported = int(predicted[external_columns].sum(axis=1).eq(0).sum())
    if expected_proteins and len(predicted) != expected_proteins:
        raise ValueError(f"Original protein result expected {expected_proteins} predicted TFs; found {len(predicted)}")
    if expected_unsupported and unsupported != expected_unsupported:
        raise ValueError(f"Original protein result expected {expected_unsupported} unsupported TFs; found {unsupported}")
    return {
        "status": "verified",
        "predicted_TF_proteins": len(predicted),
        "proteins_without_external_support": unsupported,
        "statistical_unit": "protein; preserved and not redefined by recurrence analysis",
    }


def reviewer_interpretation(curves, stats, nonfamily):
    def curve_value(frame, dataset, label):
        row = frame[
            frame.population.eq("external_AD") & frame.dataset.eq(dataset)
            & frame.threshold_label.eq(label)
        ]
        return row.iloc[0] if len(row) else None

    permissive = curve_value(curves, UNION_NAME, ">0")
    final_label = next(label for label in curves.threshold_label.unique() if label.startswith("manuscript final"))
    stringent = curve_value(curves, UNION_NAME, final_label)
    multi = curve_value(curves, MULTI_NAME, final_label)
    nonfox_final = nonfamily[
        nonfamily.dataset.eq(UNION_NAME) & nonfamily.threshold_label.eq(final_label)
    ]
    comparison = stats[stats.dataset.eq(UNION_NAME)].iloc[0] if len(stats[stats.dataset.eq(UNION_NAME)]) else None

    if comparison is None:
        background_text = "Matched-control recurrence could not be estimated."
    elif comparison.Mann_Whitney_p < .05 and comparison.Cliffs_delta_external_minus_control > 0:
        background_text = (
            "Published AD regions had higher maximum ensemble recurrence than matched non-AD TF-IDR controls "
            f"(Cliff's delta {comparison.Cliffs_delta_external_minus_control:.2f}, "
            f"95% CI {comparison.Cliffs_delta_CI_95_low:.2f} to {comparison.Cliffs_delta_CI_95_high:.2f}; "
            f"two-sided Mann-Whitney p={comparison.Mann_Whitney_p:.3g})."
        )
    else:
        background_text = (
            "Maximum ensemble recurrence was not clearly higher for published AD regions than for matched non-AD "
            f"TF-IDR controls (Cliff's delta {comparison.Cliffs_delta_external_minus_control:.2f}; "
            f"two-sided Mann-Whitney p={comparison.Mann_Whitney_p:.3g})."
        )

    if permissive is None or stringent is None:
        threshold_text = "Threshold-dependent recovery could not be summarized."
    else:
        lost = permissive.recovery_fraction - stringent.recovery_fraction
        if permissive.recovery_fraction < .5:
            threshold_text = (
                f"Only {permissive.N_recognized}/{permissive.N_eligible} deduplicated external regions "
                f"({permissive.recovery_fraction:.1%}) received support from any model, indicating that the ensemble "
                "recognizes a restricted subset of published activation-domain sequence space rather than acting as a "
                "general AD predictor."
            )
        elif lost >= .20:
            threshold_text = (
                f"Recovery decreased from {permissive.recovery_fraction:.1%} at support by any model to "
                f"{stringent.recovery_fraction:.1%} at the manuscript window threshold; this {lost:.1%} difference "
                "shows that stringent consensus removes a substantial fraction of permissively recognized ADs."
            )
        else:
            threshold_text = (
                f"Recovery changed from {permissive.recovery_fraction:.1%} at support by any model to "
                f"{stringent.recovery_fraction:.1%} at the manuscript window threshold, so stringent thresholding "
                "alone does not explain most final-catalogue misses."
            )

    if multi is not None and stringent is not None:
        multi_text = (
            f"At the manuscript window threshold, recovery was {multi.recovery_fraction:.1%} for regions supported "
            f"by at least two datasets versus {stringent.recovery_fraction:.1%} for the deduplicated union."
        )
    else:
        multi_text = "The >=2-dataset subset was too small to summarize."

    if len(nonfox_final) and stringent is not None:
        row = nonfox_final.iloc[0]
        difference = row.recovery_fraction - stringent.recovery_fraction
        family_text = (
            f"After excluding FOX/TCF/LEF, union recovery at the same threshold was "
            f"{row.N_recognized}/{row.N_eligible} ({row.recovery_fraction:.1%}), a change of "
            f"{difference:+.1%}."
        )
    else:
        family_text = "Non-FOX/TCF/LEF sensitivity could not be estimated."

    return (
        background_text + " " + threshold_text + " " + multi_text + " " + family_text + " "
        "These external datasets annotate transcriptional activation domains and are not gold standards for RG/RGG "
        "binding. The window/region recurrence analysis therefore complements, but does not redefine, the original "
        "protein-level Figure 2D result of 108 of 190 predicted TF proteins lacking mapped external support."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="External AD ensemble-recurrence analysis")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--window-votes", type=Path, required=True,
                        help="candidate_vote_fractions_reconstructed.csv from the complete candidate universe")
    parser.add_argument("--candidate-window-fasta", type=Path,
                        help="reference_proteome_dis_and_tfs_15aa.fasta")
    parser.add_argument("--candidate-region-table", type=Path,
                        help="biological_candidate_regions.csv")
    parser.add_argument("--tf-proteome-fasta", type=Path)
    parser.add_argument("--gene-map", type=Path)
    parser.add_argument("--external-fasta-dir", type=Path)
    parser.add_argument("--blast-dir", type=Path)
    parser.add_argument("--external-mapping-detail", type=Path,
                        help="Existing external_AD_mapping_detail.csv (preferred when available)")
    parser.add_argument("--external-union-clusters", type=Path,
                        help="Existing external_AD_union_clusters.csv (preferred when available)")
    parser.add_argument("--predictions", type=Path,
                        help="Optional predictions.zip; filenames are inventoried only, predictions are not recomputed")
    parser.add_argument("--membership-matrix", type=Path)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--minimum-universe-coverage", type=float, default=.80)
    parser.add_argument("--final-consensus-threshold", type=float, default=.85)
    parser.add_argument("--expected-model-count", type=int, default=168)
    parser.add_argument("--expected-window-count", type=int, default=3423499)
    parser.add_argument("--expected-candidate-regions", type=int, default=20486)
    parser.add_argument("--expected-original-proteins", type=int, default=190)
    parser.add_argument("--expected-original-unsupported", type=int, default=108)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260819)
    parser.add_argument("--chunksize", type=int, default=250000)
    args = parser.parse_args()

    root = args.project_root.resolve()
    args.candidate_window_fasta = args.candidate_window_fasta or root / "inputs/fasta/reference_proteome_dis_and_tfs_15aa.fasta"
    args.candidate_region_table = args.candidate_region_table or root / "outputs/reviewer_family_bias/prediction_universe_bias_region_level/biological_candidate_regions.csv"
    args.tf_proteome_fasta = args.tf_proteome_fasta or root / "inputs/fasta/tf_modified_headers.fasta"
    args.gene_map = args.gene_map or root / "inputs/metadata/UNIPROTIDS_GENENAME.txt"
    args.external_fasta_dir = args.external_fasta_dir or root / "inputs/fasta"
    args.blast_dir = args.blast_dir or root / "outputs/blast"
    args.membership_matrix = args.membership_matrix or root / "outputs/blast/venn_RTADs_ADdatasets_fromBLAST_membership_matrix.tsv"
    args.outdir = args.outdir or root / "outputs/reviewer_family_bias/external_AD_ensemble_recurrence"
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.external_mapping_detail is None:
        default = root / "outputs/reviewer_external_AD_recovery/external_AD_mapping_detail.csv"
        args.external_mapping_detail = default if default.is_file() else None
    if args.external_union_clusters is None:
        default = root / "outputs/reviewer_external_AD_recovery/external_AD_union_clusters.csv"
        args.external_union_clusters = default if default.is_file() else None

    required_paths = [
        args.window_votes, args.candidate_window_fasta, args.candidate_region_table,
        args.tf_proteome_fasta, args.gene_map,
    ]
    if args.external_mapping_detail is None:
        required_paths.extend([args.external_fasta_dir / item[1] for item in DATASETS])
        required_paths.extend([args.blast_dir / item[2] for item in DATASETS])
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be >=1")
    if not 0 <= args.final_consensus_threshold <= 1:
        raise ValueError("--final-consensus-threshold must be between 0 and 1")

    print("[1/8] Reading candidate-region table and TF reference", flush=True)
    candidate_regions = pd.read_csv(args.candidate_region_table)
    if candidate_regions.region_id.duplicated().any():
        raise ValueError("Candidate-region table contains duplicate region_id values")
    if args.expected_candidate_regions and len(candidate_regions) != args.expected_candidate_regions:
        raise ValueError(
            f"Expected {args.expected_candidate_regions:,} biological candidate regions, "
            f"found {len(candidate_regions):,}"
        )
    references = fasta_dict(args.tf_proteome_fasta)
    genes = read_gene_map(args.gene_map)
    print("[2/8] Reconstructing biological candidate regions from the complete window FASTA", flush=True)
    reconstructed, fasta_window_count = reconstruct_parent_regions(
        args.candidate_window_fasta, candidate_regions
    )
    if args.expected_window_count and fasta_window_count != args.expected_window_count:
        raise ValueError(
            f"Expected {args.expected_window_count:,} FASTA windows, found {fasta_window_count:,}"
        )
    print("[3/8] Mapping candidate regions and loading the existing external-AD mappings", flush=True)
    parent_map, parent_statuses = map_parent_regions(candidate_regions, reconstructed, references)
    detail, clusters, mapping_source = load_or_create_external_mapping(
        args, references, genes, parent_map
    )
    records = normalize_analysis_records(detail, clusters)
    print("[4/8] Constructing non-overlapping matched TF-IDR controls", flush=True)
    controls, control_pool = construct_matched_controls(records, parent_map, detail)

    selected_proteins = set(records.loc[records.eligible.eq(1), "protein"].astype(str))
    selected_proteins.update(controls.protein.astype(str))
    print("[5/8] Reading and validating the complete-universe recurrence table", flush=True)
    windows, recurrence_qa = load_relevant_window_votes(
        args.window_votes,
        parent_map,
        selected_proteins,
        args.expected_window_count,
        args.expected_model_count,
        args.chunksize,
    )
    print("[6/8] Calculating external-AD and matched-control recurrence", flush=True)
    external_recurrence = add_recurrence(records, windows, args.final_consensus_threshold)
    control_recurrence = add_recurrence(controls, windows, args.final_consensus_threshold)
    eligible_missing = external_recurrence.eligible.eq(1) & external_recurrence.n_mapped_windows.eq(0)
    if eligible_missing.any():
        examples = external_recurrence.loc[
            eligible_missing, ["dataset", "external_region_id", "protein"]
        ].head(10)
        raise ValueError("Eligible external regions without mapped windows:\n" + examples.to_string(index=False))
    if control_recurrence.n_mapped_windows.eq(0).any():
        examples = control_recurrence.loc[
            control_recurrence.n_mapped_windows.eq(0), ["analysis_group", "external_region_id", "protein"]
        ].head(10)
        raise ValueError("Matched controls without recurrence windows:\n" + examples.to_string(index=False))

    print("[7/8] Running threshold, matched-control, miss-stratification, and family-sensitivity analyses", flush=True)
    curves = recovery_curves(
        external_recurrence, control_recurrence, args.final_consensus_threshold
    )
    nonfamily = recovery_curves(
        external_recurrence, None, args.final_consensus_threshold, nonfamily=True
    )
    statistics = matched_control_statistics(
        external_recurrence, control_recurrence, args.final_consensus_threshold,
        args.bootstrap, args.bootstrap_seed,
    )
    misses = miss_stratification(external_recurrence)
    original_result = original_108_check(
        args.membership_matrix,
        args.expected_original_proteins,
        args.expected_original_unsupported,
    )
    prediction_inventory = inventory_predictions(args.predictions, args.expected_model_count)

    print("[8/8] Writing tables, QA, interpretation, and PDF/SVG/PNG figure", flush=True)
    primary_columns = [
        "dataset", "external_region_id", "protein", "gene", "family", "record_type",
        "dataset_support_n", "external_start", "external_end", "region_length", "eligible",
        "exclusion_reason", "n_mapped_windows", "max_positive_model_count",
        "total_applicable_models", "max_vote_fraction", "mean_vote_fraction",
        "median_vote_fraction", "fraction_windows_gt_0", "fraction_windows_ge_0_25",
        "fraction_windows_ge_0_50", "fraction_windows_ge_0_75", "fraction_windows_ge_0_85",
        "fraction_windows_ge_0_90", "fraction_windows_ge_manuscript_final",
        "final_catalogue_recovered", "mapped_window_ids", "analysis_record_id",
    ]
    for column in primary_columns:
        if column not in external_recurrence:
            external_recurrence[column] = np.nan
    external_recurrence[primary_columns].to_csv(
        args.outdir / "external_AD_window_recurrence.csv", index=False
    )
    curves.to_csv(
        args.outdir / "external_AD_recovery_by_consensus_threshold.csv", index=False
    )
    statistics.to_csv(
        args.outdir / "external_AD_vs_matched_control_recurrence.csv", index=False
    )
    control_recurrence.to_csv(
        args.outdir / "external_AD_matched_control_pairs.csv", index=False
    )
    misses.to_csv(
        args.outdir / "external_AD_final_miss_stratification.csv", index=False
    )
    nonfamily.to_csv(
        args.outdir / "external_AD_recurrence_nonFOX_sensitivity.csv", index=False
    )
    recurrence_figure(
        external_recurrence, control_recurrence, curves,
        args.final_consensus_threshold, args.outdir,
    )

    interpretation = reviewer_interpretation(curves, statistics, nonfamily)
    (args.outdir / "reviewer_interpretation.txt").write_text(interpretation + "\n")
    qa = {
        "analysis": "external activation-domain ensemble recurrence",
        "no_model_loading": True,
        "no_retraining": True,
        "no_score_recovery_rerun": True,
        "window_vote_input": str(args.window_votes.resolve()),
        "complete_candidate_window_fasta": str(args.candidate_window_fasta.resolve()),
        "candidate_region_table": str(args.candidate_region_table.resolve()),
        "external_mapping_source": mapping_source,
        "candidate_window_statistical_role": "recurrence measurements within external biological regions",
        "primary_external_statistical_unit": "eligible external AD region; coordinate clusters for union",
        "primary_recurrence_metric": "maximum vote fraction among overlapping 15-aa candidate windows",
        "threshold_rule": ">0 is strict; all numeric thresholds use >=",
        "final_consensus_threshold": args.final_consensus_threshold,
        "external_eligibility_minimum_candidate_coverage": args.minimum_universe_coverage,
        "candidate_FASTA_window_count": fasta_window_count,
        "biological_candidate_region_count": len(candidate_regions),
        "candidate_parent_mapping_statuses": parent_statuses,
        "non_AD_control_pool_regions": len(control_pool),
        "control_rule": "same protein where available; otherwise another TF; closest length/disorder; no overlap with any mapped external annotation; one control per external record within analysis group",
        "cliffs_delta_sign": "positive = external AD maximum recurrence exceeds matched control",
        "bootstrap": {"iterations": args.bootstrap, "seed": args.bootstrap_seed, "method": "matched-set resampling"},
        "recurrence_cross_checks": recurrence_qa,
        "prediction_archive_cross_check": prediction_inventory,
        "original_Figure_2D_protein_result": original_result,
        "software_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": mpl.__version__,
        },
    }
    (args.outdir / "external_AD_recurrence_metadata.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n"
    )
    qa_lines = [
        "External AD ensemble-recurrence QA",
        "==================================",
        f"window-vote input: {args.window_votes.resolve()}",
        f"complete candidate FASTA: {args.candidate_window_fasta.resolve()}",
        f"candidate region table: {args.candidate_region_table.resolve()}",
        f"external mapping: {mapping_source}",
        f"complete FASTA windows: {fasta_window_count:,}",
        f"complete recurrence rows: {recurrence_qa['complete_window_vote_rows']:,}",
        f"biological candidate regions: {len(candidate_regions):,}",
        f"model denominator(s): {recurrence_qa['model_denominators']}",
        f"vote arithmetic max absolute error: {recurrence_qa['vote_arithmetic_max_absolute_error']}",
        f"eligible source external regions: {int((external_recurrence.record_type.eq('source_external_region') & external_recurrence.eligible.eq(1)).sum()):,}",
        f"eligible deduplicated union regions: {int((external_recurrence.record_type.eq('deduplicated_union_cluster') & external_recurrence.eligible.eq(1)).sum()):,}",
        f"matched controls: {len(control_recurrence):,}",
        f"bootstrap iterations: {args.bootstrap:,}",
        f"bootstrap seed: {args.bootstrap_seed}",
        f"original Figure 2D result check: {json.dumps(original_result, sort_keys=True)}",
        "",
        "The original 108/190 result is protein-level and is not recalculated from window/region recurrence.",
        "Final-catalogue recovery is inherited from the existing same-protein external mapping.",
        "External AD datasets are not treated as RG/RGG-binding gold standards.",
    ]
    (args.outdir / "external_AD_recurrence_QA.txt").write_text("\n".join(qa_lines) + "\n")
    print(f"Complete. Outputs: {args.outdir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
