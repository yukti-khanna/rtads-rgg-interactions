#!/usr/bin/env python3
"""Cluster-bootstrap Figure 1C Hedges' g confidence intervals by parent region."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_AHD = "Starting AHD windows"
GROUP_CONTROL = "Matched TF-IDR controls"
FEATURES = [
    ("isoelectric_point", "Isoelectric point"),
    ("disorder_promoting_fraction", "Disorder-promoting fraction"),
    ("aliphatic_fraction", "Aliphatic fraction"),
    ("SCD", "Sequence charge decoration (SCD)"),
]
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20261005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Fig1C_cluster_bootstrap_statistics.csv"),
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fasta_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                ids.add(line[1:].strip().split()[0])
    if not ids:
        raise ValueError(f"No FASTA identifiers found in {path}")
    return ids


def recover_parent_id(sequence_id: str) -> str:
    """Remove only the final colon-delimited 15-aa window-position suffix."""
    match = re.fullmatch(r"(.+):(\d+)", str(sequence_id))
    if not match:
        raise ValueError(f"Cannot recover parent ID from sequence identifier: {sequence_id}")
    return match.group(1)


def hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Signed Hedges' g: Starting AHD minus matched TF-IDR control."""
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    group_a = group_a[np.isfinite(group_a)]
    group_b = group_b[np.isfinite(group_b)]
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return np.nan
    df = n_a + n_b - 2
    pooled_variance = (
        (n_a - 1) * np.var(group_a, ddof=1)
        + (n_b - 1) * np.var(group_b, ddof=1)
    ) / df
    if not np.isfinite(pooled_variance) or pooled_variance <= 0:
        return np.nan
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return correction * (np.mean(group_a) - np.mean(group_b)) / np.sqrt(pooled_variance)


def parent_clusters(frame: pd.DataFrame, feature: str) -> dict[str, np.ndarray]:
    clusters: dict[str, np.ndarray] = {}
    for parent_id, group in frame.groupby("parent_id", sort=True):
        values = group[feature].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            clusters[str(parent_id)] = values
    return clusters


def cluster_bootstrap_ci(
    ahd_clusters: dict[str, np.ndarray],
    control_clusters: dict[str, np.ndarray],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    ahd_ids = np.asarray(sorted(ahd_clusters), dtype=object)
    control_ids = np.asarray(sorted(control_clusters), dtype=object)
    bootstrap_g = np.empty(iterations, dtype=float)

    for index in range(iterations):
        sampled_ahd = rng.choice(ahd_ids, size=len(ahd_ids), replace=True)
        sampled_control = rng.choice(control_ids, size=len(control_ids), replace=True)
        # Repeatedly sampled parents contribute their complete window cluster repeatedly.
        ahd_values = np.concatenate([ahd_clusters[parent] for parent in sampled_ahd])
        control_values = np.concatenate(
            [control_clusters[parent] for parent in sampled_control]
        )
        bootstrap_g[index] = hedges_g(ahd_values, control_values)

    valid = bootstrap_g[np.isfinite(bootstrap_g)]
    if len(valid) != iterations:
        raise ValueError(
            f"Only {len(valid)} of {iterations} cluster-bootstrap replicates were finite"
        )
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high), len(valid)


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    raw_path = root / "fig1C_corrected" / "raw_feature_values.csv"
    stats_path = root / "fig1C_corrected" / "all_feature_statistics.csv"
    ahd_fasta = root / "inputs" / "fasta" / "seed_tads_15aa.fasta"
    control_fasta = (
        root / "inputs" / "fasta" / "random_lenmatched_from_allIDRs_seed_tads_15aa.fasta"
    )
    output_path = args.output if args.output.is_absolute() else root / args.output

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    if args.bootstrap_iterations < 10_000:
        raise ValueError("At least 10,000 bootstrap iterations are required")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    stats = pd.read_csv(stats_path).set_index("feature")
    required = {"group", "sequence_id", *(feature for feature, _ in FEATURES)}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Corrected raw table is missing columns: {sorted(missing)}")
    detected_groups = set(raw["group"].dropna().unique())
    if detected_groups != {GROUP_AHD, GROUP_CONTROL}:
        raise ValueError(f"Unexpected groups: {sorted(detected_groups)}")

    raw = raw.copy()
    raw["parent_id"] = raw["sequence_id"].map(recover_parent_id)

    # Confirm that the corrected observations are the exact sequence IDs in their FASTAs.
    expected_fasta = {
        GROUP_AHD: fasta_ids(ahd_fasta),
        GROUP_CONTROL: fasta_ids(control_fasta),
    }
    for group, frame in raw.groupby("group", sort=False):
        observed = set(frame["sequence_id"].astype(str))
        missing_from_fasta = observed.difference(expected_fasta[group])
        if missing_from_fasta:
            raise ValueError(
                f"{group}: corrected raw IDs absent from FASTA: {sorted(missing_from_fasta)}"
            )

    # Required pre-bootstrap parent grouping report.
    print("Unique parent IDs detected before bootstrap:")
    for group in (GROUP_AHD, GROUP_CONTROL):
        parents = sorted(raw.loc[raw["group"].eq(group), "parent_id"].unique())
        print(f"\n{group} (n={len(parents)}):")
        print("\n".join(f"  {parent}" for parent in parents))
    print("\nBeginning cluster bootstrap after parent-ID report.\n")

    rng = np.random.default_rng(args.seed)
    results: list[dict[str, float | int | str]] = []
    for feature, label in FEATURES:
        ahd_frame = raw.loc[raw["group"].eq(GROUP_AHD), ["parent_id", feature]]
        control_frame = raw.loc[
            raw["group"].eq(GROUP_CONTROL), ["parent_id", feature]
        ]
        ahd_values = ahd_frame[feature].dropna().to_numpy(dtype=float)
        control_values = control_frame[feature].dropna().to_numpy(dtype=float)
        ahd_clusters = parent_clusters(ahd_frame, feature)
        control_clusters = parent_clusters(control_frame, feature)

        window_g = hedges_g(ahd_values, control_values)
        original_g = float(stats.loc[feature, "hedges_g_starting_minus_control"])
        if not np.isclose(window_g, original_g, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{feature}: calculated window-level g={window_g} does not match "
                f"corrected statistic g={original_g}"
            )

        ci_low, ci_high, valid = cluster_bootstrap_ci(
            ahd_clusters,
            control_clusters,
            args.bootstrap_iterations,
            rng,
        )
        ahd_parent_means = np.asarray(
            [np.mean(values) for values in ahd_clusters.values()], dtype=float
        )
        control_parent_means = np.asarray(
            [np.mean(values) for values in control_clusters.values()], dtype=float
        )
        results.append(
            {
                "feature": feature,
                "feature_label": label,
                "window_level_hedges_g": original_g,
                "cluster_bootstrap_CI_low": ci_low,
                "cluster_bootstrap_CI_high": ci_high,
                "n_windows_AHD": len(ahd_values),
                "n_windows_control": len(control_values),
                "n_parent_regions_AHD": len(ahd_clusters),
                "n_parent_regions_control": len(control_clusters),
                "parent_level_hedges_g": hedges_g(
                    ahd_parent_means, control_parent_means
                ),
                "parent_level_mean_AHD": float(np.mean(ahd_parent_means)),
                "parent_level_mean_control": float(np.mean(control_parent_means)),
                "cluster_bootstrap_valid_iterations": valid,
                "bootstrap_seed": args.seed,
            }
        )

    result = pd.DataFrame(results)
    result.to_csv(output_path, index=False)

    print("Cluster-bootstrap results:")
    for row in result.itertuples(index=False):
        print(
            f"  {row.feature_label}: g = {row.window_level_hedges_g:.4f} "
            f"[{row.cluster_bootstrap_CI_low:.4f}, "
            f"{row.cluster_bootstrap_CI_high:.4f}]"
        )
    print(f"\nOutput: {output_path}")


if __name__ == "__main__":
    main()
