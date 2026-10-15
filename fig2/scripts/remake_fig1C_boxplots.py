#!/usr/bin/env python3
"""Remake corrected Figure 1C as old-style two-group raw-value boxplots.

The plotted observations and Hedges' g point estimates are unchanged.  The
displayed confidence intervals are the parent-region cluster-bootstrap
intervals from ``Fig1C_cluster_bootstrap_statistics.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_START = "Starting AHD windows"
GROUP_CONTROL = "Matched TF-IDR controls"
GROUPS = [GROUP_START, GROUP_CONTROL]
FEATURES = [
    ("isoelectric_point", "Isoelectric point"),
    ("disorder_promoting_fraction", "Disorder-promoting fraction"),
    ("aliphatic_fraction", "Aliphatic fraction"),
    ("SCD", "SCD"),
]

COLOR_BLUE = "#0072B2"
COLOR_ORANGE = "#E69F00"
COLOR_BLACK = "#000000"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.bbox": "tight",
})


def clean_numeric(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return array[np.isfinite(array)]


def draw_boxplot(ax: plt.Axes, start: np.ndarray, control: np.ndarray,
                 ylabel: str, g_value: float, ci_low: float, ci_high: float) -> None:
    groups = [start, control]
    boxes = ax.boxplot(
        groups,
        labels=["Starting rAHD\nwindows", "Matched TF-IDR\ncontrols"],
        patch_artist=True,
        widths=0.10,
        whis=1.5,
        showfliers=False,
        medianprops={"linewidth": 0.8, "color": COLOR_BLACK},
        boxprops={"linewidth": 0.6},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
    )
    for patch, color in zip(boxes["boxes"], [COLOR_BLUE, COLOR_ORANGE]):
        patch.set_facecolor(color)
        patch.set_edgecolor(COLOR_BLACK)

    rng = np.random.default_rng(123)
    for index, (values, color) in enumerate(zip(groups, [COLOR_BLUE, COLOR_ORANGE]), start=1):
        x = np.clip(rng.normal(loc=index, scale=0.03, size=len(values)), index - 0.08, index + 0.08)
        ax.scatter(x, values, s=5.5, alpha=0.30, edgecolors="none", color=color, zorder=2)

    ymin = min(values.min() for values in groups)
    ymax = max(values.max() for values in groups)
    span = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.28 * span)
    ax.text(
        0.5, 0.96, f"g = {g_value:.2f} [{ci_low:.2f}, {ci_high:.2f}]",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.0,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.88},
        zorder=4,
    )
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build(project_root: Path, output: Path, statistics_path: Path | None = None,
          overwrite: bool = False) -> list[Path]:
    """Build Figure 1C and return the generated files."""
    root = project_root.resolve()
    output = output.resolve()
    raw_path = root / "fig1C_corrected" / "raw_feature_values.csv"
    statistics_path = statistics_path or root / "Fig1C_cluster_bootstrap_statistics.csv"
    statistics_path = statistics_path.resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    statistics = pd.read_csv(statistics_path).set_index("feature")
    if set(raw["group"].dropna()) != set(GROUPS):
        raise ValueError(f"Unexpected groups in {raw_path}: {sorted(raw['group'].dropna().unique())}")

    source_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(4.8, 2.3))
    for ax, (feature, label) in zip(axes.flat, FEATURES):
        if feature not in raw.columns or feature not in statistics.index:
            raise KeyError(f"Required corrected feature is missing: {feature}")
        row = statistics.loc[feature]
        start = clean_numeric(raw.loc[raw.group.eq(GROUP_START), feature])
        control = clean_numeric(raw.loc[raw.group.eq(GROUP_CONTROL), feature])
        if len(start) != int(row.n_windows_AHD) or len(control) != int(row.n_windows_control):
            raise ValueError(f"Corrected sample-size mismatch for {feature}")
        draw_boxplot(
            ax, start, control, label,
            float(row.window_level_hedges_g),
            float(row.cluster_bootstrap_CI_low),
            float(row.cluster_bootstrap_CI_high),
        )
        for group in GROUPS:
            subset = raw.loc[raw.group.eq(group), ["sequence_id", "sequence", feature]].copy()
            subset = subset.rename(columns={feature: "raw_value"})
            display_group = "Starting rAHD windows" if group == GROUP_START else group
            for record in subset.itertuples(index=False):
                source_rows.append({
                    "feature": feature,
                    "feature_label": label,
                    "group": display_group,
                    "sequence_id": record.sequence_id,
                    "sequence": record.sequence,
                    "raw_value": record.raw_value,
                    "hedges_g_rAHD_minus_control": row.window_level_hedges_g,
                    "cluster_bootstrap_CI_low": row.cluster_bootstrap_CI_low,
                    "cluster_bootstrap_CI_high": row.cluster_bootstrap_CI_high,
                    "n_parent_regions_rAHD": row.n_parent_regions_AHD,
                    "n_parent_regions_control": row.n_parent_regions_control,
                })

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.97, wspace=0.48, hspace=0.58)
    stem = output / "Fig1C_boxplots_corrected"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)

    source_path = output / "Fig1C_boxplots_source_data.csv"
    pd.DataFrame(source_rows).to_csv(source_path, index=False)

    print("Exact input files used:")
    print(f"  {raw_path}")
    print(f"  {statistics_path}")
    print("Features and corrected cluster-bootstrap effect-size intervals:")
    for feature, label in FEATURES:
        row = statistics.loc[feature]
        print(
            f"  {label}: g = {row.window_level_hedges_g:.2f} "
            f"[{row.cluster_bootstrap_CI_low:.2f}, {row.cluster_bootstrap_CI_high:.2f}]"
        )
    generated = [stem.with_suffix(".png"), stem.with_suffix(".pdf"),
                 stem.with_suffix(".svg"), source_path]
    print("Output files:")
    for path in generated:
        print(f"  {path}")
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--statistics", type=Path,
        help="Cluster-bootstrap statistics CSV (default: project-root/Fig1C_cluster_bootstrap_statistics.csv)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir or root / "fig1C_boxplot_final"
    build(root, output, statistics_path=args.statistics, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
