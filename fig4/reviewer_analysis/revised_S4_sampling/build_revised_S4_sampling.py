#!/usr/bin/env python3
"""Build revised Supplementary Figure S4 sampling/provenance panels.

Run from:
    /Users/yuktikhanna/rtads-rgg-interactions/fig4

All outputs are written beside this script in:
    reviewer_analysis/revised_S4_sampling
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PANELS = OUT / "panels"

PILOT_25 = ROOT / "25_pairs_features_data.csv"
INITIAL_SET = ROOT / "initial_set_features_data.csv"
TASK3 = ROOT / "task3_sampling_bias"
FULL = TASK3 / "full_combinatorial_feature_landscape.csv"
FINAL_917 = TASK3 / "fig4_917_sampling_membership.csv"
INDEPENDENT = TASK3 / "heldout_23x78_features.csv"
TASK3_DISTRIBUTIONS = TASK3 / "feature_distribution_comparison_three_sets.csv"

OCN = "Opposite_Charge_Number"
Y = "B22_corr1_log10"
FEATURES = [OCN, "Total_Length", "Average_lambda", "charge_TAD", "charge_RGG", "Average_NCPR"]
FINAL_FIGSIZE = (11.2, 5.8)
PANEL_FIGSIZE = (3.0, 2.25)
DATASET_LABELS = {
    "full candidate landscape": "complete candidate landscape",
    "final 917": "feature-guided simulation panel",
    "independent 1794": "independent sequence-exclusive simulation set",
}
COLORS = {
    "complete candidate landscape": "#7a7a7a",
    "feature-guided simulation panel": "#0072B2",
    "independent sequence-exclusive simulation set": "#D55E00",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
    }
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def require_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def pearson(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 2 or tmp["x"].std() == 0 or tmp["y"].std() == 0:
        return np.nan, len(tmp)
    return float(tmp["x"].corr(tmp["y"], method="pearson")), len(tmp)


def pooled_smd(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return float((b.mean() - a.mean()) / pooled)


def save_panel(fig: mpl.figure.Figure, name: str) -> None:
    for folder in (OUT, PANELS):
        fig.savefig(folder / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
        fig.savefig(folder / f"{name}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)


def scatter_panel(ax: plt.Axes, df: pd.DataFrame, xcol: str, label: str, title: str, panel: str) -> None:
    require_columns(df, ["name", xcol, Y], title)
    r, n = pearson(df[xcol], df[Y])
    ax.scatter(df[xcol], df[Y], s=14, alpha=0.65, color="#0072B2", edgecolors="none")
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[Y], errors="coerce")
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 2 and x[ok].std() > 0:
        m, b = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(float(x[ok].min()), float(x[ok].max()), 100)
        ax.plot(xx, m * xx + b, color="#222222", lw=1.0)
    ax.set_xlabel(label)
    ax.set_ylabel(r"$log_{10}(-B_{22})$")
    ax.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
    ax.text(0.03, 0.97, f"n={n}, r={r:.2f}", transform=ax.transAxes, ha="left", va="top")


def plot_old_panels(pilot: pd.DataFrame, initial: pd.DataFrame) -> dict[str, dict[str, object]]:
    panels = {
        "S4A": (pilot, "Charge_Complementarity", "Charge complementarity", "Pilot charge complementarity"),
        "S4B": (pilot, "Average_NCPR", "Mean pair NCPR", "Pilot mean pair NCPR"),
        "S4C": (pilot, "Average_lambda", "Average lambda", "Pilot average lambda"),
        "S4D": (initial, OCN, r"Net-charge product, $-Q_{RGG}\times Q_{TAD}$", "Early net-charge-product\nrelationship"),
    }
    audit: dict[str, dict[str, object]] = {}
    for panel, (df, xcol, xlabel, title) in panels.items():
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        scatter_panel(ax, df, xcol, xlabel, title, panel)
        fig.tight_layout()
        save_panel(fig, panel)
        plt.close(fig)
        audit[panel] = {
            "x_variable": xcol,
            "y_variable": Y,
            "sample_size": int(df[[xcol, Y]].dropna().shape[0]),
            "source_file": str(PILOT_25 if panel in {"S4A", "S4B", "S4C"} else INITIAL_SET),
            "script_function": f"plot_old_panels/{panel}",
            "recommendation": "retain with relabelling as sampling-design/provenance analysis",
        }
    return audit


def summarize_dataset(label: str, df: pd.DataFrame) -> dict[str, object]:
    require_columns(df, ["tad", "rgg", OCN, "Total_Length", "Average_lambda"], label)
    return {
        "dataset_name": label,
        "n_pairs": int(len(df)),
        "n_unique_TADs": int(df["tad"].nunique()),
        "n_unique_RGGs": int(df["rgg"].nunique()),
        "mean_OCN": df[OCN].mean(),
        "median_OCN": df[OCN].median(),
        "SD_OCN": df[OCN].std(ddof=1),
        "mean_total_length": df["Total_Length"].mean(),
        "median_total_length": df["Total_Length"].median(),
        "mean_average_lambda": df["Average_lambda"].mean(),
        "median_average_lambda": df["Average_lambda"].median(),
    }


def ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def distribution_panel(ax: plt.Axes, datasets: dict[str, pd.DataFrame], feature: str, xlabel: str, title: str, panel: str) -> None:
    for label, df in datasets.items():
        x, y = ecdf(df[feature])
        ax.plot(x, y, lw=1.4, color=COLORS[label], label=f"{label} (n={len(df):,})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title(f"{panel}  {title}", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#dddddd", lw=0.45)


def plot_distribution_panels(datasets: dict[str, pd.DataFrame]) -> None:
    panels = [
        ("S4E", OCN, "Net-charge product", "Net-charge product distribution"),
        ("S4F", "Total_Length", "Total pair length", "Total length distribution"),
        ("S4G", "Average_lambda", "Average lambda", "Average lambda distribution"),
    ]
    for panel, feature, xlabel, title in panels:
        fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
        distribution_panel(ax, datasets, feature, xlabel, title, panel)
        ax.legend(frameon=False, loc="best", bbox_to_anchor=(0.0, -0.33, 1.0, 0.1), mode="expand", ncol=1, borderaxespad=0)
        fig.tight_layout()
        save_panel(fig, panel)
        plt.close(fig)


def bin_coverage_panel(ax1: plt.Axes, coverage: pd.DataFrame, panel: str) -> None:
    centers = (coverage["bin_lower"] + coverage["bin_upper"]) / 2.0
    widths = coverage["bin_upper"] - coverage["bin_lower"]
    ax1.bar(centers, coverage["candidate_count"], width=widths * 0.85, color="#c9c9c9", label="candidate pairs")
    ax1.set_yscale("log")
    ax1.set_xlabel("Net-charge product")
    ax1.set_ylabel("Candidate pairs, log scale")
    ax2 = ax1.twinx()
    ax2.plot(centers, coverage["selected_917_count"], color="#0072B2", marker="o", ms=3, lw=1.1, label="selected 917")
    ax2.set_ylabel("Selected pairs")
    ax1.set_title(f"{panel}  Reconstructed net-charge-product\nbin coverage", loc="left", fontweight="bold")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1._combined_legend_handles = handles1 + handles2
    ax1._combined_legend_labels = labels1 + labels2
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.33, 1.0, 0.1),
        mode="expand",
        ncol=1,
        borderaxespad=0,
    )


def plot_ocn_bin_coverage(full: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(np.floor(full[OCN].min() / 5) * 5, np.ceil(full[OCN].max() / 5) * 5 + 5, 5)
    if bins[0] > full[OCN].min():
        bins = np.insert(bins, 0, full[OCN].min())
    full_bin = pd.cut(full[OCN], bins=bins, include_lowest=True, right=False)
    final_bin = pd.cut(final[OCN], bins=bins, include_lowest=True, right=False)
    candidate_counts = full_bin.value_counts().sort_index()
    selected_counts = final_bin.value_counts().sort_index()
    rows = []
    for i, interval in enumerate(candidate_counts.index, start=1):
        cand = int(candidate_counts.loc[interval])
        sel = int(selected_counts.get(interval, 0))
        rows.append(
            {
                "bin_id": i,
                "bin_lower": float(interval.left),
                "bin_upper": float(interval.right),
                "candidate_count": cand,
                "selected_917_count": sel,
                "selection_fraction": sel / cand if cand else np.nan,
                "historical_or_reconstructed_flag": "reconstructed from final pair list; historical per-round bin manifest not found",
            }
        )
    coverage = pd.DataFrame(rows)
    coverage.to_csv(OUT / "S4_OCN_bin_coverage.csv", index=False)

    fig, ax1 = plt.subplots(figsize=PANEL_FIGSIZE)
    bin_coverage_panel(ax1, coverage, "S4H")
    fig.tight_layout()
    save_panel(fig, "S4H")
    plt.close(fig)
    return coverage


def write_effect_sizes(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    full = datasets["complete candidate landscape"]
    comparisons = [
        ("complete candidate landscape", "feature-guided simulation panel"),
        ("complete candidate landscape", "independent sequence-exclusive simulation set"),
    ]
    rows = []
    for base_label, other_label in comparisons:
        base = full if base_label == "complete candidate landscape" else datasets[base_label]
        other = datasets[other_label]
        for feature in [OCN, "Total_Length", "Average_lambda"]:
            rows.append(
                {
                    "feature": feature,
                    "comparison": f"{base_label} vs {other_label}",
                    "standardized_mean_difference": pooled_smd(base[feature], other[feature]),
                    "base_mean": base[feature].mean(),
                    "comparison_mean": other[feature].mean(),
                }
            )
    out = pd.DataFrame(rows)
    if TASK3_DISTRIBUTIONS.exists():
        task3 = read_csv(TASK3_DISTRIBUTIONS)
        mapping = {
            "complete candidate landscape vs feature-guided simulation panel": "Full landscape vs 917 panel",
            "complete candidate landscape vs independent sequence-exclusive simulation set": "Full landscape vs Held-out 1,794",
        }
        for comparison, task3_comparison in mapping.items():
            for feature in [OCN, "Total_Length", "Average_lambda"]:
                hit = task3.loc[
                    (task3["record_type"] == "pairwise_distance")
                    & (task3["feature"] == feature)
                    & (task3["comparison"] == task3_comparison),
                    "standardized_mean_difference",
                ]
                if not hit.empty:
                    out.loc[
                        (out["comparison"] == comparison) & (out["feature"] == feature),
                        "standardized_mean_difference",
                    ] = float(hit.iloc[0])
        out["effect_size_source"] = "Task 3 where available; recomputed fallback otherwise"
    out.to_csv(OUT / "S4_distribution_effect_sizes.csv", index=False)
    return out


def write_audit(audit: dict[str, dict[str, object]]) -> None:
    lines = [
        "# Existing S4 Panel Audit",
        "",
        "Historical panels were verified from executable code/data in `supp4.py`, `25_pairs_features_data.csv`, and `initial_set_features_data.csv`.",
        "The old panels are retained as sampling-design/provenance analyses rather than independent biological validation.",
        "",
        "| panel | purpose | x variable | y variable | n | source file | script/function | recommendation |",
        "|---|---|---|---|---:|---|---|---|",
    ]
    purposes = {
        "S4A": "Pilot-stage relationship between B22 and charge complementarity.",
        "S4B": "Pilot-stage relationship between B22 and mean pair NCPR.",
        "S4C": "Pilot-stage relationship between B22 and average lambda.",
        "S4D": "Early simulated relationship between B22 and net-charge product/OCN that motivated later OCN-bin sampling.",
    }
    for panel in ["S4A", "S4B", "S4C", "S4D"]:
        row = audit[panel]
        lines.append(
            f"| {panel} | {purposes[panel]} | {row['x_variable']} | {row['y_variable']} | "
            f"{row['sample_size']} | {row['source_file']} | {row['script_function']} | {row['recommendation']} |"
        )
    (OUT / "existing_S4_panel_audit.md").write_text("\n".join(lines) + "\n")


def write_captions(summary: pd.DataFrame, effects: pd.DataFrame, audit: dict[str, dict[str, object]]) -> None:
    get_n = dict(zip(summary["dataset_name"], summary["n_pairs"]))
    ocn_smd_917 = effects.loc[
        (effects["feature"] == OCN)
        & effects["comparison"].str.contains("feature-guided", regex=False),
        "standardized_mean_difference",
    ].iloc[0]
    ocn_smd_ind = effects.loc[
        (effects["feature"] == OCN)
        & effects["comparison"].str.contains("independent", regex=False),
        "standardized_mean_difference",
    ].iloc[0]
    length_smd_917 = effects.loc[
        (effects["feature"] == "Total_Length")
        & effects["comparison"].str.contains("feature-guided", regex=False),
        "standardized_mean_difference",
    ].iloc[0]
    lambda_smd_917 = effects.loc[
        (effects["feature"] == "Average_lambda")
        & effects["comparison"].str.contains("feature-guided", regex=False),
        "standardized_mean_difference",
    ].iloc[0]
    text = f"""# S4 Panel Caption Draft

S4A. Pilot-stage B22 versus charge complementarity for the 25-pair pilot set (n=25). This panel documents an early feature relationship considered during sampling design. It does not represent an independent validation test.

S4B. Pilot-stage B22 versus mean pair NCPR for the 25-pair pilot set (n=25). This panel documents charge-composition space explored during dataset construction. It does not establish unbiased discovery performance.

S4C. Pilot-stage B22 versus average lambda for the 25-pair pilot set (n=25). This panel documents the hydropathy/stickiness region considered during the early sampling design. It does not test independent generalisation.

S4D. Early simulated B22 versus net-charge product, equivalent to the implemented OCN value (-Q_RGG x Q_TAD), for the initial simulation set (n={audit['S4D']['sample_size']:,}). This relationship motivated use of net-charge product as a later-stage sampling coordinate. It does not imply that the final 917 pairs were randomly selected or sampled by known B22.

S4E. Empirical net-charge-product distributions for the complete candidate landscape (n={get_n['complete candidate landscape']:,}), feature-guided simulation panel (n={get_n['feature-guided simulation panel']:,}), and independent sequence-exclusive simulation set (n={get_n['independent sequence-exclusive simulation set']:,}). The feature-guided panel is modestly shifted relative to the full candidate space (SMD={ocn_smd_917:.3f}), whereas the independent set is closer to the full-space distribution (SMD={ocn_smd_ind:.3f}). This panel describes sampling coverage, not a biological validation result.

S4F. Empirical total-pair-length distributions for the complete candidate landscape, feature-guided simulation panel, and independent sequence-exclusive simulation set. The final 917-pair panel is modestly shifted relative to the full candidate landscape (SMD={length_smd_917:.3f}). This panel describes sampling composition, not independent validation.

S4G. Empirical average-lambda distributions for the complete candidate landscape, feature-guided simulation panel, and independent sequence-exclusive simulation set. The final 917-pair panel is close to the full candidate landscape for average lambda (SMD={lambda_smd_917:.3f}). This panel describes feature-space coverage, not independent validation.

S4H. Reconstructed net-charge-product bin coverage for the complete candidate landscape and final 917-pair feature-guided simulation panel. Historical per-round bin manifests were not found, so bin counts are reconstructed from the retained final pair list and full candidate landscape. The panel shows coverage across the OCN range and should not be interpreted as per-round historical selection counts.
"""
    (OUT / "S4_panel_caption_draft.md").write_text(text)


def final_multipanel(
    pilot: pd.DataFrame,
    initial: pd.DataFrame,
    datasets: dict[str, pd.DataFrame],
    coverage: pd.DataFrame,
) -> str:
    fig, axes = plt.subplots(2, 4, figsize=FINAL_FIGSIZE)
    scatter_panel(axes[0, 0], pilot, "Charge_Complementarity", "Charge complementarity", "Pilot charge complementarity", "S4A")
    scatter_panel(axes[0, 1], pilot, "Average_NCPR", "Mean pair NCPR", "Pilot mean pair NCPR", "S4B")
    scatter_panel(axes[0, 2], pilot, "Average_lambda", "Average lambda", "Pilot average lambda", "S4C")
    scatter_panel(axes[0, 3], initial, OCN, r"Net-charge product, $-Q_{RGG}\times Q_{TAD}$", "Early net-charge-product\nrelationship", "S4D")
    distribution_panel(axes[1, 0], datasets, OCN, "Net-charge product", "Net-charge product distribution", "S4E")
    distribution_panel(axes[1, 1], datasets, "Total_Length", "Total pair length", "Total length distribution", "S4F")
    distribution_panel(axes[1, 2], datasets, "Average_lambda", "Average lambda", "Average lambda distribution", "S4G")
    bin_coverage_panel(axes[1, 3], coverage, "S4H")

    dataset_handles, dataset_labels = axes[1, 0].get_legend_handles_labels()
    bin_handles = getattr(axes[1, 3], "_combined_legend_handles", axes[1, 3].get_legend_handles_labels()[0])
    bin_labels = getattr(axes[1, 3], "_combined_legend_labels", axes[1, 3].get_legend_handles_labels()[1])
    for ax in (axes[1, 0], axes[1, 1], axes[1, 2], axes[1, 3]):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.legend(
        dataset_handles,
        dataset_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.38, 0.055),
        ncol=3,
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.legend(
        bin_handles,
        bin_labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.83, 0.055),
        ncol=2,
        columnspacing=1.0,
        handlelength=1.5,
    )

    fig.subplots_adjust(left=0.055, right=0.955, bottom=0.25, top=0.93, wspace=0.50, hspace=0.78)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bbox = fig.bbox
    clipped = False
    axis_text_artists = []
    for ax in fig.axes:
        axis_text_artists.extend([ax.xaxis.label, ax.yaxis.label])
        axis_text_artists.extend(ax.get_xticklabels())
        axis_text_artists.extend(ax.get_yticklabels())
    for artist in axis_text_artists:
        if not artist.get_visible():
            continue
        try:
            bbox = artist.get_window_extent(renderer=renderer)
        except Exception:
            continue
        if bbox.width == 0 or bbox.height == 0:
            continue
        if bbox.x0 < fig_bbox.x0 - 1 or bbox.y0 < fig_bbox.y0 - 1 or bbox.x1 > fig_bbox.x1 + 1 or bbox.y1 > fig_bbox.y1 + 1:
            clipped = True
            break
    fig.savefig(OUT / "revised_Figure_S4_2x4.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "revised_Figure_S4_2x4.pdf", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return "detected before tight export" if clipped else "not detected before tight export"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)

    pilot = read_csv(PILOT_25)
    initial = read_csv(INITIAL_SET)
    full = read_csv(FULL)
    final = read_csv(FINAL_917)
    independent = read_csv(INDEPENDENT)

    datasets = {
        "complete candidate landscape": full,
        "feature-guided simulation panel": final,
        "independent sequence-exclusive simulation set": independent,
    }

    audit = plot_old_panels(pilot, initial)
    write_audit(audit)

    summary = pd.DataFrame([summarize_dataset(label, df) for label, df in datasets.items()])
    summary.to_csv(OUT / "S4_sampling_summary.csv", index=False)

    effects = write_effect_sizes(datasets)
    plot_distribution_panels(datasets)
    coverage = plot_ocn_bin_coverage(full, final)
    write_captions(summary, effects, audit)
    clipping_status = final_multipanel(pilot, initial, datasets, coverage)

    if TASK3_DISTRIBUTIONS.exists():
        task3_stats = read_csv(TASK3_DISTRIBUTIONS)
        task3_stats.to_csv(OUT / "Task3_feature_distribution_comparison_three_sets_reference.csv", index=False)

    print("Final panel order: S4A | S4B | S4C | S4D / S4E | S4F | S4G | S4H")
    print(f"Final figure dimensions: {FINAL_FIGSIZE[0]} x {FINAL_FIGSIZE[1]} inches")
    print(f"Axis-label clipping: {clipping_status}")
    print("S4E/F/G separate full-size panels: yes")
    print(f"Output directory: {OUT}")
    print("Main figure: revised_Figure_S4_2x4.png and revised_Figure_S4_2x4.pdf")
    print("Panels: S4A-S4H png/pdf")
    print("Tables: S4_sampling_summary.csv, S4_OCN_bin_coverage.csv, S4_distribution_effect_sizes.csv")
    print("Text: existing_S4_panel_audit.md, S4_panel_caption_draft.md")


if __name__ == "__main__":
    main()
