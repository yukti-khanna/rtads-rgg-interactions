#!/usr/bin/env python3
"""Restore revised Figure 4D using the historical fig4.py panel_d visual design.

Run locally from this directory:
    python build_Fig4D_final_restored.py

This D-only script writes only:
    Fig4D_final_restored.png
    Fig4D_final_restored.pdf

It does not rerun CALVADOS, does not alter finite-size corrections, and does
not overwrite any other Figure 4 or Supplementary Figure S4 panels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
TASK3 = ROOT / "task3_sampling_bias"
ORIGINAL_917_TABLE = ROOT / "sim_pairs_added_features_data_270525.csv"
HIST_CACHE_CANDIDATES = [
    ROOT / "out_fig4" / "cache" / "Fig4d_scatter_table.csv",
    ROOT / "out_fig4" / "Fig4d_scatter_table.csv",
    ROOT / "Fig4d_scatter_table.csv",
]

OCN_COL = "Opposite_Charge_Number"
Y_COL = "B22_corr1_log10"
LAMBDA_COL = "lambda_TAD"
ID_COL = "name"

AXIS_LABEL_FONTSIZE = 6.5
AXIS_TITLE_FONTSIZE = 7.0
TICK_LABEL_FONTSIZE = 6.0
LEGEND_FONTSIZE = 6.0
FIGSIZE_PANEL = (3.6, 2.3)

AXES_LINEWIDTH = 0.6
TICK_WIDTH = 0.5
LINE_WIDTH = 0.8

MARKER_SIZE = 10
SCATTER_ALPHA = 0.6
SCATTER_LINEWIDTHS = 0.0
SCATTER_CMAP = "viridis"
REGRESSION_LINEWIDTH = 0.9
REGRESSION_ALPHA = 0.85
ANNOTATION_FONTSIZE = 5.4
PNG_DPI = 600
RASTERIZE_HEAVY_ARTISTS = True


def apply_mpl_style() -> None:
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    mpl.rcParams["font.size"] = TICK_LABEL_FONTSIZE
    mpl.rcParams["axes.linewidth"] = AXES_LINEWIDTH
    mpl.rcParams["xtick.major.width"] = TICK_WIDTH
    mpl.rcParams["ytick.major.width"] = TICK_WIDTH
    mpl.rcParams["lines.linewidth"] = LINE_WIDTH
    mpl.rcParams["savefig.dpi"] = PNG_DPI
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def style_axes(ax: plt.Axes, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    for sp in ax.spines.values():
        sp.set_linewidth(AXES_LINEWIDTH)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE, width=TICK_WIDTH)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def style_colorbar(cbar: mpl.colorbar.Colorbar, label: str | None = None) -> None:
    if label:
        cbar.set_label(label, fontsize=LEGEND_FONTSIZE)
    cbar.ax.tick_params(labelsize=TICK_LABEL_FONTSIZE, width=TICK_WIDTH)
    cbar.ax.yaxis.get_offset_text().set_size(TICK_LABEL_FONTSIZE)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def load_current_plotting_table() -> pd.DataFrame:
    path = TASK3 / "fig4_917_sampling_membership.csv"
    require(path)
    current = pd.read_csv(path, low_memory=False)
    required = {"pair_id", OCN_COL, Y_COL}
    missing = sorted(required.difference(current.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    require(ORIGINAL_917_TABLE)
    original = pd.read_csv(ORIGINAL_917_TABLE, low_memory=False)
    original_required = {ID_COL, OCN_COL, Y_COL, LAMBDA_COL}
    missing = sorted(original_required.difference(original.columns))
    if missing:
        raise ValueError(f"Missing required columns in {ORIGINAL_917_TABLE}: {missing}")

    current = current[["pair_id", OCN_COL, Y_COL]].rename(columns={"pair_id": ID_COL})
    original = original[[ID_COL, OCN_COL, LAMBDA_COL, Y_COL]]
    merged = current.merge(original, on=ID_COL, suffixes=("_current", "_historical_source"))

    if len(merged) != len(current):
        raise ValueError(
            "Could not match every current 917 pair to the original plotting source table. "
            f"current n={len(current)}, matching pair IDs={len(merged)}"
        )

    max_dx = float(np.max(np.abs(merged[f"{OCN_COL}_current"] - merged[f"{OCN_COL}_historical_source"])))
    max_dy = float(np.max(np.abs(merged[f"{Y_COL}_current"] - merged[f"{Y_COL}_historical_source"])))
    tolerance = 1e-12
    if max_dx > tolerance or max_dy > tolerance:
        raise ValueError(
            "Current final 917 table differs from original plotting source; stopping before plotting. "
            f"matching pair IDs={len(merged)}, max |Δx|={max_dx}, max |Δy|={max_dy}"
        )

    table = pd.DataFrame(
        {
            ID_COL: merged[ID_COL],
            OCN_COL: merged[f"{OCN_COL}_current"],
            LAMBDA_COL: merged[LAMBDA_COL],
            Y_COL: merged[f"{Y_COL}_current"],
        }
    ).replace([np.inf, -np.inf], np.nan).dropna()
    table = table.sort_values(ID_COL).reset_index(drop=True)
    if len(table) != 917:
        raise ValueError(f"Expected 917 plotted pairs; found {len(table)} in {path}")
    return table


def find_historical_cache() -> Path | None:
    for path in HIST_CACHE_CANDIDATES:
        if path.exists():
            return path
    return None


def compare_with_historical_cache(current: pd.DataFrame) -> dict[str, str | int | float]:
    hist_path = find_historical_cache()
    if hist_path is None:
        return {
            "historical_cache": "not found",
            "matching_pair_ids": 0,
            "max_abs_diff_x": "not compared",
            "max_abs_diff_y": "not compared",
            "max_abs_diff_lambda_TAD": "not compared",
        }

    hist = pd.read_csv(hist_path, low_memory=False)
    required = {ID_COL, OCN_COL, Y_COL, LAMBDA_COL}
    missing = sorted(required.difference(hist.columns))
    if missing:
        raise ValueError(f"Missing required columns in historical cache {hist_path}: {missing}")

    hist = hist[[ID_COL, OCN_COL, LAMBDA_COL, Y_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    merged = current.merge(hist, on=ID_COL, suffixes=("_current", "_historical"))
    if len(merged) != len(current) or len(merged) != len(hist):
        raise ValueError(
            "Historical-vs-current plotting table mismatch: "
            f"current n={len(current)}, historical n={len(hist)}, matching pair IDs={len(merged)}"
        )

    max_dx = float(np.max(np.abs(merged[f"{OCN_COL}_current"] - merged[f"{OCN_COL}_historical"])))
    max_dy = float(np.max(np.abs(merged[f"{Y_COL}_current"] - merged[f"{Y_COL}_historical"])))
    max_dl = float(np.max(np.abs(merged[f"{LAMBDA_COL}_current"] - merged[f"{LAMBDA_COL}_historical"])))

    tolerance = 1e-12
    if max_dx > tolerance or max_dy > tolerance or max_dl > tolerance:
        raise ValueError(
            "Historical-vs-current plotting table values differ; stopping before plotting. "
            f"matching pair IDs={len(merged)}, max |Δx|={max_dx}, "
            f"max |Δy|={max_dy}, max |Δlambda_TAD|={max_dl}"
        )

    return {
        "historical_cache": str(hist_path),
        "matching_pair_ids": int(len(merged)),
        "max_abs_diff_x": max_dx,
        "max_abs_diff_y": max_dy,
        "max_abs_diff_lambda_TAD": max_dl,
    }


def build_panel_d_restored(table: pd.DataFrame) -> tuple[int, float]:
    x = table[OCN_COL].to_numpy(dtype=float)
    y = table[Y_COL].to_numpy(dtype=float)
    c = table[LAMBDA_COL].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=FIGSIZE_PANEL)
    sc = ax.scatter(
        x,
        y,
        c=c,
        s=MARKER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=SCATTER_LINEWIDTHS,
        cmap=SCATTER_CMAP,
        rasterized=RASTERIZE_HEAVY_ARTISTS,
    )
    style_axes(
        ax,
        xlabel="Negative net-charge product (−NCP)",
        ylabel=r"$\log_{10}\!\left(-B_{22}\right)$",
    )

    slope, intercept, pearson_r, _p, _stderr = stats.linregress(x, y)
    xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ax.plot(xx, intercept + slope * xx, color="black", lw=REGRESSION_LINEWIDTH, alpha=REGRESSION_ALPHA)
    ax.text(
        0.02,
        0.98,
        f"Pearson r = {pearson_r:.2f}\n(n = {x.size})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(sc, cax=cax)
    style_colorbar(cbar, label=r"$\lambda_{\mathrm{TAD}}$")

    fig.savefig(OUT / "Fig4D_final_restored.png", dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "Fig4D_final_restored.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return int(x.size), float(pearson_r)


def main() -> None:
    apply_mpl_style()
    table = load_current_plotting_table()
    comparison = compare_with_historical_cache(table)
    n_plotted, pearson_r = build_panel_d_restored(table)

    print("Figure 4D restored")
    print(f"n plotted: {n_plotted}")
    print(f"marker size: {MARKER_SIZE}")
    print(f"alpha: {SCATTER_ALPHA}")
    print(f"colormap: {SCATTER_CMAP}")
    print("lambda_TAD colorbar restored: yes")
    print(f"figure size: {FIGSIZE_PANEL}")
    print(f"axis-label fontsize: {AXIS_LABEL_FONTSIZE}")
    print(f"tick fontsize: {TICK_LABEL_FONTSIZE}")
    print(f"annotation fontsize: {ANNOTATION_FONTSIZE}")
    print(f"regression linewidth: {REGRESSION_LINEWIDTH}")
    print(f"Pearson r: {pearson_r:.6f}")
    print("x-axis label: Negative net-charge product (−NCP)")
    print("historical-vs-current plotting-table comparison")
    print(f"historical cache: {comparison['historical_cache']}")
    print(f"number of matching pair IDs: {comparison['matching_pair_ids']}")
    print(f"max absolute difference in x: {comparison['max_abs_diff_x']}")
    print(f"max absolute difference in y: {comparison['max_abs_diff_y']}")
    print(f"max absolute difference in lambda_TAD: {comparison['max_abs_diff_lambda_TAD']}")
    print(f"PNG dpi: {PNG_DPI}")
    print(f"output directory: {OUT}")


if __name__ == "__main__":
    main()
