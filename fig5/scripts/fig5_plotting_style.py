#!/usr/bin/env python3
"""Original manuscript scatter aesthetics for revised Figure 5 plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

COLOR_BLUE = "#0072B2"
COLOR_GREY = "#656565"
FIGSIZE = (2.4, 2.3)
DPI_PNG = 600
RASTERIZE_SCATTER = True
RASTERIZE_ZORDER = 1
RASTERIZE_DPI = 600
RNG_SEED = 12345
MAX_BG_POINTS_F = 100_000


def identity_limits(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return 0.0, 1.0
    data = np.concatenate([x[ok], y[ok]])
    dmin, dmax = float(np.min(data)), float(np.max(data))
    if dmin == dmax:
        dmin -= 0.5
        dmax += 0.5
    pad = 0.05 * (dmax - dmin)
    return dmin - pad, dmax + pad


def finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], ok


def pearsonr_safe(x, y):
    x, y, _ = finite_xy(x, y)
    if len(x) < 2 or np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return np.nan, len(x)
    return float(pd.Series(x).corr(pd.Series(y), method="pearson")), len(x)


def clean_figure_axes(fig):
    for ax in fig.axes:
        try:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        except Exception:
            pass


def plot_manuscript_scatter(
    ax,
    x,
    y,
    *,
    xlabel,
    ylabel,
    label=None,
    annotation=None,
    identity=True,
    equal_aspect=True,
    axis_limits=None,
    colour=COLOR_BLUE,
    background_mask=None,
    background_label="Background",
    foreground_label=None,
    reference_vline=None,
    reference_hline=None,
    annotation_xy=(0.97, 0.03),
    annotation_ha="right",
    annotation_va="bottom",
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if background_mask is not None:
        mask = np.asarray(background_mask, dtype=bool)[ok]
        ax.scatter(
            x[~mask], y[~mask], s=7, alpha=0.50, c=COLOR_GREY, edgecolors="none",
            label=background_label, rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER,
        )
        ax.scatter(
            x[mask], y[mask], s=7, alpha=0.80, c=colour, edgecolors="none",
            label=foreground_label or label, rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER,
        )
    else:
        ax.scatter(
            x, y, s=7, alpha=0.80, c=colour, edgecolors="none", label=label,
            rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER,
        )
    if axis_limits is None and identity:
        axis_limits = identity_limits(x, y)
    if identity and axis_limits is not None:
        ax.plot(axis_limits, axis_limits, lw=0.8, color="black", zorder=3)
    if reference_vline is not None:
        ax.axvline(reference_vline, color="black", lw=0.8, ls="--", zorder=3)
    if reference_hline is not None:
        ax.axhline(reference_hline, color="black", lw=0.8, zorder=3)
    if axis_limits is not None:
        ax.set_xlim(axis_limits)
        if identity or equal_aspect:
            ax.set_ylim(axis_limits)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if annotation:
        ax.text(annotation_xy[0], annotation_xy[1], annotation, ha=annotation_ha, va=annotation_va, transform=ax.transAxes)
    if label or background_mask is not None:
        ax.legend(frameon=False, loc="best", handletextpad=0.6, borderpad=0.3)


def plot_manuscript_landscape(
    ax,
    df,
    *,
    x_col,
    y_col,
    xlabel,
    ylabel,
    highlight_col=None,
    fit_line=True,
):
    plot_df = df[[x_col, y_col] + ([highlight_col] if highlight_col and highlight_col in df.columns else [])].dropna().copy()
    if highlight_col and highlight_col in plot_df.columns:
        is_highlight = plot_df[highlight_col].astype(bool).to_numpy()
    else:
        is_highlight = np.zeros(len(plot_df), dtype=bool)
    rng = np.random.default_rng(RNG_SEED)
    idx = np.arange(len(plot_df))
    bg = idx[~is_highlight]
    fg = idx[is_highlight]
    if len(bg) > MAX_BG_POINTS_F:
        bg = rng.choice(bg, size=MAX_BG_POINTS_F, replace=False)
    keep = np.zeros(len(plot_df), dtype=bool)
    keep[bg] = True
    keep[fg] = True
    p = plot_df.loc[keep]
    mask = is_highlight[keep]
    ax.scatter(
        p.loc[~mask, x_col], p.loc[~mask, y_col], s=7, alpha=0.50, c=COLOR_GREY,
        edgecolors="none", label="All pairs", rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER,
    )
    if mask.any():
        ax.scatter(
            p.loc[mask, x_col], p.loc[mask, y_col], s=7, alpha=0.80, c=COLOR_BLUE,
            edgecolors="none", label="RBP", rasterized=RASTERIZE_SCATTER, zorder=RASTERIZE_ZORDER,
        )
    if fit_line and len(plot_df) > 2:
        x = plot_df[x_col].to_numpy(float)
        y = plot_df[y_col].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() > 2 and not np.allclose(np.std(x[good]), 0):
            m, b = np.polyfit(x[good], y[good], 1)
            xa, xb = np.nanmin(x[good]), np.nanmax(x[good])
            ax.plot([xa, xb], [m * xa + b, m * xb + b], lw=1.0, color="black", zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best", handletextpad=0.6, borderpad=0.3)


def write_scatter_style_provenance(path: Path):
    text = """Scatter style provenance

Source script path:
/Users/yuktikhanna/rtads-rgg-interactions/fig5/fig5.py

Plotting functions/helpers reused:
identity_limits, clean_figure_axes, save_fig aesthetics, panel_C/panel_D predicted-vs-simulated scatter settings, panel_F full-landscape scatter settings.

Marker type:
matplotlib Axes.scatter circular marker default.

Marker size:
s=7 for manuscript scatter points.

Alpha:
0.50 for grey/background series; 0.80 for blue/highlight or primary series.

Edge/face handling:
edgecolors=\"none\"; face colour #656565 grey or #0072B2 Okabe-Ito blue.

Density colouring:
No density colour scale in the original manuscript scatter implementation. Large landscape uses downsampled/rasterised scatter rather than hexbin.

Identity-line styling:
black line, lw=0.8, zorder=3.

Regression-line styling:
landscape/scaling panels use a black linear-fit line, lw=1.0.

Axis limits:
Predicted-vs-simulated plots use shared identity_limits with 5% padding; x and y limits matched.

Tick formatting:
Matplotlib defaults with rcParams xtick/ytick label size 6 and outward ticks.

Spine widths:
axes.linewidth=0.6; top and right spines hidden by clean_figure_axes.

Font sizes:
font.size=6.5; axes labels=7; axes titles=7; tick labels=6; legend=6.

Metric annotation placement:
Old predicted-vs-simulated panel uses lower-right axes coordinates x=0.97, y=0.03, ha=right, va=bottom.

Rasterisation settings:
rasterized=True for scatter; zorder=1; save dpi/rasterization dpi=600.

Colourbar treatment:
No colourbar for manuscript scatter or full-landscape scatter panels.
"""
    path.write_text(text)
