#!/usr/bin/env python3
# file: make_fig6_panels_final_v7.py

"""
Fig 6 Panels (single-panel exports, 3.6×2.3 in each) following Yukti's Nature-style guide.

Focus: make Panel 6B and 6C clear, with proper labels + legend.
- Panel 6B: Predicted y separated by NMR outcome (Kd-fit / LB / NB / No Kd)
- Panel 6C: Predicted y vs experimental pKd for Kd-fit points (PPT Kd ± err),
            with partner marker legend + optional THOC4-only fit line.

Inputs:
- Table with PPT-extracted TITAN Kd and error + outcomes:
    /mnt/data/Table_S8_NMR_TITAN_summary_template_v3.csv

Exports:
- Fig6B_pred_vs_outcome.(pdf|png)
- Fig6C_pred_vs_pKd.(pdf|png)

Also writes:
- data/Fig6B_points.csv
- data/Fig6C_points.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# Style guide (Yukti × Nature-style)
# =========================
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
    "xtick.direction": "out",
    "ytick.direction": "out",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# =========================
# Constants / I/O
# =========================
FIGSIZE = (3.6, 2.3)  # keep fixed per user

DPI_PNG = 600
SAVE_TIGHT = False  # keep true physical size

OUTDIR = Path("data/Fig6_panels_final")
DATADIR = OUTDIR / "data"

TABLE_S8 = Path("Table_S8.csv")

# Okabe–Ito palette (subset)
C_BLUE   = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN  = "#009E73"
C_VERM   = "#D55E00"
C_PURPLE = "#CC79A7"
C_GREY   = "#B0B0B0"
C_BLACK  = "#000000"

# Panel labels
LBL_X_PRED = r"Predicted $y=\log_{10}(-B_{22})$"

LBL_Y_PKD  = r"Experimental $pK_d=-\log_{10}(K_d[\mathrm{M}])$"
LBL_Y_OUT  = "NMR Outcome"

# Partner markers (for Panel 6C)
PARTNER_MARKER = {
    "THOC4 N-term": "o",
    "THOC4 C-term": "^",
    "G3BP1": "s",
    "TAF15": "D",
}

# For small n, keep one clear semantic mapping by partner in Panel 6C
PARTNER_COLOR = {
    "THOC4 N-term": C_BLUE,
    "THOC4 C-term": C_BLUE,
    "G3BP1": C_GREY,
    "TAF15": C_GREY,
}

TAD_DISPLAY = {
    "atf4_tad": "ATF4",
    "foxo4_cr3": "FOXO4",
    "foxo6_cr3": "FOXO6",
    "tfe3_cter": "TFE3",
    "lef1_nter": "LEF1",
}

# =========================
# Utilities
# =========================
def ensure_dirs() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    DATADIR.mkdir(parents=True, exist_ok=True)

def _new_fig_ax() -> tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.06, h_pad=0.06, wspace=0.0, hspace=0.0)
    return fig, ax

def save_fig(fig: mpl.figure.Figure, stem: str) -> None:
    pdf = OUTDIR / f"{stem}.pdf"
    png = OUTDIR / f"{stem}.png"
    svg = OUTDIR / f"{stem}.svg"
    if SAVE_TIGHT:
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(png, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.01)
    else:
        fig.savefig(pdf)
        fig.savefig(png, dpi=DPI_PNG)
        fig.savefig(svg)
    plt.close(fig)

def clean_spines(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def canonical_outcome(x: object) -> str:
    if not isinstance(x, str):
        return "Not tested"
    s = x.strip().lower().replace("_", " ").replace("-", " ")
    if s in ("kd fit", "kdfit"):
        return "Kd Fit"
    if s in ("lb", "line broadening"):
        return "Line Broadening"
    if s in ("nb", "no binding", "nobinding"):
        return "No Binding"
    if s in ("no kd", "no kd "):
        return "No Kd"
    if "no kd" in s:
        return "No Kd"
    return x.strip()

def kd_uM_to_pKd(kd_uM: float) -> float:
    if not np.isfinite(kd_uM) or kd_uM <= 0:
        return np.nan
    return float(6.0 - np.log10(kd_uM))

def kd_err_to_pKd_err(kd_uM: float, kd_err_uM: float) -> float:
    """Symmetric pKd error via propagation: σ_pKd ≈ (σ_Kd / Kd) / ln(10)."""
    if not (np.isfinite(kd_uM) and np.isfinite(kd_err_uM)) or kd_uM <= 0 or kd_err_uM <= 0:
        return np.nan
    return float((kd_err_uM / kd_uM) / np.log(10.0))

def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 2:
        return (np.nan, n)
    xs, ys = x[m], y[m]
    if np.allclose(np.std(xs), 0) or np.allclose(np.std(ys), 0):
        return (np.nan, n)
    return float(np.corrcoef(xs, ys)[0, 1]), n

def spearmanr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 2:
        return (np.nan, n)
    xs = pd.Series(x[m]).rank(method="average").to_numpy()
    ys = pd.Series(y[m]).rank(method="average").to_numpy()
    if np.allclose(np.std(xs), 0) or np.allclose(np.std(ys), 0):
        return (np.nan, n)
    return float(np.corrcoef(xs, ys)[0, 1]), n

def linear_fit_params(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return (np.nan, np.nan)
    xs, ys = x[m], y[m]
    if np.allclose(np.std(xs), 0) or np.allclose(np.std(ys), 0):
        return (np.nan, np.nan)
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)

def load_table() -> pd.DataFrame:
    df = pd.read_csv(TABLE_S8, low_memory=False)
    df["TAD"] = df["R_TAD"].map(TAD_DISPLAY).fillna(df["R_TAD"])
    df["Outcome_clean"] = df["Outcome"].apply(canonical_outcome)
    # ensure numeric
    df["Pred_y"] = pd.to_numeric(df["Pred_log10_negB22_hybrid"], errors="coerce")
    df["Kd_uM"] = pd.to_numeric(df["Kd_uM"], errors="coerce")
    df["Kd_uM_err"] = pd.to_numeric(df["Kd_uM_err"], errors="coerce")
    # recompute pKd from PPT Kd (ignore any prefilled pKd column)
    df["pKd_PPT"] = df["Kd_uM"].apply(kd_uM_to_pKd)
    df["pKd_err_PPT"] = [kd_err_to_pKd_err(k, e) for k, e in zip(df["Kd_uM"], df["Kd_uM_err"])]
    return df

# =========================
# Panel 6B — Predicted y vs outcome (clear legend)
# =========================
def panel_6B(df: pd.DataFrame) -> None:
    order = ["Kd Fit", "Line Broadening", "No Binding", "No Kd"]
    # keep only outcomes present
    present = [o for o in order if o in df["Outcome_clean"].unique()]
    # Remove "No Kd" category from this panel (user request)
    present = [o for o in present if o != "No Kd"]
    # counts
    counts = {o: int((df["Outcome_clean"] == o).sum()) for o in present}
    ytick = [f"{o} (n={counts[o]})" for o in present]

    outcome_color = {
        "Kd Fit": C_BLUE,
        "Line Broadening": C_VERM,
        "No Binding": C_GREY,
        "No Kd": C_BLACK,
    }

    fig, ax = _new_fig_ax()
    clean_spines(ax)

    rng = np.random.default_rng(1234)
    jitter = 0.08

    # scatter as stripplot: x=Pred_y, y=category index
    rows = []
    for yi, outcome in enumerate(present):
        sub = df[df["Outcome_clean"] == outcome].copy()
        x = sub["Pred_y"].to_numpy(float)
        y = yi + rng.uniform(-jitter, jitter, size=len(sub))
        c = outcome_color.get(outcome, C_BLACK)

        ax.scatter(
            x, y,
            s=22, alpha=0.9,
            c=c,
            edgecolors="none" if outcome != "No Kd" else C_BLACK,
            label=outcome,
        )

        for _, r in sub.iterrows():
            rows.append({
                "TAD": r["TAD"],
                "partner_display": r["partner_display"],
                "pair_id": r["pair_id"],
                "Outcome": outcome,
                "Pred_y": r["Pred_y"],
            })

    # Axes
    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels(ytick)
    ax.set_xlabel(LBL_X_PRED)
    ax.set_ylabel(LBL_Y_OUT)

    # Make y categories read top->bottom as stronger->weaker outcomes
    ax.invert_yaxis()

    # Light x-grid for readability (thin, not distracting)
    ax.xaxis.grid(True, which="major", linewidth=0.4, color="#D3D3D3")
    ax.set_axisbelow(True)

    # Legend (outcomes)
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=4.8,
               markerfacecolor=outcome_color[o], markeredgecolor=C_BLACK if o == "No Kd" else outcome_color[o])
        for o in present
    ]
    leg = ax.legend(
        handles, present,
        loc="center right",
        bbox_to_anchor=(0.98, 0.50),   # right edge, vertical center
        frameon=False,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.3,
    )
    if leg is not None:
        leg.set_frame_on(False)

    # Optional subtitle inside (kept short)
    #ax.text(0.01, 0.98, "Higher predicted $y$ associates with measurable binding or LB",
    #        transform=ax.transAxes, ha="left", va="top", fontsize=6.0)

    pd.DataFrame(rows).to_csv(DATADIR / "Fig6B_points.csv", index=False)
    save_fig(fig, "Fig6B_pred_vs_outcome")

# =========================
# Panel 6C — Predicted y vs pKd (Kd-fit only; PPT Kd ± err)
# =========================
def panel_6C(df: pd.DataFrame) -> None:
    d = df[df["Outcome_clean"] == "Kd Fit"].copy()
    d = d[np.isfinite(d["Pred_y"]) & np.isfinite(d["pKd_PPT"])].copy()

    # Define the discordant point requested by user (FOXO4–G3BP1) and keep it visible
    outlier_mask = (d["TAD"] == "FOXO4") & (d["partner_display"] == "G3BP1")
    d_in = d[~outlier_mask].copy()

    fig, ax = _new_fig_ax()
    clean_spines(ax)

    # Plot by partner marker (colors are partner-class; TAD labels annotate points)
    for partner, g in d.groupby("partner_display", sort=False):
        marker = PARTNER_MARKER.get(partner, "o")
        col = PARTNER_COLOR.get(partner, C_BLACK)

        x = g["Pred_y"].to_numpy(float)
        y = g["pKd_PPT"].to_numpy(float)
        yerr = g["pKd_err_PPT"].to_numpy(float)

        # Highlight the outlier with an open marker + thicker edge
        is_out = (g["TAD"].to_numpy(str) == "FOXO4") & (g["partner_display"].to_numpy(str) == "G3BP1")
        # non-outlier points
        ax.errorbar(
            x[~is_out], y[~is_out], yerr=yerr[~is_out],
            fmt=marker, ms=5.0, mfc=col, mec=C_BLACK, mew=0.4,
            ecolor=col, elinewidth=0.8, capsize=2.0, alpha=0.95,
            linestyle="None",
        )
        # outlier point (if present in this group)
        if np.any(is_out):
            ax.errorbar(
                x[is_out], y[is_out], yerr=yerr[is_out],
                fmt=marker, ms=5.2, mfc="white", mec=C_VERM, mew=1.0,
                ecolor=C_VERM, elinewidth=0.9, capsize=2.0, alpha=1.0,
                linestyle="None",
            )

    # Point labels (TAD text); few points => readable
    for _, r in d.iterrows():
        label = r["TAD"]
        if (r["TAD"] == "FOXO4") and (r["partner_display"] == "G3BP1"):
            label = "FOXO4 (outlier)"
        ax.annotate(
            label,
            (float(r["Pred_y"]), float(r["pKd_PPT"])),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=5.8,
        )

    # Fit lines: all points (dashed) and excluding outlier (solid)
    x_all = d["Pred_y"].to_numpy(float)
    y_all = d["pKd_PPT"].to_numpy(float)
    m_all, b_all = linear_fit_params(x_all, y_all)

    x_in = d_in["Pred_y"].to_numpy(float)
    y_in = d_in["pKd_PPT"].to_numpy(float)
    m_in, b_in = linear_fit_params(x_in, y_in)

    x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
    pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.1
    xx = np.array([x_min - pad, x_max + pad])

    if np.isfinite(m_all) and np.isfinite(b_all):
        ax.plot(xx, m_all * xx + b_all, lw=0.9, color=C_BLACK, linestyle="--")
    if np.isfinite(m_in) and np.isfinite(b_in) and len(d_in) >= 3:
        ax.plot(xx, m_in * xx + b_in, lw=0.95, color=C_BLUE, linestyle="-")

    # Correlations (report both)
    pear_all, n_all = pearsonr_safe(x_all, y_all)
    spear_all, _ = spearmanr_safe(x_all, y_all)
    pear_in, n_in = pearsonr_safe(x_in, y_in)
    spear_in, _ = spearmanr_safe(x_in, y_in)

    # Save correlation summary (machine-readable)
    corr_rows = [
        {"panel": "Fig6C", "subset": "All Kd Fits", "pearson_r": pear_all, "spearman_rho": spear_all, "n": n_all},
        {"panel": "Fig6C", "subset": "Excluding FOXO4–G3BP1", "pearson_r": pear_in, "spearman_rho": spear_in, "n": n_in},
    ]
    pd.DataFrame(corr_rows).to_csv(DATADIR / "Fig6_correlation_summary.csv", index=False)

    '''ax.text(
        0.99, 0.02,
        f"All Kd Fits: r={pear_all:.2f}, ρ={spear_all:.2f} (n={n_all})\n"
        f"Excluding FOXO4–G3BP1: r={pear_in:.2f}, ρ={spear_in:.2f} (n={n_in})",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=5.8,
    )'''

    ax.set_xlabel(LBL_X_PRED)
    ax.set_ylabel(LBL_Y_PKD)

    # Legend: partner markers + fit lines + outlier cue
    handles = []
    labels = []
    for partner in ["THOC4 N-term", "THOC4 C-term", "G3BP1", "TAF15"]:
        if partner in d["partner_display"].unique():
            handles.append(Line2D([0],[0], marker=PARTNER_MARKER.get(partner, "o"),
                                  linestyle="None", markersize=5,
                                  markerfacecolor=PARTNER_COLOR.get(partner, C_BLACK),
                                  markeredgecolor=C_BLACK, markeredgewidth=0.4))
            labels.append(partner)

    handles.append(Line2D([0],[0], color=C_BLACK, lw=0.9, linestyle="--")); labels.append("Linear Fit (All)")
    handles.append(Line2D([0],[0], color=C_BLUE, lw=0.95, linestyle="-")); labels.append("Linear Fit (Excluding Outlier)")
    handles.append(Line2D([0],[0], marker="o", linestyle="None", markersize=5,
                          markerfacecolor="white", markeredgecolor=C_VERM, markeredgewidth=1.0))
    labels.append("FOXO4–G3BP1 Outlier")

    leg = ax.legend(handles, labels, loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.2)
    if leg is not None:
        leg.set_frame_on(False)

    # Save plotted data
    d_out = d[["TAD","partner_display","pair_id","Pred_y","Kd_uM","Kd_uM_err","pKd_PPT","pKd_err_PPT"]].copy()
    d_out["is_outlier"] = outlier_mask.to_numpy(bool)
    d_out.to_csv(DATADIR / "Fig6C_points.csv", index=False)

    save_fig(fig, "Fig6C_pred_vs_pKd")

def main() -> None:
    ensure_dirs()
    df = load_table()
    panel_6B(df)
    panel_6C(df)
    print(f"[i] Saved to: {OUTDIR}")

if __name__ == "__main__":
    main()
