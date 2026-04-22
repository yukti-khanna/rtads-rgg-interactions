#!/usr/bin/env python3
# file: make_figS7_simB22_panels.py

"""
Supplementary Fig S7 — Integrate simulated B22 for the NMR/TITAN validation pairs.

This script is self-contained:
  - Reads the TITAN/PPT-derived Table S8 template (with outcomes + Kd ± err + predicted y)
  - Reads the per-pair simulated B22 table
  - Merges them (pair_id ↔ name)
  - Saves a merged table and generates S7 panels (each 3.6×2.3 in)

INPUTS (hard-coded; edit if needed):
  /mnt/data/Table_S8_NMR_TITAN_summary_template_v3.csv
  /mnt/data/validated_final_features_data_pred_b22s.csv

OUTPUTS:
  /mnt/data/FigS7_simB22_panels/
    FigS7A_simB22_vs_pKd.(pdf|png)     (Kd-fit subset)
    FigS7B_pred_vs_simB22.(pdf|png)    (all validation pairs)
    FigS7C_simB22_vs_outcome.(pdf|png) (all validation pairs)
    data/
      Table_S8_with_simulated_B22.csv
      FigS7A_points.csv
      FigS7B_points.csv
      FigS7C_points.csv
      FigS7_correlation_summary.csv

Notes:
- We compute pKd from PPT Kd in µM: pKd = 6 - log10(Kd_uM)
- pKd error is propagated (approx): σ_pKd ≈ (σ_Kd/Kd) / ln(10)
- SAVE_TIGHT=False preserves the physical size (3.6×2.3 in) in exports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# Nature-style rcParams
# =========================
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.titlesize": 7.0,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 6.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# =========================
# Constants / I/O
# =========================
FIGSIZE = (3.6, 2.3)   # DO NOT CHANGE (per user)
DPI_PNG = 600
SAVE_TIGHT = False     # keep physical size

# --- Base directory = this script's folder ---
BASE = Path(__file__).resolve().parent  # /Users/yuktikhanna/calvados/fig6

# --- Inputs (CSV files live here) ---
TABLE_S8  = BASE / "Table_S8.csv"
SIM_TABLE = BASE / "validated_final_features_data_pred_b22s.csv"

# --- Outputs ---
OUTDIR  = BASE / "FigS7_simB22_panels"
DATADIR = BASE / "data"   # you said this already exists (good)
# If you prefer to keep supplementary outputs inside data:
# DATADIR = OUTDIR / "data"

# Okabe–Ito palette (subset)
C_BLUE   = "#0072B2"
C_VERM   = "#D55E00"
C_GREY   = "#B0B0B0"
C_BLACK  = "#000000"

# Partner markers (Panel S7A)
PARTNER_MARKER = {
    "THOC4 N-term": "o",
    "THOC4 C-term": "^",
    "G3BP1": "s",
    "TAF15": "D",
}

# Outcome display names/colors (Panels S7B/S7C)
OUTCOME_DISP = {
    "Kd_fit": "Kd Fit",
    "LB": "Line Broadening",
    "NB": "No Binding",
    "No_Kd": "No Kd",
}
OUTCOME_COLOR = {
    "Kd_fit": C_BLUE,
    "LB": C_VERM,
    "NB": C_GREY,
    "No_Kd": C_BLACK,
}

# If you want to drop the "No Kd" category from S7C (like Fig 6B), set False
INCLUDE_NO_KD_IN_STRIP = True


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

def clean_spines(ax: mpl.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def save_fig(fig: mpl.figure.Figure, stem: str) -> None:
    pdf = OUTDIR / f"{stem}.pdf"
    png = OUTDIR / f"{stem}.png"
    if SAVE_TIGHT:
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(png, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.01)
        fig.savefig(svg, bbox_inches="tight", pad_inches=0.01)
    else:
        fig.savefig(pdf)
        fig.savefig(png, dpi=DPI_PNG)
        fig.savefig(svg)
    plt.close(fig)

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

def identity_limits(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return (0.0, 1.0)
    data = np.concatenate([x[m], y[m]])
    dmin, dmax = float(np.min(data)), float(np.max(data))
    if dmin == dmax:
        dmin -= 0.5
        dmax += 0.5
    pad = 0.05 * (dmax - dmin)
    return dmin - pad, dmax + pad

def kd_uM_to_pKd(kd_uM: float) -> float:
    if not np.isfinite(kd_uM) or kd_uM <= 0:
        return np.nan
    return float(6.0 - np.log10(kd_uM))

def kd_err_to_pKd_err(kd_uM: float, kd_err_uM: float) -> float:
    """σ_pKd ≈ (σ_Kd/Kd)/ln(10)."""
    if not (np.isfinite(kd_uM) and np.isfinite(kd_err_uM)) or kd_uM <= 0 or kd_err_uM <= 0:
        return np.nan
    return float((kd_err_uM / kd_uM) / np.log(10.0))


# =========================
# Load + merge
# =========================
def load_merged() -> pd.DataFrame:
    if not TABLE_S8.exists():
        raise FileNotFoundError(f"Missing input: {TABLE_S8}")
    if not SIM_TABLE.exists():
        raise FileNotFoundError(f"Missing input: {SIM_TABLE}")

    t8 = pd.read_csv(TABLE_S8, low_memory=False)
    sim = pd.read_csv(SIM_TABLE, low_memory=False)

    # Merge (pair_id ↔ name)
    if "pair_id" not in t8.columns:
        raise KeyError("Table S8 must contain column 'pair_id'")
    if "name" not in sim.columns:
        raise KeyError("Simulation table must contain column 'name'")

    # Pull simulation columns (keep flexible if you later add more)
    keep_sim = [c for c in [
        "name",
        "B22_corr1_log10",
        "B22_corr2_log10",
        "B22",
        "Corrected_B22_Hummer",
        "Corrected_B22_Ganguly",
    ] if c in sim.columns]
    sim2 = sim[keep_sim].copy()

    df = t8.merge(sim2, left_on="pair_id", right_on="name", how="left")

    # Numeric coercions (PPT Kd + predicted y)
    for c in ["Kd_uM", "Kd_uM_err", "Pred_log10_negB22_hybrid", "Pred_log10_negB22_base"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Simulated y (default: corr1)
    if "B22_corr1_log10" in df.columns:
        df["Sim_log10_negB22"] = pd.to_numeric(df["B22_corr1_log10"], errors="coerce")
    else:
        df["Sim_log10_negB22"] = np.nan

    # Experimental pKd from PPT Kd
    df["pKd_PPT"] = df["Kd_uM"].apply(kd_uM_to_pKd)
    df["pKd_err_PPT"] = [kd_err_to_pKd_err(k, e) for k, e in zip(df["Kd_uM"], df["Kd_uM_err"])]

    # Save merged table
    out_table = DATADIR / "Table_S8_with_simulated_B22.csv"
    df.to_csv(out_table, index=False)
    return df


# =========================
# Panel S7A — Simulated y vs pKd (Kd-fit only)
# =========================
def panel_S7A(df: pd.DataFrame) -> dict:
    d = df[df["Outcome"].eq("Kd_fit")].copy()
    d = d[np.isfinite(d["Sim_log10_negB22"]) & np.isfinite(d["pKd_PPT"])].copy()

    x = d["Sim_log10_negB22"].to_numpy(float)
    y = d["pKd_PPT"].to_numpy(float)
    yerr = d["pKd_err_PPT"].to_numpy(float)

    r, n = pearsonr_safe(x, y)
    rho, _ = spearmanr_safe(x, y)
    m, b = linear_fit_params(x, y)

    fig, ax = _new_fig_ax()
    clean_spines(ax)

    # points by partner marker (single color)
    for partner, g in d.groupby("partner_display", sort=False):
        mk = PARTNER_MARKER.get(partner, "o")
        ax.errorbar(
            g["Sim_log10_negB22"],
            g["pKd_PPT"],
            yerr=g["pKd_err_PPT"],
            fmt=mk,
            ms=5.0,
            mfc=C_BLUE,
            mec=C_BLACK,
            mew=0.4,
            ecolor=C_BLUE,
            elinewidth=0.8,
            capsize=2.0,
            linestyle="None",
            alpha=0.95,
        )

    # annotate with short TAD labels (few points → ok)
    tad_map = {
        "foxo4_cr3": "FOXO4",
        "foxo6_cr3": "FOXO6",
        "atf4_tad": "ATF4",
        "tfe3_cter": "TFE3",
        "lef1_nter": "LEF1",
    }
    for _, row in d.iterrows():
        ax.annotate(
            tad_map.get(row.get("R_TAD", ""), str(row.get("R_TAD", ""))),
            (float(row["Sim_log10_negB22"]), float(row["pKd_PPT"])),
            xytext=(3, 2),
            textcoords="offset points",
            fontsize=5.8,
        )

    # Fit line
    if np.isfinite(m) and np.isfinite(b) and len(d) >= 2:
        xa, xb = float(np.nanmin(x)), float(np.nanmax(x))
        ax.plot([xa, xb], [m * xa + b, m * xb + b], lw=0.9, color=C_BLACK)

    ax.set_xlabel(r"Simulated $y=\log_{10}(-B_{22})$")
    ax.set_ylabel(r"Experimental $pK_d=-\log_{10}(K_d[\mathrm{M}])$")

    ax.text(
        0.99, 0.02,
        f"r={r:.2f}, ρ={rho:.2f} (n={n})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
    )

    # Marker legend (partner)
    handles, labels = [], []
    for partner in ["THOC4 N-term", "THOC4 C-term", "G3BP1", "TAF15"]:
        if partner in d["partner_display"].unique():
            handles.append(Line2D(
                [0], [0],
                marker=PARTNER_MARKER.get(partner, "o"),
                linestyle="None",
                markersize=5,
                markerfacecolor=C_BLUE,
                markeredgecolor=C_BLACK,
                markeredgewidth=0.4,
            ))
            labels.append(partner)
    if handles:
        ax.legend(handles, labels, loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.2)

    d_out = d[["pair_id", "R_TAD", "partner_display", "pKd_PPT", "pKd_err_PPT", "Sim_log10_negB22"]].copy()
    d_out.to_csv(DATADIR / "FigS7A_points.csv", index=False)

    save_fig(fig, "FigS7A_simB22_vs_pKd")

    return {"panel": "FigS7A", "subset": "Kd Fit", "pearson_r": r, "spearman_rho": rho, "n": n}


# =========================
# Panel S7B — Predicted y vs Simulated y (all validation pairs)
# =========================
def panel_S7B(df: pd.DataFrame) -> dict:
    d = df[np.isfinite(df["Pred_log10_negB22_hybrid"]) & np.isfinite(df["Sim_log10_negB22"])].copy()

    x = d["Sim_log10_negB22"].to_numpy(float)
    y = d["Pred_log10_negB22_hybrid"].to_numpy(float)

    r, n = pearsonr_safe(x, y)
    rho, _ = spearmanr_safe(x, y)

    fig, ax = _new_fig_ax()
    clean_spines(ax)

    colors = d["Outcome"].map(OUTCOME_COLOR).fillna(C_BLACK).to_numpy()

    ax.scatter(x, y, s=18, alpha=0.85, c=colors, edgecolors="none")

    dmin, dmax = identity_limits(x, y)
    ax.plot([dmin, dmax], [dmin, dmax], lw=0.8, color=C_BLACK)
    ax.set_xlim(dmin, dmax)
    ax.set_ylim(dmin, dmax)

    ax.set_xlabel(r"Simulated $y=\log_{10}(-B_{22})$")
    ax.set_ylabel(r"Predicted $y=\log_{10}(-B_{22})$")

    ax.text(
        0.99, 0.02,
        f"r={r:.2f}, ρ={rho:.2f} (n={n})",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
    )

    # Outcome legend
    order = ["Kd_fit", "LB", "NB"] + (["No_Kd"] if ("No_Kd" in d["Outcome"].unique()) else [])
    handles, labels = [], []
    for o in order:
        if o in d["Outcome"].unique():
            handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=5,
                                  markerfacecolor=OUTCOME_COLOR.get(o, C_BLACK),
                                  markeredgecolor=OUTCOME_COLOR.get(o, C_BLACK)))
            labels.append(OUTCOME_DISP.get(o, o))
    if handles:
        ax.legend(handles, labels, loc="upper left", frameon=False, handletextpad=0.4, borderpad=0.2)

    d_out = d[["pair_id", "Outcome", "Pred_log10_negB22_hybrid", "Sim_log10_negB22"]].copy()
    d_out.to_csv(DATADIR / "FigS7B_points.csv", index=False)

    save_fig(fig, "FigS7B_pred_vs_simB22")

    return {"panel": "FigS7B", "subset": "All validation pairs", "pearson_r": r, "spearman_rho": rho, "n": n}


# =========================
# Panel S7C — Simulated y vs Outcome (stripplot)
# =========================
def panel_S7C(df: pd.DataFrame) -> None:
    order = ["Kd_fit", "LB", "NB"] + (["No_Kd"] if INCLUDE_NO_KD_IN_STRIP else [])
    present = [o for o in order if o in df["Outcome"].unique()]

    counts = {o: int((df["Outcome"] == o).sum()) for o in present}

    fig, ax = _new_fig_ax()
    clean_spines(ax)

    rng = np.random.default_rng(123)
    jitter = 0.08

    rows = []
    for yi, outcome in enumerate(present):
        sub = df[df["Outcome"] == outcome].copy()
        x = pd.to_numeric(sub["Sim_log10_negB22"], errors="coerce").to_numpy(float)
        m = np.isfinite(x)
        x = x[m]
        y = yi + rng.uniform(-jitter, jitter, size=len(x))

        ax.scatter(
            x, y,
            s=20, alpha=0.9,
            c=OUTCOME_COLOR.get(outcome, C_BLACK),
            edgecolors="none",
        )
        for pair_id, xv in zip(sub.loc[m, "pair_id"].to_numpy(), x):
            rows.append({"pair_id": pair_id, "Outcome": outcome, "Sim_log10_negB22": float(xv)})

    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels([f"{OUTCOME_DISP.get(o,o)} (n={counts[o]})" for o in present])
    ax.invert_yaxis()

    ax.set_xlabel(r"Simulated $y=\log_{10}(-B_{22})$")
    ax.set_ylabel("NMR Outcome")

    ax.xaxis.grid(True, which="major", linewidth=0.4, color="#D3D3D3")
    ax.set_axisbelow(True)

    pd.DataFrame(rows).to_csv(DATADIR / "FigS7C_points.csv", index=False)
    save_fig(fig, "FigS7C_simB22_vs_outcome")


# =========================
# Main
# =========================
def main() -> None:
    ensure_dirs()
    df = load_merged()

    corr = []
    corr.append(panel_S7A(df))
    corr.append(panel_S7B(df))
    pd.DataFrame(corr).to_csv(DATADIR / "FigS7_correlation_summary.csv", index=False)

    panel_S7C(df)

    print(f"[i] Wrote outputs to: {OUTDIR}")

if __name__ == "__main__":
    main()

