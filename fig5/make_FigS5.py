#!/usr/bin/env python3
"""
Figure S4 (Panels A–D): Plateau-aware hybrid model — *mechanism & behavior* (not held-out benchmarking).

Uses ONLY your existing CSV artifacts:
  - Training (simulated):  sim_pairs_pred_with_hybrid_sel_cols_with_old.csv
  - Held-out (simulated):  taken_out_preds_with_hybrid_021125_sel_cols.csv
  - Unsampled (pred-only): unsim_preds_with_hybrid_021125_sel_cols.csv

Outputs (PNG + PDF) to: ./figS4_panels/
  - FigS4A_OCN_distribution.png/.pdf
  - FigS4B_DeltaPred_vs_OCN.png/.pdf
  - FigS4C_HighOCN_zoom_training.png/.pdf
  - FigS4D_SimpleResiduals_LowHighOCN_training.png/.pdf

Panel meanings:
  A) Where OCN=40 sits in each dataset distribution (context for “high-OCN subset”)
  B) Applied correction magnitude Δ_pred = (Hybrid − Base) vs OCN (shows gating/bounding behavior)
  C) High-OCN zoom (training): baseline compression vs hybrid expansion (descriptive, not held-out eval)
  D) Simplest bias summary (training): residual medians (IQR) in Low vs High OCN for Base vs Hybrid
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# INPUTS (edit if needed)
# -----------------------
TRAIN_CSV = Path("sim_pairs_pred_with_hybrid_sel_cols_with_old.csv")
HELD_CSV  = Path("taken_out_preds_with_hybrid_021125_sel_cols.csv")
UNSIM_CSV = Path("unsim_preds_with_hybrid_021125_sel_cols.csv")

# -----------------------
# OUTPUT
# -----------------------
OUTDIR = Path("figS4_panels")
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# SETTINGS
# -----------------------
OCN_THR = 40.0  # “high-OCN subset” threshold used in Results
X_MIN, X_MAX = -20, 140

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

rng = np.random.default_rng(0)


def clean_figure_axes(fig) -> None:
    for ax in fig.axes:
        try:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        except Exception:
            pass

def save(fig, stem: str) -> None:
    clean_figure_axes(fig)
    fig.tight_layout()
    fig.savefig(OUTDIR / f"{stem}.png", dpi=600)
    fig.savefig(OUTDIR / f"{stem}.pdf")
    plt.close(fig)


# ============================================================
# Panel A: OCN distribution (Training vs Held-out vs Unsampled)
# ============================================================
train_A = pd.read_csv(TRAIN_CSV, usecols=["Opposite_Charge_Number"])
held_A  = pd.read_csv(HELD_CSV,  usecols=["Opposite_Charge_Number"])
uns_A   = pd.read_csv(UNSIM_CSV, usecols=["Opposite_Charge_Number"])

fig = plt.figure(figsize=(3.2, 2.3))
ax = plt.gca()

bins = np.arange(X_MIN, X_MAX + 1, 5)
ax.hist(train_A["Opposite_Charge_Number"], bins=bins, density=True, alpha=0.55,
        label=f"Training (n={len(train_A):,})")
ax.hist(held_A["Opposite_Charge_Number"], bins=bins, density=True, alpha=0.45,
        label=f"Held-out (n={len(held_A):,})")
ax.hist(uns_A["Opposite_Charge_Number"], bins=bins, density=True, alpha=0.25,
        label=f"Unsampled (n={len(uns_A):,})")

ax.axvline(OCN_THR, linestyle="--", linewidth=1.0)
ax.set_xlim(X_MIN, X_MAX)
ax.set_xlabel("Opposite-Charge Number (OCN)")
ax.set_ylabel("Density")
ax.set_title("OCN distribution and high-OCN threshold")
ax.legend(frameon=True, fontsize=6, loc="upper right")

save(fig, "FigS4A_OCN_distribution")


# ============================================================
# Panel B: Applied correction magnitude vs OCN (Δ_pred)
#   Δ_pred = y_hybrid − y_base
#   (Descriptive: shows where/when correction is applied)
# ============================================================
train_B = pd.read_csv(TRAIN_CSV, usecols=["Opposite_Charge_Number", "B22_pred_old", "B22_pred_hybrid"]).dropna()
held_B  = pd.read_csv(HELD_CSV,  usecols=["Opposite_Charge_Number", "B22_pred_base", "B22_pred_hybrid"]).dropna()
uns_B   = pd.read_csv(UNSIM_CSV, usecols=["Opposite_Charge_Number", "B22_pred_base", "B22_pred_hybrid"]).dropna()

train_B["Delta_pred"] = train_B["B22_pred_hybrid"] - train_B["B22_pred_old"]
held_B["Delta_pred"]  = held_B["B22_pred_hybrid"]  - held_B["B22_pred_base"]
uns_B["Delta_pred"]   = uns_B["B22_pred_hybrid"]   - uns_B["B22_pred_base"]

# Sample unsampled for readability (still representative)
n_samp = min(50000, len(uns_B))
uns_samp = uns_B.iloc[rng.choice(len(uns_B), size=n_samp, replace=False)].copy()

def binned_median(df: pd.DataFrame, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = df["Opposite_Charge_Number"].to_numpy()
    y = df["Delta_pred"].to_numpy()
    idx = np.digitize(x, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = idx == i
        if m.any():
            med[i] = np.nanmedian(y[m])
    return centers, med

edges = np.arange(X_MIN, X_MAX + 1, 10)
c_tr, med_tr = binned_median(train_B, edges)
c_he, med_he = binned_median(held_B, edges)
c_un, med_un = binned_median(uns_B, edges)

fig = plt.figure(figsize=(3.2, 2.3))
ax = plt.gca()

ax.scatter(train_B["Opposite_Charge_Number"], train_B["Delta_pred"], s=8, alpha=0.55, label="Training")
ax.scatter(held_B["Opposite_Charge_Number"],  held_B["Delta_pred"],  s=8, alpha=0.55, label="Held-out")
ax.scatter(uns_samp["Opposite_Charge_Number"], uns_samp["Delta_pred"], s=2, alpha=0.20,
           label="Unsampled (subset)")

ax.plot(c_tr, med_tr, linewidth=1.5, label="Training median (binned)")
ax.plot(c_he, med_he, linewidth=1.5, label="Held-out median (binned)")
ax.plot(c_un, med_un, linewidth=1.5, label="Unsampled median (binned)")

ax.axvline(OCN_THR, linestyle="--", linewidth=1.0)
ax.set_xlim(X_MIN, X_MAX)
ax.set_xlabel("Opposite-Charge Number (OCN)")
ax.set_ylabel(r"$\Delta_{\mathrm{pred}}$ (Hybrid − Base)")
ax.set_title("Applied hybrid correction vs OCN")
ax.legend(frameon=True, fontsize=6, loc="upper left")

save(fig, "FigS4B_DeltaPred_vs_OCN")


# ============================================================
# Panel C: High-OCN zoom (training only)
#   Shows baseline compression in the targeted regime and the hybrid expansion
# ============================================================
train_C = pd.read_csv(TRAIN_CSV, usecols=["Opposite_Charge_Number", "B22_corr1_log10", "B22_pred_old", "B22_pred_hybrid"]).dropna()
sub = train_C.loc[train_C["Opposite_Charge_Number"] >= OCN_THR].copy()

fig = plt.figure(figsize=(3.2, 2.6))
ax = plt.gca()

ax.scatter(sub["B22_corr1_log10"], sub["B22_pred_old"], s=10, alpha=0.6, label="Base MLP (old)")
ax.scatter(sub["B22_corr1_log10"], sub["B22_pred_hybrid"], s=10, alpha=0.6, label="Hybrid")

xmin = float(np.nanmin(sub["B22_corr1_log10"]))
xmax = float(np.nanmax(sub["B22_corr1_log10"]))
ax.plot([xmin, xmax], [xmin, xmax], linewidth=1.0, linestyle="--")

ax.set_xlabel(r"Simulated $y=\log_{10}(-B_{22})$")
ax.set_ylabel(r"Predicted $y$")
ax.set_title(f"High-OCN regime (training; OCN ≥ {int(OCN_THR)})")
ax.legend(frameon=True, fontsize=7, loc="upper left")

save(fig, "FigS4C_HighOCN_zoom_training")


# ============================================================
# Panel D: Simplest “why needed” panel (training only)
#   Two regimes (Low/High OCN), and for each regime:
#     Base vs Hybrid residual medians with IQR whiskers
#   residual = y_sim − y_hat
# ============================================================
train_D = pd.read_csv(TRAIN_CSV, usecols=["Opposite_Charge_Number", "B22_corr1_log10", "B22_pred_old", "B22_pred_hybrid"]).dropna()
train_D["res_base"] = train_D["B22_corr1_log10"] - train_D["B22_pred_old"]
train_D["res_hyb"]  = train_D["B22_corr1_log10"] - train_D["B22_pred_hybrid"]
train_D["regime"] = np.where(train_D["Opposite_Charge_Number"] >= OCN_THR, "High", "Low")

def med_iqr(a: np.ndarray) -> tuple[float, float, float, int]:
    a = np.asarray(a)
    return float(np.median(a)), float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75)), int(a.size)

low_base = train_D.loc[train_D["regime"] == "Low",  "res_base"].to_numpy()
hi_base  = train_D.loc[train_D["regime"] == "High", "res_base"].to_numpy()
low_hyb  = train_D.loc[train_D["regime"] == "Low",  "res_hyb"].to_numpy()
hi_hyb   = train_D.loc[train_D["regime"] == "High", "res_hyb"].to_numpy()

mb_l, q1b_l, q3b_l, n_l = med_iqr(low_base)
mb_h, q1b_h, q3b_h, n_h = med_iqr(hi_base)
mh_l, q1h_l, q3h_l, _   = med_iqr(low_hyb)
mh_h, q1h_h, q3h_h, _   = med_iqr(hi_hyb)

x = np.array([0, 1])   # Low, High
offset = 0.12
x_base = x - offset
x_hyb  = x + offset

fig = plt.figure(figsize=(3.1, 2.0))
ax = plt.gca()

# IQR whiskers + medians
ax.vlines(x_base, [q1b_l, q1b_h], [q3b_l, q3b_h], linewidth=2.0)
ax.plot(x_base, [mb_l, mb_h], marker="o", linestyle="None", label="Base")

ax.vlines(x_hyb, [q1h_l, q1h_h], [q3h_l, q3h_h], linewidth=2.0)
ax.plot(x_hyb, [mh_l, mh_h], marker="o", linestyle="None", label="Hybrid")

ax.axhline(0, linestyle="--", linewidth=1.0)

ax.set_xticks(x)
ax.set_xticklabels([f"Low OCN\n(<40)\n(n={n_l})", f"High OCN\n(≥40)\n(n={n_h})"])
ax.set_ylabel(r"Residual ($y_{\mathrm{sim}}-\hat{y}$)")
ax.set_title("Plateau correction re-centers residuals")
ax.legend(frameon=True, fontsize=7, loc="lower right")

save(fig, "FigS4D_SimpleResiduals_LowHighOCN_training")

print("Wrote S4 panels to:", OUTDIR.resolve())
