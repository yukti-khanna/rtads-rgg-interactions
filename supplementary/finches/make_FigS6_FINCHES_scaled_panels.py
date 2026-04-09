#!/usr/bin/env python3
"""
Figure S6 (scaled): FINCHES ε vs B22 on a common 0–1 scale (easy visual comparison).

Why not epsilon/epsilon.max()?
  Your ε values include negatives (min ~ -72.7, max ~ +22.8), so ε/max ranges to ~ -3.19.
  Instead we use an "attraction-scaled" ε in [0,1]:
      eps_scaled = (eps_max - eps) / (eps_max - eps_min)
  so that more negative (more attractive) ε -> larger eps_scaled (closer to 1).

We scale y = log10(-B22) using a *global* min/max across the two plotted datasets
(training simulated and all-pairs predictions), so Panel A/B share the same scale:
      y_scaled = (y - y_min) / (y_max - y_min)

Inputs (place next to script or edit paths):
  - epsilon_values_cf.csv                         (RGG rows, TAD cols)
  - sim_pairs_pred_with_hybrid_sel_cols_with_old.csv   (training; includes B22_corr1_log10)
  - unsim_preds_with_hybrid_021125_sel_cols.csv        (all predicted; includes B22_pred_hybrid)

Outputs (in ./figS6_finches_scaled/):
  - FigS6A_scaled_FINCHES_vs_simulated_training.(png|pdf)
  - FigS6B_scaled_FINCHES_vs_predicted_allpairs.(png|pdf)
  - FigS6A_training_scaled_merged.csv
  - FigS6B_unsim_scaled_merged.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS_CSV   = "~/finches/cf/epsilon_values_cf.csv"
TRAIN_CSV = "~/calvados/predicted_datasets_021125/sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
UNSIM_CSV = "~/calvados/predicted_datasets_021125/unsim_preds_with_hybrid_021125_sel_cols.csv"

outdir = Path("figS6_finches_scaled")
outdir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

def pearson(x, y) -> float:
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1])

def minmax(x, x_min, x_max):
    denom = (x_max - x_min)
    if denom == 0:
        return np.zeros_like(x, dtype=float)
    return (x - x_min) / denom

# --- ε matrix -> long ---
eps_df = pd.read_csv(EPS_CSV, index_col=0)
eps_df.index = eps_df.index.astype(str).str.strip()
eps_df.columns = eps_df.columns.astype(str).str.strip()
eps_vals = pd.to_numeric(eps_df.stack(), errors="coerce").dropna().to_numpy()
eps_min, eps_max = float(np.nanmin(eps_vals)), float(np.nanmax(eps_vals))

eps_long = eps_df.stack(dropna=False).reset_index()
eps_long.columns = ["rgg", "tad", "epsilon"]
eps_long["rgg"] = eps_long["rgg"].astype("string").str.strip()
eps_long["tad"] = eps_long["tad"].astype("string").str.strip()
eps_long["epsilon"] = pd.to_numeric(eps_long["epsilon"], errors="coerce")
eps_long = eps_long.dropna(subset=["epsilon"])

# --- load datasets and parse names ---
train = pd.read_csv(TRAIN_CSV, usecols=["name", "B22_corr1_log10"])
unsim = pd.read_csv(UNSIM_CSV, usecols=["name", "B22_pred_hybrid"])

for df in (train, unsim):
    split = df["name"].astype("string").str.split("__", n=1, expand=True)
    df["rgg"] = split[0].str.strip()
    df["tad"] = split[1].str.strip()

train_m = train.merge(eps_long, on=["rgg", "tad"], how="inner")
unsim_m = unsim.merge(eps_long, on=["rgg", "tad"], how="inner")

# Global y scaling across both panels
y_min = float(min(train_m["B22_corr1_log10"].min(), unsim_m["B22_pred_hybrid"].min()))
y_max = float(max(train_m["B22_corr1_log10"].max(), unsim_m["B22_pred_hybrid"].max()))

# Scaled variables
train_m["eps_scaled"] = (eps_max - train_m["epsilon"]) / (eps_max - eps_min)
unsim_m["eps_scaled"] = (eps_max - unsim_m["epsilon"]) / (eps_max - eps_min)

train_m["y_scaled"] = minmax(train_m["B22_corr1_log10"], y_min, y_max)
unsim_m["y_scaled"] = minmax(unsim_m["B22_pred_hybrid"], y_min, y_max)

train_m.to_csv(outdir / "FigS6A_training_scaled_merged.csv", index=False)
unsim_m.to_csv(outdir / "FigS6B_unsim_scaled_merged.csv", index=False)

r_train = pearson(train_m["eps_scaled"], train_m["y_scaled"])
r_unsim = pearson(unsim_m["eps_scaled"], unsim_m["y_scaled"])

# --- Panel A ---
fig = plt.figure(figsize=(3.2, 2.6))
ax = plt.gca()
ax.scatter(train_m["eps_scaled"], train_m["y_scaled"], s=10, alpha=0.6)
ax.set_xlabel("FINCHES ε (scaled; more attractive → 1)")
ax.set_ylabel(r"$\log_{10}(-B_{22})$ (scaled)")
ax.set_title(f"Training (n={len(train_m):,}); r={r_train:.2f}")

x = train_m["eps_scaled"].to_numpy()
y = train_m["y_scaled"].to_numpy()
m = np.isfinite(x) & np.isfinite(y)
if m.sum() >= 3:
    coef = np.polyfit(x[m], y[m], 1)
    xx = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 200)
    yy = coef[0]*xx + coef[1]
    ax.plot(xx, yy, linewidth=1.2)

plt.tight_layout()
plt.savefig(outdir / "FigS6A_scaled_FINCHES_vs_simulated_training.png", dpi=600)
plt.savefig(outdir / "FigS6A_scaled_FINCHES_vs_simulated_training.pdf")
plt.close(fig)

# --- Panel B ---
fig = plt.figure(figsize=(3.2, 2.6))
ax = plt.gca()
hb = ax.hexbin(unsim_m["eps_scaled"], unsim_m["y_scaled"], gridsize=55, mincnt=1)
ax.set_xlabel("FINCHES ε (scaled; more attractive → 1)")
ax.set_ylabel(r"$\log_{10}(-B_{22})$ (scaled, hybrid prediction)")
ax.set_title(f"All predicted pairs (n={len(unsim_m):,}); r={r_unsim:.2f}")
cb = plt.colorbar(hb, ax=ax)
cb.set_label("Pair density")

plt.tight_layout()
plt.savefig(outdir / "FigS6B_scaled_FINCHES_vs_predicted_allpairs.png", dpi=600)
plt.savefig(outdir / "FigS6B_scaled_FINCHES_vs_predicted_allpairs.pdf")
plt.close(fig)

print("Wrote:", outdir.resolve())
