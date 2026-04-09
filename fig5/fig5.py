#!/usr/bin/env python3
# file: make_fig5_panels.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# =========================
# Nature-style rcParams
# =========================
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
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# =========================
# Constants / I/O
# =========================
OUTPUT_DIR = Path("Fig5_panels")
DATA_DIR = OUTPUT_DIR / "data"
FIGSIZE = (2.4, 2.3)
DPI_PNG = 600

COLOR_BLUE = "#0072B2"   # Okabe–Ito blue
COLOR_GREY = "#656565"      # visible grey

RNG_SEED = 12345
N_BINS = 30
MAX_BG_POINTS_F = 100_000  # non-RBP background downsampling in Panel F

# Fixed input files (hard-coded)
TRAINING_CSV = "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
HELDOUT_CSV = "taken_out_preds_with_hybrid_021125_sel_cols.csv"
UNSIM_CSV = "unsim_preds_with_hybrid_021125_sel_cols.csv"
FEATURE_IMPORTANCE_CSV = "feature_pruning_results/permutation_importance_test.csv"
RBP_UNIID_FILE = "rg_rbp_unids.txt"

# Axis labels (Title Case + correct math)
LBL_X_OCN   = "Normalized OCN"
LBL_X_SIM   = r"Simulated $y=\log_{10}(-B_{22})$"
LBL_Y_SIM   = r"Simulated $y=\log_{10}(-B_{22})$"
LBL_Y_PRED  = r"Predicted $y=\log_{10}(-B_{22})$"
LBL_Y_PRED_HYB = r"Hybrid Predicted $y=\log_{10}(-B_{22})$"

# Canonical feature list (Panel B filter; OK to use λ for CALVADOS stickiness)
DEFAULT_FEATURES = [
    "Average_lambda","Min_lambda","Average_sigma","Average_charge","Average_NCPR",
    "Charge_Complementarity","Total_Length","Average_Interaction","Average_SCD","Average_SHD",
    "Opposite_Charge_Number","RGG_length","charge_RGG","lambda_RGG","NCPR_RGG","SCD_RGG","SHD_RGG",
    "TAD_length","charge_TAD","lambda_TAD","NCPR_TAD","SCD_TAD","SHD_TAD",
    "Normalized_OCN","Charge_Asymmetry","Hydropathy_Diff","Lambda_Ratio",
]

# Pretty feature labels for Panel B (Title Case; λ reserved for stickiness features)
PRETTY_FEATURE_LABELS = {
    "Average_lambda": "Average λ",
    "Min_lambda": "Min λ",
    "Average_sigma": "Average σ",
    "Average_charge": "Average Charge",
    "Average_NCPR": "Average NCPR",
    "Charge_Complementarity": "Charge Complementarity",
    "Total_Length": "Total Length",
    "Average_Interaction": "Average Interaction",
    "Average_SCD": "Average SCD",
    "Average_SHD": "Average SHD",
    "Opposite_Charge_Number": "Opposite-Charge Count",
    "RGG_length": "RGG Length",
    "charge_RGG": "Charge (RGG)",
    "lambda_RGG": "λ (RGG)",
    "NCPR_RGG": "NCPR (RGG)",
    "SCD_RGG": "SCD (RGG)",
    "SHD_RGG": "SHD (RGG)",
    "TAD_length": "TAD Length",
    "charge_TAD": "Charge (TAD)",
    "lambda_TAD": "λ (TAD)",
    "NCPR_TAD": "NCPR (TAD)",
    "SCD_TAD": "SCD (TAD)",
    "SHD_TAD": "SHD (TAD)",
    "Normalized_OCN": "Normalized OCN",
    "Charge_Asymmetry": "Charge Asymmetry",
    "Hydropathy_Diff": "Hydropathy Difference",
    "Lambda_Ratio": "λ Ratio",
}

# =========================
# Utilities
# =========================
def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def _read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, low_memory=False)

def _read_feature_importance_table(path: str | Path) -> pd.DataFrame:
    """
    Robustly read permutation importance from:
      - CSV with headers, or
      - Series-saved CSV ['Unnamed: 0','0'], or
      - Headerless 2-col TSV/CSV (feature<sep>importance).
    Returns ['feature','importance'].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature importance file not found: {path}")

    df1 = pd.read_csv(path, low_memory=False)
    cols = [str(c) for c in df1.columns]

    cand_feat = {"feature", "Feature", "variable", "var", "name", "Name"}
    cand_imp  = {"importance","Importance","weight","score","permutation_importance",
                 "importance_mean","perm_importance"}
    lc_map = {c.lower(): c for c in cols}
    fcol = next((lc_map[c] for c in cand_feat if c.lower() in lc_map), None)
    icol = next((lc_map[c] for c in cand_imp  if c.lower() in lc_map), None)
    if fcol and icol:
        out = df1[[fcol, icol]].copy()
        out.columns = ["feature", "importance"]
        return out

    if df1.shape[1] == 2:
        a, b = df1.columns.tolist()
        if ("unnamed" in str(a).lower()) or ("index" in str(a).lower()):
            return df1.rename(columns={a: "feature", b: "importance"})[["feature","importance"]]
        if ("unnamed" in str(b).lower()) or ("index" in str(b).lower()):
            return df1.rename(columns={b: "feature", a: "importance"})[["feature","importance"]]

    df2 = pd.read_csv(path, sep=None, engine="python", header=None, names=["feature","importance"])
    return df2[["feature","importance"]]

def load_rbp_unids(path: str | Path = RBP_UNIID_FILE) -> set[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RBP UniID list not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        ids = [ln.strip() for ln in fh if ln.strip()]
    return set(ids)

def _normalize_columns(df: pd.DataFrame) -> dict[str, str]:
    return {c.lower(): c for c in df.columns}

def find_col(df: pd.DataFrame, candidates: Iterable[str], what: str) -> str:
    """Case-insensitive column autodetect with explicit error."""
    lut = _normalize_columns(df)
    for cand in candidates:
        if cand in df.columns:
            return cand
        lc = cand.lower()
        if lc in lut:
            return lut[lc]
    raise KeyError(
        f"Could not detect column for '{what}'. Tried {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )

def get_pair_col(df: pd.DataFrame) -> str:
    candidates = ["pair", "pair_id", "Pair", "pair_name", "Pair_ID", "name"]
    return find_col(df, candidates, "pair identifier")

def add_is_rbp(df: pd.DataFrame, rbp_ids: set[str]) -> pd.DataFrame:
    """Compute is_rbp from pair string (do not rely on existing is_rbp)."""
    pair_col = get_pair_col(df)

    def parse_unid(val: Optional[str]) -> Optional[str]:
        if not isinstance(val, str):
            return None
        try:
            rgg_part = val.split("__")[0]
            rgg_unid = rgg_part.split("_")[0]
            return rgg_unid
        except Exception:
            return None

    rgg_unids = df[pair_col].apply(parse_unid)
    is_rbp = rgg_unids.apply(lambda u: u in rbp_ids if isinstance(u, str) else False)
    out = df.copy()
    out["is_rbp"] = is_rbp.astype(bool)
    return out

def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """Pearson r with finite-mask and zero-variance guards."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 2:
        return (np.nan, n)
    xs = x[m]; ys = y[m]
    if np.allclose(np.std(xs), 0) or np.allclose(np.std(ys), 0):
        return (np.nan, n)
    r = np.corrcoef(xs, ys)[0, 1]
    return float(r), n

def binned_median(x: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Kept for CSV outputs; not plotted in E/F (we plot linear fits)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    xs = x[m]; ys = y[m]
    if xs.size == 0:
        return pd.DataFrame(columns=["bin_center", "median", "n"])
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        return pd.DataFrame(columns=["bin_center", "median", "n"])
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    inds = np.digitize(xs, edges, right=False) - 1
    rows = []
    for i in range(n_bins):
        sel = inds == i
        if np.any(sel):
            rows.append({"bin_center": centers[i], "median": float(np.median(ys[sel])), "n": int(np.sum(sel))})
    return pd.DataFrame(rows, columns=["bin_center", "median", "n"])

def linear_fit_params(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares slope, intercept on finite points; NaN-safe."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return (np.nan, np.nan)
    xs, ys = x[m], y[m]
    if np.allclose(np.std(xs), 0) or np.allclose(np.std(ys), 0):
        return (np.nan, np.nan)
    slope, intercept = np.polyfit(xs, ys, 1)
    return float(slope), float(intercept)

def identity_limits(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Shared min/max with small padding for identity plots."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return (0.0, 1.0)
    data = np.concatenate([x[m], y[m]])
    dmin, dmax = float(np.min(data)), float(np.max(data))
    if dmin == dmax:
        dmin -= 0.5; dmax += 0.5
    pad = 0.05 * (dmax - dmin)
    return dmin - pad, dmax + pad

def clean_figure_axes(fig) -> None:
    for ax in fig.axes:
        try:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        except Exception:
            pass

def save_fig(fig: mpl.figure.Figure, name: str) -> None:
    """Save both PDF (vector) and PNG (dpi=600), tight border."""
    clean_figure_axes(fig)
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    png_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(png_path, dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

import matplotlib.patches as mpatches

def save_fig_with_right_gutter(fig, ax, name, right_gutter_in=0.12):
    clean_figure_axes(fig)
    # right_gutter_in is extra white space on the RIGHT in inches
    # Create a zero-width rectangle just outside the right edge of the axes.
    w = fig.get_figwidth()
    gutter_frac = right_gutter_in / w  # convert inches -> axes fraction
    extra = mpatches.Rectangle((1.0, 0.0), gutter_frac, 1.0,
                               transform=ax.transAxes, fill=False, linewidth=0)
    pdf = OUTPUT_DIR / f"{name}.pdf"
    png = OUTPUT_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight", bbox_extra_artists=[extra], pad_inches=0.01)
    fig.savefig(png, dpi=DPI_PNG, bbox_inches="tight", bbox_extra_artists=[extra], pad_inches=0.01)
    plt.close(fig)

def prettify_feature_label(raw: str) -> str:
    """Curated pretty labels; fallback to Title Case."""
    if not isinstance(raw, str):
        return str(raw)
    if raw in PRETTY_FEATURE_LABELS:
        return PRETTY_FEATURE_LABELS[raw]
    return raw.replace("_", " ").title()

# =========================
# Panel B — Feature Importance (Top-10; pretty labels)
# =========================
def panel_B() -> None:
    work = _read_feature_importance_table(FEATURE_IMPORTANCE_CSV)
    work = work.dropna(subset=["feature", "importance"]).copy()
    work["importance"] = pd.to_numeric(work["importance"], errors="coerce")
    work = work[np.isfinite(work["importance"])]
    if work.empty:
        raise ValueError("No valid importance values found after parsing.")

    total = work["importance"].sum()
    if not np.isfinite(total) or total == 0:
        raise ValueError("Permutation importance sum is zero or invalid.")
    work["importance_norm"] = work["importance"] / total

    # Optional filter to canonical set
    canon = {f.lower(): f for f in DEFAULT_FEATURES}
    work["_lc"] = work["feature"].astype(str).str.lower()
    if work["_lc"].isin(canon).any():
        work = work[work["_lc"].isin(canon)].copy()
        work["feature"] = work["_lc"].map(canon)

    work["label"] = work["feature"].apply(prettify_feature_label)
    work = work.sort_values("importance_norm", ascending=False)

    top = work.head(10).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    top[["feature","label","importance","importance_norm","rank"]].to_csv(
        DATA_DIR / "Fig5B_feature_importance_data.csv", index=False
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    # tighten layout a touch
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)

    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["importance_norm"], color=COLOR_GREY)
    ax.set_yticks(y_pos, labels=top["label"])
    ax.invert_yaxis()  # largest at top
    ax.set_xlabel("Permutation Importance")
    ax.set_ylabel("")
    save_fig(fig, "Fig5B_feature_importance")


# =========================
# Panel C — Training: Predicted vs Simulated (Base vs Hybrid)
# =========================
def panel_C(rbp_ids: set[str]) -> None:
    df = _read_csv(TRAINING_CSV)

    y_sim_col = find_col(df, ["B22_corr1_log10", "y_sim", "log10_negB22_sim"], "simulated y")
    pred_base_col = find_col(df, ["B22_pred_old", "y_pred_old", "pred_old"], "base prediction")
    pred_hybrid_col = find_col(df, ["B22_pred_hybrid", "y_pred_hybrid", "pred_hybrid"], "hybrid prediction")
    pair_col = get_pair_col(df)

    df = add_is_rbp(df, rbp_ids)

    out = df[[pair_col, y_sim_col, pred_base_col, pred_hybrid_col, "is_rbp"]].copy()
    out.columns = ["pair", "sim", "pred_base", "pred_hybrid", "is_rbp"]
    out.to_csv(DATA_DIR / "Fig5C_train_base_vs_hybrid_data.csv", index=False)

    x = df[y_sim_col].to_numpy()
    y_base = df[pred_base_col].to_numpy()
    y_hyb = df[pred_hybrid_col].to_numpy()

    r_base, n_base = pearsonr_safe(x, y_base)
    r_hyb, n_hyb = pearsonr_safe(x, y_hyb)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)

    ax.scatter(x, y_base, s=7, alpha=0.5, edgecolors="none", c=COLOR_GREY,
               label=f"Base (r={r_base:.2f}, n={n_base})")
    ax.scatter(x, y_hyb, s=7, alpha=0.8, edgecolors="none", c=COLOR_BLUE,
               label=f"Hybrid (r={r_hyb:.2f}, n={n_hyb})")

    dmin, dmax = identity_limits(np.concatenate([x, x]), np.concatenate([y_base, y_hyb]))
    ax.plot([dmin, dmax], [dmin, dmax], lw=0.8, color="black")  # identity
    ax.set_xlim(dmin, dmax); ax.set_ylim(dmin, dmax)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(LBL_X_SIM)
    ax.set_ylabel(LBL_Y_PRED)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "Fig5C_train_base_vs_hybrid")

# =========================
# Panel D — Held-Out: Hybrid vs Simulated + RBP Highlight
# =========================
def panel_D(rbp_ids: set[str]) -> None:
    df = _read_csv(HELDOUT_CSV)

    y_sim_col = find_col(df, ["B22_corr1_log10", "y_sim", "log10_negB22_sim"], "simulated y")
    pred_hybrid_col = find_col(df, ["B22_pred_hybrid", "y_pred_hybrid", "pred_hybrid"], "hybrid prediction")
    pair_col = get_pair_col(df)

    df = add_is_rbp(df, rbp_ids)

    out = df[[pair_col, y_sim_col, pred_hybrid_col, "is_rbp"]].copy()
    out.columns = ["pair", "sim", "pred_hybrid", "is_rbp"]
    out.to_csv(DATA_DIR / "Fig5D_heldout_pred_vs_sim_data.csv", index=False)

    x = df[y_sim_col].to_numpy()
    y = df[pred_hybrid_col].to_numpy()
    is_rbp = df["is_rbp"].to_numpy()

    r_all, n_all = pearsonr_safe(x, y)
    r_rbp, n_rbp = pearsonr_safe(x[is_rbp], y[is_rbp])

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)

    # Non-RBP background in visible grey
    ax.scatter(x[~is_rbp], y[~is_rbp], s=7, alpha=0.50, c=COLOR_GREY, edgecolors="none", label="Non-RBP")
    # RBP overlay
    ax.scatter(x[is_rbp], y[is_rbp], s=7, alpha=0.8, c=COLOR_BLUE, edgecolors="none", label="RBP")

    dmin, dmax = identity_limits(x, y)
    ax.plot([dmin, dmax], [dmin, dmax], lw=0.8, color="black")
    ax.set_xlim(dmin, dmax); ax.set_ylim(dmin, dmax)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(LBL_X_SIM)
    ax.set_ylabel(LBL_Y_PRED)
    ax.text(0.97, 0.03, f"All Pairs: r={r_all:.2f} (n={n_all})\nRBP: r={r_rbp:.2f} (n={n_rbp})",
            ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "Fig5D_heldout_pred_vs_sim")

# =========================
# Panel E — Simulated Scaling: Normalized OCN vs Simulated y (Training + Held-Out)
# =========================
def panel_E(rbp_ids: set[str]) -> None:
    df_tr = _read_csv(TRAINING_CSV)
    df_te = _read_csv(HELDOUT_CSV)

    y_sim_tr = find_col(df_tr, ["B22_corr1_log10", "y_sim", "log10_negB22_sim"], "simulated y (train)")
    y_sim_te = find_col(df_te, ["B22_corr1_log10", "y_sim", "log10_negB22_sim"], "simulated y (held-out)")
    ocn_tr = find_col(df_tr, ["Normalized_OCN", "norm_OCN", "OCN_norm"], "normalized OCN (train)")
    ocn_te = find_col(df_te, ["Normalized_OCN", "norm_OCN", "OCN_norm"], "normalized OCN (held-out)")
    pair_tr = get_pair_col(df_tr); pair_te = get_pair_col(df_te)

    df_tr = add_is_rbp(df_tr, rbp_ids)
    df_te = add_is_rbp(df_te, rbp_ids)

    # Save points
    pts_tr = df_tr[[pair_tr, ocn_tr, y_sim_tr, "is_rbp"]].copy()
    pts_tr.insert(0, "dataset", "Training")
    pts_tr.columns = ["dataset", "pair", "ocn_norm", "y_sim", "is_rbp"]

    pts_te = df_te[[pair_te, ocn_te, y_sim_te, "is_rbp"]].copy()
    pts_te.insert(0, "dataset", "Held-Out")
    pts_te.columns = ["dataset", "pair", "ocn_norm", "y_sim", "is_rbp"]

    pts = pd.concat([pts_tr, pts_te], ignore_index=True)
    pts.to_csv(DATA_DIR / "Fig5E_points.csv", index=False)

    # Also keep binned CSV (compat), but do not plot it
    med_tr = binned_median(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy(), n_bins=N_BINS)
    med_te = binned_median(df_te[ocn_te].to_numpy(), df_te[y_sim_te].to_numpy(), n_bins=N_BINS)
    med_tr.insert(0, "dataset", "Training"); med_te.insert(0, "dataset", "Held-Out")
    pd.concat([med_tr, med_te], ignore_index=True).to_csv(DATA_DIR / "Fig5E_binned.csv", index=False)

    # Linear fits (plotted)
    m_tr, b_tr = linear_fit_params(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())
    m_te, b_te = linear_fit_params(df_te[ocn_te].to_numpy(), df_te[y_sim_te].to_numpy())
    pd.DataFrame([
        {"dataset": "Training", "slope": m_tr, "intercept": b_tr},
        {"dataset": "Held-Out", "slope": m_te, "intercept": b_te},
    ]).to_csv(DATA_DIR / "Fig5E_linear_fit.csv", index=False)

    r_tr, n_tr = pearsonr_safe(df_tr[ocn_tr].to_numpy(), df_tr[y_sim_tr].to_numpy())
    r_te, n_te = pearsonr_safe(df_te[ocn_te].to_numpy(), df_te[y_sim_te].to_numpy())

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)

    ax.scatter(df_tr[ocn_tr], df_tr[y_sim_tr], s=7, alpha=0.50, c=COLOR_GREY, edgecolors="none", label="Training")
    ax.scatter(df_te[ocn_te], df_te[y_sim_te], s=7, alpha=0.80, c=COLOR_BLUE, edgecolors="none", label="Held-Out")

    # Draw straight lines across each dataset span
    for (m, b, xvals, col) in [
        (m_tr, b_tr, df_tr[ocn_tr].to_numpy(), COLOR_GREY),
        (m_te, b_te, df_te[ocn_te].to_numpy(), COLOR_BLUE),
    ]:
        if np.isfinite(m) and np.isfinite(b) and xvals.size:
            xa, xb = np.nanmin(xvals), np.nanmax(xvals)
            ax.plot([xa, xb], [m*xa + b, m*xb + b], lw=1.0, color=col)

    ax.set_xlabel(LBL_X_OCN)
    ax.set_ylabel(LBL_Y_SIM)
    ax.text(0.97, 0.03, f"Training: r={r_tr:.2f} (n={n_tr})\nHeld-Out: r={r_te:.2f} (n={n_te})",
            ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(frameon=False, loc="best", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "Fig5E_simulated_ocn_dependence")

# =========================
# Panel F — All-Pairs Extrapolation: Normalized OCN vs Hybrid Predicted (Unsim/All)
# =========================
def panel_F(rbp_ids: set[str]) -> None:
    df = _read_csv(UNSIM_CSV)

    ocn_col = find_col(df, ["Normalized_OCN", "norm_OCN", "OCN_norm"], "normalized OCN")
    pred_hybrid_col = find_col(df, ["B22_pred_hybrid", "y_pred_hybrid", "pred_hybrid"], "hybrid prediction")
    pair_col = get_pair_col(df)

    df = add_is_rbp(df, rbp_ids)

    rng = np.random.default_rng(RNG_SEED)
    is_rbp = df["is_rbp"].to_numpy()
    idx_all = np.arange(len(df))
    idx_bg = idx_all[~is_rbp]
    idx_rbp = idx_all[is_rbp]

    if len(idx_bg) > MAX_BG_POINTS_F:
        sampled_bg = rng.choice(idx_bg, size=MAX_BG_POINTS_F, replace=False)
        plot_mask = np.zeros(len(df), dtype=bool)
        plot_mask[sampled_bg] = True
        plot_mask[idx_rbp] = True  # keep all RBP
    else:
        plot_mask = np.ones(len(df), dtype=bool)

    # Save points + sampled_flag (which were plotted)
    pts = df[[pair_col, ocn_col, pred_hybrid_col, "is_rbp"]].copy()
    pts.columns = ["pair", "ocn_norm", "pred_hybrid", "is_rbp"]
    pts["sampled_flag"] = plot_mask
    pts.to_csv(DATA_DIR / "Fig5F_points.csv", index=False)

    # Keep binned CSV (compat), not plotted
    med_all = binned_median(df[ocn_col].to_numpy(), df[pred_hybrid_col].to_numpy(), n_bins=N_BINS)
    med_all.to_csv(DATA_DIR / "Fig5F_binned.csv", index=False)

    # Linear fits (computed on ALL points)
    m_all, b_all = linear_fit_params(df[ocn_col].to_numpy(), df[pred_hybrid_col].to_numpy())
    m_rbp, b_rbp = linear_fit_params(df.loc[is_rbp, ocn_col].to_numpy(), df.loc[is_rbp, pred_hybrid_col].to_numpy())
    pd.DataFrame([
        {"subset": "All Pairs", "slope": m_all, "intercept": b_all},
        {"subset": "RBP", "slope": m_rbp, "intercept": b_rbp},
    ]).to_csv(DATA_DIR / "Fig5F_linear_fit.csv", index=False)

    # Stats for annotations from plotted subset (as requested earlier)
    df_plot = df[plot_mask].copy()
    x_all = df_plot[ocn_col].to_numpy()
    y_all = df_plot[pred_hybrid_col].to_numpy()
    is_rbp_plot = df_plot["is_rbp"].to_numpy()
    r_all, n_all = pearsonr_safe(x_all, y_all)
    r_rbp, n_rbp = pearsonr_safe(x_all[is_rbp_plot], y_all[is_rbp_plot])

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.0)

    # Background (All Pairs) in visible grey
    mask_bg = ~is_rbp_plot
    ax.scatter(x_all[mask_bg], y_all[mask_bg], s=7, alpha=0.50, c=COLOR_GREY, edgecolors="none", label="All Pairs")
    # RBP overlay in blue
    ax.scatter(x_all[~mask_bg], y_all[~mask_bg], s=7, alpha=0.80, c=COLOR_BLUE, edgecolors="none", label="RBP")

    # Straight fit line for All Pairs (computed on ALL points)
    if np.isfinite(m_all) and np.isfinite(b_all) and len(df) > 0:
        xa, xb = np.nanmin(df[ocn_col]), np.nanmax(df[ocn_col])
        ax.plot([xa, xb], [m_all*xa + b_all, m_all*xb + b_all], lw=1.0, color="black")

    ax.set_xlabel(LBL_X_OCN)
    ax.set_ylabel(LBL_Y_PRED_HYB)
    ax.text(0.97, 0.03, f"All Pairs: r={r_all:.2f} (n={n_all})\nRBP: r={r_rbp:.2f} (n={n_rbp})",
            ha="right", va="bottom", transform=ax.transAxes)
    ax.legend(frameon=False, loc="best", handletextpad=0.6, borderpad=0.3)
    save_fig(fig, "Fig5F_allpairs_ocn_dependence")

# =========================
# Main
# =========================
def main() -> None:
    ensure_dirs()
    rbp_ids = load_rbp_unids(RBP_UNIID_FILE)

    print(f"[i] Output directory: {OUTPUT_DIR.resolve()}")
    print(f"[i] Data directory:   {DATA_DIR.resolve()}")

    panel_B()
    print("Saved: Fig5B_feature_importance.(pdf|png) and data/Fig5B_feature_importance_data.csv")

    panel_C(rbp_ids)
    print("Saved: Fig5C_train_base_vs_hybrid.(pdf|png) and data/Fig5C_train_base_vs_hybrid_data.csv")

    panel_D(rbp_ids)
    print("Saved: Fig5D_heldout_pred_vs_sim.(pdf|png) and data/Fig5D_heldout_pred_vs_sim_data.csv")

    panel_E(rbp_ids)
    print("Saved: Fig5E_simulated_ocn_dependence.(pdf|png) and data/Fig5E_points.csv / data/Fig5E_binned.csv / data/Fig5E_linear_fit.csv")

    panel_F(rbp_ids)
    print("Saved: Fig5F_allpairs_ocn_dependence.(pdf|png) and data/Fig5F_points.csv / data/Fig5F_binned.csv / data/Fig5F_linear_fit.csv")

if __name__ == "__main__":
    main()
