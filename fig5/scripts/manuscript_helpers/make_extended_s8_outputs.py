#!/usr/bin/env python3
"""Append corrected new S8 controls to existing controls and regenerate S8 outputs."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FIG5_ROOT = Path(__file__).resolve().parents[2]
WORK = FIG5_ROOT / "s8_extended_controls"
OUT = FIG5_ROOT / "inputs/controls"
PRED = OUT / "new_controls_predictions.csv"
EXISTING = FIG5_ROOT / "controls" / "final_predictions"
PANEL = FIG5_ROOT / "outputs/manuscript/Figure_S6"
PANEL.mkdir(parents=True, exist_ok=True)
MODEL_MANIFEST = FIG5_ROOT / "outputs" / "final_hybrid_model" / "frozen_model" / "MODEL_MANIFEST.json"
FEATURE_ORDER = json.loads(MODEL_MANIFEST.read_text())["feature_order"]

FIGSIZE = (2.4, 2.3)
DPI = 600
GREY = "#656565"
BLUE = "#0072B2"
ORANGE = "#D55E00"
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})


def metrics(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    n = len(y)
    resid = y - p
    return {
        "n": n,
        "R2": r2_score(y, p) if n > 1 else np.nan,
        "RMSE": mean_squared_error(y, p) ** 0.5 if n else np.nan,
        "MAE": mean_absolute_error(y, p) if n else np.nan,
        "Pearson": stats.pearsonr(y, p).statistic if n > 2 else np.nan,
        "Spearman": stats.spearmanr(y, p).statistic if n > 2 else np.nan,
        "mean_residual": float(np.mean(resid)) if n else np.nan,
    }


def delta_metrics(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    n = len(x)
    return {
        "n": n,
        "Pearson": stats.pearsonr(x, y).statistic if n > 2 else np.nan,
        "Spearman": stats.spearmanr(x, y).statistic if n > 2 else np.nan,
        "RMSE_delta": mean_squared_error(x, y) ** 0.5 if n else np.nan,
        "MAE_delta": mean_absolute_error(x, y) if n else np.nan,
        "mean_delta_sim": float(np.mean(x)) if n else np.nan,
        "mean_delta_pred": float(np.mean(y)) if n else np.nan,
    }


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.2, width=0.6, pad=1.5)


def limits(x, y):
    v = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    v = v[np.isfinite(v)]
    lo = float(v.min())
    hi = float(v.max())
    pad = (hi - lo) * 0.05 if hi > lo else 0.1
    return lo - pad, hi + pad


def symlim(x, y):
    v = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    v = v[np.isfinite(v)]
    m = float(np.max(np.abs(v))) * 1.08 if len(v) else 1.0
    return -m, m


def save(fig, stem):
    fig.savefig(PANEL / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(PANEL / f"{stem}.svg", bbox_inches="tight")
    # Inset colourbars otherwise trim one pixel per edge from these PNG exports.
    png_options = {"pad_inches": mpl.rcParams["savefig.pad_inches"] + 1 / DPI} if stem in {
        "S8A_non_dataset_WT_extended", "S8D_basic_controls_extended"} else {}
    fig.savefig(PANEL / f"{stem}.png", dpi=DPI, bbox_inches="tight", **png_options)
    plt.close(fig)


def plot_scatter(df, stem, annotation=""):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = df["B22_corr1_log10"]
    y = df["final_hybrid_prediction"]
    ncp_control = stem in {"S8A_non_dataset_WT_extended", "S8D_basic_controls_extended"}
    if ncp_control:
        shared_ncp = pd.concat([
            pd.read_csv(OUT / source, usecols=["Opposite_Charge_Number"])
            for source in ["S8A_extended_source.csv", "S8D_extended_source.csv"]
        ])["Opposite_Charge_Number"]
        vmin, vmax = float(shared_ncp.min()), float(shared_ncp.max())
        norm = (mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                if vmin < 0 < vmax else mpl.colors.Normalize(vmin=vmin, vmax=vmax))
        points = ax.scatter(x, y, s=7, alpha=0.55, c=df["Opposite_Charge_Number"],
                            cmap="coolwarm", norm=norm, edgecolors="none")
    else:
        ax.scatter(x, y, s=7, alpha=0.55, color=GREY, edgecolors="none")
    lo, hi = limits(x, y)
    ax.plot([lo, hi], [lo, hi], lw=0.8, color="black")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Simulated log10(−B22)")
    ax.set_ylabel("Predicted log10(−B22)")
    m = metrics(x, y)
    ax.text(0.97, 0.03, f"n={m['n']}\nR²={m['R2']:.2f}\nRMSE={m['RMSE']:.2f}\nr={m['Pearson']:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    if annotation:
        ax.text(0.03, 0.97, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
    clean(ax)
    fig.tight_layout(pad=0.4)
    if ncp_control:
        # Inset in empty space: preserve the existing axes and export dimensions.
        cax = ax.inset_axes([0.08, 0.78, 0.35, 0.025])
        ticks = [vmin, 0, vmax] if vmin < 0 < vmax else [vmin, vmax]
        cbar = fig.colorbar(points, cax=cax, orientation="horizontal", ticks=ticks)
        cbar.set_label("−NCP", fontsize=6, labelpad=2)
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.tick_params(labelsize=5.5, length=2, width=0.5, pad=1)
        cbar.outline.set_linewidth(0.5)
    save(fig, stem)
    return m


def normalize_shuffle_class(df: pd.DataFrame) -> pd.Series:
    if "shuffle_class" in df.columns:
        s = df["shuffle_class"].copy()
    else:
        s = pd.Series(index=df.index, dtype=object)
    if "control_class" in df.columns:
        s = s.fillna(df["control_class"].map({"TAD_shuffle": "TAD/AHD shuffle", "RGG_shuffle": "RG/RGG shuffle"}))
    if "control_subtype" in df.columns:
        s = s.fillna(df["control_subtype"].map({"TAD_shuffle": "TAD/AHD shuffle", "RGG_shuffle": "RG/RGG shuffle"}))
    if "condition" in df.columns:
        s = s.fillna(df["condition"].map({"TAD_SHUFFLE_A": "TAD/AHD shuffle", "TAD_SHUFFLE_B": "TAD/AHD shuffle", "RGG_SHUFFLE": "RG/RGG shuffle"}))
    return s


def feature_distribution_audit(existing_frames: list[pd.DataFrame], new: pd.DataFrame) -> pd.DataFrame:
    existing = pd.concat(existing_frames, ignore_index=True, sort=False)
    rows = []
    for origin, df in [("accepted_existing_S8_controls", existing), ("new_controls", new)]:
        for feat in FEATURE_ORDER:
            vals = pd.to_numeric(df[feat], errors="coerce") if feat in df.columns else pd.Series(dtype=float)
            rows.append({
                "feature": feat,
                "source": origin,
                "n_nonmissing": int(vals.notna().sum()),
                "min": float(vals.min()) if vals.notna().any() else np.nan,
                "q05": float(vals.quantile(0.05)) if vals.notna().any() else np.nan,
                "median": float(vals.median()) if vals.notna().any() else np.nan,
                "q95": float(vals.quantile(0.95)) if vals.notna().any() else np.nan,
                "max": float(vals.max()) if vals.notna().any() else np.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "final_feature_distribution_existing_vs_new.csv", index=False)
    return out


def main():
    new = pd.read_csv(PRED)
    orig_wt = pd.read_csv(EXISTING / "nonbasic_WT_final_predictions.csv")
    orig_wt["control_origin"] = "original"
    orig_sh = pd.read_csv(EXISTING / "nonbasic_shuffle_final_predictions.csv")
    orig_sh["control_origin"] = "original"
    wt_parent = pd.read_csv(EXISTING / "shuffle_WT_parent_final_predictions.csv")
    wt_parent["control_origin"] = "original"
    hist = pd.read_csv(EXISTING / "histidine_pH6p5_final_predictions.csv")
    hist["control_origin"] = "original"
    basic = pd.read_csv(EXISTING / "basic_control_final_predictions_ALL80.csv")
    basic["control_origin"] = "original"

    feature_audit = feature_distribution_audit([orig_wt, orig_sh, wt_parent, hist, basic], new)

    new_wt = new[new["control_class"].isin(["WT", "external_AD_core_WT"])].copy()
    new_wt["control_origin"] = "new"
    new_basic = new[new["control_class"].eq("basic_WT")].copy()
    new_basic["control_origin"] = "new"
    new_sh = new[new["control_class"].isin(["TAD_shuffle", "RGG_shuffle"])].copy()
    new_sh["control_origin"] = "new"

    comb_wt = pd.concat([orig_wt, new_wt], ignore_index=True, sort=False)
    comb_basic = pd.concat([basic, new_basic], ignore_index=True, sort=False)

    new_dataset_wt = new[(new["input_group"].eq("dataset_new")) & (new["control_class"].eq("WT"))].copy()
    parent_pool = pd.concat([wt_parent, new_dataset_wt], ignore_index=True, sort=False).drop_duplicates("name")
    parent_map = parent_pool[["name", "B22_corr1_log10", "final_hybrid_prediction"]].rename(columns={
        "name": "parent_WT_pair_id",
        "B22_corr1_log10": "WT_B22_corr1_log10",
        "final_hybrid_prediction": "WT_final_hybrid_prediction",
    })

    comb_sh = pd.concat([orig_sh, new_sh], ignore_index=True, sort=False)
    comb_sh["shuffle_class"] = normalize_shuffle_class(comb_sh)
    mapped = comb_sh.merge(parent_map, on="parent_WT_pair_id", how="left", validate="many_to_one")
    if mapped["WT_B22_corr1_log10"].isna().any():
        missing = mapped.loc[mapped["WT_B22_corr1_log10"].isna(), "parent_WT_pair_id"].drop_duplicates().tolist()
        raise RuntimeError(f"Missing WT parent predictions: {missing[:20]}")
    mapped["shuffle_B22_corr1_log10"] = mapped["B22_corr1_log10"]
    mapped["shuffle_final_hybrid_prediction"] = mapped["final_hybrid_prediction"]
    mapped["delta_simulated"] = mapped["shuffle_B22_corr1_log10"] - mapped["WT_B22_corr1_log10"]
    mapped["delta_predicted"] = mapped["shuffle_final_hybrid_prediction"] - mapped["WT_final_hybrid_prediction"]

    mapped.to_csv(OUT / "shuffle_parent_mapping.csv", index=False)
    comb_wt.to_csv(OUT / "S8A_extended_source.csv", index=False)
    mapped.to_csv(OUT / "S8B_extended_source.csv", index=False)
    hist.to_csv(OUT / "S8C_unchanged_source.csv", index=False)
    comb_basic.to_csv(OUT / "S8D_extended_source.csv", index=False)

    rows = []
    for label, df in [("original", orig_wt), ("new", new_wt), ("combined", comb_wt)]:
        rows.append({"subset": label, **metrics(df["B22_corr1_log10"], df["final_hybrid_prediction"])})
    s8a_metrics = pd.DataFrame(rows)
    s8a_metrics.to_csv(OUT / "S8A_original_new_combined_metrics.csv", index=False)

    rows = []
    for origin in ["original", "new", "combined"]:
        df = mapped if origin == "combined" else mapped[mapped["control_origin"].eq(origin)]
        for cls in ["TAD/AHD shuffle", "RG/RGG shuffle"]:
            sub = df[df["shuffle_class"].eq(cls)]
            rows.append({"subset": origin, "shuffle_class": cls, **delta_metrics(sub["delta_simulated"], sub["delta_predicted"])})
    s8b_metrics = pd.DataFrame(rows)
    s8b_metrics.to_csv(OUT / "S8B_original_new_combined_metrics.csv", index=False)

    rows = []
    for label, df in [("original", basic), ("new", new_basic), ("combined", comb_basic)]:
        eval_df = df[df["target_evaluable"].astype(bool)] if "target_evaluable" in df.columns else df.dropna(subset=["B22_corr1_log10"])
        row = {"subset": label, "total_rows": int(len(df)), "target_evaluable_rows": int(len(eval_df)), "target_undefined_rows": int(len(df) - len(eval_df))}
        row.update(metrics(eval_df["B22_corr1_log10"], eval_df["final_hybrid_prediction"]))
        rows.append(row)
    s8d_metrics = pd.DataFrame(rows)
    s8d_metrics.to_csv(OUT / "S8D_original_new_combined_metrics.csv", index=False)

    plot_scatter(comb_wt, "S8A_non_dataset_WT_extended")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    tad = mapped[mapped["shuffle_class"].eq("TAD/AHD shuffle")]
    rgg = mapped[mapped["shuffle_class"].eq("RG/RGG shuffle")]
    ax.scatter(tad["delta_simulated"], tad["delta_predicted"], s=8, alpha=0.6, color=BLUE, edgecolors="none", label="TAD/AHD")
    ax.scatter(rgg["delta_simulated"], rgg["delta_predicted"], s=10, alpha=0.7, color=ORANGE, edgecolors="none", label="RG/RGG")
    lo, hi = symlim(mapped["delta_simulated"], mapped["delta_predicted"])
    ax.axhline(0, lw=0.6, color="0.45")
    ax.axvline(0, lw=0.6, color="0.45")
    ax.plot([lo, hi], [lo, hi], lw=0.8, color="black")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δ simulated log10(−B22)")
    ax.set_ylabel("Δ predicted log10(−B22)")
    ax.legend(loc="upper left", frameon=False)
    mt = delta_metrics(tad["delta_simulated"], tad["delta_predicted"])
    mr = delta_metrics(rgg["delta_simulated"], rgg["delta_predicted"])
    ax.text(0.97, 0.03, f"TAD: n={mt['n']}, r={mt['Pearson']:.2f}\nRGG: n={mr['n']}, r={mr['Pearson']:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    clean(ax)
    fig.tight_layout(pad=0.4)
    save(fig, "S8B_shuffle_delta_extended")

    plot_scatter(hist, "S8C_histidine_pH6p5_unchanged")
    eval_basic = comb_basic[comb_basic["target_evaluable"].astype(bool)] if "target_evaluable" in comb_basic.columns else comb_basic.dropna(subset=["B22_corr1_log10"])
    plot_scatter(eval_basic, "S8D_basic_controls_extended", annotation=f"{len(eval_basic)}/{len(comb_basic)} controls have B22 < 0\n{len(comb_basic) - len(eval_basic)} outside target domain")

    with pd.ExcelWriter(OUT / "Figure_S8_extended_source_tables.xlsx") as xw:
        comb_wt.to_excel(xw, "S8A_source", index=False)
        mapped.to_excel(xw, "S8B_source", index=False)
        hist.to_excel(xw, "S8C_source", index=False)
        comb_basic.to_excel(xw, "S8D_source", index=False)
        s8a_metrics.to_excel(xw, "S8A_metrics", index=False)
        s8b_metrics.to_excel(xw, "S8B_metrics", index=False)
        s8d_metrics.to_excel(xw, "S8D_metrics", index=False)
        feature_audit.to_excel(xw, "feature_distribution", index=False)

    new_positive = new[("Corrected_B22_Hummer" in new.columns) and (new["Corrected_B22_Hummer"] >= 0)] if "Corrected_B22_Hummer" in new.columns else pd.DataFrame()
    audit_lines = [
        "Corrected extended S8 audit",
        f"feature_order={json.dumps(FEATURE_ORDER)}",
        f"S8A_original_n={len(orig_wt)}",
        f"S8A_new_n={len(new_wt)}",
        f"S8A_combined_n={len(comb_wt)}",
        f"S8B_original_TAD_AHD_n={int(((mapped['control_origin']=='original') & (mapped['shuffle_class']=='TAD/AHD shuffle')).sum())}",
        f"S8B_original_RG_RGG_n={int(((mapped['control_origin']=='original') & (mapped['shuffle_class']=='RG/RGG shuffle')).sum())}",
        f"S8B_new_TAD_AHD_n={int(((mapped['control_origin']=='new') & (mapped['shuffle_class']=='TAD/AHD shuffle')).sum())}",
        f"S8B_new_RG_RGG_n={int(((mapped['control_origin']=='new') & (mapped['shuffle_class']=='RG/RGG shuffle')).sum())}",
        f"S8B_combined_TAD_AHD_n={int((mapped['shuffle_class']=='TAD/AHD shuffle').sum())}",
        f"S8B_combined_RG_RGG_n={int((mapped['shuffle_class']=='RG/RGG shuffle').sum())}",
        f"S8C_rows_unchanged={len(hist)}",
        f"S8D_original_total_n={len(basic)}",
        f"S8D_new_total_n={len(new_basic)}",
        f"S8D_combined_total_n={len(comb_basic)}",
        f"new_B22_nonnegative_rows={len(new_positive)}",
        "corrected_mean_lambda_ranges_new=" + json.dumps(feature_audit[(feature_audit['source']=='new_controls') & (feature_audit['feature'].isin(['mean_lambda_RGG','mean_lambda_TAD']))].set_index('feature')[['min','max']].to_dict('index'), sort_keys=True),
    ]
    if (OUT / "existing_prediction_identity_check.json").exists():
        audit_lines.append("model_identity_check=" + (OUT / "existing_prediction_identity_check.json").read_text().strip())
    (OUT / "EXTENDED_S8_CORRECTED_FINAL_AUDIT.txt").write_text("\n".join(audit_lines) + "\n")
    (OUT / "README.md").write_text(
        "Corrected extended S8 outputs generated from cluster-predicted new controls. "
        "mean_lambda_RGG and mean_lambda_TAD are computed as lambda sums divided by sequence length. "
        "S8B metrics use control_subtype/control_class, not is_shuffle. S8C is copied unchanged from accepted histidine pH6.5 predictions.\n"
    )
    print(f"Wrote corrected extended S8 outputs to {OUT}")


if __name__ == "__main__":
    import sys
    if "--ncp-controls-only" in sys.argv:
        # Replot saved sources only; do not rebuild tables or any other panel.
        wt = pd.read_csv(OUT / "S8A_extended_source.csv")
        basic = pd.read_csv(OUT / "S8D_extended_source.csv")
        eval_basic = basic[basic["target_evaluable"].astype(bool)] if "target_evaluable" in basic.columns else basic.dropna(subset=["B22_corr1_log10"])
        plot_scatter(wt, "S8A_non_dataset_WT_extended")
        plot_scatter(eval_basic, "S8D_basic_controls_extended", annotation=f"{len(eval_basic)}/{len(basic)} controls have B22 < 0\n{len(basic) - len(eval_basic)} outside target domain")
    else:
        main()
