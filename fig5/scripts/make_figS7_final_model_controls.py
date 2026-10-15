#!/usr/bin/env python3
"""Generate standalone Figure S7 final frozen-model control panels and source data.

This is plotting/source-data only. It reads definitive cluster-generated prediction
CSVs and does not run model inference or alter predictions/features.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

FIG5_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = FIG5_ROOT / "controls" / "final_predictions"
OUT = FIG5_ROOT / "controls" / "figS7_final_model_controls"
OUT.mkdir(parents=True, exist_ok=True)

FIGSIZE = (2.4, 2.3)
DPI = 600
GREY = "#656565"
BLUE = "#0072B2"
ORANGE = "#D55E00"

FILES = {
    "A": PRED_DIR / "nonbasic_WT_final_predictions.csv",
    "B_wt": PRED_DIR / "shuffle_WT_parent_final_predictions.csv",
    "B_shuf": PRED_DIR / "nonbasic_shuffle_final_predictions.csv",
    "C": PRED_DIR / "histidine_pH6p5_final_predictions.csv",
    "D": PRED_DIR / "basic_control_final_predictions_ALL80.csv",
    "audit": PRED_DIR / "FINAL_CONTROL_PREDICTION_AUDIT.txt",
}

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


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2.2, width=0.6, pad=1.5)


def identity_limits(x, y, pad_frac=0.05):
    vals = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    vals = vals[np.isfinite(vals)]
    lo, hi = float(vals.min()), float(vals.max())
    pad = (hi - lo) * pad_frac if hi > lo else 0.1
    return lo - pad, hi + pad


def symmetric_limits(x, y, pad_frac=0.08):
    vals = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    vals = vals[np.isfinite(vals)]
    m = float(np.max(np.abs(vals))) if len(vals) else 1.0
    m = m * (1 + pad_frac) if m else 0.1
    return -m, m


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]; p = p[ok]
    n = len(y)
    resid = y - p
    return {
        "n": int(n),
        "R2": float(r2_score(y, p)) if n > 1 else np.nan,
        "RMSE": float(mean_squared_error(y, p) ** 0.5) if n else np.nan,
        "MAE": float(mean_absolute_error(y, p)) if n else np.nan,
        "Pearson_r": float(stats.pearsonr(y, p).statistic) if n > 2 else np.nan,
        "Spearman_rho": float(stats.spearmanr(y, p).statistic) if n > 2 else np.nan,
        "mean_residual_bias_sim_minus_pred": float(np.mean(resid)) if n else np.nan,
        "mean_prediction_error_pred_minus_sim": float(np.mean(p - y)) if n else np.nan,
    }


def corr_metrics(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    n = len(x)
    return {
        "n": int(n),
        "Pearson_r": float(stats.pearsonr(x, y).statistic) if n > 2 else np.nan,
        "Spearman_rho": float(stats.spearmanr(x, y).statistic) if n > 2 else np.nan,
    }


def save_fig(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_sim(df, stem, extra_text=None):
    x = df["B22_corr1_log10"].astype(float)
    y = df["final_hybrid_prediction"].astype(float)
    m = metrics(x, y)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(x, y, s=7, alpha=0.55, color=GREY, edgecolors="none")
    lo, hi = identity_limits(x, y)
    ax.plot([lo, hi], [lo, hi], lw=0.8, color="black", zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Simulated log10(−B22)")
    ax.set_ylabel("Predicted log10(−B22)")
    stat = f"n={m['n']}\nR²={m['R2']:.2f}\nRMSE={m['RMSE']:.2f}\nr={m['Pearson_r']:.2f}"
    ax.text(0.97, 0.03, stat, transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    if extra_text:
        ax.text(0.03, 0.97, extra_text, transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
    clean_axes(ax)
    fig.tight_layout(pad=0.4)
    save_fig(fig, stem)
    return m


def parse_audit_value(pattern, text):
    m = re.search(pattern, text)
    return m.group(1) if m else "NA"


def main():
    for k,p in FILES.items():
        if not p.exists():
            raise FileNotFoundError(p)
    audit_text = FILES["audit"].read_text()

    # S7A
    a = pd.read_csv(FILES["A"])
    if len(a) != 80:
        raise RuntimeError(f"S7A expected 80 rows, found {len(a)}")
    a_source = a.copy()
    a_source.to_csv(OUT / "FigS7A_source.csv", index=False)
    m_a = plot_pred_vs_sim(a_source, "FigS7A_nonbasic_WT_simulated_vs_predicted")

    # S7B
    wt = pd.read_csv(FILES["B_wt"])
    sh = pd.read_csv(FILES["B_shuf"])
    if len(wt) != 80 or len(sh) != 240:
        raise RuntimeError(f"S7B expected WT=80/shuffle=240, found WT={len(wt)} shuffle={len(sh)}")
    if "parent_WT_pair_id" not in sh.columns:
        raise RuntimeError("nonbasic_shuffle file missing parent_WT_pair_id")
    wt_map = wt[["name", "B22_corr1_log10", "final_hybrid_prediction"]].rename(columns={
        "name": "parent_WT_pair_id",
        "B22_corr1_log10": "WT_B22_corr1_log10",
        "final_hybrid_prediction": "WT_final_hybrid_prediction",
    })
    b = sh.merge(wt_map, on="parent_WT_pair_id", how="left", validate="many_to_one")
    if b[["WT_B22_corr1_log10", "WT_final_hybrid_prediction"]].isna().any().any():
        missing = b.loc[b["WT_B22_corr1_log10"].isna(), "parent_WT_pair_id"].drop_duplicates().tolist()
        raise RuntimeError(f"Shuffle parent matching failed for {missing[:20]}")
    b["shuffle_pair_id"] = b["name"]
    b["shuffle_class"] = b["control_subtype"].map({"TAD_shuffle": "TAD shuffle", "RGG_shuffle": "RGG shuffle"}).fillna(b["control_subtype"])
    b["shuffle_B22_corr1_log10"] = b["B22_corr1_log10"]
    b["shuffle_final_hybrid_prediction"] = b["final_hybrid_prediction"]
    b["delta_simulated"] = b["shuffle_B22_corr1_log10"] - b["WT_B22_corr1_log10"]
    b["delta_predicted"] = b["shuffle_final_hybrid_prediction"] - b["WT_final_hybrid_prediction"]
    b_source_cols = [
        "shuffle_pair_id", "parent_WT_pair_id", "shuffle_class", "condition", "WT_B22_corr1_log10",
        "shuffle_B22_corr1_log10", "WT_final_hybrid_prediction", "shuffle_final_hybrid_prediction",
        "delta_simulated", "delta_predicted", "model_manifest_sha256", "lambda_final", "gate_value",
    ]
    b[b_source_cols].to_csv(OUT / "FigS7B_source.csv", index=False)
    tad = b[b["control_subtype"].eq("TAD_shuffle")]
    rgg = b[b["control_subtype"].eq("RGG_shuffle")]
    if len(tad) != 160 or len(rgg) != 80:
        raise RuntimeError(f"S7B expected TAD=160/RGG=80, found TAD={len(tad)} RGG={len(rgg)}")
    m_tad = corr_metrics(tad["delta_simulated"], tad["delta_predicted"])
    m_rgg = corr_metrics(rgg["delta_simulated"], rgg["delta_predicted"])
    m_pool = corr_metrics(b["delta_simulated"], b["delta_predicted"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(tad["delta_simulated"], tad["delta_predicted"], s=8, alpha=0.60, color=BLUE, edgecolors="none", label="TAD shuffle")
    ax.scatter(rgg["delta_simulated"], rgg["delta_predicted"], s=10, alpha=0.70, color=ORANGE, edgecolors="none", label="RGG shuffle")
    lo, hi = symmetric_limits(b["delta_simulated"], b["delta_predicted"])
    ax.axhline(0, lw=0.6, color="0.45", zorder=0)
    ax.axvline(0, lw=0.6, color="0.45", zorder=0)
    ax.plot([lo, hi], [lo, hi], lw=0.8, color="black", zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Δ simulated log10(−B22)")
    ax.set_ylabel("Δ predicted log10(−B22)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.2, borderpad=0.1)
    ax.text(0.97, 0.03, f"TAD: n={m_tad['n']}, r={m_tad['Pearson_r']:.2f}\nRGG: n={m_rgg['n']}, r={m_rgg['Pearson_r']:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    clean_axes(ax)
    fig.tight_layout(pad=0.4)
    save_fig(fig, "FigS7B_shuffle_delta_simulated_vs_predicted")

    # S7C
    c = pd.read_csv(FILES["C"])
    if len(c) != 390:
        raise RuntimeError(f"S7C expected 390 rows, found {len(c)}")
    c.to_csv(OUT / "FigS7C_source.csv", index=False)
    m_c = plot_pred_vs_sim(c, "FigS7C_histidine_pH6p5_simulated_vs_predicted")

    # S7D
    d = pd.read_csv(FILES["D"])
    if len(d) != 80:
        raise RuntimeError(f"S7D expected 80 rows, found {len(d)}")
    if "pair_id" in d.columns:
        blank = d["pair_id"].isna() | (d["pair_id"].astype(str).str.strip() == "")
        d.loc[blank, "pair_id"] = d.loc[blank, "name"]
    else:
        d["pair_id"] = d["name"]
    d["target_evaluable"] = d["target_evaluable"].astype(bool)
    d.to_csv(OUT / "FigS7D_source.csv", index=False)
    d_eval = d[d["target_evaluable"] & np.isfinite(d["B22_corr1_log10"])].copy()
    pos_n = int((~d["target_evaluable"]).sum())
    if len(d_eval) != 78 or pos_n != 2:
        raise RuntimeError(f"S7D expected 78 evaluable/2 undefined, found {len(d_eval)}/{pos_n}")
    m_d = plot_pred_vs_sim(d_eval, "FigS7D_basic_controls_simulated_vs_predicted", extra_text="78/80 controls have B22 < 0\n2 with B22 > 0 outside target domain")

    # Provenance values
    manifest_hashes = sorted(set(pd.concat([a, wt, sh, c, d], ignore_index=True, sort=False)["model_manifest_sha256"].dropna().astype(str)))
    lambdas = sorted(set(pd.concat([a, wt, sh, c, d], ignore_index=True, sort=False)["lambda_final"].dropna().round(15).astype(str)))
    cap = parse_audit_value(r"cap_loaded=([^\n]+)", audit_text)
    gate_center = parse_audit_value(r"gate_center=([^\n]+)", audit_text)
    gate_scale = parse_audit_value(r"gate_scale=([^\n]+)", audit_text)
    pH_feature_statement = "pH6.5 features are identical to pH7 features according to controls/HISTIDINE_FEATURE_REPRESENTATION_AUDIT.txt and FINAL_CONTROL_PREDICTION_AUDIT.txt."

    lines = [
        "FIGURE S7 FINAL MODEL CONTROL AUDIT",
        "",
        "All Figure S7 predictions were generated by the authoritative cluster-deployed frozen hybrid model. No fitting, recalibration, or model inference was performed during figure generation.",
        "The WT-vs-shuffle and pH7-vs-pH6.5 simulation-only analyses are not part of Figure S7 and are handled in the Figure 4 supplementary analysis.",
        "",
        f"model manifest SHA256 from input files: {manifest_hashes}",
        f"lambda values from input files: {lambdas}",
        f"cap loaded from cluster audit: {cap}",
        f"gate center/scale from cluster audit: {gate_center}/{gate_scale}",
        "",
        "S7A — Non-basic WT external/control pairs",
        f"source file: {OUT/'FigS7A_source.csv'}",
        f"n: {m_a['n']}",
        f"displayed metrics: {json.dumps(m_a, sort_keys=True)}",
        "",
        "S7B — Shuffle perturbation prediction",
        f"source file: {OUT/'FigS7B_source.csv'}",
        f"WT parent n: {wt['name'].nunique()}",
        f"TAD-shuffle n: {m_tad['n']}",
        f"RGG-shuffle n: {m_rgg['n']}",
        "exact matching success: all 240 shuffled pairs matched to one WT parent using parent_WT_pair_id",
        f"TAD Pearson/Spearman: {m_tad['Pearson_r']:.12g}/{m_tad['Spearman_rho']:.12g}",
        f"RGG Pearson/Spearman: {m_rgg['Pearson_r']:.12g}/{m_rgg['Spearman_rho']:.12g}",
        f"pooled statistics for audit only: {json.dumps(m_pool, sort_keys=True)}",
        "",
        "S7C — Histidine pH6.5 applicability",
        f"source file: {OUT/'FigS7C_source.csv'}",
        f"source n: {len(c)}",
        f"displayed metrics: {json.dumps(m_c, sort_keys=True)}",
        pH_feature_statement,
        "",
        "S7D — Basic/noncanonical applicability",
        f"source file: {OUT/'FigS7D_source.csv'}",
        f"total n = {len(d)}",
        f"plotted/evaluable n = {len(d_eval)}",
        f"positive-B22 n = {pos_n}",
        f"displayed metrics: {json.dumps(m_d, sort_keys=True)}",
        "confirmation: the two target-undefined rows were retained in FigS7D_source.csv with target_evaluable=false.",
    ]
    (OUT / "FIGS7_FINAL_AUDIT.txt").write_text("\n".join(lines) + "\n")
    print("Wrote Figure S7 outputs to", OUT)


if __name__ == "__main__":
    main()
