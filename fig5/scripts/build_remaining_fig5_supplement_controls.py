#!/usr/bin/env python3
"""Build remaining Figure-5 supplementary control artifacts from frozen predictions/model.

This script writes only under fig5/controls and fig5/outputs. It does not update Fig. 5,
Fig. S5, manuscript text, or any upstream repositories.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "controls"
FIGOUT = OUT / "figures"
TABOUT = OUT / "tables"
for p in (OUT, FIGOUT, TABOUT):
    p.mkdir(parents=True, exist_ok=True)

FINAL_PRED_DIR = ROOT / "outputs" / "final_hybrid_model" / "final_predictions"
FROZEN = ROOT / "outputs" / "final_hybrid_model" / "frozen_model"
MANIFEST_PATH = FROZEN / "MODEL_MANIFEST.json"
BASE_MODEL_PATH = FROZEN / "base_model_ensemble.joblib"
PREP_PATH = FROZEN / "residual_preprocessor.joblib"
RESIDUAL_PATH = FROZEN / "residual_nn.keras"

S9 = ROOT / "outputs" / "revised_fig5_supplementary_data" / "csv" / "Table_S9_reviewer_controls"
S9A_NONDATASET_WT = S9 / "S9A_non_dataset_WT.csv"
S9C_HISTIDINE = S9 / "S9C_histidine_pH.csv"
S9D_BASIC = S9 / "S9D_basic_region_controls.csv"
SHUFFLE_PRIMARY = ROOT / "shuffle_control_analysis_local" / "outputs" / "shuffle_control_analysis" / "shuffle_nonbasic_matched_effects.csv"
SHUFFLE_ALL = ROOT / "shuffle_control_analysis_local" / "outputs" / "shuffle_control_analysis" / "shuffle_matched_effects.csv"
HISTIDINE_MATCHED = ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_matched_b22.csv"

EXPECTED_FINAL_MODEL_HASH = "48b7279b7e76260c04c1f42ee766d3f385f3d27c5ad61b99b2941d0d02b0d516"

FIGSIZE = (2.4, 2.3)
DPI = 600
GREY = "#656565"
BLUE = "#0072B2"


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def soft_gate(x, center, scale):
    return 1.0 / (1.0 + np.exp(-((np.asarray(x, dtype=float) - center) / scale)))


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mean_lambda_RGG" not in df.columns and {"lambda_RGG", "RGG_length"}.issubset(df.columns):
        df["mean_lambda_RGG"] = df["lambda_RGG"] / df["RGG_length"]
    if "mean_lambda_TAD" not in df.columns and {"lambda_TAD", "TAD_length"}.issubset(df.columns):
        df["mean_lambda_TAD"] = df["lambda_TAD"] / df["TAD_length"]
    return df


def load_model():
    import tensorflow as tf
    missing = [p for p in (MANIFEST_PATH, BASE_MODEL_PATH, PREP_PATH, RESIDUAL_PATH) if not p.exists()]
    if missing:
        raise RuntimeError("Missing frozen model artifacts: " + ", ".join(str(p) for p in missing))
    manifest = load_manifest()
    return manifest, sha256(MANIFEST_PATH), joblib.load(BASE_MODEL_PATH), joblib.load(PREP_PATH), tf.keras.models.load_model(RESIDUAL_PATH)


def predict_final(df: pd.DataFrame, manifest: dict, base, prep, residual) -> pd.DataFrame:
    df = derive_features(df)
    feature_order = manifest["feature_order"]
    missing = [c for c in ["name", *feature_order] if c not in df.columns]
    if missing:
        raise RuntimeError("Missing required frozen-model columns: " + ", ".join(missing))
    if not df["name"].is_unique:
        raise RuntimeError("Control feature table has duplicate name values")
    X = df[feature_order].to_numpy(float)
    if not np.isfinite(X).all():
        raise RuntimeError("Non-finite frozen-model feature value detected")
    base_pred = np.mean([m.predict(base["scaler"].transform(X)) for m in base["models"]], axis=0)
    raw = residual.predict(prep["scaler"].transform(X), verbose=0).reshape(-1)
    minus = df[manifest["minus_NCP_feature"]].to_numpy(float)
    gate = soft_gate(minus, manifest["gate_center"], manifest["gate_scale"])
    clipped = np.clip(raw, -manifest["residual_cap"], manifest["residual_cap"])
    applied = manifest["lambda_final"] * gate * clipped
    out = pd.DataFrame({
        "pair_id": df["name"].astype(str),
        "minus_NCP": minus,
        "base_prediction": base_pred,
        "raw_residual_prediction": raw,
        "gate_value": gate,
        "applied_correction": applied,
        "final_hybrid_prediction": base_pred + applied,
    })
    if "B22_corr1_log10" in df.columns:
        out["simulated_target"] = df["B22_corr1_log10"].to_numpy(float)
        out["residual"] = out["simulated_target"] - out["final_hybrid_prediction"]
        out["absolute_error"] = out["residual"].abs()
    for col in ["condition", "TAD_variant_type", "RGG_variant_type", "B22_sign", "transformed_target_defined", "reason_target_undefined"]:
        if col in df.columns:
            out[col] = df[col].to_numpy()
    return out


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    n = len(y)
    return {
        "n": n,
        "R2": r2_score(y, p) if n > 1 else np.nan,
        "RMSE": mean_squared_error(y, p) ** 0.5 if n else np.nan,
        "MAE": mean_absolute_error(y, p) if n else np.nan,
        "Pearson_r": stats.pearsonr(y, p).statistic if n > 2 else np.nan,
        "Spearman_rho": stats.spearmanr(y, p).statistic if n > 2 else np.nan,
    }


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=2, width=0.6, labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)


def save_panel(fig, stem: str):
    for ext in ("pdf", "svg", "png"):
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = DPI
        fig.savefig(FIGOUT / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def identity_limits(x, y):
    vals = np.concatenate([np.asarray(x, float), np.asarray(y, float)])
    vals = vals[np.isfinite(vals)]
    lo, hi = vals.min(), vals.max()
    pad = (hi - lo) * 0.05 if hi > lo else 0.05
    return lo - pad, hi + pad


def plot_identity_scatter(df, xcol, ycol, stem, xlabel, ylabel, title=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(df[xcol], df[ycol], s=7, alpha=0.55, color=GREY, edgecolors="none")
    dmin, dmax = identity_limits(df[xcol], df[ycol])
    ax.plot([dmin, dmax], [dmin, dmax], lw=0.8, color="black", zorder=3)
    ax.set_xlim(dmin, dmax)
    ax.set_ylim(dmin, dmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=8)
    m = metrics(df[xcol], df[ycol])
    ax.text(0.97, 0.03, f"r={m['Pearson_r']:.2f}\nn={m['n']}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7)
    clean_axes(ax)
    fig.tight_layout(pad=0.4)
    save_panel(fig, stem)


def write_missing(name: str, reason: str, checked_paths: Iterable[Path]):
    text = [f"{name}", "", "Status: NOT GENERATED", "", reason, "", "Checked inputs:"]
    for p in checked_paths:
        text.append(f"- {p} :: exists={p.exists()}")
    (OUT / f"{name}_MISSING_DATA.txt").write_text("\n".join(text) + "\n")


def build_shuffle_simulation_panel(audit_lines):
    df = pd.read_csv(SHUFFLE_PRIMARY)
    needed = ["WT_pair_id", "shuffle_pair_id", "B22_WT", "B22_shuffle", "delta_logB22", "minus_NCP_WT", "minus_NCP_shuffle", "shuffle_type"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Shuffle source missing columns: {missing}")
    src = df[needed + [c for c in ["condition", "TAD_class"] if c in df.columns]].copy()
    src = src.rename(columns={"B22_WT": "WT_log10_minus_B22", "B22_shuffle": "shuffle_log10_minus_B22"})
    src.to_csv(TABOUT / "FigS8B_shuffle_simulation_source.csv", index=False)
    plot_identity_scatter(src, "WT_log10_minus_B22", "shuffle_log10_minus_B22", "FigS8B_shuffle_WT_vs_shuffle_simulated", "WT simulated log10(-B22)", "Shuffle simulated log10(-B22)")
    audit_lines.append(f"FigS8B generated from {SHUFFLE_PRIMARY}; n={len(src)} non-basic matched shuffle rows. No model predictions used.")


def build_histidine_simulation_panel(audit_lines):
    df = pd.read_csv(HISTIDINE_MATCHED if HISTIDINE_MATCHED.exists() else S9C_HISTIDINE)
    needed = ["pair_id", "B22_corr1_log10_pH7", "B22_corr1_log10_pH6p5", "delta_logB22"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise RuntimeError(f"Histidine source missing columns: {missing}")
    src = df[needed + [c for c in ["total_H", "delta_OCN_proxy", "TAD_id", "RGG_id"] if c in df.columns]].copy()
    src.to_csv(TABOUT / "FigS8D_histidine_pH_simulation_source.csv", index=False)
    plot_identity_scatter(src, "B22_corr1_log10_pH7", "B22_corr1_log10_pH6p5", "FigS8D_histidine_pH7_vs_pH65_simulated", "pH 7 simulated log10(-B22)", "pH 6.5 simulated log10(-B22)")
    audit_lines.append(f"FigS8D generated from {HISTIDINE_MATCHED if HISTIDINE_MATCHED.exists() else S9C_HISTIDINE}; n={len(src)} matched pH pairs. No model predictions used.")


def build_basic_model_panel(audit_lines):
    manifest, manifest_hash, base, prep, residual = load_model()
    df = pd.read_csv(S9D_BASIC)
    pred = predict_final(df, manifest, base, prep, residual)
    pred["local_model_manifest_sha256"] = manifest_hash
    pred["definitive_prediction_manifest_sha256"] = EXPECTED_FINAL_MODEL_HASH
    pred.to_csv(TABOUT / "frozen_model_basic_region_control_predictions.csv", index=False)
    valid = pred[np.isfinite(pred.get("simulated_target", np.nan))].copy()
    valid.to_csv(TABOUT / "FigS8F_basic_region_source.csv", index=False)
    plot_identity_scatter(valid, "simulated_target", "final_hybrid_prediction", "FigS8F_basic_region_predicted_vs_simulated", "Simulated log10(-B22)", "Frozen model predicted log10(-B22)")
    m = metrics(valid["simulated_target"], valid["final_hybrid_prediction"])
    audit_lines.append(f"FigS8F generated from {S9D_BASIC}; n={len(valid)} rows with simulated target and full frozen features.")
    audit_lines.append("FigS8F metrics: " + json.dumps(m, sort_keys=True))
    audit_lines.append(f"Local MODEL_MANIFEST sha256: {manifest_hash}")
    audit_lines.append(f"Definitive final-prediction manifest sha256 expected from final tables: {EXPECTED_FINAL_MODEL_HASH}")
    if manifest_hash != EXPECTED_FINAL_MODEL_HASH:
        audit_lines.append("WARNING: local frozen-model manifest hash differs from the definitive hash embedded in final prediction tables. Predictions were generated with the local frozen_model artifacts present in this fig5 workspace.")


def input_inventory() -> pd.DataFrame:
    feature_cols = []
    if MANIFEST_PATH.exists():
        feature_cols = load_manifest().get("feature_order", [])
    rows = []
    for label, path in {
        "development_final_predictions": FINAL_PRED_DIR / "development_917_OOF_final_hybrid.csv",
        "heldout_final_predictions": FINAL_PRED_DIR / "test_1794_final_hybrid_predictions.csv",
        "full_landscape_final_predictions": FINAL_PRED_DIR / "full_landscape_final_hybrid_predictions.csv",
        "S9A_non_dataset_WT": S9A_NONDATASET_WT,
        "S9C_histidine_pH": S9C_HISTIDINE,
        "S9D_basic_region_controls": S9D_BASIC,
        "shuffle_nonbasic_matched_effects": SHUFFLE_PRIMARY,
        "shuffle_all_matched_effects": SHUFFLE_ALL,
        "histidine_matched_b22": HISTIDINE_MATCHED,
    }.items():
        row = {"input": label, "path": str(path), "exists": path.exists(), "sha256": sha256(path)}
        if path.exists() and path.suffix == ".csv":
            df = pd.read_csv(path, nrows=5)
            row["columns_preview"] = ";".join(df.columns[:30])
            row["has_all_frozen_features"] = all(c in df.columns for c in feature_cols) if feature_cols else False
            row["has_old_prediction_columns"] = any("prediction" in c.lower() or "pred" in c.lower() for c in df.columns)
            row["n_rows"] = sum(1 for _ in path.open()) - 1
        rows.append(row)
    inv = pd.DataFrame(rows)
    inv.to_csv(TABOUT / "final_control_input_inventory.csv", index=False)
    return inv


def main():
    audit = []
    audit.append("Remaining Figure-5 supplementary controls rebuilt from frozen final-model resources")
    audit.append(f"fig5 root: {ROOT}")
    audit.append("")
    inv = input_inventory()
    audit.append("Input inventory written to controls/tables/final_control_input_inventory.csv")
    audit.append("")

    # Panels with enough authoritative data.
    build_shuffle_simulation_panel(audit)
    build_histidine_simulation_panel(audit)
    build_basic_model_panel(audit)

    # Panels intentionally not generated because frozen input features are absent.
    write_missing(
        "FigS8A_non_dataset_WT_model_prediction",
        "The available non-dataset WT table contains old prediction columns but does not contain the 13 frozen final-model input features. Reusing old predictions would violate the task requirement to use the already-frozen final hybrid model.",
        [S9A_NONDATASET_WT],
    )
    write_missing(
        "FigS8C_shuffle_model_prediction_effects",
        "The available shuffle matched-effect tables contain simulated WT/shuffle B22 values and old/provisional prediction deltas, but not the complete frozen-model feature rows for WT and shuffle sequences. The model-derived shuffle panel was therefore not regenerated.",
        [SHUFFLE_PRIMARY, SHUFFLE_ALL],
    )
    write_missing(
        "FigS8E_histidine_pH65_model_prediction",
        "The available histidine pH tables contain simulated pH7/pH6.5 targets and old/provisional prediction columns, but not complete frozen-model feature rows for the pH6.5 sequence-state inputs. The model-derived histidine panel was therefore not regenerated.",
        [S9C_HISTIDINE, HISTIDINE_MATCHED],
    )
    audit.append("FigS8A, FigS8C, and FigS8E were not generated because required frozen-model feature inputs are absent from the available control tables. Missing-data notes were written under controls/.")
    audit.append("")
    audit.append("No Figure 5 or Figure S5 files were modified by this script.")
    audit.append("No FINCHES files outside fig5 were modified by this script.")
    (OUT / "FINAL_CONTROL_FIGURE_AUDIT.txt").write_text("\n".join(audit) + "\n")
    (OUT / "FINCHES_AND_CONTROL_UPDATE_SUMMARY.txt").write_text(
        "Control update completed inside fig5/controls. FINCHES was not updated because writing to the existing supplementary/finches workspace is outside this fig5-only output area and requires explicit approval.\n"
        "Generated supported control panels: FigS8B simulation shuffle, FigS8D histidine simulation, FigS8F basic-region frozen-model prediction.\n"
        "Not generated: FigS8A, FigS8C, FigS8E, because available tables lack complete frozen-model feature inputs and old prediction columns were not reused.\n"
    )
    print("Wrote remaining-control outputs to", OUT)


if __name__ == "__main__":
    main()
