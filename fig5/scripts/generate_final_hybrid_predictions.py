#!/usr/bin/env python3
"""Generate definitive predictions from the frozen final Figure 5 hybrid model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "outputs" / "final_hybrid_model"
FROZEN = MODEL_ROOT / "frozen_model"
OUT = MODEL_ROOT / "final_predictions"
OUT.mkdir(parents=True, exist_ok=True)

DEV_FEATURES = ROOT / "fig5_residual_tuning_local" / "inputs" / "fig5_917_F2_P8.csv"
DEV_OOF = MODEL_ROOT / "development_oof_final_hybrid_predictions.csv"
TEST_FEATURES = ROOT / "fig5_final_model_validation_local" / "outputs" / "heldout" / "heldout_features.csv"
FULL_FEATURES = ROOT / "patches" / "datasets" / "features_unsim_pairs_for_training.csv"
RBP_IDS = ROOT / "rg_rbp_unids.txt"

MANIFEST_PATH = FROZEN / "MODEL_MANIFEST.json"
BASE_MODEL_PATH = FROZEN / "base_model_ensemble.joblib"
RESIDUAL_PREP_PATH = FROZEN / "residual_preprocessor.joblib"
RESIDUAL_MODEL_PATH = FROZEN / "residual_nn.keras"


def configure_paths(model_dir: Path | None = None, output_dir: Path | None = None) -> None:
    global FROZEN, OUT, MANIFEST_PATH, BASE_MODEL_PATH, RESIDUAL_PREP_PATH, RESIDUAL_MODEL_PATH
    if model_dir is not None:
        FROZEN = model_dir
    if output_dir is not None:
        OUT = output_dir
    MANIFEST_PATH = FROZEN / "MODEL_MANIFEST.json"
    BASE_MODEL_PATH = FROZEN / "base_model_ensemble.joblib"
    RESIDUAL_PREP_PATH = FROZEN / "residual_preprocessor.joblib"
    RESIDUAL_MODEL_PATH = FROZEN / "residual_nn.keras"
    OUT.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Frozen model manifest missing: {MANIFEST_PATH}")
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


def load_rbp_ids() -> set[str]:
    if not RBP_IDS.exists():
        return set()
    return {x.strip() for x in RBP_IDS.read_text().splitlines() if x.strip()}


def rgg_uniprot(pair_id: str) -> str:
    if not isinstance(pair_id, str):
        return ""
    return pair_id.split("__")[0].split("_")[0]


def split_pair(pair_id: str) -> tuple[str, str]:
    if isinstance(pair_id, str) and "__" in pair_id:
        a, b = pair_id.split("__", 1)
        return a, b
    return "", ""


def prepare_table(path: Path, manifest: dict, expect_n: int | None, require_target: bool, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = derive_features(df)
    features = manifest["feature_order"]
    missing = [c for c in ["name"] + features if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label}: missing required columns: {missing}")
    if expect_n is not None and len(df) != expect_n:
        raise RuntimeError(f"{label}: expected {expect_n} rows, found {len(df)}")
    if not df["name"].is_unique:
        raise RuntimeError(f"{label}: duplicate pair IDs found")
    if require_target and "B22_corr1_log10" not in df.columns:
        raise RuntimeError(f"{label}: missing simulated target B22_corr1_log10")
    if not np.isfinite(df[features].to_numpy(float)).all():
        raise RuntimeError(f"{label}: non-finite feature value detected")
    if require_target and not np.isfinite(df["B22_corr1_log10"].to_numpy(float)).all():
        raise RuntimeError(f"{label}: non-finite target detected")
    return df


def load_frozen_model():
    import tensorflow as tf
    required = [MANIFEST_PATH, BASE_MODEL_PATH, RESIDUAL_PREP_PATH, RESIDUAL_MODEL_PATH]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        present = sorted(x.name for x in FROZEN.glob("*")) if FROZEN.exists() else []
        raise RuntimeError(
            "Frozen model package is incomplete. Missing required artifact(s):\n  "
            + "\n  ".join(missing)
            + f"\n\nModel directory checked: {FROZEN}"
            + f"\nFiles present there: {present}"
            + "\n\nCopy the entire outputs/final_hybrid_model/frozen_model directory, including residual_nn.keras, "
              "base_model_ensemble.joblib, residual_preprocessor.joblib, and MODEL_MANIFEST.json."
        )
    manifest = read_manifest()
    manifest_hash = sha256(MANIFEST_PATH)
    base = joblib.load(BASE_MODEL_PATH)
    prep = joblib.load(RESIDUAL_PREP_PATH)
    residual = tf.keras.models.load_model(RESIDUAL_MODEL_PATH)
    return manifest, manifest_hash, base, prep, residual


def predict_components(df: pd.DataFrame, manifest: dict, base, prep, residual, label: str) -> pd.DataFrame:
    features = manifest["feature_order"]
    X = df[features].to_numpy(float)
    Xb = base["scaler"].transform(X)
    base_pred = np.mean([m.predict(Xb) for m in base["models"]], axis=0)
    Xr = prep["scaler"].transform(X)
    raw = residual.predict(Xr, verbose=0).reshape(-1)
    minus = df[manifest["minus_NCP_feature"]].to_numpy(float)
    gate = soft_gate(minus, manifest["gate_center"], manifest["gate_scale"])
    clipped = np.clip(raw, -manifest["residual_cap"], manifest["residual_cap"])
    uncapped = manifest["lambda_final"] * gate * raw
    applied = manifest["lambda_final"] * gate * clipped
    final = base_pred + applied
    err = float(np.max(np.abs(final - (base_pred + applied)))) if len(final) else 0.0
    if err > 1e-10:
        raise RuntimeError(f"{label}: prediction identity failed with max error {err}")
    out = pd.DataFrame({
        "pair_id": df["name"].to_numpy(),
        "minus_NCP": minus,
        "base_prediction": base_pred,
        "raw_residual_prediction": raw,
        "gate_value": gate,
        "lambda_final": manifest["lambda_final"],
        "uncapped_weighted_correction": uncapped,
        "applied_correction": applied,
        "final_hybrid_prediction": final,
    })
    return out


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    resid = y - p
    return {
        "n": int(len(y)),
        "R2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "RMSE": float(mean_squared_error(y, p) ** 0.5) if len(y) else np.nan,
        "MAE": float(mean_absolute_error(y, p)) if len(y) else np.nan,
        "Pearson": float(stats.pearsonr(y, p).statistic) if len(y) > 2 else np.nan,
        "Spearman": float(stats.spearmanr(y, p).statistic) if len(y) > 2 else np.nan,
        "mean_residual": float(np.mean(resid)) if len(y) else np.nan,
        "median_residual": float(np.median(resid)) if len(y) else np.nan,
    }


def metrics_block(name: str, df: pd.DataFrame, target_col: str, pred_col: str) -> str:
    y = df[target_col].to_numpy(float)
    p = df[pred_col].to_numpy(float)
    high = df["high_minus_NCP"].to_numpy(bool)
    return json.dumps({"all": metrics(y, p), "high_minus_NCP": metrics(y[high], p[high])}, indent=2)


def add_ids_and_rbp(out: pd.DataFrame, rbp_ids: set[str]) -> pd.DataFrame:
    pairs = out["pair_id"].map(split_pair)
    out.insert(1, "RGG_id", [x[0] for x in pairs])
    out.insert(2, "TAD_id", [x[1] for x in pairs])
    if rbp_ids:
        out["is_rbp"] = out["pair_id"].map(lambda x: rgg_uniprot(x) in rbp_ids)
    return out


def make_development_oof(manifest_hash: str) -> dict:
    df = pd.read_csv(DEV_OOF)
    if len(df) != 917 or not df["pair_id"].is_unique:
        raise RuntimeError("Development OOF final hybrid file is not 917 unique pairs")
    out = pd.DataFrame({
        "pair_id": df["pair_id"],
        "simulated_target": df["simulated_target"],
        "minus_NCP": df["minus_NCP"],
        "OOF_base_prediction": df["base_OOF_prediction"],
        "OOF_raw_residual_prediction": df["raw_crossfit_residual_prediction"],
        "OOF_gate_value": df["gate_value"],
        "OOF_residual_prediction_clipped": df["residual_prediction_clipped"],
        "OOF_applied_correction": df["applied_residual_correction"],
        "OOF_final_hybrid_prediction": df["hybrid_OOF_prediction"],
    })
    out["residual"] = out["simulated_target"] - out["OOF_final_hybrid_prediction"]
    out["absolute_error"] = np.abs(out["residual"])
    out["high_minus_NCP"] = out["minus_NCP"] >= 40
    out["model_manifest_sha256"] = manifest_hash
    out.to_csv(OUT / "development_917_OOF_final_hybrid.csv", index=False)
    write = metrics_block("development_oof", out, "simulated_target", "OOF_final_hybrid_prediction")
    (OUT / "development_OOF_metrics.txt").write_text("Development 917 OOF final hybrid metrics\n\n" + write + "\n")
    return json.loads(write)


def make_development_frozen(manifest, manifest_hash, base, prep, residual, rbp_ids) -> tuple[dict, float]:
    dev = prepare_table(DEV_FEATURES, manifest, 917, True, "development frozen")
    pred = predict_components(dev, manifest, base, prep, residual, "development frozen")
    out = add_ids_and_rbp(pred, rbp_ids)
    out.insert(3, "simulated_target", dev["B22_corr1_log10"].to_numpy(float))
    out["prediction_type"] = "in_sample_frozen_model"
    out["residual"] = out["simulated_target"] - out["final_hybrid_prediction"]
    out["absolute_error"] = np.abs(out["residual"])
    out["high_minus_NCP"] = out["minus_NCP"] >= 40
    out["model_manifest_sha256"] = manifest_hash
    out.to_csv(OUT / "development_917_frozen_model_predictions.csv", index=False)
    identity = float(np.max(np.abs(out["final_hybrid_prediction"] - (out["base_prediction"] + out["applied_correction"]))))
    return {"n": int(len(out)), "prediction_type": "in_sample_frozen_model"}, identity


def make_test(manifest, manifest_hash, base, prep, residual, rbp_ids) -> tuple[dict, float]:
    test = prepare_table(TEST_FEATURES, manifest, 1794, True, "test 1794")
    pred = predict_components(test, manifest, base, prep, residual, "test 1794")
    out = add_ids_and_rbp(pred, rbp_ids)
    out.insert(3, "simulated_target", test["B22_corr1_log10"].to_numpy(float))
    out["residual"] = out["simulated_target"] - out["final_hybrid_prediction"]
    out["absolute_error"] = np.abs(out["residual"])
    out["high_minus_NCP"] = out["minus_NCP"] >= 40
    out["model_manifest_sha256"] = manifest_hash
    if len(out) != 1794 or not out["pair_id"].is_unique or out["final_hybrid_prediction"].isna().any():
        raise RuntimeError("test 1794 output integrity check failed")
    out.to_csv(OUT / "test_1794_final_hybrid_predictions.csv", index=False)
    m = json.loads(metrics_block("test", out, "simulated_target", "final_hybrid_prediction"))
    (OUT / "test_1794_metrics.txt").write_text("Locked 1,794 test final hybrid metrics\nReporting only; no model parameter was changed.\n\n" + json.dumps(m, indent=2) + "\n")
    identity = float(np.max(np.abs(out["final_hybrid_prediction"] - (out["base_prediction"] + out["applied_correction"]))))
    return m, identity


def make_full(manifest, manifest_hash, base, prep, residual, rbp_ids, full_features_path: Path) -> tuple[dict, float]:
    full = prepare_table(full_features_path, manifest, 191386, False, "full landscape")
    pred = predict_components(full, manifest, base, prep, residual, "full landscape")
    out = add_ids_and_rbp(pred, rbp_ids)
    out["high_minus_NCP"] = out["minus_NCP"] >= 40
    out["model_manifest_sha256"] = manifest_hash
    out.to_csv(OUT / "full_landscape_final_hybrid_predictions.csv", index=False)
    applied = out["applied_correction"].to_numpy(float)
    summary = {
        "n_pairs": int(len(out)),
        "prediction_min": float(out["final_hybrid_prediction"].min()),
        "prediction_max": float(out["final_hybrid_prediction"].max()),
        "prediction_median": float(out["final_hybrid_prediction"].median()),
        "minus_NCP_min": float(out["minus_NCP"].min()),
        "minus_NCP_max": float(out["minus_NCP"].max()),
        "fraction_nonzero_applied_correction": float(np.mean(np.abs(applied) > 1e-12)),
        "applied_correction_min": float(np.min(applied)),
        "applied_correction_max": float(np.max(applied)),
        "applied_correction_median_abs": float(np.median(np.abs(applied))),
    }
    (OUT / "full_landscape_prediction_summary.txt").write_text(json.dumps(summary, indent=2) + "\n")
    identity = float(np.max(np.abs(out["final_hybrid_prediction"] - (out["base_prediction"] + out["applied_correction"]))))
    return summary, identity


def parse_args():
    ap = argparse.ArgumentParser(description="Generate final hybrid predictions from frozen model artifacts.")
    ap.add_argument("--model-dir", default=str(FROZEN), help="Frozen model directory. Defaults to outputs/final_hybrid_model/frozen_model under the fig5 root.")
    ap.add_argument("--output-dir", default=str(OUT), help="Output directory. Defaults to outputs/final_hybrid_model/final_predictions under the fig5 root.")
    ap.add_argument("--full-features", default=str(FULL_FEATURES), help="Full-landscape feature CSV. Defaults to fig5/patches/datasets/features_unsim_pairs_for_training.csv")
    return ap.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    configure_paths(model_dir=model_dir, output_dir=output_dir)
    full_features_path = Path(args.full_features)
    if not full_features_path.is_absolute():
        full_features_path = ROOT / full_features_path
    manifest, manifest_hash, base, prep, residual = load_frozen_model()
    if abs(manifest["lambda_final"] - 0.136803) > 1e-3 or abs(manifest["residual_cap"] - 0.169146) > 1e-3:
        raise RuntimeError("Frozen model calibration values do not match expected identification checks")
    rbp_ids = load_rbp_ids()
    dev_oof_metrics = make_development_oof(manifest_hash)
    dev_frozen_summary, dev_identity = make_development_frozen(manifest, manifest_hash, base, prep, residual, rbp_ids)
    test_metrics, test_identity = make_test(manifest, manifest_hash, base, prep, residual, rbp_ids)
    full_summary, full_identity = make_full(manifest, manifest_hash, base, prep, residual, rbp_ids, full_features_path)

    integrity = f"""TEST SET INTEGRITY

Frozen model manifest existed before test evaluation: YES
Frozen model manifest path: {MANIFEST_PATH}
Frozen model manifest SHA256: {manifest_hash}
lambda_final: {manifest['lambda_final']}
residual_cap: {manifest['residual_cap']}
gate_center: {manifest['gate_center']}
gate_scale: {manifest['gate_scale']}
The 1,794 test targets were not used for fitting/calibration.
No model parameter was changed after seeing test performance.
Test evaluation was reporting-only.
"""
    (OUT / "TEST_SET_INTEGRITY.txt").write_text(integrity)

    consistency = f"""PREDICTION CONSISTENCY AUDIT

Frozen model manifest SHA256: {manifest_hash}
Feature order: {manifest['feature_order']}
lambda_final: {manifest['lambda_final']}
residual_cap: {manifest['residual_cap']}
gate_center: {manifest['gate_center']}
gate_scale: {manifest['gate_scale']}

Development frozen model: same frozen model artifact as test/full landscape.
Test 1,794: same lambda, cap, gate, feature order, weights.
Full landscape: same lambda, cap, gate, feature order, weights.
Full landscape feature source: {full_features_path}
No dataset-specific recalibration was performed.

Identity errors:
Development frozen max abs(final_hybrid - base - applied_correction): {dev_identity}
Test 1,794 max abs(final_hybrid - base - applied_correction): {test_identity}
Full landscape max abs(final_hybrid - base - applied_correction): {full_identity}
"""
    (OUT / "prediction_consistency_audit.txt").write_text(consistency)

    summary = f"""FINAL PREDICTION SUMMARY

Frozen model
manifest SHA256: {manifest_hash}
lambda_final: {manifest['lambda_final']}
cap: {manifest['residual_cap']}
gate centre/scale: {manifest['gate_center']} / {manifest['gate_scale']}

Development OOF
n = 917
{json.dumps(dev_oof_metrics, indent=2)}

Development frozen-model predictions
n = {dev_frozen_summary['n']}
prediction_type = in_sample_frozen_model
These predictions are provenance/model-consistency only and are not generalisation evidence.

Locked test
n = {test_metrics['all']['n']}
{json.dumps(test_metrics, indent=2)}
The test set did not alter the model.

Full landscape
exact n = {full_summary['n_pairs']}
prediction range = [{full_summary['prediction_min']}, {full_summary['prediction_max']}]
median prediction = {full_summary['prediction_median']}
fraction with non-zero applied correction = {full_summary['fraction_nonzero_applied_correction']}
"""
    (OUT / "FINAL_PREDICTION_SUMMARY.txt").write_text(summary)
    print(f"Wrote final prediction tables to {OUT}")
    print(f"Frozen model manifest SHA256: {manifest_hash}")

if __name__ == "__main__":
    main()
