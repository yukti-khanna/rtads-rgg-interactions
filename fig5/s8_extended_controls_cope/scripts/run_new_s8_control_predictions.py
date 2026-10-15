#!/usr/bin/env python3
"""Run final frozen-model predictions for prepared new S8 controls on the cluster."""
from __future__ import annotations

import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import joblib

FIG5_ROOT = Path(__file__).resolve().parents[2]
WORK = FIG5_ROOT / "s8_extended_controls"
OUTPUT = WORK / "outputs"
MODEL_DIR = FIG5_ROOT / "outputs" / "final_hybrid_model" / "frozen_model"
INPUT_CSV = OUTPUT / "new_controls_model_input.csv"
IDENTITY_TOLERANCE = 1e-7

EXISTING_FILES = [
    FIG5_ROOT / "controls" / "final_predictions" / "nonbasic_WT_final_predictions.csv",
    FIG5_ROOT / "controls" / "final_predictions" / "shuffle_WT_parent_final_predictions.csv",
    FIG5_ROOT / "controls" / "final_predictions" / "nonbasic_shuffle_final_predictions.csv",
    FIG5_ROOT / "controls" / "final_predictions" / "histidine_pH6p5_final_predictions.csv",
    FIG5_ROOT / "controls" / "final_predictions" / "basic_control_final_predictions_ALL80.csv",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def soft_gate(x, center, scale):
    return 1.0 / (1.0 + np.exp(-((np.asarray(x, dtype=float) - center) / scale)))


def load_model():
    import tensorflow as tf
    manifest = json.loads((MODEL_DIR / "MODEL_MANIFEST.json").read_text())
    base = joblib.load(MODEL_DIR / "base_model_ensemble.joblib")
    prep = joblib.load(MODEL_DIR / "residual_preprocessor.joblib")
    residual = tf.keras.models.load_model(MODEL_DIR / "residual_nn.keras")
    hashes = {p.name: sha(p) for p in MODEL_DIR.iterdir() if p.is_file() and p.name in ["MODEL_MANIFEST.json", "base_model_ensemble.joblib", "residual_preprocessor.joblib", "residual_nn.keras", "predict_final_hybrid.py"]}
    return manifest, base, prep, residual, hashes


def predict(df, manifest, base, prep, residual):
    features = manifest["feature_order"]
    missing = [c for c in ["name", *features] if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    if df[features].isna().any().any():
        raise RuntimeError("Input feature columns contain missing values")
    X = df[features].to_numpy(float)
    base_pred = np.mean([m.predict(base["scaler"].transform(X)) for m in base["models"]], axis=0)
    raw = residual.predict(prep["scaler"].transform(X), verbose=0).reshape(-1)
    minus = df[manifest["minus_NCP_feature"]].to_numpy(float)
    gate = soft_gate(minus, manifest["gate_center"], manifest["gate_scale"])
    uncapped = manifest["lambda_final"] * gate * raw
    applied = manifest["lambda_final"] * gate * np.clip(raw, -manifest["residual_cap"], manifest["residual_cap"])
    out = df.copy()
    out["base_prediction"] = base_pred
    out["raw_residual_prediction"] = raw
    out["gate_value"] = gate
    out["lambda_final"] = manifest["lambda_final"]
    out["uncapped_correction"] = uncapped
    out["applied_correction"] = applied
    out["final_hybrid_prediction"] = base_pred + applied
    out["prediction_identity_error"] = np.abs(out["final_hybrid_prediction"] - (out["base_prediction"] + out["applied_correction"]))
    out["model_manifest_sha256"] = sha(MODEL_DIR / "MODEL_MANIFEST.json")
    out["base_model_sha256"] = sha(MODEL_DIR / "base_model_ensemble.joblib")
    out["residual_preprocessor_sha256"] = sha(MODEL_DIR / "residual_preprocessor.joblib")
    out["residual_model_sha256"] = sha(MODEL_DIR / "residual_nn.keras")
    if "B22_corr1_log10" in out.columns:
        out["residual"] = out["B22_corr1_log10"] - out["final_hybrid_prediction"]
        out["absolute_error"] = np.abs(out["residual"])
    return out


def identity_check(manifest, base, prep, residual) -> dict:
    existing = []
    for p in EXISTING_FILES:
        df = pd.read_csv(p)
        df["existing_source_file"] = p.name
        existing.append(df)
    existing = pd.concat(existing, ignore_index=True, sort=False)
    features = manifest["feature_order"]
    check = existing.dropna(subset=[*features, "base_prediction", "raw_residual_prediction", "gate_value", "applied_correction", "final_hybrid_prediction"]).drop_duplicates(subset=["name"]).copy()
    pred = predict(check[["name", *features]], manifest, base, prep, residual)
    m = check.merge(pred[["name", "base_prediction", "raw_residual_prediction", "gate_value", "applied_correction", "final_hybrid_prediction"]], on="name", suffixes=("_stored", "_new"), validate="one_to_one")
    diffs = {col: float(np.max(np.abs(m[f"{col}_stored"] - m[f"{col}_new"]))) for col in ["base_prediction", "raw_residual_prediction", "gate_value", "applied_correction", "final_hybrid_prediction"]}
    diffs["n_rows_compared"] = int(len(m))
    metric_keys = ["base_prediction", "raw_residual_prediction", "gate_value", "applied_correction", "final_hybrid_prediction"]
    diffs["identity_tolerance"] = IDENTITY_TOLERANCE
    diffs["pass_identity_tolerance"] = bool(all(diffs[k] <= IDENTITY_TOLERANCE for k in metric_keys))
    return diffs


def main():
    manifest, base, prep, residual, hashes = load_model()
    identity = identity_check(manifest, base, prep, residual)
    (OUTPUT / "existing_prediction_identity_check.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    if not identity["pass_identity_tolerance"]:
        raise RuntimeError(f"Existing prediction identity check failed: {identity}")
    df = pd.read_csv(INPUT_CSV)
    out = predict(df, manifest, base, prep, residual)
    out.to_csv(OUTPUT / "new_controls_predictions.csv", index=False)
    (OUTPUT / "PREDICTION_AUDIT.txt").write_text(
        "New S8 control prediction audit\n"
        f"model_dir={MODEL_DIR}\n"
        f"model_hashes={json.dumps(hashes, sort_keys=True)}\n"
        f"identity_check={json.dumps(identity, sort_keys=True)}\n"
        f"predicted_n={len(out)}\n"
        f"max_prediction_identity_error={float(out['prediction_identity_error'].max())}\n"
    )
    print(f"Wrote predictions to {OUTPUT / 'new_controls_predictions.csv'}")


if __name__ == "__main__":
    main()
