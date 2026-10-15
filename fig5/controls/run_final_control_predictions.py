#!/usr/bin/env python3
"""Run frozen-model inference for Figure-5 supplementary control datasets.

Run from the mirrored fig5 directory on the cluster. This script is inference-only:
it loads the frozen model artifacts, predicts prepared control inputs, performs
identity/provenance checks, and writes prediction tables. It never fits or tunes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FIG5_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = FIG5_ROOT / "outputs" / "final_hybrid_model" / "frozen_model"
INPUT_DIR = FIG5_ROOT / "controls" / "final_prediction_inputs"
OUTPUT_DIR = FIG5_ROOT / "controls" / "final_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    ("development_917", "development_917_reference_input.csv", "development_917_cluster_predictions.csv", "B22_corr1_log10"),
    ("nonbasic_WT", "nonbasic_WT_prediction_input.csv", "nonbasic_WT_final_predictions.csv", "B22_corr1_log10"),
    ("shuffle_WT_parent", "shuffle_WT_parent_prediction_input.csv", "shuffle_WT_parent_final_predictions.csv", "B22_corr1_log10"),
    ("nonbasic_shuffle", "nonbasic_shuffle_prediction_input.csv", "nonbasic_shuffle_final_predictions.csv", "B22_corr1_log10"),
    ("histidine_pH7", "histidine_pH7_prediction_input.csv", "histidine_pH7_final_predictions.csv", "B22_corr1_log10"),
    ("histidine_pH6p5", "histidine_pH6p5_prediction_input.csv", "histidine_pH6p5_final_predictions.csv", "B22_corr1_log10"),
    ("basic_control_ALL80", "basic_control_prediction_input_ALL80.csv", "basic_control_final_predictions_ALL80.csv", "B22_corr1_log10"),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(FIG5_ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def soft_gate(x, center, scale):
    return 1.0 / (1.0 + np.exp(-((np.asarray(x, dtype=float) - center) / scale)))


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]; p = p[ok]
    n = len(y)
    return {
        "n": int(n),
        "R2": float(r2_score(y, p)) if n > 1 else np.nan,
        "RMSE": float(mean_squared_error(y, p) ** 0.5) if n else np.nan,
        "MAE": float(mean_absolute_error(y, p)) if n else np.nan,
        "Pearson_r": float(stats.pearsonr(y, p).statistic) if n > 2 else np.nan,
        "Spearman_rho": float(stats.spearmanr(y, p).statistic) if n > 2 else np.nan,
    }


def load_frozen_model():
    import tensorflow as tf
    required = ["MODEL_MANIFEST.json", "base_model_ensemble.joblib", "residual_preprocessor.joblib", "residual_nn.keras"]
    missing = [MODEL_DIR / x for x in required if not (MODEL_DIR / x).exists()]
    if missing:
        raise RuntimeError("Missing frozen model artifacts: " + ", ".join(map(str, missing)))
    manifest = json.loads((MODEL_DIR / "MODEL_MANIFEST.json").read_text())
    base = joblib.load(MODEL_DIR / "base_model_ensemble.joblib")
    prep = joblib.load(MODEL_DIR / "residual_preprocessor.joblib")
    residual = tf.keras.models.load_model(MODEL_DIR / "residual_nn.keras")
    hashes = {x: sha256(MODEL_DIR / x) for x in required}
    if (MODEL_DIR / "predict_final_hybrid.py").exists():
        hashes["predict_final_hybrid.py"] = sha256(MODEL_DIR / "predict_final_hybrid.py")
    return manifest, base, prep, residual, hashes


def predict_table(input_path: Path, dataset: str, manifest: dict, base, prep, residual, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    features = manifest["feature_order"]
    missing = [c for c in features if c not in df.columns]
    if "name" not in df.columns:
        missing = ["name", *missing]
    if missing:
        raise RuntimeError(f"{dataset}: missing required columns {missing}")
    if df["name"].duplicated().any():
        raise RuntimeError(f"{dataset}: duplicate name values detected")
    feature_values = df[features].to_numpy(float)
    if not np.isfinite(feature_values).all():
        raise RuntimeError(f"{dataset}: non-finite feature values detected")
    Xb = base["scaler"].transform(feature_values)
    base_pred = np.mean([m.predict(Xb) for m in base["models"]], axis=0)
    Xr = prep["scaler"].transform(feature_values)
    raw = residual.predict(Xr, verbose=0).reshape(-1)
    minus_ncp = df[manifest["minus_NCP_feature"]].to_numpy(float)
    gate = soft_gate(minus_ncp, manifest["gate_center"], manifest["gate_scale"])
    clipped = np.clip(raw, -manifest["residual_cap"], manifest["residual_cap"])
    uncapped = manifest["lambda_final"] * gate * raw
    applied = manifest["lambda_final"] * gate * clipped
    final = base_pred + applied
    out = df.copy()
    out.insert(0, "control_dataset", dataset)
    out["base_prediction"] = base_pred
    out["raw_residual_prediction"] = raw
    out["gate_value"] = gate
    out["lambda_final"] = manifest["lambda_final"]
    out["uncapped_correction"] = uncapped
    out["applied_correction"] = applied
    out["final_hybrid_prediction"] = final
    out["prediction_identity_error"] = np.abs(out["final_hybrid_prediction"] - (out["base_prediction"] + out["applied_correction"]))
    out["model_manifest_sha256"] = sha256(MODEL_DIR / "MODEL_MANIFEST.json")
    out["base_model_sha256"] = sha256(MODEL_DIR / "base_model_ensemble.joblib")
    out["residual_preprocessor_sha256"] = sha256(MODEL_DIR / "residual_preprocessor.joblib")
    out["residual_model_sha256"] = sha256(MODEL_DIR / "residual_nn.keras")
    if target_col in out.columns:
        out["residual"] = out[target_col] - out["final_hybrid_prediction"]
        out["absolute_error"] = np.abs(out["residual"])
    if float(out["prediction_identity_error"].max()) > 1e-10:
        raise RuntimeError(f"{dataset}: final != base + applied correction")
    return out


def development_identity_check() -> dict:
    cluster = pd.read_csv(OUTPUT_DIR / "development_917_cluster_predictions.csv")
    ref = pd.read_csv(INPUT_DIR / "development_917_reference_predictions.csv")
    merged = cluster.merge(ref, left_on="name", right_on="pair_id", suffixes=("_cluster", "_reference"), validate="one_to_one")
    pairs = {
        "base_prediction": ("base_prediction_cluster", "base_prediction_reference"),
        "raw_residual_prediction": ("raw_residual_prediction_cluster", "raw_residual_prediction_reference"),
        "gate": ("gate_value_cluster", "gate_value_reference"),
        "applied_correction": ("applied_correction_cluster", "applied_correction_reference"),
        "final_hybrid_prediction": ("final_hybrid_prediction_cluster", "final_hybrid_prediction_reference"),
    }
    diffs = {k: float(np.max(np.abs(merged[a] - merged[b]))) for k, (a, b) in pairs.items()}
    lines = ["development_917_cluster_identity_check", f"rows_compared={len(merged)}"]
    lines += [f"max_abs_diff_{k}={v:.16g}" for k, v in diffs.items()]
    lines.append("This is a provenance check only. The model was not modified, retrained, or recalibrated in response to this result.")
    (OUTPUT_DIR / "development_917_cluster_identity_check.txt").write_text("\n".join(lines) + "\n")
    return diffs


def main():
    manifest, base, prep, residual, hashes = load_frozen_model()
    outputs = {}
    missing_feature_count = 0
    for dataset, input_name, output_name, target_col in DATASETS:
        input_path = INPUT_DIR / input_name
        pred = predict_table(input_path, dataset, manifest, base, prep, residual, target_col)
        pred.to_csv(OUTPUT_DIR / output_name, index=False)
        outputs[dataset] = pred
        missing_feature_count += int(pred[manifest["feature_order"]].isna().sum().sum())
    identity_diffs = development_identity_check()

    all_controls = pd.concat([outputs[k] for k in outputs if k != "development_917"], ignore_index=True, sort=False)
    all_controls.to_csv(OUTPUT_DIR / "S7_all_control_final_predictions.csv", index=False)

    nb_shuffle = outputs["nonbasic_shuffle"]
    h7 = outputs["histidine_pH7"]
    h65 = outputs["histidine_pH6p5"]
    basic = outputs["basic_control_ALL80"]
    feature_diff = (h7[manifest["feature_order"]].to_numpy(float) != h65[manifest["feature_order"]].to_numpy(float)).sum(axis=0)
    feature_diff_summary = dict(zip(manifest["feature_order"], [int(x) for x in feature_diff]))
    target_eval = basic["target_evaluable"].astype(bool) if "target_evaluable" in basic.columns else np.isfinite(basic.get("B22_corr1_log10", np.nan))

    audit = [
        "FINAL CONTROL PREDICTION AUDIT",
        f"fig5_root={FIG5_ROOT}",
        f"model_dir={rel(MODEL_DIR)}",
        f"feature_order_loaded={manifest['feature_order']}",
        f"lambda_loaded={manifest['lambda_final']}",
        f"cap_loaded={manifest['residual_cap']}",
        f"gate_center={manifest['gate_center']}",
        f"gate_scale={manifest['gate_scale']}",
        f"model_artifact_hashes={json.dumps(hashes, sort_keys=True)}",
        f"missing_feature_count={missing_feature_count}",
        f"maximum_prediction_identity_error={float(all_controls['prediction_identity_error'].max()):.16g}",
        f"development_identity_diffs={json.dumps(identity_diffs, sort_keys=True)}",
        f"nonbasic_WT n={len(outputs['nonbasic_WT'])}",
        f"shuffle_WT_parent n={len(outputs['shuffle_WT_parent'])}",
        f"nonbasic_shuffle total n={len(nb_shuffle)}",
        f"TAD shuffle n={int((nb_shuffle['control_subtype'] == 'TAD_shuffle').sum())}",
        f"RGG shuffle n={int((nb_shuffle['control_subtype'] == 'RGG_shuffle').sum())}",
        f"histidine pH7 n={len(h7)}",
        f"histidine pH6.5 n={len(h65)}",
        f"histidine feature differences pH7_vs_pH6p5={json.dumps(feature_diff_summary, sort_keys=True)}",
        f"basic total n={len(basic)}",
        f"basic target-evaluable n={int(target_eval.sum())}",
        f"basic target-undefined positive-B22 n={int((~target_eval).sum())}",
    ]
    for dataset, df in outputs.items():
        if dataset == "development_917":
            continue
        if "B22_corr1_log10" in df.columns:
            evaluable = df[df["target_evaluable"].astype(bool)] if "target_evaluable" in df.columns else df[np.isfinite(df["B22_corr1_log10"])]
            audit.append(f"{dataset} metrics={json.dumps(metrics(evaluable['B22_corr1_log10'], evaluable['final_hybrid_prediction']), sort_keys=True)}")
    audit.append("No control dataset was used to fit, recalibrate, select, or modify the frozen model.")
    (OUTPUT_DIR / "FINAL_CONTROL_PREDICTION_AUDIT.txt").write_text("\n".join(audit) + "\n")
    print(f"Wrote final control predictions to {rel(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
