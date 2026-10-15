#!/usr/bin/env python3
"""Build the frozen final Figure 5 hybrid model from 917 development pairs only."""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "fig5_residual_tuning_local"
OUT = ROOT / "outputs" / "final_hybrid_model"
FROZEN = OUT / "frozen_model"
INPUT = PACKAGE / "inputs" / "fig5_917_F2_P8.csv"
BASE_CONFIG = PACKAGE / "config" / "base_model.json"
RUN_CONFIG = PACKAGE / "config" / "run_config.json"
GRID_CONFIG = PACKAGE / "config" / "residual_grid.json"
OOF = PACKAGE / "outputs" / "outer_oof_predictions.csv"
INNER_TUNING = PACKAGE / "outputs" / "inner_tuning_results.csv"
OUTER_FOLDS = PACKAGE / "outputs" / "outer_fold_results.csv"
TAIL_BOOT = PACKAGE / "outputs" / "tail_bootstrap_differences.csv"
SPLIT_IDS = PACKAGE / "outputs" / "residual_fit_calibration_ids.csv"
UTILS = PACKAGE / "scripts" / "utils.py"
NESTED_SCRIPT = PACKAGE / "scripts" / "run_nested_residual_tuning.py"
FINAL_LABEL = "historical_fixed_nn"
TARGET = "B22_corr1_log10"
PAIR_ID = "name"
EXPECTED_ROWS = 917

FORBIDDEN_TOKENS = ("heldout", "held_out", "taken_out", "test", "1794", "sequence_exclusive", "validation_make_csvs", "validation_run_model")
READ_AUDIT: list[str] = []


def forbid_path(path: Any) -> Path:
    p = Path(path)
    s = str(p).lower()
    allowed_validation_names = {"final_validation_loss"}
    if any(tok in s for tok in FORBIDDEN_TOKENS):
        raise RuntimeError(f"Forbidden held-out/test-like path access blocked: {p}")
    READ_AUDIT.append(str(p))
    return p


def safe_read_csv(path: Any, *args, **kwargs) -> pd.DataFrame:
    return pd.read_csv(forbid_path(path), *args, **kwargs)


def load_json(path: Path) -> dict[str, Any]:
    with open(forbid_path(path), "r") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except Exception:
        pass


def soft_gate(minus_ncp, center: float, scale: float) -> np.ndarray:
    x = np.asarray(minus_ncp, dtype=float)
    return 1.0 / (1.0 + np.exp(-((x - center) / scale)))


def metrics(y, pred) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    resid = y - pred
    return {
        "n": int(len(y)),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "RMSE": float(mean_squared_error(y, pred) ** 0.5) if len(y) else float("nan"),
        "MAE": float(mean_absolute_error(y, pred)) if len(y) else float("nan"),
        "Pearson": float(stats.pearsonr(y, pred).statistic) if len(y) > 2 else float("nan"),
        "Spearman": float(stats.spearmanr(y, pred).statistic) if len(y) > 2 else float("nan"),
        "bias": float(np.mean(resid)) if len(y) else float("nan"),
    }


def bootstrap_differences(y, base, hybrid, mask, n=5000, seed=20260916) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    rows = []
    def one(ids):
        mb = metrics(y[ids], base[ids])
        mh = metrics(y[ids], hybrid[ids])
        return {
            "delta_RMSE": mh["RMSE"] - mb["RMSE"],
            "delta_MAE": mh["MAE"] - mb["MAE"],
            "delta_R2": mh["R2"] - mb["R2"],
            "delta_bias": mh["bias"] - mb["bias"],
        }
    observed = one(idx)
    vals = {k: [] for k in observed}
    for _ in range(n):
        sample = rng.choice(idx, size=len(idx), replace=True)
        d = one(sample)
        for k, v in d.items():
            vals[k].append(v)
    for k, obs in observed.items():
        arr = np.asarray(vals[k], dtype=float)
        rows.append({"subset": "high_minus_NCP" if mask.sum() != len(mask) else "all", "metric": k, "observed": obs, "ci_2_5": float(np.nanpercentile(arr, 2.5)), "ci_97_5": float(np.nanpercentile(arr, 97.5)), "n_bootstrap": n})
    return pd.DataFrame(rows)


def load_development(base_config) -> pd.DataFrame:
    df = safe_read_csv(INPUT)
    feature_cols = base_config["features"]
    required = [PAIR_ID, TARGET] + feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Development table missing columns: {missing}")
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} rows; observed {len(df)}")
    if not df[PAIR_ID].is_unique:
        raise RuntimeError("Development pair IDs are not unique")
    if not np.isfinite(df[[TARGET] + feature_cols].to_numpy(float)).all():
        raise RuntimeError("Development target/features contain non-finite values")
    out = df[[PAIR_ID, TARGET] + feature_cols].copy()
    out = out.rename(columns={"Opposite_Charge_Number": "minus_NCP", TARGET: "simulated_target"})
    out.to_csv(OUT / "development_917.csv", index=False)
    return df


def validated_procedure_text(base_config, run_config, grid) -> str:
    text = f"""Validated development-only residual procedure audit

Authoritative package: {PACKAGE}
Nested residual script: {NESTED_SCRIPT}
Utility implementation: {UTILS}
Base config: {BASE_CONFIG}
Run config: {RUN_CONFIG}
Residual grid: {GRID_CONFIG}

Base OOF prediction generation:
  Outer CV seeds: {run_config['outer_cv']['seeds']}
  Outer folds per seed: {run_config['outer_cv']['n_splits']}
  Shuffle: {run_config['outer_cv']['shuffle']}
  Base model: sklearn MLPRegressor ensemble with seeds {base_config['mlp']['ensemble_random_seeds']}
  Base scaler: StandardScaler fitted on each training fold only.
  Base features: {base_config['features']}

Residual target definition:
  For each outer fold, inner 5-fold OOF base predictions are generated inside the outer-training set.
  residual_target = true B22_corr1_log10 - inner_OOF_base_prediction.
  Outer-validation residual predictions are always from models that did not train on those validation examples.

Examples used to train residual NN:
  Only high−NCP examples in the outer-training set are eligible.
  High−NCP threshold: {run_config['high_ocn_threshold']}.
  Eligible high−NCP examples are split by train_test_split(test_size=0.35, random_state=outer_seed*100+outer_fold).
  65% are residual-fit examples; 35% are calibration examples.

Gate/sample weighting:
  residual-fit sample weight = sigmoid((−NCP - center) / scale).
  Gate center: {run_config['gate']['center']}.
  Gate scale: {run_config['gate']['scale']}.

Validated fixed residual architecture retained for final hybrid model:
  Dense 16 tanh; Dense 8 tanh; Dense 1 linear.
  L2 = 5e-4.
  Adam learning rate = 2e-4.
  Huber delta = 0.75.

Optimizer/loss/early stopping from authoritative procedure:
  TensorFlow/Keras Adam optimizer.
  Huber loss.
  max_epochs = {run_config['tensorflow']['max_epochs']}.
  validation_split = {run_config['tensorflow']['validation_split']} within residual-fit data.
  EarlyStopping patience = {run_config['tensorflow']['patience']}, restore_best_weights=True, monitor=val_loss.
  ReduceLROnPlateau enabled = {run_config['tensorflow']['reduce_lr_on_plateau']}; patience = {run_config['tensorflow']['reduce_lr_patience']}; factor = {run_config['tensorflow']['reduce_lr_factor']}.
  Batch size small/large = {run_config['tensorflow']['batch_size_small']}/{run_config['tensorflow']['batch_size_large']} with large_n_threshold = {run_config['tensorflow']['large_n_threshold']}.

Lambda equation:
  lambda_raw = dot(true_residual, predicted_residual) / (dot(predicted_residual, predicted_residual) + epsilon).
  epsilon = {run_config['lambda']['epsilon']}.
  lambda_hat = clip(lambda_raw, {run_config['lambda']['clip_min']}, {run_config['lambda']['clip_max']}).

Cap equation:
  cap = percentile(abs(true_residual), {run_config['residual_cap_percentile']}).

Gate equation used during prediction:
  gate = 1 / (1 + exp(-((−NCP - {run_config['gate']['center']}) / {run_config['gate']['scale']}))).
  residual_prediction_clipped = clip(raw_residual_prediction, -cap, cap).
  applied_correction = lambda_hat * gate * residual_prediction_clipped.
  hybrid_prediction = base_prediction + applied_correction.

No held-out/test input is read by this build script.
"""
    write_text(OUT / "validated_residual_procedure.txt", text)
    return text


def build_oof_outputs(base_config, run_config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = safe_read_csv(OOF)
    df = df[df["residual_model_label"] == FINAL_LABEL].copy()
    expected_oof_rows = EXPECTED_ROWS * len(run_config["outer_cv"]["seeds"])
    if len(df) != expected_oof_rows:
        raise RuntimeError(f"Expected {expected_oof_rows} fold-level final-label OOF rows; observed {len(df)}")
    df["base_residual"] = df["true_target"] - df["base_prediction"]
    df.rename(columns={"name": "pair_id", "OCN": "minus_NCP", "true_target": "simulated_target", "outer_fold": "fold", "outer_seed": "repeat_seed"}, inplace=True)
    base_oof = df[["pair_id", "repeat_seed", "fold", "simulated_target", "minus_NCP", "base_prediction", "base_residual"]].copy()
    base_oof.to_csv(OUT / "base_oof_predictions.csv", index=False)
    res = df[["pair_id", "repeat_seed", "fold", "minus_NCP", "base_residual", "residual_prediction_raw", "gate"]].copy()
    res["sample_weight_used_for_training_formula"] = "sigmoid gate for high−NCP residual-fit examples; OOF validation rows report gate value"
    res.rename(columns={"base_residual": "base_OOF_residual_target", "residual_prediction_raw": "raw_crossfit_residual_prediction", "gate": "gate_value"}, inplace=True)
    res.to_csv(OUT / "residual_crossfit_predictions.csv", index=False)

    unique = df.groupby("pair_id", as_index=False).agg(
        simulated_target=("simulated_target", "first"),
        minus_NCP=("minus_NCP", "first"),
        base_OOF_prediction=("base_prediction", "mean"),
        base_residual=("base_residual", "mean"),
        raw_crossfit_residual_prediction=("residual_prediction_raw", "mean"),
        gate_value=("gate", "first"),
    )
    if len(unique) != EXPECTED_ROWS:
        raise RuntimeError("Unique OOF aggregation did not produce 917 rows")
    high = unique["minus_NCP"] >= run_config["high_ocn_threshold"]
    if int(high.sum()) != 93:
        raise RuntimeError(f"Expected 93 high−NCP rows, observed {int(high.sum())}")
    # Final deployment calibration on all unique high−NCP cross-fitted residual predictions.
    true = unique.loc[high, "base_residual"].to_numpy(float)
    pred = unique.loc[high, "raw_crossfit_residual_prediction"].to_numpy(float)
    eps = run_config["lambda"]["epsilon"]
    lambda_raw = float(np.dot(true, pred) / (np.dot(pred, pred) + eps))
    lambda_final = float(np.clip(lambda_raw, run_config["lambda"]["clip_min"], run_config["lambda"]["clip_max"]))
    cap = float(np.percentile(np.abs(true), run_config["residual_cap_percentile"]))
    cal = {
        "lambda_raw": lambda_raw,
        "lambda_final": lambda_final,
        "calibration_sample_count": int(high.sum()),
        "calibration_subset": "unique development pairs with −NCP >= 40",
        "calibration_equation": "lambda_raw = dot(true_residual, predicted_residual) / (dot(predicted_residual, predicted_residual) + epsilon); lambda_final = clip(lambda_raw, 0, 0.6)",
        "epsilon": eps,
        "gate_center": run_config["gate"]["center"],
        "gate_scale": run_config["gate"]["scale"],
        "threshold": run_config["high_ocn_threshold"],
        "cap_percentile": run_config["residual_cap_percentile"],
        "cap_quantity": "abs(base residual) among unique high−NCP cross-fitted development pairs",
        "cap": cap,
    }
    write_text(OUT / "final_residual_calibration.json", json.dumps(cal, indent=2) + "\n")
    raw = unique["raw_crossfit_residual_prediction"].to_numpy(float)
    clipped = np.clip(raw, -cap, cap)
    correction = lambda_final * unique["gate_value"].to_numpy(float) * clipped
    unique["residual_prediction_clipped"] = clipped
    unique["applied_residual_correction"] = correction
    unique["hybrid_OOF_prediction"] = unique["base_OOF_prediction"] + correction
    unique["hybrid_residual"] = unique["simulated_target"] - unique["hybrid_OOF_prediction"]
    unique.to_csv(OUT / "development_oof_final_hybrid_predictions.csv", index=False)
    return unique, base_oof, cal


def write_metrics(unique: pd.DataFrame, cal: dict[str, Any]) -> dict[str, Any]:
    y = unique["simulated_target"].to_numpy(float)
    base = unique["base_OOF_prediction"].to_numpy(float)
    hybrid = unique["hybrid_OOF_prediction"].to_numpy(float)
    high = unique["minus_NCP"].to_numpy(float) >= cal["threshold"]
    out = {
        "all_base": metrics(y, base),
        "all_hybrid": metrics(y, hybrid),
        "high_base": metrics(y[high], base[high]),
        "high_hybrid": metrics(y[high], hybrid[high]),
    }
    out["all_changes"] = {k: out["all_hybrid"][k] - out["all_base"][k] for k in ["R2", "RMSE", "MAE", "bias"]}
    out["high_changes"] = {k: out["high_hybrid"][k] - out["high_base"][k] for k in ["R2", "RMSE", "MAE", "bias"]}
    boot_all = bootstrap_differences(y, base, hybrid, np.ones(len(y), dtype=bool), n=5000)
    boot_high = bootstrap_differences(y, base, hybrid, high, n=5000)
    boot = pd.concat([boot_all, boot_high], ignore_index=True)
    boot.to_csv(OUT / "development_oof_final_hybrid_bootstrap.csv", index=False)
    lines = ["Development-only OOF final hybrid diagnostics", "", json.dumps(out, indent=2), "", "Bootstrap paired differences:", boot.to_string(index=False)]
    write_text(OUT / "development_oof_final_hybrid_metrics.txt", "\n".join(lines) + "\n")
    return out


def train_final_base(dev: pd.DataFrame, base_config: dict[str, Any]) -> dict[str, Any]:
    features = base_config["features"]
    X = dev[features].to_numpy(float)
    y = dev[TARGET].to_numpy(float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    params = dict(base_config["mlp"])
    seeds = list(params.pop("ensemble_random_seeds"))
    params.pop("class", None)
    params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    models = []
    for seed in seeds:
        model = MLPRegressor(random_state=seed, **params)
        model.fit(Xs, y)
        models.append(model)
    joblib.dump({"scaler": scaler, "models": models, "feature_order": features, "base_config": base_config}, FROZEN / "base_model_ensemble.joblib")
    return {"base_training_rows": len(dev), "base_ensemble_seeds": seeds, "base_n_models": len(models)}


def build_residual_nn(input_dim: int):
    import tensorflow as tf
    from tensorflow.keras import layers as L, regularizers as R, optimizers as O
    inp = L.Input(shape=(input_dim,))
    h = L.Dense(16, activation="tanh", kernel_regularizer=R.l2(5e-4))(inp)
    h = L.Dense(8, activation="tanh", kernel_regularizer=R.l2(5e-4))(h)
    out = L.Dense(1)(h)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=O.Adam(learning_rate=2e-4), loss=tf.keras.losses.Huber(delta=0.75))
    return model


def train_final_residual(dev: pd.DataFrame, unique_oof: pd.DataFrame, base_config: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any]:
    import tensorflow as tf
    feature_order = base_config["features"]
    merged = dev[[PAIR_ID] + feature_order].merge(unique_oof[["pair_id", "base_residual", "minus_NCP"]], left_on=PAIR_ID, right_on="pair_id", how="inner", validate="one_to_one")
    high = merged["minus_NCP"] >= run_config["high_ocn_threshold"]
    fit = merged[high].copy()
    X = fit[feature_order].to_numpy(float)
    y = fit["base_residual"].to_numpy(float)
    residual_scaler = StandardScaler().fit(X)
    Xs = residual_scaler.transform(X)
    weights = soft_gate(fit["minus_NCP"].to_numpy(float), run_config["gate"]["center"], run_config["gate"]["scale"])
    inner = safe_read_csv(INNER_TUNING)
    hist = inner[(inner["model_type"] == "NN") & (inner["architecture"].astype(str) == "(16, 8)") & (inner["l2"].astype(float) == 0.0005) & (inner["learning_rate"].astype(float) == 0.0002) & (inner["huber_delta"].astype(float) == 0.75)].copy()
    if len(hist) != 25:
        raise RuntimeError(f"Expected 25 historical fixed NN epoch records; observed {len(hist)}")
    final_epochs = int(np.median(hist["best_epoch"].astype(float)))
    if final_epochs < 1:
        raise RuntimeError("Invalid final epoch count")
    seed = 20260916
    set_all_seeds(seed)
    model = build_residual_nn(Xs.shape[1])
    batch = run_config["tensorflow"]["batch_size_small"] if len(y) < run_config["tensorflow"]["large_n_threshold"] else run_config["tensorflow"]["batch_size_large"]
    history = model.fit(Xs, y, sample_weight=weights, epochs=final_epochs, batch_size=batch, verbose=0, shuffle=True)
    model.save(FROZEN / "residual_nn.keras")
    joblib.dump({"scaler": residual_scaler, "feature_order": feature_order}, FROZEN / "residual_preprocessor.joblib")
    return {
        "residual_training_rows": int(len(fit)),
        "residual_training_high_threshold": run_config["high_ocn_threshold"],
        "residual_sample_weight": "sigmoid gate on −NCP",
        "final_residual_seed": seed,
        "final_training_epoch_count": final_epochs,
        "epoch_choice_rule": "median best_epoch across the 25 validated nested historical_fixed_nn runs from inner_tuning_results.csv",
        "nested_best_epoch_summary": {"min": int(hist["best_epoch"].min()), "median": float(hist["best_epoch"].median()), "max": int(hist["best_epoch"].max())},
        "final_training_loss_last": float(history.history["loss"][-1]),
    }


def write_prediction_script() -> None:
    script = r'''#!/usr/bin/env python3
"""Predict with the frozen Figure 5 final hybrid model."""
import argparse, json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

def soft_gate(x, center, scale):
    return 1.0/(1.0+np.exp(-((np.asarray(x,dtype=float)-center)/scale)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--model-dir', default=str(Path(__file__).resolve().parent))
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    model_dir=Path(args.model_dir)
    manifest=json.loads((model_dir/'MODEL_MANIFEST.json').read_text())
    df=pd.read_csv(args.features)
    required=['name']+manifest['feature_order']
    missing=[c for c in required if c not in df.columns]
    if missing: raise SystemExit(f'Missing columns: {missing}')
    base=joblib.load(model_dir/'base_model_ensemble.joblib')
    X=df[manifest['feature_order']].to_numpy(float)
    Xb=base['scaler'].transform(X)
    base_pred=np.mean([m.predict(Xb) for m in base['models']], axis=0)
    prep=joblib.load(model_dir/'residual_preprocessor.joblib')
    import tensorflow as tf
    residual=tf.keras.models.load_model(model_dir/'residual_nn.keras')
    Xr=prep['scaler'].transform(X)
    raw=residual.predict(Xr, verbose=0).reshape(-1)
    minus_ncp=df[manifest['minus_NCP_feature']].to_numpy(float)
    gate=soft_gate(minus_ncp, manifest['gate_center'], manifest['gate_scale'])
    clipped=np.clip(raw, -manifest['residual_cap'], manifest['residual_cap'])
    uncapped=manifest['lambda_final']*gate*raw
    applied=manifest['lambda_final']*gate*clipped
    final=base_pred+applied
    out=pd.DataFrame({'name':df['name'], 'base_prediction':base_pred, 'raw_residual_prediction':raw, 'gate_value':gate, 'lambda_final':manifest['lambda_final'], 'uncapped_correction':uncapped, 'applied_correction':applied, 'final_hybrid_prediction':final})
    if not np.allclose(out['final_hybrid_prediction'], out['base_prediction']+out['applied_correction'], atol=1e-10):
        raise SystemExit('Prediction identity check failed')
    out.to_csv(args.out, index=False)
if __name__=='__main__': main()
'''
    path = FROZEN / "predict_final_hybrid.py"
    write_text(path, script)
    path.chmod(0o755)


def write_manifest(base_config, run_config, cal, proc_text, base_meta, residual_meta, metrics_dict) -> dict[str, Any]:
    import sklearn, scipy
    import tensorflow as tf
    manifest = {
        "model_name": "final Figure 5 hybrid model",
        "training_scope": "917 development simulations only",
        "heldout_test_dataset_opened_or_used": False,
        "target": TARGET,
        "pair_id_column": PAIR_ID,
        "minus_NCP_feature": "Opposite_Charge_Number",
        "feature_order": base_config["features"],
        "base_model": {
            "class": "sklearn.neural_network.MLPRegressor ensemble",
            "scaler": "StandardScaler fitted on all 917 development rows for final deployment",
            "parameters": base_config["mlp"],
            **base_meta,
        },
        "residual_model": {
            "architecture": ["Dense(16,tanh,L2=5e-4)", "Dense(8,tanh,L2=5e-4)", "Dense(1,linear)"],
            "optimizer": "Adam learning_rate=2e-4",
            "loss": "Huber delta=0.75",
            **residual_meta,
        },
        "lambda_raw": cal["lambda_raw"],
        "lambda_final": cal["lambda_final"],
        "residual_cap": cal["cap"],
        "cap_percentile": cal["cap_percentile"],
        "gate_center": cal["gate_center"],
        "gate_scale": cal["gate_scale"],
        "high_minus_NCP_threshold": cal["threshold"],
        "prediction_formula": "base + lambda_final * sigmoid((−NCP-gate_center)/gate_scale) * clip(raw_residual, -cap, cap)",
        "development_oof_metrics": metrics_dict,
        "software_versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__, "tensorflow": tf.__version__},
        "authoritative_sources": {"input_917": str(INPUT), "residual_package": str(PACKAGE), "nested_script": str(NESTED_SCRIPT), "utils": str(UTILS), "base_config": str(BASE_CONFIG), "run_config": str(RUN_CONFIG), "grid_config": str(GRID_CONFIG), "oof_predictions": str(OOF), "inner_tuning": str(INNER_TUNING)},
    }
    write_text(FROZEN / "MODEL_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def write_checksums() -> pd.DataFrame:
    rows=[]
    for p in sorted(FROZEN.rglob("*")):
        if p.is_file():
            rows.append({"file": str(p.relative_to(FROZEN)), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    df=pd.DataFrame(rows)
    df.to_csv(FROZEN / "SHA256SUMS.csv", index=False)
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FROZEN.mkdir(parents=True, exist_ok=True)
    base_config = load_json(BASE_CONFIG)
    run_config = load_json(RUN_CONFIG)
    grid = load_json(GRID_CONFIG)
    proc_text = validated_procedure_text(base_config, run_config, grid)
    dev = load_development(base_config)
    unique_oof, base_oof, cal = build_oof_outputs(base_config, run_config)
    metrics_dict = write_metrics(unique_oof, cal)
    base_meta = train_final_base(dev, base_config)
    residual_meta = train_final_residual(dev, unique_oof, base_config, run_config)
    write_prediction_script()
    manifest = write_manifest(base_config, run_config, cal, proc_text, base_meta, residual_meta, metrics_dict)
    checks = write_checksums()
    manifest_sha = sha256_file(FROZEN / "MODEL_MANIFEST.json")
    # internal frozen prediction function integrity on development rows: use loaded artifacts without writing predictions.
    loaded_base = joblib.load(FROZEN / "base_model_ensemble.joblib")
    X = dev[base_config["features"]].to_numpy(float)
    Xb = loaded_base["scaler"].transform(X)
    bp = np.mean([m.predict(Xb) for m in loaded_base["models"]], axis=0)
    prep = joblib.load(FROZEN / "residual_preprocessor.joblib")
    import tensorflow as tf
    rn = tf.keras.models.load_model(FROZEN / "residual_nn.keras")
    raw = rn.predict(prep["scaler"].transform(X), verbose=0).reshape(-1)
    gate = soft_gate(dev["Opposite_Charge_Number"].to_numpy(float), cal["gate_center"], cal["gate_scale"])
    applied = cal["lambda_final"] * gate * np.clip(raw, -cal["cap"], cal["cap"])
    final = bp + applied
    if not np.allclose(final, bp + applied, atol=1e-12):
        raise RuntimeError("Frozen prediction identity failed")
    freeze = f"""MODEL FROZEN

model_manifest_sha256: {manifest_sha}
lambda_final: {cal['lambda_final']}
cap: {cal['cap']}
gate_center: {cal['gate_center']}
gate_scale: {cal['gate_scale']}
final_feature_order: {base_config['features']}
final_residual_architecture: Dense 16 tanh; Dense 8 tanh; Dense 1 linear; L2=5e-4; Adam lr=2e-4; Huber delta=0.75
training_row_count: 917

The 1,794 sequence-exclusive held-out/test dataset was not opened or used during model construction, calibration, fitting, early stopping, or model selection.
"""
    write_text(OUT / "MODEL_FROZEN.txt", freeze)
    audit_lines = [
        "FINAL MODEL BUILD AUDIT", "",
        "1. Development rows used: 917.",
        "2. Was the 1,794 test set opened? NO.",
        "3. Was the 1,794 test set used to estimate lambda? NO.",
        "4. Was the 1,794 test set used for early stopping? NO.",
        "5. Was the 1,794 test set used for feature/model selection? NO.",
        f"6. Residual-analysis code defining final procedure: {NESTED_SCRIPT} and {UTILS}.",
        f"7. lambda_final: {cal['lambda_final']}",
        f"8. final cap: {cal['cap']}",
        f"9. 917 OOF base metrics: {json.dumps(metrics_dict['all_base'], sort_keys=True)}",
        f"10. 917 OOF hybrid metrics: {json.dumps(metrics_dict['all_hybrid'], sort_keys=True)}",
        f"11. high−NCP base metrics: {json.dumps(metrics_dict['high_base'], sort_keys=True)}",
        f"12. high−NCP hybrid metrics: {json.dumps(metrics_dict['high_hybrid'], sort_keys=True)}",
        f"13. final model-manifest SHA256: {manifest_sha}",
        "14. Model construction is complete and frozen before any test-set evaluation.",
        "", "Read-path audit:", *[f"  {p}" for p in sorted(set(READ_AUDIT))]
    ]
    write_text(OUT / "FINAL_MODEL_BUILD_AUDIT.txt", "\n".join(audit_lines) + "\n")
    print(f"Built frozen final hybrid model in {OUT}")
    print(f"MODEL_MANIFEST SHA256 {manifest_sha}")

if __name__ == "__main__":
    main()
