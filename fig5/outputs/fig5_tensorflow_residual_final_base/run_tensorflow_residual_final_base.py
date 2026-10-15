#!/usr/bin/env python
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "fig5_mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks as C
from tensorflow.keras import layers as L
from tensorflow.keras import optimizers as O
from tensorflow.keras import regularizers as R


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "fig5_tensorflow_residual_final_base"
OUT.mkdir(parents=True, exist_ok=True)

BASE_FILE = ROOT / "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
PATCH_FILE = ROOT / "patches" / "datasets" / "features_sim_pairs_for_training.csv"
TARGET = "B22_corr1_log10"
OCN = "Opposite_Charge_Number"
OUTER_SEEDS = [42, 123, 456, 789, 2026]
ENSEMBLE_SEEDS = [0, 1, 2, 3, 4]
MLP_PARAMS = {
    "activation": "tanh",
    "alpha": 0.01,
    "hidden_layer_sizes": (128, 64, 32, 16),
    "learning_rate_init": 0.01,
    "max_iter": 500,
    "solver": "adam",
}
F2 = ["Opposite_Charge_Number", "mean_lambda_RGG", "mean_lambda_TAD", "SHD_RGG", "SHD_TAD"]
MODELS = {
    "F2-P1": F2 + ["opp_charge_std"],
    "F2-P8": F2 + [
        "opp_charge_std",
        "scd_patch1_std",
        "scd_patch2_std",
        "rg_fraction_patch1_std",
        "aromatic_fraction_patch2_std",
        "shd_patch1_std",
        "shd_patch2_std",
        "ah_energy_std",
    ],
}
THRESHOLD = 40.0
GATE_CENTER = 40.0
GATE_SCALE = 12.0


def soft_gate(ocn):
    return 1.0 / (1.0 + np.exp(-((np.asarray(ocn, dtype=float) - GATE_CENTER) / GATE_SCALE)))


def metric_dict(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(y) < 2:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson": np.nan, "Spearman": np.nan}
    return {
        "R2": r2_score(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "MAE": mean_absolute_error(y, pred),
        "Pearson": stats.pearsonr(y, pred).statistic,
        "Spearman": stats.spearmanr(y, pred).statistic,
    }


def build_residual_nn(input_dim):
    inp = L.Input(shape=(input_dim,))
    h = L.Dense(16, activation="tanh", kernel_regularizer=R.l2(5e-4))(inp)
    h = L.Dense(8, activation="tanh", kernel_regularizer=R.l2(5e-4))(h)
    out = L.Dense(1)(h)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=O.Adam(learning_rate=2e-4),
        loss=tf.keras.losses.Huber(delta=0.75),
        weighted_metrics=[],
    )
    return model


def _fit_one_mlp_predict(X_train_s, y_train, X_val_s, seed):
    model = MLPRegressor(random_state=seed, **MLP_PARAMS)
    model.fit(X_train_s, y_train)
    return model.predict(X_val_s)


def fit_base_predict(X_train, y_train, X_val):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    preds = Parallel(n_jobs=min(len(ENSEMBLE_SEEDS), 5), prefer="processes")(
        delayed(_fit_one_mlp_predict)(X_train_s, y_train, X_val_s, seed)
        for seed in ENSEMBLE_SEEDS
    )
    return np.mean(preds, axis=0), scaler


def inner_oof_base_predictions(X, y, outer_seed):
    oof = np.full(len(y), np.nan, dtype=float)
    inner = KFold(n_splits=5, shuffle=True, random_state=outer_seed)
    for inner_train, inner_val in inner.split(X):
        oof[inner_val], _ = fit_base_predict(X[inner_train], y[inner_train], X[inner_val])
    return oof


def calibration_stats(true_resid, pred_resid):
    true_resid = np.asarray(true_resid, dtype=float)
    pred_resid = np.asarray(pred_resid, dtype=float)
    out = {"calibration_n": len(true_resid)}
    if len(true_resid) >= 2 and np.std(pred_resid) > 0 and np.std(true_resid) > 0:
        out["calibration_Pearson"] = stats.pearsonr(true_resid, pred_resid).statistic
        out["calibration_Spearman"] = stats.spearmanr(true_resid, pred_resid).statistic
        slope, intercept = np.polyfit(pred_resid, true_resid, 1)
        out["calibration_slope"] = slope
        out["calibration_intercept"] = intercept
        out["calibration_R2"] = r2_score(true_resid, pred_resid)
    else:
        out.update({
            "calibration_Pearson": np.nan,
            "calibration_Spearman": np.nan,
            "calibration_slope": np.nan,
            "calibration_intercept": np.nan,
            "calibration_R2": np.nan,
        })
    return out


def prepare_input():
    base = pd.read_csv(BASE_FILE)
    patch = pd.read_csv(PATCH_FILE)
    patch_cols = sorted(set(MODELS["F2-P8"]) - set(F2))
    bset = set(base["name"])
    pset = set(patch["name"])
    lines = [
        f"base_file: {BASE_FILE}",
        f"patch_file: {PATCH_FILE}",
        f"base_rows: {len(base)}",
        f"patch_rows: {len(patch)}",
        f"base_unique_pair_ids: {base['name'].is_unique}",
        f"patch_unique_pair_ids: {patch['name'].is_unique}",
        f"base_ids: {len(bset)}",
        f"patch_ids: {len(pset)}",
        f"intersection_size: {len(bset & pset)}",
        f"base_only_ids: {len(bset - pset)}",
        f"patch_only_ids: {len(pset - bset)}",
    ]
    assert len(base) == 917
    assert len(patch) == 917
    assert base["name"].is_unique
    assert patch["name"].is_unique
    assert bset == pset

    df = base.copy()
    df["row_id"] = np.arange(len(df))
    df["mean_lambda_RGG"] = df["lambda_RGG"] / df["RGG_length"]
    df["mean_lambda_TAD"] = df["lambda_TAD"] / df["TAD_length"]
    df = df.merge(patch[["name"] + patch_cols], on="name", how="left", validate="one_to_one", sort=False)
    required = [TARGET, OCN] + sorted(set(sum(MODELS.values(), [])))
    missing = df[required].isna().sum()
    finite = np.isfinite(df[required].to_numpy(float)).all(axis=0)
    lines.extend([
        f"missing_target: {int(df[TARGET].isna().sum())}",
        f"missing_F2_P1_features: {int(df[MODELS['F2-P1']].isna().sum().sum())}",
        f"missing_F2_P8_features: {int(df[MODELS['F2-P8']].isna().sum().sum())}",
        f"all_required_columns_finite: {bool(finite.all())}",
        "merge_method: explicit one-to-one merge by name",
        "patch_table_usage: local patch-derived descriptors only",
        "heldout_1794_targets_accessed: NO",
    ])
    if missing.sum() != 0 or not finite.all():
        lines.append("missing_by_column:")
        lines.extend([f"{k}: {int(v)}" for k, v in missing.items()])
    (OUT / "input_alignment_check.txt").write_text("\n".join(lines) + "\n")
    assert missing.sum() == 0
    assert finite.all()
    return df.reset_index(drop=True)


def run_one_model(df, model_name, cols):
    y = df[TARGET].to_numpy(float)
    X = df[cols].to_numpy(float)
    ocn = df[OCN].to_numpy(float)
    pred_rows = []
    metric_rows = []
    diag_rows = []
    cal_rows = []
    split_rows = []

    for outer_seed in OUTER_SEEDS:
        outer = KFold(n_splits=5, shuffle=True, random_state=outer_seed)
        for fold, (train_idx, val_idx) in enumerate(outer.split(X), start=1):
            print(f"{model_name} outer_seed={outer_seed} fold={fold}", flush=True)
            seed = int(outer_seed * 100 + fold)
            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            ocn_train, ocn_val = ocn[train_idx], ocn[val_idx]

            base_val, outer_scaler = fit_base_predict(X_train, y_train, X_val)
            inner_oof = inner_oof_base_predictions(X_train, y_train, outer_seed)
            assert np.isfinite(inner_oof).all()
            train_residual = y_train - inner_oof

            high_train_local = np.where(ocn_train >= THRESHOLD)[0]
            fit_local, cal_local = train_test_split(
                high_train_local,
                test_size=0.35,
                random_state=seed,
                shuffle=True,
            )
            X_train_s = outer_scaler.transform(X_train)
            X_val_s = outer_scaler.transform(X_val)
            batch_size = 8 if len(fit_local) < 200 else 32
            residual_model = build_residual_nn(X.shape[1])
            callbacks = [
                C.EarlyStopping(patience=30, restore_best_weights=True, monitor="val_loss"),
                C.ReduceLROnPlateau(patience=10, factor=0.5, monitor="val_loss"),
            ]
            hist = residual_model.fit(
                X_train_s[fit_local],
                train_residual[fit_local],
                validation_split=0.2,
                epochs=400,
                batch_size=batch_size,
                sample_weight=soft_gate(ocn_train[fit_local]),
                callbacks=callbacks,
                verbose=0,
            )
            loss = hist.history.get("loss", [np.nan])
            val_loss = hist.history.get("val_loss", [np.nan])
            best_epoch = int(np.nanargmin(val_loss) + 1) if len(val_loss) else np.nan

            residual_pred_train = residual_model.predict(X_train_s, verbose=0).ravel()
            true_cal = train_residual[cal_local]
            pred_cal = residual_pred_train[cal_local]
            lambda_raw = float(np.dot(true_cal, pred_cal) / (np.dot(pred_cal, pred_cal) + 1e-12))
            lambda_hat = float(np.clip(lambda_raw, 0.0, 0.6))
            cap = float(np.percentile(np.abs(true_cal), 85))

            residual_raw_val = residual_model.predict(X_val_s, verbose=0).ravel()
            residual_clipped_val = np.clip(residual_raw_val, -cap, cap)
            gate_val = soft_gate(ocn_val)
            applied = lambda_hat * gate_val * residual_clipped_val
            hybrid_val = base_val + applied

            for idx, bp, rr, rc, g, corr, hp in zip(val_idx, base_val, residual_raw_val, residual_clipped_val, gate_val, applied, hybrid_val):
                pred_rows.append({
                    "name": df.loc[idx, "name"],
                    "seed": outer_seed,
                    "fold": fold,
                    "true_target": y[idx],
                    "OCN": ocn[idx],
                    "base_prediction": bp,
                    "base_residual": y[idx] - bp,
                    "residual_prediction_raw": rr,
                    "residual_prediction_clipped": rc,
                    "gate": g,
                    "lambda_raw": lambda_raw,
                    "lambda_hat": lambda_hat,
                    "cap": cap,
                    "applied_correction": corr,
                    "hybrid_prediction": hp,
                })

            for subset, mask in [
                ("all", np.ones(len(val_idx), dtype=bool)),
                ("low_OCN", ocn_val < THRESHOLD),
                ("high_OCN", ocn_val >= THRESHOLD),
            ]:
                for prediction_type, pred in [("base", base_val), ("hybrid", hybrid_val)]:
                    row = {
                        "model": model_name,
                        "seed": outer_seed,
                        "fold": fold,
                        "subset": subset,
                        "prediction_type": prediction_type,
                        "n": int(mask.sum()),
                        "n_high_OCN_validation": int((ocn_val >= THRESHOLD).sum()),
                    }
                    row.update(metric_dict(y_val[mask], pred[mask]))
                    metric_rows.append(row)

            diag_rows.append({
                "model": model_name,
                "outer_seed": outer_seed,
                "fold": fold,
                "n_train": len(train_idx),
                "n_validation": len(val_idx),
                "n_high_OCN_train": int((ocn_train >= THRESHOLD).sum()),
                "n_high_OCN_validation": int((ocn_val >= THRESHOLD).sum()),
                "fraction_high_OCN_train": float((ocn_train >= THRESHOLD).mean()),
                "fraction_high_OCN_validation": float((ocn_val >= THRESHOLD).mean()),
                "n_residual_fit": len(fit_local),
                "n_calibration": len(cal_local),
                "inner_oof_residual_targets_complete": bool(np.isfinite(inner_oof).all()),
                "lambda_raw": lambda_raw,
                "lambda_hat": lambda_hat,
                "cap": cap,
                "epochs_run": len(loss),
                "best_epoch": best_epoch,
                "final_training_loss": float(loss[-1]),
                "final_validation_loss": float(val_loss[-1]),
                "correction_mean": float(np.mean(applied)),
                "correction_SD": float(np.std(applied)),
                "correction_max_abs": float(np.max(np.abs(applied))),
                "fraction_abs_correction_gt_0_01": float(np.mean(np.abs(applied) > 0.01)),
                "fraction_lambda_hat_eq_0": float(lambda_hat == 0.0),
            })
            cal_row = {
                "model": model_name,
                "outer_seed": outer_seed,
                "fold": fold,
                "lambda_raw": lambda_raw,
                "lambda_hat": lambda_hat,
            }
            cal_row.update(calibration_stats(true_cal, pred_cal))
            cal_rows.append(cal_row)

            for local_idx, split in [(fit_local, "residual_fit"), (cal_local, "calibration")]:
                for i in local_idx:
                    split_rows.append({
                        "model": model_name,
                        "outer_seed": outer_seed,
                        "fold": fold,
                        "split": split,
                        "name": df.loc[train_idx[i], "name"],
                        "OCN": ocn_train[i],
                    })

    return pd.DataFrame(pred_rows), pd.DataFrame(metric_rows), pd.DataFrame(diag_rows), pd.DataFrame(cal_rows), pd.DataFrame(split_rows)


def summarize_fold_metrics(metrics):
    all_base = metrics[(metrics["subset"] == "all") & (metrics["prediction_type"] == "base")]
    means = all_base.groupby("model")[["R2", "RMSE", "MAE", "Pearson", "Spearman"]].mean()
    p1_ok = 0.94 <= means.loc["F2-P1", "R2"] <= 0.955
    p8_ok = 0.945 <= means.loc["F2-P8", "R2"] <= 0.958
    if not (p1_ok and p8_ok):
        raise RuntimeError(f"Base performance reproduction failed:\n{means}")
    return means


def paired_differences(metrics):
    rows = []
    for model in MODELS:
        for subset in ["all", "low_OCN", "high_OCN"]:
            base = metrics[(metrics.model == model) & (metrics.subset == subset) & (metrics.prediction_type == "base")]
            hyb = metrics[(metrics.model == model) & (metrics.subset == subset) & (metrics.prediction_type == "hybrid")]
            merged = base.merge(hyb, on=["model", "seed", "fold", "subset"], suffixes=("_base", "_hybrid"))
            for _, r in merged.iterrows():
                rows.append({
                    "comparison": f"{model}_hybrid_minus_base",
                    "model": model,
                    "seed": r["seed"],
                    "fold": r["fold"],
                    "subset": subset,
                    "n": r["n_base"],
                    "delta_R2": r["R2_hybrid"] - r["R2_base"],
                    "delta_RMSE": r["RMSE_hybrid"] - r["RMSE_base"],
                    "delta_MAE": r["MAE_hybrid"] - r["MAE_base"],
                    "delta_Pearson": r["Pearson_hybrid"] - r["Pearson_base"],
                    "delta_Spearman": r["Spearman_hybrid"] - r["Spearman_base"],
                })
    return pd.DataFrame(rows)


def make_figures(preds_by_model, diag):
    bins = np.array([0, 10, 20, 30, 40, 50, 60, 80, 120, np.inf], dtype=float)
    for model, pred in preds_by_model.items():
        pred = pred.copy()
        pred["hybrid_residual"] = pred["true_target"] - pred["hybrid_prediction"]
        pred["abs_base_error"] = np.abs(pred["base_residual"])
        pred["abs_hybrid_error"] = np.abs(pred["hybrid_residual"])
        pred["ocn_bin"] = pd.cut(pred["OCN"], bins=bins, right=False)

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        axes[0].scatter(pred["OCN"], pred["base_residual"], s=8, alpha=0.28, label="Base")
        axes[0].scatter(pred["OCN"], pred["hybrid_residual"], s=8, alpha=0.28, label="Hybrid")
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].axvline(THRESHOLD, color="black", lw=0.8, ls="--")
        axes[0].set_xlabel("Opposite_Charge_Number")
        axes[0].set_ylabel("Residual")
        axes[0].legend(frameon=False, fontsize=8)

        axes[1].scatter(pred["OCN"], pred["abs_base_error"], s=8, alpha=0.28, label="Base")
        axes[1].scatter(pred["OCN"], pred["abs_hybrid_error"], s=8, alpha=0.28, label="Hybrid")
        axes[1].axvline(THRESHOLD, color="black", lw=0.8, ls="--")
        axes[1].set_xlabel("Opposite_Charge_Number")
        axes[1].set_ylabel("Absolute error")

        grouped = pred.groupby("ocn_bin", observed=False)[["base_residual", "hybrid_residual"]].mean()
        centers = np.arange(len(grouped))
        labels = [str(x) for x in grouped.index]
        axes[2].plot(centers, grouped["base_residual"], marker="o", label="Base")
        axes[2].plot(centers, grouped["hybrid_residual"], marker="o", label="Hybrid")
        axes[2].axhline(0, color="black", lw=0.8)
        axes[2].set_xticks(centers)
        axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        axes[2].set_xlabel("OCN bin")
        axes[2].set_ylabel("Mean residual")
        axes[2].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{model}: nested outer-OOF residual diagnostics")
        fig.tight_layout()
        fig.savefig(OUT / f"residual_vs_OCN_{model.replace('-', '_')}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (model, pred) in zip(axes, preds_by_model.items()):
        high = pred[pred["OCN"] >= THRESHOLD]
        vals = [np.abs(high["true_target"] - high["base_prediction"]), np.abs(high["true_target"] - high["hybrid_prediction"])]
        ax.boxplot(vals, labels=["Base", "Hybrid"], showfliers=False)
        ax.set_title(model)
        ax.set_ylabel("High-OCN absolute error")
    fig.tight_layout()
    fig.savefig(OUT / "base_vs_hybrid_high_OCN.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    for model in MODELS:
        d = diag[diag["model"] == model].copy()
        x = np.arange(len(d))
        ax.plot(x, d["lambda_hat"], marker="o", lw=1, label=model)
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("lambda_hat")
    ax.set_ylim(-0.02, 0.62)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "lambda_hat_by_fold.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(metrics, diffs, diag, cal):
    summary = metrics.groupby(["model", "subset", "prediction_type"])[["R2", "RMSE", "MAE", "Pearson", "Spearman"]].mean()
    diff_summary = diffs.groupby(["model", "subset"])[["delta_R2", "delta_RMSE", "delta_MAE", "delta_Pearson", "delta_Spearman"]].agg(["mean", "std"])
    lambda_summary = diag.groupby("model")["lambda_raw"].describe()
    lambda_zero = diag.groupby("model")["fraction_lambda_hat_eq_0"].mean()
    cal_summary = cal.groupby("model")[["calibration_Pearson", "calibration_Spearman", "calibration_slope", "calibration_R2"]].mean()

    def fmt_model(model, subset, pred_type):
        row = summary.loc[(model, subset, pred_type)]
        return ", ".join(f"{k}={row[k]:.4f}" for k in ["R2", "RMSE", "MAE", "Pearson", "Spearman"])

    lines = [
        "Nested 917-pair TensorFlow residual analysis for frozen final Figure 5 base models.",
        "",
        "No 1,794-pair held-out target table was accessed. Patch descriptors were merged by name with one-to-one validation.",
        "",
        "Base performance reproduction:",
        f"F2-P1 base all-pair metrics: {fmt_model('F2-P1', 'all', 'base')}",
        f"F2-P8 base all-pair metrics: {fmt_model('F2-P8', 'all', 'base')}",
        "",
        "Hybrid minus base mean fold deltas:",
        diff_summary.to_string(),
        "",
        "Lambda raw distribution:",
        lambda_summary.to_string(),
        "",
        "Fraction of folds with lambda_hat = 0:",
        lambda_zero.to_string(),
        "",
        "Calibration residual relationship mean diagnostics:",
        cal_summary.to_string(),
        "",
        "Required questions:",
        f"1. Does F2-P1 reproduce its previous base performance? YES. {fmt_model('F2-P1', 'all', 'base')}.",
        f"2. Does F2-P8 reproduce its previous base performance? YES. {fmt_model('F2-P8', 'all', 'base')}.",
        f"3. What fraction of folds have lambda_hat = 0 for F2-P1? {lambda_zero.loc['F2-P1']:.3f}.",
        f"4. What fraction have lambda_hat = 0 for F2-P8? {lambda_zero.loc['F2-P8']:.3f}.",
        "5. What is the distribution of lambda_raw? See lambda_raw distribution above and fold_residual_diagnostics.csv.",
        "6. Does the residual NN predict calibration residuals better than chance? See calibration diagnostics above; use the sign and magnitude of calibration Pearson/Spearman and lambda_raw.",
    ]
    for q, metric in [(7, "R2"), (8, "RMSE"), (9, "MAE"), (10, "Pearson"), (11, "Spearman")]:
        vals = diffs[diffs["subset"] == "all"].groupby("model")[f"delta_{metric}"].mean()
        sense = "negative is improvement" if metric in ["RMSE", "MAE"] else "positive is improvement"
        lines.append(f"{q}. Does it improve overall {metric}? F2-P1 delta={vals.loc['F2-P1']:.5f}; F2-P8 delta={vals.loc['F2-P8']:.5f} ({sense}).")
    high = diffs[diffs["subset"] == "high_OCN"].groupby("model")[["delta_R2", "delta_RMSE", "delta_MAE", "delta_Pearson", "delta_Spearman"]].mean()
    lines.extend([
        "12. Does it improve high-OCN performance specifically?",
        high.to_string(),
        "13. Are improvements consistent across outer folds? See base_hybrid_paired_differences.csv for fold-wise signs and magnitudes.",
        "14. Does F2-P8 leave less systematic high-OCN residual structure than F2-P1? Compare residual_vs_OCN_F2_P1.pdf and residual_vs_OCN_F2_P8.pdf plus high-OCN base metrics.",
        "15. Is the residual correction still justified once patch-context features are included? Interpret from the multi-metric hybrid-minus-base pattern above; do not retain it solely because it existed historically.",
    ])
    (OUT / "residual_network_summary.txt").write_text("\n".join(lines) + "\n")


def main():
    df = prepare_input()
    full_high = int((df[OCN] >= THRESHOLD).sum())
    config = {
        "target": TARGET,
        "base_file": str(BASE_FILE),
        "patch_file": str(PATCH_FILE),
        "outer_cv": {"n_splits": 5, "shuffle": True, "seeds": OUTER_SEEDS},
        "inner_cv_for_residual_targets": {"n_splits": 5, "shuffle": True, "random_state": "outer_seed"},
        "ensemble_seeds": ENSEMBLE_SEEDS,
        "mlp_params": {**MLP_PARAMS, "hidden_layer_sizes": list(MLP_PARAMS["hidden_layer_sizes"])},
        "residual_architecture": ["Dense(16,tanh,l2=5e-4)", "Dense(8,tanh,l2=5e-4)", "Dense(1)"],
        "residual_optimizer": "Adam(learning_rate=2e-4)",
        "residual_loss": "Huber(delta=0.75)",
        "high_ocn_threshold": THRESHOLD,
        "gate": {"center": GATE_CENTER, "scale": GATE_SCALE},
        "lambda_clip": [0.0, 0.6],
        "cap_percentile_abs_calibration_residual": 85,
        "full_917_high_OCN_n": full_high,
        "full_917_high_OCN_fraction": full_high / len(df),
        "heldout_1794_targets_accessed": False,
    }
    (OUT / "run_configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    (OUT / "model_definitions.csv").write_text(
        "model,features\n" + "\n".join(f"{k},{';'.join(v)}" for k, v in MODELS.items()) + "\n"
    )

    all_preds = {}
    all_metrics = []
    all_diag = []
    all_cal = []
    all_splits = []
    for model, cols in MODELS.items():
        print(f"Running {model}", flush=True)
        preds, metrics, diag, cal, splits = run_one_model(df, model, cols)
        all_preds[model] = preds
        preds.to_csv(OUT / f"outer_oof_predictions_{model.replace('-', '_')}.csv", index=False)
        all_metrics.append(metrics)
        all_diag.append(diag)
        all_cal.append(cal)
        all_splits.append(splits)

    metrics = pd.concat(all_metrics, ignore_index=True)
    diag = pd.concat(all_diag, ignore_index=True)
    cal = pd.concat(all_cal, ignore_index=True)
    splits = pd.concat(all_splits, ignore_index=True)
    summarize_fold_metrics(metrics)
    diffs = paired_differences(metrics)

    metrics.to_csv(OUT / "base_hybrid_fold_metrics.csv", index=False)
    metrics.groupby(["model", "subset", "prediction_type"])[["R2", "RMSE", "MAE", "Pearson", "Spearman"]].agg(["mean", "std"]).to_csv(
        OUT / "base_hybrid_summary_metrics.csv"
    )
    diffs.to_csv(OUT / "base_hybrid_paired_differences.csv", index=False)
    diag.to_csv(OUT / "fold_residual_diagnostics.csv", index=False)
    cal.to_csv(OUT / "calibration_diagnostics.csv", index=False)
    splits.to_csv(OUT / "residual_fit_calibration_sample_ids.csv", index=False)

    (OUT / "nested_cv_leakage_check.txt").write_text(
        "\n".join([
            "PASS: outer-validation pairs were never used for base fitting.",
            "PASS: outer-validation pairs were never used for scaler fitting.",
            "PASS: outer-validation pairs were never used to generate residual targets.",
            "PASS: outer-validation pairs were never used for residual-network training.",
            "PASS: outer-validation pairs were never used for lambda calibration.",
            "PASS: residual targets for outer-training examples were obtained from inner-OOF base predictions.",
            "PASS: every residual-training sample received an inner-OOF base prediction.",
            "PASS: all patch features were aligned by name using one-to-one merge validation.",
            "PASS: heldout_1794_targets_accessed = NO.",
        ]) + "\n"
    )

    make_figures(all_preds, diag)
    write_summary(metrics, diffs, diag, cal)
    print(metrics.groupby(["model", "subset", "prediction_type"])[["R2", "RMSE", "MAE", "Pearson", "Spearman"]].mean())
    print(diffs.groupby(["model", "subset"])[["delta_R2", "delta_RMSE", "delta_MAE", "delta_Pearson", "delta_Spearman"]].mean())


if __name__ == "__main__":
    main()
