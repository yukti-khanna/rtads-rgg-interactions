#!/usr/bin/env python
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "fig5_high_OCN_tail_analysis"
RESIDUAL_OUT = ROOT / "outputs" / "fig5_tensorflow_residual_final_base"
BASE_FILE = ROOT / "sim_pairs_pred_with_hybrid_sel_cols_with_old.csv"
TARGET = "B22_corr1_log10"
OCN = "Opposite_Charge_Number"
THRESHOLD = 40.0
SEEDS = [42, 123, 456, 789, 2026]
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
MODEL_COLS = {
    "F2": "mean_F2",
    "F2-P1": "mean_F2_P1",
    "F2-P8": "mean_F2_P8",
    "F2-P1 hybrid": "mean_F2_P1_hybrid",
    "F2-P8 hybrid": "mean_F2_P8_hybrid",
}


def metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    resid = y - pred
    if len(y) < 2:
        return {"n": len(y), "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "mean_residual": np.nan,
                "median_residual": np.nan, "abs_mean_residual": np.nan, "Pearson": np.nan, "Spearman": np.nan}
    return {
        "n": len(y),
        "R2": r2_score(y, pred),
        "RMSE": mean_squared_error(y, pred) ** 0.5,
        "MAE": mean_absolute_error(y, pred),
        "mean_residual": float(np.mean(resid)),
        "median_residual": float(np.median(resid)),
        "abs_mean_residual": float(abs(np.mean(resid))),
        "Pearson": stats.pearsonr(y, pred).statistic,
        "Spearman": stats.spearmanr(y, pred).statistic,
    }


def _fit_one_mlp_predict(Xtr, ytr, Xva, seed):
    model = MLPRegressor(random_state=seed, **MLP_PARAMS)
    model.fit(Xtr, ytr)
    return model.predict(Xva)


def fit_f2_oof():
    cached = OUT / "compatible_F2_outer_oof_predictions.csv"
    if cached.exists():
        cached_df = pd.read_csv(cached)
        if len(cached_df) == 917 * len(SEEDS) and set(cached_df["seed"]) == set(SEEDS):
            return cached_df
    base = pd.read_csv(BASE_FILE)
    base["mean_lambda_RGG"] = base["lambda_RGG"] / base["RGG_length"]
    base["mean_lambda_TAD"] = base["lambda_TAD"] / base["TAD_length"]
    X = base[F2].to_numpy(float)
    y = base[TARGET].to_numpy(float)
    rows = []
    for seed in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[train_idx])
            Xva = scaler.transform(X[val_idx])
            print(f"F2 compatible OOF seed={seed} fold={fold}", flush=True)
            preds = Parallel(n_jobs=min(5, len(ENSEMBLE_SEEDS)), prefer="processes")(
                delayed(_fit_one_mlp_predict)(Xtr, y[train_idx], Xva, es)
                for es in ENSEMBLE_SEEDS
            )
            pred = np.mean(preds, axis=0)
            for idx, p in zip(val_idx, pred):
                rows.append({
                    "name": base.loc[idx, "name"],
                    "seed": seed,
                    "fold": fold,
                    "true_target": y[idx],
                    "OCN": base.loc[idx, OCN],
                    "prediction": p,
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "compatible_F2_outer_oof_predictions.csv", index=False)
    return out


def load_predictions():
    f2 = fit_f2_oof()
    p1 = pd.read_csv(RESIDUAL_OUT / "outer_oof_predictions_F2_P1.csv")
    p8 = pd.read_csv(RESIDUAL_OUT / "outer_oof_predictions_F2_P8.csv")
    for df, label in [(f2, "F2"), (p1, "F2-P1"), (p8, "F2-P8")]:
        assert df[["name", "seed", "fold"]].duplicated().sum() == 0, label
        assert set(df["seed"]) == set(SEEDS), label
        assert len(df) == 917 * len(SEEDS), label
    merged = f2.rename(columns={"prediction": "F2"})
    keep = ["name", "seed", "fold", "base_prediction", "hybrid_prediction"]
    merged = merged.merge(
        p1[keep].rename(columns={"base_prediction": "F2_P1", "hybrid_prediction": "F2_P1_hybrid"}),
        on=["name", "seed", "fold"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        p8[keep].rename(columns={"base_prediction": "F2_P8", "hybrid_prediction": "F2_P8_hybrid"}),
        on=["name", "seed", "fold"],
        how="inner",
        validate="one_to_one",
    )
    assert len(merged) == 917 * len(SEEDS)
    return merged


def region_mask(df, region):
    if region == "ALL":
        return np.ones(len(df), dtype=bool)
    if region == "LOW":
        return df["OCN"].to_numpy(float) < THRESHOLD
    if region == "HIGH":
        return df["OCN"].to_numpy(float) >= THRESHOLD
    raise ValueError(region)


def fold_metrics(pred):
    rows = []
    models = {"F2": "F2", "F2-P1": "F2_P1", "F2-P8": "F2_P8", "F2-P1 hybrid": "F2_P1_hybrid", "F2-P8 hybrid": "F2_P8_hybrid"}
    for (seed, fold), d in pred.groupby(["seed", "fold"]):
        for region in ["ALL", "LOW", "HIGH"]:
            m = region_mask(d, region)
            for model, col in models.items():
                row = {"model": model, "seed": seed, "fold": fold, "region": region}
                row.update(metrics(d.loc[m, "true_target"], d.loc[m, col]))
                rows.append(row)
    return pd.DataFrame(rows)


def unique_pair_predictions(pred):
    agg = pred.groupby("name").agg(
        true_target=("true_target", "first"),
        OCN=("OCN", "first"),
        mean_F2=("F2", "mean"),
        mean_F2_P1=("F2_P1", "mean"),
        mean_F2_P8=("F2_P8", "mean"),
        mean_F2_P1_hybrid=("F2_P1_hybrid", "mean"),
        mean_F2_P8_hybrid=("F2_P8_hybrid", "mean"),
        n_repeats_available=("seed", "nunique"),
    ).reset_index()
    assert (agg["n_repeats_available"] == 5).all()
    return agg


def unique_metrics(unique):
    rows = []
    for region in ["ALL", "LOW", "HIGH"]:
        m = region_mask(unique, region)
        for model, col in MODEL_COLS.items():
            row = {"model": model, "region": region}
            row.update(metrics(unique.loc[m, "true_target"], unique.loc[m, col]))
            rows.append(row)
    return pd.DataFrame(rows)


def comparison_stats(unique, model_a, model_b, region="HIGH"):
    col_a = MODEL_COLS[model_a]
    col_b = MODEL_COLS[model_b]
    m = region_mask(unique, region)
    d = unique.loc[m].copy()
    y = d["true_target"].to_numpy(float)
    pa = d[col_a].to_numpy(float)
    pb = d[col_b].to_numpy(float)
    ea = np.abs(y - pa)
    eb = np.abs(y - pb)
    imp = eb - ea
    ma = metrics(y, pa)
    mb = metrics(y, pb)
    return {
        "comparison": f"{model_a}_minus_{model_b}",
        "region": region,
        "n": len(d),
        "mean_delta_absolute_error": float(np.mean(imp)),
        "median_delta_absolute_error": float(np.median(imp)),
        "delta_RMSE": ma["RMSE"] - mb["RMSE"],
        "delta_MAE": ma["MAE"] - mb["MAE"],
        "delta_bias": ma["mean_residual"] - mb["mean_residual"],
        "delta_abs_bias": ma["abs_mean_residual"] - mb["abs_mean_residual"],
        "delta_Pearson": ma["Pearson"] - mb["Pearson"],
        "delta_Spearman": ma["Spearman"] - mb["Spearman"],
        "delta_R2": ma["R2"] - mb["R2"],
        "fraction_improved": float(np.mean(imp > 1e-12)),
        "fraction_worsened": float(np.mean(imp < -1e-12)),
        "fraction_essentially_unchanged": float(np.mean(np.abs(imp) <= 1e-12)),
    }


def high_pair_errors(unique):
    h = unique[unique["OCN"] >= THRESHOLD].copy()
    for model, col in MODEL_COLS.items():
        safe = model.replace("-", "_").replace(" ", "_")
        h[f"abs_error_{safe}"] = np.abs(h["true_target"] - h[col])
        h[f"residual_{safe}"] = h["true_target"] - h[col]
    h["improvement_F2_P1_vs_F2"] = h["abs_error_F2"] - h["abs_error_F2_P1"]
    h["improvement_F2_P8_vs_F2"] = h["abs_error_F2"] - h["abs_error_F2_P8"]
    h["improvement_F2_P8_vs_F2_P1"] = h["abs_error_F2_P1"] - h["abs_error_F2_P8"]
    h["improvement_F2_P1_hybrid_vs_base"] = h["abs_error_F2_P1"] - h["abs_error_F2_P1_hybrid"]
    h["improvement_F2_P8_hybrid_vs_base"] = h["abs_error_F2_P8"] - h["abs_error_F2_P8_hybrid"]
    h["patch_helped"] = h["abs_error_F2_P8"] < h["abs_error_F2"]
    h["residual_helped"] = h["abs_error_F2_P8_hybrid"] < h["abs_error_F2_P8"]
    return h


def bootstrap(unique, comparisons, n_boot=5000, seed=20260912):
    rng = np.random.default_rng(seed)
    h = unique[unique["OCN"] >= THRESHOLD].reset_index(drop=True)
    rows = []
    n = len(h)
    y_all = h["true_target"].to_numpy(float)

    def vals_for(y, p):
        resid = y - p
        out = {
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "MAE": float(np.mean(np.abs(y - p))),
            "bias": float(np.mean(resid)),
            "R2": float(1.0 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        }
        if np.std(y) > 0 and np.std(p) > 0:
            out["Pearson"] = float(np.corrcoef(y, p)[0, 1])
        else:
            out["Pearson"] = np.nan
        out["Spearman"] = float(stats.spearmanr(y, p).statistic)
        return out

    for model_a, model_b in comparisons:
        pa_all = h[MODEL_COLS[model_a]].to_numpy(float)
        pb_all = h[MODEL_COLS[model_b]].to_numpy(float)
        obs_a = vals_for(y_all, pa_all)
        obs_b = vals_for(y_all, pb_all)
        observed = {
            "delta_RMSE": obs_a["RMSE"] - obs_b["RMSE"],
            "delta_MAE": obs_a["MAE"] - obs_b["MAE"],
            "delta_bias": obs_a["bias"] - obs_b["bias"],
            "delta_Pearson": obs_a["Pearson"] - obs_b["Pearson"],
            "delta_Spearman": obs_a["Spearman"] - obs_b["Spearman"],
            "delta_R2": obs_a["R2"] - obs_b["R2"],
        }
        values = {k: [] for k in observed}
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            ya = y_all[idx]
            va = vals_for(ya, pa_all[idx])
            vb = vals_for(ya, pb_all[idx])
            values["delta_RMSE"].append(va["RMSE"] - vb["RMSE"])
            values["delta_MAE"].append(va["MAE"] - vb["MAE"])
            values["delta_bias"].append(va["bias"] - vb["bias"])
            values["delta_Pearson"].append(va["Pearson"] - vb["Pearson"])
            values["delta_Spearman"].append(va["Spearman"] - vb["Spearman"])
            values["delta_R2"].append(va["R2"] - vb["R2"])
        for k, vals in values.items():
            vals = np.asarray(vals, dtype=float)
            rows.append({
                "comparison": f"{model_a}_minus_{model_b}",
                "metric": k,
                "observed_difference": observed[k],
                "bootstrap_mean": float(np.nanmean(vals)),
                "ci_2_5": float(np.nanpercentile(vals, 2.5)),
                "ci_97_5": float(np.nanpercentile(vals, 97.5)),
                "n_unique_high_OCN_pairs": n,
                "n_bootstrap_resamples": n_boot,
            })
    return pd.DataFrame(rows)


def trends(unique):
    rows = []
    h = unique[unique["OCN"] >= THRESHOLD]
    x = h["OCN"].to_numpy(float)
    for model, col in MODEL_COLS.items():
        resid = h["true_target"].to_numpy(float) - h[col].to_numpy(float)
        lr = stats.linregress(x, resid)
        ci = 1.96 * lr.stderr
        rows.append({
            "model": model,
            "region": "HIGH",
            "n": len(h),
            "slope_residual_vs_OCN": lr.slope,
            "slope_ci_95_low": lr.slope - ci,
            "slope_ci_95_high": lr.slope + ci,
            "intercept": lr.intercept,
            "Pearson_residual_OCN": lr.rvalue,
            "p_value": lr.pvalue,
        })
    return pd.DataFrame(rows)


def bin_metrics(unique):
    bins = [-np.inf, 20, 30, 40, 50, 60, np.inf]
    labels = ["OCN < 20", "20 <= OCN < 30", "30 <= OCN < 40", "40 <= OCN < 50", "50 <= OCN < 60", "OCN >= 60"]
    d = unique.copy()
    d["OCN_bin"] = pd.cut(d["OCN"], bins=bins, labels=labels, right=False)
    rows = []
    for label, g in d.groupby("OCN_bin", observed=False):
        for model, col in MODEL_COLS.items():
            row = {"OCN_bin": str(label), "model": model}
            met = metrics(g["true_target"], g[col])
            row.update({k: met[k] for k in ["n", "RMSE", "MAE", "mean_residual"]})
            rows.append(row)
    return pd.DataFrame(rows)


def make_figures(unique, unique_metrics_df, comparisons_df, bins_df):
    high_metrics = unique_metrics_df[unique_metrics_df["region"] == "HIGH"]
    order = list(MODEL_COLS)
    x = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        vals = [high_metrics.loc[high_metrics.model == m, metric].iloc[0] for m in order]
        ax.bar(x, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"High-OCN {metric}")
    fig.tight_layout()
    fig.savefig(OUT / "tail_RMSE_MAE_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    vals = [high_metrics.loc[high_metrics.model == m, "mean_residual"].iloc[0] for m in order]
    ax.bar(x, vals)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel("Mean residual (true - predicted)")
    ax.set_title("High-OCN bias")
    fig.tight_layout()
    fig.savefig(OUT / "tail_bias_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    h = unique[unique["OCN"] >= THRESHOLD]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model, col in MODEL_COLS.items():
        ax.scatter(h["OCN"], h["true_target"] - h[col], s=18, alpha=0.45, label=model)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Opposite_Charge_Number")
    ax.set_ylabel("Residual")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "residual_vs_OCN_models.pdf", bbox_inches="tight")
    plt.close(fig)

    high = high_pair_errors(unique)
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [
        high["improvement_F2_P1_vs_F2"],
        high["improvement_F2_P8_vs_F2"],
        high["improvement_F2_P1_hybrid_vs_base"],
        high["improvement_F2_P8_hybrid_vs_base"],
    ]
    ax.boxplot(data, labels=["P1-F2", "P8-F2", "P1 hyb-base", "P8 hyb-base"], showfliers=False)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Absolute-error improvement")
    fig.tight_layout()
    fig.savefig(OUT / "absolute_error_improvement_high_OCN.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    for ax, metric in zip(axes, ["RMSE", "MAE", "mean_residual"]):
        for model in order:
            d = bins_df[bins_df["model"] == model]
            ax.plot(range(len(d)), d[metric], marker="o", label=model)
        ax.set_title(metric)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels(d["OCN_bin"], rotation=45, ha="right", fontsize=7)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "performance_by_OCN_bin.pdf", bbox_inches="tight")
    plt.close(fig)


def write_summary(unique_metrics_df, comps, boot, trend, high_pairs, bins):
    high = unique_metrics_df[unique_metrics_df.region == "HIGH"].set_index("model")
    allm = unique_metrics_df[unique_metrics_df.region == "ALL"].set_index("model")
    low = unique_metrics_df[unique_metrics_df.region == "LOW"].set_index("model")
    n_high = int(high.loc["F2", "n"])
    n_low = int(low.loc["F2", "n"])
    n_all = int(allm.loc["F2", "n"])
    c = comps.set_index("comparison")

    def line_model(model):
        r = high.loc[model]
        return f"{model}: RMSE={r.RMSE:.4f}, MAE={r.MAE:.4f}, bias={r.mean_residual:+.4f}, abs_bias={r.abs_mean_residual:.4f}, Pearson={r.Pearson:.4f}, Spearman={r.Spearman:.4f}, R2={r.R2:.4f}"

    def comp_line(name):
        r = c.loc[name]
        return f"{name}: delta_RMSE={r.delta_RMSE:+.5f}, delta_MAE={r.delta_MAE:+.5f}, delta_abs_bias={r.delta_abs_bias:+.5f}, fraction_improved={r.fraction_improved:.3f}, median_abs_error_improvement={r.median_delta_absolute_error:+.5f}"

    def ci_excludes_zero(name, metric):
        r = boot[(boot.comparison == name) & (boot.metric == metric)].iloc[0]
        return (r.ci_2_5 > 0 and r.ci_97_5 > 0) or (r.ci_2_5 < 0 and r.ci_97_5 < 0)

    help_table = pd.crosstab(high_pairs["patch_helped"], high_pairs["residual_helped"])
    lines = [
        "High-OCN tail analysis from existing outer-OOF predictions.",
        "",
        "Only 917-pair outer-validation predictions were used. The 1,794-pair held-out dataset was not accessed. F2 was rerun only to obtain compatible OOF predictions on the same outer splits used by the TensorFlow residual analysis.",
        f"Region sizes: ALL={n_all}, LOW OCN<40={n_low}, HIGH OCN>=40={n_high}. Each unique pair contributes one repeat-averaged OOF prediction per model.",
        "",
        "Primary high-OCN metrics:",
        line_model("F2"),
        line_model("F2-P1"),
        line_model("F2-P8"),
        line_model("F2-P1 hybrid"),
        line_model("F2-P8 hybrid"),
        "",
        "High-OCN paired comparisons:",
        comp_line("F2-P1_minus_F2"),
        comp_line("F2-P8_minus_F2"),
        comp_line("F2-P8_minus_F2-P1"),
        comp_line("F2-P1 hybrid_minus_F2-P1"),
        comp_line("F2-P8 hybrid_minus_F2-P8"),
        "",
        "Residual-vs-OCN slopes in the high tail:",
        trend.to_string(index=False),
        "",
        "Patch helped vs residual helped for F2-P8 high-OCN pairs:",
        help_table.to_string(),
        "",
        "Required answers:",
        f"1. How many unique pairs have OCN >= 40? {n_high}.",
        f"2. How does plain F2 perform specifically in this tail? {line_model('F2')}.",
        f"3. Does opp_charge_std improve tail RMSE? Yes. F2-P1 minus F2 delta_RMSE={c.loc['F2-P1_minus_F2','delta_RMSE']:+.5f}; negative means lower RMSE.",
        f"4. Does it improve tail MAE? Yes. F2-P1 minus F2 delta_MAE={c.loc['F2-P1_minus_F2','delta_MAE']:+.5f}.",
        f"5. Does it reduce systematic tail bias? Yes if judged by absolute mean bias: delta_abs_bias={c.loc['F2-P1_minus_F2','delta_abs_bias']:+.5f}.",
        f"6. Does the full P8 patch panel improve further? Yes for RMSE/MAE relative to F2 and modestly relative to F2-P1: {comp_line('F2-P8_minus_F2-P1')}.",
        f"7. What fraction of high-OCN pairs benefit from P1? {c.loc['F2-P1_minus_F2','fraction_improved']:.3f}.",
        f"8. What fraction benefit from P8? {c.loc['F2-P8_minus_F2','fraction_improved']:.3f}.",
        f"9. Does the F2-P1 residual network improve tail RMSE/MAE? No. {comp_line('F2-P1 hybrid_minus_F2-P1')}.",
        f"10. Does the F2-P8 residual network improve tail RMSE/MAE? Mixed and negligible: RMSE improves slightly but MAE worsens. {comp_line('F2-P8 hybrid_minus_F2-P8')}.",
        f"11. Does either residual network reduce systematic high-OCN bias? F2-P1 hybrid does not; F2-P8 hybrid changes absolute bias by {c.loc['F2-P8 hybrid_minus_F2-P8','delta_abs_bias']:+.5f}, a very small shift.",
        "12. Does either residual network flatten residual-vs-OCN trends? No convincing flattening; slope CIs and trend table should be treated as diagnostic, and the hybrid slopes do not show a robust tail correction pattern.",
        f"13. Are tail improvements supported by paired bootstrap confidence intervals? Patch improvements are supported where the relevant RMSE/MAE CIs exclude zero; residual improvements are not supported as a consistent RMSE/MAE/MAE+bias pattern. See tail_bootstrap_differences.csv. Bootstrap RMSE CI excludes zero for P1 vs F2: {ci_excludes_zero('F2-P1_minus_F2','delta_RMSE')}; P8 vs F2: {ci_excludes_zero('F2-P8_minus_F2','delta_RMSE')}; P8 hybrid vs base: {ci_excludes_zero('F2-P8 hybrid_minus_F2-P8','delta_RMSE')}.",
        "14. Are patch gains concentrated in the high-OCN regime or distributed across OCN values? They are visible in the high-OCN bin summaries and should be read alongside performance_by_OCN_bin.csv/pdf; the main improvement is not a residual-network effect.",
        f"15. Does the residual network correct examples that remain poorly predicted after patch features? Not systematically. Among high-OCN pairs, the patch/residual help table is:\n{help_table.to_string()}",
        "16. Based specifically on its intended high-OCN use case, is the residual correction justified? No for F2-P1, and not convincingly for F2-P8. The P8 residual gives a tiny high-tail RMSE/R2 shift but worsens MAE/rank metrics in the nested residual run and does not show a strong calibrated bias/trend correction.",
    ]
    (OUT / "high_OCN_tail_summary.txt").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pred = load_predictions()
    fold = fold_metrics(pred)
    unique = unique_pair_predictions(pred)
    uniq_met = unique_metrics(unique)
    comps = pd.DataFrame([
        comparison_stats(unique, "F2-P1", "F2"),
        comparison_stats(unique, "F2-P8", "F2"),
        comparison_stats(unique, "F2-P8", "F2-P1"),
        comparison_stats(unique, "F2-P1 hybrid", "F2-P1"),
        comparison_stats(unique, "F2-P8 hybrid", "F2-P8"),
    ])
    boot = bootstrap(unique, [
        ("F2-P1", "F2"),
        ("F2-P8", "F2"),
        ("F2-P8", "F2-P1"),
        ("F2-P1 hybrid", "F2-P1"),
        ("F2-P8 hybrid", "F2-P8"),
    ])
    high_pairs = high_pair_errors(unique)
    trend = trends(unique)
    bins = bin_metrics(unique)
    help_table = pd.crosstab(high_pairs["patch_helped"], high_pairs["residual_helped"]).rename_axis(index="patch_helped", columns="residual_helped")

    fold.to_csv(OUT / "tail_fold_metrics.csv", index=False)
    unique.to_csv(OUT / "unique_pair_mean_oof_predictions.csv", index=False)
    uniq_met.to_csv(OUT / "tail_unique_pair_metrics.csv", index=False)
    comps.to_csv(OUT / "tail_paired_model_comparisons.csv", index=False)
    boot.to_csv(OUT / "tail_bootstrap_differences.csv", index=False)
    high_pairs.to_csv(OUT / "high_OCN_pair_error_comparison.csv", index=False)
    help_table.to_csv(OUT / "patch_vs_residual_help_table.csv")
    trend.to_csv(OUT / "residual_OCN_trends.csv", index=False)
    bins.to_csv(OUT / "performance_by_OCN_bin.csv", index=False)
    (OUT / "run_configuration.json").write_text(json.dumps({
        "source": "outer-OOF predictions only",
        "residual_prediction_dir": str(RESIDUAL_OUT),
        "F2": "rerun only to obtain compatible OOF predictions on seeds/folds [42,123,456,789,2026]",
        "heldout_1794_accessed": False,
        "threshold": THRESHOLD,
        "bootstrap_resamples": 5000,
    }, indent=2) + "\n")
    make_figures(unique, uniq_met, comps, bins)
    write_summary(uniq_met, comps, boot, trend, high_pairs, bins)
    print(uniq_met[uniq_met.region == "HIGH"].to_string(index=False))
    print(comps.to_string(index=False))


if __name__ == "__main__":
    main()
