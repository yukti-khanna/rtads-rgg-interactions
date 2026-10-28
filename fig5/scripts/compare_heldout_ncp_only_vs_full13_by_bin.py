#!/usr/bin/env python3
"""Compare held-out RMSE by S4D −NCP bin for −NCP-only and full 13-feature models.

The −NCP-only model uses the frozen model's MLP architecture and ensemble seeds,
trained on the 917 development simulations. The full 13-feature predictions are
the frozen predictor's base predictions on the 1,794 locked held-out pairs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "outputs/final_hybrid_model/development_917.csv"
TEST = ROOT / "outputs/final_hybrid_model/final_predictions/test_1794_final_hybrid_predictions.csv"
MANIFEST = ROOT / "outputs/final_hybrid_model/frozen_model/MODEL_MANIFEST.json"
BIN_SOURCE = (ROOT.parent / "fig4/reviewer_analysis/training_selection_provenance/"
              "S4D_feature_guided_sampling_coverage/S4D_bin_source.csv")
OUT = ROOT / "outputs/manuscript/Figure_S6"
BOOTSTRAPS = 5000
SEED = 20261018


def assign_bins(x: np.ndarray, bins: pd.DataFrame) -> np.ndarray:
    upper = bins["bin_upper_inclusive"].to_numpy(float)
    lower = bins["bin_lower_exclusive_except_first"].to_numpy(float)
    idx = np.searchsorted(upper, x, side="left")
    if np.any(idx >= len(bins)) or np.any(x < lower[idx]):
        raise ValueError("Held-out −NCP values fall outside the S4D intervals")
    return bins["bin_label"].to_numpy()[idx]


def fit_ncp_only(dev: pd.DataFrame, test: pd.DataFrame, manifest: dict) -> np.ndarray:
    feature = manifest["minus_NCP_feature"]
    target = manifest["target"] if manifest["target"] in dev.columns else "simulated_target"
    train_feature = feature if feature in dev.columns else "minus_NCP"
    x_train = dev[[train_feature]].to_numpy(float)
    y_train = dev[target].to_numpy(float)
    x_test = test[["minus_NCP"]].to_numpy(float)
    scaler = StandardScaler().fit(x_train)
    pred = []
    params = manifest["base_model"]["parameters"]
    for seed in params["ensemble_random_seeds"]:
        model = MLPRegressor(
            activation=params["activation"], alpha=params["alpha"],
            hidden_layer_sizes=tuple(params["hidden_layer_sizes"]),
            learning_rate_init=params["learning_rate_init"],
            max_iter=params["max_iter"], solver=params["solver"], random_state=seed,
        )
        model.fit(scaler.transform(x_train), y_train)
        pred.append(model.predict(scaler.transform(x_test)))
    return np.mean(np.vstack(pred), axis=0)


def cluster_bootstrap_rmse(g: pd.DataFrame, pred_col: str, rng: np.random.Generator):
    """Two-way identity bootstrap; resample RGG and TAD identities independently."""
    rg_ids = g["RGG_id"].drop_duplicates().to_numpy()
    tad_ids = g["TAD_id"].drop_duplicates().to_numpy()
    rg_index = pd.Index(rg_ids)
    tad_index = pd.Index(tad_ids)
    rg_codes = rg_index.get_indexer(g["RGG_id"])
    tad_codes = tad_index.get_indexer(g["TAD_id"])
    y = g["simulated_target"].to_numpy(float)
    pred = g[pred_col].to_numpy(float)
    out = np.empty(BOOTSTRAPS, dtype=float)
    b = 0
    while b < BOOTSTRAPS:
        rg_draws = rng.integers(0, len(rg_ids), len(rg_ids))
        tad_draws = rng.integers(0, len(tad_ids), len(tad_ids))
        rg_w = np.bincount(rg_draws, minlength=len(rg_ids))
        tad_w = np.bincount(tad_draws, minlength=len(tad_ids))
        weights = rg_w[rg_codes] * tad_w[tad_codes]
        denom = weights.sum()
        if denom == 0:
            # A bin is a subset of the full cross-product; a bootstrap draw can
            # miss it. Redraw that replicate rather than fabricate an interval.
            continue
        out[b] = np.sqrt(np.sum(weights * (pred - y) ** 2) / denom)
        b += 1
    return np.quantile(out, [0.025, 0.975])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = pd.read_csv(DEV)
    test = pd.read_csv(TEST)
    bins = pd.read_csv(BIN_SOURCE).sort_values("bin_id").reset_index(drop=True)
    manifest = json.loads(MANIFEST.read_text())
    if len(dev) != 917 or len(test) != 1794 or test.pair_id.nunique() != 1794:
        raise ValueError("Expected 917 development and 1,794 unique held-out pairs")
    dev_ncp_col = manifest["minus_NCP_feature"] if manifest["minus_NCP_feature"] in dev.columns else "minus_NCP"
    if not np.array_equal(dev[dev_ncp_col].to_numpy(float), dev["minus_NCP"].to_numpy(float)):
        raise ValueError("Development −NCP representations disagree")
    if test["model_manifest_sha256"].nunique() != 1:
        raise ValueError("Held-out predictions do not share one frozen-model manifest")

    test["ncp_only_prediction"] = fit_ncp_only(dev, test, manifest)
    test["bin_label"] = assign_bins(test["minus_NCP"].to_numpy(float), bins)
    test.to_csv(OUT / "FigureS6G_heldout_ncp_model_source.csv", index=False)
    rng = np.random.default_rng(SEED)
    rows = []
    for b in bins.itertuples(index=False):
        g = test.loc[test.bin_label.eq(b.bin_label)].copy()
        if g.empty:
            raise ValueError(f"No held-out pairs in S4D bin {b.bin_label}")
        for model, col in [("−NCP-only model", "ncp_only_prediction"),
                           ("Full 13-feature predictor", "base_prediction")]:
            y = g.simulated_target.to_numpy(float)
            p = g[col].to_numpy(float)
            ci = cluster_bootstrap_rmse(g, col, rng)
            rows.append({
                "bin_id": b.bin_id, "bin_label": b.bin_label,
                "bin_lower_exclusive_except_first": b.bin_lower_exclusive_except_first,
                "bin_upper_inclusive": b.bin_upper_inclusive,
                "model": model, "n": len(g),
                "RMSE": float(mean_squared_error(y, p) ** 0.5),
                "RMSE_CI_2.5": float(ci[0]), "RMSE_CI_97.5": float(ci[1]),
                "bootstrap": "two-way cluster bootstrap over RGG and TAD identities",
                "bootstrap_replicates": BOOTSTRAPS,
            })
    result = pd.DataFrame(rows)
    counts = result[result.model.eq("−NCP-only model")].sort_values("bin_id").n.to_numpy()
    s4_counts = bins.held_out_count.to_numpy()
    if not np.array_equal(counts, s4_counts) or counts.sum() != 1794:
        raise ValueError("Binned held-out counts do not match the S4D 1,794-pair audit")
    result.to_csv(OUT / "FigureS6G_heldout_ncp_model_rmse_by_bin.csv", index=False)
    (OUT / "FigureS6G_heldout_ncp_model_rmse_by_bin_README.txt").write_text(
        "Figure S6G source and method\n\n"
        "The 1,794 sequence-exclusive held-out pairs are grouped using the 12 shared −NCP intervals in Figure S4D.\n"
        "The −NCP-only baseline is a five-member MLPRegressor ensemble using only Opposite_Charge_Number (−NCP). "
        "It uses the frozen 13-feature base predictor's MLP architecture, hyperparameters, and seeds, with scaling "
        "fit on the 917 development pairs. It is trained on those 917 pairs only.\n"
        "The comparison series is the frozen full 13-feature base predictor's base_prediction on the held-out pairs; "
        "the optional residual correction is not included. Test targets were used only to calculate held-out RMSE.\n"
        "Points show within-bin RMSE. Error bars are percentile 95% intervals from 5,000 two-way cluster-bootstrap "
        "replicates resampling RG/RGG and AHD identities independently within each bin (seed 20261018). Pair counts "
        "are shown as translucent bars on the right-hand y-axis. Bins use the S4D convention (lower, upper], except the first includes its "
        "lower bound. The 12 bin counts were checked against the S4D held-out audit and sum to 1,794.\n"
        "No model selection, tuning, or fitting used held-out targets.\n"
    )

    colors = {"−NCP-only model": "#0072B2", "Full 13-feature predictor": "#D55E00"}
    x = np.arange(len(bins), dtype=float)
    offset = 0.13
    with plt.rc_context({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7, "axes.linewidth": 0.7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    }):
        fig, ax = plt.subplots(figsize=(9, 5))
        count_ax = ax.twinx()
        count_ax.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        count_ax.bar(x, bins.held_out_count.to_numpy(float), width=.52,
                     color="#777777", alpha=.18, linewidth=0, zorder=0)
        for j, model in enumerate(colors):
            sub = result[result.model.eq(model)].sort_values("bin_id")
            y = sub.RMSE.to_numpy(float)
            err = np.vstack([y - sub["RMSE_CI_2.5"].to_numpy(float),
                             sub["RMSE_CI_97.5"].to_numpy(float) - y])
            xpos = x + (-offset if j == 0 else offset)
            ax.errorbar(xpos, y, yerr=err, fmt="o", linestyle="none",
                        color=colors[model], ecolor=colors[model], markersize=4,
                        elinewidth=0.85, capsize=2, capthick=0.85, label=model, zorder=3)
        tick_labels = [str(v).replace("-", "−") for v in bins.bin_label]
        ax.set_xticks(x, tick_labels, rotation=38, ha="right")
        ax.set_xlim(-0.5, len(bins) - 0.5)
        ax.set_ylim(bottom=0)
        count_ax.set_ylim(0, float(bins.held_out_count.max()) * 1.15)
        count_ax.set_ylabel("Number of pairs per bin")
        count_ax.tick_params(axis="y", width=.7, length=2.5, pad=2, labelsize=7)
        count_ax.spines["top"].set_visible(False)
        ax.set_ylabel(r"Prediction RMSE, $\log_{10}(-B_{22})$")
        ax.set_xlabel("−NCP interval (S4D)", labelpad=4)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", width=0.7, length=2.5, pad=2)
        ax.legend(frameon=False, loc="upper right", ncol=2, handletextpad=0.4,
                  columnspacing=1.0, borderaxespad=0.2)
        ax.text(-0.065, 1.015, "G", transform=ax.transAxes, fontsize=7,
                fontweight="bold", va="bottom", ha="left", clip_on=False)
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.26, top=0.93)
        for ext, kwargs in [("pdf", {"dpi": 600}), ("svg", {}), ("png", {"dpi": 600})]:
            fig.savefig(OUT / f"FigureS6G_heldout_ncp_model_rmse_by_bin.{ext}",
                        **kwargs)
        plt.close(fig)
    print(result[["bin_label", "model", "n", "RMSE", "RMSE_CI_2.5", "RMSE_CI_97.5"]].to_string(index=False))
    print(f"Saved Figure S6G outputs to {OUT}")


if __name__ == "__main__":
    main()
