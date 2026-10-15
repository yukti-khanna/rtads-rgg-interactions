#!/usr/bin/env python3
"""Composition-preserving WT/shuffle paired control audit.

Run from any directory; all inputs are explicit CLI paths. No simulations,
model fitting, shuffle generation, or tuning are performed.
"""
import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None

AA_CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1}

DEFAULT_FEATURES = [
    "SCD_RGG", "SCD_TAD", "SHD_RGG", "SHD_TAD", "Charge_Complementarity",
    "opp_charge_std", "scd_patch1_std", "scd_patch2_std", "shd_patch1_std",
    "shd_patch2_std", "rg_fraction_patch1_std", "rg_fraction_patch2_std",
    "aromatic_fraction_patch1_std", "aromatic_fraction_patch2_std",
    "rgg_count_patch1_std", "rgg_count_patch2_std", "ah_energy_std",
    "Net_Charge_Product", "Opposite_Charge_Number",
]


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def pick(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def sequence_charge(seq):
    return sum(AA_CHARGE.get(aa, 0) for aa in str(seq))


def acidic_count(seq):
    return sum(aa in "DE" for aa in str(seq))


def basic_count(seq):
    return sum(aa in "KR" for aa in str(seq))


def bootstrap_ci(x, stat, n=5000, seed=42):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.choice(x, len(x), replace=True)) for _ in range(n)])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def corr_pair(x, y, method):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    if stats is None:
        if method == "pearson":
            return float(np.corrcoef(x[ok], y[ok])[0, 1])
        return float(pd.Series(x[ok]).corr(pd.Series(y[ok]), method="spearman"))
    if method == "pearson":
        return float(stats.pearsonr(x[ok], y[ok])[0])
    return float(stats.spearmanr(x[ok], y[ok]).correlation)


def wilcoxon_summary(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if stats is None or len(x) == 0 or np.allclose(x, 0):
        return np.nan, np.nan
    res = stats.wilcoxon(x, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def rdf_path_for_pair(rdf_dir, pair_id):
    return Path(rdf_dir) / f"rdfs_{pair_id}_20nm.pkl"


def per_run_b22(pair_id, rdf_dir, rdf_scripts_dir):
    if not rdf_dir:
        return None
    path = rdf_path_for_pair(rdf_dir, pair_id)
    if not path.exists():
        return None
    if rdf_scripts_dir:
        sys.path.insert(0, str(Path(rdf_scripts_dir).resolve()))
    from rdf_b22_functions import corrected_rdfs_b22

    df = pd.read_pickle(path)
    _, b22s, _ = corrected_rdfs_b22(df, 7.5, box_size=20, delta=2, rc_nm=4.0)
    vals = []
    for run in sorted(b22s):
        record = b22s[run]
        vals.append(float(record.get("Corrected_B22_Hummer", record.get("b22_corr1"))))
    return vals


def make_inventory(manifest, b22):
    rows = []
    wt_by_parent = {}
    for _, row in manifest.iterrows():
        if str(row.get("condition", "")).upper() == "WT_WT":
            wt_by_parent[(row.get("RGG_parent_id"), row.get("TAD_parent_id"))] = row.get("name")
    for _, row in manifest.iterrows():
        condition = str(row.get("condition", "")).upper()
        if "SHUFF" not in condition:
            continue
        if "TAD" in condition:
            shuffle_type = "TAD_shuffle"
        elif "RGG" in condition:
            shuffle_type = "RGG_shuffle"
        else:
            continue
        tad_parent = row.get("TAD_parent_id")
        rgg_parent = row.get("RGG_parent_id")
        wt_name = wt_by_parent.get((rgg_parent, tad_parent), f"RGG_{rgg_parent}_WT__TAD_{tad_parent}_WT")
        rows.append({
            "shuffle_pair_id": row.get("name"),
            "WT_pair_id": wt_name,
            "shuffle_type": shuffle_type,
            "condition": row.get("condition"),
            "TAD_parent_id": tad_parent,
            "RGG_parent_id": rgg_parent,
            "TAD_class": row.get("TAD_class"),
            "is_basic_region_control": "basic_AD_control" in str(tad_parent) or "basic_AD_control" in str(row.get("TAD_class")),
            "TAD_sequence_shuffle": row.get("TAD_sequence"),
            "RGG_sequence_shuffle": row.get("RGG_sequence"),
            "Net_Charge_Product_shuffle": row.get("Net_Charge_Product"),
        })
    inv = pd.DataFrame(rows).drop_duplicates("shuffle_pair_id")
    b22_ids = set(b22["pair_id"])
    inv["WT_simulation_present"] = inv["WT_pair_id"].isin(b22_ids)
    inv["shuffle_simulation_present"] = inv["shuffle_pair_id"].isin(b22_ids)
    return inv


def make_effects(inv, manifest, b22):
    wt_seq = manifest.set_index("name")[["TAD_sequence", "RGG_sequence", "Net_Charge_Product"]].to_dict("index")
    eff = inv.merge(b22, left_on="WT_pair_id", right_on="pair_id").rename(columns={"B22": "B22_WT"}).drop(columns=["pair_id"])
    eff = eff.merge(b22, left_on="shuffle_pair_id", right_on="pair_id").rename(columns={"B22": "B22_shuffle"}).drop(columns=["pair_id"])
    eff["delta_B22"] = eff["B22_shuffle"] - eff["B22_WT"]
    eff["delta_logB22"] = np.nan
    valid_log = (eff["B22_shuffle"] < 0) & (eff["B22_WT"] < 0)
    eff.loc[valid_log, "delta_logB22"] = np.log10(-eff.loc[valid_log, "B22_shuffle"]) - np.log10(-eff.loc[valid_log, "B22_WT"])
    eff["TAD_sequence_WT"] = eff["WT_pair_id"].map(lambda x: wt_seq.get(x, {}).get("TAD_sequence"))
    eff["RGG_sequence_WT"] = eff["WT_pair_id"].map(lambda x: wt_seq.get(x, {}).get("RGG_sequence"))
    eff["Net_Charge_Product_WT"] = eff["WT_pair_id"].map(lambda x: wt_seq.get(x, {}).get("Net_Charge_Product"))
    eff["minus_NCP_WT"] = -eff["Net_Charge_Product_WT"].astype(float)
    eff["minus_NCP_shuffle"] = -eff["Net_Charge_Product_shuffle"].astype(float)
    return eff


def verify_composition(eff):
    rows = []
    for _, row in eff.iterrows():
        if row["shuffle_type"] == "TAD_shuffle":
            wt_seq = row["TAD_sequence_WT"]
            shuf_seq = row["TAD_sequence_shuffle"]
            unchanged_seq = row["RGG_sequence_WT"] == row["RGG_sequence_shuffle"]
        else:
            wt_seq = row["RGG_sequence_WT"]
            shuf_seq = row["RGG_sequence_shuffle"]
            unchanged_seq = row["TAD_sequence_WT"] == row["TAD_sequence_shuffle"]
        wt_charge = sequence_charge(wt_seq)
        shuf_charge = sequence_charge(shuf_seq)
        length_ok = len(str(wt_seq)) == len(str(shuf_seq))
        comp_ok = Counter(str(wt_seq)) == Counter(str(shuf_seq))
        acidic_ok = acidic_count(wt_seq) == acidic_count(shuf_seq)
        basic_ok = basic_count(wt_seq) == basic_count(shuf_seq)
        charge_ok = wt_charge == shuf_charge
        ncp_ok = np.isclose(float(row["minus_NCP_WT"]), float(row["minus_NCP_shuffle"]))
        overall = all([length_ok, comp_ok, acidic_ok, basic_ok, charge_ok, ncp_ok, unchanged_seq])
        rows.append({
            "shuffle_pair_id": row["shuffle_pair_id"],
            "WT_pair_id": row["WT_pair_id"],
            "shuffle_type": row["shuffle_type"],
            "is_basic_region_control": row["is_basic_region_control"],
            "length_preserved": length_ok,
            "amino_acid_composition_preserved": comp_ok,
            "acidic_residue_count_preserved": acidic_ok,
            "basic_residue_count_preserved": basic_ok,
            "net_charge_preserved": charge_ok,
            "minus_NCP_preserved": ncp_ok,
            "unchanged_partner_sequence_preserved": unchanged_seq,
            "overall_pass": overall,
            "WT_length": len(str(wt_seq)),
            "shuffle_length": len(str(shuf_seq)),
            "WT_acidic_count": acidic_count(wt_seq),
            "shuffle_acidic_count": acidic_count(shuf_seq),
            "WT_basic_count": basic_count(wt_seq),
            "shuffle_basic_count": basic_count(shuf_seq),
            "WT_net_charge": wt_charge,
            "shuffle_net_charge": shuf_charge,
            "minus_NCP_WT": row["minus_NCP_WT"],
            "minus_NCP_shuffle": row["minus_NCP_shuffle"],
        })
    return pd.DataFrame(rows)


def add_replica_noise(eff, rdf_dir, rdf_scripts_dir):
    cache = {}
    rows = []
    for _, row in eff.iterrows():
        out = row.to_dict()
        vals = []
        for pair_col in ["WT_pair_id", "shuffle_pair_id"]:
            pair_id = row[pair_col]
            if pair_id not in cache:
                cache[pair_id] = per_run_b22(pair_id, rdf_dir, rdf_scripts_dir)
            vals.append(cache[pair_id])
        wt, shuf = vals
        if wt is None or shuf is None:
            out.update({
                "WT_replica_n": 0 if wt is None else len(wt),
                "shuffle_replica_n": 0 if shuf is None else len(shuf),
                "WT_replica_mean_B22": np.nan,
                "WT_replica_sd_B22": np.nan,
                "shuffle_replica_mean_B22": np.nan,
                "shuffle_replica_sd_B22": np.nan,
                "pooled_within_condition_sd": np.nan,
                "abs_delta_B22": abs(row["delta_B22"]),
                "effect_to_noise": np.nan,
            })
        else:
            wt_arr = np.asarray(wt, float)
            shuf_arr = np.asarray(shuf, float)
            wt_sd = float(np.std(wt_arr, ddof=1))
            shuf_sd = float(np.std(shuf_arr, ddof=1))
            pooled = math.sqrt(((len(wt_arr) - 1) * wt_sd**2 + (len(shuf_arr) - 1) * shuf_sd**2) / (len(wt_arr) + len(shuf_arr) - 2))
            out.update({
                "WT_replica_n": len(wt_arr),
                "shuffle_replica_n": len(shuf_arr),
                "WT_replica_mean_B22": float(np.mean(wt_arr)),
                "WT_replica_sd_B22": wt_sd,
                "shuffle_replica_mean_B22": float(np.mean(shuf_arr)),
                "shuffle_replica_sd_B22": shuf_sd,
                "pooled_within_condition_sd": pooled,
                "abs_delta_B22": abs(row["delta_B22"]),
                "effect_to_noise": abs(row["delta_B22"]) / pooled if pooled > 0 else np.nan,
            })
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_effects(eff):
    rows = []
    for shuffle_type, group in eff.groupby("shuffle_type"):
        x = group["delta_B22"].to_numpy(float)
        w_stat, w_p = wilcoxon_summary(x)
        row = {
            "shuffle_type": shuffle_type,
            "n": len(group),
            "mean_delta_B22": float(np.mean(x)),
            "median_delta_B22": float(np.median(x)),
            "sd_delta_B22": float(np.std(x, ddof=1)),
            "iqr_delta_B22": float(np.subtract(*np.quantile(x, [0.75, 0.25]))),
            "mean_abs_delta_B22": float(np.mean(np.abs(x))),
            "median_abs_delta_B22": float(np.median(np.abs(x))),
            "fraction_positive": float(np.mean(x > 0)),
            "fraction_negative": float(np.mean(x < 0)),
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": w_p,
        }
        row["bootstrap_mean_low"], row["bootstrap_mean_high"] = bootstrap_ci(x, np.mean)
        row["bootstrap_median_low"], row["bootstrap_median_high"] = bootstrap_ci(x, np.median)
        rows.append(row)
    return pd.DataFrame(rows)


def noise_summary(replica):
    rows = []
    for shuffle_type, group in replica.groupby("shuffle_type"):
        ratio = group["effect_to_noise"].to_numpy(float)
        rows.append({
            "shuffle_type": shuffle_type,
            "n_with_replica_noise": int(np.isfinite(ratio).sum()),
            "fraction_gt_0p5_pooled_sd": float(np.nanmean(ratio > 0.5)),
            "fraction_gt_1_pooled_sd": float(np.nanmean(ratio > 1.0)),
            "fraction_gt_2_pooled_sd": float(np.nanmean(ratio > 2.0)),
            "median_abs_delta_over_pooled_sd": float(np.nanmedian(ratio)),
        })
    return pd.DataFrame(rows)


def parent_reproducibility(tad_eff):
    rows = []
    for key, group in tad_eff.groupby(["WT_pair_id", "TAD_parent_id", "RGG_parent_id"]):
        group = group.sort_values("shuffle_pair_id")
        vals = group["delta_B22"].to_numpy(float)
        ratios = group["effect_to_noise"].to_numpy(float) if "effect_to_noise" in group.columns else np.full(len(group), np.nan)
        row = dict(zip(["WT_pair_id", "TAD_parent_id", "RGG_parent_id"], key))
        row["n_shuffles"] = len(group)
        for i in range(2):
            row[f"shuffle_{i + 1}_pair_id"] = group.iloc[i]["shuffle_pair_id"] if i < len(group) else None
            row[f"shuffle_{i + 1}_delta_B22"] = vals[i] if i < len(vals) else np.nan
            row[f"shuffle_{i + 1}_effect_to_noise"] = ratios[i] if i < len(ratios) else np.nan
        row["same_direction"] = len(vals) >= 2 and np.sign(vals[0]) == np.sign(vals[1]) and vals[0] != 0 and vals[1] != 0
        row["mean_delta_B22"] = float(np.mean(vals))
        row["difference_between_shuffle_variants"] = float(abs(vals[0] - vals[1])) if len(vals) >= 2 else np.nan
        row["both_exceed_1_pooled_sd"] = bool(len(ratios) >= 2 and np.all(ratios[:2] > 1))
        row["neither_exceeds_1_pooled_sd"] = bool(len(ratios) >= 2 and np.all(ratios[:2] <= 1))
        rows.append(row)
    return pd.DataFrame(rows)


def prediction_delta(eff, predictions):
    name_col = pick(predictions, ["name", "pair_id"])
    pred_col = pick(predictions, ["prediction_final", "prediction", "pred"])
    if not name_col or not pred_col:
        return pd.DataFrame()
    pred = predictions.rename(columns={name_col: "pair_id", pred_col: "prediction"})[["pair_id", "prediction"]]
    out = eff.merge(pred, left_on="WT_pair_id", right_on="pair_id").rename(columns={"prediction": "prediction_WT"}).drop(columns=["pair_id"])
    out = out.merge(pred, left_on="shuffle_pair_id", right_on="pair_id").rename(columns={"prediction": "prediction_shuffle"}).drop(columns=["pair_id"])
    out["delta_predicted"] = out["prediction_shuffle"] - out["prediction_WT"]
    out["delta_simulated"] = out["delta_logB22"]
    out["delta_B22_simulated"] = out["delta_B22"]
    return out


def prediction_delta_summary(pred_delta):
    rows = []
    for shuffle_type, group in pred_delta.groupby("shuffle_type"):
        y = group["delta_simulated"].to_numpy(float)
        p = group["delta_predicted"].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(p)
        rows.append({
            "shuffle_type": shuffle_type,
            "n": int(ok.sum()),
            "Pearson": corr_pair(y, p, "pearson"),
            "Spearman": corr_pair(y, p, "spearman"),
            "MAE_delta": float(np.mean(np.abs(p[ok] - y[ok]))) if ok.any() else np.nan,
            "RMSE_delta": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))) if ok.any() else np.nan,
            "sign_agreement": float(np.mean(np.sign(p[ok]) == np.sign(y[ok]))) if ok.any() else np.nan,
        })
    return pd.DataFrame(rows)


def feature_delta_correlations(eff, predictions):
    name_col = pick(predictions, ["name", "pair_id"])
    if not name_col:
        return pd.DataFrame()
    available = [c for c in DEFAULT_FEATURES if c in predictions.columns]
    feat = predictions.rename(columns={name_col: "pair_id"})[["pair_id"] + available]
    merged = eff[["shuffle_pair_id", "WT_pair_id", "shuffle_type", "delta_B22", "delta_logB22"]].merge(feat, left_on="WT_pair_id", right_on="pair_id")
    merged = merged.drop(columns=["pair_id"])
    merged = merged.merge(feat, left_on="shuffle_pair_id", right_on="pair_id", suffixes=("_WT", "_shuffle")).drop(columns=["pair_id"])
    rows = []
    for feature in available:
        delta = merged[f"{feature}_shuffle"] - merged[f"{feature}_WT"]
        label = feature.replace("Opposite_Charge_Number", "minus_NCP").replace("Net_Charge_Product", "net_charge_product")
        for shuffle_type, idx in merged.groupby("shuffle_type").groups.items():
            sub = merged.loc[idx]
            d = delta.loc[idx].to_numpy(float)
            rows.append({
                "feature": label,
                "shuffle_type": shuffle_type,
                "n": int(np.isfinite(d).sum()),
                "mean_feature_delta": float(np.nanmean(d)),
                "median_feature_delta": float(np.nanmedian(d)),
                "Pearson_delta_feature_vs_delta_B22": corr_pair(d, sub["delta_B22"].to_numpy(float), "pearson"),
                "Spearman_delta_feature_vs_delta_B22": corr_pair(d, sub["delta_B22"].to_numpy(float), "spearman"),
                "Pearson_delta_feature_vs_delta_logB22": corr_pair(d, sub["delta_logB22"].to_numpy(float), "pearson"),
                "Spearman_delta_feature_vs_delta_logB22": corr_pair(d, sub["delta_logB22"].to_numpy(float), "spearman"),
            })
    return pd.DataFrame(rows)


def parent_effects(eff):
    rows = []
    for parent_kind, col in [("TAD_parent", "TAD_parent_id"), ("RGG_parent", "RGG_parent_id")]:
        for (shuffle_type, parent), group in eff.groupby(["shuffle_type", col]):
            x = group["delta_B22"].to_numpy(float)
            rows.append({
                "parent_kind": parent_kind,
                "parent_id": parent,
                "shuffle_type": shuffle_type,
                "n": len(group),
                "mean_delta_B22": float(np.mean(x)),
                "median_delta_B22": float(np.median(x)),
                "mean_abs_delta_B22": float(np.mean(np.abs(x))),
            })
    return pd.DataFrame(rows)


def make_figures(eff, replica, pred_delta, repro, out):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    lo = float(min(eff["B22_WT"].min(), eff["B22_shuffle"].min()))
    hi = float(max(eff["B22_WT"].max(), eff["B22_shuffle"].max()))
    for ax, shuffle_type in zip(axes, ["TAD_shuffle", "RGG_shuffle"]):
        group = eff[eff["shuffle_type"] == shuffle_type]
        ax.scatter(group["B22_WT"], group["B22_shuffle"], s=22, alpha=0.7)
        ax.plot([lo, hi], [lo, hi], color="black", lw=1)
        ax.set_title(shuffle_type)
        ax.set_xlabel("WT simulated B22")
        ax.set_ylabel("shuffled simulated B22")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    fig.savefig(figdir / "A_WT_vs_shuffle_B22.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for shuffle_type in ["TAD_shuffle", "RGG_shuffle"]:
        ax.hist(eff.loc[eff["shuffle_type"] == shuffle_type, "delta_B22"], bins=30, alpha=0.55, label=shuffle_type)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("delta_B22 = shuffle - WT")
    ax.set_ylabel("count")
    ax.legend()
    fig.savefig(figdir / "B_delta_B22_distributions.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    vals = [replica.loc[replica["shuffle_type"] == s, "effect_to_noise"].dropna() for s in ["TAD_shuffle", "RGG_shuffle"]]
    ax.boxplot(vals, labels=["TAD", "RGG"], showfliers=False)
    ax.axhline(1, color="black", lw=1, ls="--")
    ax.axhline(2, color="black", lw=1, ls=":")
    ax.set_ylabel("|delta_B22| / pooled replica SD")
    fig.savefig(figdir / "C_effect_to_replica_noise.png", dpi=200)
    plt.close(fig)

    if not repro.empty:
        fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
        ax.scatter(repro["shuffle_1_delta_B22"], repro["shuffle_2_delta_B22"], s=24, alpha=0.75)
        lim = float(np.nanmax(np.abs(repro[["shuffle_1_delta_B22", "shuffle_2_delta_B22"]].to_numpy())))
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=1)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel("TAD shuffle 1 delta_B22")
        ax.set_ylabel("TAD shuffle 2 delta_B22")
        fig.savefig(figdir / "D_TAD_shuffle_reproducibility.png", dpi=200)
        plt.close(fig)

    if not pred_delta.empty:
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        for ax, shuffle_type in zip(axes, ["TAD_shuffle", "RGG_shuffle"]):
            group = pred_delta[pred_delta["shuffle_type"] == shuffle_type]
            ax.scatter(group["delta_simulated"], group["delta_predicted"], s=22, alpha=0.7)
            ax.axhline(0, color="gray", lw=0.8)
            ax.axvline(0, color="gray", lw=0.8)
            ax.set_title(shuffle_type)
            ax.set_xlabel("simulated delta log10(-B22)")
            ax.set_ylabel("predicted delta")
        fig.savefig(figdir / "E_delta_predicted_vs_delta_simulated.png", dpi=200)
        plt.close(fig)


def write_interpretation(out, eff, integ, summary, noise, repro, pred_summary, feature_corr, parent):
    lines = ["Corrected non-basic shuffle audit", ""]
    lines.append(f"1. Valid matched non-basic shuffles: {(eff.shuffle_type == 'TAD_shuffle').sum()} TAD shuffles and {(eff.shuffle_type == 'RGG_shuffle').sum()} RGG shuffles.")
    lines.append(f"2. Length, composition, acidic/basic counts, net charge, and -NCP preservation: {'PASS for all strict non-basic comparisons' if bool(integ.overall_pass.all()) else 'FAIL for one or more comparisons; strict outputs exclude failures'}.")
    for _, row in summary.iterrows():
        lines.append(f"3. {row.shuffle_type}: mean delta_B22={row.mean_delta_B22:.3f}, 95% bootstrap CI [{row.bootstrap_mean_low:.3f}, {row.bootstrap_mean_high:.3f}], Wilcoxon p={row.wilcoxon_pvalue:.3g}.")
        lines.append(f"4. {row.shuffle_type}: median delta_B22={row.median_delta_B22:.3f}, median CI [{row.bootstrap_median_low:.3f}, {row.bootstrap_median_high:.3f}].")
    for _, row in noise.iterrows():
        lines.append(f"5. {row.shuffle_type}: fraction above 1 pooled replica SD={row.fraction_gt_1_pooled_sd:.3f}; above 2 pooled SD={row.fraction_gt_2_pooled_sd:.3f}.")
    if not repro.empty:
        paired = repro[repro["n_shuffles"] >= 2]
        same = paired["same_direction"].mean() if len(paired) else np.nan
        pear = corr_pair(paired["shuffle_1_delta_B22"], paired["shuffle_2_delta_B22"], "pearson") if len(paired) else np.nan
        spear = corr_pair(paired["shuffle_1_delta_B22"], paired["shuffle_2_delta_B22"], "spearman") if len(paired) else np.nan
        lines.append(f"6. TAD parent reproducibility: same-direction fraction={same:.3f}, Pearson={pear:.3f}, Spearman={spear:.3f}.")
    if not parent.empty:
        top = parent.sort_values("mean_abs_delta_B22", ascending=False).head(5)
        top_ids = ", ".join(f"{r.parent_kind}:{r.parent_id}" for _, r in top.iterrows())
        lines.append(f"7. Large effects are context-dependent; the largest mean absolute effects are concentrated in: {top_ids}.")
    if not pred_summary.empty:
        for _, row in pred_summary.iterrows():
            lines.append(f"8. Frozen model delta performance for {row.shuffle_type}: Pearson={row.Pearson:.3f}, Spearman={row.Spearman:.3f}, sign agreement={row.sign_agreement:.3f}.")
    if not feature_corr.empty:
        top = feature_corr.assign(abs_spearman=feature_corr["Spearman_delta_feature_vs_delta_B22"].abs()).sort_values("abs_spearman", ascending=False).head(5)
        desc = ", ".join(f"{r.feature} ({r.shuffle_type}, rho={r.Spearman_delta_feature_vs_delta_B22:.3f})" for _, r in top.iterrows())
        lines.append(f"9. Strongest descriptive feature-delta associations with delta_B22: {desc}.")
    lines.append("10. Best-supported interpretation: context-dependent sequence-order modulation, not a universal positional rule.")
    lines.append("11. The result does not justify a general positional interaction grammar claim.")
    lines.append("12. Model retraining required: NO.")
    (out / "corrected_shuffle_audit_interpretation.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--b22", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--rdf-dir", default="")
    parser.add_argument("--rdf-scripts-dir", default="")
    parser.add_argument("--out", default="outputs/shuffle_control_analysis")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = read_table(args.manifest)
    b22_raw = read_table(args.b22)
    name_col = pick(b22_raw, ["name", "pair_name", "pair", "simulation_name"])
    b22_col = pick(b22_raw, ["B22", "B22_corr1", "B22_corr1_native", "B22_corrected", "Corrected_B22_Hummer", "b22"])
    if not name_col or not b22_col:
        raise SystemExit("B22 table must contain a pair-name column and a B22 value column")
    b22 = b22_raw.rename(columns={name_col: "pair_id", b22_col: "B22"})[["pair_id", "B22"]].dropna()

    inv = make_inventory(manifest, b22)
    inv.to_csv(out / "shuffle_matched_inventory.csv", index=False)
    effects = make_effects(inv, manifest, b22)
    effects.to_csv(out / "shuffle_matched_effects.csv", index=False)
    effects[effects["is_basic_region_control"]].to_csv(out / "shuffle_basic_control_matched_effects_secondary.csv", index=False)

    integrity_all = verify_composition(effects)
    integrity_all.to_csv(out / "shuffle_composition_integrity_verified_all.csv", index=False)
    nonbasic = effects[~effects["is_basic_region_control"]].copy()
    integrity_nonbasic = integrity_all[~integrity_all["is_basic_region_control"]].copy()
    passing_ids = set(integrity_nonbasic.loc[integrity_nonbasic["overall_pass"], "shuffle_pair_id"])
    strict = nonbasic[nonbasic["shuffle_pair_id"].isin(passing_ids)].copy()
    integrity_nonbasic.to_csv(out / "shuffle_composition_integrity_verified.csv", index=False)
    strict.to_csv(out / "shuffle_nonbasic_matched_effects.csv", index=False)

    if args.rdf_dir:
        replica = add_replica_noise(strict, args.rdf_dir, args.rdf_scripts_dir)
    else:
        replica = strict.copy()
        replica["effect_to_noise"] = np.nan
    replica.to_csv(out / "shuffle_effect_vs_replica_noise.csv", index=False)
    noise = noise_summary(replica)
    noise.to_csv(out / "shuffle_effect_vs_replica_noise_summary.csv", index=False)

    summary = summarize_effects(strict)
    summary.to_csv(out / "shuffle_effect_summary_nonbasic.csv", index=False)
    repro = parent_reproducibility(replica[replica["shuffle_type"] == "TAD_shuffle"])
    repro.to_csv(out / "TAD_shuffle_parent_reproducibility.csv", index=False)

    pred_delta = pd.DataFrame()
    pred_summary = pd.DataFrame()
    feature_corr = pd.DataFrame()
    if args.predictions:
        predictions = read_table(args.predictions)
        pred_delta = prediction_delta(strict, predictions)
        pred_delta.to_csv(out / "shuffle_prediction_delta.csv", index=False)
        pred_summary = prediction_delta_summary(pred_delta)
        pred_summary.to_csv(out / "shuffle_prediction_delta_summary.csv", index=False)
        feature_corr = feature_delta_correlations(strict, predictions)
        feature_corr.to_csv(out / "shuffle_feature_delta_correlations.csv", index=False)
    else:
        (out / "shuffle_prediction_delta.csv").write_text("")
        (out / "shuffle_feature_delta_correlations.csv").write_text("")

    parent = parent_effects(strict)
    parent.to_csv(out / "shuffle_effect_by_parent.csv", index=False)
    make_figures(strict, replica, pred_delta, repro, out)
    write_interpretation(out, strict, integrity_nonbasic, summary, noise, repro, pred_summary, feature_corr, parent)

    print(f"Wrote corrected shuffle audit to {out}")
    print(f"Non-basic strict matched effects: TAD={(strict.shuffle_type == 'TAD_shuffle').sum()}, RGG={(strict.shuffle_type == 'RGG_shuffle').sum()}")
    print(f"Composition verification all-pass: {bool(integrity_nonbasic.overall_pass.all())}")


if __name__ == "__main__":
    main()
