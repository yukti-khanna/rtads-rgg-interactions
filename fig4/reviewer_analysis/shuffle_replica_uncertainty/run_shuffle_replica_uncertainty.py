#!/usr/bin/env python3
"""Pair-specific replica uncertainty for the validated 300 WT-shuffle panel.

This script reads completed RDF pickles only. It does not run CALVADOS, alter
sequences, regenerate shuffles, fit a model, or use the generic 917-pair
uncertainty to classify any comparison.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE_PANEL = ROOT / "reviewer_analysis" / "simulation_control_panels" / "FigS4_shuffle_sim_vs_sim_source.csv"
PANEL_MANIFEST = ROOT / "reviewer_analysis" / "control_simulation_manifest" / "final_dataset_panel_manifest_400.csv"
RDF_DIR = ROOT / "controls_rdfs"

sys.path.insert(0, str(ROOT))
from rdf_b22_functions import corrected_rdfs_b22  # noqa: E402


RUNS = [f"run_{i}" for i in range(1, 6)]
LAST_NM = 7.5
BOX_NM = 20.0
DELTA_CORRECTION = 2.0
RC_NM = 4.0
BOOTSTRAP_ITERATIONS = 50_000
NULL_ITERATIONS = 100_000
CLUSTER_ITERATIONS = 20_000
RANDOM_SEED = 20261005
TRANSFORMATION_SENSITIVE_THRESHOLD = 0.01
MEAN_REPRODUCTION_TOLERANCE = 1e-8
DPI = 600

MANIFEST_OUT = OUT / "shuffle_300_replica_manifest.csv"
REPLICA_OUT = OUT / "shuffle_replica_level_B22.csv"
SOURCE_OUT = OUT / "S6D_shuffle_replica_uncertainty_source_data.csv"
INVALID_OUT = OUT / "shuffle_invalid_bootstrap_cases.csv"
FOREST_PDF = OUT / "S6D_shuffle_effect_with_replica_uncertainty.pdf"
FOREST_PNG = OUT / "S6D_shuffle_effect_with_replica_uncertainty.png"
SUMMARY_PDF = OUT / "S6D_shuffle_uncertainty_summary.pdf"
SUMMARY_PNG = OUT / "S6D_shuffle_uncertainty_summary.png"
REPORT_OUT = OUT / "SHUFFLE_REPLICA_UNCERTAINTY_REPORT.md"
MANUSCRIPT_OUT = OUT / "shuffle_uncertainty_manuscript_updates.md"

COLORS = {"AHD shuffle": "#0072B2", "RG/RGG shuffle": "#D55E00"}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def b22_replicas(path: Path) -> dict[str, float]:
    frame = pd.read_pickle(require(path))
    if frame.shape[0] != 2 or list(frame.columns) != RUNS:
        raise ValueError(f"Unexpected RDF layout in {path}: shape={frame.shape}, columns={list(frame.columns)}")
    _, b22s, _ = corrected_rdfs_b22(
        frame,
        LAST_NM,
        box_size=BOX_NM,
        delta=DELTA_CORRECTION,
        rc_nm=RC_NM,
    )
    values = {run: float(b22s[run]["b22_corr1"]) for run in RUNS}
    if len(values) != 5 or not np.all(np.isfinite(list(values.values()))):
        raise ValueError(f"Expected five finite b22_corr1 values in {path}: {values}")
    return values


def component_is_shuffle(value: object) -> bool:
    return str(value).lower() == "shuffle"


def reconstruct_manifest(panel: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    if len(panel) != 300 or panel["shuffle_pair_id"].nunique() != 300:
        raise ValueError(f"Authoritative source must contain 300 unique comparisons; found {len(panel)} rows")
    if panel["parent_WT_pair_id"].nunique() != 100:
        raise ValueError(f"Expected 100 WT parents; found {panel['parent_WT_pair_id'].nunique()}")
    expected_types = {"AHD shuffle": 200, "RG/RGG shuffle": 100}
    if panel["shuffle_type"].value_counts().to_dict() != expected_types:
        raise ValueError(f"Unexpected shuffle counts: {panel['shuffle_type'].value_counts().to_dict()}")
    if controls["pair_name"].duplicated().any():
        raise ValueError("Control manifest contains duplicate pair_name values")
    lookup = controls.set_index("pair_name", drop=False)
    rows: list[dict] = []
    for number, row in enumerate(panel.sort_values("shuffle_pair_id").itertuples(index=False), start=1):
        if row.parent_WT_pair_id not in lookup.index or row.shuffle_pair_id not in lookup.index:
            raise ValueError(f"Panel IDs absent from control manifest: {row.parent_WT_pair_id}, {row.shuffle_pair_id}")
        wt = lookup.loc[row.parent_WT_pair_id]
        sh = lookup.loc[row.shuffle_pair_id]
        tad_shuffle = component_is_shuffle(sh["TAD_variant_type"])
        rgg_shuffle = component_is_shuffle(sh["RGG_variant_type"])
        if tad_shuffle == rgg_shuffle:
            raise ValueError(f"Expected exactly one shuffled component: {row.shuffle_pair_id}")
        inferred = "AHD shuffle" if tad_shuffle else "RG/RGG shuffle"
        if inferred != row.shuffle_type:
            raise ValueError(f"Shuffle-type mismatch for {row.shuffle_pair_id}: {row.shuffle_type} vs {inferred}")
        if wt["TAD_variant_type"] != "WT" or wt["RGG_variant_type"] != "WT":
            raise ValueError(f"Parent is not WT/WT: {row.parent_WT_pair_id}")
        if wt["TAD_parent"] != sh["TAD_parent"] or wt["RGG_parent"] != sh["RGG_parent"]:
            raise ValueError(f"Parent identifiers disagree for {row.shuffle_pair_id}")
        wt_rdf = Path(row.WT_RDF).resolve()
        sh_rdf = Path(row.shuffle_RDF).resolve()
        record = {
            "comparison_id": f"CMP_{number:03d}",
            "shuffle_type": row.shuffle_type,
            "WT_pair_id": row.parent_WT_pair_id,
            "shuffle_pair_id": row.shuffle_pair_id,
            "WT_AHD_id": wt["TAD_variant"],
            "WT_RGG_id": wt["RGG_variant"],
            "shuffle_sequence_id": sh["TAD_variant"] if tad_shuffle else sh["RGG_variant"],
            "WT_AHD_sequence": wt["TAD_sequence"],
            "WT_RGG_sequence": wt["RGG_sequence"],
            "shuffle_sequence": sh["TAD_sequence"] if tad_shuffle else sh["RGG_sequence"],
            "unchanged_partner_sequence": sh["RGG_sequence"] if tad_shuffle else sh["TAD_sequence"],
            "WT_RDF": str(wt_rdf),
            "shuffle_RDF": str(sh_rdf),
        }
        for i, run in enumerate(RUNS, start=1):
            record[f"WT_replica_{i}_id"] = f"{row.parent_WT_pair_id}::{run}"
            record[f"shuffle_replica_{i}_id"] = f"{row.shuffle_pair_id}::{run}"
        rows.append(record)
    out = pd.DataFrame(rows)
    if len(out) != 300 or out["WT_pair_id"].nunique() != 100:
        raise ValueError("Reconstructed manifest count check failed")
    return out


def calculate_replica_values(manifest: pd.DataFrame) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    paths: dict[str, Path] = {}
    memberships: dict[str, list[tuple[str, str, str]]] = {}
    for row in manifest.itertuples(index=False):
        paths[row.WT_pair_id] = Path(row.WT_RDF)
        paths[row.shuffle_pair_id] = Path(row.shuffle_RDF)
        memberships.setdefault(row.WT_pair_id, []).append((row.comparison_id, "WT", row.shuffle_type))
        memberships.setdefault(row.shuffle_pair_id, []).append((row.comparison_id, "shuffle", row.shuffle_type))
    if len(paths) != 400:
        raise ValueError(f"Expected 400 unique simulation systems; found {len(paths)}")
    values: dict[str, dict[str, float]] = {}
    rows: list[dict] = []
    for system_id in sorted(paths):
        path = paths[system_id]
        values[system_id] = b22_replicas(path)
        entries = memberships[system_id]
        role = entries[0][1]
        comparison_ids = ";".join(x[0] for x in entries)
        shuffle_types = ";".join(sorted(set(x[2] for x in entries)))
        for run in RUNS:
            rows.append(
                {
                    "system_id": system_id,
                    "comparison_id": comparison_ids,
                    "WT_or_shuffle": role,
                    "shuffle_type": shuffle_types,
                    "replica": int(run.split("_")[1]),
                    "replica_identifier": f"{system_id}::{run}",
                    "B22": values[system_id][run],
                    "source_file": str(path.resolve()),
                    "calculation_status": "verified_finite_b22_corr1",
                    "rdf_processing": "corrected_rdfs_b22(last=7.5, box_size=20, delta=2, rc_nm=4.0)",
                }
            )
    return values, pd.DataFrame(rows)


def log_strength(mean_b22: np.ndarray | float) -> np.ndarray | float:
    return np.log10(-mean_b22)


def bootstrap_delta(
    wt: np.ndarray,
    sh: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
    batch_size: int = 10_000,
) -> tuple[np.ndarray, float]:
    valid_values: list[np.ndarray] = []
    invalid = 0
    for start in range(0, iterations, batch_size):
        n = min(batch_size, iterations - start)
        wmean = wt[rng.integers(0, 5, size=(n, 5))].mean(axis=1)
        smean = sh[rng.integers(0, 5, size=(n, 5))].mean(axis=1)
        valid = (wmean < 0) & (smean < 0)
        invalid += int((~valid).sum())
        valid_values.append(np.log10(-smean[valid]) - np.log10(-wmean[valid]))
    return np.concatenate(valid_values), invalid / iterations


def null_bootstrap_p(
    wt: np.ndarray,
    sh: np.ndarray,
    observed_delta: float,
    rng: np.random.Generator,
    iterations: int,
    batch_size: int = 10_000,
) -> tuple[float, float]:
    pooled_location = (wt.mean() + sh.mean()) / 2.0
    wt_null = wt - wt.mean() + pooled_location
    sh_null = sh - sh.mean() + pooled_location
    exceed = 0
    valid_total = 0
    invalid = 0
    for start in range(0, iterations, batch_size):
        n = min(batch_size, iterations - start)
        wmean = wt_null[rng.integers(0, 5, size=(n, 5))].mean(axis=1)
        smean = sh_null[rng.integers(0, 5, size=(n, 5))].mean(axis=1)
        valid = (wmean < 0) & (smean < 0)
        invalid += int((~valid).sum())
        delta = np.log10(-smean[valid]) - np.log10(-wmean[valid])
        exceed += int((np.abs(delta) >= abs(observed_delta)).sum())
        valid_total += len(delta)
    fraction_invalid = invalid / iterations
    if valid_total == 0 or fraction_invalid >= TRANSFORMATION_SENSITIVE_THRESHOLD:
        return np.nan, fraction_invalid
    return (exceed + 1) / (valid_total + 1), fraction_invalid


def adjusted_pvalues(pvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(p)
    conservative = np.where(finite, p, 1.0)
    m = len(conservative)
    order = np.argsort(conservative)
    ranked = conservative[order]
    holm_sorted = np.maximum.accumulate((m - np.arange(m)) * ranked)
    holm_sorted = np.clip(holm_sorted, 0, 1)
    holm = np.empty(m)
    holm[order] = holm_sorted
    bh_sorted = ranked * m / np.arange(1, m + 1)
    bh_sorted = np.minimum.accumulate(bh_sorted[::-1])[::-1]
    bh_sorted = np.clip(bh_sorted, 0, 1)
    bh = np.empty(m)
    bh[order] = bh_sorted
    return holm, bh


def analyze_comparisons(
    manifest: pd.DataFrame,
    replica_values: dict[str, dict[str, float]],
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    panel_lookup = panel.set_index("shuffle_pair_id")
    rng = np.random.default_rng(RANDOM_SEED)
    rows: list[dict] = []
    for row in manifest.itertuples(index=False):
        wt = np.array([replica_values[row.WT_pair_id][run] for run in RUNS], dtype=float)
        sh = np.array([replica_values[row.shuffle_pair_id][run] for run in RUNS], dtype=float)
        wt_mean = float(wt.mean())
        sh_mean = float(sh.mean())
        if wt_mean >= 0 or sh_mean >= 0:
            raise ValueError(f"Observed mean has undefined log strength: {row.comparison_id}")
        wt_strength = float(log_strength(wt_mean))
        sh_strength = float(log_strength(sh_mean))
        observed = sh_strength - wt_strength
        boot, invalid_fraction = bootstrap_delta(wt, sh, rng, BOOTSTRAP_ITERATIONS)
        if len(boot) == 0:
            raise ValueError(f"No valid bootstrap draws: {row.comparison_id}")
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        raw_p, invalid_null = null_bootstrap_p(wt, sh, observed, rng, NULL_ITERATIONS)
        out = {
            "comparison_id": row.comparison_id,
            "shuffle_type": row.shuffle_type,
            "WT_pair_id": row.WT_pair_id,
            "shuffle_pair_id": row.shuffle_pair_id,
        }
        for i, value in enumerate(wt, start=1):
            out[f"WT_rep{i}_B22"] = value
        for i, value in enumerate(sh, start=1):
            out[f"shuffle_rep{i}_B22"] = value
        out.update(
            {
                "WT_mean_B22": wt_mean,
                "shuffle_mean_B22": sh_mean,
                "WT_log_strength": wt_strength,
                "shuffle_log_strength": sh_strength,
                "Delta": observed,
                "abs_Delta": abs(observed),
                "bootstrap_mean_Delta": float(np.mean(boot)),
                "bootstrap_SE_Delta": float(np.std(boot, ddof=1)),
                "CI95_low": float(ci_low),
                "CI95_high": float(ci_high),
                "probability_Delta_gt_0": float(np.mean(boot > 0)),
                "probability_Delta_lt_0": float(np.mean(boot < 0)),
                "fraction_invalid_bootstrap": float(invalid_fraction),
                "fraction_invalid_null_bootstrap": float(invalid_null),
                "transformation_sensitive": bool(
                    invalid_fraction >= TRANSFORMATION_SENSITIVE_THRESHOLD
                    or invalid_null >= TRANSFORMATION_SENSITIVE_THRESHOLD
                ),
                "raw_bootstrap_p": raw_p,
                "CI_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            }
        )
        current = panel_lookup.loc[row.shuffle_pair_id]
        out["current_WT_mean_B22"] = float(current["WT_B22_Hummer"])
        out["current_shuffle_mean_B22"] = float(current["shuffle_B22_Hummer"])
        out["WT_mean_difference_vs_current"] = wt_mean - out["current_WT_mean_B22"]
        out["shuffle_mean_difference_vs_current"] = sh_mean - out["current_shuffle_mean_B22"]
        rows.append(out)
    result = pd.DataFrame(rows)
    # Preserve both requested names: the detailed source-data specification uses
    # raw_bootstrap_p, while the multiple-testing section calls the field raw_p.
    result["raw_p"] = result["raw_bootstrap_p"]
    holm, bh = adjusted_pvalues(result["raw_bootstrap_p"].to_numpy(float))
    result["holm_adjusted_p"] = holm
    result["holm_significant"] = result["holm_adjusted_p"] <= 0.05
    result["BH_q_value"] = bh
    result["BH_significant"] = result["BH_q_value"] <= 0.05
    max_wt = float(result["WT_mean_difference_vs_current"].abs().max())
    max_sh = float(result["shuffle_mean_difference_vs_current"].abs().max())
    reproduced = max_wt <= MEAN_REPRODUCTION_TOLERANCE and max_sh <= MEAN_REPRODUCTION_TOLERANCE
    validation = {
        "max_abs_WT_mean_difference": max_wt,
        "max_abs_shuffle_mean_difference": max_sh,
        "reproduced": reproduced,
    }
    if not reproduced:
        raise ValueError(f"Current plot values were not reproduced: {validation}")
    return result, validation


def effect_summary(frame: pd.DataFrame) -> dict[str, float]:
    d = frame["Delta"].to_numpy(float)
    a = np.abs(d)
    q1, q3 = np.percentile(d, [25, 75])
    aq1, aq3 = np.percentile(a, [25, 75])
    return {
        "n": int(len(frame)),
        "mean_signed_Delta": float(np.mean(d)),
        "median_signed_Delta": float(np.median(d)),
        "SD_signed_Delta": float(np.std(d, ddof=1)),
        "IQR_signed_Delta": float(q3 - q1),
        "median_absolute_Delta": float(np.median(a)),
        "IQR_absolute_Delta": float(aq3 - aq1),
        "absolute_Delta_p90": float(np.percentile(a, 90)),
        "absolute_Delta_p95": float(np.percentile(a, 95)),
        "RMS_Delta": float(np.sqrt(np.mean(d**2))),
        "fraction_abs_Delta_gt_0.025": float(np.mean(a > 0.025)),
        "fraction_abs_Delta_gt_0.05": float(np.mean(a > 0.05)),
        "fraction_abs_Delta_gt_0.10": float(np.mean(a > 0.10)),
    }


def cluster_bootstrap(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    parents = sorted(frame["WT_pair_id"].unique())
    if len(parents) != 100:
        raise ValueError(f"Expected 100 WT clusters; found {len(parents)}")
    parent_index = {p: i for i, p in enumerate(parents)}
    pidx = frame["WT_pair_id"].map(parent_index).to_numpy(int)
    if not np.all(np.bincount(pidx, minlength=100) == 3):
        raise ValueError("Every WT parent must contribute exactly three comparisons")
    delta = frame["Delta"].to_numpy(float)
    rng = np.random.default_rng(RANDOM_SEED + 1)
    means = np.empty(CLUSTER_ITERATIONS)
    medians = np.empty(CLUSTER_ITERATIONS)
    median_abs = np.empty(CLUSTER_ITERATIONS)
    rms = np.empty(CLUSTER_ITERATIONS)
    dorder = np.argsort(delta)
    aorder = np.argsort(np.abs(delta))
    for start in range(0, CLUSTER_ITERATIONS, 500):
        n = min(500, CLUSTER_ITERATIONS - start)
        samples = rng.integers(0, 100, size=(n, 100))
        counts = np.zeros((n, 100), dtype=np.int16)
        row_ids = np.repeat(np.arange(n), 100)
        np.add.at(counts, (row_ids, samples.ravel()), 1)
        weights = counts[:, pidx]
        means[start : start + n] = (weights @ delta) / 300.0
        rms[start : start + n] = np.sqrt((weights @ (delta**2)) / 300.0)
        dcum = np.cumsum(weights[:, dorder], axis=1)
        acum = np.cumsum(weights[:, aorder], axis=1)
        dlo = np.argmax(dcum >= 150, axis=1)
        dhi = np.argmax(dcum >= 151, axis=1)
        alo = np.argmax(acum >= 150, axis=1)
        ahi = np.argmax(acum >= 151, axis=1)
        medians[start : start + n] = (delta[dorder[dlo]] + delta[dorder[dhi]]) / 2.0
        abs_delta = np.abs(delta)
        median_abs[start : start + n] = (abs_delta[aorder[alo]] + abs_delta[aorder[ahi]]) / 2.0
    return {
        "mean_signed_Delta": tuple(np.percentile(means, [2.5, 97.5])),
        "median_signed_Delta": tuple(np.percentile(medians, [2.5, 97.5])),
        "median_absolute_Delta": tuple(np.percentile(median_abs, [2.5, 97.5])),
        "RMS_Delta": tuple(np.percentile(rms, [2.5, 97.5])),
    }


def plot_forest(frame: pd.DataFrame, summary_all: dict[str, float]) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.1), gridspec_kw={"width_ratios": [2, 1]}, sharex=True)
    for ax, shuffle_type in zip(axes, ["AHD shuffle", "RG/RGG shuffle"]):
        sub = frame[frame["shuffle_type"] == shuffle_type].sort_values("Delta").reset_index(drop=True)
        y = np.arange(len(sub))
        colors = np.where(
            sub["holm_significant"],
            "#7A0177",
            np.where(sub["CI_excludes_zero"], COLORS[shuffle_type], "#A8A8A8"),
        )
        ax.hlines(y, sub["CI95_low"], sub["CI95_high"], color=colors, linewidth=0.55, alpha=0.9)
        ax.scatter(sub["Delta"], y, c=colors, s=5.5, linewidths=0, zorder=3)
        ax.axvline(0, color="#222222", lw=0.7, ls="--")
        ax.set_title(f"{shuffle_type} (n = {len(sub)})")
        ax.set_yticks([])
        ax.set_ylabel("Comparisons sorted by observed Δ")
        ax.set_xlabel(r"Δ = $log_{10}(-\overline{B}_{22,shuffle}) - log_{10}(-\overline{B}_{22,WT})$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    lim = max(abs(frame["CI95_low"].min()), abs(frame["CI95_high"].max())) * 1.04
    for ax in axes:
        ax.set_xlim(-lim, lim)
    text = (
        f"median |Δ| = {summary_all['median_absolute_Delta']:.3f}\n"
        f"RMS Δ = {summary_all['RMS_Delta']:.3f}\n"
        f"CI excludes 0: {int(frame['CI_excludes_zero'].sum())}/300\n"
        f"Holm: {int(frame['holm_significant'].sum())}/300\n"
        f"BH-FDR: {int(frame['BH_significant'].sum())}/300"
    )
    axes[0].text(0.02, 0.98, text, transform=axes[0].transAxes, ha="left", va="top", fontsize=6,
                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
    fig.tight_layout(w_pad=1.6)
    fig.savefig(FOREST_PDF, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FOREST_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_summary(frame: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.55))
    ax = axes[0]
    for shuffle_type, group in frame.groupby("shuffle_type"):
        ax.scatter(group["abs_Delta"], group["bootstrap_SE_Delta"], s=9, alpha=0.65,
                   color=COLORS[shuffle_type], linewidths=0, label=shuffle_type)
    high = max(frame["abs_Delta"].max(), frame["bootstrap_SE_Delta"].max())
    ax.plot([0, high], [0, high], color="#777777", lw=0.7, ls="--", label="|Δ| = bootstrap SE")
    ax.set_xlim(0, high * 1.03)
    ax.set_ylim(0, high * 1.03)
    ax.set_xlabel("Absolute observed Δ")
    ax.set_ylabel("Pair-specific bootstrap SE of Δ")
    ax.legend(frameon=False, fontsize=5.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = axes[1]
    for column, color, label in [
        ("abs_Delta", "#222222", "|observed Δ|"),
        ("bootstrap_SE_Delta", "#7A7A7A", "bootstrap SE(Δ)"),
    ]:
        x = np.sort(frame[column].to_numpy(float))
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=color, lw=1.2, label=label)
    ax.set_xlabel("Log-strength units")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(SUMMARY_PDF, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(SUMMARY_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fmt_ci(ci: tuple[float, float]) -> str:
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def build_report(
    frame: pd.DataFrame,
    validation: dict,
    summaries: dict[str, dict[str, float]],
    cluster: dict[str, tuple[float, float]],
    pearson: float,
    spearman: float,
) -> None:
    all_s = summaries["All"]
    invalid = frame[(frame["fraction_invalid_bootstrap"] > 0) | (frame["fraction_invalid_null_bootstrap"] > 0)]
    lines = [
        "# Shuffle replica uncertainty analysis",
        "",
        "## Scope and validation",
        "",
        "This analysis uses the exact validated 300-comparison WT-versus-shuffle source panel: 100 WT parents, 200 AHD-shuffle comparisons, and 100 RG/RGG-shuffle comparisons. Each of the 400 unique simulation systems was processed from its completed five-run RDF pickle. No molecular-dynamics simulation was rerun, no sequence was changed, and no model prediction was used.",
        "",
        f"Replica RDFs were converted to B22 with the existing project function `corrected_rdfs_b22(last={LAST_NM}, box_size={BOX_NM:g}, delta={DELTA_CORRECTION:g}, rc_nm={RC_NM})`. Interaction strength is `log10(-mean(B22_replica1...5))`, not the mean of replica log strengths.",
        "",
        f"Maximum absolute disagreement with the current panel means was {validation['max_abs_WT_mean_difference']:.3g} for WT and {validation['max_abs_shuffle_mean_difference']:.3g} for shuffles. Current plotted values reproduced: **{validation['reproduced']}**.",
        "",
        "## A. Landscape preservation",
        "",
        f"Across all 300 comparisons, Pearson r = {pearson:.4f} and Spearman rho = {spearman:.4f}. Thus, WT and composition-preserving shuffle strengths remain strongly correlated at the landscape level.",
        "",
        "## B. Effect magnitude",
        "",
        f"The mean signed Delta was {all_s['mean_signed_Delta']:.4f}, median signed Delta {all_s['median_signed_Delta']:.4f}, median absolute Delta {all_s['median_absolute_Delta']:.4f}, and RMS Delta {all_s['RMS_Delta']:.4f} log units.",
        "",
        "| Group | n | Mean Δ | Median Δ | SD Δ | IQR Δ | Median |Δ| | IQR |Δ| | P90 |Δ| | P95 |Δ| | RMS Δ | frac |Δ|>0.025 | frac |Δ|>0.05 | frac |Δ|>0.10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {s['n']} | {s['mean_signed_Delta']:.4f} | {s['median_signed_Delta']:.4f} | "
            f"{s['SD_signed_Delta']:.4f} | {s['IQR_signed_Delta']:.4f} | {s['median_absolute_Delta']:.4f} | "
            f"{s['IQR_absolute_Delta']:.4f} | {s['absolute_Delta_p90']:.4f} | {s['absolute_Delta_p95']:.4f} | "
            f"{s['RMS_Delta']:.4f} | {s['fraction_abs_Delta_gt_0.025']:.3f} | "
            f"{s['fraction_abs_Delta_gt_0.05']:.3f} | {s['fraction_abs_Delta_gt_0.10']:.3f} |"
        )
    lines += [
        "",
        "## C. Resolution relative to simulation uncertainty",
        "",
        f"Independent replica bootstraps used {BOOTSTRAP_ITERATIONS:,} iterations per comparison with seed {RANDOM_SEED}. The median pair-specific bootstrap SE of Delta was {frame['bootstrap_SE_Delta'].median():.4f} log units, compared with the rough 0.05-log-unit expectation from the separate 917-pair convergence analysis. This comparison is contextual only; all individual classifications here use the actual WT and shuffle replicas.",
        "",
        f"Unadjusted percentile CIs excluded zero for {int(frame['CI_excludes_zero'].sum())}/300 comparisons; {int(frame['holm_significant'].sum())}/300 survived Holm family-wise correction, and {int(frame['BH_significant'].sum())}/300 survived exploratory BH-FDR correction.",
        "",
    ]
    for group in ["AHD shuffle", "RG/RGG shuffle"]:
        g = frame[frame["shuffle_type"] == group]
        lines.append(
            f"- {group}: CI excludes zero {int(g['CI_excludes_zero'].sum())}/{len(g)}; "
            f"Holm-significant {int(g['holm_significant'].sum())}/{len(g)}; "
            f"BH-significant {int(g['BH_significant'].sum())}/{len(g)}."
        )
    lines += [
        "",
        "## Bootstrap null and multiple testing",
        "",
        f"For each comparison, WT and shuffle B22 replicates were centred separately to the common B22-space location `(mean_WT + mean_shuffle)/2`, preserving the residual pattern within each group. Five null WT and five null shuffle values were then independently sampled with replacement for {NULL_ITERATIONS:,} iterations. The two-sided P value is the fraction of valid null draws with `|Delta_null| >= |Delta_observed|`, with a plus-one correction. Holm correction at alpha=0.05 was applied across all 300 tests; BH-FDR at q=0.05 is secondary.",
        "",
        "## Invalid log-transform draws",
        "",
        f"{len(invalid)} comparisons had at least one invalid bootstrap or null-bootstrap draw caused by a non-negative resampled mean B22. Comparisons with invalid frequency >= {TRANSFORMATION_SENSITIVE_THRESHOLD:.1%} were flagged as transformation-sensitive; their raw null P value was not forced and was treated conservatively as 1.0 only for the 300-test multiplicity adjustment.",
        "",
        "## WT-parent cluster bootstrap",
        "",
        f"A {CLUSTER_ITERATIONS:,}-iteration cluster bootstrap sampled the 100 WT parent blocks with replacement and retained all three associated shuffle comparisons per selected parent. The aggregate 95% CIs were:",
        "",
        f"- Mean signed Delta: {fmt_ci(cluster['mean_signed_Delta'])}",
        f"- Median signed Delta: {fmt_ci(cluster['median_signed_Delta'])}",
        f"- Median absolute Delta: {fmt_ci(cluster['median_absolute_Delta'])}",
        f"- RMS Delta: {fmt_ci(cluster['RMS_Delta'])}",
        "",
        "## Interpretation",
        "",
    ]
    corrected = int(frame["holm_significant"].sum())
    if corrected == 0:
        interpretation = (
            "Composition-preserving shuffling broadly preserves the simulated interaction landscape, and most pair-specific shifts are not distinguishable from replica-level uncertainty under the present five-replica sampling. A subset may show larger effects, but evidence for systematic sequence-order effects is limited."
        )
    else:
        interpretation = (
            "Composition-preserving shuffling broadly preserves the global interaction landscape, while a subset of individual pairs shows shuffle-induced changes that exceed replica-level simulation uncertainty, supporting pair-specific modulation by residue order."
        )
    lines.append(interpretation)
    REPORT_OUT.write_text("\n".join(lines) + "\n")


def build_manuscript_updates(
    frame: pd.DataFrame,
    summaries: dict[str, dict[str, float]],
    cluster: dict[str, tuple[float, float]],
    pearson: float,
    spearman: float,
) -> None:
    s = summaries["All"]
    ci_n = int(frame["CI_excludes_zero"].sum())
    holm_n = int(frame["holm_significant"].sum())
    bh_n = int(frame["BH_significant"].sum())
    if holm_n == 0:
        proportional = "Most pair-specific shifts were not distinguishable from replica-level uncertainty after family-wise correction, limiting evidence for systematic sequence-order effects."
    else:
        proportional = f"A subset ({holm_n}/300) remained distinguishable from replica-level uncertainty after Holm correction, supporting pair-specific sequence-order modulation within an otherwise preserved landscape."
    text = f"""# Candidate manuscript and reviewer-response updates

These are update candidates only; no manuscript or response document was edited.

## 1. Figure 3 Results paragraph

Composition-preserving shuffling broadly preserved the simulated interaction-strength landscape (Pearson r = {pearson:.3f}; Spearman rho = {spearman:.3f}; 300 matched comparisons from 100 WT parents). Shuffle-induced changes were generally modest (median |Delta| = {s['median_absolute_Delta']:.3f}; RMS Delta = {s['RMS_Delta']:.3f} log10 units). Pair-specific independent-replica bootstrapping gave 95% CIs excluding zero for {ci_n}/300 comparisons before multiplicity adjustment; {holm_n}/300 survived Holm family-wise correction and {bh_n}/300 survived exploratory BH-FDR correction. {proportional}

## 2. Supplementary Figure S6D legend

**S6D. Pair-specific uncertainty of composition-preserving shuffle effects.** Observed Delta is defined as log10(-mean B22_shuffle) - log10(-mean B22_WT), where each mean is calculated from five independent simulation replicas. Points show observed Delta and horizontal bars show 95% percentile intervals from {BOOTSTRAP_ITERATIONS:,} independent WT/shuffle replica-bootstrap iterations. Comparisons are sorted by observed Delta and separated into AHD- and RG/RGG-shuffle controls. Colour identifies unadjusted intervals excluding zero; Holm-significant comparisons are distinguished separately. Across 300 comparisons, median |Delta| = {s['median_absolute_Delta']:.3f}, RMS Delta = {s['RMS_Delta']:.3f}, {ci_n}/300 unadjusted intervals excluded zero, {holm_n}/300 survived Holm correction, and {bh_n}/300 survived exploratory BH-FDR correction.

## 3. STAR Methods: Composition-preserving sequence-shuffle controls

For each WT-shuffle comparison, B22 was evaluated independently for five WT and five shuffle simulation replicas using the same RDF processing and finite-size-corrected integration employed for the primary analysis (`corrected_rdfs_b22`, last = 7.5 nm, box size = 20 nm, delta = 2, rc = 4.0 nm). Interaction strength was calculated as log10(-mean(B22_replica1...5)); replica log values were not averaged. The shuffle effect was Delta = log10(-mean B22_shuffle) - log10(-mean B22_WT). Pair-specific uncertainty was estimated by independently resampling the five WT and five shuffle replicas with replacement for {BOOTSTRAP_ITERATIONS:,} iterations and reporting percentile 95% CIs. Non-negative bootstrap means, for which the log transform is undefined, were counted explicitly. Two-sided bootstrap P values were obtained from {NULL_ITERATIONS:,} centred-null iterations in which WT and shuffle B22 residuals were retained after centring both groups to their common B22-space mean. Holm correction across all 300 comparisons was the primary family-wise procedure; Benjamini-Hochberg FDR at q = 0.05 was secondary. Aggregate uncertainty was estimated with a {CLUSTER_ITERATIONS:,}-iteration bootstrap over the 100 WT-parent blocks, retaining all shuffle comparisons associated with each resampled parent.

## 4. Reviewer response R1-05

We added a pair-specific uncertainty analysis for the complete 300-comparison WT-shuffle panel. Rather than applying the generic uncertainty estimated from the broader 917-pair convergence analysis, we recalculated replica-level B22 for the exact five WT and five shuffle replicas underlying every comparison and independently bootstrapped each matched effect. The global WT-shuffle relationship remained strong (Pearson r = {pearson:.3f}; Spearman rho = {spearman:.3f}), with median |Delta| = {s['median_absolute_Delta']:.3f} and RMS Delta = {s['RMS_Delta']:.3f}. Unadjusted 95% CIs excluded zero for {ci_n}/300 comparisons; {holm_n}/300 survived Holm correction and {bh_n}/300 survived exploratory BH-FDR correction. {proportional}

## 5. Other shuffle-control response text

Where the response currently states that shuffling has "no effect," replace that categorical wording with: "Composition-preserving shuffling broadly preserved the global simulated interaction landscape, while pair-specific changes were evaluated relative to the replica uncertainty of the exact WT and shuffle ensembles." Report the Holm-corrected result as primary and the BH-FDR result as exploratory.

## Aggregate cluster-bootstrap context

The WT-parent cluster-bootstrap 95% CIs were {fmt_ci(cluster['mean_signed_Delta'])} for mean signed Delta, {fmt_ci(cluster['median_signed_Delta'])} for median signed Delta, {fmt_ci(cluster['median_absolute_Delta'])} for median absolute Delta, and {fmt_ci(cluster['RMS_Delta'])} for RMS Delta.
"""
    MANUSCRIPT_OUT.write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(require(SOURCE_PANEL))
    controls = pd.read_csv(require(PANEL_MANIFEST))
    manifest = reconstruct_manifest(panel, controls)
    manifest.to_csv(MANIFEST_OUT, index=False)
    replica_values, replica_table = calculate_replica_values(manifest)
    replica_table.to_csv(REPLICA_OUT, index=False)
    analyzed, validation = analyze_comparisons(manifest, replica_values, panel)
    analyzed.to_csv(SOURCE_OUT, index=False)
    invalid = analyzed[(analyzed["fraction_invalid_bootstrap"] > 0) | (analyzed["fraction_invalid_null_bootstrap"] > 0)]
    invalid.to_csv(INVALID_OUT, index=False)

    pearson = float(stats.pearsonr(analyzed["WT_log_strength"], analyzed["shuffle_log_strength"]).statistic)
    spearman = float(stats.spearmanr(analyzed["WT_log_strength"], analyzed["shuffle_log_strength"]).statistic)
    summaries = {
        "All": effect_summary(analyzed),
        "AHD shuffle": effect_summary(analyzed[analyzed["shuffle_type"] == "AHD shuffle"]),
        "RG/RGG shuffle": effect_summary(analyzed[analyzed["shuffle_type"] == "RG/RGG shuffle"]),
    }
    cluster = cluster_bootstrap(analyzed)
    plot_forest(analyzed, summaries["All"])
    plot_summary(analyzed)
    build_report(analyzed, validation, summaries, cluster, pearson, spearman)
    build_manuscript_updates(analyzed, summaries, cluster, pearson, spearman)

    median_ci_width = float((analyzed["CI95_high"] - analyzed["CI95_low"]).median())
    print(f"TOTAL MATCHED COMPARISONS: {len(analyzed)}")
    print(f"AHD SHUFFLES: {(analyzed['shuffle_type'] == 'AHD shuffle').sum()}")
    print(f"RG/RGG SHUFFLES: {(analyzed['shuffle_type'] == 'RG/RGG shuffle').sum()}")
    print("ALL HAVE 5 WT REPLICAS: YES")
    print("ALL HAVE 5 SHUFFLE REPLICAS: YES")
    print(f"CURRENT-PLOT VALUES REPRODUCED: {validation['reproduced']}")
    print(f"MEAN SIGNED DELTA: {summaries['All']['mean_signed_Delta']:.6f}")
    print(f"MEDIAN SIGNED DELTA: {summaries['All']['median_signed_Delta']:.6f}")
    print(f"MEDIAN ABSOLUTE DELTA: {summaries['All']['median_absolute_Delta']:.6f}")
    print(f"RMS DELTA: {summaries['All']['RMS_Delta']:.6f}")
    print(f"MEDIAN PAIR-SPECIFIC BOOTSTRAP SE: {analyzed['bootstrap_SE_Delta'].median():.6f}")
    print(f"MEDIAN CI WIDTH: {median_ci_width:.6f}")
    print(f"UNADJUSTED CI EXCLUDES ZERO: {int(analyzed['CI_excludes_zero'].sum())}/300")
    print(f"HOLM-SIGNIFICANT: {int(analyzed['holm_significant'].sum())}/300")
    print(f"BH-FDR-SIGNIFICANT: {int(analyzed['BH_significant'].sum())}/300")
    for label in ["AHD shuffle", "RG/RGG shuffle"]:
        group = analyzed[analyzed["shuffle_type"] == label]
        print(f"{label.upper()}:")
        print(f"  CI EXCLUDES ZERO: {int(group['CI_excludes_zero'].sum())}/{len(group)}")
        print(f"  HOLM-SIGNIFICANT: {int(group['holm_significant'].sum())}/{len(group)}")
        print(f"  BH-SIGNIFICANT: {int(group['BH_significant'].sum())}/{len(group)}")
    print("CLUSTER-BOOTSTRAP 95% CI:")
    print(f"  MEAN SIGNED DELTA: {fmt_ci(cluster['mean_signed_Delta'])}")
    print(f"  MEDIAN SIGNED DELTA: {fmt_ci(cluster['median_signed_Delta'])}")
    print(f"  MEDIAN ABSOLUTE DELTA: {fmt_ci(cluster['median_absolute_Delta'])}")
    print(f"  RMS DELTA: {fmt_ci(cluster['RMS_Delta'])}")
    print(f"COMPARISONS WITH INVALID BOOTSTRAP LOG DRAWS: {len(invalid)}")
    print(f"NEW S6D CREATED: {FOREST_PDF}")
    print(f"SOURCE DATA CREATED: {SOURCE_OUT}")
    print(f"REPORT CREATED: {REPORT_OUT}")
    print(f"MANUSCRIPT UPDATE NOTES CREATED: {MANUSCRIPT_OUT}")
    print("OUTPUT PATHS:")
    for path in [
        MANIFEST_OUT, REPLICA_OUT, SOURCE_OUT, INVALID_OUT,
        FOREST_PDF, FOREST_PNG, SUMMARY_PDF, SUMMARY_PNG,
        REPORT_OUT, MANUSCRIPT_OUT,
    ]:
        print(path)


if __name__ == "__main__":
    main()
