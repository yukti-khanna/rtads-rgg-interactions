#!/usr/bin/env python3
"""Parent-aware global finite-sampling null for the 300 WT-shuffle comparisons.

Uses only the completed replica-level B22 tables from the existing uncertainty
analysis. No trajectories, RDFs, simulations, sequences, or predictor analyses
are read or modified.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
PARENT = OUT.parent
MANIFEST_PATH = PARENT / "shuffle_300_replica_manifest.csv"
REPLICA_PATH = PARENT / "shuffle_replica_level_B22.csv"
PAIR_SOURCE_PATH = PARENT / "S6D_shuffle_replica_uncertainty_source_data.csv"
OLD_SUMMARY_PDF = PARENT / "S6D_shuffle_uncertainty_summary.pdf"

PARENT_STRUCTURE_CSV = OUT / "global_null_parent_structure.csv"
SUMMARY_CSV = OUT / "global_null_summary.csv"
INSET_SOURCE_CSV = OUT / "S6F_global_null_inset_source_data.csv"
S6F_PDF = OUT / "S6F_shuffle_effect_with_replica_uncertainty.pdf"
S6F_PNG = OUT / "S6F_shuffle_effect_with_replica_uncertainty.png"
INSET_PDF = OUT / "S6F_global_null_inset.pdf"
INSET_PNG = OUT / "S6F_global_null_inset.png"
AUDIT_SUMMARY_PDF = OUT / "shuffle_uncertainty_summary_audit.pdf"
REPORT_MD = OUT / "GLOBAL_SHUFFLE_NULL_REPORT.md"
NOTES_MD = OUT / "S6F_manuscript_notes.md"

ITERATIONS = 50_000
RANDOM_SEED = 20261005
BATCH_SIZE = 500
DPI = 600
RUNS = [1, 2, 3, 4, 5]
GROUP_COLORS = {"AHD shuffle": "#0072B2", "RG/RGG shuffle": "#D55E00"}
STAT_LABELS = {
    "median_absolute_delta": "Median |Δ|",
    "RMS_delta": "RMS Δ",
    "mean_signed_delta": "Mean signed Δ",
    "median_signed_delta": "Median signed Δ",
}


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


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(require(MANIFEST_PATH))
    replica = pd.read_csv(require(REPLICA_PATH))
    pairs = pd.read_csv(require(PAIR_SOURCE_PATH))
    if len(manifest) != 300 or manifest["comparison_id"].nunique() != 300:
        raise ValueError("Expected exactly 300 unique comparisons in the manifest")
    if manifest["WT_pair_id"].nunique() != 100:
        raise ValueError("Expected exactly 100 WT parents")
    counts = manifest["shuffle_type"].value_counts().to_dict()
    if counts != {"AHD shuffle": 200, "RG/RGG shuffle": 100}:
        raise ValueError(f"Unexpected shuffle counts: {counts}")
    if replica["system_id"].nunique() != 400 or len(replica) != 2000:
        raise ValueError(f"Expected 400 systems and 2,000 replica rows; found {replica['system_id'].nunique()}, {len(replica)}")
    per_system = replica.groupby("system_id").size()
    if not (per_system == 5).all() or not np.isfinite(replica["B22"]).all():
        raise ValueError("Every system must have exactly five finite replica B22 values")
    if len(pairs) != 300 or pairs["comparison_id"].nunique() != 300:
        raise ValueError("Pair-specific source table does not contain exactly 300 comparisons")
    merged_ids = manifest[["comparison_id", "WT_pair_id", "shuffle_pair_id"]].merge(
        pairs[["comparison_id", "WT_pair_id", "shuffle_pair_id"]],
        on=["comparison_id", "WT_pair_id", "shuffle_pair_id"],
        how="outer",
        indicator=True,
    )
    if not (merged_ids["_merge"] == "both").all():
        raise ValueError("Manifest and pair-specific source identifiers do not match")
    return manifest, replica, pairs


def replica_array(replica: pd.DataFrame, system_id: str) -> np.ndarray:
    sub = replica[replica["system_id"] == system_id].sort_values("replica")
    values = sub["B22"].to_numpy(float)
    if len(values) != 5 or not np.array_equal(sub["replica"].to_numpy(int), np.array(RUNS)):
        raise ValueError(f"Missing or duplicated replicas for {system_id}")
    return values


def reconstruct_parent_structure(
    manifest: pd.DataFrame,
    replica: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[list[str]]]:
    parent_ids = sorted(manifest["WT_pair_id"].unique())
    values = np.empty((100, 4, 5), dtype=float)
    rows: list[dict] = []
    ordered_shuffle_ids: list[list[str]] = []
    system_labels: list[str] = []
    for parent_index, parent_id in enumerate(parent_ids):
        block = manifest[manifest["WT_pair_id"] == parent_id]
        ahd_ids = sorted(block.loc[block["shuffle_type"] == "AHD shuffle", "shuffle_pair_id"])
        rgg_ids = sorted(block.loc[block["shuffle_type"] == "RG/RGG shuffle", "shuffle_pair_id"])
        if len(ahd_ids) != 2 or len(rgg_ids) != 1 or len(block) != 3:
            raise ValueError(
                f"Parent {parent_id} has AHD={len(ahd_ids)}, RG/RGG={len(rgg_ids)}, total={len(block)}"
            )
        shuffle_ids = ahd_ids + rgg_ids
        systems = [parent_id] + shuffle_ids
        ordered_shuffle_ids.append(shuffle_ids)
        for system_index, system_id in enumerate(systems):
            values[parent_index, system_index, :] = replica_array(replica, system_id)
            system_labels.append(f"{parent_id}|{system_id}")
        wt_replica_ids = ";".join(
            replica.loc[replica["system_id"] == parent_id].sort_values("replica")["replica_identifier"].astype(str)
        )
        rows.append(
            {
                "WT_parent_id": parent_id,
                "n_AHD_shuffles": len(ahd_ids),
                "n_RGG_shuffles": len(rgg_ids),
                "n_total_shuffles": len(shuffle_ids),
                "WT_replica_ids": wt_replica_ids,
                "shuffle_system_ids": ";".join(shuffle_ids),
            }
        )
    structure = pd.DataFrame(rows)
    system_means = values.mean(axis=2)
    equal_centres = system_means.mean(axis=1)
    return structure, values, equal_centres, parent_ids, ordered_shuffle_ids


def observed_statistics(pairs: pd.DataFrame) -> dict[str, float]:
    delta = pairs["Delta"].to_numpy(float)
    return {
        "median_absolute_delta": float(np.median(np.abs(delta))),
        "RMS_delta": float(np.sqrt(np.mean(delta**2))),
        "mean_signed_delta": float(np.mean(delta)),
        "median_signed_delta": float(np.median(delta)),
    }


def run_global_null(
    values: np.ndarray,
    centres: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], int, np.ndarray, int]:
    """Resample parent blocks while sharing one WT draw across three comparisons."""
    system_means = values.mean(axis=2)
    residuals = values - system_means[:, :, None]
    rng = np.random.default_rng(seed)
    collected = {key: [] for key in STAT_LABELS}
    invalid_iteration_count = 0
    invalid_system_counts = np.zeros((100, 4), dtype=np.int64)
    valid_iteration_count = 0
    for start in range(0, ITERATIONS, BATCH_SIZE):
        n = min(BATCH_SIZE, ITERATIONS - start)
        indices = rng.integers(0, 5, size=(n, 100, 4, 5))
        sampled = np.take_along_axis(residuals[None, :, :, :], indices, axis=3)
        means = centres[None, :, None] + sampled.mean(axis=3)
        invalid_system = means >= 0
        invalid_system_counts += invalid_system.sum(axis=0)
        valid_iteration = ~invalid_system.any(axis=(1, 2))
        invalid_iteration_count += int((~valid_iteration).sum())
        if not valid_iteration.any():
            continue
        valid_means = means[valid_iteration]
        wt_strength = np.log10(-valid_means[:, :, 0])
        shuffle_strength = np.log10(-valid_means[:, :, 1:])
        # The single WT draw for each parent is broadcast to all three shuffles.
        delta = (shuffle_strength - wt_strength[:, :, None]).reshape((-1, 300))
        collected["median_absolute_delta"].append(np.median(np.abs(delta), axis=1))
        collected["RMS_delta"].append(np.sqrt(np.mean(delta**2, axis=1)))
        collected["mean_signed_delta"].append(np.mean(delta, axis=1))
        collected["median_signed_delta"].append(np.median(delta, axis=1))
        valid_iteration_count += delta.shape[0]
    result = {key: np.concatenate(chunks) if chunks else np.array([], dtype=float) for key, chunks in collected.items()}
    if valid_iteration_count + invalid_iteration_count != ITERATIONS:
        raise ValueError("Global-null iteration accounting failed")
    return result, invalid_iteration_count, invalid_system_counts, valid_iteration_count


def empirical_p(null: np.ndarray, observed: float, magnitude: bool) -> float:
    if magnitude:
        exceed = int(np.sum(null >= observed))
    elif observed >= 0:
        exceed = int(np.sum(null >= observed))
    else:
        exceed = int(np.sum(null <= observed))
    return (1 + exceed) / (1 + len(null))


def summarize_null(
    null: dict[str, np.ndarray],
    observed: dict[str, float],
    invalid_iterations: int,
    analysis: str,
) -> pd.DataFrame:
    rows = []
    for key in STAT_LABELS:
        x = null[key]
        q025, q05, q50, q95, q975 = np.percentile(x, [2.5, 5, 50, 95, 97.5])
        rows.append(
            {
                "analysis": analysis,
                "statistic": key,
                "display_label": STAT_LABELS[key],
                "observed": observed[key],
                "null_mean": float(np.mean(x)),
                "null_median": float(q50),
                "null_q025": float(q025),
                "null_q05": float(q05),
                "null_q95": float(q95),
                "null_q975": float(q975),
                "empirical_p": empirical_p(x, observed[key], key in {"median_absolute_delta", "RMS_delta"}),
                "n_valid_iterations": int(len(x)),
                "n_invalid_iterations": int(invalid_iterations),
            }
        )
    return pd.DataFrame(rows)


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_global_inset(ax: plt.Axes, inset_source: pd.DataFrame, fontsize: float = 5.0) -> None:
    order = ["RMS Δ", "Median |Δ|"]
    data = inset_source.set_index("statistic").loc[order]
    y = np.arange(2)
    ax.hlines(y, data["null_q025"], data["null_q975"], color="#B5B5B5", linewidth=5.2, alpha=0.8, zorder=1)
    ax.scatter(data["null_median"], y, color="#777777", s=11, linewidths=0, zorder=2, label="Null median")
    ax.scatter(data["observed"], y, facecolor="black", edgecolor="white", linewidth=0.4, s=18, zorder=3, label="Observed")
    ax.set_yticks(y, order)
    ax.set_xlabel(r"$log_{10}$ interaction-strength units", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 0.5, width=0.4, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
    ax.legend(frameon=False, fontsize=fontsize - 1.0, loc="upper right", handletextpad=0.3, borderpad=0.1)
    xmax = max(data["observed"].max(), data["null_q975"].max()) * 1.14
    ax.set_xlim(0, xmax)


def create_figures(pairs: pd.DataFrame, inset_source: pd.DataFrame) -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.1), gridspec_kw={"width_ratios": [2, 1]}, sharex=True)
    for ax, shuffle_type in zip(axes, ["AHD shuffle", "RG/RGG shuffle"]):
        sub = pairs[pairs["shuffle_type"] == shuffle_type].sort_values("Delta").reset_index(drop=True)
        y = np.arange(len(sub))
        color = GROUP_COLORS[shuffle_type]
        ax.hlines(y, sub["CI95_low"], sub["CI95_high"], color=color, linewidth=0.55, alpha=0.46)
        ax.scatter(sub["Delta"], y, color=color, s=5.5, linewidths=0, alpha=0.88, zorder=3)
        ax.axvline(0, color="#222222", linewidth=0.7, linestyle="--")
        ax.set_title(f"{shuffle_type} (n = {len(sub)})")
        ax.set_yticks([])
        ax.set_ylabel("Comparisons sorted by observed Δ")
        ax.set_xlabel(r"Δ = $log_{10}(-B_{22,shuffle}) - log_{10}(-B_{22,WT})$")
        clean_axes(ax)
    limit = max(abs(pairs["CI95_low"].min()), abs(pairs["CI95_high"].max())) * 1.04
    for ax in axes:
        ax.set_xlim(-limit, limit)
    fig.text(0.005, 0.995, "F", ha="left", va="top", fontsize=10, fontweight="bold")
    iax = inset_axes(axes[0], width="44%", height="22%", loc="upper left", borderpad=2.8)
    iax.set_facecolor("white")
    draw_global_inset(iax, inset_source, fontsize=4.7)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.94, wspace=0.12)
    fig.savefig(S6F_PDF, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(S6F_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    draw_global_inset(ax, inset_source, fontsize=6.0)
    fig.tight_layout()
    fig.savefig(INSET_PDF, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(INSET_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def write_report(
    structure: pd.DataFrame,
    observed: dict[str, float],
    primary_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    invalid_system_counts: np.ndarray,
    centre_difference: float,
    sensitivity_consistent: bool,
) -> None:
    primary = primary_summary.set_index("statistic")
    invalid_system_total = int(invalid_system_counts.sum())
    invalid_iterations = int(primary_summary["n_invalid_iterations"].iloc[0])
    attempted = int(primary_summary["n_valid_iterations"].iloc[0] + invalid_iterations)
    valid = int(primary_summary["n_valid_iterations"].iloc[0])
    med = primary.loc["median_absolute_delta"]
    rms = primary.loc["RMS_delta"]
    sens = sensitivity_summary.set_index("statistic")
    max_null_difference = float(
        max(
            abs(primary.loc[key, column] - sens.loc[key, column])
            for key in STAT_LABELS
            for column in ["null_mean", "null_median", "null_q025", "null_q975", "empirical_p"]
        )
    )
    conclusion = (
        "Many individual WT-to-shuffle changes are comparable in magnitude to replica-level uncertainty, "
        "but the overall distribution of observed changes is broader than expected from finite-replica sampling alone."
    )
    text = f"""# Global shuffle finite-sampling null analysis

## Dependence structure

The completed panel contains exactly {structure['WT_parent_id'].nunique()} WT parents and {int(structure['n_total_shuffles'].sum())} matched WT-shuffle comparisons. Every parent contributes two AHD shuffles and one RG/RGG shuffle. Every WT and shuffle system has five replica B22 values. Within each global-null iteration, one WT mean was sampled per parent and reused for all three comparisons sharing that parent; the three shuffle systems were resampled independently.

## Null-resampling procedure

For each parent, the primary common null mean was the equal-weight mean of the four system-level means (WT, two AHD shuffles, and one RG/RGG shuffle) in B22 space. Each system's five residuals around its own observed mean were retained. During each iteration, five residuals were sampled with replacement independently for the WT and each shuffle, then added to the common parent mean. The same sampled WT mean was used for all three parent comparisons. The resulting 300 log-strength differences were summarized by median |Delta|, RMS Delta, mean signed Delta, and median signed Delta. A fixed seed of {RANDOM_SEED} was used for {attempted:,} iterations.

## Observed statistics

- Median |Delta|: {observed['median_absolute_delta']:.6f} log10 units
- RMS Delta: {observed['RMS_delta']:.6f} log10 units
- Mean signed Delta: {observed['mean_signed_delta']:.6f} log10 units
- Median signed Delta: {observed['median_signed_delta']:.6f} log10 units

## Primary global-null comparison

| Statistic | Observed | Null median | Null 95% range | Empirical P |
|---|---:|---:|---:|---:|
| Median |Delta| | {med['observed']:.6f} | {med['null_median']:.6f} | [{med['null_q025']:.6f}, {med['null_q975']:.6f}] | {med['empirical_p']:.6g} |
| RMS Delta | {rms['observed']:.6f} | {rms['null_median']:.6f} | [{rms['null_q025']:.6f}, {rms['null_q975']:.6f}] | {rms['empirical_p']:.6g} |

Magnitude-test P values are one-sided exceedance probabilities, `(1 + count(null >= observed))/(1 + n_valid)`. Signed statistics are secondary and use the observed-direction tail.

## Invalid log transforms

There were {invalid_system_total} invalid null-resampled system means across all parent/system/iteration combinations. {invalid_iterations}/{attempted} global iterations contained at least one invalid system mean and were excluded as incomplete; {valid:,} complete iterations were retained. The invalid-iteration fraction was {invalid_iterations / attempted:.6%}.

## Alternative-centre sensitivity

The sensitivity analysis used the pooled mean of all 20 replica B22 values per parent. Because all four systems have exactly five replicas, this centre is mathematically equal to the equal-system-mean centre; the maximum absolute parent-centre difference was {centre_difference:.3g}. With identical random draws, the maximum difference among reported null summaries and P values was {max_null_difference:.3g}. Sensitivity consistent: **{sensitivity_consistent}**.

## Conclusion

{conclusion}

This is a global distribution-level statement. It does not assign a definitive number of significant sequence-order effects to individual pairs.
"""
    REPORT_MD.write_text(text)


def write_notes(observed: dict[str, float], primary_summary: pd.DataFrame) -> None:
    primary = primary_summary.set_index("statistic")
    med = primary.loc["median_absolute_delta"]
    rms = primary.loc["RMS_delta"]
    simple_conclusion = (
        "the observed global distribution of shifts was broader than expected from finite-replica sampling alone"
    )
    text = f"""# Supplementary Figure S6F manuscript notes

No manuscript document was edited.

## A. Results-ready sentence

Composition-preserving shuffles largely preserved the simulated interaction-strength landscape (Pearson r = 0.975; Spearman rho = 0.958). WT-to-shuffle differences varied between pairs (median |Delta| = {observed['median_absolute_delta']:.3f} log10 units). Replica-level analysis showed that {simple_conclusion} (Supplementary Fig. S6F).

## B. Proposed S6F legend

**Supplementary Figure S6F. WT-to-shuffle effects relative to replica-level uncertainty.** Observed Delta values, defined as log10(-mean B22_shuffle) - log10(-mean B22_WT), are shown for 200 AHD-shuffle and 100 RG/RGG-shuffle comparisons, sorted within each group. Points denote observed effects and horizontal intervals denote pair-specific 95% replica-bootstrap intervals obtained by independently resampling five WT and five shuffle replicas. The inset compares the observed median |Delta| and RMS Delta with parent-aware global finite-sampling null distributions. Grey intervals show the central 95% of the null distribution, grey points show null medians, and black points show observed statistics. The global null reused a single resampled WT mean for all three shuffle comparisons sharing each WT parent. Observed median |Delta| was {observed['median_absolute_delta']:.3f} versus a null median of {med['null_median']:.3f} (95% range {med['null_q025']:.3f}-{med['null_q975']:.3f}; empirical P = {med['empirical_p']:.3g}); observed RMS Delta was {observed['RMS_delta']:.3f} versus a null median of {rms['null_median']:.3f} (95% range {rms['null_q025']:.3f}-{rms['null_q975']:.3f}; empirical P = {rms['empirical_p']:.3g}).

## C. Methods paragraph

For each matched comparison, interaction strength was calculated as log10(-mean B22) from five WT or five shuffle replicas, and pair-specific 95% intervals were obtained by independently resampling the five WT and five shuffle B22 values with replacement. To test whether the complete distribution of observed WT-to-shuffle effects could be explained by finite-replica sampling, we constructed a parent-aware global null across 100 WT parent blocks. Within each parent, the WT, two AHD-shuffle systems and one RG/RGG-shuffle system were centred to the equal-weight mean of their four system-level B22 means while preserving each system's five residuals. In each of {ITERATIONS:,} iterations, the WT residuals were resampled once per parent and the resulting WT mean was reused for all three matched comparisons; shuffle systems were resampled independently. Median |Delta| and RMS Delta were the primary global statistics. Null resamples with any non-negative system mean B22 were tracked and excluded as incomplete before log transformation.
"""
    NOTES_MD.write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, replica, pairs = load_and_validate()
    structure, values, equal_centres, parent_ids, ordered_shuffle_ids = reconstruct_parent_structure(manifest, replica)
    structure.to_csv(PARENT_STRUCTURE_CSV, index=False)
    observed = observed_statistics(pairs)

    pooled_centres = values.reshape((100, 20)).mean(axis=1)
    centre_difference = float(np.max(np.abs(equal_centres - pooled_centres)))
    primary_null, primary_invalid, invalid_system_counts, primary_valid = run_global_null(
        values, equal_centres, RANDOM_SEED
    )
    sensitivity_null, sensitivity_invalid, sensitivity_invalid_system_counts, sensitivity_valid = run_global_null(
        values, pooled_centres, RANDOM_SEED
    )
    primary_summary = summarize_null(
        primary_null, observed, primary_invalid, "equal_system_mean_centre"
    )
    sensitivity_summary = summarize_null(
        sensitivity_null, observed, sensitivity_invalid, "pooled_replica_centre"
    )
    # Keep one row per statistic, as requested, and place the alternative-centre
    # sensitivity result in explicitly prefixed companion columns.
    combined_summary = primary_summary.copy()
    sensitivity_by_stat = sensitivity_summary.set_index("statistic")
    for column in [
        "null_mean", "null_median", "null_q025", "null_q05", "null_q95", "null_q975",
        "empirical_p", "n_valid_iterations", "n_invalid_iterations",
    ]:
        combined_summary[f"sensitivity_{column}"] = combined_summary["statistic"].map(
            sensitivity_by_stat[column]
        )
    combined_summary.to_csv(SUMMARY_CSV, index=False)

    primary_indexed = primary_summary.set_index("statistic")
    inset_source = pd.DataFrame(
        [
            {
                "statistic": STAT_LABELS[key],
                "observed": primary_indexed.loc[key, "observed"],
                "null_median": primary_indexed.loc[key, "null_median"],
                "null_q025": primary_indexed.loc[key, "null_q025"],
                "null_q975": primary_indexed.loc[key, "null_q975"],
                "empirical_p": primary_indexed.loc[key, "empirical_p"],
            }
            for key in ["median_absolute_delta", "RMS_delta"]
        ]
    )
    inset_source.to_csv(INSET_SOURCE_CSV, index=False)

    max_distribution_difference = max(
        float(np.max(np.abs(primary_null[key] - sensitivity_null[key]))) for key in STAT_LABELS
    )
    sensitivity_consistent = bool(
        centre_difference < 1e-10
        and primary_invalid == sensitivity_invalid
        and primary_valid == sensitivity_valid
        and max_distribution_difference < 1e-10
    )
    if primary_valid < ITERATIONS * 0.99:
        raise ValueError(
            f"Invalid global iterations may materially affect interpretation: {primary_invalid}/{ITERATIONS}"
        )
    if not sensitivity_consistent:
        raise ValueError(
            f"Alternative centre was not consistent: centre difference={centre_difference}, "
            f"distribution difference={max_distribution_difference}"
        )

    create_figures(pairs, inset_source)
    shutil.copy2(require(OLD_SUMMARY_PDF), AUDIT_SUMMARY_PDF)
    write_report(
        structure,
        observed,
        primary_summary,
        sensitivity_summary,
        invalid_system_counts,
        centre_difference,
        sensitivity_consistent,
    )
    write_notes(observed, primary_summary)

    med = primary_indexed.loc["median_absolute_delta"]
    rms = primary_indexed.loc["RMS_delta"]
    print(f"WT PARENTS: {structure['WT_parent_id'].nunique()}")
    print(f"TOTAL MATCHED COMPARISONS: {int(structure['n_total_shuffles'].sum())}")
    print(f"AHD SHUFFLES: {int(structure['n_AHD_shuffles'].sum())}")
    print(f"RG/RGG SHUFFLES: {int(structure['n_RGG_shuffles'].sum())}")
    print(f"OBSERVED MEDIAN |DELTA|: {observed['median_absolute_delta']:.6f}")
    print(f"OBSERVED RMS DELTA: {observed['RMS_delta']:.6f}")
    print(f"GLOBAL NULL ITERATIONS ATTEMPTED: {ITERATIONS}")
    print(f"VALID GLOBAL NULL ITERATIONS: {primary_valid}")
    print(f"INVALID GLOBAL NULL ITERATIONS: {primary_invalid}")
    print(f"MEDIAN |DELTA| NULL MEDIAN: {med['null_median']:.6f}")
    print(f"MEDIAN |DELTA| NULL 95% RANGE: [{med['null_q025']:.6f}, {med['null_q975']:.6f}]")
    print(f"MEDIAN |DELTA| EMPIRICAL P: {med['empirical_p']:.8f}")
    print(f"RMS DELTA NULL MEDIAN: {rms['null_median']:.6f}")
    print(f"RMS DELTA NULL 95% RANGE: [{rms['null_q025']:.6f}, {rms['null_q975']:.6f}]")
    print(f"RMS DELTA EMPIRICAL P: {rms['empirical_p']:.8f}")
    print(f"ALTERNATIVE-CENTRE SENSITIVITY CONSISTENT: {sensitivity_consistent}")
    print("PAIR-SPECIFIC MAIN PLOT PRESERVED: YES")
    print(f"GLOBAL-NULL INSET CREATED: {INSET_PDF}")
    print(f"FINAL S6F CREATED: {S6F_PDF}")
    print(f"MANUSCRIPT NOTES CREATED: {NOTES_MD}")
    print("OUTPUT PATHS:")
    for path in [
        PARENT_STRUCTURE_CSV,
        SUMMARY_CSV,
        INSET_SOURCE_CSV,
        S6F_PDF,
        S6F_PNG,
        INSET_PDF,
        INSET_PNG,
        AUDIT_SUMMARY_PDF,
        REPORT_MD,
        NOTES_MD,
    ]:
        print(path)


if __name__ == "__main__":
    main()
