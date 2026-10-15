#!/usr/bin/env python3
"""Build standalone Figure 4 simulation-control panels from completed outputs.

This script does not run CALVADOS, does not use Figure 5 ML/model outputs, and
does not alter B22/RDF definitions. WT/shuffle B22 values are derived from the
completed RDF pickle outputs using the existing Figure 4 RDF->B22 utility and
standard parameters; pH controls use the completed Task 4 matched source table.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reviewer_analysis" / "simulation_control_panels"
CONTROL = ROOT / "reviewer_analysis" / "control_simulation_manifest"
RDF_DIR = ROOT / "controls_rdfs"
TASK4 = ROOT / "task4_ph_sensitivity"
S4_FINAL = ROOT / "reviewer_analysis" / "final_fig4_revision"
S4_REVISED = ROOT / "reviewer_analysis" / "revised_S4_sampling"

sys.path.insert(0, str(ROOT))
from rdf_b22_functions import corrected_rdfs_b22  # noqa: E402


DPI = 600
LAST_NM = 7.5
BOX_NM = 20
DELTA = 2
RNG_NOTE = "No ML predictions or Figure 5 model outputs were used."


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.png", dpi=DPI)
    plt.close(fig)


def pair_to_rdf(pair_name: str) -> Path:
    return RDF_DIR / f"rdfs_{pair_name}_20nm.pkl"


def b22_from_rdf(path: Path) -> float:
    df = pd.read_pickle(require(path))
    _, b22s, _ = corrected_rdfs_b22(df, LAST_NM, box_size=BOX_NM, delta=DELTA, rc_nm=4.0)
    vals = np.array([b22s[k]["b22_corr1"] for k in sorted(b22s)], dtype=float)
    if vals.size != 5 or not np.all(np.isfinite(vals)):
        raise ValueError(f"Expected five finite historical B22 values in {path}, got {vals}")
    return float(vals.mean())


def log10_neg_b22(value: float) -> float:
    if not np.isfinite(value) or value >= 0:
        return np.nan
    return float(np.log10(-value))


def build_shuffle_source() -> tuple[pd.DataFrame, dict]:
    manifest_path = require(CONTROL / "final_dataset_panel_manifest_400.csv")
    manifest = pd.read_csv(manifest_path)
    required = {"pair_name", "condition", "TAD_parent", "RGG_parent", "run_status"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} missing required columns: {sorted(missing)}")

    group_cols = ["TAD_parent", "RGG_parent"]
    wt = manifest[manifest["condition"] == "WT_WT"].copy()
    nonwt = manifest[manifest["condition"] != "WT_WT"].copy()
    if wt.groupby(group_cols).size().ne(1).any():
        bad = wt.groupby(group_cols).size().loc[lambda s: s.ne(1)]
        raise ValueError(f"Ambiguous WT parent rows: {bad.to_dict()}")
    if nonwt.groupby(group_cols).size().ne(3).any():
        bad = nonwt.groupby(group_cols).size().loc[lambda s: s.ne(3)]
        raise ValueError(f"Expected three shuffles per WT parent, found: {bad.to_dict()}")

    wt_lookup = wt.set_index(group_cols)["pair_name"].to_dict()
    all_pairs = sorted(set(manifest["pair_name"]))
    b22_cache = {}
    missing_rdfs = []
    for pair in all_pairs:
        rdf = pair_to_rdf(pair)
        if not rdf.exists():
            missing_rdfs.append(str(rdf))
            continue
        b22_cache[pair] = b22_from_rdf(rdf)

    rows = []
    skipped_missing_rdf_or_parent = []
    for row in nonwt.to_dict("records"):
        parent = wt_lookup[(row["TAD_parent"], row["RGG_parent"])]
        if parent not in b22_cache or row["pair_name"] not in b22_cache:
            skipped_missing_rdf_or_parent.append(row["pair_name"])
            continue
        shuffle_type = "AHD shuffle" if str(row["condition"]).startswith("TAD_SHUFFLE") else "RG/RGG shuffle"
        wt_value = log10_neg_b22(b22_cache[parent])
        shuf_value = log10_neg_b22(b22_cache[row["pair_name"]])
        rows.append(
            {
                "shuffle_pair_id": row["pair_name"],
                "parent_WT_pair_id": parent,
                "shuffle_type": shuffle_type,
                "condition": row["condition"],
                "run_status": row.get("run_status", ""),
                "WT_B22_Hummer": b22_cache[parent],
                "shuffle_B22_Hummer": b22_cache[row["pair_name"]],
                "WT_simulated_value_log10_negB22": wt_value,
                "shuffle_simulated_value_log10_negB22": shuf_value,
                "delta_simulation": shuf_value - wt_value,
                "WT_RDF": str(pair_to_rdf(parent)),
                "shuffle_RDF": str(pair_to_rdf(row["pair_name"])),
            }
        )

    out = pd.DataFrame(rows)
    before = len(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["WT_simulated_value_log10_negB22", "shuffle_simulated_value_log10_negB22", "delta_simulation"]
    )
    stats = {
        "source_manifest": str(manifest_path),
        "n_manifest_rows": int(manifest.shape[0]),
        "n_completed_RDF_pickles_in_controls_rdfs": int(len(b22_cache)),
        "n_manifest_rows_without_completed_RDF": int(len(missing_rdfs)),
        "n_planned_WT_parent_pairs": int(wt.shape[0]),
        "n_completed_WT_parent_pairs_with_evaluable_shuffle": int(out["parent_WT_pair_id"].nunique() if not out.empty else 0),
        "n_TAD_AHD_shuffles": int((out["shuffle_type"] == "AHD shuffle").sum()),
        "n_RGG_shuffles": int((out["shuffle_type"] == "RG/RGG shuffle").sum()),
        "n_successfully_simulated_evaluable_shuffle_pairs": int(out.shape[0]),
        "n_excluded_undefined_log_metric": int(before - out.shape[0]),
        "n_skipped_missing_shuffle_or_parent_RDF": int(len(skipped_missing_rdf_or_parent)),
    }
    return out, stats


def build_ph_source() -> tuple[pd.DataFrame, dict]:
    source_path = require(TASK4 / "per_pair_B22_pH7_vs_pH65.csv")
    mapping_path = require(TASK4 / "ph7_ph65_pair_mapping.csv")
    ph = pd.read_csv(source_path)
    mapping = pd.read_csv(mapping_path)
    if ph["pair_id"].duplicated().any() or mapping["pair_id"].duplicated().any():
        raise ValueError("pH input contains duplicated pair IDs; exact one-to-one match required.")
    merged = ph.merge(
        mapping[["pair_id", "pH7_RDF", "pH65_RDF", "complete_five_both"]],
        on="pair_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(ph):
        raise ValueError(f"pH mapping mismatch: {len(merged)} mapped rows for {len(ph)} pH rows")
    merged = merged[merged["complete_five_both"] == True].copy()
    out = pd.DataFrame(
        {
            "pair_id": merged["pair_id"],
            "pH7_simulated_value_log10_negB22": merged["log10_negB22_pH7"],
            "pH65_simulated_value_log10_negB22": merged["log10_negB22_pH65"],
            "delta_simulation": merged["log10_negB22_pH65"] - merged["log10_negB22_pH7"],
            "pH7_B22_Hummer": merged["mean_B22_pH7"],
            "pH65_B22_Hummer": merged["mean_B22_pH65"],
            "pH7_RDF": merged["pH7_RDF"],
            "pH65_RDF": merged["pH65_RDF"],
        }
    ).replace([np.inf, -np.inf], np.nan)
    before = len(out)
    out = out.dropna(
        subset=["pH7_simulated_value_log10_negB22", "pH65_simulated_value_log10_negB22", "delta_simulation"]
    )
    stats = {
        "source_per_pair": str(source_path),
        "source_mapping": str(mapping_path),
        "n_matched_complete_five_both": int(out.shape[0]),
        "n_excluded_undefined_log_metric": int(before - out.shape[0]),
    }
    return out, stats


def corr_stats(x: pd.Series, y: pd.Series, delta: pd.Series) -> dict:
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    delta = pd.Series(delta, dtype=float)
    return {
        "n": int(len(x)),
        "pearson_r": float(x.corr(y, method="pearson")),
        "spearman_rho": float(x.corr(y, method="spearman")),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "fraction_stronger_or_higher_y": float((delta > 0).mean()),
        "fraction_weaker_or_lower_y": float((delta < 0).mean()),
    }


def identity_scatter(df: pd.DataFrame, xcol: str, ycol: str, color_col: str | None, title: str, xlabel: str, ylabel: str, stem: str, text: str) -> None:
    fig, ax = plt.subplots(figsize=(3.25, 3.0), constrained_layout=True)
    if color_col is None:
        ax.scatter(df[xcol], df[ycol], s=10, alpha=0.65, color="#555555", linewidths=0)
    else:
        colors = {"AHD shuffle": "#4477AA", "RG/RGG shuffle": "#AA3377"}
        for key, group in df.groupby(color_col):
            ax.scatter(group[xcol], group[ycol], s=10, alpha=0.65, color=colors.get(key, "#555555"), linewidths=0, label=key)
        ax.legend(frameon=False, loc="lower right")
    lo = float(np.nanmin([df[xcol].min(), df[ycol].min()]))
    hi = float(np.nanmax([df[xcol].max(), df[ycol].max()]))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=0.7, ls=":")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.04, 0.96, text, transform=ax.transAxes, ha="left", va="top", fontsize=5.6)
    save_figure(fig, stem)


def delta_hist(df: pd.DataFrame, delta_col: str, title: str, xlabel: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(3.25, 2.35), constrained_layout=True)
    ax.hist(df[delta_col], bins=35, color="#777777", edgecolor="white", linewidth=0.35)
    ax.axvline(0, color="black", lw=0.7, ls=":")
    ax.axvline(df[delta_col].median(), color="#0072B2", lw=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    save_figure(fig, stem)


def write_notes(shuffle_stats, shuffle_corr, ph_stats, ph_corr, rdf_decision: str) -> None:
    audit = [
        "Simulation-control panel audit",
        "",
        "Source files used:",
        f"- WT/shuffle manifest: {shuffle_stats['source_manifest']}",
        f"- WT/shuffle RDF folder: {RDF_DIR}",
        f"- pH per-pair source: {ph_stats['source_per_pair']}",
        f"- pH pair mapping: {ph_stats['source_mapping']}",
        "",
        "Exact n:",
        f"- WT/shuffle manifest rows: {shuffle_stats['n_manifest_rows']}",
        f"- completed RDF pickles in controls_rdfs matched to manifest: {shuffle_stats['n_completed_RDF_pickles_in_controls_rdfs']}",
        f"- manifest rows without completed RDF: {shuffle_stats['n_manifest_rows_without_completed_RDF']}",
        f"- planned WT parent pairs: {shuffle_stats['n_planned_WT_parent_pairs']}",
        f"- completed WT parent pairs with at least one evaluable shuffle: {shuffle_stats['n_completed_WT_parent_pairs_with_evaluable_shuffle']}",
        f"- completed/evaluable AHD shuffles: {shuffle_stats['n_TAD_AHD_shuffles']}",
        f"- completed/evaluable RGG shuffles: {shuffle_stats['n_RGG_shuffles']}",
        f"- Evaluable WT/shuffle pairs: {shuffle_stats['n_successfully_simulated_evaluable_shuffle_pairs']}",
        f"- Matched pH7/pH6.5 pairs: {ph_stats['n_matched_complete_five_both']}",
        "",
        "Pair-matching method:",
        "- WT/shuffle: each non-WT row in final_dataset_panel_manifest_400.csv was matched to the single WT_WT row with identical TAD_parent and RGG_parent; ambiguity is fatal.",
        "- pH: exact one-to-one pair_id match between per_pair_B22_pH7_vs_pH65.csv and ph7_ph65_pair_mapping.csv; only complete five-replica pH7/pH6.5 rows retained.",
        "",
        "Metric definitions:",
        "- WT/shuffle simulated value: log10(-mean Hummer-corrected B22) from five historical replicas in completed controls_rdfs RDF pickles, using existing rdf_b22_functions.corrected_rdfs_b22 with last=7.5, box_size=20, delta=2.",
        "- pH simulated value: existing log10_negB22_pH7 and log10_negB22_pH65 columns from Task 4.",
        "- delta_simulation: comparison condition minus reference condition in log10(-B22) units.",
        "",
        "Excluded or undefined pairs:",
        f"- WT/shuffle undefined log metric exclusions: {shuffle_stats['n_excluded_undefined_log_metric']}",
        f"- WT/shuffle skipped for absent shuffle or parent RDF: {shuffle_stats['n_skipped_missing_shuffle_or_parent_RDF']}",
        f"- pH undefined log metric exclusions: {ph_stats['n_excluded_undefined_log_metric']}",
        "",
        "Summary statistics:",
        f"- WT/shuffle: {shuffle_corr}",
        f"- pH7/pH6.5: {ph_corr}",
        "",
        "RDF panel decision:",
        rdf_decision,
        "",
        RNG_NOTE,
        "",
    ]
    (OUT / "simulation_control_panel_audit.txt").write_text("\n".join(audit))

    layout = [
        "Simulation-control layout note",
        "",
        "Current inspected Figure 4 supplementary structure:",
        "- revised_S4_sampling and final_fig4_revision contain validated S4A-S4H sampling/design panels plus S4I independent validation.",
        "- rdf_b22_convergence_figure contains the separate 3-panel B22 reproducibility/convergence control figure.",
        "- task4_ph_sensitivity already contains pH/histidine sensitivity outputs and should be updated/re-exported from final matched data rather than duplicated from scratch.",
        "",
        "Placement decision:",
        "- Add the WT-vs-shuffle and pH7-vs-pH6.5 panels as new standalone Figure 4 supplementary simulation-control panels.",
        "- Do not overwrite validated S4A-S4I panels or the convergence figure.",
        "- These panels logically sit after replicate/convergence robustness and alongside pH/histidine sensitivity as simulation robustness controls.",
        "- Final panel letters should be assigned manually during assembly.",
        "",
        RNG_NOTE,
        "",
    ]
    (OUT / "simulation_control_layout_note.txt").write_text("\n".join(layout))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    style()
    shuffle, shuffle_counts = build_shuffle_source()
    ph, ph_counts = build_ph_source()
    shuffle.to_csv(OUT / "FigS4_shuffle_sim_vs_sim_source.csv", index=False)
    ph.to_csv(OUT / "FigS4_histidine_pH_sim_vs_sim_source.csv", index=False)

    shuffle_corr = corr_stats(
        shuffle["WT_simulated_value_log10_negB22"],
        shuffle["shuffle_simulated_value_log10_negB22"],
        shuffle["delta_simulation"],
    )
    ph_corr = corr_stats(
        ph["pH7_simulated_value_log10_negB22"],
        ph["pH65_simulated_value_log10_negB22"],
        ph["delta_simulation"],
    )

    identity_scatter(
        shuffle,
        "WT_simulated_value_log10_negB22",
        "shuffle_simulated_value_log10_negB22",
        "shuffle_type",
        "WT vs shuffled simulations",
        "WT log10(-B22)",
        "Shuffled log10(-B22)",
        "FigS4_shuffle_sim_vs_sim",
        f"n={shuffle_corr['n']}\nr={shuffle_corr['pearson_r']:.2f}\nrho={shuffle_corr['spearman_rho']:.2f}\nmean delta={shuffle_corr['mean_delta']:.3f}",
    )
    delta_hist(
        shuffle,
        "delta_simulation",
        "Shuffle - WT simulation shifts",
        "Delta log10(-B22), shuffle - WT",
        "FigS4_shuffle_delta_simulation_diagnostic",
    )
    identity_scatter(
        ph,
        "pH7_simulated_value_log10_negB22",
        "pH65_simulated_value_log10_negB22",
        None,
        "pH 7.0 vs pH 6.5 simulations",
        "pH 7.0 log10(-B22)",
        "pH 6.5 log10(-B22)",
        "FigS4_histidine_pH_sim_vs_sim",
        f"n={ph_corr['n']}\nr={ph_corr['pearson_r']:.2f}\nrho={ph_corr['spearman_rho']:.2f}\nmean delta={ph_corr['mean_delta']:.3f}\nRMSE={ph_corr['rmse']:.3f}",
    )

    rdf_decision = (
        "No additional RDF example panels were generated by this script. The global B22 simulation-vs-simulation comparisons are the primary evidence; "
        "representative RDF examples would require an additional objective example-selection rule and could be added later if the global statistics indicate a need."
    )
    write_notes(shuffle_counts, shuffle_corr, ph_counts, ph_corr, rdf_decision)

    print(f"WT/shuffle evaluable n: {shuffle_counts['n_successfully_simulated_evaluable_shuffle_pairs']}")
    print(f"WT parent pairs with evaluable shuffles: {shuffle_counts['n_completed_WT_parent_pairs_with_evaluable_shuffle']}")
    print(f"AHD shuffles evaluable n: {shuffle_counts['n_TAD_AHD_shuffles']}")
    print(f"RG/RGG shuffles evaluable n: {shuffle_counts['n_RGG_shuffles']}")
    print(f"pH7-vs-pH6.5 matched n: {ph_counts['n_matched_complete_five_both']}")
    print("RDF-based panels included: no")
    print(f"Output directory: {OUT}")
    print("Output files:")
    for name in [
        "FigS4_shuffle_sim_vs_sim.pdf",
        "FigS4_shuffle_sim_vs_sim.svg",
        "FigS4_shuffle_sim_vs_sim.png",
        "FigS4_shuffle_delta_simulation_diagnostic.pdf",
        "FigS4_histidine_pH_sim_vs_sim.pdf",
        "FigS4_histidine_pH_sim_vs_sim.svg",
        "FigS4_histidine_pH_sim_vs_sim.png",
        "FigS4_shuffle_sim_vs_sim_source.csv",
        "FigS4_histidine_pH_sim_vs_sim_source.csv",
        "simulation_control_panel_audit.txt",
        "simulation_control_layout_note.txt",
    ]:
        print(OUT / name)
    print(RNG_NOTE)


if __name__ == "__main__":
    main()
