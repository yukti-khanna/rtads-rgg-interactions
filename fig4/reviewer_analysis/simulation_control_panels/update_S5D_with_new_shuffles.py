#!/usr/bin/env python3
"""Extend the validated S5D WT-vs-shuffle panel with completed controls only.

This is a targeted updater. It reads the validated source CSV as its baseline,
uses dataset_new_pairs_to_run_140.txt as the only new-control list, reuses the
existing five-replica RDF-to-B22 implementation, and does not touch any other
supplementary panel.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_simulation_control_panels import (
    CONTROL,
    DPI,
    OUT,
    b22_from_rdf,
    corr_stats,
    log10_neg_b22,
    pair_to_rdf,
    style,
)


NEW_NAMES = CONTROL / "dataset_new_pairs_to_run_140.txt"
MANIFEST = CONTROL / "final_dataset_panel_manifest_400.csv"
SOURCE = OUT / "FigS4_shuffle_sim_vs_sim_source.csv"
AUDIT = OUT / "simulation_control_panel_audit.txt"
FIGURE_STEM = "FigS4_shuffle_sim_vs_sim"
REQUIRED_SOURCE_COLUMNS = {
    "shuffle_pair_id",
    "parent_WT_pair_id",
    "shuffle_type",
    "condition",
    "run_status",
    "WT_B22_Hummer",
    "shuffle_B22_Hummer",
    "WT_simulated_value_log10_negB22",
    "shuffle_simulated_value_log10_negB22",
    "delta_simulation",
    "WT_RDF",
    "shuffle_RDF",
}
COMPONENT_RE = re.compile(r"^(RGG_.+?)__(TAD_.+)$")
SHUF_SUFFIX_RE = re.compile(r"_SHUF\d+$")


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_names(path: Path) -> list[str]:
    names = [line.strip() for line in require(path).read_text().splitlines() if line.strip()]
    if len(names) != len(set(names)):
        duplicated = pd.Series(names)[pd.Series(names).duplicated(keep=False)].unique().tolist()
        raise ValueError(f"Duplicate entries in {path}: {duplicated}")
    return names


def classify_and_parent(pair_id: str) -> tuple[str, str]:
    match = COMPONENT_RE.fullmatch(pair_id)
    if not match:
        raise ValueError(f"Unrecognized control pair identifier: {pair_id}")
    rgg, tad = match.groups()
    rgg_is_shuffle = SHUF_SUFFIX_RE.search(rgg) is not None
    tad_is_shuffle = SHUF_SUFFIX_RE.search(tad) is not None
    if rgg_is_shuffle == tad_is_shuffle:
        raise ValueError(
            f"Expected exactly one shuffled component in {pair_id}; "
            f"RGG shuffle={rgg_is_shuffle}, TAD shuffle={tad_is_shuffle}"
        )
    if rgg_is_shuffle:
        parent = f"{SHUF_SUFFIX_RE.sub('_WT', rgg)}__{tad}"
        return "RG/RGG shuffle", parent
    parent = f"{rgg}__{SHUF_SUFFIX_RE.sub('_WT', tad)}"
    return "AHD shuffle", parent


def validate_manifest_mapping(
    pair_id: str,
    parent_id: str,
    shuffle_type: str,
    manifest_by_pair: pd.DataFrame,
) -> tuple[str, str]:
    if pair_id not in manifest_by_pair.index:
        raise ValueError(f"New shuffle is absent from final manifest: {pair_id}")
    if parent_id not in manifest_by_pair.index:
        raise ValueError(f"Reconstructed WT parent is absent from final manifest: {parent_id}")
    row = manifest_by_pair.loc[pair_id]
    parent = manifest_by_pair.loc[parent_id]
    if isinstance(row, pd.DataFrame) or isinstance(parent, pd.DataFrame):
        raise ValueError(f"Non-unique pair_name in manifest for {pair_id} or {parent_id}")
    if parent["condition"] != "WT_WT":
        raise ValueError(f"Reconstructed parent is not WT_WT in manifest: {parent_id}")
    expected_condition = "RGG_SHUFFLE" if shuffle_type == "RG/RGG shuffle" else None
    if expected_condition is not None and row["condition"] != expected_condition:
        raise ValueError(f"Manifest condition mismatch for {pair_id}: {row['condition']}")
    if shuffle_type == "AHD shuffle" and not str(row["condition"]).startswith("TAD_SHUFFLE"):
        raise ValueError(f"Manifest condition mismatch for {pair_id}: {row['condition']}")
    if row["TAD_parent"] != parent["TAD_parent"] or row["RGG_parent"] != parent["RGG_parent"]:
        raise ValueError(f"Manifest parent fields disagree with identifier mapping for {pair_id}")
    return str(row["condition"]), str(row.get("run_status", ""))


def summary_stats(df: pd.DataFrame) -> dict:
    corr = corr_stats(
        df["WT_simulated_value_log10_negB22"],
        df["shuffle_simulated_value_log10_negB22"],
        df["delta_simulation"],
    )
    corr.update(
        {
            "n_WT_parents": int(df["parent_WT_pair_id"].nunique()),
            "n_AHD_TAD": int((df["shuffle_type"] == "AHD shuffle").sum()),
            "n_RGG": int((df["shuffle_type"] == "RG/RGG shuffle").sum()),
        }
    )
    return corr


def validate_source(df: pd.DataFrame, label: str) -> None:
    missing = REQUIRED_SOURCE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
    if df["shuffle_pair_id"].duplicated().any():
        dup = df.loc[df["shuffle_pair_id"].duplicated(keep=False), "shuffle_pair_id"].tolist()
        raise ValueError(f"{label} contains duplicate shuffle IDs: {dup}")
    expected_delta = (
        df["shuffle_simulated_value_log10_negB22"] - df["WT_simulated_value_log10_negB22"]
    )
    if not np.allclose(df["delta_simulation"], expected_delta, rtol=0, atol=1e-12):
        raise ValueError(f"{label} has inconsistent delta_simulation values")
    if not set(df["shuffle_type"]).issubset({"AHD shuffle", "RG/RGG shuffle"}):
        raise ValueError(f"{label} contains an unrecognized shuffle_type")


def plot_scatter(df: pd.DataFrame, stats: dict, out_dir: Path) -> None:
    style()
    fig, ax = plt.subplots(figsize=(3.25, 3.0), constrained_layout=True)
    colors = {"AHD shuffle": "#4477AA", "RG/RGG shuffle": "#AA3377"}
    for key, group in df.groupby("shuffle_type"):
        ax.scatter(
            group["WT_simulated_value_log10_negB22"],
            group["shuffle_simulated_value_log10_negB22"],
            s=10,
            alpha=0.65,
            color=colors[key],
            linewidths=0,
            label=key,
        )
    ax.legend(frameon=False, loc="lower right")
    x = df["WT_simulated_value_log10_negB22"]
    y = df["shuffle_simulated_value_log10_negB22"]
    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    pad = 0.04 * (hi - lo if hi > lo else 1.0)
    limits = (lo - pad, hi + pad)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.plot(limits, limits, color="black", lw=0.7, ls=":")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("WT vs shuffled simulations")
    ax.set_xlabel("WT log10(-B22)")
    ax.set_ylabel("Shuffled log10(-B22)")
    annotation = (
        f"n = {stats['n']} shuffles / {stats['n_WT_parents']} WT parents\n"
        f"r = {stats['pearson_r']:.2f}\n"
        f"rho = {stats['spearman_rho']:.2f}\n"
        f"mean delta = {stats['mean_delta']:.3f}"
    )
    ax.text(0.04, 0.96, annotation, transform=ax.transAxes, ha="left", va="top", fontsize=5.6)
    fig.savefig(out_dir / f"{FIGURE_STEM}.pdf")
    fig.savefig(out_dir / f"{FIGURE_STEM}.svg")
    fig.savefig(out_dir / f"{FIGURE_STEM}.png", dpi=DPI)
    plt.close(fig)


def stats_line(stats: dict) -> str:
    return (
        f"n={stats['n']}, WT parents={stats['n_WT_parents']}, "
        f"AHD/TAD={stats['n_AHD_TAD']}, RG/RGG={stats['n_RGG']}, "
        f"Pearson r={stats['pearson_r']:.12f}, Spearman rho={stats['spearman_rho']:.12f}, "
        f"mean delta={stats['mean_delta']:.12f}, median delta={stats['median_delta']:.12f}, "
        f"fraction stronger={stats['fraction_stronger_or_higher_y']:.12f}, "
        f"fraction weaker={stats['fraction_weaker_or_lower_y']:.12f}"
    )


def main() -> None:
    names = read_names(NEW_NAMES)
    wt_names = [name for name in names if "_SHUF" not in name]
    shuffle_names = [name for name in names if "_SHUF" in name]
    if len(wt_names) + len(shuffle_names) != len(names):
        raise ValueError("Internal classification count mismatch in new-names file")

    old = pd.read_csv(require(SOURCE))
    validate_source(old, "validated source CSV")
    old_stats = summary_stats(old)
    old_ids = set(old["shuffle_pair_id"])

    manifest = pd.read_csv(require(MANIFEST))
    required_manifest = {"pair_name", "condition", "TAD_parent", "RGG_parent", "run_status"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(f"Manifest missing columns: {sorted(missing_manifest)}")
    if manifest["pair_name"].duplicated().any():
        raise ValueError("final_dataset_panel_manifest_400.csv has duplicate pair_name values")
    manifest_by_pair = manifest.set_index("pair_name")

    duplicates: list[str] = []
    additions: list[dict] = []
    excluded: list[tuple[str, str]] = []
    source_files: list[tuple[str, str, str]] = []
    b22_cache: dict[str, float] = {}

    for pair_id in shuffle_names:
        shuffle_type, parent_id = classify_and_parent(pair_id)
        condition, run_status = validate_manifest_mapping(
            pair_id, parent_id, shuffle_type, manifest_by_pair
        )
        if pair_id in old_ids:
            duplicates.append(pair_id)
            continue
        shuffle_rdf = pair_to_rdf(pair_id)
        parent_rdf = pair_to_rdf(parent_id)
        missing = [str(path) for path in (shuffle_rdf, parent_rdf) if not path.exists()]
        if missing:
            excluded.append((pair_id, "missing RDF: " + "; ".join(missing)))
            continue
        try:
            for name, path in ((pair_id, shuffle_rdf), (parent_id, parent_rdf)):
                if name not in b22_cache:
                    b22_cache[name] = b22_from_rdf(path)
            shuf_b22 = b22_cache[pair_id]
            wt_b22 = b22_cache[parent_id]
        except Exception as exc:
            excluded.append((pair_id, f"five-replica B22 evaluation failed: {exc}"))
            continue
        shuf_log = log10_neg_b22(shuf_b22)
        wt_log = log10_neg_b22(wt_b22)
        if not np.isfinite(shuf_log) or not np.isfinite(wt_log):
            excluded.append(
                (pair_id, f"undefined log10(-B22): WT B22={wt_b22}, shuffle B22={shuf_b22}")
            )
            continue
        additions.append(
            {
                "shuffle_pair_id": pair_id,
                "parent_WT_pair_id": parent_id,
                "shuffle_type": shuffle_type,
                "condition": condition,
                "run_status": run_status,
                "WT_B22_Hummer": wt_b22,
                "shuffle_B22_Hummer": shuf_b22,
                "WT_simulated_value_log10_negB22": wt_log,
                "shuffle_simulated_value_log10_negB22": shuf_log,
                "delta_simulation": shuf_log - wt_log,
                "WT_RDF": str(parent_rdf.resolve()),
                "shuffle_RDF": str(shuffle_rdf.resolve()),
            }
        )
        source_files.append((pair_id, str(parent_rdf.resolve()), str(shuffle_rdf.resolve())))

    additions_df = pd.DataFrame(additions, columns=old.columns)
    combined = pd.concat([old, additions_df], ignore_index=True)
    combined = combined.sort_values("shuffle_pair_id", kind="stable").reset_index(drop=True)
    validate_source(combined, "combined source CSV")
    after_stats = summary_stats(combined)

    if after_stats["n"] != old_stats["n"] + len(additions):
        raise ValueError("Final row count is inconsistent with old rows plus additions")
    if len(shuffle_names) != len(duplicates) + len(additions) + len(excluded):
        raise ValueError("New shuffle count does not reconcile to duplicates + additions + exclusions")
    if after_stats["n"] != after_stats["n_AHD_TAD"] + after_stats["n_RGG"]:
        raise ValueError("Final shuffle-type counts do not sum to the final comparison count")
    if after_stats["n_WT_parents"] != combined["parent_WT_pair_id"].nunique():
        raise ValueError("Final WT-parent count is internally inconsistent")

    excluded_lines = [f"- {pair}: {reason}" for pair, reason in excluded] or ["- none"]
    duplicate_lines = [f"- {pair}" for pair in duplicates] or ["- none"]
    source_lines = [
        f"- {pair}\n  WT: {wt_rdf}\n  shuffle: {shuffle_rdf}"
        for pair, wt_rdf, shuffle_rdf in source_files
    ] or ["- none"]
    prior_audit = AUDIT.read_text() if AUDIT.exists() else ""
    audit_update = "\n".join(
        [
            "",
            "S5D completed-shuffle extension",
            f"Update timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"Shuffle-only input filename: {NEW_NAMES}",
            f"Names in input file: {len(names)} ({len(wt_names)} WT references; {len(shuffle_names)} shuffle controls)",
            f"Old source table shuffle comparisons: {old_stats['n']}",
            f"Already present in old source table: {len(duplicates)}",
            f"Newly added: {len(additions)}",
            f"Excluded: {len(excluded)}",
            f"Final unique WT parent pairs: {after_stats['n_WT_parents']}",
            f"Final total shuffle comparisons: {after_stats['n']}",
            f"Final AHD/TAD-shuffle count: {after_stats['n_AHD_TAD']}",
            f"Final RG/RGG-shuffle count: {after_stats['n_RGG']}",
            "",
            "Statistics before:",
            stats_line(old_stats),
            "Statistics after:",
            stats_line(after_stats),
            "",
            "Existing duplicates not added twice:",
            *duplicate_lines,
            "",
            "Excluded/incomplete controls and exact reasons:",
            *excluded_lines,
            "",
            "Exact RDF source files used for newly added B22 values:",
            *source_lines,
            "",
            "B22 procedure confirmation:",
            "- Reused build_simulation_control_panels.b22_from_rdf without modification.",
            "- Each RDF pickle was processed with rdf_b22_functions.corrected_rdfs_b22(last=7.5, box_size=20, delta=2, rc_nm=4.0).",
            "- Exactly five finite b22_corr1 replica values were required and averaged before log10(-B22).",
            "- No CALVADOS trajectories were rerun and no B22/RDF definition was changed.",
            "- No ML predictions or Figure 5 model outputs were used.",
            "- Complete per-row WT and shuffle RDF paths are retained in FigS4_shuffle_sim_vs_sim_source.csv.",
            "",
        ]
    )
    updated_audit = prior_audit.rstrip() + "\n" + audit_update

    with tempfile.TemporaryDirectory(prefix="s5d_update_", dir=OUT) as tmp_name:
        staged = Path(tmp_name)
        combined.to_csv(staged / SOURCE.name, index=False)
        (staged / AUDIT.name).write_text(updated_audit)
        plot_scatter(combined, after_stats, staged)
        expected = [
            staged / SOURCE.name,
            staged / AUDIT.name,
            staged / f"{FIGURE_STEM}.pdf",
            staged / f"{FIGURE_STEM}.svg",
            staged / f"{FIGURE_STEM}.png",
        ]
        if not all(path.exists() and path.stat().st_size > 0 for path in expected):
            raise RuntimeError("One or more staged S5D outputs are missing or empty")

        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        backup_dir = OUT / "backups" / f"S5D_pre_update_{stamp}"
        backup_dir.mkdir(parents=True, exist_ok=False)
        for path in [
            SOURCE,
            AUDIT,
            OUT / f"{FIGURE_STEM}.pdf",
            OUT / f"{FIGURE_STEM}.svg",
            OUT / f"{FIGURE_STEM}.png",
        ]:
            if path.exists():
                shutil.copy2(path, backup_dir / path.name)
        for path in expected:
            path.replace(OUT / path.name)

    print(f"Shuffle-only input filename: {NEW_NAMES}")
    print(f"Shuffle-only input entries: {len(names)}")
    print(f"1. old shuffle comparison count: {old_stats['n']}")
    print(f"2. new shuffle names supplied: {len(shuffle_names)} (plus {len(wt_names)} WT reference names)")
    print(f"3. duplicates already present: {len(duplicates)}")
    print(f"4. newly evaluable additions: {len(additions)}")
    print(f"5. excluded/incomplete additions: {len(excluded)}")
    for pair, reason in excluded:
        print(f"   EXCLUDED {pair}: {reason}")
    print(f"6. final total shuffle comparisons: {after_stats['n']}")
    print(f"7. final unique WT parent count: {after_stats['n_WT_parents']}")
    print(f"8. final AHD/TAD-shuffle count: {after_stats['n_AHD_TAD']}")
    print(f"9. final RG/RGG-shuffle count: {after_stats['n_RGG']}")
    print(
        "10. Pearson r and Spearman rho before vs after: "
        f"r {old_stats['pearson_r']:.12f} -> {after_stats['pearson_r']:.12f}; "
        f"rho {old_stats['spearman_rho']:.12f} -> {after_stats['spearman_rho']:.12f}"
    )
    print(
        "11. mean and median delta before vs after: "
        f"mean {old_stats['mean_delta']:.12f} -> {after_stats['mean_delta']:.12f}; "
        f"median {old_stats['median_delta']:.12f} -> {after_stats['median_delta']:.12f}"
    )
    print("12. output paths:")
    for path in [
        SOURCE,
        AUDIT,
        OUT / f"{FIGURE_STEM}.pdf",
        OUT / f"{FIGURE_STEM}.svg",
        OUT / f"{FIGURE_STEM}.png",
        backup_dir,
    ]:
        print(f"   {path}")
    print(f"PNG dpi: {DPI}")
    print("No ML predictions were used.")


if __name__ == "__main__":
    main()
