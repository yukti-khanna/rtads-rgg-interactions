#!/usr/bin/env python3
"""Audit the synthetic [XRGG]7 master table and build the requested rAHD panels.

This is a plot-only/source-data workflow. It never modifies the master CSV.
Existing generated files are copied to timestamped backup folders before they
are replaced.
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "AHD_gene", "XRGG_partner", "AHD_construct_name", "AHD_residues",
    "AHD_variant", "experiment_group", "qualitative_result", "Kd_uM",
    "Kd_error_uM", "Kd_error_type", "experiment_status", "source_document",
    "source_location", "notes",
]

MAIN_SCREEN_GENES = ["FOXO4", "LEF1", "TP53", "FOXM1"]
MAIN_SCREEN_RESIDUES = {
    "FOXO4": "461-505", "LEF1": "1-101", "TP53": "1-94", "FOXM1": "685-801"
}
MAIN_SCREEN_PARTNERS = ["[RRGG]7", "[CRGG]7", "[KRGG]7", "[LRGG]7", "[ERGG]7"]
PARTNER_ORDER = MAIN_SCREEN_PARTNERS + ["[GRGG]7", "[TRGG]7", "[IRGG]7", "[NRGG]7"]
MAIN_AFFINITY_GENES = ["FOXO4", "FOXO6", "TP53", "FOXM1", "LEF1"]
ADDITIONAL_GROUPS = [
    ("Reference/bridge AHDs", ["ATF4", "TFE3"]),
    ("Independent AD-derived constructs", ["ETV4", "SPI1", "CREB3"]),
    ("Basic / charge-mismatched controls", ["TCF7L2", "EOMES"]),
]

CONTROLLED_STATUSES = [
    "Interaction", "Weak/tendency", "No detectable interaction",
    "Line broadening", "Missing/not measured",
]
STATUS_ALIASES = {
    "interaction": "Interaction",
    "interacts": "Interaction",
    "binding": "Interaction",
    "weak": "Weak/tendency",
    "weak interaction": "Weak/tendency",
    "weak/tendency": "Weak/tendency",
    "tendency": "Weak/tendency",
    "tendency to bind": "Weak/tendency",
    "no detectable interaction": "No detectable interaction",
    "no interaction": "No detectable interaction",
    "not binding": "No detectable interaction",
    "line broadening": "Line broadening",
    "line-broadening": "Line broadening",
    "lb": "Line broadening",
    "missing": "Missing/not measured",
    "missing/not measured": "Missing/not measured",
    "not measured": "Missing/not measured",
    "pending": "Missing/not measured",
    "": "Missing/not measured",
}
MISSING_STRINGS = {"", "missing", "na", "n/a", "nan", "none", "not measured", "pending"}
SYNTHETIC_PARTNER = re.compile(r"^\[[A-Z]RGG\]7$")

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#8A8A8A"
LIGHT_GREY = "#E5E5E5"

STATUS_STYLE = {
    "Interaction": {"marker": "o", "color": BLUE, "face": BLUE, "symbol": "●"},
    "Weak/tendency": {"marker": "^", "color": ORANGE, "face": ORANGE, "symbol": "▲"},
    "No detectable interaction": {"marker": "x", "color": VERMILLION, "face": "none", "symbol": "×"},
    "Line broadening": {"marker": "s", "color": PURPLE, "face": PURPLE, "symbol": "■"},
    "Missing/not measured": {"marker": "o", "color": GREY, "face": "white", "symbol": "○"},
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 6.5,
    "axes.labelsize": 6.5,
    "axes.titlesize": 7.0,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "legend.fontsize": 5.8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})


def is_missing(value: object) -> bool:
    return pd.isna(value) or str(value).strip().lower() in MISSING_STRINGS


def normalize_status(value: object) -> str | None:
    key = "" if pd.isna(value) else re.sub(r"\s+", " ", str(value).strip().lower())
    return STATUS_ALIASES.get(key)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def backup_target(path: Path, run_stamp: str) -> None:
    if not path.exists():
        return
    backup_dir = path.parent / "backups" / run_stamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    destination = backup_dir / path.name
    counter = 2
    while destination.exists():
        destination = backup_dir / f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    shutil.copy2(path, destination)


def write_csv(frame: pd.DataFrame, path: Path, run_stamp: str) -> None:
    backup_target(path, run_stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6g")


def write_text(text: str, path: Path, run_stamp: str) -> None:
    backup_target(path, run_stamp)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, stem: Path, run_stamp: str) -> list[Path]:
    outputs = []
    for suffix, kwargs in [(".pdf", {}), (".png", {"dpi": 600})]:
        path = stem.with_suffix(suffix)
        backup_target(path, run_stamp)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def load_master(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Master CSV not found: {path}")
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Master CSV is missing required columns: {missing}")
    data.insert(0, "master_row", np.arange(2, len(data) + 2))
    data["qualitative_result_normalized"] = data["qualitative_result"].map(normalize_status)
    data["Kd_uM_numeric"] = pd.to_numeric(data["Kd_uM"], errors="coerce")
    data["Kd_error_uM_numeric"] = pd.to_numeric(data["Kd_error_uM"], errors="coerce")
    return data


def ordered_partners(data: pd.DataFrame) -> list[str]:
    observed = list(dict.fromkeys(data["XRGG_partner"].tolist()))
    return [value for value in PARTNER_ORDER if value in observed] + sorted(set(observed) - set(PARTNER_ORDER))


def exact_one(data: pd.DataFrame, gene: str, partner: str, variant: str = "WT",
              residues: str | None = None) -> pd.Series | None:
    mask = data["AHD_gene"].eq(gene) & data["XRGG_partner"].eq(partner) & data["AHD_variant"].eq(variant)
    if residues is not None:
        mask &= data["AHD_residues"].eq(residues)
    rows = data[mask]
    if rows.empty:
        return None
    if len(rows) != 1:
        raise ValueError(f"Expected one row for {gene}, {partner}, {variant}, {residues}; found {len(rows)}")
    return rows.iloc[0]


def matrix_source(data: pd.DataFrame, partners: list[str]) -> pd.DataFrame:
    records = []
    for gene in MAIN_SCREEN_GENES:
        residues = MAIN_SCREEN_RESIDUES[gene]
        for partner in partners:
            row = exact_one(data, gene, partner, "WT", residues)
            records.append({
                "AHD_gene": gene,
                "AHD_residues": residues,
                "AHD_label": f"{gene} {residues}",
                "XRGG_partner": partner,
                "qualitative_result": "MISSING" if row is None else row["qualitative_result"],
                "qualitative_result_normalized": "Missing/not measured" if row is None else row["qualitative_result_normalized"],
                "experiment_present_in_master": row is not None,
                "master_row": "" if row is None else int(row["master_row"]),
                "experiment_group": "" if row is None else row["experiment_group"],
                "experiment_status": "" if row is None else row["experiment_status"],
                "source_document": "" if row is None else row["source_document"],
                "source_location": "" if row is None else row["source_location"],
                "notes": "" if row is None else row["notes"],
            })
    return pd.DataFrame(records)


def draw_matrix(source: pd.DataFrame, partners: list[str], title: str, width: float) -> plt.Figure:
    rows = [f"{gene} {MAIN_SCREEN_RESIDUES[gene]}" for gene in MAIN_SCREEN_GENES]
    fig, ax = plt.subplots(figsize=(width, 2.35))
    ax.set_xlim(-0.5, len(partners) - 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i, label in enumerate(rows):
        for j, partner in enumerate(partners):
            status = source.loc[source.AHD_label.eq(label) & source.XRGG_partner.eq(partner),
                                "qualitative_result_normalized"].iloc[0]
            style = STATUS_STYLE[status]
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white",
                                   edgecolor=LIGHT_GREY, linewidth=0.55))
            ax.scatter(j, i, s=34, marker=style["marker"], edgecolor=style["color"],
                       facecolor=style["face"], linewidth=0.9, zorder=3)
    ax.set_xticks(range(len(partners)), partners)
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", pad=7)
    handles = []
    for status in CONTROLLED_STATUSES:
        style = STATUS_STYLE[status]
        handles.append(Line2D([0], [0], marker=style["marker"], linestyle="none",
                              markeredgecolor=style["color"], markerfacecolor=style["face"],
                              markeredgewidth=0.9, markersize=4.7, label=status))
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), columnspacing=1.0, handletextpad=0.35)
    return fig


def affinity_source(data: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    columns = [
        "master_row", "AHD_gene", "AHD_construct_name", "AHD_residues", "AHD_variant",
        "XRGG_partner", "qualitative_result", "qualitative_result_normalized",
        "Kd_uM_numeric", "Kd_error_uM_numeric", "Kd_error_type", "experiment_status",
        "source_document", "source_location", "notes",
    ]
    rows = []
    for gene in genes:
        row = exact_one(data, gene, "[RRGG]7", "WT")
        if row is None:
            rows.append({column: "" for column in columns} | {
                "AHD_gene": gene, "AHD_variant": "WT", "XRGG_partner": "[RRGG]7",
                "qualitative_result_normalized": "Missing/not measured",
            })
        else:
            rows.append(row[columns].to_dict())
    return pd.DataFrame(rows).rename(columns={
        "Kd_uM_numeric": "Kd_uM", "Kd_error_uM_numeric": "Kd_error_uM"
    })


def choose_scale(source: pd.DataFrame) -> str:
    valid = source[pd.to_numeric(source["Kd_uM"], errors="coerce").gt(0)].copy()
    if len(valid) < 2:
        return "linear"
    kd = pd.to_numeric(valid["Kd_uM"], errors="coerce")
    err = pd.to_numeric(valid["Kd_error_uM"], errors="coerce")
    positive_error_lows = (err.isna() | ((kd - err) > 0)).all()
    return "log" if positive_error_lows and kd.max() / kd.min() >= 10 else "linear"


def plot_affinity_rows(source: pd.DataFrame, labels: list[str], title: str,
                       group_at_y: list[tuple[str, float]] | None = None,
                       pair_lines: list[tuple[int, int]] | None = None,
                       colors: dict[str, str] | None = None,
                       height: float = 2.8) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.35, height))
    n = len(source)
    y = np.arange(n)[::-1]
    kd = pd.to_numeric(source["Kd_uM"], errors="coerce")
    err = pd.to_numeric(source["Kd_error_uM"], errors="coerce")
    numeric = kd.gt(0)
    scale = choose_scale(source)
    if pair_lines:
        for first, second in pair_lines:
            if numeric.iloc[first] and numeric.iloc[second]:
                ax.plot([kd.iloc[first], kd.iloc[second]], [y[first], y[second]],
                        color="#BDBDBD", linewidth=0.8, zorder=1)
    for index, row in source.reset_index(drop=True).iterrows():
        color = BLUE if colors is None else colors.get(str(row["AHD_variant"]), BLUE)
        if numeric.iloc[index]:
            xerr = None if pd.isna(err.iloc[index]) else float(err.iloc[index])
            ax.errorbar(float(kd.iloc[index]), y[index], xerr=xerr, fmt="o", color=color,
                        ecolor=color, markersize=3.8, capsize=2.2, elinewidth=0.8, zorder=3)
        else:
            status = row["qualitative_result_normalized"]
            ax.text(0.98, y[index], status, transform=ax.get_yaxis_transform(), ha="right",
                    va="center", color=STATUS_STYLE[status]["color"], fontsize=5.8)
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.75, max(n - 0.25, 0.75))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if numeric.any():
        if scale == "log":
            ax.set_xscale("log")
        ax.set_xlabel("Dissociation constant, Kd (µM)")
    else:
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.spines[["bottom"]].set_visible(False)
        ax.text(0.98, 1.01, "Qualitative result; Kd not available", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5.5, color="#555555")
    if group_at_y:
        for label, y_position in group_at_y:
            if y_position < n - 0.5:
                ax.axhline(y_position, color=LIGHT_GREY, linewidth=0.6, zorder=0)
            ax.text(-0.02, y_position, label, transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=5.7, fontweight="bold")
    ax.set_title(title, loc="left", pad=7)
    return fig


def shuffle_source(data: pd.DataFrame) -> pd.DataFrame:
    rrgg = data[data["XRGG_partner"].eq("[RRGG]7")].copy()
    genes = list(dict.fromkeys(rrgg.loc[rrgg["AHD_variant"].str.startswith("shuffle"), "AHD_gene"]))
    records = []
    for gene in genes:
        subset = rrgg[rrgg.AHD_gene.eq(gene) & (rrgg.AHD_variant.eq("WT") | rrgg.AHD_variant.str.startswith("shuffle"))]
        wt = subset[subset.AHD_variant.eq("WT")]
        if len(wt) != 1:
            raise ValueError(f"WT/shuffle comparison for {gene} requires one WT row; found {len(wt)}")
        wt_kd = wt.iloc[0].Kd_uM_numeric
        for _, row in subset.iterrows():
            fold = np.nan
            if row.AHD_variant.startswith("shuffle") and pd.notna(wt_kd) and wt_kd > 0 \
                    and pd.notna(row.Kd_uM_numeric) and row.Kd_uM_numeric > 0:
                fold = row.Kd_uM_numeric / wt_kd
            records.append({
                "pair_id": f"{gene}_{row.AHD_residues}",
                "master_row": int(row.master_row),
                "AHD_gene": gene,
                "AHD_construct_name": row.AHD_construct_name,
                "AHD_residues": row.AHD_residues,
                "AHD_variant": row.AHD_variant,
                "XRGG_partner": row.XRGG_partner,
                "qualitative_result": row.qualitative_result,
                "qualitative_result_normalized": row.qualitative_result_normalized,
                "Kd_uM": row.Kd_uM_numeric,
                "Kd_error_uM": row.Kd_error_uM_numeric,
                "Kd_error_type": row.Kd_error_type,
                "Kd_shuffle_over_WT": fold,
                "experiment_status": row.experiment_status,
                "source_document": row.source_document,
                "source_location": row.source_location,
                "notes": row.notes,
            })
    return pd.DataFrame(records)


def audit_master(data: pd.DataFrame) -> dict[str, object]:
    combo = ["AHD_gene", "XRGG_partner", "AHD_construct_name", "AHD_residues", "AHD_variant"]
    duplicate_mask = data.duplicated(combo, keep=False)
    duplicates = data.loc[duplicate_mask, ["master_row"] + combo + ["qualitative_result", "Kd_uM", "Kd_error_uM"]]
    duplicate_combinations = 0 if duplicates.empty else duplicates.groupby(combo, dropna=False).ngroups
    conflicts = []
    for key, group in data.groupby(combo, dropna=False):
        statuses = sorted(set(group["qualitative_result_normalized"].dropna()) - {"Missing/not measured"})
        if len(statuses) > 1:
            conflicts.append({**dict(zip(combo, key)), "labels": " | ".join(statuses),
                              "master_rows": ", ".join(map(str, group.master_row))})
    unknown = data[data["qualitative_result_normalized"].isna()][["master_row", "qualitative_result"]]
    non_synthetic = data[~data["XRGG_partner"].str.match(SYNTHETIC_PARTNER)][
        ["master_row", "AHD_gene", "XRGG_partner", "AHD_construct_name"]]
    numeric_kd = data[data.Kd_uM_numeric.notna()]
    numeric_both = data[data.Kd_uM_numeric.notna() & data.Kd_error_uM_numeric.notna()]
    numeric_missing_error = data[data.Kd_uM_numeric.notna() & data.Kd_error_uM_numeric.isna()]
    qualitative_only = data[data.Kd_uM_numeric.isna() & data.qualitative_result_normalized.notna()
                            & ~data.qualitative_result_normalized.eq("Missing/not measured")]
    missing_results = data[data.Kd_uM_numeric.isna()
                           & data.qualitative_result_normalized.eq("Missing/not measured")]
    quantitative_panel_rows = data[data.XRGG_partner.eq("[RRGG]7")]
    quantitative_panel_incomplete = quantitative_panel_rows[
        quantitative_panel_rows.Kd_uM_numeric.isna()
        | quantitative_panel_rows.Kd_error_uM_numeric.isna()
    ]
    return {
        "duplicates": duplicates,
        "duplicate_combinations": duplicate_combinations,
        "conflicts": pd.DataFrame(conflicts),
        "unknown": unknown,
        "non_synthetic": non_synthetic,
        "numeric_kd": numeric_kd,
        "numeric_both": numeric_both,
        "numeric_missing_error": numeric_missing_error,
        "qualitative_only": qualitative_only,
        "missing_results": missing_results,
        "quantitative_panel_incomplete": quantitative_panel_incomplete,
    }


def rows_string(frame: pd.DataFrame) -> str:
    return "None" if frame.empty else ", ".join(map(str, frame.master_row.tolist()))


def audit_text(data: pd.DataFrame, audit: dict[str, object], represented: dict[int, set[str]],
               generated: list[Path], master: Path) -> str:
    unrepresented = [int(row) for row in data.master_row if int(row) not in represented]
    lines = [
        "Synthetic [XRGG]7 rAHD panel audit",
        "=================================",
        f"Master CSV (read-only): {master}",
        f"Rows: {len(data)}",
        "rAHD genes: " + ", ".join(sorted(data.AHD_gene.unique())),
        "[XRGG]7 peptide variants: " + ", ".join(ordered_partners(data)),
        "rAHD variants: " + ", ".join(sorted(data.AHD_variant.unique())),
        f"Rows with numeric Kd: {len(audit['numeric_kd'])} ({rows_string(audit['numeric_kd'])})",
        f"Rows with numeric Kd and Kd error: {len(audit['numeric_both'])} ({rows_string(audit['numeric_both'])})",
        f"Rows with numeric Kd but missing Kd error: {len(audit['numeric_missing_error'])} ({rows_string(audit['numeric_missing_error'])})",
        f"Rows with qualitative result only: {len(audit['qualitative_only'])} ({rows_string(audit['qualitative_only'])})",
        f"Rows missing results: {len(audit['missing_results'])} ({rows_string(audit['missing_results'])})",
        f"Duplicated experimental combinations: {audit['duplicate_combinations']} "
        f"({len(audit['duplicates'])} affected rows)",
        f"Conflicting normalized qualitative labels: {len(audit['conflicts'])}",
        f"Unknown qualitative labels: {len(audit['unknown'])}",
        f"Rows outside the synthetic [XRGG]7 scope: {len(audit['non_synthetic'])}",
        "",
        "Main panel A: FOXO4 461-505, LEF1 1-101, TP53 1-94 and FOXM1 685-801; "
        "[RRGG]7, [CRGG]7, [KRGG]7, [LRGG]7 and [ERGG]7.",
        "Main panel B: WT FOXO4, FOXO6, TP53, FOXM1 and LEF1 against [RRGG]7.",
        "Supplementary WT/shuffle panel: every [RRGG]7 WT/shuffle series present in the master.",
        "Supplementary additional controls: ATF4, TFE3, ETV4, SPI1, CREB3, TCF7L2 and EOMES against [RRGG]7.",
        "Supplementary full matrix: every [XRGG]7 variant for the four original rAHD constructs.",
        "",
        "[RRGG]7 rows with missing Kd and/or Kd error preventing fully quantitative affinity panels: "
        + rows_string(audit["quantitative_panel_incomplete"]),
        "Unrepresented master rows: " + ("None" if not unrepresented else ", ".join(map(str, unrepresented))),
    ]
    if not audit["duplicates"].empty:
        lines += ["", "DUPLICATED COMBINATIONS", audit["duplicates"].to_string(index=False)]
    if not audit["conflicts"].empty:
        lines += ["", "CONFLICTING QUALITATIVE LABELS", audit["conflicts"].to_string(index=False)]
    if not audit["unknown"].empty:
        lines += ["", "UNKNOWN QUALITATIVE LABELS", audit["unknown"].to_string(index=False)]
    if not audit["non_synthetic"].empty:
        lines += ["", "OUT-OF-SCOPE ROWS", audit["non_synthetic"].to_string(index=False)]
    lines += ["", "ROW ACCOUNTING"]
    for row in data.itertuples():
        placements = sorted(represented.get(int(row.master_row), set()))
        lines.append(f"master row {row.master_row}: " + ("; ".join(placements) if placements else "UNREPRESENTED"))
    lines += ["", "FILES GENERATED"] + ([str(path) for path in generated] if generated else ["None"])
    return "\n".join(lines)


def add_accounting(represented: dict[int, set[str]], source: pd.DataFrame, panel: str) -> None:
    for value in source.get("master_row", pd.Series(dtype=object)):
        if not is_missing(value):
            represented.setdefault(int(value), set()).add(panel)


def build(project_root: Path, master_path: Path) -> None:
    project_root = project_root.resolve()
    master_path = master_path.resolve()
    source_dir = project_root / "xrgg_panel_source_data"
    plot_dir = project_root / "xrgg_panels"
    run_stamp = timestamp()
    data = load_master(master_path)
    audit = audit_master(data)
    represented: dict[int, set[str]] = {}
    generated: list[Path] = []
    audit_path = source_dir / "xrgg_panel_audit.txt"

    blockers = []
    if not audit["duplicates"].empty:
        blockers.append("duplicated experimental combinations")
    if not audit["conflicts"].empty:
        blockers.append("conflicting qualitative-result labels")
    if not audit["unknown"].empty:
        blockers.append("unknown qualitative-result labels")
    if not audit["non_synthetic"].empty:
        blockers.append("rows outside the synthetic [XRGG]7 scope")
    invalid_kd = data[data.Kd_uM_numeric.notna() & data.Kd_uM_numeric.le(0)]
    invalid_error = data[data.Kd_error_uM_numeric.notna() & data.Kd_error_uM_numeric.lt(0)]
    if not invalid_kd.empty or not invalid_error.empty:
        blockers.append("non-positive Kd or negative Kd error")
    if blockers:
        report = audit_text(data, audit, represented, generated, master_path)
        report += "\n\nPLOTTING STOPPED: " + "; ".join(blockers)
        write_text(report, audit_path, run_stamp)
        print(report)
        raise ValueError("Input audit found unresolved issues; see xrgg_panel_audit.txt")

    main_matrix = matrix_source(data, MAIN_SCREEN_PARTNERS)
    main_matrix_path = source_dir / "Fig_main_XRGG_screen_A_source.csv"
    write_csv(main_matrix, main_matrix_path, run_stamp)
    generated.append(main_matrix_path)
    add_accounting(represented, main_matrix, "Main XRGG screen A")
    fig = draw_matrix(main_matrix, MAIN_SCREEN_PARTNERS, "Synthetic [XRGG]7 interaction screen", 4.75)
    generated += save_figure(fig, plot_dir / "Fig_main_XRGG_screen_A", run_stamp)

    full_partners = ordered_partners(data)
    full_matrix = matrix_source(data, full_partners)
    full_matrix_path = source_dir / "FigS_XRGG_full_matrix_source.csv"
    write_csv(full_matrix, full_matrix_path, run_stamp)
    generated.append(full_matrix_path)
    add_accounting(represented, full_matrix, "Supplementary full XRGG matrix")
    fig = draw_matrix(full_matrix, full_partners, "Synthetic [XRGG]7 interaction screen", max(5.2, 0.62 * len(full_partners)))
    generated += save_figure(fig, plot_dir / "FigS_XRGG_full_matrix", run_stamp)

    reference = affinity_source(data, MAIN_AFFINITY_GENES)
    reference_path = source_dir / "Fig_main_RRGG_reference_Kd_B_source.csv"
    write_csv(reference, reference_path, run_stamp)
    generated.append(reference_path)
    add_accounting(represented, reference, "Main RRGG reference affinity B")
    fig = plot_affinity_rows(reference, reference.AHD_gene.tolist(),
                             "Reference rAHD interactions with [RRGG]7", height=2.65)
    generated += save_figure(fig, plot_dir / "Fig_main_RRGG_reference_Kd_B", run_stamp)

    shuffles = shuffle_source(data)
    shuffle_path = source_dir / "FigS_RRGG_WT_vs_shuffle_source.csv"
    write_csv(shuffles, shuffle_path, run_stamp)
    generated.append(shuffle_path)
    add_accounting(represented, shuffles, "Supplementary RRGG WT versus shuffle")
    shuffle_labels = [f"{row.AHD_gene} {row.AHD_variant}" for row in shuffles.itertuples()]
    pair_lines = []
    for _, group in shuffles.reset_index(drop=True).groupby("pair_id", sort=False):
        wt_indices = group.index[group.AHD_variant.eq("WT")].tolist()
        for shuffle_index in group.index[group.AHD_variant.str.startswith("shuffle")]:
            if wt_indices:
                pair_lines.append((wt_indices[0], int(shuffle_index)))
    fig = plot_affinity_rows(
        shuffles, shuffle_labels, "WT and composition-preserving rAHD shuffles with [RRGG]7",
        pair_lines=pair_lines, colors={"WT": BLUE, "shuffle_1": ORANGE, "shuffle_2": GREEN},
        height=max(3.2, 0.28 * len(shuffles) + 1.1),
    )
    generated += save_figure(fig, plot_dir / "FigS_RRGG_WT_vs_shuffle", run_stamp)

    additional_genes = [gene for _, genes in ADDITIONAL_GROUPS for gene in genes]
    additional = affinity_source(data, additional_genes)
    additional.insert(0, "control_group", [group for group, genes in ADDITIONAL_GROUPS for _ in genes])
    additional_path = source_dir / "FigS_RRGG_additional_controls_source.csv"
    write_csv(additional, additional_path, run_stamp)
    generated.append(additional_path)
    add_accounting(represented, additional, "Supplementary RRGG additional controls")
    group_labels = []
    offset = 0
    n_additional = len(additional)
    for group, genes in ADDITIONAL_GROUPS:
        group_labels.append((group, n_additional - offset - 0.5))
        offset += len(genes)
    fig = plot_affinity_rows(additional, additional.AHD_gene.tolist(),
                             "Additional rAHD and control interactions with [RRGG]7",
                             group_at_y=group_labels, height=3.45)
    generated += save_figure(fig, plot_dir / "FigS_RRGG_additional_controls", run_stamp)

    unrepresented = [int(row) for row in data.master_row if int(row) not in represented]
    if unrepresented:
        report = audit_text(data, audit, represented, generated, master_path)
        report += "\n\nWARNING: Some master rows are not represented. Review before using panels."
    else:
        report = audit_text(data, audit, represented, generated + [audit_path], master_path)
    write_text(report, audit_path, run_stamp)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--project-root", type=Path, default=default_root,
                        help="Existing fig2 directory (default: parent of this script folder)")
    parser.add_argument("--master", type=Path,
                        help="Master CSV; default is inputs/tables/xrgg_AHD_parent_master.csv")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    master = args.master or project_root / "inputs" / "tables" / "xrgg_AHD_parent_master.csv"
    build(project_root, master)


if __name__ == "__main__":
    main()
