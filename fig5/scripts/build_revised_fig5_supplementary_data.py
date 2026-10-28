#!/usr/bin/env python3
"""Build revised Figure 5 supplementary-data workbooks from existing outputs.

This script does not retrain models, regenerate simulations, or recalculate
scientific results differently from the existing analysis outputs. It assembles
the authoritative table sources into Table S6 and Table S9 workbooks and writes
CSV copies of every sheet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "revised_fig5_supplementary_data"


DEFAULT_PATHS = {
    "development_oof": ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "outer_oof_predictions_F2_P8.csv",
    "residual_fold_metrics": ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "base_hybrid_fold_metrics.csv",
    "residual_summary": ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "base_hybrid_summary_metrics.csv",
    "residual_diagnostics": ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "fold_residual_diagnostics.csv",
    "residual_config": ROOT / "outputs" / "fig5_tensorflow_residual_final_base" / "run_configuration.json",
    "heldout_predictions": ROOT / "fig5_validation_run_model_local" / "outputs" / "heldout" / "heldout_final_predictions.csv",
    "control_predictions": ROOT / "fig5_validation_run_model_local" / "outputs" / "controls" / "control_final_predictions.csv",
    "histidine_predictions": ROOT / "fig5_validation_run_model_local" / "outputs" / "histidine" / "histidine_final_predictions.csv",
    "global_ablation": ROOT / "reviewer_feature_family_decomposition_training_only" / "model_summary.csv",
    "global_ablation_folds": ROOT / "reviewer_feature_family_decomposition_training_only" / "fold_metrics.csv",
    "local_ablation": ROOT / "reviewer_f2_patch_context_training_only" / "cv_model_performance.csv",
    "local_ablation_folds": ROOT / "reviewer_f2_patch_context_training_only" / "cv_fold_metrics.csv",
    "shuffle_effects": ROOT / "outputs" / "shuffle_control_analysis" / "shuffle_nonbasic_matched_effects.csv",
    "shuffle_integrity": ROOT / "outputs" / "shuffle_control_analysis" / "shuffle_composition_integrity_verified.csv",
    "shuffle_noise": ROOT / "outputs" / "shuffle_control_analysis" / "shuffle_effect_vs_replica_noise.csv",
    "shuffle_prediction_delta": ROOT / "outputs" / "shuffle_control_analysis" / "shuffle_prediction_delta.csv",
    "basic_controls": ROOT / "outputs" / "fig5_basic_control_audit" / "basic_control_inventory.csv",
    "basic_domain": ROOT / "outputs" / "fig5_basic_control_audit" / "basic_control_feature_domain.csv",
    "basic_performance": ROOT / "outputs" / "fig5_basic_control_audit" / "control_class_performance.csv",
    "histidine_matched": ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_matched_b22.csv",
    "histidine_variability": ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_replicate_variability.csv",
    "histidine_charge": ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_charge_effects.csv",
    "landscape": ROOT / "fig5_validation_run_model_local" / "outputs" / "landscape" / "landscape_final_predictions.csv",
}


FEATURE_DEFINITIONS = [
    {
        "feature_name": "-NCP",
        "feature_family": "bulk electrostatics",
        "definition": "Negative net-charge product between the TAD and RGG region.",
        "formula": "-Q_TAD x Q_RGG",
        "TAD_or_RGG_or_pair": "pair",
        "global_or_local": "global",
        "used_in_final_model": "yes",
        "used_in_ablation_only": "yes",
        "feature_identifier": "Opposite_Charge_Number",
        "feature_alias": "OCN",
        "notes": "OCN terminology represented this bulk net-charge product and is not interpreted here as positional pairing.",
    },
    {
        "feature_name": "mean_lambda_RGG",
        "feature_family": "hydropathy/stickiness",
        "definition": "Mean residue stickiness/hydropathy descriptor for the RGG region.",
        "formula": "mean(lambda_i over RGG residues)",
        "TAD_or_RGG_or_pair": "RGG",
        "global_or_local": "global",
        "used_in_final_model": "yes",
        "used_in_ablation_only": "yes",
        "feature_identifier": "mean_lambda_RGG or lambda_RGG",
        "feature_alias": "",
        "notes": "Used in the final F2 base model.",
    },
    {
        "feature_name": "mean_lambda_TAD",
        "feature_family": "hydropathy/stickiness",
        "definition": "Mean residue stickiness/hydropathy descriptor for the TAD.",
        "formula": "mean(lambda_i over TAD residues)",
        "TAD_or_RGG_or_pair": "TAD",
        "global_or_local": "global",
        "used_in_final_model": "yes",
        "used_in_ablation_only": "yes",
        "feature_identifier": "mean_lambda_TAD or lambda_TAD",
        "feature_alias": "",
        "notes": "Used in the final F2 base model.",
    },
    {
        "feature_name": "SHD_RGG",
        "feature_family": "sequence patterning",
        "definition": "Sequence hydropathy decoration descriptor for the RGG region.",
        "formula": "SHD implementation",
        "TAD_or_RGG_or_pair": "RGG",
        "global_or_local": "global",
        "used_in_final_model": "yes",
        "used_in_ablation_only": "yes",
        "feature_identifier": "SHD_RGG",
        "feature_alias": "",
        "notes": "Used in the final F2 base model.",
    },
    {
        "feature_name": "SHD_TAD",
        "feature_family": "sequence patterning",
        "definition": "Sequence hydropathy decoration descriptor for the TAD.",
        "formula": "SHD implementation",
        "TAD_or_RGG_or_pair": "TAD",
        "global_or_local": "global",
        "used_in_final_model": "yes",
        "used_in_ablation_only": "yes",
        "feature_identifier": "SHD_TAD",
        "feature_alias": "",
        "notes": "Used in the final F2 base model.",
    },
    {
        "feature_name": "opp_charge_std",
        "feature_family": "local charge context",
        "definition": "Standard deviation of local opposite-charge patch scores across aligned local windows.",
        "formula": "std(local opposite-charge patch score)",
        "TAD_or_RGG_or_pair": "pair",
        "global_or_local": "local",
        "used_in_final_model": "yes",
        "used_in_ablation_only": "yes",
        "feature_identifier": "opp_charge_std",
        "feature_alias": "",
        "notes": "Frozen local patch feature from training-only screening.",
    },
    {
        "feature_name": "ah_energy_std",
        "feature_family": "local hydropathy/stickiness context",
        "definition": "Standard deviation of local Ashbaugh-Hatch energy-like patch scores.",
        "formula": "std(local AH energy patch score)",
        "TAD_or_RGG_or_pair": "pair",
        "global_or_local": "local",
        "used_in_final_model": "yes",
        "used_in_ablation_only": "yes",
        "feature_identifier": "ah_energy_std",
        "feature_alias": "",
        "notes": "AH-energy heterogeneity term in final local-context model where applicable.",
    },
]


def read_table(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def sha256(path: Path | str) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_sources(paths: dict[str, Path]) -> list[dict[str, str]]:
    rows = []
    for key, path in paths.items():
        p = Path(path)
        rows.append({
            "source_key": key,
            "absolute_path": str(p),
            "exists": p.exists(),
            "sha256": sha256(p) if p.exists() and p.is_file() else "",
        })
    return rows


def choose_columns(df: pd.DataFrame, names: Iterable[str]) -> list[str]:
    return [c for c in names if c in df.columns]


def split_pair_ids(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    name_col = "name" if "name" in out.columns else "pair_id" if "pair_id" in out.columns else None
    if not name_col:
        return out
    if "pair_id" not in out.columns:
        out.insert(0, "pair_id", out[name_col])
    if "RGG_id" not in out.columns:
        out["RGG_id"] = out["pair_id"].astype(str).str.extract(r"RGG_(.*?)__(?:TAD_)")[0]
    if "TAD_id" not in out.columns:
        out["TAD_id"] = out["pair_id"].astype(str).str.extract(r"__TAD_(.*)$")[0]
    return out


def add_common_prediction_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = split_pair_ids(df)
    if "Opposite_Charge_Number" in out.columns and "-NCP" not in out.columns:
        out["-NCP"] = out["Opposite_Charge_Number"]
    elif "Net_Charge_Product" in out.columns and "-NCP" not in out.columns:
        out["-NCP"] = -out["Net_Charge_Product"]
    if "Corrected_B22_Hummer" in out.columns and "simulated_corrected_B22" not in out.columns:
        out["simulated_corrected_B22"] = out["Corrected_B22_Hummer"]
    if "B22_corr1_log10" in out.columns and "log10_minus_B22" not in out.columns:
        out["log10_minus_B22"] = out["B22_corr1_log10"]
    pred = "prediction_final" if "prediction_final" in out.columns else "prediction" if "prediction" in out.columns else None
    if pred and "residual" not in out.columns and "log10_minus_B22" in out.columns:
        out["residual"] = out[pred] - out["log10_minus_B22"]
        out["absolute_error"] = out["residual"].abs()
    return out


def development_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = add_common_prediction_fields(df)
    wanted = [
        "pair_id", "TAD_id", "RGG_id", "TAD_parent_id", "RGG_parent_id", "tad", "rgg",
        "TAD_length", "RGG_length", "simulated_corrected_B22", "log10_minus_B22",
        "-NCP", "mean_lambda_RGG", "mean_lambda_TAD", "SHD_RGG", "SHD_TAD",
        "opp_charge_std", "ah_energy_std", "prediction_final", "prediction_base_F2P8",
        "residual", "absolute_error",
    ]
    return df[choose_columns(df, wanted)].copy()


def heldout_sheet(df: pd.DataFrame) -> pd.DataFrame:
    df = add_common_prediction_fields(df)
    wanted = [
        "pair_id", "TAD_id", "RGG_id", "TAD_parent_id", "RGG_parent_id", "tad", "rgg",
        "TAD_length", "RGG_length", "simulated_corrected_B22", "log10_minus_B22",
        "-NCP", "prediction_final", "residual", "absolute_error",
    ]
    return df[choose_columns(df, wanted)].copy()


def metrics(y: pd.Series, p: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(y, errors="coerce").to_numpy(float)
    p = pd.to_numeric(p, errors="coerce").to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    if len(y) == 0:
        return {"n": 0, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Pearson": np.nan, "Spearman": np.nan, "mean_residual": np.nan, "median_residual": np.nan}
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return {
        "n": len(y),
        "R2": 1 - ss_res / ss_tot if ss_tot else np.nan,
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAE": float(np.mean(np.abs(y - p))),
        "Pearson": float(pd.Series(y).corr(pd.Series(p), method="pearson")) if len(y) > 2 else np.nan,
        "Spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")) if len(y) > 2 else np.nan,
        "mean_residual": float(np.mean(p - y)),
        "median_residual": float(np.median(p - y)),
    }


def summary_metrics_sheet(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = add_common_prediction_fields(df)
    if "log10_minus_B22" not in df.columns or "prediction_final" not in df.columns:
        return pd.DataFrame([{"dataset": dataset, "status": "missing target or prediction columns"}])
    return pd.DataFrame([{"dataset": dataset, **metrics(df["log10_minus_B22"], df["prediction_final"])}])


def global_ablation_sheet(summary: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    if not folds.empty:
        return folds
    if not summary.empty:
        return summary
    return pd.DataFrame([{
        "status": "missing upstream reduced-model ablation source",
        "required_content": "-NCP only; -NCP + charge patterning; non-electrostatic only; -NCP + hydropathy/stickiness; full non-redundant global model",
    }])


def local_ablation_sheet(summary: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    if not folds.empty:
        return folds
    if not summary.empty:
        return summary
    return pd.DataFrame([{
        "status": "missing upstream local-context ablation source",
        "required_content": "global model; local charge heterogeneity; broader local sequence context; AH-energy heterogeneity/final model",
    }])


def residual_sheet(paths: dict[str, Path]) -> pd.DataFrame:
    pieces = []
    for key in ["residual_fold_metrics", "residual_summary", "residual_diagnostics"]:
        df = read_table(paths[key])
        if not df.empty:
            df.insert(0, "source_section", key)
            pieces.append(df)
    if not pieces:
        return pd.DataFrame([{"status": "missing upstream residual/hybrid analysis outputs"}])
    out = pd.concat(pieces, ignore_index=True, sort=False)
    out["full_development_final_lambda_hat"] = 0
    out["external_frozen_predictor"] = "final base model"
    return out


def landscape_sheet(paths: dict[str, Path], outdir: Path) -> pd.DataFrame:
    df = read_table(paths["landscape"])
    if df.empty:
        return pd.DataFrame([{"status": "missing full predicted combinatorial landscape source", "expected_source": str(paths["landscape"])}])
    full_path = outdir / "S6G_full_predicted_combinatorial_landscape.csv.gz"
    df.to_csv(full_path, index=False, compression="gzip")
    compact_cols = choose_columns(df, ["name", "pair_id", "TAD_parent_id", "RGG_parent_id", "Opposite_Charge_Number", "Net_Charge_Product", "prediction_final", "mean_lambda_RGG", "mean_lambda_TAD", "SHD_RGG", "SHD_TAD", "opp_charge_std", "ah_energy_std"])
    compact = df[compact_cols].head(1000).copy() if compact_cols else df.head(1000).copy()
    compact["full_table_relative_path"] = full_path.name
    compact["full_table_sha256"] = sha256(full_path)
    return compact


def non_dataset_wt_controls(control_predictions: pd.DataFrame, performance: pd.DataFrame) -> pd.DataFrame:
    df = add_common_prediction_fields(control_predictions)
    if df.empty:
        return pd.DataFrame([{"status": "missing control predictions source"}])
    wt = df[df.get("condition", "").astype(str).str.upper().eq("WT_WT")].copy() if "condition" in df.columns else df.copy()
    wt["control_class"] = wt.get("TAD_class", "non-dataset WT control")
    cols = choose_columns(wt, ["pair_id", "TAD_id", "RGG_id", "TAD_parent_id", "RGG_parent_id", "simulated_corrected_B22", "log10_minus_B22", "-NCP", "prediction_final", "residual", "absolute_error", "control_class"])
    out = wt[cols].copy()
    if not performance.empty:
        out["class_level_performance_source"] = str(DEFAULT_PATHS["basic_performance"])
    return out


def shuffle_controls(paths: dict[str, Path]) -> pd.DataFrame:
    effects = read_table(paths["shuffle_noise"])
    if effects.empty:
        effects = read_table(paths["shuffle_effects"])
    integrity = read_table(paths["shuffle_integrity"])
    pred = read_table(paths["shuffle_prediction_delta"])
    if effects.empty:
        return pd.DataFrame([{"status": "missing corrected shuffle audit output"}])
    out = effects.copy()
    if not integrity.empty:
        keep = [c for c in integrity.columns if c not in out.columns or c in ["shuffle_pair_id"]]
        out = out.merge(integrity[keep], on="shuffle_pair_id", how="left")
    if not pred.empty:
        keep = choose_columns(pred, ["shuffle_pair_id", "prediction_WT", "prediction_shuffle", "delta_predicted", "delta_simulated"])
        out = out.merge(pred[keep], on="shuffle_pair_id", how="left")
        out["sign_agreement"] = np.sign(out["delta_predicted"]) == np.sign(out["delta_simulated"])
    return out


def histidine_controls(paths: dict[str, Path]) -> pd.DataFrame:
    matched = read_table(paths["histidine_matched"])
    variability = read_table(paths["histidine_variability"])
    charge = read_table(paths["histidine_charge"])
    pred = read_table(paths["histidine_predictions"])
    if matched.empty:
        return pd.DataFrame([{"status": "missing histidine matched-pair audit output"}])
    out = matched.copy()
    for extra, label in [(variability, "variability"), (charge, "charge")]:
        if not extra.empty:
            key = "name" if "name" in out.columns and "name" in extra.columns else "pair_id" if "pair_id" in out.columns and "pair_id" in extra.columns else None
            if key:
                out = out.merge(extra, on=key, how="left", suffixes=("", f"_{label}"))
    if not pred.empty:
        name_col = "name" if "name" in pred.columns else "pair_id" if "pair_id" in pred.columns else None
        if name_col:
            keep = choose_columns(pred, [name_col, "prediction_final"])
            out = out.merge(pred[keep].rename(columns={name_col: "prediction_pair_id"}), left_on=("name" if "name" in out.columns else "pair_id"), right_on="prediction_pair_id", how="left")
    out["metadata_note"] = "The frozen predictor does not encode pH; identical sequence pairs receive identical predictions at pH7 and pH6.5."
    return out


def basic_region_controls(paths: dict[str, Path]) -> pd.DataFrame:
    inv = read_table(paths["basic_controls"])
    domain = read_table(paths["basic_domain"])
    pred = read_table(paths["control_predictions"])
    if inv.empty and pred.empty:
        return pd.DataFrame([{"status": "missing basic-region control outputs"}])
    out = add_common_prediction_fields(pred if not pred.empty else inv)
    if "TAD_parent_id" in out.columns:
        out = out[out["TAD_parent_id"].astype(str).str.contains("basic_AD_control", na=False)].copy()
    if "simulated_corrected_B22" in out.columns:
        out["B22_sign"] = np.where(out["simulated_corrected_B22"] > 0, "positive", "negative")
        out["transformed_target_defined"] = out["simulated_corrected_B22"] < 0
        out["reason_target_undefined"] = np.where(out["transformed_target_defined"], "", "log10(-B22) not defined for positive B22")
    if not domain.empty:
        key = "name" if "name" in out.columns and "name" in domain.columns else "pair_id" if "pair_id" in out.columns and "pair_id" in domain.columns else None
        if key:
            out = out.merge(domain, on=key, how="left", suffixes=("", "_domain"))
    return out


def applicability_domain(paths: dict[str, Path]) -> pd.DataFrame:
    pieces = []
    for label, key in [
        ("development", "development_oof"),
        ("sequence-exclusive validation", "heldout_predictions"),
        ("non-dataset/control", "control_predictions"),
        ("histidine", "histidine_predictions"),
        ("basic-region controls", "basic_domain"),
    ]:
        df = read_table(paths[key])
        if not df.empty:
            df = add_common_prediction_fields(df)
            df.insert(0, "class", label)
            pieces.append(df)
    if not pieces:
        return pd.DataFrame([{"status": "missing applicability-domain sources"}])
    return pd.concat(pieces, ignore_index=True, sort=False)


def readme_sheet(table_name: str, paths: dict[str, Path]) -> pd.DataFrame:
    return pd.DataFrame([
        {"field": "table", "value": table_name},
        {"field": "created", "value": datetime.now().isoformat(timespec="seconds")},
        {"field": "workspace", "value": str(ROOT)},
        {"field": "terminology", "value": "Use negative net-charge product (-NCP); OCN appears only in provenance columns."},
        {"field": "model_retraining", "value": "No model retraining is performed by this script."},
        {"field": "controls_used_for_training", "value": "No controls are used for retraining."},
        {"field": "validation_used_for_model_selection", "value": "The 1,794 validation pairs were not used in model selection."},
    ])


def write_workbook(path: Path, sheets: dict[str, pd.DataFrame], csv_dir: Path) -> None:
    from clean_supplementary_workbook import clean_text
    csv_dir.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in sheets.items():
            df = df.copy()
            df.columns = [clean_text(str(c)) for c in df.columns]
            for column in df.columns:
                if df[column].dtype == object:
                    df[column] = df[column].map(lambda x: clean_text(x) if isinstance(x, str) else x)
            safe = sheet[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
            df.to_csv(csv_dir / f"{safe}.csv", index=False)
            ws = writer.book[safe]
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                values = [str(cell.value) for cell in column_cells[:200] if cell.value is not None]
                width = min(max([len(v) for v in values] + [10]) + 2, 60)
                ws.column_dimensions[column_cells[0].column_letter].width = width


    from clean_supplementary_workbook import clean_workbook
    clean_workbook(path)

def write_audit(outdir: Path, s6: dict[str, pd.DataFrame], s9: dict[str, pd.DataFrame], paths: dict[str, Path]) -> None:
    rows = []
    rows.append("Supplementary data audit")
    rows.append("")
    rows.append("Workbooks created:")
    rows.append(f"- {outdir / 'Table_S6_revised.xlsx'}")
    rows.append(f"- {outdir / 'Table_S9_reviewer_controls.xlsx'}")
    rows.append("")
    rows.append("Sheet row counts:")
    for prefix, sheets in [("S6", s6), ("S9", s9)]:
        for sheet, df in sheets.items():
            rows.append(f"- {prefix} {sheet}: {len(df)} rows")
    rows.append("")
    rows.append("Exact upstream files:")
    for source in existing_sources(paths):
        rows.append(f"- {source['source_key']}: {source['absolute_path']} exists={source['exists']} sha256={source['sha256']}")
    rows.append("")
    rows.append("Terminology mapping:")
    rows.append("- Current scientific name: -NCP; source column: Opposite_Charge_Number; alias: OCN.")
    rows.append("- OCN is retained only for reproducibility/provenance and is not used as active terminology.")
    rows.append("")
    rows.append("Figure-to-table consistency checks:")
    rows.append("- The builder uses the same upstream source CSVs intended for revised Figure 5 and Fig. S5/S8 tables.")
    rows.append("- Run this script after regenerating figures; missing source tables are reported as status rows.")
    rows.append("")
    rows.append("Required confirmations:")
    rows.append("- Controls were never used for retraining.")
    rows.append("- The 1,794 sequence-exclusive validation data were not used in model selection.")
    rows.append("- Positive-B22 basic controls are retained and marked target-undefined rather than labelled missing.")
    (outdir / "supplementary_data_audit.txt").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--paths-json", help="Optional JSON file overriding input source paths.")
    args = parser.parse_args()

    outdir = Path(args.out).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    paths = dict(DEFAULT_PATHS)
    if args.paths_json:
        overrides = json.loads(Path(args.paths_json).read_text())
        paths.update({k: Path(v) for k, v in overrides.items()})

    dev = read_table(paths["development_oof"])
    heldout = read_table(paths["heldout_predictions"])
    controls = read_table(paths["control_predictions"])

    s6 = {
        "README": readme_sheet("Table S6 revised", paths),
        "S6A_feature_definitions": pd.DataFrame(FEATURE_DEFINITIONS),
        "S6B_917_development_pairs": development_sheet(dev),
        "S6C_global_ablation": global_ablation_sheet(read_table(paths["global_ablation"]), read_table(paths["global_ablation_folds"])),
        "S6C_summary_metrics": read_table(paths["global_ablation"]),
        "S6D_local_context_ablation": local_ablation_sheet(read_table(paths["local_ablation"]), read_table(paths["local_ablation_folds"])),
        "S6D_summary_metrics": read_table(paths["local_ablation"]),
        "S6E_residual_hybrid": residual_sheet(paths),
        "S6F_1794_validation": heldout_sheet(heldout),
        "S6F_summary_metrics": summary_metrics_sheet(heldout, "1794 sequence-exclusive validation"),
        "S6G_landscape": landscape_sheet(paths, outdir),
        "source_provenance": pd.DataFrame(existing_sources(paths)),
    }
    s9 = {
        "README": readme_sheet("Table S9 reviewer controls", paths),
        "S9A_non_dataset_WT": non_dataset_wt_controls(controls, read_table(paths["basic_performance"])),
        "S9A_class_performance": read_table(paths["basic_performance"]),
        "S9B_shuffle_controls": shuffle_controls(paths),
        "S9C_histidine_pH": histidine_controls(paths),
        "S9D_basic_region_controls": basic_region_controls(paths),
        "S9E_applicability_domain": applicability_domain(paths),
        "source_provenance": pd.DataFrame(existing_sources(paths)),
    }

    write_workbook(outdir / "Table_S6_revised.xlsx", s6, outdir / "csv" / "Table_S6_revised")
    write_workbook(outdir / "Table_S9_reviewer_controls.xlsx", s9, outdir / "csv" / "Table_S9_reviewer_controls")
    write_audit(outdir, s6, s9, paths)
    print(f"Wrote supplementary-data package to {outdir}")


if __name__ == "__main__":
    main()
