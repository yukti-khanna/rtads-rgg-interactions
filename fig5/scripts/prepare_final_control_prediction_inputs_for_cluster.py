#!/usr/bin/env python3
"""Prepare Figure-5 supplementary control inputs for cluster frozen-model inference.

This script does not load the Keras residual model and does not run inference. It
only inspects/reuses existing feature tables, generates missing input rows where
needed using the validated historical feature generator, and writes audits.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

FIG5_ROOT = Path(__file__).resolve().parents[1]
CONTROLS = FIG5_ROOT / "controls"
INPUT_DIR = CONTROLS / "final_prediction_inputs"
PRED_DIR = CONTROLS / "final_predictions"
WORK_DIR = CONTROLS / "_feature_recovery_work"
for d in [CONTROLS, INPUT_DIR, PRED_DIR, WORK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_DIR = FIG5_ROOT / "outputs" / "final_hybrid_model" / "frozen_model"
MANIFEST_PATH = MODEL_DIR / "MODEL_MANIFEST.json"
S9_DIR = FIG5_ROOT / "outputs" / "revised_fig5_supplementary_data" / "csv" / "Table_S9_reviewer_controls"
S9E = S9_DIR / "S9E_applicability_domain.csv"
S9D = S9_DIR / "S9D_basic_region_controls.csv"
DEV_INPUT = FIG5_ROOT / "fig5_residual_tuning_local" / "inputs" / "fig5_917_F2_P8.csv"
DEV_REF_PRED = FIG5_ROOT / "outputs" / "final_hybrid_model" / "final_predictions" / "development_917_frozen_model_predictions.csv"
SHUFFLE_MATCHED = FIG5_ROOT / "shuffle_control_analysis_local" / "outputs" / "shuffle_control_analysis" / "shuffle_nonbasic_matched_effects.csv"
BASIC_INVENTORY = FIG5_ROOT / "outputs" / "fig5_basic_control_audit" / "basic_control_inventory.csv"
CONTROL_B22 = FIG5_ROOT / "fig5_final_model_validation_local" / "outputs" / "controls" / "control_b22_from_original_rdfs.csv"
CONTROL_MANIFEST = FIG5_ROOT / "fig5_final_model_validation_local" / "inputs" / "controls" / "manifest" / "simulation_manifest_400.csv"
HIST_TARGETS = FIG5_ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_matched_b22.csv"
PATCH_SCRIPT = FIG5_ROOT / "fig5_final_model_validation_local" / "scripts" / "historical_patch_get_features.py"
RESIDUES = FIG5_ROOT / "fig5_final_model_validation_local" / "inputs" / "histidine" / "metadata" / "residues.csv"

EXPECTED_FEATURES = [
    "Opposite_Charge_Number",
    "mean_lambda_RGG",
    "mean_lambda_TAD",
    "SHD_RGG",
    "SHD_TAD",
    "opp_charge_std",
    "scd_patch1_std",
    "scd_patch2_std",
    "rg_fraction_patch1_std",
    "aromatic_fraction_patch2_std",
    "shd_patch1_std",
    "shd_patch2_std",
    "ah_energy_std",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(FIG5_ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest_features() -> list[str]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    features = manifest["feature_order"]
    if features != EXPECTED_FEATURES:
        raise RuntimeError(f"Manifest feature order differs from expected: {features}")
    return features


def split_pair(pair_id: str) -> tuple[str, str]:
    if isinstance(pair_id, str) and "__" in pair_id:
        return tuple(pair_id.split("__", 1))
    return "", ""


def derive_means(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mean_lambda_RGG" not in df.columns and {"lambda_RGG", "RGG_length"}.issubset(df.columns):
        df["mean_lambda_RGG"] = df["lambda_RGG"] / df["RGG_length"]
    if "mean_lambda_TAD" not in df.columns and {"lambda_TAD", "TAD_length"}.issubset(df.columns):
        df["mean_lambda_TAD"] = df["lambda_TAD"] / df["TAD_length"]
    return df


def table_status(path: Path, features: list[str], pair_col: str | None = None) -> dict:
    row = {"path": rel(path), "exists": path.exists()}
    if not path.exists():
        row.update({"row_count": None, "unique_pair_count": None, "all_13_features_present": False, "simulated_target_present": False, "complete": False, "reuse_unchanged": False})
        return row
    df = pd.read_csv(path, low_memory=False)
    if pair_col is None:
        pair_col = "name" if "name" in df.columns else "pair_id" if "pair_id" in df.columns else None
    row["row_count"] = int(len(df))
    row["pair_id_column"] = pair_col or "NONE"
    row["unique_pair_count"] = int(df[pair_col].nunique()) if pair_col and pair_col in df.columns else None
    row["all_13_features_present"] = all(c in df.columns for c in features)
    row["missing_features"] = ";".join([c for c in features if c not in df.columns])
    target_cols = [c for c in ["B22_corr1_log10", "simulated_target", "B22_corr1_log10_pH7", "B22_corr1_log10_pH6p5", "Corrected_B22_Hummer", "simulated_corrected_B22"] if c in df.columns]
    row["simulated_target_present"] = bool(target_cols)
    row["target_columns"] = ";".join(target_cols)
    feature_complete = row["all_13_features_present"] and not df[features].isna().any(axis=1).any()
    unique_ok = pair_col is not None and pair_col in df.columns and df[pair_col].is_unique
    row["complete"] = bool(feature_complete and unique_ok and row["simulated_target_present"])
    row["reuse_unchanged"] = False
    row["sha256"] = sha256(path)
    return row


def require_features(df: pd.DataFrame, features: list[str], label: str) -> None:
    missing = [c for c in ["name", *features] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label}: missing required columns {missing}")
    if not df["name"].is_unique:
        raise RuntimeError(f"{label}: name column is not unique")
    if df[features].isna().any(axis=1).any():
        raise RuntimeError(f"{label}: missing values in final features")
    if not np.isfinite(df[features].to_numpy(float)).all():
        raise RuntimeError(f"{label}: non-finite final features")


def standard_input(df: pd.DataFrame, features: list[str], label: str, target_col: str = "B22_corr1_log10") -> pd.DataFrame:
    df = derive_means(df)
    require_features(df, features, label)
    out = pd.DataFrame({"name": df["name"].astype(str)})
    split = out["name"].map(split_pair)
    out["RGG_id"] = [x[0] for x in split]
    out["TAD_id"] = [x[1] for x in split]
    metadata_cols = [
        "pair_id", "parent_WT_pair_id", "shuffle_pair_id", "condition", "control_subtype",
        "shuffle_type", "TAD_variant_type", "RGG_variant_type", "TAD_parent_id", "RGG_parent_id",
        "TAD_class", "feature_name", "feature_name_generated", "B22_sign", "target_evaluable",
        "transformed_target_defined", "reason_target_undefined", "Corrected_B22_Hummer", "simulated_corrected_B22",
        "log_target_defined_pH7", "log_target_defined_pH6p5", "total_H", "delta_OCN_proxy",
    ]
    for c in metadata_cols:
        if c in df.columns:
            out[c] = df[c].to_numpy()
    if target_col and target_col in df.columns:
        out["B22_corr1_log10"] = pd.to_numeric(df[target_col], errors="coerce")
    if "target_evaluable" not in out.columns:
        out["target_evaluable"] = np.isfinite(out["B22_corr1_log10"]) if "B22_corr1_log10" in out.columns else False
    for c in features:
        out[c] = pd.to_numeric(df[c], errors="raise")
    return out


def write_if_needed(df: pd.DataFrame, path: Path, audit: list[str], label: str) -> None:
    old_hash = sha256(path) if path.exists() else None
    df.to_csv(path, index=False)
    new_hash = sha256(path)
    action = "reused/regenerated same path"
    if old_hash is None:
        action = "generated"
    elif old_hash == new_hash:
        action = "already equivalent; left unchanged by content"
    else:
        action = "updated"
    audit.append(f"{label}: {action}; {rel(path)}; n={len(df)}; sha256={new_hash}")


def write_fasta(records: dict[str, str], path: Path) -> None:
    with path.open("w") as handle:
        for key, seq in records.items():
            handle.write(f">{key}\n{seq}\n")


def recover_basic_all80(features: list[str], audit: list[str]) -> pd.DataFrame:
    s9d = pd.read_csv(S9D, low_memory=False)
    inv = pd.read_csv(BASIC_INVENTORY)
    b22 = pd.read_csv(CONTROL_B22)
    manifest = pd.read_csv(CONTROL_MANIFEST)
    missing_names = sorted(set(inv["name"]) - set(s9d["name"]))
    audit.append(f"basic 78-row table missing IDs: {missing_names}")
    if len(s9d) != 78 or len(missing_names) != 2:
        raise RuntimeError(f"Expected 78 existing basic rows and 2 missing rows; got existing={len(s9d)}, missing={len(missing_names)}")
    miss_manifest = manifest[manifest["name"].isin(missing_names)].copy()
    if len(miss_manifest) != 2:
        raise RuntimeError("Could not locate the two missing basic rows in simulation_manifest_400.csv")
    records = {}
    feature_names = []
    for _, r in miss_manifest.iterrows():
        records[str(r["RGG_variant_id"])] = str(r["RGG_sequence"])
        records[str(r["TAD_variant_id"])] = str(r["TAD_sequence"])
        feature_names.append(f"{r['RGG_variant_id']}__{r['TAD_variant_id']}")
    fasta = WORK_DIR / "missing_basic_sequences.fasta"
    names_file = WORK_DIR / "missing_basic_feature_names.txt"
    write_fasta(records, fasta)
    names_file.write_text("\n".join(feature_names) + "\n")
    # The historical implementation imports numba only for speed. Some local
    # prep environments do not have numba installed, so provide a no-op njit
    # shim in the temporary work directory. This does not alter feature logic.
    shim = WORK_DIR / "numba.py"
    shim.write_text("def njit(*args, **kwargs):\n    if args and callable(args[0]) and len(args) == 1 and not kwargs:\n        return args[0]\n    def deco(fn):\n        return fn\n    return deco\n")
    cmd = [
        sys.executable, str(PATCH_SCRIPT),
        "--names_file", str(names_file),
        "--residue_csv", str(RESIDUES),
        "--fasta_file", str(fasta),
        "--out_folder", str(WORK_DIR),
        "--out_csv", "missing_basic_generated_features.csv",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(WORK_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, env=env)
    feats = pd.read_csv(WORK_DIR / "missing_basic_generated_features.csv")
    rev = miss_manifest[["name", "condition", "TAD_variant_type", "RGG_variant_type", "TAD_parent_id", "RGG_parent_id", "TAD_class", "TAD_variant_id", "RGG_variant_id"]].copy()
    rev["feature_name"] = [f"{r['RGG_variant_id']}__{r['TAD_variant_id']}" for _, r in miss_manifest.iterrows()]
    feats = feats.merge(rev, left_on="name", right_on="feature_name", how="inner", validate="one_to_one", suffixes=("_feature", ""))
    feats = feats.drop(columns=["name_feature"]).rename(columns={"name": "name"})
    feats = feats.merge(b22[["name", "Corrected_B22_Hummer", "B22_corr1_log10"]], on="name", how="left", validate="one_to_one")
    feats["simulated_corrected_B22"] = feats["Corrected_B22_Hummer"]
    feats["target_evaluable"] = np.isfinite(feats["B22_corr1_log10"])
    feats["transformed_target_defined"] = feats["target_evaluable"]
    feats["B22_sign"] = np.where(feats["Corrected_B22_Hummer"] < 0, "negative", "positive")
    feats["reason_target_undefined"] = np.where(feats["target_evaluable"], "", "positive_B22_log10_minus_B22_undefined")
    feats = derive_means(feats)
    require_features(feats, features, "recovered missing basic rows")
    all80 = pd.concat([s9d, feats], ignore_index=True, sort=False)
    if len(all80) != 80 or all80["name"].nunique() != 80:
        raise RuntimeError(f"basic ALL80 invalid: n={len(all80)}, unique={all80['name'].nunique()}")
    if int(np.isfinite(all80["B22_corr1_log10"]).sum()) != 78:
        raise RuntimeError("basic ALL80 does not contain exactly 78 target-evaluable transformed targets")
    all80["target_evaluable"] = np.isfinite(all80["B22_corr1_log10"])
    all80["reason_target_undefined"] = np.where(all80["target_evaluable"], all80.get("reason_target_undefined", ""), "positive_B22_log10_minus_B22_undefined")
    return all80


def main() -> None:
    features = load_manifest_features()
    audit_lines = ["CONTROL FEATURE INPUT AUDIT", f"fig5_root={rel(FIG5_ROOT)}", f"model_dir={rel(MODEL_DIR)}", f"feature_order={features}", ""]
    candidate_paths = [
        S9E, S9D,
        INPUT_DIR / "nonbasic_WT_prediction_input.csv",
        INPUT_DIR / "shuffle_WT_parent_prediction_input.csv",
        INPUT_DIR / "nonbasic_shuffle_prediction_input.csv",
        INPUT_DIR / "histidine_pH7_prediction_input.csv",
        INPUT_DIR / "histidine_pH6p5_prediction_input.csv",
        INPUT_DIR / "basic_control_prediction_input_ALL80.csv",
        DEV_INPUT, DEV_REF_PRED, SHUFFLE_MATCHED, HIST_TARGETS, BASIC_INVENTORY, CONTROL_B22, CONTROL_MANIFEST,
    ]
    status_rows = [table_status(p, features) for p in candidate_paths]

    s9e = pd.read_csv(S9E, low_memory=False)
    nondata = s9e[s9e["class"].eq("non-dataset/control")].copy()
    wt = nondata[nondata["condition"].eq("WT_WT")].copy()
    wt["is_basic_parent"] = wt["TAD_parent_id"].astype(str).str.contains("basic_AD_control", regex=False)
    nonbasic_wt = wt[~wt["is_basic_parent"]].copy()
    basic_wt = wt[wt["is_basic_parent"]].copy()
    if len(nonbasic_wt) != 80 or len(basic_wt) != 20:
        raise RuntimeError(f"WT partition failed: nonbasic={len(nonbasic_wt)}, basic={len(basic_wt)}")
    nonbasic_wt["control_subtype"] = "nonbasic_WT_parent"
    dev_ids = set(pd.read_csv(DEV_INPUT, usecols=["name"])["name"])
    wt_overlap = sorted(set(nonbasic_wt["name"]) & dev_ids)
    write_if_needed(standard_input(nonbasic_wt, features, "nonbasic WT"), INPUT_DIR / "nonbasic_WT_prediction_input.csv", audit_lines, "nonbasic WT input")
    write_if_needed(standard_input(nonbasic_wt.assign(control_subtype="shuffle_WT_parent"), features, "shuffle WT parent"), INPUT_DIR / "shuffle_WT_parent_prediction_input.csv", audit_lines, "shuffle WT parent input")

    shuffles = nondata[nondata["condition"].isin(["TAD_SHUFFLE_A", "TAD_SHUFFLE_B", "RGG_SHUFFLE"])].copy()
    shuffles["is_basic_parent"] = shuffles["TAD_parent_id"].astype(str).str.contains("basic_AD_control", regex=False)
    nonbasic_shuffles = shuffles[~shuffles["is_basic_parent"]].copy()
    basic_shuffles = shuffles[shuffles["is_basic_parent"]].copy()
    matched = pd.read_csv(SHUFFLE_MATCHED)
    if set(nonbasic_shuffles["name"]) != set(matched["shuffle_pair_id"]):
        raise RuntimeError("Non-basic shuffle feature rows do not exactly match shuffle_nonbasic_matched_effects.csv")
    nonbasic_shuffles["control_subtype"] = np.where(nonbasic_shuffles["condition"].astype(str).str.startswith("TAD"), "TAD_shuffle", "RGG_shuffle")
    nonbasic_shuffles = nonbasic_shuffles.merge(matched[["shuffle_pair_id", "WT_pair_id"]], left_on="name", right_on="shuffle_pair_id", how="left", validate="one_to_one")
    nonbasic_shuffles["parent_WT_pair_id"] = nonbasic_shuffles["WT_pair_id"]
    if len(nonbasic_shuffles) != 240 or int((nonbasic_shuffles["control_subtype"] == "TAD_shuffle").sum()) != 160 or int((nonbasic_shuffles["control_subtype"] == "RGG_shuffle").sum()) != 80:
        raise RuntimeError("Non-basic shuffle counts failed")
    write_if_needed(standard_input(nonbasic_shuffles, features, "nonbasic shuffles"), INPUT_DIR / "nonbasic_shuffle_prediction_input.csv", audit_lines, "nonbasic shuffle input")

    hist_features = s9e[s9e["class"].eq("histidine")].copy()
    hist_targets = pd.read_csv(HIST_TARGETS)
    if set(hist_features["name"]) != set(hist_targets["pair_id"]):
        raise RuntimeError("Histidine feature and target IDs do not match")
    hist = hist_features.merge(hist_targets, left_on="name", right_on="pair_id", how="inner", validate="one_to_one", suffixes=("", "_target"))
    h7 = hist.copy(); h7["condition"] = "pH7.0"; h7["control_subtype"] = "histidine_pH7"; h7["B22_corr1_log10"] = h7["B22_corr1_log10_pH7"]; h7["target_evaluable"] = h7["log_target_defined_pH7"].astype(bool)
    h65 = hist.copy(); h65["condition"] = "pH6.5"; h65["control_subtype"] = "histidine_pH6p5"; h65["B22_corr1_log10"] = h65["B22_corr1_log10_pH6p5"]; h65["target_evaluable"] = h65["log_target_defined_pH6p5"].astype(bool)
    h7_input = standard_input(h7, features, "histidine pH7")
    h65_input = standard_input(h65, features, "histidine pH6p5")
    write_if_needed(h7_input, INPUT_DIR / "histidine_pH7_prediction_input.csv", audit_lines, "histidine pH7 input")
    write_if_needed(h65_input, INPUT_DIR / "histidine_pH6p5_prediction_input.csv", audit_lines, "histidine pH6.5 input")
    feature_diff = (h7_input[features].to_numpy(float) != h65_input[features].to_numpy(float)).sum(axis=0)
    diff_summary = dict(zip(features, [int(x) for x in feature_diff]))
    (CONTROLS / "HISTIDINE_FEATURE_REPRESENTATION_AUDIT.txt").write_text(
        "HISTIDINE FEATURE REPRESENTATION AUDIT\n\n"
        "Inspected final feature-generation implementation: fig5_final_model_validation_local/scripts/historical_patch_get_features.py and prepare_validation_features.py. The validated pipeline reads a fixed residue property table and has no pH/protonation-state argument. Charge-dependent features in the final 13-feature model are Opposite_Charge_Number, opp_charge_std, scd_patch1_std, and scd_patch2_std; these use the same fixed charge representation as training. Other final features are sequence hydropathy/length/local-composition/AH descriptors and are also pH-independent in this implementation.\n\n"
        "Therefore the matched histidine pH7 and pH6.5 model inputs intentionally use the same sequence-derived feature representation. The simulation target differs by pH, but the predictor input does not. This is an external target-sensitivity/applicability evaluation, not a pH-aware model recalculation.\n\n"
        f"Feature differences between written pH7 and pH6.5 inputs by column: {json.dumps(diff_summary, sort_keys=True)}\n"
    )

    all80 = recover_basic_all80(features, audit_lines)
    write_if_needed(standard_input(all80, features, "basic ALL80"), INPUT_DIR / "basic_control_prediction_input_ALL80.csv", audit_lines, "basic ALL80 input")

    dev_in = standard_input(pd.read_csv(DEV_INPUT), features, "development 917 reference")
    write_if_needed(dev_in, INPUT_DIR / "development_917_reference_input.csv", audit_lines, "development 917 reference input")
    dev_ref = pd.read_csv(DEV_REF_PRED)
    write_if_needed(dev_ref, INPUT_DIR / "development_917_reference_predictions.csv", audit_lines, "development 917 reference predictions copy")

    status_rows.extend([table_status(INPUT_DIR / f, features) for f in [
        "nonbasic_WT_prediction_input.csv", "shuffle_WT_parent_prediction_input.csv", "nonbasic_shuffle_prediction_input.csv",
        "histidine_pH7_prediction_input.csv", "histidine_pH6p5_prediction_input.csv", "basic_control_prediction_input_ALL80.csv",
        "development_917_reference_input.csv", "development_917_reference_predictions.csv",
    ]])
    status = pd.DataFrame(status_rows)
    status.to_csv(CONTROLS / "CONTROL_FEATURE_INPUT_AUDIT.csv", index=False)
    audit_lines.append("")
    audit_lines.append("Candidate file inspection table: controls/CONTROL_FEATURE_INPUT_AUDIT.csv")
    audit_lines.append(f"nonbasic WT n=80; overlap with development 917 n={len(wt_overlap)}; overlap IDs={wt_overlap}")
    audit_lines.append("shuffle WT parent n=80")
    audit_lines.append("nonbasic shuffle n=240; TAD shuffle n=160; RGG shuffle n=80")
    audit_lines.append(f"basic-control shuffles excluded from nonbasic shuffle table n={len(basic_shuffles)}")
    audit_lines.append("histidine pH7 n=390; histidine pH6.5 n=390; feature representations identical by design")
    audit_lines.append("basic ALL80 n=80; target_evaluable n=78; positive_B22_target_undefined n=2")
    audit_lines.append("No local Keras residual model was loaded or run by this preparation script.")
    (CONTROLS / "CONTROL_FEATURE_INPUT_AUDIT.txt").write_text("\n".join(audit_lines) + "\n")

    summary = [
        "FINAL CLUSTER PREDICTION PREPARATION SUMMARY",
        "",
        "Reused existing complete feature sources:",
        f"- {rel(S9E)} for non-basic WT, shuffle WT parents, non-basic shuffles, and histidine sequence-derived features.",
        f"- {rel(S9D)} for 78 existing basic/noncanonical controls.",
        f"- {rel(HIST_TARGETS)} for histidine pH7/pH6.5 targets.",
        f"- {rel(SHUFFLE_MATCHED)} for non-basic shuffle parent mapping.",
        "",
        "Generated or updated input files:",
        "- controls/final_prediction_inputs/nonbasic_WT_prediction_input.csv (n=80)",
        "- controls/final_prediction_inputs/shuffle_WT_parent_prediction_input.csv (n=80)",
        "- controls/final_prediction_inputs/nonbasic_shuffle_prediction_input.csv (n=240; 160 TAD, 80 RGG)",
        "- controls/final_prediction_inputs/histidine_pH7_prediction_input.csv (n=390)",
        "- controls/final_prediction_inputs/histidine_pH6p5_prediction_input.csv (n=390; same feature representation as pH7, pH-specific target)",
        "- controls/final_prediction_inputs/basic_control_prediction_input_ALL80.csv (n=80; 78 target-evaluable, 2 positive-B22 target-undefined)",
        "- controls/final_prediction_inputs/development_917_reference_input.csv (n=917)",
        "- controls/final_prediction_inputs/development_917_reference_predictions.csv (n=917 copy for cluster identity check)",
        "",
        "Executable paths:",
        "- controls/run_final_control_predictions.py",
        "- controls/run_final_control_predictions.sh",
        "",
        "No local Keras inference was performed during preparation.",
        "All new executable paths are resolved relative to the fig5 project root and should work unchanged in the mirrored cluster fig5 directory.",
    ]
    (CONTROLS / "FINAL_CLUSTER_PREDICTION_PREPARATION_SUMMARY.txt").write_text("\n".join(summary) + "\n")
    print("Prepared final control prediction inputs under", rel(INPUT_DIR))


if __name__ == "__main__":
    main()
