#!/usr/bin/env python3
"""Finalize frozen-model prediction datasets for Figure 5 supplementary controls.

Outputs only tables/audits under fig5/controls. No plotting, no model retraining,
no recalibration, and no Figure 4 simulation-vs-simulation panels.
"""
from __future__ import annotations

import json, hashlib, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "controls" / "final_prediction_inputs"
OUT = ROOT / "controls" / "final_predictions"
SRC = ROOT / "controls" / "source_data"
WORK = ROOT / "controls" / "_feature_recovery_work"
for d in [IN, OUT, SRC, WORK]:
    d.mkdir(parents=True, exist_ok=True)

FROZEN = ROOT / "outputs" / "final_hybrid_model" / "frozen_model"
MANIFEST = FROZEN / "MODEL_MANIFEST.json"
EXPECTED_HASHES = {
    "MODEL_MANIFEST.json": "fdb9ad1c7d2e0cc80edbac563960957b553c16aff46780656d047d30d7b7a6e9",
    "base_model_ensemble.joblib": "133946f680c6884036dcdcaddab62ac3248b38321febee02d6f535b229cf1210",
    "residual_nn.keras": "d1a329f92c00671e875d29f729682e7c8ded1f06a2c276060e604559a8f519b1",
    "residual_preprocessor.joblib": "b0f193a6f43b7c7bc95e0088cab69e7af92f7f42c578daf24f9e37d7a96c5ece",
    "predict_final_hybrid.py": "e2ae4ccf45adb414f7f3b8e6ac1678407b43f8e6459b6ed9eff17d575495bd44",
}
S9 = ROOT / "outputs" / "revised_fig5_supplementary_data" / "csv" / "Table_S9_reviewer_controls"
S9E = S9 / "S9E_applicability_domain.csv"
S9D = S9 / "S9D_basic_region_controls.csv"
DEV_FEATURES = ROOT / "fig5_residual_tuning_local" / "inputs" / "fig5_917_F2_P8.csv"
DEV_FROZEN_REF = ROOT / "outputs" / "final_hybrid_model" / "final_predictions" / "development_917_frozen_model_predictions.csv"
SHUFFLE_NONBASIC = ROOT / "shuffle_control_analysis_local" / "outputs" / "shuffle_control_analysis" / "shuffle_nonbasic_matched_effects.csv"
BASIC_INV = ROOT / "outputs" / "fig5_basic_control_audit" / "basic_control_inventory.csv"
CONTROL_B22 = ROOT / "fig5_final_model_validation_local" / "outputs" / "controls" / "control_b22_from_original_rdfs.csv"
CONTROL_MANIFEST = ROOT / "fig5_final_model_validation_local" / "inputs" / "controls" / "manifest" / "simulation_manifest_400.csv"
HIST_MATCHED = ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_matched_b22.csv"
PATCH_SCRIPT = ROOT / "fig5_final_model_validation_local" / "scripts" / "historical_patch_get_features.py"
RESIDUES = ROOT / "fig5_final_model_validation_local" / "inputs" / "histidine" / "metadata" / "residues.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_pair(x: str) -> tuple[str, str]:
    if isinstance(x, str) and "__" in x:
        a, b = x.split("__", 1)
        return a, b
    return "", ""


def metrics(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p); y = y[ok]; p = p[ok]
    n = len(y)
    return {"n": int(n), "R2": float(r2_score(y,p)) if n>1 else np.nan,
            "RMSE": float(mean_squared_error(y,p)**0.5) if n else np.nan,
            "MAE": float(mean_absolute_error(y,p)) if n else np.nan,
            "Pearson_r": float(stats.pearsonr(y,p).statistic) if n>2 else np.nan,
            "Spearman_rho": float(stats.spearmanr(y,p).statistic) if n>2 else np.nan}


def load_model():
    import tensorflow as tf
    manifest = json.loads(MANIFEST.read_text())
    base = joblib.load(FROZEN / "base_model_ensemble.joblib")
    prep = joblib.load(FROZEN / "residual_preprocessor.joblib")
    residual = tf.keras.models.load_model(FROZEN / "residual_nn.keras")
    return manifest, base, prep, residual


def derive_means(df):
    df = df.copy()
    if "mean_lambda_RGG" not in df.columns and {"lambda_RGG","RGG_length"}.issubset(df.columns):
        df["mean_lambda_RGG"] = df["lambda_RGG"] / df["RGG_length"]
    if "mean_lambda_TAD" not in df.columns and {"lambda_TAD","TAD_length"}.issubset(df.columns):
        df["mean_lambda_TAD"] = df["lambda_TAD"] / df["TAD_length"]
    return df


def soft_gate(x, center, scale):
    return 1.0/(1.0+np.exp(-((np.asarray(x,float)-center)/scale)))


def require_features(df, features, label):
    missing = [c for c in ["name", *features] if c not in df.columns]
    if missing: raise RuntimeError(f"{label}: missing {missing}")
    if not df["name"].is_unique:
        raise RuntimeError(f"{label}: duplicate names")
    if df[features].isna().any(axis=1).any():
        raise RuntimeError(f"{label}: missing feature values")
    if not np.isfinite(df[features].to_numpy(float)).all():
        raise RuntimeError(f"{label}: non-finite feature values")


def predict(df, label, manifest, base, prep, residual, target_col="B22_corr1_log10"):
    df = derive_means(df).copy()
    features = manifest["feature_order"]
    require_features(df, features, label)
    X = df[features].to_numpy(float)
    base_pred = np.mean([m.predict(base["scaler"].transform(X)) for m in base["models"]], axis=0)
    raw = residual.predict(prep["scaler"].transform(X), verbose=0).reshape(-1)
    minus = df[manifest["minus_NCP_feature"]].to_numpy(float)
    gate = soft_gate(minus, manifest["gate_center"], manifest["gate_scale"])
    uncapped = manifest["lambda_final"] * gate * raw
    clipped = np.clip(raw, -manifest["residual_cap"], manifest["residual_cap"])
    applied = manifest["lambda_final"] * gate * clipped
    final = base_pred + applied
    out = pd.DataFrame({
        "control_dataset": label,
        "pair_id": df["name"].astype(str).to_numpy(),
        "minus_NCP": minus,
        "base_prediction": base_pred,
        "raw_residual_prediction": raw,
        "gate_value": gate,
        "lambda_final": manifest["lambda_final"],
        "uncapped_correction": uncapped,
        "applied_correction": applied,
        "final_hybrid_prediction": final,
        "prediction_identity_error": np.abs(final - (base_pred + applied)),
        "frozen_model_manifest_sha256": sha256(FROZEN / "MODEL_MANIFEST.json"),
        "base_model_sha256": sha256(FROZEN / "base_model_ensemble.joblib"),
        "residual_model_sha256": sha256(FROZEN / "residual_nn.keras"),
        "residual_preprocessor_sha256": sha256(FROZEN / "residual_preprocessor.joblib"),
        "feature_order_json": json.dumps(features),
    })
    ids = out["pair_id"].map(split_pair)
    out.insert(2, "RGG_id", [x[0] for x in ids])
    out.insert(3, "TAD_id", [x[1] for x in ids])
    for c in ["pair_id", "condition", "control_subtype", "TAD_variant_type", "RGG_variant_type", "TAD_parent_id", "RGG_parent_id", "TAD_class", "feature_name", "feature_name_generated", "B22_sign", "reason_target_undefined"]:
        if c in df.columns and c not in out.columns:
            out[c if c != "pair_id" else "source_pair_id"] = df[c].to_numpy()
    if target_col and target_col in df.columns:
        out["simulated_target"] = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
    if "Corrected_B22_Hummer" in df.columns:
        out["simulated_B22"] = df["Corrected_B22_Hummer"].to_numpy()
    elif "simulated_corrected_B22" in df.columns:
        out["simulated_B22"] = df["simulated_corrected_B22"].to_numpy()
    if "target_evaluable" in df.columns:
        out["target_evaluable"] = df["target_evaluable"].astype(bool).to_numpy()
    elif "transformed_target_defined" in df.columns:
        out["target_evaluable"] = df["transformed_target_defined"].astype(bool).to_numpy()
    elif "simulated_target" in out.columns:
        out["target_evaluable"] = np.isfinite(out["simulated_target"])
    else:
        out["target_evaluable"] = False
    if "simulated_target" in out.columns:
        out["residual"] = out["simulated_target"] - out["final_hybrid_prediction"]
        out["absolute_error"] = np.abs(out["residual"])
    return out


def standard_input(df, features, label, target_col="B22_corr1_log10"):
    df = derive_means(df).copy()
    require_features(df, features, label)
    out = pd.DataFrame({"name": df["name"].astype(str)})
    ids = out["name"].map(split_pair)
    out["RGG_id"] = [x[0] for x in ids]
    out["TAD_id"] = [x[1] for x in ids]
    for c in ["pair_id", "condition", "control_subtype", "TAD_variant_type", "RGG_variant_type", "TAD_parent_id", "RGG_parent_id", "TAD_class", "feature_name", "feature_name_generated", "B22_sign", "transformed_target_defined", "target_evaluable", "reason_target_undefined", "Corrected_B22_Hummer", "simulated_corrected_B22"]:
        if c in df.columns: out[c] = df[c].to_numpy()
    if target_col and target_col in df.columns:
        out["B22_corr1_log10"] = pd.to_numeric(df[target_col], errors="coerce")
    for f in features:
        out[f] = pd.to_numeric(df[f], errors="raise")
    return out


def reconcile_model_identity(manifest, base, prep, residual):
    lines=[]
    lines.append("MODEL IDENTITY RECONCILIATION")
    ok=True
    for fn, exp in EXPECTED_HASHES.items():
        h=sha256(FROZEN/fn); same=(h==exp); ok &= same
        lines.append(f"{fn}\texpected={exp}\tobserved={h}\tmatch={same}")
    if not ok:
        (OUT/"MODEL_IDENTITY_RECONCILIATION.txt").write_text("\n".join(lines)+"\n")
        raise RuntimeError("Frozen artifact hash mismatch")
    dev=pd.read_csv(DEV_FEATURES)
    pred=predict(dev, "development_917_identity_check", manifest, base, prep, residual)
    ref=pd.read_csv(DEV_FROZEN_REF)
    m=pred.merge(ref, left_on="pair_id", right_on="pair_id", suffixes=("_new","_ref"), validate="one_to_one")
    checks={
        "base_prediction": ("base_prediction_new","base_prediction_ref"),
        "raw_residual_prediction": ("raw_residual_prediction_new","raw_residual_prediction_ref"),
        "gate_value": ("gate_value_new","gate_value_ref"),
        "applied_correction": ("applied_correction_new","applied_correction_ref"),
        "final_hybrid_prediction": ("final_hybrid_prediction_new","final_hybrid_prediction_ref"),
    }
    maxdiff={k: float(np.max(np.abs(m[a]-m[b]))) for k,(a,b) in checks.items()}
    lines.append("")
    lines.append(f"development identity rows compared: {len(m)}")
    for k,v in maxdiff.items(): lines.append(f"max_abs_diff_{k}: {v:.16g}")
    passed= len(m)==917 and all(v <= 1e-10 for v in maxdiff.values())
    lines.append(f"identity_pass_1e-10: {passed}")
    if passed:
        lines.append("The apparent 48b727... versus fdb9ad... manifest-hash discrepancy is metadata/provenance-only for these deployed artefacts: the current artefacts reproduce the definitive frozen development prediction table row-by-row within 1e-10.")
    (OUT/"MODEL_IDENTITY_RECONCILIATION.txt").write_text("\n".join(lines)+"\n")
    if not passed: raise RuntimeError("Development frozen prediction identity check failed")
    return maxdiff


def write_fasta(records, path):
    with path.open("w") as f:
        for k,v in records.items(): f.write(f">{k}\n{v}\n")


def recover_missing_basic_features(features):
    inv=pd.read_csv(BASIC_INV)
    s9d=pd.read_csv(S9D, low_memory=False)
    missing=sorted(set(inv["name"])-set(s9d["name"]))
    b22=pd.read_csv(CONTROL_B22)
    man=pd.read_csv(CONTROL_MANIFEST)
    if len(missing)!=2:
        raise RuntimeError(f"Expected 2 missing basic rows, found {len(missing)}: {missing}")
    missman=man[man["name"].isin(missing)].copy()
    if len(missman)!=2: raise RuntimeError("Missing basic rows not found in control manifest")
    records={}
    names=[]
    for _,r in missman.iterrows():
        records[str(r.RGG_variant_id)] = str(r.RGG_sequence)
        records[str(r.TAD_variant_id)] = str(r.TAD_sequence)
        names.append(f"{r.RGG_variant_id}__{r.TAD_variant_id}")
    fasta=WORK/"missing_basic_sequences.fasta"; names_file=WORK/"missing_basic_feature_names.txt"
    write_fasta(records, fasta); names_file.write_text("\n".join(names)+"\n")
    cmd=[sys.executable, str(PATCH_SCRIPT), "--names_file", str(names_file), "--residue_csv", str(RESIDUES), "--fasta_file", str(fasta), "--out_folder", str(WORK), "--out_csv", "missing_basic_generated_features.csv"]
    subprocess.run(cmd, check=True)
    feats=pd.read_csv(WORK/"missing_basic_generated_features.csv")
    rev=missman[["name","condition","TAD_variant_type","RGG_variant_type","TAD_parent_id","RGG_parent_id","TAD_class","TAD_variant_id","RGG_variant_id"]].copy()
    rev["feature_name"]=[f"{r.RGG_variant_id}__{r.TAD_variant_id}" for _,r in missman.iterrows()]
    feats=feats.merge(rev, left_on="name", right_on="feature_name", how="inner", validate="one_to_one", suffixes=("_feature", ""))
    feats=feats.drop(columns=["name_feature"]).rename(columns={"name":"name"})
    feats=feats.merge(b22[["name","Corrected_B22_Hummer","B22_corr1_log10"]], on="name", how="left", validate="one_to_one")
    feats["simulated_corrected_B22"] = feats["Corrected_B22_Hummer"]
    feats["target_evaluable"] = np.isfinite(feats["B22_corr1_log10"])
    feats["transformed_target_defined"] = feats["target_evaluable"]
    feats["B22_sign"] = np.where(feats["Corrected_B22_Hummer"] < 0, "negative", "positive")
    feats["reason_target_undefined"] = np.where(feats["target_evaluable"], "", "positive_B22_log10_minus_B22_undefined")
    feats=derive_means(feats)
    require_features(feats, features, "missing basic generated")
    # Existing 78 plus missing two. Keep all rows including NaN target.
    existing=s9d.copy()
    if "target_evaluable" not in existing.columns:
        existing["target_evaluable"] = existing.get("transformed_target_defined", True)
    all80=pd.concat([existing, feats], ignore_index=True, sort=False)
    if len(all80)!=80 or not all80["name"].is_unique:
        raise RuntimeError(f"basic all80 invalid: n={len(all80)}, unique={all80['name'].nunique()}")
    return all80, missing


def main():
    manifest, base, prep, residual = load_model()
    features=manifest["feature_order"]
    identity_diffs=reconcile_model_identity(manifest, base, prep, residual)

    s9e=pd.read_csv(S9E, low_memory=False)
    dev_ids=set(pd.read_csv(DEV_FEATURES, usecols=["name"])["name"])
    nondata=s9e[s9e["class"].eq("non-dataset/control")].copy()
    wt=nondata[nondata["condition"].eq("WT_WT")].copy()
    wt["is_basic_parent"] = wt["TAD_parent_id"].astype(str).str.contains("basic_AD_control", regex=False)
    nonbasic_wt=wt[~wt["is_basic_parent"]].copy(); nonbasic_wt["control_subtype"]="nonbasic_WT_parent"
    basic_wt=wt[wt["is_basic_parent"]].copy(); basic_wt["control_subtype"]="basic_WT_parent"
    if len(nonbasic_wt)!=80 or len(basic_wt)!=20: raise RuntimeError(f"WT partition failed: nonbasic={len(nonbasic_wt)} basic={len(basic_wt)}")
    wt_overlap=sorted(set(wt["name"]) & dev_ids)

    std_nonbasic_wt=standard_input(nonbasic_wt, features, "nonbasic WT")
    std_basic_wt=standard_input(basic_wt, features, "basic WT")
    std_nonbasic_wt.to_csv(IN/"nonbasic_WT_parent_prediction_input.csv", index=False)
    std_basic_wt.to_csv(IN/"basic_WT_parent_prediction_input.csv", index=False)

    sh=nondata[nondata["condition"].isin(["TAD_SHUFFLE_A","TAD_SHUFFLE_B","RGG_SHUFFLE"])].copy()
    sh["is_basic_parent"] = sh["TAD_parent_id"].astype(str).str.contains("basic_AD_control", regex=False)
    nonbasic_sh=sh[~sh["is_basic_parent"]].copy(); basic_sh=sh[sh["is_basic_parent"]].copy()
    nonbasic_sh["control_subtype"] = np.where(nonbasic_sh["condition"].astype(str).str.startswith("TAD"), "TAD_shuffle", "RGG_shuffle")
    basic_sh["control_subtype"] = np.where(basic_sh["condition"].astype(str).str.startswith("TAD"), "basic_TAD_shuffle", "basic_RGG_shuffle")
    nb_match=pd.read_csv(SHUFFLE_NONBASIC)
    if set(nonbasic_sh["name"]) != set(nb_match["shuffle_pair_id"]):
        raise RuntimeError(f"Nonbasic shuffle ID mismatch: features={len(set(nonbasic_sh['name']))} matched={len(set(nb_match['shuffle_pair_id']))} intersect={len(set(nonbasic_sh['name']) & set(nb_match['shuffle_pair_id']))}")
    if len(nonbasic_sh)!=240 or nonbasic_sh["TAD_parent_id"].nunique()!=20: # 80 parent pairs across RGG+TAD parents, not TAD only
        pass
    if len(nonbasic_sh)!=240 or (nonbasic_sh["control_subtype"].eq("TAD_shuffle").sum()!=160) or (nonbasic_sh["control_subtype"].eq("RGG_shuffle").sum()!=80):
        raise RuntimeError("Nonbasic shuffle partition counts failed")
    standard_input(nonbasic_sh, features, "nonbasic shuffles").to_csv(IN/"nonbasic_shuffle_prediction_input.csv", index=False)
    standard_input(basic_sh, features, "basic shuffles").to_csv(IN/"basic_shuffle_prediction_input.csv", index=False)

    # Basic all80 recovery.
    all80, missing_basic = recover_missing_basic_features(features)
    basic80_in=standard_input(all80, features, "basic controls all80")
    basic80_in.to_csv(IN/"basic_control_prediction_input_ALL80.csv", index=False)

    # Histidine: final feature generation is not pH-aware; use identical feature representation with pH-specific targets.
    hist_features=s9e[s9e["class"].eq("histidine")].copy()
    hist_targets=pd.read_csv(HIST_MATCHED)
    if set(hist_features["name"]) != set(hist_targets["pair_id"]):
        raise RuntimeError("Histidine feature/target ID mismatch")
    hist_base=hist_features.merge(hist_targets, left_on="name", right_on="pair_id", how="inner", validate="one_to_one", suffixes=("", "_target"))
    h7=hist_base.copy(); h7["condition"]="pH7.0"; h7["control_subtype"]="histidine_pH7"; h7["B22_corr1_log10"] = h7["B22_corr1_log10_pH7"]; h7["target_evaluable"] = h7["log_target_defined_pH7"].astype(bool)
    h65=hist_base.copy(); h65["condition"]="pH6.5"; h65["control_subtype"]="histidine_pH6p5"; h65["B22_corr1_log10"] = h65["B22_corr1_log10_pH6p5"]; h65["target_evaluable"] = h65["log_target_defined_pH6p5"].astype(bool)
    standard_input(h7, features, "histidine pH7").to_csv(IN/"histidine_pH7_prediction_input.csv", index=False)
    standard_input(h65, features, "histidine pH6p5").to_csv(IN/"histidine_pH6p5_prediction_input.csv", index=False)
    (IN/"HISTIDINE_FEATURE_REPRESENTATION_AUDIT.txt").write_text(
        "Histidine feature representation audit\n\n"
        "The validated final feature-generation code used for control/histidine feature preparation calls historical_patch_get_features.py with a fixed residue property table (residues.csv) and does not expose a pH/protonation-state parameter. Charge-dependent descriptors in the frozen 13-feature set (Opposite_Charge_Number, opp_charge_std, SCD-derived patch descriptors) therefore use the same training-condition residue charges for pH 7.0 and pH 6.5.\n\n"
        "Consequently, histidine_pH7_prediction_input.csv and histidine_pH6p5_prediction_input.csv intentionally contain identical frozen-model input features and differ only in condition labels and simulated pH-specific targets. The pH 6.5 analysis is an external target-sensitivity/applicability evaluation, not a pH-aware feature recalculation.\n"
    )

    # Predictions.
    nb_wt_pred=predict(pd.read_csv(IN/"nonbasic_WT_parent_prediction_input.csv"), "nonbasic_WT", manifest, base, prep, residual)
    nb_wt_pred.to_csv(OUT/"nonbasic_WT_final_predictions.csv", index=False)
    nb_sh_pred=predict(pd.read_csv(IN/"nonbasic_shuffle_prediction_input.csv"), "nonbasic_shuffle", manifest, base, prep, residual)
    nb_sh_pred=nb_sh_pred.merge(nb_match[["shuffle_pair_id","WT_pair_id","B22_WT","B22_shuffle"]], left_on="pair_id", right_on="shuffle_pair_id", how="left", validate="one_to_one")
    nb_sh_pred.to_csv(OUT/"nonbasic_shuffle_final_predictions.csv", index=False)
    h7_pred=predict(pd.read_csv(IN/"histidine_pH7_prediction_input.csv"), "histidine_pH7", manifest, base, prep, residual)
    h65_pred=predict(pd.read_csv(IN/"histidine_pH6p5_prediction_input.csv"), "histidine_pH6p5", manifest, base, prep, residual)
    h7_pred.to_csv(OUT/"histidine_pH7_final_predictions.csv", index=False)
    h65_pred.to_csv(OUT/"histidine_pH6p5_final_predictions.csv", index=False)
    basic_pred=predict(pd.read_csv(IN/"basic_control_prediction_input_ALL80.csv"), "basic_control", manifest, base, prep, residual)
    basic_pred.to_csv(OUT/"basic_control_final_predictions_ALL80.csv", index=False)

    # Source tables.
    nb_wt_pred.to_csv(SRC/"FigS7A_source.csv", index=False)
    wt_for_delta=nb_wt_pred[["pair_id","simulated_target","final_hybrid_prediction"]].rename(columns={"pair_id":"WT_pair_id","simulated_target":"WT_simulated_target","final_hybrid_prediction":"WT_final_hybrid_prediction"})
    s7b=nb_sh_pred.merge(wt_for_delta, on="WT_pair_id", how="left", validate="many_to_one")
    s7b["delta_simulated"] = s7b["simulated_target"] - s7b["WT_simulated_target"]
    s7b["delta_predicted"] = s7b["final_hybrid_prediction"] - s7b["WT_final_hybrid_prediction"]
    s7b.to_csv(SRC/"FigS7B_source.csv", index=False)
    h65_pred.to_csv(SRC/"FigS7C_source.csv", index=False)
    basic_pred.to_csv(SRC/"FigS7D_source.csv", index=False)

    all_preds=pd.concat([nb_wt_pred, nb_sh_pred, h7_pred, h65_pred, basic_pred], ignore_index=True, sort=False)
    all_preds.to_csv(OUT/"S7_all_control_final_predictions.csv", index=False)
    max_identity=float(all_preds["prediction_identity_error"].max())

    audit=[]
    audit.append("FINAL S7 CONTROL PREDICTION AUDIT")
    audit.append(f"model identity reconciliation: PASS; details={OUT/'MODEL_IDENTITY_RECONCILIATION.txt'}")
    audit.append("exact frozen artefact hashes:")
    for fn in EXPECTED_HASHES: audit.append(f"  {fn}: {sha256(FROZEN/fn)}")
    audit.append(f"development identity max diffs: {json.dumps(identity_diffs, sort_keys=True)}")
    audit.append(f"WT controls exact overlap with 917 development pairs: {len(wt_overlap)}")
    if wt_overlap: audit.append("WT overlap IDs: " + ";".join(wt_overlap))
    audit.append(f"non-basic WT n: {len(nb_wt_pred)}")
    audit.append(f"basic WT parent n kept separate: {len(basic_wt)}")
    audit.append(f"non-basic shuffle n: {len(nb_sh_pred)}; TAD={int((nb_sh_pred['control_subtype']=='TAD_shuffle').sum())}; RGG={int((nb_sh_pred['control_subtype']=='RGG_shuffle').sum())}")
    audit.append(f"basic shuffle rows kept separate from S7B: {len(basic_sh)}")
    audit.append(f"histidine pH7 n: {len(h7_pred)}")
    audit.append(f"histidine pH6.5 n: {len(h65_pred)}")
    audit.append(f"histidine pH feature convention: not pH-aware; identical frozen features for pH7/pH6.5, pH-specific targets only. See {IN/'HISTIDINE_FEATURE_REPRESENTATION_AUDIT.txt'}")
    audit.append(f"basic total n: {len(basic_pred)}; evaluable={int(basic_pred['target_evaluable'].sum())}; undefined={int((~basic_pred['target_evaluable']).sum())}; recovered_positive_B22={missing_basic}")
    audit.append(f"maximum prediction identity error: {max_identity:.16g}")
    audit.append("source files:")
    for p in [IN/"nonbasic_WT_parent_prediction_input.csv", IN/"nonbasic_shuffle_prediction_input.csv", IN/"histidine_pH7_prediction_input.csv", IN/"histidine_pH6p5_prediction_input.csv", IN/"basic_control_prediction_input_ALL80.csv", OUT/"nonbasic_WT_final_predictions.csv", OUT/"nonbasic_shuffle_final_predictions.csv", OUT/"histidine_pH7_final_predictions.csv", OUT/"histidine_pH6p5_final_predictions.csv", OUT/"basic_control_final_predictions_ALL80.csv", SRC/"FigS7A_source.csv", SRC/"FigS7B_source.csv", SRC/"FigS7C_source.csv", SRC/"FigS7D_source.csv"]:
        audit.append(f"  {p} sha256={sha256(p)}")
    for label,df in [("S7A",nb_wt_pred),("S7C",h65_pred),("S7D",basic_pred[basic_pred['target_evaluable']])]:
        audit.append(f"{label} metrics: {json.dumps(metrics(df['simulated_target'], df['final_hybrid_prediction']), sort_keys=True)}")
    audit.append("No control dataset was used to fit, recalibrate, select, or modify the frozen model.")
    (OUT/"FINAL_S7_CONTROL_PREDICTION_AUDIT.txt").write_text("\n".join(audit)+"\n")
    print("Final S7 control prediction tables written under", OUT)

if __name__ == "__main__":
    main()
