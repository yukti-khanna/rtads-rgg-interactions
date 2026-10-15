#!/usr/bin/env python3
"""Part II-A: generate definitive frozen-model predictions for Figure S8 controls.

This script is intentionally conservative. It regenerates model predictions from the
frozen final hybrid model, and it stops before plotting if any requested control
subset cannot be reconciled to the exact frozen 13-feature representation.

Run from anywhere, for example:

  python /Users/yuktikhanna/rtads-rgg-interactions/fig5/scripts/part_IIA_generate_S8_frozen_predictions.py

Optional:

  python /path/to/fig5/scripts/part_IIA_generate_S8_frozen_predictions.py \
    --fig5-root /path/to/fig5
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DEFAULT_EXPECTED_MANIFEST_SHA256 = "48b7279b7e76260c04c1f42ee766d3f385f3d27c5ad61b99b2941d0d02b0d516"


@dataclass
class Context:
    root: Path
    out: Path
    manifest: dict
    manifest_sha: str
    base_model_sha: str
    residual_preprocessor_sha: str
    residual_model_sha: str
    base: object
    prep: object
    residual: object


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, **kwargs)


def soft_gate(x, center, scale):
    return 1.0 / (1.0 + np.exp(-((np.asarray(x, dtype=float) - center) / scale)))


def metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok]
    p = p[ok]
    n = len(y)
    return {
        "n": int(n),
        "R2": float(r2_score(y, p)) if n > 1 else np.nan,
        "RMSE": float(mean_squared_error(y, p) ** 0.5) if n else np.nan,
        "MAE": float(mean_absolute_error(y, p)) if n else np.nan,
        "Pearson_r": float(stats.pearsonr(y, p).statistic) if n > 2 else np.nan,
        "Spearman_rho": float(stats.spearmanr(y, p).statistic) if n > 2 else np.nan,
    }


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mean_lambda_RGG" not in df.columns and {"lambda_RGG", "RGG_length"}.issubset(df.columns):
        df["mean_lambda_RGG"] = df["lambda_RGG"] / df["RGG_length"]
    if "mean_lambda_TAD" not in df.columns and {"lambda_TAD", "TAD_length"}.issubset(df.columns):
        df["mean_lambda_TAD"] = df["lambda_TAD"] / df["TAD_length"]
    return df


def load_context(root: Path) -> Context:
    frozen = root / "outputs" / "final_hybrid_model" / "frozen_model"
    manifest_path = frozen / "MODEL_MANIFEST.json"
    base_path = frozen / "base_model_ensemble.joblib"
    prep_path = frozen / "residual_preprocessor.joblib"
    residual_path = frozen / "residual_nn.keras"
    missing = [p for p in [manifest_path, base_path, prep_path, residual_path] if not p.exists()]
    if missing:
        raise RuntimeError("Frozen model package is incomplete. Missing:\n  " + "\n  ".join(map(str, missing)))
    import tensorflow as tf
    manifest = json.loads(manifest_path.read_text())
    out = root / "controls" / "final_predictions"
    out.mkdir(parents=True, exist_ok=True)
    return Context(
        root=root,
        out=out,
        manifest=manifest,
        manifest_sha=sha256(manifest_path),
        base_model_sha=sha256(base_path),
        residual_preprocessor_sha=sha256(prep_path),
        residual_model_sha=sha256(residual_path),
        base=joblib.load(base_path),
        prep=joblib.load(prep_path),
        residual=tf.keras.models.load_model(residual_path),
    )


def require_feature_table(df: pd.DataFrame, ctx: Context, label: str) -> pd.DataFrame:
    df = derive_features(df)
    features = ctx.manifest["feature_order"]
    missing = [c for c in ["name", *features] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{label}: missing frozen-model columns: {missing}")
    if not df["name"].is_unique:
        dup = df.loc[df["name"].duplicated(), "name"].head(10).tolist()
        raise RuntimeError(f"{label}: duplicate name values, examples: {dup}")
    bad = df[features].isna().any(axis=1)
    if bad.any():
        raise RuntimeError(f"{label}: {int(bad.sum())} rows have missing frozen-model feature values")
    if not np.isfinite(df[features].to_numpy(float)).all():
        raise RuntimeError(f"{label}: non-finite frozen-model feature value detected")
    return df


def split_pair(pair_id: str) -> tuple[str, str]:
    if isinstance(pair_id, str) and "__" in pair_id:
        a, b = pair_id.split("__", 1)
        return a, b
    return "", ""


def predict(df: pd.DataFrame, ctx: Context, dataset: str, target_col: str | None = None, target_evaluable_col: str | None = None) -> pd.DataFrame:
    df = require_feature_table(df, ctx, dataset)
    features = ctx.manifest["feature_order"]
    X = df[features].to_numpy(float)
    X_base = ctx.base["scaler"].transform(X)
    base_prediction = np.mean([m.predict(X_base) for m in ctx.base["models"]], axis=0)
    X_resid = ctx.prep["scaler"].transform(X)
    raw = ctx.residual.predict(X_resid, verbose=0).reshape(-1)
    minus_ncp = df[ctx.manifest["minus_NCP_feature"]].to_numpy(float)
    gate = soft_gate(minus_ncp, ctx.manifest["gate_center"], ctx.manifest["gate_scale"])
    clipped = np.clip(raw, -ctx.manifest["residual_cap"], ctx.manifest["residual_cap"])
    uncapped = ctx.manifest["lambda_final"] * gate * raw
    applied = ctx.manifest["lambda_final"] * gate * clipped
    final = base_prediction + applied
    out = pd.DataFrame({
        "control_dataset": dataset,
        "pair_id": df["name"].astype(str).to_numpy(),
        "minus_NCP": minus_ncp,
        "base_prediction": base_prediction,
        "raw_residual_prediction": raw,
        "gate_value": gate,
        "lambda_final": ctx.manifest["lambda_final"],
        "uncapped_correction": uncapped,
        "applied_correction": applied,
        "final_hybrid_prediction": final,
        "frozen_model_manifest_sha256": ctx.manifest_sha,
        "base_model_sha256": ctx.base_model_sha,
        "residual_preprocessor_sha256": ctx.residual_preprocessor_sha,
        "residual_model_sha256": ctx.residual_model_sha,
        "feature_order_json": json.dumps(features),
    })
    ids = out["pair_id"].map(split_pair)
    out.insert(2, "RGG_id", [x[0] for x in ids])
    out.insert(3, "TAD_id", [x[1] for x in ids])
    for src, dst in [
        ("pair_id", "source_pair_id"),
        ("condition", "condition"),
        ("TAD_variant_type", "TAD_variant_type"),
        ("RGG_variant_type", "RGG_variant_type"),
        ("TAD_parent_id", "TAD_parent_id"),
        ("RGG_parent_id", "RGG_parent_id"),
        ("feature_name", "feature_name"),
        ("feature_name_generated", "feature_name_generated"),
        ("control_subtype", "control_subtype"),
        ("class", "source_class"),
        ("B22_sign", "B22_sign"),
        ("reason_target_undefined", "reason_target_undefined"),
    ]:
        if src in df.columns and dst not in out.columns:
            out[dst] = df[src].to_numpy()
    if target_col is not None:
        out["simulated_target"] = pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
    if target_evaluable_col and target_evaluable_col in df.columns:
        out["target_evaluable"] = df[target_evaluable_col].astype(bool).to_numpy()
    else:
        out["target_evaluable"] = np.isfinite(out["simulated_target"]) if "simulated_target" in out else False
    if "simulated_target" in out:
        out["residual"] = out["simulated_target"] - out["final_hybrid_prediction"]
        out["absolute_error"] = np.abs(out["residual"])
    out["prediction_identity_error"] = np.abs(out["final_hybrid_prediction"] - (out["base_prediction"] + out["applied_correction"]))
    return out


def load_s9e(ctx: Context) -> pd.DataFrame:
    p = ctx.root / "outputs" / "revised_fig5_supplementary_data" / "csv" / "Table_S9_reviewer_controls" / "S9E_applicability_domain.csv"
    return read_csv(p, low_memory=False)


def load_s9d(ctx: Context) -> pd.DataFrame:
    p = ctx.root / "outputs" / "revised_fig5_supplementary_data" / "csv" / "Table_S9_reviewer_controls" / "S9D_basic_region_controls.csv"
    return read_csv(p, low_memory=False)


def input_dir(ctx: Context) -> Path:
    return ctx.root / "controls" / "final_prediction_inputs"


def load_prepared_input(ctx: Context, filename: str, label: str) -> pd.DataFrame:
    p = input_dir(ctx) / filename
    if not p.exists():
        raise RuntimeError(f"Prepared input missing for {label}: {p}. Run scripts/prepare_S8_final_prediction_inputs.py first.")
    return read_csv(p, low_memory=False)


def build_non_dataset_wt(ctx: Context, s9e: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = load_prepared_input(ctx, "non_dataset_WT_prediction_input.csv", "non-dataset WT")
    pred = predict(df, ctx, "non_dataset_WT", target_col="B22_corr1_log10")
    pred.to_csv(ctx.out / "non_dataset_WT_final_predictions.csv", index=False)
    return pred, {"requested_n": int(len(df)), "successfully_predicted_n": int(len(pred)), "missing_feature_n": 0}


def build_shuffle_predictions(ctx: Context, s9e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    wt = load_prepared_input(ctx, "shuffle_WT_parent_prediction_input.csv", "shuffle WT parents")
    wt_pred = predict(wt, ctx, "shuffle_WT_parent", target_col="B22_corr1_log10")
    wt_pred = wt_pred.rename(columns={"pair_id": "parent_WT_pair_id", "simulated_target": "simulated_WT_target"})
    wt_pred.to_csv(ctx.out / "shuffle_WT_parent_final_predictions.csv", index=False)

    sh = load_prepared_input(ctx, "shuffle_prediction_input.csv", "shuffle controls")
    sh_pred = predict(sh, ctx, "shuffle", target_col="B22_corr1_log10")
    sh_pred = sh_pred.rename(columns={"pair_id": "shuffle_pair_id", "simulated_target": "simulated_shuffle_target"})
    sh_pred["parent_WT_pair_id"] = sh_pred.apply(
        lambda r: f"{r.get('RGG_parent_id','')}__{r.get('TAD_parent_id','')}" if pd.notna(r.get("RGG_parent_id", np.nan)) and pd.notna(r.get("TAD_parent_id", np.nan)) else "",
        axis=1,
    )
    sh_pred.to_csv(ctx.out / "shuffle_final_predictions.csv", index=False)
    info = {
        "WT_parent_n": int(len(wt_pred)),
        "shuffle_n": int(len(sh_pred)),
        "TAD_shuffle_n": int((sh_pred["control_subtype"] == "TAD_shuffle").sum()),
        "RGG_shuffle_n": int((sh_pred["control_subtype"] == "RGG_shuffle").sum()),
    }
    return wt_pred, sh_pred, info


def build_histidine_predictions_or_stop(ctx: Context, s9e: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Require unambiguous pH-specific feature sources.

    S9E contains complete histidine feature rows, but no pH condition. This function
    only accepts tables whose rows explicitly label pH 7.0 and pH 6.5 feature states.
    """
    candidates = []
    for p in ctx.root.rglob("*.csv"):
        s = str(p).lower()
        if any(x in s for x in ["histidine", "ph65", "ph6", "ph7"]):
            try:
                df0 = pd.read_csv(p, nrows=0)
            except Exception:
                continue
            cols = set(df0.columns)
            has_all = all(c in cols for c in ctx.manifest["feature_order"])
            has_name = "name" in cols
            pH_cols = [c for c in df0.columns if "ph" in c.lower() or c.lower() == "condition"]
            candidates.append({"path": str(p), "has_name": has_name, "has_all_frozen_features": has_all, "pH_or_condition_columns": ";".join(pH_cols)})
    cand = pd.DataFrame(candidates)
    cand.to_csv(ctx.out / "histidine_pH_feature_source_candidates.csv", index=False)
    usable = []
    for row in candidates:
        if not (row["has_name"] and row["has_all_frozen_features"] and row["pH_or_condition_columns"]):
            continue
        df = pd.read_csv(row["path"], low_memory=False)
        text = " ".join(str(df[c].dropna().astype(str).unique()[:10]) for c in df.columns if "ph" in c.lower() or c.lower() == "condition").lower()
        if ("6.5" in text or "6p5" in text or "ph65" in text) and ("7" in text or "7.0" in text):
            usable.append(Path(row["path"]))
    if not usable:
        raise RuntimeError(
            "Cannot generate histidine pH7/pH6.5 frozen-model predictions: no unambiguous pH-specific feature table was found. "
            "S9E contains complete histidine frozen features but no pH condition label, and using those rows for both pH states would silently use the wrong representation. "
            f"Candidate audit written to {ctx.out / 'histidine_pH_feature_source_candidates.csv'}"
        )
    raise RuntimeError("Multiple/usable pH-specific histidine feature tables need explicit selection: " + ", ".join(map(str, usable)))


def build_basic(ctx: Context) -> tuple[pd.DataFrame, dict]:
    df = load_prepared_input(ctx, "basic_control_prediction_input.csv", "basic controls")
    target_col = "B22_corr1_log10" if "B22_corr1_log10" in df.columns else None
    pred = predict(df, ctx, "basic_control", target_col=target_col, target_evaluable_col="transformed_target_defined")
    if "simulated_corrected_B22" in df.columns:
        pred["simulated_B22"] = df["simulated_corrected_B22"].to_numpy()
    if "Corrected_B22_Hummer" in df.columns:
        pred["Corrected_B22_Hummer"] = df["Corrected_B22_Hummer"].to_numpy()
    pred.to_csv(ctx.out / "basic_control_final_predictions.csv", index=False)
    return pred, {
        "total_n": int(len(pred)),
        "successfully_predicted_n": int(len(pred)),
        "target_evaluable_n": int(pred["target_evaluable"].sum()),
        "positive_B22_or_target_undefined_n": int((~pred["target_evaluable"].astype(bool)).sum()),
    }


def write_audit(ctx: Context, records: dict, max_identity: dict, stopped: str | None = None) -> None:
    lines = []
    lines.append("S8 definitive frozen-model prediction audit")
    lines.append("")
    lines.append(f"fig5 root: {ctx.root}")
    lines.append(f"frozen model directory: {ctx.root / 'outputs/final_hybrid_model/frozen_model'}")
    lines.append(f"frozen model manifest SHA256: {ctx.manifest_sha}")
    lines.append(f"expected definitive final-prediction manifest SHA256: {DEFAULT_EXPECTED_MANIFEST_SHA256}")
    if ctx.manifest_sha != DEFAULT_EXPECTED_MANIFEST_SHA256:
        lines.append("WARNING: local MODEL_MANIFEST hash differs from the SHA256 embedded in the definitive final prediction tables.")
    lines.append(f"base model SHA256: {ctx.base_model_sha}")
    lines.append(f"residual preprocessor SHA256: {ctx.residual_preprocessor_sha}")
    lines.append(f"residual model SHA256: {ctx.residual_model_sha}")
    lines.append(f"feature order: {ctx.manifest['feature_order']}")
    lines.append(f"lambda: {ctx.manifest['lambda_final']}")
    lines.append(f"cap: {ctx.manifest['residual_cap']}")
    lines.append(f"gate centre: {ctx.manifest['gate_center']}")
    lines.append(f"gate scale: {ctx.manifest['gate_scale']}")
    lines.append("")
    for k, v in records.items():
        lines.append(k)
        lines.append(json.dumps(v, indent=2, sort_keys=True))
        if k in max_identity:
            lines.append(f"maximum numerical error in final = base + correction: {max_identity[k]:.16g}")
        lines.append("")
    if stopped:
        lines.append("STOPPED BEFORE S8 PLOTTING")
        lines.append(stopped)
        lines.append("")
    lines.append("All S8 control predictions were generated after the final hybrid model had been frozen. No S8 control simulation or prediction was used to fit, select, recalibrate, or modify the model.")
    (ctx.out / "S8_FINAL_PREDICTION_AUDIT.txt").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig5-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--allow-ambiguous-histidine", action="store_true", help="Do not use for manuscript outputs; retained only for debugging.")
    args = ap.parse_args()
    ctx = load_context(args.fig5_root.resolve())
    # Load S9E only for legacy function signatures; prediction subsets are read
    # from controls/final_prediction_inputs prepared by prepare_S8_final_prediction_inputs.py.
    s9e = load_s9e(ctx)
    records = {}
    max_identity = {}
    all_tables = []

    non, records["Non-dataset WT"] = build_non_dataset_wt(ctx, s9e)
    max_identity["Non-dataset WT"] = float(non["prediction_identity_error"].max())
    all_tables.append(non)

    wt, sh, sh_info = build_shuffle_predictions(ctx, s9e)
    records["Shuffle WT parents"] = {"n": sh_info["WT_parent_n"]}
    records["Shuffles"] = {k: v for k, v in sh_info.items() if k != "WT_parent_n"}
    max_identity["Shuffle WT parents"] = float(wt["prediction_identity_error"].max())
    max_identity["Shuffles"] = float(sh["prediction_identity_error"].max())
    all_tables.extend([wt, sh])

    basic, records["Basic controls"] = build_basic(ctx)
    max_identity["Basic controls"] = float(basic["prediction_identity_error"].max())
    all_tables.append(basic)

    stopped = None
    try:
        h7, h65, hinfo = build_histidine_predictions_or_stop(ctx, s9e)
        records["Histidine pH 7.0"] = {"n": len(h7), "feature_protonation_convention": hinfo.get("pH7_convention", "documented in source")}
        records["Histidine pH 6.5"] = {"n": len(h65), "feature_protonation_convention": hinfo.get("pH6p5_convention", "documented in source")}
        max_identity["Histidine pH 7.0"] = float(h7["prediction_identity_error"].max())
        max_identity["Histidine pH 6.5"] = float(h65["prediction_identity_error"].max())
        all_tables.extend([h7, h65])
    except Exception as e:
        stopped = str(e)
        write_audit(ctx, records, max_identity, stopped=stopped)
        print(stopped)
        raise SystemExit(2)

    master = pd.concat(all_tables, ignore_index=True, sort=False)
    if "parent_pair_id" not in master.columns:
        master["parent_pair_id"] = ""
    if "condition" not in master.columns:
        master["condition"] = ""
    if "control_subtype" not in master.columns:
        master["control_subtype"] = ""
    master.to_csv(ctx.out / "S8_all_control_final_predictions.csv", index=False)
    write_audit(ctx, records, max_identity)
    print(f"Wrote S8 frozen-model predictions to {ctx.out}")


if __name__ == "__main__":
    main()
