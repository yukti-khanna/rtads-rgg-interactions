#!/usr/bin/env python3
"""Prepare input feature CSVs for Part II-A S8 frozen-model predictions.

This script does not run the model. It only extracts/reconciles the available
control feature rows into per-dataset input files with the exact frozen feature
order and required metadata.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "controls" / "final_prediction_inputs"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = ROOT / "outputs" / "final_hybrid_model" / "frozen_model" / "MODEL_MANIFEST.json"
S9 = ROOT / "outputs" / "revised_fig5_supplementary_data" / "csv" / "Table_S9_reviewer_controls"
S9E = S9 / "S9E_applicability_domain.csv"
S9D = S9 / "S9D_basic_region_controls.csv"
S9C = S9 / "S9C_histidine_pH.csv"
HIST_MATCHED = ROOT / "outputs" / "fig5_histidine_pH_audit" / "histidine_matched_b22.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def split_pair(x: str):
    if isinstance(x, str) and '__' in x:
        a, b = x.split('__', 1)
        return a, b
    return '', ''


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def require_complete(df: pd.DataFrame, features: list[str], label: str):
    missing = [c for c in ['name', *features] if c not in df.columns]
    if missing:
        raise RuntimeError(f'{label}: missing required columns: {missing}')
    if not df['name'].is_unique:
        raise RuntimeError(f'{label}: name column is not unique')
    bad = df[features].isna().any(axis=1)
    if bad.any():
        raise RuntimeError(f'{label}: {int(bad.sum())} rows have missing frozen features')
    if not np.isfinite(df[features].to_numpy(float)).all():
        raise RuntimeError(f'{label}: non-finite frozen feature values')


def standardize(df: pd.DataFrame, features: list[str], label: str, target_col='B22_corr1_log10') -> pd.DataFrame:
    require_complete(df, features, label)
    out = pd.DataFrame()
    out['name'] = df['name'].astype(str)
    ids = out['name'].map(split_pair)
    out['RGG_id'] = [x[0] for x in ids]
    out['TAD_id'] = [x[1] for x in ids]
    for c in ['pair_id','condition','control_subtype','TAD_variant_type','RGG_variant_type','TAD_parent_id','RGG_parent_id','feature_name','feature_name_generated','class','B22_sign','transformed_target_defined','reason_target_undefined']:
        if c in df.columns:
            out[c] = df[c].to_numpy()
    if target_col and target_col in df.columns:
        out['B22_corr1_log10'] = pd.to_numeric(df[target_col], errors='coerce')
    for c in ['simulated_corrected_B22','Corrected_B22_Hummer','log10_minus_B22','-NCP']:
        if c in df.columns:
            out[c] = df[c].to_numpy()
    for f in features:
        out[f] = pd.to_numeric(df[f], errors='raise')
    return out


def write(df: pd.DataFrame, name: str):
    path = OUT / name
    df.to_csv(path, index=False)
    return path


def main():
    manifest = load_manifest()
    features = manifest['feature_order']
    audit = []
    audit.append('S8 final prediction input preparation audit')
    audit.append(f'fig5 root: {ROOT}')
    audit.append(f'manifest: {MANIFEST_PATH}')
    audit.append(f'manifest_sha256: {sha256(MANIFEST_PATH)}')
    audit.append(f'feature_order: {features}')
    audit.append('')

    s9e = pd.read_csv(S9E, low_memory=False)
    s9d = pd.read_csv(S9D, low_memory=False)

    # Non-dataset WT and WT parents for shuffle experiments.
    wt = s9e[s9e['condition'].eq('WT_WT')].copy()
    wt['control_subtype'] = 'non_dataset_WT'
    wt_in = standardize(wt, features, 'non_dataset_WT')
    p = write(wt_in, 'non_dataset_WT_prediction_input.csv')
    audit.append(f'non_dataset_WT_prediction_input.csv: n={len(wt_in)}, source={S9E}, filter=condition == WT_WT, sha256={sha256(p)}')

    wt_parent = wt.copy()
    wt_parent['control_subtype'] = 'shuffle_WT_parent'
    wt_parent_in = standardize(wt_parent, features, 'shuffle_WT_parent')
    p = write(wt_parent_in, 'shuffle_WT_parent_prediction_input.csv')
    audit.append(f'shuffle_WT_parent_prediction_input.csv: n={len(wt_parent_in)}, source={S9E}, filter=condition == WT_WT, sha256={sha256(p)}')

    # Shuffles.
    sh = s9e[s9e['condition'].isin(['TAD_SHUFFLE_A','TAD_SHUFFLE_B','RGG_SHUFFLE'])].copy()
    sh['control_subtype'] = np.where(sh['condition'].astype(str).str.startswith('TAD'), 'TAD_shuffle', 'RGG_shuffle')
    sh_in = standardize(sh, features, 'shuffle_controls')
    p = write(sh_in, 'shuffle_prediction_input.csv')
    audit.append(f'shuffle_prediction_input.csv: n={len(sh_in)}, TAD_shuffle_n={(sh_in.control_subtype == "TAD_shuffle").sum()}, RGG_shuffle_n={(sh_in.control_subtype == "RGG_shuffle").sum()}, source={S9E}, sha256={sha256(p)}')

    # Basic controls.
    basic = s9d.copy()
    if 'control_subtype' not in basic.columns:
        basic['control_subtype'] = basic['condition'] if 'condition' in basic.columns else 'basic_control'
    basic_in = standardize(basic, features, 'basic_controls')
    p = write(basic_in, 'basic_control_prediction_input.csv')
    target_eval = int(basic_in.get('transformed_target_defined', pd.Series([True]*len(basic_in))).astype(bool).sum()) if len(basic_in) else 0
    audit.append(f'basic_control_prediction_input.csv: n={len(basic_in)}, target_evaluable_n={target_eval}, target_undefined_n={len(basic_in)-target_eval}, source={S9D}, sha256={sha256(p)}')

    # Histidine pH simulation targets only, not model-ready feature inputs.
    if HIST_MATCHED.exists():
        hist_targets = pd.read_csv(HIST_MATCHED)
        keep = [c for c in ['pair_id','TAD_id','RGG_id','B22_corr1_log10_pH7','B22_corr1_log10_pH6p5','log_target_defined_pH7','log_target_defined_pH6p5','delta_B22','delta_logB22','n_H_TAD','n_H_RGG','total_H','Q_TAD_pH7','Q_TAD_pH65','Q_RGG_pH7','Q_RGG_pH65','delta_Q_TAD','delta_Q_RGG','OCN_net_charge_product_pH7','OCN_net_charge_product_pH65','delta_OCN_proxy'] if c in hist_targets.columns]
        p = write(hist_targets[keep], 'histidine_pH_matched_targets_ONLY_not_model_features.csv')
        audit.append(f'histidine_pH_matched_targets_ONLY_not_model_features.csv: n={len(hist_targets)}, source={HIST_MATCHED}, sha256={sha256(p)}')

    # Candidate audit for pH feature tables.
    candidates = []
    for path in ROOT.rglob('*.csv'):
        s = str(path).lower()
        if any(x in s for x in ['histidine','ph65','ph6','ph7']):
            try:
                hdr = pd.read_csv(path, nrows=0)
            except Exception:
                continue
            cols = set(hdr.columns)
            candidates.append({
                'path': str(path),
                'has_name': 'name' in cols,
                'has_all_13_frozen_features': all(f in cols for f in features),
                'feature_count': sum(f in cols for f in features),
                'pH_or_condition_columns': ';'.join([c for c in hdr.columns if 'ph' in c.lower() or c.lower() == 'condition']),
            })
    cand = pd.DataFrame(candidates).sort_values(['has_all_13_frozen_features','feature_count','path'], ascending=[False,False,True])
    p = OUT / 'histidine_pH_feature_source_candidates.csv'
    cand.to_csv(p, index=False)
    audit.append(f'histidine_pH_feature_source_candidates.csv written; no pH7/pH6.5 model-ready inputs were created because no unambiguous pH-specific table with name + all 13 frozen features was identified automatically.')

    (OUT / 'HISTIDINE_PH_INPUTS_MISSING_OR_AMBIGUOUS.txt').write_text(
        'Histidine pH model-prediction inputs were not created.\n\n'
        'Reason: existing matched histidine pH tables provide pH7 and pH6.5 targets, but not the full frozen 13-feature rows. '
        'S9E contains complete histidine feature rows, but these rows do not label an explicit pH feature/protonation condition, so they cannot be safely used as both pH7 and pH6.5 inputs.\n\n'
        f'Candidate audit: {p}\n'
        f'Target-only input: {OUT / "histidine_pH_matched_targets_ONLY_not_model_features.csv"}\n'
    )
    audit.append('Histidine pH model-ready inputs: NOT CREATED; see HISTIDINE_PH_INPUTS_MISSING_OR_AMBIGUOUS.txt')

    (OUT / 'S8_FINAL_PREDICTION_INPUT_AUDIT.txt').write_text('\n'.join(audit) + '\n')
    print(f'Prepared S8 prediction inputs in {OUT}')


if __name__ == '__main__':
    main()
