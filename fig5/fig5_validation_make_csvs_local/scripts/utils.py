from pathlib import Path
import json, hashlib, os, pickle, re
import numpy as np
import pandas as pd
FEATURES = json.loads((Path(__file__).resolve().parents[1]/'config/model_config.json').read_text())['features']
TARGET = json.loads((Path(__file__).resolve().parents[1]/'config/model_config.json').read_text())['target']
def root(): return Path(__file__).resolve().parents[1]
def ensure_dirs():
    for d in ['artifacts/final_model','artifacts/checkpoints','outputs/heldout','outputs/histidine','outputs/controls','outputs/figures','outputs/summary','logs','provenance']:(root()/d).mkdir(parents=True,exist_ok=True)
def sha256(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def write_checkpoint(name, outputs):
    cp=root()/'artifacts/checkpoints'; cp.mkdir(exist_ok=True,parents=True)
    rows=[]
    for o in outputs:
        p=root()/o
        if not p.exists(): raise FileNotFoundError(p)
        rows.append({'path':o,'sha256':sha256(p),'size':p.stat().st_size})
    (cp/f'{name}.json').write_text(json.dumps({'stage':name,'outputs':rows},indent=2))
def checkpoint_ok(name):
    p=root()/f'artifacts/checkpoints/{name}.json'
    if not p.exists(): return False
    d=json.loads(p.read_text())
    for row in d.get('outputs',[]):
        q=root()/row['path']
        if not q.exists() or sha256(q)!=row['sha256']: return False
    return True
def read_csv(p): return pd.read_csv(root()/p)
def derive_means(df):
    df=df.copy()
    if 'mean_lambda_RGG' not in df and {'lambda_RGG','RGG_length'} <= set(df.columns): df['mean_lambda_RGG']=df['lambda_RGG']/df['RGG_length']
    if 'mean_lambda_TAD' not in df and {'lambda_TAD','TAD_length'} <= set(df.columns): df['mean_lambda_TAD']=df['lambda_TAD']/df['TAD_length']
    return df
def merge_by_name(base, patch, patch_cols):
    assert base['name'].is_unique; assert patch['name'].is_unique
    if set(base['name']) != set(patch['name']):
        raise SystemExit(f'ID mismatch: base={base.name.nunique()} patch={patch.name.nunique()} intersection={len(set(base.name)&set(patch.name))}')
    return base.merge(patch[['name']+patch_cols], on='name', how='left', validate='one_to_one')
def load_development():
    base=derive_means(read_csv('inputs/development/sim_pairs_pred_with_hybrid_sel_cols_with_old.csv'))
    patch=read_csv('inputs/development/features_sim_pairs_for_training.csv')
    if len(base)!=917 or len(patch)!=917: raise SystemExit(f'Expected 917 dev rows, got base={len(base)} patch={len(patch)}')
    patch_cols=[c for c in FEATURES if c in patch.columns and c not in base.columns]
    # Always use local patch descriptors from patch table; keep global columns from base table.
    for c in ['opp_charge_std','scd_patch1_std','scd_patch2_std','rg_fraction_patch1_std','aromatic_fraction_patch2_std','shd_patch1_std','shd_patch2_std','ah_energy_std']:
        if c not in patch.columns: raise SystemExit(f'Missing dev patch feature {c}')
    df=base.merge(patch[['name','opp_charge_std','scd_patch1_std','scd_patch2_std','rg_fraction_patch1_std','aromatic_fraction_patch2_std','shd_patch1_std','shd_patch2_std','ah_energy_std']],on='name',how='left',validate='one_to_one')
    require_features(df)
    return df
def require_features(df):
    miss=[c for c in FEATURES if c not in df.columns]
    if miss: raise SystemExit('Missing required features: '+', '.join(miss))
    if TARGET in df.columns and not np.isfinite(df[TARGET]).all(): raise SystemExit('Non-finite target values')
    if not np.isfinite(df[FEATURES].to_numpy(float)).all(): raise SystemExit('Non-finite feature values')
def metrics(y,p):
    from scipy import stats
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    y=np.asarray(y,float); p=np.asarray(p,float); n=len(y)
    mask=np.isfinite(y) & np.isfinite(p)
    y=y[mask]; p=p[mask]; n=len(y)
    mse=mean_squared_error(y,p) if n else np.nan
    return {'n':n,'R2':r2_score(y,p) if n>1 else np.nan,'RMSE':float(np.sqrt(mse)) if n else np.nan,'MAE':mean_absolute_error(y,p) if n else np.nan,'Pearson':stats.pearsonr(y,p)[0] if n>2 else np.nan,'Spearman':stats.spearmanr(y,p).correlation if n>2 else np.nan,'mean_residual':float(np.mean(p-y)) if n else np.nan,'median_residual':float(np.median(p-y)) if n else np.nan}
def parse_pair_from_rdf_stem(stem): return stem.replace('rdfs_','').replace('_20nm','')

def config():
    return json.loads((root()/'config/run_config.json').read_text())
def resolve_input_path(path_value):
    p=Path(path_value)
    if p.is_absolute(): return p
    return root()/p
