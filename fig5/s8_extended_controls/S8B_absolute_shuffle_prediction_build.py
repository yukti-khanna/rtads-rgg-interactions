#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path('/Users/yuktikhanna/rtads-rgg-interactions/fig5')
SRC=ROOT/'s8_extended_controls/outputs/S8B_extended_source.csv'
OUT=ROOT/'s8_extended_controls/S8B_absolute_shuffle_prediction'
OUT.mkdir(parents=True, exist_ok=True)
FIGSIZE=(2.4,2.3); DPI=600; BLUE='#0072B2'; ORANGE='#D55E00'
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7})

def metric_abs(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); ok=np.isfinite(y)&np.isfinite(p); y=y[ok]; p=p[ok]
    n=len(y); resid=y-p
    return {'n':n,'R2':r2_score(y,p) if n>1 else np.nan,'RMSE':mean_squared_error(y,p)**0.5 if n else np.nan,'MAE':mean_absolute_error(y,p) if n else np.nan,'Pearson':stats.pearsonr(y,p).statistic if n>2 else np.nan,'Spearman':stats.spearmanr(y,p).statistic if n>2 else np.nan,'mean_residual':float(np.mean(resid)) if n else np.nan}

def metric_pair(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    n=len(x); diff=y-x
    out={'n':n,'Pearson':stats.pearsonr(x,y).statistic if n>2 else np.nan,'Spearman':stats.spearmanr(x,y).statistic if n>2 else np.nan,'RMSE':mean_squared_error(x,y)**0.5 if n else np.nan,'MAE':mean_absolute_error(x,y) if n else np.nan}
    return out

def delta_metric(ds,dp):
    out=metric_pair(ds,dp)
    ds=np.asarray(ds,float); dp=np.asarray(dp,float); ok=np.isfinite(ds)&np.isfinite(dp); ds=ds[ok]; dp=dp[ok]
    out['mean_delta_sim']=float(np.mean(ds)) if len(ds) else np.nan
    out['mean_delta_pred']=float(np.mean(dp)) if len(dp) else np.nan
    out['sign_agreement']=float(np.mean(np.sign(ds)==np.sign(dp))) if len(ds) else np.nan
    return out

def sim_similarity(wt,shuf):
    out=metric_pair(wt,shuf)
    wt=np.asarray(wt,float); shuf=np.asarray(shuf,float); ok=np.isfinite(wt)&np.isfinite(shuf); wt=wt[ok]; shuf=shuf[ok]
    out['mean_absolute_shuffle_effect']=float(np.mean(np.abs(shuf-wt))) if len(wt) else np.nan
    return out

def clean(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(direction='out',length=2.2,width=0.6,pad=1.5)

def limits(x,y):
    v=np.concatenate([np.asarray(x,float),np.asarray(y,float)]); v=v[np.isfinite(v)]
    lo=float(v.min()); hi=float(v.max()); pad=(hi-lo)*0.05 if hi>lo else .1
    return lo-pad, hi+pad

def main():
    df=pd.read_csv(SRC)
    # Normalize and audit required identities.
    expected_classes={'TAD/AHD shuffle','RG/RGG shuffle'}
    classes=set(df['shuffle_class'].dropna())
    if classes != expected_classes:
        raise RuntimeError(f'Unexpected shuffle classes: {classes}')
    df['shuffle_pair_id_audited']=df.get('shuffle_pair_id',df['name']).fillna(df['name'])
    # for new rows shuffle_pair_id may be NaN; use name.
    df.loc[df['shuffle_pair_id_audited'].isna(),'shuffle_pair_id_audited']=df.loc[df['shuffle_pair_id_audited'].isna(),'name']
    if df['parent_WT_pair_id'].isna().any():
        raise RuntimeError('Missing parent_WT_pair_id')
    parent_counts=df.groupby('shuffle_pair_id_audited')['parent_WT_pair_id'].nunique()
    if (parent_counts != 1).any():
        raise RuntimeError('At least one shuffle pair does not have a unique WT parent')
    left=df['name'].str.split('__',n=1).str[0]
    right=df['name'].str.split('__',n=1).str[1]
    tad=df['shuffle_class'].eq('TAD/AHD shuffle')
    rgg=df['shuffle_class'].eq('RG/RGG shuffle')
    audit={
        'total_rows': int(len(df)),
        'TAD_AHD_shuffle_rows': int(tad.sum()),
        'RG_RGG_shuffle_rows': int(rgg.sum()),
        'TAD_AHD_has_WT_RGG_partner': bool((left[tad].str.contains('_WT',regex=False) & ~left[tad].str.contains('_SHUF',regex=False) & right[tad].str.contains('_SHUF',regex=False)).all()),
        'RG_RGG_has_WT_AHD_partner': bool((right[rgg].str.contains('_WT',regex=False) & ~right[rgg].str.contains('_SHUF',regex=False) & left[rgg].str.contains('_SHUF',regex=False)).all()),
        'unique_shuffle_pair_ids': int(df['shuffle_pair_id_audited'].nunique()),
        'unique_WT_parent_ids': int(df['parent_WT_pair_id'].nunique()),
        'missing_WT_sim_target': int(df['WT_B22_corr1_log10'].isna().sum()),
        'missing_WT_prediction': int(df['WT_final_hybrid_prediction'].isna().sum()),
        'missing_shuffle_sim_target': int(df['B22_corr1_log10'].isna().sum()),
        'missing_shuffle_prediction': int(df['final_hybrid_prediction'].isna().sum()),
        'delta_simulated_recomputed_max_abs_diff': float(np.nanmax(np.abs((df['B22_corr1_log10']-df['WT_B22_corr1_log10'])-df['delta_simulated']))),
        'delta_predicted_recomputed_max_abs_diff': float(np.nanmax(np.abs((df['final_hybrid_prediction']-df['WT_final_hybrid_prediction'])-df['delta_predicted']))),
    }
    if not audit['TAD_AHD_has_WT_RGG_partner'] or not audit['RG_RGG_has_WT_AHD_partner']:
        raise RuntimeError('Shuffle orientation audit failed')
    # Recover raw WT B22 from the accepted WT-parent file plus new dataset WT rows.
    existing_parent = pd.read_csv(ROOT/'controls/final_predictions/shuffle_WT_parent_final_predictions.csv')
    new_pred_path = ROOT/'s8_extended_controls/outputs/new_controls_predictions.csv'
    new_pred = pd.read_csv(new_pred_path) if new_pred_path.exists() else pd.DataFrame()
    if not new_pred.empty and {'input_group','control_class'}.issubset(new_pred.columns):
        new_parent = new_pred[(new_pred['input_group'].eq('dataset_new')) & (new_pred['control_class'].eq('WT'))].copy()
    else:
        new_parent = pd.DataFrame(columns=existing_parent.columns)
    parent_raw = pd.concat([existing_parent, new_parent], ignore_index=True, sort=False).drop_duplicates('name')
    raw_col = 'Corrected_B22_Hummer' if 'Corrected_B22_Hummer' in parent_raw.columns else 'simulated_corrected_B22'
    parent_raw = parent_raw[['name', raw_col]].rename(columns={'name':'parent_WT_pair_id', raw_col:'simulated_WT_B22'})
    df = df.merge(parent_raw, on='parent_WT_pair_id', how='left', validate='many_to_one')
    audit['missing_raw_WT_B22'] = int(df['simulated_WT_B22'].isna().sum())
    audited=pd.DataFrame({
        'shuffle_pair_id':df['shuffle_pair_id_audited'],
        'name':df['name'],
        'WT_parent_pair_id':df['parent_WT_pair_id'],
        'shuffle_class':df['shuffle_class'],
        'control_origin':df['control_origin'],
        'simulated_WT_B22':df['simulated_WT_B22'],
        'simulated_shuffled_B22':df['Corrected_B22_Hummer'],
        'simulated_WT_target':df['WT_B22_corr1_log10'],
        'simulated_shuffled_target':df['B22_corr1_log10'],
        'predicted_WT_target':df['WT_final_hybrid_prediction'],
        'predicted_shuffled_target':df['final_hybrid_prediction'],
        'delta_simulated':df['delta_simulated'],
        'delta_predicted':df['delta_predicted'],
    })
    audited.to_csv(OUT/'S8B_shuffle_source_audited.csv',index=False)
    rows=[]
    for subset in ['original','new','combined']:
        subdf=df if subset=='combined' else df[df['control_origin'].eq(subset)]
        for cls in ['TAD/AHD shuffle','RG/RGG shuffle']:
            sub=subdf[subdf['shuffle_class'].eq(cls)]
            rows.append({'subset':subset,'shuffle_class':cls,**metric_abs(sub['B22_corr1_log10'],sub['final_hybrid_prediction'])})
    abs_metrics=pd.DataFrame(rows); abs_metrics.to_csv(OUT/'S8B_absolute_metrics_original_new_combined.csv',index=False)
    rows=[]
    for subset in ['original','new','combined']:
        subdf=df if subset=='combined' else df[df['control_origin'].eq(subset)]
        for cls in ['TAD/AHD shuffle','RG/RGG shuffle']:
            sub=subdf[subdf['shuffle_class'].eq(cls)]
            rows.append({'subset':subset,'shuffle_class':cls,**delta_metric(sub['delta_simulated'],sub['delta_predicted'])})
    delta_metrics=pd.DataFrame(rows); delta_metrics.to_csv(OUT/'S8B_delta_metrics_original_new_combined.csv',index=False)
    rows=[]
    for cls in ['TAD/AHD shuffle','RG/RGG shuffle']:
        sub=df[df['shuffle_class'].eq(cls)]
        rows.append({'shuffle_class':cls,**sim_similarity(sub['WT_B22_corr1_log10'],sub['B22_corr1_log10'])})
    sim_metrics=pd.DataFrame(rows); sim_metrics.to_csv(OUT/'S8B_WT_vs_shuffle_simulation_metrics.csv',index=False)
    # Plot combined absolute shuffled-pair prediction.
    fig,ax=plt.subplots(figsize=FIGSIZE)
    for cls,color,label,s,alpha in [('TAD/AHD shuffle',BLUE,'AHD/TAD shuffle',8,0.6),('RG/RGG shuffle',ORANGE,'RG/RGG shuffle',10,0.7)]:
        sub=df[df['shuffle_class'].eq(cls)]
        ax.scatter(sub['B22_corr1_log10'],sub['final_hybrid_prediction'],s=s,alpha=alpha,color=color,edgecolors='none',label=label)
    lo,hi=limits(df['B22_corr1_log10'],df['final_hybrid_prediction'])
    ax.plot([lo,hi],[lo,hi],lw=.8,color='black')
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('Simulated log10(−B22)')
    ax.set_ylabel('Predicted log10(−B22)')
    ax.legend(loc='upper left',frameon=False)
    mt=abs_metrics[(abs_metrics['subset']=='combined')&(abs_metrics['shuffle_class']=='TAD/AHD shuffle')].iloc[0]
    mr=abs_metrics[(abs_metrics['subset']=='combined')&(abs_metrics['shuffle_class']=='RG/RGG shuffle')].iloc[0]
    ax.text(.97,.03,f"TAD: n={int(mt['n'])}, r={mt['Pearson']:.2f}\nRGG: n={int(mr['n'])}, r={mr['Pearson']:.2f}",transform=ax.transAxes,ha='right',va='bottom',fontsize=7)
    clean(ax); fig.tight_layout(pad=.4)
    fig.savefig(OUT/'S8B_shuffle_absolute_prediction.pdf',bbox_inches='tight')
    fig.savefig(OUT/'S8B_shuffle_absolute_prediction.png',dpi=DPI,bbox_inches='tight')
    plt.close(fig)
    readme=(
        '# S8B absolute shuffled-pair prediction correction\n\n'
        'This directory replaces the primary S8B plot with simulated versus final-predicted log10(-B22) for the shuffled pair itself. '
        'The original delta_simulated versus delta_predicted perturbation analysis is retained as a secondary diagnostic table. '
        'No simulations, RDF integrations, features, model predictions, or other S8 panels were regenerated in this task.\n\n'
        f'Audit: `{json.dumps(audit, sort_keys=True)}`\n'
    )
    (OUT/'README.md').write_text(readme)
    (OUT/'S8B_shuffle_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    print(json.dumps(audit,indent=2,sort_keys=True))
    print(abs_metrics.to_string(index=False))
    print(delta_metrics.to_string(index=False))
    print(sim_metrics.to_string(index=False))

if __name__=='__main__': main()
