#!/usr/bin/env python3
"""Replica uncertainty for the fixed 345 predictor-shuffle comparisons (new S7E/F).

Read-only inputs: accepted predictor-control CSVs and completed Fig4 RDF pickles.
No model loading, feature calculation, simulation, or fitting.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'fig5_shuffle_uncertainty'
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = ROOT / 's8_extended_controls/S8B_absolute_shuffle_prediction/S8B_shuffle_source_audited.csv'
OLD_METRICS = SOURCE.with_name('S8B_delta_metrics_original_new_combined.csv')
RDF_FUNCTION = ROOT / 'fig5_final_model_validation_local/scripts/rdf_b22_functions.py'
RDF_DIR = ROOT.parent / 'fig4/controls_rdfs'
N_BOOT = 20_000
N_NULL = 20_000
SEED = 20261005
RUNS = [f'run_{i}' for i in range(1, 6)]
GROUPS = ['AHD', 'RG/RGG']
COLORS = {'AHD':'#0072B2', 'RG/RGG':'#D55E00'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(frame, name):
    frame.to_csv(OUT / name, index=False)

def point_metrics(x, y):
    x, y = np.asarray(x,float), np.asarray(y,float)
    return dict(n=len(x), Pearson=float(stats.pearsonr(x,y).statistic),
                Spearman=float(stats.spearmanr(x,y).statistic),
                RMSE=float(np.sqrt(np.mean((x-y)**2))), MAE=float(np.mean(np.abs(x-y))),
                mean_delta_sim=float(np.mean(y)), median_delta_sim=float(np.median(y)),
                median_abs_delta_sim=float(np.median(np.abs(y))), RMS_delta_sim=float(np.sqrt(np.mean(y*y))))

def corr_rows(x, y):
    x = np.asarray(x,float); y = np.asarray(y,float)
    xc=x-x.mean(axis=-1,keepdims=True); yc=y-y.mean(axis=-1,keepdims=True)
    den=np.sqrt(np.sum(xc*xc,axis=-1)*np.sum(yc*yc,axis=-1))
    return np.sum(xc*yc,axis=-1)/den

def reconstruct():
    src=pd.read_csv(SOURCE,float_precision='round_trip')
    if len(src)!=345 or not src.shuffle_pair_id.is_unique:
        raise ValueError('Expected exactly 345 unique predictor shuffle comparisons')
    src['shuffle_type']=src.shuffle_class.map({'TAD/AHD shuffle':'AHD','RG/RGG shuffle':'RG/RGG'})
    if src.shuffle_type.value_counts().to_dict()!={'AHD':230,'RG/RGG':115}:
        raise ValueError('Unexpected predictor shuffle class counts')
    spec=importlib.util.spec_from_file_location('accepted_rdf_b22',RDF_FUNCTION)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ids=sorted(set(src.shuffle_pair_id)|set(src.WT_parent_pair_id))
    values={}; problems=[]; inventory=[]; replica_rows=[]
    for name in ids:
        path=RDF_DIR/f'rdfs_{name}_20nm.pkl'
        try:
            rdf=pd.read_pickle(path)
            if rdf.shape!=(2,5) or set(rdf.columns)!=set(RUNS):
                raise ValueError(f'Expected five runs; shape={rdf.shape}, columns={list(rdf.columns)}')
            _, b22s, _=module.corrected_rdfs_b22(rdf,7.5,box_size=20,delta=2,rc_nm=4.0)
            v=np.array([b22s[run]['b22_corr1'] for run in RUNS],float)
            if not np.isfinite(v).all():raise ValueError('Non-finite replica B22')
            values[name]=v
            inventory.append(dict(pair_id=name,path=str(path.resolve()),sha256=sha(path),n_replicas=5))
            for i,x in enumerate(v,1):
                replica_rows.append(dict(pair_id=name,replica=i,B22=float(x),source_file=str(path.resolve())))
        except Exception as exc:
            problems.append(dict(pair_id=name,source_file=str(path.resolve()),problem=str(exc),
                                 affected_shuffle_pairs=';'.join(src.loc[(src.shuffle_pair_id==name)|(src.WT_parent_pair_id==name),'shuffle_pair_id'])))
    save(pd.DataFrame(problems,columns=['pair_id','source_file','problem','affected_shuffle_pairs']),'missing_replica_data.csv')
    save(pd.DataFrame(inventory),'rdf_input_inventory.csv')
    if problems:
        (OUT/'FIG5_SHUFFLE_DELTA_UNCERTAINTY_REPORT.md').write_text('# Reconstruction stopped\n\nExact five-replica data are unavailable for all 345 comparisons. See missing_replica_data.csv; no reduced-set analysis was performed.\n')
        raise RuntimeError(f'{len(problems)} systems failed exact replica reconstruction; see missing_replica_data.csv')
    rows=[]
    for r in src.itertuples(index=False):
        wr,wa=r.WT_parent_pair_id.split('__'); sr,sa=r.shuffle_pair_id.split('__')
        if not (wr.endswith('_WT') and wa.endswith('_WT')):raise ValueError(f'Non-WT parent: {r.WT_parent_pair_id}')
        if r.shuffle_type=='AHD':
            if wr!=sr or '_SHUF' not in sa or '_SHUF' in sr:raise ValueError(f'Partner orientation mismatch: {r.shuffle_pair_id}')
            if sa.split('_SHUF')[0]!=wa.removesuffix('_WT'):raise ValueError(f'AHD parent mismatch: {r.shuffle_pair_id}')
        else:
            if wa!=sa or '_SHUF' not in sr or '_SHUF' in sa:raise ValueError(f'Partner orientation mismatch: {r.shuffle_pair_id}')
            if sr.split('_SHUF')[0]!=wr.removesuffix('_WT'):raise ValueError(f'RGG parent mismatch: {r.shuffle_pair_id}')
        w=values[r.WT_parent_pair_id];s=values[r.shuffle_pair_id]
        mw,ms=float(w.mean()),float(s.mean())
        if mw>=0 or ms>=0:raise ValueError(f'Undefined point target: {r.shuffle_pair_id}')
        yw,ys=np.log10(-mw),np.log10(-ms)
        record=dict(shuffle_type=r.shuffle_type,WT_pair_id=r.WT_parent_pair_id,shuffle_pair_id=r.shuffle_pair_id,
                    WT_AHD_id=wa,WT_RGG_id=wr,shuffled_sequence_id=sa if r.shuffle_type=='AHD' else sr,
                    control_origin=r.control_origin,prediction_WT=r.predicted_WT_target,prediction_shuffle=r.predicted_shuffled_target,
                    delta_pred=r.predicted_shuffled_target-r.predicted_WT_target,mean_WT_B22=mw,mean_shuffle_B22=ms,
                    WT_target=yw,shuffle_target=ys,delta_sim=ys-yw,
                    WT_n_replicas=5,shuffle_n_replicas=5,
                    WT_RDF=str((RDF_DIR/f'rdfs_{r.WT_parent_pair_id}_20nm.pkl').resolve()),
                    shuffle_RDF=str((RDF_DIR/f'rdfs_{r.shuffle_pair_id}_20nm.pkl').resolve()),
                    mean_WT_B22_difference=mw-r.simulated_WT_B22,mean_shuffle_B22_difference=ms-r.simulated_shuffled_B22,
                    WT_target_difference=yw-r.simulated_WT_target,shuffle_target_difference=ys-r.simulated_shuffled_target,
                    delta_sim_difference=(ys-yw)-r.delta_simulated,
                    delta_pred_difference=(r.predicted_shuffled_target-r.predicted_WT_target)-r.delta_predicted)
        for i in range(5):record[f'WT_replica_{i+1}_B22']=w[i];record[f'shuffle_replica_{i+1}_B22']=s[i]
        rows.append(record)
    data=pd.DataFrame(rows)
    if data.groupby('WT_pair_id').prediction_WT.agg(lambda x: x.max()-x.min()).max()>1e-12:
        raise ValueError('Shared WT parent has inconsistent stored predictions')
    for col in ['mean_WT_B22_difference','mean_shuffle_B22_difference']:
        if data[col].abs().max()>1e-8:raise ValueError(f'Failed source reproduction: {col}')
    for col in ['WT_target_difference','shuffle_target_difference','delta_sim_difference','delta_pred_difference']:
        if data[col].abs().max()>1e-12:raise ValueError(f'Failed source reproduction: {col}')
    save(data,'predictor_shuffle_replica_data.csv')
    save(pd.DataFrame(replica_rows),'predictor_shuffle_replica_values_long.csv')
    parents=data.groupby('WT_pair_id',sort=True).agg(n_comparisons=('shuffle_pair_id','size'),
               n_AHD=('shuffle_type',lambda x:int((x=='AHD').sum())),
               n_RGG=('shuffle_type',lambda x:int((x=='RG/RGG').sum())),
               shuffle_pair_ids=('shuffle_pair_id',lambda x:';'.join(x))).reset_index()
    save(parents,'predictor_shuffle_parent_structure.csv')
    old=pd.read_csv(OLD_METRICS)
    current=[]
    for group in GROUPS+['all_exploratory']:
        d=data if group=='all_exploratory' else data[data.shuffle_type==group]
        m=point_metrics(d.delta_pred,d.delta_sim)
        m.update(shuffle_type=group,role='exploratory' if group=='all_exploratory' else 'class_specific')
        if group in GROUPS:
            previous=old[(old.subset=='combined')&(old.shuffle_class==('TAD/AHD shuffle' if group=='AHD' else 'RG/RGG shuffle'))].iloc[0]
            for col in ['Pearson','Spearman','RMSE','MAE']:
                m['previous_'+col]=previous[col];m[col+'_reproduction_difference']=m[col]-previous[col]
                if abs(m[col]-previous[col])>1e-12:raise ValueError(f'Current {group} {col} not reproduced')
        current.append(m)
    current=pd.DataFrame(current);save(current,'current_delta_analysis.csv')
    return data,parents,current,values

def bootstrap_targets(data,values,n,rng):
    """Resample each unique WT once per draw, then map its mean to its children."""
    parents=sorted(data.WT_pair_id.unique()); lookup={p:i for i,p in enumerate(parents)}
    wi=np.array([lookup[p] for p in data.WT_pair_id]); w=np.array([values[p] for p in parents])
    s=np.array([values[p] for p in data.shuffle_pair_id])
    if not data.shuffle_pair_id.is_unique:raise ValueError('Shuffle simulations must be unique')
    # Batch allocation is bounded, while storing all 20,000 target draws for audit.
    result=np.empty((n,len(data)))
    for start in range(0,n,1000):
        size=min(1000,n-start)
        widx=rng.integers(0,5,(size,len(w),5));sidx=rng.integers(0,5,(size,len(s),5))
        wm=w[np.arange(len(w))[None,:,None],widx].mean(axis=2)
        sm=s[np.arange(len(s))[None,:,None],sidx].mean(axis=2)
        if (wm>=0).any() or (sm>=0).any():
            raise RuntimeError('A bootstrap mean B22 is nonnegative. Stopping without dropping comparisons or substituting a target.')
        result[start:start+size]=np.log10(-sm)-np.log10(-wm[:,wi])
    return result

def calculate_uncertainty(data,current,values):
    seeds=np.random.SeedSequence(SEED).spawn(3)
    draws=bootstrap_targets(data,values,N_BOOT,np.random.default_rng(seeds[0]))
    null_targets=bootstrap_targets(data,values,N_NULL,np.random.default_rng(seeds[1]))
    rng=np.random.default_rng(seeds[2]);summary=[];null_rows=[];signals=[]
    boot_columns={'iteration':np.arange(1,N_BOOT+1)};null_columns={'iteration':np.arange(1,N_NULL+1)}
    source=data.copy()
    source['bootstrap_mean_delta_sim']=draws.mean(axis=0)
    source['bootstrap_SE_delta_sim']=draws.std(axis=0,ddof=1)
    source['bootstrap_q025_delta_sim']=np.quantile(draws,.025,axis=0)
    source['bootstrap_q975_delta_sim']=np.quantile(draws,.975,axis=0)
    for group in GROUPS:
        mask=data.shuffle_type.eq(group).to_numpy();d=data.loc[mask];x=d.delta_pred.to_numpy()
        y=draws[:,mask];ry=stats.rankdata(y,axis=1);rx=stats.rankdata(x)
        distributions={'Pearson':corr_rows(x,y),'Spearman':corr_rows(rx,ry),
                       'RMSE':np.sqrt(np.mean((y-x)**2,axis=1)),'MAE':np.mean(np.abs(y-x),axis=1)}
        old=current[current.shuffle_type==group].iloc[0]
        for metric,dist in distributions.items():
            if not np.isfinite(dist).all():raise RuntimeError('Undefined bootstrap metric; no silent omissions permitted')
            summary.append(dict(shuffle_type=group,n=len(d),metric=metric,point_estimate=old[metric],
                                bootstrap_median=float(np.median(dist)),q025=float(np.quantile(dist,.025)),q975=float(np.quantile(dist,.975)),
                                fraction_gt_zero=float(np.mean(dist>0)) if metric in ['Pearson','Spearman'] else np.nan,n_bootstrap=N_BOOT))
            boot_columns[f'{group}_{metric}']=dist
        # Each null iteration has a fresh shared-WT bootstrap vector and an independent within-class permutation.
        ny=null_targets[:,mask];nry=stats.rankdata(ny,axis=1)
        pearson=np.empty(N_NULL);spearman=np.empty(N_NULL)
        for start in range(0,N_NULL,1000):
            size=min(1000,N_NULL-start)
            perm=np.array([rng.permutation(len(x)) for _ in range(size)])
            pearson[start:start+size]=corr_rows(x[perm],ny[start:start+size])
            spearman[start:start+size]=corr_rows(rx[perm],nry[start:start+size])
        for metric,dist in [('Pearson',pearson),('Spearman',spearman)]:
            count=int((np.abs(dist)>=abs(old[metric])).sum())
            null_rows.append(dict(shuffle_type=group,n=len(d),metric=metric,observed_correlation=old[metric],
                                  empirical_two_sided_p=(1+count)/(1+N_NULL),n_extreme=count,n_permutations=N_NULL,
                                  null_median=float(np.median(dist)),null_q025=float(np.quantile(dist,.025)),null_q975=float(np.quantile(dist,.975))))
            null_columns[f'{group}_{metric}']=dist
        se=source.loc[mask,'bootstrap_SE_delta_sim'].to_numpy();delta=np.abs(d.delta_sim.to_numpy())
        signals.append(dict(shuffle_type=group,n=len(d),median_abs_delta_sim=float(np.median(delta)),
                            median_bootstrap_SE_delta_sim=float(np.median(se)),RMS_delta_sim=float(np.sqrt(np.mean(delta**2))),
                            fraction_abs_delta_gt_SE=float(np.mean(delta>se)),fraction_abs_delta_gt_2SE=float(np.mean(delta>2*se))))
    summary=pd.DataFrame(summary);null=pd.DataFrame(null_rows);signals=pd.DataFrame(signals)
    save(summary,'predictor_delta_uncertainty_summary.csv');save(null,'predictor_delta_null_test.csv')
    save(signals,'predictor_delta_signal_vs_uncertainty.csv');save(source,'FigS7EF_shuffle_delta_uncertainty_source.csv')
    # Compatibility source alias requested in the task; new panel labels are S7E/F.
    save(source,'FigS8B_shuffle_delta_uncertainty_source.csv')
    save(pd.DataFrame(boot_columns),'bootstrap_metric_draws.csv');save(pd.DataFrame(null_columns),'null_metric_draws.csv')
    np.savez_compressed(OUT/'bootstrap_delta_draws.npz',delta_sim=draws,shuffle_pair_id=data.shuffle_pair_id.to_numpy(str))
    return source,summary,null,signals,boot_columns

def draw_panel(ax,group,letter,data,current,summary):
    d=data[data.shuffle_type==group];m=current[current.shuffle_type==group].iloc[0]
    ci=summary[(summary.shuffle_type==group)&(summary.metric=='Pearson')].iloc[0]
    ax.scatter(d.delta_pred,d.delta_sim,color=COLORS[group],s=8 if group=='AHD' else 10,alpha=.6 if group=='AHD' else .7,edgecolors='none',rasterized=False)
    span=max(abs(data.delta_pred).max(),abs(data.delta_sim).max())*1.1
    ax.axhline(0,color='#888888',lw=.5,zorder=0);ax.axvline(0,color='#888888',lw=.5,zorder=0)
    ax.plot([-span,span],[-span,span],color='black',lw=.8,zorder=0)
    ax.set(xlim=(-span,span),ylim=(-span,span),xlabel=r'Predicted $\Delta$',ylabel=r'Simulated $\Delta$')
    ax.set_aspect('equal',adjustable='box')
    ax.set_title(f'{group} shuffles',fontsize=8,pad=5)
    ax.text(-.18,1.07,letter,transform=ax.transAxes,fontsize=10,fontweight='bold')
    ax.text(.97,.03,f'n = {len(d)}\nr = {m.Pearson:.3f}, ρ = {m.Spearman:.3f}\nr replica 95% interval:\n[{ci.q025:.3f}, {ci.q975:.3f}]',
            transform=ax.transAxes,ha='right',va='bottom',fontsize=6.2)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out',length=2.2,width=.6,pad=1.5)

def figures(data,current,summary):
    plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','font.size':8,
                         'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6})
    stems=[]
    for group,letter,tag in [('AHD','E','AHD'),('RG/RGG','F','RGG')]:
        fig,ax=plt.subplots(figsize=(2.4,2.3));draw_panel(ax,group,letter,data,current,summary)
        fig.subplots_adjust(left=.24,right=.98,bottom=.20,top=.87)
        stem=f'FigS7{letter}_{tag}_shuffle_delta_uncertainty'
        for extension in ['pdf','png','svg']:fig.savefig(OUT/f'{stem}.{extension}',dpi=600)
        plt.close(fig);stems.append(stem)
    fig,axes=plt.subplots(1,2,figsize=(5.1,2.5))
    for ax,group,letter in zip(axes,GROUPS,['E','F']):draw_panel(ax,group,letter,data,current,summary)
    fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.87,wspace=.45)
    for stem in ['FigS7EF_shuffle_delta_uncertainty_audit','FigS8B_shuffle_delta_uncertainty_audit']:
        for extension in ['pdf','png']:fig.savefig(OUT/f'{stem}.{extension}',dpi=600)
    plt.close(fig)
    return stems

def write_report(data,parents,current,summary,null,signals,provenance):
    def row(frame,group,metric=None):
        m=frame.shuffle_type.eq(group)
        if metric is not None:m &= frame.metric.eq(metric)
        return frame[m].iloc[0]
    supported={g:all(row(summary,g,k).q025>0 and row(null,g,k).empirical_two_sided_p<.05 for k in ['Pearson','Spearman']) for g in GROUPS}
    case='A' if all(supported.values()) else 'B' if supported['RG/RGG'] and not supported['AHD'] else 'C' if not any(supported.values()) else 'AHD_only'
    overall={'A':'retain with uncertainty qualification for both classes','B':'retain with qualification for RG/RGG only; AHD perturbations are not reliably resolved','C':'delete the positive perturbation-prediction claim and primary delta correlation analysis for both classes; retain absolute shuffled-strength validation','AHD_only':'retain with qualification for AHD only; RG/RGG perturbations are not reliably resolved'}[case]
    lines=['# Predictor-shuffle uncertainty: new Supplementary Fig. S7E/F','',
           f'**Recommendation: {overall}.**','',
           '## Reconstruction and unchanged sources',
           'The exact accepted predictor-control source contains 345 unique shuffle comparisons: 230 AHD and 115 RG/RGG. There are 115 unique WT parents; each contributes two AHD shuffles and one RG/RGG shuffle. Each comparison changes exactly one partner, retaining its exact WT partner. All 460 unique simulations have five finite replicas (2,300 unique replica values).',
           'Replica B22 values were recovered directly from the existing RDFs, using the existing Figure-5 `corrected_rdfs_b22(last=7.5, box_size=20, delta=2, rc_nm=4.0)` implementation and its `b22_corr1` values. The replica B22 values were averaged first and only then transformed as log10(-mean B22). No simulation, model inference, feature generation, model fitting, or calibration was performed.',
           'The Fig. 4 directory was accessed read-only for raw RDFs. No values or uncertainty estimates from the separate 300-comparison simulation-robustness analysis were reused. Former S6 numbering is now S7; these proposed additions are E and F. Existing panels and manuscript text were not edited.',
           f'Maximum absolute mean-B22 reproduction difference: {max(data.mean_WT_B22_difference.abs().max(),data.mean_shuffle_B22_difference.abs().max()):.6g}. Maximum transformed-delta reproduction difference: {data.delta_sim_difference.abs().max():.6g}.',
           '', '## Reproduced fixed-target analysis',
           '| Shuffle class | n | Pearson r | Spearman rho | RMSE | MAE |', '|---|---:|---:|---:|---:|---:|']
    for g in GROUPS+['all_exploratory']:
        m=row(current,g);lines.append(f'| {g} | {int(m.n)} | {m.Pearson:.6f} | {m.Spearman:.6f} | {m.RMSE:.6f} | {m.MAE:.6f} |')
    lines += ['', 'The class-specific Pearson, Spearman, RMSE and MAE reproduce the existing combined original+new statistics within 1e-12. Combined-class statistics are exploratory only.',
              '', '## Replica-bootstrap procedure and scope',
              f'{N_BOOT:,} bootstrap iterations; master seed {SEED}; NumPy SeedSequence child streams 0 (uncertainty), 1 (null target bootstrap), and 2 (null permutations). Within each draw, five WT replicas are resampled once per unique WT parent; that parent mean is shared across all three comparisons. Five replicas are independently resampled for each shuffled simulation. Predictions and the comparison set remain fixed. The bootstrap SD (ddof=1 across iterations) is reported as the Monte Carlo estimate of SE(delta_sim).',
              'All input replica B22 values are negative; every resampled mean is therefore negative, and no bootstrap observations, pairs or iterations require exclusion for the logarithm. Percentile intervals describe propagation of finite-replica uncertainty conditional on these fixed sequences; they do not include resampling sequences, model-development uncertainty, or generalization to other sequences.',
              '', '## Uncertainty-aware performance distributions',
              '| Class | Metric | Point estimate | Bootstrap median | 95% interval | Fraction > 0 |',
              '|---|---|---:|---:|---|---:|']
    for r in summary.itertuples(index=False):
        fraction=f'{r.fraction_gt_zero:.5f}' if np.isfinite(r.fraction_gt_zero) else 'not applicable'
        lines.append(f'| {r.shuffle_type} | {r.metric} | {r.point_estimate:.6f} | {r.bootstrap_median:.6f} | [{r.q025:.6f}, {r.q975:.6f}] | {fraction} |')
    lines += ['', 'Fractions above zero are descriptive bootstrap proportions, not p-values.',
              'These requested percentile propagation intervals need not contain the original point estimate. Resampling perturbs the noisy empirical target vector; the nonlinear correlations can be attenuated and the error metrics increased. For example, the RG/RGG original Pearson r is 0.616, whereas its bootstrap median is 0.553. These are not bias-corrected confidence intervals for the unobserved, noise-free correlation, nor a measurement-error correction. The evidence here is persistence of positive association under the specified replica uncertainty procedure.', '', '## Secondary permutation-null test',
              f'{N_NULL:,} iterations independently permuted delta_pred within each shuffle class and paired it with a fresh shared-WT bootstrap target vector. The observed reference is the original fixed-target correlation. Empirical two-sided p = (1 + count(abs(null) >= abs(observed))) / (1 + {N_NULL}).',
              'This requested comparison-level permutation assumes within-class exchangeability of predictor perturbations. Shared-WT target dependence is preserved, but comparison-level predictor permutations do not preserve predictor parent blocks. Accordingly these are secondary, nominal p-values, not proof of a causal or sequence-generalizable effect or a guarantee of conservativeness under every dependence structure. The four tests are reported without multiplicity adjustment.',
              '', '| Class | Metric | Observed | Two-sided permutation p | Null 95% interval |', '|---|---|---:|---:|---|']
    for r in null.itertuples(index=False):lines.append(f'| {r.shuffle_type} | {r.metric} | {r.observed_correlation:.6f} | {r.empirical_two_sided_p:.6g} | [{r.null_q025:.6f}, {r.null_q975:.6f}] |')
    lines += ['', '## Signal magnitude versus replica uncertainty',
              '| Class | Median abs(delta_sim) | Median bootstrap SE | RMS delta_sim | Fraction abs(delta)>SE | Fraction abs(delta)>2SE |',
              '|---|---:|---:|---:|---:|---:|']
    for r in signals.itertuples(index=False):lines.append(f'| {r.shuffle_type} | {r.median_abs_delta_sim:.6f} | {r.median_bootstrap_SE_delta_sim:.6f} | {r.RMS_delta_sim:.6f} | {r.fraction_abs_delta_gt_SE:.5f} | {r.fraction_abs_delta_gt_2SE:.5f} |')
    lines += ['', 'These thresholds are descriptive; they are not counts of statistically significant pairs.', '', '## Class-specific interpretation']
    for g in GROUPS:
        a=row(summary,g,'Pearson');b=row(summary,g,'Spearman')
        lines.append(f'- **{g}:** '+('Retain with qualification: both uncertainty-aware association intervals remain positive and both secondary nominal permutation tests support non-random association.' if supported[g] else 'Do not claim reliable recovery of pair-specific shuffle perturbations; retain any delta plot only as a diagnostic of this limitation.')+f' Pearson interval [{a.q025:.3f}, {a.q975:.3f}]; Spearman interval [{b.q025:.3f}, {b.q975:.3f}].')
    lines += ['',f'Operational reading of the requested framework: case {case}. “Supported” here requires both percentile lower bounds > 0 and both secondary nominal p < 0.05; this is a transparent report rule, not feature or model selection. Absolute shuffled-interaction-strength prediction is a separate analysis and remains unchanged.',
              '', '### Manuscript candidate text (alternatives; no manuscript edited)',
              '**A — if both classes remain supported:** With finite-replica uncertainty propagated through matched WT–shuffle comparisons, predicted perturbations retained positive associations with simulated perturbations in both shuffle classes. The result supports qualified recovery of shuffle effects, while individual small differences remain subject to sampling uncertainty.',
              '**B — if only RG/RGG remains supported:** Predicted shuffle-induced changes retained a positive association with simulation-derived changes for RG/RGG shuffles after propagating replica uncertainty. AHD shuffle perturbations were not resolved reliably, despite accurate absolute prediction of shuffled interaction strengths.',
              '**C — if neither remains sufficiently resolved:** The predictor accurately recovers absolute shuffled interaction strengths, but these data do not establish reliable recovery of the smaller WT-to-shuffle perturbations once finite-replica uncertainty is considered. We therefore limit the validation claim to absolute interaction strengths.',
              '', '## Files and reproducibility',
              'Run `python scripts/run_predictor_shuffle_uncertainty.py` from the fig5 clone. It refuses to overwrite a completed report. Use `--audit-only` to reconstruct and verify input data without bootstrapping.',
              f'- Exact predictor source: `{SOURCE}`',f'- Prior delta metrics: `{OLD_METRICS}`',
              f'- Existing RDF calculation: `{RDF_FUNCTION}`',f'- Read-only RDF folder: `{RDF_DIR.resolve()}`',
              '- Per-file hashes and software details: `analysis_provenance.json` and `rdf_input_inventory.csv`.',
              '- New individual audit panels: `FigS7E_AHD_shuffle_delta_uncertainty` and `FigS7F_RGG_shuffle_delta_uncertainty` (PDF/SVG/600-dpi PNG, 2.4 x 2.3 inches, vector points).',
              '- The combined audit is `FigS7EF_shuffle_delta_uncertainty_audit`. The requested `FigS8B_shuffle_delta_uncertainty_audit` and source filenames are compatibility aliases in this new directory only; no existing S8B is overwritten.',
              '- Full fixed-target metrics, uncertainty summaries, null results, parent structure, all 20,000 metric draws and compressed per-pair bootstrap delta draws are retained.']
    formatted=[]
    for line in lines:
        if line.startswith('- ') and formatted and formatted[-1] and not formatted[-1].startswith('- '):
            formatted.append('')
        formatted.append(line)
        if line.startswith('#'):formatted.append('')
    (OUT/'FIG5_SHUFFLE_DELTA_UNCERTAINTY_REPORT.md').write_text('\n'.join(formatted)+'\n')
    console=['TOTAL SHUFFLE COMPARISONS: 345','AHD SHUFFLES: 230','RG/RGG SHUFFLES: 115',
             f'UNIQUE WT PARENTS: {len(parents)}','ALL PAIRS HAVE 5 WT + 5 SHUFFLE REPLICAS: YES']
    for g,tag in [('AHD','AHD'),('RG/RGG','RGG')]:
        a=row(summary,g,'Pearson');b=row(summary,g,'Spearman');m=row(current,g);s=row(signals,g)
        console += [f'CURRENT {tag} PEARSON: {m.Pearson:.9f}',f'{tag} PEARSON BOOTSTRAP 95% CI: [{a.q025:.6f}, {a.q975:.6f}]',
                    f'{tag} SPEARMAN BOOTSTRAP 95% CI: [{b.q025:.6f}, {b.q975:.6f}]',
                    f'{tag} PERMUTATION P: Pearson={row(null,g,"Pearson").empirical_two_sided_p:.6g}; Spearman={row(null,g,"Spearman").empirical_two_sided_p:.6g}',
                    f'{tag} MEDIAN |DELTA_SIM|: {s.median_abs_delta_sim:.6f}',f'{tag} MEDIAN SE(DELTA_SIM): {s.median_bootstrap_SE_delta_sim:.6f}',
                    f'RECOMMENDATION FOR {tag} DELTA ANALYSIS: '+('retain with qualification' if supported[g] else 'do not claim recovery; diagnostic only')]
    console += [f'OVERALL MANUSCRIPT RECOMMENDATION: {overall}']
    (OUT/'console_summary.txt').write_text('\n'.join(console)+'\n')
    print('\n'.join(console),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audit-only',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'FIG5_SHUFFLE_DELTA_UNCERTAINTY_REPORT.md').exists():
        raise SystemExit('A report already exists; refusing to overwrite this completed analysis.')
    data,parents,current,values=reconstruct()
    print(f'Reconstruction passed: {len(data)} comparisons, {len(parents)} WT parents, five replicas for all systems.',flush=True)
    provenance=dict(seed=SEED,n_bootstrap=N_BOOT,n_permutations=N_NULL,python=sys.version,platform=platform.platform(),
                    numpy=np.__version__,pandas=pd.__version__,scipy=__import__('scipy').__version__,matplotlib=matplotlib.__version__,
                    generated_utc=datetime.now(timezone.utc).isoformat(),sources={str(p.resolve()):sha(p) for p in [SOURCE,OLD_METRICS,RDF_FUNCTION,Path(__file__)]},
                    raw_rdf_inventory_sha256=sha(OUT/'rdf_input_inventory.csv'),replica_calculation='corrected_rdfs_b22(last=7.5,box_size=20,delta=2,rc_nm=4.0): b22_corr1',
                    source_predictions_reused_unchanged=True,model_loaded=False,simulation_rerun=False,
                    all_replica_B22_negative=all((v<0).all() for v in values.values()))
    provenance['all_replica_B22_negative']=bool(provenance['all_replica_B22_negative'])
    (OUT/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    if args.audit_only:return
    print('Running 20,000 shared-WT bootstrap draws and 20,000 within-class permutation draws...',flush=True)
    source,summary,null,signals,draws=calculate_uncertainty(data,current,values)
    figures(data,current,summary)
    for path,original_hash in provenance['sources'].items():
        if sha(path)!=original_hash:raise RuntimeError(f'Input changed during analysis: {path}')
    write_report(data,parents,current,summary,null,signals,provenance)
    paths=sorted(str(p.resolve()) for p in OUT.iterdir() if p.is_file())
    (OUT/'output_files.txt').write_text('\n'.join(paths+[str((OUT/'output_files.txt').resolve())])+'\n')
    print('\nGENERATED OUTPUTS:\n'+'\n'.join(paths+[str((OUT/'output_files.txt').resolve())]))

if __name__=='__main__':main()
