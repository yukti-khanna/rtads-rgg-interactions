from pathlib import Path
import hashlib, json, platform, subprocess, sys, time, warnings
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, spearmanr
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/Users/yuktikhanna/rtads-rgg-interactions/fig5')
REPO = Path('/Users/yuktikhanna/idr-interaction-predict')
AUDIT = ROOT / 'reviewer_feature_redundancy_audit'
OUT = ROOT / 'reviewer_reduced_model_ablation'
TRAIN = ROOT / 'sim_pairs_pred_with_hybrid_sel_cols_with_old.csv'
HELD = ROOT / 'taken_out_preds_with_hybrid_021125_sel_cols.csv'
TARGET = 'B22_corr1_log10'
SEEDS = [0, 1, 2, 3, 4]
BOOTSTRAP_SEED = 20260911
N_BOOT = 2000
PARAMS = dict(activation='tanh', alpha=0.01, hidden_layer_sizes=(128,64,32,16),
              learning_rate_init=0.01, max_iter=500, solver='adam')
FEATURES = {
    'M1': ['Opposite_Charge_Number'],
    'M2': ['Opposite_Charge_Number','SCD_RGG','SCD_TAD'],
    'M3': ['mean_lambda_RGG','mean_lambda_TAD','SHD_RGG','SHD_TAD','RGG_length','TAD_length'],
    'M4': ['Opposite_Charge_Number','SCD_RGG','SCD_TAD','mean_lambda_RGG','mean_lambda_TAD','SHD_RGG','SHD_TAD','RGG_length','TAD_length'],
    'M1b': ['Normalized_OCN'],
}
LABELS = {'M1':'OCN only','M2':'OCN + SCD','M3':'Hydropathy + length','M4':'Combined non-redundant','M1b':'Normalized OCN only'}
PRIMARY = ['M1','M2','M3','M4']

assert not OUT.exists(), f'Refusing to overwrite {OUT}'
assert AUDIT.exists()
preexisting = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file()}
input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [TRAIN,HELD]}
train = pd.read_csv(TRAIN, float_precision='round_trip')
held = pd.read_csv(HELD, float_precision='round_trip')
assert train.shape[0] == 917 and held.shape[0] == 1794
assert TARGET in train and TARGET in held
assert np.isfinite(train[TARGET]).sum() == 917 and np.isfinite(held[TARGET]).sum() == 1794
assert train['RGG_length'].gt(0).all() and train['TAD_length'].gt(0).all()
assert held['RGG_length'].gt(0).all() and held['TAD_length'].gt(0).all()
for df in [train, held]:
    df['mean_lambda_RGG'] = df['lambda_RGG'] / df['RGG_length']
    df['mean_lambda_TAD'] = df['lambda_TAD'] / df['TAD_length']
all_features = sorted(set(sum(FEATURES.values(), [])))
for df in [train, held]:
    assert np.isfinite(df[all_features + [TARGET]].to_numpy(dtype=float)).all()
assert set(train['rgg']).isdisjoint(held['rgg']) and set(train['tad']).isdisjoint(held['tad'])
assert held['rgg'].nunique() == 78 and held['tad'].nunique() == 23
assert len(held[['rgg','tad']].drop_duplicates()) == 78*23

OUT.mkdir()
def save_csv(name, df):
    path=OUT/name
    assert not path.exists()
    with path.open('x') as handle: df.to_csv(handle,index=False,float_format='%.17g')
def save_text(name, text):
    path=OUT/name
    assert not path.exists()
    with path.open('x') as handle: handle.write(text)

feature_rows=[]
for model, cols in FEATURES.items():
    for order,col in enumerate(cols,1):
        feature_rows.append(dict(model=model,model_label=LABELS[model],primary_model=model in PRIMARY,
                                 feature_order=order,feature=col,definition={
            'Opposite_Charge_Number':'-charge_RGG * charge_TAD',
            'Normalized_OCN':'Opposite_Charge_Number / Total_Length',
            'mean_lambda_RGG':'lambda_RGG / RGG_length',
            'mean_lambda_TAD':'lambda_TAD / TAD_length',
        }.get(col,'stored historical Figure 5 value; see Task 1 audit')))
save_csv('feature_sets.csv',pd.DataFrame(feature_rows))

derived=[]
for dataset,df in [('training',train),('heldout',held)]:
    for side in ['RGG','TAD']:
        calc=df[f'lambda_{side}']/df[f'{side}_length']; stored_avg_component=calc
        derived.append(dict(dataset=dataset,derived_feature=f'mean_lambda_{side}',formula=f'lambda_{side} / {side}_length',
                            n=len(df),n_zero_length=int((df[f'{side}_length']==0).sum()),n_missing=int(df[f'mean_lambda_{side}'].isna().sum()),
                            n_infinite=int(np.isinf(df[f'mean_lambda_{side}']).sum()),minimum=df[f'mean_lambda_{side}'].min(),maximum=df[f'mean_lambda_{side}'].max(),
                            verification='all lengths positive; direct division used; original lambda column preserved'))
    expected=(df.mean_lambda_RGG+df.mean_lambda_TAD)/2
    derived.append(dict(dataset=dataset,derived_feature='Average_lambda cross-check',formula='(mean_lambda_RGG + mean_lambda_TAD)/2 compared with stored Average_lambda',
                        n=len(df),n_zero_length=0,n_missing=0,n_infinite=0,minimum=np.abs(expected-df.Average_lambda).min(),maximum=np.abs(expected-df.Average_lambda).max(),
                        verification='maximum absolute discrepancy; reflects stored decimal precision'))
save_csv('derived_feature_verification.csv',pd.DataFrame(derived))

def metrics(y,p):
    return dict(n=len(y),pearson_r=pearsonr(y,p).statistic,spearman_rho=spearmanr(y,p).statistic,
                r2=r2_score(y,p),rmse=mean_squared_error(y,p)**0.5,mae=mean_absolute_error(y,p))

pred_train={}; pred_held={}; training_log=[]; model_details=[]
for model,cols in FEATURES.items():
    Xtr=train[cols].to_numpy(float); Xh=held[cols].to_numpy(float)
    scaler=StandardScaler().fit(Xtr)
    pt=[];ph=[]
    for seed in SEEDS:
        estimator=MLPRegressor(random_state=seed,**PARAMS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            estimator.fit(scaler.transform(Xtr),train[TARGET].to_numpy(float))
        pt.append(estimator.predict(scaler.transform(Xtr)))
        ph.append(estimator.predict(scaler.transform(Xh)))
        training_log.append(dict(model=model,seed=seed,n_iter=estimator.n_iter_,loss=estimator.loss_,
                                 converged_before_max_iter=estimator.n_iter_<PARAMS['max_iter'],
                                 warnings=' | '.join(str(w.message) for w in caught)))
    pred_train[model]=np.mean(pt,axis=0);pred_held[model]=np.mean(ph,axis=0)
    model_details.append(dict(model=model,n_features=len(cols),n_training_rows=len(Xtr),n_heldout_rows=len(Xh),
                              rows_dropped_training=0,rows_dropped_heldout=0,scaler_fit_rows=len(Xtr),
                              scaler_mean=json.dumps(scaler.mean_.tolist()),scaler_scale=json.dumps(scaler.scale_.tolist())))
save_csv('model_training_diagnostics.csv',pd.DataFrame(training_log))
save_csv('model_scaler_metadata.csv',pd.DataFrame(model_details))

mt=[];mh=[]
for model in FEATURES:
    mt.append(dict(model=model,model_label=LABELS[model],primary_model=model in PRIMARY,**metrics(train[TARGET],pred_train[model])))
    mh.append(dict(model=model,model_label=LABELS[model],primary_model=model in PRIMARY,**metrics(held[TARGET],pred_held[model])))
mt=pd.DataFrame(mt);mh=pd.DataFrame(mh)
save_csv('model_metrics_training.csv',mt);save_csv('model_metrics_heldout.csv',mh)
base=mh.set_index('model').loc['M1']
diff=[]
for model in ['M2','M3','M4','M1b']:
    row=mh.set_index('model').loc[model]
    diff.append(dict(model=model,model_label=LABELS[model],reference_model='M1',delta_r2=row.r2-base.r2,
                     delta_rmse=row.rmse-base.rmse,delta_mae=row.mae-base.mae,
                     improvement_rmse=base.rmse-row.rmse,improvement_mae=base.mae-row.mae))
save_csv('model_metric_differences_vs_ocn.csv',pd.DataFrame(diff))

predcols=['name','rgg','tad',TARGET,'mean_lambda_RGG','mean_lambda_TAD']
ptr=train[predcols].copy();phd=held[predcols].copy()
for model in FEATURES:
    ptr[f'prediction_{model}']=pred_train[model];ptr[f'residual_{model}']=train[TARGET]-pred_train[model]
    phd[f'prediction_{model}']=pred_held[model];phd[f'residual_{model}']=held[TARGET]-pred_held[model]
save_csv('predictions_training.csv',ptr);save_csv('predictions_heldout.csv',phd)

# Arrange complete matrix; index arrays preserve multiplicity when IDs are sampled with replacement.
rgg_ids=pd.Index(sorted(held.rgg.unique()));tad_ids=pd.Index(sorted(held.tad.unique()))
def matrix(values):
    tmp=pd.DataFrame({'rgg':held.rgg,'tad':held.tad,'v':values})
    return tmp.pivot(index='rgg',columns='tad',values='v').loc[rgg_ids,tad_ids].to_numpy(float)
y_matrix=matrix(held[TARGET].to_numpy(float));pred_matrices={m:matrix(p) for m,p in pred_held.items()}
rng=np.random.default_rng(BOOTSTRAP_SEED);boot=[]
for replicate in range(N_BOOT):
    ri=rng.integers(0,len(rgg_ids),len(rgg_ids));ti=rng.integers(0,len(tad_ids),len(tad_ids))
    y=y_matrix[np.ix_(ri,ti)].ravel()
    rep_metrics={m:metrics(y,pm[np.ix_(ri,ti)].ravel()) for m,pm in pred_matrices.items()}
    b=rep_metrics['M1']
    for model,m in rep_metrics.items():
        boot.append(dict(replicate=replicate,model=model,pearson_r=m['pearson_r'],r2=m['r2'],rmse=m['rmse'],
                         delta_r2_vs_M1=m['r2']-b['r2'],delta_rmse_vs_M1=m['rmse']-b['rmse']))
boot=pd.DataFrame(boot);save_csv('bootstrap_metrics.csv',boot)
bs=[]
for model in FEATURES:
    sub=boot[boot.model==model];point=mh.set_index('model').loc[model]
    for metric in ['pearson_r','r2','rmse','delta_r2_vs_M1','delta_rmse_vs_M1']:
        vals=sub[metric]; point_value=0.0 if model=='M1' and metric.startswith('delta_') else (point[metric] if metric in point else (point.r2-base.r2 if metric=='delta_r2_vs_M1' else point.rmse-base.rmse))
        lo,hi=np.percentile(vals,[2.5,97.5])
        bs.append(dict(model=model,model_label=LABELS[model],metric=metric,point_estimate=point_value,
                       bootstrap_median=vals.median(),ci95_low=lo,ci95_high=hi,n_bootstrap=N_BOOT,
                       interval_excludes_zero=bool(lo>0 or hi<0) if metric.startswith('delta_') else np.nan))
bs=pd.DataFrame(bs);save_csv('bootstrap_summary.csv',bs)

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors={'M1':'#0072B2','M2':'#E69F00','M3':'#009E73','M4':'#CC79A7','M1b':'#777777'}
y=held[TARGET].to_numpy(float);lo=min(y.min(),*(pred_held[m].min() for m in PRIMARY));hi=max(y.max(),*(pred_held[m].max() for m in PRIMARY));pad=.04*(hi-lo);lims=(lo-pad,hi+pad)
fig,axes=plt.subplots(2,2,figsize=(9,8),sharex=True,sharey=True)
for ax,model in zip(axes.flat,PRIMARY):
    p=pred_held[model];m=mh.set_index('model').loc[model]
    ax.scatter(y,p,s=8,alpha=.32,color=colors[model],edgecolors='none',rasterized=True)
    ax.plot(lims,lims,'k--',lw=1);ax.set_xlim(lims);ax.set_ylim(lims);ax.set_aspect('equal','box')
    ax.set_title(f'{model}: {LABELS[model]}',loc='left',fontweight='bold')
    ax.text(.04,.96,f'Pearson r = {m.pearson_r:.3f}\nR² = {m.r2:.3f}\nRMSE = {m.rmse:.3f}\nn = {int(m.n):,}',transform=ax.transAxes,va='top')
for ax in axes[-1,:]:ax.set_xlabel('Simulated B22_corr1_log10')
for ax in axes[:,0]:ax.set_ylabel('Ensemble-predicted B22_corr1_log10')
fig.suptitle('Independent sequence-exclusive held-out simulations',fontsize=14)
fig.tight_layout(rect=[0,0,1,.96])
for ext in ['pdf','png']:fig.savefig(OUT/f'heldout_predicted_vs_simulated.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)

def interval(model,metric):
    return bs[(bs.model==model)&(bs.metric==metric)].iloc[0]
x=np.arange(4);vals=[mh.set_index('model').loc[m].r2 for m in PRIMARY];cis=[interval(m,'r2') for m in PRIMARY]
err=np.array([[v-c.ci95_low for v,c in zip(vals,cis)],[c.ci95_high-v for v,c in zip(vals,cis)]])
fig,ax=plt.subplots(figsize=(7,4.8));bars=ax.bar(x,vals,color=[colors[m] for m in PRIMARY],width=.7)
ax.errorbar(x,vals,yerr=err,fmt='none',ecolor='black',capsize=4,lw=1.2);ax.axhline(0,color='black',lw=.8)
ax.set_xticks(x,[f'{m}\n{LABELS[m]}' for m in PRIMARY]);ax.set_ylabel('Held-out R²');ax.set_title('Base-MLP ablation performance\n95% two-axis matrix-bootstrap intervals')
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v,f'{v:.3f}',ha='center',va='bottom' if v>=0 else 'top')
fig.tight_layout();
for ext in ['pdf','png']:fig.savefig(OUT/f'ablation_performance.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
save_csv('ablation_performance_table.csv',pd.DataFrame([dict(model=m,model_label=LABELS[m],heldout_r2=v,ci95_low=c.ci95_low,ci95_high=c.ci95_high) for m,v,c in zip(PRIMARY,vals,cis)]))

models=['M2','M3','M4'];vals=[mh.set_index('model').loc[m].r2-base.r2 for m in models];cis=[interval(m,'delta_r2_vs_M1') for m in models]
err=np.array([[v-c.ci95_low for v,c in zip(vals,cis)],[c.ci95_high-v for v,c in zip(vals,cis)]])
fig,ax=plt.subplots(figsize=(6.5,4.8));bars=ax.bar(np.arange(3),vals,color=[colors[m] for m in models],width=.68)
ax.errorbar(np.arange(3),vals,yerr=err,fmt='none',ecolor='black',capsize=4,lw=1.2);ax.axhline(0,color='black',lw=1)
ax.set_xticks(np.arange(3),[f'{m}\n{LABELS[m]}' for m in models]);ax.set_ylabel('ΔR² relative to M1 OCN-only');ax.set_title('Incremental held-out performance\n95% two-axis matrix-bootstrap intervals')
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v,f'{v:+.3f}',ha='center',va='bottom' if v>=0 else 'top')
fig.tight_layout();
for ext in ['pdf','png']:fig.savefig(OUT/f'delta_performance_vs_ocn.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)

diagnostics=['Opposite_Charge_Number','SCD_TAD','SCD_RGG','mean_lambda_TAD','mean_lambda_RGG','TAD_length','RGG_length']
fig,axes=plt.subplots(len(diagnostics),2,figsize=(10,18))
for i,f in enumerate(diagnostics):
    for j,model in enumerate(['M1','M4']):
        ax=axes[i,j];res=y-pred_held[model];xv=held[f].to_numpy(float);r=pearsonr(xv,res).statistic;rho=spearmanr(xv,res).statistic
        ax.scatter(xv,res,s=7,alpha=.25,color=colors[model],edgecolors='none',rasterized=True);ax.axhline(0,color='black',lw=.8)
        ax.set_xlabel(f);ax.set_ylabel(f'{model} residual (simulated − predicted)');ax.text(.98,.95,f'r={r:.3f}\nρ={rho:.3f}',ha='right',va='top',transform=ax.transAxes)
        if i==0:ax.set_title(f'{model}: {LABELS[model]}',fontweight='bold')
fig.suptitle('Held-out residual diagnostics (descriptive; no causal interpretation)',fontsize=14)
fig.tight_layout(rect=[0,0,1,.98])
for ext in ['pdf','png']:fig.savefig(OUT/f'residual_diagnostics.{ext}',dpi=240,bbox_inches='tight')
plt.close(fig)

summary_lines=[]
for model in FEATURES:
    m=mh.set_index('model').loc[model]; cr=interval(model,'r2'); cp=interval(model,'pearson_r'); cm=interval(model,'rmse')
    summary_lines.append(f'{model} ({LABELS[model]}): n={int(m.n)}, Pearson r={m.pearson_r:.6f} [95% matrix-bootstrap CI {cp.ci95_low:.6f}, {cp.ci95_high:.6f}], Spearman rho={m.spearman_rho:.6f}, R2={m.r2:.6f} [CI {cr.ci95_low:.6f}, {cr.ci95_high:.6f}], RMSE={m.rmse:.6f} [CI {cm.ci95_low:.6f}, {cm.ci95_high:.6f}], MAE={m.mae:.6f}.')
def comp(model):
    d=pd.DataFrame(diff).set_index('model').loc[model]; dr=interval(model,'delta_r2_vs_M1'); dm=interval(model,'delta_rmse_vs_M1')
    supported=(dr.ci95_low>0 and dm.ci95_high<0)
    return f'{model} vs M1: ΔR2={d.delta_r2:+.6f} [CI {dr.ci95_low:+.6f}, {dr.ci95_high:+.6f}]; ΔRMSE={d.delta_rmse:+.6f} [CI {dm.ci95_low:+.6f}, {dm.ci95_high:+.6f}]; ΔMAE={d.delta_mae:+.6f}. Both ΔR2 improvement and RMSE reduction supported={supported}.'
m1=mh.set_index('model').loc['M1'];m2=mh.set_index('model').loc['M2'];m3=mh.set_index('model').loc['M3'];m4=mh.set_index('model').loc['M4'];m1b=mh.set_index('model').loc['M1b']
dr2=interval('M2','delta_r2_vs_M1');dm2=interval('M2','delta_rmse_vs_M1');dr4=interval('M4','delta_r2_vs_M1');dm4=interval('M4','delta_rmse_vs_M1')
def support(dr,dm):return dr.ci95_low>0 and dm.ci95_high<0
if m4.r2-m1.r2>=.10: classification='substantial additional information beyond bulk electrostatics'
elif support(dr4,dm4) and m4.r2>m1.r2:classification='bulk electrostatics plus a small reproducible contribution from other sequence properties'
else:classification='almost entirely bulk electrostatics within the resolution of this ablation'
summary=f'''FIGURE 5 TASK 2: REDUCED-MODEL BASE-MLP ABLATION

Scope and guardrails
Only the stored 917-pair Figure 5 feature table was used for fitting and StandardScaler estimation. The 1,794-pair 78-RGG x 23-TAD sequence-exclusive matrix was used once as external evaluation data after fitting. No held-out result selected features, hyperparameters, seeds, thresholds, or models. No residual correction or OCN-dependent gate was used. No B22 or feature table was regenerated, and the unsimulated landscape was not read. Each feature set used five MLPRegressors with seeds 0-4 and predictions were averaged. Architecture: {PARAMS}. Target: {TARGET}; finite values 917/917 training and 1,794/1,794 held out. All models used every row; zero rows dropped.

Held-out results (primary)
{chr(10).join(summary_lines)}

Differences from OCN-only
{chr(10).join(comp(m) for m in ['M2','M3','M4','M1b'])}

Answers to reviewer-focused questions
1. OCN alone explains R2={m1.r2:.6f} of held-out response variance ({100*m1.r2:.1f}% under the ordinary coefficient-of-determination definition). Pearson correlation alone must not be described as explained variance.

2. Charge patterning beyond OCN: M2 changes held-out R2 by {m2.r2-m1.r2:+.6f}, RMSE by {m2.rmse-m1.rmse:+.6f}, and MAE by {m2.mae-m1.mae:+.6f}. Matrix-bootstrap support using joint criteria (ΔR2 CI entirely above zero and ΔRMSE CI entirely below zero): {support(dr2,dm2)}. SCD remains an electrostatic patterning descriptor.

3. Hydropathy/stickiness/patterning/length without bulk charge: M3 achieves R2={m3.r2:.6f}, Pearson r={m3.pearson_r:.6f}, RMSE={m3.rmse:.6f}, and MAE={m3.mae:.6f}. This establishes predictive capacity without the excluded bulk-charge features; it does not establish causality or complete electrostatic independence of all physics represented by simulations.

4-5. Combined model: M4 changes R2 by {m4.r2-m1.r2:+.6f}, RMSE by {m4.rmse-m1.rmse:+.6f}, and MAE by {m4.mae-m1.mae:+.6f}. Matrix-bootstrap joint support: {support(dr4,dm4)}. Magnitudes and intervals, rather than a binary significance label, are the primary interpretation.

6. Raw versus normalized OCN: raw M1 R2={m1.r2:.6f}, RMSE={m1.rmse:.6f}; normalized M1b R2={m1b.r2:.6f}, RMSE={m1b.rmse:.6f}. M1b minus M1: ΔR2={m1b.r2-m1.r2:+.6f}, ΔRMSE={m1b.rmse-m1.rmse:+.6f}. This is a pre-specified sensitivity comparison, not model selection.

7. Conservative classification: {classification}. This classification uses a transparent descriptive threshold of ΔR2 >= 0.10 for “substantial”; that threshold was not supplied by the reviewer and is not a formal hypothesis test. The defensible prose should emphasize the observed point differences and two-axis intervals. The analysis can test whether the current predictor is “just recovering charge complementarity,” but it cannot identify mechanisms or make electrostatic dominance novel.

Matrix-aware uncertainty
For each of {N_BOOT:,} replicates (seed {BOOTSTRAP_SEED}), 78 RGG IDs were sampled with replacement and 23 TAD IDs were independently sampled with replacement. The 78x23 indexed Cartesian cells were then flattened. Repeated sampled IDs were retained with full multiplicity through repeated row/column indices; nothing was deduplicated. Every model was evaluated on the same cells within a replicate, enabling paired differences from M1. Percentile 2.5th and 97.5th quantiles form the reported 95% intervals. This two-axis cluster bootstrap reflects dependence induced by both shared RGG and shared TAD partners. It does not correct for model-training uncertainty, sequence homology, or the deliberate composition of the 78x23 panel. No naive pair-level bootstrap was used.

Interpretive limits
Training-set scores are apparent fit scores, not cross-validated estimates. The ensemble is fit once per seed and feature set according to the pre-specified design; confidence intervals resample only held-out matrix axes. MLP optimization is stochastic but fixed by seeds. Residual correlations are descriptive diagnostics and do not imply causality. The historical SCD/SHD formulas remain limited by Task 1 provenance; this task uses stored audited values. Derived mean lambdas are transparent ratios and leave original columns unchanged.
'''
save_text('analysis_summary.txt',summary)

commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C',str(REPO),'status','--short'],text=True).strip()
audit_files=['audit_summary.txt','feature_definitions.csv','implementation_discrepancies.csv','exact_relationships.csv','feature_groups.csv','selected10_correlations.csv']
provenance=f'''Figure 5 Task 2 reduced-model ablation provenance
Repository: https://github.com/yukti-khanna/idr-interaction-predict
Local repository: {REPO}
Commit SHA: {commit}
Working tree status before/after analysis: {status or '(clean for tracked files)'}

Inputs
{TRAIN}\n  bytes={TRAIN.stat().st_size}\n  SHA256={input_hashes[str(TRAIN)]}\n  dimensions={train.shape[0]} rows x {train.shape[1]-2} original columns (plus 2 analysis-only derived columns in memory)
{HELD}\n  bytes={HELD.stat().st_size}\n  SHA256={input_hashes[str(HELD)]}\n  dimensions={held.shape[0]} rows x {held.shape[1]-2} original columns (plus 2 analysis-only derived columns in memory)
Task 1 audit used: {AUDIT}
Task 1 files inspected: {', '.join(audit_files)}
Historical model scripts inspected: {REPO/'scripts/mlp_with_plateau_residual_calibrated.py'}; {REPO/'scripts/mlp_feature_pruning.py'}

Runtime
Python executable: {sys.executable}
Python: {sys.version}
Platform: {platform.platform()}
numpy={np.__version__}; pandas={pd.__version__}; scipy={scipy.__version__}; scikit-learn={sklearn.__version__}; matplotlib={matplotlib.__version__}

Model specification
Feature lists/order: feature_sets.csv
Seeds: {SEEDS}
Hyperparameters: {PARAMS}
Ensemble: unweighted arithmetic mean of five predictions
Scaler: one StandardScaler per feature set, fit on all 917 training rows only
Target: {TARGET}; no alternate target transformation
Rows: every model used 917 training and 1,794 held-out; zero rows dropped
Bootstrap: two-axis RGG/TAD matrix bootstrap, seed={BOOTSTRAP_SEED}, replicates={N_BOOT}; multiplicity preserved
No residual model, gate, early model selection, feature selection, held-out tuning, permutation importance, B22 generation, feature regeneration, or unsimulated predictions.
'''
save_text('input_provenance.txt',provenance)

# Scope-preservation and output validation.
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in preexisting.items())
required=['input_provenance.txt','feature_sets.csv','derived_feature_verification.csv','model_metrics_training.csv','model_metrics_heldout.csv','model_metric_differences_vs_ocn.csv','bootstrap_metrics.csv','bootstrap_summary.csv','predictions_training.csv','predictions_heldout.csv','heldout_predicted_vs_simulated.pdf','heldout_predicted_vs_simulated.png','ablation_performance.pdf','ablation_performance.png','delta_performance_vs_ocn.pdf','delta_performance_vs_ocn.png','residual_diagnostics.pdf','residual_diagnostics.png','analysis_summary.txt']
assert all((OUT/f).is_file() and (OUT/f).stat().st_size>0 for f in required)
assert len(boot)==N_BOOT*len(FEATURES)
verification=f'''PASS: all {len(preexisting)} pre-existing Figure 5 files retain their original SHA256 hashes.\nPASS: input CSV hashes match pre-run values.\nPASS: target finite 917/917 training and 1794/1794 held-out.\nPASS: all requested features finite and all lengths positive; zero rows dropped.\nPASS: held-out IDs are disjoint and form the complete 78 x 23 Cartesian matrix.\nPASS: {N_BOOT} two-axis bootstrap replicates x {len(FEATURES)} models; multiplicity preserved by repeated row/column indexing.\nPASS: all 19 user-requested outputs exist and are non-empty.\n'''
save_text('verification.txt',verification)
print(summary)
print(verification)
