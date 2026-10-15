#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT=Path(__file__).resolve().parents[2]
PRED=ROOT/'inputs/predictor/final_predictions'
MODEL=ROOT/'inputs/predictor'
MAIN=ROOT/'outputs/manuscript/Figure_4'
SUPP=ROOT/'outputs/manuscript/Figure_S6'
SUBDIR=os.environ.get('FIG5_OUTPUT_SUBDIR')
if SUBDIR:
    MAIN=MAIN/SUBDIR
    SUPP=SUPP/SUBDIR
RASTERIZE_POINTS=os.environ.get('FIG5_RASTERIZE_POINTS','0') == '1'
PANEL_MODE=os.environ.get('FIG5_PANEL_MODE','all')
MAIN.mkdir(parents=True, exist_ok=True); SUPP.mkdir(parents=True, exist_ok=True)
DEV_OOF=PRED/'development_917_OOF_final_hybrid.csv'
TEST=PRED/'test_1794_final_hybrid_predictions.csv'
FULL=PRED/'full_landscape_final_hybrid_predictions.csv'
DEV_DIAG=MODEL/'development_oof_final_hybrid_predictions.csv'
BOOT=MODEL/'development_oof_final_hybrid_bootstrap.csv'
RBP=ROOT/'conventional_rbp_uniprot_ids.txt'
EXPECTED_HASH='48b7279b7e76260c04c1f42ee766d3f385f3d27c5ad61b99b2941d0d02b0d516'
LAMBDA=0.1368029406764465; CAP=0.16914558161466975; GATE_CENTER=40; GATE_SCALE=12
FIGSIZE=(2.4,2.3); DPI=int(os.environ.get('FIG5_DPI','600')); BLUE='#0072B2'; GREY='#656565'; ORANGE='#D55E00'; MAX_BG=100000
PANEL_F_FIGSIZE=(float(os.environ.get('FIG5_PANEL_F_WIDTH_IN', FIGSIZE[0])), float(os.environ.get('FIG5_PANEL_F_HEIGHT_IN', FIGSIZE[1])))
mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":7,"axes.labelsize":7,"axes.titlesize":7,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","axes.linewidth":0.6,"xtick.major.width":0.6,"ytick.major.width":0.6,"xtick.major.size":2.5,"ytick.major.size":2.5,"xtick.direction":"out","ytick.direction":"out"})

def clean(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(length=2.5,width=0.6,pad=1.5)

def save(fig, outdir, stem):
    paths=[]
    for ext in ['pdf','svg','png']:
        p=outdir/f'{stem}.{ext}'
        # dpi is intentionally passed for PDF/SVG too because Matplotlib uses it
        # for rasterized artists embedded inside vector containers.
        kw={'bbox_inches':'tight','dpi':DPI}
        fig.savefig(p, **kw); paths.append(str(p))
    plt.close(fig); return paths

def rbp_ids(): return {x.strip() for x in RBP.read_text().splitlines() if x.strip()} if RBP.exists() else set()
def rgg_uniprot(pair): return str(pair).split('__')[0].split('_')[0]
def add_rbp(df):
    ids=rbp_ids(); out=df.copy(); out['is_rbp']=out['pair_id'].map(lambda x: rgg_uniprot(x) in ids); return out

def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float); resid=y-p
    return {'n':int(len(y)),'R2':float(r2_score(y,p)),'RMSE':float(mean_squared_error(y,p)**0.5),'MAE':float(mean_absolute_error(y,p)),'Pearson':float(stats.pearsonr(y,p).statistic),'Spearman':float(stats.spearmanr(y,p).statistic),'mean_residual':float(np.mean(resid)),'median_residual':float(np.median(resid))}

def identity_limits(x,y):
    lo=float(np.nanmin([np.nanmin(x),np.nanmin(y)])); hi=float(np.nanmax([np.nanmax(x),np.nanmax(y)])); pad=0.04*(hi-lo); return lo-pad, hi+pad

def scatter_pred(df, xcol, ycol, outdir, stem, source_name, panel_label, color_by_rbp=True):
    df=add_rbp(df); df.to_csv(outdir/source_name, index=False)
    x=df[xcol].to_numpy(float); y=df[ycol].to_numpy(float); isr=df['is_rbp'].to_numpy(bool); m=metrics(x,y)
    fig,ax=plt.subplots(figsize=FIGSIZE, constrained_layout=True); fig.set_constrained_layout_pads(w_pad=0.02,h_pad=0.02,wspace=0,hspace=0)
    if color_by_rbp:
        ax.scatter(x[~isr],y[~isr],s=7,alpha=0.50,c=GREY,edgecolors='none',label='Non-RBP',rasterized=RASTERIZE_POINTS,zorder=1)
        ax.scatter(x[isr],y[isr],s=7,alpha=0.80,c=BLUE,edgecolors='none',label='RBP',rasterized=RASTERIZE_POINTS,zorder=1)
    else:
        ax.scatter(x,y,s=7,alpha=0.50,c=GREY,edgecolors='none',rasterized=RASTERIZE_POINTS,zorder=1)
    dmin,dmax=identity_limits(x,y); ax.plot([dmin,dmax],[dmin,dmax],lw=0.8,color='black',zorder=3); ax.set_xlim(dmin,dmax); ax.set_ylim(dmin,dmax); ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel(r"Simulated $y=\log_{10}(-B_{22})$"); ax.set_ylabel(r"Predicted $y=\log_{10}(-B_{22})$")
    ax.text(0.97,0.03,f"{panel_label}: r={m['Pearson']:.2f} (n={m['n']})\nR²={m['R2']:.2f}",ha='right',va='bottom',transform=ax.transAxes)
    
    if color_by_rbp:
        ax.legend(frameon=False,loc='upper left',handletextpad=0.6,borderpad=0.3)
    clean(ax)
    paths=save(fig,outdir,stem); return m,paths

def make_main():
    dev=pd.read_csv(DEV_OOF); test=pd.read_csv(TEST); full=pd.read_csv(FULL)
    assert len(dev)==917 and len(test)==1794 and len(full)==191386
    assert dev['model_manifest_sha256'].iloc[0]==EXPECTED_HASH and test['model_manifest_sha256'].iloc[0]==EXPECTED_HASH and full['model_manifest_sha256'].iloc[0]==EXPECTED_HASH
    if PANEL_MODE == 'F_only':
        d_m,d_paths={},[]
        e_m,e_paths={},[]
    else:
        d_src=dev[['pair_id','simulated_target','OOF_final_hybrid_prediction','minus_NCP','high_minus_NCP','model_manifest_sha256']].copy()
        d_m,d_paths=scatter_pred(d_src,'simulated_target','OOF_final_hybrid_prediction',MAIN,'Fig5D_development_OOF','Fig5D_development_OOF_final_hybrid_source.csv','Training', color_by_rbp=False)
        e_src=test[['pair_id','RGG_id','TAD_id','simulated_target','final_hybrid_prediction','minus_NCP','high_minus_NCP','is_rbp','model_manifest_sha256']].copy()
        # scatter_pred recomputes is_rbp from pair_id for consistency
        e_m,e_paths=scatter_pred(e_src.drop(columns=['is_rbp']),'simulated_target','final_hybrid_prediction',MAIN,'Fig5E_heldout_1794','Fig5E_test_1794_final_hybrid_source.csv','Held-Out', color_by_rbp=False)
    f_src=full[['pair_id','RGG_id','TAD_id','minus_NCP','final_hybrid_prediction','is_rbp','model_manifest_sha256']].copy()
    f_src.to_csv(MAIN/'Fig5F_full_landscape_final_hybrid_source.csv', index=False)

    x=f_src['minus_NCP'].to_numpy(float)
    y=f_src['final_hybrid_prediction'].to_numpy(float)
    pearson_all=float(stats.pearsonr(x, y).statistic)
    spearman_all=float(stats.spearmanr(x, y).statistic)
    rbp_src=f_src[f_src['is_rbp'].astype(bool)]
    non_src=f_src[~f_src['is_rbp'].astype(bool)]
    pearson_rbp=float(stats.pearsonr(rbp_src['minus_NCP'], rbp_src['final_hybrid_prediction']).statistic) if len(rbp_src)>2 else np.nan
    spearman_rbp=float(stats.spearmanr(rbp_src['minus_NCP'], rbp_src['final_hybrid_prediction']).statistic) if len(rbp_src)>2 else np.nan
    pearson_non=float(stats.pearsonr(non_src['minus_NCP'], non_src['final_hybrid_prediction']).statistic) if len(non_src)>2 else np.nan
    spearman_non=float(stats.spearmanr(non_src['minus_NCP'], non_src['final_hybrid_prediction']).statistic) if len(non_src)>2 else np.nan

    # Density colouring for the full 191,386-point landscape. Because −NCP is
    # discrete/integer-like, a 2D histogram gives a stable visual density without
    # hiding the actual points.
    xbins=np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 220)
    ybins=np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), 220)
    hist, xe, ye=np.histogram2d(x, y, bins=[xbins, ybins])
    xi=np.clip(np.searchsorted(xe, x, side='right')-1, 0, hist.shape[0]-1)
    yi=np.clip(np.searchsorted(ye, y, side='right')-1, 0, hist.shape[1]-1)
    density=hist[xi, yi]
    order=np.argsort(density)
    density_source=f_src[['pair_id','minus_NCP','final_hybrid_prediction','is_rbp']].copy()
    density_source['plot_density_bin_count']=density
    density_source.to_csv(MAIN/'Fig5F_density_colored_source.csv', index=False)

    slope, intercept, r_value, p_value, std_err=stats.linregress(x, y)
    fit_x=np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 300)
    fit_y=intercept+slope*fit_x
    pd.DataFrame([{'slope':slope,'intercept':intercept,'pearson_r':r_value,'p_value':p_value,'std_err':std_err,'n':len(f_src)}]).to_csv(MAIN/'Fig5F_minus_NCP_linear_fit.csv', index=False)

    fig,ax=plt.subplots(figsize=PANEL_F_FIGSIZE, constrained_layout=True); fig.set_constrained_layout_pads(w_pad=0.02,h_pad=0.02,wspace=0,hspace=0)
    sc=ax.scatter(x[order], y[order], c=np.log10(density[order]), cmap='viridis', s=4, alpha=0.75, edgecolors='none', zorder=1, rasterized=RASTERIZE_POINTS)
    xpad=0.04*(f_src.minus_NCP.max()-f_src.minus_NCP.min()); ypad=0.04*(f_src.final_hybrid_prediction.max()-f_src.final_hybrid_prediction.min())
    ax.set_xlim(float(f_src.minus_NCP.min()-xpad), float(f_src.minus_NCP.max()+xpad))
    ax.set_ylim(float(f_src.final_hybrid_prediction.min()-ypad), float(f_src.final_hybrid_prediction.max()+ypad))
    ax.set_xlabel('-NCP'); ax.set_ylabel(r"Predicted $y=\log_{10}(-B_{22})$")
    ax.text(0.97,0.03,f"r={pearson_all:.2f}, ρ={spearman_all:.2f}\nn={len(f_src):,}",ha='right',va='bottom',transform=ax.transAxes,fontsize=6)
    cbar=fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label('log10 local density', fontsize=6)
    cbar.ax.tick_params(labelsize=5, length=2, width=0.5, pad=1)
    clean(ax)
    f_paths=save(fig,MAIN,'Fig5F_full_landscape')
    f_m={'n':len(f_src),'prediction_min':float(f_src.final_hybrid_prediction.min()),'prediction_max':float(f_src.final_hybrid_prediction.max()),'minus_NCP_min':float(f_src.minus_NCP.min()),'minus_NCP_max':float(f_src.minus_NCP.max()),'pearson_minus_NCP_prediction_all':pearson_all,'spearman_minus_NCP_prediction_all':spearman_all,'linear_fit_slope':float(slope),'linear_fit_intercept':float(intercept),'RBP_n':int(len(rbp_src)),'RBP_pearson_minus_NCP_prediction':pearson_rbp,'RBP_spearman_minus_NCP_prediction':spearman_rbp,'non_RBP_n':int(len(non_src)),'non_RBP_pearson_minus_NCP_prediction':pearson_non,'non_RBP_spearman_minus_NCP_prediction':spearman_non}
    pd.DataFrame([f_m]).to_csv(MAIN/'Fig5F_minus_NCP_prediction_statistics.csv',index=False)
    return {'D':(d_m,d_paths),'E':(e_m,e_paths),'F':(f_m,f_paths)}

def bootstrap_ci(vals, seed=2026):
    vals=np.asarray(vals,float); obs=float(np.mean(vals));
    if len(vals)<2: return obs,obs,obs
    rng=np.random.default_rng(seed); bs=[np.mean(vals[rng.integers(0,len(vals),len(vals))]) for _ in range(5000)]
    return obs,float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))

def make_bins(df):
    low=df[df.minus_NCP<40]; high=df[df.minus_NCP>=40]
    low_edges=sorted(set(np.quantile(low.minus_NCP,[0,.2,.4,.6,.8,1])))
    edges=sorted(set([float(x) for x in low_edges if x<40]+[40.0,60.0,90.0,float(high.minus_NCP.max())+1e-9]))
    return pd.cut(df.minus_NCP,bins=edges,include_lowest=True,duplicates='drop').astype(str)

def binned(df,col):
    rows=[]
    for b,g in df.groupby('bin_assignment'):
        mean,lo,hi=bootstrap_ci(g[col].to_numpy())
        rows.append({'bin_assignment':b,'bin_center_minus_NCP':float(g.minus_NCP.median()),'bin_n':len(g),f'{col}_bin_mean':mean,f'{col}_bin_ci_2_5':lo,f'{col}_bin_ci_97_5':hi})
    return pd.DataFrame(rows).sort_values('bin_center_minus_NCP')

def plot_resid(df, summary, col, ylabel, stem, ylim, source_name):
    df.merge(summary,on='bin_assignment',how='left').to_csv(SUPP/source_name,index=False)
    fig,ax=plt.subplots(figsize=FIGSIZE)
    low=df[~df.high_minus_NCP]; high=df[df.high_minus_NCP]
    ax.scatter(low.minus_NCP,low[col],s=7,alpha=.50,color=GREY,edgecolors='none',zorder=1,label='−NCP < 40',rasterized=RASTERIZE_POINTS)
    ax.scatter(high.minus_NCP,high[col],s=7,alpha=.80,color=BLUE,edgecolors='none',zorder=2,label='−NCP ≥ 40',rasterized=RASTERIZE_POINTS)
    ax.plot(summary.bin_center_minus_NCP, summary[f'{col}_bin_mean'], 'o-', markersize=2.5, lw=.8, color=ORANGE, zorder=4)
    ax.axhline(0,lw=.8,color='black'); ax.axvline(40,lw=.8,color='black',ls='--'); ax.set_ylim(*ylim); ax.set_xlabel('−NCP'); ax.set_ylabel(ylabel); ax.legend(frameon=False,loc='upper left',handletextpad=.2,borderpad=.1); clean(ax)
    return save(fig,SUPP,stem)

def make_s5():
    df=pd.read_csv(DEV_DIAG); boot=pd.read_csv(BOOT)
    assert len(df)==917
    df['high_minus_NCP']=df.minus_NCP>=40; assert int(df.high_minus_NCP.sum())==93
    df['bin_assignment']=make_bins(df)
    base_sum=binned(df,'base_residual'); hyb_sum=binned(df,'hybrid_residual'); corr_sum=binned(df,'applied_residual_correction')
    vals=np.r_[df.base_residual.to_numpy(float), df.hybrid_residual.to_numpy(float)]; pad=.08*(vals.max()-vals.min()); ylim=(float(vals.min()-pad),float(vals.max()+pad))
    a_paths=plot_resid(df,base_sum,'base_residual','Base-component OOF residual','FigS5A_base_residual_vs_minusNCP',ylim,'FigS5A_source.csv')
    b_paths=plot_resid(df,hyb_sum,'hybrid_residual','Full-hybrid OOF residual','FigS5B_hybrid_residual_vs_minusNCP',ylim,'FigS5B_source.csv')
    # S5C final high-NCP two-point metric comparison
    high=df[df.high_minus_NCP]
    base_m=metrics(high.simulated_target, high.base_OOF_prediction); hyb_m=metrics(high.simulated_target, high.hybrid_OOF_prediction)
    get=lambda metric: boot[(boot.subset=='high_minus_NCP')&(boot.metric==metric)].iloc[0]
    rmse_ci=get('delta_RMSE'); bias_ci=get('delta_bias')
    csrc=pd.DataFrame([{'component':'Base component','n':base_m['n'],'RMSE':base_m['RMSE'],'R2':base_m['R2'],'MAE':base_m['MAE'],'bias':base_m['mean_residual']},{'component':'Full hybrid','n':hyb_m['n'],'RMSE':hyb_m['RMSE'],'R2':hyb_m['R2'],'MAE':hyb_m['MAE'],'bias':hyb_m['mean_residual']}])
    csrc['delta_RMSE_hybrid_minus_base']=hyb_m['RMSE']-base_m['RMSE']; csrc['delta_RMSE_ci_2_5']=float(rmse_ci.ci_2_5); csrc['delta_RMSE_ci_97_5']=float(rmse_ci.ci_97_5); csrc['delta_bias_hybrid_minus_base']=hyb_m['mean_residual']-base_m['mean_residual']; csrc['delta_bias_ci_2_5']=float(bias_ci.ci_2_5); csrc['delta_bias_ci_97_5']=float(bias_ci.ci_97_5)
    csrc.to_csv(SUPP/'FigS5C_source.csv',index=False)
    fig,ax=plt.subplots(figsize=FIGSIZE)
    ax.bar([0,1],[base_m['RMSE'],hyb_m['RMSE']],color=[GREY,BLUE],alpha=.85,edgecolor='none',width=.6)
    ax.set_xticks([0,1],['Base\ncomponent','Full\nhybrid']); ax.set_ylabel('High−NCP RMSE')
    ax.text(.98,.05,f"ΔRMSE = {hyb_m['RMSE']-base_m['RMSE']:.5f}\n95% CI [{rmse_ci.ci_2_5:.5f}, {rmse_ci.ci_97_5:.5f}]\nΔbias = {hyb_m['mean_residual']-base_m['mean_residual']:.5f}",transform=ax.transAxes,ha='right',va='bottom',fontsize=6)
    clean(ax); c_paths=save(fig,SUPP,'FigS5C_high_minusNCP_RMSE_ablation')
    d_paths=plot_resid(df,corr_sum,'applied_residual_correction','Hybrid − base prediction','FigS5D_applied_residual_correction',(float(-1.15*np.max(np.abs(df.applied_residual_correction))),float(1.15*np.max(np.abs(df.applied_residual_correction)))),'FigS5D_source.csv')
    return {'S5A':({'n':len(df)},a_paths),'S5B':({'n':len(df)},b_paths),'S5C':({'base':base_m,'hybrid':hyb_m,'delta_RMSE':hyb_m['RMSE']-base_m['RMSE'],'delta_bias':hyb_m['mean_residual']-base_m['mean_residual'],'delta_RMSE_CI':[float(rmse_ci.ci_2_5),float(rmse_ci.ci_97_5)],'delta_bias_CI':[float(bias_ci.ci_2_5),float(bias_ci.ci_97_5)]},c_paths),'S5D':({'n':len(df)},d_paths)}

def main():
    main_info=make_main(); s5_info={} if PANEL_MODE == 'F_only' else make_s5()
    audit=['FINAL FIGURE DATA AUDIT','',f'model manifest SHA256: {EXPECTED_HASH}',f'lambda_final: {LAMBDA}',f'cap: {CAP}',f'gate centre/scale: {GATE_CENTER}/{GATE_SCALE}','']
    audit += ['Main D source: development_917_OOF_final_hybrid.csv',json.dumps(main_info['D'][0],indent=2),'Main E source: test_1794_final_hybrid_predictions.csv',json.dumps(main_info['E'][0],indent=2),'Main F source: full_landscape_final_hybrid_predictions.csv',json.dumps(main_info['F'][0],indent=2)]
    audit += ['S5 A/B/D source: development_oof_final_hybrid_predictions.csv','S5C source: final development-only hybrid metrics/bootstrap',json.dumps({k:v[0] for k,v in s5_info.items()},indent=2),'No panel uses the older historical hybrid-prediction tables.','No model parameter was modified during figure generation.','The locked test set was used only for Panel E reporting.']
    (MODEL/'FINAL_FIGURE_DATA_AUDIT.txt').write_text('\n'.join(audit)+'\n')
    print('Updated final Figure 5 D/E/F and S5 A-D panels')
if __name__=='__main__': main()
