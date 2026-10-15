#!/usr/bin/env python3
"""Assemble publication-facing Figure 5, Figures S10-S11 and Table S11.

Panel 5A is intentionally a placeholder for the author's NMR spectra.
This script consumes only curated/current fig6 results and never imports Excel.
"""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parent
REV=ROOT/'revision_analysis'
OUT=ROOT/'final_figure_package'
FIG=OUT/'figures'; TAB=OUT/'tables'; SRC=OUT/'source_data'
PRED='Pred_log10_negB22_final'; SIM='Sim_log10_negB22_final'
BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'; GREY='#777777'; PALE='#E8E8E8'
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})

LABELS={'Kd_fit':'Kd fit','LB':'LB','NB':'ND','Binding_qualitative':'Qualitative interaction','Kd_pending':'Measurement pending','Ambiguous':'Unresolved response'}
FULL_LABELS={'Kd_fit':'Quantitative Kd fit','LB':'Line broadening without an extractable Kd','NB':'No detectable interaction under the tested conditions','Binding_qualitative':'Qualitative interaction','Kd_pending':'Measurement pending','Ambiguous':'Unresolved response'}

def kd_text(kd,err):
 if not np.isfinite(kd): return ''
 if np.isfinite(err):
  decimals=1 if err < 10 else 0
  return f'{kd:.{decimals}f}±{err:.{decimals}f}'
 return f'{kd:g}'

def data():
 from run_revision_analysis import data as load
 return load()

def savefig(fig,name):
 FIG.mkdir(parents=True,exist_ok=True);fig.tight_layout()
 # Place identity labels after the final axes layout, so collision checks
 # use the coordinates that are actually exported.
 from plot_master_validation import identity_labels
 for ax,frame,col in getattr(fig,'_identity_panels',[]):
  identity_labels(fig,ax,frame,col,'pKd',yerr='pKd_error')
 for ext in ['pdf','svg','png']:fig.savefig(FIG/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)

def auc(y,s):
 y=np.asarray(y,bool);s=np.asarray(s,float)
 return float((rankdata(s)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))

def roc_points(y,s):
 y=np.asarray(y,bool);s=np.asarray(s,float);thr=np.r_[np.inf,np.sort(np.unique(s))[::-1],-np.inf];rows=[]
 for t in thr:
  q=s>=t;rows.append((np.mean(q[y]),np.mean(~q[~y])))
 return np.array([(1-sp,se) for se,sp in rows])

def panel_letter(ax,letter):ax.text(-.16,1.15,letter,transform=ax.transAxes,fontweight='bold',fontsize=12,va='top')

def outcome_matrix(ax,d):
 z=d[(~d.is_shuffle) & d.Outcome.isin(['NB','LB','Kd_fit'])].copy()
 row_order=['lef1_nter','foxo4_cr3','foxo6_cr3','atf4_tad','tfe3_cter','tp53','foxm1','sox12']
 col_order=['thoc4_nter','thoc4_cter','g3bp1_rgg','taf15_rgg','rbmx','snrnp70_s','tceal6','rab44']
 rows=[x for x in row_order if x in set(z.AHD_name)];cols=[x for x in col_order if x in set(z.RGG_name)]
 codes={'NB':0,'LB':1,'Kd_fit':2}
 arr=np.full((len(rows),len(cols)),np.nan);texts={}
 for _,r in z.iterrows():
  i=rows.index(r.AHD_name);j=cols.index(r.RGG_name);arr[i,j]=codes.get(r.Outcome,4)
  texts[i,j]=kd_text(r.Kd_uM,r.Kd_error_uM) if r.Outcome=='Kd_fit' else {'LB':'LB','NB':'ND'}.get(r.Outcome,'')
 colors=['#E6E6E6','#E69F00','#56B4E9'];cmap=ListedColormap(colors);cmap.set_bad('#FAFAFA')
 ax.imshow(np.ma.masked_invalid(arr),aspect='auto',cmap=cmap,vmin=0,vmax=2)
 pretty_ahd={'lef1_nter':'LEF1','foxo4_cr3':'FOXO4','foxo6_cr3':'FOXO6','atf4_tad':'ATF4','tfe3_cter':'TFE3','tp53':'p53','foxm1':'FOXM1','sox12':'SOX12'}
 pretty_rgg={'thoc4_nter':'THOC4-N','thoc4_cter':'THOC4-C','g3bp1_rgg':'G3BP1','taf15_rgg':'TAF15','rbmx':'RBMX','snrnp70_s':'SNRNP70','tceal6':'TCEAL6','rab44':'RAB44'}
 ax.set(xticks=range(len(cols)),xticklabels=[pretty_rgg.get(x,x) for x in cols],yticks=range(len(rows)),yticklabels=[pretty_ahd.get(x,x) for x in rows]);ax.tick_params(axis='x',rotation=42,labelsize=7);ax.tick_params(axis='y',labelsize=7,length=0)
 ax.set_xticks(np.arange(-.5,len(cols),1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True);ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',bottom=False,left=False)
 for (i,j),t in texts.items():ax.text(j,i,t,ha='center',va='center',fontsize=5.5,fontweight='bold' if t not in ['ND','LB'] else 'normal',color='#17324D' if arr[i,j]==2 else '#222222')
 ax.set_title('WT experimental outcomes',fontsize=10,pad=27,fontweight='bold',loc='left')
 ax.text(.5,1.04,'Kd values are shown as estimate ± fit error (µM)',transform=ax.transAxes,ha='center',fontsize=7,color='#555555')
 ax.legend(handles=[Patch(facecolor=colors[2],label='Kd fit'),Patch(facecolor=colors[1],label='LB'),Patch(facecolor=colors[0],label='ND'),Patch(facecolor='#FAFAFA',edgecolor='#CCCCCC',label='Not measured')],loc='lower right',bbox_to_anchor=(1,1.105),ncol=4,frameon=False,fontsize=6.7,handlelength=1.3,columnspacing=1.2,borderaxespad=0)
 for spine in ax.spines.values():spine.set_visible(False)
 z[['pair_id','pair_display_name','AHD_name','RGG_name','Outcome','Kd_uM','Kd_error_uM']].to_csv(SRC/'Fig5A_outcome_matrix.csv',index=False)

def score_outcomes(ax,d,col=PRED,source='Fig5B_scores_by_outcome',primary_stat=False):
 order=['NB','LB','Kd_fit'];z=d[d.Outcome.isin(order)&np.isfinite(d[col])].copy();colors={'NB':'#999999','LB':'#E69F00','Kd_fit':'#56B4E9'}
 rng=np.random.default_rng(7)
 counts=[]
 for k,o in enumerate(order):
  g=z[z.Outcome.eq(o)]
  x=k+rng.uniform(-.14,.14,len(g));ax.scatter(x,g[col],s=27,c=colors[o],marker='o',edgecolor='white',linewidth=.35,alpha=.9)
  ax.plot([k-.19,k+.19],[g[col].median()]*2,c='black',lw=1.3)
  counts.append(f'{LABELS[o]}\n(n={len(g)})')
 label='Predicted' if col==PRED else 'Simulated'
 ax.set(xticks=range(3),xticklabels=counts,xlim=(-.5,2.5),ylabel=label+r' $\log_{10}(-B_{22})$')
 ax.set_title(label+' scores by NMR outcome',pad=24 if primary_stat else None)
 if primary_stat:
  stats=pd.read_csv(REV/'tables/discrimination_statistics.csv')
  r=stats[(stats.comparison.eq('Kd_vs_NB')) &
          (stats.series.eq('Predicted')) &
          (stats.cluster.eq('AHD_parent'))].iloc[0]
  ax.text(.5,1.015,
          f'Kd fit vs ND: AUROC {r.AUROC:.2f} '
          f'(95% CI {r.CI95_low:.2f}-{r.CI95_high:.2f}; AHD-grouped bootstrap)',
          transform=ax.transAxes,ha='center',va='bottom',fontsize=6.5,color='#444444')
 z[['pair_id','pair_display_name','Outcome',col]].to_csv(SRC/(source+'.csv'),index=False)

def primary_roc(ax,d,source='Fig5C_primary_ROC'):
 z=d[d.Outcome.isin(['Kd_fit','NB']) & np.isfinite(d[PRED]) & np.isfinite(d[SIM])].copy();y=z.Outcome.eq('Kd_fit').values
 for col,label,color in [(PRED,'Predicted',BLUE),(SIM,'Simulated',ORANGE)]:
  q=roc_points(y,z[col]);ax.plot(q[:,0],q[:,1],label=f'{label}: AUROC {auc(y,z[col]):.2f}',color=color,lw=1.8)
 ax.plot([0,1],[0,1],':',c='#777777',lw=1)
 ax.set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1.02),title='Kd fit versus ND')
 ax.legend(frameon=False,fontsize=7,loc='lower right')
 z[['pair_id','pair_display_name','Outcome',PRED,SIM]].to_csv(SRC/(source+'.csv'),index=False)

def predictor_primary_roc(ax,d,source='Fig5C_predictor_ROC'):
 z=d[d.Outcome.isin(['Kd_fit','NB']) & np.isfinite(d[PRED])].copy();y=z.Outcome.eq('Kd_fit').to_numpy()
 q=roc_points(y,z[PRED]);estimate=auc(y,z[PRED])
 stats=pd.read_csv(REV/'tables/discrimination_statistics.csv')
 ahd=stats[(stats.comparison.eq('Kd_vs_NB')) & (stats.series.eq('Predicted')) & (stats.cluster.eq('AHD_parent'))].iloc[0]
 rgg=stats[(stats.comparison.eq('Kd_vs_NB')) & (stats.series.eq('Predicted')) & (stats.cluster.eq('RGG_parent'))].iloc[0]
 ax.plot(q[:,0],q[:,1],color=BLUE,lw=2,label=f'Predicted: AUROC {estimate:.2f}')
 ax.plot([0,1],[0,1],':',c='#777777',lw=1,label='Chance')
 ax.set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1.02),
        title='Predictor discrimination: Kd fit vs ND')
 ax.legend(frameon=False,fontsize=7,loc='lower right')
 ax.text(.03,.97,
         f'{int(y.sum())} Kd fit vs {int((~y).sum())} ND\n'
         f'95% CI, AHD grouped: {ahd.CI95_low:.2f}-{ahd.CI95_high:.2f}\n'
         f'95% CI, RGG grouped: {rgg.CI95_low:.2f}-{rgg.CI95_high:.2f}',
         transform=ax.transAxes,va='top',fontsize=6.6,color='#333333')
 out=z[['pair_id','pair_display_name','Outcome',PRED]].copy();out['positive']=y
 out.to_csv(SRC/(source+'.csv'),index=False)

def outcome_concordance(ax,d,cluster='AHD_parent',source='Fig5C_AUROC_concordance'):
 stats=pd.read_csv(REV/'tables/discrimination_statistics.csv')
 comparisons=[('Kd_vs_NB',['Kd_fit'],'Kd fit vs ND'),
              ('Kd_LB_vs_NB',['Kd_fit','LB'],'Kd fit + LB vs ND'),
              ('LB_vs_NB',['LB'],'LB vs ND')]
 plotted=[];members=[];labels=[]
 for k,(name,positive,label) in enumerate(comparisons):
  valid=d.Outcome.isin(positive+['NB']) & np.isfinite(d[PRED]) & np.isfinite(d[SIM])
  z=d[valid];y=z.Outcome.isin(positive).to_numpy()
  labels.append(f'{label}\n{int(y.sum())} vs {int((~y).sum())} pairs')
  mem=d[['pair_id','pair_display_name','Outcome',PRED,SIM]].copy()
  mem['comparison']=name;mem['included']=valid
  mem['reason']=np.where(valid,'included','outcome outside comparison or score unavailable')
  members.append(mem)
  for off,series,col,color,marker in [(-.14,'Predicted',PRED,BLUE,'o'),(.14,'Simulated',SIM,ORANGE,'^')]:
   estimate=auc(y,z[col])
   r=stats[(stats.comparison==name)&(stats.series==series)&(stats.cluster==cluster)].iloc[0]
   if int(r.n)!=len(z) or int(r.positive)!=int(y.sum()) or not np.isclose(r.AUROC,estimate):
    raise ValueError('Regenerate revision statistics: current panel membership differs from cached statistics')
   ax.errorbar(estimate,k+off,xerr=[[estimate-r.CI95_low],[r.CI95_high-estimate]],fmt=marker,ms=6,color=color,elinewidth=1.5,capsize=2.5,mec='white',mew=.6,zorder=3)
   plotted.append(r.to_dict())
 ax.axhspan(-.39,.39,color='#F0F4F7',zorder=-2)
 ax.axvline(.5,color='#777777',ls=':',lw=1,zorder=-1)
 ax.set(yticks=range(3),yticklabels=labels,xlim=(.45,1.025),ylim=(2.48,-.48),
        xticks=[.5,.6,.7,.8,.9,1.0],xlabel='AUROC')
 ax.set_title('Experimental outcome discrimination',pad=33)
 ax.tick_params(axis='y',length=0,labelsize=8)
 ax.spines['left'].set_visible(False);ax.spines['bottom'].set_bounds(.5,1)
 ax.get_yticklabels()[0].set_fontweight('bold')
 ax.legend(handles=[Line2D([],[],color=BLUE,marker='o',label='Predicted'),
                    Line2D([],[],color=ORANGE,marker='^',label='Simulated')],
           loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=7,ncol=2,borderaxespad=0)
 ax.text(.01,-.21,f'95% intervals grouped by {cluster.split("_")[0]} parent · dotted line: chance',
         transform=ax.transAxes,fontsize=7,color='#555555')
 pd.DataFrame(plotted).to_csv(SRC/(source+'_statistics.csv'),index=False)
 pd.concat(members,ignore_index=True).to_csv(SRC/(source+'_membership.csv'),index=False)

def add_trend(ax,x,y,color,ls='-'):
 x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
 if ok.sum()<2:return
 slope,intercept=np.polyfit(x[ok],y[ok],1);xx=np.linspace(x[ok].min(),x[ok].max(),100)
 ax.plot(xx,slope*xx+intercept,color=color,ls=ls,lw=1.5,alpha=.9)

def intensity_combined(ax,d):
 z=d[d.I_ratio_mean_below_threshold.notna()].copy()
 for col,label,color,marker in [(PRED,'Predicted',BLUE,'o'),(SIM,'Simulated',ORANGE,'^')]:
  ax.scatter(z[col],z.I_ratio_mean_below_threshold,c=color,marker=marker,s=30,alpha=.85,label=label,edgecolor='white',linewidth=.35)
  add_trend(ax,z[col],z.I_ratio_mean_below_threshold,color)
  rho=spearmanr(z[col],z.I_ratio_mean_below_threshold).statistic
  ax.plot([],[],color=color,label=f'{label} ρ={rho:.2f}')
 ax.set(xlabel=r'Computational $\log_{10}(-B_{22})$',ylabel='Mean retained intensity ratio',title=f'Exploratory intensity comparison (n={len(z)})')
 ax.legend(frameon=False,fontsize=6.5,loc='best',ncol=2)
 z[['pair_id','pair_display_name','Outcome','I_ratio_mean_below_threshold',PRED,SIM]].to_csv(SRC/'FigS10E_intensity_comparison.csv',index=False)

def pkd(ax,d,col=PRED,source='Fig5D_fitted_affinities'):
 from plot_master_validation import limits
 z=d[d.Outcome.eq('Kd_fit') & np.isfinite(d[col]) & np.isfinite(d.pKd)].copy()
 marker_cycle=['o','^','s','*','D','v','P','X','<','>','h']
 markers={name:marker_cycle[i%len(marker_cycle)] for i,name in enumerate(sorted(z.RGG_name.dropna().unique()))}
 markers.update({'thoc4_nter':'o','thoc4_cter':'^','g3bp1_rgg':'s','rbmx':'*'})
 partners={'thoc4_nter':'THOC4-N','thoc4_cter':'THOC4-C','g3bp1_rgg':'G3BP1','rbmx':'RBMX'}
 for partner,g in z.groupby('RGG_name',sort=True):
  color=BLUE if partner.startswith('thoc4') else '#999999'
  ax.errorbar(g[col],g.pKd,yerr=g.pKd_error,fmt=markers.get(partner,'D'),
              ms=6 if partner!='rbmx' else 8,mfc=color,mec='#222222',mew=.6,
              ecolor='#999999',elinewidth=.8,capsize=2,ls='none',label=partners.get(partner,partner),zorder=3)
 add_trend(ax,z[col],z.pKd,'black',ls='--')
 rho=spearmanr(z[col],z.pKd).statistic
 label='Predicted' if col==PRED else 'Simulated'
 err=z.pKd_error.fillna(0)
 ax.set(xlabel=label+r' $\log_{10}(-B_{22})$',ylabel='Experimental pKd',
        xlim=limits(z[col],pad=.12),ylim=limits(np.r_[z.pKd-err,z.pKd+err],pad=.14))
 ax.set_title(label+' score versus fitted affinity',pad=40)
 ax.legend(frameon=False,fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=2,columnspacing=1)
 ax.text(.02,.97,f'n = {len(z)}    Spearman ρ = {rho:.2f}',transform=ax.transAxes,fontsize=7,va='top')
 fig=ax.figure
 fig._identity_panels=getattr(fig,'_identity_panels',[])+[(ax,z,col)]
 z[['pair_id','pair_display_name','AHD_name','RGG_name','Kd_uM','Kd_error_uM','pKd','pKd_error',col]].to_csv(SRC/(source+'.csv'),index=False)

def figure5(d):
 for stale in [SRC/'Fig5A_outcome_matrix.csv',SRC/'Fig5A_scores_by_outcome.csv',
               SRC/'Fig5C_scores_by_outcome.csv',SRC/'Fig5C_primary_ROC.csv',
               SRC/'Fig5C_predictor_ROC.csv',SRC/'Fig5C_intensity_comparison.csv',
               SRC/'Fig5C_fitted_affinities.csv']:
  if stale.exists():stale.unlink()
 for ext in ['pdf','svg','png']:
  stale=FIG/f'Figure5_with_panelB_placeholder.{ext}'
  if stale.exists():stale.unlink()
 fig=plt.figure(figsize=(12,8));gs=fig.add_gridspec(2,2,height_ratios=[1,1],hspace=.43,wspace=.28)
 a=fig.add_subplot(gs[0,0]);a.axis('off');a.add_patch(plt.Rectangle((.05,.09),.9,.82,fill=False,ls='--',lw=1.2,color=GREY,transform=a.transAxes));a.text(.5,.60,'PANEL A PLACEHOLDER',ha='center',va='center',weight='bold',transform=a.transAxes);a.text(.5,.40,'Representative NMR evidence:\nKd-fit spectrum + titration curve\nline broadening · no detectable interaction',ha='center',va='center',color=GREY,transform=a.transAxes);panel_letter(a,'A')
 b=fig.add_subplot(gs[0,1]);score_outcomes(b,d,PRED,'Fig5B_scores_by_outcome');panel_letter(b,'B')
 c=fig.add_subplot(gs[1,0]);outcome_concordance(c,d,'AHD_parent','Fig5C_AUROC_concordance');panel_letter(c,'C')
 dd=fig.add_subplot(gs[1,1]);pkd(dd,d,PRED,'Fig5D_fitted_affinities');panel_letter(dd,'D')
 savefig(fig,'Figure5_with_panelA_placeholder')

def fig_s10(d):
 for stale in [SRC/'FigS10C_primary_ROC.csv',
               SRC/'FigS10D_RGG_grouped_concordance_statistics.csv',
               SRC/'FigS10D_RGG_grouped_concordance_membership.csv',
               SRC/'FigS10E_held_family_thresholds.csv',
               SRC/'FigS10F_intensity_comparison.csv']:
  if stale.exists():stale.unlink()
 for ext in ['pdf','svg','png']:
  stale=FIG/f'FigureS10_complete_NMR_and_statistics.{ext}'
  if stale.exists():stale.unlink()
 from histidine_ph_controls import load_controls
 sim,pred,metrics,provenance=load_controls()
 sim.to_csv(SRC/'FigS10A_histidine_pH_simulation.csv',index=False)
 pred.to_csv(SRC/'FigS10B_histidine_pH_prediction.csv',index=False)
 for name in ['FigS10A_simulated_outcomes.csv','FigS10B_simulated_affinities.csv','FigS10C_intensity_comparison.csv']:
  (SRC/name).unlink(missing_ok=True)
 fig=plt.figure(figsize=(12,12));gs=fig.add_gridspec(3,2,hspace=.48,wspace=.3)
 for j,(frame,x,y,xlabel,ylabel) in enumerate([
  (sim,'simulated_pH7','simulated_pH6p5','pH 7.0 simulated score','pH 6.5 simulated score'),
  (pred,'simulated_pH6p5','predicted_pH6p5','pH 6.5 simulated score','Frozen predicted score')]):
  ax=fig.add_subplot(gs[0,j]);ax.scatter(frame[x],frame[y],s=16,color=GREY,alpha=.55,edgecolors='none')
  lo=min(frame[x].min(),frame[y].min())-.05;hi=max(frame[x].max(),frame[y].max())+.05
  ax.plot([lo,hi],[lo,hi],color='black',ls=':',lw=1);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
  ax.set_xlabel(xlabel+' log₁₀(−B₂₂)');ax.set_ylabel(ylabel+' log₁₀(−B₂₂)')
  m=metrics.iloc[j];ax.text(.04,.96,f'n = {int(m.n)}\nr = {m.Pearson_r:.3f}\nρ = {m.Spearman_rho:.3f}\nRMSE = {m.RMSE:.3f}',transform=ax.transAxes,va='top')
  panel_letter(ax,'AB'[j])
 c=fig.add_subplot(gs[1,0]);score_outcomes(c,d,SIM,'FigS10C_simulated_outcomes');panel_letter(c,'C')
 e=fig.add_subplot(gs[1,1]);pkd(e,d,SIM,'FigS10D_simulated_affinities');panel_letter(e,'D')
 f=fig.add_subplot(gs[2,:]);intensity_combined(f,d);panel_letter(f,'E')
 savefig(fig,'FigureS10_simulation_and_intensity')


def fig_s11(d):
 for stale in ['FigureS11_shuffle_source.csv','FigureS11_synthetic_reference_source.csv']:
  p=SRC/stale
  if p.exists():p.unlink()
 fig=plt.figure(figsize=(14,11));gs=fig.add_gridspec(3,6,height_ratios=[1,1.35,1.15],hspace=.62,wspace=.78)
 a=fig.add_subplot(gs[0,:3]);b=fig.add_subplot(gs[0,3:]);c=fig.add_subplot(gs[1,:]);energy_axes=[fig.add_subplot(gs[2,0:2]),fig.add_subplot(gs[2,2:4]),fig.add_subplot(gs[2,4:6])]

 thresholds=pd.read_csv(REV/'tables/threshold_summary.csv')
 primary=thresholds[(thresholds.comparison.eq('Kd_vs_NB'))&(thresholds.series.eq('Predicted'))&(thresholds.evaluation.eq('in-sample exploratory'))].iloc[0]
 cutoff=float(primary.threshold);z=d[d.Outcome.isin(['Kd_fit','NB'])&np.isfinite(d[PRED])].copy().sort_values(['Outcome',PRED]);rng=np.random.default_rng(14)
 z['outcome_y']=z.Outcome.map({'NB':0,'Kd_fit':1}).astype(float)+rng.uniform(-.12,.12,len(z));z['above_threshold']=z[PRED]>=cutoff;z['correct']=z.above_threshold.eq(z.Outcome.eq('Kd_fit'))
 colors=z.Outcome.map({'NB':'#999999','Kd_fit':'#56B4E9'});edges=np.where(z.correct,'white','#D55E00')
 a.axvspan(z[PRED].min()-.05,cutoff,color='#F2F2F2',zorder=-2);a.axvspan(cutoff,z[PRED].max()+.05,color='#EAF4FA',zorder=-2);a.axvline(cutoff,color='#222222',ls='--',lw=1.2)
 a.scatter(z[PRED],z.outcome_y,c=colors,edgecolor=edges,linewidth=np.where(z.correct,.4,1.4),s=35)
 a.set(yticks=[0,1],yticklabels=[f'ND (n={sum(z.Outcome.eq("NB"))})',f'Kd fit (n={sum(z.Outcome.eq("Kd_fit"))})'],xlabel=r'Predicted $\log_{10}(-B_{22})$',title='Exploratory Kd-fit versus ND threshold',ylim=(-.35,1.35))
 a.text(cutoff,.5,f'  threshold = {cutoff:.2f}',rotation=90,va='center',ha='left',fontsize=7);a.text(.02,.96,f'Balanced accuracy = {primary.balanced_accuracy:.2f}\nOrange outlines: misclassified pairs',transform=a.transAxes,va='top',fontsize=7)
 z[['pair_id','pair_display_name','Outcome',PRED,'above_threshold','correct']].to_csv(SRC/'FigureS11A_threshold_classification.csv',index=False)

 robust=thresholds[thresholds.series.eq('Predicted')].copy();eval_order=['in-sample exploratory','leave-AHD_parent-out','leave-RGG_parent-out'];comparison_order=['Kd_vs_NB','Kd_LB_vs_NB','LB_vs_NB'];labels={'Kd_vs_NB':'Kd fit vs ND','Kd_LB_vs_NB':'Kd fit + LB vs ND','LB_vs_NB':'LB vs ND'};eval_labels=['In-sample','Leave AHD out','Leave RGG out']
 for name,color,marker in zip(comparison_order,[BLUE,GREEN,ORANGE],['o','s','^']):
  g=robust[robust.comparison.eq(name)].set_index('evaluation').reindex(eval_order);b.plot(range(3),g.balanced_accuracy,marker=marker,color=color,lw=1.3,label=labels[name])
 b.axhline(.5,color='#777777',ls=':',lw=1);b.set(xticks=range(3),xticklabels=eval_labels,ylabel='Balanced accuracy',ylim=(.45,.94),title='Threshold robustness across held-family evaluations');b.legend(frameon=False,fontsize=6.5,loc='lower left');b.tick_params(axis='x',rotation=12)
 robust.to_csv(SRC/'FigureS11B_threshold_robustness.csv',index=False)

 sh=pd.read_csv(REV/'figures/WT_shuffle_comparison.csv').sort_values(['label']).reset_index(drop=True);outcome_colors={'Kd_fit':'#56B4E9','LB':'#E69F00','NB':'#999999','Kd_pending':'#FFFFFF'};outcome_edges={'Kd_fit':'#2474A6','LB':'#A65E00','NB':'#555555','Kd_pending':'#777777'}
 complete=~sh.WT_outcome.eq('Kd_pending')&~sh.shuffle_outcome.eq('Kd_pending');changed=complete&sh.WT_outcome.ne(sh.shuffle_outcome)
 for i,r in sh.iterrows():
  c.plot([0,1],[i,i],color='#D0D0D0',lw=1.3,zorder=0)
  for xval,outcome in [(0,r.WT_outcome),(1,r.shuffle_outcome)]:
   c.scatter(xval,i,s=75,facecolor=outcome_colors.get(outcome,'#FFFFFF'),edgecolor=outcome_edges.get(outcome,'#555555'),linewidth=1,zorder=2)
   c.text(xval+(.045 if xval==0 else -.045),i,('Pending' if outcome=='Kd_pending' else LABELS.get(outcome,outcome)),ha='left' if xval==0 else 'right',va='center',fontsize=6.5)
 c.set(xticks=[0,1],xticklabels=['WT response','Shuffle response'],yticks=range(len(sh)),yticklabels=sh.label,xlim=(-.28,1.28),ylim=(len(sh)-.5,-.5),title=f'Pair-resolved experimental response to sequence shuffling · {int(changed.sum())}/{int(complete.sum())} complete pairs changed category')
 c.tick_params(axis='y',length=0,labelsize=7);c.spines['left'].set_visible(False);c.spines['bottom'].set_visible(False);c.tick_params(axis='x',length=0)
 for tick in c.get_yticklabels():
  if 'LEF1' in tick.get_text():tick.set_fontweight('bold');tick.set_color(ORANGE)
 c.legend(handles=[Line2D([],[],marker='o',ls='',mfc=outcome_colors[o],mec=outcome_edges[o],label=('Pending' if o=='Kd_pending' else LABELS[o])) for o in ['Kd_fit','LB','NB','Kd_pending']],loc='lower center',bbox_to_anchor=(.5,-.22),ncol=4,frameon=False,fontsize=7)
 sh.assign(complete_pair=complete,outcome_changed=changed).to_csv(SRC/'FigureS11C_shuffle_outcomes.csv',index=False)

 tr=pd.read_csv(REV/'tables/trajectory_pair_summary.csv');lef=tr[tr.AHD_name.eq('lef1_nter')];partners=sorted(lef.RGG_name.unique());summary=[]
 metrics=[('total_kJ_mol','total_kJ_mol_block_SEM','Mean total energy\n(kJ/mol)','All-frame interaction energy','higher'),('contact_08','contact_08_block_SEM','Contact occupancy\n(r < 0.8 nm)','Encounter frequency','lower'),('total_kJ_mol_contact_conditioned',None,'Contact-conditioned energy\n(kJ/mol)','Energy during encounters','higher')]
 partner_labels={'thoc4_nter':'THOC4-N','thoc4_cter':'THOC4-C','g3bp1_rgg':'G3BP1','taf15_rgg':'TAF15'}
 for ax,(metric,error,ylabel,title,direction) in zip(energy_axes,metrics):
  for j,p in enumerate(partners):
   g=tr[tr.RGG_name.eq(p)].copy();lg=g[g.AHD_name.eq('lef1_nter')].iloc[0];comp=g[~g.AHD_name.eq('lef1_nter')];jit=np.linspace(-.12,.12,max(len(comp),1))
   if len(comp):ax.errorbar(j+jit,comp[metric],yerr=comp[error] if error else None,fmt='o',ms=4,color='#999999',ecolor='#BBBBBB',elinewidth=.7,capsize=1.5,ls='none')
   ax.errorbar(j,lg[metric],yerr=lg[error] if error else None,fmt='o',ms=7,color=ORANGE,ecolor=ORANGE,elinewidth=.9,capsize=2,ls='none',zorder=3)
   delta=float(lg[metric]-comp[metric].median());summary.append({'metric':metric,'RGG_name':p,'LEF1_value':lg[metric],'LEF1_block_SEM':lg[error] if error else np.nan,'comparator_median':comp[metric].median(),'LEF1_minus_comparator_median':delta,'n_comparators':len(comp),'direction_consistent':delta>0 if direction=='higher' else delta<0})
  q=pd.DataFrame([r for r in summary if r['metric']==metric]);n=int(q.direction_consistent.sum());total=len(q);p_sign=2*(.5**total) if n in [0,total] and total else np.nan
  for row in summary:
   if row['metric']==metric:row['consistent_partners']=n;row['tested_partners']=total;row['exact_two_sided_sign_p']=p_sign
  ax.set(xticks=range(len(partners)),xticklabels=[partner_labels.get(p,p.replace('_',' ')) for p in partners],ylabel=ylabel,title=title);ax.tick_params(axis='x',rotation=35,labelsize=6.5);ax.set_title(title,pad=24);ax.text(.5,1.015,f'LEF1 direction: {n}/{total} partners · sign p = {p_sign:.3f}',transform=ax.transAxes,fontsize=6.5,ha='center',va='bottom',color='#555555')
 fig.legend(handles=[Line2D([],[],marker='o',ls='',color=ORANGE,label='LEF1'),Line2D([],[],marker='o',ls='',color='#999999',label='Matched WT AHDs')],loc='lower center',bbox_to_anchor=(.5,-.005),ncol=2,frameon=False,fontsize=7)
 profiles=[]
 for p in partners:
  pair=lef[lef.RGG_name.eq(p)].iloc[0].pair_id;m=pd.read_csv(REV/'trajectories/cache'/pair/'residue_maps.csv.gz');q=m.groupby(['AHD_index','AHD_residue'],as_index=False).total_kJ_mol.sum();q['partner']=p;profiles.append(q)
 for ax,l in zip([a,b,c,*energy_axes],'ABCDEF'):panel_letter(ax,l)
 lef.to_csv(SRC/'FigureS11_LEF1_trajectory_source.csv',index=False);pd.concat(profiles).to_csv(SRC/'FigureS11_LEF1_residue_profiles.csv',index=False);pd.DataFrame(summary).to_csv(SRC/'FigureS11D-F_LEF1_matched_metrics.csv',index=False)
 savefig(fig,'FigureS11_shuffle_and_LEF1_context')

def _section(title,frame,note=''):
 split=json.loads(frame.to_json(orient='split'))
 return {'title':title,'note':note,'columns':split['columns'],'rows':split['data']}

def _describe_column(column):
 descriptions={
  'pair_id':'Unique oriented machine identifier, RGG partner followed by AHD construct.',
  'pair_display_name':'Human-readable AHD–RG/RGG pair name.',
  'Outcome':'Curated experimental outcome used by the analyses; raw labels are not used.',
  'Experimental_description':'Expanded description of the curated experimental outcome.',
  'Kd_uM':'Fitted dissociation constant in micromolar.',
  'Kd_error_uM':'Reported fitting error for Kd in micromolar; its source uncertainty type is preserved.',
  'pKd':'6 − log10(Kd in micromolar).',
  'pKd_error':'First-order propagation of the reported Kd error to pKd.',
  'highest_ligand_equivalents':'Highest ligand-to-labelled-protein molar equivalent measured for this pair.',
  'fraction_bound_highest_ligand':'Manually supplied bound fraction at the highest ligand equivalent; range 0–1.',
  'percent_bound_highest_ligand':'Manually supplied bound percentage at the highest ligand equivalent; equals 100 × fraction when both are supplied.',
  'I_ratio_threshold':'Intensity-ratio threshold measurement supplied by the experimental table.',
  'I_ratio_mean_below_threshold':'Mean retained intensity ratio below the supplied threshold.',
  PRED:'Frozen predictor output, log10(−B22).',
  SIM:'Simulation-derived log10(−B22) after the established replica aggregation.',
  'AUROC':'Area under the ROC curve; equal to positive–negative concordance with half credit for score ties.',
  'positive_negative_concordance':'Probability that a positive pair scores above a negative pair, with half credit for ties.',
  'CI95_low':'Lower endpoint of the 95% grouped-bootstrap interval.',
  'CI95_high':'Upper endpoint of the 95% grouped-bootstrap interval.',
  'threshold':'Exploratory score cutoff selected by balanced accuracy.',
  'balanced_accuracy':'Mean of sensitivity and specificity.',
  'experimental_status':'Measurement availability or resolution status.',
  'Notes':'Curated experimental or provenance note.',
  'pair_id_parent':'Matched WT pair identifier for a shuffled construct.',
  'record_type':'Section identifier used when related analyses are exported in one CSV.'}
 if column in descriptions:return descriptions[column]
 if column in ['delta_B22','delta_logB22','delta_pH6p5_minus_pH7']:return 'pH6.5 value minus pH7 value on the named simulation scale.'
 if column.startswith('delta_'):return 'Shuffle value minus matched WT value for '+column.removeprefix('delta_')+'.'
 if column.startswith('WT_'):return 'Matched WT value for '+column.removeprefix('WT_')+'.'
 if column.startswith('shuffle_'):return 'Shuffled-construct value for '+column.removeprefix('shuffle_')+'.'
 if column.endswith('_source') or 'source' in column:return 'Source or provenance field retained from the analysis.'
 if column.endswith('_sequence'):return 'Exact amino-acid sequence used by the computational analysis.'
 if column.startswith('base_'):return 'Base predictor output or error; this is the output used by the supplied histidine audit. Residual = predicted minus simulated; delta absolute error = pH6.5 minus pH7.'
 if column.startswith('final_hybrid_'):return 'Final hybrid predictor output or error; this is the output used by the existing histidine prediction panel. Residual = predicted minus simulated; delta absolute error = pH6.5 minus pH7.'
 if column in ['SD_pH7','SD_pH65']:return 'Within-condition standard deviation of corrected B22 across five simulation replicas; not a standard error.'
 if column=='pooled_within_condition_SD':return 'Square root of the mean of the two within-condition B22 variances.'
 if column=='standardized_condition_difference':return 'Mean B22 at pH6.5 minus pH7, divided by pooled within-condition SD; descriptive, not a significance test.'
 if column in ['n_H_TAD','n_H_RGG','total_H']:return 'Histidine count in the AHD, RG/RGG partner, or both constructs respectively.'
 return 'See the corresponding sheet heading, figure caption, and analysis methods for this field.'

def tables(d):
 TAB.mkdir(parents=True,exist_ok=True);SRC.mkdir(parents=True,exist_ok=True)
 for stale in [TAB/'Table_S11_complete_NMR_dataset.csv']:
  if stale.exists():stale.unlink()
 for pattern in ['Table_S11*.csv','Table_S12*.csv','Table_S13*.csv','Table_S14*.csv','Table_S15*.csv']:
  for stale in TAB.glob(pattern):stale.unlink()
 exp=d[d.experimental_status.ne('no experimental measurement supplied')].copy();exp['Experimental_description']=exp.Outcome.map(FULL_LABELS)
 cols=['pair_id','pair_display_name','AHD_name','RGG_name','AHD_length','RGG_length','AHD_variant','Outcome','Experimental_description','Kd_uM','Kd_error_uM','pKd','pKd_error','highest_ligand_equivalents','fraction_bound_highest_ligand','percent_bound_highest_ligand','I_ratio_threshold','I_ratio_mean_below_threshold',PRED,SIM,'experimental_status','Notes']
 nmr=exp[[c for c in cols if c in exp]].copy()
 nmr.to_csv(TAB/'Table_S11_NMR_outcomes_and_Kd.csv',index=False)

 pred_outcome=pd.read_csv(REV/'paper_figures/prediction_vs_outcome_statistics.csv');pred_outcome.insert(0,'series','Predicted');pred_outcome.insert(0,'analysis','outcome group tests')
 sim_outcome=pd.read_csv(REV/'paper_figures/simulation_vs_outcome_statistics.csv');sim_outcome.insert(0,'series','Simulated');sim_outcome.insert(0,'analysis','outcome group tests')
 pred_pkd=pd.read_csv(REV/'paper_figures/prediction_vs_pKd_statistics.csv');pred_pkd.insert(0,'series','Predicted');pred_pkd.insert(0,'analysis','pKd association')
 sim_pkd=pd.read_csv(REV/'paper_figures/simulation_vs_pKd_statistics.csv');sim_pkd.insert(0,'series','Simulated');sim_pkd.insert(0,'analysis','pKd association')
 validation=pd.concat([pred_outcome,sim_outcome,pred_pkd,sim_pkd],ignore_index=True,sort=False)
 validation.to_csv(TAB/'Table_S11_validation_statistics.csv',index=False)

 discrimination=pd.read_csv(REV/'tables/discrimination_statistics.csv')
 rank=pd.read_csv(REV/'tables/rank_statistics.csv');rank=rank[rank.panel.eq('pKd')].copy()
 auroc_export=pd.concat([discrimination.assign(record_type='outcome discrimination'),rank.assign(record_type='pKd rank concordance')],ignore_index=True,sort=False)
 auroc_export.to_csv(TAB/'Table_S11_AUROC_and_concordance.csv',index=False)

 thresholds=pd.read_csv(REV/'tables/threshold_summary.csv');held=pd.read_csv(REV/'tables/threshold_heldout_predictions.csv')
 thresholds.to_csv(TAB/'Table_S11_threshold_sensitivity.csv',index=False);held.to_csv(TAB/'Table_S11_threshold_heldout_predictions.csv',index=False)
 ir_cols=['pair_id','pair_display_name','Experimental_description','I_ratio_threshold','I_ratio_mean_below_threshold','I_ratio_threshold_rank_ascending','I_ratio_mean_below_threshold_rank_ascending',PRED,SIM,'I_ratio_source','I_ratio_source_location']
 intensity=exp[exp.I_ratio_mean_below_threshold.notna()][[c for c in ir_cols if c in exp]].copy();intensity.to_csv(TAB/'Table_S11_intensity_ratios.csv',index=False)
 shuffle=pd.read_csv(REV/'figures/WT_shuffle_comparison.csv');shuffle.to_csv(TAB/'Table_S11_shuffle_comparisons.csv',index=False)

 trajectory_path=REV/'tables/trajectory_pair_summary.csv'
 if not trajectory_path.exists():raise FileNotFoundError('Cached trajectory summary is required; run with --with-trajectories to rebuild it')
 trajectory=pd.read_csv(trajectory_path);lef1=trajectory[trajectory.AHD_name.eq('lef1_nter')].copy()
 if lef1.empty:raise ValueError('Cached trajectory summary contains no LEF1 pairs')
 lef1.to_csv(TAB/'Table_S11_LEF1_trajectory_summary.csv',index=False)
 maps=[]
 for _,r in lef1.iterrows():
  path=REV/'trajectories/cache'/r.pair_id/'residue_maps.csv.gz'
  if not path.exists():raise FileNotFoundError(f'Cached residue map missing for {r.pair_id}; run with --with-trajectories')
  q=pd.read_csv(path);q.insert(0,'pair_id',r.pair_id);maps.append(q)
 residue_maps=pd.concat(maps,ignore_index=True);residue_maps.to_csv(TAB/'Table_S11_LEF1_residue_maps.csv',index=False)
 matched_path=SRC/'FigureS11D-F_LEF1_matched_metrics.csv'
 matched_metrics=pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()

 guide=pd.DataFrame([
  {'field':'highest_ligand_equivalents','definition':'Highest ligand-to-labelled-protein molar equivalent measured for the pair.','allowed_values':'positive number or blank','entry_rule':'Enter the actual maximum for each pair; maxima may differ between pairs.'},
  {'field':'fraction_bound_highest_ligand','definition':'Manually determined fraction of labelled protein bound at the highest ligand equivalent.','allowed_values':'0 to 1 or blank','entry_rule':'Enter only when supported by the experimental analysis; the build does not infer it from equivalents.'},
  {'field':'percent_bound_highest_ligand','definition':'The same bound fraction expressed as a percentage.','allowed_values':'0 to 100 or blank','entry_rule':'When both forms are supplied, this must equal 100 × fraction_bound_highest_ligand.'}
 ])
 guide.to_csv(TAB/'Table_S11_bound_fraction_guide.csv',index=False)
 bound_entries=nmr[nmr.Outcome.eq('Kd_fit')][['pair_id','pair_display_name','Kd_uM','Kd_error_uM','highest_ligand_equivalents','fraction_bound_highest_ligand','percent_bound_highest_ligand']].copy()

 from histidine_ph_controls import load_controls
 phsim,phpred,phmetrics,phsources=load_controls()
 from histidine_ph_controls import load_audit_tables
 audit_sim,audit_pred,audit_metrics,audit_sources=load_audit_tables()
 if set(phsim.pair_id)!=set(audit_sim.pair_id):raise ValueError('Panel/audit histidine pair mismatch')
 for name,frame in [('histidine_matched_simulations',audit_sim),('histidine_prediction_audit',audit_pred),('histidine_model_metrics',audit_metrics)]:
  frame.to_csv(TAB/f'Table_S11_{name}.csv',index=False)
 phsources=pd.concat([phsources,audit_sources],ignore_index=True).drop_duplicates()
 for name,frame in [('pH_simulation_comparison',phsim),('pH_predictor_comparison',phpred),('pH_control_statistics',phmetrics),('pH_control_provenance',phsources)]:
  frame.to_csv(TAB/f'Table_S11_{name}.csv',index=False)
 dictionary_rows=[]
 sheet_for=[('NMR outcomes and Kd',nmr),('Validation statistics',validation),('AUROC and concordance',pd.concat([discrimination,rank],ignore_index=True,sort=False)),('Threshold sensitivity',pd.concat([thresholds,held],ignore_index=True,sort=False)),('Intensity ratios',intensity),('Shuffle comparisons',shuffle),('LEF1 context',pd.concat([lef1,matched_metrics],ignore_index=True,sort=False)),('Bound-fraction guide',guide)]
 sheet_for += [('pH simulation comparison',audit_sim),('pH predictor comparison',audit_pred),('pH predictor comparison',audit_metrics)]
 for sheet,frame in sheet_for:
  for col in frame.columns:dictionary_rows.append({'sheet':sheet,'column':col,'description':_describe_column(col)})
 dictionary=pd.DataFrame(dictionary_rows).drop_duplicates();dictionary.to_csv(TAB/'Table_S11_data_dictionary.csv',index=False)

 payload=[
  {'sheet':'NMR outcomes and Kd','description':'Curated experimental outcomes and fitted affinities. Raw labels are not used for classification.','sections':[_section('Complete curated NMR dataset',nmr)]},
  {'sheet':'Validation statistics','description':'Omnibus, planned pairwise, and fitted-affinity association statistics.','sections':[_section('Outcome-group tests',pd.concat([pred_outcome,sim_outcome],ignore_index=True,sort=False)),_section('pKd association tests',pd.concat([pred_pkd,sim_pkd],ignore_index=True,sort=False))]},
  {'sheet':'AUROC and concordance','description':'Outcome discrimination and quantitative rank-concordance results with grouped-bootstrap uncertainty.','sections':[_section('Outcome discrimination',discrimination),_section('pKd rank statistics',rank)]},
  {'sheet':'Threshold sensitivity','description':'Exploratory balanced-accuracy cutoffs and held-family-out predictions; these are not validated binding boundaries.','sections':[_section('Threshold summary',thresholds),_section('Held-family-out predictions',held)]},
  {'sheet':'Intensity ratios','description':'Exploratory intensity attenuation measurements; lower retained intensity indicates greater attenuation and is not converted to Kd.','sections':[_section('Intensity-ratio measurements',intensity)]},
  {'sheet':'Shuffle comparisons','description':'Composition-verified shuffled constructs matched to the same WT and RG/RGG partner.','sections':[_section('Matched WT–shuffle comparisons',shuffle)]},
  {'sheet':'LEF1 context','description':'Pair-level trajectory summaries and partner-matched LEF1 comparisons used in Figure S11. Full residue-pair maps and collapsed residue profiles remain in accompanying CSVs.','sections':[_section('LEF1 trajectory summaries',lef1),_section('Partner-matched LEF1 metrics',matched_metrics)]},
  {'sheet':'Bound-fraction guide','description':'Manual fields for reporting saturation at each pair’s highest ligand equivalent.','sections':[_section('Field definitions',guide),_section('Kd-fit entry placeholders',bound_entries)]},
  {'sheet':'Data dictionary','description':'Column definitions for Supplementary Table S11.','sections':[_section('Workbook column dictionary',dictionary)]}
 ]
 payload[-1:-1]=[
  {'sheet':'pH simulation comparison','description':'Matched histidine pH controls, separate from NMR measurements. Five replicas per condition; SD is within-condition replicate variability. Deltas are pH6.5 minus pH7.','sections':[_section('Matched simulations and replicate variability',audit_sim),_section('Agreement statistics',phmetrics[phmetrics.panel.eq('S10A')]),_section('Source provenance',phsources)]},
  {'sheet':'pH predictor comparison','description':'Base and final hybrid predictions are explicitly distinguished. The supplied audit uses base_prediction; the existing figure uses final_hybrid_prediction. Residuals here are predicted minus simulated. No pH-specific prediction claim or retraining.','sections':[_section('Matched predictions at both pH conditions',audit_pred),_section('Model-specific agreement statistics',audit_metrics),_section('Existing final-hybrid panel source',phpred),_section('Existing panel agreement statistics',phmetrics[phmetrics.panel.eq('S10B')]),_section('Source provenance',phsources)]}]
 (OUT/'tables_payload.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False))
 # Proposed unified main Table 1: quantitative/qualitative [RRGG]7 experiments
 # from Figure 1 plus the integrated natural-partner measurements.
 syn=pd.read_csv(ROOT.parent/'fig2/inputs/tables/xrgg_AHD_parent_master.csv')
 syn=syn[syn.XRGG_partner.eq('[RRGG]7') & syn.AHD_variant.eq('WT')].copy()
 syn_result=syn.qualitative_result.replace({'Interaction':'Quantitative Kd fit'}).where(syn.Kd_uM.notna(),syn.qualitative_result)
 syn_result=syn_result.replace({'Line broadening':'Interaction-associated line broadening without an extractable Kd','No detectable interaction':'No detectable interaction under the tested conditions'})
 syn_display=[kd_text(k,e)+' µM' if np.isfinite(k) else {'Line broadening':'LB','No detectable interaction':'ND'}.get(q,q) for k,e,q in zip(syn.Kd_uM,syn.Kd_error_uM,syn.qualitative_result)]
 s=pd.DataFrame({'assay_context':'Synthetic [RRGG]7 peptide','AHD_construct':syn.AHD_construct_name,'AHD_variant':syn.AHD_variant,'partner':syn.XRGG_partner,'display_result':syn_display,'experimental_result':syn_result,'Kd_uM':syn.Kd_uM,'Kd_error_uM':syn.Kd_error_uM,'experimental_source':syn.source_document,'notes':syn.notes})
 variants=pd.read_csv(ROOT/'fig6_parent_master.csv').set_index('pair_id').AHD_variant
 natural_display=[kd_text(k,e)+' µM' if o=='Kd_fit' and np.isfinite(k) else LABELS.get(o,o) for o,k,e in zip(exp.Outcome,exp.Kd_uM,exp.Kd_error_uM)]
 expwt=exp[~exp.is_shuffle].copy().reset_index(drop=True)
 natural_display=[kd_text(k,e)+' µM' if o=='Kd_fit' and np.isfinite(k) else LABELS.get(o,o) for o,k,e in zip(expwt.Outcome,expwt.Kd_uM,expwt.Kd_error_uM)]
 n=pd.DataFrame({'assay_context':'Natural RG/RGG partner','AHD_construct':expwt.AHD_name,'AHD_variant':expwt.pair_id.map(variants),'partner':expwt.RGG_name,'display_result':natural_display,'experimental_result':expwt.Experimental_description,'Kd_uM':expwt.Kd_uM,'Kd_error_uM':expwt.Kd_error_uM,'experimental_source':expwt.experimental_status,'notes':expwt.Notes})
 pd.concat([s,n],ignore_index=True).to_csv(TAB/'PROPOSED_Table_1_combined_experimental_outcomes.csv',index=False)

def write_package_texts():
 (OUT/'FIGURE_STORYLINE.md').write_text('''# Figure 5 evidence sequence

The figure follows one question at a time.

1. **What experimental responses were observed?** Figure 5A shows representative NMR evidence in the same order used in the Results: a quantitatively fitted interaction, interaction-associated line broadening without an extractable Kd, and no detectable interaction under the tested conditions.
2. **How do the frozen predictions distribute across those outcomes?** Figure 5B shows all measured outcome groups. LB remains visible but is not forced into an affinity order.
3. **How robust is outcome discrimination to the treatment of line broadening?** Figure 5C reports group-versus-group AUROC, equivalent to positive-negative pairwise concordance, for Kd fit versus ND, Kd fit plus LB versus ND, and LB versus ND. The first comparison is primary; the two LB-inclusive comparisons are sensitivity analyses that do not assign LB an affinity rank.
4. **Among pairs with fitted Kd values, does the predictor recover relative affinity?** Figure 5D shows the descriptive pKd association, including all fitted measurements and their supplied fitting errors. The intended claim is coarse rank ordering, not calibrated affinity prediction.

Figure S10 asks whether simulation-derived scores show the same qualitative patterns and keeps the intensity-ratio analysis explicitly exploratory. Figure S11 first makes the exploratory predictor threshold and held-family sensitivity explicit, then shows pair-resolved WT-to-shuffle experimental changes and tests whether LEF1 differs from partner-matched AHDs in encounter frequency and interaction energy.

The main AUROC panel shows AHD-parent-grouped uncertainty for predicted and simulated scores. RGG-parent-grouped sensitivity, complete bootstrap statistics, held-family threshold checks, panel membership, exclusions, complete outcomes, intensity measurements, shuffle comparisons, LEF1 trajectory summaries and highest-titration bound fractions are reported in the descriptive sheets of Supplementary Table S11.
''')
 (OUT/'CAPTIONS.md').write_text('''# Final figure captions

## Figure 5 Experimental evaluation of predicted AHD–RG/RGG interaction propensities
(A) Representative NMR evidence supplied by the authors, ordered from a quantitatively fitted interaction to interaction-associated line broadening without an extractable Kd and no detectable interaction under the tested conditions. The fitted example should include its titration curve and fitted model. (B) Frozen predicted interaction scores grouped by experimental response. Each point is a measured pair; horizontal bars show group medians and labels report pair counts. LB is retained as a separate qualitative response and is not assigned an affinity order. Pending measurements are excluded. (C) Group-versus-group AUROC, equivalent to positive-negative pairwise concordance with half credit for ties, for Kd fit versus ND (primary comparison), Kd fit plus LB versus ND, and LB versus ND. Blue circles show predicted scores and orange triangles show simulation-derived scores. Bars give 95% percentile intervals from 2,000 bootstrap resamples grouped by AHD parent; RGG-parent-grouped sensitivity, panel membership and exclusions are reported in Supplementary Table S11. The dotted line marks chance concordance. The LB-inclusive comparisons treat broadening as qualitative interaction evidence and do not impose an affinity ordering on LB. Targeted sampling and shared constructs limit generalization. (D) Experimental pKd versus predicted interaction scores for all quantitatively fitted measurements. Text labels identify the AHD and marker shapes identify the RG/RGG partner. Error bars propagate the supplied Kd fitting errors to pKd by first-order error propagation, preserving their original uncertainty interpretation. The dashed line is an ordinary-least-squares descriptive fit; the reported coefficient is Spearman rank correlation. This line is not a calibration model. The corresponding simulation comparison is Figure S10D. Complete WT outcomes and Kd estimates with fitting errors are reported in Table 1; shuffle measurements are reported in Supplementary Table S11 and Figure S11. These results support qualitative discrimination and coarse affinity ordering rather than precise affinity prediction.

## Figure S10 Simulation comparisons and exploratory intensity attenuation
(A) Simulation-derived scores grouped by ND, LB and Kd-fit outcomes; bars show medians. (B) Experimental pKd versus simulated score for the same fitted measurements as Figure 5D. AHD labels and RG/RGG marker shapes identify each pair, error bars propagate supplied Kd fitting errors, and the dashed line is a descriptive least-squares fit. Simulation uncertainty is not displayed; the supplied fit errors are not total experimental uncertainty. (C) Exploratory mean-below-threshold intensity ratios versus predicted and simulated scores. Lines are descriptive least-squares fits and coefficients are Spearman correlations. Lower retained intensity indicates greater attenuation; these measurements are not converted to Kd and do not establish specificity. Ranked intensity measurements, pair identities, AUROC sensitivity analyses and held-family threshold checks are reported in Supplementary Table S11.

## Figure S11 Sequence order and partner-context analyses
(A) Frozen predicted scores for Kd-fit and ND pairs relative to the exploratory balanced-accuracy cutoff. Orange outlines identify pairs misclassified at this cutoff. The cutoff is a visualization of the selected validation panel and is not a universal binding boundary. (B) Balanced accuracy for Kd fit versus ND, Kd fit plus LB versus ND, and LB versus ND under in-sample, leave-AHD-parent-out and leave-RGG-partner-out threshold selection. (C) Pair-resolved experimental responses for composition-preserving shuffles and their matched WT constructs. Pending denotes measurement status rather than an experimental response. Five of the six pairs with completed responses in both constructs change category; LEF1–THOC4-N changes from ND to LB. Similar ND-to-LB changes in other constructs mean that the LEF1 transition is evidence of sequence-order dependence but is not unique to LEF1. (D) All-frame mean total interchain energy, (E) contact occupancy using the 0.8-nm cutoff, and (F) contact-conditioned total energy for LEF1 and matched WT AHD comparators against the same four provenance-resolved partners. Error bars show contiguous-block sensitivity where available and are not independent-replica errors. LEF1 follows the weaker-interaction direction relative to the comparator median for all four partners in each metric; the exact two-sided sign test gives p=0.125, so this is a consistent descriptive pattern rather than a statistically significant general effect. Energies are model-dependent ensemble observables and do not establish an experimental binding interface.
''')
 (OUT/'README.md').write_text('''# Final Figure 5 package

Run `python build_final_figure_package.py` from `fig6/` to regenerate the analysis, main Figure 5, Supplementary Figures S10-S11, source-data CSVs and the single `Supplementary_Table_S11.xlsx` workbook. Add `--with-trajectories` only when the trajectory calculations need to be rebuilt; the default reuses the validated cached trajectory summaries and residue maps.

The curated `fig6_parent_master.csv` is the experimental source of truth. The build never calls the Excel importer, retrains the predictor or edits manuscript files. Panel 5A remains reserved for author-supplied spectra and the fitted curve. Panel 5B uses its established renderer unchanged.

Supplementary Table S11 uses descriptive sheets for the NMR dataset, validation statistics, AUROC and concordance, threshold sensitivity, intensity ratios, shuffle comparisons, LEF1 context, highest-titration bound-fraction fields and the data dictionary. Detailed residue-pair maps remain available as an accompanying CSV.
''')
 old=OUT/'PANEL_B_INSTRUCTIONS.txt'
 if old.exists():old.unlink()
 (OUT/'PANEL_A_INSTRUCTIONS.txt').write_text('Panel 5A is reserved for author-supplied representative NMR evidence in the order used by the Results narrative: one Kd-fit example with its titration curve and fitted model, one interaction-associated line-broadening example without an extractable Kd, and one no-detectable-interaction example. Define LB and ND in the caption, use the corrected titration-arrow direction, and match the dashed placeholder. Complete spectra and titration series can remain in the supplement.\n')
 from histidine_ph_controls import RESULTS,CAPTION
 (OUT/'HISTIDINE_PH_CROSS_REFERENCE.txt').write_text('Histidine controls are now Figure S10A,B, with data in the two pH-control sheets of Supplementary Table S11. Previous figure-folder sources are read-only.\n')
 captions=(OUT/'CAPTIONS.md').read_text()
 start=captions.index('## Figure S10');end=captions.index('## Figure S11',start)
 (OUT/'CAPTIONS.md').write_text(captions[:start]+CAPTION+'\n\n'+captions[end:])
 (OUT/'HISTIDINE_RESULTS_INSERT.md').write_text('# Results insertion\n\nPlace after the NMR premise and response definitions, before outcome discrimination.\n\n'+RESULTS+'\n\nSubsequent references: simulated outcomes S10C; simulated pKd S10D; intensity ratios S10E.\n')
 story=OUT/'FIGURE_STORYLINE.md'
 story.write_text(story.read_text().replace('Figure S10 asks whether', 'Figure S10A,B first test the histidine pH adjustment and frozen-model agreement at pH 6.5. Figure S10C–E then ask whether'))

def main():
 for p in [FIG,TAB,SRC]:p.mkdir(parents=True,exist_ok=True)
 stale=SRC/'FigureS10_source.csv'
 if stale.exists():stale.unlink()
 d=data();figure5(d);fig_s10(d);fig_s11(d);tables(d);write_package_texts()
 manifest=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file():manifest.append({'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size})
 pd.DataFrame(manifest).to_csv(OUT/'PACKAGE_MANIFEST.csv',index=False)
 print('Prepared',OUT)

if __name__=='__main__':main()
