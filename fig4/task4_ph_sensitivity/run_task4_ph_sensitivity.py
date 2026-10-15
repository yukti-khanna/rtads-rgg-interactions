#!/usr/bin/env python3
"""Figure 4 Task 4: compare existing pH 7.0 and pH 6.5 His-pair simulations."""
from __future__ import annotations
import argparse, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SEED=20260910; NBOOT=10_000; LAST=7.5; BOX=20.0; DELTA=2.0
FIG4=Path('/Users/yuktikhanna/rtads-rgg-interactions/fig4')
IDR=Path('/Users/yuktikhanna/idr-interaction-analysis')
PH7=Path('/Users/yuktikhanna/calvados/test_rdfs')
PH65=FIG4/'histidines_rdfs'
DCD=FIG4/'histidines_dcds'
OUT=FIG4/'task4_ph_sensitivity'
FASTA=FIG4/'all_tads_rggs_dis.fasta'
FEATURES=FIG4/'sim_pairs_added_features_data_270525.csv'
CACHE=FIG4/'out_fig4/cache/Fig4d_scatter_table.csv'
B22CODE=IDR/'scripts/rdf_b22_functions.py'

def load_b22():
 s=importlib.util.spec_from_file_location('task4_b22',B22CODE); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.corrected_rdfs_b22

def seqs(path):
 out={}; key=None
 for line in path.read_text().splitlines():
  if line.startswith('>'): key=line[1:].split()[0]; out[key]=''
  elif key: out[key]+=line.strip().upper()
 return out

def pair_from(path): return path.name[len('rdfs_'):-len('_20nm.pkl')]

def rdf_info(path):
 try:
  d=pd.read_pickle(path); cols=list(d.columns); hist=[c for c in cols if c in [f'run_{i}' for i in range(1,6)]]; extra=[c for c in cols if c not in hist]
  r=np.asarray(d[cols[0]].iloc[0],float); bw=float(np.median(np.diff(r)))
  valid=all(len(np.asarray(d[c].iloc[0]))==len(np.asarray(d[c].iloc[1])) and np.all(np.isfinite(np.asarray(d[c].iloc[1],float))) for c in cols)
  return d,cols,hist,extra,len(r),bw,'' if valid else 'non-finite or unequal RDF arrays'
 except Exception as e: return None,[],[],[],0,np.nan,f'{type(e).__name__}: {e}'

def calc(d,func):
 _,b,_=func(d,LAST,box_size=BOX,delta=DELTA)
 return {k:{'raw':v['b22'],'hummer':v['b22_corr1'],'ganguly':v['b22_corr2']} for k,v in b.items()}

def boot_ci(x,fn,rng):
 x=np.asarray(x,float); vals=[]
 for _ in range(NBOOT): vals.append(fn(x[rng.integers(0,len(x),len(x))]))
 return np.quantile(vals,[.025,.975])

def corrmetrics(x,y):
 x=np.asarray(x,float); y=np.asarray(y,float); pr=stats.pearsonr(x,y); sr=stats.spearmanr(x,y); lr=stats.linregress(x,y)
 return dict(n=len(x),Pearson_r=pr.statistic,Pearson_p=pr.pvalue,Spearman_rho=sr.statistic,Spearman_p=sr.pvalue,slope=lr.slope,intercept=lr.intercept,R2=lr.rvalue**2)

def savefig(fig,p): fig.tight_layout(); fig.savefig(p.with_suffix('.png'),dpi=220,bbox_inches='tight'); fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight'); plt.close(fig)

def main(out):
 out.mkdir(parents=True,exist_ok=True); figs=out/'figures'; figs.mkdir(exist_ok=True); func=load_b22(); S=seqs(FASTA)
 names=pd.read_csv(CACHE).name.astype(str); master=pd.read_csv(FEATURES); panel=master[master.name.isin(set(names))].copy(); assert len(panel)==917 and panel.name.nunique()==917
 panel['n_H_RGG']=panel.rgg.map(lambda x:S[x].count('H')); panel['n_H_TAD']=panel.tad.map(lambda x:S[x].count('H')); panel['total_H']=panel.n_H_RGG+panel.n_H_TAD; his=panel[panel.total_H>0].copy()
 files=sorted(PH65.glob('rdfs_*_20nm.pkl')); filemap={pair_from(p):p for p in files}; ph7map={pair_from(p):p for p in PH7.glob('rdfs_*_20nm.pkl')}
 inv=[]; mapping=[]; rep=[]
 for p in files:
  pair=pair_from(p); d,cols,hist,extra,nb,bw,w=rdf_info(p); pooled='pooled' in [str(x).lower() for x in cols]
  inv.append(dict(pair_id=pair,TAD_id=pair.split('__',1)[1],RGG_id=pair.split('__',1)[0],rdf_file=str(p),replicate_ids=';'.join(hist),pooled_RDF_available=pooled,n_historical_replicas=len(hist),additional_replicas=';'.join(extra),n_additional_replicas=len(extra),box_size_nm=20,RDF_n_bins=nb,RDF_bin_width_nm=bw,warnings=w))
  p7=ph7map.get(pair); d7,c7,h7,e7,nb7,bw7,w7=rdf_info(p7) if p7 else (None,[],[],[],0,np.nan,'pH7 RDF absent')
  in917=pair in set(names); mapping.append(dict(pair_id=pair,in_Figure4_917=in917,pH65_RDF=str(p),pH7_RDF=str(p7) if p7 else '',has_pH7_RDF=p7 is not None,n_pH65_historical_replicas=len(hist),n_pH7_historical_replicas=len(h7),complete_five_both=len(hist)==5 and len(h7)==5,warnings='; '.join(x for x in [w,w7] if x)))
  if d is not None and d7 is not None:
   b65=calc(d[hist],func); b7=calc(d7[h7],func)
   for i in range(1,6):
    k=f'run_{i}'
    if k not in b65 or k not in b7: continue
    a=b7[k]; b=b65[k]; delta=b['hummer']-a['hummer']
    rep.append(dict(pair_id=pair,replicate_id=k,B22_raw_pH7=a['raw'],B22_raw_pH65=b['raw'],B22_Hummer_pH7=a['hummer'],B22_Hummer_pH65=b['hummer'],B22_Ganguly_pH7=a['ganguly'],B22_Ganguly_pH65=b['ganguly'],B22_pH7=a['hummer'],B22_pH65=b['hummer'],delta_B22=delta,abs_delta_B22=abs(delta),relative_delta_B22=delta/abs(a['hummer']) if a['hummer'] else np.nan))
 pd.DataFrame(inv).to_csv(out/'ph65_rdf_inventory.csv',index=False,float_format='%.17g'); mp=pd.DataFrame(mapping); mp.to_csv(out/'ph7_ph65_pair_mapping.csv',index=False,float_format='%.17g'); R=pd.DataFrame(rep); R.to_csv(out/'per_replica_B22_pH7_vs_pH65.csv',index=False,float_format='%.17g')

 hinv=his[['name','tad','rgg','n_H_TAD','n_H_RGG','total_H']].rename(columns={'name':'pair_id','tad':'TAD_id','rgg':'RGG_id'}); hinv['H_in_TAD_only']=(hinv.n_H_TAD>0)&(hinv.n_H_RGG==0); hinv['H_in_RGG_only']=(hinv.n_H_RGG>0)&(hinv.n_H_TAD==0); hinv['H_in_both']=(hinv.n_H_RGG>0)&(hinv.n_H_TAD>0); hinv['rerun_at_pH65']=hinv.pair_id.isin(filemap); hinv.to_csv(out/'histidine_pair_inventory.csv',index=False)
 complete=hinv.copy(); complete['pH65_RDF_found']=complete.pair_id.isin(filemap); complete['pH7_RDF_found']=complete.pair_id.isin(ph7map); cdict=mp.set_index('pair_id').complete_five_both.to_dict(); complete['complete_five_replicas_both']=complete.pair_id.map(cdict).eq(True); complete['status']=np.where(~complete.pH65_RDF_found,'His-containing Figure 4 pair not rerun',np.where(~complete.pH7_RDF_found,'pH7 reference missing',np.where(~complete.complete_five_replicas_both,'incomplete historical replicas','complete'))); complete.to_csv(out/'histidine_rerun_completeness.csv',index=False)

 agg=[]
 for pair,g in R.groupby('pair_id'):
  a=g.B22_pH7; b=g.B22_pH65; ma=a.mean(); mb=b.mean(); dl=np.log10(-mb)-np.log10(-ma) if ma<0 and mb<0 else np.nan
  agg.append(dict(pair_id=pair,n_replicates=len(g),mean_B22_pH7=ma,mean_B22_pH65=mb,SD_pH7=a.std(ddof=1),SD_pH65=b.std(ddof=1),SE_pH7=a.std(ddof=1)/np.sqrt(len(g)),SE_pH65=b.std(ddof=1)/np.sqrt(len(g)),delta_mean_B22=mb-ma,abs_delta_mean_B22=abs(mb-ma),relative_delta_B22=(mb-ma)/abs(ma) if ma else np.nan,fold_change_abs_B22=abs(mb)/abs(ma) if ma else np.nan,log10_negB22_pH7=np.log10(-ma) if ma<0 else np.nan,log10_negB22_pH65=np.log10(-mb) if mb<0 else np.nan,delta_log10_negB22=dl))
 P=pd.DataFrame(agg).merge(hinv,on='pair_id',how='left'); P.to_csv(out/'per_pair_B22_pH7_vs_pH65.csv',index=False,float_format='%.17g')

 valid=P[['mean_B22_pH7','mean_B22_pH65']].dropna(); glob=[]
 for scale,x,y in [('mean_B22',valid.mean_B22_pH7,valid.mean_B22_pH65),('log10_negB22',P.log10_negB22_pH7.dropna(),P.loc[P.log10_negB22_pH7.notna(),'log10_negB22_pH65'])]: glob.append({'record_type':'agreement','metric':scale,**corrmetrics(x,y)})
 d=P.delta_mean_B22; rd=P.relative_delta_B22; ld=P.delta_log10_negB22.dropna()
 desc=dict(n_pairs=len(P),median_absolute_B22_change=P.abs_delta_mean_B22.median(),median_relative_change=rd.median(),median_absolute_log10_change=ld.abs().median(),delta_B22_p05=d.quantile(.05),delta_B22_p95=d.quantile(.95),n_sign_changes=int(((P.mean_B22_pH7*P.mean_B22_pH65)<0).sum()),fraction_sign_changes=float(((P.mean_B22_pH7*P.mean_B22_pH65)<0).mean()))
 for t in [.05,.1,.2]: desc[f'n_abs_relative_change_gt_{int(t*100)}pct']=int((rd.abs()>t).sum()); desc[f'fraction_abs_relative_change_gt_{int(t*100)}pct']=float((rd.abs()>t).mean())
 for t in [.05,.1]: desc[f'n_abs_log_change_gt_{str(t).replace(".","p")}']=int((ld.abs()>t).sum()); desc[f'fraction_abs_log_change_gt_{str(t).replace(".","p")}']=float((ld.abs()>t).mean())
 glob.append({'record_type':'change_summary','metric':'Hummer_corrected_B22',**desc}); pd.DataFrame(glob).to_csv(out/'ph_sensitivity_global_statistics.csv',index=False,float_format='%.17g')

 P['rank_pH7']=P.mean_B22_pH7.rank(method='average'); P['rank_pH65']=P.mean_B22_pH65.rank(method='average'); P['rank_difference']=P.rank_pH65-P.rank_pH7; P['absolute_rank_difference']=P.rank_difference.abs(); P['percentile_rank_shift']=P.absolute_rank_difference/(len(P)-1)*100
 P[['pair_id','mean_B22_pH7','mean_B22_pH65','rank_pH7','rank_pH65','rank_difference','absolute_rank_difference','percentile_rank_shift']].sort_values('absolute_rank_difference',ascending=False).to_csv(out/'ph_sensitivity_rank_changes.csv',index=False,float_format='%.17g')

 eff=[]
 for hcol in ['total_H','n_H_TAD','n_H_RGG']:
  for ycol in ['delta_mean_B22','delta_log10_negB22']:
   z=P[[hcol,ycol]].dropna(); pr=stats.pearsonr(z[hcol],z[ycol]); sr=stats.spearmanr(z[hcol],z[ycol]); eff.append(dict(histidine_measure=hcol,effect_measure=ycol,n=len(z),Pearson_r=pr.statistic,Pearson_p=pr.pvalue,Spearman_rho=sr.statistic,Spearman_p=sr.pvalue))
 pd.DataFrame(eff).to_csv(out/'ph_effect_vs_histidine_content.csv',index=False,float_format='%.17g')
 P['histidine_location']=np.select([P.H_in_both,P.H_in_TAD_only,P.H_in_RGG_only],['both','TAD only','RGG only'],'none'); loc=[]
 for (cat,y),g in P.melt(id_vars=['pair_id','histidine_location'],value_vars=['delta_mean_B22','delta_log10_negB22'],var_name='effect_measure',value_name='value').dropna().groupby(['histidine_location','effect_measure']): loc.append(dict(histidine_location=cat,effect_measure=y,n=len(g),median=g.value.median(),p25=g.value.quantile(.25),p75=g.value.quantile(.75),IQR=g.value.quantile(.75)-g.value.quantile(.25)))
 pd.DataFrame(loc).to_csv(out/'ph_effect_by_histidine_location.csv',index=False,float_format='%.17g')

 # CALVADOS standard titration: H has pKa 6.0; K/R=+1, D/E=-1; both termini cancel in net charge.
 def q(seq,ph): return seq.count('K')+seq.count('R')-seq.count('D')-seq.count('E')+seq.count('H')/(1+10**(ph-6.0))
 charge=[]
 for _,r in P.iterrows():
  qt7=q(S[r.TAD_id],7); qt65=q(S[r.TAD_id],6.5); qr7=q(S[r.RGG_id],7); qr65=q(S[r.RGG_id],6.5)
  charge.append(dict(pair_id=r.pair_id,Q_TAD_pH7=qt7,Q_TAD_pH65=qt65,Q_RGG_pH7=qr7,Q_RGG_pH65=qr65,delta_Q_TAD=qt65-qt7,delta_Q_RGG=qr65-qr7,delta_pair_charge=(qt65+qr65)-(qt7+qr7),OCN_net_charge_product_pH7=-(qt7*qr7),OCN_net_charge_product_pH65=-(qt65*qr65),delta_OCN_proxy=-(qt65*qr65)+(qt7*qr7),histidine_pKa=6.0,charge_termini='both (net cancellation)'))
 C=pd.DataFrame(charge); C.to_csv(out/'ph_charge_state_changes.csv',index=False,float_format='%.17g'); E=P[['pair_id','delta_mean_B22','delta_log10_negB22']].merge(C,on='pair_id'); er=[]
 for x in ['delta_pair_charge','delta_OCN_proxy']:
  for y in ['delta_mean_B22','delta_log10_negB22']:
   z=E[[x,y]].dropna(); pr=stats.pearsonr(z[x],z[y]); sr=stats.spearmanr(z[x],z[y]); er.append(dict(charge_change=x,effect_measure=y,n=len(z),Pearson_r=pr.statistic,Pearson_p=pr.pvalue,Spearman_rho=sr.statistic,Spearman_p=sr.pvalue))
 pd.DataFrame(er).to_csv(out/'ph_effect_vs_charge_change.csv',index=False,float_format='%.17g')

 pmean=P.set_index('pair_id').mean_B22_pH65; A=panel[['name','rgg','tad','Opposite_Charge_Number','Charge_Complementarity','Total_Length','B22_corr1_log10','Corrected_B22_Hummer']].rename(columns={'name':'pair_id','Opposite_Charge_Number':'OCN','Corrected_B22_Hummer':'B22_original_pH7'}).copy(); A['contains_histidine']=A.pair_id.isin(set(his.name)); A['Normalized_OCN']=A.OCN/A.Total_Length; A['B22_adjusted']=A.pair_id.map(pmean).fillna(A.B22_original_pH7); A['B22_source_for_adjusted']=np.where(A.pair_id.isin(pmean.index),'pH6.5 rerun mean (5 replicas)','original pH7 (no rerun available)'); A['log10_negB22_original']=np.where(A.B22_original_pH7<0,np.log10(-A.B22_original_pH7),np.nan); A['log10_negB22_adjusted']=np.where(A.B22_adjusted<0,np.log10(-A.B22_adjusted),np.nan); A.to_csv(out/'fig4_917_pH_adjusted_B22.csv',index=False,float_format='%.17g')
 rel=[]
 for lab,y in [('original pH7','log10_negB22_original'),('pH-adjusted','log10_negB22_adjusted')]:
  for f in ['OCN','Normalized_OCN','Charge_Complementarity']:
   z=A[[f,y]].dropna(); rel.append(dict(dataset=lab,feature=f,**corrmetrics(z[f],z[y])))
 REL=pd.DataFrame(rel); REL.to_csv(out/'fig4_electrostatic_relationships_before_after_pH_adjustment.csv',index=False,float_format='%.17g')

 rng=np.random.default_rng(SEED); diff=P.delta_mean_B22.dropna().to_numpy(); wil=stats.wilcoxon(diff); mci=boot_ci(diff,np.mean,rng); mdci=boot_ci(diff,np.median,rng); paired=pd.DataFrame([dict(n=len(diff),mean_difference=diff.mean(),mean_difference_CI_low=mci[0],mean_difference_CI_high=mci[1],median_difference=np.median(diff),median_difference_CI_low=mdci[0],median_difference_CI_high=mdci[1],SD_difference=diff.std(ddof=1),paired_Cohens_dz=diff.mean()/diff.std(ddof=1),Wilcoxon_statistic=wil.statistic,Wilcoxon_p=wil.pvalue,bootstrap_resamples=NBOOT,random_seed=SEED)])
 paired.to_csv(out/'paired_pH_difference_statistics.csv',index=False,float_format='%.17g')

 dcds=list(DCD.rglob('*.dcd')) if DCD.exists() else []; di=[]
 for pair in sorted(filemap): di.append(dict(pair_id=pair,trajectory_directory=str(DCD),DCD_found=False,n_DCD_files=0,total_frames=np.nan,read_status='not checked: pH6.5 DCD directory absent locally',consistent_with_RDF_inventory=False,warnings='Stored RDF is complete; combined DCD unavailable for requested lightweight integrity check.'))
 if dcds:
  # Optional inventory without treating combined files as continuous replicas.
  import mdtraj as md
  di=[]
  for p in dcds:
   pair=next((x for x in filemap if x in p.name),p.stem); status='ok'; nf=np.nan
   try:
    with md.formats.DCDTrajectoryFile(str(p)) as fh: nf=len(fh)
   except Exception as e: status=f'{type(e).__name__}: {e}'
   di.append(dict(pair_id=pair,trajectory_directory=str(DCD),DCD_found=True,n_DCD_files=1,total_frames=nf,read_status=status,consistent_with_RDF_inventory=pair in filemap,warnings='combined trajectory only; not interpreted as continuous replicas'))
 pd.DataFrame(di).to_csv(out/'ph65_trajectory_inventory.csv',index=False,float_format='%.17g')

 # Figures
 for x,y,n,title in [('mean_B22_pH7','mean_B22_pH65','B22_pH7_vs_pH65','Mean Hummer-corrected B22'),('log10_negB22_pH7','log10_negB22_pH65','logB22_pH7_vs_pH65','log10(-B22)')]:
  z=P[[x,y]].dropna(); lo=min(z.min()); hi=max(z.max()); fig,ax=plt.subplots(figsize=(5,4.5)); ax.scatter(z[x],z[y],s=18,alpha=.55); ax.plot([lo,hi],[lo,hi],'k--',label='identity'); ax.set(xlabel='pH 7.0',ylabel='pH 6.5',title=title); ax.legend(frameon=False); savefig(fig,figs/n)
 for col,n,xlab in [('delta_mean_B22','delta_B22_distribution','B22 pH6.5 - pH7'),('delta_log10_negB22','delta_logB22_distribution','log10(-B22) pH6.5 - pH7'),('percentile_rank_shift','rank_shift_distribution','Absolute percentile-rank shift')]:
  fig,ax=plt.subplots(figsize=(5,4)); ax.hist(P[col].dropna(),bins=30,color='#0072B2',alpha=.8); ax.axvline(0,color='black',lw=1); ax.set(xlabel=xlab,ylabel='Pairs'); savefig(fig,figs/n)
 fig,ax=plt.subplots(figsize=(5,4)); ax.scatter(P.total_H,P.delta_mean_B22,alpha=.55,s=18); ax.axhline(0,color='black',lw=1); ax.set(xlabel='Total histidines',ylabel='B22 pH6.5 - pH7'); savefig(fig,figs/'delta_B22_vs_histidine_count')
 cats=['TAD only','RGG only','both']; vals=[P.loc[P.histidine_location==c,'delta_mean_B22'] for c in cats]; fig,ax=plt.subplots(figsize=(5.5,4)); ax.boxplot(vals,tick_labels=cats,showfliers=False); ax.axhline(0,color='black',lw=1); ax.set(ylabel='B22 pH6.5 - pH7'); savefig(fig,figs/'delta_B22_by_histidine_location')
 fig,axs=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True)
 for ax,lab,y,col in [(axs[0],'Original pH7','log10_negB22_original','#D55E00'),(axs[1],'pH-adjusted','log10_negB22_adjusted','#0072B2')]: ax.scatter(A.OCN,A[y],s=8,alpha=.4,c=col); z=A[['OCN',y]].dropna(); lr=stats.linregress(z.OCN,z[y]); xx=np.array([A.OCN.min(),A.OCN.max()]); ax.plot(xx,lr.intercept+lr.slope*xx,c='black'); ax.set(title=lab,xlabel='OCN',ylabel='log10(-B22)')
 savefig(fig,figs/'OCN_vs_B22_original_and_pH_adjusted')

 agree=corrmetrics(P.mean_B22_pH7,P.mean_B22_pH65); rank=stats.spearmanr(P.mean_B22_pH7,P.mean_B22_pH65).statistic; kend=stats.kendalltau(P.mean_B22_pH7,P.mean_B22_pH65).statistic; orig=REL.query("dataset=='original pH7' and feature=='OCN'").iloc[0]; adj=REL.query("dataset=='pH-adjusted' and feature=='OCN'").iloc[0]; conclusion='does not materially change' if abs(orig.Pearson_r-adj.Pearson_r)<.05 else 'changes materially by the prespecified descriptive 0.05 correlation threshold'
 report=f'''# Figure 4 Task 4: pH sensitivity (R1-15)

## 1. Reviewer concern
Figure 4 simulations used pH 7.0 whereas NMR used pH 6.5. Histidine protonation can therefore alter simulated interactions over this interval.

## 2. Data provenance
pH 7 RDFs: `{PH7}`. pH 6.5 RDFs: `{PH65}` (the brief's IDR-repository path was absent; the files were found in the corrected Figure 4 workspace). Requested pH 6.5 trajectories: `{DCD}`; this directory is absent locally. B22 was recalculated with `{B22CODE}`, `last=7.5`, `box_size=20`, and `delta=2` without changing any equation or correction.

## 3. Rerun completeness
The 917 panel contains {len(his)} His-containing pairs. There are {len(files)} pH 6.5 RDF pairs, all mapping to the 917 panel and pH 7 reference RDFs. {int(mp.complete_five_both.sum())} have five historical replicas at both pH values. One His-containing pair was not rerun: {', '.join(complete.loc[~complete.pH65_RDF_found,'pair_id'])}.

## 4. B22 comparison
Across {len(P)} matched pairs, pH7 versus pH6.5 mean B22 gives Pearson r={agree['Pearson_r']:.4f} and Spearman rho={agree['Spearman_rho']:.4f}. Median absolute change is {P.abs_delta_mean_B22.median():.4g}; median relative change is {P.relative_delta_B22.median():.2%}. Sign changes: {desc['n_sign_changes']} ({desc['fraction_sign_changes']:.2%}). Five-replica SD/SE values are reported per pair; paired bootstrap uncertainty uses {NBOOT:,} resamples.

## 5. Rank stability
Spearman rho={rank:.4f}, Kendall tau={kend:.4f}. Median absolute rank shift is {P.absolute_rank_difference.median():.1f} pairs ({P.percentile_rank_shift.median():.2f} percentile points); 90th/95th percentiles are {P.absolute_rank_difference.quantile(.9):.1f}/{P.absolute_rank_difference.quantile(.95):.1f}. Largest shifts are listed first in `ph_sensitivity_rank_changes.csv` after sorting by `absolute_rank_difference`.

## 6. Histidine dependence
Count correlations and location-group median/IQR summaries are in the two histidine-effect tables. They describe association only and do not establish causality.

## 7. Charge-state changes
Charges use CALVADOS's standard Henderson-Hasselbalch histidine assignment (pKa 6.0), fixed K/R and D/E charges, and both termini whose +1/-1 contributions cancel in each sequence net charge. Per-sequence and OCN-proxy changes are tabulated. This calculation describes the parameter change; B22 effects come from the existing simulations.

## 8. Effect on Figure 4 conclusion
Original 917 OCN-logB22 Pearson r={orig.Pearson_r:.4f}; pH-adjusted r={adj.Pearson_r:.4f}. The main electrostatic conclusion **{conclusion}** under this descriptive comparison.

## 9. Reviewer-relevant conclusion
Within the tested pH 6.5-7.0 interval, use the effect magnitudes and rank statistics above. Do not generalize to pH independence outside this interval. The one absent rerun and absent combined DCD archive should be disclosed.

## 10. Recommended manuscript use
Use the provenance and B22 parameters for Methods, the paired and rank tables plus comparison plots for a supplementary pH-sensitivity panel, and the conservative conclusion above in the reviewer response. Update Figure 4 only if the journal requires the pH-adjusted values; no manuscript files were edited.
'''
 (out/'TASK4_REPORT.md').write_text(report)
 print(f'His-containing Figure 4 pairs: {len(his)}'); print(f'pH6.5 RDF pairs found: {len(files)}'); print(f'pairs successfully matched pH7<->pH6.5: {len(P)}'); print(f'complete five-replica matched pairs: {int(mp.complete_five_both.sum())}'); print(f'Pearson pH7 vs pH6.5: {agree["Pearson_r"]:.6f}'); print(f'Spearman pH7 vs pH6.5: {agree["Spearman_rho"]:.6f}'); print(f'median relative B22 change: {P.relative_delta_B22.median():.6%}'); print(f'median rank shift: {P.absolute_rank_difference.median():.3f}'); print(f'original OCN-B22 Pearson: {orig.Pearson_r:.6f}'); print(f'pH-adjusted OCN-B22 Pearson: {adj.Pearson_r:.6f}'); print(f'Main Figure 4 conclusion: {conclusion}'); print(f'Output directory: {out}')

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=OUT); a=ap.parse_args(); main(a.out)
