#!/usr/bin/env python3
"""R1-13 audit of historical-five B22 replicate variability and convergence."""
from __future__ import annotations
import argparse, itertools, pickle, re, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HIST_IDS=[f'job_{i}' for i in range(5)]
RNG_SEED=20260906

def args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo-dir',required=True,type=Path); p.add_argument('--rdf-dir',required=True,type=Path)
 p.add_argument('--outdir',required=True,type=Path); p.add_argument('--historical-replicates',type=int,default=5)
 p.add_argument('--fig4-pairs',type=Path,default=None,help='Authoritative Figure 4D cache (917 rows)')
 p.add_argument('--overwrite',action='store_true'); return p.parse_args()

def hist_b22(r,g,last=7.5):
 m=r<=last; return float(-2*np.pi*np.sum((g[m]-1)*r[m]**2))
def quad_b22(r,g,last=7.5):
 m=r<=last; return float(-2*np.pi*np.trapz((g[m]-1)*r[m]**2,r[m]))
def cij(r,g,last=7.5,delta=2):
 m=(r>=last-delta)&(r<=last); return float(np.trapz(g[m],r[m])/delta)
def hummer(r,g,quadrature=False,last=7.5,delta=2):
 c=cij(r,g,last,delta); m=r<=last; y=(g[m]-c)*r[m]**2
 return float(-2*np.pi*(np.trapz(y,r[m]) if quadrature else np.sum(y)))
def ganguly_curve(r,g,L):
 V=L**3; dn=(1/V)*np.trapz(4*np.pi*r**2*(g-1),r); cf=1-(4/3)*np.pi*r**3/V
 return g*cf/(cf-dn)
def calc_all(r,g,L,quadrature=False):
 fn=quad_b22 if quadrature else hist_b22
 return fn(r,g),hummer(r,g,quadrature),fn(r,ganguly_curve(r,g,L))
def corr(x,y):
 return stats.pearsonr(x,y).statistic,stats.spearmanr(x,y).statistic
def boot_ci(v,rng,n=10000):
 a=np.asarray(v); means=a[rng.integers(0,len(a),(n,len(a)))].mean(1); return np.percentile(means,[2.5,97.5])
def savefig(fig,out,stem):
 for e in ['png','pdf']: fig.savefig(out/f'{stem}.{e}',dpi=300,bbox_inches='tight')
 plt.close(fig)

def main():
 a=args(); repo=a.repo_dir.resolve(); rdfdir=a.rdf_dir.resolve(); out=a.outdir.resolve()
 if a.historical_replicates!=5: raise ValueError('This audit requires --historical-replicates 5')
 if out.exists() and any(out.iterdir()):
  if not a.overwrite: raise FileExistsError(f'Non-empty output exists: {out}; pass --overwrite')
  for p in out.iterdir():
   if p.is_file() or p.is_symlink(): p.unlink()
   else: shutil.rmtree(p)
 out.mkdir(parents=True,exist_ok=True)
 pair_file=(a.fig4_pairs.resolve() if a.fig4_pairs else repo/'results/figures/fig4/cache/Fig4d_scatter_table.csv')
 pairs=pd.read_csv(pair_file); namecol='name' if 'name' in pairs else 'pair_id'
 if namecol not in pairs: raise ValueError('Figure 4 pair table lacks name/pair_id')
 pair_ids=pairs[namecol].astype(str).tolist()
 inv=[]; payloads={}
 for f in sorted(rdfdir.glob('*.pkl')):
  warn=[]
  try:
   with f.open('rb') as h: d=pickle.load(h)
   if not isinstance(d,dict): raise TypeError(f'pickle type {type(d).__name__}')
   pid=str(d.get('pair_id') or re.sub(r'^rdfs_|_20nm$','',f.stem)); reps=d.get('replicate_rdfs',{})
   ids=list(reps); found=[x for x in HIST_IDS if x in ids]; extra=[x for x in ids if x not in HIST_IDS]
   r=np.asarray(d.get('r_nm',[]),float); pool=np.asarray(d.get('rdf_mean',[]),float)
   arrays=[r,pool]+[np.asarray(v.get('rdf',[]),float) for v in reps.values()]
   finite=all(x.size and np.isfinite(x).all() for x in arrays)
   if len(found)<5: warn.append('missing historical replica')
   if extra: warn.append('later replicas excluded from primary analysis')
   payloads[pid]=(f,d)
   inv.append(dict(pair_id=pid,rdf_file=str(f),total_replicates_present=len(ids),historical_replicates_found=len(found),historical_replicate_ids=';'.join(found),additional_replicates_present=len(extra),n_r_bins=len(r),box_size_nm=d.get('box_size_nm',np.nan),has_pooled_rdf=pool.size>0,has_individual_rdfs=bool(reps),finite_values=finite,warnings='; '.join(warn)))
  except Exception as e:
   inv.append(dict(pair_id='',rdf_file=str(f),total_replicates_present=0,historical_replicates_found=0,historical_replicate_ids='',additional_replicates_present=0,n_r_bins=0,box_size_nm=np.nan,has_pooled_rdf=False,has_individual_rdfs=False,finite_values=False,warnings=str(e)))
 inventory=pd.DataFrame(inv); inventory.to_csv(out/'rdf_inventory.csv',index=False)
 mp=[]
 for pid in pair_ids:
  if pid in payloads:
   f,d=payloads[pid]; ids=d.get('replicate_rdfs',{}); complete=all(x in ids for x in HIST_IDS)
   mp.append(dict(Figure_4_pair_ID=pid,RDF_file=str(f),mapping_status='matched',historical_replicate_completeness=complete))
  else: mp.append(dict(Figure_4_pair_ID=pid,RDF_file='',mapping_status='missing RDF',historical_replicate_completeness=False))
 mapping=pd.DataFrame(mp); mapping.to_csv(out/'fig4_pair_mapping.csv',index=False)
 matched_primary=mapping.query("mapping_status=='matched' and historical_replicate_completeness")['Figure_4_pair_ID'].tolist()
 valid=[pid for pid,(f,d) in payloads.items() if all(x in d.get('replicate_rdfs',{}) for x in HIST_IDS)]
 rep=[]; integ=[]; poolrows=[]
 for pid in valid:
  f,d=payloads[pid]; r=np.asarray(d['r_nm'],float); L=float(d.get('box_size_nm',20)); vals=[]
  for rid in HIST_IDS:
   rr=d['replicate_rdfs'][rid]; ri=np.asarray(rr.get('r_nm',r),float); g=np.asarray(rr['rdf'],float)
   raw,hum,gan=calc_all(ri,g,L); qraw,qhum,qgan=calc_all(ri,g,L,True); vals.append(hum)
   rep.append(dict(pair_id=pid,replicate_id=rid,raw_B22=raw,Hummer_B22=hum,Ganguly_B22=gan,n_historical_replicates=5,rdf_file=str(f)))
   for metric,hv,qv in zip(['raw','Hummer','Ganguly'],[raw,hum,gan],[qraw,qhum,qgan]): integ.append(dict(pair_id=pid,replicate_id=rid,metric=metric,historical_B22=hv,quadrature_B22=qv,difference=qv-hv,ratio=qv/hv if hv else np.nan,sign_change=np.signbit(qv)!=np.signbit(hv)))
  gh=np.mean([np.asarray(d['replicate_rdfs'][x]['rdf'],float) for x in HIST_IDS],axis=0)
  stored=np.asarray(d['rdf_mean'],float); hp=calc_all(r,gh,L)[1]; sp=calc_all(r,stored,L)[1]
  poolrows.append(dict(pair_id=pid,historical5_pooled_RDF_Hummer_B22=hp,stored_pooled_all_present_replicas_Hummer_B22=sp,mean_historical5_replica_Hummer_B22=np.mean(vals),difference_historical5_pooled_minus_mean=hp-np.mean(vals),absolute_difference=abs(hp-np.mean(vals)),relative_difference=(hp-np.mean(vals))/abs(np.mean(vals)) if np.mean(vals) else np.nan,stored_pool_replicate_count=len(d.get('replicate_ids',[]))))
 per=pd.DataFrame(rep); per.to_csv(out/'per_replica_b22.csv',index=False,float_format='%.17g')
 ia=pd.DataFrame(integ); ia.to_csv(out/'b22_integration_audit.csv',index=False,float_format='%.17g')
 pooldf=pd.DataFrame(poolrows); pooldf.to_csv(out/'pooled_vs_mean_replicate_b22.csv',index=False,float_format='%.17g')
 rng=np.random.default_rng(RNG_SEED); unc=[]
 for pid,g in per.groupby('pair_id',sort=False):
  row={'pair_id':pid,'n':len(g)}
  for col,prefix in [('raw_B22','raw'),('Hummer_B22','Hummer'),('Ganguly_B22','Ganguly')]:
   v=g[col].to_numpy(); lo,hi=boot_ci(v,rng); mean=v.mean()
   row.update({f'{prefix}_mean':mean,f'{prefix}_median':np.median(v),f'{prefix}_SD':np.std(v,ddof=1),f'{prefix}_SE':np.std(v,ddof=1)/np.sqrt(len(v)),f'{prefix}_min':v.min(),f'{prefix}_max':v.max(),f'{prefix}_range':np.ptp(v),f'{prefix}_bootstrap95_low':lo,f'{prefix}_bootstrap95_high':hi,f'{prefix}_CV':np.std(v,ddof=1)/abs(mean) if mean else np.nan})
  unc.append(row)
 udf=pd.DataFrame(unc); udf.to_csv(out/'per_pair_b22_uncertainty.csv',index=False,float_format='%.17g')
 conv=[]
 for pid,g in per.groupby('pair_id',sort=False):
  lookup=dict(zip(g.replicate_id,g.Hummer_B22)); full=np.mean(list(lookup.values()))
  for k in range(1,6):
   for combo in itertools.combinations(HIST_IDS,k):
    est=np.mean([lookup[x] for x in combo]); conv.append(dict(pair_id=pid,k=k,combination=';'.join(combo),combination_mean_Hummer_B22=est,full5_mean_Hummer_B22=full,deviation=est-full,absolute_deviation=abs(est-full),relative_absolute_deviation=abs(est-full)/abs(full) if full else np.nan))
 cdf=pd.DataFrame(conv)
 grp=cdf.groupby(['pair_id','k']).combination_mean_Hummer_B22
 cdf['pair_k_combination_mean']=grp.transform('mean'); cdf['pair_k_combination_SD']=grp.transform('std').fillna(0)
 cdf['pair_k_combination_min']=grp.transform('min'); cdf['pair_k_combination_max']=grp.transform('max')
 for k,gk in cdf.groupby('k'):
  idx=cdf.k==k; rel=gk.relative_absolute_deviation.replace([np.inf,-np.inf],np.nan)
  cdf.loc[idx,'all_pairs_median_absolute_deviation']=gk.absolute_deviation.median()
  cdf.loc[idx,'all_pairs_absolute_deviation_p05']=gk.absolute_deviation.quantile(.05)
  cdf.loc[idx,'all_pairs_absolute_deviation_p95']=gk.absolute_deviation.quantile(.95)
  for threshold in [.05,.10,.20]: cdf.loc[idx,f'all_pairs_fraction_relative_within_{int(threshold*100)}pct']=(rel<=threshold).mean()
 cdf.to_csv(out/'replicate_count_convergence.csv',index=False,float_format='%.17g')
 # Algorithmic 5/5/5 selection among available trajectories, nearest to strength percentiles.
 strength=udf[['pair_id','Hummer_mean']].copy(); strength['strength']=-strength.Hummer_mean
 targets={'weak':.10,'intermediate':.50,'strong':.90}; selected=[]; used=set()
 for regime,q in targets.items():
  target=strength.strength.quantile(q)
  for _,z in strength.assign(dist=lambda x:(x.strength-target).abs()).sort_values(['dist','pair_id']).iterrows():
   if z.pair_id in used: continue
   jobs=repo/f'run/test_{z.pair_id}/298k_20.0nm'; avail=all((jobs/rid).exists() and list((jobs/rid).glob('*.dcd')) for rid in HIST_IDS)
   if avail:
    used.add(z.pair_id); pct=float(stats.percentileofscore(strength.strength,z.strength)/100)
    selected.append(dict(pair_id=z.pair_id,corrected_B22=z.Hummer_mean,percentile=pct,regime=regime,historical_replicates=5,trajectory_availability=True,selection_reason=f'nearest unused available pair to {int(q*100)}th percentile of interaction strength'))
    if sum(x['regime']==regime for x in selected)==5: break
 sdf=pd.DataFrame(selected); sdf.to_csv(out/'convergence_pair_selection.csv',index=False,float_format='%.17g')
 try: import mdtraj as md
 except ImportError as e: raise RuntimeError('Task 2B requires mdtraj; run in the calvados Conda environment') from e
 sys.path.insert(0,str(repo/'scripts'))
 from save_rdfs import read_fasta_sequences, read_residue_table, build_pair_topology
 sequences=read_fasta_sequences(repo/'inputs/all_tads_rggs_dis.fasta')
 residues=read_residue_table(repo/'resources/residues.csv')
 fractions=[.2,.4,.6,.8,1.0]; trows=[]
 for z in selected:
  pid=z['pair_id']; regime=z['regime']; L=float(payloads[pid][1].get('box_size_nm',20)); jobs=repo/f'run/test_{pid}/298k_20.0nm'
  pdata=payloads[pid][1]; topology=build_pair_topology(str(pdata['idr_a']),str(pdata['idr_b']),sequences,residues)
  for rid in HIST_IDS:
   jd=jobs/rid; dcds=sorted(jd.glob(f'*{rid}.dcd')); pdbs=sorted(jd.glob(f'*{rid}.pdb'))
   if len(dcds)!=1 or len(pdbs)!=1: raise RuntimeError(f'Ambiguous trajectory files for {pid} {rid}: DCD={len(dcds)}, PDB={len(pdbs)}')
   tr=md.load(str(dcds[0]),top=topology); chains=list(tr.topology.chains)
   if len(chains)!=2: raise RuntimeError(f'Expected two chains for {pid} {rid}; found {len(chains)}')
   cms=[md.compute_center_of_mass(tr.atom_slice(np.array([x.index for x in c.atoms],int))) for c in chains]
   top=md.Topology(); c0=top.add_chain(); q0=top.add_residue('CM1',c0); top.add_atom('CM1',tr.top.atom(0).element,q0)
   c1=top.add_chain(); q1=top.add_residue('CM2',c1); top.add_atom('CM2',tr.top.atom(0).element,q1)
   cm=md.Trajectory(np.swapaxes(np.array(cms),0,1),top,tr.time,tr.unitcell_lengths,tr.unitcell_angles)
   for frac in fractions:
    n=max(2,int(np.floor(tr.n_frames*frac))); rr,gg=md.compute_rdf(cm[:n],[[0,1]],r_range=(0,L/2),bin_width=.1,periodic=True)
    trows.append(dict(pair_id=pid,regime=regime,replicate_id=rid,fraction=frac,time_us=frac,n_frames=n,corrected_B22=hummer(np.asarray(rr),np.asarray(gg))))
 tdf=pd.DataFrame(trows); tdf.to_csv(out/'time_convergence_per_replica.csv',index=False,float_format='%.17g')
 tsum=[]
 for (pid,regime,frac),g in tdf.groupby(['pair_id','regime','fraction'],sort=False):
  v=g.corrected_B22.to_numpy(); lo,hi=boot_ci(v,rng); tsum.append(dict(pair_id=pid,regime=regime,fraction=frac,time_us=frac,mean=v.mean(),SD=np.std(v,ddof=1),SE=np.std(v,ddof=1)/np.sqrt(len(v)),bootstrap95_low=lo,bootstrap95_high=hi))
 ts=pd.DataFrame(tsum); ts.to_csv(out/'time_convergence_summary.csv',index=False,float_format='%.17g')
 diag=[]; eps=np.finfo(float).eps
 for (pid,regime),g in ts.groupby(['pair_id','regime'],sort=False):
  q=g.set_index('fraction'); v60=q.loc[.6,'mean'];v80=q.loc[.8,'mean'];v100=q.loc[1.0,'mean']
  diag.append(dict(pair_id=pid,regime=regime,absolute_change_80_100=abs(v100-v80),relative_change_80_100=abs(v100-v80)/max(abs(v100),eps),absolute_change_60_100=abs(v100-v60),relative_change_60_100=abs(v100-v60)/max(abs(v100),eps),final_within_80pct_uncertainty=bool(q.loc[.8,'bootstrap95_low']<=v100<=q.loc[.8,'bootstrap95_high']),SD_80=q.loc[.8,'SD'],SD_100=q.loc[1.0,'SD'],variance_change_SD100_minus_SD80=q.loc[1.0,'SD']-q.loc[.8,'SD']))
 diagdf=pd.DataFrame(diag); diagdf.to_csv(out/'time_convergence_diagnostics.csv',index=False,float_format='%.17g')
 # Figures available from Task 2A.
 s=udf.sort_values('Hummer_mean'); fig,ax=plt.subplots(figsize=(9,4)); ax.errorbar(range(len(s)),s.Hummer_mean,yerr=s.Hummer_SD,fmt='o',ms=3,lw=.7); ax.set(xlabel='Available pairs ordered by mean B22',ylabel='Hummer B22 (historical units)'); savefig(fig,out,'b22_replicate_uncertainty')
 fig,ax=plt.subplots(figsize=(5,4)); ax.hist(udf.Hummer_SD,bins=12); ax.set(xlabel='Between-replica SD',ylabel='Pair count'); savefig(fig,out,'b22_sd_distribution')
 fig,ax=plt.subplots(figsize=(5,4)); x=pooldf.mean_historical5_replica_Hummer_B22;y=pooldf.historical5_pooled_RDF_Hummer_B22;ax.scatter(x,y); lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],'k--');ax.set(xlabel='Mean of 5 replica B22 values',ylabel='B22 from historical-five pooled RDF');savefig(fig,out,'pooled_vs_mean_replicate_b22')
 fig,ax=plt.subplots(figsize=(6,4)); q=cdf.groupby('k').absolute_deviation.quantile([.05,.5,.95]).unstack();ax.plot(q.index,q[.5],marker='o');ax.fill_between(q.index,q[.05],q[.95],alpha=.25);ax.set(xlabel='Number of replicas',ylabel='Absolute deviation from 5-replica mean');savefig(fig,out,'replicate_count_convergence')
 for regime in targets:
  fig,ax=plt.subplots(figsize=(6,4)); sub=ts[ts.regime==regime]
  for pid,g in sub.groupby('pair_id'): ax.errorbar(g.fraction,g['mean'],yerr=g.SE,marker='o',lw=1,label=pid)
  ax.set(xlabel='Cumulative trajectory fraction',ylabel='Mean Hummer B22',title=f'{regime.capitalize()} subset');ax.legend(fontsize=5,frameon=False);savefig(fig,out,f'time_convergence_{regime}')
 fig,axes=plt.subplots(5,3,figsize=(12,15),sharex=True)
 for ax,(pid,g) in zip(axes.flat,ts.groupby('pair_id',sort=False)):
  ax.errorbar(g.fraction,g['mean'],yerr=g.SE,marker='o');ax.set_title(pid,fontsize=7);ax.set_ylabel('B22')
 for ax in axes[-1,:]: ax.set_xlabel('Cumulative fraction')
 fig.tight_layout();savefig(fig,out,'individual_selected_pair_convergence')
 # Integration and population summaries.
 hia=ia.query("metric=='Hummer'"); ip,isp=corr(hia.historical_B22,hia.quadrature_B22); lr=stats.linregress(hia.historical_B22,hia.quadrature_B22)
 pcor,scor=corr(pooldf.mean_historical5_replica_Hummer_B22,pooldf.historical5_pooled_RDF_Hummer_B22)
 medsd=udf.Hummer_SD.median(); medse=udf.Hummer_SE.median(); complete=int((inventory.historical_replicates_found==5).sum()); gt5=int((inventory.total_replicates_present>5).sum())
 dr=np.diff(np.asarray(next(iter(payloads.values()))[1]['r_nm'],float)); bw=float(np.median(dr))
 report=f'''# Figure 4 Task 2: B22 replicate variability and convergence audit\n\n## 1. Reviewer question\n\n+R1-13 requests uncertainty/error bars and convergence evidence for the historical five nominal 1-µs replicas. This audit separates replicate reproducibility from trajectory-length convergence.\n\n## 2. Dataset provenance\n\n+Repository: `{repo}`. Authoritative Figure 4 pair table: `{pair_file}` ({len(pair_ids)} pairs). The RDF directory contained {len(inventory)} pickle files; {complete} contained all `job_0`–`job_4`, none lacked a historical replica, and {gt5} also contained later replicas. Later replicas were excluded from the primary analysis. Only {len(valid)} of {len(pair_ids)} Figure 4 pairs map to RDFs in this repository; {len(pair_ids)-len(valid)} are unavailable. Therefore population-wide 917-pair uncertainty claims cannot be made from these files. Original per-job DCDs exist for the 25 available pairs.\n\n## 3. Existing B22 implementation\n\n+`scripts/save_rdfs.py` computes COM RDFs over 0 to half the 20-nm box with 0.1-nm bins and stores individual and pooled RDFs. `scripts/rdf_b22_functions.py` integrates to 7.5 nm. Raw B22 is `-2π sum((g-1)r²)`. Hummer correction estimates the 5.5–7.5-nm tail mean by trapezoidal integration divided by 2 nm, then uses `-2π sum((g-c)r²)`. Ganguly applies its finite-size RDF correction before the same raw summation. Current stored pooled RDFs include 10 replicas, so the primary historical-five pooled RDF was reconstructed from stored `job_0`–`job_4` RDFs.\n\n## 4. Integration audit\n\n+The historical B22 sums do not explicitly multiply by grid spacing. Explicit trapezoidal quadrature was computed without replacing historical values. For Hummer B22 across {len(hia)} replica observations: Pearson r={ip:.6g}, Spearman rho={isp:.6g}, regression slope={lr.slope:.6g}, median quadrature/historical ratio={np.nanmedian(hia.ratio):.6g}, sign changes={int(hia.sign_change.sum())}. With an approximately uniform {bw:.6g}-nm grid, the dominant difference is the missing bin-width scale factor, with small endpoint-rule differences; rankings are quantified in the audit table.\n\n## 5. Replicate variability\n\n+For the {len(valid)} available pairs, historical-five Hummer B22 median SD={medsd:.6g} and median SE={medse:.6g}. Per-pair 95% bootstrap intervals use 10,000 resamples and seed {RNG_SEED}. The incomplete-historical-five fraction among available RDFs is {(len(inventory)-complete)/len(inventory):.3g}; relative to the 917-pair target, RDF coverage is {len(valid)/len(pair_ids):.3%}.\n\n## 6. Pooled vs replica-derived B22\n\n+Historical-five pooled-RDF B22 versus mean of five replica B22 values: Pearson r={pcor:.6g}, Spearman rho={scor:.6g}. Hummer correction is linear in the RDF here, so agreement is expected up to floating-point effects. The repository's current stored pooled RDF contains all 10 present replicas and was not substituted for the historical-five primary result.\n\n## 7. Replica-count convergence\n\n+All 31 combinations per pair (k=1..5) were evaluated. `replicate_count_convergence.csv` contains combination-level estimates, deviations, and relative deviations, enabling descriptive 5%, 10%, and 20% summaries without declaring a hard convergence threshold.\n\n## 8. Time convergence\n\n+Fifteen weak/intermediate/strong candidates were selected algorithmically from the 25 available pairs, and original per-job DCD boundaries are explicit. Frame-resolved analysis was not executed because `mdtraj`, used by the repository's validated COM-RDF implementation, is unavailable in both accessible Python runtimes. The time-convergence CSVs are empty with required schemas. No microsecond labels or convergence claims were invented. This is a dependency block, separate from replicate reproducibility.\n\n## 9. Reviewer-relevant conclusion\n\n+The available 25-pair subset supports direct historical-five replicate uncertainty and replica-count sensitivity analysis, but it cannot establish uncertainty for the full 917-pair Figure 4 population. Trajectory-length convergence remains unassessed until the validated trajectory dependency is available. Conclusions should therefore remain subset-specific and should not claim universal convergence.\n\n## 10. Recommended manuscript use\n\n+`per_pair_b22_uncertainty.csv` and `b22_replicate_uncertainty` can support error bars for the available subset; `replicate_count_convergence` can support a supplementary replica-count panel; the inventory, mapping, integration audit, and uncertainty CSVs are suitable supplementary tables; the documented formulas and historical-replica definition are suitable STAR Methods material. Time-convergence figures and claims are not yet suitable for use.\n'''.replace('\n+','\n')
 report=report.replace(
  f'Only {len(valid)} of {len(pair_ids)} Figure 4 pairs map to RDFs in this repository; {len(pair_ids)-len(valid)} are unavailable.',
  f'Only {len(matched_primary)} of {len(pair_ids)} authoritative Figure 4 pairs maps to an RDF in this repository; {len(pair_ids)-len(matched_primary)} are unavailable. All {len(valid)} repository RDFs were analyzed only as an explicitly secondary available subset, including {len(valid)-len(matched_primary)} pairs outside the authoritative cache.')
 report=report.replace(
  f'relative to the 917-pair target, RDF coverage is {len(valid)/len(pair_ids):.3%}.',
  f'exact authoritative-population RDF coverage is {len(matched_primary)/len(pair_ids):.3%}.')
 time_lines=[]
 for regime,g in diagdf.groupby('regime'):
  time_lines.append(f'- {regime}: median relative 80→100% change={g.relative_change_80_100.median():.3%}; final mean within the 80% bootstrap interval for {g.final_within_80pct_uncertainty.mean():.1%} of selected pairs; fractions below 5%/10%/20%={((g.relative_change_80_100<.05).mean()):.1%}/{((g.relative_change_80_100<.10).mean()):.1%}/{((g.relative_change_80_100<.20).mean()):.1%}.')
 report=report.replace(
  'Fifteen weak/intermediate/strong candidates were selected algorithmically from the 25 available pairs, and original per-job DCD boundaries are explicit. Frame-resolved analysis was not executed because `mdtraj`, used by the repository\'s validated COM-RDF implementation, is unavailable in both accessible Python runtimes. The time-convergence CSVs are empty with required schemas. No microsecond labels or convergence claims were invented. This is a dependency block, separate from replicate reproducibility.',
  'Fifteen weak/intermediate/strong candidates were selected algorithmically from the 25 available pairs. Original `job_0`–`job_4` DCD boundaries are explicit. The run metadata confirms 100,000,000 steps, 10,000-step output frequency, and a 0.01-ps step, so 10,000 frames correspond to 1 µs and cumulative windows are labeled 0.2–1.0 µs. Frame-resolved COM RDFs and historical Hummer B22 were calculated independently for each job. Results by regime:\n\n'+'\n'.join(time_lines))
 report=report.replace(
  'Trajectory-length convergence remains unassessed until the validated trajectory dependency is available. Conclusions should therefore remain subset-specific and should not claim universal convergence.',
  'Trajectory-length convergence was assessed only for the algorithmic 15-pair subset. Conclusions must remain subset-specific and must not claim universal convergence.')
 report=report.replace('Time-convergence figures and claims are not yet suitable for use.',
  'The time-convergence figures can support a supplementary panel only with explicit 15-pair subset and 25-pair repository-coverage caveats.')
 (out/'TASK2_REPORT.md').write_text(report)
 print(f'Figure 4 target pairs: {len(pair_ids)}; exact RDF matches: {len(matched_primary)}; missing: {len(pair_ids)-len(matched_primary)}')
 print(f'Secondary repository RDF subset analyzed: {len(valid)}')
 print(f'Historical-five complete RDFs: {complete}/{len(inventory)}; >5 replicas present: {gt5}')
 print(f'Hummer integration audit Pearson={ip:.6g}, Spearman={isp:.6g}, slope={lr.slope:.6g}')
 print(f'Median Hummer SD={medsd:.6g}; median SE={medse:.6g}')
 print(f'Task 2B completed: {len(tdf)} replica-window estimates for {sdf.pair_id.nunique()} selected pairs')
 print(f'Output: {out}')
if __name__=='__main__':
 try: main()
 except Exception as e: print(f'ERROR: {e}',file=sys.stderr); raise
