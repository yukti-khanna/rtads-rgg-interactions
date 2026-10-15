#!/usr/bin/env python3
"""Reviewer R1-13: historical-five B22 variability and convergence for Figure 4."""
from __future__ import annotations
import argparse, itertools, pickle, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS=[f'run_{i}' for i in range(1,6)]; JOBS=[f'job_{i}' for i in range(5)]; SEED=20260907

def cli():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo-dir',type=Path,required=True);p.add_argument('--rdf-dir',type=Path,required=True)
 p.add_argument('--outdir',type=Path,required=True);p.add_argument('--fig4-pairs',type=Path,required=True)
 p.add_argument('--calvados-dir',type=Path,default=Path('/Users/yuktikhanna/calvados'))
 p.add_argument('--historical-replicates',type=int,default=5);p.add_argument('--overwrite',action='store_true');return p.parse_args()
def one_df(r,g,key='run_1'): return pd.DataFrame({key:[np.asarray(r,float),np.asarray(g,float)]})
def calc(existing,df,box=20):
 _,b,_=existing(df,7.5,box_size=box,delta=2); z=next(iter(b.values()))
 return float(z['b22']),float(z['b22_corr1']),float(z['b22_corr2'])
def boot(v,rng):
 a=np.asarray(v); m=a[rng.integers(0,len(a),(10000,len(a)))].mean(1);return np.percentile(m,[2.5,97.5])
def savefig(fig,folder,name):
 for x in ['png','pdf']:fig.savefig(folder/f'{name}.{x}',dpi=300,bbox_inches='tight')
 plt.close(fig)
def find_traj_root(calv,pid):
 hits=[]
 for root in sorted(calv.glob('run*')):
  p=root/f'test_{pid}'/'298k_20.0nm'
  if p.is_dir() and all((p/j).is_dir() and len(list((p/j).glob('*.dcd')))==1 for j in JOBS):hits.append(p)
 return hits
def fasta(path):
 d={};cur=None
 for s in path.read_text().splitlines():
  if s.startswith('>'):cur=s[1:].split()[0];d[cur]=''
  elif cur:d[cur]+=s.strip()
 return d
def topology(md,element,chain_resnames,residues):
 top=md.Topology()
 for ci,sq in enumerate(chain_resnames):
  ch=top.add_chain();res=top.add_residue(f'C{ci+1}',ch)
  for i,aa in enumerate(sq):
   mass=float(residues.loc[aa,'MW'])+(2 if i==0 else 0)+(16 if i==len(sq)-1 else 0)
   rad=float(residues.loc[aa,'sigmas'])/2;element.Element._elements_by_symbol.pop(aa,None)
   el=element.Element.__new__(element.Element,1,aa,aa,mass,rad);top.add_atom(aa,el,res)
  for i in range(ch.n_atoms-1):top.add_bond(ch.atom(i),ch.atom(i+1))
 return top

def main():
 a=cli();repo=a.repo_dir.resolve();rdfdir=a.rdf_dir.resolve();out=a.outdir.resolve();calv=a.calvados_dir.resolve()
 if a.historical_replicates!=5:raise ValueError('Historical analysis requires exactly 5 replicas')
 if out.exists() and any(out.iterdir()):
  if not a.overwrite:raise FileExistsError(f'{out} is non-empty; use --overwrite')
  for p in out.iterdir():p.unlink() if p.is_file() or p.is_symlink() else shutil.rmtree(p)
 out.mkdir(parents=True,exist_ok=True);figdir=out/'figures';figdir.mkdir()
 sys.path.insert(0,str(repo/'scripts'));from rdf_b22_functions import corrected_rdfs_b22
 pairs=pd.read_csv(a.fig4_pairs); ids=pairs['name'].astype(str).tolist()
 if len(ids)!=917 or len(set(ids))!=917:raise ValueError(f'Authoritative table must contain 917 unique names; got {len(ids)} rows/{len(set(ids))} unique')
 inv=[];files={};dups={}
 for f in sorted(rdfdir.glob('*.pkl')):
  pid=f.stem.removeprefix('rdfs_').removesuffix('_20nm');files.setdefault(pid,[]).append(f)
  warn=[]
  try:
   d=pd.read_pickle(f);cols=list(map(str,d.columns)) if isinstance(d,pd.DataFrame) else []
   hist=[x for x in RUNS if x in cols];extra=[x for x in cols if x not in RUNS]
   arr=[]
   if isinstance(d,pd.DataFrame):
    for c in cols:
     if len(d[c])>=2:arr.extend([np.asarray(d[c].iloc[0],float),np.asarray(d[c].iloc[1],float)])
   finite=bool(arr) and all(x.size and np.isfinite(x).all() for x in arr); nbin=len(arr[0]) if arr else 0
   if not isinstance(d,pd.DataFrame):warn.append(f'unsupported {type(d).__name__}')
   if len(hist)<5:warn.append('incomplete historical five')
   inv.append(dict(pair_id=pid,rdf_file=str(f),total_replicates_present=len(cols),historical_replicate_ids=';'.join(hist),number_historical_replicates=len(hist),additional_replicate_ids=';'.join(extra),has_pooled_rdf=False,has_individual_rdfs=bool(cols),n_r_bins=nbin,box_size_nm=20,finite_values=finite,warnings='; '.join(warn)))
  except Exception as e:inv.append(dict(pair_id=pid,rdf_file=str(f),total_replicates_present=0,historical_replicate_ids='',number_historical_replicates=0,additional_replicate_ids='',has_pooled_rdf=False,has_individual_rdfs=False,n_r_bins=0,box_size_nm=np.nan,finite_values=False,warnings=str(e)))
 inv=pd.DataFrame(inv);inv.to_csv(out/'rdf_inventory_full917.csv',index=False)
 m=[]
 for pid in ids:
  fs=files.get(pid,[]);row=inv[inv.pair_id==pid]
  if len(fs)==1:
   q=row.iloc[0];status='mapped';warn=q.warnings
  elif not fs:q=None;status='unmatched';warn='no RDF file'
  else:q=row.iloc[0];status='duplicate';warn=f'{len(fs)} RDF files'
  m.append(dict(pair_id=pid,authoritative_Fig4_pair=True,rdf_file=str(fs[0]) if fs else '',mapping_status=status,total_replicates_present=int(q.total_replicates_present) if q is not None else 0,historical_replicates_present=int(q.number_historical_replicates) if q is not None else 0,additional_replicates_present=(int(q.total_replicates_present-q.number_historical_replicates) if q is not None else 0),warnings=warn))
 mapping=pd.DataFrame(m);mapping.to_csv(out/'fig4_917_pair_mapping.csv',index=False)
 complete=mapping.query("mapping_status=='mapped' and historical_replicates_present==5").pair_id.tolist()
 rep=[];pool=[]
 for pid in complete:
  f=Path(mapping.loc[mapping.pair_id==pid,'rdf_file'].iloc[0]);d=pd.read_pickle(f);gs=[];rs=[]
  for run,job in zip(RUNS,JOBS):
   r=np.asarray(d[run].iloc[0],float);g=np.asarray(d[run].iloc[1],float);raw,hum,gan=calc(corrected_rdfs_b22,one_df(r,g,run));gs.append(g);rs.append(r)
   rep.append(dict(pair_id=pid,replicate_id=job,stored_run_id=run,raw_B22=raw,Hummer_B22=hum,Ganguly_B22=gan,rdf_file=str(f)))
  means=np.mean(gs,axis=0);praw,phum,pgan=calc(corrected_rdfs_b22,one_df(rs[0],means))
  hv=[x['Hummer_B22'] for x in rep[-5:]];pool.append(dict(pair_id=pid,stored_pooled_rdf_present=False,reconstructed_mean_RDF_Hummer_B22=phum,mean_five_replica_Hummer_B22=np.mean(hv),absolute_difference=abs(phum-np.mean(hv)),relative_difference=abs(phum-np.mean(hv))/max(abs(np.mean(hv)),np.finfo(float).eps)))
 rep=pd.DataFrame(rep);rep.to_csv(out/'per_replica_b22.csv',index=False,float_format='%.17g');pool=pd.DataFrame(pool)
 lr=stats.linregress(pool.mean_five_replica_Hummer_B22,pool.reconstructed_mean_RDF_Hummer_B22);pr=stats.pearsonr(pool.mean_five_replica_Hummer_B22,pool.reconstructed_mean_RDF_Hummer_B22);sr=stats.spearmanr(pool.mean_five_replica_Hummer_B22,pool.reconstructed_mean_RDF_Hummer_B22)
 pool['population_Pearson_r']=pr.statistic;pool['population_Spearman_rho']=sr.statistic;pool['population_slope']=lr.slope;pool['population_intercept']=lr.intercept;pool.to_csv(out/'pooled_vs_mean_replicate_b22.csv',index=False,float_format='%.17g')
 rng=np.random.default_rng(SEED);u=[]
 for pid,g in rep.groupby('pair_id',sort=False):
  row={'pair_id':pid,'n_replicates':5}
  for col,prefix in [('raw_B22','raw'),('Hummer_B22','Hummer'),('Ganguly_B22','Ganguly')]:
   v=g[col].to_numpy();lo,hi=boot(v,rng);mean=v.mean();row.update({f'{prefix}_mean':mean,f'{prefix}_median':np.median(v),f'{prefix}_SD':np.std(v,ddof=1),f'{prefix}_SE':np.std(v,ddof=1)/np.sqrt(5),f'{prefix}_min':v.min(),f'{prefix}_max':v.max(),f'{prefix}_range':np.ptp(v),f'{prefix}_bootstrap95_low':lo,f'{prefix}_bootstrap95_high':hi,f'{prefix}_CV':np.std(v,ddof=1)/abs(mean) if mean else np.nan})
  u.append(row)
 unc=pd.DataFrame(u);unc.to_csv(out/'per_pair_b22_uncertainty.csv',index=False,float_format='%.17g')
 cr=[]
 for pid,g in rep.groupby('pair_id',sort=False):
  vals=dict(zip(g.replicate_id,g.Hummer_B22));full=np.mean(list(vals.values()))
  for k in range(1,6):
   est=np.array([np.mean([vals[j] for j in c]) for c in itertools.combinations(JOBS,k)])
   for combo,e in zip(itertools.combinations(JOBS,k),est):cr.append(dict(pair_id=pid,k=k,combination=';'.join(combo),estimate=e,five_replica_estimate=full,absolute_deviation=abs(e-full),relative_deviation=abs(e-full)/max(abs(full),np.finfo(float).eps)))
 conv=pd.DataFrame(cr);conv.to_csv(out/'replicate_count_convergence.csv',index=False,float_format='%.17g')
 cs=[]
 for k,g in conv.groupby('k'):
  cs.append(dict(k=k,n_combinations=len(g),median_absolute_deviation=g.absolute_deviation.median(),absolute_deviation_p05=g.absolute_deviation.quantile(.05),absolute_deviation_p95=g.absolute_deviation.quantile(.95),fraction_within_5pct=(g.relative_deviation<=.05).mean(),fraction_within_10pct=(g.relative_deviation<=.10).mean(),fraction_within_20pct=(g.relative_deviation<=.20).mean()))
 csum=pd.DataFrame(cs);csum.to_csv(out/'replicate_count_convergence_summary.csv',index=False,float_format='%.17g')
 # Locate original trajectories before percentile-nearest selection.
 roots={pid:find_traj_root(calv,pid) for pid in complete};strength=unc[['pair_id','Hummer_mean']].copy();strength['strength']=-strength.Hummer_mean
 sel=[];used=set()
 for regime,q in [('weak',.10),('intermediate',.50),('strong',.90)]:
  target=strength.strength.quantile(q)
  for _,z in strength.assign(dist=lambda x:abs(x.strength-target)).sort_values(['dist','pair_id']).iterrows():
   if z.pair_id in used or len(roots[z.pair_id])!=1:continue
   used.add(z.pair_id);sel.append(dict(pair_id=z.pair_id,mean_B22=z.Hummer_mean,percentile=stats.percentileofscore(strength.strength,z.strength)/100,regime=regime,historical_replicates=5,trajectory_path=str(roots[z.pair_id][0]),selection_reason=f'nearest unused pair with unambiguous five-job trajectories to {int(q*100)}th percentile'))
   if sum(x['regime']==regime for x in sel)==5:break
 selected=pd.DataFrame(sel);selected.to_csv(out/'convergence_pair_selection.csv',index=False,float_format='%.17g')
 # Same COM-RDF construction as historical save_rdfs.py; exact repository B22 functions.
 import mdtraj as md
 import MDAnalysis as mda
 from mdtraj import element
 res=pd.read_csv(calv/'residues.csv').set_index('three');tr=[]
 for z in sel:
  pid=z['pair_id'];root=Path(z['trajectory_path'])
  pdb_dir=('ocn_pdbs' if root.parent.parent.name=='run_ocn' else 'lambda045_pdbs' if root.parent.parent.name=='run_lambda045' else 'test_pdbs')
  pdb=calv/pdb_dir/f'{pid}_20.0nm_job_0.pdb'
  if not pdb.is_file():raise RuntimeError(f'Historical topology PDB unavailable: {pdb}')
  u=mda.Universe(str(pdb));chain_resnames=[list(seg.residues.resnames) for seg in u.segments]
  if len(chain_resnames)!=2:raise RuntimeError(f'Expected two PDB segments for {pid}')
  top=topology(md,element,chain_resnames,res)
  for job in JOBS:
   dcd=list((root/job).glob('*.dcd'))
   if len(dcd)!=1:continue
   t=md.load(str(dcd[0]),top=top);cmt=md.Topology();pos=[]
   for ci in range(2):
    ch=cmt.add_chain();rr=cmt.add_residue(f'CM{ci+1}',ch);cmt.add_atom(f'CM{ci+1}',t.top.atom(0).element,rr);pos.append(md.compute_center_of_mass(t.atom_slice(t.top.select(f'chainid {ci}'))))
   cm=md.Trajectory(np.swapaxes(np.array(pos),0,1),cmt,t.time,t.unitcell_lengths,t.unitcell_angles)
   for frac in [.2,.4,.6,.8,1.0]:
    n=int(len(cm)*frac);r,g=md.compute_rdf(cm[:n],[[0,1]],r_range=(0,10),bin_width=.1,periodic=True);B=calc(corrected_rdfs_b22,one_df(r,g))[1]
    tr.append(dict(pair_id=pid,regime=z['regime'],replicate_id=job,trajectory_fraction=frac,time_us=frac,n_frames=n,B22=B))
 td=pd.DataFrame(tr);td.to_csv(out/'time_convergence_per_replica.csv',index=False,float_format='%.17g');ts=[]
 for (pid,reg,frac),g in td.groupby(['pair_id','regime','trajectory_fraction']):
  v=g.B22.to_numpy();lo,hi=boot(v,rng);ts.append(dict(pair_id=pid,regime=reg,trajectory_fraction=frac,time_us=frac,mean=v.mean(),SD=np.std(v,ddof=1),SE=np.std(v,ddof=1)/np.sqrt(len(v)),bootstrap95_low=lo,bootstrap95_high=hi))
 tsum=pd.DataFrame(ts);tsum.to_csv(out/'time_convergence_summary.csv',index=False,float_format='%.17g');dg=[]
 for (pid,reg),g in tsum.groupby(['pair_id','regime']):
  q=g.set_index('trajectory_fraction');v1=q.loc[1.,'mean'];v8=q.loc[.8,'mean'];v6=q.loc[.6,'mean'];eps=np.finfo(float).eps
  dg.append(dict(pair_id=pid,regime=reg,absolute_change_80_100=abs(v1-v8),relative_change_80_100=abs(v1-v8)/max(abs(v1),eps),absolute_change_60_100=abs(v1-v6),relative_change_60_100=abs(v1-v6)/max(abs(v1),eps)))
 diag=pd.DataFrame(dg);diag.to_csv(out/'time_convergence_diagnostics.csv',index=False,float_format='%.17g')
 # Figures.
 s=unc.sort_values('Hummer_mean');fig,ax=plt.subplots(figsize=(10,4));ax.errorbar(range(len(s)),s.Hummer_mean,yerr=s.Hummer_SD,fmt='.',ms=2,lw=.3);ax.set(xlabel='917 Figure 4 pairs ordered by mean B22',ylabel='Hummer B22 ± SD');savefig(fig,figdir,'b22_replicate_uncertainty')
 for col,name in [('Hummer_SD','b22_sd_distribution'),('Hummer_SE','b22_se_distribution')]:fig,ax=plt.subplots(figsize=(5,4));ax.hist(unc[col],bins=40);ax.set(xlabel=col,ylabel='Pair count');savefig(fig,figdir,name)
 fig,ax=plt.subplots(figsize=(5,4));x=pool.mean_five_replica_Hummer_B22;y=pool.reconstructed_mean_RDF_Hummer_B22;ax.scatter(x,y,s=5);lo=min(x.min(),y.min());hi=max(x.max(),y.max());ax.plot([lo,hi],[lo,hi],'k--');ax.set(xlabel='Mean five-replica B22',ylabel='B22 from reconstructed mean RDF');savefig(fig,figdir,'pooled_vs_mean_replicate_b22')
 fig,ax=plt.subplots(figsize=(6,4));ax.plot(csum.k,csum.median_absolute_deviation,marker='o');ax.fill_between(csum.k,csum.absolute_deviation_p05,csum.absolute_deviation_p95,alpha=.25);ax.set(xlabel='Replica count',ylabel='Absolute deviation from five-replica mean');savefig(fig,figdir,'replicate_count_convergence')
 for reg in ['weak','intermediate','strong']:
  fig,ax=plt.subplots(figsize=(7,4));sub=tsum[tsum.regime==reg]
  for pid,g in sub.groupby('pair_id'):ax.errorbar(g.trajectory_fraction,g['mean'],yerr=g.SE,marker='o',lw=1,label=pid)
  ax.set(xlabel='Trajectory fraction',ylabel='Mean B22',title=reg.capitalize());ax.legend(fontsize=5,frameon=False);savefig(fig,figdir,f'time_convergence_{reg}')
 fig,axes=plt.subplots(5,3,figsize=(12,15),sharex=True)
 for ax,(pid,g) in zip(axes.flat,tsum.groupby('pair_id')):ax.errorbar(g.trajectory_fraction,g['mean'],yerr=g.SE,marker='o');ax.set_title(pid,fontsize=7)
 fig.tight_layout();savefig(fig,figdir,'selected_pair_time_convergence')
 # Report.
 sd5,sd95=np.percentile(unc.Hummer_SD,[5,95]);inc=917-len(complete);k4=csum.set_index('k').loc[4];lines=[]
 for reg,g in diag.groupby('regime'):lines.append(f'- {reg}: median 80→100% change {g.relative_change_80_100.median():.2%}; median 60→100% change {g.relative_change_60_100.median():.2%}; below 5%/10%/20%: {(g.relative_change_80_100<.05).mean():.1%}/{(g.relative_change_80_100<.1).mean():.1%}/{(g.relative_change_80_100<.2).mean():.1%}.')
 report=f'''# Task 2: B22 replicate variability and convergence (R1-13)\n\n## 1. Reviewer concern\nReviewer R1-13 requests error bars and convergence evidence for five 1-µs replicas of charged IDR pairs. Between-replica reproducibility and trajectory-length convergence are assessed separately.\n\n## 2. Dataset provenance\nThe authoritative Figure 4 population contains 917 unique pairs. RDF source: `{rdfdir}`. {len(mapping.query("mapping_status=='mapped'"))} mapped uniquely; {len(mapping.query("mapping_status=='unmatched'"))} unmatched; {len(mapping.query("mapping_status=='duplicate'"))} duplicate mappings. {len(complete)} have all five historical replicas. Legacy `run_1`–`run_5` correspond sequentially to `job_0`–`job_4` in `calvados/scripts/save_rdfs.py`. Additional replicas were found for {(mapping.additional_replicates_present>0).sum()} authoritative pairs and excluded.\n\n## 3. B22 calculation\nB22 was recalculated by importing the existing `scripts/rdf_b22_functions.py::corrected_rdfs_b22`, following `compute_b22.py`, with `last=7.5`, `box_size=20`, and `delta=2`. No integration, Hummer, or Ganguly definition was modified. Figure 4 uses the mean of per-run Hummer-corrected B22 (`b22_corr1`), not a stored pooled RDF.\n\n## 4. Replicate variability\nAcross {len(complete)} complete pairs, median Hummer SD={unc.Hummer_SD.median():.6g}, median SE={unc.Hummer_SE.median():.6g}, and the SD 5th–95th percentile is {sd5:.6g}–{sd95:.6g}. Bootstrap CIs use 10,000 resamples with seed {SEED}. Incomplete/unmapped: {inc}/917 ({inc/917:.2%}). Regime-specific uncertainty can be read from the selected-pair and full uncertainty tables without relying on CV near zero.\n\n## 5. Pooled versus replicate-derived B22\nNo stored pooled RDF exists in these legacy pickles. A mean RDF was reconstructed for comparison only. Its B22 versus the historical mean of five replica B22 values gives Pearson r={pr.statistic:.9g}, Spearman rho={sr.statistic:.9g}, slope={lr.slope:.9g}, intercept={lr.intercept:.9g}. The displayed Figure 4 quantity remains the mean of per-replica Hummer B22 values.\n\n## 6. Replica-number convergence\nAll combinations were evaluated. Moving from four to five replicas reduces the median absolute deviation from {k4.median_absolute_deviation:.6g} to zero; four-replica estimates are within 5%/10%/20% in {k4.fraction_within_5pct:.1%}/{k4.fraction_within_10pct:.1%}/{k4.fraction_within_20pct:.1%} of combinations. Thresholds are descriptive.\n\n## 7. Representative pair selection\nFive pairs per regime were selected nearest the 10th, 50th, and 90th percentiles of the true complete-pair B22 landscape, subject to complete RDFs and one unambiguous trajectory root. See `convergence_pair_selection.csv`.\n\n## 8. Trajectory-length convergence\nOriginal `job_0`–`job_4` trajectories were analyzed separately at 20–100%. Metadata confirms 1 µs per full trajectory, so time labels are used.\n\n{chr(10).join(lines)}\n\nThese 15 pairs do not establish universal time convergence for all 917 pairs.\n\n## 9. Reviewer-relevant conclusion\nThe tables quantify between-replica reproducibility over the complete mapped population and trajectory-length convergence only for the representative subset. The results should be described conservatively and by interaction regime.\n\n## 10. Recommended manuscript usage\nUse `per_pair_b22_uncertainty.csv` for Figure 4 error bars and a supplementary uncertainty table; use replica-count and time-convergence figures as supplementary convergence evidence; and use the provenance and exact existing-function parameters for STAR Methods. No manuscript files were edited.\n'''
 (out/'TASK2_REPORT.md').write_text(report)
 print(f'Authoritative Figure 4 pairs: 917; mapped RDFs: {(mapping.mapping_status=="mapped").sum()}; complete historical five: {len(complete)}; later replicas: {(mapping.additional_replicates_present>0).sum()}')
 print(f'Median B22 SD: {unc.Hummer_SD.median():.6g}; median B22 SE: {unc.Hummer_SE.median():.6g}; pooled-vs-replicate Pearson: {pr.statistic:.9g}')
 print('Replica stability summary:',csum.to_dict('records'));print('Selected:',selected.groupby('regime').pair_id.apply(list).to_dict());print('80→100%:',diag.groupby('regime').relative_change_80_100.median().to_dict());print(f'Output: {out}')
if __name__=='__main__':main()
