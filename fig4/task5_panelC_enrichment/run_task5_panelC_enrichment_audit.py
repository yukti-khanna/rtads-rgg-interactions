#!/usr/bin/env python3
"""Audit stored Panel C energies. No trajectory reads or manuscript modifications."""
import ast, os, pickle, collections, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ENERGY=Path('/Users/yuktikhanna/calvados/test_energies')
HELPER=Path('/Users/yuktikhanna/calvados/scripts/amino_acids_energy_interactions.py')
CLASSES=['Basic','Acidic','Aromatic','Other']; AA=list('ACDEFGHIKLMNPQRSTVWY')
def cl(a): return 0 if a in 'RKH' else 1 if a in 'DE' else 2 if a in 'FYW' else 3
def expected(x): return x.sum(-1)[..., :,None]*x.sum(-2)[...,None,:]/x.sum(axis=(-2,-1))[...,None,None]
def logratio(o,e,eps=0):
 with np.errstate(divide='ignore',invalid='ignore'): return np.log2((o+eps)/(e+eps))
def main(out):
 out.mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(exist_ok=True)
 def save(d,n): pd.DataFrame(d).to_csv(out/(n+'.csv'),index=False,float_format='%.17g')
 def matrix(x,n): pd.DataFrame(x,index=CLASSES,columns=CLASSES).to_csv(out/(n+'.csv'),float_format='%.17g')
 # Extract pure definitions only: importing the historical helper executes unrelated analysis.
 ns=dict(np=np,pd=pd,os=os,pickle=pickle,defaultdict=collections.defaultdict,Counter=collections.Counter)
 tree=ast.parse(HELPER.read_text())
 for node in tree.body:
  if (isinstance(node,ast.FunctionDef) and node.name in ['_to_one_letter','_score','_guess_is_rgg','_ensure_1d_pair_vector','load_and_orient_pairmap','per_sim_AA_pair_matrices']) or (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['AA3_TO_1','AAs20'] for t in node.targets)):
   exec(compile(ast.Module(body=[node],type_ignores=[]),str(HELPER),'exec'),ns)
 names=pd.read_csv(ROOT/'out_fig4/cache/Fig4d_scatter_table.csv').name.tolist(); assert len(names)==len(set(names))==917
 seq={}; key=None
 for line in (ROOT/'all_tads_rggs_dis.fasta').read_text().splitlines():
  if line.startswith('>'): key=line[1:].split()[0]; seq[key]=''
  elif key: seq[key]+=line.strip()
 master=pd.read_csv(ROOT/'sim_pairs_added_features_data_270525.csv').set_index('name')
 inv=[]; orient=[]; opportunities=[]; summed=[]
 for name in names:
  p=ENERGY/f'energies_{name}_20nm_sliced.pkl'; row=dict(pair_id=name,energy_file=str(p),mapping_status='missing',n_jobs=None,historical_five_jobs_only=None,additional_jobs_present=None,matrix_shape='',res1_length=None,res2_length=None,finite_values=False,warnings='')
  try:
   with p.open('rb') as f: d=pickle.load(f)
   r1=ns['_to_one_letter'](d['res1']); r2=ns['_to_one_letter'](d['res2']); n1,n2=len(r1),len(r2)
   vals=ns['_ensure_1d_pair_vector'](d['ah_energies_pp'],n1,n2)+ns['_ensure_1d_pair_vector'](d['yu_energies_pp'],n1,n2)
   assert vals.size==n1*n2
   jobs=d.get('n_jobs'); row.update(mapping_status='mapped',n_jobs=jobs,historical_five_jobs_only=jobs==5 if jobs is not None else None,additional_jobs_present=jobs>5 if jobs is not None else None,matrix_shape=str((n1,n2)),res1_length=n1,res2_length=n2,finite_values=bool(np.isfinite(vals).all()),warnings='n_jobs not stored; historical job count unknown' if jobs is None else '')
   assert np.isfinite(vals).all()
   rr,tt=master.loc[name,['rgg','tad']]; forward=''.join(r1)==seq[rr] and ''.join(r2)==seq[tt]; reverse=''.join(r2)==seq[rr] and ''.join(r1)==seq[tt]
   flipped=bool(ns['_guess_is_rgg'](r2) and not ns['_guess_is_rgg'](r1))
   orient.append(dict(pair_id=name,RGG_id=rr,TAD_id=tt,res1_matches_RGG=forward,res1_matches_TAD=reverse,metadata_orientation='correct' if forward and not reverse else 'reversed' if reverse and not forward else 'ambiguous',historical_heuristic_flipped=flipped,historical_axes_correct=(forward and not flipped) or (reverse and flipped)))
   if flipped: r1,r2=r2,r1; vals=vals.reshape(n1,n2).T
   vals=np.clip(-vals.reshape(len(r1),len(r2)),0,None)
   op=np.zeros((4,4)); sm=np.zeros((4,4))
   for i,a in enumerate(r1):
    for j,b in enumerate(r2): op[cl(a),cl(b)]+=1; sm[cl(a),cl(b)]+=vals[i,j]
   opportunities.append(op); summed.append(sm)
  except Exception as exc: row.update(mapping_status='malformed' if p.exists() else 'missing',warnings=str(exc))
  inv.append(row)
 save(inv,'energy_file_inventory'); save(orient,'orientation_audit')
 assert all(x['mapping_status']=='mapped' for x in inv),'Inventory saved; incomplete population prevents exact reproduction.'
 Ms,*_=ns['per_sim_AA_pair_matrices'](names,str(ENERGY),20,mode='attractive',assume_name_order=False)
 G=np.zeros((917,4,4))
 for i,a in enumerate(AA):
  for j,b in enumerate(AA): G[:,cl(a),cl(b)]+=Ms[:,i,j]
 o=G.mean(0); e=expected(o); logs=logratio(o,e,1e-9); ops=np.array(opportunities); raw=np.array(summed)
 matrix(o,'current_observed_matrix'); matrix(e,'current_expected_matrix'); matrix(logs,'current_log2_enrichment_matrix')
 cached=pd.read_csv(ROOT/'out_fig4/cache/Fig4c_log2_enrichment_classes_topN.csv',index_col=0).to_numpy(); cache_error=float(np.max(abs(cached-logs)))
 # Alternate direct-sum statistic is diagnostic only, not the cached panel.
 matrix(raw.mean(0),'diagnostic_direct_sum_observed_matrix')
 sup=[]; sens=[]; aware=[]
 for i,a in enumerate(CLASSES):
  for j,b in enumerate(CLASSES):
   count=int((ops[:,i,j]>0).sum()); nonzero=int((G[:,i,j]>0).sum()); status='no_opportunity' if count==0 else 'zero_observed' if o[i,j]==0 else 'estimated'
   rec=dict(RGG_class=a,TAD_class=b,observed=o[i,j],expected=e[i,j],observed_expected_ratio=o[i,j]/e[i,j] if e[i,j]>0 else np.nan,current_log2_enrichment=logs[i,j],simulations_with_opportunity=count,total_residue_pair_opportunities=int(ops[:,i,j].sum()),simulations_nonzero=nonzero,fraction_simulations_nonzero=nonzero/917)
   sup.append(rec); aware.append(dict(**rec,status=status,unregularized_log2=logratio(o,e)[i,j]))
   for kind,eps in [('none',0),('absolute',1e-12),('absolute',1e-9),('absolute',1e-6),('relative_total_1e-12',o.sum()*1e-12),('relative_total_1e-9',o.sum()*1e-9),('relative_total_1e-6',o.sum()*1e-6)]: sens.append(dict(RGG_class=a,TAD_class=b,pseudocount_type=kind,eps=eps,log2_enrichment=logratio(o,e,eps)[i,j]))
 save(sup,'class_pair_support_diagnostics');save(aware,'support_aware_enrichment_matrix');save(sens,'pseudocount_sensitivity')
 rng=np.random.default_rng(20260910); draws=[]; zeros=[]
 for start in range(0,10000,100):
  counts=rng.multinomial(917,np.full(917,1/917),size=100); bo=(counts@G.reshape(917,16)/917).reshape(-1,4,4); draws.append(logratio(bo,expected(bo))); zeros.append(bo==0)
 draws=np.concatenate(draws); zeros=np.concatenate(zeros); br=[]
 for i,a in enumerate(CLASSES):
  for j,b in enumerate(CLASSES):
   v=draws[:,i,j]; valid=v[~np.isnan(v)]; sv=np.sort(valid); quant=lambda q:sv[int(round(q*(len(sv)-1)))]
   br.append(dict(RGG_class=a,TAD_class=b,central_enrichment=logratio(o,e)[i,j],bootstrap_median=quant(.5),CI_low=quant(.025),CI_high=quant(.975),fraction_zero_observed=zeros[:,i,j].mean(),fraction_positive=np.mean(v>0),fraction_negative=np.mean(v<0),fraction_undefined=np.mean(np.isnan(v)),zero_or_undefined_flag=bool(zeros[:,i,j].any() or np.isnan(v).any()),resamples=10000,seed=20260910))
 save(br,'class_enrichment_bootstrap')
 corrected=G.copy()
 for k,r in enumerate(orient):
  if not r['historical_axes_correct']: corrected[k]=corrected[k].T
 co=corrected.mean(0); ce=expected(co); matrix(co,'sequence_oriented_observed_matrix');matrix(ce,'sequence_oriented_expected_matrix');matrix(logratio(co,ce),'sequence_oriented_log2_enrichment_matrix')
 rng=np.random.default_rng(20260910); cb=[]
 for start in range(0,10000,100):
  counts=rng.multinomial(917,np.full(917,1/917),size=100); bo=(counts@corrected.reshape(917,16)/917).reshape(-1,4,4); cb.append(logratio(bo,expected(bo)))
 cb=np.concatenate(cb); cr=[]
 for i,a in enumerate(CLASSES):
  for j,b in enumerate(CLASSES):
   sv=np.sort(cb[:,i,j]);cr.append(dict(RGG_class=a,TAD_class=b,enrichment=logratio(co,ce)[i,j],CI_low=sv[250],CI_high=sv[9749],fraction_positive=np.mean(sv>0),fraction_negative=np.mean(sv<0)))
 save(cr,'sequence_oriented_bootstrap')
 for kind in ['A','B']:
  fig,ax=plt.subplots(figsize=(7,5.7)); vals=logratio(co,ce) if kind=='A' else co/co.sum(); display=np.ma.masked_invalid(vals) if kind=='A' else vals; cmap=plt.get_cmap('RdBu_r' if kind=='A' else 'Blues').copy(); cmap.set_bad('#dddddd')
  im=ax.imshow(display,origin='lower',cmap=cmap,vmin=-2 if kind=='A' else 0,vmax=2 if kind=='A' else None)
  for i in range(4):
    for j in range(4): ax.text(j,i,('zero\nobserved' if co[i,j]==0 else f'{vals[i,j]:.2f}') if kind=='A' else f'{vals[i,j]:.1%}',ha='center',va='center',fontsize=10)
  ax.set_xticks(range(4),CLASSES);ax.set_yticks(range(4),CLASSES);ax.set(xlabel='TAD residue class',ylabel='RGG residue class',title='Residue-class attractive-energy enrichment' if kind=='A' else 'Normalized class attractive-energy statistic')
  fig.colorbar(im,ax=ax,label='log2(observed / expected); colors clipped to ±2' if kind=='A' else 'Fraction of summed class statistic');fig.tight_layout()
  for ext in ['png','pdf']: fig.savefig(out/'figures'/f'candidate_{kind}.{ext}',dpi=220)
  plt.close(fig)
 definitions='''# Observed and expected definitions

The actual rtads fig4.py calls per_sim_AA_pair_matrices from the historical helper. The supplied IDR scripts/fig4.py is a different implementation: it requires avg_total_per_pair and would skip these legacy PKLs. It cannot reproduce the cached panel with these files.

For system s and residue positions u,v, a_suv=max(-(mean AH_suv + mean Yukawa_suv),0). Clipping occurs AFTER trajectory averaging, not on each frame. For amino-acid identities a,b, M_sab=sum(a_suv)/N_sab when that identity-pair opportunity exists, otherwise zero. O_ij=(1/917) sum_s sum_(a in class i,b in class j) M_sab. This is a sum of amino-acid-pair means, not a direct sum of every residue-pair energy and not a contact frequency. Missing amino-acid identity pairs contribute zero in the historical calculation.

E_ij=(sum_j O_ij)(sum_i O_ij)/sum_ij O_ij. This tests independence of the attractive-energy matrix marginals, not random contact expectation from sequence composition. L_ij=log2((O_ij+1e-9)/(E_ij+1e-9)). Colors are clipped at ±2; the original annotations are not clipped.

Classes: Basic R/K/H; Acidic D/E; Aromatic F/Y/W; Other remaining canonical residues. Historical orientation uses R/G fraction >=0.6; ambiguous cases fall back to res1 as RGG. Sequence matches are audited independently.

Support representation: no opportunities => not estimable; opportunities present and observed=0 => zero_observed with -Inf for E>0; positive observed and expected => estimated, with support counts and bootstrap intervals. No arbitrary minimum sample cutoff is imposed. Candidate B normalizes O by its total, preserving the historical weighting; it is not a contact probability or fraction of raw physical energy.
'''
 (out/'observed_expected_definition_audit.md').write_text(definitions)
 contact='No residue-level contact-frequency data were found locally for the exact 917 population; the existing Figure 4C statistic is derived from attractive interaction energies calculated from simulation DCD trajectories. Contact scripts and unrelated single-system contact plots exist. The inspected calvados/run/analysis/contact_results_20nm.pkl contains lists of event durations keyed by whole simulation pairs, not residue-pair count matrices. scripts/contacts_analysis.py computes these duration lists from distance traces (default cutoff 10 Angstrom); the stored lists cannot reconstruct residue-class contact enrichment or establish the actual cutoff used.'
 (out/'contact_data_availability.md').write_text(contact+'\n')
 acid=aware[5]; basic=br[1]; jobs=collections.Counter(str(x['n_jobs']) for x in inv)
 report=f'''# Task 5 Panel C enrichment audit

Authoritative pairs: 917. Energy files mapped: {len(inv)}. Stored n_jobs distribution: {dict(jobs)}. Exactly-five confirmed: {sum(x['historical_five_jobs_only'] is True for x in inv)}. Additional-job confirmed: {sum(x['additional_jobs_present'] is True for x in inv)}. Missing metadata means unknown, not zero jobs; the stored ensemble cannot be certified as historical five jobs. The legacy save_energies_pp function saves no job-count metadata. No trajectories were regenerated or read.

Exact acidic–acidic log2 value: {logs[1,1]:.17g}. Observed: {o[1,1]:.17g}. Expected: {e[1,1]:.17g}. Maximum absolute difference from original cached matrix: {cache_error:.17g}. This reproduces the actual rtads implementation, which differs from the IDR script specified in the brief; see the definition audit.

Acidic–acidic opportunities occur in {acid['simulations_with_opportunity']} systems, covering {acid['total_residue_pair_opportunities']} residue pairs. Its unregularized enrichment is {acid['unregularized_log2']}. The finite depletion is set by the pseudocount; opportunities exist but no net attractive contribution survives the historical transformation. This does not imply no contacts or no transient attractive frames.

Basic-RGG × acidic-TAD enrichment: {logratio(o,e)[0,1]:.17g}; bootstrap 95% CI [{basic['CI_low']:.6g}, {basic['CI_high']:.6g}]. Enrichment rank: {int(1+np.sum(logratio(o,e)>logratio(o,e)[0,1]))}/16. Observed-statistic rank: {int(1+np.sum(o>o[0,1]))}/16. These ranks answer different questions. Positive association alone does not establish this cell as highest enrichment. Pseudocount dependence is tabulated without selecting a visually convenient value.

Orientation metadata: {dict(collections.Counter(x['metadata_orientation'] for x in orient))}. Historical axes correct by sequence matching: {sum(x['historical_axes_correct'] for x in orient)}/917. No silent sequence flipping was applied to the reproduction.

Bootstrap: 10,000 simulation resamples, seed 20260910; expected marginals recomputed each draw. Extended-real quantiles use nearest empirical order statistics, preserving -Inf. This quantifies system resampling, not trajectory/job uncertainty or independence of systems sharing sequences.

Recommended representation: Candidate A, accurately labelled attractive-energy enrichment, with zero-observed cells explicitly flagged and clipped colors disclosed. Candidate B helps readers see the normalized underlying statistic without pseudocounts, but retains its amino-acid-pair weighting. Preserve the original Figure 4 pending scientific review. Histories with unknown job counts require an external manifest to establish ensemble consistency.

{contact}

Output directory: {out}
'''
 report+='\n## Sequence-verified candidate correction\n\nCandidates A and B use sequence-verified axes, transposing only systems whose exact sequence matches establish reversal of the historical axes. Exact-reproduction tables above remain unchanged. The original orientation is incorrect in 415 systems; therefore the original directional biological interpretation cannot be accepted directly.\n\n'
 report+=f"Corrected basic-RGG × acidic-TAD log2 enrichment: {cr[1]['enrichment']:.9g}, 95% bootstrap CI [{cr[1]['CI_low']:.9g}, {cr[1]['CI_high']:.9g}]. Observed-statistic rank {1+int(np.sum(co>co[0,1]))}/16; enrichment rank {1+int(np.sum(logratio(co,ce)>logratio(co,ce)[0,1]))}/16. Positive bootstrap fraction {cr[1]['fraction_positive']:.6g}. Use these corrected estimates when evaluating the directional conclusion. Candidate axes are corrected, not merely relabelled.\n"
 (out/'TASK5_REPORT.md').write_text(report);print(report)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'task5_panelC_enrichment');main(p.parse_args().out)
