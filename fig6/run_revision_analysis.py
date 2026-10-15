#!/usr/bin/env python3
"""Figure 6 reviewer analyses. Never imports Excel or changes curated input.
Run with calvados Python: run_revision_analysis.py --stage all
Stages: inventory, plots, trajectories, report, all. Trajectories resume hashed caches.
"""
from pathlib import Path
import argparse, json, hashlib, collections, importlib.util
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'revision_analysis'
PRED='Pred_log10_negB22_final'; SIM='Sim_log10_negB22_final'
SEED=20261007
STYLE={'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42}
plt.rcParams.update(STYLE)

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(d,name):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False);return p

def data():
 from validation_master import MasterBuilder,load_config
 cfg=load_config('validation_sources.json')
 cfg['experiments']['path']='fig6_parent_master.csv'
 d=MasterBuilder(cfg,outdir=str(OUT/'master')).build()
 d['Outcome']=d.Outcome.replace({'Kd fit':'Kd_fit'})
 for c in ['AHD_parent','RGG_parent']:
  d[c]=d[c].astype(str)
 return d

def export(fig,name,frame,caption,stats=None):
 p=OUT/'figures'/name;p.parent.mkdir(parents=True,exist_ok=True)
 fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(p.with_suffix('.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig);frame.to_csv(p.with_suffix('.csv'),index=False)
 p.with_suffix('.caption.txt').write_text(caption+'\n')
 if stats is not None:stats.to_csv(p.parent/(name+'_statistics.csv'),index=False)

def auc(y,s):
 y=np.asarray(y,bool);s=np.asarray(s,float)
 if not y.any() or y.all():return np.nan
 return float((rankdata(s)[y].sum()-y.sum()*(y.sum()+1)/2)/(y.sum()*(~y).sum()))
def concordance(y,s):
 y=np.asarray(y,float);s=np.asarray(s,float);i,j=np.triu_indices(len(y),1);dy=y[i]-y[j];ds=s[i]-s[j];ok=dy!=0
 return float(np.mean((dy[ok]*ds[ok]>0)+.5*(ds[ok]==0))) if ok.any() else np.nan

def boot(d,ycol,scol,group,fn):
 rng=np.random.default_rng(SEED);groups=list(d.groupby(group).indices.values());vals=[]
 for _ in range(2000):
  ix=np.concatenate([groups[k] for k in rng.integers(len(groups),size=len(groups))]);z=d.iloc[ix];v=fn(z[ycol].values,z[scol].values)
  if np.isfinite(v):vals.append(v)
 lo,hi=np.percentile(vals,[2.5,97.5]) if vals else (np.nan,np.nan)
 return dict(CI95_low=lo,CI95_high=hi,valid_resamples=len(vals),requested_resamples=2000,cluster=group,seed=SEED)
def metrics(y,z):
 y=np.asarray(y,bool);z=np.asarray(z,bool);tp=int((y&z).sum());tn=int((~y&~z).sum());fp=int((~y&z).sum());fn=int((y&~z).sum())
 se=tp/(tp+fn) if tp+fn else np.nan;sp=tn/(tn+fp) if tn+fp else np.nan
 return dict(TP=tp,TN=tn,FP=fp,FN=fn,sensitivity=se,specificity=sp,balanced_accuracy=(se+sp)/2)
def threshold(y,s):
 u=np.unique(s);ts=np.r_[u[0]-1e-8,(u[1:]+u[:-1])/2,u[-1]+1e-8]
 scores=np.array([metrics(y,s>=t)['balanced_accuracy'] for t in ts]);best=ts[np.isclose(scores,np.nanmax(scores))]
 return float(best[len(best)//2]),ts,scores

def ppi(d):
 ids={'lef1_nter':'Q9UJU2','foxo4_cr3':'P98177','foxo6_cr3':'A8MYZ6','atf4_tad':'P18848','tfe3_cter':'P19532','tp53':'P04637','foxm1':'Q08050','sox12':'O15370','thoc4_nter':'Q86V81','thoc4_cter':'Q86V81','taf15_rgg':'Q92804','g3bp1_rgg':'Q13283','rbmx':'P38159','rab44':'Q7Z6P3','tceal6':'Q6IPX3','snrnp70_s':'P08621','q8n319':'Q8N319'}
 path=ROOT.parent/'fig1/input/ppis.txt';edges=collections.Counter()
 for line in path.open():
  t=line.split()
  if len(t)>=2:edges[tuple(sorted(t[:2]))]+=1
 source=Path('/Users/yuktikhanna/Documents/pendrive/Documents/RBP-TF-main/human_annotated_PPIs_unids_w_methods_evidence_nopred.txt')
 records=collections.defaultdict(list)
 if source.exists():
  for line in source.open():
   t=line.rstrip('\n').split('\t')
   if len(t)>=4 and 'exp' in [s.strip() for s in t[3].split(';')]:records[tuple(sorted(t[:2]))].append(t)
 save(pd.DataFrame([dict(path=str(path),sha256=sha(path)),dict(path=str(source),sha256=sha(source) if source.exists() else '',status='available' if source.exists() else 'missing annotated evidence')]),'tables/ppi_source_provenance.csv')
 spec=importlib.util.spec_from_file_location('assay',ROOT.parent/'fig1/scripts/r2_06_assay_sensitivity.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 rows=[];support=[]
 for _,r in d[~d.is_shuffle & d.experimental_status.ne('no experimental measurement supplied')].iterrows():
  a,b=ids.get(r.AHD_name),ids.get(r.RGG_name);key=tuple(sorted([a,b])) if a and b else None
  rec=records.get(key,[])
  rows.append(dict(pair_id=r.pair_id,AHD_uniprot=a,RGG_uniprot=b,ppi_present=bool(edges.get(key,0)),source_records=edges.get(key,0),status='protein-level association in this dataset' if edges.get(key,0) else 'not found in this dataset',methods='; '.join(sorted({x[2] for x in rec})),evidence_scope='full protein; no fragment or shuffle attribution'))
  for z in rec:support.append(dict(pair_id=r.pair_id,proteinA=z[0],proteinB=z[1],methods=z[2],evidence=z[3],source=str(source)))
 save(pd.DataFrame(rows),'tables/ppi_evidence.csv');save(pd.DataFrame(support),'tables/ppi_source_records.csv')
 # Reuse the existing assay classification rules when exposed by the Figure 1 module.
 if support:
  methods=sorted({m.strip() for z in support for m in z['methods'].split(';') if m.strip()})
  out=[]
  for m in methods:
   category,rule,rationale=mod.classify_assay(m)
   out.append(dict(method=m,classification=category,rule=rule,rationale=rationale))
  save(pd.DataFrame(out),'tables/ppi_assay_classification.csv')

def inventory(d):
 paths=[ROOT/'fig6_parent_master.csv',ROOT/'preds_val_updated/predictions_minimal.csv',ROOT.parent/'fig2/inputs/tables/xrgg_AHD_parent_master.csv',ROOT.parent/'fig1/input/ppis.txt',ROOT.parent/'fig4/energy_funcs.py',ROOT.parent/'fig4/per_pair_energies.py']
 save(pd.DataFrame([dict(path=str(p),sha256=sha(p)) for p in paths]),'tables/input_hashes.csv')
 # Verify shuffled amino-acid composition and retain both sequences.
 ix=d.set_index('pair_id');rows=[]
 for _,r in d[d.is_shuffle].iterrows():
  parent=r.parent_pair_id
  ok=parent in ix.index and collections.Counter(r.AHD_sequence)==collections.Counter(ix.loc[parent].AHD_sequence) and r.RGG_sequence==ix.loc[parent].RGG_sequence
  rows.append(dict(pair_id=r.pair_id,parent_pair_id=parent,composition_verified=ok,experimental_row=r.experimental_status!='no experimental measurement supplied',AHD_sequence=r.AHD_sequence,WT_sequence=ix.loc[parent].AHD_sequence if parent in ix.index else ''))
 save(pd.DataFrame(rows),'tables/shuffle_composition_audit.csv')
 # Descriptor rows are simulation/prediction constructs, not assumed experimental boundaries.
 rows=[]
 bounds={'tp53':(1,94),'foxm1':(685,801),'foxo4_cr3':(461,505),'foxo6_cr3':(418,472),'lef1_nter':(1,101),'atf4_tad':(41,141),'tfe3_cter':(475,575),'sox12':(57,72)}
 for _,r in d.iterrows():
  a=dict(pair_id=r.pair_id,AHD_name=r.AHD_name,RGG_name=r.RGG_name)
  for role in ['AHD','RGG']:
   seq=r[role+'_sequence'];length=len(seq);charge=sum(seq.count(x) for x in 'KR')-sum(seq.count(x) for x in 'DE')
   a.update({role+'_length':length,role+'_nominal_charge_KR_DE':charge,role+'_nominal_charge_density':charge/length,role+'_RG_per_residue':seq.count('RG')/length,role+'_aromatic_fraction_FWY':sum(seq.count(x) for x in 'FWY')/length,role+'_sequence_sha256':hashlib.sha256(seq.encode()).hexdigest()})
  for col in ['minus_NCP','SHD_RGG','SHD_AHD','local_charge_complementarity_std','scd_RGG_patch_std','scd_AHD_patch_std','ah_energy_patch_std']:a[col]=r.get(col,np.nan)
  a['nominal_minus_NCP']=-a['AHD_nominal_charge_KR_DE']*a['RGG_nominal_charge_KR_DE'];b=bounds.get(r.AHD_parent)
  a['documented_AHD_bounds']=f'{b[0]}-{b[1]}' if b else '';a['AHD_length_matches_documented_bounds']=a['AHD_length']==b[1]-b[0]+1 if b else False
  a['boundary_evidence']='manuscript construct table; length agreement does not verify experimental sample sequence'
  rows.append(a)
 save(pd.DataFrame(rows),'tables/sequence_descriptors_and_boundary_audit.csv')
 ppi(d)
 from revision_trajectories import inventory as trajectory_inventory
 trajectory_inventory(d)

def discrimination(d):
 stats=[];thresholds=[];folds=[];members=[]
 for name,pos in [('Kd_vs_NB',['Kd_fit']),('Kd_LB_vs_NB',['Kd_fit','LB']),('LB_vs_NB',['LB'])]:
  valid=d.Outcome.isin(pos+['NB']) & np.isfinite(d[PRED]) & np.isfinite(d[SIM])
  mem=d[['pair_id','Outcome']].copy();mem['panel']=name;mem['included']=valid;mem['reason']=np.where(valid,'included','outcome outside comparison or score unavailable');members.append(mem)
  z=d[valid].copy();z['positive']=z.Outcome.isin(pos)
  fig,ax=plt.subplots(figsize=(5.2,4.5));figt,axt=plt.subplots(figsize=(5.2,4.5));panelstats=[]
  for label,col,color in [('Predicted',PRED,'#0072B2'),('Simulated',SIM,'#D55E00')]:
   y=z.positive.values;s=z[col].values;v=auc(y,s);t,ts,sc=threshold(y,s)
   points=[metrics(y,s>=tt) for tt in ts[::-1]]
   line,=ax.plot([1-p['specificity'] for p in points],[p['sensitivity'] for p in points],label=f'{label}: AUC = {v:.3f}',c=color)
   axt.plot(ts,sc,label=label,c=color);axt.axvline(t,c=color,ls=':',lw=.8)
   thresholds.append(dict(comparison=name,series=label,threshold=t,evaluation='in-sample exploratory',**metrics(y,s>=t)))
   for group in ['AHD_parent','RGG_parent']:
    ci=boot(z,'positive',col,group,auc);row=dict(comparison=name,series=label,n=len(z),positive=int(sum(y)),negative=int(sum(~y)),AUROC=v,positive_negative_concordance=v,**ci);stats.append(row);panelstats.append(row)
    if group=='AHD_parent':line.set_label(f'{label}: {v:.3f} [{ci["CI95_low"]:.3f}, {ci["CI95_high"]:.3f}]')
    pred=np.full(len(z),np.nan)
    for g in z[group].unique():
     hold=z[group].eq(g).values
     if len(np.unique(y[~hold]))<2:continue
     tt,_,_=threshold(y[~hold],s[~hold]);pred[hold]=s[hold]>=tt
     for k in np.flatnonzero(hold):folds.append(dict(comparison=name,series=label,grouping=group,held_out=g,pair_id=z.iloc[k].pair_id,threshold=tt,observed=int(y[k]),predicted=int(pred[k])))
    ok=np.isfinite(pred);thresholds.append(dict(comparison=name,series=label,threshold=np.nan,evaluation='leave-'+group+'-out',n_evaluated=int(ok.sum()),**metrics(y[ok],pred[ok].astype(bool))))
  ax.plot([0,1],[0,1],'k:',lw=.8);ax.set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1.02),title=f'{name.replace("_"," ")} (n={len(z)})');ax.legend(loc='lower right',frameon=False)
  caption='Curated outcomes only; all fits retained. AUROC equals positive-negative concordance with half credit for ties. Cluster-bootstrap intervals use 2000 AHD-parent and RGG-parent resamples separately. Selected panel; shared constructs limit generalization. Simulation associations outside the audited subset are exploratory.'
  export(fig,'ROC_'+name,z,caption,pd.DataFrame(panelstats))
  axt.set(xlabel=r'Cutoff on $\log_{10}(-B_{22})$',ylabel='In-sample balanced accuracy',title='Exploratory threshold sensitivity');axt.legend(frameon=False)
  curve=[]
  for label,col in [('Predicted',PRED),('Simulated',SIM)]:
   _,ts,sc=threshold(z.positive.values,z[col].values)
   curve.extend(dict(series=label,threshold=t,balanced_accuracy=v) for t,v in zip(ts,sc))
  export(figt,'threshold_'+name,pd.DataFrame(curve),'Cutoffs maximize in-sample balanced accuracy; no universal binding boundary is inferred. Held-family-out performance is exported separately.')
 save(pd.concat(members),'tables/discrimination_membership.csv');save(pd.DataFrame(stats),'tables/discrimination_statistics.csv');save(pd.DataFrame(thresholds),'tables/threshold_summary.csv');save(pd.DataFrame(folds),'tables/threshold_heldout_predictions.csv')
 # Audited simulation-only sensitivity on the same rows for both scores.
 rows=[]
 for name,pos in [('Kd_vs_NB',['Kd_fit']),('Kd_LB_vs_NB',['Kd_fit','LB']),('LB_vs_NB',['LB'])]:
  z=d[d.Outcome.isin(pos+['NB']) & d.simulation_accuracy_eligible.astype(str).str.lower().eq('true')].copy()
  for label,col in [('Predicted',PRED),('Simulated',SIM)]:rows.append(dict(comparison=name,series=label,n=len(z),AUROC=auc(z.Outcome.isin(pos),z[col]),scope='existing audited simulation subset'))
 save(pd.DataFrame(rows),'tables/audited_subset_discrimination.csv')

def rankstats(d):
 rows=[]
 for panel,ycol,sel in [('pKd','pKd',d.Outcome.eq('Kd_fit')),('intensity','I_ratio_mean_below_threshold',d.I_ratio_mean_below_threshold.notna())]:
  z=d[sel & np.isfinite(d[PRED]) & np.isfinite(d[SIM])].copy()
  save(z,f'tables/{panel}_rank_members.csv')
  for label,col in [('Predicted',PRED),('Simulated',SIM)]:
   for metric,fun in [('Spearman',lambda y,s: float(spearmanr(y,s).statistic) if len(np.unique(y))>1 and len(np.unique(s))>1 else np.nan),('concordance',concordance)]:
    for group in ['AHD_parent','RGG_parent']:
     # Low intensity corresponds to greater response; reverse only concordance direction.
     temp=z.copy()
     if panel=='intensity' and metric=='concordance':temp[ycol]=-temp[ycol]
     rows.append(dict(panel=panel,series=label,n=len(z),metric=metric,estimate=fun(temp[ycol],temp[col]),**boot(temp,ycol,col,group,fun)))
 save(pd.DataFrame(rows),'tables/rank_statistics.csv')

def context(d):
 src=ROOT.parent/'fig2/inputs/tables/xrgg_AHD_parent_master.csv';s=pd.read_csv(src)
 s=s[s.AHD_variant.eq('WT') & s.XRGG_partner.eq('[RRGG]7') & s.AHD_gene.isin(['LEF1','FOXO4','FOXO6','TP53','FOXM1'])].sort_values('Kd_uM').copy()
 s['assay_context']='synthetic RRGG; do not pool with natural partner Kd';save(s,'tables/synthetic_reference.csv')
 fig,ax=plt.subplots(figsize=(5.7,3.4))
 for i,(_,r) in enumerate(s.iterrows()):ax.errorbar(r.Kd_uM,i,xerr=r.Kd_error_uM,fmt='o',color='#D55E00' if r.AHD_gene=='LEF1' else '#0072B2',capsize=3)
 ax.set(yticks=range(len(s)),yticklabels=s.AHD_gene,xscale='log',xlabel=r'[RRGG]$_7$ $K_d$ (µM)')
 export(fig,'synthetic_reference_Kd',s,'Five WT reference constructs. Error bars are supplied TITAN fit errors. LEF1 is weakest among these five. Figure 1 Methods report different initial-screen buffers for FOXM1 versus the other original seeds; quantitative-run condition equivalence requires laboratory confirmation.')
 csp=ROOT.parent/'fig2/outputs/fig1/Fig1A_XRGG_CSP_matrix_source.csv'
 if csp.exists():save(pd.read_csv(csp),'tables/reference_CSP_matrix.csv')
 lef=d[d.AHD_parent.eq('lef1_nter')].copy().sort_values(['RGG_name','is_shuffle']);fig,ax=plt.subplots(figsize=(8,5))
 labels=[]
 for i,(_,r) in enumerate(lef.iterrows()):
  ax.plot([r[PRED],r[SIM]],[i,i],color='#bbbbbb');ax.scatter(r[PRED],i,c='#0072B2',marker='o');ax.scatter(r[SIM],i,c='#D55E00',marker='^')
  labels.append(r.pair_display_name+' | '+(str(r.Outcome) if pd.notna(r.Outcome) else 'not supplied'))
 ax.scatter([],[],c='#0072B2',label='Predicted');ax.scatter([],[],c='#D55E00',marker='^',label='Simulated');ax.legend(frameon=False);ax.set(yticks=range(len(lef)),yticklabels=labels,xlabel=r'$\log_{10}(-B_{22})$');export(fig,'LEF1_natural_context',lef,'All LEF1 computational pairs retained, including unmeasured and shuffle rows. NB is an assay outcome, not zero simulated attraction.')
 z=d[~d.is_shuffle & d.RGG_name.isin(lef.RGG_name)].copy();partners=sorted(z.RGG_name.unique())
 ncols=min(3,max(1,int(np.ceil(np.sqrt(len(partners))))));nrows=max(1,int(np.ceil(len(partners)/ncols)))
 fig,axs=plt.subplots(nrows,ncols,figsize=(4.3*ncols,3.4*nrows),squeeze=False)
 for ax in axs.flat[len(partners):]:ax.axis('off')
 for ax,partner in zip(axs.flat,partners):
  g=z[z.RGG_name.eq(partner)].sort_values('AHD_name');xx=np.arange(len(g));ax.scatter(xx,g[PRED],label='Predicted',c='#0072B2');ax.scatter(xx,g[SIM],label='Simulated',c='#D55E00',marker='^');ax.set(title=partner,xticks=xx,xticklabels=g.AHD_name.str.replace('_cr3|_tad|_nter|_cter','',regex=True));ax.tick_params(axis='x',rotation=65)
  for i,(_,r) in enumerate(g.iterrows()):
   if r.AHD_name=='lef1_nter':ax.axvspan(i-.3,i+.3,color='#F0E442',alpha=.25,zorder=0)
 axs.flat[0].legend(fontsize=7);fig.supylabel(r'$\log_{10}(-B_{22})$');export(fig,'matched_AHD_scores',z,f'All available WT AHDs against the {len(partners)} LEF1 partners currently represented; yellow shading identifies LEF1. Independent of experimental outcome.')

def shuffles(d):
 audit=pd.read_csv(OUT/'tables/shuffle_composition_audit.csv');z=d[d.is_shuffle & d.experimental_status.ne('no experimental measurement supplied')].copy();ix=d.set_index('pair_id');rows=[]
 for _,r in z.iterrows():
  passed=audit.set_index('pair_id').loc[r.pair_id,'composition_verified']
  if not passed:continue
  w=ix.loc[r.parent_pair_id];a=dict(pair_id=r.pair_id,parent_pair_id=r.parent_pair_id,label=r.pair_display_name,WT_outcome=w.Outcome,shuffle_outcome=r.Outcome)
  for col in [PRED,SIM]:a['WT_'+col]=w[col];a['shuffle_'+col]=r[col];a['delta_'+col]=r[col]-w[col]
  a['experimental_delta_status']='not calculated: paired condition equivalence not recorded'
  rows.append(a)
 out=pd.DataFrame(rows);fig,axs=plt.subplots(1,2,figsize=(12,6),sharex=True,sharey=True)
 for ax,col,title in zip(axs,[PRED,SIM],['Predicted','Simulated']):
  for i,r in out.iterrows():ax.plot([r['WT_'+col],r['shuffle_'+col]],[i,i],c='#aaaaaa');ax.scatter(r['WT_'+col],i,facecolor='white',edgecolor='#0072B2');ax.scatter(r['shuffle_'+col],i,c='#D55E00')
  ax.set(title=title,xlabel=r'$\log_{10}(-B_{22})$');ax.scatter([],[],facecolor='white',edgecolor='#0072B2',label='WT');ax.scatter([],[],c='#D55E00',label='Shuffle');ax.legend(frameon=False)
 axs[0].set_yticks(range(len(out)),[r.label+' | '+str(r.WT_outcome)+' → '+str(r.shuffle_outcome) for _,r in out.iterrows()]);export(fig,'WT_shuffle_comparison',out,'Exact matched partner; composition checked. WT open, shuffle filled. Outcomes are not placed in an affinity order. Missing experimental deltas are not inferred.')

def plots(d):
 from plot_master_validation import MasterFigures
 from validation_master import load_config
 cfg=load_config('validation_sources.json');cfg['diagnostics_dir']=str(OUT/'diagnostics')
 dd=d.copy();dd['Outcome']=dd.Outcome.replace({'Kd_fit':'Kd fit'})
 MasterFigures(dd,cfg,str(OUT)).run(diagnostics=False)
 discrimination(d);rankstats(d);context(d);shuffles(d)

def report(d):
 p=OUT/'REVIEWER_RESPONSE_DRAFT.md'
 text='''# NMR validation reviewer response draft

This draft distinguishes completed analyses from remaining experimental work. Raw labels are ignored; curated outcomes and numeric Kd values are authoritative.

## R1-18 and R1-03: LEF1, seed coverage and scope

LEF1 binds synthetic [RRGG]7 with Kd 44.292 ± 4.422 µM, the weakest of the five reference WT constructs. Its WT-to-shuffle Kd changes to 64.372 ± 6.089 µM in the synthetic-repeat dataset. The supplied quantitative CSP matrix likewise shows the smallest RRGG response for LEF1 among the four initial seeds. These different observables are not interchangeable. Together they demonstrate heterogeneity among seed constructs rather than a uniform class of strong binders.

The original four natural-partner LEF1 outcomes remain NB, whereas the expanded panel includes LEF1–RAB44 LB and a qualitative-binding THOC4-N shuffle. This supports partner and sequence-context dependence. It does not establish that the weak repeat-peptide affinity causes natural-partner NB. Differences in construct boundaries and assay conditions must remain explicit. p53 and FOXM1 are now represented, with pending measurements shown as pending.

## R2-31: discrimination and rank concordance

All supplied fitted Kd values are retained, including p53–G3BP1 and FOXO4–G3BP1. Primary discrimination compares Kd fits with NB; secondary comparisons pool Kd fits with LB or assess LB alone. LB is not assigned an ordinal affinity rank. AUROC is the positive-negative concordance with half credit for score ties. Confidence intervals resample AHD parents and RGG parents separately; shared constructs and targeted selection limit generalization. Quantitative affinity comparisons include Spearman and pairwise concordance with grouped-bootstrap uncertainty. Exploratory cutoffs are evaluated using held-family-out threshold selection, not presented as universal binding boundaries. NB has no invented numeric Kd or censoring bound.

## R2-29: composition-preserving shuffles

The paired panel uses all experimentally tracked shuffles passing exact composition checks, matched to WT against the same RGG partner. Prediction and simulation changes are plotted with curated outcomes. A computational change is not an experimentally established affinity change. Paired experimental deltas require matching measurements and condition records.

## R2-28: prior corroboration

The PPI audit reports archived protein-level associations and assay labels separately from fragment-level binding. LEF1–ALYREF/THOC4, p53–ALYREF/THOC4, p53–G3BP1 and FOXM1–RBMX are present in the inspected edge list. THOC4 N and C share a protein accession but remain separate constructs. Missing edges are described as not found in this dataset, never proof of novelty or noninteraction. Selection rationale cannot be retrospectively inferred from experimental outcomes.

## R2-25/27 and R1-19: constructs and mechanism

Frozen current predictions and B22 values are reused. The trajectory audit binds analysis to sequence-checked topology and serialized simulation force parameters where available. Energy components are model-dependent descriptions of sampled configurations, not additive B22 contributions or experimental mechanism proofs. Full-panel simulation associations are exploratory where historical provenance remains unresolved; the audited-subset discrimination table is retained separately.

## Outstanding experimental/source work

Highest-titration bound fractions require absolute protein and ligand concentrations and an appropriate binding model. Eight equivalents alone is insufficient. Spectrum-arrow verification requires the original spectrum/concentration ordering. Displayed Kd should be rounded to uncertainty-supported precision while source values retain full precision. Exact experimental boundary and identity discrepancies must be resolved against laboratory construct records. These tasks are not reported as completed. Manuscript DOCX files were not edited.
'''
 for fn in ['discrimination_statistics.csv','rank_statistics.csv','threshold_summary.csv']:
  f=OUT/'tables'/fn
  if f.exists():text+='\n## '+fn+'\n\n```text\n'+pd.read_csv(f).to_string(index=False)+'\n```\n'
 f=OUT/'trajectories/inventory.csv'
 if f.exists():text+='\n## Trajectory execution status\n\n'+pd.read_csv(f).status.value_counts().to_string()+'\n'
 f=OUT/'trajectories/summary.csv'
 if f.exists():text+='\nCompleted trajectory pairs: '+str(len(pd.read_csv(f)))+'. See per-pair provenance and energy verification tables.\n'
 p.write_text(text)
 imgs=list((OUT/'figures').glob('*.png'))+list((OUT/'supplementary_figures').glob('*.png'))
 if imgs:
  fig,axs=plt.subplots(int(np.ceil(len(imgs)/3)),3,figsize=(15,4*np.ceil(len(imgs)/3)))
  for ax in np.asarray(axs).flat:ax.axis('off')
  for ax,im in zip(np.asarray(axs).flat,imgs):ax.imshow(plt.imread(im));ax.set_title(im.stem,fontsize=8)
  fig.tight_layout();fig.savefig(OUT/'revision_overview.png',dpi=150);plt.close(fig)

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=['inventory','plots','trajectories','report','all'],default='all');ap.add_argument('--run-root',type=Path,action='append',help='Additional archived simulation root containing test_PAIR/298k_20.0nm/job_* folders');args=ap.parse_args()
 if args.run_root:
  import revision_trajectories
  revision_trajectories.RUNROOTS.extend(p.expanduser().resolve() for p in args.run_root)
 OUT.mkdir(exist_ok=True);before=sha(ROOT/'fig6_parent_master.csv');d=data()
 if args.stage in ['inventory','all']:inventory(d)
 if args.stage in ['plots','all']:
  if not (OUT/'tables/shuffle_composition_audit.csv').exists():inventory(d)
  plots(d)
 if args.stage in ['trajectories','all']:
  from revision_trajectories import run
  run(d)
 report(d)
 assert before==sha(ROOT/'fig6_parent_master.csv'),'Curated input changed'
 print('Completed',args.stage,'in',OUT,flush=True)
