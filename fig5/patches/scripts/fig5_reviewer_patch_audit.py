#!/usr/bin/env python3
"""Figure 5 feature-only audit. Never loads targets, trains models or writes inputs."""
import argparse, ast, hashlib, json, os, sys, re, platform
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def local(c):
    return bool(re.search(r'patch[12]_',c) or c.startswith(('ah_energy_','ah_density_','opp_charge_','hydropathy_diff_','charge_mismatch_')) or c in ['top_patch_gap','count_strong_patches','corr_oppcharge_ah'])

def read(p,cols=None):
    return pd.read_csv(p,usecols=cols,float_precision='round_trip')

def save(df,name): df.to_csv(out/name,index=False)

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--patch-root',type=Path,default=Path(__file__).resolve().parents[1])
ap.add_argument('--fig5-root',type=Path,required=True)
ap.add_argument('--predictor-root',type=Path,required=True)
ap.add_argument('--output',type=Path)
ap.add_argument('--inventory-only',action='store_true')
a=ap.parse_args(); root=a.patch_root.resolve(); fig=a.fig5_root.resolve(); out=a.output or root/'outputs/fig5_reviewer_patch_audit'
out.mkdir(parents=True,exist_ok=True)
if (out/'patch_audit_summary.txt').exists(): raise SystemExit('Refusing to overwrite completed audit; provide a new --output.')
paths=dict(training=fig/'sim_pairs_pred_with_hybrid_sel_cols_with_old.csv',heldout=fig/'taken_out_preds_with_hybrid_021125_sel_cols.csv',landscape=fig/'unsim_preds_with_hybrid_021125_sel_cols.csv')
ids={k:set(read(p,['name']).name) for k,p in paths.items()}
refs='Opposite_Charge_Number Normalized_OCN charge_RGG charge_TAD NCPR_RGG NCPR_TAD Average_NCPR Charge_Complementarity RGG_length TAD_length Total_Length SCD_RGG SCD_TAD Average_SCD SHD_RGG SHD_TAD Average_SHD lambda_RGG lambda_TAD Average_lambda Hydropathy_Diff Lambda_Ratio Average_Interaction Min_lambda Average_sigma Average_charge Average_Aromatics Charge_Asymmetry'.split()
inv=[]; overlaps=[]; tabmap=[]; patchfiles=[]; union=set()
for p in sorted(root.rglob('*.csv')):
    if 'outputs' in p.relative_to(root).parts: continue
    cols=list(read(p,lambda c:False).columns) if False else list(pd.read_csv(p,nrows=0).columns)
    names=read(p,['name']) if 'name' in cols else None
    n=len(names) if names is not None else sum(len(x) for x in pd.read_csv(p,chunksize=10000))
    tabmap.append(f'{p.relative_to(root)}: {n} x {len(cols)}; modified {datetime.fromtimestamp(p.stat().st_mtime).isoformat()}')
    if not any(local(c) for c in cols): continue
    patchfiles.append(p); s=set(names.name) if names is not None else set(); union |= s
    matches={k:len(s&v) for k,v in ids.items()}
    inv.append(dict(path=str(p),filename=p.name,rows=n,columns=len(cols),pair_identifier='name' if names is not None else '',RGG__TAD=bool(s) and all(str(x).count('__')==1 for x in s),unique_pair_ids=len(s),duplicates=n-len(s),row_type='raw patch-patch' if 'patch1_index' in cols else 'summarized protein pair',parent_dataset='; '.join(f'{k}: {v}' for k,v in matches.items()),patch_columns=sum(map(local,cols)),sha256=sha(p)))
    for k,v in ids.items(): overlaps.append(dict(patch_file=str(p),dataset=k,reference_pairs=len(v),patch_unique_pairs=len(s),overlap=len(s&v),reference_missing=len(v-s),patch_outside_reference=len(s-v)))
for k,v in ids.items(): overlaps.append(dict(patch_file='UNION_ALL_CANDIDATES',dataset=k,reference_pairs=len(v),patch_unique_pairs=len(union),overlap=len(union&v),reference_missing=len(v-union),patch_outside_reference=len(union-v)))
save(pd.DataFrame(inv),'patch_file_inventory.csv');save(pd.DataFrame(overlaps),'pair_overlap_summary.csv')
selected=[root/'datasets/features_sim_pairs_for_training.csv',root/'datasets/features_for_training.csv',root/'features/one_pair_for_training_added_features_data.csv',root/'scripts/plot_features.py',root/'scripts/get_features.py',root/'scripts/energy_funcs.py',root/'scripts/merge_datasets.py',root/'features/residues.csv',root/'features/all_tads_rggs_dis.fasta',*paths.values()]
hashes={str(p):sha(p) for p in selected}
maptext=['PROJECT MAP (written before feature analysis)',f'Output: {out}',
'patches/scripts: historical generators, merging/filtering, plotting and legacy predictors. New audit code lives here.',
'patches/features: 80 computed shards, example schema, residue table, FASTA and batch submission scripts.',
'patches/datasets: assembled landscape, simulation and unsimulated subsets, B22-merged subset.',
'patches/datsets: empty historical misspelled output directory.',
'patches/preds and figs: historical predictions and figures; not used as primary sources.',
'fig5: current authoritative pair/global CSVs; fig5.py confirms filenames. Panels, supplementary figures and prior global audit are retained.',
'idr-interaction-predict: scripts (feature generation/training/inference), models/pretrained, models_old, config, resources, inputs and sample_rdfs. No models loaded.',
'No pre-existing outputs directory was present in these three scoped roots; using patches/outputs with its existing scripts structure.',
'Authoritative tables:']
for k,p in paths.items(): maptext.append(f'{k}: {p}; {len(ids[k])} unique pairs x {len(pd.read_csv(p,nrows=0).columns)} columns')
maptext+=['PATCH CSV INPUTS',*tabmap,'SELECTED INPUT SHA256',*(f'{v}  {k}' for k,v in hashes.items()),'RECURSIVE WORKSPACE TREE (all entries except .git and caches; no analysis of unrelated figures)']
for base in [fig.parent,a.predictor_root,root]:
    maptext.append(str(base))
    for dp,ds,fs in os.walk(base):
        ds[:]=sorted(d for d in ds if d not in ['.git','__pycache__','.DS_Store','node_modules'])
        rel=Path(dp).relative_to(base)
        maptext.append(f'  {rel}/ ({len(fs)} files)')
        maptext.extend(f'    {f}' for f in sorted(fs))
(out/'project_structure.txt').write_text('\n'.join(maptext))
(out/'input_provenance.txt').write_text('\n'.join([f'Python {sys.version}',f'numpy {np.__version__}; pandas {pd.__version__}', 'Target and prediction columns never loaded into audit frames. Exact pair IDs only; no normalization or reversed-name matching.', 'Primary: training subset plus full patch landscape for membership/copy checks. All candidate CSVs inventoried; prediction exports are secondary copies.',*(f'{v}  {k}' for k,v in hashes.items())]))
print('Inventory overlaps:',[(k,len(v&union),len(v)) for k,v in ids.items()],flush=True)
if a.inventory_only: sys.exit(0)
# Feature analysis starts only after the project map has been written.
train_path=root/'datasets/features_sim_pairs_for_training.csv'
header=list(pd.read_csv(train_path,nrows=0).columns); features=[c for c in header if local(c)]
P=read(train_path,['name',*features]).set_index('name').sort_index()
G=read(paths['training'],lambda c:c=='name' or c in refs).set_index('name').loc[P.index]
assert len(P)==917 and P.index.is_unique and set(P.index)==ids['training']
# Check all copies without using any B22 or model output values.
copy_checks=[]
for p in patchfiles:
    if not any(r['patch_file']==str(p) and r['dataset']=='training' and r['overlap']>0 for r in overlaps): continue
    h=list(pd.read_csv(p,nrows=0).columns); shared=[c for c in features if c in h]
    d=read(p,['name',*shared]).set_index('name'); common=P.index.intersection(d.index)
    if d.index.is_unique and len(common):
        x=P.loc[common,shared].to_numpy();y=d.loc[common,shared].to_numpy()
        close=np.isclose(x,y,rtol=1e-10,atol=1e-12,equal_nan=True)
        copy_checks.append(dict(file=str(p),matched_training_pairs=len(common),features=len(shared),mismatching_cells=int((~close).sum()),max_abs_difference=float(np.nanmax(abs(x-y)))))
save(pd.DataFrame(copy_checks),'patch_copy_consistency.csv')
# Function-only AST execution prevents historical top-level file writes, RDF work and plotting.
from scipy.integrate import quad
def read_fasta(path):
    seqs={}; key=None
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            key=line[1:].split()[0];seqs[key]=''
        elif key is not None: seqs[key]+=line.strip()
    return seqs

def functions_only(path,env,names=None):
    tree=ast.parse(path.read_text());nodes=[]
    for n in tree.body:
        if isinstance(n,ast.FunctionDef) and (names is None or n.name in names):
            n.decorator_list=[];nodes.append(n)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),env)

rp=read(root/'features/residues.csv').set_index('one')
seqs=read_fasta(root/'features/all_tads_rggs_dis.fasta')
env=dict(np=np,pd=pd,quad=quad,residue_properties=rp,findall=re.findall)
functions_only(root/'scripts/energy_funcs.py',env,['lj_potential','ah_potential','ah_scaled'])
functions_only(root/'scripts/plot_features.py',env)
ah=env['make_ah_intgrl_map'](rp);env['ah_intgrl_map']=ah
# Memoize unchanged pure historical single-window functions, reducing repeated work.
from functools import lru_cache
for fn in ['calc_SHD','calc_SCD','rg_fraction','aromatic_fraction','rgg_motif_count','sequence_entropy']:
    env[fn]=lru_cache(maxsize=None)(env[fn])
# Use three deterministic training pairs covering the training OCN distribution.
ordered=G.sort_values(['Opposite_Charge_Number'],kind='stable').index
examples=[ordered[i] for i in [0,len(ordered)//2,len(ordered)-1]]
checks=[];algebra=[];raws={}
for name in examples:
    r,t=name.split('__');raw=env['patch_features'](seqs[r],seqs[t],rp,ah,patch_size=5);raws[name]=raw
    sm=env['summarize_patch_features'](raw)
    for c,v in sm.items():
        if c in P:
            stored=P.loc[name,c]
            checks.append(dict(pair=name,source='plot_features.py',feature=c,stored=stored,recomputed=v,absolute_error=abs(stored-v),agrees=bool(np.isclose(stored,v,rtol=1e-9,atol=1e-11,equal_nan=True))))
    q1=np.array([rp.loc[list(p),'q'].sum() for p in env['get_patches'](seqs[r])]);q2=np.array([rp.loc[list(p),'q'].sum() for p in env['get_patches'](seqs[t])])
    algebra.append(dict(pair=name,patch_pairs=len(raw),local_ocn_product_max_error=float(abs(raw.opp_charge.to_numpy()+np.outer(q1,q2).ravel()).max()),ah_density_rescaling_max_error=float(abs(raw.ah_energy-25*raw.ah_density).max()),ocn_mean_weighted_product_error=abs(raw.opp_charge.mean()+q1.mean()*q2.mean()),ocn_variance_factorization_error=abs(raw.opp_charge.var(ddof=0)-(np.mean(q1*q1)*np.mean(q2*q2)-q1.mean()**2*q2.mean()**2))))
# Verify optimized generator independently for the same small examples, including extra kappa/top5.
from collections import Counter
aas=list(rp.index)
e2=dict(np=np,pd=pd,quad=quad,Counter=Counter,aas=aas,aa_to_idx={x:i for i,x in enumerate(aas)},LAMB=rp.lambdas.to_numpy(),SIGM=rp.sigmas.to_numpy(),QVAL=rp.q.to_numpy(),AH=np.array([[ah[x,y] for y in aas] for x in aas]))
functions_only(root/'scripts/get_features.py',e2)
for name in examples:
    r,t=name.split('__');s1,s2=seqs[r],seqs[t]
    sm=e2['patch_pair_summaries'](list(s1),list(s2),e2['encode_seq'](s1),e2['encode_seq'](s2))
    for c,v in sm.items():
        stored=P.loc[name,c]
        checks.append(dict(pair=name,source='get_features.py',feature=c,stored=stored,recomputed=v,absolute_error=abs(stored-v),agrees=bool(np.isclose(stored,v,rtol=1e-9,atol=1e-11,equal_nan=True))))
save(pd.DataFrame(checks),'historical_recomputation_checks.csv');save(pd.DataFrame(algebra),'representative_algebra_checks.csv')
print('Recomputation:',pd.DataFrame(checks).groupby('source').agrees.agg(['count','sum']).to_dict(),flush=True)
# Diagnostic-only variables are separate, never written over historical columns.
D=pd.DataFrame(index=P.index)
D['diagnostic_number_of_patch_pairs']=(G.RGG_length-4).clip(lower=0)*(G.TAD_length-4).clip(lower=0)
D['diagnostic_fraction_strong_patches']=P.count_strong_patches/D.diagnostic_number_of_patch_pairs.replace(0,np.nan)
D['diagnostic_opp_charge_range']=P.opp_charge_max-P.opp_charge_min
D['diagnostic_ah_energy_range']=P.ah_energy_max-P.ah_energy_min
save(D.reset_index(),'new_diagnostic_features_training.csv')

rawdefs={
'ah_energy':('calc_ah_between_sequences','sum of 25 residue-pair AH radial integrals / 25','AH interaction','Composition of both local windows; order within a window is ignored.'),
'opp_charge':('patch_opp_charges','-(sum q in RGG window)*(sum q in TAD window)','electrostatics','Exact local net-charge product; not a distinct residue-pairing metric.'),
'scd':('calc_SCD','sum(i>j) q_i*q_j*sqrt(i-j)/5; R,K=+1; D,E=-1; others=0; no terminal charge','charge patterning','Local charge composition and order both contribute; not independent of composition.'),
'shd':('calc_SHD','sum(i<j) (lambda_i+lambda_j)/(j-i)/5','stickiness patterning','Linear position-weighted lambda sum at fixed window length.'),
'rg_fraction':('rg_fraction','count of R or G residues / 5','RG clustering','Local R+G density, not specifically an RG motif.'),
'aromatic_fraction':('aromatic_fraction','count of F,Y,W residues / 5','aromatic clustering','Local composition; overlapping-window distribution reflects placement.'),
'rgg_count':('rgg_motif_count','str.count("RG") within each 5-aa window','RG motif clustering','Counts RG, not RGG; boundary-crossing motifs omitted from a window.'),
'hydropathy_diff':('hydropathy_difference','mean(lambda_RGG_window)-mean(lambda_TAD_window)','stickiness difference','Signed difference; joint variance is sum of marginal window-mean variances.'),
'charge_mismatch':('charge_mismatch','abs(Q1+Q2)-0.5*(sum(abs(q1))+sum(abs(q2)))','electrostatics','Uses absolute residue charges; distinct formula from global charge_complementarity in plot_features.py.'),
'ah_density':('ah_density','calc_ah_between_sequences / 25 = ah_energy / 25','AH interaction','Exact redundant rescaling for fixed 5-aa windows.'),
'entropy':('sequence_entropy','-sum_a p_a*log2(p_a) over residue types in window','local complexity','Composition diversity, not order within the window.'),
'kappa':('get_features.py:calc_kappa','sum_ij q_i*q_j*abs(i-j)/(Q^2*N); zero if Q=0 or no charges','charge patterning','Absent from plot_features.py. Custom historical formula; do not equate with standard kappa; discontinuity at Q=0.')}

def split_feature(c):
    for s in ['top10_mean','top5_mean','mean','min','max','std']:
        if c.endswith('_'+s): return c[:-len(s)-1],s
    return c,'meta'
def category(c):
    b,_=split_feature(c);b=re.sub('_patch[12]$','',b)
    if b in rawdefs:return rawdefs[b][2]
    return {'top_patch_gap':'AH interaction','count_strong_patches':'AH interaction','corr_oppcharge_ah':'electrostatics–AH coupling'}.get(c,'diagnostic')
expected=list(env['summarize_patch_features'](raws[examples[0]]))
allstored=set()
for p in patchfiles:allstored.update(c for c in pd.read_csv(p,nrows=0).columns if local(c))
defs=[]
for c in sorted(allstored|set(expected)):
    b,s=split_feature(c);base=re.sub('_patch[12]$','',b);side='RGG' if '_patch1' in b else 'TAD' if '_patch2' in b else 'pair-level'
    if base in rawdefs:
        fn,formula,cat,caution=rawdefs[base]
        summary={'mean':'arithmetic mean','min':'minimum','max':'maximum','std':'population standard deviation (ddof=0)','top10_mean':'mean of largest min(10,n) entries','top5_mean':'mean of largest min(5,n) entries'}.get(s,s)
        dep='Window means use terminally downweighted composition; not generally equal to global composition.' if s=='mean' else 'Marginal single-sequence descriptor repeated over partner windows.' if side!='pair-level' else ''
        if base=='ah_density':dep='Every corresponding statistic = ah_energy statistic / 25.'
        if s.startswith('top'):caution+=' Largest numerical tail; for negative AH this is least attractive. Original single-side top10 repeats values over all partner windows.'
        origin='plot_features.py:'+fn+' + summarize_patch_features' if c in expected else 'get_features.py:patch_pair_summaries' if c in features else 'UNESTABLISHED exact producing script; extension of historical formulas only'
    else:
        cat=category(c);summary='meta';dep='';fn=c;origin='plot_features.py:'+c
        formula={'top_patch_gap':'largest AH energy minus second largest AH energy','count_strong_patches':'number of AH energy values strictly below -1.0','corr_oppcharge_ah':'Pearson correlation of flattened local OCN and AH arrays'}[c]
        caution={'top_patch_gap':'Weak-attraction tail; ties give zero.','count_strong_patches':'Opportunity count scales as (L_RGG-4)*(L_TAD-4); fixed threshold may never be reached.','corr_oppcharge_ah':'plot_features returns NaN for constants; get_features explicitly returns zero.'}[c]
    defs.append(dict(stored_feature=c,underlying_raw_patch_quantity=b,summary_statistic=s,**{'RGG/TAD/pair-level':side},implemented_definition=formula+'; '+summary,broad_category=cat,known_dependency=dep,interpretation=f'{side} {cat}: {s}',caution=caution,source_function=origin,present_training=c in features,expected_plot_features=c in expected,present_any_supplied_file=c in allstored))
save(pd.DataFrame(defs),'patch_feature_definitions.csv')
# Store complete schema differences, including nonpatch columns, for every candidate.
schemas=[]
for p in patchfiles:
    h=list(pd.read_csv(p,nrows=0).columns); loc=[c for c in h if local(c)]
    schemas.append(dict(file=str(p),missing_expected_patch_columns=';'.join(sorted(set(expected)-set(loc))),extra_patch_columns=';'.join(sorted(set(loc)-set(expected))),nonpatch_columns=';'.join(c for c in h if not local(c))))
save(pd.DataFrame(schemas),'patch_schema_differences.csv')
# Constant detection: both exact and numerical (absolute population variance <=1e-12).
stat=pd.DataFrame(dict(feature=features,variance=[P[c].var(ddof=0) for c in features],unique_values=[P[c].nunique() for c in features],missing_fraction=[P[c].isna().mean() for c in features],minimum=[P[c].min() for c in features],maximum=[P[c].max() for c in features]))
stat['near_zero_variance']=stat.variance<=1e-12;save(stat,'patch_feature_statistics_training.csv')
active=[c for c in features if not stat.set_index('feature').loc[c,'near_zero_variance']]
pr=P.corr(method='pearson');sp=P.corr(method='spearman')
constant=[c for c in features if c not in active]
pr.loc[constant,:]=np.nan;pr.loc[:,constant]=np.nan;sp.loc[constant,:]=np.nan;sp.loc[:,constant]=np.nan
corr=[];high=[];exact=[]
for i,c in enumerate(features):
    for j in range(i+1,len(features)):
        d=features[j]; row=dict(dataset='training',feature1=c,feature2=d,pearson=pr.loc[c,d],spearman=sp.loc[c,d],n_complete=int(P[[c,d]].dropna().shape[0]))
        corr.append(row)
        if abs(row['pearson'])>=.9 or abs(row['spearman'])>=.9:high.append(row)
        x=P[c].to_numpy();y=P[d].to_numpy();ok=np.isfinite(x)&np.isfinite(y)
        if ok.any() and np.array_equal(x[ok],y[ok]):exact.append(dict(feature1=c,feature2=d,relationship='observed exact duplicate',slope=1.,intercept=0.,max_error=0.,proof='empirical training equality; not universal identity'))
        elif c in active and d in active:
            slope,intercept=np.polyfit(x[ok],y[ok],1);err=np.max(abs(y[ok]-(slope*x[ok]+intercept)))
            if np.allclose(y[ok],slope*x[ok]+intercept,rtol=1e-10,atol=1e-12):exact.append(dict(feature1=c,feature2=d,relationship='observed affine transform',slope=slope,intercept=intercept,max_error=err,proof='empirical training affine relationship; not universal identity'))
save(pd.DataFrame(corr),'patch_correlations_training.csv');save(pd.DataFrame(high),'patch_high_correlation_pairs.csv')
for s in ['mean','min','max','std','top10_mean']:
    exact.append(dict(feature1='ah_energy_'+s,feature2='ah_density_'+s,relationship='algebraic fixed-window rescaling',slope=1/25,intercept=0,max_error=0,proof='ah_density divides already averaged AH by 5*5; absent main training schema'))
save(pd.DataFrame(exact),'exact_patch_relationships.csv')
# Full pairwise feature/global correlations; no target values read or referenced.
pg=[]
for c in features:
    for g in G:
        valid=pd.concat([P[c],G[g]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
        good=c in active and valid[g].nunique()>1
        pg.append(dict(patch_feature=c,global_feature=g,pearson=valid[c].corr(valid[g],method='pearson') if good else np.nan,spearman=valid[c].corr(valid[g],method='spearman') if good else np.nan,n_complete=len(valid)))
pg=pd.DataFrame(pg);save(pg,'patch_global_correlations_training.csv')
# Held-out absence is explicit; header-only required files contain no fabricated statistics.
pd.DataFrame(columns=pd.DataFrame(corr).columns).to_csv(out/'patch_correlations_heldout.csv',index=False)
pd.DataFrame(columns=pg.columns).to_csv(out/'patch_global_correlations_heldout.csv',index=False)
(out/'heldout_patch_data_unavailable.txt').write_text('0/1794 exact held-out pair IDs overlap the union of every supplied candidate patch-feature CSV. No held-out statistics or plots are possible; no descriptors were regenerated and no held-out targets were read.\n')
strong=[]
for c in features:
    sub=pg[pg.patch_feature==c].copy();sub['strength']=sub[['pearson','spearman']].abs().max(axis=1)
    if sub.strength.notna().any():
        z=sub.sort_values(['strength','global_feature'],ascending=[False,True]).iloc[0]
        desc='strongly correlated' if z.strength>=.9 else 'partially overlapping; potentially distinct local information, not proof of independence'
        strong.append(dict(feature=c,strongest_global_correlate=z.global_feature,pearson_training=z.pearson,spearman_training=z.spearman,pearson_heldout=np.nan,spearman_heldout=np.nan,category=category(c),interpretation=desc,strongest_by='max absolute Pearson or Spearman'))
    else:strong.append(dict(feature=c,strongest_global_correlate='',pearson_training=np.nan,spearman_training=np.nan,pearson_heldout=np.nan,spearman_heldout=np.nan,category=category(c),interpretation='constant or near-zero variance; not informative here',strongest_by='max absolute Pearson or Spearman'))
strong=pd.DataFrame(strong);save(strong,'patch_global_relationship_summary.csv')
# Correlation graph groups describe connected components, not proven interchangeability.
parent={c:c for c in active}
def find(c):
    while parent[c]!=c:c=parent[c]
    return c
for r in high:
    c,d=r['feature1'],r['feature2'];parent[find(d)]=find(c)
groups=[]
for c in features:
    groups.append(dict(feature=c,category=category(c),correlation_component=find(c) if c in parent else 'constant_or_near_zero',group_definition='training graph edges: abs Pearson or Spearman >=0.90; transitive component, not necessarily pairwise correlated',status='eligible for mechanistic consideration' if c in active else 'exclude constant/near-zero'))
save(pd.DataFrame(groups),'patch_feature_groups.csv')
# Plots include both correlation metrics; unavailable numerical-constant values are grey.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def heat(mats,names,stem,title):
    width=26 if len(mats[0].columns)>40 else 19
    figp,axes=plt.subplots(1,2,figsize=(width,19),layout='constrained')
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dddddd')
    for ax,m,n in zip(axes,mats,names):
        im=ax.imshow(m.to_numpy(),vmin=-1,vmax=1,cmap=cmap,aspect='auto',interpolation='nearest')
        ax.set_xticks(range(len(m.columns)),m.columns,rotation=90,fontsize=6)
        ax.set_yticks(range(len(m.index)),m.index,fontsize=6);ax.set_title(n)
    figp.colorbar(im,ax=axes,shrink=.6,label='Correlation coefficient; grey = undefined / near-constant')
    figp.suptitle(title,fontsize=16)
    figp.savefig(out/(stem+'.pdf'));figp.savefig(out/(stem+'.png'),dpi=170);plt.close(figp)
heat([pr,sp],['Pearson r','Spearman rho'],'patch_feature_correlation_heatmap_training','Historical local descriptors — 917 training pairs')
heat([pg.pivot(index='patch_feature',columns='global_feature',values=s).reindex(features) for s in ['pearson','spearman']],['Pearson r','Spearman rho'],'patch_vs_global_correlation_heatmap_training','Local descriptors versus stored Figure 5 globals — 917 training pairs')
print('Training descriptors:',len(features),'near-zero:',constant,flush=True)
print(strong[strong.feature.str.endswith('_std')].to_string(index=False),flush=True)
# Descriptive multivariate projection, not a B22 model or predictive evaluation.
# Residual variance quantifies what a linear span of stored globals does not reproduce in-sample.
blocks={
'OCN_only':['Opposite_Charge_Number'],
'charge_and_length':['Opposite_Charge_Number','Normalized_OCN','charge_RGG','charge_TAD','NCPR_RGG','NCPR_TAD','RGG_length','TAD_length','Total_Length'],
'all_stored_globals':list(G.columns)}
projection=[]
for block,cols in blocks.items():
    cols=[c for c in cols if c in G and G[c].std()>1e-12]
    X=G[cols]; ok=np.isfinite(X).all(axis=1);X=X.loc[ok];X=(X-X.mean())/X.std(ddof=0)
    X=np.column_stack([np.ones(len(X)),X.to_numpy()]);u,s,v=np.linalg.svd(X,full_matrices=False);rank=int((s>s[0]*1e-10).sum());basis=u[:,:rank]
    for c in active:
        y=P.loc[ok,c].to_numpy()
        if not np.isfinite(y).all():continue
        res=y-basis@(basis.T@y)
        projection.append(dict(feature=c,global_block=block,design_rank=rank,n=len(y),residual_variance_fraction=float(np.var(res)/np.var(y)),caution='In-sample linear descriptive projection only; remaining variance can reflect nonlinear global dependence, omitted composition, noise or parameter drift. No target or predictive model.'))
save(pd.DataFrame(projection),'global_linear_span_diagnostic_training.csv')
# Within identical OCN bins: variation refutes dependence on OCN alone, not all globals.
within=[]
for c in features:
    d=pd.DataFrame({'ocn':G.Opposite_Charge_Number,'value':P[c]})
    q=d.groupby('ocn').value.agg(['size','min','max']);q=q[q['size']>1]
    within.append(dict(feature=c,repeated_ocn_groups=len(q),groups_with_variable_feature=int(((q['max']-q['min'])>1e-10).sum()),largest_within_ocn_range=float((q['max']-q['min']).max())))
save(pd.DataFrame(within),'within_OCN_variation_training.csv')
# Existing global descriptors in patch files are never substituted for Figure 5 globals.
histglobals=read(train_path,lambda c:c=='name' or c in G.columns).set_index('name').loc[P.index]
globaldiff=[]
for c in histglobals:
    x=G[c].to_numpy();y=histglobals[c].to_numpy()
    globaldiff.append(dict(feature=c,max_abs_difference=float(np.nanmax(abs(x-y))),equal_within_tolerance=bool(np.allclose(x,y,rtol=1e-9,atol=1e-11,equal_nan=True))))
save(pd.DataFrame(globaldiff),'stored_global_definition_comparison.csv')
# Verify one deterministic row of the 10-pair extended schema; it is not training data.
example_path=root/'features/one_pair_for_training_added_features_data.csv'
ex=read(example_path,lambda c:c=='name' or local(c)).set_index('name').sort_index();ename=ex.index[0]
r,t=ename.split('__');eraw=env['patch_features'](seqs[r],seqs[t],rp,ah,patch_size=5);esm=env['summarize_patch_features'](eraw)
exchecks=[]
for c,v in esm.items():
    if c in ex:exchecks.append(dict(pair=ename,feature=c,stored=ex.loc[ename,c],recomputed=v,agrees=bool(np.isclose(ex.loc[ename,c],v,rtol=1e-9,atol=1e-11,equal_nan=True)),absolute_error=abs(ex.loc[ename,c]-v)))
save(pd.DataFrame(exchecks),'extended_example_recomputation_checks.csv')
# Additional exact relations on the extended table, clearly outside the training panel.
exrel=[]
for suffix in ['mean','min','max','std','top5_mean','top10_mean']:
    er=(ex['ah_energy_'+suffix]-25*ex['ah_density_'+suffix]).abs().max()
    exrel.append(dict(dataset='10 landscape examples',relationship='ah_energy_'+suffix+' = 25 * ah_density_'+suffix,max_error=er,agrees=er<1e-10))
save(pd.DataFrame(exrel),'extended_example_exact_relationships.csv')
# Quantify normalization and constant thresholds without changing stored variables.
diagcorr=[]
for c in D:
    for g in G:
        diagcorr.append(dict(diagnostic_feature=c,global_feature=g,pearson=D[c].corr(G[g]) if D[c].std()>1e-12 and G[g].std()>1e-12 else np.nan,spearman=D[c].corr(G[g],method='spearman') if D[c].std()>1e-12 and G[g].std()>1e-12 else np.nan))
save(pd.DataFrame(diagcorr),'diagnostic_global_correlations_training.csv')
# Report actual resource differences relative to files named by the historical script,
# located via project roots (never hard-coded historical absolute paths).
resources=[]
for label,p in [('original_residues',root.parent/'residues.csv'),('original_fasta',root.parent/'all_tads_rggs_dis.fasta'),('rgg_headers',root.parent/'rg_tract_headers.txt'),('tad_headers',root.parent/'tad_headers.txt')]:
    if p.exists():
        hashes[str(p)]=sha(p);resources.append(f'{label}: {p}; SHA256 {hashes[str(p)]}')
        if label=='original_residues':
            rr=read(p).set_index('one');resources.append('Original vs shard residue table maximum absolute differences: '+str({c:float((rr.loc[rp.index,c]-rp[c]).abs().max()) for c in ['q','sigmas','lambdas']}))
        if label=='original_fasta':
            ss=read_fasta(p);resources.append(f'Original vs shard FASTA: {len(ss)} vs {len(seqs)} IDs; different shared sequences: {sum(ss[k]!=seqs[k] for k in ss.keys()&seqs.keys())}')
    else:resources.append(f'{label}: unavailable at project-relative expected location {p}')
(out/'historical_resource_comparison.txt').write_text('\n'.join(resources))
with (out/'project_structure.txt').open('a') as f: f.write('\nRESOURCE PROVENANCE SUPPLEMENT\n'+'\n'.join(resources)+'\n')
print('Extended example checks:',pd.DataFrame(exchecks).agrees.value_counts().to_dict(),flush=True)
# Mechanistically specified shortlist, pruned only by TRAINING feature relationships.
# No held-out feature or target values enter this decision.
preferred=['opp_charge_std','scd_patch1_std','scd_patch2_std','rg_fraction_patch1_std','rgg_count_patch1_std','aromatic_fraction_patch1_std','aromatic_fraction_patch2_std','shd_patch1_std','shd_patch2_std','ah_energy_std']
chosen=[];candidates=[]
for c in preferred:
    if c not in features:continue
    z=strong.set_index('feature').loc[c]
    other=[d for d in active if d!=c]
    strength=pd.concat([pr.loc[c,other].abs(),sp.loc[c,other].abs()],axis=1).max(axis=1)
    nearest=strength.idxmax() if strength.notna().any() else ''
    conflict=[d for d in chosen if max(abs(pr.loc[c,d]),abs(sp.loc[c,d]))>=.9]
    accept=c in active and not conflict and c!='aromatic_fraction_patch1_std'
    if accept:chosen.append(c)
    rationale=('Candidate for later training-only ablation: local distribution varies within identical OCN groups; pairwise and multivariate diagnostic evidence is descriptive, not independence or predictive value.' if accept else 'Use as an alternative within the same mechanistic group; correlated with already proposed '+', '.join(conflict))
    if c=='aromatic_fraction_patch1_std':
        rationale='Defer: Spearman correlation with local aromatic mean is approximately 0.992; mainly tracks aromatic density here. Stored global aromatic fraction is truncated, so weak correlations with current globals cannot establish clustering beyond composition.'
    candidates.append(dict(feature=c,category=category(c),interpretation=f'Population SD across overlapping 5-aa windows: {category(c)}; '+('RGG side' if 'patch1' in c else 'TAD side' if 'patch2' in c else 'all RGG × TAD windows'),strongest_global_correlate=z.strongest_global_correlate,pearson_training=z.pearson_training,spearman_training=z.spearman_training,pearson_heldout=np.nan,spearman_heldout=np.nan,redundant_with_patch_feature=nearest,recommended_for_ablation=accept,rationale=rationale,strongest_patch_pearson=pr.loc[c,nearest] if nearest else np.nan,strongest_patch_spearman=sp.loc[c,nearest] if nearest else np.nan,entry_recommendation='direct descriptor as representative of its mechanistic group; no joint inclusion of highly correlated mean/max alternatives' if accept else 'defer; local mean/composition alternative requires separate justification',strongly_correlated_selected_features=';'.join(conflict),global_linear_residual_fraction=next(r['residual_variance_fraction'] for r in projection if r['feature']==c and r['global_block']=='all_stored_globals')))
save(pd.DataFrame(candidates),'candidate_local_context_features.csv')
# Report numerical constants and training-observed duplicate redundancy separately from identities.
redundant=set(constant)
for r in exact:
    if r['relationship'].startswith('observed') and r['feature1'] not in redundant:redundant.add(r['feature2'])
review=pd.DataFrame(checks)
implementation='''IMPLEMENTATION AUDIT
Authoritative interpretation: supplied scripts/plot_features.py. Primary stored schema instead matches scripts/get_features.py, which is named by features/submit_feats.sh and submit_feats_o.sh. A schema match and numerical spot checks support compatibility, not a complete historical run manifest.

A. Local OCN: let P_i=sum positive charges and M_i=-sum negative charges. P1*M2+M1*P2-P1*P2-M1*M2 = -(P1-M1)*(P2-M2) = -Q1*Q2. This is a local net-charge product. Larger positive values indicate stronger charge complementarity. Mean = -mean(Q1)*mean(Q2); variance = E[Q1^2]*E[Q2^2]-E[Q1]^2*E[Q2]^2. Thus dispersion reflects marginal local-charge distributions, not aligned contacts or interchain positional matching.
For overlapping windows, mean(Q)=sum_i w_i*q_i/(L-4), where w_i counts windows containing residue i. Termini are underweighted; this need not equal 5*Q_global/L. No general exact identity with stored global OCN follows. Fixed global OCN can have different local summary values.

B. calc_ah_between_sequences already divides the sum of 25 integrals by 25. ah_density divides again: density=energy/25. Mean/min/max/std/top10 therefore have the same scaling. Main 917-pair files do not contain density columns; do not invent missing training values. Extended 10-pair stored values checked separately.

C. top10_mean (and get_features top5) selects the numerically largest values, not the 10% tail. With negative AH attraction it selects the LEAST attractive tail. ah_energy_min is the strongest attraction. top_patch_gap is largest minus second largest AH energy, also at the weak-attraction tail; ties yield zero. These historical values were retained. Largest positive OCN instead does select strongest charge complementarity. Single-side original top10 summaries repeat a window for each partner window, which can make them equal to the maximum and can introduce partner-length dependence in the top-k statistic.

D. count_strong_patches counts AH < -1.0. The number of opportunities is max(L_RGG-4,0)*max(L_TAD-4,0); its range and expected magnitude depend on length. Newly derived diagnostic_fraction_strong_patches divides by that opportunity count and is stored separately. Normalization does not prove length independence or remove compositional confounding.

E. rgg_motif_count counts non-overlapping occurrences of the two-letter string RG, not RGG. RG has no self-overlap; local R+G fraction counts either residue and is not motif density.

F. plot_features.calc_aromatics returns int(fraction(F,Y,W)), truncating ordinary fractions to zero (floating-point summation can also affect a nominal all-aromatic case). Local aromatic_fraction returns the actual fraction. get_features uses the untruncated global fraction, so its global aromatic column does not share the plot_features bug. Historical values are unchanged.

Additional issues: get_features omits 27 plot_features descriptors and adds 10 (two top5 and eight custom kappa summaries). Its zero-variance corr_oppcharge_ah returns zero where plot_features would produce NaN. Its Charge_Complementarity is global OCN, unlike plot_features abs(Q1+Q2)-0.5*(abs(Q1)+abs(Q2)); use only Figure 5 globals for comparisons. Its custom kappa is not established as standard kappa. Encoding removes unknown residues; all verification sequences must therefore be checked against the table. Local SCD uses hard-coded charges whereas local OCN uses table q. With these tables R/K/D/E charges agree and H=0.

Window summaries describe distributions, not retained absolute positions. AH and local fractions ignore order within a window; SHD at fixed window length is a linear positional weighting of lambda. Overlapping-window dispersion can reflect local composition/arrangement but is not evidence of order beyond composition unless composition-matched controls support that later.

Recomputation uses AST-extracted unchanged functions with decorators omitted and pure single-window results memoized. Top-level historical code is never executed. No RDFs, B22 calculations, models or outputs are invoked. Three deterministic training pairs span minimum/median/maximum OCN, with exact names and all results recorded. One lexicographically first extended example is checked separately; it is not used to choose candidates.
'''
implementation+='\nShared/optimized spot checks:\n'+review.groupby('source').agrees.agg(['count','sum']).to_string()+'\n\nResource provenance:\n'+'\n'.join(resources)
(out/'patch_implementation_audit.txt').write_text(implementation)
# Explicit answers to all 13 requested summary questions.
summary=[]
summary.append('FIGURE 5 HISTORICAL LOCAL-CONTEXT FEATURE AUDIT — diagnostic only\n')
summary.append(f'1. Files: {len(patchfiles)} candidate CSVs: 80 features/features_shard*_of_80.csv; datasets/features_for_training.csv; features_sim_pairs_for_training.csv; features_unsim_pairs_for_training.csv; features_with_b22s_sim_pairs_for_training.csv; features/one_pair_for_training_added_features_data.csv; and three preds/unsim_pred_*.csv. All contain summarized pair rows, not raw patch-patch rows. Full dimensions, IDs and duplicates are in patch_file_inventory.csv.')
summary.append('2. Exact membership: training 917/917; held-out 0/1794; prediction landscape 191386/191386. The full patch table has 192303 = 917 + 191386 unique pairs. The B22-merged subset has only 843 training pairs and was not used as primary input. The extended-schema example contains 10 landscape pairs and no training/held-out pairs. The landscape must not be substituted for the independent held-out panel.')
summary.append(f'3. Definitions: not the identical complete schema. plot_features expects {len(expected)} descriptors; primary files supply {len(features)}, with {len(set(expected)-set(features))} absent and {len(set(features)-set(expected))} extra. Shared formulas and optimized extra formulas were spot-checked. Results: '+str(review.groupby('source').agrees.agg(['count','sum']).to_dict())+'. Read historical_recomputation_checks.csv for individual matches/mismatches. The exact producing run is not proven. Shard residue lambdas differ from the originally referenced residue table. Extended example agreement: '+str(pd.DataFrame(exchecks).agrees.value_counts().to_dict())+'.')
summary.append('4. Exact dependencies: local OCN is a product of local net charges; its mean/variance factorize into marginal charge moments. Local fraction means and AH means are window-coverage-weighted composition summaries, not generally exact functions of unweighted global composition or OCN. Local SHD is a position-weighted linear lambda sum. AH density is redundant scaling. Constant/empirically identical training columns are enumerated separately, with no claim of universal biological identity.')
summary.append('5. Yes: local OCN = -Q_patch1*Q_patch2. Algebra and representative numerical errors are in representative_algebra_checks.csv. It is not an independent residue-pairing mechanism.')
summary.append('6. No: ah_density = ah_energy/25 for fixed 5-aa windows; corresponding summary statistics rescale identically. Density is absent from the training schema and present only in the extended example.')
summary.append('7. Yes, relative to strongest negative AH attraction: top10_mean/top5_mean and top_patch_gap refer to the largest/least-attractive energies. ah_energy_min refers to the strongest-attraction extreme. No historical value was redefined.')
summary.append(f'8. count_strong_patches has length-dependent opportunities. Training count range: {P.count_strong_patches.min():g}–{P.count_strong_patches.max():g}; opportunity range: {D.diagnostic_number_of_patch_pairs.min():g}–{D.diagnostic_number_of_patch_pairs.max():g}. The derived fraction is separate. RG motif counts are per fixed window; their raw totals are not full-sequence motif counts. Extreme summaries additionally depend on the number of sampled windows.')
summary.append('9. Strongest local correlations by global reference (actual Pearson and Spearman):')
for g in ['Opposite_Charge_Number','charge_RGG','charge_TAD','RGG_length','TAD_length','SCD_RGG','SCD_TAD','SHD_RGG','SHD_TAD','lambda_RGG','lambda_TAD','Average_lambda']:
    z=pg[pg.global_feature==g].copy();z['strength']=z[['pearson','spearman']].abs().max(axis=1);z=z.sort_values('strength',ascending=False).head(3)
    summary.append('  '+g+': '+'; '.join(f'{r.patch_feature} (r={r.pearson:.3f}, rho={r.spearman:.3f})' for r in z.itertuples()))
summary.append('Per-descriptor strongest relationships are in patch_global_relationship_summary.csv; the full table retains every coefficient. Correlation graph groups are transitive components, not proof that all members are interchangeable.')
summary.append('10. Dispersion and extremes with variation at fixed OCN warrant consideration as potentially distinct local context. The proposed panel below spans charge heterogeneity/patterning, RG and aromatic density, and lambda/AH heterogeneity where training redundancy permits. This does not establish independence from all composition or nonlinear global transforms. global_linear_span_diagnostic_training.csv reports in-sample residual variance after descriptive projection on global blocks; this is not predictive evaluation. within_OCN_variation_training.csv tests dependence on OCN alone. No B22 targets or model scores were used.')
summary.append('11. Not assessable: no supplied patch rows match held-out sequence space. Required held-out correlation CSVs are header-only and have an explicit unavailable-data note. No held-out plots were manufactured.')
summary.append('12. RGG aromatic SD is deferred because its Spearman correlation with local aromatic mean is approximately 0.992; a weak correlation with the truncated stored global aromatic column is not evidence for clustering. Proposed next-ablation panel ('+str(len(chosen))+'): '+', '.join(chosen)+'. Selection uses training feature relationships and mechanistic interpretability only. Candidate table includes alternatives, strongest global/patch coefficients and grouping advice. Hydropathy_diff and charge_mismatch summaries cannot be proposed from this training schema; AH density should not be added as an independent descriptor. Custom kappa is deferred because of its nonstandard definition; local entropy is a possible later alternative, not added simply to increase panel size.')
summary.append('13. Report before use: schema/version and lambda-parameter differences; zero/NaN correlation convention; AH tail names; double-normalized density; RG vs RGG naming; global aromatic integer truncation in plot_features; changed Charge_Complementarity in get_features; custom kappa definition; length/opportunity dependence; global single-side lambda_RGG/TAD columns represent sums in historical code, while Average_lambda averages per-residue means. Numerical spot checks support stored values only for tested examples and parameter tables.')
summary.append(f'\nAudited {len(features)} training descriptors; {len(constant)} near-zero variance at population variance <=1e-12: '+', '.join(constant)+f'. {len(redundant)} unique columns flagged as near-constant/observed redundant for candidate consideration (none removed from historical files). '+str(sum(r['relationship'].startswith('observed') for r in exact))+' observed duplicate/affine feature-pair relationships; five additional algebraic AH density relationships apply to the absent-density schema. Tolerances: rtol=1e-10, atol=1e-12 for affine checks; numeric spot checks rtol=1e-9, atol=1e-11. Correlations are descriptive without significance tests; repeated sequences make rows non-independent.')
summary.append('\nScope stop: no retraining, no model comparison, no permutation importance, no B22 recalculation, no panel edits. No predictive value is claimed. Original data and scripts are unchanged; SHA256 verification follows.')
# Verify all actual inputs, including every inventoried patch file, remained unchanged.
all_hashes={**hashes,**{r['path']:r['sha256'] for r in inv}}
changed=[p for p,h in all_hashes.items() if sha(Path(p))!=h]
assert not changed,changed
summary.append(f'Input integrity: {len(all_hashes)} source files rehashed unchanged.')
(out/'patch_audit_summary.txt').write_text('\n\n'.join(summary)+'\n')
with (out/'input_provenance.txt').open('a') as f:
    f.write('\n\nHISTORICAL RESOURCE COMPARISON\n'+'\n'.join(resources)+'\nAll '+str(len(all_hashes))+' audited source hashes rechecked unchanged.\nAudit script SHA256: '+sha(Path(__file__))+'\n')
print('Selected inputs:',train_path,paths['training'],root/'datasets/features_for_training.csv')
print('Dimensions:',len(P),'training pairs x',len(features),'local descriptors;',len(G.columns),'stored global descriptors')
print('Pair overlap: training 917/917; heldout 0/1794; landscape 191386/191386')
print('Flagged exact/affine or near-constant redundant columns:',len(redundant),'(no historical columns removed)')
print('Candidate local-context features:',', '.join(chosen))
print('Output:',out)
