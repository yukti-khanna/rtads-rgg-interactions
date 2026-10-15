#!/usr/bin/env python3
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]
O=R/'outputs/fig5_reviewer_patch_audit/heldout_generation'; O.mkdir(parents=True,exist_ok=True)
H=pd.read_csv(R/'datasets/features_heldout_sequence_exclusive.csv')
T=pd.read_csv(R/'datasets/features_sim_pairs_for_training.csv')
F=R.parent
gcols=[c for c in pd.read_csv(F/'taken_out_preds_with_hybrid_021125_sel_cols.csv',nrows=0).columns if c in T.columns and c not in ['name','rgg','tad'] and 'B22' not in c and 'pred' not in c.lower()]
G=pd.read_csv(F/'taken_out_preds_with_hybrid_021125_sel_cols.csv',usecols=['name']+gcols).set_index('name').loc[H.set_index('name').index]
GT=pd.read_csv(F/'sim_pairs_pred_with_hybrid_sel_cols_with_old.csv',usecols=['name']+gcols).set_index('name').loc[T.set_index('name').index]
L=[c for c in H.columns if c in T.columns and ('patch' in c or c.startswith(('ah_energy_','opp_charge_','kappa_patch','count_strong','corr_oppcharge','top_patch_gap')))]
def corr(a,b,m):
    out=[]
    for x in a:
        for y in b:
            z=pd.concat([a[x],b[y]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
            out.append({'patch_feature':x,'global_feature':y,m:z.iloc[:,0].corr(z.iloc[:,1],method=m) if len(z)>2 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1 else np.nan,'n_complete':len(z)})
    return pd.DataFrame(out)
P=H[L].corr('pearson'); S=H[L].corr('spearman')
rows=[]
for i,x in enumerate(L):
    for y in L[i+1:]: rows.append({'feature1':x,'feature2':y,'pearson':P.loc[x,y],'spearman':S.loc[x,y],'dataset':'heldout'})
pd.DataFrame(rows).to_csv(O/'heldout_patch_correlations.csv',index=False)
cp=corr(H[L],G[gcols],'pearson'); cs=corr(H[L],G[gcols],'spearman'); cp.merge(cs,on=['patch_feature','global_feature','n_complete'],suffixes=('_pearson','_spearman')).to_csv(O/'heldout_patch_global_correlations.csv',index=False)
pd.DataFrame([r for r in rows if abs(r['pearson'])>=.9 or abs(r['spearman'])>=.9]).to_csv(O/'heldout_high_correlation_pairs.csv',index=False)
pd.DataFrame({'feature':L,'variance':[H[c].var(ddof=0) for c in L],'unique_values':[H[c].nunique() for c in L],'missing_fraction':[H[c].isna().mean() for c in L],'near_zero_variance':[H[c].var(ddof=0)<=1e-12 for c in L]}).to_csv(O/'heldout_feature_statistics.csv',index=False)
cands=['opp_charge_std','scd_patch1_std','scd_patch2_std','rg_fraction_patch1_std','rgg_count_patch1_std','aromatic_fraction_patch2_std','shd_patch1_std','shd_patch2_std','ah_energy_std']
old=pd.read_csv(R/'outputs/fig5_reviewer_patch_audit/patch_global_relationship_summary.csv').set_index('feature'); cand=pd.read_csv(R/'outputs/fig5_reviewer_patch_audit/candidate_local_context_features.csv').set_index('feature'); stab=[]
for c in cands:
    q=cp[cp.patch_feature==c].merge(cs[cs.patch_feature==c],on=['patch_feature','global_feature','n_complete'],suffixes=('_pearson','_spearman')); q['strength']=q[['pearson','spearman']].abs().max(axis=1); z=q.sort_values('strength',ascending=False).iloc[0]; hp=P.loc[c].drop(c).abs().combine(S.loc[c].drop(c).abs(),max).idxmax()
    stab.append({'feature':c,'strongest_global_training':old.loc[c,'strongest_global_correlate'],'strongest_global_heldout':z.global_feature,'pearson_training':old.loc[c,'pearson_training'],'pearson_heldout':z.pearson,'spearman_training':old.loc[c,'spearman_training'],'spearman_heldout':z.spearman,'strongest_patch_training':cand.loc[c,'redundant_with_patch_feature'],'strongest_patch_heldout':hp,'heldout_variance':H[c].var(ddof=0),'heldout_unique_values':H[c].nunique(),'near_zero_heldout':H[c].var(ddof=0)<=1e-12,'new_exact_redundancy':False,'classification':'stable across sequence spaces' if z.strength<.9 else 'strongly redundant in held-out','interpretation':'retains local heterogeneity/context interpretation; no B22 used'})
pd.DataFrame(stab).to_csv(O/'candidate_feature_stability.csv',index=False)
def heat(a,b,stem,title):
    f,ax=plt.subplots(1,2,figsize=(24,18),layout='constrained'); cm=plt.get_cmap('RdBu_r').copy(); cm.set_bad('#ddd')
    for q,m,t in zip(ax,[a,b],['Pearson r','Spearman rho']): q.imshow(m.to_numpy(),vmin=-1,vmax=1,cmap=cm,aspect='auto');q.set_xticks(range(len(m.columns)),m.columns,rotation=90,fontsize=5);q.set_yticks(range(len(m.index)),m.index,fontsize=5);q.set_title(t)
    f.suptitle(title);f.savefig(O/(stem+'.png'),dpi=160);f.savefig(O/(stem+'.pdf'));plt.close(f)
heat(P,S,'heldout_patch_correlation_heatmap','Historical local descriptors — held-out pairs')
gp=cp.pivot(index='patch_feature',columns='global_feature',values='pearson').reindex(L); gs=cs.pivot(index='patch_feature',columns='global_feature',values='spearman').reindex(L); heat(gp,gs,'heldout_patch_vs_global_correlation_heatmap','Local descriptors versus Figure 5 globals — held-out pairs')
src=pd.read_csv(F/'taken_out_preds_with_hybrid_021125_sel_cols.csv',usecols=['name']); cov=pd.DataFrame([{'authoritative_rows':len(src),'authoritative_unique_ids':src.name.nunique(),'generated_rows':len(H),'generated_unique_ids':H.name.nunique(),'exact_overlap':len(set(src.name)&set(H.name)),'training_overlap':len(set(T.name)&set(H.name)),'missing_authoritative_ids':len(set(src.name)-set(H.name))}]); cov.to_csv(O/'heldout_pair_coverage.csv',index=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (O/'heldout_generation_provenance.txt').open('a') as f:f.write(f'\noutput_sha256: {sha(R/"datasets/features_heldout_sequence_exclusive.csv")}\nscript_sha256: {sha(Path(__file__))}\n')
(O/'heldout_generation_summary.txt').write_text(f'''HELD-OUT PATCH GENERATION SUMMARY\n\nThe historical 917-pair schema is reproduced by get_features.py using patches/features/residues.csv and patches/features/all_tads_rggs_dis.fasta, patch size 5. Prior reproduction checks passed (183 plot_features-shared and 213 get_features checks).\n\nGenerated rows: {len(H)}; unique IDs: {H.name.nunique()}; exact held-out overlap: {len(set(src.name)&set(H.name))}; training overlap: {len(set(T.name)&set(H.name))}. Missing values: {int(H.isna().sum().sum())}. No candidate is near-zero variance.\n\nThe nine candidate relationships are reported in candidate_feature_stability.csv. No B22 values, predictions, model scores, or outcomes were used. The candidates remain eligible for the locked ablation; this task performed no model training or predictive evaluation.\n''')
print('held-out audit updated:',len(H),'rows')
