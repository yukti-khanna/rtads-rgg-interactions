import hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; O=R/"outputs/fig5_patch_feature_importance"; O.mkdir(parents=True,exist_ok=True); P=R/"datasets/features_sim_pairs_for_training.csv"; F=R.parent; Y="B22_corr1_log10"
patch=pd.read_csv(P); target=pd.read_csv(F/"sim_pairs_pred_with_hybrid_sel_cols_with_old.csv",usecols=["name",Y]); df=patch.merge(target,on="name",validate="one_to_one")
raw=[c for c in patch.columns if c not in ["name","rgg","tad"] and ("patch" in c or c.startswith(("ah_energy_","opp_charge_","kappa_patch","count_strong","corr_oppcharge","top_patch_gap")))]
near=set(pd.read_csv(R/"outputs/fig5_reviewer_patch_audit/patch_feature_statistics_training.csv").query("near_zero_variance").feature)
def reason(c):
 if c in near:return "near-zero variance"
 if c.startswith("kappa_patch"):return "custom/non-standard kappa"
 if c in ["count_strong_patches"]:return "length/opportunity-dependent count"
 if c.startswith(("ah_energy_top5","ah_energy_top10")):return "numerically largest AH tail, not strongest negative attraction"
 return ""
def cat(c):
 for pre,n in [("opp_charge","local electrostatics"),("scd_patch1","RGG charge patterning"),("scd_patch2","TAD charge patterning"),("shd_patch1","RGG hydropathy patterning"),("shd_patch2","TAD hydropathy patterning"),("rg_fraction_patch1","RG localization"),("rgg_count_patch1","RG localization"),("aromatic_fraction_patch1","RGG aromatic localization"),("aromatic_fraction_patch2","TAD aromatic localization"),("ah_energy","local AH interaction"),("entropy_patch1","RGG complexity"),("entropy_patch2","TAD complexity")]:
  if c.startswith(pre):return n
 return "other patch descriptor"
include=[c for c in raw if not reason(c) and not c.startswith("ah_density_")]; groups={c:cat(c) for c in include}; fam={g:[c for c in include if groups[c]==g] for g in set(groups.values())}
pd.DataFrame([{"feature":c,"include_initial_screen":c in include,"exclusion_reason":reason(c) or ("deterministic AH density rescaling" if c.startswith("ah_density_") else ""),"feature_category":cat(c),"historical_definition":"from get_features.py historical patch schema","caution":""} for c in raw]).to_csv(O/"patch_feature_screening_inventory.csv",index=False)
pd.DataFrame([{"feature":c,"patch_category":groups[c]} for c in include]).to_csv(O/"patch_feature_groups.csv",index=False)
y=df[Y].to_numpy(); X=df[include].to_numpy(float)
u=[]
for c in include:
 x=df[c].to_numpy(); ok=np.isfinite(x)&np.isfinite(y); r=pearsonr(x[ok],y[ok]).statistic; s=spearmanr(x[ok],y[ok]).statistic; u.append({"feature":c,"pearson_B22_training":r,"spearman_B22_training":s,"univariate_R2":r*r,"missing_fraction":1-ok.mean(),"variance":np.nanvar(x),"unique_values":pd.Series(x).nunique()})
u=pd.DataFrame(u);u["abs_pearson_rank"]=u.pearson_B22_training.abs().rank(ascending=False);u["abs_spearman_rank"]=u.spearman_B22_training.abs().rank(ascending=False);u.to_csv(O/"patch_univariate_b22_associations.csv",index=False)
params=dict(activation="tanh",alpha=.01,hidden_layer_sizes=(100,50,25),learning_rate_init=.01,max_iter=500,solver="adam")
kf=KFold(5,shuffle=True,random_state=42); indiv=[]; grouped=[]; ablation=[]; conditioned=[]
def train(tr,te,cols,seed):
 ix=[include.index(c) for c in cols]; sc=StandardScaler().fit(X[tr][:,ix]); m=MLPRegressor(**params,random_state=seed);m.fit(sc.transform(X[tr][:,ix]),y[tr]);return m,sc,m.predict(sc.transform(X[te][:,ix]))
for fold,(tr,te) in enumerate(kf.split(X),1):
 m,sc,pred=train(tr,te,include,42+fold); br=r2_score(y[te],pred); bm=np.sqrt(mean_squared_error(y[te],pred)); rng=np.random.default_rng(1000+fold)
 for c in include:
  vals=[]; j=include.index(c)
  for rep in range(3):
   z=X[te].copy();z[:,j]=z[rng.permutation(len(te)),j]; q=m.predict(sc.transform(z)); vals.append((br-r2_score(y[te],q),np.sqrt(mean_squared_error(y[te],q))-bm))
  a=np.array(vals);indiv.append({"feature":c,"fold":fold,"importance_r2_mean":a[:,0].mean(),"importance_r2_sd":a[:,0].std(ddof=1),"importance_rmse_mean":a[:,1].mean(),"positive":(a[:,0]>0).mean()})
 for g,cols in fam.items():
  vals=[]; js=[include.index(c) for c in cols]
  for rep in range(3):
   z=X[te].copy();z[:,js]=z[rng.permutation(len(te))][:,js];q=m.predict(sc.transform(z));vals.append((br-r2_score(y[te],q),np.sqrt(mean_squared_error(y[te],q))-bm))
  a=np.array(vals);grouped.append({"group":g,"fold":fold,"importance_r2_mean":a[:,0].mean(),"importance_r2_sd":a[:,0].std(ddof=1),"importance_rmse_mean":a[:,1].mean(),"positive":(a[:,0]>0).mean()})
 for g,cols in fam.items():
  keep=[c for c in include if c not in cols];_,_,q=train(tr,te,keep,42+fold);ablation.append({"group":g,"fold":fold,"full_r2":br,"without_group_r2":r2_score(y[te],q),"delta_r2":br-r2_score(y[te],q),"full_rmse":bm,"without_group_rmse":np.sqrt(mean_squared_error(y[te],q)),"delta_rmse":np.sqrt(mean_squared_error(y[te],q))-bm,"without_group_mae":mean_absolute_error(y[te],q),"without_group_pearson":pearsonr(y[te],q).statistic,"without_group_spearman":spearmanr(y[te],q).statistic})
 ocn=df["Opposite_Charge_Number"].to_numpy(); XX=np.column_stack([X,ocn]); s2=StandardScaler().fit(XX[tr]); m2=MLPRegressor(**params,random_state=42+fold);m2.fit(s2.transform(XX[tr]),y[tr]); b=r2_score(y[te],m2.predict(s2.transform(XX[te]))); rr=np.random.default_rng(2000+fold)
 for c in include:
  v=[];j=include.index(c)
  for rep in range(3):z=XX[te].copy();z[:,j]=z[rr.permutation(len(te)),j];v.append(b-r2_score(y[te],m2.predict(s2.transform(z))))
  conditioned.append({"feature":c,"fold":fold,"importance_after_global_OCN_mean":np.mean(v),"importance_after_global_OCN_sd":np.std(v,ddof=1)})
ind=pd.DataFrame(indiv); agg=ind.groupby("feature").agg(permutation_importance_mean=("importance_r2_mean","mean"),permutation_importance_sd=("importance_r2_mean","std"),permutation_importance_median=("importance_r2_mean","median"),positive_importance_fold_fraction=("positive","mean"),rmse_increase_mean=("importance_rmse_mean","mean")).reset_index();agg.to_csv(O/"patch_permutation_importance_cv.csv",index=False)
grp=pd.DataFrame(grouped).groupby("group").agg(grouped_family_importance=("importance_r2_mean","mean"),grouped_importance_sd=("importance_r2_mean","std"),positive_fold_fraction=("positive","mean"),grouped_rmse_increase=("importance_rmse_mean","mean")).reset_index();grp.to_csv(O/"patch_grouped_permutation_importance_cv.csv",index=False)
abl=pd.DataFrame(ablation).groupby("group").mean(numeric_only=True).reset_index();abl.to_csv(O/"patch_group_ablation_cv.csv",index=False)
co=pd.DataFrame(conditioned).groupby("feature").agg(importance_after_global_OCN=("importance_after_global_OCN_mean","mean"),importance_after_global_OCN_sd=("importance_after_global_OCN_mean","std")).reset_index();co.to_csv(O/"patch_importance_conditioned_on_global_OCN.csv",index=False)
sel=agg.merge(u[["feature","pearson_B22_training","spearman_B22_training"]],on="feature").merge(co,on="feature"); sel["patch_category"]=sel.feature.map(groups);sel["historical_definition"]="get_features.py 5-aa overlapping-window descriptor";sel["drop_group_effect"]=sel.patch_category.map(abl.set_index("group").delta_r2);sel["grouped_family_importance"]=sel.patch_category.map(grp.set_index("group").grouped_family_importance);sel["selected_for_M4"]=False;sel["rationale"]="Training-only screen; final lock deferred to review";sel.to_csv(O/"selected_patch_features_for_M4.csv",index=False)
(O/"input_provenance.txt").write_text(f"Patch input: {P}\nTarget input: {F/'sim_pairs_pred_with_hybrid_sel_cols_with_old.csv'}\nRows: {len(df)}\nHeld-out B22 was not loaded.\n")
(O/"target_definition.txt").write_text("Target is B22_corr1_log10, the corrected/log10 simulated B22 column used by Figure 5 scripts.\n")
(O/"model_definition.txt").write_text("StandardScaler within each fold; MLPRegressor tanh, (100,50,25), alpha=.01, Adam, learning_rate_init=.01, max_iter=500, seed 42; shuffled 5-fold CV seed 42; 3 permutation repeats.\n")
for data,key,name,title in [(agg,"permutation_importance_mean","patch_feature_importance_cv","Patch feature permutation importance"),(grp,"grouped_family_importance","patch_group_importance_cv","Grouped patch permutation importance"),(abl,"delta_r2","patch_group_ablation_cv","Patch group drop-group effect")]:
 z=data.sort_values(key);f,ax=plt.subplots(figsize=(10,8));ax.barh(z.iloc[:,0],z[key]);ax.set_title(title);f.tight_layout();f.savefig(O/(name+".png"),dpi=160);f.savefig(O/(name+".pdf"));plt.close(f)
(O/"patch_feature_importance_summary.txt").write_text(f"Training-only feature screening completed on {len(df)} pairs. Target: {Y}. Held-out 1,794 B22 values accessed: NO. {len(include)} patch descriptors screened after exclusions. See CSV outputs for univariate, individual permutation, grouped, ablation and OCN-conditioned results. No model was evaluated on held-out data.\n")
print("completed",len(df),"rows",len(include),"features")
