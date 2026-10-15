#!/usr/bin/env python3
"""Figure 4 Task 3: sampling-bias and sequence-exclusive validation audit.

This script does not simulate or fit models. It uses the manuscript catalogues,
the exact 917-pair Figure 4 cache, and the existing simulation-derived held-out
B22 table. All random procedures are deterministic.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SEED = 20260910
N_BOOT = 10_000

FIG4 = Path("/Users/yuktikhanna/rtads-rgg-interactions/fig4")
CAL = Path("/Users/yuktikhanna/calvados")
IDR = Path("/Users/yuktikhanna/idr-interaction-analysis")
DEFAULT_OUT = FIG4 / "task3_sampling_bias"

FULL_SOURCE = CAL / "all_pairs_features_data.csv"
CAT_FASTA = CAL / "lists/all_tads_rggs_dis.fasta"
TAD_LIST = CAL / "lists/tad_headers.txt"
RGG_LIST = CAL / "lists/rg_tract_headers.txt"
HELD_LIST = CAL / "lists/tads_rgs_taken_out.txt"
HELD_SIM = CAL / "taken_out/datasets/taken_out_features_data_sim.csv"
HELD_SELECTION_SCRIPT = CAL / "names_from_diff_lists.py"
FIG4_CACHE = FIG4 / "out_fig4/cache/Fig4d_scatter_table.csv"
FIG4_FEATURES = FIG4 / "sim_pairs_added_features_data_270525.csv"

FEATURES = [
    "Opposite_Charge_Number", "Normalized_OCN", "Charge_Complementarity",
    "charge_TAD", "charge_RGG", "Average_NCPR", "Average_lambda",
    "TAD_length", "RGG_length", "Total_Length",
]
EXTRA_SELECTION = ["Average_sigma", "Average_charge"]


def read_ids(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text().splitlines() if x.strip()]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()


def canonical_features(df: pd.DataFrame) -> pd.DataFrame:
    d = clean(df)
    d = d.rename(columns={
        "name": "pair_id", "rgg_length": "RGG_length", "tad_length": "TAD_length",
        "charge_rgg": "charge_RGG", "charge_tad": "charge_TAD",
        "Average_ncpr": "Average_NCPR", "Average_ncpr": "Average_NCPR",
        "Total_Sequence_Length": "Total_Length",
    })
    if "pair_id" not in d and {"rgg", "tad"}.issubset(d):
        d["pair_id"] = d["rgg"].astype(str) + "__" + d["tad"].astype(str)
    d["Opposite_Charge_Number"] = -(d["charge_RGG"] * d["charge_TAD"])
    d["Normalized_OCN"] = d["Opposite_Charge_Number"] / d["Total_Length"]
    return d


def describe(x: pd.Series) -> dict:
    a = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    q = np.quantile(a, [.05, .25, .5, .75, .95])
    return {"n": len(a), "mean": a.mean(), "median": q[2],
            "SD": a.std(ddof=1), "IQR": q[3]-q[1], "p05": q[0],
            "p25": q[1], "p75": q[3], "p95": q[4]}


def distance(a: pd.Series, b: pd.Series) -> dict:
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(float)
    pooled = np.sqrt(((len(x)-1)*x.var(ddof=1)+(len(y)-1)*y.var(ddof=1))/(len(x)+len(y)-2))
    return {"standardized_mean_difference": (y.mean()-x.mean())/pooled if pooled else np.nan,
            "KS_statistic": stats.ks_2samp(x, y).statistic,
            "Wasserstein_distance": stats.wasserstein_distance(x, y)}


def percentile_ci(vals: list[float]) -> tuple[float, float]:
    return tuple(np.quantile(np.asarray(vals, float), [.025, .975]))


def bootstrap_corr(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot=N_BOOT):
    n = len(x); pr=[]; sr=[]
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        xb, yb = x[ix], y[ix]
        if np.std(xb) == 0 or np.std(yb) == 0: continue
        pr.append(np.corrcoef(xb, yb)[0,1])
        sr.append(stats.spearmanr(xb, yb).statistic)
    return percentile_ci(pr), percentile_ci(sr)


def corr_row(label: str, feature: str, d: pd.DataFrame, rng: np.random.Generator) -> dict:
    z=d[[feature,"B22_corr1_log10"]].dropna(); x=z[feature].to_numpy(float); y=z.B22_corr1_log10.to_numpy(float)
    p=stats.pearsonr(x,y); s=stats.spearmanr(x,y); pci,sci=bootstrap_corr(x,y,rng)
    return {"dataset":label,"feature":feature,"n":len(z),"Pearson_r":p.statistic,
            "Pearson_CI_low":pci[0],"Pearson_CI_high":pci[1],"Pearson_p":p.pvalue,
            "Spearman_rho":s.statistic,"Spearman_CI_low":sci[0],
            "Spearman_CI_high":sci[1],"Spearman_p":s.pvalue}


def plot_ecdfs(sets: dict[str,pd.DataFrame], features: list[str], path: Path, cols=3):
    rows=int(np.ceil(len(features)/cols)); fig,axs=plt.subplots(rows,cols,figsize=(4.1*cols,3.15*rows)); axs=np.atleast_1d(axs).ravel()
    colors={"Full landscape":"#aaaaaa","917 panel":"#D55E00","Held-out 1,794":"#0072B2"}
    for ax,f in zip(axs,features):
        for lab,d in sets.items():
            x=np.sort(d[f].dropna().to_numpy(float)); y=np.arange(1,len(x)+1)/len(x)
            ax.plot(x,y,label=lab,color=colors[lab],lw=1.8,alpha=.9)
        ax.set(title=f.replace('_',' '),ylabel="ECDF"); ax.grid(alpha=.18)
    for ax in axs[len(features):]: ax.axis('off')
    axs[0].legend(frameon=False,fontsize=8); fig.tight_layout(); savefig(fig,path)


def savefig(fig, path: Path):
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)


def main(out: Path):
    out.mkdir(parents=True,exist_ok=True); figs=out/'figures'; figs.mkdir(exist_ok=True)
    tad_ids=read_ids(TAD_LIST); rgg_ids=read_ids(RGG_LIST); held_ids=read_ids(HELD_LIST)
    held_tads=held_ids[:23]; held_rggs=held_ids[23:]

    full=canonical_features(pd.read_csv(FULL_SOURCE))
    f4_names=clean(pd.read_csv(FIG4_CACHE))["name"].astype(str)
    f4_master=canonical_features(pd.read_csv(FIG4_FEATURES))
    panel=f4_master[f4_master.pair_id.isin(set(f4_names))].copy()
    held=canonical_features(pd.read_csv(HELD_SIM))

    assert len(tad_ids)==230 and len(rgg_ids)==1008
    assert len(full)==230*1008==231840 and full.pair_id.nunique()==len(full)
    assert len(panel)==panel.pair_id.nunique()==917 and set(panel.pair_id)==set(f4_names)
    assert len(held_tads)==23 and len(held_rggs)==78
    expected={r+'__'+t for r in held_rggs for t in held_tads}
    assert len(expected)==1794 and set(held.pair_id)==expected and held.pair_id.nunique()==1794
    assert held["Corrected_B22_Hummer"].notna().all() and held["B22_corr1_log10"].notna().all()
    assert np.allclose(held.B22_corr1_log10, np.log10(-held.Corrected_B22_Hummer))
    for d in (full,panel,held):
        missing=[f for f in FEATURES+EXTRA_SELECTION if f not in d]
        if missing: raise ValueError(f"Missing required features: {missing}")

    train_tads=set(panel.tad); train_rggs=set(panel.rgg)
    ov_t=sorted(set(held_tads)&train_tads); ov_r=sorted(set(held_rggs)&train_rggs)
    exclus=[]
    for typ,ids,train in [("TAD",held_tads,train_tads),("RGG",held_rggs,train_rggs)]:
        exclus += [{"sequence_type":typ,"sequence_id":x,"appears_in_917":x in train,
                    "sequence_exclusive":x not in train} for x in ids]
    pd.DataFrame(exclus).to_csv(out/'heldout_sequence_exclusivity.csv',index=False,float_format='%.17g')

    prov=[
      {"dataset":"Dataset A: full combinatorial candidate landscape","path":str(FULL_SOURCE),"n_TADs":full.tad.nunique(),"n_RGGs":full.rgg.nunique(),"n_pairs":len(full),"has_sequences":CAT_FASTA.exists(),"has_features":True,"has_B22":False,"selection_method_documented":"not applicable: complete cross-product","notes":"230 x 1,008 exact complete catalogue cross-product; sequence-derived features only."},
      {"dataset":"Dataset B: feature-guided Figure 4 panel","path":str(FIG4_CACHE),"n_TADs":panel.tad.nunique(),"n_RGGs":panel.rgg.nunique(),"n_pairs":len(panel),"has_sequences":True,"has_features":True,"has_B22":True,"selection_method_documented":"partial executable record in fig4/get_features_all.py","notes":"Exact 917 cache; historical code records feature filters and random_state=42 for some stages, but not a single end-to-end selection manifest."},
      {"dataset":"Dataset C: 23 x 78 sequence-exclusive held-out pool","path":str(HELD_SIM),"n_TADs":held.tad.nunique(),"n_RGGs":held.rgg.nunique(),"n_pairs":len(held),"has_sequences":True,"has_features":True,"has_B22":True,"selection_method_documented":"saved IDs and cross-product script; original random draw not found","notes":"All 1,794 cross-combinations have simulation-derived Hummer-corrected B22. Sequence exclusivity confirmed. Random pre-fit selection provenance is not reproducible from available code/seed."},
    ]
    pd.DataFrame(prov).to_csv(out/'dataset_provenance.csv',index=False,float_format='%.17g')

    keep=['pair_id','tad','rgg']+FEATURES+EXTRA_SELECTION
    full[keep].to_csv(out/'full_combinatorial_feature_landscape.csv',index=False,float_format='%.17g')
    panel_out=panel[keep+['B22_corr1_log10','Corrected_B22_Hummer']].copy(); panel_out['sampled_917']=True
    panel_out.to_csv(out/'fig4_917_sampling_membership.csv',index=False,float_format='%.17g')
    held[keep+['B22_corr1_log10','Corrected_B22_Hummer']].to_csv(out/'heldout_23x78_features.csv',index=False,float_format='%.17g')

    rows=[]
    for f in FEATURES+EXTRA_SELECTION:
        row={"feature":f}
        for pre,d in [("full",full),("panel_917",panel)]: row.update({f"{pre}_{k}":v for k,v in describe(d[f]).items()})
        row.update(distance(full[f],panel[f])); rows.append(row)
    pd.DataFrame(rows).to_csv(out/'feature_distribution_comparison_full_vs_917.csv',index=False,float_format='%.17g')

    sets={"Full landscape":full,"917 panel":panel,"Held-out 1,794":held}
    rows=[]
    for f in FEATURES:
        for lab,d in sets.items(): rows.append({"record_type":"summary","feature":f,"dataset":lab,"comparison":"",**describe(d[f])})
        for a,b in [("Full landscape","917 panel"),("Full landscape","Held-out 1,794"),("917 panel","Held-out 1,794")]:
            rows.append({"record_type":"pairwise_distance","feature":f,"dataset":"","comparison":f"{a} vs {b}",**distance(sets[a][f],sets[b][f])})
    pd.DataFrame(rows).to_csv(out/'feature_distribution_comparison_three_sets.csv',index=False,float_format='%.17g')

    cov=[]
    for f in FEATURES:
        edges=np.quantile(full[f],[i/10 for i in range(11)]); edges=np.unique(edges)
        for lab,d in [("917 panel",panel),("Held-out 1,794",held)]:
            bins=pd.cut(d[f],bins=np.r_[-np.inf,edges[1:-1],np.inf],labels=False,include_lowest=True)
            pct=stats.percentileofscore(full[f],d[f],kind='rank') if False else None
            ranks=np.searchsorted(np.sort(full[f].to_numpy()),d[f].to_numpy(),side='right')/len(full)*100
            for dec in range(10):
                n=int((bins==dec).sum()); cov.append({"record_type":"decile","feature":f,"dataset":lab,"full_decile":dec+1,"count":n,"fraction":n/len(d)})
            cov.append({"record_type":"coverage_summary","feature":f,"dataset":lab,"deciles_occupied":int(bins.nunique()),"minimum_full_percentile_represented":ranks.min(),"maximum_full_percentile_represented":ranks.max(),"fraction_in_extreme_10pct_tails":float(((ranks<=10)|(ranks>=90)).mean())})
    coverage=pd.DataFrame(cov); coverage.to_csv(out/'feature_decile_coverage.csv',index=False,float_format='%.17g')

    rng=np.random.default_rng(SEED); corrs=[]
    for lab,d in [("917 feature-guided",panel),("held-out sequence-exclusive",held)]:
        for f in ["Opposite_Charge_Number","Normalized_OCN","Charge_Complementarity"]: corrs.append(corr_row(lab,f,d,rng))
    pd.DataFrame(corrs).to_csv(out/'electrostatic_correlations_917_vs_heldout.csv',index=False,float_format='%.17g')

    # Common OCN quartiles defined by the full landscape.
    q=np.unique(np.quantile(full.Opposite_Charge_Number,[0,.25,.5,.75,1])); strat=[]
    for lab,d in [("917 feature-guided",panel),("held-out sequence-exclusive",held)]:
        sb=pd.cut(d.Opposite_Charge_Number,bins=np.r_[-np.inf,q[1:-1],np.inf],labels=False)
        for k,g in d.groupby(sb,observed=True):
            x=g.Opposite_Charge_Number.to_numpy(); y=g.B22_corr1_log10.to_numpy()
            valid=len(g)>=10 and np.std(x)>0 and np.std(y)>0
            strat.append({"dataset":lab,"stratum":int(k)+1,"lower_edge":q[int(k)],"upper_edge":q[int(k)+1],"n":len(g),
                          "Pearson_r":stats.pearsonr(x,y).statistic if valid else np.nan,
                          "Spearman_rho":stats.spearmanr(x,y).statistic if valid else np.nan,
                          "slope":stats.linregress(x,y).slope if valid else np.nan,"sufficient_variation":valid})
    pd.DataFrame(strat).to_csv(out/'OCN_B22_stratified_correlations.csv',index=False,float_format='%.17g')

    # Transparent 3D coarsened exact matching; target held-out proportions, no B22 use.
    keys=[]
    for f in ["Opposite_Charge_Number","Total_Length","Average_lambda"]:
        e=np.unique(np.quantile(held[f],[0,.2,.4,.6,.8,1])); key=f+'_bin'; keys.append(key)
        for d in (held,panel): d[key]=pd.cut(d[f],bins=np.r_[-np.inf,e[1:-1],np.inf],labels=False)
    chunks=[]
    for kval,hg in held.groupby(keys):
        mask=np.ones(len(panel),bool)
        for k,v in zip(keys,kval): mask &= panel[k].to_numpy()==v
        pg=panel.loc[mask]
        if len(pg): chunks.append(pg.sample(n=min(len(pg),len(hg)),random_state=SEED+sum(map(int,kval))))
    matched=pd.concat(chunks).drop_duplicates('pair_id')
    sens=[]
    for lab,d in [("full 917",panel),("matched 917 subset",matched),("held-out",held)]:
        x=d.Opposite_Charge_Number; y=d.B22_corr1_log10
        sens.append({"dataset":lab,"n":len(d),"Pearson_r":stats.pearsonr(x,y).statistic,"Spearman_rho":stats.spearmanr(x,y).statistic,
                     "matching_algorithm":"5x5x5 quantile-bin coarsened exact matching to held-out; min available per occupied cell","random_seed":SEED})
    pd.DataFrame(sens).to_csv(out/'matched_sampling_sensitivity.csv',index=False,float_format='%.17g')

    # Two-way pigeonhole bootstrap: independently resample TAD and RGG clusters.
    cb=[]; rr=np.random.default_rng(SEED+9)
    for f in ["Opposite_Charge_Number","Normalized_OCN","Charge_Complementarity"]:
        pivx=held.pivot(index='tad',columns='rgg',values=f).loc[held_tads,held_rggs].to_numpy()
        pivy=held.pivot(index='tad',columns='rgg',values='B22_corr1_log10').loc[held_tads,held_rggs].to_numpy()
        pv=[]; sv=[]
        for _ in range(N_BOOT):
            i=rr.integers(0,len(held_tads),len(held_tads)); j=rr.integers(0,len(held_rggs),len(held_rggs))
            x=pivx[np.ix_(i,j)].ravel(); y=pivy[np.ix_(i,j)].ravel()
            pv.append(np.corrcoef(x,y)[0,1]); sv.append(stats.spearmanr(x,y).statistic)
        pair=next(r for r in corrs if r['dataset']=="held-out sequence-exclusive" and r['feature']==f)
        pci=percentile_ci(pv); sci=percentile_ci(sv)
        cb.append({"feature":f,"n_pairs":1794,"n_TAD_clusters":23,"n_RGG_clusters":78,
                   "pair_bootstrap_Pearson_CI_low":pair['Pearson_CI_low'],"pair_bootstrap_Pearson_CI_high":pair['Pearson_CI_high'],
                   "pair_bootstrap_Spearman_CI_low":pair['Spearman_CI_low'],"pair_bootstrap_Spearman_CI_high":pair['Spearman_CI_high'],
                   "two_way_cluster_Pearson_CI_low":pci[0],"two_way_cluster_Pearson_CI_high":pci[1],
                   "two_way_cluster_Spearman_CI_low":sci[0],"two_way_cluster_Spearman_CI_high":sci[1],
                   "bootstrap_resamples":N_BOOT,"random_seed":SEED+9})
    pd.DataFrame(cb).to_csv(out/'heldout_clustered_bootstrap_correlations.csv',index=False,float_format='%.17g')

    plot_ecdfs(sets,FEATURES,figs/'feature_distributions_full_917_heldout')
    for f,n in [("Opposite_Charge_Number","OCN_distribution_three_sets"),("Charge_Complementarity","charge_complementarity_distribution_three_sets"),("Average_NCPR","average_NCPR_distribution_three_sets"),("Average_lambda","average_lambda_distribution_three_sets")]: plot_ecdfs(sets,[f],figs/n,cols=1)
    plot_ecdfs(sets,["TAD_length","RGG_length","Total_Length"],figs/'length_distribution_three_sets')
    c=coverage[coverage.record_type=='decile']; fig,axs=plt.subplots(2,5,figsize=(15,6),sharey=True)
    for ax,(f,g) in zip(axs.ravel(),c.groupby('feature',sort=False)):
        for lab,col in [("917 panel","#D55E00"),("Held-out 1,794","#0072B2")]:
            z=g[g.dataset==lab]; ax.plot(z.full_decile,z.fraction,marker='o',label=lab,color=col)
        ax.set_title(f.replace('_',' '),fontsize=9); ax.set_xticks([1,5,10]); ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7,frameon=False); fig.tight_layout(); savefig(fig,figs/'feature_decile_coverage')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,yf in zip(axs,["Average_lambda","Average_NCPR"]):
        hb=ax.hexbin(full.Opposite_Charge_Number,full[yf],gridsize=45,mincnt=1,cmap='Greys',alpha=.55)
        ax.scatter(panel.Opposite_Charge_Number,panel[yf],s=7,c='#D55E00',alpha=.45,label='917')
        ax.scatter(held.Opposite_Charge_Number,held[yf],s=7,c='#0072B2',alpha=.35,label='held-out')
        ax.set(xlabel='OCN',ylabel=yf.replace('_',' ')); ax.legend(frameon=False,fontsize=8)
    fig.tight_layout(); savefig(fig,figs/'sampling_feature_space_projection')
    for f,n in [("Opposite_Charge_Number","OCN_vs_B22_comparison"),("Normalized_OCN","normalized_OCN_vs_B22_comparison"),("Charge_Complementarity","charge_complementarity_vs_B22_comparison")]:
        fig,axs=plt.subplots(1,2,figsize=(9,4),sharey=True)
        xmin=min(panel[f].min(),held[f].min()); xmax=max(panel[f].max(),held[f].max())
        pad=.03*(xmax-xmin) if xmax>xmin else 1
        for ax,(lab,d,col) in zip(axs,[("917 feature-guided",panel,'#D55E00'),("Held-out sequence-exclusive",held,'#0072B2')]):
            ax.scatter(d[f],d.B22_corr1_log10,s=8,alpha=.35,c=col); slope,inter=np.polyfit(d[f],d.B22_corr1_log10,1); xx=np.array([d[f].min(),d[f].max()]); ax.plot(xx,inter+slope*xx,c='black'); ax.set(title=lab,xlabel=f.replace('_',' '),ylabel='log10(-B22 Hummer)',xlim=(xmin-pad,xmax+pad))
        fig.tight_layout(); savefig(fig,figs/n)
    for lab,d,n,col in [("917 feature-guided",panel,"OCN_vs_B22_917",'#D55E00'),("Held-out sequence-exclusive",held,"OCN_vs_B22_heldout",'#0072B2')]:
        fig,ax=plt.subplots(figsize=(4.7,4)); ax.scatter(d.Opposite_Charge_Number,d.B22_corr1_log10,s=9,alpha=.4,c=col); slope,inter=np.polyfit(d.Opposite_Charge_Number,d.B22_corr1_log10,1); xx=np.array([d.Opposite_Charge_Number.min(),d.Opposite_Charge_Number.max()]); ax.plot(xx,inter+slope*xx,c='black'); ax.set(title=lab,xlabel='Opposite Charge Number',ylabel='log10(-B22 Hummer)'); fig.tight_layout(); savefig(fig,figs/n)

    cdf=pd.DataFrame(corrs); oc917=cdf[(cdf.dataset=='917 feature-guided')&(cdf.feature=='Opposite_Charge_Number')].iloc[0]; och=cdf[(cdf.dataset=='held-out sequence-exclusive')&(cdf.feature=='Opposite_Charge_Number')].iloc[0]
    report=f"""# Figure 4 Task 3 — sampling-bias audit

## Executive result

The exact full candidate landscape contains **{len(full):,} pairs** ({full.tad.nunique()} TADs × {full.rgg.nunique():,} RGGs). The exact Figure 4 panel contains **917 unique pairs**. The held-out dataset is the complete **23 × 78 = 1,794-pair** cross-product and all 1,794 pairs have actual simulation-derived Hummer-corrected B22 values.

The held-out sequences are fully exclusive from the 917 panel: **{len(ov_t)} overlapping TADs and {len(ov_r)} overlapping RGGs**. However, the available files do not contain executable code plus a seed that reproduces the original choice of the 23 and 78 sequences. They establish saved-list provenance and exclusivity, not a reproducible claim of uniform random pre-fit selection.

OCN versus log10(-B22) is Pearson r = **{oc917.Pearson_r:.3f}** (95% pair-bootstrap CI {oc917.Pearson_CI_low:.3f} to {oc917.Pearson_CI_high:.3f}) in the feature-guided 917 panel and r = **{och.Pearson_r:.3f}** (95% pair-bootstrap CI {och.Pearson_CI_low:.3f} to {och.Pearson_CI_high:.3f}) in the sequence-exclusive held-out simulations. See the clustered-bootstrap table for inference respecting the crossed 23-by-78 sequence design.

## Provenance interpretation

- Dataset A is a complete sequence-derived feature cross-product; it has no B22.
- Dataset B is the exact Figure 4 cache mapped without duplicates to its feature/B22 table. Historical scripts document feature-guided stages and seed 42 in some sampling blocks, but no single end-to-end manifest was found.
- Dataset C has a saved 101-ID list, executable cross-product construction, simulation preparation/analysis scripts, PDB evidence, and a complete simulation-derived B22 table. The original selection draw of the 23 and 78 IDs is not reproducible from an available seed/code record.

## Methods

Feature definitions follow the existing repository tables. OCN is `-(charge_RGG × charge_TAD)`; normalized OCN is OCN divided by total length; charge complementarity, Average NCPR, Average lambda, length, sigma, and average charge retain repository definitions. Distribution comparisons report effect sizes rather than relying on large-n p-values. Deciles are defined on Dataset A. Correlation CIs use {N_BOOT:,} deterministic pair bootstraps (seed {SEED}). The held-out sequence-aware CI uses {N_BOOT:,} two-way pigeonhole bootstraps, independently resampling 23 TAD and 78 RGG identities (seed {SEED+9}). Matching uses 5-bin coarsened exact matching on OCN, total length, and Average lambda without B22.

## Caveat

Sequence exclusivity is confirmed. Independence from feature-guided *pair* selection is supported structurally because none of the held-out sequences occurs in the 917 panel. The stronger statement that held-out sequences were selected randomly before model fitting or before B22 inspection is **not documented reproducibly** by the available files and should not be asserted without an external historical record.
"""
    (out/'TASK3_REPORT.md').write_text(report)
    print(report)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,default=DEFAULT_OUT); a=ap.parse_args(); main(a.out)
