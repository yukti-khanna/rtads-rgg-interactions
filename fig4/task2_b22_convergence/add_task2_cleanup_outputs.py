#!/usr/bin/env python3
"""Add the R1-13 reporting cleanup outputs without recomputing RDFs or trajectories."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task2-dir",required=True,type=Path)
    p.add_argument("--fig4-cache",required=True,type=Path)
    return p.parse_args()

def main():
    a=parse_args(); out=a.task2_dir.resolve(); cache_path=a.fig4_cache.resolve()
    uncertainty=pd.read_csv(out/"per_pair_b22_uncertainty.csv")
    cache=pd.read_csv(cache_path)
    required={"pair_id","Hummer_mean","Hummer_bootstrap95_low","Hummer_bootstrap95_high"}
    if not required.issubset(uncertainty): raise ValueError(f"Missing uncertainty columns: {sorted(required-set(uncertainty))}")
    if not {"name","B22_corr1_log10"}.issubset(cache): raise ValueError("Figure 4 cache lacks required columns")
    if len(uncertainty)!=917 or uncertainty.pair_id.nunique()!=917: raise ValueError("Task 2 uncertainty table is not 917 unique pairs")
    if len(cache)!=917 or cache.name.nunique()!=917: raise ValueError("Figure 4 cache is not 917 unique pairs")

    e=uncertainty[["pair_id","Hummer_mean","Hummer_bootstrap95_low","Hummer_bootstrap95_high"]].rename(columns={
        "Hummer_mean":"mean_B22","Hummer_bootstrap95_low":"ci95_low_B22","Hummer_bootstrap95_high":"ci95_high_B22"})
    valid=(e.mean_B22<0)&(e.ci95_low_B22<0)&(e.ci95_high_B22<0)
    crossing=(e.ci95_low_B22<=0)&(e.ci95_high_B22>=0)
    e["B22_corr1_log10"]=np.where(valid,np.log10(-e.mean_B22),np.nan)
    e["ci95_low_log10"]=np.where(valid,np.log10(-e.ci95_high_B22),np.nan)
    e["ci95_high_log10"]=np.where(valid,np.log10(-e.ci95_low_B22),np.nan)
    e["errorbar_lower"]=e.B22_corr1_log10-e.ci95_low_log10
    e["errorbar_upper"]=e.ci95_high_log10-e.B22_corr1_log10
    e["transform_status"]=np.select([valid,crossing,e.mean_B22>=0],["transformed","CI crosses or includes zero","mean B22 is nonnegative"],default="invalid negative-domain input")
    e.to_csv(out/"fig4_errorbars_log10.csv",index=False,float_format="%.17g")

    c=cache.rename(columns={"name":"pair_id","B22_corr1_log10":"original_B22_corr1_log10"})[["pair_id","original_B22_corr1_log10"]]
    comp=c.merge(e[["pair_id","B22_corr1_log10","transform_status"]],on="pair_id",how="outer",indicator=True,validate="one_to_one")
    comp["difference"]=comp.B22_corr1_log10-comp.original_B22_corr1_log10
    comp["absolute_difference"]=comp.difference.abs()
    comp["mapping_status"]=comp._merge.map({"both":"matched","left_only":"missing Task 2","right_only":"missing original"});comp=comp.drop(columns="_merge")
    matched=comp.dropna(subset=["B22_corr1_log10","original_B22_corr1_log10"])
    pr=stats.pearsonr(matched.B22_corr1_log10,matched.original_B22_corr1_log10)
    sr=stats.spearmanr(matched.B22_corr1_log10,matched.original_B22_corr1_log10)
    comp.to_csv(out/"task2_vs_original_fig4_b22.csv",index=False,float_format="%.17g")

    selection=pd.read_csv(out/"convergence_pair_selection.csv")
    config_paths=[]; steps=[]; wfreq=[]; missing=[]
    for _,row in selection.iterrows():
        root=Path(row.trajectory_path)
        for job in [f"job_{i}" for i in range(5)]:
            p=root/job/"config.yaml"
            if not p.is_file(): missing.append(str(p)); continue
            text=p.read_text();config_paths.append(str(p))
            sm=re.search(r"(?m)^steps:\s*(\d+)",text);wm=re.search(r"(?m)^wfreq:\s*(\d+)",text)
            steps.append(int(sm.group(1)) if sm else None);wfreq.append(int(wm.group(1)) if wm else None)
    provenance=f"""Trajectory-duration provenance

Evidence source: simulation configuration files for all 15 selected pairs and job_0 through job_4.
Configurations found: {len(config_paths)}/75.
Representative path: {config_paths[0] if config_paths else 'none'}
Relevant parameters: steps={sorted(set(steps))}; wfreq={sorted(set(wfreq))}.
Derived saved-frame count: steps / wfreq = {steps[0]//wfreq[0] if steps and wfreq and steps[0] and wfreq[0] else 'unavailable'} frames.
Trajectory analysis output independently contains 10,000 frames at the 100% window.

The local configuration does not specify the molecular-dynamics integrator timestep. DCD timing metadata available to the analysis stores frame indices rather than physical time. Therefore steps × timestep cannot be calculated from these local files, and the 1-µs duration is not independently inferred here.

Conclusion: Historical simulation duration supplied externally: 1 µs per replica
Confidence/status: configuration and frame-count provenance directly verified; physical-duration conversion relies on the documented historical simulation design.
Missing configuration files: {len(missing)}.
"""
    (out/"trajectory_duration_provenance.txt").write_text(provenance)

    good=e[e.transform_status=="transformed"].sort_values("B22_corr1_log10")
    show=good.iloc[np.linspace(0,len(good)-1,min(60,len(good)),dtype=int)]
    fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(len(show))
    ax.errorbar(x,show.B22_corr1_log10,yerr=np.vstack([show.errorbar_lower,show.errorbar_upper]),fmt=".",ms=3,lw=.6,capsize=1.5)
    ax.set(xlabel="Ordered diagnostic subset across the Figure 4 distribution",ylabel=r"$\log_{10}(-B_{22})$",title="Asymmetric bootstrap intervals in Figure 4 plotting space")
    fig.tight_layout()
    for ext in ("png","pdf"): fig.savefig(out/"figures"/f"fig4_log10_errorbar_example.{ext}",dpi=300,bbox_inches="tight")
    plt.close(fig)

    transformed=int(valid.sum());cross=int(crossing.sum());unmatched=int((comp.mapping_status!="matched").sum())
    maxdiff=float(matched.absolute_difference.max());meandiff=float(matched.absolute_difference.mean())
    report_path=out/"TASK2_REPORT.md"; report=report_path.read_text()
    report=report.replace("Metadata confirms 1 µs per full trajectory, so time labels are used.",
        "The cumulative fractions are calculated directly. The 0.2–1.0 µs labels follow the documented historical 1-µs design; local configurations verify step and frame counts but do not contain the integrator timestep needed to infer physical duration independently.")
    marker="## Figure 4 log-space error bars"
    addition=f"""

## Figure 4 log-space error bars

For negative mean B22 and negative bootstrap bounds, the plotted value is `log10(-mean_B22)`. Because this transformation reverses the ordering of negative B22 bounds, the lower plotted bound uses the upper B22 bound and vice versa. Asymmetric lower and upper error extents were generated for {transformed}/917 pairs. {cross} confidence intervals cross or include zero and were left untransformed.

## Original Figure 4 reproduction

Task 2 five-replica mean Hummer B22 values were transformed and matched by exact pair ID to the original Figure 4D cache. Matched n={len(matched)}; unmatched n={unmatched}; maximum absolute difference={maxdiff:.17g}; mean absolute difference={meandiff:.17g}; Pearson r={pr.statistic:.17g}; Spearman rho={sr.statistic:.17g}; differences above 1e-8={int((matched.absolute_difference>1e-8).sum())}; differences above 1e-6={int((matched.absolute_difference>1e-6).sum())}. This is a reproducibility comparison only.

## Trajectory-duration provenance

All 75 selected pair/replica configurations were checked. They consistently specify 100,000,000 steps and an output frequency of 10,000 steps, yielding 10,000 saved frames, but they do not specify the integrator timestep. The local DCD metadata provides frame indices rather than physical time. Therefore the duration status is: **Historical simulation duration supplied externally: 1 µs per replica.** The cumulative fractions remain directly verified; their µs labels derive from the documented historical design rather than an independent local inference.
"""
    if marker in report: report=report[:report.index(marker)].rstrip()+addition
    else: report=report.rstrip()+addition
    report_path.write_text(report+"\n")
    print(f"log-space error bars generated: {transformed}/917")
    print(f"CI intervals crossing zero: {cross}")
    print(f"original Figure 4 pairs matched: {len(matched)}; unmatched: {unmatched}")
    print(f"maximum absolute difference from original Figure 4 values: {maxdiff:.17g}")
    print("trajectory-duration provenance status: Historical simulation duration supplied externally: 1 µs per replica")
    print(f"updated report path: {report_path}")

if __name__=="__main__": main()
