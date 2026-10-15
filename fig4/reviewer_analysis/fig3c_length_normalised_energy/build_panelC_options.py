#!/usr/bin/env python3
"""Alternative displays of Main Figure 3 panel C's saved contributions."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

OUT=Path(os.environ.get('FIG3C_OUTPUT_DIR', Path(__file__).resolve().parent))
CLASSES=['Basic','Acidic','Aromatic','Other']
SEED=20261005
NBOOT=10000

def plot(matrix, stem, label, fractional=False, secondary=False, logarithmic=False, compact=False):
    fig,ax=plt.subplots(figsize=(3.4,2.8) if compact else (4.1,3.3))
    data=matrix.copy()
    top=np.unravel_index(np.argmax(matrix),matrix.shape)
    cmap=plt.get_cmap('YlOrRd').copy(); cmap.set_bad('#E5E5E5')
    if secondary:
        data=np.ma.array(data,mask=False); data.mask[top]=True
    if logarithmic:
        data=np.ma.masked_less_equal(data,0)
        positive=matrix[matrix>0]
        kwargs={'norm':LogNorm(vmin=positive.min(),vmax=positive.max())}
    else:
        kwargs={'vmin':0,'vmax':data.max()}
    edges=np.arange(5)-.5
    im=ax.pcolormesh(edges,edges,data,cmap=cmap,shading='flat',edgecolors='white',linewidth=.8,antialiased=False,**kwargs)
    ax.set(xlim=(-.5,3.5),ylim=(3.5,-.5),aspect='equal',xlabel='AHD residue class',ylabel='RG/RGG residue class')
    ax.set_xticks(range(4),CLASSES); ax.set_yticks(range(4),CLASSES)
    for i in range(4):
        for j in range(4):
            value=matrix[i,j]
            text=f'{value:.1f}%' if fractional else f'{value:.3g}'
            if secondary and (i,j)==top:
                text+='*'
                color='black'
            else:
                scale=float(im.norm(value)) if value>0 else 0
                color='white' if scale>.65 else 'black'
            ax.text(j,i,text,ha='center',va='center',fontsize=5.8 if compact else 6.5,color=color)
    if secondary:
        ax.add_patch(Rectangle((top[1]-.5,top[0]-.5),1,1,fill=False,edgecolor='#555555',linewidth=1.1))
        fig.text(.5,.015,'* Secondary scaling excluding dominant cell',ha='center',fontsize=5.5 if compact else 6)
    ax.text(-.25,1.06,'C',transform=ax.transAxes,fontsize=10,fontweight='bold')
    bar=fig.colorbar(im,ax=ax,fraction=.047,pad=.05)
    bar.set_label(label,fontsize=6 if compact else 7)
    fig.tight_layout(rect=(0,.045 if secondary else 0,1,1))
    fig.savefig(OUT/(stem+'.pdf'),bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/(stem+'.png'),dpi=600,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.linewidth':.6,'pdf.fonttype':42})
    source=pd.read_csv(OUT/'Fig3C_length_normalised_energy_per_pair.csv')
    index=pd.MultiIndex.from_product([CLASSES,CLASSES],names=['RGG_class','AHD_class'])
    values=source.pivot(index='pair_id',columns=['RGG_class','AHD_class'],values='length_normalised_attractive_energy').reindex(columns=index).to_numpy(float)
    assert values.shape==(917,16) and np.isfinite(values).all() and (values>=0).all()
    totals=values.sum(axis=1)
    if (totals<=0).any():
        raise ValueError(f'{(totals<=0).sum()} simulations have no attractive energy; fractional average requires an explicit eligibility decision.')
    fractions=values/totals[:,None]
    mean=fractions.mean(axis=0); direct=values.mean(axis=0)
    np.testing.assert_allclose(mean.sum(),1,atol=1e-12)
    rng=np.random.default_rng(SEED); boot=np.empty((NBOOT,16))
    for start in range(0,NBOOT,100):
        indices=rng.integers(0,917,size=(min(100,NBOOT-start),917))
        boot[start:start+len(indices)]=fractions[indices].mean(axis=1)
    low,high=np.percentile(boot,[2.5,97.5],axis=0)
    rank=np.argsort(np.argsort(-mean,kind='stable'),kind='stable')+1
    first=np.bincount(np.argmax(boot,axis=1),minlength=16)/NBOOT
    rows=[]; bootstrap=[]
    for k in range(16):
        base={'RGG_class':CLASSES[k//4],'AHD_class':CLASSES[k%4]}
        rows.append(dict(base,mean_fraction=mean[k],mean_percentage=100*mean[k],rank=rank[k]))
        bootstrap.append(dict(base,observed_mean_fraction=mean[k],observed_mean_percentage=100*mean[k],bootstrap_q025_fraction=low[k],bootstrap_q975_fraction=high[k],rank=rank[k],bootstrap_fraction_rank1=first[k]))
    pd.DataFrame(rows).to_csv(OUT/'Fig3C_fractional_contribution_source.csv',index=False)
    pd.DataFrame(bootstrap).to_csv(OUT/'Fig3C_fractional_contribution_bootstrap.csv',index=False)
    key=pd.DataFrame([dict(bootstrap[k],bootstrap_q025_percentage=100*low[k],bootstrap_q975_percentage=100*high[k],ratio_basic_acidic_to_acidic_basic=mean[1]/mean[4],basic_acidic_is_top=bool(rank[1]==1)) for k in [1,4]])
    key.to_csv(OUT/'Fig3C_fractional_key_cells.csv',index=False)
    plot(direct.reshape(4,4),'Fig3C_contribution_reference','Length-normalised attractive-energy contribution')
    fractional_label='Mean fractional contribution to\nattractive interaction energy (%)'
    plot(mean.reshape(4,4)*100,'Fig3C_fractional_contribution_main',fractional_label,fractional=True)
    plot(mean.reshape(4,4)*100,'Fig3C_fractional_contribution_secondary_scale',fractional_label,fractional=True,secondary=True)
    plot(mean.reshape(4,4)*100,'Fig3C_fractional_contribution_secondary_scale_inset',fractional_label,fractional=True,secondary=True,compact=True)
    plot(direct.reshape(4,4),'Fig3C_contribution_logscale','Length-normalised attractive-energy contribution\n(log colour scale; zero cells masked)',logarithmic=True)
    t=int(np.argmax(mean)); dt=int(np.argmax(direct))
    component_report=(OUT/'Fig3C_length_normalised_energy_summary.md').read_text()
    yukawa='Dominant signal mainly Yukawa: YES' in component_report
    summary=f'''# Main Figure 3 panel C display options

1. Direct contribution dominant cell: {CLASSES[dt//4]}-RG/RGG × {CLASSES[dt%4]}-AHD.
2. Fractional contribution dominant cell: {CLASSES[t//4]}-RG/RGG × {CLASSES[t%4]}-AHD; its fractions sum to {mean.sum():.12f} across all cells. All 917 pairs have positive total attractive energy.
3. The preferred fractional panel is easier to interpret numerically through percentages, although a dominant cell still controls its linear colour range.
4. The secondary scale reveals weaker contributors while preserving the dominant cell as a grey, outlined box with its actual percentage and an asterisk identifying exclusion from colour scaling.
5. The log-scale version is acceptable for comparing nonzero magnitudes; zero cells are masked grey and annotated 0. No positive floor or pseudocount is used.
6. Recommended main panel: fractional contribution. Recommended optional secondary panel: fractional secondary-scale inset.
7. Basic-RG/RGG × Acidic-AHD: {100*mean[1]:.3f}% (95% CI {100*low[1]:.3f}–{100*high[1]:.3f}%); rank {rank[1]}/16; ranked first in {first[1]:.2%} of bootstraps. Ratio to the reverse cell: {mean[1]/mean[4]:.4f}.
8. Acidic-RG/RGG × Basic-AHD: {100*mean[4]:.3f}% (95% CI {100*low[4]:.3f}–{100*high[4]:.3f}%); rank {rank[4]}/16.
9. Dominant direct signal mainly Yukawa: {'YES' if yukawa else 'NO'}, from the existing component check. Bootstrap: 10,000 equal-weight resamples of per-pair fractional matrices; seed {SEED}.
'''
    (OUT/'Fig3C_panelC_options_summary.md').write_text(summary)
    print('TOTAL PAIRS: 917')
    print(f'DIRECT CONTRIBUTION TOP CELL: {CLASSES[dt//4]} x {CLASSES[dt%4]}\nDIRECT CONTRIBUTION TOP VALUE: {direct[dt]:.9g}')
    print(f'FRACTIONAL CONTRIBUTION TOP CELL: {CLASSES[t//4]} x {CLASSES[t%4]}\nFRACTIONAL CONTRIBUTION TOP PERCENTAGE: {100*mean[t]:.6f}\nFRACTIONAL CONTRIBUTION TOP 95% CI: [{100*low[t]:.6f}, {100*high[t]:.6f}]%\nBOOTSTRAP FRACTION RANKED FIRST: {first[t]:.6f}')
    for k,title in [(1,'BASIC-RG/RGG x ACIDIC-AHD'),(4,'ACIDIC-RG/RGG x BASIC-AHD')]:
        print(f'{title}:\nPERCENTAGE: {100*mean[k]:.6f}\n95% CI: [{100*low[k]:.6f}, {100*high[k]:.6f}]%\nRANK: {rank[k]}')
    print(f'RATIO BASIC-RG/RGG x ACIDIC-AHD TO ACIDIC-RG/RGG x BASIC-AHD: {mean[1]/mean[4]:.6f}')
    print('SECONDARY-SCALE VERSION CREATED: YES\nLOG-SCALE VERSION CREATED: YES\nRECOMMENDED MAIN PANEL: fractional contribution\nRECOMMENDED OPTIONAL SECONDARY PANEL: fractional secondary-scale inset')
    for p in sorted(OUT.glob('Fig3C*')):
        if 'length_normalised' not in p.name: print(p)

if __name__=='__main__': main()
