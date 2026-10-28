"""S8E: p53–THOC4-N salt-dependent CSPs; row matching confirmed by user."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'final_figure_package'
FILES={150:('p53_ref_150_nacl_thoc4n','p53_ref_150_nacl_thoc4n_8eq'),450:('p53_ref_450_nacl_thoc4n_ref','p53_ref_450_nacl_thoc4n_8eq')}
def build():
 rows=[];hashes={};n=None
 for salt,paths in FILES.items():
  frames=[]
  for name in paths:
   path=ROOT/name;d=pd.read_csv(path)
   if list(d.columns)!=['#','Position F1','Position F2'] or d['#'].duplicated().any():raise ValueError(f'Invalid peak list: {name}')
   if not np.isfinite(d[['Position F1','Position F2']].to_numpy(float)).all():raise ValueError(name)
   if n is None:n=len(d)
   if len(d)!=n:raise ValueError('Row-matched lists must have equal lengths')
   frames.append(d);hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
  ref,bound=frames
  for i in range(n):
   h=float(bound.iloc[i,1]-ref.iloc[i,1]);nn=float(bound.iloc[i,2]-ref.iloc[i,2])
   rows.append(dict(peak_order=i+1,reference_peak_id=int(ref.iloc[i,0]),bound_peak_id=int(bound.iloc[i,0]),salt_mM=salt,ligand_equivalents=8,H_ref_ppm=ref.iloc[i,1],N_ref_ppm=ref.iloc[i,2],H_8eq_ppm=bound.iloc[i,1],N_8eq_ppm=bound.iloc[i,2],delta_H_ppm=h,delta_N_ppm=nn,CSP_ppm=np.hypot(h,nn/7)))
 source=pd.DataFrame(rows);matrix=source.pivot(index='salt_mM',columns='peak_order',values='CSP_ppm').loc[[150,450]]
 paired=matrix.T.rename(columns={150:'CSP_150_ppm',450:'CSP_450_ppm'})
 paired['delta_450_minus_150_ppm']=paired.CSP_450_ppm-paired.CSP_150_ppm
 stats=source.groupby('salt_mM').CSP_ppm.agg(['count','mean','median','max']).reset_index()
 stats['peaks_above_display_cutoff']=[int((matrix.loc[s]>.001).sum()) for s in stats.salt_mM]
 folder=OUT/'source_data';folder.mkdir(parents=True,exist_ok=True)
 source.to_csv(folder/'FigureS8E_salt_CSP_all_peaks.csv',index=False);paired.to_csv(folder/'FigureS8E_salt_paired_changes.csv');stats.to_csv(folder/'FigureS8E_salt_summary.csv',index=False)
 display=matrix.loc[:,matrix.max()>.001];order=display.loc[150].sort_values(ascending=False,kind='stable').index;display=display[order].mask(lambda x:x<=.001)
 display.to_csv(folder/'FigureS8E_salt_heatmap_matrix.csv')
 plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'svg.fonttype':'none','pdf.fonttype':42})
 from matplotlib.font_manager import findfont
 findfont('Arial',fallback_to_default=False)
 fig,ax=plt.subplots(figsize=(9/2.54,5/2.54));cmap=LinearSegmentedColormap.from_list('salt',[(1,1,1),(1,.6,.2),(.3,.15,.1)]);cmap.set_bad('white')
 im=ax.imshow(display,aspect='auto',cmap=cmap,vmin=0,vmax=np.nanmax(display),interpolation='nearest')
 ticks=list(range(0,len(order),max(1,int(np.ceil(len(order)/18)))))
 ax.set_xticks(ticks,[str(order[i]) for i in ticks],rotation=90);ax.set_yticks([0,1],['150','450']);ax.set_ylabel('NaCl (mM)');ax.set_xlabel('Matched peak index (ranked by 150 mM CSP)');ax.tick_params(length=0)
 for spine in ax.spines.values():spine.set_visible(False)
 fig.colorbar(im,ax=ax,fraction=.035,pad=.025,label='CSP (ppm)');fig.text(.025,.96,'E',weight='bold',va='top');fig.subplots_adjust(left=.15,right=.86,bottom=.30,top=.88)
 dest=OUT/'figures/manuscript_panels';dest.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf','svg']:fig.savefig(dest/f'FigureS8E.{ext}',dpi=300)
 plt.close(fig)
 med=stats.set_index('salt_mM')['median'];lower=int((paired.delta_450_minus_150_ppm<0).sum())
 caption=f'''(E) Salt-dependent CSPs for p53 in the presence of eight equivalents of THOC4-N at 150 and 450 mM NaCl. Each ligand-containing spectrum is compared with its salt-matched reference. The same {n} peaks were matched by user-confirmed row order; labels indicate matched peak indices, not residue assignments. CSP = sqrt[(delta H)^2 + (delta N/7)^2]. Columns are ranked by the 150 mM CSP and share a colour scale. Values at or below 0.001 ppm are masked for display only; columns are retained if either condition exceeds this cutoff. Summary statistics use all matched peaks. No replicate-level uncertainty is available.\n'''
 (OUT/'S8E_SALT_LEGEND.md').write_text(caption)
 interpretation=f'Median CSPs were {med[150]:.5f} ppm at 150 mM and {med[450]:.5f} ppm at 450 mM NaCl; {lower} of {n} matched peaks had smaller CSPs at 450 mM. '
 interpretation+='Reduced perturbations at higher salt are consistent with a salt-sensitive contribution to the p53–THOC4-N response. They do not by themselves quantify a change in Kd, demonstrate loss of binding, or establish a uniquely electrostatic mechanism. Peaks are correlated observations within spectra, not independent experimental replicates.\n'
 (OUT/'S8E_SALT_INTERPRETATION.md').write_text(interpretation)
 (folder/'FigureS8E_salt_provenance.json').write_text(json.dumps(dict(input_sha256=hashes,pairing='row order confirmed by user',labels='peak indices, not residue assignments',formula='sqrt(dH^2+(dN/7)^2)',display_cutoff_ppm=.001,reference_method='fig2/scripts/plot_p53_rrgg_salt_csp_heatmap.py',n_peaks=n),indent=2))
 return source,stats
if __name__=='__main__':build()

def write_manuscript_order():
 mapping={'S8A':'Simulation-derived outcome groups (WT)','S8B':'Predicted threshold (WT)','S8C':'Held-group threshold evaluation (WT)','S8D':'Simulation-derived scores versus pKd (WT)','S8E':'p53–THOC4-N salt-dependent CSPs','S8F':'Exploratory intensity ratios (WT and shuffles)','S8G':'Histidine pH 7.0 versus 6.5 simulations','S8H':'Predictions versus pH 6.5 simulations','S9A':'Matched WT–shuffle outcomes','S9B':'Matched LEF1 all-frame energy','S9C':'Matched LEF1 contact occupancy','S9D':'Matched LEF1 contact-conditioned energy'}
 (OUT/'MANUSCRIPT_PANEL_ORDER.md').write_text('# Current manuscript panel order\n\n'+ '\n'.join(f'- {k}: {v}' for k,v in mapping.items())+'\n\nUse figures/manuscript_panels for current labels. Historical S10/S11 filenames are retained as internal compatibility outputs. Table numbering remains S11. In the manuscript, former S8E becomes S8F, former S8F becomes S8G, and former S8G becomes S8H. S8A–D and S9A–D retain their manuscript meanings.\n')
 captions='''# Figure S8 | Complementary NMR comparisons, salt dependence and histidine pH controls

(A) Simulation-derived scores for WT pairs grouped by curated experimental response. Points represent pairs; bars show medians.

(B) Predicted scores for WT Kd-fit and ND pairs relative to the cutoff selected to maximise balanced accuracy. Orange outlines identify misclassified pairs. LB is excluded. This exploratory cutoff is not a universal binding boundary.

(C) Balanced accuracy for Kd fit versus ND, Kd fit plus LB versus ND, and LB versus ND among WT pairs. Cutoffs are evaluated within the full panel and after withholding an AHD parent or RG/RGG partner during selection. Held-out predictions are pooled for evaluation.

(D) Simulation-derived scores versus pKd for the WT fitted interactions. Error bars propagate Kd fitting errors. The dashed line is a descriptive linear fit; the annotation reports Spearman correlation.

'''+(OUT/'S8E_SALT_LEGEND.md').read_text()+'''
(F) Mean-below-threshold intensity ratios versus predicted and simulated scores for measured WT and shuffle pairs. Lower ratios indicate greater attenuation. Lines are descriptive fits; annotations report Spearman correlations. This exploratory comparison does not establish Kd or an affinity ordering among LB responses.

(G) Simulation-derived log10(−B22) at pH 7.0 versus pH 6.5 for 390 matched histidine-containing pairs. The line denotes identity. Annotations report agreement statistics.

(H) Frozen predicted scores versus pH 6.5 simulation-derived scores for the same 390 pairs. The line denotes identity rather than a fitted regression. Source records and statistics are in Supplementary Table S11.

# Figure S9 | Sequence rearrangement and LEF1 partner context

(A) Curated experimental responses for composition-verified shuffle controls and their matched WT constructs against the same partner. Changes in response category are not numerical affinity changes.

(B) All-frame mean total interchain energy for LEF1 and WT comparators matched by RG/RGG partner.

(C) Fraction of frames with at least one interchain residue-bead distance below 0.8 nm under periodic boundary conditions. This operational occupancy is not an experimental bound fraction.

(D) Total interchain energy averaged over contact-containing frames. For B–D, error bars describe contiguous-block sensitivity, not independent-replica uncertainty. Energies are model-dependent observables, not additive contributions to B22.
'''
 (OUT/'MANUSCRIPT_SUPPLEMENTARY_LEGENDS.md').write_text(captions)
