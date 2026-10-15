"""Render a design preview only; do not modify the final figure package."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figure_previews' / 'panel_c_simple'
OUT.mkdir(parents=True, exist_ok=True)
stats = pd.read_csv(ROOT / 'revision_analysis/tables/discrimination_statistics.csv')
# Show one consistent grouping in this draft. AHD groups keep WT/shuffles
# and measurements sharing the same AHD parent together. RGG sensitivity
# remains necessary in the supplement; this choice is not based on CI width.
stats = stats[stats.cluster.eq('AHD_parent')].copy()
plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                     'svg.fonttype':'none', 'pdf.fonttype':42})
fig, ax = plt.subplots(figsize=(6.2, 3.9))
fig.subplots_adjust(left=.30, right=.95, bottom=.22, top=.77)
colors = {'Predicted':'#0072B2', 'Simulated':'#D55E00'}
comparisons = [('Kd_vs_NB','Kd fit vs ND'),
               ('Kd_LB_vs_NB','Kd fit + LB vs ND'),
               ('LB_vs_NB','LB vs ND')]
ax.axhspan(-.39,.39, color='#F0F4F7', zorder=0)
ax.axvline(.5, color='#858585', ls=(0,(2,3)), lw=.9, zorder=1)
labels=[]
for i,(comparison,label) in enumerate(comparisons):
    g=stats[stats.comparison.eq(comparison)]
    labels.append(f'{label}\n{int(g.iloc[0].positive)} vs {int(g.iloc[0].negative)} pairs')
    for series, offset, marker in [('Predicted',-.14,'o'),('Simulated',.14,'^')]:
        r=g[g.series.eq(series)].iloc[0]
        ax.errorbar(r.AUROC, i+offset,
                    xerr=[[r.AUROC-r.CI95_low],[r.CI95_high-r.AUROC]],
                    fmt=marker, markersize=6, color=colors[series],
                    elinewidth=1.5, capsize=2.5, markeredgecolor='white',
                    markeredgewidth=.6, zorder=3)
ax.set(yticks=range(3), yticklabels=labels, ylim=(2.48,-.48),
       xlim=(.45,1.025), xticks=[.5,.6,.7,.8,.9,1.0], xlabel='AUROC')
ax.tick_params(axis='y',length=0,pad=12)
ax.tick_params(axis='x',length=3,color='#777777')
ax.spines[['left','top','right']].set_visible(False)
ax.spines['bottom'].set_bounds(.5,1.)
ax.spines['bottom'].set_color('#777777')
ax.get_yticklabels()[0].set_fontweight('bold')
fig.text(.05,.935,'C', fontsize=14, weight='bold')
fig.text(.30,.935,'Experimental outcome discrimination',fontsize=11,weight='bold')
fig.legend(handles=[Line2D([],[],color=colors[s], marker=m,ls='-',
                           markersize=5,label=s) for s,m in
                    [('Predicted','o'),('Simulated','^')]],
           loc='upper left',bbox_to_anchor=(.30,.89),ncol=2,
           frameon=False,fontsize=8.5,borderaxespad=0,handlelength=1.6)
fig.text(.30,.06,'95% intervals grouped by AHD parent · dotted line: chance',
         fontsize=7.5,color='#555555')
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'Panel_C_preview.{ext}',dpi=220)
plt.close(fig)
stats.to_csv(OUT/'Panel_C_preview_statistics.csv',index=False)
(OUT/'README.txt').write_text(
    'Design preview only. Final figure package has not been changed.\n'
    'Intervals: AHD-parent percentile bootstrap, 2,000 requested resamples.\n'
    'RGG-parent grouping remains a necessary supplementary sensitivity.\n'
    'Primary comparison shaded. No affinity ordering assigned to LB.\n'
    'Uses current outcomes, including provisional LB assignments for '
    'p53–THOC4-C and shuffled LEF1–THOC4-N, pending experimental confirmation.\n')
print(OUT)
