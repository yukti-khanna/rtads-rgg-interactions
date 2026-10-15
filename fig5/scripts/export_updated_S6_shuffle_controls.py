#!/usr/bin/env python3
"""Export newer-control shuffle panels from accepted predictions only."""
from pathlib import Path
import importlib.util
import hashlib
import json
import shutil
import os

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/revised_figS6_shuffle_controls'
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def plot_absolute_b():
    a = load('absolute_shuffle_b', ROOT / 's8_extended_controls/S8B_absolute_shuffle_prediction_build.py')
    df = a.pd.read_csv(OUT / 'S8B_shuffle_source_audited.csv')
    a.plt.rcParams.update({'svg.fonttype':'none', 'axes.linewidth':.6})
    fig, ax = a.plt.subplots(figsize=a.FIGSIZE)
    rows = []
    for cls, label, color, size, alpha in [('TAD/AHD shuffle','rAHD',a.BLUE,8,.6), ('RG/RGG shuffle','RG/RGG',a.ORANGE,10,.7)]:
        d = df[df.shuffle_class.eq(cls)]
        x, y = d.simulated_shuffled_target, d.predicted_shuffled_target
        ax.scatter(x,y,s=size,alpha=alpha,color=color,edgecolors='none',label=f'{label} shuffle',rasterized=False)
        rows.append({'shuffle_class':cls, **a.metric_abs(x,y)})
    lo, hi = a.limits(df.simulated_shuffled_target,df.predicted_shuffled_target)
    ax.plot([lo,hi],[lo,hi],lw=.8,color='black')
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel=r'Simulated $\log_{10}(-B_{22})$',ylabel=r'Predicted $\log_{10}(-B_{22})$')
    ax.set_aspect('equal',adjustable='box')
    ax.legend(loc='upper left',frameon=False,fontsize=6.5)
    blocks=[]
    for label, m in zip(['rAHD','RG/RGG'],rows):
        blocks.append(f"{label}: n = {m['n']}\nR² = {m['R2']:.3f}, r = {m['Pearson']:.3f}\nRMSE = {m['RMSE']:.3f}")
    ax.text(.98,.03,'\n\n'.join(blocks),transform=ax.transAxes,ha='right',va='bottom',fontsize=6)
    a.clean(ax)
    fig.tight_layout(pad=.4)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT / f'FigS6B_shuffle_absolute_prediction.{ext}',dpi=600)
    a.plt.close(fig)
    a.pd.DataFrame(rows).to_csv(OUT / 'FigS6B_absolute_prediction_statistics.csv',index=False)

def main():
    absolute = load('absolute_shuffle', ROOT / 's8_extended_controls/S8B_absolute_shuffle_prediction_build.py')
    uncertainty = load('shuffle_uncertainty', ROOT / 'scripts/run_predictor_shuffle_uncertainty.py')
    pd, np, plt = absolute.pd, absolute.np, absolute.plt
    absolute.OUT = OUT
    plt.rcParams['svg.fonttype'] = 'none'
    absolute.main()
    for ext in ['pdf', 'png']:
        (OUT / f'S8B_shuffle_absolute_prediction.{ext}').rename(OUT / f'FigS6B_shuffle_absolute_prediction.{ext}')
    plot_absolute_b()

    source = pd.read_csv(OUT / 'S8B_shuffle_source_audited.csv')
    cache = ROOT / 'fig5_shuffle_uncertainty'
    replicas = pd.read_csv(cache / 'predictor_shuffle_replica_data.csv')
    summary = pd.read_csv(cache / 'predictor_delta_uncertainty_summary.csv')
    current = pd.read_csv(cache / 'current_delta_analysis.csv')
    assert len(source) == 345 and source.shuffle_pair_id.is_unique
    assert set(source.shuffle_pair_id) == set(replicas.shuffle_pair_id)
    joined = source.merge(replicas, on='shuffle_pair_id', validate='one_to_one')
    assert (joined.WT_parent_pair_id == joined.WT_pair_id).all()
    differences = {}
    for a, b in [('simulated_WT_target','WT_target'), ('simulated_shuffled_target','shuffle_target'),
                 ('predicted_WT_target','prediction_WT'), ('predicted_shuffled_target','prediction_shuffle'),
                 ('delta_simulated','delta_sim'), ('delta_predicted','delta_pred')]:
        differences[a] = float(np.max(np.abs(joined[a] - joined[b])))
        assert differences[a] < 1e-12, (a, differences[a])
    assert source.groupby('shuffle_class').size().to_dict() == {'TAD/AHD shuffle':230, 'RG/RGG shuffle':115}
    # Reuse the completed 20,000-replica-resample uncertainty estimates only
    # after establishing identical pair IDs, parents, targets and predictions.
    for filename in ['predictor_delta_uncertainty_summary.csv', 'current_delta_analysis.csv']:
        shutil.copy2(cache / filename, OUT / filename)
    plt.rcParams.update({'axes.linewidth':.6, 'pdf.fonttype':42, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(1, 2, figsize=(5.1,2.5))
    for ax, group in zip(axes, uncertainty.GROUPS):
        uncertainty.draw_panel(ax, group, '', replicas, current, summary)
    fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.87,wspace=.45)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT / f'FigS6C_shuffle_delta_uncertainty.{ext}', dpi=600)
    plt.close(fig)

    # Updated version of the old B perturbation view, retained as a diagnostic.
    fig, ax = plt.subplots(figsize=(2.4,2.3))
    for cls, color, size, alpha in [('TAD/AHD shuffle',absolute.BLUE,8,.6), ('RG/RGG shuffle',absolute.ORANGE,10,.7)]:
        d = source[source.shuffle_class.eq(cls)]
        ax.scatter(d.delta_simulated,d.delta_predicted,s=size,alpha=alpha,color=color,edgecolors='none',label=cls)
    lo, hi = absolute.limits(source.delta_simulated,source.delta_predicted)
    ax.axhline(0,color='#888888',lw=.5,zorder=0)
    ax.axvline(0,color='#888888',lw=.5,zorder=0)
    ax.plot([lo,hi],[lo,hi],color='black',lw=.8,zorder=0)
    ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel=r'Simulated $\Delta$',ylabel=r'Predicted $\Delta$')
    ax.set_aspect('equal',adjustable='box')
    ax.legend(loc='upper left',frameon=False,fontsize=6)
    ahd=current[current.shuffle_type.eq('AHD')].iloc[0]
    rgg=current[current.shuffle_type.eq('RG/RGG')].iloc[0]
    ax.text(.97,.03,f'AHD: n=230, r={ahd.Pearson:.3f}\nRG/RGG: n=115, r={rgg.Pearson:.3f}',transform=ax.transAxes,ha='right',va='bottom',fontsize=6.5)
    absolute.clean(ax)
    fig.tight_layout(pad=.4)
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT / f'FigS6B_shuffle_delta_diagnostic.{ext}',dpi=600)
    plt.close(fig)
    inputs = [absolute.SRC, cache/'predictor_shuffle_replica_data.csv', cache/'predictor_delta_uncertainty_summary.csv']
    (OUT/'input_audit.json').write_text(json.dumps({'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'cached_uncertainty_target_prediction_max_differences':differences},indent=2))
    (OUT/'README.md').write_text('''# Updated Figure S6 shuffle controls

230 AHD shuffles (160 original + 70 new), 115 RG/RGG shuffles (80 original + 35 new), and 115 unique WT parents.

- FigS6B_shuffle_absolute_prediction: absolute simulated versus predicted shuffled-pair strength, using the previously requested primary validation view.
- FigS6B_shuffle_delta_diagnostic: updated version of the old B difference-versus-difference view; retained as a secondary diagnostic, not an additional primary panel.
- FigS6C_shuffle_delta_uncertainty: existing extended-control uncertainty analysis re-exported, with identical statistics. Its axes follow the existing uncertainty script (predicted delta on x, simulated delta on y).

PDFs are vector; PNGs are exported at 600 dpi. Existing plotting functions were reused from s8_extended_controls/S8B_absolute_shuffle_prediction_build.py and scripts/run_predictor_shuffle_uncertainty.py. Original/new/combined statistics and audited source tables retain their historical S8 filenames in this directory.

The cached uncertainty analysis already included all 345 shuffles. Pair IDs, WT parents, targets and predictions were checked against the latest source before reusing its 20,000-resample results. The displayed interval propagates finite-replica simulation uncertainty conditional on these pairs; it is not a population confidence interval.

No simulations, RDF/B22 calculations, feature generation, model inference or fitting were run. Other panels and the manuscript DOCX were not modified.
''')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--b-only', action='store_true', help='Replot only absolute S6B from the audited source.')
    args = parser.parse_args()
    plot_absolute_b() if args.b_only else main()
