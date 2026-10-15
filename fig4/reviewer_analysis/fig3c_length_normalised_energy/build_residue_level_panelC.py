#!/usr/bin/env python3
"""Decompose existing corrected Panel C energy averages; no trajectory reads."""
import argparse
import ast
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AA = list('ACDEFGHIKLMNPQRSTVWY')
CLASSES = ['Basic', 'Acidic', 'Aromatic', 'Other']
SEED = 20261005

def group(a):
    return 'Basic' if a in 'RKH' else 'Acidic' if a in 'DE' else 'Aromatic' if a in 'FYW' else 'Other'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Completed full-trajectory Panel C output directory.')
    args = parser.parse_args()
    source = args.input_dir.resolve(strict=True)
    out = source/'residue_level'
    helper = Path('/Users/yuktikhanna/calvados/scripts/amino_acids_energy_interactions.py')
    ns = dict(np=np)
    for node in ast.parse(helper.read_text()).body:
        if (isinstance(node, ast.FunctionDef) and node.name in ['_to_one_letter', '_ensure_1d_pair_vector']) or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'AA3_TO_1' for t in node.targets)):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(helper), 'exec'), ns)
    audit = pd.read_csv(ROOT/'task5_panelC_enrichment/orientation_audit_final.csv').set_index('pair_id')
    master = pd.read_csv(ROOT/'sim_pairs_added_features_data_270525.csv').set_index('name')
    sequences = {}; key = None
    for line in (ROOT/'all_tads_rggs_dis.fasta').read_text().splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0]; sequences[key] = ''
        elif key:
            sequences[key] += line.strip()
    names = audit.index.tolist()
    assert len(names) == len(set(names)) == 917
    raw = []; normalised = []; ah_components = []; yu_components = []
    counts_r = []; counts_a = []; lengths = []
    for name in names:
        path = source/'energy_cache'/f'energies_{name}_20nm_sliced.pkl'
        with path.open('rb') as stream:
            data = pickle.load(stream)
        r1, r2 = [ns['_to_one_letter'](data[k]) for k in ['res1', 'res2']]
        rid, aid = master.loc[name, ['rgg', 'tad']]
        forward = ''.join(r1) == sequences[rid] and ''.join(r2) == sequences[aid]
        reverse = ''.join(r2) == sequences[rid] and ''.join(r1) == sequences[aid]
        if forward == reverse or audit.loc[name, 'corrected_RGG_side'] != ('res1' if forward else 'res2'):
            raise ValueError(f'Exact-sequence orientation mismatch: {name}')
        ah, yu = [ns['_ensure_1d_pair_vector'](data[k], len(r1), len(r2)).reshape(len(r1), len(r2)) for k in ['ah_energies_pp', 'yu_energies_pp']]
        if not np.isfinite(ah).all() or not np.isfinite(yu).all():
            raise ValueError(f'Nonfinite energies: {name}')
        if reverse:
            r1, r2, ah, yu = r2, r1, ah.T, yu.T
        if not set(r1).issubset(AA) or not set(r2).issubset(AA):
            raise ValueError(f'Noncanonical residues: {name}')
        idx1 = np.array([AA.index(a) for a in r1]); idx2 = np.array([AA.index(a) for a in r2])
        bins = (idx1[:, None]*20 + idx2[None, :]).ravel()
        attractive = np.maximum(-(ah+yu), 0)
        collapse = lambda e: np.bincount(bins, weights=e.ravel(), minlength=400)
        energy = collapse(attractive)
        np.testing.assert_allclose(energy.sum(), attractive.sum(), atol=1e-12)
        if energy.sum() <= 0:
            raise ValueError(f'Undefined fractions for zero-attraction system: {name}; no exclusion allowed.')
        denominator = len(r1)*len(r2)
        raw.append(energy); normalised.append(energy/denominator)
        # Signed decomposition on the same total-attractive residue-pair support.
        # Components add exactly to total; repulsive offsets can be negative.
        ah_components.append(collapse(np.where(attractive>0, -ah, 0))/denominator)
        yu_components.append(collapse(np.where(attractive>0, -yu, 0))/denominator)
        counts_r.append(np.bincount(idx1, minlength=20)); counts_a.append(np.bincount(idx2, minlength=20))
        lengths.append((len(r1), len(r2)))
    raw, c = np.array(raw), np.array(normalised)
    f = c/c.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(f.sum(axis=1), 1, atol=1e-12)
    mean = f.mean(axis=0)
    class_indices = np.array([CLASSES.index(group(a))*4+CLASSES.index(group(b)) for a in AA for b in AA])
    reconstructed = np.bincount(class_indices, weights=mean, minlength=16)
    existing = pd.read_csv(source/'Fig3C_fractional_contribution_source.csv').set_index(['RGG_class', 'AHD_class'])
    class_order = pd.MultiIndex.from_product([CLASSES, CLASSES])
    reference = existing.reindex(class_order).mean_fraction.to_numpy()
    if not np.isfinite(reference).all():
        raise ValueError('Incomplete grouped reference table.')
    np.testing.assert_allclose(reconstructed, reference, rtol=1e-11, atol=1e-12)
    # Also compare each system against the saved direct contribution table.
    old = pd.read_csv(source/'Fig3C_length_normalised_energy_per_pair.csv')
    old = old.pivot(index='pair_id', columns=['RGG_class', 'AHD_class'], values='length_normalised_attractive_energy').reindex(index=names, columns=class_order).to_numpy()
    rec_c = np.array([np.bincount(class_indices, weights=row, minlength=16) for row in c])
    np.testing.assert_allclose(rec_c, old, rtol=1e-11, atol=1e-12)
    diff = float(np.max(np.abs(reconstructed-reference)))
    print(f'917 systems validated; grouped reconstruction max difference {diff:.3g}. Bootstrapping...', flush=True)
    rng = np.random.default_rng(SEED)
    boot = np.empty((10000, 400))
    # Multinomial multiplicities are exactly equivalent to pair resampling.
    for start in range(0, 10000, 100):
        weights = rng.multinomial(917, np.full(917, 1/917), size=100)
        boot[start:start+100] = weights@f/917
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    ranks = pd.Series(mean).rank(method='min', ascending=False).astype(int).to_numpy()
    cr, ca = np.array(counts_r), np.array(counts_a)
    opportunities = (cr[:, :, None]*ca[:, None, :]).reshape(917, 400)
    labels = pd.DataFrame([(a,b) for a in AA for b in AA], columns=['RGG_aa', 'AHD_aa'])
    aggregate = labels.assign(mean_raw_attractive_energy=raw.mean(axis=0),
        mean_length_normalised_attractive_energy=c.mean(axis=0),
        mean_fraction_total_attractive_energy=mean, mean_percentage_total_attractive_energy=100*mean,
        n_systems_with_both_residue_types=(opportunities>0).sum(axis=0),
        total_residue_pair_opportunities=opportunities.sum(axis=0), rank_by_fractional_contribution=ranks)
    bootstrap = labels.assign(observed_mean_fraction=mean, observed_mean_percentage=100*mean,
                             bootstrap_q025=low, bootstrap_q975=high, rank=ranks)
    ranked = labels.assign(rank=ranks, mean_fraction=mean, mean_percentage=100*mean,
                           bootstrap_q025=low, bootstrap_q975=high,
                           RGG_class=[group(a) for a in labels.RGG_aa], AHD_class=[group(a) for a in labels.AHD_aa])
    ranked = ranked.sort_values('mean_fraction', ascending=False, kind='stable')
    consistency = pd.DataFrame([(a,b) for a in CLASSES for b in CLASSES], columns=['RGG_class','AHD_class']).assign(
        direct_mean_fraction=reference, reconstructed_mean_fraction=reconstructed,
        absolute_difference=np.abs(reference-reconstructed),
        direct_percentage=reference*100, reconstructed_percentage=reconstructed*100)
    out.mkdir(exist_ok=True)
    per = pd.DataFrame(dict(pair_id=np.repeat(names,400), RGG_aa=np.tile(labels.RGG_aa,917),
        AHD_aa=np.tile(labels.AHD_aa,917), RGG_length=np.repeat(np.array(lengths)[:,0],400),
        AHD_length=np.repeat(np.array(lengths)[:,1],400), RGG_aa_count=np.repeat(cr,20,axis=1).ravel(),
        AHD_aa_count=np.tile(ca,(1,20)).ravel(), raw_attractive_energy=raw.ravel(),
        length_normalised_attractive_energy=c.ravel(), fraction_total_attractive_energy=f.ravel()))
    tables = {'per_pair':per, '20x20_source':aggregate, 'bootstrap':bootstrap}
    for suffix, table in tables.items():
        table.to_csv(out/f'Fig3C_residue_level_{suffix}.csv', index=False)
    ranked.to_csv(out/'Fig3C_residue_pair_ranked.csv', index=False)
    consistency.to_csv(out/'Fig3C_residue_to_class_consistency.csv', index=False)
    ah_mean, yu_mean = np.mean(ah_components, axis=0), np.mean(yu_components, axis=0)
    np.testing.assert_allclose(ah_mean+yu_mean, c.mean(axis=0), atol=1e-12)
    top = ranked.index[:20]
    components = labels.loc[top].assign(rank=ranks[top], mean_total_contribution=c.mean(axis=0)[top],
        mean_yukawa_contribution=yu_mean[top], mean_AH_contribution=ah_mean[top],
        dominant_component=np.where(yu_mean[top]>ah_mean[top], 'Yukawa','Ashbaugh-Hatch'))
    components.to_csv(out/'Fig3C_top_residue_pairs_energy_components.csv', index=False)
    dominant = (ranked.RGG_class=='Basic') & (ranked.AHD_class=='Acidic')
    non = ranked[~dominant]; other = ranked[(ranked.RGG_class=='Other') | (ranked.AHD_class=='Other')]
    aromatic = ranked[(ranked.RGG_class=='Aromatic') | (ranked.AHD_class=='Aromatic')].mean_percentage.sum()
    describe = lambda table: ', '.join(f'{r.RGG_aa}-{r.AHD_aa} {r.mean_percentage:.3f}%' for r in table.itertuples())
    lines = ['# Main Figure 3 Panel C residue-level decomposition', '',
        f'917 equally weighted systems; all 400 identities retained. Bootstrap: 10,000 pair resamples; seed {SEED}. CI columns are fractions, not percentages.',
        f'Input: {source}. Saved trajectory-averaged energy caches only; exact-sequence orientation verified.', '',
        'Top 10 overall: '+describe(ranked.head(10)), '', 'Top 20 overall: '+describe(ranked.head(20)), '',
        f'Basic-RG/RGG × Acidic-AHD total: {ranked[dominant].mean_percentage.sum():.3f}%.',
        'Top 20 outside that cell: '+describe(non.head(20)), '',
        'Largest Other-involving pairs: '+describe(other.head(10)), '',
        f'Aromatic-containing pairs contribute {aromatic:.3f}% (reported quantitatively; no biological importance threshold inferred).',
        f'Grouped reconstruction maximum absolute difference: {diff:.3g} in fraction units ({100*diff:.3g} percentage points).', '',
        'Signed AH/Yukawa components use only residue pairs with negative total energy and add exactly to the clipped total. A negative component is a repulsive offset, not separately clipped attraction.', '']
    for g1,g2 in [('Basic','Acidic'),('Acidic','Basic'),('Basic','Other'),('Other','Acidic'),('Aromatic','Acidic'),('Basic','Aromatic'),('Other','Other')]:
        cell = ranked[(ranked.RGG_class==g1)&(ranked.AHD_class==g2)]
        positive = cell[cell.mean_fraction>0]
        if len(positive):
            threshold = .8*positive.mean_fraction.sum()
            count = np.searchsorted(positive.mean_fraction.cumsum().to_numpy(),threshold)+1
            main = positive.head(count)
        else:
            main = positive
        lines.append(f'{g1}-RG/RGG × {g2}-AHD ({cell.mean_percentage.sum():.3f}%): pairs accounting for at least 80% of this cell: '+(describe(main) or 'none (zero cell)'))
    (out/'Fig3C_residue_level_summary.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.family':'sans-serif','font.size':8,'axes.linewidth':.6,'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(7.2,6.3), layout='constrained')
    matrix = mean.reshape(20,20)*100
    im = ax.pcolormesh(np.arange(21)-.5, np.arange(21)-.5, matrix,
                      cmap='YlOrRd', vmin=0, edgecolors='white', linewidth=.3, antialiased=False)
    ax.set(xlim=(-.5,19.5),ylim=(19.5,-.5),aspect='equal',xlabel='AHD amino-acid identity',ylabel='RG/RGG amino-acid identity')
    ax.set_xticks(range(20),AA); ax.set_yticks(range(20),AA)
    ax.set_title('Residue-level attractive-energy contributions',fontsize=10)
    for i in range(20):
        for j in range(20):
            v=matrix[i,j]
            if v>=.1:
                ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=5,
                        color='white' if v>matrix.max()*.65 else 'black')
    fig.colorbar(im, ax=ax, shrink=.8, pad=.025).set_label('Mean fractional contribution to\nattractive interaction energy (%)')
    fig.savefig(out/'Fig3C_residue_level_fractional_heatmap.pdf')
    fig.savefig(out/'Fig3C_residue_level_fractional_heatmap.png', dpi=600)
    plt.close(fig)
    print('TOTAL PAIRS: 917\nTOTAL AA PAIR CLASSES: 400\nTOP 10 AA PAIRS:')
    print(ranked.head(10)[['rank','RGG_aa','AHD_aa','mean_percentage']].to_string(index=False))
    print(f'BASIC-RG/RGG × ACIDIC-AHD TOTAL %: {ranked[dominant].mean_percentage.sum():.6f}')
    print('TOP AA PAIRS WITHIN BASIC-RG/RGG × ACIDIC-AHD:',describe(ranked[dominant].head(10)))
    print('TOP NON-BASIC×ACIDIC AA PAIRS:',describe(non.head(10)))
    print('TOP OTHER-INVOLVING AA PAIRS:',describe(other.head(10)))
    print(f'AROMATIC-CONTAINING CONTRIBUTIONS IMPORTANT: {"YES" if aromatic>=1 else "NO"} (descriptive display threshold ≥1% combined contribution; actual {aromatic:.3f}%)')
    print(f'4×4 RECONSTRUCTION MAX ABS DIFFERENCE: {diff:.12g}\n20×20 HEATMAP CREATED: YES')
    for p in sorted(out.iterdir()): print(p)

if __name__=='__main__':
    main()
