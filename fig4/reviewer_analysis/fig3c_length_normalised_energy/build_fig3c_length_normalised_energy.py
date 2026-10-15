#!/usr/bin/env python3
"""Main Figure 3 panel C from stored, sequence-oriented residue-pair energies."""
import ast
import collections
import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('FIG3C_OUTPUT_DIR', Path(__file__).resolve().parent))
CLASSES = ['Basic', 'Acidic', 'Aromatic', 'Other']
SEED = 20261005
NBOOT = 10000
def cls(a):
    return 0 if a in 'RKH' else 1 if a in 'DE' else 2 if a in 'FYW' else 3

def main():
    helper = Path('/Users/yuktikhanna/calvados/scripts/amino_acids_energy_interactions.py')
    ns = dict(np=np, pd=pd, os=os, pickle=pickle, defaultdict=collections.defaultdict, Counter=collections.Counter)
    for node in ast.parse(helper.read_text()).body:
        if (isinstance(node, ast.FunctionDef) and node.name in ['_to_one_letter', '_ensure_1d_pair_vector']) or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'AA3_TO_1' for t in node.targets)):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(helper), 'exec'), ns)
    audit = pd.read_csv(ROOT/'task5_panelC_enrichment/orientation_audit_final.csv').set_index('pair_id')
    master = pd.read_csv(ROOT/'sim_pairs_added_features_data_270525.csv').set_index('name')
    names = audit.index.tolist()
    assert len(names) == len(set(names)) == 917
    seq = {}; key = None
    for line in (ROOT/'all_tads_rggs_dis.fasta').read_text().splitlines():
        if line.startswith('>'):
            key = line[1:].split()[0]; seq[key] = ''
        elif key:
            seq[key] += line.strip()
    matrices = []; component_ah = []; component_yu = []; rows = []
    for name in names:
        path = Path(os.environ.get('FIG3C_ENERGY_DIR', '/Users/yuktikhanna/calvados/test_energies'))/f'energies_{name}_20nm_sliced.pkl'
        with path.open('rb') as f:
            data = pickle.load(f)
        r1 = ns['_to_one_letter'](data['res1']); r2 = ns['_to_one_letter'](data['res2'])
        rgg_id, ahd_id = master.loc[name, ['rgg', 'tad']]
        forward = ''.join(r1) == seq[rgg_id] and ''.join(r2) == seq[ahd_id]
        reverse = ''.join(r2) == seq[rgg_id] and ''.join(r1) == seq[ahd_id]
        assert forward != reverse, f'Ambiguous exact-sequence orientation: {name}'
        assert audit.loc[name, 'corrected_RGG_side'] == ('res1' if forward else 'res2'), name
        ah = ns['_ensure_1d_pair_vector'](data['ah_energies_pp'],len(r1),len(r2)).reshape(len(r1),len(r2))
        yu = ns['_ensure_1d_pair_vector'](data['yu_energies_pp'],len(r1),len(r2)).reshape(len(r1),len(r2))
        assert np.isfinite(ah).all() and np.isfinite(yu).all(), name
        if reverse:
            r1, r2 = r2, r1; ah = ah.T; yu = yu.T
        c1 = np.array([cls(a) for a in r1]); c2 = np.array([cls(a) for a in r2])
        denominator = len(r1)*len(r2)
        def collapse(energy):
            attractive = np.maximum(-energy, 0)
            result = np.zeros((4,4))
            for i in range(4):
                for j in range(4):
                    result[i,j] = attractive[np.ix_(c1==i,c2==j)].sum()
            np.testing.assert_allclose(result.sum(), attractive.sum(), rtol=1e-12)
            return result
        raw = collapse(ah+yu)
        matrices.append(raw/denominator)
        component_ah.append(collapse(ah)/denominator)
        component_yu.append(collapse(yu)/denominator)
        for i in range(4):
            for j in range(4):
                rows.append(dict(pair_id=name, RGG_class=CLASSES[i], AHD_class=CLASSES[j], RGG_length=len(r1), AHD_length=len(r2), raw_attractive_energy=raw[i,j], length_normalised_attractive_energy=raw[i,j]/denominator))
    values = np.array(matrices).reshape(917,16)
    mean = values.mean(axis=0)
    rng = np.random.default_rng(SEED)
    boot = np.empty((NBOOT,16))
    for start in range(0,NBOOT,100):
        idx = rng.integers(0,917,size=(min(100,NBOOT-start),917))
        boot[start:start+len(idx)] = values[idx].mean(axis=1)
    low, high = np.percentile(boot,[2.5,97.5],axis=0)
    ranks = np.argsort(np.argsort(-mean,kind='stable'),kind='stable')+1
    rank1 = np.bincount(np.argmax(boot,axis=1),minlength=16)/NBOOT
    table = pd.DataFrame([dict(RGG_class=CLASSES[k//4], AHD_class=CLASSES[k%4], mean_length_normalised_contribution=mean[k], bootstrap_q025=low[k], bootstrap_q975=high[k], rank=ranks[k], bootstrap_fraction_rank1=rank1[k]) for k in range(16)])
    table.to_csv(OUT/'Fig3C_length_normalised_energy_source.csv',index=False)
    pd.DataFrame(rows).to_csv(OUT/'Fig3C_length_normalised_energy_per_pair.csv',index=False)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':7,'axes.labelsize':7,'xtick.labelsize':6,'ytick.labelsize':6,'axes.linewidth':0.6,'pdf.fonttype':42})
    fig, ax = plt.subplots(figsize=(3.9,3.2))
    edge = np.arange(5)-.5
    im = ax.pcolormesh(edge,edge,mean.reshape(4,4),cmap='YlOrRd',vmin=0,vmax=mean.max(),shading='flat',edgecolors='white',linewidth=.8,antialiased=False)
    ax.set(xlim=(-.5,3.5),ylim=(3.5,-.5),aspect='equal',xlabel='AHD residue class',ylabel='RG/RGG residue class')
    ax.set_xticks(range(4),CLASSES); ax.set_yticks(range(4),CLASSES)
    for i in range(4):
        for j in range(4):
            v=mean.reshape(4,4)[i,j]
            ax.text(j,i,f'{v:.3g}',ha='center',va='center',fontsize=6.5,color='white' if v>mean.max()*.65 else 'black')
    ax.text(-.25,1.06,'C',transform=ax.transAxes,fontweight='bold',fontsize=10)
    cbar=fig.colorbar(im,ax=ax,fraction=.047,pad=.05)
    cbar.set_label('Length-normalised attractive-energy contribution')
    fig.tight_layout()
    fig.savefig(OUT/'Fig3C_length_normalised_energy.pdf',bbox_inches='tight',pad_inches=.04)
    fig.savefig(OUT/'Fig3C_length_normalised_energy.png',dpi=600,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    top=int(np.argmax(mean)); b=1; a=4
    ah_mean=np.mean(component_ah,axis=0); yu_mean=np.mean(component_yu,axis=0)
    main_yukawa=bool(yu_mean.reshape(16)[top]>ah_mean.reshape(16)[top])
    report=f'''# Main Figure 3 panel C: length-normalised attractive energy

1. Top-ranked combination: {CLASSES[top//4]}-RG/RGG × {CLASSES[top%4]}-AHD.
2. Mean contribution: {mean[top]:.9g}; bootstrap 95% CI [{low[top]:.9g}, {high[top]:.9g}].
3. Basic-RG/RGG × Acidic-AHD: {mean[b]:.9g}; 95% CI [{low[b]:.9g}, {high[b]:.9g}]; rank {ranks[b]}/16.
4. Acidic-RG/RGG × Basic-AHD: {mean[a]:.9g}; 95% CI [{low[a]:.9g}, {high[a]:.9g}]; rank {ranks[a]}/16.
5. Basic/Acidic to Acidic/Basic ratio: {mean[b]/mean[a]:.9g}.
6. Basic-RG/RGG × Acidic-AHD ranks first in {rank1[b]:.2%} of 10,000 pair-bootstrap iterations (seed {SEED}).
7. Dominant signal mainly Yukawa: {'YES' if main_yukawa else 'NO'}. Separately clipped component contributions for the top cell: Yukawa {yu_mean.reshape(16)[top]:.9g}, Ashbaugh-Hatch {ah_mean.reshape(16)[top]:.9g}. These independently clipped component magnitudes are interpretive and need not add to the clipped total.
8. The panel supports replacing the marginal enrichment display with direct attractive-energy contribution magnitude. Each of the 917 simulations contributes equally after division by its full RG/RGG length × AHD length; exact full-sequence orientation was verified against the corrected audit. No class-opportunity denominator or enrichment transformation was used.
'''
    (OUT/'Fig3C_length_normalised_energy_summary.md').write_text(report)
    print('TOTAL PAIRS: 917')
    print(f'TOP CONTRIBUTION CELL: {CLASSES[top//4]}-RG/RGG x {CLASSES[top%4]}-AHD\nTOP CONTRIBUTION VALUE: {mean[top]:.9g}\nTOP CONTRIBUTION 95% CI: [{low[top]:.9g}, {high[top]:.9g}]')
    print(f'BASIC-RG/RGG x ACIDIC-AHD:\nVALUE: {mean[b]:.9g}\n95% CI: [{low[b]:.9g}, {high[b]:.9g}]\nRANK: {ranks[b]}\nBOOTSTRAP FRACTION RANKED FIRST: {rank1[b]:.6f}')
    print(f'ACIDIC-RG/RGG x BASIC-AHD:\nVALUE: {mean[a]:.9g}\n95% CI: [{low[a]:.9g}, {high[a]:.9g}]\nRANK: {ranks[a]}')
    print(f'RATIO BASIC-RGG/ACIDIC-AHD TO ACIDIC-RGG/BASIC-AHD: {mean[b]/mean[a]:.9g}\nDOMINANT SIGNAL MAINLY YUKAWA: {"YES" if main_yukawa else "NO"}\nREPLACEMENT MAIN FIGURE 3 PANEL C CREATED: YES')
    for p in sorted(OUT.glob('Fig3C*')):
        print(p)

if __name__=='__main__':
    main()
