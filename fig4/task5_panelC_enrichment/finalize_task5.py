"""Targeted cleanup: stored-energy verification only; preserve original outputs."""
import ast, collections, hashlib, os, pickle
from pathlib import Path
import numpy as np
import pandas as pd

out=Path(__file__).resolve().parent; root=out.parent
existing={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='TASK5_REPORT.md'}
helper=Path('/Users/yuktikhanna/calvados/scripts/amino_acids_energy_interactions.py')
ns=dict(np=np,pd=pd,os=os,pickle=pickle,defaultdict=collections.defaultdict,Counter=collections.Counter)
for node in ast.parse(helper.read_text()).body:
    if (isinstance(node,ast.FunctionDef) and node.name in ['_to_one_letter','_score','_guess_is_rgg','_ensure_1d_pair_vector','load_and_orient_pairmap','per_sim_AA_pair_matrices']) or (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['AA3_TO_1','AAs20'] for t in node.targets)):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),ns)
names=pd.read_csv(root/'out_fig4/cache/Fig4d_scatter_table.csv').name.tolist()
master=pd.read_csv(root/'sim_pairs_added_features_data_270525.csv').set_index('name')
old=pd.read_csv(out/'orientation_audit.csv').set_index('pair_id')
seq={}; key=None
for line in (root/'all_tads_rggs_dis.fasta').read_text().splitlines():
    if line.startswith('>'): key=line[1:].split()[0];seq[key]=''
    elif key: seq[key]+=line.strip()
rows=[]; corrected=[]; historical=[]; aa=list('ACDEFGHIKLMNPQRSTVWY'); ix={a:i for i,a in enumerate(aa)}
for name in names:
    p=Path('/Users/yuktikhanna/calvados/test_energies')/f'energies_{name}_20nm_sliced.pkl'
    with p.open('rb') as f: d=pickle.load(f)
    r1=ns['_to_one_letter'](d['res1']);r2=ns['_to_one_letter'](d['res2']);rgg,tad=master.loc[name,['rgg','tad']]
    forward=''.join(r1)==seq[rgg] and ''.join(r2)==seq[tad]
    reverse=''.join(r2)==seq[rgg] and ''.join(r1)==seq[tad]
    assert forward != reverse, name
    flip=bool(ns['_guess_is_rgg'](r2) and not ns['_guess_is_rgg'](r1))
    hside='res2' if flip else 'res1';cside='res1' if forward else 'res2';correct=hside==cside
    assert correct==old.loc[name,'historical_axes_correct']
    rows.append(dict(pair_id=name,historical_RGG_side=hside,corrected_RGG_side=cside,historical_TAD_side='res1' if flip else 'res2',corrected_TAD_side='res2' if forward else 'res1',historical_correct=correct,source_of_correct_assignment='Exact res1/res2 sequence matches to rgg/tad fields in sim_pairs_added_features_data_270525.csv and all_tads_rggs_dis.fasta; population from Fig4d_scatter_table.csv',notes=f'RGG={rgg}; TAD={tad}; unique full-sequence match; no content heuristic in corrected assignment'))
    v=ns['_ensure_1d_pair_vector'](d['ah_energies_pp'],len(r1),len(r2))+ns['_ensure_1d_pair_vector'](d['yu_energies_pp'],len(r1),len(r2))
    v=np.clip(-v.reshape(len(r1),len(r2)),0,None);sm=np.zeros((20,20));cnt=np.zeros((20,20))
    for i,a in enumerate(r1):
        for j,b in enumerate(r2):sm[ix[a],ix[b]]+=v[i,j];cnt[ix[a],ix[b]]+=1
    m=np.divide(sm,cnt,out=np.zeros_like(sm),where=cnt>0)
    historical.append(m.T if flip else m);corrected.append(m if forward else m.T)
assert len(rows)==917
final=pd.DataFrame(rows); final.to_csv(out/'orientation_audit_final.csv',index=False)
hm=np.mean(historical,axis=0); target=out/'historical_20x20_mean_matrix.csv'
if target.exists(): np.testing.assert_allclose(pd.read_csv(target,index_col=0),hm,rtol=1e-12,atol=1e-15)
else: pd.DataFrame(hm,index=aa,columns=aa).to_csv(target,float_format='%.17g')
def collapse(m):
    g=np.zeros((4,4));cls=lambda a:0 if a in 'RKH' else 1 if a in 'DE' else 2 if a in 'FYW' else 3
    for i,a in enumerate(aa):
        for j,b in enumerate(aa):g[cls(a),cls(b)]+=m[i,j]
    return g
co=collapse(np.mean(corrected,axis=0));ce=co.sum(1)[:,None]*co.sum(0)[None,:]/co.sum()
with np.errstate(divide='ignore'): logs=np.log2(co/ce)
for x,n in [(co,'sequence_oriented_observed_matrix'),(ce,'sequence_oriented_expected_matrix'),(logs,'sequence_oriented_log2_enrichment_matrix'),(collapse(hm),'current_observed_matrix')]:np.testing.assert_allclose(x,pd.read_csv(out/(n+'.csv'),index_col=0).to_numpy(),rtol=1e-12,atol=1e-15)
b=pd.read_csv(out/'sequence_oriented_bootstrap.csv').query("RGG_class=='Basic' and TAD_class=='Acidic'").iloc[0]
assert np.isclose(b.enrichment,logs[0,1])
cross=pd.crosstab(final.corrected_RGG_side,final.historical_RGG_side)
errors=int((~final.historical_correct).sum())
reconciliation=f'''# Orientation count reconciliation

“Reversed” in orientation_audit.csv describes the **stored PKL order**: res1 is the TAD sequence and res2 is the RGG sequence, established by exact full-sequence identity against authoritative pair metadata and FASTA. This occurs in 480/917 files. The other 437 store res1=RGG and res2=TAD. Stored order alone is not an analysis error.

“Incorrect historical orientation” describes the **axes after the historical heuristic**. The heuristic chooses res2 as RGG only when its R/G fraction is at least 0.6 and res1's is below 0.6; otherwise it chooses res1. The historical assignment disagrees with the exact sequence assignment in {errors}/917 systems. Thus 502 historical assignments are correct.

Cross-tabulation (rows: sequence-verified RGG side; columns: historical RGG side):

{cross.to_string()}

The off-diagonal counts are errors. Quote **415/917 historical Figure 4C axis assignments were incorrect** when discussing the panel problem. Quote 480 only when describing PKL storage order. Corrected assignments use the rgg/tad IDs in the exact 917 pair definitions and full-sequence matching, never the R/G-content heuristic. All 917 matches are unambiguous.
'''
(out/'orientation_count_reconciliation.md').write_text(reconciliation)
(out/'historical_pipeline_provenance.md').write_text('''# Historical Panel C pipeline provenance

Simulation PDB/topology and residue-level DCD trajectories → calc_per_pair_average_energy in per_pair_energies.py → AH and Yukawa energies for each interchain residue pair, averaged over available trajectory frames → legacy save_energies_pp in calvados/scripts/plot_energies.py → energies_<pair>_20nm_sliced.pkl containing ah_energies_pp, yu_energies_pp, pairs, res1, res2 and aggregate energy/error fields → calvados/scripts/amino_acids_energy_interactions.py → per_sim_AA_pair_matrices(..., mode="attractive", assume_name_order=False) → rtads-rgg-interactions/fig4/fig4.py → arithmetic mean over the exact 917 systems of the 20×20 AA matrices → collapse to the 4×4 residue-class matrix → expected matrix from attractive-energy row and column marginals → log2((observed+1e-9)/(expected+1e-9)).

The energy function selects segid A as res1 and segid B as res2. It calculates distances with periodic boundaries, evaluates AH and Yukawa energies per frame, and averages those energies. The historical helper clips negative mean total energy to positive attractive magnitude, then divides each AA-identity-pair sum by its number of residue opportunities within each system (zero when absent). The class matrix sums these AA-pair means; it is not direct contact counting or an unnormalized sum of all residue-pair energies.

Original orientation uses the R/G-content heuristic (fraction >=0.6, fallback to res1). The final correction instead matches complete residue sequences against authoritative rgg/tad IDs and the simulation FASTA for every pair.

The IDR repository's newer compute_pair_energies.py/fig4.py pipeline uses a different schema (avg_total_per_pair and n_jobs). It is not the schema or computation that reproduces the historical cached panel. All 917 legacy PKLs lack n_jobs and job IDs. Their exact five-job versus additional-job provenance cannot be established from those files; no numerical job count is inferred. No PDB/DCD files or trajectory energies were regenerated for this cleanup.
''')
section=f'''## Final recommendation for Figure 4C

1. Remove the −20.87 annotation from the revised panel. Keep it only in the historical reproduction. It is the pseudocount-dependent finite representation of zero observed attractive contribution with positive expected contribution; unregularized enrichment is −Inf. Display “zero observed” explicitly.
2. Call the panel **residue-class attractive interaction-energy enrichment**, not contact enrichment. Expected values derive from the energy-statistic marginals, not sequence-only contact frequencies.
3. Replace the heuristic orientation. Exact sequence matches establish {errors}/917 incorrect historical assignments; 480 refers only to reversed PKL storage order.
4. Use **sequence-oriented, support-aware Candidate A** (figures/candidate_A.pdf) for the revised Figure 4C, with the statistic, weighting, zero-observed treatment and ±2 color clipping explained in its caption. Candidate B remains a diagnostic of the normalized underlying quantity. Disclose unknown historical job counts.
5. Preferential Basic-RGG × Acidic-TAD contributions remain supported: observed={co[0,1]:.17g}, expected={ce[0,1]:.17g}, log2 enrichment={logs[0,1]:.17g}; stored 10,000-resample bootstrap CI [{b.CI_low:.17g}, {b.CI_high:.17g}], positive fraction={b.fraction_positive:.1f}. This cell ranks 1/16 by observed interaction statistic and 2/16 by enrichment. Do not describe it as the highest enrichment. This supports an energy-based preference, not a measured contact-frequency claim.

Cleanup verification independently reread stored energy arrays and exact sequences, reproduced the corrected 4×4 and historical 20×20 matrices, and checked the saved bootstrap estimate against the verified central value. The existing bootstrap was not rerun. All historical matrices, candidate figures and prior analysis outputs were preserved.
'''
report=out/'TASK5_REPORT.md';text=report.read_text();heading='## Final recommendation for Figure 4C'
if heading in text:text=text[:text.index(heading)].rstrip()+'\n\n'
report.write_text(text.rstrip()+'\n\n'+section)
for p,h in existing.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==h,f'Existing output changed: {p}'
print(reconciliation);print(section);print('Output directory:',out)
