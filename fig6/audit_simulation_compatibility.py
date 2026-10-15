#!/usr/bin/env python3
"""Per-pair provenance/convention audit; never bypass target-scale uncertainty."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from workflow_common import ROOT, OUT, DIAG, sha256, write_json
from calculate_validated_b22 import load_curves, legacy_calculation, integrated_calculation
from compare_old_new_predictions import NEW


def topology_sequences(path):
    names = pd.read_csv('/Users/yuktikhanna/calvados/fig6/residues.csv')
    code = dict(zip(names.three.str.upper(), names.one))
    chains, seen = {}, set()
    for line in path.read_text().splitlines():
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            chain, residue = line[21], line[22:27]
            if (chain, residue) in seen:
                continue
            seen.add((chain, residue))
            chains.setdefault(chain, []).append(code[line[17:20].strip().upper()])
    return [''.join(s) for s in chains.values()]


def main():
    dest = OUT / 'b22'
    inventory = pd.read_csv(OUT / 'complete_pair_inventory.csv')
    historical = pd.read_csv(ROOT / 'Table_S8.csv').set_index('pair_id')
    rows, candidates, sequence_audit = [], [], []
    oldsource = DIAG / 'historical_model/same_construct_predictions.csv'
    same = pd.read_csv(oldsource).set_index('pair_id') if oldsource.exists() else None
    runroots = [Path('/Users/yuktikhanna/calvados/fig6/run_validated_final'),
                Path('/Users/yuktikhanna/calvados/validated/run_validated')]
    for _, row in inventory.iterrows():
        pid = row.pair_id
        topology_matches, topology_sources, primary_matches = [], [], []
        for root in runroots:
            for identifier in [pid, '__'.join(pid.split('__')[::-1])]:
                folder = root / ('test_' + identifier) / '298k_20.0nm'
                for p in sorted(folder.glob('job_*/top.pdb')):
                    try:
                        seqs = topology_sequences(p)
                        match = sorted(seqs) == sorted([row.AHD_sequence, row.RGG_sequence])
                        topology_matches.append(match)
                        if root == runroots[0]:
                            primary_matches.append(match)
                        topology_sources.append(str(p))
                        sequence_audit.append(dict(pair_id=pid, topology_file=str(p), root=str(root),
                            topology_sha256=sha256(p), sequence_lengths=';'.join(str(len(s)) for s in seqs),
                            sequences=';'.join(seqs), new_AHD_length=len(row.AHD_sequence), new_RGG_length=len(row.RGG_sequence),
                            exact_pair_sequence_match=match, primary_current_run_root=root == runroots[0]))
                    except (KeyError, ValueError):
                        topology_matches.append(False)
        # Older run_validated/ contains a different FOXO4 construct. Its mismatch
        # is recorded, not allowed to veto the consistent current final-run root.
        exact_topology = bool(primary_matches and all(primary_matches))
        reproduced_current = reproduced_archive = False
        cutoff_match = []
        if pid in historical.index:
            target = historical.loc[pid]
            paths = [(ROOT / row.rdf_file, 'current_supplied')]
            for folder in [Path('/Users/yuktikhanna/idr-interaction-predict/sample_rdfs'), Path('/Users/yuktikhanna/calvados/fig6/validated_final_rdfs')]:
                f = folder / Path(row.rdf_file).name
                if f.exists():
                    paths.append((f, 'local_archived_candidate'))
            for source, label in paths:
                curves, box, _ = load_curves(source, 20)
                for cutoff in [6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10]:
                    vals = [legacy_calculation(r, g, cutoff) for _, r, g, _ in curves]
                    raw, corrected = np.mean(np.asarray(vals)[:, :2], axis=0)
                    reproduces = bool(np.isclose(raw, target.Sim_B22_raw, rtol=1e-6, atol=1e-4)
                                      and np.isclose(corrected, target.Sim_B22_corr_Hummer, rtol=1e-6, atol=1e-4))
                    candidates.append(dict(pair_id=pid, source=str(source), source_kind=label, source_sha256=sha256(source),
                        method='legacy_sum', cutoff_nm=cutoff, n_replicas=len(curves), raw_B22=raw, corrected_B22=corrected,
                        raw_difference=raw-target.Sim_B22_raw, corrected_difference=corrected-target.Sim_B22_corr_Hummer,
                        reproduces_historical_raw_and_corrected=reproduces))
                    if reproduces and cutoff == 7.5:
                        reproduced_current |= label == 'current_supplied'
                        reproduced_archive |= label == 'local_archived_candidate'
                    if reproduces:
                        cutoff_match.append(cutoff)
                # Explicitly test replica selection without selecting a subset
                # after the fact as the new principal target.
                if label == 'current_supplied' and len(curves) <= 6:
                    vals = np.array([legacy_calculation(r, g, 7.5)[:2] for _, r, g, _ in curves])
                    for count in range(1, len(curves)):
                        for combo in itertools.combinations(range(len(curves)), count):
                            raw, corrected = vals[list(combo)].mean(axis=0)
                            if np.isclose(raw, target.Sim_B22_raw, rtol=1e-6, atol=1e-4) and np.isclose(corrected, target.Sim_B22_corr_Hummer, rtol=1e-6, atol=1e-4):
                                candidates.append(dict(pair_id=pid, source=str(source), source_kind='replica_subset_diagnostic_only',
                                    method='legacy_sum', cutoff_nm=7.5, n_replicas=count,
                                    replica_ids=';'.join(curves[i][0] for i in combo), raw_B22=raw, corrected_B22=corrected,
                                    reproduces_historical_raw_and_corrected=True))
        legacy_scale_supported = reproduced_current or reproduced_archive
        eligible = bool(exact_topology and legacy_scale_supported and same is not None and pid in same.index)
        reasons = []
        if not exact_topology:
            reasons.append('exact supplied construct sequences not verified in available run topology')
        if not legacy_scale_supported:
            reasons.append('no 7.5 nm raw/corr1 numerical reproduction in supplied or archived candidate RDFs')
        if same is None or pid not in same.index:
            reasons.append('same-construct old final predictor unavailable')
        rows.append(dict(pair_id=pid, adjacent_topology_exact_match=exact_topology,
                         topology_files=';'.join(topology_sources),
                         topology_evidence='adjacent run topology; no embedded RDF-generator manifest',
                         current_legacy_reproduces=reproduced_current, archived_legacy_reproduces=reproduced_archive,
                         historical_matching_cutoffs=';'.join(map(str, sorted(set(cutoff_match)))),
                         old_same_construct_available=same is not None and pid in same.index,
                         paired_simulation_accuracy_eligible=eligible, exclusion_reason='; '.join(reasons)))
    audit = pd.DataFrame(rows)
    pd.DataFrame(sequence_audit).to_csv(dest / 'topology_construct_sequence_audit.csv', index=False)
    audit.to_csv(dest / 'per_pair_simulation_eligibility.csv', index=False)
    comparisons = pd.DataFrame(candidates)
    comparisons.to_csv(dest / 'convention_source_replica_investigation.csv', index=False)
    accuracy = []
    if audit.paired_simulation_accuracy_eligible.any():
        sim = pd.read_csv(dest / 'per_pair_b22.csv')
        sim = sim[sim.method.eq('legacy_sum') & sim.integration_boundary_nm.eq(7.5)].set_index('pair_id')
        eligible_ids = audit[audit.paired_simulation_accuracy_eligible].pair_id
        data = same.loc[eligible_ids].join(sim[['log10_neg_mean_B22']], validate='one_to_one')
        data = data[np.isfinite(data.log10_neg_mean_B22) & np.isfinite(data.Old_same_construct_final) & np.isfinite(data[NEW])]
        data.to_csv(dest / 'paired_simulation_accuracy_source.csv')
        for model, col in [('old_same_construct', 'Old_same_construct_final'), ('new', NEW)]:
            y, p = data.log10_neg_mean_B22.to_numpy(), data[col].to_numpy()
            e = p-y
            denom = np.sum((y-y.mean())**2)
            accuracy.append(dict(predictor=model, n=len(y), R_squared=1-np.sum(e**2)/denom if denom > 0 else np.nan,
                RMSE=np.sqrt(np.mean(e**2)), MAE=abs(e).mean(), mean_signed_error=e.mean(),
                Pearson_r=stats.pearsonr(y,p).statistic if len(y)>=3 else np.nan,
                Spearman_rho=stats.spearmanr(y,p).statistic if len(y)>=3 else np.nan,
                convention='legacy_sum 7.5 nm corr1, log after replica mean', eligibility='see per-pair audit; identical rows'))
    pd.DataFrame(accuracy, columns=['predictor','n','R_squared','RMSE','MAE','mean_signed_error','Pearson_r','Spearman_rho','convention','eligibility']).to_csv(dest / 'paired_simulation_accuracy_metrics.csv', index=False)
    from expanded_validation_analysis import kw_test, pairwise_rank_test, binary_effect, holm, interval, SEED, B
    summaries=pd.read_csv(dest/'per_pair_b22.csv')
    summaries=summaries[summaries.method.eq('legacy_sum') & summaries.integration_boundary_nm.eq(7.5)]
    experimental=pd.read_csv(OUT/'experimental_validation_updated.csv')
    plotted=experimental.merge(summaries[['pair_id','log10_neg_mean_B22']],on='pair_id',validate='one_to_one')
    plotted=plotted[np.isfinite(plotted.log10_neg_mean_B22) & plotted.Outcome.isin(['Kd_fit','LB','NB'])]
    rng=np.random.default_rng(SEED)
    h,p,eps=kw_test(plotted.log10_neg_mean_B22.to_numpy(),plotted.Outcome.to_numpy(),rng)
    statistical_rows=[dict(predictor='simulation',comparison='Kd-fit / LB / NB omnibus',n=len(plotted),statistic=h,
        effect=eps,effect_name='bias-adjusted epsilon squared',p_raw=p,p_adjusted=p,
        test='permutation Kruskal-Wallis',permutations=B,adjustment='none; omnibus')]
    pair_rows=[]
    for a,b in [('Kd_fit','LB'),('Kd_fit','NB'),('LB','NB')]:
        aa=plotted[plotted.Outcome.eq(a)].log10_neg_mean_B22.to_numpy()
        bb=plotted[plotted.Outcome.eq(b)].log10_neg_mean_B22.to_numpy()
        p,count=pairwise_rank_test(aa,bb)
        effect=binary_effect(aa,bb)[1]
        pair_rows.append(dict(predictor='simulation',comparison=f'{a} vs {b}',n=len(aa)+len(bb),statistic=effect,
            effect=effect,effect_name='rank-biserial',p_raw=p,test='exact two-sided permutation Mann-Whitney',
            permutations=count,adjustment='Holm across three comparisons'))
    for row,p in zip(pair_rows,holm([r['p_raw'] for r in pair_rows])):
        row['p_adjusted']=p
        statistical_rows.append(row)
    pd.DataFrame(statistical_rows).to_csv(dest/'simulation_outcome_statistics.csv',index=False)
    aa=plotted[plotted.Outcome.eq('Kd_fit')].log10_neg_mean_B22.to_numpy()
    bb=plotted[plotted.Outcome.eq('NB')].log10_neg_mean_B22.to_numpy()
    boot=[binary_effect(rng.choice(aa,len(aa)),rng.choice(bb,len(bb)))[0] for _ in range(10000)]
    pd.DataFrame([dict(predictor='simulation',estimate=binary_effect(aa,bb)[0],n=len(aa)+len(bb),n_positive=len(aa),
        n_negative=len(bb),exclusions='LB excluded from binary diagnostic',**interval(boot))]).to_csv(dest/'simulation_outcome_auroc.csv',index=False)
    write_json(dest / 'subset_accuracy_status.json', dict(total_pairs=len(audit), adjacent_topology_matches=int(audit.adjacent_topology_exact_match.sum()),
        legacy_7_5_reproduction_pairs=int((audit.current_legacy_reproduces|audit.archived_legacy_reproduces).sum()),
        eligible_accuracy_pairs=int(audit.paired_simulation_accuracy_eligible.sum()),
        conclusion='Accuracy restricted per pair; prediction agreement never blocked by simulation discrepancies. Adjacent topology is contextual sequence evidence; an RDF generation manifest would strengthen linkage.'))
    print(audit.paired_simulation_accuracy_eligible.value_counts().to_dict())


if __name__ == '__main__':
    main()
