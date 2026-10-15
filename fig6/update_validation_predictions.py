#!/usr/bin/env python3
"""Audit frozen supplied predictions and create new tables without altering inputs.

Run: python update_validation_predictions.py [--panel-fasta PATH]
--panel-fasta means the user has established that this FASTA generated the OLD
Table_S8 panel predictions. A nearby FASTA or matching length alone is not proof.
"""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from workflow_common import ROOT, OUT, inside, read_fasta, seqhash, sha256, unique, pkd, rdf_inventory, write_json


def historical_roles(d, new):
    """Resolve BOTH tokens and feature roles; old tables reversed AHD/RGG labels."""
    ahd, rgg = set(new.AHD_name), set(new.RGG_name)
    rows = []
    for _, z in d.iterrows():
        parts = str(z['name']).split('__')
        if len(parts) != 2:
            raise ValueError(f"Invalid historical pair identifier: {z['name']}")
        aa, rr = [v for v in parts if v in ahd], [v for v in parts if v in rgg]
        valid = len(aa) == len(rr) == 1 and aa[0] != rr[0]
        a, r = (aa[0], rr[0]) if valid else ('', '')
        lengths = {str(z['rgg']): z['RGG_length'], str(z['tad']): z['TAD_length']}
        rows.append(dict(historical_name=z['name'],
                         pair_id=f'{r}__{a}' if valid else '',
                         AHD_name=a, RGG_name=r,
                         old_AHD_length=lengths.get(a, np.nan),
                         old_RGG_length=lengths.get(r, np.nan),
                         mapping_status='both_role_identities_verified' if valid else 'unmatched_role_identity',
                         role_columns_reversed=valid and str(z['rgg']) == a))
    return pd.concat([d.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', default='updated_validation')
    ap.add_argument('--panel-fasta', type=Path,
                    help='Verified FASTA underlying old Table_S8 predictions, NOT an assumed candidate')
    args = ap.parse_args()
    out = inside(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    qcdir = out / 'qc'
    qcdir.mkdir(exist_ok=True)
    pred = pd.read_csv(ROOT / 'preds_val_updated/predictions_minimal.csv')
    resolved = pd.read_csv(ROOT / 'preds_val_updated/resolved_pairs.csv')
    prepared = pd.read_csv(ROOT / 'preds_val_updated/prepared_pairs.csv')
    full = pd.read_csv(ROOT / 'preds_val_updated/predictions.csv')
    registry = pd.read_csv(ROOT / 'preds_val_updated/sequence_registry_audit.csv')
    rejected = pd.read_csv(ROOT / 'preds_val_updated/rejected_pairs.csv')
    old = pd.read_csv(ROOT / 'Table_S8.csv')
    fasta = read_fasta(ROOT / 'validated_final_updated.fasta')
    fasta.to_csv(qcdir / 'fasta_registry.csv', index=False)
    issues = []
    for label, d in [('minimal', pred), ('resolved', resolved), ('prepared', prepared), ('full', full), ('experimental', old)]:
        for _, z in d[d.pair_id.duplicated(keep=False) | d.pair_id.isna()].iterrows():
            issues.append(dict(source=label, pair_id=z.pair_id, issue='duplicate_or_missing_pair_id', severity='fatal'))
    pd.DataFrame(issues, columns=['source', 'pair_id', 'issue', 'severity']).to_csv(qcdir / 'duplicate_pairs.csv', index=False)
    for label, d in [('minimal', pred), ('resolved', resolved), ('prepared', prepared), ('full', full), ('experimental', old)]:
        unique(d, 'pair_id', label)
    for d in (resolved, prepared, full):
        if set(d.pair_id) != set(pred.pair_id):
            raise ValueError('Supplied prediction tables disagree on pair identities')
    if not np.isfinite(pred[['base_prediction', 'final_hybrid_prediction']].to_numpy(float)).all():
        raise ValueError('Nonfinite supplied predictions')
    audit = []
    for _, z in pred.iterrows():
        if z.pair_id != f'{z.RGG_name}__{z.AHD_name}':
            raise ValueError(f'Oriented pair ID and roles disagree: {z.pair_id}')
        for role in ['AHD', 'RGG']:
            seq, name = z[f'{role}_sequence'], z[f'{role}_name']
            f = fasta[fasta.name.eq(name)]
            r = resolved.set_index('pair_id').loc[z.pair_id]
            registry_rows = registry[(registry.role == role) & (registry.name == name)]
            ok = (not f.empty and f.sequence.eq(seq).all() and not f.invalid_residue.any()
                  and len(seq) == r[f'{role}_length'] and seqhash(seq) == r[f'{role}_sequence_sha256']
                  and len(registry_rows) == 1 and registry_rows.iloc[0].sequence_sha256 == seqhash(seq))
            for source, d in [('prepared', prepared), ('full', full), ('resolved', resolved)]:
                rr = d.set_index('pair_id').loc[z.pair_id]
                ok = ok and rr[f'{role}_name'] == name and rr[f'{role}_sequence'] == seq
            audit.append(dict(pair_id=z.pair_id, role=role, name=name, length=len(seq), sha256=seqhash(seq),
                              fasta_record_count=len(f), exact_sequence_verified=bool(ok)))
    ad = pd.DataFrame(audit)
    ad.to_csv(qcdir / 'pair_sequence_audit.csv', index=False)
    if not ad.exact_sequence_verified.all():
        raise ValueError('Sequence audit failed; see pair_sequence_audit.csv')
    for col in ['base_prediction', 'final_hybrid_prediction', 'applied_correction']:
        x = pred.set_index('pair_id')[col].sort_index()
        y = full.set_index('pair_id')[col].sort_index()
        np.testing.assert_allclose(x, y, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pred.base_prediction + pred.applied_correction,
                               pred.final_hybrid_prediction, rtol=0, atol=1e-12)
    manifest = json.loads((ROOT / 'preds_val_updated/run_manifest.json').read_text())
    integrity = []
    for filename, expected in manifest.get('prepared_output_hashes', {}).items():
        f = ROOT / 'preds_val_updated' / filename
        integrity.append(dict(file=filename, expected_sha256=expected, actual_sha256=sha256(f), passed=sha256(f) == expected))
    pd.DataFrame(integrity).to_csv(qcdir / 'manifest_hash_checks.csv', index=False)
    if integrity and not all(z['passed'] for z in integrity):
        raise ValueError('Prepared input hashes differ from supplied run manifest')
    names = [s.strip() for s in (ROOT / 'validated_final_names_updated.txt').read_text().splitlines() if s.strip()]
    names_qc = pd.DataFrame({'pair_id': names})
    names_qc['duplicate'] = names_qc.pair_id.duplicated(keep=False)
    names_qc['has_prediction'] = names_qc.pair_id.isin(pred.pair_id)
    names_qc.to_csv(qcdir / 'requested_pairs.csv', index=False)
    if names_qc.duplicate.any() or set(names) != set(pred.pair_id):
        raise ValueError('Requested pair list does not match predictions')
    if not old.pair_id.eq(old.RGG_partner + '__' + old.R_TAD).all():
        raise ValueError('Historical experimental pair IDs disagree with construct roles')
    pred = pred.rename(columns={'base_prediction': 'New_base_log10_negB22',
                                'final_hybrid_prediction': 'New_final_log10_negB22',
                                'applied_correction': 'New_applied_correction'})
    pred['AHD_length'] = pred.AHD_sequence.str.len()
    pred['RGG_length'] = pred.RGG_sequence.str.len()
    pred['AHD_parent'] = pred.AHD_name.str.replace(r'_shuffle_\d+$', '', regex=True)
    pred['RGG_parent'] = pred.RGG_name.str.replace(r'_shuffle_\d+$', '', regex=True)
    pred['is_shuffle'] = pred.AHD_name.str.contains('_shuffle_') | pred.RGG_name.str.contains('_shuffle_')
    pred['duplicate_sequence_pair'] = pred.duplicated(['AHD_sequence', 'RGG_sequence'], keep=False)
    updated = old.merge(pred, on='pair_id', how='left', validate='one_to_one', indicator='prediction_merge')
    updated['pKd'] = pkd(updated.Kd_uM)
    updated['pKd_error'] = updated.Kd_uM_err / (updated.Kd_uM * np.log(10))
    updated.loc[~(updated.Kd_uM.gt(0) & updated.Kd_uM_err.ge(0)), 'pKd_error'] = np.nan
    historical = {}
    for filename in ['validated_final_features_data_pred_b22s.csv', 'tables/validated_features_data.csv',
                     'tables/validated_pred_with_hybrid_021125.csv', 'tables/validated_pred_with_hybrid_021125_sel_cols.csv']:
        h = historical_roles(pd.read_csv(ROOT / filename), pred)
        new_lengths = pred[['pair_id', 'AHD_length', 'RGG_length']].rename(
            columns={'AHD_length': 'new_AHD_length', 'RGG_length': 'new_RGG_length'})
        h = h.merge(new_lengths, on='pair_id', how='left', validate='many_to_one')
        h['historical_length_status'] = np.where(
            h.old_AHD_length.eq(h.new_AHD_length) & h.old_RGG_length.eq(h.new_RGG_length),
            'length_match_sequence_unverified', 'changed_length_or_unmatched_pair')
        h.to_csv(qcdir / (Path(filename).stem + '_mapped.csv'), index=False)
        good = h[h.mapping_status.eq('both_role_identities_verified')].copy()
        unique(good, 'pair_id', filename)
        historical[filename] = good
    panel_features = historical['validated_final_features_data_pred_b22s.csv'].set_index('pair_id')
    verified_old_fasta = read_fasta(args.panel_fasta) if args.panel_fasta else None
    if verified_old_fasta is not None:
        verified_old_fasta.to_csv(qcdir / 'verified_panel_fasta.csv', index=False)
    statuses = []
    for _, z in updated.iterrows():
        if z.prediction_merge != 'both':
            statuses.append('unmatched_new_prediction')
            continue
        h = panel_features.loc[z.pair_id] if z.pair_id in panel_features.index else None
        length_match = h is not None and h.old_AHD_length == z.AHD_length and h.old_RGG_length == z.RGG_length
        status = 'length_match_sequence_unverified' if length_match else 'changed_length_or_missing_historical_input'
        if verified_old_fasta is not None:
            matches, complete = [], True
            for role in ['AHD', 'RGG']:
                f = verified_old_fasta[verified_old_fasta.name.eq(z[f'{role}_name'])]
                if f.empty or f.conflicting_name.any() or f.invalid_residue.any():
                    complete = False
                else:
                    matches.append(f.sequence.eq(z[f'{role}_sequence']).all())
            if complete:
                status = 'exact_sequence_match' if all(matches) else 'changed_construct'
        statuses.append(status)
    updated['sequence_match_status'] = statuses
    # Nearby historical FASTAs are audited as candidates, never asserted as the
    # inputs to the panel prediction run without the explicit --panel-fasta.
    candidates = [Path('/Users/yuktikhanna/calvados/fig6/validated_final.fasta'),
                  Path('/Users/yuktikhanna/calvados/validated/validated_1.fasta')]
    candidate_rows = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        cf = read_fasta(candidate)
        cf['source'] = str(candidate)
        cf.to_csv(qcdir / (candidate.parent.name + '_' + candidate.stem + '_candidate_registry.csv'), index=False)
        for _, z in pred.iterrows():
            for role in ['AHD', 'RGG']:
                records = cf[cf.name.eq(z[f'{role}_name'])]
                candidate_rows.append(dict(pair_id=z.pair_id, role=role, source=str(candidate),
                    name=z[f'{role}_name'], records=len(records),
                    sequence_equals_new=bool(not records.empty and records.sequence.eq(z[f'{role}_sequence']).all()),
                    conflicting_name=bool(records.conflicting_name.any()),
                    attribution='candidate only; old-panel prediction linkage not established'))
    pd.DataFrame(candidate_rows).to_csv(qcdir / 'historical_candidate_sequence_audit.csv', index=False)
    # All historical measurements/annotations remain byte-equivalent in meaning after CSV parsing.
    pd.testing.assert_frame_equal(updated[old.columns].reset_index(drop=True), old.reset_index(drop=True))
    updated.to_csv(out / 'experimental_validation_updated.csv', index=False)
    rdf = rdf_inventory(ROOT / 'validated_final_rdfs', ROOT / 'rdf_pair_aliases.csv')
    unique(rdf, 'pair_id', 'RDF inventory')
    inventory = pred.merge(old[['pair_id', 'Outcome', 'Notes', 'Kd_uM', 'Kd_uM_err']],
                           on='pair_id', how='left', validate='one_to_one')
    inventory['experiment_present'] = inventory.pair_id.isin(old.pair_id)
    inventory['Outcome'] = inventory.Outcome.where(inventory.experiment_present, 'not tested')
    inventory['missing_experimental_annotation'] = inventory.experiment_present & inventory.Outcome.isna()
    inventory = inventory.merge(rdf, on='pair_id', how='left', validate='one_to_one')
    inventory['simulation_construct_status'] = 'RDF filename identity only; sequence not embedded/verified'
    inventory.to_csv(out / 'complete_pair_inventory.csv', index=False)
    mapping = pd.DataFrame({'pair_id': sorted(set(pred.pair_id) | set(old.pair_id) | set(rdf.pair_id))})
    mapping['has_new_prediction'] = mapping.pair_id.isin(pred.pair_id)
    mapping['has_experiment'] = mapping.pair_id.isin(old.pair_id)
    mapping['has_rdf'] = mapping.pair_id.isin(rdf.pair_id)
    mapping = mapping.merge(updated[['pair_id', 'sequence_match_status']], on='pair_id', how='left', validate='one_to_one')
    mapping['validation_inclusion'] = np.where(mapping.has_experiment & mapping.has_new_prediction, 'include', 'not in experimental validation')
    mapping.to_csv(qcdir / 'pair_mapping.csv', index=False)
    rdf[~rdf.pair_id.isin(pred.pair_id)].to_csv(qcdir / 'orphan_rdfs.csv', index=False)
    agreement = []
    for filename, h in historical.items():
        if 'B22_pred_hybrid' not in h:
            continue
        m = old[['pair_id', 'Pred_log10_negB22_hybrid']].merge(h[['pair_id', 'B22_pred_hybrid', 'old_AHD_length', 'old_RGG_length']],
                                                          on='pair_id', validate='one_to_one')
        m['difference_from_old_panel'] = m.B22_pred_hybrid - m.Pred_log10_negB22_hybrid
        m['source'] = filename
        agreement.append(m)
    if agreement:
        pd.concat(agreement).to_csv(qcdir / 'historical_prediction_agreement.csv', index=False)
    hashes = {str(f.relative_to(ROOT)): sha256(f) for f in sorted((ROOT / 'preds_val_updated').glob('*')) if f.is_file()}
    for f in [ROOT / 'Table_S8.csv', ROOT / 'validated_final_updated.fasta', ROOT / 'validated_final_names_updated.txt']:
        hashes[str(f.relative_to(ROOT))] = sha256(f)
    current_git = []
    inspection = json.loads((ROOT / 'inspection_provenance.json').read_text())
    for repo in inspection['repositories']:
        p = Path(repo['path'])
        if p.exists():
            def git(*cmd):
                return subprocess.check_output(['git', '-C', str(p), *cmd], text=True).strip()
            current_git.append(dict(path=str(p), commit=git('rev-parse', 'HEAD'), status=git('status', '--short')))
    report = dict(predictions=len(pred), rejected=len(rejected), experimental_rows=len(old),
                  positive_numerical_Kd=int(old.Kd_uM.gt(0).sum()), rdf_files=len(rdf),
                  matched_experiments=int(updated.prediction_merge.eq('both').sum()),
                  untested_pairs=int((~inventory.experiment_present).sum()),
                  historical_table_rows={k: len(v) for k, v in historical.items()},
                  sequence_status_counts=updated.sequence_match_status.value_counts().to_dict(),
                  input_hashes=hashes, current_reference_git=current_git,
                  verified_panel_fasta=str(args.panel_fasta) if args.panel_fasta else None,
                  frozen_prediction_source='predictions_minimal.csv; no fitting/retransformation',
                  old_panel_prediction_source='Table_S8.csv::Pred_log10_negB22_hybrid')
    write_json(qcdir / 'prediction_update_summary.json', report)
    print(f'Prepared {len(pred)}-pair inventory and {len(updated)} experimental rows in {out}')


if __name__ == '__main__':
    main()
