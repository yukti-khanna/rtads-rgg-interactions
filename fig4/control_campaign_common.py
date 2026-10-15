"""Shared exact-name exclusion, traceability, and campaign validation helpers."""
import argparse
import hashlib
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_DIR = Path('reviewer_analysis/control_simulation_manifest')
DEFAULT_EXCLUSION = DEFAULT_DIR / 'simulation_pair_names_400.txt'
STATEMENT = ('Systems present in the previous simulation list were removed from eligibility before random selection. Replacement systems were selected solely on the basis of uniform random sampling among the remaining biologically eligible candidates. No previous simulation outcome, machine-learning prediction, B22 value, OCN, charge, hydropathy, length, RG density, or other physicochemical ranking variable was used for selection.')
NAMING = {}
PREVIOUS_SEQUENCES = {}

def configure_naming(directory):
    """Read historical sequence metadata only; never interaction descriptors/outcomes."""
    NAMING.clear()
    PREVIOUS_SEQUENCES.clear()
    path = directory / 'simulation_sequence_lookup_400.csv'
    if not path.exists(): return
    table = pd.read_csv(path, usecols=['variant_id','molecule_type','parent_id','variant_type','shuffle_number','sequence'])
    for row in table.to_dict('records'):
        number = None if row['variant_type'] == 'WT' else int(row['shuffle_number'])
        key = (row['molecule_type'], row['parent_id'], number)
        vid = row['variant_id']
        if key in NAMING and NAMING[key] != vid: fail(f'conflicting previous variant identifiers for {key}')
        if vid in PREVIOUS_SEQUENCES and PREVIOUS_SEQUENCES[vid] != row['sequence']: fail(f'conflicting previous sequence for {vid}')
        if '__' in vid or not re.fullmatch(r'[A-Za-z0-9_.-]+', vid): fail(f'unsafe historical identifier: {vid}')
        NAMING[key] = vid
        PREVIOUS_SEQUENCES[vid] = row['sequence']

def fail(message):
    raise SystemExit('ERROR: ' + message)

def seed_for(namespace, identifier, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}|{namespace}|{identifier}'.encode()).digest()[:4], 'big')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def args_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--input-dir', type=Path, default=DEFAULT_DIR)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_DIR)
    p.add_argument('--existing-pairs', type=Path, default=DEFAULT_EXCLUSION)
    p.add_argument('--seed', type=int, default=20260906)
    return p

def exclusion(path):
    if not path.is_file():
        fail(f'required exclusion input not found: {path}')
    return {s.strip() for s in path.read_text().splitlines() if s.strip()}

def read_table(path, columns):
    if not path.is_file():
        fail(f'required input not found: {path}')
    try:
        df = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)
    except Exception as e:
        fail(f'cannot read {path}: {e}')
    missing = set(columns) - set(df.columns)
    if missing:
        fail(f'{path} missing required columns: {sorted(missing)}')
    return df

def variant(molecule, parent, number=None):
    key = (molecule, parent, number)
    if key in NAMING: return NAMING[key]
    token = re.sub(r'[^A-Za-z0-9.-]+', '_', str(parent)).strip('_')
    return f'{molecule}_{token}_' + ('WT' if number is None else f'SHUF{int(number):02d}')

def candidate(tad, rgg, condition, excluded, tad_sh=None, rgg_sh=None):
    tn = None if tad_sh is None else int(tad_sh['shuffle_number'])
    rn = None if rgg_sh is None else int(rgg_sh['shuffle_number'])
    tv, rv = variant('TAD', tad['parent_id'], tn), variant('RGG', rgg['RGG_ID'], rn)
    name = rv + '__' + tv
    tseq = tad['sequence'] if tad_sh is None else tad_sh['sequence']
    rseq = rgg['sequence'] if rgg_sh is None else rgg_sh['sequence']
    for vid,seq in [(tv,tseq),(rv,rseq)]:
        if vid in PREVIOUS_SEQUENCES and PREVIOUS_SEQUENCES[vid] != seq:
            fail(f'regenerated sequence differs from historical identifier {vid}; do not reuse its identifier')
    return dict(pair_name=name, TAD_parent=tad['parent_id'], TAD_gene=tad['gene'],
        TAD_class=tad['class'], TAD_variant=tv, TAD_variant_type='WT' if tn is None else 'shuffle',
        TAD_shuffle_number=tn, RGG_parent=rgg['RGG_ID'], RGG_UniProt=rgg['UniProt_ID'],
        RGG_variant=rv, RGG_variant_type='WT' if rn is None else 'shuffle', RGG_shuffle_number=rn,
        condition=condition, TAD_sequence=tad['sequence'] if tad_sh is None else tad_sh['sequence'],
        RGG_sequence=rgg['sequence'] if rgg_sh is None else rgg_sh['sequence'],
        shuffle_seed=None if condition == 'WT_WT' else int((tad_sh if tad_sh is not None else rgg_sh)['seed']),
        TAD_shuffle_seed=None if tad_sh is None else int(tad_sh['seed']),
        RGG_shuffle_seed=None if rgg_sh is None else int(rgg_sh['seed']),
        already_used=name in excluded, already_run_or_running=name in excluded)

def verify_source(args, excluded):
    meta_path = args.input_dir / 'candidate_exclusion_metadata.json'
    meta = json.loads(meta_path.read_text())
    if meta['exclusion_sha256'] != sha(args.existing_pairs) or meta['number_loaded'] != len(excluded):
        fail('exclusion input changed since Script 2; rebuild candidate universes before selection')
    return meta

def audit_update(args, campaign, details, paths):
    path = args.output_dir / 'control_selection_audit_final.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    digest = sha(args.existing_pairs)
    if data.get('exclusion_sha256', digest) != digest:
        fail('audit belongs to a different exclusion input; use a distinct output directory')
    data.update(global_seed=args.seed, exclusion_pairs_filename=str(args.existing_pairs.resolve()),
        exclusion_sha256=digest, number_exclusion_pair_names_loaded=len(exclusion(args.existing_pairs)),
        python_version=platform.python_version(), pandas_version=pd.__version__, numpy_version=np.__version__)
    meta = json.loads((args.input_dir / 'candidate_exclusion_metadata.json').read_text())
    data['number_shuffle_candidates_removed'] = meta['shuffle_candidates_removed']
    data['number_basic_control_candidates_removed'] = meta['basic_control_candidates_removed']
    data['existing_pair_category_counts'] = meta['existing_pair_category_counts']
    data['excluded_candidate_pairs'] = pd.read_csv(args.input_dir / 'excluded_existing_pairs_from_candidates.csv').to_dict('records')
    data['native_RTAD_eligible_pool_count'] = len(pd.read_csv(args.input_dir / 'native_RTAD_eligible_pool.csv'))
    data['all_input_sha256'] = json.loads((args.input_dir / 'generation_input_audit.json').read_text())['original_input_sha256']
    for filename in ['native_RTAD_eligible_pool.csv','external_acidic_AD_parents.csv','basic_control_TAD_parents.csv',
                     'shuffle_TAD_variants.csv','shuffle_RGG_variants.csv','eligible_conventional_RGGs.csv',
                     'shuffle_candidate_universe.parquet','shuffle_WT_reference_universe.csv',
                     'basic_control_candidate_universe.csv','candidate_exclusion_metadata.json',
                     'existing_400_pair_classification.csv']:
        data['all_input_sha256'][str((args.input_dir / filename).resolve())] = sha(args.input_dir / filename)
    data['all_input_sha256'][str(args.existing_pairs.resolve())] = digest
    naming_path = args.input_dir / 'simulation_sequence_lookup_400.csv'
    if naming_path.exists(): data['all_input_sha256'][str(naming_path.resolve())] = sha(naming_path)
    data['stage1_generation_seeds'] = {}
    for filename, parent_col in [('shuffle_TAD_variants.csv', 'parent_id'), ('shuffle_RGG_variants.csv', 'RGG_parent_id')]:
        table = pd.read_csv(args.input_dir / filename, usecols=[parent_col, 'seed']).drop_duplicates()
        data['stage1_generation_seeds'][filename] = {r[parent_col]: int(r['seed']) for r in table.to_dict('records')}
    data[campaign] = dict(details, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                         final_output_sha256={str(p): sha(p) for p in paths})
    path.write_text(json.dumps(data, indent=2) + '\n')

def outputs(args, manifest, stem, expected_count, expected_sequences, excluded):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = manifest.rename(columns={'pair_name': 'name'})
    columns = ['name', 'campaign', 'condition', 'TAD_parent', 'TAD_gene', 'TAD_class', 'TAD_variant',
               'TAD_variant_type', 'TAD_shuffle_number', 'RGG_parent', 'RGG_UniProt', 'RGG_variant',
               'RGG_variant_type', 'RGG_shuffle_number', 'TAD_sequence', 'RGG_sequence',
               'TAD_shuffle_seed', 'RGG_shuffle_seed', 'shuffle_seed', 'selection_seed', 'already_used', 'already_run_or_running']
    manifest = manifest[columns]
    names = manifest['name'].tolist()
    if len(names) != expected_count or len(set(names)) != expected_count or set(names) & excluded:
        fail('final count, uniqueness, or exclusion overlap validation failed')
    if manifest['already_run_or_running'].any():
        fail('an already-run pair entered the selection')
    records = {}
    lookup = []
    for molecule in ('TAD', 'RGG'):
        for vid, group in manifest.groupby(f'{molecule}_variant', sort=True):
            sequences = group[f'{molecule}_sequence'].unique()
            if len(sequences) != 1 or not re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]+', sequences[0]):
                fail(f'invalid or inconsistent sequence for {vid}')
            if vid in records:
                fail(f'duplicate identifier: {vid}')
            records[vid] = sequences[0]
            row = group.iloc[0]
            lookup.append(dict(variant_id=vid, molecule_type=molecule, parent_id=row[f'{molecule}_parent'],
                gene_or_uniprot=row['TAD_gene' if molecule == 'TAD' else 'RGG_UniProt'],
                biological_class=row['TAD_class'] if molecule == 'TAD' else 'conventional_RGG',
                variant_type=row[f'{molecule}_variant_type'],
                shuffle_number=row[f'{molecule}_shuffle_number'],
                shuffle_seed=row[f'{molecule}_shuffle_seed'], sequence=sequences[0], length=len(sequences[0])))
    components = set()
    for name in names:
        if name.count('__') != 1:
            fail(f'pair does not split exactly once: {name}')
        r, t = name.split('__')
        if not r.startswith('RGG_') or not t.startswith('TAD_'):
            fail(f'incorrect pair component order: {name}')
        components.update((r, t))
    if components != set(records) or len(records) != expected_sequences:
        fail('FASTA membership or expected identifier count failed')
    pair_path = out / f'{stem[0]}_pairs_{stem[1]}_new.txt'
    fasta_path = out / f'{stem[0]}_sequences_{stem[1]}_new.fasta'
    manifest_path = out / f'{stem[0]}_manifest_{stem[1]}_new.csv'
    if args.existing_pairs.resolve() in {p.resolve() for p in (pair_path, fasta_path, manifest_path)}:
        fail('output would overwrite the exclusion input')
    pair_path.write_text('\n'.join(names) + '\n')
    fasta_path.write_text(''.join(f'>{vid}\n{seq}\n' for vid, seq in records.items()))
    manifest.to_csv(manifest_path, index=False)
    pd.DataFrame(lookup).to_csv(out / f'{stem[0]}_sequence_lookup_{stem[1]}_new.csv', index=False)
    saved = pair_path.read_text().splitlines()
    headers = [s[1:] for s in fasta_path.read_text().splitlines() if s.startswith('>')]
    if set(saved) != set(pd.read_csv(manifest_path)['name']) or set(saved) & exclusion(args.existing_pairs):
        fail('serialized pair/manifest/exclusion validation failed')
    if set(headers) != components or len(headers) != len(set(headers)):
        fail('serialized FASTA validation failed')
    others = ([out/'basic_control_pairs_200_new.txt', out/'basic_control_pairs_100_new.txt']
              if stem[0] == 'shuffle' else [out/'shuffle_pairs_400_new.txt'])
    for other in others:
        if other.exists() and set(saved) & set(other.read_text().splitlines()):
            fail('the two new campaigns overlap')
    return pair_path, fasta_path, manifest_path

def validate_shuffles(df, tad_parents, rgg_parents):
    tad = tad_parents.set_index('parent_id')['sequence'].to_dict()
    rgg = rgg_parents.set_index('RGG_ID')['sequence'].to_dict()
    for row in df.to_dict('records'):
        for molecule, parents in [('TAD', tad), ('RGG', rgg)]:
            seq, wt = row[f'{molecule}_sequence'], parents[row[f'{molecule}_parent']]
            if Counter(seq) != Counter(wt):
                fail(f'composition invariant failed: {row["pair_name"]}')
            if row[f'{molecule}_variant_type'] == 'WT' and seq != wt:
                fail('WT partner sequence changed')
            if row[f'{molecule}_variant_type'] != 'WT' and seq == wt:
                fail('shuffle equals WT')
