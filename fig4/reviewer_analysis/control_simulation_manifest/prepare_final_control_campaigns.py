#!/usr/bin/env python3
"""Prepare (but never submit) 400 NEW dataset shuffles and 175 NEW external WT controls.

Run locally: python prepare_final_control_campaigns.py
Dependencies: numpy, pandas, openpyxl. No sibling Python modules are required.
All paths default relative to this script, independently of the shell working directory.
Only sequence/annotation/naming columns are read; no interaction outcomes are loaded.
"""
from __future__ import annotations

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

SCRIPT_DIR = Path(__file__).resolve().parent
FIG4_DIR = SCRIPT_DIR.parent.parent
GLOBAL_SEED = 20260906
AA = set('ACDEFGHIKLMNPQRSTVWY')
EXTERNAL = ('ETV4', 'SPI1', 'CREB3')
BASIC = ('TCF7L2', 'TET3', 'EOMES', 'LEF1')
CONTROLS = set(EXTERNAL + BASIC)


def fail(message):
    raise SystemExit(f'ERROR: {message}')


def check(condition, message):
    if not condition:
        fail(message)


def seed_for(namespace, identifier, seed):
    return int.from_bytes(hashlib.sha256(f'{seed}|{namespace}|{identifier}'.encode()).digest()[:4], 'big')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def sequence(value, context):
    check(pd.notna(value), f'missing sequence: {context}')
    seq = re.sub(r'\s+', '', str(value)).upper()
    check(bool(seq) and set(seq) <= AA, f'invalid amino-acid sequence: {context}')
    return seq


def fasta(path):
    records, current, parts = {}, None, []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith('>'):
            if current is not None:
                records[current] = sequence(''.join(parts), current)
            current = line[1:].split()[0] if line[1:].strip() else ''
            check(current and current not in records, f'empty/duplicate FASTA ID in {path}:{number}')
            parts = []
        else:
            check(current is not None, f'sequence before FASTA header: {path}:{number}')
            parts.append(line)
    if current is not None:
        check(current not in records, f'duplicate FASTA ID: {current}')
        records[current] = sequence(''.join(parts), current)
    check(bool(records), f'empty FASTA: {path}')
    return records


def accession(identifier):
    check('_' in identifier, f'region identifier lacks accession separator: {identifier}')
    return identifier.split('_', 1)[0]


def descriptors(seq):
    c = Counter(seq)
    q = c['K'] + c['R'] - c['D'] - c['E']
    return dict(length=len(seq), net_charge=q, NCPR=q / len(seq), aromatic_count=sum(c[x] for x in 'FWY'))


def excel_metadata(path, sheet, header, columns):
    try:
        df = pd.read_excel(path, sheet_name=sheet, header=header, usecols=lambda c: c in columns)
    except Exception as exc:
        fail(f'cannot read sequence/annotation columns from {path}: {exc}')
    check(set(columns) <= set(df.columns), f'{path} missing columns: {sorted(set(columns)-set(df.columns))}')
    return df


def resolve_variant(fig4, override, alternatives):
    if override is not None:
        return override.resolve()
    for name in alternatives:
        path = fig4 / name
        if path.is_file():
            return path
    fail(f'missing input; tried: {", ".join(str(fig4 / n) for n in alternatives)}')


class Naming:
    """Recover exact historical IDs and sequences without loading outcome columns."""
    def __init__(self):
        self.entries = {}
        self.id_sequences = {}
        self.input_paths = []

    def add(self, molecule, parent, kind, number, identifier, seq, seed=None):
        if pd.isna(parent) or pd.isna(identifier) or pd.isna(seq):
            return
        number = None if str(kind).upper() == 'WT' else int(number)
        identifier = str(identifier)
        check('__' not in identifier and re.fullmatch(r'[A-Za-z0-9_.-]+', identifier),
              f'historical identifier unsafe or incompatible with exact pair splitting: {identifier}')
        check(identifier.startswith(molecule + '_'), f'historical ID prefix mismatch: {identifier}')
        key = (molecule, str(parent), number)
        seq = sequence(seq, identifier)
        entry = dict(variant_id=identifier, sequence=seq, shuffle_seed=None if pd.isna(seed) else int(seed))
        if key in self.entries:
            old = self.entries[key]
            check(old['variant_id'] == identifier and old['sequence'] == seq, f'conflicting historical mapping: {key}')
            if old['shuffle_seed'] is not None:
                entry['shuffle_seed'] = old['shuffle_seed']
        check(identifier not in self.id_sequences or self.id_sequences[identifier] == seq,
              f'historical identifier maps to different sequences: {identifier}')
        self.entries[key] = entry
        self.id_sequences[identifier] = seq

    def load(self, directory):
        lookup = directory / 'simulation_sequence_lookup_400.csv'
        if lookup.is_file():
            allowed = {'variant_id', 'molecule_type', 'parent_id', 'variant_type', 'shuffle_number', 'sequence', 'seed', 'shuffle_seed'}
            table = pd.read_csv(lookup, usecols=lambda c: c in allowed)
            required = allowed - {'seed', 'shuffle_seed'}
            check(required <= set(table.columns), f'missing naming columns in {lookup}')
            for row in table.to_dict('records'):
                self.add(row['molecule_type'], row['parent_id'], row['variant_type'], row['shuffle_number'],
                         row['variant_id'], row['sequence'], row.get('shuffle_seed', row.get('seed')))
            self.input_paths.append(lookup)
        manifest = directory / 'simulation_manifest_400.csv'
        if manifest.is_file():
            allowed = {f'{m}_{field}' for m in ('TAD', 'RGG') for field in
                       ('parent_id', 'variant_id', 'variant_type', 'shuffle_number', 'sequence', 'seed', 'shuffle_seed')}
            table = pd.read_csv(manifest, usecols=lambda c: c in allowed)
            for m in ('TAD', 'RGG'):
                required = {f'{m}_{f}' for f in ('parent_id', 'variant_id', 'variant_type', 'shuffle_number', 'sequence')}
                check(required <= set(table.columns), f'missing historical {m} naming columns in {manifest}')
            for row in table.to_dict('records'):
                for m in ('TAD', 'RGG'):
                    self.add(m, row[f'{m}_parent_id'], row[f'{m}_variant_type'], row[f'{m}_shuffle_number'],
                             row[f'{m}_variant_id'], row[f'{m}_sequence'], row.get(f'{m}_shuffle_seed', row.get(f'{m}_seed')))
            self.input_paths.append(manifest)

    def identifier(self, molecule, parent, number=None):
        key = (molecule, parent, number)
        if key in self.entries:
            return self.entries[key]['variant_id']
        token = re.sub(r'[^A-Za-z0-9_.-]+', '_', parent)
        token = re.sub(r'_+', '_', token).strip('_')
        return f'{molecule}_{token}_' + ('WT' if number is None else f'SHUF{number:02d}')

    def wt(self, molecule, parent):
        vid = self.identifier(molecule, parent['parent_id'])
        seq = parent['sequence']
        check(vid not in self.id_sequences or self.id_sequences[vid] == seq, f'WT sequence differs from historical ID: {vid}')
        return dict(variant_id=vid, sequence=seq, shuffle_number=None, shuffle_seed=None,
                    variant_type='WT', origin='input_WT')


def make_shuffles(parent, molecule, naming, seed, count):
    """Preserve existing shuffle IDs/sequences, fill gaps by full monomer permutations."""
    seq, parent_id = parent['sequence'], parent['parent_id']
    check(len(set(seq)) > 1, f'cannot shuffle monomeric sequence: {parent_id}')
    generation_seed = seed_for(molecule + '_shuffle', parent_id, seed)
    rng = np.random.default_rng(generation_seed)
    chars = np.array(list(seq), dtype='U1')
    generated = set()
    for _ in range(100000):
        shuffled = ''.join(rng.permutation(chars))
        if shuffled != seq:
            generated.add(shuffled)
        if len(generated) == count:
            break
    check(len(generated) == count, f'cannot generate {count} unique non-WT permutations for {parent_id}')
    generated = sorted(generated)  # Same legacy deterministic numbering where recoverable.
    records, used = {}, set()
    for number in range(1, count + 1):
        previous = naming.entries.get((molecule, parent_id, number))
        if previous is not None:
            check(previous['sequence'] != seq and Counter(previous['sequence']) == Counter(seq),
                  f'historical shuffle violates composition/WT rules: {previous["variant_id"]}')
            check(previous['sequence'] not in used, f'duplicate historical shuffle within parent: {parent_id}')
            records[number] = dict(previous, shuffle_number=number, variant_type='shuffle', origin='historical_mapping')
            used.add(previous['sequence'])
    for number in range(1, count + 1):
        if number in records:
            continue
        shuffled = generated[number - 1]
        attempts = 0
        while shuffled in used or shuffled == seq:
            attempts += 1
            check(attempts <= 100000, f'cannot fill unique shuffle variants for {parent_id}')
            shuffled = ''.join(rng.permutation(chars))
        records[number] = dict(variant_id=naming.identifier(molecule, parent_id, number), sequence=shuffled,
                               shuffle_number=number, shuffle_seed=generation_seed, variant_type='shuffle', origin='generated')
        used.add(shuffled)
    for record in records.values():
        check(Counter(record['sequence']) == Counter(seq) and descriptors(record['sequence']) == descriptors(seq),
              f'shuffle invariant failure: {record["variant_id"]}')
    check(len(used) == count, f'duplicate shuffle sequences: {parent_id}')
    return [records[n] for n in sorted(records)], generation_seed


def load_parents(paths):
    all_records = fasta(paths['all_fasta'])
    rg_records = fasta(paths['rgg_fasta'])
    check(len(all_records) == 1238 and len(rg_records) == 1008, 'catalogue counts must be 1238 total and 1008 RG regions')
    check(set(rg_records) <= set(all_records) and all(all_records[k] == s for k, s in rg_records.items()),
          'RG catalogue ID/sequence cross-check failed')
    rtads = {k: s for k, s in all_records.items() if k not in rg_records}
    check(len(rtads) == 230, 'predicted R-TAD classification must yield exactly 230 records')
    conventional = {line.strip().split()[0] for line in paths['conventional'].read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith('#')}
    rggs = [dict(parent_id=k, UniProt_ID=accession(k), sequence=s, gene=accession(k), biological_class='dataset_conventional_RGG')
            for k, s in sorted(rg_records.items()) if accession(k) in conventional]
    check(bool(rggs), 'no conventional RGG dataset regions found')
    cols = ['Gene name', 'UniProt ID', 'Category', 'External dataset count', 'Representative sequence', 'Overlapping R-TAD region ID(s)']
    annotation = excel_metadata(paths['tad_selection'], 'Selected region clusters', 0, cols)
    natives = []
    for row in annotation.to_dict('records'):
        gene = str(row['Gene name']).strip()
        if gene in CONTROLS or pd.isna(row['Overlapping R-TAD region ID(s)']):
            continue
        count = pd.to_numeric(row['External dataset count'], errors='coerce')
        if pd.isna(count) or count < 2:
            continue
        ids = sorted({s.strip() for s in str(row['Overlapping R-TAD region ID(s)']).split(';') if s.strip()})
        check(set(ids) <= set(rtads), f'documented R-TAD overlap not in catalogue: {gene}/{ids}')
        if not ids:
            continue
        rid = ids[0]  # Non-feature deterministic region tie-break, one candidate per protein.
        check(accession(rid) == str(row['UniProt ID']).strip(), f'accession mismatch for {gene}')
        check(sum(s == rtads[rid] for s in rtads.values()) == 1, f'ambiguous sequence-to-catalogue mapping: {rid}')
        natives.append(dict(parent_id=rid, gene=gene, UniProt_ID=accession(rid), sequence=rtads[rid],
                            biological_class='dataset_RTAD', external_dataset_count=int(count), RTAD_overlap_status=f'documented: {rid}'))
    native_df = pd.DataFrame(natives)
    check(not native_df.empty, 'native eligible pool is empty')
    native_df = native_df.sort_values(['UniProt_ID', 'parent_id']).drop_duplicates('UniProt_ID').drop_duplicates('gene')
    natives = native_df.to_dict('records')
    check(len(natives) >= 20, f'need 20 eligible native genes/proteins; only {len(natives)} available')
    external = excel_metadata(paths['external'], 0, 2, ['Protein', 'UniProt ID', 'Core representative sequence'])
    controls = []
    for gene in EXTERNAL:
        rows = external[external.Protein.astype(str).str.strip() == gene]
        ann = annotation[annotation['Gene name'].astype(str).str.strip() == gene]
        check(len(rows) == len(ann) == 1, f'external core/annotation must map exactly once: {gene}')
        row, note = rows.iloc[0], ann.iloc[0]
        seq = sequence(row['Core representative sequence'], gene)
        check(seq == sequence(note['Representative sequence'], gene), f'external representative mismatch: {gene}')
        check(pd.isna(note['Overlapping R-TAD region ID(s)']) and '+ R-TAD' not in str(note['Category'])
              and pd.notna(note['External dataset count']) and float(note['External dataset count']) >= 2,
              f'cannot verify externally supported non-dataset AD: {gene}')
        uid = str(row['UniProt ID']).strip()
        controls.append(dict(parent_id=f'{uid}_external_AD_core', gene=gene, UniProt_ID=uid, sequence=seq,
                             biological_class='nondataset_external_AD'))
    basic = excel_metadata(paths['basic'], 'Basic controls', 2, ['Gene', 'UniProt ID', 'Start', 'End', 'Length', 'Sequence'])
    for gene in BASIC:
        rows = basic[basic.Gene.astype(str).str.strip() == gene]
        check(len(rows) == 1, f'expected exactly one supplied basic control: {gene}')
        row = rows.iloc[0]
        seq = sequence(row.Sequence, gene)
        check(len(seq) == int(row.Length), f'supplied basic length mismatch: {gene}')
        uid = str(row['UniProt ID']).strip()
        controls.append(dict(parent_id=f'{uid}_{int(row.Start)}_{int(row.End)}_basic_AD_control', gene=gene,
                             UniProt_ID=uid, sequence=seq, biological_class='nondataset_basic_control'))
    check(len(controls) == 7 and {p['gene'] for p in controls} == CONTROLS, 'non-dataset control identities invalid')
    check(all(p['parent_id'] not in rtads and p['sequence'] not in set(rtads.values()) for p in controls),
          'a supposed non-dataset control maps to the 230-sequence dataset; classification must be resolved')
    return natives, rggs, controls, rtads, rg_records


def pair(tad_variant, rgg_variant):
    return rgg_variant['variant_id'] + '__' + tad_variant['variant_id']


def system(tad, rgg, tv, rv, condition, campaign, selection_seed):
    check(not (tv['variant_type'] == rv['variant_type'] == 'shuffle'), 'double shuffle forbidden')
    return dict(name=pair(tv, rv), campaign=campaign, condition=condition, TAD_parent=tad['parent_id'],
        TAD_gene=tad['gene'], TAD_class=tad['biological_class'], TAD_variant=tv['variant_id'],
        TAD_variant_type=tv['variant_type'], TAD_shuffle_number=tv['shuffle_number'], TAD_sequence=tv['sequence'],
        RGG_parent=rgg['parent_id'], RGG_UniProt=rgg['UniProt_ID'], RGG_variant=rv['variant_id'],
        RGG_variant_type=rv['variant_type'], RGG_shuffle_number=rv['shuffle_number'], RGG_sequence=rv['sequence'],
        TAD_shuffle_seed=tv['shuffle_seed'], RGG_shuffle_seed=rv['shuffle_seed'], selection_seed=selection_seed,
        already_used=False)


def validate_campaign(rows, count, expected_identifiers, old_pairs):
    names = [r['name'] for r in rows]
    check(len(names) == len(set(names)) == count and not set(names) & old_pairs, 'count, uniqueness, or exclusion validation failed')
    records = {}
    for row in rows:
        check(row['name'].count('__') == 1, f'pair must split exactly once: {row["name"]}')
        rg, tad = row['name'].split('__')
        check(rg.startswith('RGG_') and tad.startswith('TAD_'), 'pair order must be RGG__TAD')
        check((rg, tad) == (row['RGG_variant'], row['TAD_variant']), 'manifest pair components inconsistent')
        for vid, seq in [(rg, row['RGG_sequence']), (tad, row['TAD_sequence'])]:
            check(re.fullmatch(r'[A-Za-z0-9_.-]+', vid), f'unsafe FASTA identifier: {vid}')
            check(vid not in records or records[vid] == seq, f'identifier maps to different sequences: {vid}')
            records[vid] = sequence(seq, vid)
    components = {part for name in names for part in name.split('__')}
    check(set(records) == components and len(records) == expected_identifiers, 'FASTA exact membership/count failed')
    return records


def write_campaign(out, stem, rows, records):
    paths = [out / stem['pairs'], out / stem['fasta'], out / stem['manifest']]
    names = [r['name'] for r in rows]
    paths[0].write_text('\n'.join(names) + '\n')
    paths[1].write_text(''.join(f'>{vid}\n{seq}\n' for vid, seq in sorted(records.items())))
    pd.DataFrame(rows).to_csv(paths[2], index=False)
    check(paths[0].read_text().splitlines() == pd.read_csv(paths[2])['name'].tolist(), 'serialized pair/manifest mismatch')
    headers = [s[1:] for s in paths[1].read_text().splitlines() if s.startswith('>')]
    check(len(headers) == len(set(headers)) == len(records) and set(headers) == set(records), 'serialized FASTA validation failed')
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fig4-dir', type=Path, default=FIG4_DIR)
    parser.add_argument('--input-dir', type=Path, default=SCRIPT_DIR, help='Historical exclusion/naming directory')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR)
    parser.add_argument('--existing-pairs', type=Path)
    parser.add_argument('--external-controls', type=Path)
    parser.add_argument('--basic-controls', type=Path)
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED)
    parser.add_argument('--candidate-shuffles', type=int, default=20)
    parser.add_argument('--max-draws', type=int, default=10000)
    args = parser.parse_args()
    check(args.candidate_shuffles >= 3 and args.max_draws > 0, 'need >=3 candidate shuffles and a positive draw limit')
    fig4, inputs, out = args.fig4_dir.resolve(), args.input_dir.resolve(), args.output_dir.resolve()
    paths = dict(all_fasta=fig4 / 'all_tads_rggs_dis.fasta', rgg_fasta=fig4 / 'combined_rg_regions_ll_7_all.fasta',
        conventional=fig4 / 'conventional_rbp_uniprot_ids.txt', tad_selection=fig4 / 'tad_region_selection.xlsx',
        external=resolve_variant(fig4, args.external_controls, ['ETV4_SPI1_CREB3_core_longest_with_B22.xlsx', 'ETV4_SPI1_CREB3_core_and_longest_sequences.xlsx']),
        basic=resolve_variant(fig4, args.basic_controls, ['basic_AD_IDR_controls (1).xlsx', 'basic_AD_IDR_controls.xlsx']),
        exclusion=(args.existing_pairs or inputs / 'simulation_pair_names_400.txt').resolve())
    for path in paths.values():
        check(path.is_file(), f'required input missing: {path}')
    print(f'External sequence workbook: {paths["external"]}\nBasic sequence workbook: {paths["basic"]}')
    initial_hashes = {str(p): sha256(p) for p in paths.values()}
    natives, rggs, controls, rtads, rg_records = load_parents(paths)
    naming = Naming()
    naming.load(inputs)
    for p in naming.input_paths:
        initial_hashes[str(p)] = sha256(p)
    # Only dataset parents reach this generator. Non-dataset controls are WT-only.
    twt = {p['parent_id']: naming.wt('TAD', p) for p in natives + controls}
    rwt = {p['parent_id']: naming.wt('RGG', p) for p in rggs}
    tsh, rsh, generation_seeds = {}, {}, {}
    for molecule, parents, target in [('TAD', natives, tsh), ('RGG', rggs, rsh)]:
        for p in parents:
            target[p['parent_id']], s = make_shuffles(p, molecule, naming, args.seed, args.candidate_shuffles)
            generation_seeds[f'{molecule}|{p["parent_id"]}'] = s
    # Enumerate exact biological candidates first; then exclude before ALL selection RNG.
    dataset_candidates = set()
    for t in natives:
        tid = t['parent_id']
        for r in rggs:
            rid = r['parent_id']
            dataset_candidates.update(pair(v, rwt[rid]) for v in tsh[tid])
            dataset_candidates.update(pair(twt[tid], v) for v in rsh[rid])
    nondataset_candidates = {pair(twt[t['parent_id']], rwt[r['parent_id']]) for t in controls for r in rggs}
    raw_old = paths['exclusion'].read_text().splitlines()
    old_pairs = {s.strip() for s in raw_old if s.strip()}
    dataset_available = dataset_candidates - old_pairs
    nondataset_available = nondataset_candidates - old_pairs
    rejected_variants, rejected_draws = [], []
    proteins = sorted({r['UniProt_ID'] for r in rggs})
    check(len(proteins) >= 25, f'need >=25 conventional proteins; only {len(proteins)} available')
    native_seed = seed_for('final_dataset_native_selection', '20', args.seed)
    protein_seed = seed_for('final_dataset_RGG_protein_selection', '5', args.seed)
    native_rng, protein_rng = np.random.default_rng(native_seed), np.random.default_rng(protein_seed)
    region_seeds = {}
    for draw in range(1, args.max_draws + 1):
        selected_tads = [natives[i] for i in native_rng.choice(len(natives), 20, replace=False)]
        chosen_proteins = [proteins[i] for i in protein_rng.choice(len(proteins), 5, replace=False)]
        selected_rggs = []
        for uid in chosen_proteins:
            regions = [r for r in rggs if r['UniProt_ID'] == uid]
            s = seed_for('final_dataset_RGG_region', f'{draw}|{uid}', args.seed)
            region_seeds[f'{draw}|{uid}'] = s
            selected_rggs.append(regions[int(np.random.default_rng(s).integers(len(regions)))])
        valid_t, valid_r, reasons = {}, {}, []
        for t in selected_tads:
            tid = t['parent_id']
            valid_t[tid] = []
            for v in tsh[tid]:
                names = [pair(v, rwt[r['parent_id']]) for r in selected_rggs]
                conflicts = sorted(set(names) & old_pairs)
                if all(n in dataset_available for n in names):
                    valid_t[tid].append(v)
                else:
                    rejected_variants.append(dict(draw=draw, molecule='TAD', parent_id=tid,
                        variant_id=v['variant_id'], conflicting_pairs=conflicts, reason='previous_pair_conflict'))
            if len(valid_t[tid]) < 3:
                reasons.append(f'{tid}: only {len(valid_t[tid])} shuffles new across all five RGGs; need 3')
        for r in selected_rggs:
            rid = r['parent_id']
            valid_r[rid] = []
            for v in rsh[rid]:
                names = [pair(twt[t['parent_id']], v) for t in selected_tads]
                conflicts = sorted(set(names) & old_pairs)
                if all(n in dataset_available for n in names):
                    valid_r[rid].append(v)
                else:
                    rejected_variants.append(dict(draw=draw, molecule='RGG', parent_id=rid,
                        variant_id=v['variant_id'], conflicting_pairs=conflicts, reason='previous_pair_conflict'))
            if not valid_r[rid]:
                reasons.append(f'{rid}: no shuffle new across all 20 selected TADs')
        if not reasons:
            break
        rejected_draws.append(dict(draw=draw, TAD_parents=[p['parent_id'] for p in selected_tads],
            RGG_parents=[p['parent_id'] for p in selected_rggs], reasons=reasons,
            rule='redraw joint balanced parent context solely because of exact prior-run conflicts'))
    else:
        fail(f'no feasible parent set after {args.max_draws} exact-conflict-only draws; no final outputs written. Increase --candidate-shuffles or --max-draws.')
    rows_a, selected_t_variants, selected_r_variants, selection_seeds = [], [], [], {}
    for t in selected_tads:
        tid = t['parent_id']
        s = seed_for('final_three_TAD_shuffles', tid, args.seed)
        selection_seeds[f'TAD|{tid}'] = s
        candidates = sorted(valid_t[tid], key=lambda v: v['shuffle_number'])
        for order, i in enumerate(np.random.default_rng(s).choice(len(candidates), 3, replace=False), 1):
            v = candidates[int(i)]
            selected_t_variants.append(dict(parent_id=tid, gene=t['gene'], **v, selection_seed=s, selection_order=order))
            rows_a.extend(system(t, r, v, rwt[r['parent_id']], 'TAD_SHUFFLE', 'dataset_shuffle', s) for r in selected_rggs)
    for r in selected_rggs:
        rid = r['parent_id']
        s = seed_for('final_one_RGG_shuffle', rid, args.seed)
        selection_seeds[f'RGG|{rid}'] = s
        candidates = sorted(valid_r[rid], key=lambda v: v['shuffle_number'])
        v = candidates[int(np.random.default_rng(s).integers(len(candidates)))]
        selected_r_variants.append(dict(parent_id=rid, UniProt_ID=r['UniProt_ID'], **v, selection_seed=s))
        rows_a.extend(system(t, r, twt[t['parent_id']], v, 'RGG_SHUFFLE', 'dataset_shuffle', s) for t in selected_tads)
    # A region must support a complete seven-control WT block BEFORE the protein draw.
    block_eligible = []
    rejected_blocks = []
    for r in rggs:
        names = [pair(twt[t['parent_id']], rwt[r['parent_id']]) for t in controls]
        if all(n in nondataset_available for n in names):
            block_eligible.append(r)
        else:
            rejected_blocks.append(dict(parent_id=r['parent_id'], conflicting_pairs=sorted(set(names) & old_pairs)))
    block_proteins = sorted({r['UniProt_ID'] for r in block_eligible})
    check(len(block_proteins) >= 25, f'need 25 proteins with complete seven-control NEW WT blocks; only {len(block_proteins)} eligible. No final outputs written.')
    control_seed = seed_for('final_nondataset_WT_proteins', '25', args.seed)
    chosen = [block_proteins[i] for i in np.random.default_rng(control_seed).choice(len(block_proteins), 25, replace=False)]
    control_rggs, control_region_seeds = [], {}
    for uid in chosen:
        regions = [r for r in block_eligible if r['UniProt_ID'] == uid]
        s = seed_for('final_nondataset_WT_region', uid, args.seed)
        control_region_seeds[uid] = s
        control_rggs.append(regions[int(np.random.default_rng(s).integers(len(regions)))])
    rows_b = [system(t, r, twt[t['parent_id']], rwt[r['parent_id']], 'WT_WT', 'nondataset_WT', control_seed)
              for t in controls for r in control_rggs]
    # Validate ALL final designs before any output file is written.
    records_a = validate_campaign(rows_a, 400, 90, old_pairs)
    records_b = validate_campaign(rows_b, 175, 32, old_pairs)
    check(Counter(r['condition'] for r in rows_a) == {'TAD_SHUFFLE': 300, 'RGG_SHUFFLE': 100}, 'dataset condition totals invalid')
    check(len(selected_tads) == 20 and len({t['parent_id'] for t in selected_tads}) == 20
          and len({t['gene'] for t in selected_tads}) == len({t['UniProt_ID'] for t in selected_tads}) == 20,
          'dataset TAD parent/gene/protein uniqueness failed')
    check(all(t['parent_id'] in rtads and t['gene'] not in CONTROLS for t in selected_tads), 'non-dataset TAD in shuffle campaign')
    check(len(selected_rggs) == len({r['UniProt_ID'] for r in selected_rggs}) == 5
          and all(r['parent_id'] in rg_records for r in selected_rggs), 'dataset RGG membership/protein count failed')
    check(len(selected_t_variants) == 60 and len(selected_r_variants) == 5, 'retained shuffle variant counts failed')
    df_a, df_b = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
    check(df_a[df_a.condition == 'TAD_SHUFFLE'].groupby('TAD_variant').RGG_parent.nunique().eq(5).all(), 'TAD shuffle partner reuse failed')
    check(df_a[df_a.condition == 'RGG_SHUFFLE'].groupby('RGG_variant').TAD_parent.nunique().eq(20).all(), 'RGG shuffle partner reuse failed')
    check(len(control_rggs) == len({r['UniProt_ID'] for r in control_rggs}) == 25, 'non-dataset common RGG protein uniqueness failed')
    check(set(df_b.TAD_gene) == CONTROLS and df_b.TAD_parent.nunique() == 7
          and df_b.groupby('TAD_parent').size().eq(25).all() and df_b.groupby('RGG_parent').TAD_parent.nunique().eq(7).all(),
          'non-dataset seven-by-25 balance failed')
    check(df_b.TAD_variant_type.eq('WT').all() and df_b.RGG_variant_type.eq('WT').all(), 'non-dataset shuffling forbidden')
    check(not set(df_a.name) & set(df_b.name), 'campaign pair lists overlap')
    for p in [*paths.values(), *naming.input_paths]:
        check(sha256(p) == initial_hashes[str(p)], f'input changed during preparation: {p}')
    output_names = ['final_dataset_shuffle_pairs_400_new.txt', 'final_dataset_shuffle_sequences_400_new.fasta',
        'final_dataset_shuffle_manifest_400_new.csv', 'final_dataset_shuffle_TAD_parents.csv',
        'final_dataset_shuffle_RGG_parents.csv', 'final_selected_TAD_shuffles.csv', 'final_selected_RGG_shuffles.csv',
        'final_nondataset_WT_pairs_175_new.txt', 'final_nondataset_WT_sequences_175_new.fasta',
        'final_nondataset_WT_manifest_175_new.csv', 'final_nondataset_selected_RGGs_25.csv', 'final_control_selection_audit.json']
    protected = {p.resolve() for p in [*paths.values(), *naming.input_paths, Path(__file__)]}
    check(not {(out / n).resolve() for n in output_names} & protected, 'an output path would overwrite an input/script')
    out.mkdir(parents=True, exist_ok=True)
    campaign_paths = write_campaign(out, dict(pairs=output_names[0], fasta=output_names[1], manifest=output_names[2]), rows_a, records_a)
    campaign_paths += write_campaign(out, dict(pairs=output_names[7], fasta=output_names[8], manifest=output_names[9]), rows_b, records_b)
    for filename, table in [(output_names[3], selected_tads), (output_names[4], selected_rggs),
                            (output_names[5], selected_t_variants), (output_names[6], selected_r_variants), (output_names[10], control_rggs)]:
        documented = [dict(r, **descriptors(r['sequence'])) for r in table]
        pd.DataFrame(documented).to_csv(out / filename, index=False)
    check(sha256(paths['exclusion']) == initial_hashes[str(paths['exclusion'])], 'exclusion input unexpectedly changed')
    audit = dict(global_seed=args.seed, input_paths={k: str(p) for k, p in paths.items()}, input_sha256=initial_hashes,
        old_exclusion_sha256=initial_hashes[str(paths['exclusion'])], old_exclusion_counts=dict(original_lines=len(raw_old),
            nonempty=sum(bool(s.strip()) for s in raw_old), unique=len(old_pairs)),
        catalogue_counts=dict(predicted_RTADs=230, compact_RGGs=1008, total=1238), eligible_predicted_RTAD_count=len(natives),
        eligible_conventional_RGG_count=len(rggs), eligible_conventional_protein_count=len(proteins),
        selected_20_dataset_RTADs=selected_tads, selected_5_dataset_RGGs=selected_rggs,
        selected_60_TAD_shuffles=selected_t_variants, selected_5_RGG_shuffles=selected_r_variants,
        excluded_dataset_candidates=sorted(dataset_candidates & old_pairs), excluded_nondataset_candidates=sorted(nondataset_candidates & old_pairs),
        rejected_variants=rejected_variants, rejected_parent_draws=rejected_draws, rejected_nondataset_blocks=rejected_blocks,
        successful_parent_draw=draw, candidate_shuffles_per_parent=args.candidate_shuffles,
        seven_nondataset_controls=controls, complete_seven_control_eligible_RGGs=len(block_eligible),
        complete_seven_control_eligible_proteins=len(block_proteins), selected_25_nondataset_RGGs=control_rggs,
        seeds=dict(native_selection=native_seed, dataset_protein_selection=protein_seed, dataset_region_draws=region_seeds,
                   generation=generation_seeds, variant_selection=selection_seeds, nondataset_protein_selection=control_seed,
                   nondataset_region_selection=control_region_seeds),
        naming_inputs=[str(p) for p in naming.input_paths], python_version=platform.python_version(),
        pandas_version=pd.__version__, numpy_version=np.__version__, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        output_sha256={str(out / n): sha256(out / n) for n in output_names if n != output_names[-1]},
        selection_rule='Exact prior-run availability and uniform identifier-only random sampling; no outcomes/features; no simulations launched.')
    (out / output_names[-1]).write_text(json.dumps(audit, indent=2) + '\n')
    print(f'DATASET SHUFFLES: 20 dataset TADs; 5 common RGG proteins; 60 TAD shuffles; 5 RGG shuffles; 300 + 100 = 400 NEW pairs; previous-pair overlap = 0; FASTA IDs = 90')
    print(f'NON-DATASET WT: seven controls; 25 common RGG proteins; 175 NEW pairs; previous-pair overlap = 0; FASTA IDs = 32')
    for path in campaign_paths:
        print(path)
    print(f'Audit: {out / output_names[-1]}\nNo simulations, RDFs, B22 calculations, or ML predictions were run.')


if __name__ == '__main__':
    main()
