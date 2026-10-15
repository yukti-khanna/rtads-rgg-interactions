#!/usr/bin/env python3
"""Extend the existing backbone to 400 TOTAL dataset and 175 TOTAL non-dataset pairs.

Run locally: python prepare_final_control_panels.py
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



def variant_record(naming, molecule, parent, number):
    entry = naming.entries.get((molecule, parent, number))
    check(entry is not None, f'cannot recover exact historical sequence: {molecule}/{parent}/{number}')
    return dict(entry, shuffle_number=number, variant_type='WT' if number is None else 'shuffle',
                origin='historical_sequence_mapping')


def panel_row(tad, rgg, tv, rv, condition, status, source, selection_seed=None):
    row = system(tad, rgg, tv, rv, condition, 'dataset_panel' if tad['biological_class']=='dataset_RTAD' else 'nondataset_WT_panel', selection_seed)
    name = row.pop('name')
    row.pop('already_used')
    return dict(pair_name=name, run_status=status, source=source, **row)


def pair_lists(rows):
    return [r['pair_name'] for r in rows]


def fasta_records(rows, expected):
    records = {}
    for row in rows:
        check(row['pair_name'].count('__') == 1, f'invalid exact pair name: {row["pair_name"]}')
        rg, tad = row['pair_name'].split('__')
        check((rg,tad)==(row['RGG_variant'],row['TAD_variant']) and rg.startswith('RGG_') and tad.startswith('TAD_'), 'pair naming/order mismatch')
        for vid,seq in [(rg,row['RGG_sequence']),(tad,row['TAD_sequence'])]:
            check(re.fullmatch(r'[A-Za-z0-9_.-]+',vid) and '__' not in vid, f'unsafe identifier: {vid}')
            check(vid not in records or records[vid]==seq, f'identifier maps to inconsistent sequences: {vid}')
            records[vid]=sequence(seq,vid)
    check(len(records)==expected, f'expected {expected} sequence IDs for NEW pairs; found {len(records)}')
    components={part for row in rows for part in row['pair_name'].split('__')}
    check(set(records)==components, 'FASTA includes missing/unused identifiers')
    return records


def check_matched(rows, parent_sequences, rgg_sequences):
    for row in rows:
        for molecule,mapping in [('TAD',parent_sequences),('RGG',rgg_sequences)]:
            seq,wt=row[f'{molecule}_sequence'],mapping[row[f'{molecule}_parent']]
            check(Counter(seq)==Counter(wt) and descriptors(seq)==descriptors(wt), f'matched composition/length/charge/NCPR/aromatics failed: {row["pair_name"]}')
            if row[f'{molecule}_variant_type']=='WT':
                check(seq==wt, f'WT partner sequence changed: {row["pair_name"]}')
            else:
                check(seq!=wt, f'shuffle equals WT: {row["pair_name"]}')


def candidate_tad_shuffles(parent,naming,candidate_table,seed):
    pid=parent['parent_id']
    if candidate_table is None or not candidate_table['parent_id'].eq(pid).any():
        variants,generation_seed=make_shuffles(parent,'TAD',naming,seed,20)
        return variants,dict(origin='generated_full_monomer_permutations',generation_seed=generation_seed)
    rows=candidate_table[candidate_table.parent_id==pid].sort_values('shuffle_number').to_dict('records')
    check(len(rows)==20, f'existing candidate table must contain 20 shuffles for {pid}; found {len(rows)}')
    variants=[]
    for row in rows:
        n=int(row['shuffle_number'])
        check(1<=n<=20, f'invalid candidate shuffle number for {pid}: {n}')
        seq=sequence(row['sequence'],f'{pid}/{n}')
        vid=str(row['variant_id'])
        if vid==f'{pid}__shuffle_{n:02d}':
            vid=naming.identifier('TAD',pid,n)  # Explicit legacy candidate-table convention, not fuzzy matching.
        check(vid.startswith('TAD_') and '__' not in vid and re.fullmatch(r'[A-Za-z0-9_.-]+',vid), f'incompatible simulation identifier: {vid}')
        previous=naming.entries.get(('TAD',pid,n))
        check(previous is None or (previous['variant_id']==vid and previous['sequence']==seq), f'candidate differs from historical mapping: {vid}')
        check(seq!=parent['sequence'] and Counter(seq)==Counter(parent['sequence']) and descriptors(seq)==descriptors(parent['sequence']), f'invalid exact-composition candidate: {vid}')
        raw_seed=row.get('shuffle_seed',row.get('seed'))
        variants.append(dict(variant_id=vid,sequence=seq,shuffle_number=n,
            shuffle_seed=None if pd.isna(raw_seed) else int(raw_seed),variant_type='shuffle',origin='existing_shuffle_TAD_variants.csv'))
    check(len({v['sequence'] for v in variants})==20 and len({v['variant_id'] for v in variants})==20 and len({v['shuffle_number'] for v in variants})==20,
          f'duplicate candidate sequence/ID/number for {pid}')
    return variants,dict(origin='existing_shuffle_TAD_variants.csv')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fig4-dir',type=Path,default=FIG4_DIR)
    parser.add_argument('--input-dir',type=Path,default=SCRIPT_DIR)
    parser.add_argument('--output-dir',type=Path,default=SCRIPT_DIR)
    parser.add_argument('--existing-pairs',type=Path)
    parser.add_argument('--external-controls',type=Path)
    parser.add_argument('--basic-controls',type=Path)
    parser.add_argument('--seed',type=int,default=GLOBAL_SEED)
    args=parser.parse_args()
    fig4,inputs,out=args.fig4_dir.resolve(),args.input_dir.resolve(),args.output_dir.resolve()
    paths=dict(all_fasta=fig4/'all_tads_rggs_dis.fasta',rgg_fasta=fig4/'combined_rg_regions_ll_7_all.fasta',
        conventional=fig4/'conventional_rbp_uniprot_ids.txt',tad_selection=fig4/'tad_region_selection.xlsx',
        external=resolve_variant(fig4,args.external_controls,['ETV4_SPI1_CREB3_core_longest_with_B22.xlsx','ETV4_SPI1_CREB3_core_and_longest_sequences.xlsx']),
        basic=resolve_variant(fig4,args.basic_controls,['basic_AD_IDR_controls (1).xlsx','basic_AD_IDR_controls.xlsx']),
        exclusion=(args.existing_pairs or inputs/'simulation_pair_names_400.txt').resolve())
    for path in paths.values():
        check(path.is_file(),f'required input not found: {path}')
    input_hashes={str(p):sha256(p) for p in paths.values()}
    natives,rggs,controls,rtads,rg_records=load_parents(paths)
    naming=Naming()
    naming.load(inputs)
    historical_fasta=inputs/'simulation_sequences_400.fasta'
    if historical_fasta.is_file():
        for vid,seq in fasta(historical_fasta).items():
            match=re.fullmatch(r'(TAD|RGG)_(.+)_(WT|SHUF(\d+))',vid)
            check(match is not None,f'cannot parse historical sequence identifier: {vid}')
            molecule,parent,label,number=match.groups()
            naming.add(molecule,parent,'WT' if label=='WT' else 'shuffle',None if number is None else int(number),vid,seq)
        naming.input_paths.append(historical_fasta)
    for path in naming.input_paths:
        input_hashes[str(path)]=sha256(path)
    raw_old=paths['exclusion'].read_text().splitlines()
    old_pairs={s.strip() for s in raw_old if s.strip()}
    parent_map={p['parent_id']:p for p in natives+controls}
    rgg_map={p['parent_id']:p for p in rggs}
    id_map={v['variant_id']:(m,p,n) for (m,p,n),v in naming.entries.items()}
    historical_labels={}
    manifest=inputs/'simulation_manifest_400.csv'
    if manifest.is_file():
        table=pd.read_csv(manifest,usecols=lambda c:c in {'name','condition'})
        if {'name','condition'}<=set(table.columns):
            historical_labels=dict(zip(table['name'],table['condition']))
    classified=[]
    diagnostics=[]
    for name in sorted(old_pairs):
        category='UNMAPPED'; condition=''; tad_id=rg_id=''
        if name.count('__')!=1:
            diagnostics.append(f'{name}: pair must split exactly once on __')
        else:
            rg_id,tad_id=name.split('__')
            tm,rm=id_map.get(tad_id),id_map.get(rg_id)
            if not tm or not rm:
                diagnostics.append(f'{name}: exact sequence mapping not recovered')
            elif tm[0]!='TAD' or rm[0]!='RGG':
                diagnostics.append(f'{name}: component molecule types reversed')
            elif tm[1] not in parent_map or rm[1] not in rgg_map:
                diagnostics.append(f'{name}: parent absent from biological/conventional source pools')
            else:
                t,r=parent_map[tm[1]],rgg_map[rm[1]]
                ts,rs=tm[2] is not None,rm[2] is not None
                if t['gene'] in CONTROLS:
                    category='OLD_NOT_USED_IN_FINAL_DESIGN' if ts or rs else 'VALID_NONDATASET_WT'
                    condition='NONDATASET_SHUFFLED' if ts or rs else 'WT_WT'
                elif tm[1] in rtads:
                    if ts and rs:
                        diagnostics.append(f'{name}: unexpected double-shuffled dataset pair')
                    else:
                        category='VALID_DATASET'
                        condition='TAD_SHUFFLE' if ts else 'RGG_SHUFFLE' if rs else 'WT_WT'
        classified.append(dict(pair_name=name,category=category,condition=condition,TAD_variant=tad_id,RGG_variant=rg_id))
    counts=Counter(row['category'] for row in classified)
    conditions=Counter(row['condition'] for row in classified if row['category']=='VALID_DATASET')
    expected={'VALID_DATASET':260,'VALID_NONDATASET_WT':35,'OLD_NOT_USED_IN_FINAL_DESIGN':105}
    check(len(old_pairs)==400 and dict(counts)==expected and conditions=={'WT_WT':65,'TAD_SHUFFLE':130,'RGG_SHUFFLE':65} and not diagnostics,
        f'OLD-400 CLASSIFICATION FAILED before selection. Unique={len(old_pairs)}; categories={dict(counts)}; dataset conditions={dict(conditions)}; expected={expected}. Diagnostics:\n'+'\n'.join(diagnostics[:30]))
    valid_old_dataset=[row for row in classified if row['category']=='VALID_DATASET']
    valid_old_non=[row for row in classified if row['category']=='VALID_NONDATASET_WT']
    unused=[row['pair_name'] for row in classified if row['category']=='OLD_NOT_USED_IN_FINAL_DESIGN']
    existing_tids=sorted({id_map[row['TAD_variant']][1] for row in valid_old_dataset})
    existing_rids=sorted({id_map[row['RGG_variant']][1] for row in valid_old_dataset})
    check(len(existing_tids)==13 and len(existing_rids)==5,'old dataset backbone must contain exactly 13 TAD and five RGG parents')
    existing_tads=[parent_map[p] for p in existing_tids]
    existing_rggs=[rgg_map[p] for p in existing_rids]
    check(len({p['UniProt_ID'] for p in existing_rggs})==5 and all(p['parent_id'] in rg_records for p in existing_rggs),
        'the five backbone RGGs must be dataset conventional regions from distinct proteins')
    check(all(p['parent_id'] in rtads and p['gene'] not in CONTROLS for p in existing_tads),'non-dataset TAD in old dataset backbone')
    existing_rg_shuffles={}
    for rid in existing_rids:
        numbers={id_map[row['RGG_variant']][2] for row in valid_old_dataset if id_map[row['RGG_variant']][1]==rid and row['condition']=='RGG_SHUFFLE'}
        check(len(numbers)==1,f'expected exactly one historical RGG shuffle for {rid}; found {numbers}')
        existing_rg_shuffles[rid]=variant_record(naming,'RGG',rid,next(iter(numbers)))
    tad_ab={}
    ab_rules={}
    for tid in existing_tids:
        rows=[row for row in valid_old_dataset if id_map[row['TAD_variant']][1]==tid and row['condition']=='TAD_SHUFFLE']
        numbers=sorted({id_map[row['TAD_variant']][2] for row in rows})
        check(len(numbers)==2,f'{tid}: old backbone must have exactly two TAD shuffle variants')
        labels={}
        for number in numbers:
            found={historical_labels.get(row['pair_name']) for row in rows if id_map[row['TAD_variant']][2]==number}
            labels[number]=next(iter(found)) if len(found)==1 else None
        has_historical_ab=any(historical_labels.get(row['pair_name']) in {'TAD_SHUFFLE_A','TAD_SHUFFLE_B'} for row in rows)
        check(not has_historical_ab or set(labels.values())=={'TAD_SHUFFLE_A','TAD_SHUFFLE_B'},
              f'{tid}: inconsistent historical A/B labels across partners; cannot silently relabel the backbone')
        if set(labels.values())=={'TAD_SHUFFLE_A','TAD_SHUFFLE_B'}:
            tad_ab[tid]={number:labels[number] for number in numbers}
            ab_rules[tid]='recovered historical manifest condition labels'
        else:
            tad_ab[tid]=dict(zip(numbers,['TAD_SHUFFLE_A','TAD_SHUFFLE_B']))
            ab_rules[tid]='deterministic shuffle-number order; historical A/B labels unavailable'
    reused_a,reused_b=[],[]
    old_source=str(paths['exclusion'])
    for row in valid_old_dataset+valid_old_non:
        tm,rm=id_map[row['TAD_variant']],id_map[row['RGG_variant']]
        tv,rv=variant_record(naming,*tm),variant_record(naming,*rm)
        condition=tad_ab[tm[1]][tm[2]] if row['condition']=='TAD_SHUFFLE' else row['condition']
        target=reused_a if row['category']=='VALID_DATASET' else reused_b
        target.append(panel_row(parent_map[tm[1]],rgg_map[rm[1]],tv,rv,condition,'REUSED_EXISTING',old_source))
    old_df=pd.DataFrame(reused_a)
    expected_conditions={'WT_WT','TAD_SHUFFLE_A','TAD_SHUFFLE_B','RGG_SHUFFLE'}
    for tid in existing_tids:
        for rid in existing_rids:
            block=old_df[(old_df.TAD_parent==tid)&(old_df.RGG_parent==rid)]
            check(len(block)==4 and set(block.condition)==expected_conditions,f'incomplete old dataset context: {tid}/{rid}')
        check(old_df[(old_df.TAD_parent==tid)&old_df.condition.eq('TAD_SHUFFLE_A')].TAD_variant.nunique()==1 and
              old_df[(old_df.TAD_parent==tid)&old_df.condition.eq('TAD_SHUFFLE_B')].TAD_variant.nunique()==1,
              f'historical TAD shuffle not reused across five RGGs: {tid}')
    old_non_df=pd.DataFrame(reused_b)
    check(set(old_non_df.TAD_gene)==CONTROLS and set(old_non_df.RGG_parent)==set(existing_rids) and
          old_non_df.groupby('TAD_parent').size().eq(5).all() and old_non_df.groupby('RGG_parent').TAD_parent.nunique().eq(7).all(),
          'old non-dataset WT backbone must be exactly seven controls x same five RGGs')
    check_matched(reused_a+reused_b,{p['parent_id']:p['sequence'] for p in natives+controls},
                  {p['parent_id']:p['sequence'] for p in rggs})
    # Hard structural/count checks above occur before any new random selection.
    candidate_path=inputs/'shuffle_TAD_variants.csv'
    candidate_table=None
    if candidate_path.is_file():
        allowed={'parent_id','variant_id','shuffle_number','shuffle_seed','seed','sequence'}
        candidate_table=pd.read_csv(candidate_path,usecols=lambda c:c in allowed)
        check({'parent_id','variant_id','shuffle_number','sequence'}<=set(candidate_table.columns),f'missing deterministic candidate columns: {candidate_path}')
        input_hashes[str(candidate_path)]=sha256(candidate_path)
    old_genes={p['gene'] for p in existing_tads}
    old_proteins={p['UniProt_ID'] for p in existing_tads}
    remaining=[p for p in natives if p['parent_id'] not in existing_tids and p['gene'] not in old_genes and p['UniProt_ID'] not in old_proteins]
    twt={p['parent_id']:naming.wt('TAD',p) for p in natives+controls}
    rwt={p['parent_id']:naming.wt('RGG',p) for p in rggs}
    rejected=[]; viable_tads=[]; viable_variants={}; candidate_origins={}
    for t in remaining:
        tid=t['parent_id']
        fixed=[pair(twt[tid],rwt[rid]) for rid in existing_rids]+[pair(twt[tid],existing_rg_shuffles[rid]) for rid in existing_rids]
        conflicts=sorted(set(fixed)&old_pairs)
        if conflicts:
            rejected.append(dict(kind='dataset_TAD_parent',parent_id=tid,reason='required WT/reference block already present',conflicting_pairs=conflicts))
            continue
        variants,origin=candidate_tad_shuffles(t,naming,candidate_table,args.seed)
        candidate_origins[tid]=origin
        viable=[]
        for v in variants:
            conflicts=sorted({pair(v,rwt[rid]) for rid in existing_rids}&old_pairs)
            if conflicts:
                rejected.append(dict(kind='dataset_TAD_shuffle',parent_id=tid,variant_id=v['variant_id'],reason='existing exact pair',conflicting_pairs=conflicts))
            else:
                viable.append(v)
        if len(viable)<2:
            rejected.append(dict(kind='dataset_TAD_parent',parent_id=tid,reason=f'only {len(viable)} TAD shuffles new across all five retained RGGs'))
        else:
            viable_tads.append(t); viable_variants[tid]=viable
    check(len(viable_tads)>=7,f'need seven additional eligible complete-block dataset parents; only {len(viable_tads)} available')
    native_seed=seed_for('reuse_panel_additional_dataset_TADs','7',args.seed)
    additional_tads=[viable_tads[int(i)] for i in np.random.default_rng(native_seed).choice(len(viable_tads),7,replace=False)]
    new_a,new_tad_shuffles,variant_seeds=[],[],{}
    for t in additional_tads:
        tid=t['parent_id']; variants=viable_variants[tid]
        s=seed_for('reuse_panel_two_TAD_shuffles',tid,args.seed)
        variant_seeds[tid]=s
        selected=[variants[int(i)] for i in np.random.default_rng(s).choice(len(variants),2,replace=False)]
        for label,v in zip(['TAD_SHUFFLE_A','TAD_SHUFFLE_B'],selected):
            new_tad_shuffles.append(dict(parent_id=tid,gene=t['gene'],condition=label,**v,selection_seed=s))
        for r in existing_rggs:
            rid=r['parent_id']
            block=[('WT_WT',twt[tid],rwt[rid]),('TAD_SHUFFLE_A',selected[0],rwt[rid]),
                   ('TAD_SHUFFLE_B',selected[1],rwt[rid]),('RGG_SHUFFLE',twt[tid],existing_rg_shuffles[rid])]
            new_a.extend(panel_row(t,r,tv,rv,c,'NEW_TO_RUN','additional_dataset_TAD_complete_block',s) for c,tv,rv in block)
    existing_rgg_proteins={p['UniProt_ID'] for p in existing_rggs}
    eligible_rggs=[]
    for r in rggs:
        if r['UniProt_ID'] in existing_rgg_proteins:
            continue  # Preserve 25 distinct proteins, including the five already run.
        conflicts=sorted({pair(twt[t['parent_id']],rwt[r['parent_id']]) for t in controls}&old_pairs)
        if conflicts:
            rejected.append(dict(kind='nondataset_RGG_block',parent_id=r['parent_id'],reason='incomplete seven-control NEW block',conflicting_pairs=conflicts))
        else: eligible_rggs.append(r)
    eligible_proteins=sorted({p['UniProt_ID'] for p in eligible_rggs})
    check(len(eligible_proteins)>=20,f'need 20 additional conventional proteins distinct from retained five; only {len(eligible_proteins)} eligible')
    protein_seed=seed_for('reuse_panel_additional_nondataset_RGG_proteins','20',args.seed)
    proteins=[eligible_proteins[int(i)] for i in np.random.default_rng(protein_seed).choice(len(eligible_proteins),20,replace=False)]
    additional_rggs,region_seeds=[],{}
    for uid in proteins:
        regions=[r for r in eligible_rggs if r['UniProt_ID']==uid]
        s=seed_for('reuse_panel_additional_nondataset_RGG_region',uid,args.seed); region_seeds[uid]=s
        additional_rggs.append(regions[int(np.random.default_rng(s).integers(len(regions)))])
    new_b=[panel_row(t,r,twt[t['parent_id']],rwt[r['parent_id']],'WT_WT','NEW_TO_RUN',
                    'additional_conventional_RGG_complete_seven_control_block',protein_seed) for t in controls for r in additional_rggs]
    final_a,final_b=reused_a+new_a,reused_b+new_b
    check(len(reused_a)==260 and len(new_a)==140 and len(final_a)==400,'dataset reused/new/final counts must be 260/140/400')
    check(len(reused_b)==35 and len(new_b)==140 and len(final_b)==175,'non-dataset reused/new/final counts must be 35/140/175')
    for reused,new,final in [(reused_a,new_a,final_a),(reused_b,new_b,final_b)]:
        rp,npairs,fp=set(pair_lists(reused)),set(pair_lists(new)),set(pair_lists(final))
        check(rp<=old_pairs and not npairs&old_pairs and not rp&npairs and len(fp)==len(final),'reused/new pair membership or uniqueness failed')
    check(not (set(pair_lists(final_a))|set(pair_lists(final_b)))&set(unused),'unwanted historical non-dataset shuffles leaked into final panels')
    df_a,df_b=pd.DataFrame(final_a),pd.DataFrame(final_b)
    check(df_a.run_status.value_counts().to_dict()=={'REUSED_EXISTING':260,'NEW_TO_RUN':140} and
          df_b.run_status.value_counts().to_dict()=={'NEW_TO_RUN':140,'REUSED_EXISTING':35},'manifest run-status counts invalid')
    check(df_a.TAD_parent.nunique()==20 and df_a.RGG_parent.nunique()==5 and set(df_a.RGG_parent)==set(existing_rids),'final dataset parent structure failed')
    for (tid,rid),block in df_a.groupby(['TAD_parent','RGG_parent']):
        check(len(block)==4 and set(block.condition)==expected_conditions,f'incomplete final four-condition context: {tid}/{rid}')
    check(df_a.groupby('TAD_parent').size().eq(20).all() and df_a.TAD_parent.map(lambda p:p in rtads).all() and not set(df_a.TAD_gene)&CONTROLS,'final dataset membership/balance failed')
    check(df_a.condition.value_counts().to_dict()=={c:100 for c in expected_conditions},'final dataset must have 100 systems per condition')
    check(df_b.TAD_parent.nunique()==7 and set(df_b.TAD_gene)==CONTROLS and df_b.RGG_parent.nunique()==25 and
          df_b.RGG_UniProt.nunique()==25 and df_b.groupby('TAD_parent').size().eq(25).all() and
          df_b.groupby('RGG_parent').TAD_parent.nunique().eq(7).all() and df_b.TAD_variant_type.eq('WT').all() and df_b.RGG_variant_type.eq('WT').all(),
          'non-dataset WT seven-by-25 final structure failed')
    check(not set(pair_lists(final_a))&set(pair_lists(final_b)),'final panels overlap')
    check_matched(final_a+final_b,{p['parent_id']:p['sequence'] for p in natives+controls},{p['parent_id']:p['sequence'] for p in rggs})
    # Validate exact unique ID-to-sequence membership for full panels AND NEW submission FASTAs.
    fasta_records(final_a,70); fasta_records(final_b,32)
    records_a=fasta_records(new_a,31)  # Seven WT + fourteen shuffled TADs + five WT and five existing shuffled RGGs.
    records_b=fasta_records(new_b,27)  # Seven WT controls + twenty additional WT RGGs.
    for key,value in input_hashes.items():
        check(sha256(Path(key))==value,f'input changed during preparation: {key}')
    filenames=['final_dataset_panel_400.txt','dataset_reused_pairs_260.txt','dataset_new_pairs_to_run_140.txt',
        'final_dataset_panel_manifest_400.csv','dataset_new_sequences_to_run.fasta','additional_dataset_TAD_parents_7.csv',
        'additional_dataset_TAD_shuffles_14.csv','final_nondataset_WT_panel_175.txt','nondataset_reused_pairs_35.txt',
        'nondataset_new_pairs_to_run_140.txt','final_nondataset_WT_manifest_175.csv','nondataset_new_sequences_to_run.fasta',
        'additional_nondataset_RGGs_20.csv','FINAL_CONTROL_PANEL_REPORT.md','final_reuse_control_selection_audit.json']
    protected={Path(p).resolve() for p in input_hashes}|{Path(__file__).resolve()}
    check(not {(out/name).resolve() for name in filenames}&protected,'output would overwrite input or script')
    out.mkdir(parents=True,exist_ok=True)
    for filename,rows in [(filenames[0],final_a),(filenames[1],reused_a),(filenames[2],new_a),
                          (filenames[7],final_b),(filenames[8],reused_b),(filenames[9],new_b)]:
        (out/filename).write_text(''.join(n+'\n' for n in pair_lists(rows)))
    for filename,rows in [(filenames[3],final_a),(filenames[10],final_b)]:
        pd.DataFrame(rows).to_csv(out/filename,index=False)
    for filename,records in [(filenames[4],records_a),(filenames[11],records_b)]:
        (out/filename).write_text(''.join(f'>{vid}\n{seq}\n' for vid,seq in sorted(records.items())))
        headers=[line[1:] for line in (out/filename).read_text().splitlines() if line.startswith('>')]
        check(len(headers)==len(set(headers))==len(records) and set(headers)==set(records),'serialized submission FASTA membership failed')
    for filename,rows in [(filenames[5],additional_tads),(filenames[6],new_tad_shuffles),(filenames[12],additional_rggs)]:
        pd.DataFrame([dict(row,**descriptors(row['sequence'])) for row in rows]).to_csv(out/filename,index=False)
    for list_file,manifest_file,rows in [(filenames[0],filenames[3],final_a),(filenames[7],filenames[10],final_b)]:
        check((out/list_file).read_text().splitlines()==pd.read_csv(out/manifest_file).pair_name.tolist()==pair_lists(rows),'serialized panel/manifest mismatch')
    check(not set((out/filenames[2]).read_text().splitlines())&old_pairs and not set((out/filenames[9]).read_text().splitlines())&old_pairs,'serialized NEW submission list overlaps old pairs')
    report=('OLD SIMULATIONS\nTotal old pairs: 400\nReusable dataset-control pairs: 260\nReusable non-dataset WT pairs: 35\nOld pairs not used in corrected design: 105\n\n'
        'FINAL DATASET-SHUFFLE PANEL\nExisting dataset TAD parents retained: 13\nAdditional dataset TAD parents selected: 7\nTotal dataset TAD parents: 20\nRGG parents retained: 5\nExisting simulations reused: 260\nNew simulations to run: 140\nFINAL PANEL TOTAL: 400\n\n'
        'FINAL NON-DATASET WT PANEL\nNon-dataset TAD controls: 7\nExisting RGG parents retained: 5\nAdditional RGG parents selected: 20\nFinal RGG parents: 25\nExisting simulations reused: 35\nNew simulations to run: 140\nFINAL PANEL TOTAL: 175\n\n'
        'TOTAL NEW SIMULATIONS ACTUALLY REQUIRED: 280\nNO MODEL FITTING OR B22-BASED SELECTION PERFORMED')
    explanation=('Old pairs were retained solely by sequence identity, dataset membership, and their predefined perturbation condition. No B22, prediction, OCN, simulation outcome, charge, hydropathy, length or other ranking was used. '
        'Only dataset_new_pairs_to_run_140.txt and nondataset_new_pairs_to_run_140.txt are new simulation submission lists. Full panel lists also contain already-run pairs and must not be submitted wholesale. '
        'No non-dataset TAD was shuffled, no RGG paired with a non-dataset TAD was shuffled, and no RGG shuffle was newly generated. '
        'The original simulation_pair_names_400.txt and all biological source inputs are preserved. No simulations, B22 calculations or model fitting were performed.')
    (out/filenames[13]).write_text('# Final reuse-based control panels\n\n'+report+'\n\n'+explanation+'\n')
    audit=dict(seed=args.seed,source_paths={k:str(v) for k,v in paths.items()},source_sha256=input_hashes,
        old_pair_count=400,old_valid_dataset_pairs=260,old_valid_nondataset_WT_pairs=35,old_not_used_pairs=105,
        old_classification=classified,old_unused_pair_names=unused,existing_13_dataset_TADs=existing_tads,
        additional_7_dataset_TADs=additional_tads,five_existing_RGGs=existing_rggs,existing_five_RGG_shuffle_variants=existing_rg_shuffles,
        selected_14_new_TAD_shuffle_variants=new_tad_shuffles,additional_20_nondataset_RGGs=additional_rggs,
        native_eligible_pool_count=len(natives),additional_dataset_parents_eligible_after_status_checks=len(viable_tads),
        additional_nondataset_proteins_eligible=len(eligible_proteins),rejected_candidates=rejected,
        rejection_method='preselection complete-block eligibility filtering by exact prior-run status only; uniform sampling among viable candidates',
        candidate_origins=candidate_origins,historical_TAD_A_B_label_rules=ab_rules,
        seeds=dict(additional_dataset_parent_selection=native_seed,new_TAD_shuffle_selection=variant_seeds,
                   additional_nondataset_protein_selection=protein_seed,additional_nondataset_region_selection=region_seeds),
        final_counts=dict(dataset_reused=260,dataset_new=140,dataset_total=400,nondataset_reused=35,nondataset_new=140,nondataset_total=175,total_new=280),
        output_sha256={str(out/name):sha256(out/name) for name in filenames[:-1]},
        python_version=platform.python_version(),pandas_version=pd.__version__,numpy_version=np.__version__,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),scientific_rule=explanation)
    (out/filenames[14]).write_text(json.dumps(audit,indent=2)+'\n')
    check(sha256(paths['exclusion'])==input_hashes[str(paths['exclusion'])],'authoritative old list changed')
    print(f'External workbook used: {paths["external"]}\nBasic workbook used: {paths["basic"]}\n')
    print(report)
    print(f'\nNew dataset submission list: {out/filenames[2]}\nDataset submission FASTA: {out/filenames[4]}')
    print(f'New non-dataset submission list: {out/filenames[9]}\nNon-dataset submission FASTA: {out/filenames[11]}')


if __name__=='__main__':
    main()
