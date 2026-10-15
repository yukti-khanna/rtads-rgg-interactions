#!/usr/bin/env python3
"""Identify sequence pairs containing histidine.

Input pair format: sequence_id_1__sequence_id_2
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd


def read_fasta(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    header = None
    parts: list[str] = []
    with path.open() as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    if header in records:
                        raise ValueError(f'Duplicate FASTA header: {header}')
                    records[header] = ''.join(parts).upper()
                header = line[1:].split()[0]
                if not header:
                    raise ValueError(f'Empty FASTA header at line {line_number}')
                parts = []
            else:
                if header is None:
                    raise ValueError(f'Sequence before FASTA header at line {line_number}')
                parts.append(line)
    if header is not None:
        if header in records:
            raise ValueError(f'Duplicate FASTA header: {header}')
        records[header] = ''.join(parts).upper()
    if not records:
        raise ValueError(f'No FASTA records found in {path}')
    return records


def parse_pair(line: str, line_number: int) -> tuple[str, str]:
    fields = line.split('__')
    if len(fields) != 2 or not all(fields):
        raise ValueError(
            f'Invalid pair at line {line_number}: {line!r}. '
            "Expected 'sequence_id_1__sequence_id_2'."
        )
    return fields[0], fields[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Extract all sequence pairs containing at least one histidine.'
    )
    parser.add_argument('--fasta', required=True, type=Path)
    parser.add_argument('--pairs', required=True, type=Path)
    parser.add_argument('--outdir', type=Path, default=Path('histidine_pair_analysis'))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sequences = read_fasta(args.fasta)

    seq_rows = []
    for seq_id, seq in sequences.items():
        h_count = seq.count('H')
        seq_rows.append({
            'sequence_id': seq_id,
            'length': len(seq),
            'histidine_count': h_count,
            'has_histidine': h_count > 0,
            'sequence': seq,
        })
    pd.DataFrame(seq_rows).sort_values('sequence_id').to_csv(
        args.outdir / 'sequence_histidine_counts.tsv', sep='\t', index=False
    )

    pair_rows = []
    missing_ids: set[str] = set()
    with args.pairs.open() as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            first_id, second_id = parse_pair(line, line_number)
            if first_id not in sequences:
                missing_ids.add(first_id)
                continue
            if second_id not in sequences:
                missing_ids.add(second_id)
                continue
            h1 = sequences[first_id].count('H')
            h2 = sequences[second_id].count('H')
            if h1 and h2:
                category = 'both'
            elif h1:
                category = 'first_only'
            elif h2:
                category = 'second_only'
            else:
                category = 'neither'
            pair_rows.append({
                'pair': line,
                'first_id': first_id,
                'second_id': second_id,
                'first_histidine_count': h1,
                'second_histidine_count': h2,
                'total_histidine_count': h1 + h2,
                'category': category,
                'has_any_histidine': bool(h1 or h2),
            })

    if missing_ids:
        raise KeyError('IDs missing from FASTA:\n' + '\n'.join(sorted(missing_ids)))

    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        raise ValueError('No valid pairs found.')

    pair_df.to_csv(args.outdir / 'pair_histidine_summary.tsv', sep='\t', index=False)

    outputs = {
        'pairs_with_at_least_one_histidine.txt': pair_df['has_any_histidine'],
        'pairs_with_histidine_in_both.txt': pair_df['category'].eq('both'),
        'pairs_without_histidine.txt': pair_df['category'].eq('neither'),
    }
    for filename, mask in outputs.items():
        with (args.outdir / filename).open('w') as out:
            for pair in pair_df.loc[mask, 'pair']:
                out.write(pair + '\n')

    counts = pair_df['category'].value_counts()
    total = len(pair_df)
    any_h = int(pair_df['has_any_histidine'].sum())
    summary = [
        ('total_pairs', total),
        ('pairs_with_any_histidine', any_h),
        ('fraction_with_any_histidine', f'{any_h / total:.6f}'),
        ('first_only', int(counts.get('first_only', 0))),
        ('second_only', int(counts.get('second_only', 0))),
        ('both', int(counts.get('both', 0))),
        ('neither', int(counts.get('neither', 0))),
    ]
    with (args.outdir / 'histidine_pair_counts.txt').open('w') as out:
        for key, value in summary:
            out.write(f'{key}\t{value}\n')

    for key, value in summary:
        print(f'{key}\t{value}')
    print(f'Outputs written to: {args.outdir.resolve()}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
