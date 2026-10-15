#!/usr/bin/env python3
"""Build the validation master and figures from existing calculated sources.

This workflow never recalculates RDF/B22, recovers a model, or trains a model.
Experimental files replace previous annotations rather than accumulating them.
"""
import argparse
import json
import sys
import pandas as pd
from validation_master import ROOT, MasterBuilder, load_config, path


def configure(args):
    config = load_config(args.config)
    exp = config['experiments']
    if args.experiments:
        exp['path'] = str(path(args.experiments))
        # Only standardized unit-bearing headings are recognized automatically.
        # Arbitrary Pair/KD headings require explicit column and unit options.
        source_exists = path(args.experiments).exists()
        headings = pd.read_csv(path(args.experiments), nrows=0).columns if source_exists else pd.Index([])
        if 'pair_id' in headings:
            exp['columns']['pair_id'] = 'pair_id'
        if 'Kd_uM' in headings:
            exp['columns']['Kd'] = 'Kd_uM'
            exp['kd_units'] = 'uM'
        elif source_exists and args.kd_column:
            if not args.kd_units:
                raise ValueError('A replacement Kd heading without units requires --kd-units explicitly')
        elif source_exists and exp['columns'].get('Kd') in headings and exp.get('kd_units'):
            # A deliberately saved nonstandard mapping and its explicit units
            # are reusable; users need not repeat the same CLI options each run.
            pass
        elif source_exists and any(str(c).lower().startswith('kd') and 'err' not in str(c).lower() for c in headings):
            raise ValueError('Replacement file has a nonstandard Kd heading; specify --kd-column and --kd-units explicitly')
        elif source_exists:
            exp['columns']['Kd'] = None
        exp['columns']['Kd_error'] = 'Kd_error_uM' if 'Kd_error_uM' in headings else ('Kd_uM_err' if 'Kd_uM_err' in headings else (exp['columns'].get('Kd_error') if exp['columns'].get('Kd_error') in headings else None))
        if 'Kd_error_uM' in headings or 'Kd_uM_err' in headings:
            exp['kd_error_units'] = 'uM'
        elif args.kd_error_column:
            exp['kd_error_units'] = args.kd_error_units or args.kd_units or exp['kd_units']
        for c in ['Outcome', 'Notes']:
            exp['columns'][c] = c if c in headings else (exp['columns'].get(c) if exp['columns'].get(c) in headings else None)
        # A prior selection based on source row numbers cannot carry across files.
        exp['select_measurements'] = {}
        if args.selection_config:
            exp['select_measurements'] = json.loads(path(args.selection_config).read_text())
    elif args.selection_config:
        exp['select_measurements'] = json.loads(path(args.selection_config).read_text())
    for key, attr in [('pair_id', 'pair_column'), ('Kd', 'kd_column'), ('Kd_error', 'kd_error_column'), ('Outcome', 'outcome_column'), ('Notes', 'notes_column')]:
        value = getattr(args, attr)
        if value:
            exp['columns'][key] = value
    if args.kd_units:
        exp['kd_units'] = args.kd_units
    if args.kd_error_units:
        exp['kd_error_units'] = args.kd_error_units
    if args.uncertainty_type:
        exp['uncertainty_type'] = args.uncertainty_type
    elif args.experiments:
        exp['uncertainty_type'] = 'reported error; type unspecified in replacement input'
    if args.measurement_id_column:
        exp['measurement_id_column'] = args.measurement_id_column
    for role in ['predictions', 'simulations', 'features']:
        if getattr(args, role):
            config[role]['path'] = str(path(getattr(args, role)))
    if args.feature_row_ids:
        config['features']['row_ids_path'] = str(path(args.feature_row_ids))
    if args.feature_manifest:
        config['features']['manifest_path'] = str(path(args.feature_manifest))
    if args.outdir:
        config['outdir'] = args.outdir
    if args.diagnostics_dir:
        config['diagnostics_dir'] = args.diagnostics_dir
    if args.bootstrap is not None:
        config['statistics']['bootstrap'] = args.bootstrap
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiments', help='Replacement experimental CSV; defaults to selected current Table_S8.csv')
    for role in ['predictions', 'simulations', 'features']:
        ap.add_argument('--' + role, help='Override source path; column definitions remain in configuration')
    ap.add_argument('--config', default='validation_sources.json')
    ap.add_argument('--outdir', help='Output directory inside fig6')
    ap.add_argument('--diagnostics-dir', help='Optional isolated diagnostics directory inside fig6')
    for option in ['pair-column', 'kd-column', 'kd-error-column', 'outcome-column', 'notes-column', 'measurement-id-column']:
        ap.add_argument('--' + option)
    ap.add_argument('--kd-units', choices=['M', 'mM', 'uM', 'nM'])
    ap.add_argument('--kd-error-units', choices=['M', 'mM', 'uM', 'nM'], help='Optional distinct units for the reported error')
    ap.add_argument('--uncertainty-type', help='Reported interpretation, e.g. SD, SEM, CI half-width, or unspecified')
    ap.add_argument('--selection-config', help='JSON mapping pair_id to one measurement ID (or 1-based input row)')
    ap.add_argument('--feature-row-ids', help='Explicit mapping for a replacement anonymous feature matrix')
    ap.add_argument('--feature-manifest', help='Matching feature manifest for a replacement anonymous matrix')
    ap.add_argument('--bootstrap', type=int)
    ap.add_argument('--master-only', action='store_true', help='Build CSV/QC without plotting')
    ap.add_argument('--paper-only', action='store_true', help='Skip expanded old/new diagnostic figure regeneration')
    ap.add_argument('--verify', action='store_true', help='Run isolated reusable-update fixtures before writing scientific outputs')
    ap.add_argument('--skip-b22', action='store_true', help=argparse.SUPPRESS)
    args = ap.parse_args()
    config = configure(args)
    settings = config['statistics']
    if settings['bootstrap'] < 1 or settings['permutations'] < 1 or settings['exact_spearman_max_n'] > 9:
        raise ValueError('Use positive resampling counts and exact_spearman_max_n <= 9')
    if args.verify:
        import unittest
        from verify_master_workflow import ReusableWorkflowChecks
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReusableWorkflowChecks))
        if not result.wasSuccessful():
            raise SystemExit('Reusable workflow fixture checks failed; scientific outputs were not regenerated')
    builder = MasterBuilder(config)
    master = builder.build()
    if not args.master_only:
        from plot_master_validation import MasterFigures
        MasterFigures(master, config, config['outdir']).run(diagnostics=not args.paper_only)
    from write_master_report import write_report
    write_report(master, builder.out, figures_generated=not args.master_only and not args.paper_only)
    summary = json.loads((builder.out / 'master_summary.json').read_text())
    print(json.dumps(summary, indent=2))
    print(f'Master CSV: {builder.out / "validation_master.csv"}')
    if not args.master_only:
        print(f'Paper panels: {builder.out / "paper_figures"}')
    print('No RDF/B22 calculation, historical recovery, or model fitting was performed.')


if __name__ == '__main__':
    main()
