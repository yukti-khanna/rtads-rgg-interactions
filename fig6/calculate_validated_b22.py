#!/usr/bin/env python3
"""Calculate replica-resolved B22 with explicit legacy and physical conventions.

python calculate_validated_b22.py --help
Default inputs/outputs are relative to this fig6 folder. Only trusted local
pickles should be loaded. The 7.5 nm legacy sum is not labelled physical nm^3.
Integrated 7.5 nm and L/2 values are separate and never substituted into the
frozen predictor comparison. No association constant or Kd is calculated.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from workflow_common import ROOT, inside, rdf_inventory, sha256, log_negative, write_json


def trapezoidal_integral(y, x):
    """Same trapezoidal rule on NumPy 1.x (trapz) and 2.x (trapezoid)."""
    integrate = getattr(np, 'trapezoid', None)
    if integrate is None:
        integrate = np.trapz
    return integrate(y, x)


class Payload:
    """Storage-only compatibility for historical RDFResult dataclasses."""
    pass


class RDFUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'RDFResult':
            return Payload
        return super().find_class(module, name)


def load_curves(path, filename_box):
    with Path(path).open('rb') as handle:
        loaded = RDFUnpickler(handle).load()
    if isinstance(loaded, pd.DataFrame):
        if loaded.columns.duplicated().any() or len(loaded) != 2:
            raise ValueError('Expected unique run columns and two RDF rows')
        if not all(str(c).startswith('run_') for c in loaded.columns):
            raise ValueError('Unrecognized DataFrame columns; do not assume averaged curves are replicas')
        return [(str(c), np.asarray(loaded[c].iloc[0], float),
                 np.asarray(loaded[c].iloc[1], float), False) for c in loaded], filename_box, 'filename_20nm_metadata'
    d = loaded if isinstance(loaded, dict) else vars(loaded)
    box = float(d.get('box_size_nm', d.get('box_size', filename_box)))
    if not np.isclose(box, filename_box):
        raise ValueError('Embedded box metadata disagrees with filename')
    box_source = 'embedded_metadata' if 'box_size_nm' in d or 'box_size' in d else 'filename_metadata'
    if d.get('replicate_rdfs'):
        rows = []
        for rid, curve in d['replicate_rdfs'].items():
            if not isinstance(curve, dict):
                raise ValueError('Expected replicate_rdfs[id] with radial/RDF arrays')
            r = curve.get('r_nm', curve.get('r', d.get('r_nm')))
            g = curve.get('rdf', curve.get('g_r', curve.get('rdf_mean')))
            if r is None or g is None:
                raise ValueError('Replica has no recognized r or g array')
            rows.append((str(rid), np.asarray(r, float), np.asarray(g, float), False))
        return rows, box, box_source
    if 'r_nm' in d and 'rdf_mean' in d:
        return [('averaged_rdf', np.asarray(d['r_nm'], float), np.asarray(d['rdf_mean'], float), True)], box, box_source
    raise ValueError(f'Unsupported pickle structure: {type(loaded).__name__}, keys {list(d)}')


def validate_curve(r, g, box, cutoff, delta):
    if r.ndim != 1 or g.ndim != 1 or len(r) != len(g) or len(r) < 3:
        raise ValueError('RDF/r arrays must be equal one-dimensional arrays with >=3 bins')
    if not np.isfinite(r).all() or not np.isfinite(g).all():
        raise ValueError('Nonfinite radial/RDF data')
    if (r < 0).any() or (g < 0).any() or not (np.diff(r) > 0).all():
        raise ValueError('Negative radius/RDF or nonmonotonic coordinates')
    if not np.isfinite(box) or box <= 0 or cutoff > box / 2 + 1e-6 or cutoff <= delta or delta <= 0:
        raise ValueError('Invalid box, integration cutoff, or tail width')
    spacing = np.diff(r)
    dr = float(np.median(spacing))
    uniform = bool(np.allclose(spacing, dr, rtol=1e-4, atol=1e-6))
    # Centre-of-bin RDFs reach half a bin beyond the first/last sampled radius.
    upper_edge = r[-1] + spacing[-1] / 2
    if upper_edge < cutoff - 1e-5 or r[0] - spacing[0] / 2 > 1e-5:
        raise ValueError('RDF bin edges do not cover 0..cutoff')
    tail = (r >= cutoff - delta) & (r <= cutoff)
    if tail.sum() < 2 or r[0] > cutoff - delta:
        raise ValueError('Insufficient tail-window coverage')
    return dict(radial_spacing_nm=dr, radial_spacing_min_nm=float(spacing.min()),
                radial_spacing_max_nm=float(spacing.max()), uniform_spacing=uniform,
                expected_0_1nm_spacing=bool(np.isclose(dr, .1, atol=1e-5)),
                radial_min_nm=float(r[0]), radial_max_nm=float(r[-1]),
                radial_outer_edge_nm=float(upper_edge), tail_bin_count=int(tail.sum()),
                legacy_tail_sample_span_nm=float(np.ptp(r[tail])))


def legacy_calculation(r, g, cutoff=7.5, delta=2):
    """Exact repository convention, including mask-only tail integration / delta."""
    tail = (r >= cutoff - delta) & (r <= cutoff)
    c = trapezoidal_integral(g[tail], r[tail]) / delta
    m = r <= cutoff
    raw = -2 * np.pi * np.sum((g[m] - 1) * r[m] ** 2)
    corrected = -2 * np.pi * np.sum((g[m] - c) * r[m] ** 2)
    return float(raw), float(corrected), float(c)


def window_coordinates(r, g, lower, upper):
    """Interpolate endpoints; constant extrapolation only across a half-bin edge.

    validate_curve guarantees coverage. We do not extend beyond available bins.
    At r=0 the r^2 integrand vanishes; at L/2 the last-bin value supplies the
    missing half-bin. This explicit endpoint convention is not legacy code.
    """
    m = (r > lower) & (r < upper)
    x = np.r_[lower, r[m], upper]
    y = np.interp(x, r, g)
    return x, y


def integrated_calculation(r, g, cutoff, delta=2):
    tail_r, tail_g = window_coordinates(r, g, cutoff - delta, cutoff)
    c = trapezoidal_integral(tail_g, tail_r) / delta
    x, y = window_coordinates(r, g, 0, cutoff)
    raw = -2 * np.pi * trapezoidal_integral((y - 1) * x ** 2, x)
    corrected = -2 * np.pi * trapezoidal_integral((y - c) * x ** 2, x)
    return float(raw), float(corrected), float(c)


def summarize(rep):
    rows = []
    for (pair, method, cutoff), d in rep.groupby(['pair_id', 'method', 'integration_boundary_nm'], sort=False):
        vals = d.corrected_B22.to_numpy(float)
        averaged = bool(d.averaged_only.any())
        count = 0 if averaged else len(vals)
        sd = float(np.std(vals, ddof=1)) if count >= 2 else np.nan
        mean = float(np.mean(vals))
        reason = '' if mean < 0 else ('zero_mean_B22' if mean == 0 else 'positive_mean_B22')
        rows.append(dict(pair_id=pair, method=method, integration_boundary_nm=cutoff,
                         tail_width_nm=float(d.tail_width_nm.iloc[0]), units=d.units.iloc[0],
                         raw_B22_mean=float(d.raw_B22.mean()), corrected_B22_mean=mean,
                         corrected_B22_SD=sd, corrected_B22_SEM=sd / np.sqrt(count) if count >= 2 else np.nan,
                         n_replicas=count, n_rdf_curves=len(vals), averaged_only=averaged,
                         uncertainty_status='unavailable_averaged_RDF' if averaged else ('available' if count >= 2 else 'unavailable_single_replica'),
                         log10_neg_mean_B22=float(log_negative([mean])[0]), transform_missing_reason=reason,
                         rdf_file=d.rdf_file.iloc[0], rdf_sha256=d.rdf_sha256.iloc[0],
                         correction='primary_tail_offset', endpoint_convention=d.endpoint_convention.iloc[0]))
    return pd.DataFrame(rows)


def compare_historical(summary, out, cutoff, rtol, atol):
    legacy = summary[summary.method.eq('legacy_sum') & np.isclose(summary.integration_boundary_nm, cutoff)].copy()
    hist = pd.read_csv(ROOT / 'validated_final_features_data_pred_b22s.csv')
    hist = hist.rename(columns={'name': 'pair_id'})
    comparison = hist.merge(legacy, on='pair_id', how='left', validate='one_to_one')
    comparison['corrected_difference'] = comparison.corrected_B22_mean - comparison.Corrected_B22_Hummer
    comparison['raw_difference'] = comparison.raw_B22_mean - comparison.B22
    comparison['reproduces_historical_corrected'] = np.isclose(comparison.corrected_B22_mean, comparison.Corrected_B22_Hummer, rtol=rtol, atol=atol)
    comparison['reproduces_historical_raw'] = np.isclose(comparison.raw_B22_mean, comparison.B22, rtol=rtol, atol=atol)
    comparison['cutoff_status'] = 'candidate historical 7.5 nm; numerical reproduction required'
    comparison['construct_status'] = 'historical feature roles/lengths available; RDF sequence provenance not embedded'
    comparison['discrepancy_followup'] = np.where(comparison.reproduces_historical_corrected & comparison.reproduces_historical_raw,
        'legacy correction/cutoff/replica averaging reproduced; does not independently verify sequences',
        'inspect RDF version, exact sequences, cutoff, correction, bin spacing and replica selection')
    comparison.to_csv(out / 'historical_reproduction.csv', index=False)
    t8 = pd.read_csv(ROOT / 'Table_S8.csv')
    t = t8[['pair_id', 'Sim_B22_raw', 'Sim_B22_corr_Hummer']].merge(legacy, on='pair_id', how='left', validate='one_to_one')
    t['reproduces_panel_raw'] = np.isclose(t.raw_B22_mean, t.Sim_B22_raw, rtol=rtol, atol=atol)
    t['reproduces_panel_corrected'] = np.isclose(t.corrected_B22_mean, t.Sim_B22_corr_Hummer, rtol=rtol, atol=atol)
    t.to_csv(out / 'historical_panel_reproduction.csv', index=False)
    older_comparisons = []
    for filename in ['validated_features_data', 'validated_pred_with_hybrid_021125', 'validated_pred_with_hybrid_021125_sel_cols']:
        mapped = ROOT / 'updated_validation/qc' / (filename + '_mapped.csv')
        if not mapped.exists():
            continue
        old_table = pd.read_csv(mapped)
        old_table = old_table[old_table.mapping_status.eq('both_role_identities_verified')]
        z = old_table.merge(legacy, on='pair_id', how='left', validate='one_to_one')
        z['source'] = filename
        z['corrected_difference'] = z.corrected_B22_mean - z.Corrected_B22_Hummer
        z['raw_difference'] = z.raw_B22_mean - z.B22
        z['interpretation'] = np.where(z.historical_length_status.eq('length_match_sequence_unverified'),
            'lengths match; exact sequences and numerical convention remain separate provenance questions',
            'construct length changed or pair missing; difference cannot be assigned to numerical convention alone')
        older_comparisons.append(z)
    if older_comparisons:
        pd.concat(older_comparisons, ignore_index=True).to_csv(out / 'older_construct_simulation_comparisons.csv', index=False)
    passed = (len(comparison) > 0 and comparison.reproduces_historical_corrected.all()
              and comparison.reproduces_historical_raw.all() and t.reproduces_panel_raw.all() and t.reproduces_panel_corrected.all())
    return dict(overlapping_pairs=int(comparison.corrected_B22_mean.notna().sum()),
                historical_pairs=len(hist), all_historical_values_reproduced=bool(passed),
                rtol=rtol, atol=atol, frozen_target_scale='legacy radial sum, corr1, not physical nm^3',
                simulation_prediction_comparison_allowed=bool(passed and np.isclose(cutoff, 7.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rdf-dir', default='validated_final_rdfs')
    ap.add_argument('--output-dir', default='updated_validation/b22')
    ap.add_argument('--legacy-cutoff', type=float, default=7.5)
    ap.add_argument('--tail-width', type=float, default=2)
    ap.add_argument('--analysis-repo', type=Path, default=None,
                    help='Read-only helper reference; defaults to inspected local repository')
    ap.add_argument('--reproduction-rtol', type=float, default=1e-6)
    ap.add_argument('--reproduction-atol', type=float, default=1e-4)
    args = ap.parse_args()
    rdfdir, out = inside(args.rdf_dir), inside(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inventory = rdf_inventory(rdfdir, ROOT / 'rdf_pair_aliases.csv')
    if inventory.empty:
        raise ValueError('No RDF files found')
    if inventory.pair_id.duplicated().any():
        raise ValueError('Multiple source RDFs resolve to one pair')
    inventory.to_csv(out / 'rdf_inventory.csv', index=False)
    rep, qc = [], []
    for _, item in inventory.iterrows():
        f = ROOT / item.rdf_file
        try:
            curves, box, box_source = load_curves(f, item.box_size_nm)
            if not curves:
                raise ValueError('Empty replica dictionary')
            for rid, r, g, averaged in curves:
                for method, cutoff in [('legacy_sum', args.legacy_cutoff), ('integrated_same_cutoff', args.legacy_cutoff), ('integrated_L_half', box / 2)]:
                    validation = validate_curve(r, g, box, cutoff, args.tail_width)
                    raw, corrected, offset = (legacy_calculation(r, g, cutoff, args.tail_width) if method == 'legacy_sum'
                                               else integrated_calculation(r, g, cutoff, args.tail_width))
                    rep.append(dict(pair_id=item.pair_id, rdf_pair_id=item.rdf_pair_id, replica_id=rid,
                                    averaged_only=averaged, method=method, integration_boundary_nm=cutoff,
                                    tail_width_nm=args.tail_width, tail_offset=offset,
                                    raw_B22=raw, corrected_B22=corrected,
                                    units='legacy_sum_scale_no_dr' if method == 'legacy_sum' else 'nm^3',
                                    correction='primary_tail_offset', box_size_nm=box, box_source=box_source,
                                    rdf_file=item.rdf_file, rdf_sha256=sha256(f),
                                    endpoint_convention='masked_bin_centres' if method == 'legacy_sum' else 'interpolated_endpoints_constant_half_bin_edges',
                                    **validation))
            qc.append(dict(pair_id=item.pair_id, rdf_file=item.rdf_file, status='passed', reason='', n_curves=len(curves)))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            # Never silently discard failed files or partial replicas.
            rep = [z for z in rep if z['pair_id'] != item.pair_id]
            qc.append(dict(pair_id=item.pair_id, rdf_file=item.rdf_file, status='failed', reason=str(exc), n_curves=0))
    qc = pd.DataFrame(qc)
    qc.to_csv(out / 'rdf_qc.csv', index=False)
    if not rep:
        raise ValueError('No valid RDF data; inspect rdf_qc.csv')
    rep = pd.DataFrame(rep)
    rep.to_csv(out / 'per_replica_b22.csv', index=False)
    summary = summarize(rep)
    summary.to_csv(out / 'per_pair_b22.csv', index=False)
    report = compare_historical(summary, out, args.legacy_cutoff, args.reproduction_rtol, args.reproduction_atol)
    # Verify each legacy replica against the exact local reference functions.
    config = json.loads((ROOT / 'inspection_provenance.json').read_text())
    repo = args.analysis_repo or Path(config['repositories'][0]['path'])
    helper_verified = None
    helper = repo / 'scripts/rdf_b22_functions.py'
    if helper.exists():
        import importlib.util
        sys.dont_write_bytecode = True  # reference repositories stay read-only
        spec = importlib.util.spec_from_file_location('read_only_b22_reference', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for _, item in inventory[inventory.pair_id.isin(qc[qc.status.eq('passed')].pair_id)].iterrows():
            curves, _, _ = load_curves(ROOT / item.rdf_file, item.box_size_nm)
            for rid, r, g, _ in curves:
                row = rep[(rep.pair_id == item.pair_id) & (rep.replica_id == rid) & (rep.method == 'legacy_sum')].iloc[0]
                np.testing.assert_allclose(row.raw_B22, module.calculate_b2(r, g, args.legacy_cutoff), rtol=1e-12)
                reference, _ = module.calculate_bij_hummer(r, g, args.legacy_cutoff, args.tail_width)
                np.testing.assert_allclose(row.corrected_B22, reference, rtol=1e-12)
        helper_verified = True
    report.update(rdf_files=len(inventory), failed_files=int(qc.status.eq('failed').sum()),
                  reference_helper_verified=helper_verified,
                  reference_helper=str(helper), reference_helper_sha256=sha256(helper) if helper.exists() else None,
                  legacy_cutoff_nm=args.legacy_cutoff, tail_width_nm=args.tail_width,
                  physical_calculation='Trapezoidal integration over radial coordinates; full 2 nm endpoint-interpolated tail; cutoff 7.5 or L/2',
                  primary_log_transform='log10(-mean corrected B22); not mean replica logs',
                  legacy_tail_bias='Masked centre samples can span 1.9 nm while divisor stays 2 nm. Preserved only in legacy outputs.',
                  target_manifest_provenance='Supplied run manifest is authoritative; full new training RDF provenance was not supplied.')
    if qc.status.eq('failed').any():
        report['simulation_prediction_comparison_allowed'] = False
    write_json(out / 'numerical_convention_qc.json', report)
    (out / 'CONVENTIONS.md').write_text(
        '# Numerical conventions\n\n'
        'Legacy: -2*pi*sum((g-c)*r^2), without dr; c=trapezoid(masked tail)/delta. '
        'This reproduces repository corr1/Hummer-labelled targets and preserves the shorter bin-centre tail span. '
        'It is not a physical nm^3 integral. Historical cutoff evidence is 7.5 nm, pending numerical reproduction.\n\n'
        'Integrated: -2*pi*trapezoid((g-c)*r^2,r), with exact window endpoints and constant extension over outer half bins. '
        'r is in nm, so these outputs are in nm^3. The full tail window avoids the legacy constant-RDF offset artifact. '
        '7.5 nm isolates numerical changes; L/2 additionally implements the manuscript cutoff. '
        'Neither calculation replaces the frozen target scale or generates a Kd.\n\n'
        'Five stored run columns are retained as five RDF replicas; independence follows their run labels and is not independently '
        'established from trajectories. A mean-only payload has n_replicas=0 (unknown), n_rdf_curves=1 and no uncertainty. '
        'Negative mean B22 is transformed after averaging. Positive/zero means are retained with a missing log.\n\n'
        'See historical_reproduction.csv and historical_panel_reproduction.csv for actual local-run agreement. '
        'The supplied frozen-model target scale is inferred from local historical code; exact new training provenance remains unresolved.\n')
    if qc.status.eq('failed').any():
        raise SystemExit('Some RDFs failed QC; outputs preserved, downstream simulation panels blocked')
    print(f'Wrote {len(rep)} replica/method rows and {len(summary)} pair/method rows to {out}')


if __name__ == '__main__':
    main()
