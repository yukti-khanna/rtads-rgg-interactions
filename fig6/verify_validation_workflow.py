#!/usr/bin/env python3
"""Local numerical and output checks. Prepared but not executed by Codex.

python verify_validation_workflow.py           # numerical tests
python verify_validation_workflow.py --outputs # numerical tests plus generated tables
"""
import argparse
import json
import unittest
import numpy as np
import pandas as pd
from workflow_common import ROOT, OUT, pkd, log_negative
from calculate_validated_b22 import legacy_calculation, integrated_calculation, summarize, validate_curve
from compare_old_new_predictions import concordance


class NumericalChecks(unittest.TestCase):
    def test_ideal_rdf_and_legacy_tail_span(self):
        r = np.arange(.05, 10, .1)
        g = np.ones(len(r))
        validate_curve(r, g, 20, 7.5, 2)
        raw, corrected, c = integrated_calculation(r, g, 7.5)
        self.assertAlmostEqual(c, 1)
        self.assertAlmostEqual(raw, 0)
        self.assertAlmostEqual(corrected, 0)
        raw, legacy, clegacy = legacy_calculation(r, g)
        self.assertAlmostEqual(raw, 0)
        self.assertAlmostEqual(clegacy, .95, places=10)
        # Legacy artifact is intentional reproduction, not physical attraction.
        self.assertLess(legacy, 0)

    def test_uniform_background_cancels(self):
        r = np.arange(.05, 10, .1)
        g = np.full(len(r), 1.3)
        raw, corrected, c = integrated_calculation(r, g, 10)
        expected = -2 * np.pi * .3 * 10**3 / 3
        self.assertAlmostEqual(c, 1.3)
        self.assertAlmostEqual(corrected, 0, places=9)
        self.assertLess(abs(raw - expected) / abs(expected), 1e-4)

    def test_analytic_radial_integral_and_refinement(self):
        # A smooth excess entirely inside the tail: polynomial integrates exactly
        # to amplitude*a^3*8/105 before the -2*pi factor.
        a, amplitude, cutoff = 3., 2., 7.5
        expected = -2 * np.pi * amplitude * a**3 * 8 / 105
        errors = []
        for step in [.1, .025]:
            r = np.arange(step / 2, 10, step)
            g = 1 + amplitude * np.where(r < a, (1 - (r / a)**2)**2, 0)
            _, b, c = integrated_calculation(r, g, cutoff)
            self.assertAlmostEqual(c, 1)
            errors.append(abs(b - expected))
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[1] / abs(expected), 1e-4)

    def test_spacing_factor_is_not_optional(self):
        r = np.arange(0, 7.5 + .05, .1)
        g = 1 + np.exp(-r**2)
        raw = legacy_calculation(r, g)[0]
        physical_raw = integrated_calculation(r, g, 7.5)[0]
        # Endpoints have negligible integrand for this curve; sum*dr matches integral.
        self.assertAlmostEqual(raw * .1, physical_raw, places=9)

    def test_replica_mean_before_log_and_uncertainty(self):
        rep = pd.DataFrame([dict(pair_id='a__b', method='legacy_sum', integration_boundary_nm=7.5,
            tail_width_nm=2, units='legacy_sum_scale_no_dr', raw_B22=v, corrected_B22=v,
            averaged_only=False, rdf_file='example.pkl', rdf_sha256='example', endpoint_convention='masked_bin_centres')
            for v in [-10., -100., 20.]])
        row = summarize(rep).iloc[0]
        self.assertEqual(row.n_replicas, 3)
        self.assertAlmostEqual(row.corrected_B22_mean, -30)
        self.assertAlmostEqual(row.log10_neg_mean_B22, np.log10(30))
        self.assertAlmostEqual(row.corrected_B22_SD, np.std([-10, -100, 20], ddof=1))
        self.assertAlmostEqual(row.corrected_B22_SEM, row.corrected_B22_SD / np.sqrt(3))
        rep = rep.iloc[:1].copy()
        rep['averaged_only'] = True
        row = summarize(rep).iloc[0]
        self.assertEqual(row.n_replicas, 0)
        self.assertTrue(np.isnan(row.corrected_B22_SD))

    def test_nonnegative_values_keep_missing_log(self):
        values = log_negative([-100, 0, 20, np.nan])
        self.assertEqual(values[0], 2)
        self.assertTrue(np.isnan(values[1:]).all())

    def test_pkd_units_and_ties(self):
        np.testing.assert_allclose(pkd([1, 100, 1000]), [6, 4, 3])
        self.assertTrue(np.isnan(pkd([0, -1])).all())
        c = concordance(np.array([1, 1, 2]), np.array([1, 2, 2]))
        self.assertEqual(c['pKd_ties'], 1)
        self.assertEqual(c['prediction_ties'], 1)
        self.assertEqual(c['ranking_concordance'], .75)

    def test_invalid_radial_data_rejected(self):
        for r, g in [(np.array([0, .2, .1]), np.ones(3)),
                     (np.array([0, .1, .2]), np.array([1, np.nan, 1]))]:
            with self.assertRaises(ValueError):
                validate_curve(r, g, 20, 7.5, 2)


def verify_outputs():
    old = pd.read_csv(ROOT / 'Table_S8.csv')
    new = pd.read_csv(OUT / 'experimental_validation_updated.csv')
    pd.testing.assert_frame_equal(old, new[old.columns])
    pred = pd.read_csv(ROOT / 'preds_val_updated/predictions_minimal.csv').set_index('pair_id')
    inv = pd.read_csv(OUT / 'complete_pair_inventory.csv').set_index('pair_id')
    assert len(inv) == len(pred) and set(inv.index) == set(pred.index)
    np.testing.assert_allclose(inv.loc[pred.index, 'New_final_log10_negB22'], pred.final_hybrid_prediction)
    np.testing.assert_allclose(new.pKd, pkd(new.Kd_uM), equal_nan=True)
    assert inv.loc[~inv.experiment_present, 'Outcome'].eq('not tested').all()
    matched = pd.read_csv(ROOT / 'prediction_diagnostics/matched_pairs.csv')
    expected = new[(new.Outcome == 'Kd_fit') & new.Kd_uM.gt(0) & np.isfinite(new.Pred_log10_negB22_hybrid) & np.isfinite(new.New_final_log10_negB22)]
    assert set(matched.pair_id) == set(expected.pair_id)
    if 'g3bp1_rgg__foxo4_cr3' in set(expected.pair_id):
        assert 'g3bp1_rgg__foxo4_cr3' in set(matched.pair_id)
    rep = pd.read_csv(OUT / 'b22/per_replica_b22.csv')
    pair = pd.read_csv(OUT / 'b22/per_pair_b22.csv')
    reconstructed = summarize(rep).sort_values(['pair_id', 'method', 'integration_boundary_nm']).reset_index(drop=True)
    stored = pair.sort_values(['pair_id', 'method', 'integration_boundary_nm']).reset_index(drop=True)
    for col in ['raw_B22_mean', 'corrected_B22_mean', 'corrected_B22_SD', 'corrected_B22_SEM', 'log10_neg_mean_B22']:
        np.testing.assert_allclose(reconstructed[col], stored[col], equal_nan=True)
    assert stored.loc[stored.corrected_B22_mean.ge(0), 'log10_neg_mean_B22'].isna().all()
    if (OUT / 'b22/numerical_convention_qc.json').exists():
        qc = json.loads((OUT / 'b22/numerical_convention_qc.json').read_text())
        assert qc['failed_files'] == 0, 'Some RDF files failed'
    (OUT / 'qc/verification_status.txt').write_text('Numerical tests and generated table checks passed. Visual panel inspection remains required.\n')
    print('Generated table checks passed; visual panel inspection remains required')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outputs', action='store_true')
    args = ap.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NumericalChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.outputs:
        verify_outputs()
