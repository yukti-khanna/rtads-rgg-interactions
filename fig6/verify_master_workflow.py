#!/usr/bin/env python3
"""Isolated synthetic fixtures; never write into scientific output directories."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from validation_master import ROOT, ESSENTIAL, MasterBuilder, csv_write
from validation_statistics import spearman_test


class ReusableWorkflowChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='.validation-fixture-', dir=ROOT)
        self.root = Path(self.temp.name)
        ids = ['rgg__a', 'rgg__b', 'rgg__c']
        self.ids = ids
        csv_write(pd.DataFrame(dict(pair_id=ids, final_hybrid_prediction=[4., 4.1, 4.2], base_prediction=[4., 4.1, 4.2], applied_correction=[0., 0., 0.], AHD_sequence=['AAA', 'CCC', 'DDD'], RGG_sequence=['RRR'] * 3)), self.root / 'pred.csv')
        csv_write(pd.DataFrame(dict(pair_id=ids, AHD_name=['a', 'b', 'c'], RGG_name=['rgg'] * 3, AHD_sequence=['AAA', 'CCC', 'DDD'], RGG_sequence=['RRR'] * 3)), self.root / 'seq.csv')
        # Deliberately reversed feature rows: joining by identity must preserve values.
        csv_write(pd.DataFrame(dict(pair_id=ids[::-1], test_feature=[30., 20., 10.])), self.root / 'features.csv')
        csv_write(pd.DataFrame(dict(pair_id=ids, b22=[-100., -200., 0.], sd=[10., 20., 1.], sem=[5., 10., .5], replicas=[4, 4, 4])), self.root / 'sim.csv')
        self.settings = dict(seed=42, permutations=199, bootstrap=50, exact_spearman_max_n=9, exact_pairwise_max_partitions=2000)
        self.config = dict(outdir=str(self.root / 'outputs'), experiments=dict(path=str(self.root / 'exp.csv'),
            columns=dict(pair_id='pair_id', Kd='Kd_uM', Kd_error='Kd_error_uM', Outcome='Outcome', Notes='Notes'), kd_units='uM', uncertainty_type='reported SD', select_measurements={}),
            predictions=dict(path=str(self.root / 'pred.csv'), pair_column='pair_id', final_column='final_hybrid_prediction', base_column='base_prediction', correction_column='applied_correction'),
            simulations=dict(path=str(self.root / 'sim.csv'), pair_column='pair_id', filters={}, b22_column='b22', log_column='absent', sd_column='sd', sem_column='sem', replicas_column='replicas', transformation='log10(-replica mean)'),
            features=dict(path=str(self.root / 'features.csv'), pair_column='pair_id', row_ids_path=None, feature_row_column='feature_row', manifest_path=None),
            sequence_registry=str(self.root / 'seq.csv'), additional_features=[], statistics=self.settings)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, rows):
        csv_write(pd.DataFrame(rows, columns=['pair_id', 'Kd_uM', 'Kd_error_uM', 'Outcome', 'Notes']), self.root / 'exp.csv')
        return MasterBuilder(self.config).build().set_index('pair_id')

    def test_new_annotation_and_unmatched_pair_survive(self):
        master = self.build([['rgg__b', 10., 2., 'Kd fit', 'new measurement'], ['unknown__new', 1., .1, 'Kd fit', 'new pair']])
        self.assertEqual(len(master), 4)
        self.assertEqual(master.loc['rgg__b', 'Pred_log10_negB22_final'], 4.1)
        self.assertEqual(master.loc['rgg__b', 'test_feature'], 20.)
        self.assertAlmostEqual(master.loc['rgg__b', 'pKd'], 5.)
        self.assertAlmostEqual(master.loc['rgg__b', 'pKd_error'], 2 / (10 * np.log(10)))
        self.assertTrue(np.isnan(master.loc['unknown__new', 'Pred_log10_negB22_final']))
        self.assertIn('missing simulation', master.loc['unknown__new', 'missing_data_reason'])
        self.assertEqual(list(master.reset_index().columns[:8]), ESSENTIAL)

    def test_conflicting_duplicates_are_not_selected(self):
        master = self.build([['rgg__a', 10., 1., 'Kd fit', 'condition A'], ['rgg__a', 50., 5., 'Kd fit', 'condition B']])
        self.assertTrue(np.isnan(master.loc['rgg__a', 'Kd_uM']))
        self.assertIn('selection required', master.loc['rgg__a', 'experimental_status'])
        support = pd.read_csv(self.root / 'outputs/experimental_measurements.csv')
        self.assertEqual(len(support), 2)
        self.assertFalse(support.selected_for_pair.any())

    def test_explicit_duplicate_selection(self):
        self.config['experiments']['select_measurements'] = {'rgg__a': 2}
        master = self.build([['rgg__a', 10., 1., 'Kd fit', 'A'], ['rgg__a', 50., 5., 'Kd fit', 'B']])
        self.assertEqual(master.loc['rgg__a', 'Kd_uM'], 50.)

    def test_replacement_removes_stale_annotations(self):
        self.build([['rgg__a', 10., 1., 'Kd fit', 'old']])
        master = self.build([['rgg__b', 20., 1., 'Kd fit', 'replacement']])
        self.assertTrue(np.isnan(master.loc['rgg__a', 'Kd_uM']))
        self.assertEqual(master.loc['rgg__a', 'experimental_status'], 'no experimental measurement supplied')

    def test_nonnegative_simulation_and_missing_are_distinct(self):
        master = self.build([['unknown__new', 1., .1, 'Kd fit', '']])
        self.assertEqual(master.loc['rgg__c', 'Sim_B22_final'], 0)
        self.assertTrue(np.isnan(master.loc['rgg__c', 'Sim_log10_negB22_final']))
        self.assertIn('available', master.loc['rgg__c', 'simulation_status'])
        self.assertFalse(master.loc['rgg__a', 'simulation_accuracy_eligible'])
        self.assertTrue(np.isfinite(master.loc['rgg__a', 'Sim_log10_negB22_final']))

    def test_anonymous_features_follow_explicit_row_map(self):
        csv_write(pd.DataFrame(dict(test_feature=[30., 10., 20.])), self.root / 'features.csv')
        csv_write(pd.DataFrame(dict(pair_id=['rgg__b', 'rgg__c', 'rgg__a'], feature_row=[2, 0, 1])), self.root / 'ids.csv')
        self.config['features']['row_ids_path'] = str(self.root / 'ids.csv')
        master = self.build([['rgg__a', 10., 1., 'Kd fit', '']])
        self.assertEqual(master.loc['rgg__a', 'test_feature'], 10.)
        self.assertEqual(master.loc['rgg__b', 'test_feature'], 20.)
        self.assertEqual(master.loc['rgg__c', 'test_feature'], 30.)

    def test_exact_display_alias_joins_without_approximate_matching(self):
        master = self.build([['A–RGG', 10., 1., 'Kd fit', 'display name'], ['a', 20., 2., 'Kd fit', 'ambiguous short name']])
        self.assertEqual(master.loc['rgg__a', 'Kd_uM'], 10.)
        self.assertIn('a', master.index)
        self.assertTrue(np.isnan(master.loc['a', 'Pred_log10_negB22_final']))

    def test_missing_feature_map_does_not_discard_other_sources(self):
        csv_write(pd.DataFrame(dict(test_feature=[30., 10., 20.])), self.root / 'features.csv')
        self.config['features']['row_ids_path'] = str(self.root / 'absent_ids.csv')
        master = self.build([['rgg__a', 10., 1., 'Kd fit', '']])
        self.assertEqual(master.loc['rgg__a', 'Pred_log10_negB22_final'], 4.)
        self.assertEqual(master.loc['rgg__a', 'feature_status'], 'missing current features')
        self.assertTrue((self.root / 'outputs/unresolved_feature_rows.csv').exists())

    def test_updated_values_change_limits_and_statistics(self):
        from plot_master_validation import limits, MasterFigures, PRED
        before = self.build([['rgg__a', 100., 10., 'Kd fit', ''], ['rgg__b', 10., 1., 'Kd fit', '']]).reset_index()
        after = self.build([['rgg__a', 100., 10., 'Kd fit', ''], ['rgg__b', 10., 1., 'Kd fit', ''], ['rgg__c', .01, .001, 'Kd fit', '']]).reset_index()
        figure_config = dict(self.config, diagnostics_dir=str(self.root / 'diagnostics'), plot_limits={})
        plot = MasterFigures(after, figure_config, self.root / 'outputs')
        selected = plot.select('fixture', plot.fitted_criteria(PRED))
        self.assertEqual(len(selected), 3)
        a = before.loc[before.pKd.notna(), 'pKd']
        b = after.loc[after.pKd.notna(), 'pKd']
        self.assertGreater(limits(b)[1], limits(a)[1])
        self.assertTrue(np.isnan(spearman_test([4., 4.1], a, self.settings)[1]))
        rho, p, method, count = spearman_test(selected[PRED], selected.pKd, self.settings)
        self.assertAlmostEqual(rho, 1.)
        self.assertAlmostEqual(p, 1 / 3)
        self.assertEqual(count, 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
