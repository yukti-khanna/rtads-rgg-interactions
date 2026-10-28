#!/usr/bin/env python3
"""Local checks on synthetic data; does not process the scientific trajectories."""
import unittest
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from run_revision_analysis import auc,concordance,threshold,metrics,boot
from revision_trajectories import accumulate
OUTPUTS='--outputs' in sys.argv
if OUTPUTS:sys.argv.remove('--outputs')

class RevisionChecks(unittest.TestCase):
 def test_auc_concordance_ties(self):
  self.assertAlmostEqual(auc([0,0,1,1],[1,2,2,3]),.875)
  self.assertAlmostEqual(concordance([1,2,3],[1,1,3]),5/6)
  self.assertTrue(np.isnan(auc([1,1],[2,3])))
 def test_threshold_orientation(self):
  y=np.array([False,False,True,True]);s=np.array([1.,2.,3.,4.]);t,_,_=threshold(y,s)
  self.assertEqual(t,2.5);self.assertEqual(metrics(y,s>=t)['balanced_accuracy'],1)
 def test_cluster_bootstrap_reproducible(self):
  d=pd.DataFrame({'group':['A','A','B','B','C','C'],'y':[0,1,0,1,0,1],'s':[0,1,0,2,1,2]})
  a=boot(d,'y','s','group',auc);b=boot(d,'y','s','group',auc)
  self.assertEqual(a,b);self.assertEqual(a['valid_resamples'],2000)
 def test_energy_sign_cutoff_and_aggregation(self):
  # Far interactions vanish, opposite charges attract, equal charges repel.
  sig=np.full((1,2),.5);lam=np.full((1,2),.7);qq=np.array([[-1.,1.]])
  par=np.array([.8368,2.,1.770853714757,1.273814397064,np.exp(-1.273814397064*4)/4,4.])
  a=np.zeros((1,2));y=a.copy();c=a.copy();cond=a.copy();cl=np.zeros((16,4));classes=np.zeros((1,2),dtype=int)
  v=accumulate(np.array([[7.,7.]]),sig,lam,qq,par,a,y,c,cond,classes,cl)
  self.assertLess(y[0,0],0);self.assertGreater(y[0,1],0);self.assertAlmostEqual(v[2],a.sum()+y.sum());self.assertEqual(v[3],1)
  np.testing.assert_allclose(cond,a+y)
  aa=np.zeros((1,2));yy=aa.copy();cc=aa.copy();co=aa.copy();cs=np.zeros((16,4))
  v=accumulate(np.array([[50.,50.]]),sig,lam,qq,par,aa,yy,cc,co,classes,cs)
  np.testing.assert_allclose(v,0)
 def test_bound_fraction_validation_rules(self):
  from build_final_figure_package import BOUND_COLUMNS
  source=pd.read_csv(Path(__file__).resolve().parent/'fig6_parent_master.csv')
  self.assertTrue(set(BOUND_COLUMNS).issubset(source.columns))
  fraction=pd.to_numeric(source.fraction_bound_highest_ligand,errors='coerce')
  percent=pd.to_numeric(source.percent_bound_highest_ligand,errors='coerce')
  equivalents=pd.to_numeric(source.highest_ligand_equivalents,errors='coerce')
  self.assertTrue(fraction.dropna().between(0,1).all())
  self.assertTrue(percent.dropna().between(0,100).all())
  self.assertTrue(equivalents.dropna().gt(0).all())
  both=fraction.notna()&percent.notna()
  np.testing.assert_allclose(percent[both],100*fraction[both],atol=.05,rtol=0)

@unittest.skipUnless(OUTPUTS,'Use --outputs after the local scientific run')
class OutputChecks(unittest.TestCase):
 def test_curated_input_and_panel_membership(self):
  from run_revision_analysis import ROOT,OUT,sha
  hashes=pd.read_csv(OUT/'tables/input_hashes.csv').set_index('path')
  self.assertEqual(sha(ROOT/'fig6_parent_master.csv'),hashes.loc[str(ROOT/'fig6_parent_master.csv'),'sha256'])
  source=pd.read_csv(ROOT/'fig6_parent_master.csv')
  plotted=pd.read_csv(OUT/'paper_figures/prediction_vs_pKd.csv')
  expected=set(source.loc[source.AHD_variant.eq('WT')&source.Outcome.eq('Kd_fit')&source.Kd_uM.gt(0),'pair_id'])
  self.assertEqual(set(plotted.pair_id),expected)
  self.assertTrue(source.loc[source.Outcome.eq('NB'),'Kd_uM'].isna().all())
  corrected=source.set_index('pair_id').loc['g3bp1_rgg__tp53']
  self.assertEqual(corrected.Outcome,'NB')
  self.assertNotIn('g3bp1_rgg__tp53',set(plotted.pair_id))
 def test_shuffle_and_concordance_exports(self):
  from run_revision_analysis import ROOT,OUT
  source=pd.read_csv(ROOT/'fig6_parent_master.csv');audit=pd.read_csv(OUT/'tables/shuffle_composition_audit.csv')
  fig=pd.read_csv(OUT/'figures/WT_shuffle_comparison.csv')
  expected=set(source.loc[source.AHD_variant.str.startswith('shuffle'),'pair_id'])
  passed=set(audit.loc[audit.composition_verified,'pair_id'])
  self.assertEqual(set(fig.pair_id),expected&passed)
  stats=pd.read_csv(OUT/'tables/discrimination_statistics.csv')
  np.testing.assert_allclose(stats.AUROC,stats.positive_negative_concordance)
  self.assertTrue(stats.CI95_low.between(0,1).all());self.assertTrue(stats.CI95_high.between(0,1).all())
 def test_primary_WT_and_intensity_scope(self):
  root=Path(__file__).resolve().parent
  master=pd.read_csv(root/'fig6_parent_master.csv')
  wt=set(master.loc[master.AHD_variant.eq('WT'),'pair_id'])
  for name in ['pKd_rank_members.csv','threshold_heldout_predictions.csv']:
   frame=pd.read_csv(root/'revision_analysis/tables'/name)
   self.assertTrue(set(frame.pair_id)<=wt)
  intensity=pd.read_csv(root/'revision_analysis/tables/intensity_rank_members.csv')
  self.assertEqual(set(intensity.pair_id),set(master.loc[master.I_ratio_mean_below_threshold.notna(),'pair_id']))
  self.assertTrue(bool(set(intensity.pair_id)-wt))
 def test_energy_aggregation(self):
  from run_revision_analysis import OUT
  path=OUT/'trajectories/summary.csv'
  if not path.exists():self.skipTest('Trajectory stage not yet run')
  for _,r in pd.read_csv(path).iterrows():
   maps=pd.read_csv(OUT/'trajectories/cache'/r.pair_id/'residue_maps.csv.gz')
   self.assertAlmostEqual(maps.total_kJ_mol.sum(),r.total_kJ_mol,places=6)
   self.assertEqual(int(r.frames),len(pd.read_csv(OUT/'trajectories/cache'/r.pair_id/'frame_summary.csv.gz')))
 def test_single_table_s10_workbook_and_membership(self):
  from run_revision_analysis import ROOT
  out=ROOT/'final_figure_package'
  inspect=(out/'Supplementary_Table_S11.xlsx.inspect.ndjson').read_text()
  for sheet in ['NMR outcomes and Kd','Validation statistics','AUROC and concordance','Threshold sensitivity','Intensity ratios','Shuffle comparisons','LEF1 context','Bound-fraction guide','Data dictionary']:
   self.assertIn(sheet,inspect)
  source=pd.read_csv(ROOT/'fig6_parent_master.csv')
  table=pd.read_csv(out/'tables/Table_S11_NMR_outcomes_and_Kd.csv')
  self.assertEqual(set(source.pair_id),set(table.pair_id))
  self.assertTrue(set(['highest_ligand_equivalents','fraction_bound_highest_ligand','percent_bound_highest_ligand']).issubset(table.columns))
 def test_revised_s11_threshold_shuffle_and_lef1_panels(self):
  from run_revision_analysis import OUT,ROOT
  package=ROOT/'final_figure_package';source=package/'source_data'
  threshold=pd.read_csv(source/'FigureS11A_threshold_classification.csv')
  expected=pd.read_csv(ROOT/'fig6_parent_master.csv');expected=set(expected.loc[expected.AHD_variant.eq('WT')&expected.Outcome.isin(['Kd_fit','NB']),'pair_id'])
  self.assertEqual(set(threshold.pair_id),expected)
  robust=pd.read_csv(source/'FigureS11B_threshold_robustness.csv')
  self.assertEqual(set(robust.evaluation),{'in-sample exploratory','leave-AHD_parent-out','leave-RGG_parent-out'})
  shuffle=pd.read_csv(source/'FigureS11C_shuffle_outcomes.csv')
  complete=shuffle.WT_outcome.ne('Kd_pending') & shuffle.shuffle_outcome.ne('Kd_pending')
  changed=complete & shuffle.WT_outcome.ne(shuffle.shuffle_outcome)
  np.testing.assert_array_equal(shuffle.complete_pair,complete)
  np.testing.assert_array_equal(shuffle.outcome_changed,changed)
  lef=pd.read_csv(source/'FigureS11D-F_LEF1_matched_metrics.csv')
  self.assertEqual(set(lef.metric),{'total_kJ_mol','contact_08','total_kJ_mol_contact_conditioned'})
  self.assertEqual(lef.groupby('metric').size().to_dict(),{'contact_08':4,'total_kJ_mol':4,'total_kJ_mol_contact_conditioned':4})
  self.assertTrue(lef.direction_consistent.all())
  self.assertFalse((source/'FigureS11_synthetic_reference_source.csv').exists())

if __name__=='__main__':unittest.main(verbosity=2)
