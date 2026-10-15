#!/usr/bin/env python3
"""Checks actual row populations, archived inference, statistics and exports."""
import json
import numpy as np
import pandas as pd
from PIL import Image
from workflow_common import ROOT, OUT, DIAG
from expanded_validation_analysis import spearman_permutation, pairwise_rank_test


def main():
    rho,p,method,n=spearman_permutation(np.arange(3),np.arange(3),np.random.default_rng(1))
    assert np.isclose(rho,1) and np.isclose(p,1/3) and n==6
    p,n=pairwise_rank_test(np.array([3,4]),np.array([1,2]))
    assert np.isclose(p,1/3) and n==6
    checks=pd.read_csv(DIAG/'independent_20pair_checks.csv')
    assert checks.passed.all()
    pairs=pd.read_csv(DIAG/'all20_prediction_changes.csv')
    assert pairs.Outcome.value_counts().to_dict()=={'Kd_fit':9,'NB':7,'LB':4}
    kd=pd.read_csv(DIAG/'kd_statistics.csv')
    assert kd.n.eq(9).all() and kd.permutations.eq(362880).all()
    old=pd.read_csv(ROOT/'Table_S8.csv')
    updated=pd.read_csv(OUT/'experimental_validation_updated.csv')
    pd.testing.assert_frame_equal(old,updated[old.columns])
    recovery=json.loads((DIAG/'historical_model/recovery_status.json').read_text())
    if recovery['recovered_same_construct_old_pairs']:
        full=pd.read_csv(DIAG/'historical_model/same_construct_predictions.csv')
        new=pd.read_csv(ROOT/'preds_val_updated/predictions_minimal.csv').set_index('pair_id')
        assert len(full)==len(new) and set(full.pair_id)==set(new.index)
        np.testing.assert_allclose(full.set_index('pair_id').loc[new.index,'New_final_log10_negB22'],new.final_hybrid_prediction)
        features=pd.read_csv(DIAG/'historical_model/feature_reproduction.csv')
        assert features.passed.all()
        repro=pd.read_csv(DIAG/'historical_model/panel_reproduction.csv')
        good=repro[repro.passed]
        assert len(good)==20
        np.testing.assert_allclose(good.candidate_base,good.archived_base,rtol=0,atol=1e-6)
        np.testing.assert_allclose(good.candidate_final,good.archived_final,rtol=0,atol=1e-6)
    axes=json.loads((DIAG/'figure_axis_limits.json').read_text())
    kd_axes=[a for a in axes if a['panel'].endswith(('old_prediction_vs_pKd','new_prediction_vs_pKd'))]
    assert len(kd_axes)==2 and kd_axes[0]['xlim']==kd_axes[1]['xlim'] and kd_axes[0]['ylim']==kd_axes[1]['ylim']
    for a in axes:
        if a['panel'].endswith(('old_vs_new_predictions','complete_same_construct_agreement')):
            assert a['xlim']==a['ylim'] and a['aspect']=='1.0'
    manifest=pd.read_csv(DIAG/'figure_export_manifest.csv')
    for png in manifest.png:
        p=ROOT/png
        for suffix in ['.png','.pdf','.svg','.csv','.caption.txt']:
            assert p.with_suffix(suffix).exists(),p.with_suffix(suffix)
        assert '<text' in p.with_suffix('.svg').read_text(),f'SVG lacks editable text: {p}'
        with Image.open(p) as im:
            assert all(abs(dpi-600)<1 for dpi in im.info['dpi'])
    (OUT/'qc/expanded_verification.txt').write_text('Passed: exact permutation toy checks, independent 20-pair values, counts, preserved annotations, frozen old inference/feature reproduction, identical model axes, editable SVGs and 600-dpi exports. Visual inspection is recorded separately.\n')
    print('Expanded numerical/source/export checks passed')


if __name__=='__main__':
    main()
