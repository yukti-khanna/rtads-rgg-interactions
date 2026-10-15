"""Independent numerical checks and figure-only refresh; no model use."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image

OUT=Path(__file__).resolve().parent
SCRIPT=OUT.parent/'scripts/run_predictor_shuffle_uncertainty.py'
spec=importlib.util.spec_from_file_location('shuffle_analysis',SCRIPT)
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)
data=pd.read_csv(OUT/'predictor_shuffle_replica_data.csv',float_precision='round_trip')
source=pd.read_csv(OUT/'FigS7EF_shuffle_delta_uncertainty_source.csv',float_precision='round_trip')
current=pd.read_csv(OUT/'current_delta_analysis.csv',float_precision='round_trip')
parents=pd.read_csv(OUT/'predictor_shuffle_parent_structure.csv')
summary=pd.read_csv(OUT/'predictor_delta_uncertainty_summary.csv')
null=pd.read_csv(OUT/'predictor_delta_null_test.csv')
signals=pd.read_csv(OUT/'predictor_delta_signal_vs_uncertainty.csv')
boot=pd.read_csv(OUT/'bootstrap_metric_draws.csv',float_precision='round_trip')
ndraw=pd.read_csv(OUT/'null_metric_draws.csv',float_precision='round_trip')
saved=np.load(OUT/'bootstrap_delta_draws.npz',allow_pickle=False)
draws=saved['delta_sim']
assert saved['shuffle_pair_id'].tolist()==data.shuffle_pair_id.tolist()
assert draws.shape==(20000,345)
assert len(parents)==115 and parents.n_comparisons.eq(3).all()
assert parents.n_AHD.eq(2).all() and parents.n_RGG.eq(1).all()
assert data.WT_n_replicas.eq(5).all() and data.shuffle_n_replicas.eq(5).all()
np.testing.assert_allclose(draws.std(axis=0,ddof=1),source.bootstrap_SE_delta_sim,rtol=0,atol=1e-14)
np.testing.assert_allclose(np.quantile(draws,[.025,.975],axis=0).T,source[['bootstrap_q025_delta_sim','bootstrap_q975_delta_sim']],rtol=0,atol=1e-14)
max_difference=0.0
for group in analysis.GROUPS:
    mask=data.shuffle_type.eq(group).to_numpy();x=data.loc[mask,'delta_pred'].to_numpy()
    # Independent scipy calls compare selected resampled draws to the vectorized implementation.
    for index in [0,1,11,99,999,9999,19999]:
        y=draws[index,mask]
        expected={'Pearson':stats.pearsonr(x,y).statistic,'Spearman':stats.spearmanr(x,y).statistic,
                  'RMSE':np.sqrt(np.mean((x-y)**2)),'MAE':np.mean(np.abs(x-y))}
        for metric,val in expected.items():
            diff=abs(val-boot.loc[index,f'{group}_{metric}']);max_difference=max(max_difference,diff)
            assert diff<1e-12,(group,index,metric,diff)
    for metric in ['Pearson','Spearman']:
        obs=current.loc[current.shuffle_type.eq(group),metric].iloc[0]
        calculated=(1+int((ndraw[f'{group}_{metric}'].abs()>=abs(obs)).sum()))/20001
        reported=null[(null.shuffle_type==group)&(null.metric==metric)].empirical_two_sided_p.iloc[0]
        assert abs(calculated-reported)<1e-15
for col in ['delta_sim','delta_pred']:
    np.testing.assert_allclose(data[col],source[col],rtol=0,atol=1e-15)

# Synthetic diagnostic: two constant-shuffle children sharing one fluctuating WT
# must have identical target-delta draws; an independent WT must differ.
toy=pd.DataFrame({'WT_pair_id':['parent1','parent1','parent2'],'shuffle_pair_id':['s1','s2','s3']})
values={'parent1':np.array([-10,-20,-30,-40,-50]),'parent2':np.array([-10,-20,-30,-40,-50]),
        's1':np.repeat(-25.,5),'s2':np.repeat(-25.,5),'s3':np.repeat(-25.,5)}
test=analysis.bootstrap_targets(toy,values,500,np.random.default_rng(197))
np.testing.assert_array_equal(test[:,0],test[:,1])
assert not np.array_equal(test[:,0],test[:,2])

# Refresh presentation/report after the annotation-placement clarification only.
analysis.figures(data,current,summary)
prov=json.loads((OUT/'analysis_provenance.json').read_text())
prov['sources'][str(SCRIPT.resolve())]=analysis.sha(SCRIPT)
prov['verification_script_sha256']=analysis.sha(Path(__file__))
prov['verification']={'n_bootstrap':20000,'n_null':20000,'metric_max_absolute_difference_vs_scipy':float(max_difference),
                      'shared_WT_synthetic_check':'passed','full_comparison_counts':'passed','five_replica_counts':'passed',
                      'source_inputs_unchanged':True}
for path,expected in prov['sources'].items():assert analysis.sha(path)==expected,path
for row in pd.read_csv(OUT/'rdf_input_inventory.csv').itertuples():assert analysis.sha(row.path)==row.sha256,row.path
analysis.write_report(data,parents,current,summary,null,signals,prov)
for name in ['FigS7E_AHD_shuffle_delta_uncertainty.png','FigS7F_RGG_shuffle_delta_uncertainty.png']:
    with Image.open(OUT/name) as im:
        assert im.size==(1440,1380)
        assert all(abs(d-600)<.1 for d in im.info['dpi'])
prov['verification']['panel_resolution']='1440 x 1380 pixels, 600 dpi, 2.4 x 2.3 inches'
(OUT/'analysis_provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
(OUT/'verification_results.json').write_text(json.dumps(prov['verification'],indent=2)+'\n')
paths=sorted(str(p.resolve()) for p in OUT.iterdir() if p.is_file())
(OUT/'output_files.txt').write_text('\n'.join(paths)+'\n')
print('Independent verification passed:',json.dumps(prov['verification'],indent=2))
