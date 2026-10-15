"""Read-only import of the existing histidine controls for Figure S10."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent

def load_audit_tables():
    """Relevant audit measurements only; distinguish base from final hybrid output."""
    folder = ROOT.parent/'fig5/outputs/fig5_histidine_pH_audit'
    matched_path = folder/'histidine_matched_b22.csv'
    m = pd.read_csv(matched_path).set_index('pair_id', verify_integrity=True).sort_index()
    paths = [ROOT.parent/f'fig5/inputs/controls/histidine_{condition}_final_predictions.csv'
             for condition in ['pH7','pH6p5']]
    inputs = [pd.read_csv(p).set_index('pair_id', verify_integrity=True).sort_index() for p in paths]
    for frame,condition,target in zip(inputs,['pH7','pH6p5'],['B22_corr1_log10_pH7','B22_corr1_log10_pH6p5']):
        if not frame.index.equals(m.index):raise ValueError('Audit/prediction pair identities differ')
        if not np.allclose(frame.base_prediction,m['prediction_'+condition],rtol=0,atol=1e-10):
            raise ValueError('Audit predictions no longer correspond to base_prediction')
        if not np.allclose(frame.B22_corr1_log10,m[target],rtol=0,atol=1e-10):
            raise ValueError('Audit/prediction targets differ')
    sim = m[['TAD_id','RGG_id','n_replicates','mean_B22_pH7','mean_B22_pH65',
             'SD_pH7','SD_pH65','B22_corr1_log10_pH7','B22_corr1_log10_pH6p5',
             'delta_B22','delta_logB22','n_H_TAD','n_H_RGG','total_H']].copy()
    sim['pooled_within_condition_SD']=np.sqrt((sim.SD_pH7**2+sim.SD_pH65**2)/2)
    sim['standardized_condition_difference']=sim.delta_B22/sim.pooled_within_condition_SD
    pred=m[['TAD_id','RGG_id','B22_corr1_log10_pH7','B22_corr1_log10_pH6p5']].copy()
    metrics=[]
    for frame,condition,target in zip(inputs,['pH7','pH6p5'],['B22_corr1_log10_pH7','B22_corr1_log10_pH6p5']):
        for kind,column in [('base','base_prediction'),('final_hybrid','final_hybrid_prediction')]:
            x=m[target];y=frame[column];res=y-x
            pred[f'{kind}_prediction_{condition}']=y
            pred[f'{kind}_residual_{condition}']=res
            metrics.append(dict(model_output=column,condition=condition,n=len(x),R2=1-np.sum(res**2)/np.sum((x-x.mean())**2),
                                RMSE=np.sqrt(np.mean(res**2)),MAE=np.mean(abs(res)),Pearson_r=np.corrcoef(x,y)[0,1],
                                Spearman_rho=spearmanr(x,y).statistic,mean_residual_predicted_minus_simulated=res.mean()))
    for kind in ['base','final_hybrid']:
        pred[f'{kind}_delta_absolute_error']=abs(pred[f'{kind}_residual_pH6p5'])-abs(pred[f'{kind}_residual_pH7'])
    for condition in ['pH7','pH6p5']:
        reported=pd.read_csv(folder/f'histidine_prediction_metrics_{condition}.csv').iloc[0]
        calc=next(row for row in metrics if row['condition']==condition and row['model_output']=='base_prediction')
        for key in ['R2','RMSE','MAE']:
            if not np.isclose(calc[key],reported[key],atol=1e-10):raise ValueError('Audit metric mismatch: '+key)
    sources=pd.DataFrame([{'source':str(p.relative_to(ROOT.parent)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                          for p in [matched_path,*paths]])
    return sim.reset_index(),pred.reset_index(),pd.DataFrame(metrics),sources

def load_controls():
    paths = [ROOT.parent/'fig4/reviewer_analysis/simulation_control_panels/FigS4_histidine_pH_sim_vs_sim_source.csv',
             ROOT.parent/'fig5/controls/tables/FigS8D_histidine_pH_simulation_source.csv',
             ROOT.parent/'fig5/controls/final_predictions/histidine_pH6p5_final_predictions.csv']
    a,b,c = [pd.read_csv(p).set_index('pair_id', verify_integrity=True).sort_index() for p in paths]
    if not (a.index.equals(b.index) and a.index.equals(c.index)):
        raise ValueError('Histidine control pair identities differ between source tables')
    for x,y in [(a.pH7_simulated_value_log10_negB22,b.B22_corr1_log10_pH7),
                (a.pH65_simulated_value_log10_negB22,b.B22_corr1_log10_pH6p5),
                (a.pH65_simulated_value_log10_negB22,c.B22_corr1_log10)]:
        if not np.allclose(x,y,rtol=0,atol=1e-10):
            raise ValueError('Histidine simulation values disagree between sources')
    sim=pd.DataFrame({'RGG_id':b.RGG_id,'AHD_id':b.TAD_id,'total_H':b.total_H,
        'simulated_pH7':a.pH7_simulated_value_log10_negB22,
        'simulated_pH6p5':a.pH65_simulated_value_log10_negB22})
    sim['delta_pH6p5_minus_pH7']=sim.simulated_pH6p5-sim.simulated_pH7
    pred=pd.DataFrame({'RGG_id':c.RGG_id,'AHD_id':c.TAD_id,
        'simulated_pH6p5':c.B22_corr1_log10,'predicted_pH6p5':c.final_hybrid_prediction})
    pred['residual_predicted_minus_simulated']=pred.predicted_pH6p5-pred.simulated_pH6p5
    for col in ['model_manifest_sha256','base_model_sha256','residual_model_sha256']:
        pred[col]=c[col]
    metrics=[]
    for panel,x,y in [('S10A',sim.simulated_pH7,sim.simulated_pH6p5),('S10B',pred.simulated_pH6p5,pred.predicted_pH6p5)]:
        if not np.isfinite(np.column_stack([x,y])).all():raise ValueError('Nonfinite pH-control scores')
        metrics.append(dict(panel=panel,n=len(x),Pearson_r=np.corrcoef(x,y)[0,1],Spearman_rho=spearmanr(x,y).statistic,
            mean_y_minus_x=np.mean(y-x),RMSE=np.sqrt(np.mean((y-x)**2)),R2=1-np.sum((y-x)**2)/np.sum((x-x.mean())**2)))
    provenance=pd.DataFrame([{'source':str(p.relative_to(ROOT.parent)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths])
    return sim.reset_index(),pred.reset_index(),pd.DataFrame(metrics),provenance

RESULTS = '''The NMR titrations were performed at pH 6.5, whereas the reference simulations used pH 7.0. We therefore examined whether adjusting histidine protonation to the experimental pH altered the simulated interaction landscape. Across 390 matched pairs, scores at the two pH values remained closely correlated (Pearson r = 0.98; Spearman ρ = 0.96), with a mean change of approximately 0.001 and an RMSE of 0.061 in log₁₀(−B₂₂) (Fig. S10A). The frozen predictor also retained agreement with the pH 6.5 simulations (R² = 0.95; RMSE = 0.07; Pearson r = 0.98; Fig. S10B). These controls support the robustness of the broad computational ranking to the modelled histidine-protonation adjustment, although they do not establish that individual experimental affinities are insensitive to pH.'''

CAPTION = '''## Figure S10 Histidine pH controls, simulation comparisons and exploratory intensity attenuation
(A) Simulated log₁₀(−B₂₂) at pH 7.0 versus pH 6.5 for 390 matched computational-control pairs. (B) Frozen predicted scores versus pH 6.5 simulated scores for the same pairs. Lines in A and B indicate identity, not fitted regressions. Annotations report sample sizes and agreement statistics. These computational controls are distinct from the experimental NMR panel. (C) Simulation-derived scores grouped by ND, LB and Kd-fit outcomes; bars show medians. (D) Experimental pKd versus simulated scores for the same fitted measurements as Figure 5D. Error bars propagate supplied Kd fitting errors, and the dashed line is a descriptive least-squares fit. (E) Exploratory mean-below-threshold intensity ratios versus predicted and simulated scores. Lines are descriptive fits and coefficients are Spearman correlations. Intensity ratios are not converted to Kd or bound fractions. Pair identities, pH-control sources, agreement statistics and experimental panel data are reported in Supplementary Table S11.'''
