#!/usr/bin/env python3
"""Report actual expanded results, provenance and visual inspection state."""
import json
import pandas as pd
from workflow_common import ROOT, OUT, DIAG


def main():
    changes=pd.read_csv(DIAG/'prediction_change_summary.csv')
    all20=changes[changes['set'].eq('all_experimental')].iloc[0]
    kd=pd.read_csv(DIAG/'kd_statistics.csv')
    auc=pd.read_csv(DIAG/'outcome_auroc.csv')
    outcomes=pd.read_csv(DIAG/'outcome_statistics.csv')
    pairs=pd.read_csv(DIAG/'all20_prediction_changes.csv')
    recovery=json.loads((DIAG/'historical_model/recovery_status.json').read_text())
    simstatus=json.loads((OUT/'b22/subset_accuracy_status.json').read_text())
    simmetrics=pd.read_csv(OUT/'b22/paired_simulation_accuracy_metrics.csv')
    visualfile=DIAG/'visual_inspection.json'
    visual=json.loads(visualfile.read_text()) if visualfile.exists() else {'status':'inspection needed after regeneration'}
    complete=pd.read_csv(DIAG/'complete_same_construct_summary.csv') if (DIAG/'complete_same_construct_summary.csv').exists() else pd.DataFrame()
    lines=['# Corrected validation results','','## 1. Prediction changes across all experimental pairs','',
        f"All {int(all20.n)} finite old/new experimental pairs were included, regardless of Kd availability. "
        f"Mean new−old = {all20.mean_delta:+.6f} (descriptive paired-bootstrap 95% CI {all20.mean_delta_CI95_low:+.6f} to {all20.mean_delta_CI95_high:+.6f}); "
        f"median = {all20.median_delta:+.6f}; mean absolute change = {all20.mean_absolute_change:.6f}; "
        f"range = {all20.minimum_delta:+.6f} to {all20.maximum_delta:+.6f}. "
        f"Old/new agreement: Pearson r={all20.pearson_r:.6f}, Spearman ρ={all20.spearman_rho:.6f}. "
        f"Exact two-sided paired sign-flip test of mean change: p={all20.shift_pvalue:.6f}. A systematic shift is not an accuracy improvement.",'',
        'Independently supplied checks (n, r, ρ, mean and median change) all match within 1e-6. Ranking changes per pair are exported.','',
        '## 2. Kd agreement and outcome separation','',
        'All nine fitted Kd pairs, including FOXO4–G3BP1, are primary. pKd=6−log10(Kd_uM), with first-order uncertainty σKd/(Kd ln10).', '']
    for model in ['old','new']:
        s=kd[kd.predictor.eq(model)].set_index('metric')
        au=auc[auc.predictor.eq(model)].iloc[0]
        om=outcomes[outcomes.predictor.eq(model)].iloc[0]
        lines.append(f"- {model}: Kd Pearson r={s.loc['pearson_r','estimate']:.6f}; Spearman ρ={s.loc['spearman_rho','estimate']:.6f}, "
                     f"exact two-sided permutation p={s.loc['spearman_rho','spearman_two_sided_p']:.6f} (9!=362,880 permutations). "
                     f"Outcome omnibus H={om.statistic:.4f}, permutation p={om.p_raw:.4f}. Kd-fit versus NB AUROC={au.estimate:.6f}.")
    ad=auc[auc.predictor.eq('new_minus_old')].iloc[0]
    lines += ['',f"Thus the directions are mixed: fitted Kd agreement improves, but Kd-fit/NB AUROC decreases by {ad.estimate:.6f} "
             f"(paired 95% CI {ad.CI95_low:.6f} to {ad.CI95_high:.6f}). LB is excluded only from this binary diagnostic. "
             'Outcome counts are nine Kd-fit, four LB, seven NB. Kd-fit/LB separation increases; LB/NB effect stays the same. '
             'Holm-adjusted pairwise p-values and effects, including nonsignificant results, are in outcome_statistics.csv; '
             'paired effect differences are in paired_outcome_effect_differences.csv. Separate model p-values are not a test of model superiority.','',
             'All p-values and row-bootstrap intervals are exploratory: pairs share constructs, so independent-row exchangeability is not assured. '
             'No affinity ordering is imposed on LB, NB, Kd-fit or No Kd.','',
             '## 3. Partners and largest construct differences','']
    for _,r in changes[changes['set'].str.startswith('partner_display:')].iterrows():
        lines.append(f"- {r['set'].split(':',1)[1]}: mean change {r.mean_delta:+.6f}; n={int(r.n)}.")
    lines += ['', 'All five TAF15 predictions decrease. LEF1 predictions increase for THOC4 N-term, THOC4 C-term and G3BP1. '
             'Largest absolute changes:', '', pairs.assign(abs_delta=pairs.delta.abs()).sort_values('abs_delta',ascending=False)[['pair_id','Outcome','delta']].head(6).to_string(index=False),'',
             '## 4. Complete exact-construct old/new comparison','',
             f"Recovered historical final pipeline predictions for {recovery['recovered_same_construct_old_pairs']} of {recovery['total_new_pairs']} exact new-run construct pairs. "
             'These newly inferred same-construct old predictions are kept separate from archived Table_S8 predictions. '
             'models_old/ contained a different 22-feature base pipeline and standalone model/scaler. The actual matching archived base/residual weights '
             'were in models/pretrained/mlp_plateau_calibrated_bundle_sel_cols.pkl.','',
             'The archived panel convention applies the saved clipped residual without the public script’s soft gate. '
             'Three fixed application conventions were tested; ungated application reproduced all 20 archived base/final values within 1e-6. '
             'All ten historical feature columns were reconstructed from local pure functions/residue parameters and matched to numerical precision. '
             'No weights, lambda, cap, new-run gate or preprocessing parameters were fitted or recalibrated. '
             'Dense Keras weights were read directly for inference, with only supported sequential Dense graphs accepted. '
             'Artifact hashes, application conventions, loading warnings, exact features and reproduction errors are retained.','']
    if not complete.empty:
        lines += [complete[complete['set'].isin(['complete_same_construct','WT','shuffled'])].to_string(index=False),'',
                  'Matched WT–shuffle source tables preserve shuffle identity and WT-parent relationships; named heatmaps and a paired shuffle-effect panel are exported.','']
    lines += ['## 5. Simulation accuracy and unresolved provenance','',
        f"The legacy convention was reproduced for {simstatus['legacy_7_5_reproduction_pairs']} pairs using local archived RDF candidates. "
        f"A {simstatus['eligible_accuracy_pairs']}-pair subset has exact supplied sequences in the corresponding current final-run topologies and "
        'compatible legacy 7.5 nm corr1 target evidence. Both predictors use exactly those rows, with logs taken after replica averaging:', '',
        simmetrics.to_string(index=False),'',
        'The newer supplied RDFs do not globally reproduce the older simulation table. Source/version differences are investigated per pair and cutoff; '
        'archived RDF candidates reproduce the historical values, so a discrepancy in the newer RDF version does not imply a new target convention. '
        'Older run_validated/ FOXO4 inputs are shorter than current final-run inputs and are kept separate; they do not veto matching current topologies. '
        'See topology_construct_sequence_audit.csv, convention_source_replica_investigation.csv and per_pair_simulation_eligibility.csv.','',
        'Accuracy improves on the supported simulation subset, but is not evidence of accuracy across all 73 pairs or calibrated Kd prediction. '
        'The other 53 pairs lack sufficient construct/run provenance for this paired accuracy analysis. '
        'RDF-to-topology linkage is contextual (matching named run directories), not an embedded generator manifest. '
        'Exact supplied new-model training RDF provenance also remains incomplete. Physical nm³ integrals remain separately exported and are not substituted into frozen predictions.','',
        '## Outputs, numbering and visual review','',
        'Individual PDFs, editable SVGs, 600-dpi PNGs, exact plotted data, statistics and captions are in updated_validation/figures/ and prediction_diagnostics/figures/. '
        'The readable diagnostic_overview.png accompanies individual panels. Figure 5 panel letters remain unverified; descriptive filenames avoid occupied S7/S8 numbering. '
        'Original data, annotations and DOCX files are unchanged. Prior generated outputs are saved under workflow_history/before_expanded_diagnostics_20261001/.','',
        f"Visual inspection status: {visual.get('status')}. See visual_inspection.json and VISUAL_QA.md for inspected filenames and corrections.",'',
        'Proposed manuscript/legend corrections: describe historical target sums without nm³ units; distinguish 7.5 nm legacy sums from L/2 physical integrals; '
        'include all nine Kd fits as primary; state shared-construct limits; report mixed Kd/outcome findings rather than universal superiority; '
        'identify the recovered archived correction convention and sequence-supported simulation subset. No manuscript document was edited.']
    text='\n'.join(lines)+'\n'
    (OUT/'RESULTS_REPORT.md').write_text(text)
    (DIAG/'COMPARISON_REPORT.md').write_text(text)
    (DIAG/'METHODS.md').write_text('''# Statistical and provenance methods

All old/new comparisons use identical eligible rows. Archived old values remain separate from newly generated same-construct old inference. Kd fits use positive numeric Kd and retain all nine pairs. Spearman p-values enumerate all 9! labelled-row permutations, use |rho| for a two-sided test and count ties inclusively. Pearson and Spearman confidence intervals are row-bootstrap descriptive intervals; degenerate draws are recorded. Primary old/new metric differences use paired row draws.

Kruskal–Wallis uses average ranks, standard tie correction and 19,999 random label permutations, with (extreme+1)/(B+1). Pairwise tests enumerate all assignments to the two observed group sizes using |U−n1*n2/2|. Holm adjustment applies to the three pairwise comparisons per predictor. Rank-biserial effects equal 2*AUROC−1, so positive means the first named group has higher predictions. Pairwise effects have descriptive bootstrap intervals; differences use the same within-category rows for old/new. Omnibus epsilon squared is bias adjusted and can be negative.

The binary diagnostic includes only Kd-fit (positive, n=9) and NB (negative, n=7), where higher prediction is expected for Kd-fit; LB is excluded, not ordered. AUROC confidence intervals and new−old differences use paired outcome-stratified bootstrap draws. Outcome plots show counts and numeric adjusted p-values.

Mean/median paired changes use 10,000 paired bootstrap draws. The systematic shift test enumerates sign flips for n<=20, otherwise uses 19,999 Monte Carlo sign flips; its symmetric-difference/exchangeability assumption is exploratory with shared constructs. It tests prediction shifts, not prediction quality. No old/new correlation significance is used to claim superiority.

Simulation accuracy uses only per-pair eligible, negative-mean legacy 7.5 nm corr1 targets with supported exact topology sequences. R²=1−SSE/SST, RMSE, MAE, signed prediction-minus-target error, Pearson and Spearman are calculated against the simulation target (the same quantity), never pKd. Paired bootstrap differences use identical rows. RDF/adjacent topology matching is contextual evidence, not an embedded generator manifest; full new training target provenance remains incomplete. Physical integrated nm³ outputs are separate.

Seed=20261001. Every resampling confidence interval assumes row-level resampling; shared AHD/RGG constructs reduce effective independence. Results are descriptive and exploratory, not decisive model selection. Short plot axes omit implementation details; companion captions explain the legacy target convention.
''')


if __name__=='__main__':
    main()
