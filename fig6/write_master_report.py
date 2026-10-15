"""Write a short results report from current master-derived statistics only."""
import json
import numpy as np
import pandas as pd
from validation_master import csv_write


def show(value):
    return f'{value:.3f}' if pd.notna(value) and np.isfinite(value) else 'unavailable'


def write_report(master, out, figures_generated):
    summary = json.loads((out / 'master_summary.json').read_text())
    selection = json.loads((out / 'source_selection.json').read_text())
    lines = ['# Master-based validation results', '',
        f"Combined {summary['pairs']} unique pair identities: {summary['experiments']} supplied experimental annotations, "
        f"{summary['predictions']} finite final predictions, {summary['simulations']} simulated B22 values and {summary['features']} current feature rows. "
        f"{summary['accuracy_eligible']} pairs pass the existing simulation comparison audit bound to selected values and sequences.", '',
        '## Sources', '']
    for source in selection['sources']:
        lines.append(f"- {source['role']}: `{source['path']}` ({source['status']}; SHA256 `{source.get('sha256')}`).")
    if selection['missing_sources']:
        lines.extend(['', 'Missing sources: ' + '; '.join(selection['missing_sources']) + '.'])
    reports = out / 'analysis_reports'
    if figures_generated and (reports / 'experimental_old_new_summary.csv').exists():
        changes = pd.read_csv(reports / 'experimental_old_new_summary.csv')
        s = changes[changes['set'].eq('experimental pairs')].iloc[0]
        lines.extend(['', '## Prediction changes across experimental pairs', '',
            f"Matched n={int(s.n)}; mean new-minus-old {show(s.mean_delta)} "
            f"(95% paired row-bootstrap CI {show(s.mean_delta_CI95_low)} to {show(s.mean_delta_CI95_high)}), "
            f"median {show(s.median_delta)}, mean absolute change {show(s.mean_absolute_change)}; "
            f"range {show(s.minimum_delta)} to {show(s.maximum_delta)}. Old/new Pearson r={show(s.Pearson_r)}, Spearman rho={show(s.Spearman_rho)}. "
            'These describe prediction changes, not better accuracy.', '',
            '### Partner summaries', ''])
        for _, row in changes[changes['set'].str.startswith('partner_display:')].iterrows():
            lines.append(f"- {row['set'].split(': ', 1)[-1]}: n={int(row.n)}, mean change {show(row.mean_delta)}.")
        rows_path = reports / 'master_experimental_named_paired_predictions_all_rows.csv'
        if rows_path.exists():
            rows = pd.read_csv(rows_path)
            for _, row in rows.reindex(rows.delta.abs().sort_values(ascending=False).index).head(5).iterrows():
                lines.append(f"- Large change: {row.pair_display_name}, new-minus-old {show(row.delta)}.")
        lines.extend(['', '## Agreement with Kd and outcome separation', ''])
        kd_file = reports / 'paired_Kd_correlation_comparison.csv'
        if kd_file.exists():
            for _, row in pd.read_csv(kd_file).iterrows():
                lines.append(f"- {row.metric}: n={int(row.n)}, old {show(row.old)}, new {show(row.new)}, paired difference {show(row.difference_new_minus_old)} "
                             f"(95% CI {show(row.CI95_low)} to {show(row.CI95_high)}).")
        auc_file = reports / 'paired_outcome_effect_differences.csv'
        if auc_file.exists():
            for _, row in pd.read_csv(auc_file).iterrows():
                lines.append(f"- {row.comparison}, {row.metric}: difference {show(row.estimate)} (95% CI {show(row.CI95_low)} to {show(row.CI95_high)}).")
        lines.append('Use matched effect/correlation differences, not the two models’ separate p-values, to assess direction. '
                     'Intervals and permutation p-values are exploratory because constructs are shared. LB is excluded from binary Kd fit/NB AUROC.')
        complete_file = reports / 'complete_WT_shuffle_summary.csv'
        if complete_file.exists():
            complete = pd.read_csv(complete_file)
            n = int(complete.loc[complete['set'].eq('complete'), 'n'].iloc[0])
            lines.extend(['', '## Complete same-construct model comparison', '',
                f'{n} pairs have finite old/new predictions and matching historical/current sequences. Newly recovered same-construct old predictions remain separate from archived panel values. Neither model is retrained or recalibrated.'])
            for _, row in complete[~complete['set'].eq('complete')].iterrows():
                lines.append(f"- {row['set']}: n={int(row.n)}, mean new-minus-old {show(row.mean_delta)}.")
        simulation_file = reports / 'paired_simulation_accuracy.csv'
        if simulation_file.exists():
            lines.extend(['', '## Simulation accuracy and remaining limitations', ''])
            for _, row in pd.read_csv(simulation_file).iterrows():
                if row.predictor in ['old', 'new']:
                    lines.append(f"- {row.predictor}: n={int(row.n)}, R²={show(row.R_squared)}, RMSE={show(row.RMSE)}, MAE={show(row.MAE)}, "
                                 f"bias={show(row.bias)}, Pearson r={show(row.Pearson_r)}, Spearman rho={show(row.Spearman_rho)}.")
            lines.append('Both predictors use identical eligible rows. Values outside the eligible subset are still retained in the master. '
                         'Topology/RDF linkage is contextual, and complete model training-target lineage remains unresolved; excluded rows cannot support an accuracy claim.')
    else:
        lines.extend(['', 'Expanded comparative statistics were not regenerated in this invocation. No stale diagnostic tables are interpreted as current results.'])
    lines.extend(['', '## Figure exports', '',
        '`paper_figures/` contains five current-model panels and their exact rows, statistics, captions and overview when plotting is enabled. '
        'Analysis membership and exclusion reasons are in `analysis_reports/`. '
        'Automatic checks do not certify visual layout; inspect the regenerated PNGs locally at publication size. '
        'No agent visual inspection is claimed for this locally executed workflow.'])
    (out / 'MASTER_RESULTS_REPORT.md').write_text('\n'.join(lines) + '\n')
