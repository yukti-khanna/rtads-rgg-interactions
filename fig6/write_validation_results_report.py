#!/usr/bin/env python3
"""Write a concise results report from actual outputs; never invent results."""
import json
import pandas as pd
from workflow_common import ROOT, OUT


def main():
    qc = json.loads((OUT / 'qc/prediction_update_summary.json').read_text())
    metric = pd.read_csv(ROOT / 'prediction_diagnostics/metric_comparison.csv')
    delta = pd.read_csv(ROOT / 'prediction_diagnostics/paired_bootstrap_differences.csv')
    bfile = OUT / 'b22/numerical_convention_qc.json'
    b = json.loads(bfile.read_text()) if bfile.exists() else None
    agreement = pd.read_csv(OUT / 'qc/historical_prediction_agreement.csv')
    lines = ['# Validation update results', '',
        f"Frozen supplied run: {qc['predictions']} predictions; {qc['rejected']} rejected rows. "
        f"Historical experiments: {qc['experimental_rows']} rows, {qc['positive_numerical_Kd']} positive numerical Kd values. "
        f"Matched experimental rows: {qc['matched_experiments']}; untested predicted pairs: {qc['untested_pairs']}.", '',
        'Experimental measurements, errors, outcomes and notes were preserved. New final/base columns are copied directly '
        'from predictions_minimal.csv. No prediction recalibration, target re-transformation or model fitting occurred.', '',
        'Old-panel predictions are Table_S8::Pred_log10_negB22_hybrid. Other historical tables are audited separately:', '']
    for filename, d in agreement.groupby('source'):
        lines.append(f"- {filename}: n={len(d)}, max absolute difference from panel={d.difference_from_old_panel.abs().max():.6g}.")
    lines += ['', 'Primary paired comparison (all eligible fitted pairs, including FOXO4-G3BP1):', '',
              metric[metric.analysis.eq('all_eligible_primary')].to_string(index=False), '',
              'Paired new-minus-old differences with descriptive row-bootstrap intervals:', '',
              delta[delta.analysis.eq('all_eligible_primary')].to_string(index=False), '',
              'Small n and shared constructs limit inference. No decisive superiority claim is made. '
              'Exact pair identities and sequence-stratified results are in prediction_diagnostics.', '',
              f"Old/new sequence status counts: {qc['sequence_status_counts']}. Equal lengths do not establish exact old-run identity. "
              'Nearby historical FASTAs are candidate evidence only unless explicitly linked with --panel-fasta.', '']
    if b:
        lines += [f"RDF files: {b['rdf_files']}; failed QC: {b['failed_files']}; overlapping historical pairs: {b['overlapping_pairs']}. "
                  f"All historical values reproduced: {b['all_historical_values_reproduced']}; "
                  f"reference helpers verified: {b['reference_helper_verified']}.", '',
                  'Legacy sums retain the missing-dr convention and centre-only tail span. '
                  'Integrated outputs include dr and full-window endpoints, separately at 7.5 nm and L/2, in nm^3. '
                  'The principal transform is log10(-mean corrected B22). Nonnegative means remain untransformed. '
                  'No simulated experimental Kd is calculated.', '']
    else:
        lines += ['B22 calculation has not been run; simulation results remain unavailable.', '']
    lines += ['Proposed manuscript/legend corrections (DOCX files unchanged):', '',
              '- State that frozen predictions are in the historical corr1 sum target scale; avoid assigning physical nm³ to that scale.',
              '- Reconcile the manuscript L/2 integral with historical 7.5 nm sums and the tail-span artifact; name each calculation explicitly.',
              '- Report all eligible fitted pairs as primary; label any FOXO4-G3BP1 exclusion as sensitivity only.',
              '- Describe fit lines as descriptive; do not claim a calibrated Kd predictor or use direct RMSE/R² across pKd and log10(-B22).',
              '- Preserve NB, LB, No Kd and not tested as separate annotations without an invented affinity order.',
              '- Verify Figure 5C/5D against the assembled current layout before assigning letters. Use descriptive supplementary filenames; S7/S8 are already occupied.', '',
              'Unresolved provenance: exact original panel prediction FASTA linkage; exact supplied new-model training RDF/target provenance; '
              'independent confirmation of RDF construct sequences and replica/job linkage from trajectories. '
              'Numerical reproduction does not resolve sequence identity. Inspect historical reproduction CSVs for discrepancies.', '',
              'Visual verification is pending local inspection of every PNG. Editable SVG text and 600 dpi PNG exports are prepared by the plotting script.']
    (OUT / 'RESULTS_REPORT.md').write_text('\n'.join(lines) + '\n')
    print(f'Wrote {OUT / "RESULTS_REPORT.md"}')


if __name__ == '__main__':
    from write_expanded_report import main as expanded_main
    expanded_main()
