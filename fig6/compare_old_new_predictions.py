#!/usr/bin/env python3
"""Paired descriptive old/new prediction comparison against measured pKd.

Primary analysis includes every eligible Kd-fit pair, including FOXO4-G3BP1.
Both predictors are log10(-B22), not calibrated Kd predictors; no direct RMSE
or predictive R^2 is computed. Bootstrap intervals are descriptive row-based
intervals; shared constructs make rows dependent and reduce effective n.
"""
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from workflow_common import ROOT, inside, unique, pkd, write_json

OLD = 'Pred_log10_negB22_hybrid'
NEW = 'New_final_log10_negB22'


def correlation(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def concordance(x, y):
    """Comparable pKd pairs only. Prediction ties get 1/2; pKd ties excluded.

    Equality is exact, with no invented tolerance/affinity ranking. Returns
    comparable count, prediction ties, outcome ties, concordant and discordant.
    """
    i, j = np.triu_indices(len(x), 1)
    dx, dy = np.asarray(x)[i] - np.asarray(x)[j], np.asarray(y)[i] - np.asarray(y)[j]
    comparable = dy != 0
    ties = (dx == 0) & comparable
    yes = (dx * dy > 0) & comparable
    no = (dx * dy < 0) & comparable
    n = int(comparable.sum())
    return dict(ranking_concordance=float((yes.sum() + .5 * ties.sum()) / n) if n else np.nan,
                comparable_pair_count=n, prediction_ties=int(ties.sum()), pKd_ties=int((~comparable).sum()),
                concordant_pairs=int(yes.sum()), discordant_pairs=int(no.sum()))


def metrics(x, y):
    return dict(pearson_r=correlation(x, y), spearman_rho=correlation(rankdata(x), rankdata(y)), **concordance(x, y))


def paired_bootstrap(old, new, y, iterations, rng):
    differences = {k: [] for k in ['pearson_r', 'spearman_rho', 'ranking_concordance']}
    for _ in range(iterations):
        ix = rng.integers(0, len(y), len(y))
        a, b = metrics(old[ix], y[ix]), metrics(new[ix], y[ix])
        for key in differences:
            differences[key].append(b[key] - a[key])
    rows = []
    a, b = metrics(old, y), metrics(new, y)
    for key, values in differences.items():
        v = np.asarray(values)
        valid = np.isfinite(v)
        ci = np.percentile(v[valid], [2.5, 97.5]) if valid.any() else [np.nan, np.nan]
        rows.append(dict(metric=key, new_minus_old=b[key] - a[key], descriptive_CI95_low=ci[0],
                         descriptive_CI95_high=ci[1], bootstrap_samples=iterations,
                         valid_bootstrap_samples=int(valid.sum()), degenerate_bootstrap_samples=int((~valid).sum())))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', default='updated_validation/experimental_validation_updated.csv')
    ap.add_argument('--output-dir', default='prediction_diagnostics')
    ap.add_argument('--bootstrap', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261001)
    args = ap.parse_args()
    if args.bootstrap < 1:
        raise ValueError('--bootstrap must be positive')
    out = inside(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(inside(args.table))
    unique(d, 'pair_id', 'updated experimental table')
    d['pKd'] = pkd(d.Kd_uM)
    d['pKd_error'] = d.Kd_uM_err / (d.Kd_uM * np.log(10))
    d.loc[~(d.Kd_uM.gt(0) & d.Kd_uM_err.ge(0)), 'pKd_error'] = np.nan
    eligible = (d.Outcome.eq('Kd_fit') & d.Kd_uM.gt(0) & np.isfinite(d.Kd_uM)
                & np.isfinite(d[OLD]) & np.isfinite(d[NEW]))
    d['comparison_included'] = eligible
    d['exclusion_reason'] = np.where(eligible, '', 'requires Kd_fit, finite old/new predictions and positive numerical Kd')
    d['new_minus_old'] = d[NEW] - d[OLD]
    d.to_csv(out / 'all_experimental_pairs.csv', index=False)
    matched = d[eligible].copy()
    matched.to_csv(out / 'matched_pairs.csv', index=False)
    sets = [('all_eligible_primary', matched),
            ('historical_FOXO4_G3BP1_exclusion_sensitivity', matched[~matched.pair_id.eq('g3bp1_rgg__foxo4_cr3')]),
            ('exact_sequence_match_predictor_only', matched[matched.sequence_match_status.eq('exact_sequence_match')]),
            ('changed_construct', matched[matched.sequence_match_status.isin(['changed_construct', 'changed_length_or_missing_historical_input'])]),
            ('sequence_identity_unverified', matched[matched.sequence_match_status.eq('length_match_sequence_unverified')])]
    metric_rows, delta_rows, memberships = [], [], []
    rng = np.random.default_rng(args.seed)
    for label, subset in sets:
        for pid in subset.pair_id:
            memberships.append(dict(analysis=label, pair_id=pid))
        ids = ';'.join(subset.pair_id)
        for name, col in [('old_panel_final', OLD), ('new_frozen_final', NEW)]:
            vals = metrics(subset[col].to_numpy(), subset.pKd.to_numpy()) if len(subset) else dict(pearson_r=np.nan, spearman_rho=np.nan, ranking_concordance=np.nan)
            metric_rows.append(dict(analysis=label, predictor=name, n=len(subset), exact_pair_ids=ids,
                                    status='descriptive' if len(subset) >= 3 else 'insufficient_pairs', **vals))
        if len(subset) >= 3:
            for row in paired_bootstrap(subset[OLD].to_numpy(), subset[NEW].to_numpy(), subset.pKd.to_numpy(), args.bootstrap, rng):
                delta_rows.append(dict(analysis=label, n=len(subset), **row))
    pd.DataFrame(metric_rows).to_csv(out / 'metric_comparison.csv', index=False)
    pd.DataFrame(delta_rows, columns=['analysis', 'n', 'metric', 'new_minus_old', 'descriptive_CI95_low',
        'descriptive_CI95_high', 'bootstrap_samples', 'valid_bootstrap_samples', 'degenerate_bootstrap_samples']).to_csv(out / 'paired_bootstrap_differences.csv', index=False)
    pd.DataFrame(memberships, columns=['analysis', 'pair_id']).to_csv(out / 'analysis_membership.csv', index=False)
    write_json(out / 'comparison_settings.json', dict(seed=args.seed, bootstrap=args.bootstrap,
        primary_n=len(matched), pair_ids=matched.pair_id.tolist(), old_column=OLD, new_column=NEW,
        primary_includes_FOXO4_G3BP1=bool(matched.pair_id.eq('g3bp1_rgg__foxo4_cr3').any()),
        tie_rule='Exact pKd ties excluded; exact prediction ties count 0.5',
        interpretation='Descriptive paired row resampling; small sample and shared constructs; not independent-population or calibrated Kd inference'))
    primary = pd.DataFrame(metric_rows).query("analysis == 'all_eligible_primary'")
    (out / 'COMPARISON_REPORT.md').write_text(
        '# Old/new prediction diagnostic\n\n'
        f'Primary matched n={len(matched)}; identical rows for both predictors. FOXO4-G3BP1 is included whenever eligible.\n\n'
        + primary.to_string(index=False) + '\n\n'
        'Exact pair identities are in matched_pairs.csv and analysis_membership.csv. '
        'pKd=6-log10(Kd_uM); first-order symmetric uncertainty is sigma_Kd/(Kd*ln(10)). '
        'This assumes the reported Kd errors can be used as symmetric small errors; their original interpretation is preserved, not reclassified.\n\n'
        'Row-based paired bootstrap draws the same rows for old, new and pKd. Confidence intervals summarize new-minus-old '
        'metric differences; degenerate draws are counted separately for each metric. Shared AHD and RGG constructs '
        'make observations dependent. These intervals do not establish decisive model superiority or independent generalization.\n\n'
        'The exact-sequence subset is the only predictor-only comparison. Equal lengths alone do not establish that subset; '
        'the default leaves old-panel sequence identity unverified. Changed constructs and uncertain identities are reported separately. '
        'Differences against the older 25-row tables are audited separately and are not substituted for old-panel predictions.\n\n'
        'NB, LB, No Kd and not tested receive no invented affinity values or ordinal ranking. '
        'No direct RMSE or predictive R^2 compares log10(-B22) with pKd. Figure fit lines are descriptive. '
        'The new supplied final predictions remain primary regardless of these diagnostic results.\n')
    print(f'Wrote paired diagnostics for n={len(matched)} to {out}')


if __name__ == '__main__':
    main()
