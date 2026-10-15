#!/usr/bin/env python3
"""Same-construct complete-set, WT/shuffle and simulation performance changes."""
import numpy as np
import pandas as pd
from scipy import stats
from workflow_common import OUT, DIAG
from expanded_validation_analysis import paired_summary, interval, SEED
from compare_old_new_predictions import NEW


def main():
    rng = np.random.default_rng(SEED)
    source = DIAG / 'historical_model/same_construct_predictions.csv'
    if not source.exists():
        return
    d = pd.read_csv(source)
    old = 'Old_same_construct_final'
    d['delta'] = d[NEW] - d[old]
    d['old_rank'] = d[old].rank(ascending=False)
    d['new_rank'] = d[NEW].rank(ascending=False)
    d['rank_change'] = d.new_rank - d.old_rank
    d.to_csv(DIAG / 'complete_same_construct_changes.csv', index=False)
    rows = [paired_summary(d, 'complete_same_construct', rng, old=old)]
    for shuffle, group in d.groupby('is_shuffle'):
        rows.append(paired_summary(group, 'shuffled' if shuffle else 'WT', rng, old=old))
    for partner, group in d.groupby('RGG_name'):
        if len(group) >= 3:
            rows.append(paired_summary(group, 'RGG:' + partner, rng, old=old))
    pd.DataFrame(rows).to_csv(DIAG / 'complete_same_construct_summary.csv', index=False)
    indexed = d.set_index('pair_id')
    changes = []
    for _, row in d[d.is_shuffle].iterrows():
        parent = row.RGG_parent + '__' + row.AHD_parent
        if parent not in indexed.index:
            continue
        wt = indexed.loc[parent]
        changes.append(dict(pair_id=row.pair_id, parent_pair_id=parent,
            AHD_name=row.AHD_name, RGG_name=row.RGG_name,
            old_shuffle_minus_WT=row[old] - wt[old], new_shuffle_minus_WT=row[NEW] - wt[NEW],
            difference_in_shuffle_effect=(row[NEW] - wt[NEW]) - (row[old] - wt[old])))
    changes = pd.DataFrame(changes)
    changes.to_csv(DIAG / 'matched_WT_shuffle_changes.csv', index=False)
    if not changes.empty:
        pd.DataFrame([dict(n=len(changes), mean_old_shuffle_effect=changes.old_shuffle_minus_WT.mean(),
            mean_new_shuffle_effect=changes.new_shuffle_minus_WT.mean(),
            mean_difference=changes.difference_in_shuffle_effect.mean(),
            interpretation='Same exact WT/shuffle constructs in both models; no inferred experimental affinity ordering')]).to_csv(DIAG / 'WT_shuffle_summary.csv', index=False)
    sim = OUT / 'b22/paired_simulation_accuracy_source.csv'
    if sim.exists():
        x = pd.read_csv(sim)
        y = x.log10_neg_mean_B22.to_numpy()
        a, b = x[old].to_numpy(), x[NEW].to_numpy()
        def metric(p, yy):
            e = p - yy
            denominator = ((yy - yy.mean())**2).sum()
            return {'R_squared': 1 - np.sum(e**2) / denominator if denominator > 0 else np.nan,
                    'RMSE': np.sqrt(np.mean(e**2)), 'MAE': abs(e).mean(), 'mean_signed_error': e.mean(),
                    'Pearson_r': stats.pearsonr(p, yy).statistic if np.ptp(p)>0 and np.ptp(yy)>0 else np.nan,
                    'Spearman_rho': stats.spearmanr(p, yy).statistic if np.ptp(p)>0 and np.ptp(yy)>0 else np.nan}
        observed_a, observed_b = metric(a, y), metric(b, y)
        draws = {key: [] for key in observed_a}
        for _ in range(10000):
            ix = rng.integers(0, len(y), len(y))
            aa, bb = metric(a[ix], y[ix]), metric(b[ix], y[ix])
            for key in draws:
                draws[key].append(bb[key] - aa[key])
        pd.DataFrame([dict(metric=key, n=len(y), new_minus_old=observed_b[key] - observed_a[key],
            method='paired identical-row bootstrap, exploratory shared-construct subset', **interval(values))
            for key, values in draws.items()]).to_csv(OUT / 'b22/paired_simulation_accuracy_differences.csv', index=False)


if __name__ == '__main__':
    main()
