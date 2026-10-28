#!/usr/bin/env python3
"""Compare out-of-fold development and held-out performance within fixed −NCP bins."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
FIG4_ROOT = ROOT.parent / 'fig4'
PREDICTIONS = ROOT / 'outputs' / 'final_hybrid_model' / 'final_predictions'
BIN_SOURCE = (FIG4_ROOT / 'reviewer_analysis' / 'training_selection_provenance' /
              'S4D_feature_guided_sampling_coverage' / 'S4D_bin_source.csv')
OUT = ROOT / 'outputs' / 'manuscript' / 'performance_by_minus_NCP_bin'
FIGURE_OUT = ROOT / 'outputs' / 'manuscript' / 'Figure_S6'
BOOTSTRAPS = 5000
SEED = 20261017


def assign_bins(values: pd.Series, bins: pd.DataFrame) -> np.ndarray:
    lower = bins['bin_lower_exclusive_except_first'].to_numpy(float)
    upper = bins['bin_upper_inclusive'].to_numpy(float)
    x = values.to_numpy(float)
    if np.isnan(x).any() or (x < lower[0]).any() or (x > upper[-1]).any():
        raise ValueError('Some −NCP values fall outside the displayed bins.')
    idx = np.searchsorted(upper, x, side='left')
    if ((x < lower[idx]) & (idx > 0)).any():
        raise ValueError('Some −NCP values were not covered by the displayed bins.')
    return bins['bin_label'].to_numpy()[idx]


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, rng: np.random.Generator):
    n = len(y)
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
    errors = pred[indices] - y[indices]
    rmse = np.sqrt(np.mean(errors ** 2, axis=1))
    mae = np.mean(np.abs(errors), axis=1)
    return np.quantile(rmse, [.025, .975]), np.quantile(mae, [.025, .975])


def build():
    bins = pd.read_csv(BIN_SOURCE).sort_values('bin_id').reset_index(drop=True)
    sampling = bins[[
        'bin_id', 'bin_label', 'candidate_count', 'candidate_fraction',
        'training_count', 'training_fraction',
        'representation_ratio_training_to_candidate_fraction',
        'held_out_count', 'held_out_fraction',
    ]].copy()
    sampling = sampling.rename(columns={
        'candidate_count': 'candidate_pairs_n',
        'candidate_fraction': 'candidate_pairs_fraction',
        'training_count': '917_pairs_n',
        'training_fraction': '917_pairs_fraction',
        'representation_ratio_training_to_candidate_fraction': '917_to_candidate_fraction_ratio',
        'held_out_count': '1794_pairs_n',
        'held_out_fraction': '1794_pairs_fraction',
    })
    sampling['expected_917_n_if_candidate_proportional'] = 917 * sampling['candidate_pairs_fraction']
    sampling['917_minus_expected_n'] = sampling['917_pairs_n'] - sampling['expected_917_n_if_candidate_proportional']
    sampling['917_minus_candidate_share_percentage_points'] = 100 * (sampling['917_pairs_fraction'] - sampling['candidate_pairs_fraction'])
    cohorts = [
        ('917 development (OOF)', 'development_917_OOF_final_hybrid.csv',
         'OOF_final_hybrid_prediction'),
        ('1,794 held-out', 'test_1794_final_hybrid_predictions.csv',
         'final_hybrid_prediction'),
    ]
    rng = np.random.default_rng(SEED)
    rows = []
    audit = {}
    for cohort, filename, prediction_col in cohorts:
        df = pd.read_csv(PREDICTIONS / filename)
        required = {'pair_id', 'minus_NCP', 'simulated_target', prediction_col}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f'{filename} is missing columns: {sorted(missing)}')
        if df['pair_id'].duplicated().any():
            raise ValueError(f'{filename} contains duplicate pair IDs.')
        df['bin_label'] = assign_bins(df['minus_NCP'], bins)
        audit[cohort] = {'n': int(len(df)), 'unique_pair_ids': int(df['pair_id'].nunique()),
                         'bin_counts': {str(k): int(v) for k, v in df['bin_label'].value_counts().items()}}
        for binrow in bins.itertuples(index=False):
            g = df.loc[df['bin_label'].eq(binrow.bin_label)]
            y = g['simulated_target'].to_numpy(float)
            pred = g[prediction_col].to_numpy(float)
            err = pred - y
            if len(y):
                rmse = float(np.sqrt(mean_squared_error(y, pred)))
                mae = float(mean_absolute_error(y, pred))
                rmse_ci, mae_ci = bootstrap_ci(y, pred, rng)
                r2 = float(r2_score(y, pred)) if len(y) >= 2 and np.var(y) > 0 else np.nan
                pearson = float(pearsonr(y, pred).statistic) if len(y) >= 2 and np.std(y) > 0 and np.std(pred) > 0 else np.nan
                spearman = float(spearmanr(y, pred).statistic) if len(y) >= 2 else np.nan
                bias = float(np.mean(err))
            else:
                rmse = mae = r2 = pearson = spearman = bias = np.nan
                rmse_ci = mae_ci = [np.nan, np.nan]
            rows.append({
                'bin_id': int(binrow.bin_id), 'bin_label': binrow.bin_label,
                'bin_lower_exclusive_except_first': float(binrow.bin_lower_exclusive_except_first),
                'bin_upper_inclusive': float(binrow.bin_upper_inclusive),
                'cohort': cohort, 'n': int(len(g)), 'RMSE': rmse,
                'RMSE_CI_2.5': float(rmse_ci[0]), 'RMSE_CI_97.5': float(rmse_ci[1]),
                'MAE': mae, 'MAE_CI_2.5': float(mae_ci[0]), 'MAE_CI_97.5': float(mae_ci[1]),
                'R2': r2, 'Pearson_r': pearson, 'Spearman_rho': spearman,
                'mean_prediction_minus_target': bias,
            })
    result = pd.DataFrame(rows)
    if result.groupby('cohort')['n'].sum().to_dict() != {c[0]: audit[c[0]]['n'] for c in cohorts}:
        raise RuntimeError('Bin counts do not sum to cohort totals.')
    development_counts = result[result['cohort'].eq(cohorts[0][0])].sort_values('bin_id')['n'].to_numpy()
    heldout_counts = result[result['cohort'].eq(cohorts[1][0])].sort_values('bin_id')['n'].to_numpy()
    if not np.array_equal(development_counts, sampling['917_pairs_n'].to_numpy()):
        raise RuntimeError('The 917 prediction bin counts do not match the S4D sampling audit.')
    if not np.array_equal(heldout_counts, sampling['1794_pairs_n'].to_numpy()):
        raise RuntimeError('The 1,794 prediction bin counts do not match the S4D sampling audit.')
    id_sets = []
    for _, filename, _ in cohorts:
        id_sets.append(set(pd.read_csv(PREDICTIONS / filename, usecols=['pair_id'])['pair_id']))
    overlap = len(id_sets[0] & id_sets[1])
    audit['development_test_pair_id_overlap'] = overlap
    if overlap:
        raise ValueError(f'Development and held-out pair IDs overlap: {overlap}')

    rmse_by_bin = result.pivot(index='bin_id', columns='cohort', values='RMSE').sort_index()
    mae_by_bin = result.pivot(index='bin_id', columns='cohort', values='MAE').sort_index()
    dev_weights = sampling['917_pairs_n'].to_numpy(float) / 917
    test_weights = sampling['1794_pairs_n'].to_numpy(float) / 1794
    overall_rows = []
    for cohort, filename, prediction_col in cohorts:
        df = pd.read_csv(PREDICTIONS / filename)
        err = df[prediction_col].to_numpy(float) - df['simulated_target'].to_numpy(float)
        cohort_rmse = float(np.sqrt(np.mean(err ** 2)))
        cohort_mae = float(np.mean(np.abs(err)))
        overall_rows.append({
            'cohort': cohort, 'n': int(len(df)), 'observed_mix_RMSE': cohort_rmse,
            'observed_mix_MAE': cohort_mae,
            'RMSE_weighted_to_917_bin_mix': float(np.sqrt(np.sum(dev_weights * rmse_by_bin[cohort].to_numpy() ** 2))),
            'RMSE_weighted_to_1794_bin_mix': float(np.sqrt(np.sum(test_weights * rmse_by_bin[cohort].to_numpy() ** 2))),
            'MAE_weighted_to_917_bin_mix': float(np.sum(dev_weights * mae_by_bin[cohort].to_numpy())),
            'MAE_weighted_to_1794_bin_mix': float(np.sum(test_weights * mae_by_bin[cohort].to_numpy())),
        })
    cohort_summary = pd.DataFrame(overall_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    FIGURE_OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / 'performance_by_minus_NCP_bin.csv', index=False, float_format='%.8g')
    sampling.to_csv(OUT / 'sampling_representation_by_minus_NCP_bin.csv', index=False, float_format='%.8g')
    cohort_summary.to_csv(OUT / 'cohort_summary.csv', index=False, float_format='%.8g')
    (OUT / 'audit.txt').write_text(
        'Bin-wise performance comparison\n'
        'Bins match the attached S4D panel: each interval is (lower, upper], except the first includes its lower bound.\n'
        'Sampling comparison uses the S4D training-eligible candidate pool (192,303 pairs), selected development sample (917 pairs), and held-out sample (1,794 pairs).\n'
        'The representation ratio is the 917-pair share divided by the candidate-pool share; it describes sampling coverage and does not imply causation.\n'
        'The cohort summary reports observed error and errors reweighted to each cohort’s bin mix; reweighting is descriptive and only adjusts across these coarse bins.\n'
        'Development metrics use out-of-fold predictions for the 917 development pairs.\n'
        'Held-out metrics use final frozen-hybrid predictions for the 1,794 test pairs.\n'
        'Error = prediction − simulated target; intervals are pair-resampling percentile 95% confidence intervals.\n'
        'Intervals are descriptive and do not adjust for dependence from repeated protein-region identities; no between-cohort significance tests were run.\n'
        f'Bootstrap replicates: {BOOTSTRAPS}; seed: {SEED}.\n'
        f'Pair-ID overlap between cohorts: {overlap}.\n'
        + '\n'.join(f'{cohort}: n={info["n"]}; per-bin counts={info["bin_counts"]}' for cohort, info in audit.items() if isinstance(info, dict))
        + '\n'
    )

    colors = {'917 development (OOF)': '#0878B5', '1,794 held-out': '#E17C05'}
    plot_data = pd.read_csv(OUT / 'performance_by_minus_NCP_bin.csv')
    x = np.arange(len(bins), dtype=float)
    offset = .13
    with plt.rc_context({
        'font.family': 'Arial',
        'font.size': 7,
        'axes.labelsize': 7,
        'axes.linewidth': .7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'axes.unicode_minus': False,
    }):
        fig, ax = plt.subplots(figsize=(9, 5))
        count_ax = ax.twinx()
        count_ax.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)
        count_width = .22
        for j, (cohort, _, _) in enumerate(cohorts):
            sub = plot_data[plot_data['cohort'].eq(cohort)].sort_values('bin_id')
            if len(sub) != len(bins):
                raise RuntimeError(f'{cohort} does not have one row per −NCP bin.')
            xpos = x + (-offset if j == 0 else offset)
            count_ax.bar(xpos, sub['n'].to_numpy(float), width=count_width,
                         color=colors[cohort], alpha=.18, linewidth=0, zorder=0)
            y = sub['RMSE'].to_numpy(float)
            lo = sub['RMSE_CI_2.5'].to_numpy(float)
            hi = sub['RMSE_CI_97.5'].to_numpy(float)
            yerr = np.vstack([y - lo, hi - y])
            ax.errorbar(
                xpos, y, yerr=yerr, fmt='o', linestyle='none',
                color=colors[cohort], ecolor=colors[cohort],
                markersize=4.0, markeredgewidth=.45, elinewidth=.9,
                capsize=2.0, capthick=.9, label=cohort, zorder=3,
            )

        tick_labels = [str(v).replace('-', '−') for v in bins['bin_label']]
        ax.set_xticks(x, tick_labels, rotation=40, ha='right')
        ax.set_xlim(-.48, len(bins) - .52)
        upper = float(plot_data['RMSE_CI_97.5'].max())
        ax.set_ylim(0, upper * 1.16)
        ax.set_yticks(np.arange(0, .201, .05))
        ax.set_ylabel(r'Prediction RMSE, $\log_{10}(-B_{22})$')
        ax.set_xlabel('−NCP bin', labelpad=3)
        count_ax.set_ylim(0, float(sampling[['917_pairs_n', '1794_pairs_n']].to_numpy().max()) * 1.15)
        count_ax.set_ylabel('Number of pairs per bin')
        count_ax.tick_params(axis='y', width=.7, length=2.5, pad=2, labelsize=7)
        count_ax.spines['top'].set_visible(False)
        ax.grid(axis='y', color='#dddddd', linewidth=.55, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', width=.8, length=3, pad=3)
        ax.legend(frameon=False, ncol=2, loc='upper right', handletextpad=.35,
                  columnspacing=1.2, borderaxespad=.15)
        ax.text(-.065, 1.015, 'F', transform=ax.transAxes, fontsize=7,
                fontweight='bold', va='bottom', ha='left', clip_on=False)
        fig.subplots_adjust(left=.10, right=.90, bottom=.26, top=.93)

        # All PDF plot elements remain vector. The PNG export is 600 dpi at the
        # declared 9 × 5 inch final size; SVG keeps editable vector text.
        fig.savefig(OUT / 'performance_by_minus_NCP_bin.svg')
        fig.savefig(OUT / 'performance_by_minus_NCP_bin.png', dpi=600)
        for ext, kwargs in [('pdf', {'dpi': 600}), ('svg', {}), ('png', {'dpi': 600})]:
            fig.savefig(FIGURE_OUT / f'FigureS6F_performance_by_minus_NCP_bin.{ext}',
                        **kwargs)
        plt.close(fig)
    print(result[['bin_label','cohort','n','RMSE','MAE','R2','mean_prediction_minus_target']].to_string(index=False))
    print('\nS4D sampling representation (selected share / eligible-candidate share):')
    print(sampling[['bin_label','candidate_pairs_n','917_pairs_n','1794_pairs_n',
                    '917_to_candidate_fraction_ratio','917_minus_candidate_share_percentage_points']].to_string(index=False))
    print('\nOverall and bin-mix-standardized RMSE/MAE:')
    print(cohort_summary.to_string(index=False))
    print(f'Wrote outputs to {OUT}')
    return result, sampling


if __name__ == '__main__':
    build()
