#!/usr/bin/env python3
"""Compare predicted attraction for final AHD candidates with/without AD matches.
Run inside the existing Figure 5 Python environment; no model fitting or inference.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata, t as student_t

FIG5 = Path(__file__).resolve().parents[1]
FIG2 = FIG5.parent / 'fig2'
SCORE = 'final_hybrid_prediction'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results')
    p.add_argument('--bootstrap', type=int, default=5000)
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--omit-development', '--deployment-only', dest='deployment_only', action='store_true',
                   help='Omit frozen development rows; retain unsimulated and held-out predictions.')
    p.add_argument('--rg-list', type=Path,
                   help='Optional file with one RGG_id per line, to restrict the common partner pool.')
    return p.parse_args()


def rank_tests(values, matched, clusters):
    """Nominal MWU plus a rank-regression test with TF-cluster robust variance."""
    y = rankdata(values, method='average')
    x = np.column_stack([np.ones(len(y)), matched.astype(float)])
    inv = np.linalg.inv(x.T @ x)
    beta = inv @ x.T @ y
    residual = y - x @ beta
    unique = np.unique(clusters)
    meat = np.zeros((2, 2))
    for cluster in unique:
        score = x[clusters == cluster].T @ residual[clusters == cluster]
        meat += np.outer(score, score)
    n, g = len(y), len(unique)
    if g < 2 or n <= 2:
        raise ValueError('Insufficient independent TFs for the cluster-robust rank test.')
    covariance = inv @ meat @ inv * (g / (g - 1)) * ((n - 1) / (n - 2))
    se = np.sqrt(max(0., covariance[1, 1]))
    cluster_p = float(2 * student_t.sf(abs(beta[1] / se), df=g - 1)) if se else (1. if beta[1] == 0 else 0.)
    mwu = mannwhitneyu(values[matched], values[~matched], alternative='two-sided', method='asymptotic')
    return {'MWU_AHD_level_nominal_p': float(mwu.pvalue),
            'TF_cluster_robust_rank_test_p': cluster_p,
            'TF_cluster_robust_rank_test_df': g - 1}


def roc_curve(values, matched):
    order = np.argsort(-values, kind='stable')
    scores, labels = values[order], matched[order].astype(int)
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp, fp = np.cumsum(labels)[ends], np.cumsum(1 - labels)[ends]
    return pd.DataFrame({'threshold': np.r_[np.inf, scores[ends]],
                         'false_positive_rate': np.r_[0., fp / (~matched).sum()],
                         'true_positive_rate': np.r_[0., tp / matched.sum()]})


def fasta(path):
    result = {}
    name = None
    for line in path.read_text().splitlines():
        if line.startswith('>'):
            name = line[1:].split()[0]
            if name in result:
                raise ValueError(f'Duplicate FASTA identifier: {name}')
            result[name] = ''
        elif line.strip():
            if name is None:
                raise ValueError('Sequence before FASTA header')
            result[name] += line.strip().upper()
    return result


def weighted_median(values, weights):
    """Median of integer-weighted observations, including the even-size case."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    n = int(weights.sum())
    if not n:
        return np.nan
    cumulative = np.cumsum(weights)
    left = values[np.searchsorted(cumulative, (n - 1) // 2 + 1)]
    right = values[np.searchsorted(cumulative, n // 2 + 1)]
    return float((left + right) / 2)


def histogram_overlap(a, b, bins):
    if max(a.max(), b.max()) == min(a.min(), b.min()):
        return 1.
    edges = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), bins + 1)
    ha = np.histogram(a, edges)[0] / len(a)
    hb = np.histogram(b, edges)[0] / len(b)
    return float(np.minimum(ha, hb).sum())


def main():
    args = parse_args()
    if args.bootstrap < 1000:
        raise ValueError('Use at least 1000 bootstrap resamples.')
    files = {
        'catalogue_fasta': FIG2 / 'inputs/fasta/pred_tads_frags.fasta',
        'region_correspondence': FIG2 / 'outputs/reviewer_external_AD_recovery/predicted_region_external_support.csv',
        'deployment_predictions': FIG5 / 'outputs/final_hybrid_model/final_predictions/full_landscape_final_hybrid_predictions.csv',
        'heldout_predictions': FIG5 / 'outputs/final_hybrid_model/final_predictions/test_1794_final_hybrid_predictions.csv',
    }
    if not args.deployment_only:
        files['frozen_development_predictions'] = FIG5 / 'outputs/final_hybrid_model/final_predictions/development_917_frozen_model_predictions.csv'
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    sequences = fasta(files['catalogue_fasta'])
    support = pd.read_csv(files['region_correspondence'])
    if not support.region_id.is_unique or set(support.region_id) != set(sequences):
        raise ValueError('Support table does not map exactly to the candidate FASTA.')
    if len(sequences) != 230 or support.protein.nunique() != 190:
        raise ValueError('Expected the final 230-region / 190-TF catalogue.')
    if not support.overlap_external_union.isin([0, 1]).all():
        raise ValueError('Region correspondence must be binary.')
    support['TAD_id'] = support.region_id.str.replace(':', '_', regex=False)
    support['sequence'] = support.region_id.map(sequences)
    support['catalogue_group'] = np.where(support.overlap_external_union.eq(1), 'Matched', 'Unmatched')
    if not support.TAD_id.is_unique:
        raise ValueError('AHD identifier conversion produced duplicates.')

    frames = []
    for key in ['deployment_predictions', 'frozen_development_predictions', 'heldout_predictions']:
        if key in files:
            frame = pd.read_csv(files[key])
            frame['prediction_source'] = key
            frame['partner_cohort'] = 'heldout' if key == 'heldout_predictions' else 'development_landscape'
            frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    if predictions.duplicated(['TAD_id', 'RGG_id']).any():
        raise ValueError('Duplicate AHD–RG pairs across prediction sources; do not average them.')
    if not np.isfinite(predictions[SCORE]).all():
        raise ValueError('Non-finite predicted scores.')
    manifests = predictions.model_manifest_sha256.dropna().unique()
    if len(manifests) != 1 or predictions.model_manifest_sha256.isna().any():
        raise ValueError('Prediction rows do not share one identified frozen model.')
    if not set(predictions.TAD_id) <= set(support.TAD_id):
        raise ValueError('Prediction table contains AHD IDs outside the final catalogue.')
    requested = None
    if args.rg_list:
        files['requested_RG_pool'] = args.rg_list.resolve()
        requested = {line.strip() for line in args.rg_list.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')}
    if predictions.groupby('TAD_id').partner_cohort.nunique().gt(1).any():
        raise ValueError('An AHD spans prediction cohorts; check the source definitions.')
    pools, portions = {}, []
    for cohort, frame in predictions.groupby('partner_cohort'):
        partner_sets = frame.groupby('TAD_id').RGG_id.agg(set)
        pool = set.intersection(*partner_sets.tolist())
        if requested is not None:
            pool &= requested
        if not pool:
            raise ValueError(f'No partners shared within {cohort}; check any --rg-list restriction.')
        portion = frame[frame.RGG_id.isin(pool)].copy()
        if not portion.groupby('TAD_id').size().eq(len(pool)).all():
            raise ValueError(f'The common partner grid is incomplete in {cohort}.')
        pools[cohort] = sorted(pool)
        portions.append(portion)
    kept = pd.concat(portions, ignore_index=True)
    matrix = kept.pivot(index='TAD_id', columns='RGG_id', values=SCORE)
    ahd = support.set_index('TAD_id').loc[matrix.index].copy()
    ahd['partner_cohort'] = kept.groupby('TAD_id').partner_cohort.first()
    ahd['median_score'] = matrix.median(axis=1)
    ahd['q10_score'] = matrix.quantile(.10, axis=1)
    ahd['q90_score'] = matrix.quantile(.90, axis=1)
    ahd['maximum_score'] = matrix.max(axis=1)
    group = ahd.overlap_external_union.eq(1).to_numpy()
    a = ahd.loc[~group, 'median_score'].to_numpy()
    b = ahd.loc[group, 'median_score'].to_numpy()
    if min(len(a), len(b)) < 2:
        raise ValueError('At least two AHDs from each group are required.')

    # One summary per AHD; resample whole TFs jointly, keeping mixed-status TFs.
    proteins = sorted(ahd.protein.unique())
    index = {p: i for i, p in enumerate(proteins)}
    clusters = np.array([index[p] for p in ahd.protein])
    signs = np.sign(a[:, None] - b[None, :])
    rng = np.random.default_rng(args.seed)
    boot_delta, boot_difference = [], []
    for _ in range(args.bootstrap):
        counts = np.bincount(rng.integers(len(proteins), size=len(proteins)), minlength=len(proteins))
        wa, wb = counts[clusters[~group]], counts[clusters[group]]
        if wa.sum() == 0 or wb.sum() == 0:
            continue
        boot_delta.append(float(wa @ signs @ wb / (wa.sum() * wb.sum())))
        boot_difference.append(weighted_median(a, wa) - weighted_median(b, wb))
    if len(boot_delta) < .99 * args.bootstrap:
        raise ValueError('Too many bootstrap samples lack one group; investigate group coverage.')
    delta = float(signs.mean())
    tagged = kept.merge(ahd[['catalogue_group']], left_on='TAD_id', right_index=True, validate='many_to_one')
    partner = tagged.groupby(['partner_cohort', 'RGG_id', 'catalogue_group'])[SCORE].median().unstack()
    partner = partner.rename(columns={'Unmatched': 'unmatched_median_score', 'Matched': 'matched_median_score'})
    partner['unmatched_minus_matched'] = partner.unmatched_median_score - partner.matched_median_score
    reference_band = np.quantile(b, [.1, .9])
    omitted = support[~support.TAD_id.isin(matrix.index)].copy()
    result = {
        'score_definition': 'Frozen-model log10(-B22); higher predicts stronger attraction, not Kd.',
        'group_definition': 'Direct candidate-region eligible BLAST correspondence, not archived 108/82 protein membership.',
        'catalogue_unmatched_AHDs': int(support.overlap_external_union.eq(0).sum()),
        'catalogue_matched_AHDs': int(support.overlap_external_union.eq(1).sum()),
        'included_unmatched_AHDs': len(a), 'included_matched_AHDs': len(b),
        'included_TFs': len(proteins), 'omitted_AHDs': len(omitted),
        'common_RG_partners_by_cohort': {key: len(value) for key, value in pools.items()}, 'comparison_pairs': len(kept),
        'source_row_counts': {f.prediction_source.iloc[0]: len(f) for f in frames},
        'unmatched_score_median': float(np.median(a)),
        'matched_score_median': float(np.median(b)),
        'median_difference_unmatched_minus_matched': float(np.median(a) - np.median(b)),
        'median_difference_CI95': np.quantile(boot_difference, [.025, .975]).tolist(),
        'cliffs_delta_unmatched_minus_matched': delta,
        'cliffs_delta_CI95': np.quantile(boot_delta, [.025, .975]).tolist(),
        'AUROC_matched_positive': (1 - delta) / 2,
        'AUROC_TF_bootstrap_CI95': ((1 - np.quantile(boot_delta, [.975, .025])) / 2).tolist(),
        'probability_unmatched_exceeds_matched_half_ties': (delta + 1) / 2,
        'histogram_overlap_by_bin_count': {str(n): histogram_overlap(a, b, n) for n in [10, 20, 40]},
        'matched_score_10_90_band': reference_band.tolist(),
        'fraction_unmatched_inside_matched_10_90_band': float(((a >= reference_band[0]) & (a <= reference_band[1])).mean()),
        'fraction_RG_partners_with_lower_unmatched_group_median': float(partner.unmatched_minus_matched.lt(0).mean()),
        'valid_bootstrap_resamples': len(boot_delta), 'seed': args.seed,
        'model_manifest_sha256': str(manifests[0]),
        'source_files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()},
    }
    result.update(rank_tests(ahd.median_score.to_numpy(), group, clusters))
    cohort_rows = []
    for cohort, frame in ahd.groupby('partner_cohort'):
        ca = frame.loc[frame.catalogue_group.eq('Unmatched'), 'median_score'].to_numpy()
        cb = frame.loc[frame.catalogue_group.eq('Matched'), 'median_score'].to_numpy()
        row = {'partner_cohort': cohort, 'unmatched_AHDs': len(ca), 'matched_AHDs': len(cb), 'RG_partners': len(pools[cohort])}
        if len(ca) >= 2 and len(cb) >= 2:
            row.update(AUROC_matched_positive=float((1 - np.sign(ca[:, None] - cb[None, :]).mean()) / 2),
                       unmatched_median=float(np.median(ca)), matched_median=float(np.median(cb)))
            row.update(rank_tests(frame.median_score.to_numpy(), frame.catalogue_group.eq('Matched').to_numpy(), frame.protein.to_numpy()))
        cohort_rows.append(row)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    ahd.reset_index().to_csv(out / 'AHD_level_scores.csv', index=False)
    partner.reset_index().to_csv(out / 'RG_partner_comparison.csv', index=False)
    omitted.to_csv(out / 'omitted_AHDs.csv', index=False)
    pd.DataFrame([{'partner_cohort': key, 'RGG_id': rg} for key, pool in pools.items() for rg in pool]).to_csv(out / 'common_RG_pool.csv', index=False)
    pd.DataFrame(cohort_rows).to_csv(out / 'within_dataset_comparisons.csv', index=False)
    roc = roc_curve(ahd.median_score.to_numpy(), group)
    roc.to_csv(out / 'ROC_curve.csv', index=False)
    pd.DataFrame({'cliffs_delta': boot_delta, 'median_difference': boot_difference}).to_csv(out / 'TF_cluster_bootstrap.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
    edges = np.linspace(min(a.min(), b.min()) - .005, max(a.max(), b.max()) + .005, 21)
    for values, label, colour in [(a, 'Without catalogue match', '#D97732'), (b, 'With catalogue match', '#2878A5')]:
        axes[0].hist(values, bins=edges, density=True, alpha=.45, color=colour, label=label)
        x = np.sort(values)
        axes[1].step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=colour, label=label)
    axes[0].set(xlabel='AHD median predicted attraction', ylabel='Density')
    axes[1].set(xlabel='AHD median predicted attraction', ylabel='Cumulative fraction')
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].plot(roc.false_positive_rate, roc.true_positive_rate, label=f'AUROC = {result["AUROC_matched_positive"]:.3f}')
    axes[2].plot([0, 1], [0, 1], color='grey', linewidth=1, linestyle='--')
    axes[2].set(xlabel='False-positive rate', ylabel='True-positive rate')
    axes[2].legend(frameon=False)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(out / 'comparison.png', dpi=200)
    fig.savefig(out / 'comparison.pdf')
    plt.close(fig)
    report = f'''# Predicted attraction by candidate AD catalogue correspondence

Included: {len(a)} unmatched and {len(b)} matched candidate AHDs, collected across unsimulated, simulated-development and held-out prediction sources. Omitted candidates: {len(omitted)}; see omitted_AHDs.csv. Common RG counts within each cohort: {result['common_RG_partners_by_cohort']}.

Median AHD scores: unmatched {np.median(a):.4f}; matched {np.median(b):.4f}.
Unmatched-minus-matched median difference: {result['median_difference_unmatched_minus_matched']:.4f}, protein-bootstrap 95% CI {result['median_difference_CI95']}.
Cliff’s delta: {delta:.3f}, protein-bootstrap 95% CI {result['cliffs_delta_CI95']}.
AUROC (matched candidates positive; higher score predicts matched): {result['AUROC_matched_positive']:.3f}, protein-bootstrap 95% CI {result['AUROC_TF_bootstrap_CI95']}.
TF-cluster robust rank-test p: {result['TF_cluster_robust_rank_test_p']:.4g}.
Nominal AHD-level Mann–Whitney p: {result['MWU_AHD_level_nominal_p']:.4g}; this nominal test does not account for multiple AHDs per TF, so prioritise the cluster-aware results.
Probability an unmatched AHD exceeds a matched AHD, counting ties as half: {(delta + 1) / 2:.1%}.
Histogram overlap (20 bins): {result['histogram_overlap_by_bin_count']['20']:.1%}; inspect the 10/40-bin sensitivities in summary.json.
Fraction of unmatched AHDs within the matched group’s 10th–90th percentile score band: {result['fraction_unmatched_inside_matched_10_90_band']:.1%}.

## Interpretation

A negative difference/delta indicates weaker predicted attraction in the unmatched group; overlap measures describe how much the distributions still share. Neither a small difference nor high overlap proves equivalence, and neither establishes experimental binding. No binding cutoff is applied. Histogram overlap is bin-dependent and descriptive.

Each AHD score is its median over available predictions, with an identical RG pool within each cohort. The combined comparison uses all source cohorts as requested; development-landscape and held-out sequences have different RG partner pools, so inspect within_dataset_comparisons.csv before attributing a combined difference to AHD catalogue correspondence. The primary unit is an AHD; 5,000 resamples by default draw whole TFs jointly and preserve multiple and mixed-status AHDs. Pair rows and RG-partner medians are not treated as independent biological replicates. AUROC here measures separation of catalogue-correspondence groups, not prediction of binding or model validation. It is not flipped automatically to exceed 0.5.

The default combines deployment, frozen in-sample development, and held-out predictions from the same identified model. It does not use OOF scores, retrain models or recalibrate predictions. This is a comparison of model predictions, not independent validation: charge helped determine the predictions themselves. Candidates absent from these sources are excluded explicitly. --omit-development omits development rows while retaining held-out rows; inspect its resulting within-cohort partner counts.

The source candidate-region correspondence is BLAST-based (same protein, identity ≥60%, shorter-sequence coverage ≥80%, E≤0.001), not a functional activation assay or the archived 108/82 combined protein-membership definition. Source hashes and precise coverage are in summary.json.
'''
    (out / 'report.md').write_text(report)
    print(report)
    print(f'Results saved to {out}')


if __name__ == '__main__':
    main()
