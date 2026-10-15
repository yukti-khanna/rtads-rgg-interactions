#!/usr/bin/env python3
"""Expanded paired diagnostics. P-values are exploratory: constructs are shared."""
import itertools
import math
import numpy as np
import pandas as pd
from scipy import stats
from workflow_common import ROOT, OUT, DIAG, write_json
from compare_old_new_predictions import OLD, NEW, metrics

SEED = 20261001
B = 19999


def spearman_permutation(x, y, rng, draws=B):
    a, b = stats.rankdata(x), stats.rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    observed = np.dot(a, b) / denom
    if len(x) <= 9:
        count = extreme = 0
        iterator = itertools.permutations(range(len(y)))
        while True:
            batch = list(itertools.islice(iterator, 8192))
            if not batch:
                break
            values = b[np.asarray(batch)] @ a / denom
            extreme += int((abs(values) >= abs(observed) - 1e-12).sum())
            count += len(batch)
        return observed, extreme / count, 'exact two-sided Spearman permutation (labelled rows)', count
    values = np.array([np.dot(a, rng.permutation(b)) / denom for _ in range(draws)])
    return observed, (1 + (abs(values) >= abs(observed) - 1e-12).sum()) / (draws + 1), 'two-sided Monte Carlo Spearman permutation', draws


def interval(values):
    v = np.asarray(values, float)
    finite = v[np.isfinite(v)]
    lo, hi = np.percentile(finite, [2.5, 97.5]) if len(finite) else [np.nan, np.nan]
    return dict(CI95_low=lo, CI95_high=hi, resamples=len(v), degenerate_resamples=int((~np.isfinite(v)).sum()))


def shift_test(delta, rng):
    n, obs = len(delta), abs(np.mean(delta))
    extreme = 0
    if n <= 20:
        count = 2**n
        for start in range(0, count, 8192):
            signs = 2 * ((np.arange(start, min(start + 8192, count))[:, None] >> np.arange(n)) & 1) - 1
            extreme += int((abs(signs @ delta / n) >= obs - 1e-12).sum())
        return extreme / count, 'exact two-sided paired sign-flip mean test', count
    signs = rng.choice([-1, 1], (B, n))
    return (1 + (abs(signs @ delta / n) >= obs - 1e-12).sum()) / (B + 1), 'Monte Carlo paired sign-flip mean test', B


def paired_summary(d, label, rng, old=OLD, new=NEW):
    a, b = d[old].to_numpy(), d[new].to_numpy()
    delta = b - a
    draws = delta[rng.integers(0, len(delta), (10000, len(delta)))]
    ci, median_ci = interval(draws.mean(axis=1)), interval(np.median(draws, axis=1))
    p, test, permutations = shift_test(delta, rng)
    return dict(set=label, n=len(d), pearson_r=stats.pearsonr(a, b).statistic,
                spearman_rho=stats.spearmanr(a, b).statistic, mean_delta=delta.mean(), median_delta=np.median(delta),
                mean_absolute_change=abs(delta).mean(), minimum_delta=delta.min(), maximum_delta=delta.max(),
                mean_delta_CI95_low=ci['CI95_low'], mean_delta_CI95_high=ci['CI95_high'],
                median_delta_CI95_low=median_ci['CI95_low'], median_delta_CI95_high=median_ci['CI95_high'],
                shift_pvalue=p, test=test, permutations=permutations, adjustment='none; exploratory',
                exclusions='only nonfinite old/new; no Kd requirement',
                agreement_concordance=metrics(a, b)['ranking_concordance'])


def kw_test(values, labels, rng):
    categories = list(pd.unique(labels))
    ranks = stats.rankdata(values)
    n = len(values)
    _, counts = np.unique(values, return_counts=True)
    correction = 1 - np.sum(counts**3 - counts) / (n**3 - n)
    masks = np.array([labels == c for c in categories]).T
    sizes = masks.sum(axis=0)
    def h(r):
        sums = np.asarray(r) @ masks
        return (12 / (n * (n + 1)) * np.sum(sums**2 / sizes, axis=-1) - 3 * (n + 1)) / correction
    obs = float(h(ranks))
    permutations = np.array([rng.permutation(ranks) for _ in range(B)])
    pv = (1 + (h(permutations) >= obs - 1e-12).sum()) / (B + 1)
    return obs, pv, (obs - len(categories) + 1) / (n - len(categories))


def binary_effect(a, b):
    differences = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    auc = ((differences > 0).sum() + .5 * (differences == 0).sum()) / differences.size
    return auc, 2 * auc - 1


def pairwise_rank_test(a, b):
    values = np.r_[a, b]
    rank = stats.rankdata(values)
    n, na = len(values), len(a)
    center = na * len(b) / 2
    obs = abs(rank[:na].sum() - na * (na + 1) / 2 - center)
    extreme = count = 0
    iterator = itertools.combinations(range(n), na)
    while True:
        batch = list(itertools.islice(iterator, 8192))
        if not batch:
            break
        vals = abs(rank[np.asarray(batch)].sum(axis=1) - na * (na + 1) / 2 - center)
        extreme += int((vals >= obs - 1e-12).sum())
        count += len(batch)
    return extreme / count, count


def holm(p):
    p = np.asarray(p)
    ix = np.argsort(p)
    adjusted = np.maximum.accumulate(p[ix] * np.arange(len(p), 0, -1))
    result = np.empty(len(p))
    result[ix] = np.minimum(adjusted, 1)
    return result


def main():
    DIAG.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    d = pd.read_csv(OUT / 'experimental_validation_updated.csv')
    d = d[np.isfinite(d[OLD]) & np.isfinite(d[NEW])].copy()
    d['delta'] = d[NEW] - d[OLD]
    d['old_rank'] = d[OLD].rank(ascending=False, method='average')
    d['new_rank'] = d[NEW].rank(ascending=False, method='average')
    d['rank_change_new_minus_old'] = d.new_rank - d.old_rank
    d.to_csv(DIAG / 'all20_prediction_changes.csv', index=False)
    summaries = [paired_summary(d, 'all_experimental', rng)]
    for field in ['partner_display', 'Outcome']:
        for group, sub in d.groupby(field, sort=False):
            summaries.append(paired_summary(sub, f'{field}:{group}', rng))
    pd.DataFrame(summaries).to_csv(DIAG / 'prediction_change_summary.csv', index=False)
    expected = dict(n=20, pearson_r=.503565, spearman_rho=.518797, mean_delta=.099186, median_delta=.093941)
    checks = [dict(quantity=k, independently_supplied=v, recalculated=summaries[0][k],
                   passed=bool(np.isclose(summaries[0][k], v, atol=1e-6, rtol=0))) for k, v in expected.items()]
    pd.DataFrame(checks).to_csv(DIAG / 'independent_20pair_checks.csv', index=False)
    kd = d[d.Outcome.eq('Kd_fit') & d.Kd_uM.gt(0) & np.isfinite(d.pKd)].copy()
    kd_stats = []
    for model, col in [('old', OLD), ('new', NEW)]:
        rho, pv, method, count = spearman_permutation(kd[col], kd.pKd, rng)
        draws = rng.integers(0, len(kd), (10000, len(kd)))
        bootstrap = {k: [] for k in ['pearson_r', 'spearman_rho', 'ranking_concordance']}
        for ix in draws:
            m = metrics(kd[col].to_numpy()[ix], kd.pKd.to_numpy()[ix])
            for key in bootstrap:
                bootstrap[key].append(m[key])
        for key, values in bootstrap.items():
            point = metrics(kd[col], kd.pKd)[key]
            kd_stats.append(dict(predictor=model, n=len(kd), metric=key, estimate=point,
                                 spearman_two_sided_p=pv, permutation_method=method, permutations=count,
                                 test='exact Spearman permutation; other metrics descriptive', adjustment='none',
                                 exclusions='non-Kd-fit or missing positive Kd/prediction',
                                 shared_constructs='p-values and row-bootstrap intervals exploratory', **interval(values)))
    pd.DataFrame(kd_stats).to_csv(DIAG / 'kd_statistics.csv', index=False)
    outcome_stats, counts = [], d.Outcome.value_counts().to_dict()
    categorical = d[d.Outcome.isin(['Kd_fit', 'LB', 'NB'])].copy()
    for model, col in [('old', OLD), ('new', NEW)]:
        h, pv, epsilon = kw_test(categorical[col].to_numpy(), categorical.Outcome.to_numpy(), rng)
        outcome_stats.append(dict(predictor=model, comparison='Kd-fit / LB / NB omnibus', n=len(categorical),
            statistic=h, effect=epsilon, effect_name='bias-adjusted epsilon squared (can be negative)',
            p_raw=pv, p_adjusted=pv, adjustment='none; omnibus', test='permutation Kruskal-Wallis',
            permutations=B, CI95_low=np.nan, CI95_high=np.nan,
            exclusions='No Kd and not tested excluded from three-category diagnostic'))
        rows = []
        for a, b in [('Kd_fit', 'LB'), ('Kd_fit', 'NB'), ('LB', 'NB')]:
            aa, bb = categorical[categorical.Outcome.eq(a)][col].to_numpy(), categorical[categorical.Outcome.eq(b)][col].to_numpy()
            p, npart = pairwise_rank_test(aa, bb)
            _, effect = binary_effect(aa, bb)
            boot = [binary_effect(rng.choice(aa, len(aa)), rng.choice(bb, len(bb)))[1] for _ in range(10000)]
            rows.append(dict(predictor=model, comparison=f'{a} vs {b}', n=len(aa) + len(bb), n_a=len(aa), n_b=len(bb),
                statistic=effect, effect=effect, effect_name='rank-biserial/Cliff delta; positive means first group higher',
                p_raw=p, test='exact two-sided permutation Mann-Whitney rank statistic', permutations=npart,
                adjustment='Holm across three pairwise tests per predictor', exclusions='other outcome categories',
                **interval(boot)))
        for row, p in zip(rows, holm([r['p_raw'] for r in rows])):
            row['p_adjusted'] = p
            outcome_stats.append(row)
    pd.DataFrame(outcome_stats).to_csv(DIAG / 'outcome_statistics.csv', index=False)
    binary = d[d.Outcome.isin(['Kd_fit', 'NB'])].copy()
    a = binary[binary.Outcome.eq('Kd_fit')]
    b = binary[binary.Outcome.eq('NB')]
    aucs = {}
    for model, col in [('old', OLD), ('new', NEW)]:
        aucs[model] = binary_effect(a[col], b[col])[0]
    ix_a = rng.integers(0, len(a), (10000, len(a)))
    ix_b = rng.integers(0, len(b), (10000, len(b)))
    boot = {model: np.array([binary_effect(a[col].to_numpy()[i], b[col].to_numpy()[j])[0]
            for i, j in zip(ix_a, ix_b)]) for model, col in [('old', OLD), ('new', NEW)]}
    aurows = [dict(predictor=model, estimate=aucs[model], metric='Kd-fit versus NB AUROC', n=len(binary), n_positive=len(a), n_negative=len(b),
                  exclusions='LB and No Kd excluded; no ordinal placement of LB', test='descriptive AUROC; shared constructs',
                  adjustment='none', **interval(boot[model])) for model in aucs]
    aurows.append(dict(predictor='new_minus_old', estimate=aucs['new'] - aucs['old'], metric='paired AUROC difference',
        n=len(binary), n_positive=len(a), n_negative=len(b), exclusions='LB and No Kd excluded',
        test='same within-outcome rows resampled for both predictors', adjustment='none', **interval(boot['new'] - boot['old'])))
    pd.DataFrame(aurows).to_csv(DIAG / 'outcome_auroc.csv', index=False)
    # Same within-category row draws for old/new separation effect differences;
    # do not infer superiority from separate p-values.
    effect_differences = []
    for first, second in [('Kd_fit', 'LB'), ('Kd_fit', 'NB'), ('LB', 'NB')]:
        aa = categorical[categorical.Outcome.eq(first)]
        bb = categorical[categorical.Outcome.eq(second)]
        observed = binary_effect(aa[NEW], bb[NEW])[1] - binary_effect(aa[OLD], bb[OLD])[1]
        differences = []
        for _ in range(10000):
            ia, ib = rng.integers(0,len(aa),len(aa)), rng.integers(0,len(bb),len(bb))
            new_effect = binary_effect(aa[NEW].to_numpy()[ia],bb[NEW].to_numpy()[ib])[1]
            old_effect = binary_effect(aa[OLD].to_numpy()[ia],bb[OLD].to_numpy()[ib])[1]
            differences.append(new_effect-old_effect)
        effect_differences.append(dict(comparison=f'{first} vs {second}', n=len(aa)+len(bb),
            new_minus_old_rank_biserial=observed, method='paired within-category row bootstrap; exploratory shared constructs',
            **interval(differences)))
    pd.DataFrame(effect_differences).to_csv(DIAG/'paired_outcome_effect_differences.csv',index=False)
    binary.to_csv(DIAG / 'auroc_included_pairs.csv', index=False)
    write_json(DIAG / 'expanded_analysis_settings.json', dict(seed=SEED, Monte_Carlo_permutations=B, bootstrap=10000,
        outcome_counts=counts, expected_checks_passed=all(z['passed'] for z in checks),
        limitations='All p-values exploratory; shared constructs violate independent-row exchangeability. Paired resampling compares models on identical rows, not independent constructs.',
        category_order='display only; no ordinal affinity interpretation',
        TAF15_all_decrease=bool(d[d.RGG_partner.eq('taf15_rgg')].delta.lt(0).all()),
        LEF1_THOC4_G3BP1_increase=bool(d[d.R_TAD.eq('lef1_nter') & d.RGG_partner.isin(['thoc4_nter', 'thoc4_cter', 'g3bp1_rgg'])].delta.gt(0).all())))
    print(pd.DataFrame(summaries).head().to_string(index=False))


if __name__ == '__main__':
    main()
