"""Reproducible statistics for eligible master-table rows, including small sets."""
import itertools
import math
import numpy as np
import pandas as pd
from scipy import stats


def finite_pairs(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def corr(x, y, kind):
    x, y = finite_pairs(x, y)
    if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float((stats.spearmanr(x, y) if kind == 'spearman' else stats.pearsonr(x, y))[0])


def spearman_test(x, y, settings):
    x, y = finite_pairs(x, y)
    rho = corr(x, y, 'spearman')
    if len(x) < 3 or not np.isfinite(rho):
        return rho, np.nan, 'unavailable: fewer than 3 pairs or constant values', 0
    a, b = stats.rankdata(x), stats.rankdata(y)
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    extreme = count = 0
    if len(x) <= settings.get('exact_spearman_max_n', 9):
        it = itertools.permutations(range(len(y)))
        while True:
            batch = list(itertools.islice(it, 8192))
            if not batch:
                break
            values = b[np.asarray(batch)] @ a / denominator
            extreme += int((abs(values) >= abs(rho) - 1e-12).sum())
            count += len(batch)
        return rho, extreme / count, 'exact two-sided Spearman permutation of labelled rows', count
    rng = np.random.default_rng(settings['seed'])
    count = settings['permutations']
    for _ in range(count):
        extreme += abs(np.dot(a, rng.permutation(b)) / denominator) >= abs(rho) - 1e-12
    return rho, (extreme + 1) / (count + 1), 'two-sided Monte Carlo Spearman permutation; plus-one correction', count


def agreement_metrics(observed, predicted):
    a, b = finite_pairs(observed, predicted)
    if not len(a):
        return dict(n=0, R_squared=np.nan, RMSE=np.nan, MAE=np.nan, bias=np.nan, Pearson_r=np.nan, Spearman_rho=np.nan)
    residual = b - a
    denominator = np.sum((a - a.mean())**2)
    return dict(n=len(a), R_squared=1 - np.sum(residual**2) / denominator if denominator > 0 and len(a) >= 2 else np.nan,
                RMSE=np.sqrt(np.mean(residual**2)), MAE=np.mean(abs(residual)), bias=np.mean(residual),
                Pearson_r=corr(a, b, 'pearson'), Spearman_rho=corr(a, b, 'spearman'))


def paired_change(old, new, settings):
    a, b = finite_pairs(old, new)
    d = b - a
    result = dict(n=len(d), Pearson_r=corr(a, b, 'pearson'), Spearman_rho=corr(a, b, 'spearman'))
    for key, func in [('mean_delta', np.mean), ('median_delta', np.median), ('mean_absolute_change', lambda v: np.mean(abs(v))), ('minimum_delta', np.min), ('maximum_delta', np.max)]:
        result[key] = float(func(d)) if len(d) else np.nan
    result.update(mean_delta_CI95_low=np.nan, mean_delta_CI95_high=np.nan, shift_pvalue=np.nan, shift_test='unavailable', permutations=0)
    if not len(d):
        return result
    rng = np.random.default_rng(settings['seed'])
    draws = d[rng.integers(0, len(d), (settings['bootstrap'], len(d)))].mean(axis=1)
    result['mean_delta_CI95_low'], result['mean_delta_CI95_high'] = np.percentile(draws, [2.5, 97.5])
    obs = abs(d.mean())
    extreme = 0
    if len(d) <= 20:
        count = 2**len(d)
        for start in range(0, count, 8192):
            signs = 2 * ((np.arange(start, min(start + 8192, count))[:, None] >> np.arange(len(d))) & 1) - 1
            extreme += int((abs(signs @ d / len(d)) >= obs - 1e-12).sum())
        pv, method = extreme / count, 'exact two-sided paired sign-flip mean test'
    else:
        count = settings['permutations']
        for _ in range(count):
            extreme += abs(np.mean(rng.choice([-1, 1], len(d)) * d)) >= obs - 1e-12
        pv, method = (extreme + 1) / (count + 1), 'Monte Carlo two-sided paired sign-flip mean test'
    result.update(shift_pvalue=pv, shift_test=method, permutations=count)
    return result


def effect(a, b):
    if not len(a) or not len(b):
        return np.nan
    differences = np.asarray(a)[:, None] - np.asarray(b)[None, :]
    return float(np.mean(differences > 0) - np.mean(differences < 0))


def pairwise(a, b, settings):
    if not len(a) or not len(b):
        return np.nan, 0, 'unavailable: category absent'
    values = np.r_[a, b]
    rank = stats.rankdata(values)
    na, nb = len(a), len(b)
    center = na * nb / 2
    observed = abs(rank[:na].sum() - na * (na + 1) / 2 - center)
    partitions = math.comb(na + nb, na)
    if partitions <= settings['exact_pairwise_max_partitions']:
        extreme = 0
        for ix in itertools.combinations(range(na + nb), na):
            value = abs(rank[list(ix)].sum() - na * (na + 1) / 2 - center)
            extreme += value >= observed - 1e-12
        return extreme / partitions, partitions, 'exact two-sided permutation Mann–Whitney rank statistic'
    rng = np.random.default_rng(settings['seed'])
    count = settings['permutations']
    extreme = sum(abs(rng.permutation(rank)[:na].sum() - na * (na + 1) / 2 - center) >= observed - 1e-12 for _ in range(count))
    return (extreme + 1) / (count + 1), count, 'Monte Carlo two-sided permutation Mann–Whitney rank statistic'


def outcome_tests(d, column, settings):
    categories = ['Kd fit', 'LB', 'NB']
    sub = d[d.Outcome.isin(categories) & np.isfinite(d[column])]
    groups = [sub.loc[sub.Outcome.eq(c), column].to_numpy() for c in categories]
    counts = dict(zip(categories, map(len, groups)))
    present = [g for g in groups if len(g)]
    h = pv = epsilon = np.nan
    method, count = 'unavailable: fewer than two categories or constant values', 0
    if len(present) >= 2 and np.ptp(sub[column].to_numpy()) > 0:
        values = sub[column].to_numpy()
        labels = sub.Outcome.to_numpy()
        names = pd.unique(labels)
        ranks = stats.rankdata(values)
        masks = np.array([labels == c for c in names]).T
        sizes = masks.sum(axis=0)
        n = len(values)
        _, ties = np.unique(values, return_counts=True)
        correction = 1 - np.sum(ties**3 - ties) / (n**3 - n)
        def statistic(r):
            sums = np.asarray(r) @ masks
            return (12 / (n * (n + 1)) * np.sum(sums**2 / sizes, axis=-1) - 3 * (n + 1)) / correction
        h = float(statistic(ranks))
        rng = np.random.default_rng(settings['seed'])
        count = settings['permutations']
        extreme = sum(statistic(rng.permutation(ranks)) >= h - 1e-12 for _ in range(count))
        pv = (extreme + 1) / (count + 1)
        epsilon = (h - len(present) + 1) / (n - len(present)) if n > len(present) else np.nan
        method = 'Monte Carlo permutation Kruskal–Wallis; plus-one correction'
    rows = [dict(comparison='omnibus', n=len(sub), statistic=h, effect=epsilon, effect_name='bias-adjusted epsilon squared', p_raw=pv,
                 p_adjusted=pv, test=method, permutations=count, adjustment='none; omnibus', counts=str(counts))]
    rng = np.random.default_rng(settings['seed'])
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        a, b = groups[i], groups[j]
        p, draws, test = pairwise(a, b, settings)
        lo = hi = np.nan
        if len(a) and len(b):
            bootstrap = [effect(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(settings['bootstrap'])]
            lo, hi = np.percentile(bootstrap, [2.5, 97.5])
        rows.append(dict(comparison=f'{categories[i]} vs {categories[j]}', n=len(a) + len(b), n_a=len(a), n_b=len(b),
            effect=effect(a, b), effect_name='rank-biserial / Cliff delta; first group minus second', CI95_low=lo, CI95_high=hi,
            p_raw=p, p_adjusted=np.nan, test=test, permutations=draws, adjustment='Holm over three planned pairwise comparisons'))
    # Missing categories leave their tests unavailable; keep family size three.
    order = sorted(range(1, 4), key=lambda i: rows[i]['p_raw'] if np.isfinite(rows[i]['p_raw']) else 1)
    previous = 0
    for rank, ix in enumerate(order):
        p = rows[ix]['p_raw']
        previous = max(previous, min(1, (3 - rank) * p)) if np.isfinite(p) else previous
        rows[ix]['p_adjusted'] = previous if np.isfinite(p) else np.nan
    a, b = groups[0], groups[2]
    auc = (effect(a, b) + 1) / 2
    lo = hi = np.nan
    if len(a) and len(b):
        samples = [(effect(rng.choice(a, len(a)), rng.choice(b, len(b))) + 1) / 2 for _ in range(settings['bootstrap'])]
        lo, hi = np.percentile(samples, [2.5, 97.5])
    rows.append(dict(comparison='Kd fit vs NB AUROC', n=len(a) + len(b), n_a=len(a), n_b=len(b), statistic=auc,
        effect=auc, CI95_low=lo, CI95_high=hi, test='descriptive ranking; LB excluded', p_raw=np.nan, p_adjusted=np.nan, adjustment='none'))
    return pd.DataFrame(rows)


def paired_correlations(d, old, new, target, settings):
    d = d[np.isfinite(d[old]) & np.isfinite(d[new]) & np.isfinite(d[target])]
    rng = np.random.default_rng(settings['seed'])
    rows = []
    for kind in ['pearson', 'spearman']:
        a, b = corr(d[old], d[target], kind), corr(d[new], d[target], kind)
        samples = []
        if len(d):
            for _ in range(settings['bootstrap']):
                sub = d.iloc[rng.integers(0, len(d), len(d))]
                samples.append(corr(sub[new], sub[target], kind) - corr(sub[old], sub[target], kind))
        valid = np.asarray(samples)[np.isfinite(samples)]
        lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
        rows.append(dict(metric=kind + ' correlation with pKd', n=len(d), old=a, new=b, difference_new_minus_old=b - a,
                         CI95_low=lo, CI95_high=hi, resamples=settings['bootstrap'],
                         test='paired row bootstrap; identical rows for both models; shared constructs; exploratory'))
    return pd.DataFrame(rows)


def paired_performance(d, old, new, target, settings):
    """Both predictors evaluated/resampled on exactly the same rows."""
    d = d[np.isfinite(d[old]) & np.isfinite(d[new]) & np.isfinite(d[target])]
    results = []
    point = {model: agreement_metrics(d[target], d[col]) for model, col in [('old', old), ('new', new)]}
    rng = np.random.default_rng(settings['seed'])
    keys = ['R_squared', 'RMSE', 'MAE', 'bias', 'Pearson_r', 'Spearman_rho']
    samples = {k: [] for k in keys}
    if len(d):
        for _ in range(settings['bootstrap']):
            sub = d.iloc[rng.integers(0, len(d), len(d))]
            a, b = agreement_metrics(sub[target], sub[old]), agreement_metrics(sub[target], sub[new])
            for k in keys:
                samples[k].append(b[k] - a[k])
    for model, values in point.items():
        results.append(dict(predictor=model, **values, test='descriptive; same eligible rows'))
    for k in keys:
        valid = np.asarray(samples[k])[np.isfinite(samples[k])]
        lo, hi = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
        results.append(dict(predictor='new minus old', metric=k, estimate=point['new'][k] - point['old'][k], n=len(d),
            CI95_low=lo, CI95_high=hi, test='paired row bootstrap; shared constructs; exploratory', resamples=settings['bootstrap']))
    return pd.DataFrame(results)
