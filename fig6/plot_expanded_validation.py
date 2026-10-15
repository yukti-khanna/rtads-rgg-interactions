#!/usr/bin/env python3
"""Readable validation and expanded diagnostic panels, with exact source/stats.

Every export is PDF, text-editable SVG and 600 dpi PNG. Captions describe the
legacy target scale and exploratory statistics; axis labels stay short.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from workflow_common import ROOT, OUT, DIAG
from compare_old_new_predictions import OLD, NEW
from expanded_validation_analysis import paired_summary, SEED

STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 6.5, 'axes.labelsize': 6.5, 'axes.titlesize': 7, 'xtick.labelsize': 6,
    'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.linewidth': .6,
    'xtick.major.width': .5, 'ytick.major.width': .5, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
COLORS = {'Kd_fit': '#0072B2', 'LB': '#E69F00', 'NB': '#B0B0B0', 'No Kd': '#000000', 'not tested': '#CC79A7'}
PARTNERS = ['THOC4 N-term', 'THOC4 C-term', 'G3BP1', 'TAF15']
MARKERS = {'THOC4 N-term': 'o', 'THOC4 C-term': '^', 'G3BP1': 's', 'TAF15': 'D'}
SHORT_PARTNER = {'thoc4_nter': 'THOC4 N', 'thoc4_cter': 'THOC4 C', 'g3bp1_rgg': 'G3BP1', 'taf15_rgg': 'TAF15',
    'snrnp70_s': 'SNRNP70 short', 'rbmx': 'RBMX', 'tceal6': 'TCEAL6', 'q8n319': 'Q8N319', 'rab44': 'RAB44'}
SHORT_AHD = {'foxo4_cr3': 'FOXO4', 'foxo6_cr3': 'FOXO6', 'atf4_tad': 'ATF4', 'tfe3_cter': 'TFE3',
             'lef1_nter': 'LEF1', 'foxm1': 'FOXM1', 'tp53': 'p53', 'sox12': 'SOX12'}
PRED = r'Predicted $\log_{10}(-B_{22})$'
OLDLABEL = r'Old predicted $\log_{10}(-B_{22})$'
NEWLABEL = r'New predicted $\log_{10}(-B_{22})$'
DELTA = r'$\Delta$ predicted $\log_{10}(-B_{22})$, new $-$ old'
CAPTION = ('Predictions use the historical corr1 radial-sum target convention; legacy sums have no assigned physical units. '
           'Pairs share constructs: permutation p-values and row-bootstrap intervals are exploratory. '
           'Old/new correlation describes agreement, not comparative accuracy. NB, LB and Kd-fit are categories, not an affinity order.')
EXPORTED = []
AXIS_RECORDS = []


def ahd_name(name):
    parent, *shuffle = str(name).split('_shuffle_')
    result = SHORT_AHD.get(parent, parent.upper())
    return result + (' shuffle ' + shuffle[0] if shuffle else '')


def pair_name(row):
    a = row.get('AHD_name', row.get('R_TAD', ''))
    r = row.get('RGG_name', row.get('RGG_partner', ''))
    return ahd_name(a) + '–' + SHORT_PARTNER.get(r, str(r).upper())


def relevant_limits(values, pad=.07, zero=False):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if zero:
        v = np.r_[v, 0]
    lo, hi = v.min(), v.max()
    spacing = (hi - lo) * pad if hi > lo else max(abs(lo) * pad, .05)
    return lo - spacing, hi + spacing


def axes(size=(4.5, 3.3), note_rows=2):
    fig = plt.figure(figsize=size)
    # Reserve a physical gap for ticks and the x-axis label before the statistics
    # strip. A fixed fractional gap became too small in compact panels.
    gap_inches = .46
    available = size[1] * .76
    hspace = 2 * gap_inches / (available - gap_inches)
    gs = fig.add_gridspec(2, 1, height_ratios=[max(size[1] - 1.2, 1), .23 * note_rows], hspace=hspace)
    ax, note = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    note.axis('off')
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    fig.subplots_adjust(left=.16, right=.96, bottom=.08, top=.84)
    return fig, ax, note


def pformat(p):
    return f'{p:.3g}' if np.isfinite(p) else 'NA'


def export(fig, path, source, statistics=None, caption=CAPTION):
    path.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(path.with_suffix('.csv'), index=False)
    if statistics is not None:
        statistics.to_csv(path.parent / (path.name + '_statistics.csv'), index=False)
    path.with_suffix('.caption.txt').write_text(caption + '\n')
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(path.with_suffix('.' + ext), dpi=600)
    for i, ax in enumerate(fig.axes):
        if ax.axison:
            AXIS_RECORDS.append(dict(panel=str(path.relative_to(ROOT)), axes=i,
                xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()), aspect=str(ax.get_aspect())))
    EXPORTED.append(str(path.with_suffix('.png').relative_to(ROOT)))
    plt.close(fig)


def adjust_identity_labels(fig, ax, d, xcol, ycol):
    """Greedy placement with expanded candidate grid and leader lines.

    Labels are protein identities; partners use the shape legend. Candidate
    scoring excludes point and error-bar boxes, and penalizes label collisions.
    Runtime visual inspection is the final check, not this heuristic.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    domain = ax.get_window_extent(renderer)
    occupied, pointboxes = [], []
    for _, row in d.iterrows():
        x, y = ax.transData.transform([row[xcol], row[ycol]])
        err = row.get('pKd_error', 0)
        low, high = ax.transData.transform([[row[xcol], row[ycol] - (err if np.isfinite(err) else 0)],
                                           [row[xcol], row[ycol] + (err if np.isfinite(err) else 0)]])[:, 1]
        pointboxes.append(Bbox.from_extents(x - 5, min(low, y - 5), x + 5, max(high, y + 5)))
    for _, row in d.sort_values(ycol, ascending=False).iterrows():
        label = ahd_name(row.get('R_TAD', row.get('AHD_name', '')))
        candidates = []
        for radius in [7, 13, 20, 28, 38, 48]:
            for dx, dy, ha in [(radius, 5, 'left'), (-radius, 5, 'right'), (radius, -8, 'left'),
                               (-radius, -8, 'right'), (0, radius, 'center'), (0, -radius, 'center')]:
                a = ax.annotate(label, (row[xcol], row[ycol]), xytext=(dx, dy), textcoords='offset points',
                                ha=ha, va='center', fontsize=6)
                box = a.get_window_extent(renderer).expanded(1.05, 1.15)
                outside = not (domain.contains(box.x0, box.y0) and domain.contains(box.x1, box.y1))
                collisions = sum(box.overlaps(b) for b in occupied + pointboxes)
                candidates.append((1000 * outside + 100 * collisions + abs(dx) + abs(dy), dx, dy, ha, box))
                a.remove()
        _, dx, dy, ha, box = min(candidates, key=lambda v: v[0])
        ax.annotate(label, (row[xcol], row[ycol]), xytext=(dx, dy), textcoords='offset points', ha=ha,
                    va='center', fontsize=6, bbox={'facecolor':'white','edgecolor':'none','pad':.25,'alpha':.9},
                    arrowprops={'arrowstyle': '-', 'lw': .35, 'color': '#777777',
                    'shrinkA': 2, 'shrinkB': 4} if abs(dx) + abs(dy) > 15 else None)
        occupied.append(box)


def kd_panel(d, col, path, model, kd_statistics, xlim, ylim, xerr=None):
    fig, ax, note = axes()
    for partner in PARTNERS:
        g = d[d.partner_display.eq(partner)]
        if g.empty:
            continue
        color = '#0072B2' if partner.startswith('THOC4') else '#B0B0B0'
        ax.errorbar(g[col], g.pKd, yerr=g.pKd_error, xerr=g[xerr] if xerr else None,
                    fmt=MARKERS[partner], ms=5, mfc=color, mec='black', mew=.4,
                    ecolor=color, elinewidth=.8, capsize=2, linestyle='None', label=partner)
    ax.set(xlim=xlim, ylim=ylim, xlabel=PRED if model in ['old', 'new'] else r'Simulated $\log_{10}(-B_{22})$', ylabel='Experimental pKd')
    # Fit cannot extend the axes. Every eligible pair is included.
    slope, intercept = np.polyfit(d[col], d.pKd, 1)
    xx = np.array([max(xlim[0], d[col].min()), min(xlim[1], d[col].max())])
    ax.plot(xx, slope * xx + intercept, 'k--', lw=.8)
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.01), columnspacing=1)
    summary = kd_statistics[kd_statistics.predictor.eq(model)]
    if not summary.empty:
        r = summary[summary.metric.eq('pearson_r')].iloc[0]
        rho = summary[summary.metric.eq('spearman_rho')].iloc[0]
        note.text(0, .65, f"n = {len(d)}    Pearson r = {r.estimate:.3f}    Spearman ρ = {rho.estimate:.3f}    p = {pformat(rho.spearman_two_sided_p)}", fontsize=6.5)
        note.text(0, .22, 'Exact two-sided Spearman permutation; exploratory (shared constructs).', fontsize=6)
    adjust_identity_labels(fig, ax, d, col, 'pKd')
    export(fig, path, d, summary, CAPTION + ' pKd errors use first-order Kd error propagation. Fit line is descriptive. All eligible Kd pairs retained.')


def outcome_panel(d, col, path, model, outcome_stats, auroc, common_xlim):
    categories = [c for c in ['Kd_fit', 'LB', 'NB', 'No Kd'] if c in set(d.Outcome)]
    fig, ax, note = axes(size=(4.8, 3.7), note_rows=4)
    plotted = []
    for i, category in enumerate(categories):
        g = d[d.Outcome.eq(category)].sort_values(['partner_display', 'R_TAD']).copy()
        g['display_y'] = i + (np.linspace(-.18, .18, len(g)) if len(g) > 1 else 0)
        ax.scatter(g[col], g.display_y, s=18, color=COLORS[category])
        plotted.append(g)
    ax.set(xlabel=PRED if model!='simulation' else r'Simulated $\log_{10}(-B_{22})$', ylabel='Experimental outcome', xlim=common_xlim, ylim=(-.45, len(categories) - .55))
    ax.set_yticks(range(len(categories)), [f'{c.replace("Kd_fit","Kd-fit")} (n={int(d.Outcome.eq(c).sum())})' for c in categories])
    ax.xaxis.grid(True, color='#dddddd', lw=.4)
    st = outcome_stats[outcome_stats.predictor.eq(model)]
    omnibus = st.iloc[0]
    note.text(0, .92, f"Permutation Kruskal–Wallis: H = {omnibus.statistic:.2f}, p = {pformat(omnibus.p_raw)}", fontsize=6.5)
    for i, (_, row) in enumerate(st.iloc[1:].iterrows()):
        note.text(0, .72 - .17 * i, f"{row.comparison.replace('Kd_fit', 'Kd-fit')}: Holm-adjusted p = {pformat(row.p_adjusted)}", fontsize=6.2)
    auc = auroc[auroc.predictor.eq(model)].iloc[0]
    note.text(0, .08, f"Kd-fit vs NB AUROC = {auc.estimate:.3f} [95% CI {auc.CI95_low:.3f}, {auc.CI95_high:.3f}], n={int(auc.n)}; LB excluded.", fontsize=6)
    export(fig, path, pd.concat(plotted), pd.concat([st, auroc[auroc.predictor.eq(model)]], ignore_index=True),
           CAPTION + ' Pairwise tests use exact two-sided Mann–Whitney permutation statistics; Holm correction across three comparisons per predictor. '
           'Higher scores expected for Kd-fit than NB only in the binary AUROC diagnostic; LB is excluded from that diagnostic.')


def ordered_pairs(d):
    d = d.copy()
    d['partner_order'] = d.partner_display.map(dict(zip(PARTNERS, range(len(PARTNERS))))).fillna(9)
    d['pair_label'] = d.apply(pair_name, axis=1)
    return d.sort_values(['partner_order', 'R_TAD']).reset_index(drop=True)


def paired_dot(d, summary, path, delta=False):
    d = ordered_pairs(d)
    fig, ax, note = axes(size=(6.5, 7.2), note_rows=2)
    fig.subplots_adjust(left=.29, right=.82, top=.91, bottom=.035)
    y = np.arange(len(d))
    if delta:
        ax.set_xlim(relevant_limits(d.delta, zero=True))
        ax.axvline(0, color='black', lw=.8, ls='--')
        ax.scatter(d.delta, y, s=18, c=d.Outcome.map(COLORS))
        ax.set_xlabel(DELTA)
    else:
        ax.set_xlim(relevant_limits(np.r_[d[OLD], d[NEW]]))
        ax.hlines(y, np.minimum(d[OLD], d[NEW]), np.maximum(d[OLD], d[NEW]), color='#aaaaaa', lw=.6)
        ax.scatter(d[OLD], y, marker='o', s=20, facecolor='white', edgecolor='#0072B2', linewidth=.8, label='Old panel')
        ax.scatter(d[NEW], y, marker='D', s=20, color='#D55E00', label='New frozen')
        ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.02))
        ax.set_xlabel(PRED)
    ax.set_yticks(y, d.pair_label)
    ax.set_ylim(len(d) - .4, -.6)
    ax.text(1.03, 1.025, 'Outcome', transform=ax.transAxes, weight='bold', fontsize=6.5)
    for i, row in d.iterrows():
        ax.text(1.03, i, row.Outcome.replace('Kd_fit', 'Kd-fit'), transform=ax.get_yaxis_transform(),
                va='center', color='#555555' if row.Outcome=='NB' else COLORS.get(row.Outcome, 'black'), fontsize=6.5)
        if i > 0 and d.iloc[i - 1].partner_display != row.partner_display:
            ax.axhline(i - .5, color='#dddddd', lw=.5)
    s = summary.iloc[0]
    note.text(0, .75, f"n={int(s.n)}    r={s.pearson_r:.3f}    ρ={s.spearman_rho:.3f}    mean Δ={s.mean_delta:+.3f} [95% CI {s.mean_delta_CI95_low:+.3f}, {s.mean_delta_CI95_high:+.3f}]", fontsize=6.2)
    note.text(0, .25, f"Paired sign-flip p={pformat(s.shift_pvalue)}; exploratory shift, not evidence of better accuracy.", fontsize=6)
    export(fig, path, d, summary)


def agreement_scatter(d, summary, path, old=OLD, new=NEW):
    fig, ax, note = axes(size=(4.5, 4), note_rows=2)
    lim = relevant_limits(np.r_[d[old], d[new]])
    ax.set(xlim=lim, ylim=lim, xlabel=OLDLABEL, ylabel=NEWLABEL, aspect='equal')
    ax.plot(lim, lim, 'k--', lw=.7)
    for category, g in d.groupby('Outcome', sort=False):
        ax.scatter(g[old], g[new], s=18, c=COLORS.get(category, '#CC79A7'), label=f'{category} (n={len(g)})')
    ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.02))
    s = summary.iloc[0]
    note.text(0, .75, f"n={int(s.n)}    Pearson r={s.pearson_r:.3f}    Spearman ρ={s.spearman_rho:.3f}", fontsize=6.5)
    note.text(0, .22, f"Mean Δ={s.mean_delta:+.3f} [95% CI {s.mean_delta_CI95_low:+.3f}, {s.mean_delta_CI95_high:+.3f}]", fontsize=6.5)
    export(fig, path, d, summary)


def named_heatmap(d, path, old, new):
    d = d.copy()
    d['AHD_label'] = d.AHD_name.map(ahd_name)
    d['RGG_label'] = d.RGG_name.map(lambda x: SHORT_PARTNER.get(x, x.upper()))
    ahds = sorted(d.AHD_label.unique(), key=lambda s: ('shuffle' in s, s))
    rggs = sorted(d.RGG_label.unique())
    matrices = [d.pivot(index='AHD_label', columns='RGG_label', values=col).reindex(index=ahds, columns=rggs) for col in [old, new]]
    delta = matrices[1] - matrices[0]
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.7))
    common = relevant_limits(np.r_[d[old], d[new]], pad=0)
    bound = np.nanmax(abs(delta.to_numpy()))
    for ax, matrix, title in zip(axs, matrices + [delta], ['Old: same constructs', 'New: same constructs', 'New − old']):
        change = title == 'New − old'
        im = ax.imshow(matrix, aspect='auto', cmap='coolwarm' if change else 'viridis',
                       vmin=-bound if change else common[0], vmax=bound if change else common[1])
        ax.set_xticks(range(len(rggs)), rggs, rotation=55, ha='right')
        ax.set_yticks(range(len(ahds)), ahds)
        ax.set_title(title)
        for i in range(len(ahds)):
            for j in range(len(rggs)):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f'{value:+.2f}' if change else f'{value:.2f}', ha='center', va='center', fontsize=5,
                            color='black' if change else 'white')
        fig.colorbar(im, ax=ax, shrink=.75)
    fig.suptitle(f'Matched exact constructs (n={len(d)}); blank cells were not supplied. Shuffles named explicitly.', fontsize=8)
    fig.tight_layout()
    export(fig, path, d)


def overview(paths, path):
    ncols, n = 3, len(paths)
    nrows = int(np.ceil(n / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows), squeeze=False)
    for ax, file in zip(axs.flat, paths):
        ax.imshow(plt.imread(ROOT / file))
        ax.set_title(Path(file).stem.replace('_', ' '), fontsize=8)
        ax.axis('off')
    for ax in list(axs.flat)[n:]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-simulation', action='store_true',
                        help='Generate prediction panels without using existing simulation outputs')
    args = parser.parse_args()
    EXPORTED.clear()
    AXIS_RECORDS.clear()
    plt.rcParams.update(STYLE)
    d = pd.read_csv(DIAG / 'all20_prediction_changes.csv')
    kd = d[d.Outcome.eq('Kd_fit') & d.Kd_uM.gt(0)].copy()
    kd_stats = pd.read_csv(DIAG / 'kd_statistics.csv')
    outcome_stats = pd.read_csv(DIAG / 'outcome_statistics.csv')
    auroc = pd.read_csv(DIAG / 'outcome_auroc.csv')
    change = pd.read_csv(DIAG / 'prediction_change_summary.csv')
    summary = change[change['set'].eq('all_experimental')]
    xlim = relevant_limits(np.r_[kd[OLD], kd[NEW]])
    ylim = relevant_limits(np.r_[kd.pKd - kd.pKd_error, kd.pKd + kd.pKd_error])
    mainfig, diagfig = OUT / 'figures', DIAG / 'figures'
    kd_panel(kd, OLD, diagfig / 'old_prediction_vs_pKd', 'old', kd_stats, xlim, ylim)
    kd_panel(kd, NEW, diagfig / 'new_prediction_vs_pKd', 'new', kd_stats, xlim, ylim)
    kd_panel(kd, NEW, mainfig / 'validation_prediction_vs_pKd', 'new', kd_stats, xlim, ylim)
    outcome_limits = relevant_limits(np.r_[d[OLD], d[NEW]])
    outcome_panel(d, OLD, diagfig / 'old_prediction_vs_outcome', 'old', outcome_stats, auroc, outcome_limits)
    outcome_panel(d, NEW, diagfig / 'new_prediction_vs_outcome', 'new', outcome_stats, auroc, outcome_limits)
    outcome_panel(d, NEW, mainfig / 'validation_prediction_vs_outcome', 'new', outcome_stats, auroc, outcome_limits)
    paired_dot(d, summary, diagfig / 'all20_named_paired_predictions')
    paired_dot(d, summary, diagfig / 'all20_named_prediction_deltas', delta=True)
    agreement_scatter(d, summary, diagfig / 'old_vs_new_predictions')
    # Simulation plots retain descriptive names and convention labels in captions.
    simfile = OUT / 'b22/per_pair_b22.csv'
    if not args.skip_simulation and simfile.exists():
        sim = pd.read_csv(simfile)
        sim = sim[sim.method.eq('legacy_sum') & np.isclose(sim.integration_boundary_nm, 7.5)]
        j = d.merge(sim, on='pair_id', validate='one_to_one')
        skd = j[j.Outcome.eq('Kd_fit') & np.isfinite(j.log10_neg_mean_B22)].copy()
        skd['sim_log_error'] = skd.corrected_B22_SEM / (abs(skd.corrected_B22_mean) * np.log(10))
        # Statistics on these plots are calculated against current RDF values,
        # not copied from prediction statistics.
        from expanded_validation_analysis import spearman_permutation
        from scipy import stats
        rho, p, method, count = spearman_permutation(skd.log10_neg_mean_B22, skd.pKd, np.random.default_rng(SEED))
        sst = pd.DataFrame([dict(predictor='simulation', metric=m, estimate=v, n=len(skd), spearman_two_sided_p=p,
            permutation_method=method, permutations=count) for m, v in [('pearson_r', stats.pearsonr(skd.log10_neg_mean_B22, skd.pKd).statistic), ('spearman_rho', rho)]])
        kd_panel(skd, 'log10_neg_mean_B22', mainfig / 'validation_simulation_vs_pKd', 'simulation', sst,
                 relevant_limits(np.r_[skd.log10_neg_mean_B22 - skd.sim_log_error, skd.log10_neg_mean_B22 + skd.sim_log_error]),
                 relevant_limits(np.r_[skd.pKd - skd.pKd_error, skd.pKd + skd.pKd_error]), xerr='sim_log_error')
        sim_outcomes=pd.read_csv(OUT/'b22/simulation_outcome_statistics.csv')
        sim_auc=pd.read_csv(OUT/'b22/simulation_outcome_auroc.csv')
        outcome_panel(j,'log10_neg_mean_B22',mainfig/'validation_simulation_vs_outcome','simulation',sim_outcomes,sim_auc,
                      relevant_limits(j.log10_neg_mean_B22))
    complete = DIAG / 'historical_model/same_construct_predictions.csv'
    if complete.exists():
        c = pd.read_csv(complete)
        oldcol = 'Old_same_construct_final'
        full_stats = pd.read_csv(DIAG / 'complete_same_construct_summary.csv')
        s = full_stats[full_stats['set'].eq('complete_same_construct')]
        agreement_scatter(c, s, diagfig / 'complete_same_construct_agreement', oldcol, NEW)
        named_heatmap(c, diagfig / 'complete_same_construct_heatmaps', oldcol, NEW)
        shuffles = pd.read_csv(DIAG / 'matched_WT_shuffle_changes.csv')
        if not shuffles.empty:
            shuffles['pair_label'] = shuffles.apply(pair_name, axis=1)
            shuffles = shuffles.sort_values(['RGG_name', 'AHD_name']).reset_index(drop=True)
            fig, ax, note = axes(size=(7, 6.4), note_rows=2)
            fig.subplots_adjust(left=.37, right=.96, top=.91, bottom=.04)
            y = np.arange(len(shuffles))
            a, b = shuffles.old_shuffle_minus_WT, shuffles.new_shuffle_minus_WT
            ax.set_xlim(relevant_limits(np.r_[a,b], zero=True))
            ax.axvline(0, color='#777777', ls='--', lw=.7)
            ax.hlines(y, np.minimum(a,b), np.maximum(a,b), color='#aaaaaa', lw=.6)
            ax.scatter(a, y, s=20, marker='o', facecolor='white', edgecolor='#0072B2', label='Old')
            ax.scatter(b, y, s=20, marker='D', color='#D55E00', label='New')
            ax.set_yticks(y, shuffles.pair_label)
            ax.set_ylim(len(y)-.4, -.6)
            ax.set_xlabel(r'Predicted $\log_{10}(-B_{22})$ change, shuffle $-$ WT')
            ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5,1.01))
            note.text(0,.6, f'n={len(shuffles)} matched WT–shuffle groups; exact parent IDs in source CSV.', fontsize=6.5)
            export(fig, diagfig / 'matched_WT_shuffle_effects', shuffles, pd.read_csv(DIAG / 'WT_shuffle_summary.csv'))
    simsource = OUT / 'b22/paired_simulation_accuracy_source.csv'
    if not args.skip_simulation and simsource.exists():
        acc = pd.read_csv(OUT / 'b22/paired_simulation_accuracy_metrics.csv')
        x = pd.read_csv(simsource)
        lim = relevant_limits(np.r_[x.log10_neg_mean_B22, x.Old_same_construct_final, x[NEW]])
        for model, col, name in [('old_same_construct','Old_same_construct_final','old_vs_simulation_verified_subset'),
                                 ('new',NEW,'new_vs_simulation_verified_subset')]:
            fig, ax, note = axes(size=(4.8,4), note_rows=3)
            ax.set(xlim=lim, ylim=lim, aspect='equal', xlabel=r'Simulated $\log_{10}(-B_{22})$', ylabel=PRED)
            ax.plot(lim,lim,'k--',lw=.7)
            for category, g in x.groupby('Outcome'):
                ax.scatter(g.log10_neg_mean_B22, g[col], s=18, color=COLORS.get(category,'black'), label=category)
            ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5,1.01))
            st = acc[acc.predictor.eq(model)]
            a = st.iloc[0]
            note.text(0,.8, f'n={int(a.n)}   R²={a.R_squared:.3f}   RMSE={a.RMSE:.3f}   MAE={a.MAE:.3f}', fontsize=6.5)
            note.text(0,.4, f'r={a.Pearson_r:.3f}   ρ={a.Spearman_rho:.3f}   mean signed error={a.mean_signed_error:+.3f}', fontsize=6.5)
            note.text(0,.02, 'Same eligible rows; legacy scale; adjacent topology sequence evidence.', fontsize=6)
            export(fig,diagfig/name,x,st,CAPTION+' Accuracy is against simulations on a sequence-supported subset, not experimental Kd. RDF/topology linkage is contextual, not an embedded generator manifest.')
            if model=='new':
                # The same rendered panel is also the main simulation-validation export.
                import shutil
                target=mainfig/'validation_prediction_vs_simulation_verified_subset'
                for suffix in ['.png','.pdf','.svg','.csv','.caption.txt']:
                    shutil.copyfile((diagfig/name).with_suffix(suffix),target.with_suffix(suffix))
                st.to_csv(mainfig/'validation_prediction_vs_simulation_verified_subset_statistics.csv',index=False)
                EXPORTED.append(str(target.with_suffix('.png').relative_to(ROOT)))
        blocked=mainfig/'simulation_prediction_comparison_BLOCKED.txt'
        if blocked.exists():
            blocked.unlink()
    overview([p for p in EXPORTED if p.startswith('prediction_diagnostics')], DIAG / 'diagnostic_overview.png')
    pd.DataFrame({'png': EXPORTED, 'visual_inspection_status': 'awaiting inspection'}).to_csv(DIAG / 'figure_export_manifest.csv', index=False)
    (DIAG / 'figure_axis_limits.json').write_text(json.dumps(AXIS_RECORDS,indent=2)+'\n')
    print(f'Exported {len(EXPORTED)} panels plus diagnostic overview')


if __name__ == '__main__':
    main()
