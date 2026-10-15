"""All current paper panels and updated diagnostics read validation_master.csv.

No figure uses old manually maintained plotting tables. Paper limits are
independent; shared limits are used only for paired diagnostic comparisons.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
from validation_master import ROOT, output_path, csv_write, pretty_ahd, pretty_rgg, boolean
from validation_statistics import spearman_test, corr, outcome_tests, agreement_metrics, paired_change, paired_performance, paired_correlations, effect

PRED = 'Pred_log10_negB22_final'
SIM = 'Sim_log10_negB22_final'
OLD = 'Old_same_construct_final'
PRED_LABEL = r'Predicted $\log_{10}(-B_{22})$'
SIM_LABEL = r'Simulated $\log_{10}(-B_{22})$'
STYLE = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
         'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
         'legend.fontsize': 7, 'axes.linewidth': .6, 'pdf.fonttype': 42, 'svg.fonttype': 'none'}
COLORS = {'Kd fit': '#0072B2', 'LB': '#E69F00', 'NB': '#999999', 'not tested': '#CC79A7'}
MARKERS = {'THOC4 N-term': 'o', 'THOC4 C-term': '^', 'G3BP1': 's', 'TAF15': 'D'}
CAPTION = ('B22 comparison uses the established 7.5 nm corr1 corrected radial sum without dr; no physical units are assigned to this sum. '
           'Predictions are already log10(-B22); simulation logs follow replica averaging. '
           'Kd is converted from explicitly specified units; pKd=6-log10(Kd in uM). '
           'pKd error is first-order propagation of the reported Kd error, preserving its original uncertainty interpretation. '
           'Pairs share constructs; permutation tests and row-bootstrap intervals are exploratory. '
           'LB denotes the reported low-binding category and NB the reported no-binding category; neither implies a fitted Kd. '
           'Untested pairs are never assigned NB. Prediction agreement and shifts do not establish accuracy.')


def limits(values, pad=.07, zero=False):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if zero:
        v = np.r_[v, 0]
    if not len(v):
        return (0, 1)
    lo, hi = v.min(), v.max()
    gap = (hi - lo) * pad if hi > lo else max(abs(lo) * pad, .05)
    return float(lo - gap), float(hi + gap)


def ptext(p):
    return f'{p:.3g}' if np.isfinite(p) else 'NA'


def axes(size=(4.8, 3.8)):
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=.18, right=.96, bottom=.18, top=.77)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    return fig, ax


def identity_labels(fig, ax, d, x, y, xerr=None, yerr=None):
    """Avoid labels, points, both uncertainty directions and the axes boundary."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    domain = ax.get_window_extent(renderer)
    occupied, obstacles = [], []
    for _, row in d.iterrows():
        dx = row.get(xerr, 0) if xerr else 0
        dy = row.get(yerr, 0) if yerr else 0
        dx = dx if np.isfinite(dx) else 0
        dy = dy if np.isfinite(dy) else 0
        corners = ax.transData.transform([[row[x] - dx, row[y] - dy], [row[x] + dx, row[y] + dy]])
        obstacles.append(Bbox.from_extents(corners[0, 0] - 5, corners[0, 1] - 5, corners[1, 0] + 5, corners[1, 1] + 5))
    # Multiple constructs with the same short AHD name within a partner need
    # their full machine-derived pair name rather than an ambiguous short label.
    names = d.AHD_name.map(pretty_ahd)
    ambiguous = d.assign(_label=names).duplicated(['partner_display', '_label'], keep=False)
    for ix, row in d.sort_values(y, ascending=False).iterrows():
        label = row.pair_display_name if ambiguous.loc[ix] else pretty_ahd(row.AHD_name)
        candidates = []
        for radius in [8, 15, 24, 34, 46, 60]:
            for dx, dy, ha in [(radius, 5, 'left'), (-radius, 5, 'right'), (radius, -9, 'left'), (-radius, -9, 'right'), (0, radius, 'center'), (0, -radius, 'center')]:
                a = ax.annotate(label, (row[x], row[y]), xytext=(dx, dy), textcoords='offset points', ha=ha, va='center', fontsize=7)
                box = a.get_window_extent(renderer).expanded(1.06, 1.15)
                outside = not (domain.contains(box.x0, box.y0) and domain.contains(box.x1, box.y1))
                score = 10000 * outside + 1000 * sum(box.overlaps(b) for b in occupied + obstacles) + abs(dx) + abs(dy)
                candidates.append((score, dx, dy, ha, box))
                a.remove()
        _, dx, dy, ha, box = min(candidates, key=lambda c: c[0])
        ax.annotate(label, (row[x], row[y]), xytext=(dx, dy), textcoords='offset points', ha=ha, va='center', fontsize=7,
            bbox=dict(facecolor='white', edgecolor='none', pad=.2, alpha=.9),
            arrowprops=dict(arrowstyle='-', lw=.35, color='#777777', shrinkA=2, shrinkB=4) if abs(dx) + abs(dy) > 15 else None,
            annotation_clip=False)
        occupied.append(box)


class MasterFigures:
    def __init__(self, master, config, outdir):
        self.d = master.copy()
        self.cfg = config
        self.settings = config['statistics']
        self.out = output_path(outdir)
        self.paper = self.out / 'paper_figures'
        self.supp = self.out / 'supplementary_figures'
        self.diag = output_path(config['diagnostics_dir'])
        self.reports = self.out / 'analysis_reports'
        self.exports, self.axis_records, self.members = [], [], []
        partners = sorted(master.partner_display.dropna().unique())
        marker_cycle = ['v', 'P', 'X', '<', '>', 'h', '*']
        self.markers = dict(MARKERS)
        for name in partners:
            if name not in self.markers:
                self.markers[name] = marker_cycle[len(self.markers) % len(marker_cycle)]

    def select(self, panel, criteria):
        reasons = pd.Series('', index=self.d.index, dtype=object)
        for ok, reason in criteria:
            failed = ~pd.Series(ok, index=self.d.index).fillna(False).astype(bool)
            reasons.loc[failed] = reasons.loc[failed].map(lambda s: s + '; ' if s else '') + reason
        included = reasons.eq('')
        membership = self.d[['pair_id', 'pair_display_name']].copy()
        membership['panel'] = panel
        membership['included'] = included
        membership['exclusion_reason'] = reasons
        self.members.append(membership)
        csv_write(membership, self.reports / (panel + '_membership.csv'))
        return self.d[included].copy()

    def fitted_criteria(self, col, simulation=False):
        criteria = [(self.d.Outcome.eq('Kd fit'), 'not a Kd fit'),
                    (np.isfinite(self.d.pKd) & self.d.Kd_uM.gt(0), 'positive fitted Kd unavailable'),
                    (np.isfinite(self.d[col]), 'plotted quantity unavailable'),
                    (~self.d.experimental_status.str.contains('selection required', na=False), 'experimental selection unresolved'),
                    (self.d.experimental_construct_status.ne('sequence conflict'), 'experimental sequence conflict')]
        if not simulation:
            criteria.append((self.d.prediction_sequence_status.ne('sequence conflict or missing'), 'prediction sequence conflict or missing'))
        return criteria

    def axis_limits(self, panel, axis, values, automatic):
        specified = self.cfg.get('plot_limits', {}).get(panel, {}).get(axis)
        if specified is None:
            return automatic
        lo, hi = map(float, specified)
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        if lo >= hi or (len(values) and (lo > values.min() or hi < values.max())):
            raise ValueError(f'{panel}.{axis}: manual limits would crop displayed data or uncertainty')
        return lo, hi

    def export(self, fig, stem, source, statistics, caption=CAPTION):
        stem.parent.mkdir(parents=True, exist_ok=True)
        csv_write(source, stem.with_suffix('.csv'))
        csv_write(statistics, stem.parent / (stem.name + '_statistics.csv'))
        stem.with_suffix('.caption.txt').write_text(caption + '\n')
        for extension in ['pdf', 'svg', 'png']:
            fig.savefig(stem.with_suffix('.' + extension), dpi=600, bbox_inches='tight', pad_inches=.08)
        for i, ax in enumerate(fig.axes):
            self.axis_records.append(dict(panel=str(stem.relative_to(ROOT)), axes=i, xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()), aspect=str(ax.get_aspect())))
        self.exports.append(dict(panel=stem.name, path=str(stem.relative_to(ROOT)), png=str(stem.with_suffix('.png').relative_to(ROOT)), n=len(source), visual_inspection='not executed by agent; local run requested'))
        plt.close(fig)

    def kd(self, d, col, stem, simulation=False, common=None):
        fig, ax = axes()
        xerr = 'Sim_log_error' if simulation else None
        if simulation:
            d['Sim_log_error'] = d.Sim_B22_SEM / (abs(d.Sim_B22_final) * np.log(10))
        xe = d[xerr].fillna(0) if xerr else np.zeros(len(d))
        ye = d.pKd_error.fillna(0)
        xv, yv = np.r_[d[col] - xe, d[col] + xe], np.r_[d.pKd - ye, d.pKd + ye]
        xl, yl = common if common else (limits(xv), limits(yv))
        xl, yl = self.axis_limits(stem.name, 'x', xv, xl), self.axis_limits(stem.name, 'y', yv, yl)
        ax.set(xlim=xl, ylim=yl, xlabel=SIM_LABEL if simulation else PRED_LABEL, ylabel='Experimental pKd')
        for partner, g in d.groupby('partner_display', sort=False):
            ax.errorbar(g[col], g.pKd, yerr=g.pKd_error.fillna(0), xerr=g[xerr].fillna(0) if xerr else None,
                fmt=self.markers.get(partner, 'o'), ms=5, mfc='#0072B2' if partner.startswith('THOC4') else '#999999',
                mec='black', mew=.4, ecolor='#999999', elinewidth=.7, capsize=2, linestyle='None', label=partner)
        if len(d) >= 2 and np.ptp(d[col]) > 0:
            m, b = np.polyfit(d[col], d.pKd, 1)
            xx = np.array([d[col].min(), d[col].max()])
            ax.plot(xx, m * xx + b, 'k--', lw=.7)
        rho, p, test, draws = spearman_test(d[col], d.pKd, self.settings)
        ax.text(0, 1.04, f'n = {len(d)}   ρ = {ptext(rho)}   p = {ptext(p)}', transform=ax.transAxes, fontsize=8)
        if not d.empty:
            ax.legend(frameon=False, ncol=2 if d.partner_display.nunique() <= 4 else 3, loc='lower center', bbox_to_anchor=(.5, 1.11), columnspacing=.8, handletextpad=.4)
            identity_labels(fig, ax, d, col, 'pKd', xerr=xerr, yerr='pKd_error')
        else:
            ax.text(.5, .5, 'No eligible pairs', ha='center', transform=ax.transAxes)
        st = pd.DataFrame([dict(n=len(d), Spearman_rho=rho, Spearman_two_sided_p=p, Pearson_r=corr(d[col], d.pKd, 'pearson'), test=test,
            permutations=draws, seed=self.settings['seed'], exclusions='see panel membership report', uncertainty='reported Kd error propagated without redefining its type')])
        self.export(fig, stem, d, st, CAPTION + f' Spearman method: {test}. Dashed fitted line is descriptive; all eligible rows included. Unavailable error bars are omitted; their source values remain NA.')

    def outcome(self, d, col, stem, simulation=False, common_xlim=None):
        fig, ax = axes()
        fig.subplots_adjust(left=.23)
        st = outcome_tests(d, col, self.settings)
        categories = [c for c in ['Kd fit', 'LB', 'NB'] if c in set(d.Outcome)]
        for i, category in enumerate(categories):
            g = d[d.Outcome.eq(category)].sort_values(['partner_display', 'pair_id']).copy()
            g['outcome_display_y'] = i + (np.linspace(-.18, .18, len(g)) if len(g) > 1 else 0)
            d.loc[g.index, 'outcome_display_y'] = g.outcome_display_y
            ax.scatter(g[col], g.outcome_display_y, s=22, c=COLORS[category])
        xl = self.axis_limits(stem.name, 'x', d[col], common_xlim or limits(d[col]))
        ax.set(xlim=xl, ylim=(-.45, max(len(categories) - .55, .55)), xlabel=SIM_LABEL if simulation else PRED_LABEL, ylabel='NMR outcome')
        ax.set_yticks(range(len(categories)), [f'{c} (n={int(d.Outcome.eq(c).sum())})' for c in categories])
        ax.xaxis.grid(True, lw=.4, color='#dddddd')
        pv = st.iloc[0].p_raw
        ax.text(0, 1.04, f'n = {len(d)}   omnibus p = {ptext(pv)}', transform=ax.transAxes, fontsize=8)
        if d.empty:
            ax.text(.5, .5, 'No eligible pairs', ha='center', transform=ax.transAxes)
        results = '; '.join(f'{r.comparison}: Holm-adjusted p={ptext(r.p_adjusted)}' for _, r in st.iloc[1:4].iterrows())
        self.export(fig, stem, d, st, CAPTION + ' Omnibus: permutation Kruskal–Wallis. Pairwise: two-sided permutation Mann–Whitney rank tests, Holm adjustment across three planned comparisons. ' + results + '. LB is excluded from the separate Kd fit versus NB AUROC diagnostic; no affinity order is imposed.')

    def accuracy(self, d, col, stem):
        fig, ax = axes(size=(4.8, 4.2))
        values = np.r_[d[SIM], d[col]]
        common = limits(values)
        xl = self.axis_limits(stem.name, 'x', values, common)
        yl = self.axis_limits(stem.name, 'y', values, common)
        if xl != yl:
            raise ValueError(f'{stem.name}: identity panels require identical x/y limits')
        ax.set(xlim=xl, ylim=yl, aspect='equal', xlabel=SIM_LABEL, ylabel=PRED_LABEL)
        ax.plot(xl, xl, 'k--', lw=.7)
        for category, g in d.groupby(d.Outcome.fillna('not tested')):
            ax.scatter(g[SIM], g[col], s=22, c=COLORS.get(category, '#CC79A7'), label=category)
        metrics = agreement_metrics(d[SIM], d[col])
        ax.text(0, 1.04, f'n = {len(d)}   R² = {ptext(metrics["R_squared"])}   RMSE = {ptext(metrics["RMSE"])}', transform=ax.transAxes, fontsize=8)
        if len(d):
            ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(.5, 1.11), ncol=3)
        else:
            ax.text(.5, .5, 'No eligible pairs', transform=ax.transAxes, ha='center')
        self.export(fig, stem, d, pd.DataFrame([dict(**metrics, test='accuracy against simulated values; eligible rows only', exclusions='see membership', bias_definition='prediction minus simulation')]),
            CAPTION + ' Comparison eligibility follows the existing per-pair convention/sequence audit, additionally bound to selected values and sequences. Adjacent topology/RDF linkage remains contextual; exact training-target lineage is incompletely documented.')

    def intensity_correlation(self):
        panel = 'intensity_ratio_vs_predicted_and_simulated'
        required = 'I_ratio_mean_below_threshold'
        if required not in self.d:
            raise ValueError(f'{required} is absent from the selected experimental source')
        d = self.select(panel, [
            (np.isfinite(pd.to_numeric(self.d[required], errors='coerce')), 'mean-below-threshold intensity ratio unavailable'),
            (np.isfinite(self.d[PRED]), 'prediction unavailable'),
            (np.isfinite(self.d[SIM]), 'simulation unavailable')]).copy()
        d[required] = pd.to_numeric(d[required], errors='coerce')
        fig, ax = axes(size=(4.8, 4.2))
        colors = {'Predicted': '#0072B2', 'Simulated': '#D55E00'}
        markers = {'Predicted': 'o', 'Simulated': '^'}
        rows, stats = [], []
        for _, row in d.iterrows():
            ax.plot([row[PRED], row[SIM]], [row[required], row[required]], color='#cccccc', lw=.55, zorder=1)
        for label, column in [('Predicted', PRED), ('Simulated', SIM)]:
            x = d[column].to_numpy(float)
            y = d[required].to_numpy(float)
            ax.scatter(x, y, s=28, marker=markers[label], color=colors[label], edgecolor='black', linewidth=.35, label=label, zorder=2)
            if len(d) >= 2 and np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                xx = np.linspace(x.min(), x.max(), 100)
                ax.plot(xx, slope * xx + intercept, color=colors[label], lw=.8, ls='--')
            rho, p, test, draws = spearman_test(x, y, self.settings)
            stats.append(dict(series=label, n=len(d), Pearson_r=corr(x, y, 'pearson'), Spearman_rho=rho,
                              Spearman_two_sided_p=p, test=test, permutations=draws, seed=self.settings['seed']))
            for _, z in d.iterrows():
                rows.append(dict(pair_id=z.pair_id, pair_display_name=z.pair_display_name, Outcome=z.Outcome,
                                 series=label, log10_negB22=z[column], I_ratio_mean_below_threshold=z[required]))
        s = pd.DataFrame(stats)
        ax.set_xlabel(r'Predicted or simulated $\log_{10}(-B_{22})$')
        ax.set_ylabel('Mean intensity ratio below threshold')
        ax.legend(frameon=False, loc='upper right')
        ax.text(0, 1.04, '   '.join(f'{r["series"]}: ρ = {ptext(r.Spearman_rho)}, p = {ptext(r.Spearman_two_sided_p)}' for _, r in s.iterrows()),
                transform=ax.transAxes, fontsize=7)
        source = pd.DataFrame(rows)
        self.export(fig, self.supp / panel, source, s,
                    CAPTION + ' Each construct contributes paired predicted and simulated values at one experimental mean-below-threshold intensity ratio; grey segments connect the pair. Dashed lines are descriptive linear fits. Spearman tests are two-sided.')

    def paired(self, d, old, stem, delta=False, new=PRED, xlabel=None):
        d = d.sort_values(['partner_display', 'AHD_name', 'pair_id']).copy()
        d['delta'] = d[new] - d[old]
        d['old_rank'] = d[old].rank(ascending=False, method='average')
        d['new_rank'] = d[new].rank(ascending=False, method='average')
        d['rank_change'] = d.new_rank - d.old_rank
        summary = pd.DataFrame([paired_change(d[old], d[new], self.settings)])
        csv_write(d, self.reports / (stem.name + '_all_rows.csv'))
        for page, start in enumerate(range(0, max(len(d), 1), 25), 1):
            sub = d.iloc[start:start + 25].copy()
            fig, ax = axes(size=(7.2, max(3.5, .27 * len(sub) + 1.9)))
            fig.subplots_adjust(left=.32, right=.81, bottom=.12, top=.86)
            y = np.arange(len(sub))
            if delta:
                ax.set_xlim(limits(d.delta, zero=True))
                ax.axvline(0, color='black', ls='--', lw=.7)
                ax.scatter(sub.delta, y, s=22, c=sub.Outcome.map(COLORS).fillna('#CC79A7'))
                ax.set_xlabel(r'$\Delta$ predicted $\log_{10}(-B_{22})$, new $-$ old')
            else:
                ax.set_xlim(limits(np.r_[d[old], d[new]]))
                ax.hlines(y, np.minimum(sub[old], sub[new]), np.maximum(sub[old], sub[new]), color='#aaaaaa', lw=.6)
                ax.scatter(sub[old], y, marker='o', s=22, facecolor='white', edgecolor='#0072B2', label='Old')
                ax.scatter(sub[new], y, marker='D', s=22, color='#D55E00', label='New')
                ax.set_xlabel(xlabel or PRED_LABEL)
                ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(.5, 1.07))
            ax.set_yticks(y, sub.pair_display_name)
            ax.set_ylim(max(len(sub) - .4, .6), -.6)
            for i, (_, row) in enumerate(sub.iterrows()):
                ax.text(1.025, i, row.Outcome if pd.notna(row.Outcome) else 'not tested', transform=ax.get_yaxis_transform(), va='center', fontsize=7)
                if i and sub.iloc[i - 1].partner_display != row.partner_display:
                    ax.axhline(i - .5, color='#dddddd', lw=.5)
            s = summary.iloc[0]
            ax.text(0, 1.025, f'n = {len(d)}   r = {ptext(s.Pearson_r)}   ρ = {ptext(s.Spearman_rho)}   mean Δ = {s.mean_delta:+.3f}', transform=ax.transAxes, fontsize=7)
            dest = stem if len(d) <= 25 else stem.parent / (stem.name + f'_page{page:02d}')
            self.export(fig, dest, sub, summary, CAPTION + f' Mean paired change 95% CI [{s.mean_delta_CI95_low}, {s.mean_delta_CI95_high}]; {s.shift_test}, p={s.shift_pvalue}. Rows grouped by RG/RGG partner; pages share limits.')

    def agreement(self, d, old, stem):
        fig, ax = axes(size=(4.8, 4.2))
        common = limits(np.r_[d[old], d[PRED]])
        ax.set(xlim=common, ylim=common, aspect='equal', xlabel=r'Old predicted $\log_{10}(-B_{22})$', ylabel=r'New predicted $\log_{10}(-B_{22})$')
        ax.plot(common, common, 'k--', lw=.7)
        for category, g in d.groupby(d.Outcome.fillna('not tested')):
            ax.scatter(g[old], g[PRED], s=22, c=COLORS.get(category, '#CC79A7'), label=category)
        s = paired_change(d[old], d[PRED], self.settings)
        ax.text(0, 1.04, f'n = {len(d)}   r = {ptext(s["Pearson_r"])}   ρ = {ptext(s["Spearman_rho"])}   mean Δ = {s["mean_delta"]:+.3f}', transform=ax.transAxes, fontsize=7.5)
        if len(d):
            ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(.5, 1.11))
        self.export(fig, stem, d, pd.DataFrame([s]))

    def heatmap(self, d, stem):
        if d.empty:
            return
        d = d.copy()
        d['AHD_display'] = d.AHD_name.map(pretty_ahd)
        d['RGG_display'] = d.RGG_name.map(pretty_rgg)
        ahds = sorted(d.AHD_display.unique(), key=lambda s: ('shuffle' in s, s))
        rggs = sorted(d.RGG_display.unique())
        old = d.pivot(index='AHD_display', columns='RGG_display', values=OLD).reindex(index=ahds, columns=rggs)
        new = d.pivot(index='AHD_display', columns='RGG_display', values=PRED).reindex(index=ahds, columns=rggs)
        delta = new - old
        common = limits(np.r_[d[OLD], d[PRED]], pad=0)
        bound = max(float(np.nanmax(abs(delta.to_numpy()))), .001)
        for page, start in enumerate(range(0, len(ahds), 24), 1):
            fig, axs = plt.subplots(1, 3, figsize=(max(12, len(rggs) * 1.25), max(4.5, .28 * min(24, len(ahds) - start) + 2)))
            for ax, matrix, title in zip(axs, [old, new, delta], ['Old', 'New', 'New − old']):
                matrix = matrix.iloc[start:start + 24]
                change = title == 'New − old'
                image = ax.imshow(matrix, aspect='auto', cmap='coolwarm' if change else 'viridis', vmin=-bound if change else common[0], vmax=bound if change else common[1])
                ax.set_xticks(range(len(rggs)), rggs, rotation=55, ha='right')
                ax.set_yticks(range(len(matrix)), matrix.index)
                ax.set_title(title)
                for i in range(len(matrix)):
                    for j in range(len(rggs)):
                        value = matrix.iloc[i, j]
                        if np.isfinite(value):
                            ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=6.5,
                                    color='black' if change else ('white' if value < np.mean(common) else 'black'))
                fig.colorbar(image, ax=ax, fraction=.045, pad=.03)
            fig.tight_layout()
            sub = d[d.AHD_display.isin(ahds[start:start + 24])]
            dest = stem if len(ahds) <= 24 else stem.parent / (stem.name + f'_page{page:02d}')
            self.export(fig, dest, sub, pd.DataFrame([paired_change(d[OLD], d[PRED], self.settings)]))

    def paired_outcome_differences(self, d, old):
        rows = []
        rng = np.random.default_rng(self.settings['seed'])
        for a, b in [('Kd fit', 'LB'), ('Kd fit', 'NB'), ('LB', 'NB')]:
            aa, bb = d[d.Outcome.eq(a)], d[d.Outcome.eq(b)]
            oa, ob, na, nb = aa[old].to_numpy(), bb[old].to_numpy(), aa[PRED].to_numpy(), bb[PRED].to_numpy()
            observed = effect(na, nb) - effect(oa, ob)
            boot = []
            if len(aa) and len(bb):
                for _ in range(self.settings['bootstrap']):
                    ia = rng.integers(0, len(aa), len(aa))
                    ib = rng.integers(0, len(bb), len(bb))
                    boot.append(effect(na[ia], nb[ib]) - effect(oa[ia], ob[ib]))
            lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
            rows.append(dict(comparison=f'{a} vs {b}', metric='paired rank-biserial difference', n=len(aa) + len(bb), estimate=observed,
                             CI95_low=lo, CI95_high=hi, test='same within-category rows resampled for both predictors; shared constructs'))
            if a == 'Kd fit' and b == 'NB':
                rows.append(dict(comparison='Kd fit vs NB', metric='paired AUROC difference', n=len(aa) + len(bb), estimate=observed / 2,
                    CI95_low=lo / 2, CI95_high=hi / 2, test='same within-category rows resampled; LB excluded'))
        return pd.DataFrame(rows)

    def diagnostics(self):
        exp = self.select('diagnostic_experimental_old_new', [
            (self.d.experimental_status.ne('no experimental measurement supplied'), 'no experiment supplied'),
            (~self.d.experimental_status.str.contains('selection required', na=False), 'experimental selection unresolved'),
            (np.isfinite(self.d.Historical_experimental_prediction), 'old prediction unavailable'),
            (np.isfinite(self.d[PRED]), 'new prediction unavailable'),
            (self.d.prediction_sequence_status.ne('sequence conflict or missing'), 'prediction sequence conflict or missing'),
            (self.d.experimental_construct_status.ne('sequence conflict'), 'experimental sequence conflict')])
        archived = 'Historical_experimental_prediction'
        if archived not in exp:
            exp[archived] = np.nan
        self.paired(exp, archived, self.diag / 'master_experimental_named_paired_predictions')
        self.paired(exp, archived, self.diag / 'master_experimental_named_deltas', delta=True)
        self.agreement(exp, archived, self.diag / 'master_experimental_old_new_agreement')
        summary = [dict(set='experimental pairs', **paired_change(exp[archived], exp[PRED], self.settings))]
        for field in ['partner_display', 'Outcome']:
            for name, g in exp.groupby(field):
                summary.append(dict(set=f'{field}: {name}', **paired_change(g[archived], g[PRED], self.settings)))
        csv_write(pd.DataFrame(summary), self.reports / 'experimental_old_new_summary.csv')
        kd = exp[exp.Outcome.eq('Kd fit') & np.isfinite(exp.pKd) & exp.experimental_construct_status.ne('sequence conflict')].copy()
        common = (limits(np.r_[kd[archived], kd[PRED]]), limits(np.r_[kd.pKd - kd.pKd_error.fillna(0), kd.pKd + kd.pKd_error.fillna(0)]))
        for label, col in [('old', archived), ('new', PRED)]:
            self.kd(kd, col, self.diag / ('master_' + label + '_prediction_vs_pKd'), common=common)
            self.outcome(exp[exp.Outcome.isin(['Kd fit', 'LB', 'NB'])].copy(), col,
                         self.diag / ('master_' + label + '_prediction_vs_outcome'), common_xlim=limits(np.r_[exp[archived], exp[PRED]]))
        csv_write(paired_correlations(kd, archived, PRED, 'pKd', self.settings), self.reports / 'paired_Kd_correlation_comparison.csv')
        csv_write(self.paired_outcome_differences(exp, archived), self.reports / 'paired_outcome_effect_differences.csv')
        complete = self.select('diagnostic_complete_old_new', [
            (np.isfinite(self.d[OLD]), 'same-construct old prediction unavailable'),
            (np.isfinite(self.d[PRED]), 'new prediction unavailable'),
            (self.d.historical_sequence_status.eq('exact sequences match'), 'historical construct identity unresolved'),
            (self.d.prediction_sequence_status.ne('sequence conflict or missing'), 'prediction sequence conflict or missing')])
        self.agreement(complete, OLD, self.diag / 'master_complete_old_new_agreement')
        self.heatmap(complete, self.diag / 'master_complete_named_heatmaps')
        groups = [dict(set='complete', **paired_change(complete[OLD], complete[PRED], self.settings))]
        for shuffled, g in complete.groupby('is_shuffle'):
            groups.append(dict(set='shuffle' if shuffled else 'WT', **paired_change(g[OLD], g[PRED], self.settings)))
        csv_write(pd.DataFrame(groups), self.reports / 'complete_WT_shuffle_summary.csv')
        c = complete.set_index('pair_id')
        shuffle_rows = []
        for _, row in complete[complete.is_shuffle].iterrows():
            if row.parent_pair_id in c.index:
                parent = c.loc[row.parent_pair_id]
                shuffle_rows.append(dict(pair_id=row.pair_id, pair_display_name=row.pair_display_name,
                    parent_pair_id=row.parent_pair_id, partner_display=row.partner_display, AHD_name=row.AHD_name, Outcome=row.Outcome,
                    old_shuffle_minus_WT=row[OLD] - parent[OLD], new_shuffle_minus_WT=row[PRED] - parent[PRED]))
        shuffle = pd.DataFrame(shuffle_rows)
        csv_write(shuffle, self.reports / 'matched_WT_shuffle_changes.csv')
        if len(shuffle):
            self.paired(shuffle, 'old_shuffle_minus_WT', self.diag / 'master_matched_WT_shuffle_changes',
                        new='new_shuffle_minus_WT', xlabel=r'Predicted $\log_{10}(-B_{22})$ change, shuffle $-$ WT')
        sim = self.select('diagnostic_paired_simulation_accuracy', [
            (boolean(self.d.simulation_accuracy_eligible), 'simulation comparison eligibility unresolved'),
            (np.isfinite(self.d[SIM]), 'simulation log unavailable'),
            (np.isfinite(self.d[OLD]) & np.isfinite(self.d[PRED]), 'old or new prediction unavailable'),
            (self.d.historical_sequence_status.eq('exact sequences match'), 'historical construct identity unresolved')])
        for label, col in [('old', OLD), ('new', PRED)]:
            self.accuracy(sim, col, self.diag / ('master_' + label + '_prediction_vs_simulation'))
        csv_write(paired_performance(sim, OLD, PRED, SIM, self.settings), self.reports / 'paired_simulation_accuracy.csv')

    def run(self, diagnostics=True):
        plt.rcParams.update(STYLE)
        self.reports.mkdir(parents=True, exist_ok=True)
        d = self.select('prediction_vs_pKd', self.fitted_criteria(PRED))
        self.kd(d, PRED, self.paper / 'prediction_vs_pKd')
        d = self.select('prediction_vs_outcome', [(self.d.Outcome.isin(['Kd fit', 'LB', 'NB']), 'no eligible annotated outcome'),
            (np.isfinite(self.d[PRED]), 'prediction unavailable'), (~self.d.experimental_status.str.contains('selection required', na=False), 'experimental selection unresolved'),
            (self.d.experimental_construct_status.ne('sequence conflict'), 'experimental sequence conflict'),
            (self.d.prediction_sequence_status.ne('sequence conflict or missing'), 'prediction sequence conflict or missing')])
        self.outcome(d, PRED, self.paper / 'prediction_vs_outcome')
        d = self.select('simulation_vs_pKd', self.fitted_criteria(SIM, simulation=True))
        self.kd(d, SIM, self.paper / 'simulation_vs_pKd', simulation=True)
        d = self.select('simulation_vs_outcome', [(self.d.Outcome.isin(['Kd fit', 'LB', 'NB']), 'no eligible annotated outcome'),
            (np.isfinite(self.d[SIM]), 'simulation log unavailable'),
            (self.d.experimental_construct_status.ne('sequence conflict'), 'experimental sequence conflict'),
            (~self.d.experimental_status.str.contains('selection required', na=False), 'experimental selection unresolved')])
        self.outcome(d, SIM, self.paper / 'simulation_vs_outcome', simulation=True)
        d = self.select('prediction_vs_simulation', [(boolean(self.d.simulation_accuracy_eligible), 'simulation comparison eligibility unresolved'),
            (np.isfinite(self.d[SIM]) & np.isfinite(self.d[PRED]), 'prediction or simulation log unavailable')])
        self.accuracy(d, PRED, self.paper / 'prediction_vs_simulation')
        self.intensity_correlation()
        if diagnostics:
            self.diagnostics()
        csv_write(pd.concat(self.members, ignore_index=True), self.reports / 'analysis_membership.csv')
        csv_write(pd.DataFrame(self.exports), self.out / 'master_figure_manifest.csv')
        (self.out / 'master_figure_axis_limits.json').write_text(json.dumps(self.axis_records, indent=2) + '\n')
        fig, axs = plt.subplots(2, 3, figsize=(12, 7))
        for ax, name in zip(axs.flat, ['prediction_vs_pKd', 'prediction_vs_outcome', 'simulation_vs_pKd', 'simulation_vs_outcome', 'prediction_vs_simulation']):
            ax.imshow(plt.imread(self.paper / (name + '.png')))
            ax.set_title(name.replace('_', ' '), fontsize=9)
            ax.axis('off')
        axs.flat[-1].axis('off')
        fig.tight_layout()
        fig.savefig(self.paper / 'paper_overview.png', dpi=180)
        plt.close(fig)
        (self.out / 'MASTER_METHODS.md').write_text(CAPTION + '\n\n'
            'Every source and its SHA256 is recorded in source_selection.json. Experimental duplicate rows are preserved; unselected conflicts have no pair-level Kd. '
            'Paper prediction/pKd limits use only the current displayed predictor values. Shared limits occur only in old/new diagnostic comparisons. '
            'Simulated values are retained even when comparison provenance is unresolved. Accuracy comparisons bind the existing audit to the selected simulation values and exact sequences. '
            'No RDF, B22 or model recovery is rerun by this entrypoint. Anonymous feature rows require an explicit feature_row map and matching manifest hashes. '
            'Additional archived descriptors remain separately named, including their original lengths and matching status.\n')
