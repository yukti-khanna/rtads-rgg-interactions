"""Identity-based union of existing computational sources and replacement experiments.

No RDF calculation or model inference occurs here. All writes are confined to
fig6; inputs may be read from elsewhere. Ambiguous measurements stay in the
supporting table and have no selected pair-level measurement by default.
"""
import copy
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ESSENTIAL = ['pair_id', 'pair_display_name', 'Kd_uM', 'Kd_error_uM',
             'Pred_log10_negB22_final', 'Sim_log10_negB22_final', 'Outcome', 'Notes']
PARTNERS = {'thoc4_nter': 'THOC4 N-term', 'thoc4_cter': 'THOC4 C-term',
            'g3bp1_rgg': 'G3BP1', 'taf15_rgg': 'TAF15', 'snrnp70_s': 'SNRNP70 short',
            'rbmx': 'RBMX', 'tceal6': 'TCEAL6', 'q8n319': 'Q8N319', 'rab44': 'RAB44'}
AHD = {'foxo4_cr3': 'FOXO4', 'foxo6_cr3': 'FOXO6', 'atf4_tad': 'ATF4',
       'tfe3_cter': 'TFE3', 'lef1_nter': 'LEF1', 'foxm1': 'FOXM1', 'tp53': 'p53', 'sox12': 'SOX12'}


def path(value):
    p = Path(value).expanduser()
    return (ROOT / p).resolve() if not p.is_absolute() else p.resolve()


def output_path(value):
    p = path(value)
    if not p.is_relative_to(ROOT):
        raise ValueError(f'Outputs must stay inside fig6: {p}')
    return p


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def csv_write(df, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, na_rep='NA')


def unique(df, label, key='pair_id'):
    if key not in df or df[key].isna().any() or df[key].duplicated().any():
        raise ValueError(f'{label}: missing or duplicate {key}; resolve computational identity conflicts explicitly')


def pretty_ahd(name):
    if pd.isna(name):
        return ''
    parent, *shuffle = str(name).split('_shuffle_')
    return AHD.get(parent, parent.upper()) + (f' shuffle {shuffle[0]}' if shuffle else '')


def pretty_rgg(name):
    if pd.isna(name):
        return ''
    parent, *shuffle = str(name).split('_shuffle_')
    return PARTNERS.get(parent, parent.upper()) + (f' shuffle {shuffle[0]}' if shuffle else '')


def boolean(series):
    return series.astype(str).str.lower().isin(['true', '1'])


def load_config(filename):
    return json.loads(path(filename).read_text())


class MasterBuilder:
    def __init__(self, config, outdir=None):
        self.config = copy.deepcopy(config)
        self.out = output_path(outdir or config['outdir'])
        self.sources, self.mapping, self.columns = [], [], {}
        self.missing = []
        self.aliases = {}

    def read(self, value, role, required=False):
        if not value:
            return None
        p = path(value)
        if not p.exists():
            self.sources.append(dict(role=role, path=str(p), status='missing', sha256=None))
            self.missing.append(f'{role}: {p}')
            if required:
                raise FileNotFoundError(p)
            return None
        self.sources.append(dict(role=role, path=str(p), status='available', sha256=digest(p)))
        return pd.read_csv(p)

    def identify(self, df, role, column='pair_id'):
        if column not in df:
            raise ValueError(f'{role}: pair identifier column {column!r} absent')
        df = df.copy()
        raw = df[column].astype('string').str.strip()
        if raw.isna().any() or raw.eq('').any():
            raise ValueError(f'{role}: blank pair identifiers')
        df['pair_id'] = raw.map(lambda x: self.aliases.get(x, x))
        for original, canonical in zip(raw, df.pair_id):
            self.mapping.append(dict(source=role, source_pair_id=original, pair_id=canonical,
                                     rule='verified explicit alias' if original in self.aliases else 'exact pair ID'))
        return df

    def define(self, column, source, source_column=None, transform='none', model_input=False, note=''):
        self.columns[column] = dict(column=column, source=source, source_column=source_column or column,
                                    transformation=transform, matching_rule='exact pair ID or verified alias',
                                    model_input=model_input, notes=note)

    def join(self, master, frame, label):
        if frame is None:
            return master
        unique(frame, label)
        overlap = set(master.columns) & set(frame.columns) - {'pair_id'}
        if overlap:
            raise ValueError(f'{label}: unexpected overlapping output columns {sorted(overlap)}')
        return master.merge(frame, on='pair_id', how='outer', validate='one_to_one')

    def experimental(self):
        spec = self.config['experiments']
        raw = self.read(spec['path'], 'experiments')
        empty = pd.DataFrame(columns=['pair_id', 'Kd_uM', 'Kd_error_uM', 'Outcome', 'Notes', 'experimental_status'])
        if raw is None:
            csv_write(empty, self.out / 'experimental_measurements.csv')
            csv_write(empty, self.out / 'experimental_conflicts.csv')
            return empty
        col = spec['columns']
        raw = self.identify(raw, 'experiments', col['pair_id'])
        raw['experiment_input_row'] = np.arange(1, len(raw) + 1)
        units = spec.get('kd_units')
        factor = {'M': 1e6, 'mM': 1e3, 'uM': 1., 'µM': 1., 'μM': 1., 'nM': .001}.get(units)
        if factor is None:
            raise ValueError('Explicit Kd units required: M, mM, uM, or nM')
        invalid_numeric = pd.Series(False, index=raw.index)
        for field, target in [('Kd', 'Kd_uM'), ('Kd_error', 'Kd_error_uM')]:
            field_units = spec.get('kd_error_units', units) if field == 'Kd_error' else units
            field_factor = {'M': 1e6, 'mM': 1e3, 'uM': 1., 'µM': 1., 'μM': 1., 'nM': .001}.get(field_units)
            if field_factor is None:
                raise ValueError('Explicit Kd/error units required: M, mM, uM, or nM')
            source = col.get(field)
            if source:
                if source not in raw:
                    raise ValueError(f'Configured experimental column {source!r} missing; provide an explicit column mapping')
                values = pd.to_numeric(raw[source], errors='coerce')
                bad = raw[source].notna() & values.isna()
                invalid_numeric |= bad
                raw['reported_' + field] = raw[source]
                raw[target] = values * field_factor
            else:
                raw[target] = np.nan
            self.define(target, spec['path'], source, f'convert {field_units} to uM by factor {field_factor}; preserve reported uncertainty')
        for field in ['Outcome', 'Notes']:
            source = col.get(field)
            raw[field] = raw[source] if source and source in raw else pd.NA
            self.define(field, spec['path'], source)
        additional = spec.get('additional_columns', [])
        missing_additional = [c for c in additional if c not in raw]
        if missing_additional:
            raise ValueError(f'Configured experimental metadata columns missing: {missing_additional}')
        for c in additional:
            self.define(c, spec['path'], c, note='experimental metadata retained without transformation')
        raw['Outcome'] = raw.Outcome.replace({'Kd_fit': 'Kd fit', 'Kd-fit': 'Kd fit'})
        # A finite positive fitted Kd can supply its category; missing Kd never implies NB.
        raw.loc[raw.Outcome.isna() & raw.Kd_uM.gt(0), 'Outcome'] = 'Kd fit'
        raw['uncertainty_type'] = spec.get('uncertainty_type', 'reported; unspecified')
        raw['experimental_status'] = 'annotation supplied'
        raw.loc[raw.Outcome.eq('NB'), 'experimental_status'] = 'explicit NB'
        raw.loc[raw.Outcome.eq('LB') & raw.Kd_uM.isna(), 'experimental_status'] = 'LB without fitted Kd'
        raw.loc[raw.Kd_uM.gt(0), 'experimental_status'] = 'fitted Kd supplied'
        invalid = (raw.Kd_uM.notna() & ~raw.Kd_uM.gt(0)) | raw.Kd_error_uM.lt(0)
        invalid |= invalid_numeric
        invalid |= (raw.Kd_uM.notna() & ~np.isfinite(raw.Kd_uM)) | (raw.Kd_error_uM.notna() & ~np.isfinite(raw.Kd_error_uM))
        invalid |= raw.Kd_uM.notna() & raw.Outcome.isin(['NB', 'LB'])
        raw.loc[invalid, 'experimental_status'] = 'invalid or contradictory measurement; selection required'
        raw['selected_for_pair'] = False
        rows = []
        for key, g in raw.groupby('pair_id', sort=False):
            selected = g if len(g) == 1 else g.iloc[0:0]
            rule = spec.get('select_measurements', {}).get(key)
            if rule is not None:
                idcol = spec.get('measurement_id_column') or 'experiment_input_row'
                if idcol not in g:
                    raise ValueError(f'Measurement selection column {idcol} missing')
                selected = g[g[idcol].astype(str).eq(str(rule))]
                if len(selected) != 1:
                    raise ValueError(f'{key}: explicit measurement selection is not unique')
            if len(selected) != 1 or selected.experimental_status.str.startswith('invalid').any():
                rows.append(dict(pair_id=key, Kd_uM=np.nan, Kd_error_uM=np.nan,
                    Outcome=pd.NA, Notes='See experimental_measurements.csv; explicit measurement selection required',
                    experimental_status='duplicate/conflicting measurements; selection required', uncertainty_type=spec.get('uncertainty_type'),
                    **{c: pd.NA for c in additional}))
            else:
                raw.loc[selected.index, 'selected_for_pair'] = True
                keep = ['pair_id', 'Kd_uM', 'Kd_error_uM', 'Outcome', 'Notes', 'experimental_status', 'uncertainty_type', 'experiment_input_row'] + additional
                row = selected.iloc[0][keep].to_dict()
                for role in ['AHD', 'RGG']:
                    if f'{role}_sequence' in selected:
                        row[f'experiment_{role}_sequence'] = selected.iloc[0][f'{role}_sequence']
                rows.append(row)
        csv_write(raw, self.out / 'experimental_measurements.csv')
        csv_write(raw[raw.pair_id.duplicated(keep=False) | invalid], self.out / 'experimental_conflicts.csv')
        return pd.DataFrame(rows) if rows else empty

    def build(self):
        self.out.mkdir(parents=True, exist_ok=True)
        alias = self.read(self.config.get('aliases_path'), 'explicit aliases')
        if alias is not None and not alias.empty:
            required = {'source_pair_id', 'pair_id', 'evidence', 'verified'}
            if not required.issubset(alias):
                raise ValueError(f'Alias table requires {required}')
            if not boolean(alias.verified).all() or alias.evidence.isna().any() or alias.evidence.astype(str).str.strip().eq('').any():
                raise ValueError('Aliases require explicit verified=true and nonempty construct evidence')
            unique(alias, 'aliases', 'source_pair_id')
            self.aliases = dict(zip(alias.source_pair_id, alias.pair_id))
        registry = self.read(self.config.get('sequence_registry'), 'sequence registry')
        if registry is not None:
            registry = self.identify(registry, 'sequence registry')
            unique(registry, 'sequence registry')
            if not registry.pair_id.eq(registry.RGG_name + '__' + registry.AHD_name).all():
                raise ValueError('Sequence registry roles disagree with oriented pair IDs')
            display = registry[['pair_id', 'AHD_name', 'RGG_name', 'AHD_sequence', 'RGG_sequence']].copy()
            display['source_pair_id'] = [pretty_ahd(a) + '–' + pretty_rgg(r) for a, r in zip(display.AHD_name, display.RGG_name)]
            display['evidence'] = 'exact complete display name generated from oriented sequence registry; no approximate matching'
            display['verified'] = True
            ambiguous = display.source_pair_id.duplicated(keep=False)
            csv_write(display[ambiguous], self.out / 'ambiguous_display_aliases.csv')
            for _, row in display[~ambiguous].iterrows():
                if row.source_pair_id in self.aliases and self.aliases[row.source_pair_id] != row.pair_id:
                    raise ValueError(f'Explicit alias conflicts with registry display name: {row.source_pair_id}')
                self.aliases[row.source_pair_id] = row.pair_id
            csv_write(display[~ambiguous], self.out / 'verified_display_aliases.csv')
            if alias is not None:
                ri = registry.set_index('pair_id')
                for _, row in alias.iterrows():
                    if row.pair_id in ri.index:
                        for role in ['AHD', 'RGG']:
                            col = role + '_sequence'
                            if col in row and pd.notna(row[col]) and row[col] != ri.loc[row.pair_id, col]:
                                raise ValueError(f'Alias sequence conflicts with registry: {row.source_pair_id}, {role}')
        exp = self.experimental()
        master = exp
        ps = self.config['predictions']
        pred = self.read(ps['path'], 'predictions')
        if pred is not None:
            pred = self.identify(pred, 'predictions', ps['pair_column'])
            unique(pred, 'predictions')
            p = pred[['pair_id']].copy()
            for target, key in [('Pred_log10_negB22_final', 'final_column'), ('New_base_prediction', 'base_column'), ('New_applied_correction', 'correction_column')]:
                if ps[key] not in pred:
                    raise ValueError(f'Prediction column {ps[key]} missing')
                p[target] = pd.to_numeric(pred[ps[key]], errors='raise')
                self.define(target, ps['path'], ps[key], 'none; already on supplied logarithmic target scale')
            p['prediction_source'] = str(path(ps['path']))
            master = self.join(master, p, 'predictions')
        ss = self.config['simulations']
        sim = self.read(ss['path'], 'simulations')
        if sim is not None:
            for col, value in ss['filters'].items():
                if col not in sim:
                    raise ValueError(f'Simulation convention column {col} missing; refuse alternative convention')
                mask = np.isclose(pd.to_numeric(sim[col], errors='coerce'), value) if isinstance(value, (float, int)) else sim[col].eq(value)
                sim = sim[mask].copy()
            sim = self.identify(sim, 'simulations', ss['pair_column'])
            unique(sim, 'selected simulation convention')
            s = sim[['pair_id']].copy()
            for target, key in [('Sim_B22_final', 'b22_column'), ('Sim_B22_SD', 'sd_column'), ('Sim_B22_SEM', 'sem_column'), ('Sim_n_replicas', 'replicas_column')]:
                s[target] = pd.to_numeric(sim[ss[key]], errors='raise') if ss[key] in sim else np.nan
                self.define(target, ss['path'], ss[key], note='selected 7.5 nm corr1 replica-mean sum; no physical units assigned')
            s['Sim_log10_negB22_final'] = np.nan
            neg = s.Sim_B22_final.lt(0)
            s.loc[neg, 'Sim_log10_negB22_final'] = np.log10(-s.loc[neg, 'Sim_B22_final'])
            if ss.get('log_column') in sim:
                np.testing.assert_allclose(s.Sim_log10_negB22_final, pd.to_numeric(sim[ss['log_column']], errors='raise'), atol=1e-10, rtol=0, equal_nan=True)
            self.define('Sim_log10_negB22_final', ss['path'], ss['b22_column'], ss['transformation'])
            s['simulation_source'] = str(path(ss['path']))
            s['simulation_convention'] = json.dumps(ss['filters'], sort_keys=True)
            s['Sim_n_replicas_reported'] = s.Sim_n_replicas
            s.loc[s.Sim_n_replicas.le(0), 'Sim_n_replicas'] = np.nan
            for col in ['uncertainty_status', 'rdf_file', 'rdf_sha256', 'endpoint_convention', 'averaged_only', 'units']:
                if col in sim:
                    s['simulation_' + col] = sim[col]
            master = self.join(master, s, 'simulations')
        if registry is not None:
            keep = ['pair_id'] + [c for c in registry if c.startswith(('AHD_', 'RGG_'))]
            master = self.join(master, registry[keep], 'sequence registry')
        # Verify supplied prediction sequences against the selected sequence registry.
        master['prediction_sequence_status'] = 'no sequences available'
        if pred is not None:
            check = pred.set_index('pair_id')
            for ix, row in master.iterrows():
                if row.pair_id not in check.index:
                    continue
                supplied = check.loc[row.pair_id]
                complete_sequences = all(pd.notna(supplied.get(f'{r}_name')) and pd.notna(supplied.get(f'{r}_sequence')) for r in ['AHD', 'RGG'])
                # Existing known registry identities are checked, never overwritten.
                registered = registry is not None and row.pair_id in set(registry.pair_id)
                if registered:
                    match = all(row.get(f'{r}_sequence') == supplied.get(f'{r}_sequence') and row.get(f'{r}_name') == supplied.get(f'{r}_name') for r in ['AHD', 'RGG'])
                    master.loc[ix, 'prediction_sequence_status'] = 'exact sequences match' if match else 'sequence conflict or missing'
                elif complete_sequences and row.pair_id == f'{supplied.RGG_name}__{supplied.AHD_name}':
                    for role in ['AHD', 'RGG']:
                        for suffix in ['name', 'sequence']:
                            master.loc[ix, f'{role}_{suffix}'] = supplied[f'{role}_{suffix}']
                        master.loc[ix, f'{role}_length'] = len(supplied[f'{role}_sequence'])
                    master.loc[ix, 'prediction_sequence_status'] = 'sequences supplied in selected prediction; no independent registry'
                else:
                    master.loc[ix, 'prediction_sequence_status'] = 'sequence conflict or missing'
        fs = self.config['features']
        features = self.read(fs['path'], 'features')
        ids = None
        if features is not None and fs['pair_column'] not in features:
            ids = self.read(fs.get('row_ids_path'), 'feature row identifiers')
            if ids is None:
                csv_write(features.assign(unresolved_reason='explicit feature-row identity map unavailable'), self.out / 'unresolved_feature_rows.csv')
                self.missing.append('current feature identities: anonymous matrix cannot be joined without its explicit row map')
                features = None
            else:
                ids = self.identify(ids, 'feature row identifiers', fs['pair_column'])
                unique(ids, 'feature row identifiers')
                rowcol = fs['feature_row_column']
                numbers = pd.to_numeric(ids[rowcol], errors='raise')
                if numbers.duplicated().any() or not np.array_equal(np.sort(numbers), np.arange(len(features))):
                    raise ValueError('Feature row IDs must explicitly map every zero-based feature row once')
                indexed = features.copy()
                indexed[rowcol] = np.arange(len(features))
                features = ids[['pair_id', rowcol]].merge(indexed, on=rowcol, validate='one_to_one').drop(columns=rowcol)
        elif features is not None:
            features = self.identify(features, 'features', fs['pair_column'])
        model_inputs = []
        if fs.get('manifest_path') and path(fs['manifest_path']).exists():
            self.sources.append(dict(role='prediction manifest', path=str(path(fs['manifest_path'])), status='available', sha256=digest(path(fs['manifest_path']))))
            manifest = json.loads(path(fs['manifest_path']).read_text())
            model_inputs = manifest.get('feature_order', [])
            if ids is not None:
                for name, source in [('feature_matrix.csv', fs['path']), ('feature_row_ids.csv', fs['row_ids_path'])]:
                    expected = manifest.get('prepared_output_hashes', {}).get(name)
                    # Hashes bind the anonymous matrix to its row map. Overrides
                    # with keyed features do not require the original matrix hash.
                    if expected and digest(path(source)) != expected:
                        raise ValueError(f'{name}: differs from manifest; supply keyed features or a matching manifest/map')
        if features is not None:
            features['feature_source'] = str(path(fs['path']))
            for c in features:
                if c not in ['pair_id', 'feature_source']:
                    self.define(c, fs['path'], model_input=c in model_inputs,
                                note='anonymous matrix mapped by explicit feature_row identifier' if ids is not None else 'keyed feature table')
            master = self.join(master, features, 'features')
        # Preserve supplementary metadata and any additional calculated columns.
        for source_key, prefix in [('prediction_details', 'Prediction_detail_'), ('prepared_pairs', 'Prepared_')]:
            extra = self.read(self.config.get(source_key), source_key)
            if extra is None:
                continue
            extra = self.identify(extra, source_key)
            extra = extra.rename(columns={c: prefix + c for c in extra if c != 'pair_id'})
            master = self.join(master, extra, source_key)
            for c in extra:
                if c != 'pair_id':
                    self.define(c, self.config[source_key], c[len(prefix):], note='separate supplied metadata/descriptor; never overrides selected quantities')
        old = self.read(self.config.get('historical_predictions'), 'recovered historical predictions')
        if old is not None:
            old = self.identify(old, 'recovered historical predictions')
            h = old[['pair_id', 'Old_same_construct_final', 'Old_same_construct_base']].copy()
            h['Historical_prediction_source'] = str(path(self.config['historical_predictions']))
            h['historical_sequence_status'] = 'unverified'
            registered = master.set_index('pair_id')
            for ix, row in old.iterrows():
                if row.pair_id in registered.index:
                    match = all(pd.notna(registered.loc[row.pair_id].get(f'{r}_sequence')) and registered.loc[row.pair_id].get(f'{r}_sequence') == row.get(f'{r}_sequence') for r in ['AHD', 'RGG'])
                    h.loc[ix, 'historical_sequence_status'] = 'exact sequences match' if match else 'sequence conflict or missing'
            master = self.join(master, h, 'historical predictions')
        hf = self.read(self.config.get('historical_features'), 'historical features')
        if hf is not None:
            hf = self.identify(hf, 'historical features')
            hf = hf.rename(columns={c: 'Historical_feature_' + c for c in hf if c != 'pair_id'})
            master = self.join(master, hf, 'historical features')
            for c in hf:
                if c != 'pair_id':
                    self.define(c, self.config['historical_features'], c.removeprefix('Historical_feature_'), note='historical definition; distinct from current model inputs')
        archive = self.read(self.config.get('archived_panel'), 'archived panel predictions')
        if archive is not None:
            archive = self.identify(archive, 'archived panel predictions')
            if 'Pred_log10_negB22_hybrid' in archive:
                master = self.join(master, archive[['pair_id', 'Pred_log10_negB22_hybrid']].rename(columns={'Pred_log10_negB22_hybrid': 'Archived_panel_prediction'}), 'archived panel predictions')
        for spec in self.config.get('additional_features', []):
            extra = self.read(spec['path'], spec['prefix'])
            if extra is None:
                continue
            # Previously exported explicit orientation mappings; retain unresolved
            # records separately rather than inventing a canonical construct ID.
            unresolved = extra[extra.pair_id.isna()].copy()
            csv_write(unresolved, self.out / (spec['prefix'] + 'unresolved.csv'))
            extra = extra[extra.pair_id.notna()].copy()
            extra = self.identify(extra, spec['prefix'])
            extra = extra.rename(columns={c: spec['prefix'] + c for c in extra if c != 'pair_id'})
            master = self.join(master, extra, spec['prefix'])
            for c in extra:
                if c != 'pair_id':
                    self.define(c, spec['path'], c.removeprefix(spec['prefix']), note=spec['matching'])
        for col in ESSENTIAL + ['Sim_B22_final', 'Sim_B22_SD', 'Sim_B22_SEM', 'Sim_n_replicas', 'feature_source', 'prediction_source', 'simulation_source', 'Old_same_construct_final', 'historical_sequence_status']:
            if col not in master:
                master[col] = np.nan if col in ['Kd_uM', 'Kd_error_uM', 'Pred_log10_negB22_final', 'Sim_log10_negB22_final', 'Sim_B22_final', 'Sim_B22_SD', 'Sim_B22_SEM', 'Sim_n_replicas', 'Old_same_construct_final'] else pd.NA
        for col in ['Kd_uM', 'Kd_error_uM', 'Pred_log10_negB22_final', 'Sim_log10_negB22_final',
                    'Sim_B22_final', 'Sim_B22_SD', 'Sim_B22_SEM', 'Sim_n_replicas', 'Old_same_construct_final', 'Archived_panel_prediction']:
            if col in master:
                master[col] = pd.to_numeric(master[col], errors='raise').astype(float)
        for role in ['AHD', 'RGG']:
            name = f'{role}_name'
            if name not in master:
                master[name] = pd.NA
            # Parsing exact machine IDs supplies display labels only, not sequence evidence.
            parts = master.pair_id.str.split('__')
            parsed = parts.map(lambda p: p[1 if role == 'AHD' else 0] if len(p) == 2 else pd.NA)
            master[name] = master[name].fillna(parsed)
            master[f'{role}_parent'] = master[name].astype('string').str.replace(r'_shuffle_\d+$', '', regex=True)
            seqcol = f'{role}_sequence'
            if seqcol not in master:
                master[seqcol] = pd.NA
            if f'{role}_length' not in master:
                master[f'{role}_length'] = master[seqcol].str.len()
        master['is_shuffle'] = master.AHD_name.astype('string').str.contains('_shuffle_', na=False) | master.RGG_name.astype('string').str.contains('_shuffle_', na=False)
        master['parent_pair_id'] = master.RGG_parent + '__' + master.AHD_parent
        master['pair_display_name'] = [pretty_ahd(a) + '–' + pretty_rgg(r) if pd.notna(a) and pd.notna(r) else pid for pid, a, r in zip(master.pair_id, master.AHD_name, master.RGG_name)]
        master['partner_display'] = master.RGG_name.map(pretty_rgg)
        master['Historical_experimental_prediction'] = master.get('Archived_panel_prediction', pd.Series(np.nan, index=master.index))
        master['Historical_experimental_prediction_source'] = np.where(master.Historical_experimental_prediction.notna(), 'archived panel prediction', 'unavailable')
        fallback = master.Historical_experimental_prediction.isna() & master.historical_sequence_status.eq('exact sequences match')
        master.loc[fallback, 'Historical_experimental_prediction'] = master.loc[fallback, 'Old_same_construct_final']
        master.loc[fallback & master.Historical_experimental_prediction.notna(), 'Historical_experimental_prediction_source'] = 'recovered same-construct old prediction; not an archived panel value'
        master['experimental_status'] = master.experimental_status.fillna('no experimental measurement supplied')
        for role in ['AHD', 'RGG']:
            seq = master[f'{role}_sequence']
            actual_lengths = seq.str.len()
            lengths = pd.to_numeric(master[f'{role}_length'], errors='coerce')
            if (seq.notna() & lengths.notna() & actual_lengths.ne(lengths)).any():
                raise ValueError(f'{role}: sequence/length conflict in selected sources')
            hashcol = role + '_sequence_sha256'
            actual_hashes = seq.map(lambda s: hashlib.sha256(s.encode()).hexdigest() if pd.notna(s) else pd.NA)
            if hashcol in master:
                if (seq.notna() & master[hashcol].notna() & master[hashcol].ne(actual_hashes)).any():
                    raise ValueError(f'{role}: sequence/hash conflict in selected sources')
                master[hashcol] = master[hashcol].fillna(actual_hashes)
            else:
                master[hashcol] = actual_hashes
            master[f'{role}_length'] = lengths.fillna(actual_lengths)
            self.define(role + '_sequence', self.config.get('sequence_registry') or self.config['predictions']['path'],
                        note='exact registry sequence, or explicitly supplied selected prediction sequence when registry identity absent')
        master['experimental_construct_status'] = 'exact pair ID; supplied sequences absent'
        for role in ['AHD', 'RGG']:
            col = f'experiment_{role}_sequence'
            if col in master:
                conflict = master[col].notna() & master[f'{role}_sequence'].notna() & master[col].ne(master[f'{role}_sequence'])
                master.loc[conflict, 'experimental_construct_status'] = 'sequence conflict'
        master['prediction_status'] = np.where(np.isfinite(master.Pred_log10_negB22_final), 'available', 'missing prediction')
        master['simulation_status'] = np.where(np.isfinite(master.Sim_B22_final), 'available', 'missing simulation')
        master.loc[master.Sim_B22_final.ge(0), 'simulation_status'] = 'available; nonnegative B22; log unavailable'
        master['feature_status'] = np.where(master.feature_source.notna(), 'available', 'missing current features')
        if features is not None:
            featurecols = [c for c in features if c not in ['pair_id', 'feature_source']]
            master.loc[master.feature_source.notna() & master[featurecols].isna().any(axis=1), 'feature_status'] = 'partially available'
        master['pKd'] = np.nan
        positive = master.Kd_uM.gt(0) & np.isfinite(master.Kd_uM)
        master.loc[positive, 'pKd'] = 6 - np.log10(master.loc[positive, 'Kd_uM'])
        master['pKd_error'] = master.Kd_error_uM / (master.Kd_uM * np.log(10))
        master.loc[~positive | master.Kd_error_uM.lt(0), 'pKd_error'] = np.nan
        self.define('pair_id', 'union of selected sources', 'configured pair ID columns',
                    'exact oriented ID or exported verified alias; never approximate matching')
        self.define('pair_display_name', self.config.get('sequence_registry') or self.config['predictions']['path'],
                    'AHD_name, RGG_name', 'readable identity-derived names; N/C and shuffle distinctions retained')
        self.define('pKd', self.config['experiments']['path'], self.config['experiments']['columns'].get('Kd'), '6-log10(Kd_uM), positive selected measurements only')
        self.define('pKd_error', self.config['experiments']['path'], self.config['experiments']['columns'].get('Kd_error'),
                    'reported Kd_error_uM/(Kd_uM*ln(10)); no SD/SEM reinterpretation')
        for c in ['Old_same_construct_final', 'Old_same_construct_base']:
            self.define(c, self.config.get('historical_predictions'), note='existing recovered inference; not recomputed; sequence matching status separate')
        self.define('Archived_panel_prediction', self.config.get('archived_panel'), 'Pred_log10_negB22_hybrid', note='archived panel only; not same-construct inference')
        master['simulation_accuracy_eligible'] = False
        master['simulation_eligibility_reason'] = 'comparison provenance unavailable'
        elig = self.read(ss.get('eligibility_path'), 'simulation eligibility')
        ref = self.read(ss.get('eligibility_reference_path'), 'eligibility quantity/sequence reference')
        if elig is not None and ref is not None:
            unique(elig, 'simulation eligibility')
            unique(ref, 'eligibility reference')
            ei, ri = elig.set_index('pair_id'), ref.set_index('pair_id')
            for ix, row in master.iterrows():
                key = row.pair_id
                if key not in ei.index:
                    continue
                er = ei.loc[key]
                reason = er.get('exclusion_reason')
                if not boolean(pd.Series([er.paired_simulation_accuracy_eligible])).iloc[0]:
                    master.loc[ix, 'simulation_eligibility_reason'] = reason if pd.notna(reason) else 'existing audit excludes pair'
                    continue
                if key not in ri.index:
                    continue
                rr = ri.loc[key]
                value_matches = np.isclose(row.Sim_log10_negB22_final, rr.log10_neg_mean_B22, rtol=0, atol=1e-10)
                seq_matches = all(pd.notna(row.get(f'{r}_sequence')) and row.get(f'{r}_sequence') == rr.get(f'{r}_sequence') for r in ['AHD', 'RGG'])
                ok = value_matches and seq_matches and row.prediction_sequence_status == 'exact sequences match'
                master.loc[ix, 'simulation_accuracy_eligible'] = ok
                master.loc[ix, 'simulation_eligibility_reason'] = 'existing per-pair audit; quantity and exact sequence binding passed' if ok else 'audit does not bind selected quantity/sequences; comparison excluded'
        master['missing_data_reason'] = master.apply(lambda r: '; '.join([
            text for condition, text in [(r.experimental_status == 'no experimental measurement supplied', 'no experiment supplied'),
                (r.prediction_status == 'missing prediction', 'missing prediction'), (r.simulation_status == 'missing simulation', 'missing simulation'),
                (r.feature_status == 'missing current features', 'missing current features'),
                (pd.notna(r.Sim_B22_final) and r.Sim_B22_final >= 0, 'nonnegative simulation B22: logarithm undefined'),
                ('selection required' in str(r.experimental_status), 'experimental selection required')]
            if condition]), axis=1)
        unique(master, 'master union')
        master = master[ESSENTIAL + [c for c in master if c not in ESSENTIAL]].sort_values('pair_id').reset_index(drop=True)
        for c in master:
            if c not in self.columns:
                self.define(c, 'selected source tables / derived metadata', transform='derived or directly retained', note='see source_selection.json and source-specific status columns')
        csv_write(master, self.out / 'validation_master.csv')
        csv_write(pd.DataFrame(self.columns.values()), self.out / 'column_dictionary.csv')
        csv_write(pd.DataFrame(self.mapping).drop_duplicates(), self.out / 'pair_matching.csv')
        known = set()
        for source in [pred, registry, features, sim]:
            if source is not None:
                known.update(source.pair_id.dropna())
        csv_write(master.loc[~master.pair_id.isin(known), ['pair_id', 'pair_display_name']].assign(reason='No exact identity in selected current computational sources; retained without approximate matching'), self.out / 'unmatched_pair_identities.csv')
        csv_write(master[['pair_id', 'experimental_status', 'prediction_status', 'simulation_status', 'feature_status', 'missing_data_reason', 'simulation_accuracy_eligible', 'simulation_eligibility_reason']], self.out / 'missing_data_report.csv')
        csv_write(master[['pair_id', 'pair_display_name']].assign(verified=True, evidence='generated display name from exact machine ID; no approximate matching'), self.out / 'display_name_mapping.csv')
        (self.out / 'source_selection.json').write_text(json.dumps(dict(sources=self.sources, resolved_config=self.config, missing_sources=self.missing, model_input_columns=model_inputs), indent=2) + '\n')
        (self.out / 'resolved_validation_config.json').write_text(json.dumps(self.config, indent=2) + '\n')
        summary = dict(pairs=len(master), experiments=int(master.experimental_status.ne('no experimental measurement supplied').sum()),
            predictions=int(np.isfinite(master.Pred_log10_negB22_final).sum()), simulations=int(np.isfinite(master.Sim_B22_final).sum()),
            features=int(master.feature_source.notna().sum()), accuracy_eligible=int(master.simulation_accuracy_eligible.sum()), missing_sources=self.missing)
        (self.out / 'master_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return master
