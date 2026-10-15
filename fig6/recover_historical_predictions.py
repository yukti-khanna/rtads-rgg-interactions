#!/usr/bin/env python3
"""Recover frozen old model inference only after reproducing archived panels.

References are read-only. Keras Dense weights are read directly from the saved
archive, never fitted; unsupported graphs fail. Numerical equivalence to the
archived base AND final panel predictions is required before predicting 73 pairs.
"""
import ast
import io
import json
import pickle
import sys
import warnings
import zipfile
import hashlib
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from workflow_common import ROOT, OUT, DIAG, sha256, write_json

sys.dont_write_bytecode = True
REPO = Path('/Users/yuktikhanna/idr-interaction-predict')
SELECTED = ['Opposite_Charge_Number', 'Normalized_OCN', 'charge_TAD', 'Average_SHD',
            'lambda_RGG', 'Lambda_Ratio', 'lambda_TAD', 'TAD_length', 'Average_Interaction', 'SHD_TAD']


class DenseArchive:
    def __init__(self, archive):
        data = archive.getvalue()
        cache = DIAG / 'historical_model/cache'
        cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(data).hexdigest()
        archivefile, weightsfile = cache / (key + '.keras'), cache / (key + '.npz')
        archivefile.write_bytes(data)
        # h5py is available in calvados; modern NumPy pickle loading occurs in
        # idr_features. This bridge only reads numerical HDF5 arrays.
        if not weightsfile.exists():
            subprocess.run(['/Users/yuktikhanna/miniconda3/envs/calvados/bin/python3',
                str(ROOT / 'extract_frozen_dense.py'), str(archivefile), str(weightsfile)], check=True)
        weights = np.load(weightsfile, allow_pickle=False)
        self.config = json.loads(str(weights['config']))
        layers = self.config['config']['layers']
        self.layers = []
        previous = None
        dense_number = 0
        for layer in layers:
                cls, cfg = layer['class_name'], layer['config']
                if cls == 'InputLayer':
                    previous = cfg['name']
                    continue
                if cls != 'Dense':
                    raise ValueError(f'Unsupported frozen residual graph: {cls}')
                nodes = layer.get('inbound_nodes', [])
                source = nodes[0]['args'][0]['config']['keras_history'][0]
                if source != previous:
                    raise ValueError('Residual graph is not a simple sequential Dense chain')
                key = 'dense' + (f'_{dense_number}' if dense_number else '')
                kernel, bias = weights[key + '/0'], weights[key + '/1']
                self.layers.append((kernel, bias, cfg['activation']))
                previous = cfg['name']
                dense_number += 1

    def predict(self, x, verbose=0):
        y = np.asarray(x, np.float32)
        for kernel, bias, activation in self.layers:
            y = y @ kernel + bias
            if activation == 'relu':
                y = np.maximum(y, 0)
            elif activation == 'tanh':
                y = np.tanh(y)
            elif activation != 'linear':
                raise ValueError(f'Unsupported residual activation: {activation}')
        return y


class FunctionalProxy:
    _unpickle_model = staticmethod(lambda archive: DenseArchive(archive))


def load_frozen(path):
    """Process-local deserialization compatibility, no artifact modifications."""
    import joblib
    from joblib.numpy_pickle import NumpyUnpickler
    import numpy.random._pickle as rng_pickle
    # NumPy 2 moved _core and changed BitGenerator pickle state; retain actual
    # RNG state rather than substituting a dummy estimator or random weights.
    if int(np.__version__.split('.')[0]) < 2:
        import numpy.core as core
        import numpy.core.numeric as numeric
        import numpy.core.multiarray as multiarray
        sys.modules.setdefault('numpy._core', core)
        sys.modules.setdefault('numpy._core.numeric', numeric)
        sys.modules.setdefault('numpy._core.multiarray', multiarray)
        rng_pickle.BitGenerators[str(np.random.MT19937)] = np.random.MT19937
        rng_pickle.BitGenerators[np.random.MT19937] = np.random.MT19937
    original_find = NumpyUnpickler.find_class
    original_build = NumpyUnpickler.dispatch[pickle.BUILD[0]]
    original_ctor = rng_pickle.__randomstate_ctor
    def finder(self, module, name):
        if module.startswith('keras.') and name == 'Functional':
            return FunctionalProxy
        return original_find(self, module, name)
    def build(self):
        instance, state = self.stack[-2:]
        if isinstance(instance, np.random.BitGenerator) and isinstance(state, tuple) and isinstance(state[0], dict):
            self.stack.pop()
            instance.state = state[0]
            return
        original_build(self)
    NumpyUnpickler.find_class = finder
    NumpyUnpickler.dispatch = NumpyUnpickler.dispatch.copy()
    NumpyUnpickler.dispatch[pickle.BUILD[0]] = build
    rng_pickle.__randomstate_ctor = lambda arg='MT19937', *rest: np.random.RandomState(arg) if isinstance(arg, np.random.BitGenerator) else original_ctor(arg, *rest)
    try:
        return joblib.load(path)
    finally:
        NumpyUnpickler.find_class = original_find
        NumpyUnpickler.dispatch[pickle.BUILD[0]] = original_build
        rng_pickle.__randomstate_ctor = original_ctor


def inference(bundle, d, convention='repository_soft_gate'):
    cols = bundle.get('feature_cols', SELECTED)
    if len(cols) != bundle['scaler'].n_features_in_:
        raise ValueError(f'Feature order unavailable for {bundle["scaler"].n_features_in_}-feature scaler')
    x = bundle['scaler'].transform(d[cols].to_numpy(float))
    base = np.asarray(bundle['base_mlp'].predict(x), float)
    cap, lam = float(bundle['cap']), float(bundle['lambda_hat'])
    if lam == 0:
        return base, base
    residual = np.asarray(bundle['residual_nn'].predict(x, verbose=0), float).ravel()
    gate = 1 / (1 + np.exp(-(d.Opposite_Charge_Number.to_numpy() - 40) / 12))
    if convention == 'archived_ungated_correction':
        gate = np.ones(len(d))
    elif convention == 'historical_threshold_mask':
        gate = (d.Opposite_Charge_Number.to_numpy() >= float(bundle['threshold'])).astype(float)
    return base, base + lam * gate * np.clip(residual, -cap, cap)


def historical_features(inv):
    """Extract only pure historical functions; never execute reference workflows."""
    from scipy.integrate import quad
    ref = Path('/Users/yuktikhanna/calvados/fig6')
    residues = pd.read_csv(ref / 'residues.csv').set_index('one')
    namespace = dict(np=np, quad=quad, residue_properties=residues)
    names = {'lj_potential', 'ah_potential', 'ah_scaled', 'make_ah_intgrl_map', 'calc_ah_between_sequences',
             'calculate_property', 'get_qs_fast', 'calc_SCD', 'make_lambda_map', 'calc_SHD'}
    for path in [ref / 'energy_funcs.py', ref / 'get_features_all.py']:
        tree = ast.parse(path.read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in functions:
            node.decorator_list = []
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    ahmap = namespace['make_ah_intgrl_map'](residues)
    rows = []
    for _, row in inv.iterrows():
        a, r = row.AHD_sequence, row.RGG_sequence
        la = residues.loc[list(a), 'lambdas'].sum()
        lr = residues.loc[list(r), 'lambdas'].sum()
        qa = residues.loc[list(a), 'q'].sum()
        qr = residues.loc[list(r), 'q'].sum()
        shda, shdr = namespace['calc_SHD'](a), namespace['calc_SHD'](r)
        rows.append(dict(pair_id=row.pair_id, Opposite_Charge_Number=-qa * qr,
            Normalized_OCN=-qa * qr / (len(a) + len(r)), charge_TAD=qa,
            Average_SHD=(shda + shdr) / 2, lambda_RGG=lr,
            # Preserve the actual legacy feature CSV expression, including its
            # epsilon position; changing it would change historical inputs.
            Lambda_Ratio=la / lr + 1e-6, lambda_TAD=la, TAD_length=len(a),
            Average_Interaction=namespace['calc_ah_between_sequences'](r, a, ahmap), SHD_TAD=shda))
    return pd.DataFrame(rows), {str(ref / name): sha256(ref / name) for name in ['get_features_all.py', 'energy_funcs.py', 'residues.csv']}


def main():
    if int(np.__version__.split('.')[0]) < 2:
        worker = Path('/Users/yuktikhanna/miniconda3/envs/idr_features/bin/python')
        if worker.exists():
            subprocess.run([str(worker), str(Path(__file__).resolve())], check=True)
            return
        raise RuntimeError('Historical bundle recovery requires NumPy >=2 for safe loading; use a compatible inference environment')
    dest = DIAG / 'historical_model'
    dest.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(ROOT / 'Table_S8.csv').set_index('pair_id')
    feat = pd.read_csv(ROOT / 'validated_final_features_data_pred_b22s.csv').rename(columns={'name': 'pair_id'}).set_index('pair_id').loc[old.index]
    attempts, candidates, predictions = [], [], []
    for file in [*sorted((REPO / 'models_old').glob('*.pkl')), *sorted((REPO / 'models/pretrained').glob('*.pkl'))]:
        entry = dict(artifact=str(file), sha256=sha256(file), status='', reason='')
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                bundle = load_frozen(file)
            entry['load_warnings'] = '; '.join(str(w.message) for w in caught)
            if not isinstance(bundle, dict):
                entry.update(status='not_complete_pipeline', reason='Standalone estimator/scaler does not define final historical hybrid prediction')
                attempts.append(entry)
                continue
            base, final = inference(bundle, feat)
            convention_checks = []
            for convention in ['repository_soft_gate', 'archived_ungated_correction', 'historical_threshold_mask']:
                bb, ff = inference(bundle, feat, convention)
                convention_checks.append((float(abs(ff - old.Pred_log10_negB22_hybrid).max()), convention, bb, ff))
            _, convention, base, final = min(convention_checks, key=lambda z: z[0])
            maxbase = float(abs(base - old.Pred_log10_negB22_base).max())
            maxfinal = float(abs(final - old.Pred_log10_negB22_hybrid).max())
            entry.update(max_base_error=maxbase, max_final_error=maxfinal,
                         recovered_application_convention=convention,
                         tested_conventions=';'.join(f'{c}:{error:.9g}' for error,c,_,_ in convention_checks),
                         lambda_hat=float(bundle['lambda_hat']), cap=float(bundle['cap']),
                         feature_columns=';'.join(bundle.get('feature_cols', SELECTED)))
            # Same panel model requires both archived base and final values.
            passed = maxbase < 1e-6 and maxfinal < 1e-6
            entry['status'] = 'reproduces_old_panel' if passed else 'different_model_or_preprocessing'
            for pid, b, f in zip(old.index, base, final):
                predictions.append(dict(pair_id=pid, artifact=str(file), candidate_base=b, candidate_final=f,
                                        archived_base=old.loc[pid, 'Pred_log10_negB22_base'],
                                        archived_final=old.loc[pid, 'Pred_log10_negB22_hybrid'], passed=passed))
            if passed:
                candidates.append((file, bundle, convention))
        except Exception as exc:
            entry.update(status='unrecoverable', reason=f'{type(exc).__name__}: {exc}')
        attempts.append(entry)
    pd.DataFrame(attempts).to_csv(dest / 'model_recovery_attempts.csv', index=False)
    pd.DataFrame(predictions).to_csv(dest / 'panel_reproduction.csv', index=False)
    inv = pd.read_csv(OUT / 'complete_pair_inventory.csv')
    result = dict(total_new_pairs=len(inv), recovered_same_construct_old_pairs=0,
                  archived_old_panel_overlap=len(old), status='historical pipeline not verified',
                  required='Frozen old-panel final predictor and exact preprocessing reproducing both old base and final predictions')
    if candidates:
        file, bundle, convention = candidates[0]
        features, source_hashes = historical_features(inv)
        merged = features.set_index('pair_id').loc[feat.index]
        errors = []
        for col in SELECTED:
            errors.append(dict(feature=col, maximum_absolute_difference=float(abs(merged[col] - feat[col]).max()),
                               passed=bool(np.allclose(merged[col], feat[col], rtol=1e-9, atol=1e-8))))
        pd.DataFrame(errors).to_csv(dest / 'feature_reproduction.csv', index=False)
        features.to_csv(dest / 'same_construct_historical_features.csv', index=False)
        if all(e['passed'] for e in errors):
            base, final = inference(bundle, features, convention)
            same = inv.copy()
            same['Old_same_construct_base'] = base
            same['Old_same_construct_final'] = final
            same['Old_pipeline_artifact'] = str(file)
            same['old_prediction_provenance'] = 'frozen old pipeline; exact supplied new sequences; archived reproduction passed'
            same.to_csv(dest / 'same_construct_predictions.csv', index=False)
            result.update(recovered_same_construct_old_pairs=len(same), status='verified old pipeline and feature reproduction',
                          model_artifact=str(file), model_sha256=sha256(file), source_hashes=source_hashes)
            result['recovered_application_convention'] = convention
            result['convention_evidence'] = 'Exact archived base and final values reproduced on all 20 rows; no model weights, lambda or cap fitted. Public soft-gate application differs from archived panel convention.'
        else:
            result.update(status='panel model recovered but historical feature reconstruction failed',
                          required='Exact original feature construction matching historical CSV; see feature_reproduction.csv')
    write_json(dest / 'recovery_status.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
