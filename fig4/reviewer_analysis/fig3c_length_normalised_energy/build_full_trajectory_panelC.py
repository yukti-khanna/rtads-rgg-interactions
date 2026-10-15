#!/usr/bin/env python3
"""Refresh energy caches from full DCDs, then generate proposed Panel C options.

Requires explicit historical parameter selection. Never overwrites old caches.
No simulation or B22 calculation is performed. All trajectory frames are used.
"""
import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import MDAnalysis as mda

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def load_universe(topology, dcd):
    if topology.suffix == '.pdb':
        return mda.Universe(str(topology), str(dcd))
    # The existing cache records the exact atom/residue ordering used for energies.
    with topology.open('rb') as stream:
        saved = pickle.load(stream)
    a, b = np.asarray(saved['res1']), np.asarray(saved['res2'])
    expected = np.array([(i, j) for i in range(1, len(a)+1) for j in range(1, len(b)+1)])
    if not np.array_equal(saved['pairs'], expected):
        raise ValueError(f'Cannot establish atom ordering from {topology}')
    n = len(a)+len(b)
    u = mda.Universe.empty(n, n_residues=n, n_segments=2,
                          atom_resindex=np.arange(n),
                          residue_segindex=np.r_[np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)])
    u.add_TopologyAttr('resnames', np.r_[a, b])
    u.add_TopologyAttr('resids', np.r_[np.arange(1, len(a)+1), np.arange(1, len(b)+1)])
    u.add_TopologyAttr('segids', ['A', 'B'])
    u.load_new(str(dcd))
    return u

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', required=True, type=Path,
                        help='Confirmed historical residues.csv; do not guess.')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate inputs without calculating energies or plotting.')
    args = parser.parse_args()
    params_path = args.params.resolve(strict=True)
    param_hash = hashlib.sha256(params_path.read_bytes()).hexdigest()
    out = HERE / ('full_trajectory_' + param_hash[:12])
    cache = out / 'energy_cache'
    sys.path.insert(0, '/Users/yuktikhanna/calvados/scripts')
    from per_pair_energies import calc_per_pair_average_energy
    params = pd.read_csv(params_path).set_index('three')
    if not params.index.is_unique or not {'sigmas', 'lambdas', 'q'}.issubset(params.columns):
        raise ValueError('Parameter table needs unique residue names and sigmas/lambdas/q columns.')
    if not np.isfinite(params[['sigmas', 'lambdas', 'q']].to_numpy(float)).all():
        raise ValueError('Nonfinite residue parameters.')
    names = pd.read_csv(ROOT/'task5_panelC_enrichment/orientation_audit_final.csv').pair_id.tolist()
    assert len(names) == len(set(names)) == 917
    base = Path('/Users/yuktikhanna/calvados')
    sources = []
    # Validate the complete input inventory before any energy calculations.
    for name in names:
        dcd = base/'combined_dcds'/f'combined_trajectory_{name}_20nm_sliced.dcd'
        pdb = base/'test_pdbs'/f'{name}.pdb'
        if not pdb.is_file():
            pdb = base/'test_pdbs'/f'{name}_20.0nm_job_0.pdb'
        if not pdb.is_file():
            pdb = base/'test_energies'/f'energies_{name}_20nm_sliced.pkl'
        if not dcd.is_file() or not pdb.is_file():
            raise FileNotFoundError(f'Missing trajectory or topology: {dcd}, {pdb}')
        sources.append((name, pdb, dcd))
    if args.check_only:
        frame_counts = {}
        for name, pdb, dcd in sources:
            u = load_universe(pdb, dcd)
            if len(u.trajectory) < 100:
                raise ValueError(f'Unexpected short trajectory: {name}')
            a, b = u.select_atoms('segid A'), u.select_atoms('segid B')
            if not len(a) or not len(b) or len(a)+len(b) != len(u.atoms):
                raise ValueError(f'Invalid chain selections: {name}')
            unknown = (set(a.resnames) | set(b.resnames)) - set(params.index)
            if unknown:
                raise ValueError(f'Missing parameters for {name}: {unknown}')
            frame_counts[len(u.trajectory)] = frame_counts.get(len(u.trajectory), 0)+1
            u.trajectory.close()
        print(f'CHECK PASSED: 917 trajectories/topologies; parameters: {params_path}')
        print(f'Frame counts: {sorted(frame_counts.items())}\nPlanned output: {out}')
        print('No energies calculated; no figures saved. Historical parameter choice still requires confirmation.')
        return
    cache.mkdir(parents=True, exist_ok=True)
    audit = []
    print(f'Parameters: {params_path}\nSHA256: {param_hash}\nOutput: {out}', flush=True)
    for number, (name, pdb, dcd) in enumerate(sources, 1):
        destination = cache/f'energies_{name}_20nm_sliced.pkl'
        signature = dict(parameter_sha256=param_hash, trajectory=str(dcd),
                         trajectory_size=dcd.stat().st_size,
                         trajectory_mtime_ns=dcd.stat().st_mtime_ns,
                         topology_sha256=hashlib.sha256(pdb.read_bytes()).hexdigest())
        reuse = False
        if destination.exists():
            with destination.open('rb') as stream:
                saved = pickle.load(stream)
            reuse = saved.get('provenance') == signature
        if not reuse:
            u = load_universe(pdb, dcd)
            if len(u.trajectory) < 100:
                raise ValueError(f'Unexpected short trajectory: {name}')
            ah, yu, pairs, res1, res2 = calc_per_pair_average_energy(
                u, params, 0.8368, 298, 0.15, rc_lj=2.0, rc_yu=4.0)
            if not np.isfinite(ah).all() or not np.isfinite(yu).all():
                raise ValueError(f'Nonfinite energies: {name}')
            saved = dict(ah_energies_pp=ah, yu_energies_pp=yu, pairs=pairs,
                         res1=res1, res2=res2, provenance=signature,
                         frames=len(u.trajectory))
            temporary = destination.with_suffix('.pkl.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(saved, stream)
            temporary.replace(destination)
            u.trajectory.close()
        total = np.maximum(-(saved['ah_energies_pp'] + saved['yu_energies_pp']), 0).sum()
        audit.append(dict(pair_id=name, frames=saved['frames'],
                          total_attractive_energy=float(total), **signature))
        print(f'{number}/917 {name}: {saved["frames"]} frames, attractive total={total:.9g}'
              + (' [reused]' if reuse else ''), flush=True)
    pd.DataFrame(audit).to_csv(out/'full_trajectory_energy_audit.csv', index=False)
    (out/'energy_definition.json').write_text(json.dumps(dict(
        parameters=str(params_path), parameter_sha256=param_hash,
        epsilon_kj_mol=0.8368, temperature_k=298, ionic_molar=0.15,
        AH_cutoff_nm=2.0, Yukawa_cutoff_nm=4.0,
        frames='all stored frames; no additional slicing',
        method='existing calc_per_pair_average_energy; clip after averaging AH+Yukawa',
        old_caches_unchanged=True, B22_recomputed=False), indent=2))
    env = dict(os.environ, FIG3C_OUTPUT_DIR=str(out), FIG3C_ENERGY_DIR=str(cache))
    for script in ['build_fig3c_length_normalised_energy.py', 'build_panelC_options.py']:
        subprocess.run([sys.executable, str(HERE/script)], env=env, check=True)

if __name__ == '__main__':
    main()
