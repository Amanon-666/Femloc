"""Post-freeze verification and source-only fixed-probe diagnosis. No training."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.uji import load_floors
from models.cross_floor import prior_map
from models.radio_map import cells
from models.learned_map import adapt, match
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import build_map, mde
from scripts.run_learned_map import (array_prior, building, episode, error, load_models,
                                     save, sha, source_tasks)


@torch.no_grad()
def audit(output, device):
    cfg = json.loads((output / 'config.json').read_text())
    provenance = json.loads((output / 'provenance.json').read_text())
    frozen = json.loads((output / 'frozen.json').read_text())
    results = json.loads((output / 'results.json').read_text())
    assert all(sha(p) == h for p, h in provenance['source_sha256'].items())
    assert results['frozen_sha256'] == sha(output / 'frozen.json')
    assert frozen['config_sha256'] == sha(output / 'config.json')
    assert provenance['train_sha256'] == sha(cfg['train_path'])
    assert results['validation_sha256'] == sha(cfg['validation_path'])
    train, val = load_floors(cfg['train_path']), load_floors(cfg['validation_path'])
    checks, source_probe, baseline_diffs, row_count = [], {}, [], 0
    for held in range(3):
        rec = json.loads((output / f'fold_{held}.json').read_text())
        assert sha(output / f'fold_{held}.json') == frozen['folds'][str(held)]['spec_sha256']
        for key, pr in [('deploy', rec['deployment_prior']), *rec['training_priors'].items()]:
            names = pr['observation_floors']
            assert all(building(f) != held for f in names)
            assert all(a in names and b in names for a, b in pr['pairs'])
            if key != 'deploy':
                assert key not in names
                assert all(key not in pair for pair in pr['pairs'])
            checks.append({'held_building': held, 'prior': key, 'floors': names, 'pairs': pr['pairs']})
        field, params = rec['selection']['field'], rec['selection']['match']
        tasks = source_tasks(train, held, rec, device)
        for seed in cfg['seeds']:
            models = load_models(output, frozen['folds'][str(held)], seed, device)
            for t in tasks:
                values = {name: [] for name in ('tpm', 'random', 'supervised', 'meta')}
                rng = np.random.default_rng(7741)
                for ep in range(5):
                    s, q = t['pool'].draw(rng, 10)
                    q = np.random.default_rng(9041 + ep).choice(q, min(len(q), 128), replace=False)
                    support, qx, qy = episode(t, s, q)
                    for name in values:
                        mapped, _ = adapt(t['context'], support, field, models.get(name))
                        values[name].append(float(error(match(qx, mapped, *params), qy)))
                source_probe[f'held{held}_seed{seed}_{t["name"]}'] = {k: float(np.mean(v)) for k, v in values.items()}
        for lo, up in PAIRS:
            if building(up) != held:
                continue
            rows = json.loads((output / f'evaluation_{up}.json').read_text())
            assert len(rows) == cfg['evaluation_episodes'] * len(cfg['ks']) * len(cfg['seeds'])
            row_count += len(rows)
            f = train[up]
            id_map = {int(v): i for i, v in enumerate(f['row_ids'])}
            by_episode = {}
            for r in rows:
                s = np.array([id_map[i] for i in r['support_row_ids']])
                q = np.array([id_map[i] for i in r['query_row_ids']])
                assert len(s) == r['k'] * cfg['scans']
                keys, n = np.unique(f['xy'][s], axis=0, return_counts=True)
                assert len(keys) == r['k'] and np.all(n == cfg['scans'])
                assert not set(map(tuple, keys)) & set(map(tuple, f['xy'][q]))
                by_episode.setdefault(r['episode'], {})[(r['k'], r['seed'])] = r
                for value in r['methods'].values():
                    assert all(np.isfinite(v) for v in value.values())
                    if 'normal_residual' in value:
                        assert value['normal_residual'] < 1e-10
            for ep_rows in by_episode.values():
                reference = ep_rows[(max(cfg['ks']), cfg['seeds'][0])]
                for seed in cfg['seeds']:
                    previous = set()
                    for k in sorted(cfg['ks']):
                        r = ep_rows[(k, seed)]
                        assert r['query_row_ids'] == reference['query_row_ids']
                        assert r['support_row_ids'] == reference['support_row_ids'][:k * cfg['scans']]
                        assert previous <= set(r['support_row_ids'])
                        previous = set(r['support_row_ids'])

            # Independently replay first K10 episode with the original NumPy/SciPy code.
            r = by_episode[0][(10, cfg['seeds'][0])]
            s = np.array([id_map[i] for i in r['support_row_ids']])
            q = np.array([id_map[i] for i in r['query_row_ids']])
            old = cells(train[lo])
            base = prior_map(old, array_prior(rec['deployment_prior']['parameters']))
            mapped = build_map('scm', base, f['rssi'][s], f['xy'][s], field, {'anchor_lookup_neighbors': 3})
            for metric, qx, qy in [('query_mde', f['rssi'][q], f['xy'][q]),
                                  ('validation_mde', val[up]['rssi'], val[up]['xy'])]:
                replay = mde(tobit_match(qx, *mapped, *params), qy)
                delta = abs(replay - r['methods']['tpm'][metric])
                assert delta < 1e-7, (up, metric, delta)
                baseline_diffs.append({'floor': up, 'metric': metric, 'absolute_difference_m': delta})
    report = {'verified_rows': row_count, 'prior_permissions': checks,
              'baseline_replay': baseline_diffs,
              'baseline_max_difference_m': max(v['absolute_difference_m'] for v in baseline_diffs),
              'source_probe_fixed_k10': source_probe,
              'source_probe_equal': {m: float(np.mean([v[m] for v in source_probe.values()]))
                                     for m in next(iter(source_probe.values()))},
              'note': 'Source probe is post-freeze diagnostic only; no checkpoint or hyperparameter selection.'}
    save(output / 'audit.json', report)
    print(json.dumps({k: report[k] for k in ('verified_rows', 'baseline_max_difference_m', 'source_probe_equal')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(2)
    audit(args.output, args.device)
