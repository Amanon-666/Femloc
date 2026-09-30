"""Frozen-budget, building-held-out pilot for a learned map adaptation rule.

Prepare uses only source buildings. Train freezes all folds before evaluation
opens official validation or constructs held-building tasks. No unlabeled pool.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.uji import load_floors
from models.cross_floor import fit_transfer, prior_map
from models.radio_map import cells, fit_observation
from models.learned_map import MapContext, MapEncoder, adapt, match, prior_with_support, tensor
from scripts.evaluate_scm_tobit import PAIRS
from scripts.evaluate_signal_calibrated_map import dbm


def save(path, value):
    def native(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=native, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def building(name):
    return int(name[1:name.index('F')])


def source_names(held):
    return sorted({f for pair in PAIRS for f in pair if building(f) != held})


def array_prior(tf):
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in tf.items()}


def fit_source_prior(source_cells, held_building, excluded_upper=None):
    permitted = sorted(f for f in source_cells if f != excluded_upper)
    assert all(building(f) != held_building for f in permitted)
    pairs = [(a, b) for a, b in PAIRS if a in permitted and b in permitted]
    assert pairs
    obs = fit_observation([source_cells[f] for f in permitted])
    tf = fit_transfer([(source_cells[a], source_cells[b]) for a, b in pairs], obs)
    return {'parameters': tf, 'observation_floors': permitted, 'pairs': pairs,
            'held_building': held_building, 'excluded_upper': excluded_upper}


class Pool:
    def __init__(self, floor):
        self.floor = floor
        _, self.inverse = np.unique(floor['xy'], axis=0, return_inverse=True)
        self.inverse = self.inverse.ravel()
        self.rows = [np.flatnonzero(self.inverse == i) for i in range(self.inverse.max() + 1)]
        self.eligible = np.array([i for i, r in enumerate(self.rows) if len(r) >= 3])

    def draw(self, rng, k):
        chosen = rng.choice(self.eligible, k, replace=False)
        support = np.concatenate([rng.choice(self.rows[i], 3, replace=False) for i in chosen])
        query = np.flatnonzero(~np.isin(self.inverse, chosen))
        return support, query


def task(name, old_cells, tf, new, device):
    context = MapContext(old_cells, prior_map(old_cells, array_prior(tf))[0], device)
    return {'name': name, 'context': context, 'pool': Pool(new)}


def episode(t, s, q):
    f, c = t['pool'].floor, t['context']
    assert not set(map(tuple, f['xy'][s])) & set(map(tuple, f['xy'][q]))
    support = c.support(f['rssi'][s], f['xy'][s])
    return support, tensor(f['rssi'][q], c.device), tensor(f['xy'][q] - c.origin, c.device)


def error(pred, truth):
    return (pred - truth).square().sum(-1).sqrt().mean()


@torch.no_grad()
def select(tasks, cfg):
    scores = np.zeros((len(tasks), len(cfg['fields']), len(cfg['matches'])))
    for i, t in enumerate(tasks):
        rng = np.random.default_rng(41)
        for ep in range(cfg['selection_episodes']):
            s, q = t['pool'].draw(rng, 10)
            q = np.random.default_rng(4000 + ep).choice(q, min(len(q), cfg['selection_query_cap']), replace=False)
            support, qx, qy = episode(t, s, q)
            for fi, field in enumerate(cfg['fields']):
                mapped, _ = adapt(t['context'], support, field)
                for mi, params in enumerate(cfg['matches']):
                    scores[i, fi, mi] += float(error(match(qx, mapped, *params), qy)) / cfg['selection_episodes']
    equal = scores.mean(0)
    fi, mi = np.unravel_index(equal.argmin(), equal.shape)
    return {'field': cfg['fields'][fi], 'match': cfg['matches'][mi],
            'source_mde': float(equal[fi, mi]), 'source_floors': [t['name'] for t in tasks],
            'grid_per_floor': scores, 'grid_equal': equal}


def source_tasks(train, held, record, device):
    names = source_names(held)
    c = {f: cells(train[f]) for f in names}
    return [task(up, c[lo], record['training_priors'][up]['parameters'], train[up], device)
            for lo, up in PAIRS if lo in names and up in names]


def prepare(train, cfg, output, device):
    for held in range(3):
        names = source_names(held)
        c = {f: cells(train[f]) for f in names}
        rec = {'held_building': held, 'source_floors': names, 'training_priors': {}}
        for lo, up in PAIRS:
            if lo not in names or up not in names:
                continue
            print('PRIOR', held, 'source task', up, flush=True)
            rec['training_priors'][up] = fit_source_prior(c, held, up)
        print('PRIOR', held, 'deployment', flush=True)
        rec['deployment_prior'] = fit_source_prior(c, held)
        tasks = source_tasks(train, held, rec, device)
        rec['selection'] = select(tasks, cfg)
        save(output / f'fold_{held}.json', rec)
        print('SELECTED', held, rec['selection']['field'], rec['selection']['match'], rec['selection']['source_mde'], flush=True)


def train_fold(tasks, rec, cfg, output, held, seed, device):
    torch.manual_seed(seed)
    initial = MapEncoder().double().to(device)
    meta, supervised = copy.deepcopy(initial), copy.deepcopy(initial)
    torch.manual_seed(seed + 10000)
    head = nn.Linear(4, 1).double().to(device)
    om = torch.optim.Adam(meta.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    os = torch.optim.Adam(list(supervised.parameters()) + list(head.parameters()),
                          lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    rng = np.random.default_rng(1729 + seed)
    field, params = rec['selection']['field'], rec['selection']['match']
    curve, start = [], time.monotonic()
    for step in range(1, cfg['steps'] + 1):
        t = tasks[rng.integers(len(tasks))]
        k = int(rng.choice(cfg['ks']))
        s, q = t['pool'].draw(rng, k)
        q = rng.choice(q, min(len(q), cfg['train_query_cap']), replace=False)
        support, qx, qy = episode(t, s, q)
        om.zero_grad()
        mapped, _ = adapt(t['context'], support, field, meta)
        pred = match(qx, mapped, *params)
        meta_loss = ((pred - qy).square().sum(-1) + 1e-6).sqrt().mean() / 20.
        meta_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(meta.parameters(), cfg['grad_clip'])
        assert torch.isfinite(meta_loss) and torch.isfinite(grad)
        om.step()

        os.zero_grad()
        base_q, z_q = t['context'].at(t['pool'].floor['xy'][q])
        target = (tensor(dbm(t['pool'].floor['rssi'][q]), device) - base_q) / 20.
        residual_pred = head(supervised(z_q)).squeeze(-1)
        supervised_loss = (residual_pred - target).square().mean()
        supervised_loss.backward()
        gs = torch.nn.utils.clip_grad_norm_(list(supervised.parameters()) + list(head.parameters()), cfg['grad_clip'])
        assert torch.isfinite(supervised_loss) and torch.isfinite(gs)
        os.step()
        if step == 1 or step % 100 == 0 or step == cfg['steps']:
            row = {'step': step, 'source_task': t['name'], 'k': k,
                   'meta_query_mde': float(meta_loss.detach()) * 20,
                   'meta_gradient_norm': float(grad), 'supervised_mse': float(supervised_loss.detach()),
                   'elapsed_s': time.monotonic() - start}
            curve.append(row)
            print('TRAIN', held, seed, row, flush=True)
    checkpoint = output / f'model_B{held}_seed{seed}.pt'
    torch.save({'random': initial.cpu().state_dict(), 'meta': meta.cpu().state_dict(),
                'supervised': supervised.cpu().state_dict(), 'supervised_head': head.cpu().state_dict(),
                'seed': seed, 'held_building': held, 'steps': cfg['steps']}, checkpoint)
    save(output / f'training_B{held}_seed{seed}.json', curve)
    return {'path': checkpoint.name, 'sha256': sha(checkpoint), 'training_seconds': time.monotonic() - start}


def train_all(train, cfg, output, device):
    assert not (output / 'frozen.json').exists()
    frozen = {'config_sha256': sha(output / 'config.json'), 'folds': {}}
    for held in range(3):
        rec = json.loads((output / f'fold_{held}.json').read_text())
        tasks = source_tasks(train, held, rec, device)
        frozen['folds'][str(held)] = {'spec_sha256': sha(output / f'fold_{held}.json'), 'models': {}}
        for seed in cfg['seeds']:
            frozen['folds'][str(held)]['models'][str(seed)] = train_fold(tasks, rec, cfg, output, held, seed, device)
    save(output / 'frozen.json', frozen)
    print('ALL_FROZEN', sha(output / 'frozen.json'), flush=True)


def load_models(output, frozen_fold, seed, device):
    spec = frozen_fold['models'][str(seed)]
    assert sha(output / spec['path']) == spec['sha256']
    state = torch.load(output / spec['path'], map_location='cpu', weights_only=True)
    models = {}
    for method in ('random', 'supervised', 'meta'):
        encoder = MapEncoder().double().to(device)
        encoder.load_state_dict(state[method])
        models[method] = encoder.eval()
    return models


@torch.no_grad()
def evaluate(train, cfg, output, device):
    frozen = json.loads((output / 'frozen.json').read_text())
    assert frozen['config_sha256'] == sha(output / 'config.json')
    validation = load_floors(cfg['validation_path'])
    all_rows = []
    for held in range(3):
        assert frozen['folds'][str(held)]['spec_sha256'] == sha(output / f'fold_{held}.json')
        rec = json.loads((output / f'fold_{held}.json').read_text())
        field, params = rec['selection']['field'], rec['selection']['match']
        models = {seed: load_models(output, frozen['folds'][str(held)], seed, device) for seed in cfg['seeds']}
        for lo, up in PAIRS:
            if building(up) != held:
                continue
            old = cells(train[lo])
            t = task(up, old, rec['deployment_prior']['parameters'], train[up], device)
            c, f, pool = t['context'], train[up], t['pool']
            identity_context = MapContext(old, c.old_encoded, device)
            val_x = tensor(validation[up]['rssi'], device)
            val_y = tensor(validation[up]['xy'] - c.origin, device)
            rng = np.random.default_rng(1001)
            rows = []
            for ep in range(cfg['evaluation_episodes']):
                s10, q = pool.draw(rng, max(cfg['ks']))
                q = np.random.default_rng(2001 + ep).choice(q, min(len(q), cfg['query_cap']), replace=False)
                for k in cfg['ks']:
                    s = s10[:k * cfg['scans']]
                    support, qx, qy = episode(t, s, q)
                    for seed in cfg['seeds']:
                        row = {'building': held, 'floor': up, 'episode': ep, 'seed': seed, 'k': k,
                               'support_row_ids': f['row_ids'][s].tolist(), 'query_row_ids': f['row_ids'][q].tolist(),
                               'methods': {}}
                        meta_pred = None
                        for name in ('prior_support', 'tpm', 'random', 'supervised', 'meta'):
                            if str(device).startswith('cuda'):
                                torch.cuda.synchronize()
                            begin = time.monotonic()
                            if name == 'prior_support':
                                mapped, diagnostics = prior_with_support(c, support), {}
                            else:
                                mapped, diagnostics = adapt(c, support, field, models[seed].get(name), diagnostics=True)
                            if str(device).startswith('cuda'):
                                torch.cuda.synchronize()
                            elapsed = time.monotonic() - begin
                            pred = match(qx, mapped, *params)
                            val_pred = match(val_x, mapped, *params)
                            row['methods'][name] = {'query_mde': float(error(pred, qy)),
                                                    'validation_mde': float(error(val_pred, val_y)),
                                                    'adapt_seconds': elapsed, **diagnostics}
                            if name == 'meta':
                                meta_pred = pred
                        if k == 10:
                            keys, inv = np.unique(f['xy'][s], axis=0, return_inverse=True)
                            shuffled_xy = keys[np.random.default_rng(ep + 21).permutation(len(keys))][inv.ravel()]
                            shuffled = c.support(f['rssi'][s], shuffled_xy)
                            smap, sinfo = adapt(c, shuffled, field, models[seed]['meta'], diagnostics=True)
                            spred = match(qx, smap, *params)
                            id_support = identity_context.support(f['rssi'][s], f['xy'][s])
                            imap, _ = adapt(identity_context, id_support, field, models[seed]['meta'])
                            row['diagnostics'] = {
                                'shuffled_query_mde': float(error(spred, qy)),
                                'shuffled_prediction_move_m': float(error(spred, meta_pred)),
                                'shuffled_field_support_rmse': sinfo['field_support_rmse'],
                                'identity_prior_query_mde_no_retraining': float(error(match(qx, imap, *params), qy)),
                                'identity_prior_validation_mde_no_retraining': float(error(match(val_x, imap, *params), val_y))}
                        rows.append(row)
            save(output / f'evaluation_{up}.json', rows)
            all_rows.extend(rows)
            print('EVALUATED', up, summarize(rows), flush=True)
    report = {'frozen_sha256': sha(output / 'frozen.json'), 'validation_sha256': sha(cfg['validation_path']),
              'metrics': summarize(all_rows),
              'note': 'Building held out of this run fitting/selection, but historically inspected development data. No U.'}
    save(output / 'results.json', report)
    print('DONE', report, flush=True)


def summarize(rows):
    result = {}
    for k in sorted({r['k'] for r in rows}):
        result[str(k)] = {}
        for metric in ('query_mde', 'validation_mde'):
            floors = {}
            for f in sorted({r['floor'] for r in rows}):
                subset = [r for r in rows if r['floor'] == f and r['k'] == k]
                floors[f] = {name: float(np.mean([r['methods'][name][metric] for r in subset]))
                             for name in subset[0]['methods']}
            buildings = {b: [v for f, v in floors.items() if building(f) == b] for b in sorted({building(f) for f in floors})}
            result[str(k)][metric] = {
                'per_floor': floors,
                'floor_equal': {m: float(np.mean([v[m] for v in floors.values()])) for m in rows[0]['methods']},
                'building_equal': {m: float(np.mean([np.mean([v[m] for v in vs]) for vs in buildings.values()]))
                                   for m in rows[0]['methods']}}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/learned_map.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('all', 'prepare', 'train', 'evaluate'), default='all')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    cfg = json.loads(Path(args.config).read_text())
    if args.stage in ('all', 'prepare'):
        args.output.mkdir(parents=True, exist_ok=False)
        save(args.output / 'config.json', cfg)
        paths = ['models/learned_map.py', 'scripts/run_learned_map.py', 'models/cross_floor.py',
                 'models/radio_map.py', 'data/uji.py', 'scripts/evaluate_signal_calibrated_map.py',
                 'scripts/evaluate_scm_tobit.py', 'configs/learned_map.json', 'docs/research/LEARNED_MAP_DERIVATION.md']
        provenance = {'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      'source_sha256': {s: sha(s) for s in paths}, 'train_sha256': sha(cfg['train_path']),
                      'torch_version': torch.__version__, 'numpy_version': np.__version__, 'device': args.device,
                      'torch_dtype': 'float64', 'torch_threads': 2, 'deterministic_algorithms': True}
        save(args.output / 'provenance.json', provenance)
    else:
        assert cfg == json.loads((args.output / 'config.json').read_text())
        provenance = json.loads((args.output / 'provenance.json').read_text())
        assert all(sha(s) == h for s, h in provenance['source_sha256'].items())
        assert sha(cfg['train_path']) == provenance['train_sha256']
    train = load_floors(cfg['train_path'])
    if args.stage in ('all', 'prepare'):
        prepare(train, cfg, args.output, args.device)
    if args.stage in ('all', 'train'):
        train_all(train, cfg, args.output, args.device)
    if args.stage in ('all', 'evaluate'):
        evaluate(train, cfg, args.output, args.device)


if __name__ == '__main__':
    main()
