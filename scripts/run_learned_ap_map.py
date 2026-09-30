"""按整栋建筑留出训练 AP 可靠度，统一比较旧图、SCM、SCM-T 与双通道地图。"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.uji import load_floors
from models.learned_ap_map import APReliability, build, build_scm_t, cells
from scripts.evaluate_adjacent_map import episode
from scripts.evaluate_scm_tobit import PAIRS, tobit_match
from scripts.evaluate_signal_calibrated_map import FLOOR, build_map, lower_map, wknn


def metrics(predicted, truth):
    errors = np.linalg.norm(predicted - truth, axis=1)
    _, inv = np.unique(truth, axis=0, return_inverse=True)
    inv = inv.ravel()
    return {'scan_mde': float(errors.mean()),
            'position_mde': float((np.bincount(inv, errors) / np.bincount(inv)).mean())}


def make_episode(train, maps, old, lo, up, support, query, cfg):
    floor = train[up]
    assert len(support) == cfg['support_positions'] * cfg['scans_per_position']
    assert len(np.unique(floor['xy'][support], axis=0)) == cfg['support_positions']
    assert set(map(tuple, floor['xy'][support])).isdisjoint(map(tuple, floor['xy'][query]))
    if cfg.get('observation_model', 'two_channel') == 'scm_t':
        radio_map = build_scm_t(maps[lo], old[lo], floor['rssi'][support], floor['xy'][support], cfg, cfg['device'])
    else:
        radio_map = build(maps[lo], floor['rssi'][support], floor['xy'][support], cfg, cfg['device'])
    return {'lower': lo, 'upper': up, 'support': support, 'query': query, 'map': radio_map,
            'rssi': floor['rssi'][query], 'truth': floor['xy'][query],
            'y': torch.as_tensor(floor['xy'][query] - radio_map.origin, device=cfg['device'], dtype=torch.float32),
            'old': old[lo]}


def pools(train, maps, old, cfg, repeats, seed, cap=None):
    result = {}
    for lo, up in PAIRS:
        rng = np.random.default_rng(seed)
        eps = []
        for _ in range(repeats):
            support, query = episode(train[up], rng, cfg['support_positions'], cfg['scans_per_position'])
            if cap is not None:
                query = rng.choice(query, min(len(query), cap), replace=False)
            eps.append(make_episode(train, maps, old, lo, up, support, query, cfg))
        result[up] = eps
    return result


def manifest(pool, train):
    return {name: [{'lower': ep['lower'], 'support': train[name]['row_ids'][ep['support']].tolist(),
                    'query': train[name]['row_ids'][ep['query']].tolist()} for ep in eps]
            for name, eps in pool.items()}


def building_mean(rows, names):
    return float(np.mean([np.mean([rows[name] for name in names if name[:2] == b])
                          for b in sorted({name[:2] for name in names})]))


def train_model(pool, development, names, cfg, seed, directory):
    """固定预算训练；只在拟合建筑的新 episode 上记录曲线，不选最好 checkpoint。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 10000)
    model = APReliability(cfg).to(cfg['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    buildings = sorted({name[:2] for name in names})
    curve = []
    for step in range(cfg['steps'] + 1):
        if step in (0, 100, 300, cfg['steps']):
            dev = {name: float(np.mean([metrics(model.predict(ep['rssi'], ep['map']), ep['truth'])['scan_mde']
                                        for ep in development[name]])) for name in names}
            row = {'step': step, 'source_development_mde': building_mean(dev, names), 'floors': dev}
            curve.append(row)
            print(directory.name, seed, 'step', step, 'source-dev', round(row['source_development_mde'], 3), flush=True)
        if step == cfg['steps']:
            break
        b = rng.choice(buildings)
        name = rng.choice([name for name in names if name[:2] == b])
        ep = pool[name][rng.integers(len(pool[name]))]
        optimizer.zero_grad()
        prediction = model(ep['rssi'], ep['map'])
        loss = torch.sqrt((prediction - ep['y']).square().sum(1) + 1e-6).mean()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    directory.mkdir(parents=True)
    torch.save({'state_dict': model.state_dict(), 'fit_floors': names, 'seed': seed, 'steps': cfg['steps']},
               directory / f'seed_{seed}.pt')
    (directory / f'curve_{seed}.json').write_text(json.dumps(curve, indent=2) + '\n')
    return model


def evaluate_episode(ep, train, cfg, frozen, models, raw=None, truth=None):
    """同一锚点和 Query 上做配对比较；标签只用于这里的误差计算。"""
    raw = ep['rssi'] if raw is None else raw
    truth = ep['truth'] if truth is None else truth
    new = train[ep['upper']]
    s = ep['support']
    scm = build_map('scm', ep['old'], new['rssi'][s], new['xy'][s],
                    (cfg['length_scale_m'], cfg['noise_ratio']), cfg)
    fixed_name = 'scm_t_frozen' if cfg.get('observation_model') == 'scm_t' else 'two_channel_frozen'
    predictions = {'old_map': wknn(raw, *ep['old'], 1000, 10),
                   'scm': wknn(raw, *scm, 8, 10),
                   'scm_t': tobit_match(raw, *scm, 16, -80),
                   fixed_name: frozen.predict(raw, ep['map'])}
    result = {method: metrics(pred, truth) for method, pred in predictions.items()}
    result['learned'] = [metrics(model.predict(raw, ep['map']), truth) for model in models]
    return result


def reduce_episodes(rows):
    out = {method: {m: float(np.mean([row[method][m] for row in rows]))
                    for m in ('scan_mde', 'position_mde')}
           for method in rows[0] if method != 'learned'}
    out['learned'] = [{m: float(np.mean([row['learned'][seed][m] for row in rows]))
                       for m in ('scan_mde', 'position_mde')}
                      for seed in range(len(rows[0]['learned']))]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/learned_ap_map.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    torch.set_num_threads(4)
    started = time.time()
    train = load_floors(cfg['train_path'])
    names = sorted({name for pair in PAIRS for name in pair})
    maps = {name: cells(train[name]['rssi'], train[name]['xy'], unknown_strength=cfg['unknown_strength_dbm'])
            for name in names}
    old = {name: lower_map(train[name]) for name in names}
    print('preparing historical maps', flush=True)
    fit = pools(train, maps, old, cfg, cfg['training_episodes_per_pair'], cfg['training_seed'], cfg['training_query_cap'])
    dev = pools(train, maps, old, cfg, cfg['development_episodes_per_pair'], cfg['development_seed'], 128)
    evaluation = pools(train, maps, old, cfg, cfg['evaluation_episodes_per_pair'], cfg['evaluation_seed'])
    (args.output / 'manifest.json').write_text(json.dumps({key: manifest(value, train)
        for key, value in [('fit', fit), ('source_development', dev), ('evaluation', evaluation)]}, indent=1) + '\n')
    frozen = APReliability(cfg).to(cfg['device'])
    print('parameters', sum(p.numel() for p in frozen.parameters()), flush=True)
    # validation 此时才装载，所有模型拟合函数均没有此参数。
    validation = load_floors(cfg['validation_path'])
    report = {'held_buildings': {}, 'targets': {}, 'config': cfg}
    for b in ('B0', 'B1', 'B2'):
        fit_names = [name for name in fit if name[:2] != b]
        assert all(name[:2] != b for name in fit_names)
        models = [train_model(fit, dev, fit_names, cfg, seed, args.output / f'held_{b}_seed_{seed}')
                  for seed in cfg['model_seeds']]
        block = {'fit_floors': fit_names, 'floors': {}}
        for name in evaluation:
            if name[:2] != b:
                continue
            eps = evaluation[name]
            same_day = [evaluate_episode(ep, train, cfg, frozen, models) for ep in eps]
            later = [evaluate_episode(ep, train, cfg, frozen, models, validation[name]['rssi'], validation[name]['xy']) for ep in eps]
            block['floors'][name] = {'same_day': reduce_episodes(same_day), 'official': reduce_episodes(later),
                                    'episodes_same_day': same_day, 'episodes_official': later,
                                    'official_rows': validation[name]['row_ids'].tolist()}
            print('HELD', b, name, json.dumps(block['floors'][name]['official']), flush=True)
        report['held_buildings'][b] = block
        (args.output / 'partial.json').write_text(json.dumps(report, indent=1) + '\n')
    models = [train_model(fit, dev, list(fit), cfg, seed, args.output / f'all_source_seed_{seed}')
              for seed in cfg['model_seeds']]
    for name in cfg['targets']:
        lo, up = FLOOR[name]
        ids = {int(row): index for index, row in enumerate(train[name]['row_ids'])}
        target_eps, target_manifest = [], []
        for seed in (0, 1, 2):
            man = json.loads((Path(cfg['manifest_root']) / f'seed_{seed}' / 'manifest.json').read_text())[name]
            s = np.array([ids[int(row)] for row in man['support']])
            q = np.array([ids[int(row)] for row in man['unseen_position_query']])
            target_eps.append(make_episode(train, maps, old, lo, up, s, q, cfg))
            target_manifest.append({'seed': seed, 'support': man['support'], 'query': man['unseen_position_query']})
        same_day = [evaluate_episode(ep, train, cfg, frozen, models) for ep in target_eps]
        later = [evaluate_episode(ep, train, cfg, frozen, models, validation[name]['rssi'], validation[name]['xy']) for ep in target_eps]
        report['targets'][name] = {'same_day': reduce_episodes(same_day), 'official': reduce_episodes(later),
                                 'episodes_same_day': same_day, 'episodes_official': later, 'manifest': target_manifest,
                                 'official_rows': validation[name]['row_ids'].tolist()}
        print('TARGET', name, json.dumps(report['targets'][name]['official']), flush=True)
    report['elapsed_seconds'] = time.time() - started
    (args.output / 'results.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    (args.output / 'completed.json').write_text(json.dumps({'models': 12, 'steps_each': cfg['steps'],
        'held_floors': 7, 'evaluation_episodes_each': cfg['evaluation_episodes_per_pair'], 'targets': 3,
        'elapsed_seconds': report['elapsed_seconds']}, indent=2) + '\n')
    print('COMPLETED', round(report['elapsed_seconds'], 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
