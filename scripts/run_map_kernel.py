"""读取固定划分，按建筑留出训练地图核，并完成配对评价。"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from data.uji import load_floors
from models.learned_ap_map import cells
from models.map_kernel import LearnedMapKernel, prepare
from scripts.evaluate_scm_tobit import PAIRS
from scripts.evaluate_signal_calibrated_map import FLOOR, lower_map
from scripts.run_learned_ap_map import building_mean, evaluate_episode, metrics, reduce_episodes


def make_episode(train, statistics, old, lo, up, support, query, cfg):
    """把已保存 row ids 接到地图构建和 source 损失所需的张量。"""
    floor = train[up]
    positions, counts = np.unique(floor['xy'][support], axis=0, return_counts=True)
    assert len(positions) == cfg['support_positions']
    assert np.all(counts == cfg['scans_per_position'])
    assert len(np.unique(floor['groups'][support])) == len(support)
    assert set(map(tuple, positions)).isdisjoint(map(tuple, floor['xy'][query]))
    radio_map = prepare(statistics[lo], old[lo], floor['rssi'][support], floor['xy'][support], cfg)
    return {'lower': lo, 'upper': up, 'support': support, 'query': query, 'map': radio_map,
            'rssi': floor['rssi'][query], 'truth': floor['xy'][query], 'old': old[lo],
            'y': torch.as_tensor(floor['xy'][query] - radio_map.origin,
                                 dtype=torch.float64, device=cfg['device'])}


def read_pools(cfg, train, statistics, old, manifest):
    """读取上一轮划分，不在训练或评价函数中重新抽取样本。"""
    row_indices = {name: {int(row): i for i, row in enumerate(floor['row_ids'])}
                   for name, floor in train.items()}
    pools = {}
    for split in ('fit', 'source_development', 'evaluation'):
        pools[split] = {}
        for up, rows in manifest[split].items():
            assert up not in cfg['targets']
            pools[split][up] = [make_episode(train, statistics, old, row['lower'], up,
                np.array([row_indices[up][r] for r in row['support']]),
                np.array([row_indices[up][r] for r in row['query']]), cfg) for row in rows]
    return pools


def train_model(pool, development, names, cfg, seed, directory):
    """仅拟合共享核矩阵；source 曲线记录不负责选择步骤。"""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 10000)
    model = LearnedMapKernel(cfg).to(cfg['device'])
    initial = model.matrix.detach().clone()
    optimizer = torch.optim.Adam([model.matrix], lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    buildings = sorted({name[:2] for name in names})
    curve, draws, losses = [], [], []
    first_gradient = None
    directory.mkdir(parents=True)
    started = time.perf_counter()
    for step in range(cfg['steps'] + 1):
        if step in cfg['curve_steps']:
            dev = {name: float(np.mean([metrics(model.predict(ep['rssi'], ep['map']), ep['truth'])['scan_mde']
                                        for ep in development[name]])) for name in names}
            matrix = model.matrix.detach()
            row = {'step': step, 'source_development_mde': building_mean(dev, names), 'floors': dev,
                   'matrix': matrix.cpu().tolist(), 'metric': (matrix.T @ matrix).cpu().tolist(),
                   'matrix_change_norm': float((matrix - initial).norm()),
                   'metric_change_norm': float((matrix.T @ matrix - initial.T @ initial).norm()),
                   'recent_train_loss': None if not losses else float(np.mean(losses[-100:]))}
            curve.append(row)
            print(directory.name, 'step', step, 'source-dev', round(row['source_development_mde'], 3),
                  'metric-change', round(row['metric_change_norm'], 5), flush=True)
        if step == cfg['steps']:
            break
        building = rng.choice(buildings)
        name = rng.choice([name for name in names if name[:2] == building])
        index = int(rng.integers(len(pool[name])))
        draws.append([str(name), index])
        ep = pool[name][index]
        optimizer.zero_grad()
        prediction = model(ep['rssi'], ep['map'])
        loss = torch.linalg.vector_norm(prediction - ep['y'], dim=1).mean()
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(model.matrix.grad).all()
        if first_gradient is None:
            first_gradient = model.matrix.grad.detach().cpu().tolist()
        optimizer.step()
        losses.append(float(loss.detach()))
    seconds = time.perf_counter() - started
    torch.save({'state_dict': model.state_dict(), 'fit_floors': names, 'seed': seed, 'steps': cfg['steps']},
               directory / 'model.pt')
    (directory / 'curve.json').write_text(json.dumps(curve, indent=2, allow_nan=False) + '\n')
    (directory / 'draws.json').write_text(json.dumps(draws) + '\n')
    (directory / 'completed.json').write_text(json.dumps({'seed': seed, 'steps': len(losses),
        'fit_floors': names, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'first_gradient': first_gradient, 'last_loss': losses[-1], 'seconds': seconds,
        'matrix': model.matrix.detach().cpu().tolist()}, indent=2, allow_nan=False) + '\n')
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/map_kernel.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    manifest_text = Path(cfg['source_manifest']).read_text()
    manifest = json.loads(manifest_text)
    (args.output / 'manifest.json').write_text(manifest_text)
    torch.set_num_threads(4)
    started = time.perf_counter()
    train = load_floors(cfg['train_path'])
    historical = sorted({name for pair in PAIRS for name in pair})
    statistics = {name: cells(train[name]['rssi'], train[name]['xy']) for name in historical}
    old = {name: lower_map(train[name]) for name in historical}
    pools = read_pools(cfg, train, statistics, old, manifest)
    fit, dev, evaluation = (pools[split] for split in ('fit', 'source_development', 'evaluation'))
    for split, count in (('fit', cfg['training_episodes_per_pair']),
                         ('source_development', cfg['development_episodes_per_pair']),
                         ('evaluation', cfg['evaluation_episodes_per_pair'])):
        assert len(pools[split]) == 7 and all(len(rows) == count for rows in pools[split].values())
    frozen = LearnedMapKernel(cfg).to(cfg['device'])
    fitted = {}
    for building in ('B0', 'B1', 'B2'):
        names = [name for name in fit if name[:2] != building]
        assert all(name[:2] != building and name not in cfg['targets'] for name in names)
        fitted[building] = [train_model(fit, dev, names, cfg, seed,
            args.output / f'held_{building}_seed_{seed}') for seed in cfg['model_seeds']]
    fitted['all_source'] = [train_model(fit, dev, list(fit), cfg, seed,
        args.output / f'all_source_seed_{seed}') for seed in cfg['model_seeds']]

    # 所有权重固定后才加载外部评价数据。
    validation = load_floors(cfg['validation_path'])
    report = {'held_buildings': {}, 'targets': {}, 'config': cfg,
              'source_manifest_sha256': hashlib.sha256(manifest_text.encode()).hexdigest(),
              'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    for building in ('B0', 'B1', 'B2'):
        names = [name for name in fit if name[:2] != building]
        block = {'fit_floors': names, 'floors': {}}
        for name, eps in evaluation.items():
            if name[:2] != building:
                continue
            same_day = [evaluate_episode(ep, train, cfg, frozen, fitted[building]) for ep in eps]
            later = [evaluate_episode(ep, train, cfg, frozen, fitted[building],
                     validation[name]['rssi'], validation[name]['xy']) for ep in eps]
            block['floors'][name] = {'same_day': reduce_episodes(same_day), 'official': reduce_episodes(later),
                'episodes_same_day': same_day, 'episodes_official': later,
                'official_rows': validation[name]['row_ids'].tolist()}
            print('HELD', building, name, json.dumps(block['floors'][name]['official']), flush=True)
        report['held_buildings'][building] = block
        (args.output / 'partial.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    target_ids = json.loads(Path(cfg['target_manifest']).read_text())
    (args.output / 'target_manifest.json').write_text(Path(cfg['target_manifest']).read_text())
    for name in cfg['targets']:
        lo, up = FLOOR[name]
        ids = {int(row): i for i, row in enumerate(train[name]['row_ids'])}
        eps, target_manifest = [], []
        for man in target_ids[name]:
            seed = man['seed']
            support = np.array([ids[r] for r in man['support']])
            query = np.array([ids[r] for r in man['query']])
            eps.append(make_episode(train, statistics, old, lo, up, support, query, cfg))
            target_manifest.append({'seed': seed, 'support': man['support'], 'query': man['query']})
        same_day = [evaluate_episode(ep, train, cfg, frozen, fitted['all_source']) for ep in eps]
        later = [evaluate_episode(ep, train, cfg, frozen, fitted['all_source'],
                 validation[name]['rssi'], validation[name]['xy']) for ep in eps]
        timing = []
        for model in fitted['all_source']:
            for ep in eps:
                with torch.no_grad():
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    model.update(ep['map'])
                    torch.cuda.synchronize()
                    timing.append(1000 * (time.perf_counter() - tick))
        report['targets'][name] = {'same_day': reduce_episodes(same_day), 'official': reduce_episodes(later),
            'episodes_same_day': same_day, 'episodes_official': later, 'manifest': target_manifest,
            'official_rows': validation[name]['row_ids'].tolist(), 'kernel_solve_ms': timing}
        print('TARGET', name, json.dumps(report['targets'][name]['official']), flush=True)
    report['elapsed_seconds'] = time.perf_counter() - started
    (args.output / 'results.json').write_text(json.dumps(report, indent=1, allow_nan=False) + '\n')
    print('TRAIN_AND_EVALUATE_COMPLETED', round(report['elapsed_seconds'], 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
