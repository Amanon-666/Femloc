"""训练源楼层DKT并以30条目标扫描条件化；保存任务、权重与逐点预测。"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .data import coordinate_scale, episode, load
from .model import DKT


def tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def run(config, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    floors = load(config['data_path'])
    sources = sorted(set(floors) - set(config['targets']))
    scale = coordinate_scale(floors, sources)
    provenance = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      data_sha256=hashlib.sha256(Path(config['data_path']).read_bytes()).hexdigest(),
                      sources=sources, coordinate_scale_m=scale, torch=torch.__version__)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    k, r = config['support_positions'], config['scans_per_position']
    results = []
    for seed in config['seeds']:
        folder = out / f'seed_{seed}'
        folder.mkdir()
        torch.manual_seed(seed)
        models = {'DKT': DKT().double().to(config['device']),
                  'RBF-GP': DKT(False).double().to(config['device'])}
        optimizers = {name: torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
                      for name, model in models.items()}
        rng = np.random.default_rng(seed)
        started = time.monotonic()
        with (folder / 'train.jsonl').open('w') as log:
            for step in range(1, config['updates'] + 1):
                name = sources[int(rng.integers(len(sources)))]
                floor = floors[name]
                support, query = episode(floor, rng, k, r, config['train_query_positions'])
                idx = np.concatenate([support, query])
                origin = floor['xy'][support].mean(0)
                x = tensor(floor['x'][idx], config['device'])
                y = tensor((floor['xy'][idx] - origin) / scale, config['device'])
                row = dict(step=step, floor=name)
                for method, model in models.items():
                    optimizers[method].zero_grad()
                    loss = model.loss(x, y)
                    if not torch.isfinite(loss):
                        raise ValueError(f'Nonfinite marginal loss: {seed}/{step}/{method}')
                    loss.backward()
                    optimizers[method].step()
                    row[method] = loss.item()
                log.write(json.dumps(row) + '\n')
                if step % 500 == 0:
                    log.flush()
                    print(f'seed={seed} step={step} loss={row["DKT"]:.4f}', flush=True)
        for method, model in models.items():
            torch.save(model.state_dict(), folder / f'{method}.pt')
        # 目标任务的随机流独立于训练，两个算法使用完全相同的Support/Query。
        rng = np.random.default_rng(10000 + seed)
        manifests = []
        for target in config['targets']:
            floor = floors[target]
            for repeat in range(config['evaluation_episodes']):
                support, query = episode(floor, rng, k, r)
                origin = floor['xy'][support].mean(0)
                sx = tensor(floor['x'][support], config['device'])
                sy = tensor((floor['xy'][support] - origin) / scale, config['device'])
                qx = tensor(floor['x'][query], config['device'])
                manifests.append(dict(target=target, episode=repeat,
                                      support=floor['ids'][support].tolist(), query=floor['ids'][query].tolist()))
                for method, model in models.items():
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    start = time.monotonic()
                    mean, variance = model.predict(sx, sy, qx)
                    if torch.cuda.is_available(): torch.cuda.synchronize()
                    elapsed = time.monotonic() - start
                    prediction = mean.cpu().numpy() * scale + origin
                    variance = variance.cpu().numpy() * scale ** 2
                    assert np.isfinite(prediction).all() and np.isfinite(variance).all()
                    error = np.linalg.norm(prediction - floor['xy'][query], axis=1)
                    pos = floor['pos'][query]
                    position_error = np.mean([error[pos == j].mean() for j in np.unique(pos)])
                    row = dict(seed=seed, target=target, episode=repeat, method=method,
                               support_scans=len(support), support_positions=k, query_scans=len(query),
                               mde=float(error.mean()), position_mde=float(position_error),
                               rmse=float(np.sqrt(np.mean(error ** 2))), seconds=elapsed,
                               target_gradient_steps=0)
                    results.append(row)
                    np.savez_compressed(folder / f'{target}_{repeat}_{method}.npz',
                                        row_ids=floor['ids'][query], mean=prediction, variance=variance,
                                        truth=floor['xy'][query])
            selected = [x['mde'] for x in results if x['seed']==seed and x['target']==target and x['method']=='DKT']
            print(f'seed={seed} target={target} DKT_MDE={np.mean(selected):.3f}', flush=True)
        (folder / 'manifest.json').write_text(json.dumps(manifests))
        (folder / 'completed.json').write_text(json.dumps(dict(seed=seed, updates=config['updates'], seconds=time.monotonic()-started)))
    (out / 'results.json').write_text(json.dumps(results, indent=2))
    (out / 'completed.json').write_text(json.dumps(dict(status='complete', seeds=config['seeds'], results=len(results))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/dkt_uji.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), Path(args.output))
