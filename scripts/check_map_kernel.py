"""检查作者类、几何初始化、梯度路径与 AP 置换不变量。"""
import argparse
import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from data.uji import load_floors
from models.learned_ap_map import cells
from models.map_kernel import LearnedMapKernel
from scripts.evaluate_scm_tobit import tobit_match
from scripts.evaluate_signal_calibrated_map import build_map, lower_map
from scripts.run_map_kernel import make_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/map_kernel.json')
    parser.add_argument('--output', type=Path, default=Path('outputs/map_kernel_check.json'))
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(4)
    class_ast = lambda path: ast.dump(next(node for node in ast.parse(Path(path).read_text()).body
                                        if isinstance(node, ast.ClassDef) and node.name == 'ExactGPLayer'))
    assert class_ast(cfg['upstream_snapshot']) == class_ast('models/dkt_reference.py')
    train = load_floors(cfg['train_path'])
    row = json.loads(Path(cfg['source_manifest']).read_text())['fit']['B0F1'][0]
    lo, up = row['lower'], 'B0F1'
    ids = {int(r): i for i, r in enumerate(train[up]['row_ids'])}
    support = np.array([ids[r] for r in row['support']])
    query = np.array([ids[r] for r in row['query']])
    old = {lo: lower_map(train[lo])}
    statistics = {lo: cells(train[lo]['rssi'], train[lo]['xy'])}
    ep = make_episode(train, statistics, old, lo, up, support, query, cfg)
    model = LearnedMapKernel(cfg).to(cfg['device'])
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 8
    expected, positions = build_map('scm', old[lo], train[up]['rssi'][support], train[up]['xy'][support],
                                    (cfg['length_scale_m'], cfg['noise_ratio']), cfg)
    expected_prediction = tobit_match(ep['rssi'], expected, positions,
                                     cfg['initial_sigma_db'], cfg['detection_theta_dbm'])
    values = model.update(ep['map']).detach().cpu().numpy()
    map_error = float(np.abs(values - expected[:, ep['map'].aps]).max())
    predicted = model.predict(ep['rssi'], ep['map'])
    prediction_error = float(np.abs(predicted - expected_prediction).max())
    assert map_error < 1e-8 and prediction_error < 1e-6
    loss = torch.linalg.vector_norm(model(ep['rssi'], ep['map']) - ep['y'], dim=1).mean()
    loss.backward()
    gradient = model.matrix.grad.detach().clone()
    assert torch.isfinite(gradient).all() and gradient[:, 2:].norm() > 1e-8
    flat = int(gradient[:, 2:].abs().argmax())
    i, j = flat // 2, flat % 2 + 2
    epsilon = 1e-5
    losses = []
    for offset in (epsilon, -epsilon):
        with torch.no_grad():
            model.matrix[i, j] += offset
            losses.append(float(torch.linalg.vector_norm(model(ep['rssi'], ep['map']) - ep['y'], dim=1).mean()))
            model.matrix[i, j] -= offset
    finite_difference = (losses[0] - losses[1]) / (2 * epsilon)
    assert np.isclose(finite_difference, float(gradient[i, j]), rtol=1e-3, atol=1e-6)
    before_metric = model.matrix.detach().T @ model.matrix.detach()
    optimizer = torch.optim.Adam([model.matrix], lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    optimizer.step()
    after_metric = model.matrix.detach().T @ model.matrix.detach()
    changed_values = model.update(ep['map']).detach().cpu().numpy()
    assert not torch.equal(before_metric, after_metric) and np.max(np.abs(changed_values - values)) > 1e-8
    permutation = np.random.default_rng(7).permutation(len(ep['map'].aps))
    radio_map = ep['map']
    permuted = replace(radio_map, aps=radio_map.aps[permutation],
        old_inputs=radio_map.old_inputs[permutation], support_inputs=radio_map.support_inputs[permutation],
        residual=radio_map.residual[permutation], old_values=radio_map.old_values[permutation],
        anchor_values=radio_map.anchor_values[:, permutation])
    permutation_error = float(np.abs(model.predict(ep['rssi'], radio_map) - model.predict(ep['rssi'], permuted)).max())
    assert permutation_error < 1e-6
    result = {'source_episode': up, 'source_support_rows': row['support'], 'source_query_rows': row['query'],
              'author_class_ast_equal': True, 'trainable_parameters': 8,
              'initial_map_max_error_db': map_error, 'initial_prediction_max_error_m': prediction_error,
              'profile_gradient_norm': float(gradient[:, 2:].norm()), 'gradient': gradient.cpu().tolist(),
              'finite_difference_component': [i, j], 'analytic': float(gradient[i, j]), 'finite_difference': finite_difference,
              'one_step_map_change_max_db': float(np.abs(changed_values - values).max()),
              'ap_permutation_max_error_m': permutation_error}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
