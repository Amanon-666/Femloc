"""核验完整训练和配对指标，并输出地图核学习的结果报告。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.map_kernel import LearnedMapKernel
from scripts.run_learned_ap_map import building_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads((args.output / 'results.json').read_text())
    cfg, seeds = data['config'], data['config']['model_seeds']
    assert (args.output / 'manifest.json').read_text() == Path(cfg['source_manifest']).read_text()
    assert (args.output / 'target_manifest.json').read_text() == Path(cfg['target_manifest']).read_text()
    names = [f'held_{b}_seed_{s}' for b in ('B0', 'B1', 'B2') for s in seeds]
    names += [f'all_source_seed_{s}' for s in seeds]
    curves = {}
    for name in names:
        directory = args.output / name
        done = json.loads((directory / 'completed.json').read_text())
        checkpoint = torch.load(directory / 'model.pt', map_location='cpu', weights_only=True)
        curve = json.loads((directory / 'curve.json').read_text())
        assert done['steps'] == checkpoint['steps'] == cfg['steps'] and done['trainable_parameters'] == 8
        assert checkpoint['state_dict']['matrix'].shape == (2, 4)
        model = LearnedMapKernel(cfg)
        model.load_state_dict(checkpoint['state_dict'])
        # 核约束的无穷上界是合法配置，有限性检查针对实际模型参数。
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        assert torch.isfinite(model.covariance.outputscale).all()
        assert torch.isfinite(model.covariance.base_kernel.lengthscale).all()
        assert [row['step'] for row in curve] == cfg['curve_steps']
        assert all(np.isfinite(row['source_development_mde']) for row in curve)
        if name.startswith('held_'):
            assert all(floor[:2] != name[5:7] for floor in done['fit_floors'])
        assert not set(done['fit_floors']) & set(cfg['targets'])
        curves[name] = [row['source_development_mde'] for row in curve]
    held = {name: row for block in data['held_buildings'].values() for name, row in block['floors'].items()}
    assert len(held) == 7 and len(data['targets']) == 3
    max_equivalence = 0.0
    for row in {**held, **data['targets']}.values():
        for scope in ('same_day', 'official'):
            for ep in row[f'episodes_{scope}']:
                for method, values in ep.items():
                    for item in values if method == 'learned' else [values]:
                        assert all(np.isfinite(value) for value in item.values())
                max_equivalence = max(max_equivalence, max(abs(ep['scm_t_frozen'][m] - ep['scm_t'][m])
                    for m in ('scan_mde', 'position_mde')))
    assert max_equivalence < 1e-5
    lines = ['# 学习旧地图残差传播核：结果', '',
        f"12 个模型各完成 {cfg['steps']} 个 source 步；每个模型 8 个参数。目标端 0 梯度步。",
        '复用上一轮 10x3 Support/Query manifest；3 折建筑留出、3 source 抽样种子。',
        '所有权重固定后评价官方 validation。± 为 3 个抽样种子的样本标准差，不是独立场地置信区间。', '',
        f"训练与评价使用的实现提交：`{data['code_commit']}`。",
        '同批次 Query 与当前 Support 的位置隔离；官方 validation 使用对应楼层全部行。',
        'old_map 是不更新的旧图，scm 是目标锚点修正，scm_t 加入固定缺失观测匹配；scm_t_frozen 是同一算法的可微等价实现。', '',
        '## 建筑等权主结果（米）', '',
        '|方法|同批次未知位置：扫描 / 位置|官方 validation：扫描 / 位置|', '|---|---:|---:|']
    totals = {}
    for method in ('old_map', 'scm', 'scm_t', 'scm_t_frozen', 'learned'):
        totals[method], text = {}, []
        for scope in ('same_day', 'official'):
            formatted = []
            for metric in ('scan_mde', 'position_mde'):
                values = [building_mean({name: (row[scope][method][i][metric] if method == 'learned'
                    else row[scope][method][metric]) for name, row in held.items()}, list(held))
                    for i in range(len(seeds))]
                totals[method][f'{scope}_{metric}'] = values
                formatted.append(f'{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}'
                                 if method == 'learned' else f'{values[0]:.2f}')
            text.append(' / '.join(formatted))
        lines.append(f'|{method}|{text[0]}|{text[1]}|')
    deltas = np.array(totals['learned']['official_scan_mde']) - np.array(totals['scm_t_frozen']['official_scan_mde'])
    wins = 0
    lines += ['', '## 每层官方 validation', '',
        '|楼层|固定几何核|学习核|学习−固定|', '|---|---:|---:|---:|']
    for name, row in held.items():
        baseline = row['official']['scm_t_frozen']['scan_mde']
        learned = np.array([r['scan_mde'] for r in row['official']['learned']])
        wins += learned.mean() < baseline
        lines.append(f'|{name}|{baseline:.2f}|{learned.mean():.2f} ± {learned.std(ddof=1):.2f}|{learned.mean()-baseline:+.2f}|')
    lines += ['', f'建筑等权配对差：{deltas.mean():+.3f} ± {deltas.std(ddof=1):.3f} m；改善 {wins}/7 层。负数表示学习更好。', '',
        '## 三个既有目标楼层（探索性附录）', '',
        '|楼层|同日固定 / 学习|官方固定 / 学习|核求解中位数 ms|', '|---|---:|---:|---:|']
    for name, row in data['targets'].items():
        text = []
        for scope in ('same_day', 'official'):
            values = np.array([r['scan_mde'] for r in row[scope]['learned']])
            text.append(f"{row[scope]['scm_t_frozen']['scan_mde']:.2f} / {values.mean():.2f} ± {values.std(ddof=1):.2f}")
        lines.append(f"|{name}|{text[0]}|{text[1]}|{np.median(row['kernel_solve_ms']):.3f}|")
    lines += ['', '## 源开发曲线（不用于挑目标结果）', '',
        '|拟合范围|0步|100步|300步|600步|', '|---|---:|---:|---:|---:|']
    for prefix in ('held_B0', 'held_B1', 'held_B2', 'all_source'):
        values = np.array([curves[f'{prefix}_seed_{s}'] for s in seeds])
        lines.append('|' + prefix + '|' + '|'.join(f'{mean:.2f} ± {std:.2f}'
            for mean, std in zip(values.mean(0), values.std(0, ddof=1))) + '|')
    lines += ['', '## 本轮判断', '',
        f'- 官方 validation 的建筑等权配对差为 {deltas.mean():+.3f} m，改善 {wins}/7 层，未形成整体定位收益。',
        '- 梯度检查、实际矩阵变化与源开发曲线下降表明训练已发生；效果弱不能解释成没有更新参数。',
        '- 这轮结果不支持“从旧 AP 分布学习八参数相关性，可整体胜过已有几何传播”的假设。',
        '- 当前明确的收益仍来自完整旧图、少量目标锚点修正和固定匹配。保留固定 SCM-T 为后续比较参照。',
        '- 学习只改变旧点上的 RSS 值；候选坐标与匹配器保持不变。预测始终是候选坐标的加权均值，不能超出其凸包。',
        '- 三个最终楼层仅为附录；不能用其中两个小幅改善来覆盖七层建筑留出主比较。',
        '- 本轮不根据最终 Query 追加预算、挑最好步骤或叠加模块。', '',
        '## 解释范围', '',
        '- 本轮仅训练地图更新相关性；匹配器、候选、目标数据量与固定几何核共用。',
        '- 实际复用作者 DKT 协方差类，但输入与训练损失已改造，因此不是 DKT/FeMLoc 原文复现。',
        '- 只有三建筑、七对相关转移；固定几何核和匹配参数有使用全部历史建筑调优的历史。',
        '- validation 与三个最终目标历史上已查看，属于探索性；用户、设备与时间影响未排除。',
        '- 需要完整相邻旧图及水平坐标对应，不能外推成无旧图的未知整栋建筑定位。',
        '- 核求解计时不含旧图读取、上下文准备或 Query 定位，不能与 MAML 一步耗时直接比较。', '',
        f"总耗时 {data['elapsed_seconds']:.1f} s；冻结实现与原 NumPy SCM-T 最大指标差 {max_equivalence:.3g} m。", '']
    Path('docs/MAP_KERNEL_RESULTS.md').write_text('\n'.join(lines))
    summary = {'building_equal': totals, 'paired_learned_minus_frozen_official': deltas.tolist(),
        'improved_floors': int(wins), 'initial_equivalence_max_mde_difference_m': max_equivalence,
        'models': len(names), 'steps_each': cfg['steps'], 'code_commit': data['code_commit'],
        'source_curves': curves, 'elapsed_seconds': data['elapsed_seconds']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (args.output / 'completed.json').write_text(json.dumps({'models': len(names), 'steps_each': cfg['steps'],
        'held_floors': len(held), 'evaluation_episodes_each': cfg['evaluation_episodes_per_pair'],
        'targets': len(data['targets']), 'metrics_finite': True, 'report': 'docs/MAP_KERNEL_RESULTS.md'}, indent=2) + '\n')
    print('\n'.join(lines[:25]), flush=True)


if __name__ == '__main__':
    main()
