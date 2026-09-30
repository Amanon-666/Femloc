"""把配对结果汇成简短报告，分别保留扫描和位置评价及模型种子差异。"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_learned_ap_map import building_mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--report', type=Path, default=Path('docs/LEARNED_AP_MAP_RESULTS.md'))
    args = parser.parse_args()
    data = json.loads((args.output / 'results.json').read_text())
    held = {name: row for block in data['held_buildings'].values() for name, row in block['floors'].items()}
    assert len(held) == 7
    fixed = 'scm_t_frozen' if data['config'].get('observation_model') == 'scm_t' else 'two_channel_frozen'
    lines = ['# 学习 AP 可靠度：训练与结果', '',
             '12 个模型全部完成，每个 600 步；三折整栋建筑留出、每折三模型种子；目标端 0 梯度步。',
             '历史每层 20 组固定 10×3 锚点，所有方法共享 manifest。± 为三个模型种子的样本标准差，未表示独立场地不确定性。', '',
             '## 建筑等权主结果（米）', '',
             '|方法|同批次未标注位置：扫描 / 位置|官方 validation：扫描 / 位置|',
             '|---|---:|---:|']
    totals = {}
    for method in ('old_map', 'scm', 'scm_t', fixed, 'learned'):
        row = []
        totals[method] = {}
        for task in ('same_day', 'official'):
            vals = []
            for metric in ('scan_mde', 'position_mde'):
                values = [building_mean({name: (v[task][method][seed][metric] if method == 'learned'
                                                else v[task][method][metric]) for name, v in held.items()}, list(held))
                          for seed in range(3)]
                totals[method][task + '_' + metric] = values
                vals.append(f'{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}' if method == 'learned' else f'{values[0]:.2f}')
            row.append(' / '.join(vals))
        lines.append(f'|{method}|{row[0]}|{row[1]}|')
    lines += ['', '## 每层官方 validation 的配对结果', '',
              '|楼层|SCM-T|双通道冻结|双通道学习|学习−冻结|', '|---|---:|---:|---:|---:|']
    wins = 0
    for name, value in held.items():
        v = value['official']
        learned = np.array([r['scan_mde'] for r in v['learned']])
        gain = learned - v[fixed]['scan_mde']
        wins += gain.mean() < 0
        lines.append(f"|{name}|{v['scm_t']['scan_mde']:.2f}|{v[fixed]['scan_mde']:.2f}|"
                     f'{learned.mean():.2f} ± {learned.std(ddof=1):.2f}|{gain.mean():+.2f}|')
    deltas = np.array(totals['learned']['official_scan_mde']) - np.array(totals[fixed]['official_scan_mde'])
    lines += ['', f'学习相对冻结的官方 validation 配对差：{deltas.mean():+.3f} ± {deltas.std(ddof=1):.3f} m；改善 {wins}/7 层。',
              '负数表示训练降低误差。每折拟合只包含另外两栋，留出建筑旧图仅在部署时作为输入。', '',
              '## 三个既有目标层（探索性）', '',
              '|楼层|SCM-T|双通道冻结|双通道学习|', '|---|---:|---:|---:|']
    for name, value in data['targets'].items():
        v = value['official']
        learned = np.array([r['scan_mde'] for r in v['learned']])
        lines.append(f"|{name}|{v['scm_t']['scan_mde']:.2f}|{v[fixed]['scan_mde']:.2f}|"
                     f'{learned.mean():.2f} ± {learned.std(ddof=1):.2f}|')
    lines += ['', '## 解释范围', '',
              '- 训练改的是候选–AP 关系的检出证据权重和强度不确定度，共 163 个参数；地图始终是推理输入。',
              '- 几何核与地图更新方式固定；学习没有增加目标标注，也没有使用 Query 标签适应。',
              '- 固定 SCM 参数曾在全部历史建筑选择，SCM-T 是强参考；冻结/学习双通道才是本轮训练收益的直接配对比较。',
              '- 学习选择、官方 validation 及三个最终楼层均处在已有研究环境，结果属探索性；不宣称已排除时间、设备、用户或覆盖影响。',
              '- 仅三栋建筑、七对历史转移；需要相邻旧地图及对齐坐标。', '',
              f"总耗时 {data['elapsed_seconds']:.1f} 秒。原始曲线、逐 episode 指标、row ids 和权重在 `{args.output}`。", '']
    if fixed == 'scm_t_frozen':
        lines[0] = '# SCM-T 地图上的 AP 可靠度学习：训练与结果'
        lines = [line.replace('双通道冻结', 'SCM-T冻结').replace('双通道学习', 'SCM-T学习')
                 .replace('冻结/学习双通道', '冻结/学习SCM-T') for line in lines]
    args.report.write_text('\n'.join(lines))
    summary = {'building_equal': totals, 'paired_learned_minus_frozen_official': deltas.tolist(),
               'improved_floors': int(wins), 'elapsed_seconds': data['elapsed_seconds']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print('\n'.join(lines[:24]), flush=True)


if __name__ == '__main__':
    main()
