"""按训练seed先汇总episode，再报告重复均值及样本标准差。"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    config=json.loads((args.output/'config.json').read_text())
    rows=json.loads((args.output/'results.json').read_text())
    expected=len(config['seeds'])*len(config['targets'])*config['evaluation_episodes']*2
    assert len(rows)==expected
    for seed in config['seeds']:
        folder=args.output/f'seed_{seed}'
        logs=[json.loads(x) for x in (folder/'train.jsonl').read_text().splitlines()]
        assert [x['step'] for x in logs]==list(range(1,config['updates']+1))
        assert all(np.isfinite(x['DKT']) and np.isfinite(x['RBF-GP']) for x in logs)
    assert all(x['support_scans']==30 and x['target_gradient_steps']==0 and np.isfinite(x['mde']) for x in rows)
    lines=['# DKT-UJI 首轮结果','',f"每个训练seed执行{config['updates']}次源任务更新；每目标20组Support，每组10个位置×3条扫描，测试阶段0次参数更新。",'',
           '误差单位米；先平均每个训练seed内20个episode，再计算三个seed的均值±样本标准差。episode重叠，不当作60个独立训练重复。','',
           '|目标|DKT MDE|原始RSSI RBF-GP MDE|DKT−RBF配对差|','|---|---:|---:|---:|']
    for target in config['targets']:
        values={m:np.array([np.mean([r['mde'] for r in rows if r['target']==target and r['seed']==seed and r['method']==m]) for seed in config['seeds']]) for m in ['DKT','RBF-GP']}
        items=[values['DKT'],values['RBF-GP'],values['DKT']-values['RBF-GP']]
        lines.append('|'+target+'|'+'|'.join(f'{v.mean():.3f} ± {v.std(ddof=1):.3f}' for v in items)+'|')
    lines+=['','## 各seed结果','', '|seed|目标|方法|扫描MDE|位置等权MDE|RMSE|后验预测秒/episode|','|---|---|---|---:|---:|---:|---:|']
    for seed in config['seeds']:
        for target in config['targets']:
            for method in ['DKT','RBF-GP']:
                r=[x for x in rows if x['seed']==seed and x['target']==target and x['method']==method]
                vals=[np.mean([x[key] for x in r]) for key in ['mde','position_mde','rmse','seconds']]
                lines.append(f'|{seed}|{target}|{method}|'+ '|'.join(f'{v:.4f}' for v in vals)+'|')
    lines+=['','## 范围','', '- 这是DKT官方机制的UJI适配，不是原论文正弦/QMUL数值复现。',
            '- 首轮固定5000更新，不按目标结果选模型；不宣称源训练已经收敛。',
            '- 三建筑各留一层，不能称跨整栋建筑；设备、用户、时间未被控制。',
            '- 未使用官方validationData；Query位置完全未进入对应Support，推断函数不接收Query标签。',
            '- 固定520维AP身份槽位；源域未见AP和仅13楼层的任务多样性仍是限制。',
            '- 方差是带观测噪声的每坐标GP方差，尚未验证置信区间校准。','']
    report=Path('docs/DKT_UJI_RESULTS.md')
    report.write_text('\n'.join(lines))
    print(report)


if __name__=='__main__': main()
