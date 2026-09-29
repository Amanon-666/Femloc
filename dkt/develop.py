"""仅从原source中留出开发楼层，选择可迁移深核的训练阶段。"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from .data import coordinate_scale, episode, load
from .model import DKT
from .run import tensor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    base=json.loads(Path('configs/dkt_uji.json').read_text())
    cfg=json.loads(Path('configs/dkt_development.json').read_text())
    out=args.output
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg['seed'])
    floors={n:f for n,f in load(base['data_path']).items() if n not in base['targets']}
    sources=sorted(set(floors)-set(cfg['development_floors']))
    assert len(sources)==7 and not set(base['targets']) & set(floors)
    scale=coordinate_scale(floors,sources)
    (out/'config.json').write_text(json.dumps(dict(base=base,development=cfg,sources=sources,scale=scale,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()),indent=2))
    device=base['device']
    models={m:DKT(m=='DKT').double().to(device) for m in ['DKT','RBF-GP']}
    optimizers={m:torch.optim.Adam(model.parameters(),lr=base['learning_rate']) for m,model in models.items()}
    rng=np.random.default_rng(30000+cfg['seed'])
    tasks=[]
    for name in cfg['development_floors']:
        for repeat in range(cfg['episodes_per_floor']):
            s,q=episode(floors[name],rng,base['support_positions'],base['scans_per_position'])
            tasks.append(dict(floor=name,episode=repeat,support=s,query=q))
    (out/'manifest.json').write_text(json.dumps([dict(floor=t['floor'],episode=t['episode'],
        support=floors[t['floor']]['ids'][t['support']].tolist(),query=floors[t['floor']]['ids'][t['query']].tolist()) for t in tasks]))
    rows=[]

    def evaluate(step):
        for task in tasks:
            f=floors[task['floor']]
            s,q=task['support'],task['query']
            origin=f['xy'][s].mean(0)
            sx,qx=[tensor(f['x'][idx],device) for idx in (s,q)]
            sy=tensor((f['xy'][s]-origin)/scale,device)
            for method,model in models.items():
                mean,var=model.predict(sx,sy,qx)
                pred=mean.cpu().numpy()*scale+origin
                assert np.isfinite(pred).all()
                rows.append(dict(step=step,floor=task['floor'],episode=task['episode'],method=method,
                                 mde=float(np.linalg.norm(pred-f['xy'][q],axis=1).mean())))
        for method,model in models.items():
            torch.save(model.state_dict(),out/f'{method}_{step}.pt')
        (out/'results.json').write_text(json.dumps(rows,indent=2))
        print(f'step={step} DKT_dev_MDE='+str(np.mean([r['mde'] for r in rows if r['step']==step and r['method']=='DKT'])),flush=True)

    evaluate(0)
    rng=np.random.default_rng(cfg['seed'])
    with (out/'train.jsonl').open('w') as log:
        for step in range(1,max(cfg['checkpoints'])+1):
            name=sources[int(rng.integers(len(sources)))]
            f=floors[name]
            s,q=episode(f,rng,base['support_positions'],base['scans_per_position'],base['train_query_positions'])
            idx=np.concatenate([s,q])
            x=tensor(f['x'][idx],device)
            y=tensor((f['xy'][idx]-f['xy'][s].mean(0))/scale,device)
            row=dict(step=step,floor=name)
            for method,model in models.items():
                optimizers[method].zero_grad()
                loss=model.loss(x,y)
                if not torch.isfinite(loss): raise ValueError(f'Nonfinite loss at {step}/{method}')
                loss.backward()
                optimizers[method].step()
                row[method]=loss.item()
            log.write(json.dumps(row)+'\n')
            if step in cfg['checkpoints']:
                log.flush()
                evaluate(step)
    scores={step:float(np.mean([r['mde'] for r in rows if r['method']=='DKT' and r['step']==step])) for step in cfg['checkpoints']}
    chosen=min(scores,key=lambda step:(scores[step],step))
    selection=dict(selected_updates=chosen,scores=scores,criterion='equal-floor mean MDE on source-only development tasks')
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    lines=['# DKT 源楼层内部开发结果','', '单seed开发运行；固定三个开发楼层，每层20个位置隔离的30扫描任务。最终三个目标未参与训练或选择。','',
           '|步数|开发楼层|DKT MDE(m)|RBF-GP MDE(m)|','|---|---|---:|---:|']
    for step in cfg['checkpoints']:
        for name in cfg['development_floors']+['等权平均']:
            values=[np.mean([r['mde'] for r in rows if r['step']==step and r['method']==m and (name=='等权平均' or r['floor']==name)]) for m in models]
            lines.append(f'|{step}|{name}|{values[0]:.3f}|{values[1]:.3f}|')
    lines+=['',f'预定规则选择步数：**{chosen}**。开发seed不是三次独立复现，不能从本表宣称稳定优势。','']
    Path('docs/DKT_DEVELOPMENT_RESULTS.md').write_text('\n'.join(lines))
    (out/'completed.json').write_text(json.dumps(dict(status='complete',training_steps=max(cfg['checkpoints']),results=len(rows))))


if __name__=='__main__': main()
