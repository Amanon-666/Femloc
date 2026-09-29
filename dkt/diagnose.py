"""只读诊断已有DKT：核相关、后验回归中心、噪声与源/目标差异。"""
import json
from pathlib import Path

import numpy as np
import torch

from .data import episode, load
from .model import DKT


def main():
    root = Path('outputs/dkt_v1')
    config = json.loads((root/'config.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    floors = load(config['data_path'])
    scale = provenance['coordinate_scale_m']
    torch.set_num_threads(1)
    device = config['device']
    source_seen = np.concatenate([floors[n]['x'] for n in provenance['sources']]).any(0)
    records, parameters, trend = [], [], []
    for seed in config['seeds']:
        folder=root/f'seed_{seed}'
        logs=[json.loads(x) for x in (folder/'train.jsonl').read_text().splitlines()]
        for start in range(0,len(logs),1000):
            trend.append(dict(seed=seed, end=start+1000, **{m:float(np.mean([x[m] for x in logs[start:start+1000]])) for m in ['DKT','RBF-GP']}))
        models={m:DKT(m=='DKT').double().to(device) for m in ['DKT','RBF-GP']}
        for method,model in models.items():
            model.load_state_dict(torch.load(folder/f'{method}.pt',map_location=device,weights_only=True))
            model.eval()
            parameters.append(dict(seed=seed,method=method,axes=[dict(lengthscale=g.covar_module.base_kernel.lengthscale.item(),
                outputscale=g.covar_module.outputscale.item(),noise=g.likelihood.noise.item(),mean=g.mean_module.constant.item()) for g in model.gps]))
        tasks=json.loads((folder/'manifest.json').read_text())
        rng=np.random.default_rng(20000+seed)
        for name in provenance['sources']:
            for repeat in range(5):
                s,q=episode(floors[name],rng,10,3)
                tasks.append(dict(target=name,episode=repeat,support=floors[name]['ids'][s].tolist(),query=floors[name]['ids'][q].tolist()))
        for task in tasks:
            f=floors[task['target']]
            s=np.flatnonzero(np.isin(f['ids'],task['support']))
            q=np.flatnonzero(np.isin(f['ids'],task['query']))
            origin=f['xy'][s].mean(0)
            sx,qx=[torch.as_tensor(f['x'][i],device=device) for i in (s,q)]
            sy=torch.as_tensor((f['xy'][s]-origin)/scale,device=device)
            for method,model in models.items():
                with torch.no_grad():
                    mean,var=model.predict(sx,sy,qx)
                    z,zq=model.feature_extractor(sx),model.feature_extractor(qx)
                    corr=model.gps[0].covar_module.base_kernel(zq,z).to_dense()
                    nn=corr.argmax(1).cpu().numpy()
                pred=mean.cpu().numpy()*scale+origin
                err=np.linalg.norm(pred-f['xy'][q],axis=1)
                records.append(dict(seed=seed,floor=task['target'],seen=task['target'] in provenance['sources'],method=method,
                    mde=float(err.mean()),centroid_mde=float(np.linalg.norm(f['xy'][q]-origin,axis=1).mean()),
                    nn_mde=float(np.linalg.norm(f['xy'][s[nn]]-f['xy'][q],axis=1).mean()),
                    prediction_offset=float(np.linalg.norm(pred-origin,axis=1).mean()),
                    max_correlation=float(corr.max(1).values.mean()),
                    std_radius=float(np.sqrt(var.cpu().numpy().sum(1)).mean()*scale),
                    novel_ap_fraction=float((f['x'][q][:,~source_seen]>0).sum()/max((f['x'][q]>0).sum(),1))))
    result=dict(parameters=parameters,training_trend=trend,records=records)
    Path('outputs/dkt_v1/diagnostics.json').write_text(json.dumps(result,indent=2))
    lines=['# 已有DKT模型机制诊断','', '源楼层是已参与训练的环境，只作拟合参照，不是泛化验证。所有统计均为描述性。','',
           '|环境|方法|MDE|Support中心MDE|最近特征邻居MDE|预测偏离中心(m)|最大核相关|预测标准差半径(m)|新AP观测占比|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['source']+config['targets']:
        for method in models:
            selected=[x for x in records if x['method']==method and (x['seen'] if name=='source' else x['floor']==name)]
            keys=['mde','centroid_mde','nn_mde','prediction_offset','max_correlation','std_radius','novel_ap_fraction']
            lines.append(f'|{name}|{method}|'+ '|'.join(f'{np.mean([x[k] for x in selected]):.4f}' for k in keys)+'|')
    lines+=['','## 训练loss分段均值','', '|seed|结束步|DKT|RBF-GP|','|---|---:|---:|---:|']
    lines += [f"|{x['seed']}|{x['end']}|{x['DKT']:.4f}|{x['RBF-GP']:.4f}|" for x in trend]
    lines+=['','## 每坐标核参数','', '|seed|方法|轴|长度尺度|信号方差|噪声方差|常数均值|','|---|---|---:|---:|---:|---:|---:|']
    for p in parameters:
        for axis,v in enumerate(p['axes']):
            lines.append(f"|{p['seed']}|{p['method']}|{axis}|"+'|'.join(f'{v[k]:.6f}' for k in ['lengthscale','outputscale','noise','mean'])+'|')
    Path('docs/DKT_DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:14]))


if __name__=='__main__': main()
