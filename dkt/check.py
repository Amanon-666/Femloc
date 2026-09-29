"""最小验证：源码提取一致、位置隔离、梯度训练、官方GP后验公式及零目标更新。"""
import ast
import json
from pathlib import Path

import numpy as np
import torch

from .data import episode, load
from .model import DKT


def main():
    for filename, classname in [('train_DKT.py', 'Feature'), ('DKT_regression.py', 'ExactGPLayer')]:
        trees = [ast.parse(Path(p).read_text()) for p in
                 [f'third_party/dkt/{filename}', 'dkt/upstream_classes.py']]
        nodes = [next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name==classname) for tree in trees]
        assert ast.dump(nodes[0]) == ast.dump(nodes[1])
    torch.set_num_threads(1)
    torch.manual_seed(0)
    config = json.loads(Path('configs/dkt_uji.json').read_text())
    floor = load(config['data_path'])['B0F0']
    s, q = episode(floor, np.random.default_rng(0), 10, 3, 10)
    assert len(s)==30 and len(np.unique(floor['pos'][s]))==10
    assert not set(floor['pos'][s]) & set(floor['pos'][q])
    for position in np.unique(floor['pos'][s]):
        assert len(np.unique(floor['x'][s[floor['pos'][s]==position]], axis=0))==3
    sx, qx = [torch.tensor(floor['x'][idx], device=config['device']) for idx in (s,q)]
    sy = torch.tensor((floor['xy'][s]-floor['xy'][s].mean(0))/30., device=config['device'])
    model = DKT().double().to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss = model.loss(sx, sy)
    loss.backward()
    assert model.feature_extractor.layer1.weight.grad.norm() > 0
    optimizer.step()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    mean, variance = model.predict(sx, sy, qx)
    assert mean.shape==(30,2) and variance.shape==(30,2)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    # 用直接Cholesky解检查库的后验均值与观测方差，而不重新实现训练算法。
    with torch.no_grad():
        z, zq = model.feature_extractor(sx), model.feature_extractor(qx)
        for axis, gp in enumerate(model.gps):
            covariance = gp.covar_module(z).to_dense() + gp.likelihood.noise * torch.eye(len(s),device=sx.device)
            cross = gp.covar_module(zq,z).to_dense()
            chol = torch.linalg.cholesky(covariance)
            weights = torch.cholesky_solve((sy[:,axis]-gp.mean_module(z)).unsqueeze(1),chol)
            expected = gp.mean_module(zq)+ (cross@weights).squeeze(1)
            expected_var = gp.covar_module(zq,diag=True) - (cross * torch.cholesky_solve(cross.T,chol).T).sum(1) + gp.likelihood.noise
            torch.testing.assert_close(mean[:,axis],expected,atol=1e-7,rtol=1e-7)
            torch.testing.assert_close(variance[:,axis],expected_var,atol=1e-7,rtol=1e-7)
    print(json.dumps(dict(upstream_classes_exact=True,support_scans=30,position_disjoint=True,
                          posterior_formula_matches=True,target_parameter_updates=0,loss=float(loss.detach()))))


if __name__=='__main__': main()
