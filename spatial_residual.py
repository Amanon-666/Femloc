"""Support spatial statistics -> identity-initialized monotone RSS map -> fixed raw GP.

The network is shared across APs. No target gradients or Query statistics are used.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from adkf_uji import DATA, Model, tensor, load, coordinate_scale, eligible, episode, posterior

BASE = Path('/home/panyushuo/projects/panyushuo/DKT-ADKF-20261013/outputs/adkf_lobo')
KS = [5, 10, 20]
CHECKS = [0, 300, 1000]
DRAWS = 5
FEATURES = ['visible_position_fraction', 'within_position_sd', 'repeat_observation_fraction',
            'ridge_training_r2', 'ridge_slope_norm', 'insufficient_positions',
            'geometry_rank_ratio', 'visible_rss_mean']


def descriptor(sx, sy):
    """Eight explicit statistics; regress only observed RSS at unique positions.

    sy uses a source-only metric scale. Missing RSS is 0 and is never used as a
    physical RSS observation in ridge or repeat-noise estimates.
    """
    _, groups = torch.unique(sy, dim=0, return_inverse=True)
    k = int(groups.max()) + 1
    vis = sx > 0
    count = torch.zeros(k, sx.shape[1])
    sums = torch.zeros_like(count)
    squares = torch.zeros_like(count)
    count.index_add_(0, groups, vis.to(sx.dtype))
    sums.index_add_(0, groups, sx * vis)
    squares.index_add_(0, groups, sx.square() * vis)
    observed = count > 0
    means = sums / count.clamp_min(1)
    n = observed.sum(0)
    n_safe = n.clamp_min(1)
    points = torch.stack([sy[groups == j][0] for j in range(k)])
    center = torch.einsum('ka,kd->ad', observed.to(sx.dtype), points) / n_safe[:, None]
    r = points[None] - center[:, None]
    w = observed.T.to(sx.dtype)
    mean = (means * observed).sum(0) / n_safe
    signal = means.T - mean[:, None]
    gram = torch.einsum('ak,aki,akj->aij', w, r, r) / n_safe[:, None, None]
    rhs = torch.einsum('ak,aki,ak->ai', w, r, signal) / n_safe[:, None]
    beta = torch.linalg.solve(gram + .1 * torch.eye(2)[None], rhs[..., None])[..., 0]
    beta = beta * (n >= 3)[:, None]
    fitted = torch.einsum('aki,ai->ak', r, beta)
    total = (signal.square() * w).sum(-1)
    residual = ((signal - fitted).square() * w).sum(-1)
    r2 = torch.where(total > 1e-12, 1 - residual / total.clamp_min(1e-12), 0.).clamp(0, 1)
    r2 = r2 * (n >= 3)
    repeated = count > 1
    within_sse = (squares - sums.square() / count.clamp_min(1)).clamp_min(0)
    within_df = ((count - 1).clamp_min(0) * repeated).sum(0)
    sd = (within_sse.sum(0) / within_df.clamp_min(1)).sqrt()
    eigen = torch.linalg.eigvalsh(gram).clamp_min(0)
    rank = eigen[:, 0] / eigen[:, 1].clamp_min(1e-12)
    return torch.stack([n / k, sd * 10., repeated.sum(0) / k, r2,
                        beta.norm(dim=-1), (n < 3).to(sx.dtype), rank, mean], -1).detach()


class ResidualMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(len(FEATURES), 16), nn.Tanh(), nn.Linear(16, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, stats):
        # Four positive slopes. At zero logits each slope=1, so phi(x)=x exactly.
        # A task-constant additive AP offset would cancel from every RBF distance.
        logits = self.net(stats)
        logslopes = logits.tanh() * (stats[:, 0] > 0)[:, None]
        widths = (x[..., None] - torch.arange(4) / 4).clamp(0., .25)
        return x + ((logslopes.exp() - 1.)[None] * widths).sum(-1)


def prediction(model, gp, sx, sy, qx):
    stats = descriptor(sx, sy)
    s, q = model(sx, stats), model(qx, stats)
    return torch.stack([posterior(s, sy[:, axis], q, gp[axis])[0] for axis in range(2)], -1)


def base_gp(run, final):
    choice = json.loads((BASE / run / 'selected.json').read_text())
    folder = BASE / run / 'final' if final else BASE / run
    state = torch.load(folder / f"checkpoint_{choice['raw_source']}.pt", weights_only=True)
    return state['raw_source']['gp'].detach().clone()


def evaluate(model, gp, floors, valid, names, scale, manifests, methods, wknn):
    rows = []
    for spec in manifests:
        name, draw, k = spec['floor'], spec['draw'], spec['k']
        if name not in names or k not in KS or draw >= DRAWS:
            continue
        f, v = floors[name], valid[name]
        id_s = {int(j): i for i, j in enumerate(f['ids'])}
        id_v = {int(j): i for i, j in enumerate(v['ids'])}
        s = np.array([id_s[j] for j in spec['support']])
        qr = np.array([id_s[j] for j in spec['internal_query']])
        vr = np.array([id_v[j] for j in spec['official_query']])
        center = f['xy'][s].mean(0)
        sx, sy = tensor(f['x'][s]), tensor((f['xy'][s] - center) / scale)
        for split, xq, yq in [('internal', f['x'][qr], f['xy'][qr]),
                              ('official', v['x'][vr], v['xy'][vr])]:
            qx = tensor(xq)
            for method in methods:
                start = time.perf_counter()
                with torch.no_grad():
                    if method == 'wknn':
                        # Aggregate three scans into one physical anchor before matching.
                        _, group = torch.unique(sy, dim=0, return_inverse=True)
                        px = torch.stack([sx[group == j].mean(0) for j in range(int(group.max()) + 1)])
                        py = torch.stack([sy[group == j][0] for j in range(len(px))])
                        dist = (qx[:, None] - px[None]).square().sum(-1)
                        nearest = dist.topk(min(wknn[0], len(px)), largest=False)
                        weight = nearest.values.clamp_min(1e-12).pow(-wknn[1] / 2)
                        weight = weight / weight.sum(-1, keepdim=True)
                        pred = (weight[..., None] * py[nearest.indices]).sum(1)
                    else:
                        active = model if method == 'spatial_residual' else ResidualMap()
                        pred = prediction(active, gp, sx, sy, qx)
                err = np.linalg.norm(pred.numpy() * scale + center - yq, axis=1)
                rows.append(dict(floor=name, draw=draw, k=k, split=split, method=method,
                                 mde=float(err.mean()), nquery=len(err),
                                 predict_seconds=time.perf_counter() - start))
    return rows


def dev_manifest(floors, valid, names):
    from adkf_uji import fixed_queries
    specs = []
    for name in names:
        f, v = floors[name], valid[name]
        sp, qp = fixed_queries(f, 771)
        qr = np.concatenate([f['pools'][j] for j in qp])
        for draw in range(2):
            rng = np.random.default_rng(12000 + draw)
            order = rng.permutation(sp)
            scans = {int(j): rng.choice(f['pools'][j], 3, False) for j in order[:20]}
            max_s = np.concatenate([scans[int(j)] for j in order[:20]])
            mask = ~np.any(np.all(v['xy'][:, None] == f['xy'][max_s][None], -1), 1)
            for k in KS:
                s = np.concatenate([scans[int(j)] for j in order[:k]])
                specs.append(dict(floor=name, draw=draw, k=k, support=f['ids'][s].tolist(),
                                  internal_query=f['ids'][qr].tolist(), official_query=v['ids'][mask].tolist()))
    return specs


def train(floors, names, scale, gp, seed, checks, folder, dev=None):
    torch.manual_seed(seed)
    model = ResidualMap()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    states, scores = {}, {}
    start = time.time()
    for step in range(max(checks) + 1):
        if step in checks:
            states[step] = copy.deepcopy(model.state_dict())
            torch.save(states[step], folder / f'map_{step}.pt')
            if dev is not None:
                valid, devnames, specs = dev
                rows = evaluate(model, gp, floors, valid, devnames, scale, specs, ['spatial_residual'], (3, 1.))
                scores[step] = float(np.mean([r['mde'] for r in rows]))
                print('SOURCE_DEV', step, scores[step], flush=True)
        if step == max(checks):
            break
        name = names[int(rng.integers(len(names)))]
        task = episode(floors[name], rng, scale, k=int(rng.choice(KS)), augment=True)
        opt.zero_grad()
        pred = prediction(model, gp, task[0], task[1], task[2])
        loss = (pred - task[3]).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite source prediction loss')
        loss.backward()
        opt.step()
        if (step + 1) % 100 == 0:
            print(json.dumps(dict(step=step + 1, seconds=round(time.time() - start, 2), loss=float(loss.detach()))), flush=True)
    return states, scores


def preflight(floors, sources, scale, gp):
    torch.manual_seed(0)
    task = episode(floors[sources[0]], np.random.default_rng(20261014), scale, k=5, augment=True)
    sx, sy, qx, qy = task[:4]
    model = ResidualMap()
    stats = descriptor(sx, sy)
    assert torch.equal(model(sx, stats), sx)
    assert torch.equal(model(qx, stats), qx)
    raw = torch.stack([posterior(sx, sy[:, a], qx, gp[a])[0] for a in range(2)], -1)
    pred = prediction(model, gp, sx, sy, qx)
    assert torch.equal(raw, pred)
    loss = (pred - qy).square().mean()
    loss.backward()
    initial_hidden = max(float(p.grad.abs().max()) for p in model.net[0].parameters())
    initial_last = max(float(p.grad.abs().max()) for p in model.net[-1].parameters())
    assert initial_hidden == 0 and initial_last > 0
    torch.optim.Adam(model.parameters(), lr=1e-3).step()
    model.zero_grad()
    (prediction(model, gp, sx, sy, qx) - qy).square().mean().backward()
    second_hidden = max(float(p.grad.abs().max()) for p in model.net[0].parameters())
    assert second_hidden > 0
    perm = torch.randperm(sx.shape[1])
    assert torch.allclose(prediction(model, gp, sx, sy, qx),
                          prediction(model, gp, sx[:, perm], sy, qx[:, perm]), atol=1e-10)
    order = torch.randperm(len(sx))
    assert torch.allclose(prediction(model, gp, sx, sy, qx),
                          prediction(model, gp, sx[order], sy[order], qx), atol=1e-10)
    assert torch.allclose(descriptor(sx, sy), descriptor(sx, sy + tensor([100., -80.])), atol=1e-10)
    constant = descriptor(sx, torch.zeros_like(sy))
    assert torch.isfinite(constant).all() and torch.equal(constant[:, 3:5], torch.zeros_like(constant[:, 3:5]))
    assert torch.equal(prediction(model, gp, sx, torch.zeros_like(sy), qx), torch.zeros_like(qy))
    grid = torch.linspace(0, 1, 101)[:, None].repeat(1, sx.shape[1])
    mapped = model(grid, stats)
    assert torch.all(mapped[1:] > mapped[:-1])
    assert torch.equal(model(torch.zeros_like(sx), stats), torch.zeros_like(sx))
    audit = dict(identity_max_error=float((pred - raw).detach().abs().max()),
                 initial_last_gradient=initial_last, initial_hidden_gradient=initial_hidden,
                 hidden_gradient_after_first_step=second_hidden,
                 valid_statistics=True, monotone=True, ap_permutation=True, support_permutation=True,
                 coordinate_translation=True, constant_axis=True,
                 note='Source-only implementation evidence, not a performance experiment')
    # Only source queries are used for these mathematical checks.
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--building', type=int, required=True)
    parser.add_argument('--mode', choices=['preflight', 'run'], default='preflight')
    args = parser.parse_args()
    floors, valid = load(DATA / 'trainingData.csv'), load(DATA / 'validationData.csv')
    run = f'B{args.building}_seed0'
    prior = json.loads((BASE / run / 'protocol.json').read_text())
    source, devs, trained, targets = prior['sources'], prior['dev'], prior['train'], prior['target']
    scale = coordinate_scale(floors, source)
    out = Path(__file__).parent / 'outputs' / 'spatial_residual' / run
    out.mkdir(parents=True, exist_ok=True)
    spec = dict(building=args.building, seed=0, sources=source, source_train=trained,
                source_dev=devs, targets=targets, coordinate_scale=scale, ks=KS,
                draws=DRAWS, scans_per_position=3, checks=CHECKS, features=FEATURES,
                ridge_statistic_regularization=.1, raw_gp_parameters='Frozen permitted source checkpoint',
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / 'protocol.json').write_text(json.dumps(spec, indent=2))
    gp_dev, gp_final = base_gp(run, False), base_gp(run, True)
    if args.mode == 'preflight':
        audit = preflight(floors, trained, scale, gp_dev)
        (out / 'source_preflight.json').write_text(json.dumps(audit, indent=2))
        print(audit, flush=True)
        return
    specs = dev_manifest(floors, valid, devs)
    folder = out / 'source_dev'; folder.mkdir(exist_ok=True)
    states, scores = train(floors, trained, scale, gp_dev, 0, CHECKS, folder, (valid, devs, specs))
    chosen = min(scores, key=scores.get)
    grid = {}
    identity = ResidualMap()
    for neighbors in [1, 3, 5]:
        for power in [1., 2.]:
            r = evaluate(identity, gp_dev, floors, valid, devs, scale, specs, ['wknn'], (neighbors, power))
            grid[f'{neighbors},{power}'] = float(np.mean([x['mde'] for x in r]))
    wkkey = min(grid, key=grid.get); neighbors, power = wkkey.split(',')
    wk = (int(neighbors), float(power))
    (out / 'selected.json').write_text(json.dumps(dict(step=chosen, dev_scores=scores,
                                                      wknn=wk, wknn_grid=grid), indent=2))
    final = out / 'final'; final.mkdir(exist_ok=True)
    states, _ = train(floors, source, scale, gp_final, 0, [chosen], final)
    model = ResidualMap(); model.load_state_dict(states[chosen])
    original = json.loads((BASE / run / 'manifest.json').read_text())
    selected_manifest = [x for x in original if x['k'] in KS and x['draw'] < DRAWS]
    rows = evaluate(model, gp_final, floors, valid, targets, scale, selected_manifest,
                    ['raw_source', 'wknn', 'spatial_residual'], wk)
    (out / 'manifest.json').write_text(json.dumps(selected_manifest))
    (out / 'results.json').write_text(json.dumps(rows, indent=2))
    print('DONE', run, chosen, len(rows), flush=True)


if __name__ == '__main__':
    main()
