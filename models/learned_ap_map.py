"""从旧地图和少量锚点构造双通道地图，学习每个 AP 对候选位置的可靠度。"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.gufu_reference import rssi2weight
from scripts.evaluate_signal_calibrated_map import lookup


def cells(rssi, xy, fallback=None):
    """按坐标汇总不同指纹的检出比例和条件 RSSI；100 不进入强度均值。"""
    _, first = np.unique(np.column_stack([rssi, xy]), axis=0, return_index=True)
    rssi, xy = rssi[first], xy[first]
    positions, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = inv.ravel()
    seen = rssi != 100
    counts, sums = (np.zeros((len(positions), 520)) for _ in range(2))
    np.add.at(counts, inv, seen)
    np.add.at(sums, inv, np.where(seen, rssi, 0))
    if fallback is None:
        fallback = np.divide(sums.sum(0), counts.sum(0), out=np.full(520, -90.0),
                             where=counts.sum(0) > 0)
    probability = counts / np.bincount(inv)[:, None]
    strength = np.divide(sums, counts, out=np.broadcast_to(fallback, sums.shape).copy(), where=counts > 0)
    return positions, probability, strength, counts > 0, fallback


def kernel(a, b, length):
    return np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / (2 * length ** 2))


def field(to_xy, from_xy, values, length, noise):
    """小型 GP 线性求解，同时返回修正和锚点覆盖度。"""
    cross = kernel(to_xy, from_xy, length)
    weights = np.linalg.solve(kernel(from_xy, from_xy, length) + noise * np.eye(len(from_xy)), cross.T)
    return cross @ np.linalg.solve(kernel(from_xy, from_xy, length) + noise * np.eye(len(from_xy)), values), \
        np.clip((cross * weights.T).sum(1), 0, 1)


@dataclass
class APMap:
    aps: np.ndarray
    probability: torch.Tensor
    strength: torch.Tensor
    features: torch.Tensor
    positions: torch.Tensor
    origin: np.ndarray


def build(old, support_rssi, support_xy, config, device):
    """只读取旧地图和 Support；两个通道经同 AP 的空间残差关系更新。"""
    pos, p0, m0, old_seen, fallback = old
    sy, ps, ms, support_seen, _ = cells(support_rssi, support_xy, fallback)
    neighbors = config['anchor_lookup_neighbors']
    ap0 = lookup(sy, pos, p0, neighbors)
    am0 = lookup(sy, pos, m0, neighbors)
    candidates = np.vstack([pos, sy])
    base_p, base_m = np.vstack([p0, ap0]), np.vstack([m0, am0])
    delta_p, coverage_p = field(candidates, sy, ps - ap0, config['length_scale_m'], config['noise_ratio'])
    probability = np.clip(base_p + delta_p, config['probability_epsilon'], 1 - config['probability_epsilon'])
    strength, coverage_m = base_m.copy(), np.zeros_like(base_m)
    active = old_seen.any(0) | support_seen.any(0)
    for a in np.flatnonzero(support_seen.any(0)):
        selected = support_seen[:, a]
        change, coverage = field(candidates, sy[selected], ms[selected, a] - am0[selected, a],
                                 config['length_scale_m'], config['noise_ratio'])
        strength[:, a] += change
        coverage_m[:, a] = coverage
    strength = np.clip(strength, -110, 0)
    probability[-len(sy):] = np.clip(ps, config['probability_epsilon'], 1 - config['probability_epsilon'])
    strength[-len(sy):] = ms
    coverage_m[-len(sy):] = support_seen
    coverage_p[-len(sy):] = 1
    features = np.stack([base_p, probability, rssi2weight(110, base_m) / 110, rssi2weight(110, strength) / 110,
                         np.broadcast_to(coverage_p[:, None], base_m.shape), coverage_m,
                         np.broadcast_to(support_seen.mean(0), base_m.shape)], -1)
    aps, origin = np.flatnonzero(active), pos.mean(0)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
    return APMap(aps, tensor(probability[:, aps]), tensor(strength[:, aps]), tensor(features[:, aps]),
                 tensor(candidates - origin), origin)


class APReliability(nn.Module):
    """AP 共享网络输出检出证据权重与强度噪声；候选坐标由后验加权输出。"""
    def __init__(self, config):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(7, config['hidden_width']), nn.SiLU(),
                                     nn.Linear(config['hidden_width'], 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.temperature = nn.Parameter(torch.tensor(0.0))
        self.minimum_sigma = config['minimum_sigma_db']
        self.initial_sigma = config['initial_sigma_db']

    def forward(self, rssi, radio_map):
        raw = torch.as_tensor(rssi[:, radio_map.aps], device=radio_map.strength.device, dtype=torch.float32)
        detected = (raw != 100).float()
        x = torch.where(raw != 100, raw, 0)
        h = self.network(radio_map.features)
        importance = F.softplus(h[..., 0]) / math.log(2)
        sigma = self.minimum_sigma + (self.initial_sigma - self.minimum_sigma) * F.softplus(h[..., 1]) / math.log(2)
        p, m = radio_map.probability, radio_map.strength
        precision = sigma.square().reciprocal()
        score = detected @ (importance * p.log()).T + (1 - detected) @ (importance * torch.log1p(-p)).T
        distance = x.square() @ precision.T - 2 * x @ (precision * m).T \
            + detected @ (precision * m.square() + 2 * sigma.log()).T
        score = (score - 0.5 * distance) / (F.softplus(self.temperature) / math.log(2))
        return torch.softmax(score, 1) @ radio_map.positions

    def predict(self, rssi, radio_map):
        """分块前向，输出绝对水平坐标；不接收 Query 标签。"""
        with torch.no_grad():
            blocks = [self(block, radio_map).cpu().numpy() + radio_map.origin
                      for block in np.array_split(rssi, max(1, math.ceil(len(rssi) / 256)))]
        return np.concatenate(blocks)
