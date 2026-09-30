"""以共享相关性核将少量目标锚点的变化传播到完整旧地图。"""
from dataclasses import dataclass

import gpytorch
import numpy as np
import torch
from torch import nn

from models.dkt_reference import ExactGPLayer
from scripts.evaluate_signal_calibrated_map import lookup, position_map


@dataclass
class KernelMap:
    aps: np.ndarray
    origin: np.ndarray
    old_inputs: torch.Tensor
    support_inputs: torch.Tensor
    residual: torch.Tensor
    old_values: torch.Tensor
    anchor_values: torch.Tensor
    positions: torch.Tensor


def prepare(old_statistics, coded_old, support_rssi, support_xy, cfg):
    """只用旧图与 Support 建立地图上下文，不接收 Query 或其坐标。"""
    old_xy, probability, _, old_seen, _ = old_statistics
    old_values, coded_xy = coded_old
    assert np.array_equal(old_xy, coded_xy)
    anchor, anchor_xy, _ = position_map(support_rssi, support_xy)
    assert len(anchor_xy) == cfg['support_positions']
    aps = np.flatnonzero(old_seen.any(0) | (support_rssi != 100).any(0))
    origin = old_xy.mean(0)
    base_anchor = lookup(anchor_xy, old_xy, old_values, cfg['anchor_lookup_neighbors'])
    anchor_probability = lookup(anchor_xy, old_xy, probability, cfg['anchor_lookup_neighbors'])
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=cfg['device'])

    def inputs(xy, values, frequency):
        relative = (xy - origin) / cfg['coordinate_scale_m']
        coordinates = np.broadcast_to(relative[:, None], (len(xy), len(aps), 2))
        features = np.concatenate((coordinates, ((values[:, aps] + 110) / 110)[..., None],
                                   frequency[:, aps, None]), axis=-1)
        return tensor(features).permute(1, 0, 2).contiguous()

    return KernelMap(aps, origin, inputs(old_xy, old_values, probability),
                     inputs(anchor_xy, base_anchor, anchor_probability),
                     tensor(((anchor - base_anchor)[:, aps] / 110).T[..., None]),
                     tensor(((old_values[:, aps] + 110) / 110).T),
                     tensor(anchor[:, aps]), tensor(np.vstack((old_xy, anchor_xy)) - origin))


class LearnedMapKernel(nn.Module):
    """八参数核、可微残差求解与固定 SCM-T 匹配器。"""
    def __init__(self, cfg):
        super().__init__()
        ratio = cfg['coordinate_scale_m'] / cfg['length_scale_m']
        self.matrix = nn.Parameter(torch.tensor([[ratio, 0, 0, 0], [0, ratio, 0, 0]], dtype=torch.float64))
        reference = ExactGPLayer(torch.zeros(10, 2), torch.zeros(10),
                                 gpytorch.likelihoods.GaussianLikelihood(), kernel='rbf')
        self.covariance = reference.covar_module.double()
        self.covariance.initialize(outputscale=1.0)
        self.covariance.base_kernel.initialize(lengthscale=1.0)
        self.covariance.requires_grad_(False)
        self.noise = cfg['noise_ratio']
        self.sigma = cfg['initial_sigma_db']
        self.theta = cfg['detection_theta_dbm']

    def update(self, radio_map):
        """按 AP 批量解十个锚点的系统，并保留真实 Support 原型。"""
        support = radio_map.support_inputs @ self.matrix.T
        old = radio_map.old_inputs @ self.matrix.T
        gram = self.covariance(support, support).to_dense()
        cross = self.covariance(old, support).to_dense()
        eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
        coefficients = torch.linalg.solve(gram + self.noise * eye, radio_map.residual)
        values = radio_map.old_values + (cross @ coefficients).squeeze(-1)
        virtual = (110 * values.T - 110).clamp(-110, 0)
        return torch.cat((virtual, radio_map.anchor_values), dim=0)

    def match(self, raw, values, radio_map):
        """固定证据分数计算候选坐标均值，不训练 AP 权重或匹配尺度。"""
        raw = torch.as_tensor(raw[:, radio_map.aps], device=values.device, dtype=values.dtype)
        detected = (raw != 100).to(values.dtype)
        x = torch.where(raw != 100, raw, 0)
        distance = x.square().sum(1, keepdim=True) + detected @ values.square().T - 2 * x @ values.T
        score = -distance / (2 * self.sigma ** 2) + (1 - detected) @ torch.special.log_ndtr(
            (self.theta - values) / self.sigma).T
        return score.softmax(1) @ radio_map.positions

    def forward(self, raw, radio_map):
        return self.match(raw, self.update(radio_map), radio_map)

    def predict(self, raw, radio_map):
        """一次更新地图，再分块预测全部扫描并恢复绝对坐标。"""
        with torch.no_grad():
            values = self.update(radio_map)
            blocks = [self.match(block, values, radio_map).cpu().numpy() + radio_map.origin
                      for block in np.array_split(raw, max(1, (len(raw) + 255) // 256))]
        return np.concatenate(blocks)
