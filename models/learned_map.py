"""AP-shared learned residual propagation; no Query labels enter adaptation.

The GP and Tobit formulas are differentiable ports of this repository's TPM.
See docs/research/LEARNED_MAP_DERIVATION.md for the objective and limitations.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from scripts.evaluate_signal_calibrated_map import lookup, position_map


def descriptors(old, prior):
    """Seven bounded, AP-shared descriptors; no AP ID or absolute coordinates."""
    detect = old['k'] / old['n'][:, None]
    observed = np.where(old['k'] > 0, old['s1'] / np.maximum(old['k'], 1), -110.)
    encoded = old['s1'] / old['n'][:, None] - 110. * (1 - detect)
    norm = lambda x: (x + 110.) / 110.
    shape = prior.shape
    z = np.stack([norm(prior), norm(encoded), detect, norm(observed),
                  np.broadcast_to(norm(observed.max(0)), shape),
                  np.broadcast_to((old['k'] > 0).mean(0), shape),
                  np.broadcast_to(norm(encoded.mean(0)), shape)], axis=-1)
    return z, encoded


def tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def squared_distance(a, b):
    return (a.square().sum(-1, keepdim=True)
            + b.square().sum(-1).unsqueeze(-2)
            - 2 * a @ b.transpose(-1, -2)).clamp_min(0)


@dataclass
class Support:
    xy: torch.Tensor
    z: torch.Tensor
    anchor: torch.Tensor
    prior: torch.Tensor

    @property
    def residual(self):
        return self.anchor - self.prior


class MapContext:
    """Only the old floor and a source-fitted prior are available here."""
    def __init__(self, old, prior, device='cpu'):
        self.device = device
        self.positions = np.asarray(old['xy'])
        self.origin = self.positions.mean(0)
        self.base = np.asarray(prior)
        self.description, self.old_encoded = descriptors(old, self.base)
        self.xy = tensor(self.positions - self.origin, device)
        self.f = tensor(self.base, device)
        self.z = tensor(self.description, device)
        self.active = torch.as_tensor(old['k'].sum(0) > 0, device=device)

    def at(self, xy):
        base = lookup(xy, self.positions, self.base, 3)
        z = lookup(xy, self.positions, self.description.reshape(len(self.positions), -1), 3)
        return tensor(base, self.device), tensor(z.reshape(len(xy), *self.description.shape[1:]), self.device)

    def support(self, rssi, xy):
        if len(xy) == 0:
            return Support(self.xy[:0], self.z[:0], self.f[:0], self.f[:0])
        anchor, keys, _ = position_map(rssi, xy)
        prior, z = self.at(keys)
        return Support(tensor(keys - self.origin, self.device), z, tensor(anchor, self.device), prior)


class MapEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, 16), nn.Tanh(), nn.Linear(16, 4))

    def forward(self, x):
        return self.net(x)


def kernels(context, support, length, encoder=None):
    spatial_ps = torch.exp(-squared_distance(context.xy, support.xy) / (2 * length**2))
    spatial_ss = torch.exp(-squared_distance(support.xy, support.xy) / (2 * length**2))
    aps = context.f.shape[1]
    if encoder is None:
        gate = torch.ones((aps, len(context.xy), len(support.xy)),
                          dtype=context.f.dtype, device=context.device)
        return spatial_ps.expand(aps, -1, -1), spatial_ss.expand(aps, -1, -1), gate
    zp = encoder(context.z).transpose(0, 1)
    zs = encoder(support.z).transpose(0, 1)
    gate = torch.exp(-squared_distance(zp, zs) / 2)
    return spatial_ps * gate, spatial_ss * torch.exp(-squared_distance(zs, zs) / 2), gate


def adapt(context, support, field, encoder=None, diagnostics=False):
    """Source network is fixed at deployment. Adaptation is a batched KxK solve."""
    if len(support.xy) == 0:
        return (context.f, context.xy), {}
    kps, kss, gate = kernels(context, support, field[0], encoder)
    eye = torch.eye(len(support.xy), dtype=context.f.dtype, device=context.device)
    system = kss + field[1] * eye
    rhs = support.residual.T.unsqueeze(-1)
    alpha = torch.linalg.solve(system, rhs)
    delta = (kps @ alpha).squeeze(-1).T
    virtual = (context.f + delta).clamp(-110., 0.)
    mapped = (torch.cat([virtual, support.anchor]), torch.cat([context.xy, support.xy]))
    info = {}
    if diagnostics:
        reconstructed_s = support.prior + (kss @ alpha).squeeze(-1).T
        active_gate = gate[context.active]
        info = {'field_support_rmse': float((reconstructed_s - support.anchor).square().mean().sqrt()),
                'map_delta_rms': float(delta.square().mean().sqrt()),
                'active_gate_mean': float(active_gate.mean()),
                'active_gate_min': float(active_gate.min()),
                'normal_residual': float((system @ alpha - rhs).norm() / rhs.norm().clamp_min(1e-12))}
    return mapped, info


def prior_with_support(context, support):
    return torch.cat([context.f, support.anchor]), torch.cat([context.xy, support.xy])


def match(qraw, mapped, sigma, theta):
    """Same Tobit-inspired energy as evaluate_scm_tobit, in centered coordinates."""
    protos, xy = mapped
    det = (qraw != 100).to(protos.dtype)
    x = torch.where(qraw == 100, 0., qraw)
    d2 = x.square().sum(1, keepdim=True) + det @ protos.square().T - 2 * x @ protos.T
    score = -0.5 / sigma**2 * d2 + (1 - det) @ torch.special.log_ndtr((theta - protos) / sigma).T
    return torch.softmax(score, dim=1) @ xy
