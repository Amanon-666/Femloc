"""共享 RSSI 回归网络及可微的 MetaLoc 内层更新。"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call


def network(hidden):
    widths = [520, *hidden, 2]
    layers = []
    for a, b in zip(widths, widths[1:]):
        layers.append(nn.Linear(a, b))
        if b != 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def adapted_parameters(model, sx, sy, steps, lr, second_order):
    """Support 梯度步；源训练保留二阶图，目标测试不保留。"""
    weights = OrderedDict(model.named_parameters())
    for _ in range(steps):
        loss = F.mse_loss(functional_call(model, weights, (sx,)), sy)
        grad = torch.autograd.grad(loss, tuple(weights.values()), create_graph=second_order)
        weights = OrderedDict((name, value - lr * g)
                              for (name, value), g in zip(weights.items(), grad))
        if not second_order:
            weights = OrderedDict((name, value.detach().requires_grad_())
                                  for name, value in weights.items())
    return weights


def predictions_after_steps(model, sx, sy, query_x, steps, lr):
    """目标端仅输入 Support 标签；返回第 0 至最后一步的 Query 预测。"""
    weights = OrderedDict(model.named_parameters())
    predictions = []
    for step in range(steps + 1):
        with torch.no_grad():
            predictions.append(functional_call(model, weights, (query_x,)).cpu().numpy())
        if step < steps:
            loss = F.mse_loss(functional_call(model, weights, (sx,)), sy)
            grad = torch.autograd.grad(loss, tuple(weights.values()))
            weights = OrderedDict((name, (value - lr * g).detach().requires_grad_())
                                  for (name, value), g in zip(weights.items(), grad))
    return predictions
