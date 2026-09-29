"""用旧楼层地图作先验，闭式求解新楼层 Support 的坐标残差。"""
import torch
from torch import nn


def encoder():
    return nn.Sequential(nn.Linear(522, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())


def corrected(model, support_x, support_base, support_y, query_x, query_base, penalty):
    """只用 Support 真坐标决定残差解，Query 标签始终不进入适应。"""
    support_feature = model(torch.cat((support_x, support_base), dim=1))
    feature_mean = support_feature.mean(0, keepdim=True)
    feature_scale = ((support_feature - feature_mean).square().mean() + 1e-6).sqrt()
    support_feature = (support_feature - feature_mean) / feature_scale
    query_feature = (model(torch.cat((query_x, query_base), dim=1)) - feature_mean) / feature_scale
    width = support_feature.shape[1]
    kernel = 1 + support_feature @ support_feature.T / width
    cross = 1 + query_feature @ support_feature.T / width
    residual = support_y - support_base
    coefficients = torch.linalg.solve(
        kernel + penalty * torch.eye(len(support_x), device=kernel.device), residual)
    return query_base + cross @ coefficients
