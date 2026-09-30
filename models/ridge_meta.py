"""用 Support 闭式拟合坐标映射，训练编码器供新楼层快速适应。"""
import torch
from torch import nn


def encoder(hidden):
    widths = [520, *hidden]
    layers = []
    for a, b in zip(widths, widths[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers)


def ridge_predict(model, support_x, support_y, query_x, penalty):
    """仅由 Support 特征及坐标决定线性解；Query 只进入预测。"""
    return ridge_solve(model(support_x), support_y, model(query_x), penalty)


def ridge_solve(support_features, support_y, query_features, penalty):
    """中心化、全特征统一 RMS 缩放、带非惩罚截距的岭回归。"""
    return apply_ridge(query_features, fit_ridge(support_features, support_y, penalty))


def fit_ridge(support_features, support_y, penalty):
    """Support 一次求解；返回后续 Query 共用的中心、缩放和系数。"""
    feature_mean = support_features.mean(0, keepdim=True)
    feature_scale = ((support_features - feature_mean).square().mean() + 1e-6).sqrt()
    support_features = (support_features - feature_mean) / feature_scale
    width = support_features.shape[1]
    gram = support_features @ support_features.T / width
    coordinate_mean = support_y.mean(0, keepdim=True)
    coefficients = torch.linalg.solve(
        gram + penalty * torch.eye(len(support_features), device=gram.device, dtype=gram.dtype),
        support_y - coordinate_mean,
    )
    return feature_mean, feature_scale, support_features, coordinate_mean, coefficients


def apply_ridge(query_features, state):
    """Query 不影响 Support 解，可逐条或整批预测。"""
    feature_mean, feature_scale, support_features, coordinate_mean, coefficients = state
    query_features = (query_features - feature_mean) / feature_scale
    return coordinate_mean + (query_features @ support_features.T / support_features.shape[1]) @ coefficients
