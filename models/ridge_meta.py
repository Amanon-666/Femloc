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
    support_features = model(support_x)
    feature_mean = support_features.mean(0, keepdim=True)
    feature_scale = ((support_features - feature_mean).square().mean() + 1e-6).sqrt()
    support_features = (support_features - feature_mean) / feature_scale
    query_features = (model(query_x) - feature_mean) / feature_scale
    width = support_features.shape[1]
    gram = support_features @ support_features.T / width
    coordinate_mean = support_y.mean(0, keepdim=True)
    coefficients = torch.linalg.solve(
        gram + penalty * torch.eye(len(support_x), device=gram.device),
        support_y - coordinate_mean,
    )
    return coordinate_mean + (query_features @ support_features.T / width) @ coefficients
