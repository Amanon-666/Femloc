"""用目标楼层少量已知坐标直接求相对位置映射。"""

import torch


def predict(support_features, support_xy, query_features, relative_penalty):
    """在 Support 上定中心、求岭回归，再预测 Query 相对坐标。"""
    center_x = support_features.mean(dim=0)
    center_y = support_xy.mean(dim=0)
    x = support_features - center_x
    gram = x @ x.T
    penalty = relative_penalty * gram.diag().mean().clamp_min(1e-8)
    coefficients = torch.linalg.solve(
        gram + penalty * torch.eye(len(x), device=x.device, dtype=x.dtype),
        support_xy - center_y,
    )
    return center_y + (query_features - center_x) @ x.T @ coefficients


if __name__ == "__main__":
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    q = torch.tensor([[.5, .5]])
    y = torch.tensor([[1., 3.], [2., 4.], [3., 5.]])
    shift = torch.tensor([100., -50.])
    torch.testing.assert_close(predict(x, y + shift, q, .1), predict(x, y, q, .1) + shift)
