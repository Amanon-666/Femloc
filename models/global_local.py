"""用全局线性关系和局部无线相似性共同定位新楼层。"""

import torch


def predict(support_features, support_xy, query_features, relative_penalty):
    """仅由 Support 求映射；线性部分允许向锚点覆盖区域外推断。"""
    center_x = support_features.mean(dim=0)
    center_y = support_xy.mean(dim=0)
    support = support_features - center_x
    query = query_features - center_x

    linear = support @ support.T
    linear_query = query @ support.T
    linear_scale = linear.diag().mean().clamp_min(1e-8)
    linear = linear / linear_scale
    linear_query = linear_query / linear_scale

    # 用 Support 自身的典型特征距离定尺度，不读取 Query 坐标。
    bandwidth = torch.pdist(support_features).square().median().detach().clamp_min(1e-8)
    radial = torch.exp(-torch.cdist(support_features, support_features).square() / bandwidth)
    radial_query = torch.exp(-torch.cdist(query_features, support_features).square() / bandwidth)
    row_mean = radial.mean(dim=1, keepdim=True)
    column_mean = radial.mean(dim=0, keepdim=True)
    grand_mean = radial.mean()
    radial_query = radial_query - radial_query.mean(dim=1, keepdim=True) - column_mean + grand_mean
    radial = radial - row_mean - column_mean + grand_mean
    radial_scale = radial.diag().mean().clamp_min(1e-8)
    radial = radial / radial_scale
    radial_query = radial_query / radial_scale

    gram = (linear + radial) / 2
    cross = (linear_query + radial_query) / 2
    penalty = relative_penalty * gram.diag().mean().clamp_min(1e-8)
    coefficients = torch.linalg.solve(
        gram + penalty * torch.eye(len(support), device=support.device, dtype=support.dtype),
        support_xy - center_y,
    )
    return center_y + cross @ coefficients


if __name__ == "__main__":
    features = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    coordinates = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    query = torch.tensor([[0.5, 0.5], [1.5, 0.5]])
    shift = torch.tensor([100., -50.])
    base = predict(features, coordinates, query, .1)
    torch.testing.assert_close(predict(features, coordinates + shift, query, .1), base + shift)
    assert torch.isfinite(base).all()
