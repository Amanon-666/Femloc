"""官方DKT回归机制的二维UJI适配：共享特征、两个独立标量GP。"""
import gpytorch
import torch
from torch import nn

from .upstream_classes import ExactGPLayer, Feature


class DKT(nn.Module):
    def __init__(self, learned_features=True):
        super().__init__()
        if learned_features:
            self.feature_extractor = Feature()
            self.feature_extractor.layer1 = nn.Linear(520, 40)
            width = 40
        else:
            self.feature_extractor = nn.Identity()
            width = 520
        self.gps = nn.ModuleList([
            ExactGPLayer(torch.zeros(1, width), torch.zeros(1),
                         gpytorch.likelihoods.GaussianLikelihood(), kernel='rbf')
            for _ in range(2)
        ])

    def loss(self, x, y):
        self.train()
        z = self.feature_extractor(x)
        losses = []
        for axis, gp in enumerate(self.gps):
            gp.set_train_data(inputs=z, targets=y[:, axis], strict=False)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
            losses.append(-mll(gp(z), gp.train_targets))
        return torch.stack(losses).mean()

    @torch.no_grad()
    def predict(self, sx, sy, qx):
        self.eval()
        support = self.feature_extractor(sx)
        query = self.feature_extractor(qx)
        means, variances = [], []
        for axis, gp in enumerate(self.gps):
            # 与官方sines测试流程一致，清除上一任务的预测缓存后绑定新Support。
            gp.train()
            gp.set_train_data(inputs=support, targets=sy[:, axis], strict=False)
            gp.eval()
            prediction = gp.likelihood(gp(query))
            means.append(prediction.mean)
            variances.append(prediction.variance)
        return torch.stack(means, -1), torch.stack(variances, -1)
