"""提供固定Adam配置和跨通信轮持续的无放回Support批流。"""
import numpy as np
import torch


def adam(parameters, lr, config):
    return torch.optim.Adam(parameters, lr=lr, **config["adam"])


class BatchStream:
    def __init__(self, size, batch_size, seed):
        self.size, self.batch_size = size, batch_size
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(size)
        self.cursor = 0
        self.visits = np.zeros(size, dtype=np.int64)

    def next(self):
        if self.cursor == self.size:
            self.order, self.cursor = self.rng.permutation(self.size), 0
        indices = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += len(indices)
        self.visits[indices] += 1
        return indices


def step(model, x, y, optimizers):
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), y)
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    loss.backward()
    for optimizer in optimizers:
        optimizer.step()
    return float(loss.detach())
