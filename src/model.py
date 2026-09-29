"""Matched dense and AP-set encoders with a two-coordinate regression head."""

import torch
from torch import nn

from src.set_blocks import PMA, SAB


class DenseEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(520, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.width = 128

    def forward(self, batch: dict) -> torch.Tensor:
        return self.network(batch["dense"])


class SetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(521, 32)
        self.input = nn.Linear(33, 64)
        self.blocks = nn.ModuleList([SAB(64, 4), SAB(64, 4)])
        self.pool = PMA(64, 4)
        self.width = 64

    def forward(self, batch: dict) -> torch.Tensor:
        embedded = self.embedding(batch["ap_ids"])
        values = torch.cat((embedded, batch["ap_values"].unsqueeze(-1)), dim=-1)
        values = self.input(values)
        for block in self.blocks:
            values = block(values, batch["ap_mask"])
        return self.pool(values, batch["ap_mask"]).squeeze(1)


class Locator(nn.Module):
    def __init__(self, kind: str):
        super().__init__()
        self.featurizer = DenseEncoder() if kind == "mlp" else SetEncoder()
        self.head = nn.Linear(self.featurizer.width, 2)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.featurizer(batch))
