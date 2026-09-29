"""Masked MAB, SAB and PMA adapted from the official Set Transformer MIT code."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class MAB(nn.Module):
    def __init__(self, dim_q: int, dim_k: int, dim_v: int, heads: int):
        super().__init__()
        self.dim_v = dim_v
        self.heads = heads
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        self.fc_o = nn.Linear(dim_v, dim_v)

    def forward(self, query: torch.Tensor, key: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(key)
        split = self.dim_v // self.heads
        qh = torch.cat(q.split(split, dim=2), dim=0)
        kh = torch.cat(k.split(split, dim=2), dim=0)
        vh = torch.cat(v.split(split, dim=2), dim=0)
        valid = key_mask.repeat(self.heads, 1).unsqueeze(1)
        attention = torch.softmax((qh.bmm(kh.transpose(1, 2)) / math.sqrt(self.dim_v)).masked_fill(~valid, -1e9), dim=2)
        output = torch.cat((qh + attention.bmm(vh)).split(q.size(0), dim=0), dim=2)
        return output + F.relu(self.fc_o(output))


class SAB(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.mab = MAB(dim, dim, dim, heads)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mab(values, values, mask)


class PMA(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.xavier_uniform_(self.seed)
        self.mab = MAB(dim, dim, dim, heads)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mab(self.seed.repeat(values.size(0), 1, 1), values, mask)
