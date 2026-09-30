"""检查 AP 共同置换不变性、30 条锚点接口和定位梯度。"""
import json
from pathlib import Path

import numpy as np
import torch

from models.learned_ap_map import APReliability, build, cells


cfg = json.loads(Path('configs/learned_ap_map.json').read_text())
cfg['device'] = 'cpu'
torch.manual_seed(0)
rng = np.random.default_rng(1)
xy = np.c_[np.repeat(np.arange(12), 4) * 5.0, np.zeros(48)]
raw = np.full((48, 520), 100, dtype=np.int16)
raw[:, :8] = rng.integers(-90, -40, (48, 8))
raw[::3, 3:6] = 100
support = np.concatenate([np.arange(i * 4, i * 4 + 3) for i in range(10)])
query = np.arange(40, 48)
model = APReliability(cfg)
mapped = build(cells(raw, xy), raw[support], xy[support], cfg, 'cpu')
prediction = model(raw[query], mapped)
loss = prediction.square().mean()
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
perm = rng.permutation(520)
other = build(cells(raw[:, perm], xy), raw[support][:, perm], xy[support], cfg, 'cpu')
assert np.allclose(model.predict(raw[query], mapped), model.predict(raw[query][:, perm], other), atol=1e-4)
assert torch.isfinite(prediction).all() and prediction.shape == (8, 2)
assert sum(p.numel() for p in model.parameters()) == 163
print('AP permutation, gradient, finite coordinates and 163 parameters: passed')
