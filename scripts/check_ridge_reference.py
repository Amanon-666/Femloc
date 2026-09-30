"""实际检查岭回归梯度、等价解、坐标平移及源/目标位置隔离。"""
import json
from pathlib import Path

import numpy as np
import torch

from data.episodes import episode_tensors, prepare_floor, sample_episode, spatial_support_split
from data.uji import load_floors
from models.ridge_meta import encoder, ridge_predict, ridge_solve


def main():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    sx = torch.randn(8, 5, dtype=torch.float64, requires_grad=True)
    qx = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    sy = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)
    penalty = .1
    assert torch.autograd.gradcheck(lambda a, b, c: ridge_solve(a, b, c, penalty), (sx, sy, qx))
    center, scale = sx.mean(0), ((sx - sx.mean(0)).square().mean() + 1e-6).sqrt()
    z, zq = (sx - center) / scale / np.sqrt(5), (qx - center) / scale / np.sqrt(5)
    primal = torch.linalg.solve(z.T @ z + penalty * torch.eye(5), z.T @ (sy - sy.mean(0)))
    prediction = ridge_solve(sx, sy, qx, penalty)
    torch.testing.assert_close(prediction, sy.mean(0) + zq @ primal)
    shift = torch.tensor([100., -50.], dtype=torch.float64)
    torch.testing.assert_close(ridge_solve(sx, sy + shift, qx, penalty), prediction + shift)
    torch.testing.assert_close(ridge_solve(sx, sy, qx[:1], penalty), prediction[:1])

    config = json.loads(Path("configs/ridge_reference.json").read_text())
    raw = load_floors(config["data_path"])
    eligibility = {}
    for name, data in raw.items():
        floor = prepare_floor(data, config["device"])
        counts = [len(rows) for rows in floor["positions"].values()]
        eligibility[name] = {"positions": len(counts), "support_ge3": sum(n >= 3 for n in counts),
                             "query_ge5": sum(n >= 5 for n in counts), "legacy_ge8": sum(n >= 8 for n in counts)}
        rng = np.random.default_rng(7)
        support, query = sample_episode(floor, rng, 10, 3, 5, unseen_query=True)
        assert len(support) == 30 and len(query) == 50
        assert not ({floor["position_keys"][i] for i in support} & {floor["position_keys"][i] for i in query})
        target_support, target_query = spatial_support_split(floor, rng, 10, 3)
        assert len(target_support) == 30
        assert not ({floor["position_keys"][i] for i in target_support} & {floor["position_keys"][i] for i in target_query})
    floor = prepare_floor(raw["B0F0"], config["device"])
    support, query = sample_episode(floor, np.random.default_rng(7), 10, 3, 5, unseen_query=True)
    model = encoder(config["hidden"]).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["outer_lr"])
    sx, sy, qx, qy = episode_tensors(floor, support, query, config["coordinate_scale_m"])
    loss = (ridge_predict(model, sx, sy, qx, .1) - qy).square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    print(json.dumps({"dual_equals_primal": True, "gradcheck": True, "translation_equivariant": True,
                      "query_batch_independent": True, "position_disjoint": True,
                      "actual_source_step_mse": float(loss.item()), "eligibility": eligibility}, indent=2))


if __name__ == "__main__":
    main()
