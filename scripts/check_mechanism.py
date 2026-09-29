"""只在source环境检查分组权限、批遍历、全局梯度步和配对适应。"""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from data.uji import Transform, load_floors, split_groups
from models.femloc import private, shared
from training.adaptation import adapt
from training.common import BatchStream
from training.federated import Client, meta_round


def main():
    config = json.loads(Path("configs/exp1.json").read_text())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    floors = load_floors(config["data_path"])
    for floor in floors.values():
        s, q = split_groups(floor["groups"], config["split_folds"], 0)
        assert set(floor["groups"][s]).isdisjoint(floor["groups"][q])
        assert len(s) + len(q) == len(floor["row_ids"])
    stream = BatchStream(65, config["batch_size"], 0)
    assert sorted(np.concatenate([stream.next() for _ in range(3)]).tolist()) == list(range(65))
    global_model = shared(config, 0)
    clients = [Client(name, floors[name], *split_groups(floors[name]["groups"], config["split_folds"], 0),
                      global_model, {"epochs": 1, "lr": config["ae"]["candidate_lrs"][0]}, config, 0)
               for name in ("B0F0", "B1F0")]
    initial = [p.detach().clone() for p in global_model.parameters()]
    records = meta_round(global_model, clients, config)
    expected = [torch.zeros_like(p) for p in global_model.parameters()]
    total = sum(len(c.query) for c in clients)
    for c in clients:
        loss = torch.nn.functional.mse_loss(c.model(c.qx), c.qy)
        gradients = torch.autograd.grad(loss, tuple(c.model.shared.parameters()))
        for accumulator, grad in zip(expected, gradients):
            accumulator.add_(grad, alpha=len(c.query) / total)
    for before, after, grad in zip(initial, global_model.parameters(), expected):
        torch.testing.assert_close(after, before - config["meta"]["outer_lr"] * grad)
    assert records["global_update_norm"] > 0
    dev = floors["B2F0"]
    s, q = split_groups(dev["groups"], config["split_folds"], 0)
    prep = Transform.fit(dev["rssi"][s], dev["xy"][s])
    sx, qx = [prep.x(dev["rssi"][ids], config["device"]) for ids in (s, q)]
    sy = prep.y(dev["xy"][s], config["device"])
    e, _, m = private(sx.shape[1], config, 0)
    config["adapt"]["steps"] = config["meta"]["inner_steps"]
    saved = deepcopy(e.state_dict())
    first = adapt(e, global_model, m, sx, sy, qx, config, 0)
    second = adapt(e, global_model, m, sx, sy, qx, config, 0)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[2], second[2])
    for key, value in e.state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    assert first[0].shape == (config["adapt"]["steps"] + 1, len(q), 2)
    result = {"status": "passed", "device": config["device"], "source_clients": [c.name for c in clients],
              "adaptation_development_floor": "B2F0", "target_query_used": False,
              "checks": ["duplicate groups isolated", "batch visits cover all rows", "Eq.11 weighted gradient matches", "identical paired runs match", "private initial weights unchanged", "prediction shape"],
              "global_update_norm": records["global_update_norm"]}
    out = Path("outputs/mechanism_check.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
