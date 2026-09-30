"""共用源 episode，分别训练闭式适应表征和普通多楼层监督表征。"""
from copy import deepcopy
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.episodes import episode_tensors, sample_episode
from models.ridge_meta import encoder, ridge_predict


def source_tasks(floors, names, rng, config):
    """每源 episode 含 10×3 标定扫描与另外 10×5 定位扫描。"""
    tasks = []
    for name in rng.choice(names, config["tasks_per_iteration"], replace=False):
        floor = floors[name]
        support, query = sample_episode(floor, rng, config["positions_per_task"],
                                        config["support_per_position"], config["query_per_position"],
                                        unseen_query=True)
        tensors = episode_tensors(floor, support, query, config["coordinate_scale_m"])
        tasks.append((name, support, query, *tensors))
    return tasks


def train_encoder(floors, names, config, seed, method, penalty, checkpoints, progress_path):
    """Meta 的 Query 损失穿过求解器；Sup 用每楼层固定坐标原点和私有线性头。"""
    torch.manual_seed(seed)
    model = encoder(config["hidden"]).to(config["device"])
    parameters = list(model.parameters())
    if method == "Sup-Ridge":
        heads = nn.ModuleDict({name: nn.Linear(config["hidden"][-1], 2) for name in names}).to(config["device"])
        parameters += list(heads.parameters())
        origins = {name: floors[name]["xy"].mean(0) for name in names}
    optimizer = torch.optim.Adam(parameters, lr=config["outer_lr"])
    rng = np.random.default_rng(seed)
    sampling_hash = hashlib.sha256()
    coverage = {name: set() for name in names}
    snapshots = {0: deepcopy(model.state_dict())} if 0 in checkpoints else {}
    with progress_path.open("w") as stream:
        for step in range(1, max(checkpoints) + 1):
            tasks = source_tasks(floors, names, rng, config)
            optimizer.zero_grad()
            losses = []
            for name, support, query, sx, sy, qx, qy in tasks:
                rows = np.concatenate([support, query])
                coverage[name].update(rows.tolist())
                sampling_hash.update(name.encode())
                sampling_hash.update(floors[name]["row_ids"][rows].astype("<i8").tobytes())
                if method == "Meta-Ridge":
                    prediction = ridge_predict(model, sx, sy, qx, penalty)
                    losses.append(F.mse_loss(prediction, qy))
                else:
                    labels = ((floors[name]["xy"][rows] - origins[name]) / config["coordinate_scale_m"]).float()
                    losses.append(F.mse_loss(heads[name](model(torch.cat([sx, qx]))), labels))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()
            if step in checkpoints:
                snapshots[step] = deepcopy(model.state_dict())
            if step == 1 or step % 100 == 0 or step == max(checkpoints):
                record = {"method": method, "seed": seed, "lambda": penalty,
                          "step": step, "source_mse": float(loss.item()),
                          "sampling_sha256": sampling_hash.hexdigest()}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                print("source", record, flush=True)
    progress_path.with_suffix(".coverage.json").write_text(json.dumps({
        "source_floors": names, "source_steps": max(checkpoints),
        "sampling_sha256": sampling_hash.hexdigest(),
        "unique_rows_seen": {name: len(rows) for name, rows in coverage.items()}
    }, indent=2) + "\n")
    return model, snapshots
