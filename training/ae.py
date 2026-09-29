"""在source内部选择AE重建预算，再按固定预算训练各环境AE。"""
import math
from collections import defaultdict

import numpy as np
import torch

from data.uji import Transform, split_groups
from models.femloc import private
from training.common import BatchStream, adam, step


def train_ae(encoder, decoder, x, epochs, lr, seed, config, held=None):
    model = torch.nn.Sequential(encoder, decoder)
    optimizer = adam(model.parameters(), lr, config)
    batches = BatchStream(len(x), config["batch_size"], seed)
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        for _ in range(math.ceil(len(x) / config["batch_size"])):
            indices = batches.next()
            losses.append(step(model, x[indices], x[indices], [optimizer]))
        if held is not None:
            with torch.no_grad():
                mse = float(torch.nn.functional.mse_loss(model(held), held))
        else:
            mse = None
        history.append({"epoch": epoch, "train_batch_mse": float(np.mean(losses)), "held_mse": mse})
    return history, batches.visits


def calibrate(floors, config, seed=0):
    """不读取target：按source留出重建/均值基线误差选择lr和epoch。"""
    prepared = []
    for name, floor in floors.items():
        if name in config["targets"]:
            continue
        s, _ = split_groups(floor["groups"], config["split_folds"], seed)
        a, v = split_groups(floor["groups"][s], config["split_folds"], seed)
        fit, held = s[a], s[v]
        prep = Transform.fit(floor["rssi"][fit], floor["xy"][fit])
        x, z = (prep.x(floor["rssi"][ids], config["device"]) for ids in (fit, held))
        prepared.append((name, x, z, floor["row_ids"][fit], floor["row_ids"][held]))
    max_batches = max(math.ceil(len(x) / config["batch_size"]) for _, x, _, _, _ in prepared)
    update_budget = config["meta"]["rounds"] * config["meta"]["inner_steps"]
    growth = config["ae"]["epoch_growth"]
    epochs = []
    e = 1
    while e * max_batches <= update_budget:
        epochs.append(e)
        e *= growth
    records, scores = [], defaultdict(list)
    for name, x, z, fit_rows, held_rows in prepared:
        zero = float(z.square().mean())
        mean = float((z - x.mean(dim=0)).square().mean())
        for lr in config["ae"]["candidate_lrs"]:
            encoder, decoder, _ = private(x.shape[1], config, seed)
            history, _ = train_ae(encoder, decoder, x, max(epochs), lr, seed, config, z)
            for epoch in epochs:
                mse = history[epoch - 1]["held_mse"]
                scores[(lr, epoch)].append(mse / mean)
                records.append({"floor": name, "lr": lr, "epoch": epoch, "held_mse": mse,
                                "zero_mse": zero, "mean_mse": mean, "relative_to_mean": mse / mean})
            print(f"AE calibration {name} lr={lr} complete", flush=True)
    selected = min(scores, key=lambda pair: (np.mean(scores[pair]), pair[1], pair[0]))
    return {"lr": selected[0], "epochs": selected[1], "selection_score": float(np.mean(scores[selected])),
            "criterion": "mean source-floor heldout reconstruction MSE / per-AP-mean baseline MSE",
            "epoch_candidates": epochs, "source_only": True,
            "row_ids": {name: {"fit": f.tolist(), "held": v.tolist()} for name, _, _, f, v in prepared},
            "records": records}
