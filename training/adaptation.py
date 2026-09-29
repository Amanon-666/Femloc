"""配对MI/RI目标适应；Query只传RSSI进行冻结推理，不接收标签。"""
from copy import deepcopy

import numpy as np
import torch

from models.femloc import Localizer
from training.common import BatchStream, adam, step


def adapt(encoder, shared_model, mapper, sx, sy, query_x, config, seed):
    model = Localizer(deepcopy(encoder), deepcopy(shared_model), deepcopy(mapper))
    optimizer = adam([
        {"params": model.encoder.parameters(), "lr": config["meta"]["encoder_lr"]},
        {"params": model.shared.parameters(), "lr": config["meta"]["shared_lr"]},
        {"params": model.mapper.parameters(), "lr": config["meta"]["mapper_lr"]}
    ], config["meta"]["shared_lr"], config)
    batches = BatchStream(len(sx), config["batch_size"], seed)
    predictions, losses = [], []
    for index in range(config["adapt"]["steps"] + 1):
        if index:
            ids = batches.next()
            losses.append(step(model, sx[ids], sy[ids], [optimizer]))
        with torch.no_grad():
            prediction = model(query_x)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite target prediction")
            predictions.append(prediction.cpu().numpy())
    return np.stack(predictions), losses, batches.visits, model
