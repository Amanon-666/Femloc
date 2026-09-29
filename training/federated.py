"""执行FeMLoc局部适应及按完整Query规模加权的一阶全局更新。"""
from copy import deepcopy

import numpy as np
import torch

from data.uji import Transform
from models.femloc import Localizer, private
from training.ae import train_ae
from training.common import BatchStream, adam, step


class Client:
    """持有一个source环境的私有参数、优化器和连续Support批流。"""
    def __init__(self, name, floor, support, query, global_model, ae_config, config, seed):
        self.name, self.support, self.query = name, support, query
        self.prep = Transform.fit(floor["rssi"][support], floor["xy"][support])
        self.sx = self.prep.x(floor["rssi"][support], config["device"])
        self.sy = self.prep.y(floor["xy"][support], config["device"])
        self.qx = self.prep.x(floor["rssi"][query], config["device"])
        self.qy = self.prep.y(floor["xy"][query], config["device"])
        encoder, decoder, mapper = private(self.sx.shape[1], config, seed)
        self.ae_history, self.ae_visits = train_ae(encoder, decoder, self.sx, ae_config["epochs"],
                                                  ae_config["lr"], seed, config)
        self.model = Localizer(encoder, deepcopy(global_model), mapper)
        self.optimizer = adam([
            {"params": encoder.parameters(), "lr": config["meta"]["encoder_lr"]},
            {"params": mapper.parameters(), "lr": config["meta"]["mapper_lr"]}
        ], config["meta"]["encoder_lr"], config)
        self.batches = BatchStream(len(support), config["batch_size"], seed)
        self.query_visits = np.zeros(len(query), dtype=np.int64)


def meta_round(global_model, clients, config):
    gradients = [torch.zeros_like(p) for p in global_model.parameters()]
    query_count = sum(len(client.query) for client in clients)
    records = []
    for client in clients:
        model = client.model
        model.shared.load_state_dict(global_model.state_dict())
        shared_optimizer = adam(model.shared.parameters(), config["meta"]["shared_lr"], config)
        losses = []
        for _ in range(config["meta"]["inner_steps"]):
            ids = client.batches.next()
            losses.append(step(model, client.sx[ids], client.sy[ids], [client.optimizer, shared_optimizer]))
        query_loss = 0.0
        for start in range(0, len(client.query), config["batch_size"]):
            end = min(start + config["batch_size"], len(client.query))
            loss = torch.nn.functional.mse_loss(model(client.qx[start:end]), client.qy[start:end])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite source query loss: {client.name}")
            grads = torch.autograd.grad(loss, tuple(model.shared.parameters()))
            for accumulator, grad in zip(gradients, grads):
                accumulator.add_(grad.detach(), alpha=(end - start) / query_count)
            query_loss += float(loss.detach()) * (end - start) / len(client.query)
        client.query_visits += 1
        records.append({"floor": client.name, "support_mse": float(np.mean(losses)), "query_mse": query_loss,
                        "query_weight": len(client.query) / query_count})
    norm = torch.sqrt(sum(g.square().sum() for g in gradients))
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite global gradient")
    with torch.no_grad():
        for parameter, grad in zip(global_model.parameters(), gradients):
            parameter.add_(grad, alpha=-config["meta"]["outer_lr"])
    return {"clients": records, "global_update_norm": float(norm) * config["meta"]["outer_lr"]}
