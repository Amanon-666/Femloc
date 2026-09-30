"""固定位置划分、配对定位误差与少量标定/完整地图两类对照。"""
import json
from statistics import mean, stdev
import time

import numpy as np
import torch

from data.episodes import episode_tensors, spatial_support_split
from models.ridge_meta import apply_ridge, fit_ridge, ridge_predict


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def make_splits(floors, names, count, config, offset):
    """一次生成具体行索引；所有算法读取同一个划分。"""
    splits = {}
    for floor_index, name in enumerate(names):
        for split in range(count):
            rng = np.random.default_rng(config["split_seed"] + offset + 1000 * floor_index + split)
            splits[f"{name}/{split}"] = spatial_support_split(
                floors[name], rng, config["positions_per_task"], config["support_per_position"])
    return splits


def manifest(floors, splits):
    return {key: {"support": floors[key.split("/")[0]]["row_ids"][support].tolist(),
                  "unseen_query": floors[key.split("/")[0]]["row_ids"][query].tolist()}
            for key, (support, query) in splits.items()}


@torch.no_grad()
def development_mde(model, floors, splits, config, penalty):
    by_floor = {name: [] for name in config["development_floors"]}
    for key, (support, query) in splits.items():
        name = key.split("/")[0]
        sx, sy, qx, qy = episode_tensors(floors[name], support, query, config["coordinate_scale_m"])
        prediction = ridge_predict(model, sx, sy, qx, penalty)
        by_floor[name].append(float(torch.linalg.vector_norm(prediction - qy, dim=1).mean().item()
                                    * config["coordinate_scale_m"]))
    return {name: mean(values) for name, values in by_floor.items()}


def support_knn(sx, sy, qx, k):
    distance, index = torch.cdist(qx, sx).topk(min(k, len(sx)), largest=False)
    weight = 1 / (distance + 1e-8)
    return ((weight / weight.sum(1, keepdim=True))[:, :, None] * sy[index]).sum(1)


def full_map(floor):
    """源楼层每个位置的唯一观测均值与原始坐标。"""
    xy = list(floor["positions"])
    prototypes = torch.stack([floor["x"][floor["positions"][pos]].mean(0) for pos in xy])
    return prototypes, torch.tensor(xy, dtype=torch.float64, device=prototypes.device)


def map_knn(qx, radio_map, k, beta):
    prototypes, xy = radio_map
    # 保留旧地图的平方 RSSI 距离、前 k 候选 softmax 定位规则。
    distance = (qx.square().sum(1)[:, None] + prototypes.square().sum(1)[None]
                - 2 * qx @ prototypes.T).clamp_min(0)
    distance, index = distance.topk(min(k, len(prototypes)), largest=False)
    weight = torch.softmax(-beta * (distance - distance[:, :1]), dim=1).double()
    return (weight[:, :, None] * xy[index]).sum(1)


@torch.no_grad()
def assess(model, method, floor, validation, support, query, config, penalty, radio_map):
    """适应函数只接 Support；真值在预测后才用于误差计算。"""
    origin = floor["xy"][support].mean(0)
    sx = floor["x"][support]
    sy = ((floor["xy"][support] - origin) / config["coordinate_scale_m"]).float()
    is_ridge = method in ("Meta-Ridge", "Sup-Ridge", "RI-Ridge", "Raw-Ridge")
    if is_ridge:
        torch.cuda.synchronize()
        start = time.perf_counter()
        state = fit_ridge(model(sx), sy, penalty)
        torch.cuda.synchronize()
        adaptation_ms = (time.perf_counter() - start) * 1000
    else:
        adaptation_ms = None
    results = []
    for scope, data, rows in (("unseen_position", floor, query),
                              ("official_validation", validation, np.arange(len(validation["x"]))),
                              ("support_fit", floor, support)):
        qx = data["x"][rows]
        if is_ridge:
            prediction = apply_ridge(model(qx), state).double() * config["coordinate_scale_m"] + origin
        elif method == "Support-WKNN":
            prediction = support_knn(sx, sy, qx, config["support_wknn_k"]).double() * config["coordinate_scale_m"] + origin
        else:
            prediction = map_knn(qx, radio_map, config["map_wknn_k"], config["map_wknn_beta"])
        errors = torch.linalg.vector_norm(prediction - data["xy"][rows], dim=1).cpu().numpy()
        results.append({"scope": scope, "n_query": len(rows), "mde_m": float(errors.mean()),
                        "median_m": float(np.median(errors)), "p90_m": float(np.quantile(errors, .9)),
                        "adaptation_ms": adaptation_ms, "target_gradient_steps": 0})
    return results


def summarize(results, config, selection):
    """先对 3 个训练种子求均值，再报告 5 个标定划分的均值/样本标准差。"""
    rows = []
    for name in config["targets"]:
        for scope in ("unseen_position", "official_validation"):
            values = {}
            methods = sorted({r["method"] for r in results})
            for method in methods:
                values[method] = [mean(r["mde_m"] for r in results if r["target"] == name
                                       and r["scope"] == scope and r["method"] == method and r["split"] == split)
                                  for split in range(config["final_splits"])]
                rows.append({"target": name, "scope": scope, "method": method,
                             "mean_mde_m": mean(values[method]), "support_split_sd_m": stdev(values[method])})
            for control in ("Sup-Ridge", "RI-Ridge", "Raw-Ridge", "Support-WKNN", "Old-Map"):
                delta = [a-b for a, b in zip(values["Meta-Ridge"], values[control])]
                rows.append({"target": name, "scope": scope, "comparison": f"Meta-Ridge - {control}",
                             "mean_delta_m": mean(delta), "support_split_sd_m": stdev(delta)})
    return {"selection": selection["selected"], "results": rows}
