"""检查现有 RSSI 特征能否由少量新楼层锚点映射到坐标。"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist

from data.episodes import prepare_floor
from data.uji import load_floors
from models.rss_maml import network


def ridge_predict(support_features, support_xy, query_features, strength):
    """支持集定中心的岭回归；输出随坐标整体平移而同步平移。"""
    feature_center = support_features.mean(axis=0)
    location_center = support_xy.mean(axis=0)
    x = support_features - feature_center
    kernel = x @ x.T
    penalty = strength * np.trace(kernel) / len(x)
    weights = np.linalg.solve(kernel + penalty * np.eye(len(x)), support_xy - location_center)
    return location_center + (query_features - feature_center) @ x.T @ weights


def choose_strength(features, xy):
    """仅在给出的十个位置内部做整位置留一验证。"""
    positions, group = np.unique(xy, axis=0, return_inverse=True)
    candidates = (0.001, 0.01, 0.1, 1.0, 10.0)
    scores = []
    for strength in candidates:
        errors = []
        for position_id in range(len(positions)):
            train = group != position_id
            held = ~train
            predicted = ridge_predict(features[train], xy[train], features[held], strength)
            errors.append(np.linalg.norm(predicted - xy[held], axis=1).mean())
        scores.append(float(np.mean(errors)))
    best = int(np.argmin(scores))
    return candidates[best], scores[best]


def wknn(support_rss, support_xy, query_rss):
    """与目标端其它方法共享三十条标注扫描的三近邻参照。"""
    distances = cdist(query_rss, support_rss)
    nearest = np.argpartition(distances, 3, axis=1)[:, :3]
    weight = 1 / np.maximum(np.take_along_axis(distances, nearest, axis=1), 1e-8)
    weight /= weight.sum(axis=1, keepdims=True)
    return (support_xy[nearest] * weight[..., None]).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    floors = {name: prepare_floor(raw, config["device"])
              for name, raw in load_floors(config["data_path"]).items()}
    records = []
    for seed in config["seeds"]:
        source = args.checkpoints / f"seed_{seed}"
        saved = torch.load(source / "source_models.pt", map_location="cpu", weights_only=True)
        manifest = json.loads((source / "manifest.json").read_text())
        for method in ("TL", "MI"):
            model = network(config["hidden"]).to(config["device"])
            model.load_state_dict(saved[method])
            model.eval()
            for target in config["targets"]:
                floor = floors[target]
                lookup = {int(row): i for i, row in enumerate(floor["row_ids"])}
                support = np.array([lookup[row] for row in manifest[target]["support"]])
                query = np.array([lookup[row] for row in manifest[target]["unseen_position_query"]])
                assert set(floor["position_keys"][i] for i in support).isdisjoint(
                    floor["position_keys"][i] for i in query)
                with torch.no_grad():
                    features = model[:-1](floor["x"]).cpu().numpy().astype(np.float64)
                    base = model(floor["x"]).cpu().numpy().astype(np.float64)
                xy = floor["xy"].cpu().numpy()
                sx, sy = features[support], xy[support]
                qx, qy = features[query], xy[query]
                strength, loo_mde = choose_strength(sx, sy)
                ridge = ridge_predict(sx, sy, qx, strength)
                ridge_support = ridge_predict(sx, sy, sx, strength)
                offset = sy.mean(axis=0) + config["coordinate_scale_m"] * (
                    base[query] - base[support].mean(axis=0))
                nearest = wknn(floor["x"][support].cpu().numpy(), sy,
                               floor["x"][query].cpu().numpy())
                error = lambda prediction, truth: float(np.linalg.norm(prediction - truth, axis=1).mean())
                records.append({"seed": seed, "target": target, "source_model": method,
                                "support_rows": len(support), "query_rows": len(query),
                                "ridge_strength": strength, "support_loo_mde": loo_mde,
                                "ridge_support_mde": error(ridge_support, sy),
                                "ridge_query_mde": error(ridge, qy),
                                "offset_query_mde": error(offset, qy),
                                "wknn_query_mde": error(nearest, qy)})
                print(records[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
