"""在旧楼层转移上学习 RSSI 距离，目标楼层只使用固定的十个坐标锚点。"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.uji import load_floors
from scripts.evaluate_adjacent_map import (correction, episode, make_map,
                                           rss_features, source_pairs,
                                           source_predictions)


class CrossFloorMetric(nn.Module):
    """用少量共享 AP 权重将目标扫描软匹配到旧楼层坐标图。"""

    def __init__(self):
        super().__init__()
        self.ap_logits = nn.Parameter(torch.zeros(520))
        self.log_beta = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, scans, fingerprints, centered_positions):
        weights = 2 * torch.sigmoid(self.ap_logits)
        left = (scans.square() * weights).sum(dim=1, keepdim=True)
        right = (fingerprints.square() * weights).sum(dim=1).unsqueeze(0)
        distance = (left + right - 2 * (scans * weights) @ fingerprints.T).clamp_min(0)
        attention = torch.softmax(-self.log_beta.exp() * distance, dim=1)
        return attention @ centered_positions


def prepared_pair(lower_floor, upper_floor, device, seed):
    """上层按位置隔离训练和验证；下层完整 radio map 只作输入。"""
    positions, fingerprints = make_map(lower_floor)
    origin = positions.mean(axis=0)
    unique, inverse = np.unique(upper_floor["xy"], axis=0, return_inverse=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    train_position = order[:int(len(unique) * 0.8)]
    train = np.flatnonzero(np.isin(inverse, train_position))
    val = np.flatnonzero(~np.isin(inverse, train_position))
    return {
        "fingerprints": torch.as_tensor(fingerprints, device=device),
        "positions": torch.as_tensor((positions - origin) / 100, dtype=torch.float32, device=device),
        "scans": torch.as_tensor(rss_features(upper_floor["rssi"]), device=device),
        "truth": torch.as_tensor((upper_floor["xy"] - origin) / 100, dtype=torch.float32, device=device),
        "origin": origin,
        "train": train,
        "val": val,
        "train_positions": int(len(train_position)),
        "val_positions": int(len(unique) - len(train_position)),
    }


@torch.no_grad()
def source_validation(model, pairs):
    """七个源楼层转移等权，不按扫描数加权。"""
    model.eval()
    scores = []
    for pair in pairs:
        errors = []
        for block in np.array_split(pair["val"], max(1, (len(pair["val"]) + 255) // 256)):
            prediction = model(pair["scans"][block], pair["fingerprints"], pair["positions"])
            errors.append(torch.linalg.vector_norm((prediction - pair["truth"][block]) * 100, dim=1))
        scores.append(torch.cat(errors).mean().item())
    return float(np.mean(scores)), scores


def learn_metric(pairs, seed, device, source_steps):
    """模型选择只看旧楼层的位置隔离验证集。"""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = CrossFloorMetric().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    initial, _ = source_validation(model, pairs)
    best = (initial, 0, copy.deepcopy(model.state_dict()))
    history = [{"step": 0, "source_val_mde_m": initial}]
    for step in range(1, source_steps + 1):
        model.train()
        pair = pairs[int(rng.integers(len(pairs)))]
        chosen = rng.choice(pair["train"], size=128, replace=True)
        prediction = model(pair["scans"][chosen], pair["fingerprints"], pair["positions"])
        weights = 2 * torch.sigmoid(model.ap_logits)
        loss = (prediction - pair["truth"][chosen]).square().sum(dim=1).mean()
        loss = loss + 0.001 * (weights - 1).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.log_beta.clamp_(0, math.log(100))
        if step % 100 == 0:
            score, per_pair = source_validation(model, pairs)
            history.append({"step": step, "source_val_mde_m": score,
                            "per_pair_mde_m": per_pair})
            print(f"seed={seed} step={step} source_val={score:.3f}m", flush=True)
            if score < best[0]:
                best = (score, step, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[2])
    model.eval()
    return model, {"initial_mde_m": initial, "best_mde_m": best[0],
                   "selected_step": best[1], "history": history}


@torch.no_grad()
def map_predictions(model, lower_floor, upper_floor, device):
    """冻结权重，以旧地图估计每条上层扫描的坐标。"""
    positions, fingerprints = make_map(lower_floor)
    origin = positions.mean(axis=0)
    prototype = torch.as_tensor(fingerprints, device=device)
    centered = torch.as_tensor((positions - origin) / 100, dtype=torch.float32, device=device)
    scans = rss_features(upper_floor["rssi"])
    predictions = []
    for block in np.array_split(scans, max(1, (len(scans) + 255) // 256)):
        x = torch.as_tensor(block, device=device)
        predictions.append(model(x, prototype, centered).cpu().numpy().astype(np.float64) * 100 + origin)
    return np.concatenate(predictions)


def calibrate_residual(floors, predictions, pairs, config, seed):
    """只用旧楼层的 20 个模拟少样本 episode 选择局部校正参数。"""
    candidates = [None] + [(ell, noise) for ell in config["length_scales_m"]
                           for noise in config["noise_ratios"]]
    rng = np.random.default_rng(config["source_episode_seed"] + seed)
    pair_scores = {}
    for lower, upper in pairs:
        floor = floors[upper]
        source_xy = predictions[upper]
        errors = {str(c): [] for c in candidates}
        for _ in range(config["source_episodes_per_pair"]):
            support, query = episode(floor, rng, config["support_positions"], config["scans_per_position"])
            truth = floor["xy"][query]
            for c in candidates:
                estimate = source_xy[query] if c is None else correction(floor, source_xy, support, query, *c)
                errors[str(c)].append(float(np.linalg.norm(estimate - truth, axis=1).mean()))
        pair_scores[f"{lower}->{upper}"] = {key: float(np.mean(value)) for key, value in errors.items()}
    means = {str(c): float(np.mean([scores[str(c)] for scores in pair_scores.values()]))
             for c in candidates}
    chosen = min(candidates, key=lambda c: (means[str(c)], c is not None,
                                           0 if c is None else -c[1], 0 if c is None else -c[0]))
    return chosen, means, pair_scores


def evaluate_targets(floors, metric_maps, hard_maps, config, seed, chosen):
    """所有方法共用既有 manifest，Query 位置不在 Support。"""
    rows = []
    for target in config["targets"]:
        lower = f"{target[:2]}F{int(target[-1]) - 1}"
        floor = floors[target]
        ids = {int(row): i for i, row in enumerate(floor["row_ids"])}
        manifest = json.loads((Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json").read_text())[target]
        support = np.array([ids[int(row)] for row in manifest["support"]])
        query = np.array([ids[int(row)] for row in manifest["unseen_position_query"]])
        assert len(support) == 30 and len(np.unique(floor["xy"][support], axis=0)) == 10
        assert set(map(tuple, floor["xy"][support])).isdisjoint(map(tuple, floor["xy"][query]))
        truth = floor["xy"][query]
        learned = metric_maps[target]
        hard = hard_maps[target]
        adapted = learned[query] if chosen is None else correction(floor, learned, support, query, *chosen)
        rows.append({"target": target, "source": lower, "seed": seed,
                     "query_count": int(len(query)),
                     "hard_uniform_mde_m": float(np.linalg.norm(hard[query] - truth, axis=1).mean()),
                     "learned_map_mde_m": float(np.linalg.norm(learned[query] - truth, axis=1).mean()),
                     "learned_plus_anchors_mde_m": float(np.linalg.norm(adapted - truth, axis=1).mean())})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adjacent_map.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/cross_floor_metric_v1"))
    parser.add_argument("--source-steps", type=int, default=1000)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else
                          "cuda:0" if torch.cuda.is_available() else "cpu")
    floors = load_floors(config["data_path"])
    transitions = source_pairs(floors, set(config["targets"]))
    assert len(transitions) == 7
    args.output.mkdir(parents=True, exist_ok=False)
    print("device", device, "source_transitions", transitions, flush=True)
    all_rows = []
    for seed in config["seeds"]:
        prepared = [prepared_pair(floors[lower], floors[upper], device, 7731 + seed * 100 + i)
                    for i, (lower, upper) in enumerate(transitions)]
        model, selection = learn_metric(prepared, seed, device, args.source_steps)
        predictions = {upper: map_predictions(model, floors[lower], floors[upper], device)
                       for lower, upper in transitions}
        chosen, scores, per_pair = calibrate_residual(floors, predictions, transitions, config, seed)
        target_metric = {}
        target_hard = {}
        for target in config["targets"]:
            lower = f"{target[:2]}F{int(target[-1]) - 1}"
            target_metric[target] = map_predictions(model, floors[lower], floors[target], device)
            target_hard[target] = source_predictions(floors[target], make_map(floors[lower]))
        rows = evaluate_targets(floors, target_metric, target_hard, config, seed, chosen)
        all_rows.extend(rows)
        torch.save(model.state_dict(), args.output / f"seed_{seed}_metric.pt")
        (args.output / f"seed_{seed}_source_selection.json").write_text(json.dumps({
            "source_transitions": transitions, "source_position_counts": [
                {"train": p["train_positions"], "val": p["val_positions"]} for p in prepared],
            "metric": selection, "ap_weight_mean": float((2 * torch.sigmoid(model.ap_logits)).mean().item()),
            "ap_weight_std": float((2 * torch.sigmoid(model.ap_logits)).std().item()),
            "beta": float(model.log_beta.exp().item()), "residual_choice": chosen,
            "residual_source_scores": scores, "residual_source_pairs": per_pair
        }, indent=2) + "\n")
        print("seed", seed, "checkpoint", selection["selected_step"], "residual", chosen,
              "targets", [(r["target"], round(r["learned_map_mde_m"], 3),
                           round(r["learned_plus_anchors_mde_m"], 3)) for r in rows], flush=True)
    (args.output / "results.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    for target in config["targets"]:
        rows = [r for r in all_rows if r["target"] == target]
        print(target, {key: round(float(np.mean([r[key] for r in rows])), 3)
                       for key in ("hard_uniform_mde_m", "learned_map_mde_m", "learned_plus_anchors_mde_m")},
              flush=True)


if __name__ == "__main__":
    main()
