"""在 RSSI 空间利用十个目标锚点修正旧楼层地图。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from data.uji import load_floors
from scripts.evaluate_adjacent_map import episode, rss_features, source_pairs
from scripts.learn_cross_floor_metric import CrossFloorMetric, map_predictions


def correct(floor, old_map_xy, support, query, gamma, noise):
    """每个物理位置先合并三次扫描，再在无线空间核回归地图残差。"""
    positions, inverse = np.unique(floor["xy"][support], axis=0, return_inverse=True)
    x = rss_features(floor["rssi"])
    fingerprints = np.zeros((len(positions), 520), dtype=np.float64)
    mapped = np.zeros((len(positions), 2), dtype=np.float64)
    np.add.at(fingerprints, inverse, x[support])
    np.add.at(mapped, inverse, old_map_xy[support])
    counts = np.bincount(inverse)
    fingerprints /= counts[:, None]
    mapped /= counts[:, None]
    residual = positions - mapped
    d_train = ((fingerprints[:, None] - fingerprints[None, :]) ** 2).sum(axis=2)
    kernel = np.exp(-gamma * d_train)
    d_query = ((x[query, None] - fingerprints[None, :]) ** 2).sum(axis=2)
    cross = np.exp(-gamma * d_query)
    return old_map_xy[query] + cross @ np.linalg.solve(kernel + noise * np.eye(len(positions)), residual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adjacent_map.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/support_rssi_v1"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else
                          "cuda:0" if torch.cuda.is_available() else "cpu")
    floors = load_floors(config["data_path"])
    source = source_pairs(floors, set(config["targets"]))
    model = CrossFloorMetric().to(device).eval()
    with torch.no_grad():
        model.log_beta.fill_(math.log(10))
    old_pred = {hi: map_predictions(model, floors[lo], floors[hi], device) for lo, hi in source}
    candidates = [None] + [(g, noise) for g in (2, 5, 10, 20) for noise in (0.3, 1, 3)]
    source_scores = {}
    target_rows = []
    for seed in config["seeds"]:
        rng = np.random.default_rng(config["source_episode_seed"] + seed)
        per_pair = {}
        for lower, upper in source:
            floor = floors[upper]
            mapped = old_pred[upper]
            scores = {str(c): [] for c in candidates}
            for _ in range(config["source_episodes_per_pair"]):
                support, query = episode(floor, rng, 10, 3)
                for c in candidates:
                    pred = mapped[query] if c is None else correct(floor, mapped, support, query, *c)
                    scores[str(c)].append(float(np.linalg.norm(pred - floor["xy"][query], axis=1).mean()))
            per_pair[f"{lower}->{upper}"] = {key: float(np.mean(value)) for key, value in scores.items()}
        means = {str(c): float(np.mean([r[str(c)] for r in per_pair.values()])) for c in candidates}
        chosen = min(candidates, key=lambda c: (means[str(c)], c is not None,
                                                0 if c is None else c[1], 0 if c is None else c[0]))
        source_scores[seed] = {"choice": chosen, "equal_pair_mde_m": means, "per_pair_mde_m": per_pair}
        for target in config["targets"]:
            lower = f"{target[:2]}F{int(target[-1]) - 1}"
            floor = floors[target]
            mapped = map_predictions(model, floors[lower], floor, device)
            ids = {int(row): i for i, row in enumerate(floor["row_ids"])}
            manifest = json.loads((Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json").read_text())[target]
            support = np.array([ids[int(row)] for row in manifest["support"]])
            query = np.array([ids[int(row)] for row in manifest["unseen_position_query"]])
            assert len(support) == 30 and len(np.unique(floor["xy"][support], axis=0)) == 10
            assert set(map(tuple, floor["xy"][support])).isdisjoint(map(tuple, floor["xy"][query]))
            pred = mapped[query] if chosen is None else correct(floor, mapped, support, query, *chosen)
            target_rows.append({"seed": seed, "target": target, "source": lower,
                                "source_map_mde_m": float(np.linalg.norm(mapped[query] - floor["xy"][query], axis=1).mean()),
                                "rssi_anchor_mde_m": float(np.linalg.norm(pred - floor["xy"][query], axis=1).mean())})
        print("seed", seed, "source_choice", chosen, "source_mde", means[str(chosen)], flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source_selection.json").write_text(json.dumps(source_scores, indent=2) + "\n")
    (args.output / "results.json").write_text(json.dumps(target_rows, indent=2) + "\n")
    for target in config["targets"]:
        rows = [r for r in target_rows if r["target"] == target]
        print(target, "old_map", round(float(np.mean([r["source_map_mde_m"] for r in rows])), 3),
              "rssi_anchors", round(float(np.mean([r["rssi_anchor_mde_m"] for r in rows])), 3))


if __name__ == "__main__":
    main()
