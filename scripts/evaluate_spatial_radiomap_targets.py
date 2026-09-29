"""用固定目标 Support 更新旧图，在同一 Query 比较旧图和少样本基线。"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.uji import load_floors
from scripts.evaluate_adjacent_map import make_map, rss_features
from scripts.evaluate_spatial_radiomap_source import (distances, key, mde,
                                                       soft_predict, update_map)


def wknn(query, support_x, support_xy):
    """目标三近邻 RSSI 指纹，逆距离加权坐标。"""
    distance = np.sqrt(distances(query, support_x))
    nearest = np.argsort(distance, axis=1)[:, :3]
    chosen = np.take_along_axis(distance, nearest, axis=1)
    weight = 1 / (chosen + 1e-6)
    return (weight[:, :, None] * support_xy[nearest]).sum(1) / weight.sum(1)[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spatial_radiomap.json")
    parser.add_argument("--selection", type=Path, default=Path("outputs/spatial_radiomap_source/selection.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/spatial_radiomap_targets"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    selected = json.loads(args.selection.read_text())
    assert selected["target_evaluation_gate_passed"]
    candidates = [(length, shrink) for length in config["length_scales_m"]
                  for shrink in config["shrinkage"]]
    pair = next(pair for pair in candidates if key(pair) == selected["selected"])
    train_path = Path(config["data_path"])
    floors = load_floors(train_path)
    validation = load_floors(train_path.with_name("validationData.csv"))
    results = []
    for target in config["targets"]:
        old_name = f"{target[:2]}F{int(target[-1]) - 1}"
        floor = floors[target]
        official = validation[target]
        positions, fingerprints = make_map(floors[old_name])
        train_x = rss_features(floor["rssi"])
        official_x = rss_features(official["rssi"])
        ids = {int(row): i for i, row in enumerate(floor["row_ids"])}
        for seed in config["seeds"]:
            manifest_path = Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json"
            manifest = json.loads(manifest_path.read_text())[target]
            support = np.array([ids[int(row)] for row in manifest["support"]])
            query = np.array([ids[int(row)] for row in manifest["unseen_position_query"]])
            support_xy = floor["xy"][support]
            assert len(support) == 30 and len(np.unique(support_xy, axis=0)) == 10
            assert set(map(tuple, support_xy)).isdisjoint(map(tuple, floor["xy"][query]))
            updated = update_map(floor, support, positions, fingerprints, *pair)
            unique_xy, group = np.unique(support_xy, axis=0, return_inverse=True)
            shuffled_maps = [update_map(floor, support, positions, fingerprints, *pair,
                                        support_xy=unique_xy[(group + shift) % len(unique_xy)])
                             for shift in range(1, len(unique_xy))]
            for split, x, truth in [
                ("unseen_position", train_x[query], floor["xy"][query]),
                ("official_validation", official_x, official["xy"]),
            ]:
                old_pred = soft_predict(x, fingerprints, positions, config["beta"])
                new_pred = soft_predict(x, updated, positions, config["beta"])
                support_pred = wknn(x, train_x[support], support_xy)
                shuffled_scores = [mde(soft_predict(x, shuffled, positions, config["beta"]), truth)
                                   for shuffled in shuffled_maps]
                results.append({"target": target, "source": old_name, "seed": seed,
                                "split": split, "query_count": len(x),
                                "support_row_ids": manifest["support"],
                                "query_row_ids": (manifest["unseen_position_query"] if split == "unseen_position"
                                                  else official["row_ids"].tolist()),
                                "old_map_mde_m": mde(old_pred, truth),
                                "updated_map_mde_m": mde(new_pred, truth),
                                "target_wknn_mde_m": mde(support_pred, truth),
                                "shuffled_label_map_mde_m": float(np.mean(shuffled_scores)),
                                "shuffled_label_map_sd_m": float(np.std(shuffled_scores, ddof=1))})
        print(target, [(split, round(np.mean([r["old_map_mde_m"] for r in results
                                              if r["target"] == target and r["split"] == split]), 3),
                        round(np.mean([r["updated_map_mde_m"] for r in results
                                       if r["target"] == target and r["split"] == split]), 3))
                       for split in ("unseen_position", "official_validation")], flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("selected", key(pair), "target_rows", len(results), flush=True)


if __name__ == "__main__":
    main()
