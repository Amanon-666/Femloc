"""冻结源地图及源端参数，在 UJI 官方 validationData 上评估。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from data.uji import load_floors
from scripts.evaluate_adjacent_map import correction, make_map, source_predictions
from scripts.evaluate_support_rssi import correct
from scripts.learn_cross_floor_metric import CrossFloorMetric, map_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adjacent_map.json")
    parser.add_argument("--validation", default="/home/panyushuo/projects/panyushuo/UJIIndoorLoc/data/raw/UJIndoorLoc/validationData.csv")
    parser.add_argument("--output", type=Path, default=Path("outputs/official_validation_v1"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    soft_selection = json.loads(Path("outputs/softmap_v1/source_selection.json").read_text())
    rssi_selection = json.loads(Path("outputs/support_rssi_v1/source_selection.json").read_text())
    torch.set_num_threads(1)
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else
                          "cuda:0" if torch.cuda.is_available() else "cpu")
    model = CrossFloorMetric().to(device).eval()
    with torch.no_grad():
        model.log_beta.fill_(math.log(soft_selection["beta"]))
    train = load_floors(config["data_path"])
    validation = load_floors(args.validation)
    rows = []
    for target in config["targets"]:
        lower = f"{target[:2]}F{int(target[-1]) - 1}"
        floor = validation[target]
        predicted = map_predictions(model, train[lower], floor, device)
        hard = source_predictions(floor, make_map(train[lower]))
        truth = floor["xy"]
        train_positions = set(map(tuple, train[target]["xy"]))
        val_positions = set(map(tuple, truth))
        no_anchor_mde = float(np.linalg.norm(predicted - truth, axis=1).mean())
        hard_mde = float(np.linalg.norm(hard - truth, axis=1).mean())
        for seed in config["seeds"]:
            path = Path(config["manifest_root"]) / f"seed_{seed}" / "manifest.json"
            manifest = json.loads(path.read_text())[target]
            ids = {int(row_id): i for i, row_id in enumerate(train[target]["row_ids"])}
            support = np.array([ids[int(row_id)] for row_id in manifest["support"]])
            support_floor = {"rssi": train[target]["rssi"][support],
                             "xy": train[target]["xy"][support]}
            support_pred = map_predictions(model, train[lower], support_floor, device)
            count = len(support)
            # validation 坐标在适应函数中置零，只在最终误差计算时使用。
            combined = {"rssi": np.concatenate([support_floor["rssi"], floor["rssi"]]),
                        "xy": np.concatenate([support_floor["xy"], np.zeros_like(truth)])}
            mapped = np.concatenate([support_pred, predicted])
            support_indices = np.arange(count)
            query_indices = np.arange(count, len(mapped))
            spatial_params = soft_selection["residual_choices"][str(seed)]["choice"]
            rssi_params = rssi_selection[str(seed)]["choice"]
            spatial = correction(combined, mapped, support_indices, query_indices, *spatial_params)
            rssi = correct(combined, mapped, support_indices, query_indices, *rssi_params)
            rows.append({"target": target, "source": lower, "seed": seed,
                         "validation_count": len(truth),
                         "validation_unique_positions": len(val_positions),
                         "train_validation_position_overlap": len(train_positions & val_positions),
                         "support_validation_position_overlap": len(set(map(tuple, support_floor["xy"])) & val_positions),
                         "hard_map_mde_m": hard_mde, "soft_map_mde_m": no_anchor_mde,
                         "spatial_anchor_mde_m": float(np.linalg.norm(spatial - truth, axis=1).mean()),
                         "rssi_anchor_mde_m": float(np.linalg.norm(rssi - truth, axis=1).mean())})
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    for target in config["targets"]:
        group = [r for r in rows if r["target"] == target]
        print(target, "n", group[0]["validation_count"],
              {key: round(float(np.mean([r[key] for r in group])), 3)
               for key in ("hard_map_mde_m", "soft_map_mde_m", "spatial_anchor_mde_m", "rssi_anchor_mde_m")})


if __name__ == "__main__":
    main()
