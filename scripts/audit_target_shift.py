"""在固定目标 manifest 上诊断 SCM v1 的距离、信号和定位变化。"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.uji import load_floors
from scripts.audit_field_vs_ranking import BINS, KNN, PARAMS, field_at_query, summarize_rows
from scripts.evaluate_signal_calibrated_map import FLOOR, build_map, lower_map, wknn


def evaluate(raw, xy, old, virtual, support_rssi, support_xy):
    old_field, scm_field = field_at_query(old, support_rssi, support_xy, xy)
    old_pred = wknn(raw, *old, *KNN)
    scm_pred = wknn(raw, *virtual, *KNN)
    distance = np.sqrt(((xy[:, None] - np.unique(support_xy, axis=0)[None]) ** 2)
                       .sum(-1)).min(1)
    old_map_dist = np.sqrt(((xy[:, None] - old[1][None]) ** 2).sum(-1)).min(1)
    out = {"n": len(xy), "mean_nearest_support_m": float(distance.mean()),
           "median_nearest_support_m": float(np.median(distance)),
           "mean_nearest_old_map_m": float(old_map_dist.mean())}
    out["all"] = summarize_rows(raw, xy, old_field, scm_field, old_pred, scm_pred,
                                np.ones(len(xy), bool))
    for name, left, right in BINS:
        out[name] = summarize_rows(raw, xy, old_field, scm_field, old_pred, scm_pred,
                                   (distance >= left) & (distance < right))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/signal_calibrated_map.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    train = load_floors(config["train_path"])
    val = load_floors(config["validation_path"])
    result = {}
    for target in config["targets"]:
        lower, upper = FLOOR[target]
        old, new, validation = lower_map(train[lower]), train[upper], val[upper]
        row_to_i = {int(row): i for i, row in enumerate(new["row_ids"])}
        result[target] = {}
        for seed in config["manifest_seeds"]:
            manifest = json.loads((Path(config["manifest_root"]) /
                                   f"seed_{seed}" / "manifest.json").read_text())[target]
            s = np.array([row_to_i[int(row)] for row in manifest["support"]])
            q = np.array([row_to_i[int(row)] for row in manifest["unseen_position_query"]])
            s_raw, s_xy = new["rssi"][s], new["xy"][s]
            assert len(np.unique(s_xy, axis=0)) == 10
            assert not set(map(tuple, s_xy)) & set(map(tuple, new["xy"][q]))
            virtual = build_map("scm", old, s_raw, s_xy, PARAMS, config)
            result[target][str(seed)] = {
                "same_day": evaluate(new["rssi"][q], new["xy"][q], old,
                                     virtual, s_raw, s_xy),
                "official_validation": evaluate(validation["rssi"], validation["xy"],
                                                old, virtual, s_raw, s_xy),
            }
        print(target, flush=True)
        for dataset in ("same_day", "official_validation"):
            rows = [result[target][str(seed)][dataset] for seed in config["manifest_seeds"]]
            print(" ", dataset,
                  "nearest support", round(np.mean([r["mean_nearest_support_m"] for r in rows]), 2),
                  "RSSI RMSE", round(np.mean([r["all"]["old_rssi_rmse_db"] for r in rows]), 2),
                  "→", round(np.mean([r["all"]["scm_rssi_rmse_db"] for r in rows]), 2),
                  "MDE", round(np.mean([r["all"]["old_mde_m"] for r in rows]), 2),
                  "→", round(np.mean([r["all"]["scm_mde_m"] for r in rows]), 2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
