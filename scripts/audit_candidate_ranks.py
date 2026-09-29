"""比较旧图与 SCM 对同一批旧图空间候选的无线排序。"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.uji import load_floors
from scripts.audit_field_vs_ranking import BINS, KNN, PARAMS
from scripts.evaluate_signal_calibrated_map import FLOOR, NO_SIGNAL, build_map, dbm, lower_map, wknn


def candidate_distances(raw, prototypes):
    a = np.clip((dbm(raw) - NO_SIGNAL) / -NO_SIGNAL, 0, 1)
    b = np.clip((prototypes - NO_SIGNAL) / -NO_SIGNAL, 0, 1)
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None]
                      - 2 * a @ b.T, 0)


def summarize(raw, xy, old, virtual, support_xy):
    n_old = len(old[1])
    physical = np.sqrt(((xy[:, None] - old[1][None]) ** 2).sum(-1))
    true_candidate = physical.argmin(1)
    geo_dist = physical[np.arange(len(xy)), true_candidate]
    to_support = np.sqrt(((xy[:, None] - np.unique(support_xy, axis=0)[None]) ** 2)
                         .sum(-1)).min(1)
    old_dist = candidate_distances(raw, old[0])
    scm_dist = candidate_distances(raw, virtual[0][:n_old])
    old_rank = 1 + (old_dist < old_dist[np.arange(len(xy)), true_candidate, None]).sum(1)
    scm_rank = 1 + (scm_dist < scm_dist[np.arange(len(xy)), true_candidate, None]).sum(1)
    old_pred = wknn(raw, *old, *KNN)
    field_pred = wknn(raw, virtual[0][:n_old], old[1], *KNN)
    full_pred = wknn(raw, *virtual, *KNN)
    old_err = np.linalg.norm(old_pred - xy, axis=1)
    field_err = np.linalg.norm(field_pred - xy, axis=1)
    full_err = np.linalg.norm(full_pred - xy, axis=1)
    result = {}
    for name, left, right in (("all", 0, float("inf")), *BINS):
        mask = (to_support >= left) & (to_support < right)
        if not mask.any():
            continue
        result[name] = {
            "n": int(mask.sum()),
            "nearest_old_coordinate_m": float(geo_dist[mask].mean()),
            "within_5m_old_coordinate_fraction": float((geo_dist[mask] <= 5).mean()),
            "old_true_candidate_rank_mean": float(old_rank[mask].mean()),
            "scm_true_candidate_rank_mean": float(scm_rank[mask].mean()),
            "old_true_candidate_top8_fraction": float((old_rank[mask] <= 8).mean()),
            "scm_true_candidate_top8_fraction": float((scm_rank[mask] <= 8).mean()),
            "old_mde_m": float(old_err[mask].mean()),
            "scm_field_only_mde_m": float(field_err[mask].mean()),
            "scm_full_mde_m": float(full_err[mask].mean()),
        }
    return result


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
            assert not set(map(tuple, s_xy)) & set(map(tuple, new["xy"][q]))
            virtual = build_map("scm", old, s_raw, s_xy, PARAMS, config)
            result[target][str(seed)] = {
                "same_day": summarize(new["rssi"][q], new["xy"][q], old, virtual, s_xy),
                "official_validation": summarize(validation["rssi"], validation["xy"],
                                                old, virtual, s_xy),
            }
        for dataset in ("same_day", "official_validation"):
            rows = [result[target][str(seed)][dataset]["all"] for seed in config["manifest_seeds"]]
            print(target, dataset, "rank", round(np.mean([r["old_true_candidate_rank_mean"] for r in rows]), 2),
                  "→", round(np.mean([r["scm_true_candidate_rank_mean"] for r in rows]), 2),
                  "field MDE", round(np.mean([r["scm_field_only_mde_m"] for r in rows]), 2),
                  "full MDE", round(np.mean([r["scm_full_mde_m"] for r in rows]), 2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
