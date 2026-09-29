"""从旧楼层选择软地图温度，评估零梯度及十锚点校正。"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from data.uji import load_floors
from scripts.evaluate_adjacent_map import make_map, source_pairs, source_predictions
from scripts.learn_cross_floor_metric import (CrossFloorMetric, calibrate_residual,
                                              evaluate_targets, map_predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adjacent_map.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/softmap_v1"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else
                          "cuda:0" if torch.cuda.is_available() else "cpu")
    floors = load_floors(config["data_path"])
    pairs = source_pairs(floors, set(config["targets"]))
    beta_candidates = [1, 2, 5, 10, 20, 50, 100]
    source_scores = {}
    model = CrossFloorMetric().to(device).eval()
    for beta in beta_candidates:
        with torch.no_grad():
            model.log_beta.fill_(math.log(beta))
        per_pair = {}
        for lower, upper in pairs:
            predicted = map_predictions(model, floors[lower], floors[upper], device)
            per_pair[f"{lower}->{upper}"] = float(np.linalg.norm(predicted - floors[upper]["xy"], axis=1).mean())
        source_scores[beta] = {"equal_pair_mde_m": float(np.mean(list(per_pair.values()))),
                               "per_pair_mde_m": per_pair}
    chosen_beta = min(beta_candidates, key=lambda beta: (source_scores[beta]["equal_pair_mde_m"], beta))
    with torch.no_grad():
        model.log_beta.fill_(math.log(chosen_beta))
    source_predictions_by_floor = {upper: map_predictions(model, floors[lower], floors[upper], device)
                                   for lower, upper in pairs}
    target_soft = {}
    target_hard = {}
    for target in config["targets"]:
        lower = f"{target[:2]}F{int(target[-1]) - 1}"
        target_soft[target] = map_predictions(model, floors[lower], floors[target], device)
        target_hard[target] = source_predictions(floors[target], make_map(floors[lower]))
    args.output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    choices = {}
    for seed in config["seeds"]:
        correction_choice, correction_scores, _ = calibrate_residual(
            floors, source_predictions_by_floor, pairs, config, seed)
        choices[seed] = {"choice": correction_choice, "source_scores_mde_m": correction_scores}
        all_rows.extend(evaluate_targets(floors, target_soft, target_hard, config, seed, correction_choice))
    (args.output / "source_selection.json").write_text(json.dumps({
        "beta": chosen_beta, "source_scores": source_scores,
        "target_gradient_steps": 0, "residual_choices": choices}, indent=2) + "\n")
    (args.output / "results.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    print("source beta", chosen_beta, "equal-pair MDE", source_scores[chosen_beta]["equal_pair_mde_m"])
    for target in config["targets"]:
        rows = [r for r in all_rows if r["target"] == target]
        print(target, {key: round(float(np.mean([r[key] for r in rows])), 3)
                       for key in ("hard_uniform_mde_m", "learned_map_mde_m", "learned_plus_anchors_mde_m")})


if __name__ == "__main__":
    main()
