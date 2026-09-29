"""以已知位置留一定位误差选校准强度，再审计历史跨层转移。"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from data.uji import load_floors
from scripts.evaluate_adjacent_map import make_map, source_pairs
from scripts.evaluate_ap_offset_source import (best_option, calibrated_predict,
                                               estimate_offset, mde, option_name, prepare)


def select_from_support(floor, support, data, options, beta):
    """每次留出整个物理位置，避免同位置扫描泄漏到校准中。"""
    _, groups = np.unique(floor["xy"][support], axis=0, return_inverse=True)
    scores = {}
    for lam in options:
        held_errors = []
        for group in range(groups.max() + 1):
            fit = support[groups != group]
            held = support[groups == group]
            if lam is None:
                predicted = data["train_baseline"][held]
            else:
                offset = estimate_offset(floor, fit, data["positions"], data["prototypes"], lam)
                predicted = calibrated_predict(data["train_d"][held],
                                               data["train_observed"][held],
                                               data["prototypes"], data["positions"],
                                               offset, beta)
            held_errors.append(mde(predicted, floor["xy"][held]))
        scores[option_name(lam)] = float(np.mean(held_errors))
    return best_option(options, scores), scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/support_ranked_map.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/support_ranked_source"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    train_path = Path(config["data_path"])
    floors = load_floors(train_path)
    validation = load_floors(train_path.with_name("validationData.csv"))
    source_episodes = json.loads(Path(config["source_episode_path"]).read_text())
    options = config["lambda_candidates"]
    beta = config["beta"]
    by_pair = {}
    detailed = {}
    for lower, upper in source_pairs(floors, set(config["targets"])):
        name = f"{lower}->{upper}"
        floor = floors[upper]
        data = prepare(floors[lower], floor, validation[upper], beta)
        ids = {int(row): i for i, row in enumerate(floor["row_ids"])}
        episodes = []
        for item in source_episodes[name]:
            support = np.array([ids[row] for row in item["support_row_ids"]])
            query = np.array([ids[row] for row in item["query_row_ids"]])
            selected, cv_scores = select_from_support(floor, support, data, options, beta)
            offset = estimate_offset(floor, support, data["positions"], data["prototypes"], selected)
            train_pred = calibrated_predict(data["train_d"][query],
                                            data["train_observed"][query],
                                            data["prototypes"], data["positions"],
                                            offset, beta)
            official_pred = calibrated_predict(data["official_d"], data["official_observed"],
                                               data["prototypes"], data["positions"],
                                               offset, beta)
            episodes.append({"selected_lambda": selected, "support_cv_mde_m": cv_scores,
                             "query_baseline_mde_m": mde(data["train_baseline"][query], floor["xy"][query]),
                             "query_calibrated_mde_m": mde(train_pred, floor["xy"][query]),
                             "official_baseline_mde_m": mde(data["official_baseline"], validation[upper]["xy"]),
                             "official_calibrated_mde_m": mde(official_pred, validation[upper]["xy"])})
        old = float(np.mean([v["query_baseline_mde_m"] for v in episodes]))
        new = float(np.mean([v["query_calibrated_mde_m"] for v in episodes]))
        official_old = float(np.mean([v["official_baseline_mde_m"] for v in episodes]))
        official_new = float(np.mean([v["official_calibrated_mde_m"] for v in episodes]))
        by_pair[name] = {"baseline_mde_m": old, "calibrated_mde_m": new,
                         "improvement_m": old - new,
                         "official_baseline_mde_m": official_old,
                         "official_calibrated_mde_m": official_new,
                         "official_improvement_m": official_old - official_new,
                         "selected_counts": dict(Counter(option_name(v["selected_lambda"]) for v in episodes))}
        detailed[name] = episodes
        print(name, "gain", round(old-new, 3), "official_gain", round(official_old-official_new, 3),
              "choices", by_pair[name]["selected_counts"], flush=True)
    gain = float(np.mean([v["improvement_m"] for v in by_pair.values()]))
    count = sum(v["improvement_m"] > 0 for v in by_pair.values())
    official_gain = float(np.mean([v["official_improvement_m"] for v in by_pair.values()]))
    gate = gain >= 1 and count >= 5 and official_gain >= 0
    result = {"method": "support_position_leave_one_out_lambda_selection",
              "source_pairs": by_pair, "source_equal_pair_gain_m": gain,
              "source_improved_pairs": count, "source_official_equal_pair_gain_m": official_gain,
              "target_evaluation_gate_passed": gate}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "episodes.json").write_text(json.dumps(detailed) + "\n")
    print("GAIN", round(gain, 3), "PAIRS", count, "OFFICIAL_GAIN", round(official_gain, 3),
          "GATE", gate, flush=True)


if __name__ == "__main__":
    main()
