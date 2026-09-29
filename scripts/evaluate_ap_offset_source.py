"""在历史跨层任务上选择 AP 偏移收缩量并执行事前门槛。"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.uji import load_floors
from scripts.evaluate_adjacent_map import episode, make_map, rss_features, source_pairs


def distances(x, prototypes):
    d = (x * x).sum(1)[:, None] + (prototypes * prototypes).sum(1)[None, :] - 2 * x @ prototypes.T
    return np.maximum(d, 0)


def soft_predict(d, positions, beta):
    weight = np.exp(-beta * (d - d.min(axis=1, keepdims=True)))
    return weight @ positions / weight.sum(axis=1, keepdims=True)


def estimate_offset(floor, support, positions, prototypes, lam, support_xy=None):
    """用已知锚点坐标配对旧图，估计每个 AP 的共同信号差。"""
    if lam is None:
        return np.zeros(prototypes.shape[1], dtype=np.float32)
    y = floor["xy"][support] if support_xy is None else support_xy
    nearest = ((y[:, None, :] - positions[None, :, :]) ** 2).sum(2).argmin(1)
    old = prototypes[nearest]
    x = rss_features(floor["rssi"][support])
    valid = (floor["rssi"][support] != 100) & (old > 0)
    count = valid.sum(0)
    total = ((x - old) * valid).sum(0)
    return np.divide(total, count + lam, out=np.zeros_like(total), where=count + lam > 0)


def calibrated_predict(d, observed, prototypes, positions, offset, beta):
    """把信号校正直接写进旧图距离，改变候选位置排序。"""
    corrected_d = d + 2 * (observed * offset) @ prototypes.T
    return soft_predict(corrected_d, positions, beta)


def mde(pred, truth):
    return float(np.linalg.norm(pred - truth, axis=1).mean())


def prepare(lower, upper, validation, beta):
    positions, prototypes = make_map(lower)
    x = rss_features(upper["rssi"])
    d = distances(x, prototypes)
    official = validation
    official_d = distances(rss_features(official["rssi"]), prototypes)
    return {"positions": positions, "prototypes": prototypes, "train_d": d,
            "train_observed": upper["rssi"] != 100,
            "train_baseline": soft_predict(d, positions, beta),
            "official_d": official_d,
            "official_observed": official["rssi"] != 100,
            "official_baseline": soft_predict(official_d, positions, beta)}


def option_name(lam):
    return "none" if lam is None else str(lam)


def best_option(options, scores):
    return min(options, key=lambda lam: (scores[option_name(lam)], 0 if lam is None else 1,
                                         0 if lam is None else -lam))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ap_offset_map.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/ap_offset_source"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    training_path = Path(config["data_path"])
    floors = load_floors(training_path)
    validation = load_floors(training_path.with_name("validationData.csv"))
    pairs = source_pairs(floors, set(config["targets"]))
    options = config["lambda_candidates"]
    beta = config["beta"]
    rng = np.random.default_rng(config["source_episode_seed"])
    pair_scores = {}
    saved_episodes = {}
    prepared = {}
    for lower, upper in pairs:
        name = f"{lower}->{upper}"
        data = prepare(floors[lower], floors[upper], validation[upper], beta)
        prepared[name] = data
        upper_floor = floors[upper]
        scores = {option_name(lam): [] for lam in options}
        manifests = []
        for _ in range(config["source_episodes_per_pair"]):
            support, query = episode(upper_floor, rng, config["support_positions"],
                                     config["scans_per_position"])
            manifests.append({"support_row_ids": upper_floor["row_ids"][support].tolist(),
                              "query_row_ids": upper_floor["row_ids"][query].tolist()})
            for lam in options:
                if lam is None:
                    pred = data["train_baseline"][query]
                else:
                    offset = estimate_offset(upper_floor, support, data["positions"],
                                             data["prototypes"], lam)
                    pred = calibrated_predict(data["train_d"][query],
                                              data["train_observed"][query],
                                              data["prototypes"], data["positions"],
                                              offset, beta)
                scores[option_name(lam)].append(mde(pred, upper_floor["xy"][query]))
        pair_scores[name] = {key: float(np.mean(value)) for key, value in scores.items()}
        saved_episodes[name] = manifests
        print(name, {k: round(v, 3) for k, v in pair_scores[name].items()}, flush=True)

    # 用其它六个转移选收缩量，审计第七个转移上的增益。
    oof = {}
    for name in pair_scores:
        training_scores = {option_name(lam): float(np.mean([
            row[option_name(lam)] for other, row in pair_scores.items() if other != name
        ])) for lam in options}
        chosen = best_option(options, training_scores)
        old = pair_scores[name]["none"]
        new = pair_scores[name][option_name(chosen)]
        oof[name] = {"selected_lambda": chosen, "baseline_mde_m": old,
                     "calibrated_mde_m": new, "improvement_m": old - new}
    oof_gain = float(np.mean([row["improvement_m"] for row in oof.values()]))
    improved_pairs = sum(row["improvement_m"] > 0 for row in oof.values())
    all_scores = {option_name(lam): float(np.mean([
        row[option_name(lam)] for row in pair_scores.values()
    ])) for lam in options}
    selected = best_option(options, all_scores)

    # 晚期官方验证只作为历史源转移的第二道门槛，不反过来调参数。
    official_scores = {}
    for lower, upper in pairs:
        name = f"{lower}->{upper}"
        data = prepared[name]
        upper_floor = floors[upper]
        baseline = mde(data["official_baseline"], validation[upper]["xy"])
        calibrated = []
        for item in saved_episodes[name]:
            ids = {int(row): i for i, row in enumerate(upper_floor["row_ids"])}
            support = np.array([ids[row] for row in item["support_row_ids"]])
            offset = estimate_offset(upper_floor, support, data["positions"],
                                     data["prototypes"], selected)
            pred = calibrated_predict(data["official_d"], data["official_observed"],
                                      data["prototypes"], data["positions"], offset, beta)
            calibrated.append(mde(pred, validation[upper]["xy"]))
        official_scores[name] = {"baseline_mde_m": baseline,
                                 "calibrated_mde_m": float(np.mean(calibrated)),
                                 "improvement_m": baseline - float(np.mean(calibrated))}
    official_gain = float(np.mean([row["improvement_m"] for row in official_scores.values()]))
    gate = oof_gain >= 1 and improved_pairs >= 5 and official_gain >= 0
    result = {"selected_lambda": selected, "all_source_scores_mde_m": all_scores,
              "pair_scores_mde_m": pair_scores, "leave_one_transition_out": oof,
              "oof_macro_improvement_m": oof_gain, "oof_improved_pairs": improved_pairs,
              "source_official_validation": official_scores,
              "source_official_macro_improvement_m": official_gain,
              "target_evaluation_gate_passed": gate}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "source_episodes.json").write_text(json.dumps(saved_episodes) + "\n")
    print("SELECTED", option_name(selected), "OOF_GAIN", round(oof_gain, 3),
          "PAIRS", improved_pairs, "OFFICIAL_GAIN", round(official_gain, 3),
          "GATE", gate, flush=True)


if __name__ == "__main__":
    main()
