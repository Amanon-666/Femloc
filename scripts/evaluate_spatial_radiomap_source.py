"""用历史整层转移审计少量锚点更新旧楼层信号地图。"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.uji import load_floors
from scripts.evaluate_adjacent_map import episode, make_map, rss_features, source_pairs


def distances(x, fingerprints):
    squared = ((x * x).sum(1)[:, None] +
               (fingerprints * fingerprints).sum(1)[None, :] - 2 * x @ fingerprints.T)
    return np.maximum(squared, 0)


def soft_predict(x, fingerprints, positions, beta):
    d = distances(x, fingerprints)
    weight = np.exp(-beta * (d - d.min(axis=1, keepdims=True)))
    return weight @ positions / weight.sum(axis=1, keepdims=True)


def mde(predicted, truth):
    return float(np.linalg.norm(predicted - truth, axis=1).mean())


def update_map(floor, support, positions, fingerprints, length_scale, shrinkage, support_xy=None):
    """用十个锚点的信号差，在旧图坐标上生成局部更新后的指纹。"""
    known_xy = floor["xy"][support] if support_xy is None else support_xy
    anchors, groups = np.unique(known_xy, axis=0, return_inverse=True)
    scans = rss_features(floor["rssi"][support])
    means = np.zeros((len(anchors), scans.shape[1]), dtype=np.float32)
    np.add.at(means, groups, scans)
    means /= np.bincount(groups)[:, None]
    closest = ((anchors[:, None, :] - positions[None, :, :]) ** 2).sum(2).argmin(1)
    signal_difference = means - fingerprints[closest]
    spatial_d2 = ((positions[:, None, :] - anchors[None, :, :]) ** 2).sum(2)
    weight = np.exp(-spatial_d2 / (2 * length_scale ** 2)).astype(np.float32)
    correction = weight @ signal_difference / (shrinkage + weight.sum(1, keepdims=True))
    return np.clip(fingerprints + correction, 0, 1).astype(np.float32)


def best_pair(candidates, scores):
    return min(candidates, key=lambda pair: (scores[key(pair)], -pair[1], pair[0]))


def key(pair):
    return f"{pair[0]}m_tau{pair[1]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/spatial_radiomap.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/spatial_radiomap_source"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    train_path = Path(config["data_path"])
    floors = load_floors(train_path)
    validation = load_floors(train_path.with_name("validationData.csv"))
    pairs = source_pairs(floors, set(config["targets"]))
    candidates = [(length, shrink) for length in config["length_scales_m"]
                  for shrink in config["shrinkage"]]
    beta = config["beta"]
    rng = np.random.default_rng(config["source_episode_seed"])
    pair_scores = {}
    saved_episodes = {}
    for lower, upper in pairs:
        name = f"{lower}->{upper}"
        old = floors[lower]
        new = floors[upper]
        positions, fingerprints = make_map(old)
        x = rss_features(new["rssi"])
        original = soft_predict(x, fingerprints, positions, beta)
        source_scores = {key(pair): [] for pair in candidates}
        baseline_scores = []
        manifests = []
        for _ in range(config["source_episodes_per_pair"]):
            support, query = episode(new, rng, config["support_positions"],
                                     config["scans_per_position"])
            manifests.append({"support_row_ids": new["row_ids"][support].tolist(),
                              "query_row_ids": new["row_ids"][query].tolist()})
            baseline_scores.append(mde(original[query], new["xy"][query]))
            for pair in candidates:
                updated = update_map(new, support, positions, fingerprints, *pair)
                predicted = soft_predict(x[query], updated, positions, beta)
                source_scores[key(pair)].append(mde(predicted, new["xy"][query]))
        pair_scores[name] = {"baseline_mde_m": float(np.mean(baseline_scores)),
                             **{k: float(np.mean(v)) for k, v in source_scores.items()}}
        saved_episodes[name] = manifests
        print(name, "baseline", round(pair_scores[name]["baseline_mde_m"], 3),
              "best", key(best_pair(candidates, pair_scores[name])),
              round(min(pair_scores[name][key(p)] for p in candidates), 3), flush=True)

    # 留出整对楼层转移，再由其它六对选择更新尺度。
    oof = {}
    for name, row in pair_scores.items():
        training_scores = {key(pair): float(np.mean([
            score[key(pair)] for other, score in pair_scores.items() if other != name
        ])) for pair in candidates}
        selected = best_pair(candidates, training_scores)
        old = row["baseline_mde_m"]
        new = row[key(selected)]
        oof[name] = {"selected": key(selected), "baseline_mde_m": old,
                     "updated_mde_m": new, "improvement_m": old - new}
    oof_gain = float(np.mean([row["improvement_m"] for row in oof.values()]))
    count = sum(row["improvement_m"] > 0 for row in oof.values())
    all_scores = {key(pair): float(np.mean([row[key(pair)] for row in pair_scores.values()]))
                  for pair in candidates}
    final_pair = best_pair(candidates, all_scores)

    # 历史源楼层的较晚官方验证只作门槛，不能调整已选参数。
    official = {}
    for lower, upper in pairs:
        name = f"{lower}->{upper}"
        new = floors[upper]
        positions, fingerprints = make_map(floors[lower])
        val_x = rss_features(validation[upper]["rssi"])
        baseline = mde(soft_predict(val_x, fingerprints, positions, beta), validation[upper]["xy"])
        ids = {int(row): i for i, row in enumerate(new["row_ids"])}
        updated_scores = []
        for item in saved_episodes[name]:
            support = np.array([ids[row] for row in item["support_row_ids"]])
            updated = update_map(new, support, positions, fingerprints, *final_pair)
            updated_scores.append(mde(soft_predict(val_x, updated, positions, beta),
                                      validation[upper]["xy"]))
        changed = float(np.mean(updated_scores))
        official[name] = {"baseline_mde_m": baseline, "updated_mde_m": changed,
                          "improvement_m": baseline - changed}
    official_gain = float(np.mean([row["improvement_m"] for row in official.values()]))
    gate = oof_gain >= 1 and count >= 5 and official_gain >= 0
    result = {"selected": key(final_pair), "pair_scores_mde_m": pair_scores,
              "leave_one_transition_out": oof, "oof_equal_pair_gain_m": oof_gain,
              "oof_improved_pairs": count, "source_official_validation": official,
              "source_official_equal_pair_gain_m": official_gain,
              "target_evaluation_gate_passed": gate}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "source_episodes.json").write_text(json.dumps(saved_episodes) + "\n")
    print("SELECTED", key(final_pair), "OOF_GAIN", round(oof_gain, 3),
          "PAIRS", count, "OFFICIAL_GAIN", round(official_gain, 3),
          "GATE", gate, flush=True)


if __name__ == "__main__":
    main()
