"""审计冻结 SCM v1 的信号场预测与定位收益是否一致。"""
import argparse
import json
from pathlib import Path

import numpy as np

from data.uji import load_floors
from scripts.evaluate_signal_calibrated_map import (
    NO_SIGNAL, build_map, dbm, episodes, gp_mean, lookup, lower_map, mde,
    position_map, wknn,
)

PAIRS = [(f"B{b}F{f}", f"B{b}F{f + 1}")
         for b, top in ((0, 3), (1, 3), (2, 4)) for f in range(top - 1)]
SEED = 2718
EPISODES = 12
QUERY_CAP = 300
PARAMS = (60.0, 0.3)
KNN = (8, 10)
BINS = [("near", 0, 20), ("middle", 20, 40), ("far", 40, float("inf"))]


def field_at_query(old, support_rssi, support_xy, query_xy):
    """只为离线误差审计在真实 Query 坐标处评价旧场和 SCM 场。"""
    old_proto, old_xy = old
    anchor, anchor_xy, _ = position_map(support_rssi, support_xy)
    residual = anchor - lookup(anchor_xy, old_xy, old_proto, 3)
    q_pos, q_inv = np.unique(query_xy, axis=0, return_inverse=True)
    baseline = lookup(q_pos, old_xy, old_proto, 3)
    adapted = np.clip(baseline + gp_mean(q_pos, anchor_xy, residual, *PARAMS),
                      NO_SIGNAL, 0)
    return baseline[q_inv], adapted[q_inv]


def summarize_rows(raw, truth_xy, old_field, scm_field, old_pred, scm_pred,
                   rows):
    if not np.any(rows):
        return None
    truth = dbm(raw[rows])
    detected = raw[rows] != 100
    old_err = truth - old_field[rows]
    scm_err = truth - scm_field[rows]
    return {
        "n": int(rows.sum()),
        "old_rssi_rmse_db": float(np.sqrt(np.mean(old_err ** 2))),
        "scm_rssi_rmse_db": float(np.sqrt(np.mean(scm_err ** 2))),
        "old_detected_rmse_db": float(np.sqrt(np.mean(old_err[detected] ** 2))),
        "scm_detected_rmse_db": float(np.sqrt(np.mean(scm_err[detected] ** 2))),
        "old_mde_m": mde(old_pred[rows], truth_xy[rows]),
        "scm_mde_m": mde(scm_pred[rows], truth_xy[rows]),
    }


def evaluate_pair(old, new, rng, config):
    records = []
    for support, query in episodes(new["xy"], 10, EPISODES, 3, rng):
        query = rng.choice(query, min(QUERY_CAP, len(query)), replace=False)
        s_raw, s_xy = new["rssi"][support], new["xy"][support]
        q_raw, q_xy = new["rssi"][query], new["xy"][query]
        old_field, scm_field = field_at_query(old, s_raw, s_xy, q_xy)
        virtual = build_map("scm", old, s_raw, s_xy, PARAMS, config)
        old_pred = wknn(q_raw, *old, *KNN)
        scm_pred = wknn(q_raw, *virtual, *KNN)
        distance = np.sqrt(((q_xy[:, None] - np.unique(s_xy, axis=0)[None]) ** 2)
                           .sum(-1)).min(1)
        record = {"all": summarize_rows(q_raw, q_xy, old_field, scm_field,
                                        old_pred, scm_pred, np.ones(len(query), bool))}
        for name, left, right in BINS:
            record[name] = summarize_rows(q_raw, q_xy, old_field, scm_field,
                                          old_pred, scm_pred,
                                          (distance >= left) & (distance < right))
        records.append(record)
    return records


def aggregate(records):
    out = {}
    for group in ("all", "near", "middle", "far"):
        usable = [r[group] for r in records if r[group] is not None]
        if usable:
            out[group] = {key: float(np.mean([r[key] for r in usable]))
                          for key in usable[0] if key != "n"}
            out[group]["episodes_with_queries"] = len(usable)
            out[group]["mean_queries_per_episode"] = float(np.mean([r["n"] for r in usable]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/signal_calibrated_map.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    train = load_floors(config["train_path"])
    rng = np.random.default_rng(SEED)
    by_pair = {}
    for lower, upper in PAIRS:
        records = evaluate_pair(lower_map(train[lower]), train[upper], rng, config)
        by_pair[f"{lower}->{upper}"] = aggregate(records)
        row = by_pair[f"{lower}->{upper}"]["all"]
        print(f"{lower}->{upper}: RSSI {row['old_rssi_rmse_db']:.2f}→"
              f"{row['scm_rssi_rmse_db']:.2f} dB, MDE {row['old_mde_m']:.2f}→"
              f"{row['scm_mde_m']:.2f} m", flush=True)
    macro = {}
    for group in ("all", "near", "middle", "far"):
        usable = [x[group] for x in by_pair.values() if group in x]
        macro[group] = {key: float(np.mean([r[key] for r in usable]))
                        for key in usable[0] if key not in ("episodes_with_queries", "mean_queries_per_episode")}
    building_macro = {}
    for group in ("all", "near", "middle", "far"):
        building_rows = []
        for building in ("B0", "B1", "B2"):
            rows = [pair[group] for name, pair in by_pair.items()
                    if name.startswith(building) and group in pair]
            building_rows.append({key: float(np.mean([row[key] for row in rows]))
                                  for key in macro[group]})
        building_macro[group] = {key: float(np.mean([row[key] for row in building_rows]))
                                 for key in macro[group]}
    result = {"protocol": {"pairs": PAIRS, "seed": SEED, "episodes_per_pair": EPISODES,
                           "query_cap": QUERY_CAP, "field": PARAMS, "wknn": KNN},
              "pairs": by_pair, "equal_pair_macro": macro,
              "equal_building_macro": building_macro}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("macro", macro["all"], flush=True)


if __name__ == "__main__":
    main()
