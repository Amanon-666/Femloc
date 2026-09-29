"""训练可由少量锚点直接求坐标映射的 RSSI 表征。"""

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch.nn import functional as F

from data.episodes import episode_tensors, prepare_floor, sample_episode, target_split
from data.uji import load_floors
from models.ridge import predict
from models.rss_maml import network


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def splits(floor, rng, repeats, config):
    """预先固定少量锚点及异位置 Query，所有方法共用。"""
    items = []
    for _ in range(repeats):
        support, _, query = target_split(floor, rng, config["positions_per_task"],
                                         config["support_per_position"], config["query_per_position"])
        items.append((support, query))
    return items


def mean_distance(prediction, truth, scale):
    return float(torch.linalg.vector_norm(prediction - truth, dim=-1).mean().item() * scale)


def nearest_predict(sx, sy, qx):
    """三近邻只使用目标楼层 Support，作为没有源训练的参照。"""
    distances = torch.cdist(qx, sx)
    nearest = distances.topk(3, largest=False)
    weights = nearest.values.clamp_min(1e-8).reciprocal()
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (sy[nearest.indices] * weights[..., None]).sum(dim=1)


def evaluate(encoder, floor, support, query_x, query_y, config):
    """Query 坐标只用于最后计分，从不传入映射求解器。"""
    sx = floor["x"][support]
    origin = floor["xy"][support].mean(dim=0)
    sy = ((floor["xy"][support] - origin) / config["coordinate_scale_m"]).float()
    truth = ((query_y - origin) / config["coordinate_scale_m"]).float()
    with torch.no_grad():
        prediction = predict(encoder(sx), sy, encoder(query_x), config["relative_penalty"])
    return mean_distance(prediction, truth, config["coordinate_scale_m"])


def score_floor(encoder, floor, support, query, config):
    return evaluate(encoder, floor, support, floor["x"][query], floor["xy"][query], config)


def train_one(seed, floors, official, config, output):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = config["device"]
    encoder = network(config["hidden"])[:-1].to(device)
    initial = deepcopy(encoder).eval()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=config["outer_lr"])
    excluded = set(config["development_targets"] + config["confirmation_targets"])
    sources = [name for name in sorted(floors) if name not in excluded]
    assert len(sources) == 7

    # 完整楼层留出用于选择训练轮数；确认楼层不参与这个选择。
    split_rng = np.random.default_rng(seed + 10000)
    development = {name: splits(floors[name], split_rng, config["development_episodes"], config)
                   for name in config["development_targets"]}
    confirmation = {name: splits(floors[name], split_rng, config["confirmation_episodes"], config)
                    for name in config["confirmation_targets"]}
    manifest = {}
    for label, cases in (("development", development), ("confirmation", confirmation)):
        manifest[label] = {name: [{"support": floors[name]["row_ids"][s].tolist(),
                                  "unseen_query": floors[name]["row_ids"][q].tolist()}
                                 for s, q in tasks] for name, tasks in cases.items()}
    save_json(output / "manifest.json", manifest)

    def development_score():
        encoder.eval()
        values = [mean(score_floor(encoder, floors[name], s, q, config) for s, q in tasks)
                  for name, tasks in development.items()]
        encoder.train()
        return float(mean(values))

    history = [{"step": 0, "development_mde": development_score()}]
    best = history[0]["development_mde"]
    best_step = 0
    best_state = deepcopy(encoder.state_dict())
    for step in range(1, config["source_steps"] + 1):
        optimizer.zero_grad()
        losses = []
        for name in rng.choice(sources, config["tasks_per_step"], replace=False):
            floor = floors[name]
            support, query = sample_episode(floor, rng, config["positions_per_task"],
                                            config["support_per_position"],
                                            config["query_per_position"], unseen_query=True)
            sx, sy, qx, qy = episode_tensors(floor, support, query, config["coordinate_scale_m"])
            estimated = predict(encoder(sx), sy, encoder(qx), config["relative_penalty"])
            losses.append(F.mse_loss(estimated, qy))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if step % config["evaluation_interval"] == 0:
            value = development_score()
            history.append({"step": step, "source_query_mse": float(loss.item()),
                            "development_mde": value})
            if value < best:
                best, best_step = value, step
                best_state = deepcopy(encoder.state_dict())
            print(f"seed={seed} step={step} dev={value:.3f}m best={best:.3f}m", flush=True)
    encoder.load_state_dict(best_state)
    encoder.eval()
    torch.save(best_state, output / "best_encoder.pt")
    save_json(output / "training_curve.json", history)

    records = []
    for name, tasks in confirmation.items():
        floor = floors[name]
        for episode_id, (support, query) in enumerate(tasks):
            sx = floor["x"][support]
            origin = floor["xy"][support].mean(dim=0)
            sy = ((floor["xy"][support] - origin) / config["coordinate_scale_m"]).float()
            for set_name, qx, qxy in (("unseen_position", floor["x"][query], floor["xy"][query]),
                                      ("official_validation", official[name]["x"], official[name]["xy"])):
                truth = ((qxy - origin) / config["coordinate_scale_m"]).float()
                with torch.no_grad():
                    baselines = {
                        "trained_ridge": predict(encoder(sx), sy, encoder(qx), config["relative_penalty"]),
                        "random_ridge": predict(initial(sx), sy, initial(qx), config["relative_penalty"]),
                        "raw_ridge": predict(sx, sy, qx, config["relative_penalty"]),
                        "wknn": nearest_predict(sx, sy, qx),
                    }
                record = {"seed": seed, "target": name, "episode": episode_id,
                          "set": set_name, "support_rows": len(support), "query_rows": len(qx),
                          **{method: mean_distance(pred, truth, config["coordinate_scale_m"])
                             for method, pred in baselines.items()}}
                records.append(record)
                print(record, flush=True)
    save_json(output / "results.json", records)
    save_json(output / "completed.json", {"seed": seed, "source_steps": config["source_steps"],
                                          "best_step": best_step, "best_development_mde": best,
                                          "confirmation_episodes": config["confirmation_episodes"]})
    return records


def summarize(records, config, output):
    lines = ["# Support 条件映射的跨楼层初步结果", "",
             "源端只训练信号特征；每次预测用目标楼层 10 个位置×3 条扫描直接求坐标映射，无目标梯度步。",
             "训练轮数由三个完整留出的开发楼层选择；三层确认目标不参与训练、选择或早停。", "",
             "|目标|评价集|训练特征+岭回归|随机特征+岭回归|原始 RSSI+岭回归|WKNN|",
             "|---|---|---:|---:|---:|---:|"]
    methods = ("trained_ridge", "random_ridge", "raw_ridge", "wknn")
    for name in config["confirmation_targets"]:
        for set_name in ("unseen_position", "official_validation"):
            values = {}
            for method in methods:
                seed_means = [mean(r[method] for r in records if r["seed"] == seed
                                   and r["target"] == name and r["set"] == set_name)
                              for seed in config["seeds"]]
                values[method] = f"{mean(seed_means):.2f} ± {stdev(seed_means):.2f}"
            lines.append(f"|{name}|{set_name}|" + "|".join(values[m] for m in methods) + "|")
    lines += ["", "数字为 UJI 投影坐标中的平均二维误差；± 为三个训练种子均值的样本标准差。",
              "`unseen_position` 为 trainingData 内与 Support 位置互斥的扫描；官方 validation 同时包含时间、设备、用户及覆盖变化。",
              "这些楼层曾出现在其它分支的源训练或研究讨论中，因此这里只能作为本方法的探索性确认。", ""]
    Path("docs/SUPPORT_RIDGE_RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "config.json", config)
    save_json(args.output / "provenance.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_csv_sha256": hashlib.sha256(Path(config["data_path"]).read_bytes()).hexdigest(),
        "validation_csv_sha256": hashlib.sha256(Path(config["validation_path"]).read_bytes()).hexdigest(),
    })
    floors = {name: prepare_floor(raw, config["device"])
              for name, raw in load_floors(config["data_path"]).items()}
    official = {name: prepare_floor(raw, config["device"])
                for name, raw in load_floors(config["validation_path"]).items()
                if name in config["confirmation_targets"]}
    assert len(official) == len(config["confirmation_targets"])
    all_results = []
    for seed in config["seeds"]:
        output = args.output / f"seed_{seed}"
        output.mkdir()
        all_results.extend(train_one(seed, floors, official, config, output))
    summarize(all_results, config, args.output)
    save_json(args.output / "completed.json", {"seeds": config["seeds"],
                                                 "result_rows": len(all_results), "status": "complete"})


if __name__ == "__main__":
    main()
