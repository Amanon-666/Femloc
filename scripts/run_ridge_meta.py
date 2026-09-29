"""先用历史整层留出选择，再训练和评价 RidgeMeta 少样本定位。"""
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
from models.ridge_meta import encoder, ridge_predict
from models.rss_maml import network


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_data(config):
    floors = {name: prepare_floor(raw, config["device"])
              for name, raw in load_floors(config["data_path"]).items()}
    validation = {name: prepare_floor(raw, config["device"])
                  for name, raw in load_floors(config["validation_path"]).items()}
    return floors, validation


def source_tasks(floors, names, rng, config):
    tasks = []
    for name in rng.choice(names, config["tasks_per_iteration"], replace=False):
        floor = floors[name]
        support, query = sample_episode(floor, rng, config["positions_per_task"],
                                        config["support_per_position"], config["query_per_position"],
                                        unseen_query=True)
        sx, sy, qx, qy = episode_tensors(floor, support, query, config["coordinate_scale_m"])
        tasks.append((floor, support, query, sx, sy, qx, qy))
    return tasks


def development_splits(floors, config):
    """提前固定三个历史楼层的 Support 和全部异位置 Query。"""
    splits = {}
    for floor_index, name in enumerate(config["development_floors"]):
        floor = floors[name]
        for split_index in range(config["development_splits"]):
            rng = np.random.default_rng(20000 + 1000 * floor_index + split_index)
            support, _, unseen = target_split(floor, rng, config["positions_per_task"],
                                              config["support_per_position"], config["query_per_position"])
            splits[f"{name}/{split_index}"] = (support, unseen)
    return splits


@torch.no_grad()
def evaluate_splits(model, floors, splits, config, penalty):
    by_floor = {name: [] for name in config["development_floors"]}
    for key, (support, query) in splits.items():
        name, _ = key.split("/")
        sx, sy, qx, qy = episode_tensors(floors[name], support, query,
                                         config["coordinate_scale_m"])
        prediction = ridge_predict(model, sx, sy, qx, penalty)
        mde = torch.linalg.vector_norm(prediction - qy, dim=1).mean().item()
        by_floor[name].append(mde * config["coordinate_scale_m"])
    return {name: mean(values) for name, values in by_floor.items()}


def train_meta(floors, names, config, seed, penalty, checkpoints, progress_path):
    """源 Query 监督通过 30×30 线性求解传回 RSSI 编码器。"""
    torch.manual_seed(seed)
    model = encoder(config["hidden"]).to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["outer_lr"])
    rng = np.random.default_rng(seed)
    snapshots = {}
    with progress_path.open("w") as progress:
        for step in range(1, max(checkpoints) + 1):
            tasks = source_tasks(floors, names, rng, config)
            optimizer.zero_grad()
            loss = torch.stack([
                F.mse_loss(ridge_predict(model, sx, sy, qx, penalty), qy)
                for _, _, _, sx, sy, qx, qy in tasks
            ]).mean()
            loss.backward()
            optimizer.step()
            if step in checkpoints:
                snapshots[step] = deepcopy(model.state_dict())
                record = {"step": step, "query_mse": float(loss.item())}
                progress.write(json.dumps(record) + "\n")
                progress.flush()
                print(f"meta seed={seed} lambda={penalty} step={step} loss={loss.item():.5f}", flush=True)
    return model, snapshots


def run_development(floors, config, root):
    names = sorted(set(floors) - set(config["targets"]) - set(config["development_floors"]))
    assert len(names) == 7
    splits = development_splits(floors, config)
    save_json(root / "development_manifest.json", {
        key: {"support": floors[key.split("/")[0]]["row_ids"][support].tolist(),
              "unseen": floors[key.split("/")[0]]["row_ids"][query].tolist()}
        for key, (support, query) in splits.items()
    })
    scores = []
    for penalty in config["lambda_candidates"]:
        for seed in config["development_seeds"]:
            prefix = f"lambda_{penalty}_seed_{seed}"
            model, snapshots = train_meta(floors, names, config, seed, penalty,
                                          config["checkpoints"], root / f"{prefix}.jsonl")
            for step, weights in snapshots.items():
                model.load_state_dict(weights)
                floors_mde = evaluate_splits(model, floors, splits, config, penalty)
                score = {"lambda": penalty, "step": step, "seed": seed,
                         "floor_mde": floors_mde, "macro_mde": mean(floors_mde.values())}
                scores.append(score)
                print("development", score, flush=True)
    options = []
    for penalty in config["lambda_candidates"]:
        for step in config["checkpoints"]:
            matched = [row["macro_mde"] for row in scores
                       if row["lambda"] == penalty and row["step"] == step]
            options.append({"lambda": penalty, "step": step,
                            "mean_mde": mean(matched), "seed_mde": matched})
    selected = min(options, key=lambda row: (row["mean_mde"], row["step"], row["lambda"]))
    save_json(root / "development_scores.json", scores)
    save_json(root / "selection.json", {"source_floors": names, "options": options,
                                         "selected": selected})
    save_json(root / "development.completed.json", {"selected": selected})
    print("selected", selected, flush=True)


def train_final(floors, config, seed, penalty, steps, out):
    """Meta 与普通源监督接触同一批源任务，适应时共用同一求解器。"""
    names = sorted(set(floors) - set(config["targets"]))
    assert len(names) == 10
    torch.manual_seed(seed)
    initial = network(config["hidden"]).to(config["device"])
    meta = deepcopy(initial[:-1])
    supervised = deepcopy(initial)
    random_features = deepcopy(initial[:-1])
    source_origin = torch.cat([floors[name]["xy"] for name in names]).mean(0)
    meta_opt = torch.optim.Adam(meta.parameters(), lr=config["outer_lr"])
    supervised_opt = torch.optim.Adam(supervised.parameters(), lr=config["outer_lr"])
    rng = np.random.default_rng(seed)
    with (out / "source_progress.jsonl").open("w") as progress:
        for step in range(1, steps + 1):
            tasks = source_tasks(floors, names, rng, config)
            meta_opt.zero_grad()
            meta_loss = torch.stack([
                F.mse_loss(ridge_predict(meta, sx, sy, qx, penalty), qy)
                for _, _, _, sx, sy, qx, qy in tasks
            ]).mean()
            meta_loss.backward()
            meta_opt.step()

            supervised_opt.zero_grad()
            losses = []
            for floor, support, query, sx, _, qx, _ in tasks:
                raw_y = torch.cat((floor["xy"][support], floor["xy"][query]))
                labels = ((raw_y - source_origin) / config["coordinate_scale_m"]).float()
                losses.append(F.mse_loss(supervised(torch.cat((sx, qx))), labels))
            supervised_loss = torch.stack(losses).mean()
            supervised_loss.backward()
            supervised_opt.step()
            if step == 1 or step % 100 == 0 or step == steps:
                record = {"step": step, "meta_query_mse": float(meta_loss.item()),
                          "supervised_source_mse": float(supervised_loss.item())}
                progress.write(json.dumps(record) + "\n")
                progress.flush()
                print(f"final seed={seed} step={step} meta={meta_loss.item():.5f} "
                      f"supervised={supervised_loss.item():.5f}", flush=True)
    torch.save({"RidgeMeta": meta.state_dict(), "TL-Ridge": supervised[:-1].state_dict(),
                "RI-Ridge": random_features.state_dict()}, out / "source_models.pt")
    return {"RidgeMeta": meta, "TL-Ridge": supervised[:-1], "RI-Ridge": random_features}


@torch.no_grad()
def assess(model, floor, validation, support, matched, unseen, config, penalty):
    origin = floor["xy"][support].mean(0)
    sy = ((floor["xy"][support] - origin) / config["coordinate_scale_m"]).float()
    sx = floor["x"][support]
    result = {}
    for label, data, rows in (("support", floor, support),
                              ("same_position", floor, matched),
                              ("unseen_position", floor, unseen),
                              ("official_validation", validation, np.arange(len(validation["x"])))):
        truth = ((data["xy"][rows] - origin) / config["coordinate_scale_m"]).float()
        predicted = ridge_predict(model, sx, sy, data["x"][rows], penalty)
        result[label] = float(torch.linalg.vector_norm(predicted - truth, dim=1).mean().item()
                              * config["coordinate_scale_m"])
    return result


def run_final(floors, validation, config, root, selection):
    penalty = selection["selected"]["lambda"]
    steps = selection["selected"]["step"]
    reference = Path(config["reference_manifest_root"])
    all_results = []
    for seed in config["final_seeds"]:
        out = root / f"seed_{seed}"
        out.mkdir()
        models = train_final(floors, config, seed, penalty, steps, out)
        manifest = json.loads((reference / f"seed_{seed}" / "manifest.json").read_text())
        seed_results = []
        for name in config["targets"]:
            floor = floors[name]
            by_id = {int(row_id): i for i, row_id in enumerate(floor["row_ids"])}
            support = np.array([by_id[i] for i in manifest[name]["support"]])
            matched = np.array([by_id[i] for i in manifest[name]["same_position_query"]])
            unseen = np.array([by_id[i] for i in manifest[name]["unseen_position_query"]])
            assert not ({floor["position_keys"][i] for i in support} &
                        {floor["position_keys"][i] for i in unseen})
            for method, model in models.items():
                metrics = assess(model, floor, validation[name], support, matched, unseen,
                                 config, penalty)
                record = {"seed": seed, "target": name, "method": method, **metrics}
                assert all(np.isfinite(value) for value in metrics.values())
                seed_results.append(record)
                all_results.append(record)
                print("target", record, flush=True)
        save_json(out / "target_results.json", seed_results)
        save_json(out / "completed.json", {"seed": seed, "source_steps": steps,
                                           "targets": config["targets"]})
    save_json(root / "target_results.json", all_results)
    write_report(root, all_results, selection, config)
    save_json(root / "final.completed.json", {"seeds": config["final_seeds"],
                                               "targets": config["targets"], "steps": steps})


def write_report(root, results, selection, config):
    lines = ["# RidgeMeta 跨楼层少样本实验", "",
             "每个新楼层只使用 10 个位置 × 3 扫描的 30 条标注。RidgeMeta / TL-Ridge / RI-Ridge 均用一次 Support 岭回归，目标梯度步数为 0。正则强度和源轮数只由历史整层留出选定。", "",
             f"源端选择：λ={selection['selected']['lambda']}，源轮数={selection['selected']['step']}，"
             f"开发楼层宏平均 MDE={selection['selected']['mean_mde']:.3f} m。", "",
             "|目标|评价|RidgeMeta (m)|TL-Ridge (m)|RI-Ridge (m)|Meta−TL (m)|",
             "|---|---|---:|---:|---:|---:|"]
    for name in config["targets"]:
        for key, label in (("support", "Support"), ("same_position", "同位置未用扫描"),
                           ("unseen_position", "异位置"), ("official_validation", "官方 validation")):
            by_method = {method: [r[key] for r in results if r["target"] == name and r["method"] == method]
                         for method in ("RidgeMeta", "TL-Ridge", "RI-Ridge")}
            delta = [a-b for a, b in zip(by_method["RidgeMeta"], by_method["TL-Ridge"])]
            fmt = lambda xs: f"{mean(xs):.3f} ± {stdev(xs):.3f}"
            lines.append(f"|{name}|{label}|{fmt(by_method['RidgeMeta'])}|{fmt(by_method['TL-Ridge'])}|"
                         f"{fmt(by_method['RI-Ridge'])}|{fmt(delta)}|")
    lines += ["", "负的 Meta−TL 代表 RidgeMeta 更好。三个种子的标准差只反映本次训练及固定三组目标划分，不能代表更广泛环境。",
              "官方 validation 与目标 Support 不同日，且设备、用户、采集覆盖混合变化；这里没有隔离这些因素。目标楼层已经在前期研究中多次被查看，本结果是探索性，不是盲测。",
              "主方法是受 R2-D2 启发的 UJI 回归设计，不是 FeMLoc、MetaLoc 或 R2-D2 原论文的精确复现。", ""]
    Path("docs/RIDGEMETA_RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("development", "final"))
    parser.add_argument("--config", default="configs/ridge_meta.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "provenance.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_sha256": hashlib.sha256(Path(config["data_path"]).read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(Path(config["validation_path"]).read_bytes()).hexdigest()})
    save_json(args.output / "config.json", config)
    floors, validation = load_data(config)
    if args.phase == "development":
        run_development(floors, config, args.output)
    else:
        selection = json.loads(Path(config["selection_path"]).read_text())
        run_final(floors, validation, config, args.output, selection)


if __name__ == "__main__":
    main()
