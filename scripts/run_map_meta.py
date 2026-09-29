"""历史跨楼层转移训练 MapMeta，整层留出选择后测试固定目标。"""
import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from torch.nn import functional as F

from data.episodes import prepare_floor, sample_episode, target_split
from data.uji import load_floors
from models.map_meta import corrected, encoder


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def lower_map(raw, floor, scale):
    """旧楼层每个物理位置聚合全部扫描，形成位置与平均指纹。"""
    positions, inverse = np.unique(raw["xy"], axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    features = np.zeros((len(positions), 520), dtype=np.float32)
    np.add.at(features, inverse, floor["x"].cpu().numpy())
    features /= counts[:, None]
    origin = positions.mean(axis=0)
    return {"fingerprints": torch.as_tensor(features, device=floor["x"].device),
            "positions": torch.as_tensor((positions - origin) / scale, dtype=torch.float32,
                                         device=floor["x"].device),
            "origin": torch.as_tensor(origin, dtype=torch.float64, device=floor["x"].device)}


@torch.no_grad()
def map_base(old_map, x, beta):
    """输出相对于旧地图原点的坐标，地图不读取新楼层标签。"""
    prototype = old_map["fingerprints"]
    predicted = []
    for block in x.split(256):
        distance = (block.square().sum(1, keepdim=True) + prototype.square().sum(1)[None, :]
                    - 2 * block @ prototype.T).clamp_min(0)
        predicted.append(torch.softmax(-beta * distance, dim=1) @ old_map["positions"])
    return torch.cat(predicted)


def source_pairs(floors, config, development):
    available = sorted(set(floors) - set(config["targets"]))
    if development:
        available = [name for name in available if name not in config["development_floors"]]
    return [(f"{upper[:2]}F{int(upper[-1]) - 1}", upper) for upper in available
            if f"{upper[:2]}F{int(upper[-1]) - 1}" in available]


def select_beta(raw, floors, pairs, config):
    maps = {lower: lower_map(raw[lower], floors[lower], config["coordinate_scale_m"])
            for lower, _ in pairs}
    scores = {}
    for beta in config["beta_candidates"]:
        per_pair = {}
        for lower, upper in pairs:
            old = maps[lower]
            base = map_base(old, floors[upper]["x"], beta)
            truth = ((floors[upper]["xy"] - old["origin"]) /
                     config["coordinate_scale_m"]).float()
            per_pair[f"{lower}->{upper}"] = float(torch.linalg.vector_norm(base-truth, dim=1).mean().item()
                                                   * config["coordinate_scale_m"])
        scores[str(beta)] = {"pair_mde": per_pair, "macro_mde": mean(per_pair.values())}
    beta = min(config["beta_candidates"], key=lambda candidate: (scores[str(candidate)]["macro_mde"], candidate))
    return beta, scores


def make_pair(raw, floors, lower, upper, beta, config):
    old = lower_map(raw[lower], floors[lower], config["coordinate_scale_m"])
    current = floors[upper]
    return {"floor": current, "map": old,
            "base": map_base(old, current["x"], beta),
            "truth": ((current["xy"] - old["origin"]) / config["coordinate_scale_m"]).float()}


def make_episode(pair, rng, config):
    floor = pair["floor"]
    support, query = sample_episode(floor, rng, config["positions_per_task"],
                                    config["support_per_position"], config["query_per_position"],
                                    unseen_query=True)
    return (floor["x"][support], pair["base"][support], pair["truth"][support],
            floor["x"][query], pair["base"][query], pair["truth"][query])


def dev_splits(pairs, config):
    splits = {}
    for floor_index, name in enumerate(config["development_floors"]):
        floor = pairs[name]["floor"]
        for split_index in range(config["development_splits"]):
            rng = np.random.default_rng(20000 + 1000 * floor_index + split_index)
            support, _, unseen = target_split(floor, rng, config["positions_per_task"],
                                              config["support_per_position"], config["query_per_position"])
            splits[f"{name}/{split_index}"] = (support, unseen)
    return splits


@torch.no_grad()
def evaluate_dev(model, pairs, splits, config, penalty):
    per_floor = {name: [] for name in config["development_floors"]}
    for key, (support, query) in splits.items():
        name = key.split("/")[0]
        pair = pairs[name]
        floor = pair["floor"]
        predicted = corrected(model, floor["x"][support], pair["base"][support],
                              pair["truth"][support], floor["x"][query],
                              pair["base"][query], penalty)
        mde = torch.linalg.vector_norm(predicted-pair["truth"][query], dim=1).mean().item()
        per_floor[name].append(mde * config["coordinate_scale_m"])
    return {name: mean(values) for name, values in per_floor.items()}


@torch.no_grad()
def evaluate_map_only(pairs, splits, config):
    per_floor = {name: [] for name in config["development_floors"]}
    for key, (_, query) in splits.items():
        name = key.split("/")[0]
        pair = pairs[name]
        mde = torch.linalg.vector_norm(pair["base"][query]-pair["truth"][query], dim=1).mean().item()
        per_floor[name].append(mde * config["coordinate_scale_m"])
    return {name: mean(values) for name, values in per_floor.items()}


def train(pairs, config, seed, penalty, checkpoints, progress_path):
    """每轮抽四个历史转移；源 Query 残差目标通过线性求解反传。"""
    torch.manual_seed(seed)
    model = encoder().to(config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["outer_lr"])
    rng = np.random.default_rng(seed)
    names = sorted(pairs)
    snapshots = {}
    with progress_path.open("w") as progress:
        for step in range(1, max(checkpoints)+1):
            chosen = rng.choice(names, config["tasks_per_iteration"], replace=False)
            tasks = [make_episode(pairs[name], rng, config) for name in chosen]
            optimizer.zero_grad()
            loss = torch.stack([
                F.mse_loss(corrected(model, sx, sb, sy, qx, qb, penalty), qy)
                for sx, sb, sy, qx, qb, qy in tasks
            ]).mean()
            loss.backward()
            optimizer.step()
            if step in checkpoints:
                snapshots[step] = deepcopy(model.state_dict())
                progress.write(json.dumps({"step": step, "query_mse": float(loss.item())})+"\n")
                progress.flush()
                print(f"seed={seed} lambda={penalty} step={step} loss={loss.item():.5f}", flush=True)
    return model, snapshots


def development(raw, floors, config, root):
    train_transfers = source_pairs(floors, config, development=True)
    assert len(train_transfers) == 4
    beta, beta_scores = select_beta(raw, floors, train_transfers, config)
    save_json(root / "beta_selection.json", {"train_transfers": train_transfers,
                                              "beta": beta, "scores": beta_scores})
    train_pairs = {upper: make_pair(raw, floors, lower, upper, beta, config)
                   for lower, upper in train_transfers}
    dev_pairs = {upper: make_pair(raw, floors, f"{upper[:2]}F{int(upper[-1])-1}",
                                 upper, beta, config)
                 for upper in config["development_floors"]}
    splits = dev_splits(dev_pairs, config)
    save_json(root / "development_manifest.json", {
        key: {"support": dev_pairs[key.split("/")[0]]["floor"]["row_ids"][support].tolist(),
              "unseen": dev_pairs[key.split("/")[0]]["floor"]["row_ids"][query].tolist()}
        for key, (support, query) in splits.items()})
    baseline = evaluate_map_only(dev_pairs, splits, config)
    records = []
    for penalty in config["lambda_candidates"]:
        for seed in config["development_seeds"]:
            model, snapshots = train(train_pairs, config, seed, penalty,
                                     config["checkpoints"], root / f"lambda_{penalty}_seed_{seed}.jsonl")
            for step, weights in snapshots.items():
                model.load_state_dict(weights)
                per_floor = evaluate_dev(model, dev_pairs, splits, config, penalty)
                record = {"lambda": penalty, "step": step, "seed": seed,
                          "floor_mde": per_floor, "macro_mde": mean(per_floor.values())}
                records.append(record)
                print("development", record, flush=True)
    choices = []
    for penalty in config["lambda_candidates"]:
        for step in config["checkpoints"]:
            selected = [r for r in records if r["lambda"] == penalty and r["step"] == step]
            floors_mean = {name: mean(r["floor_mde"][name] for r in selected)
                           for name in config["development_floors"]}
            choices.append({"lambda": penalty, "step": step, "floor_mde": floors_mean,
                            "macro_mde": mean(floors_mean.values())})
    chosen = min(choices, key=lambda r: (r["macro_mde"], r["step"], r["lambda"]))
    delta = {name: baseline[name]-chosen["floor_mde"][name]
             for name in config["development_floors"]}
    gate = mean(delta.values()) >= 1 and sum(value > 0 for value in delta.values()) >= 2
    save_json(root / "development_results.json", {"beta": beta, "map_only_mde": baseline,
                                                    "scores": records, "options": choices,
                                                    "selected": chosen, "map_minus_meta": delta,
                                                    "passed_gate": gate})
    save_json(root / "development.completed.json", {"selected": chosen,
                                                       "passed_gate": gate})
    print("selection", {"beta": beta, "map_only": baseline, "chosen": chosen,
                        "map_minus_meta": delta, "passed_gate": gate}, flush=True)


def assess_target(model, pair, validation, support, matched, unseen, penalty, config):
    floor = pair["floor"]
    metrics = {}
    for label, current, rows in (("support", pair, support),
                                 ("same_position", pair, matched),
                                 ("unseen_position", pair, unseen),
                                 ("official_validation", validation, np.arange(len(validation["floor"]["x"])))):
        x = current["floor"]["x"][rows]
        base = current["base"][rows]
        truth = current["truth"][rows]
        with torch.no_grad():
            corrected_xy = corrected(model, floor["x"][support], pair["base"][support],
                                     pair["truth"][support], x, base, penalty)
            metrics[label] = {
                "map_only": float(torch.linalg.vector_norm(base-truth, dim=1).mean().item()
                                  * config["coordinate_scale_m"]),
                "MapMeta": float(torch.linalg.vector_norm(corrected_xy-truth, dim=1).mean().item()
                                 * config["coordinate_scale_m"])}
    return metrics


def final(raw, floors, validation_floors, config, root, development_root):
    development_result = json.loads((development_root / "development_results.json").read_text())
    if not development_result["passed_gate"]:
        raise ValueError("Historical held-floor gate failed; target evaluation is not authorized by the method")
    beta = development_result["beta"]
    chosen = development_result["selected"]
    transfers = source_pairs(floors, config, development=False)
    assert len(transfers) == 7
    train_pairs = {upper: make_pair(raw, floors, lower, upper, beta, config)
                   for lower, upper in transfers}
    all_records = []
    for seed in config["final_seeds"]:
        out = root / f"seed_{seed}"
        out.mkdir()
        model, _ = train(train_pairs, config, seed, chosen["lambda"], [chosen["step"]],
                         out / "source_progress.jsonl")
        torch.save(model.state_dict(), out / "model.pt")
        manifest = json.loads((Path(config["reference_manifest_root"]) /
                               f"seed_{seed}" / "manifest.json").read_text())
        seed_records = []
        for name in config["targets"]:
            lower = f"{name[:2]}F{int(name[-1])-1}"
            pair = make_pair(raw, floors, lower, name, beta, config)
            current_val = validation_floors[name]
            validation_pair = {"floor": current_val, "map": pair["map"],
                               "base": map_base(pair["map"], current_val["x"], beta),
                               "truth": ((current_val["xy"]-pair["map"]["origin"]) /
                                         config["coordinate_scale_m"]).float()}
            ids = {int(row_id): i for i, row_id in enumerate(pair["floor"]["row_ids"])}
            support = np.array([ids[i] for i in manifest[name]["support"]])
            matched = np.array([ids[i] for i in manifest[name]["same_position_query"]])
            unseen = np.array([ids[i] for i in manifest[name]["unseen_position_query"]])
            assert not ({pair["floor"]["position_keys"][i] for i in support} &
                        {pair["floor"]["position_keys"][i] for i in unseen})
            results = assess_target(model, pair, validation_pair, support, matched, unseen,
                                    chosen["lambda"], config)
            record = {"seed": seed, "target": name, "metrics": results}
            assert all(np.isfinite(value) for row in results.values() for value in row.values())
            seed_records.append(record)
            all_records.append(record)
            print("target", record, flush=True)
        save_json(out / "target_results.json", seed_records)
        save_json(out / "completed.json", {"seed": seed, "steps": chosen["step"],
                                           "targets": config["targets"]})
    save_json(root / "target_results.json", all_records)
    write_report(all_records, config, beta, chosen, development_result)
    save_json(root / "final.completed.json", {"beta": beta, "selection": chosen,
                                               "seeds": config["final_seeds"]})


def write_report(records, config, beta, chosen, dev):
    lines = ["# MapMeta 跨楼层少样本结果", "",
             f"源转移选 β={beta}；历史整层留出选 λ={chosen['lambda']}、训练 {chosen['step']} 轮。"
             "新楼层 10 位置×3 扫描，目标梯度步数 0。", "",
             f"开发楼层零标注地图平均 {mean(dev['map_only_mde'].values()):.3f} m；"
             f"MapMeta 平均 {chosen['macro_mde']:.3f} m。", "",
             "|目标|评价|旧地图零标注 (m)|MapMeta 30标注 (m)|MapMeta−旧地图 (m)|",
             "|---|---|---:|---:|---:|"]
    for target in config["targets"]:
        for key, label in (("support", "Support"), ("same_position", "同位置"),
                           ("unseen_position", "异位置"),
                           ("official_validation", "官方 validation")):
            selected = [r["metrics"][key] for r in records if r["target"] == target]
            old = [row["map_only"] for row in selected]
            new = [row["MapMeta"] for row in selected]
            delta = [n-o for n,o in zip(new,old)]
            fmt = lambda xs: f"{mean(xs):.3f} ± {stdev(xs):.3f}" if max(xs)-min(xs)>1e-8 else f"{mean(xs):.3f}"
            lines.append(f"|{target}|{label}|{fmt(old)}|{fmt(new)}|{fmt(delta)}|")
    lines += ["", "负的差值表示 30 条目标标注有增益。相邻旧楼层完整地图是两方法共享的输入；MapMeta 的收益只能由配对差衡量。",
              "目标三个楼层已被前期工作反复查看，不能视为盲测。官方 validation 的时间、设备和用户变化混合，未单独控制。",
              "这不是 FeMLoc、MetaLoc 或 R2-D2 原文精确复现。", ""]
    Path("docs/MAPMETA_RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("development", "final"))
    parser.add_argument("--config", default="configs/map_meta.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--development-output", type=Path, default=Path("outputs/map_meta/development"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "config.json", config)
    save_json(args.output / "provenance.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_sha256": hashlib.sha256(Path(config["data_path"]).read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(Path(config["validation_path"]).read_bytes()).hexdigest()})
    raw = load_floors(config["data_path"])
    floors = {name: prepare_floor(value, config["device"]) for name, value in raw.items()}
    if args.phase == "development":
        development(raw, floors, config, args.output)
    else:
        validation_raw = load_floors(config["validation_path"])
        validation = {name: prepare_floor(value, config["device"])
                      for name, value in validation_raw.items()}
        final(raw, floors, validation, config, args.output, args.development_output)


if __name__ == "__main__":
    main()
