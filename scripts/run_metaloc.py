"""运行 MetaLoc 式少样本源域训练与目标楼层逐步适应。"""
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
from torch.func import functional_call

from data.episodes import episode_tensors, prepare_floor, sample_episode, target_split
from data.uji import load_floors
from models.rss_maml import adapted_parameters, network, predictions_after_steps


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def source_train(floors, config, seed, out):
    """四源 episode/轮：MI 优化适应后 Query，TL 普通源监督。"""
    torch.manual_seed(seed)
    initial = network(config["hidden"]).to(config["device"])
    mi, tl = deepcopy(initial), deepcopy(initial)
    mi_optim = torch.optim.Adam(mi.parameters(), lr=config["outer_lr"])
    tl_optim = torch.optim.Adam(tl.parameters(), lr=config["outer_lr"])
    rng = np.random.default_rng(seed)
    names = [name for name in sorted(floors) if name not in config["targets"]]
    assert len(names) == 10
    with (out / "source_loss.jsonl").open("w") as stream:
        for iteration in range(1, config["source_iterations"] + 1):
            tasks = []
            for name in rng.choice(names, config["tasks_per_iteration"], replace=False):
                floor = floors[name]
                support, query = sample_episode(floor, rng, config["positions_per_task"],
                                                config["support_per_position"], config["query_per_position"],
                                                unseen_query=True)
                tasks.append(episode_tensors(floor, support, query, config["coordinate_scale_m"]))

            mi_optim.zero_grad()
            losses = []
            for sx, sy, qx, qy in tasks:
                fast = adapted_parameters(mi, sx, sy, config["inner_steps"],
                                          config["inner_lr"], second_order=True)
                losses.append(F.mse_loss(functional_call(mi, fast, (qx,)), qy))
            mi_loss = torch.stack(losses).mean()
            mi_loss.backward()
            mi_optim.step()

            tl_optim.zero_grad()
            tl_loss = torch.stack([F.mse_loss(tl(torch.cat((sx, qx))), torch.cat((sy, qy)))
                                   for sx, sy, qx, qy in tasks]).mean()
            tl_loss.backward()
            tl_optim.step()

            if iteration == 1 or iteration % 100 == 0 or iteration == config["source_iterations"]:
                record = {"iteration": iteration, "MI_query_mse": mi_loss.item(),
                          "TL_source_mse": tl_loss.item()}
                assert all(np.isfinite(v) for v in record.values())
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(f"seed={seed} iteration={iteration} MI={mi_loss.item():.5f} TL={tl_loss.item():.5f}", flush=True)
    torch.save({"MI": mi.state_dict(), "TL": tl.state_dict(), "RI": initial.state_dict()}, out / "source_models.pt")
    return {"MI": mi, "TL": tl, "RI": initial}


def mean_distance(predictions, truth, scale):
    return [float(np.linalg.norm((p - truth) * scale, axis=1).mean()) for p in predictions]


def test_targets(floors, config, seed, out, models):
    """目标端只用 30 条 Support 更新，分开评价已采样与未采样位置。"""
    curves, manifest = {}, {}
    for target_index, name in enumerate(config["targets"]):
        floor = floors[name]
        rng = np.random.default_rng(seed + 10000 * (target_index + 1))
        support, matched, unseen = target_split(floor, rng, config["positions_per_task"],
                                                 config["support_per_position"], config["query_per_position"])
        sx, sy, mx, my = episode_tensors(floor, support, matched, config["coordinate_scale_m"])
        ux = floor["x"][unseen]
        uy = ((floor["xy"][unseen] - floor["xy"][support].mean(0)) /
              config["coordinate_scale_m"]).float()
        manifest[name] = {"support": floor["row_ids"][support].tolist(),
                          "same_position_query": floor["row_ids"][matched].tolist(),
                          "unseen_position_query": floor["row_ids"][unseen].tolist()}
        print(f"seed={seed} target={name} support={len(support)} matched={len(matched)} unseen={len(unseen)}", flush=True)
        qx = torch.cat((mx, ux))
        for method, trained in models.items():
            model = deepcopy(trained)
            predictions = predictions_after_steps(model, sx, sy, qx,
                                                  config["target_steps"], config["inner_lr"])
            matched_mde = mean_distance([p[:len(matched)] for p in predictions],
                                        my.cpu().numpy(), config["coordinate_scale_m"])
            unseen_mde = mean_distance([p[len(matched):] for p in predictions],
                                       uy.cpu().numpy(), config["coordinate_scale_m"])
            curves[f"{name}/{method}"] = {"same_position_mde": matched_mde,
                                          "unseen_position_mde": unseen_mde}
            print(f"seed={seed} target={name} {method} step10 unseen={unseen_mde[-1]:.3f}m", flush=True)
    save_json(out / "manifest.json", manifest)
    save_json(out / "curves.json", curves)
    return curves


def write_report(root, config):
    lines = ["# MetaLoc 式少样本跨楼层结果", "",
             "目标标注固定为 10 个位置×3 条扫描。每条曲线为同一目标划分上 0–10 步的适应结果；异位置评价为主指标。", "",
             "|目标|步数|评价|RI MDE (m)|TL MDE (m)|MI MDE (m)|MI−TL (m)|",
             "|---|---:|---|---:|---:|---:|---:|"]
    all_curves = {seed: json.loads((root / f"seed_{seed}" / "curves.json").read_text())
                  for seed in config["seeds"]}
    for name in config["targets"]:
        for step in (0, 1, 3, 5, 10):
            for key, label in (("unseen_position_mde", "异位置"), ("same_position_mde", "同位置")):
                values = {method: [all_curves[seed][f"{name}/{method}"][key][step]
                                   for seed in config["seeds"]] for method in ("RI", "TL", "MI")}
                delta = [a - b for a, b in zip(values["MI"], values["TL"])]
                fmt = lambda xs: f"{mean(xs):.3f} ± {stdev(xs):.3f}"
                lines.append(f"|{name}|{step}|{label}|{fmt(values['RI'])}|{fmt(values['TL'])}|"
                             f"{fmt(values['MI'])}|{fmt(delta)}|")
    lines += ["", "负的 MI−TL 表示 MI 误差更小。三个随机种子只能显示当前目标与预算下的趋势。",
              "全部目标数据权限、划分与方法差异见 `docs/METALOC_FEWSHOT_DESIGN.md` 和各 seed 的 manifest。",
              "源训练复用 MetaLoc 的 episode/内外层思想；UJI 回归网络与数据划分为本分支设计，不是作者代码的原样运行。", ""]
    Path("docs/METALOC_FEWSHOT_RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/metaloc_fewshot.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "config.json", config)
    save_json(args.output / "provenance.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_sha256": hashlib.sha256(Path(config["data_path"]).read_bytes()).hexdigest(),
        "metaloc_reference": "StatFusion/MetaLoc@2a3f7ae6dffcf7ebfc72a82b8091cc23db1aead9"})
    floors = {name: prepare_floor(raw, config["device"])
              for name, raw in load_floors(config["data_path"]).items()}
    for seed in config["seeds"]:
        out = args.output / f"seed_{seed}"
        out.mkdir()
        models = source_train(floors, config, seed, out)
        test_targets(floors, config, seed, out, models)
        save_json(out / "completed.json", {"seed": seed, "source_iterations": config["source_iterations"],
                                           "target_steps": config["target_steps"]})
    write_report(args.output, config)
    save_json(args.output / "completed.json", {"status": "complete", "seeds": config["seeds"],
                                               "targets": config["targets"]})


if __name__ == "__main__":
    main()
