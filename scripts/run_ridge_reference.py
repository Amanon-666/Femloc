"""在历史楼层选定方法参数后重训，自动评价固定的少量标定任务。"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import subprocess

import torch
from torch import nn

from data.episodes import prepare_floor
from data.uji import load_floors
from evaluation.ridge_reference import assess, development_mde, full_map, make_splits, manifest, save_json, summarize
from models.ridge_meta import encoder
from scripts.report_ridge_reference import write_report
from training.ridge_reference import train_encoder


def development(floors, config, root):
    """各方法独立选 λ 和源训练步数，评价楼层、位置划分完全相同。"""
    names = sorted(set(floors) - set(config["targets"]) - set(config["development_floors"]))
    assert len(names) == 7
    splits = make_splits(floors, config["development_floors"], config["development_splits"], config, 0)
    save_json(root / "manifest.json", manifest(floors, splits))
    scores = []

    def score(model, method, seed, step, penalty):
        floor_mde = development_mde(model, floors, splits, config, penalty)
        record = {"method": method, "seed": seed, "step": step, "lambda": penalty,
                  "floor_mde": floor_mde, "macro_mde": mean(floor_mde.values())}
        scores.append(record)
        print("development", record, flush=True)

    for penalty in config["lambda_candidates"]:
        for seed in config["development_seeds"]:
            model, snapshots = train_encoder(floors, names, config, seed, "Meta-Ridge", penalty,
                                              config["checkpoints"], root / f"meta_{penalty}_seed_{seed}.jsonl")
            for step, state in snapshots.items():
                model.load_state_dict(state)
                score(model, "Meta-Ridge", seed, step, penalty)
                if step == 0:
                    score(model, "RI-Ridge", seed, 0, penalty)
    for seed in config["development_seeds"]:
        model, snapshots = train_encoder(floors, names, config, seed, "Sup-Ridge", None,
                                          config["checkpoints"], root / f"sup_seed_{seed}.jsonl")
        for step, state in snapshots.items():
            model.load_state_dict(state)
            for penalty in config["lambda_candidates"]:
                score(model, "Sup-Ridge", seed, step, penalty)
    for penalty in config["lambda_candidates"]:
        score(nn.Identity(), "Raw-Ridge", 0, 0, penalty)
    options = []
    for method in ("Meta-Ridge", "Sup-Ridge", "RI-Ridge", "Raw-Ridge"):
        for penalty in config["lambda_candidates"]:
            steps = config["checkpoints"] if method in ("Meta-Ridge", "Sup-Ridge") else [0]
            for step in steps:
                matched = [row["macro_mde"] for row in scores if row["method"] == method
                           and row["lambda"] == penalty and row["step"] == step]
                options.append({"method": method, "lambda": penalty, "source_steps": step,
                                "mean_mde_m": mean(matched), "seed_mde_m": matched})
    selected = {method: min((row for row in options if row["method"] == method),
                             key=lambda row: (row["mean_mde_m"], row["source_steps"], row["lambda"]))
                for method in ("Meta-Ridge", "Sup-Ridge", "RI-Ridge", "Raw-Ridge")}
    selection = {"source_floors": names, "options": options, "selected": selected}
    save_json(root / "scores.json", scores)
    save_json(root / "selection.json", selection)
    save_json(root / "completed.json", {"selected": selected})
    print("selected", selected, flush=True)
    return selection


def final(floors, config, root, selection):
    """使用十个源楼层训练；最终验证数据在开发选择完成后才加载。"""
    validation = {name: prepare_floor(raw, config["device"])
                  for name, raw in load_floors(config["validation_path"]).items()}
    names = sorted(set(floors) - set(config["targets"]))
    assert len(names) == 10
    splits = make_splits(floors, config["targets"], config["final_splits"], config, 100000)
    save_json(root / "manifest.json", manifest(floors, splits))
    maps = {target: full_map(floors[f"{target[:2]}F{int(target[3:]) - 1}"])
            for target in config["targets"]}
    results = []
    for seed in config["final_seeds"]:
        out = root / f"seed_{seed}"
        out.mkdir()
        torch.manual_seed(seed)
        models = {"RI-Ridge": encoder(config["hidden"]).to(config["device"]), "Raw-Ridge": nn.Identity()}
        for method in ("Meta-Ridge", "Sup-Ridge"):
            pick = selection["selected"][method]
            models[method], _ = train_encoder(floors, names, config, seed, method, pick["lambda"],
                                               [pick["source_steps"]], out / f"{method}.jsonl")
        torch.save({method: model.state_dict() for method, model in models.items()}, out / "source_models.pt")
        seed_results = []
        for key, (support, query) in splits.items():
            target, split = key.split("/")
            for method in (*models, "Support-WKNN", "Old-Map"):
                penalty = selection["selected"][method]["lambda"] if method in models else None
                records = assess(models.get(method), method, floors[target], validation[target], support, query,
                                 config, penalty, maps[target])
                for metrics in records:
                    seed_results.append({"seed": seed, "target": target, "split": int(split),
                                         "method": method, **metrics})
        save_json(out / "results.json", seed_results)
        save_json(out / "completed.json", {"seed": seed, "source_floors": names,
                                           "source_steps": {m: selection["selected"][m]["source_steps"]
                                                            for m in ("Meta-Ridge", "Sup-Ridge")},
                                           "records": len(seed_results)})
        results.extend(seed_results)
        print(f"final seed={seed} completed {len(seed_results)} metric records", flush=True)
    save_json(root / "results.json", results)
    summary = summarize(results, config, selection)
    save_json(root / "summary.json", summary)
    write_report(root, results, summary, config)
    save_json(root / "completed.json", {"seeds": config["final_seeds"], "targets": config["targets"],
                                        "splits_per_target": config["final_splits"], "records": len(results)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ridge_reference.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "config.json", config)
    save_json(args.output / "provenance.json", {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "training_sha256": hashlib.sha256(Path(config["data_path"]).read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256(Path(config["validation_path"]).read_bytes()).hexdigest()})
    floors = {name: prepare_floor(raw, config["device"])
              for name, raw in load_floors(config["data_path"]).items()}
    development_root, final_root = args.output / "development", args.output / "final"
    development_root.mkdir()
    final_root.mkdir()
    selection = development(floors, config, development_root)
    final(floors, config, final_root, selection)
    save_json(args.output / "completed.json", {"development": True, "final": True})


if __name__ == "__main__":
    main()
