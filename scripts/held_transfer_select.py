"""整对旧楼层转移留出验证，随后训练固定步数并评估三个目标。"""
import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

from data.uji import load_floors
from scripts.evaluate_adjacent_map import make_map, source_pairs, source_predictions
from scripts.learn_cross_floor_metric import (CrossFloorMetric, calibrate_residual,
    evaluate_targets, map_predictions, prepared_pair)


@torch.no_grad()
def held_error(model, pair):
    model.eval()
    errors = []
    for block in np.array_split(np.arange(len(pair["scans"])),
                                max(1, (len(pair["scans"]) + 255) // 256)):
        pred = model(pair["scans"][block], pair["fingerprints"], pair["positions"])
        errors.append(torch.linalg.vector_norm((pred - pair["truth"][block]) * 100, dim=1))
    return float(torch.cat(errors).mean().item())


def train_step(model, optimizer, pair, rng):
    chosen = rng.choice(pair["train"], size=128, replace=True)
    pred = model(pair["scans"][chosen], pair["fingerprints"], pair["positions"])
    w = 2 * torch.sigmoid(model.ap_logits)
    loss = (pred - pair["truth"][chosen]).square().sum(dim=1).mean()
    loss = loss + 0.001 * (w - 1).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        model.log_beta.clamp_(0, math.log(100))


def model_seed(seed, device):
    torch.manual_seed(seed)
    return CrossFloorMetric().to(device)


def select_steps(pairs, prepared, device):
    """留出楼层的全部记录都不参与该 fold 的梯度更新。"""
    curves = {}
    for held, (lower, upper) in enumerate(pairs):
        train_pairs = [prepared[j] for j, (lo, hi) in enumerate(pairs)
                       if j != held and lo != upper and hi != upper]
        rng = np.random.default_rng(0)
        model = model_seed(0, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scores = {0: held_error(model, prepared[held])}
        for step in range(1, 3001):
            model.train()
            train_step(model, optimizer, train_pairs[int(rng.integers(len(train_pairs)))], rng)
            if step % 100 == 0:
                scores[step] = held_error(model, prepared[held])
        curves[f"{lower}->{upper}"] = scores
        print("held", f"{lower}->{upper}", "best", min(scores.items(), key=lambda x: (x[1], x[0])), flush=True)
    means = {step: float(np.mean([scores[step] for scores in curves.values()]))
             for step in range(0, 3001, 100)}
    chosen = min(means, key=lambda step: (means[step], step))
    return chosen, means, curves


def fit_all(prepared, steps, seed, device):
    rng = np.random.default_rng(seed)
    model = model_seed(seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(steps):
        pair = prepared[int(rng.integers(len(prepared))) ]
        train_step(model, optimizer, pair, rng)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adjacent_map.json")
    parser.add_argument("--output", type=Path, default=Path("outputs/held_transfer_v1"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(1)
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else
                          "cuda:0" if torch.cuda.is_available() else "cpu")
    floors = load_floors(config["data_path"])
    pairs = source_pairs(floors, set(config["targets"]))
    assert len(pairs) == 7
    prepared = [prepared_pair(floors[lo], floors[hi], device, 7731 + i)
                for i, (lo, hi) in enumerate(pairs)]
    for p in prepared:
        p["train"] = np.arange(len(p["scans"]))
    args.output.mkdir(parents=True, exist_ok=False)
    print("selecting by seven held source transfers", flush=True)
    chosen, means, curves = select_steps(pairs, prepared, device)
    (args.output / "source_selection.json").write_text(json.dumps({
        "selected_steps": chosen, "source_mean_mde_m": means,
        "held_transfer_curves_mde_m": curves, "pairs": pairs}, indent=2) + "\n")
    print("chosen source steps", chosen, "cross-transfer MDE", means[chosen], flush=True)
    all_rows = []
    for seed in config["seeds"]:
        model = fit_all(prepared, chosen, seed, device)
        old_predictions = {hi: map_predictions(model, floors[lo], floors[hi], device)
                           for lo, hi in pairs}
        residual_choice, residual_scores, _ = calibrate_residual(floors, old_predictions, pairs, config, seed)
        target_metric = {}
        target_hard = {}
        for target in config["targets"]:
            lower = f"{target[:2]}F{int(target[-1]) - 1}"
            target_metric[target] = map_predictions(model, floors[lower], floors[target], device)
            target_hard[target] = source_predictions(floors[target], make_map(floors[lower]))
        rows = evaluate_targets(floors, target_metric, target_hard, config, seed, residual_choice)
        all_rows.extend(rows)
        torch.save(model.state_dict(), args.output / f"seed_{seed}_metric.pt")
        (args.output / f"seed_{seed}_source_residual.json").write_text(json.dumps({
            "choice": residual_choice, "source_scores_mde_m": residual_scores,
            "beta": float(model.log_beta.exp().item())}, indent=2) + "\n")
        print("seed", seed, "residual", residual_choice,
              [(r["target"], round(r["learned_map_mde_m"], 3),
                round(r["learned_plus_anchors_mde_m"], 3)) for r in rows], flush=True)
    (args.output / "results.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    for target in config["targets"]:
        rows = [r for r in all_rows if r["target"] == target]
        print(target, {key: round(float(np.mean([r[key] for r in rows])), 3)
                       for key in ("hard_uniform_mde_m", "learned_map_mde_m", "learned_plus_anchors_mde_m")},
              flush=True)


if __name__ == "__main__":
    main()
