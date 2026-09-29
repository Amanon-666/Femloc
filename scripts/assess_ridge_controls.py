"""在历史留出楼层上检查闭式适应的普通监督和随机特征参照。"""
import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

from models.rss_maml import network
from scripts.run_ridge_meta import development_splits, evaluate_splits, load_data, save_json, source_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ridge_meta.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    selected = json.loads(Path(config["selection_path"]).read_text())["selected"]
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    floors, _ = load_data(config)
    names = sorted(set(floors) - set(config["targets"]) - set(config["development_floors"]))
    splits = development_splits(floors, config)
    origin = torch.cat([floors[name]["xy"] for name in names]).mean(0)
    results = []
    for seed in config["development_seeds"]:
        torch.manual_seed(seed)
        initial = network(config["hidden"]).to(config["device"])
        supervised = deepcopy(initial)
        random_features = deepcopy(initial[:-1])
        optimizer = torch.optim.Adam(supervised.parameters(), lr=config["outer_lr"])
        rng = np.random.default_rng(seed)
        for step in range(1, selected["step"] + 1):
            tasks = source_tasks(floors, names, rng, config)
            losses = []
            for floor, support, query, sx, _, qx, _ in tasks:
                raw_y = torch.cat((floor["xy"][support], floor["xy"][query]))
                labels = ((raw_y - origin) / config["coordinate_scale_m"]).float()
                losses.append(F.mse_loss(supervised(torch.cat((sx, qx))), labels))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for method, model in (("TL-Ridge", supervised[:-1]), ("RI-Ridge", random_features)):
            floor_mde = evaluate_splits(model, floors, splits, config, selected["lambda"])
            record = {"seed": seed, "method": method, "floor_mde": floor_mde,
                      "macro_mde": mean(floor_mde.values())}
            results.append(record)
            print(record, flush=True)
    save_json(args.output / "results.json", {"selected": selected, "controls": results})
    save_json(args.output / "completed.json", {"development_seeds": config["development_seeds"]})


if __name__ == "__main__":
    main()
