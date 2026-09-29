"""执行独立EXP1：source-only AE选择、十客户端训练、三个目标配对适应。"""
import argparse
import csv
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from data.uji import Transform, load_floors, split_groups
from evaluation.metrics import evaluate, first_hit
from models.femloc import private, shared
from training.adaptation import adapt
from training.ae import calibrate, train_ae
from training.federated import Client, meta_round


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def run_seed(floors, config, selection, seed, root):
    started = perf_counter()
    out = root / f"seed_{seed}"
    out.mkdir()
    splits = {name: split_groups(floor["groups"], config["split_folds"], seed) for name, floor in floors.items()}
    write_json(out / "splits.json", {name: {"support": floors[name]["row_ids"][s].tolist(),
                                           "query": floors[name]["row_ids"][q].tolist()}
                                      for name, (s, q) in splits.items()})
    global_model = shared(config, seed)
    initial = deepcopy(global_model)
    clients = []
    for name, floor in floors.items():
        if name in config["targets"]:
            continue
        client = Client(name, floor, *splits[name], global_model, selection, config, seed)
        clients.append(client)
        print(f"seed={seed} source={name} support={len(client.support)} query={len(client.query)} AP={len(client.prep.aps)} ready", flush=True)
    write_json(out / "source_preprocessing.json", {c.name: c.prep.state() for c in clients})
    write_json(out / "source_ae.json", {c.name: c.ae_history for c in clients})
    with (out / "source.jsonl").open("w") as log:
        for round_id in range(1, config["meta"]["rounds"] + 1):
            record = {"round": round_id, **meta_round(global_model, clients, config)}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            if round_id % 100 == 0:
                print(f"seed={seed} round={round_id} update={record['global_update_norm']:.6g}", flush=True)
    torch.save({"meta": global_model.state_dict(), "random": initial.state_dict()}, out / "shared.pt")
    with (out / "source_coverage.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["floor", "row_id", "role", "ae_visits", "inner_visits", "outer_visits"])
        for c in clients:
            floor = floors[c.name]
            for i, row_id in enumerate(floor["row_ids"][c.support]):
                writer.writerow([c.name, row_id, "support", c.ae_visits[i], c.batches.visits[i], 0])
            for i, row_id in enumerate(floor["row_ids"][c.query]):
                writer.writerow([c.name, row_id, "query", 0, 0, c.query_visits[i]])
    curves, summaries = {}, []
    for name in config["targets"]:
        floor = floors[name]
        support, query = splits[name]
        prep = Transform.fit(floor["rssi"][support], floor["xy"][support])
        sx, qx = (prep.x(floor["rssi"][ids], config["device"]) for ids in (support, query))
        sy = prep.y(floor["xy"][support], config["device"])
        encoder, decoder, mapper = private(sx.shape[1], config, seed)
        ae_history, ae_visits = train_ae(encoder, decoder, sx, selection["epochs"], selection["lr"], seed, config)
        write_json(out / f"{name}_preprocessing.json", {**prep.state(), **prep.diagnostics(floor["rssi"][query])})
        write_json(out / f"{name}_ae.json", ae_history)
        visits_pair = []
        for method, initialization in (("RI", initial), ("MI", global_model)):
            pred, losses, visits, model = adapt(encoder, initialization, mapper, sx, sy, qx, config, seed)
            prediction = pred.astype(np.float64) + prep.origin
            curve = evaluate(prediction, floor["xy"][query])
            curves[f"{name}/{method}"] = curve
            np.savez_compressed(out / f"{name}_{method}_predictions.npz", predictions=prediction,
                                query_row_ids=floor["row_ids"][query], truth=floor["xy"][query])
            torch.save(model.state_dict(), out / f"{name}_{method}.pt")
            write_json(out / f"{name}_{method}_support_loss.json", losses)
            visits_pair.append(visits)
            for step in config["adapt"]["report_steps"]:
                summaries.append({"seed": seed, "floor": name, "method": method, **curve[step]})
            print(f"seed={seed} target={name} {method} final_mde={curve[-1]['mde']:.4f}", flush=True)
        assert np.array_equal(*visits_pair)
        write_json(out / f"{name}_coverage.json", {"support_row_ids": floor["row_ids"][support].tolist(),
                                                   "ae_visits": ae_visits.tolist(), "adapt_visits_per_method": visits_pair[0].tolist()})
    write_json(out / "curves.json", curves)
    write_json(out / "summary.json", summaries)
    hits = {name: {method: {str(a): first_hit(curves[f"{name}/{method}"], a)
                            for a in config["adapt"]["thresholds"][name]} for method in ("MI", "RI")}
            for name in config["targets"]}
    write_json(out / "thresholds.json", hits)
    write_json(out / "completed.json", {"seed": seed, "rounds": config["meta"]["rounds"],
                                         "adapt_steps": config["adapt"]["steps"], "targets": config["targets"],
                                         "seconds": perf_counter() - started})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exp1.json")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True)
    write_json(args.output / "config.json", config)
    write_json(args.output / "provenance.json", {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                                 "data_sha256": hashlib.sha256(Path(config["data_path"]).read_bytes()).hexdigest(),
                                                 "torch": torch.__version__, "numpy": np.__version__})
    floors = load_floors(config["data_path"])
    selection = calibrate(floors, config, seed=config["seeds"][0])
    write_json(args.output / "ae_selection.json", selection)
    if selection["selection_score"] >= 1:
        raise RuntimeError("Source AE calibration did not improve the per-AP mean baseline; training stopped")
    print(f"AE selected lr={selection['lr']} epochs={selection['epochs']} score={selection['selection_score']:.5f}", flush=True)
    for seed in config["seeds"]:
        run_seed(floors, config, selection, seed, args.output)
    write_json(args.output / "completed.json", {"seeds": config["seeds"], "targets": config["targets"], "status": "complete"})


if __name__ == "__main__":
    main()
