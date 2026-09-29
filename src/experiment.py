"""Train a source-floor regressor, then evaluate DG and fixed-shot DA."""

from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data import read_csv, source_floors, support_indices, tensor_batch
from src.gga import erm_step, gga_step, regression_loss
from src.model import Locator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def error_metres(model, scans, indices: np.ndarray, origin: np.ndarray, device: torch.device,
                 known_aps: np.ndarray) -> dict:
    model.eval()
    differences = []
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            batch = tensor_batch(scans, indices[start:start + 256], origin, device, known_aps)
            prediction = model(batch) * 100 + torch.from_numpy(origin).to(device)
            truth = torch.from_numpy(scans.xy[indices[start:start + 256]]).to(device)
            differences.extend(torch.linalg.vector_norm(prediction - truth, dim=1).cpu().numpy().tolist())
    return {
        "n": len(differences),
        "mean_m": float(np.mean(differences)),
        "median_m": float(np.median(differences)),
        "p90_m": float(np.percentile(differences, 90)),
    }


def run(config_path: str, target: str, method: str, seed: int, output_root: str) -> dict:
    cfg = json.loads(Path(config_path).read_text())
    if target not in cfg["targets"] or method not in cfg["methods"] or seed not in cfg["seeds"]:
        raise ValueError("Requested target, method or seed is outside the frozen config")
    set_seed(seed)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.time()
    train = read_csv(cfg["train_csv"])
    official_test = read_csv(cfg["test_csv"])
    floor_names = source_floors(train, target)
    domain_indices = [train.floor_indices(name) for name in floor_names]
    assert all(len(indices) > 0 for indices in domain_indices)
    assert not any(np.intersect1d(indices, train.floor_indices(target)).size for indices in domain_indices)
    target_train_indices = train.floor_indices(target)
    target_test_indices = official_test.floor_indices(target)
    if len(target_test_indices) == 0:
        raise ValueError(f"No official validation scans for {target}")

    source_indices = np.concatenate(domain_indices)
    origin = train.xy[source_indices].mean(axis=0)
    known_aps = (train.raw[source_indices] != 100).any(axis=0)
    encoder_kind = method.split("_")[1]
    model = Locator(encoder_kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    generator = np.random.default_rng(seed)
    output = Path(output_root) / target / method / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    gga_history = []
    last_loss = None

    model.train()
    for step in range(cfg["train_steps"]):
        batches = [
            tensor_batch(train, generator.choice(indices, size=cfg["batch_per_source_floor"], replace=True), origin, device, known_aps)
            for indices in domain_indices
        ]
        if method.startswith("gga") and cfg["gga_start_step"] <= step <= cfg["gga_end_step"]:
            last_loss, before, after, accepted = gga_step(model, optimizer, batches, cfg)
            gga_history.append({"step": step, "cosine_before": before, "cosine_after": after, "accepted": accepted})
        else:
            last_loss = erm_step(model, optimizer, batches)
        if (step + 1) % 100 == 0:
            print(json.dumps({"target": target, "method": method, "seed": seed,
                              "source_step": step + 1, "source_loss": last_loss,
                              "elapsed_s": round(time.time() - started, 1)}), flush=True)
        if step + 1 in cfg["checkpoint_steps"]:
            torch.save(model.state_dict(), output / f"source_step_{step + 1}.pt")

    # Targets are read for reporting only after the full, fixed source training budget.
    dg_train = error_metres(model, train, target_train_indices, origin, device, known_aps)
    dg_official = error_metres(model, official_test, target_test_indices, origin, device, known_aps)
    source_metrics = {
        name: error_metres(model, train, indices, origin, device, known_aps)
        for name, indices in zip(floor_names, domain_indices)
    }
    support = support_indices(train, target, cfg["da_support_positions"], cfg["da_scans_per_position"], seed)
    support_batch = tensor_batch(train, support, origin, device, known_aps)
    adapted = copy.deepcopy(model)
    for parameter in adapted.featurizer.parameters():
        parameter.requires_grad_(False)
    head_optimizer = torch.optim.Adam(adapted.head.parameters(), lr=cfg["da_learning_rate"])
    da_curve = {"0": dg_official}
    adapted.train()
    for step in range(1, cfg["da_steps"] + 1):
        head_optimizer.zero_grad()
        loss = regression_loss(adapted, support_batch)
        loss.backward()
        head_optimizer.step()
        if step in cfg["da_report_steps"]:
            da_curve[str(step)] = error_metres(adapted, official_test, target_test_indices, origin, device, known_aps)
            adapted.train()

    result = {
        "target": target, "method": method, "seed": seed,
        "source_floors": floor_names,
        "source_rows": {name: len(indices) for name, indices in zip(floor_names, domain_indices)},
        "source_seen_ap_count": int(known_aps.sum()),
        "source_steps": cfg["train_steps"],
        "source_loss_final": last_loss,
        "source_metrics": source_metrics,
        "dg_target_train": dg_train,
        "dg_official_validation": dg_official,
        "da_support_row_ids": support.tolist(),
        "da_support_positions": cfg["da_support_positions"],
        "da_scans_per_position": cfg["da_scans_per_position"],
        "da_curve_official_validation": da_curve,
        "gga_history": gga_history,
        "elapsed_s": round(time.time() - started, 1),
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result
