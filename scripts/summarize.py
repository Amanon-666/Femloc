"""Summarize fixed DG/DA comparisons without selecting on official validation."""

import json
import statistics
import sys
from pathlib import Path

import numpy as np


config = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/v1.json")
output = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/v1")
cfg = json.loads(config.read_text())
records = {}
for target in cfg["targets"]:
    for seed in cfg["seeds"]:
        for method in cfg["methods"]:
            path = output / target / method / f"seed_{seed}" / "result.json"
            if not path.exists():
                raise FileNotFoundError(path)
            record = json.loads(path.read_text())
            assert record["source_steps"] == cfg["train_steps"]
            assert len(record["gga_history"]) == (cfg["gga_end_step"] - cfg["gga_start_step"] + 1 if method.startswith("gga") else 0)
            assert set(record["da_curve_official_validation"]) == set(map(str, cfg["da_report_steps"]))
            records[target, seed, method] = record
    for seed in cfg["seeds"]:
        supports = {tuple(records[target, seed, method]["da_support_row_ids"]) for method in cfg["methods"]}
        assert len(supports) == 1, (target, seed, "methods used different support rows")


def mean_sd(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0
    return f"{mean:.2f} ± {sd:.2f}"


lines = [
    "# GGA-UJI cross-floor V1 results",
    "",
    "Within each building, the target floor is absent from all source training. Official",
    "`validationData.csv` is the primary held-out test. DG is source-only; DA uses ten",
    "target training positions × three scans and updates only the regression head for 50 steps.",
    "Values are mean horizontal Euclidean error in metres, mean ± sample SD over three seeds.",
    "Neither the official validation rows nor their labels select a checkpoint.",
    "",
    "| Target | Method | Target train DG | Official DG | DA 1 | DA 5 | DA 10 | DA 20 | DA 50 |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for target in cfg["targets"]:
    for method in cfg["methods"]:
        runs = [records[target, seed, method] for seed in cfg["seeds"]]
        values = [
            [run["dg_target_train"]["mean_m"] for run in runs],
            [run["dg_official_validation"]["mean_m"] for run in runs],
        ] + [[run["da_curve_official_validation"][str(step)]["mean_m"] for run in runs] for step in [1, 5, 10, 20, 50]]
        lines.append(f"| {target} | {method} | " + " | ".join(map(mean_sd, values)) + " |")
lines.extend(["", "## Paired GGA − ERM error (negative favours GGA)", "",
              "| Encoder | Stage | Per-target paired differences, m | Equal-floor mean, m |",
              "|---|---|---|---:|"])
for kind in ["mlp", "set"]:
    for stage, field in [("DG", None), ("DA 10", "10"), ("DA 50", "50")]:
        floor_diffs = []
        for target in cfg["targets"]:
            diffs = []
            for seed in cfg["seeds"]:
                gga = records[target, seed, f"gga_{kind}"]
                erm = records[target, seed, f"erm_{kind}"]
                g = gga["dg_official_validation"]["mean_m"] if field is None else gga["da_curve_official_validation"][field]["mean_m"]
                e = erm["dg_official_validation"]["mean_m"] if field is None else erm["da_curve_official_validation"][field]["mean_m"]
                diffs.append(g - e)
            floor_diffs.append(statistics.mean(diffs))
        per_floor = ", ".join(f"{target}: {diff:+.2f}" for target, diff in zip(cfg["targets"], floor_diffs))
        lines.append(f"| {kind} | {stage} | {per_floor} | {statistics.mean(floor_diffs):+.2f} |")
lines.extend([
    "",
    "## Interpretation limit",
    "",
    "The official test is later data and its user, phone and spatial coverage can differ.",
    "Its error combines cross-floor generalization with temporal and sampling shifts.",
    "This is a coordinate-regression transfer of GGA, not its original image-classification",
    "benchmark. Set blocks are adapted from the official Set Transformer code with masks.",
])
report = Path("docs/RESULTS_V1.md")
report.write_text("\n".join(lines) + "\n")
print(report)
print("\n".join(lines[-18:]))
