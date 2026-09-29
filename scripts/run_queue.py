"""Run one encoder's complete frozen grid on one GPU."""

import argparse
import json
from pathlib import Path

from src.experiment import run


parser = argparse.ArgumentParser()
parser.add_argument("--kind", choices=["mlp", "set"], required=True)
parser.add_argument("--config", default="configs/v1.json")
parser.add_argument("--output", default="outputs/v1")
args = parser.parse_args()
config = json.loads(Path(args.config).read_text())
output = Path(args.output)
for target in config["targets"]:
    for seed in config["seeds"]:
        for prefix in ["erm", "gga"]:
            method = f"{prefix}_{args.kind}"
            path = output / target / method / f"seed_{seed}" / "result.json"
            if not path.exists():
                run(args.config, target, method, seed, args.output)
            print(json.dumps({"completed_cell": [target, method, seed]}), flush=True)
(output / f"queue_{args.kind}.completed").write_text("complete\n")
