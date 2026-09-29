"""Run every frozen target/method/seed cell once, then summarize all results."""

import json
import subprocess
import sys
from pathlib import Path


config = Path(sys.argv[1] if len(sys.argv) > 1 else "configs/v1.json")
output = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/v1")
cfg = json.loads(config.read_text())
for target in cfg["targets"]:
    for seed in cfg["seeds"]:
        for method in cfg["methods"]:
            result = output / target / method / f"seed_{seed}" / "result.json"
            if result.exists():
                continue
            subprocess.run(
                [sys.executable, "-m", "scripts.run", "--config", str(config),
                 "--target", target, "--method", method, "--seed", str(seed),
                 "--output", str(output)],
                check=True,
            )
subprocess.run([sys.executable, "-m", "scripts.summarize", str(config), str(output)], check=True)
