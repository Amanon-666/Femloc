"""Run one fixed target, method and seed from the experiment config."""

import argparse

from src.experiment import run


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/v1.json")
parser.add_argument("--target", required=True)
parser.add_argument("--method", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--output", default="outputs/v1")
args = parser.parse_args()
run(args.config, args.target, args.method, args.seed, args.output)
