#!/usr/bin/env bash
# 顺序执行必要检查和完整实验，保存退出状态供自动值守读取。
set -eu
cd "$(dirname "$0")/.."
mkdir -p outputs
trap 'code=$?; printf "%s\n" "$code" > outputs/ridge_reference_v1.failed' ERR
env/bin/python -u -m scripts.check_ridge_reference > outputs/ridge_reference_v1.check.json
env/bin/python -u -m scripts.run_ridge_reference --output outputs/ridge_reference_v1
