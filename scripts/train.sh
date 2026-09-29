#!/usr/bin/env bash
# 在独立环境顺序运行完整实验，失败保留日志与退出码。
set -eu
cd "$(dirname "$0")/.."
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
run_dir="$1"
if env/bin/python -u -m scripts.run --output "$run_dir"; then
  env/bin/python -m scripts.summarize "$run_dir"
else
  status=$?
  printf '%s\n' "$status" > "${run_dir}.failed"
  exit "$status"
fi
