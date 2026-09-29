#!/usr/bin/env bash
set -euo pipefail
output="$1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
trap 'printf "exit=%s\n" "$?" > "$output.failed"' ERR
env/bin/python -m dkt.run --output "$output"
env/bin/python -m dkt.summarize "$output"
