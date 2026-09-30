#!/usr/bin/env bash
# 固定预算运行地图核学习，完成后核验并自动生成配对报告。
set -euo pipefail
cd /home/panyushuo/projects/panyushuo/FeMLoc-Learned-Map-Kernel
out=outputs/map_kernel_v1
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "exit=%s time=%s\n" "$code" "$(date -u +%FT%TZ)" > outputs/map_kernel_v1.failed; fi' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
env/bin/python -u -m scripts.run_map_kernel --config configs/map_kernel.json --output "$out"
env/bin/python -m scripts.summarize_map_kernel "$out"
