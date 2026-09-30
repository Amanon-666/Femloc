#!/usr/bin/env bash
# 完成训练、评价和汇总；失败留下状态，供值守判断。
set -euo pipefail
cd /home/panyushuo/projects/panyushuo/FeMLoc-Learned-AP-Map
out=outputs/learned_ap_map_v1
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "exit=%s time=%s\n" "$code" "$(date -u +%FT%TZ)" > outputs/learned_ap_map_v1.failed; fi' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
../DKT-UJI/env/bin/python -u -m scripts.run_learned_ap_map --output "$out"
../DKT-UJI/env/bin/python -m scripts.summarize_learned_ap_map "$out"
