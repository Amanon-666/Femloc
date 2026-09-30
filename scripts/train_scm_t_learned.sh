#!/usr/bin/env bash
# 固定 SCM-T 地图，训练 AP 可靠度，完成后自动生成报告。
set -euo pipefail
cd /home/panyushuo/projects/panyushuo/FeMLoc-Learned-AP-Map
out=outputs/scm_t_learned_v1
trap 'code=$?; if [ "$code" -ne 0 ]; then printf "exit=%s time=%s\n" "$code" "$(date -u +%FT%TZ)" > outputs/scm_t_learned_v1.failed; fi' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1
../DKT-UJI/env/bin/python -u -m scripts.run_learned_ap_map --config configs/scm_t_learned.json --output "$out"
../DKT-UJI/env/bin/python -m scripts.summarize_learned_ap_map "$out" --report docs/SCM_T_LEARNED_RESULTS.md
