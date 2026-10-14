#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
runtime=/home/panyushuo/projects/panyushuo/DKT-UJI/env/bin/python
mkdir -p outputs/spatial_residual
git rev-parse HEAD > outputs/spatial_residual/frozen_commit.txt
date -u '+STARTED %Y-%m-%dT%H:%M:%SZ'
pids=()
for b in 0 1 2; do
  "$runtime" spatial_residual.py --building "$b" --mode run > "outputs/spatial_residual/B${b}_seed0.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
date -u '+FINISHED %Y-%m-%dT%H:%M:%SZ'
echo "EXIT_STATUS $status"
exit "$status"
