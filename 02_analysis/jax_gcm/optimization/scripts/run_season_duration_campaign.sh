#!/bin/bash
set -e

cd ~/code_uqam/jax-climate-experiments/02_analysis/jax_gcm/optimization/scripts

for DATE in 2000-01-01 2000-04-01 2000-07-01 2000-10-01
do
  for DAYS in 5 10 15
  do
    echo "======================================="
    echo "Running init=$DATE days=$DAYS"
    echo "======================================="

    python optimize_sst_field_windows.py \
      --init_time ${DATE}T00:00:00 \
      --total_days $DAYS \
      --n_steps 15 \
      --lr 0.5 \
      --lambda_l2 1e-4 \
      --lambda_smooth 0.02
  done
done
