#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/SVR-Depth/velodepth
mkdir -p outputs/p2_3_v5_26b_long_interval_refresh

python -m py_compile experiments/p1/p2_3_v5_26b_long_interval_refresh.py


python experiments/p1/p2_3_v5_26b_long_interval_refresh.py \
  --rgb /root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb \
  --frames 9 \
  --start-index 0 \
  --target-ages 1,2,4,8 \
  --resolution_level 0 \
  --halos 32,64 \
  --warmup 1 \
  --repeats 3 \
  --stream-warmup-rollouts 1 \
  --boundary-band 4 \
  2>&1 | tee outputs/p2_3_v5_26b_long_interval_refresh/run.log
