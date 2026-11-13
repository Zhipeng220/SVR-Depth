#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth
mkdir -p outputs/p2_3_v5_25b_halo_sensitivity
python -m py_compile experiments/p1/p2_3_v5_25b_halo_sensitivity.py
python experiments/p1/p2_3_v5_25b_halo_sensitivity.py \
  --rgb /root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb \
  --frames 5 \
  --start-index 0 \
  --resolution_level 0 \
  --halos 0,32,64 \
  --warmup 2 \
  --repeats 5 \
  --stream-warmup 2 \
  --stream-repeats 5 \
  2>&1 | tee outputs/p2_3_v5_25b_halo_sensitivity/run.log
