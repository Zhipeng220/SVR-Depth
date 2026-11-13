#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/SVR-Depth/velodepth
mkdir -p outputs/p2_3_v5_25_tile_halo_feasibility


python experiments/p1/p2_3_v5_25_tile_halo_feasibility.py \
  --rgb /root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb \
  --frames 3 \
  --resolution_level 0 \
  --halo 32 \
  --warmup 1 \
  --repeats 3 \
  2>&1 | tee outputs/p2_3_v5_25_tile_halo_feasibility/run.log
