#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth
source /root/miniconda3/etc/profile.d/conda.sh
conda activate svrdepth
python -m py_compile experiments/p1/e2_s1_screen.py
mkdir -p outputs/e2_s1_screen
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python experiments/p1/e2_s1_screen.py \
  --frames 46 \
  --anchors 9,37 \
  --horizons 1,4,8 \
  --resolution-level 0 \
  --tolerance 1e-6 \
  --out-dir outputs/e2_s1_screen/crowd2_a9_a37 \
  2>&1 | tee outputs/e2_s1_screen/e2_s1_screen.log
