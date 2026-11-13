#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd /home/svrdepth/SVR-Depth/velodepth
conda activate svrdepth

OUT="outputs/e2_s4_state_conditioned_repair/crowd2_multi_anchor_run01"
mkdir -p "$(dirname "$OUT")"

PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python experiments/p1/e2_s4_state_conditioned_repair.py \
  --data-root "$HOME/datasets/rgbd_bonn_crowd2" \
  --frames 46 \
  --anchors 9,18,27,37 \
  --horizons 1,4,8 \
  --tiles 0,1,2,3 \
  --resolution-level 0 \
  --tolerance 1e-6 \
  --out-dir "$OUT" \
  2>&1 | tee "${OUT}.log"
