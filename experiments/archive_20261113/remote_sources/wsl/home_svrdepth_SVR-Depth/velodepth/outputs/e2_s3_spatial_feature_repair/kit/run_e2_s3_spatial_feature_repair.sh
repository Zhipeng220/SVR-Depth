#!/usr/bin/env bash
set -euo pipefail
cd /home/svrdepth/SVR-Depth/velodepth
if ! command -v conda >/dev/null 2>&1; then
  source /root/miniconda3/etc/profile.d/conda.sh
fi
conda activate svrdepth
OUT="outputs/e2_s3_spatial_feature_repair/crowd2_a9_a37_run01"
mkdir -p "$(dirname "$OUT")"
if [ -d "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "ERROR: $OUT is not empty; choose a new --out-dir to preserve prior results." >&2
  exit 2
fi
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python experiments/p1/e2_s3_spatial_feature_repair.py \
  --frames 46 \
  --anchors 9,37 \
  --horizons 1,4,8 \
  --tiles 0,1,2,3 \
  --resolution-level 0 \
  --tolerance 1e-6 \
  --out-dir "$OUT" \
  2>&1 | tee outputs/e2_s3_spatial_feature_repair/e2_s3_spatial_feature_repair.log
