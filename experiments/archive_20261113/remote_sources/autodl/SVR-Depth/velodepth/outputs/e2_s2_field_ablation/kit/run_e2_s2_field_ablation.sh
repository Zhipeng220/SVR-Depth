#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth
conda activate svrdepth 2>/dev/null || source /root/miniconda3/etc/profile.d/conda.sh && conda activate svrdepth
OUT="outputs/e2_s2_field_ablation/crowd2_a9_a37_run01"
mkdir -p "$(dirname "$OUT")"
if [ -d "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
  echo "ERROR: $OUT is not empty; use a new --out-dir to preserve prior runs." >&2
  exit 2
fi
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
python experiments/p1/e2_s2_field_ablation.py \
  --frames 46 \
  --anchors 9,37 \
  --horizons 1,4,8 \
  --resolution-level 0 \
  --tolerance 1e-6 \
  --out-dir "$OUT" \
  2>&1 | tee outputs/e2_s2_field_ablation/e2_s2_field_ablation.log
