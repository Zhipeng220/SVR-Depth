#!/usr/bin/env bash
set -euo pipefail

REPO="/root/autodl-tmp/SVR-Depth/velodepth"
SCRIPT="$REPO/experiments/p1/p2_3_v5_21_locked_model_review.py"
DATA_ROOT="${1:-/root/autodl-tmp/datasets/rgbd_bonn_synchronous}"

cd "$REPO"

if [[ ! -s "$SCRIPT" ]]; then
  echo "ERROR: missing Python script: $SCRIPT" >&2
  exit 2
fi

echo "===== P2.3 v5.21: locked-model cross-sequence review ====="
echo "Sequence: synchronous"
echo "Data root: $DATA_ROOT"
echo "Weights will remain frozen."

python -m py_compile "$SCRIPT"
python "$SCRIPT" all --data-root "$DATA_ROOT" --sequence synchronous --frames 86
