#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

PYTHON="/root/miniconda3/envs/svrdepth/bin/python"
SCRIPT="experiments/p1/p2_3_v5_15_restart_aware_counterfactual.py"
DATA="experiments/p1/data_p2_3_v5_14"
TRAIN="$DATA/p2_2_2_valid_dataset.csv"
OUT="outputs/p2_3_v5_15_smoke/crowd2_32"

if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python)"; fi
for f in "$SCRIPT" "$TRAIN" experiments/p1/p2_3_online_replay_v5_4_budget_sweep.py \
  experiments/p1/p1_multihorizon_counterfactual.py experiments/p1/stream_state.py \
  experiments/p1/stream_state_snapshot.py experiments/p1/velodepth_stream.py; do
  [[ -s "$f" ]] || { echo "ERROR: missing required file: $f"; exit 2; }
done

[[ -s /root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb.txt ]] || {
  echo "ERROR: crowd2 RGB index missing"; exit 3;
}
[[ -s /root/autodl-tmp/datasets/rgbd_bonn_crowd2/depth.txt ]] || {
  echo "ERROR: crowd2 depth index missing"; exit 3;
}

rm -rf "$OUT"
mkdir -p "$OUT"
"$PYTHON" -m py_compile "$SCRIPT"
"$PYTHON" "$SCRIPT" --self-test
"$PYTHON" -c "import torch; print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'This replay requires a CUDA GPU.'"

echo "===== P2.3 v5.15 SMOKE: crowd2, 32 frames ====="
"$PYTHON" "$SCRIPT" \
  --data-root /root/autodl-tmp/datasets/rgbd_bonn_crowd2 \
  --sequence crowd2 \
  --frames 32 \
  --train-dataset "$TRAIN" \
  --out-dir "$OUT" \
  --horizon 16 \
  --gamma 0.9 \
  --candidate-stride 4 \
  --candidate-offset 1 \
  --min-refresh-gap 4 \
  --fixed-interval 8 \
  --budget-actions 7 \
  --policies auto,budget_fixed,causal_budget_state_only,causal_budget_motion_plus_learned \
  2>&1 | tee outputs/p2_3_v5_15_smoke/crowd2_32_run.log

ZIP="outputs/p2_3_v5_15_smoke_review.zip"
rm -f "$ZIP"
zip -q -r "$ZIP" outputs/p2_3_v5_15_smoke \
  "$SCRIPT" experiments/p1/analyze_p2_3_v5_15_restart_aware.py \
  experiments/p1/run_p2_3_v5_15_smoke.sh \
  experiments/p1/p2_3_online_replay_v5_4_budget_sweep.py \
  experiments/p1/p1_multihorizon_counterfactual.py experiments/p1/stream_state.py \
  experiments/p1/stream_state_snapshot.py experiments/p1/velodepth_stream.py

echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
