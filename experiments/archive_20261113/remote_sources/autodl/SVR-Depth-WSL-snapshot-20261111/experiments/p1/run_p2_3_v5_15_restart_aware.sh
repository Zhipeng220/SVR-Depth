#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

PYTHON="/root/miniconda3/envs/svrdepth/bin/python"
SCRIPT="experiments/p1/p2_3_v5_15_restart_aware_counterfactual.py"
ANALYZER="experiments/p1/analyze_p2_3_v5_15_restart_aware.py"
DATA="experiments/p1/data_p2_3_v5_14"
OUT="outputs/p2_3_v5_15_restart_aware"
TRAIN="$DATA/p2_2_2_valid_dataset.csv"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python)"
fi
for f in "$SCRIPT" "$ANALYZER" "$TRAIN" "$DATA/p2_3_v5_5_all_same_state_anchor_labels.csv" \
  experiments/p1/p2_3_online_replay_v5_4_budget_sweep.py \
  experiments/p1/p1_multihorizon_counterfactual.py \
  experiments/p1/stream_state.py experiments/p1/stream_state_snapshot.py experiments/p1/velodepth_stream.py; do
  [[ -s "$f" ]] || { echo "ERROR: missing required file: $f"; exit 2; }
done

rm -rf "$OUT"
mkdir -p "$OUT"
"$PYTHON" -m py_compile "$SCRIPT" "$ANALYZER"
"$PYTHON" "$SCRIPT" --self-test
"$PYTHON" -c "import torch; print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'This replay requires a CUDA GPU.'"

for seq in balloon2 crowd2 crowd3; do
  case "$seq" in
    balloon2) DATA_ROOT="/root/autodl-tmp/datasets/rgbd_bonn_balloon2" ;;
    crowd2)   DATA_ROOT="/root/autodl-tmp/datasets/rgbd_bonn_crowd2" ;;
    crowd3)   DATA_ROOT="/root/autodl-tmp/datasets/rgbd_bonn_crowd3" ;;
  esac
  [[ -s "$DATA_ROOT/rgb.txt" && -s "$DATA_ROOT/depth.txt" ]] || {
    echo "ERROR: RGB-D index files missing for $seq at $DATA_ROOT"; exit 3;
  }
  echo "===== P2.3 v5.15 RESTART-AWARE REPLAY: $seq ====="
  "$PYTHON" "$SCRIPT" \
    --data-root "$DATA_ROOT" \
    --sequence "$seq" \
    --frames 86 \
    --train-dataset "$TRAIN" \
    --out-dir "$OUT/${seq}_86" \
    --horizon 16 \
    --gamma 0.9 \
    --candidate-stride 4 \
    --candidate-offset 1 \
    --min-refresh-gap 4 \
    --fixed-interval 8 \
    --budget-actions 8 \
    --policies auto,budget_fixed,causal_budget_state_only,causal_budget_motion_plus_learned \
    2>&1 | tee "$OUT/${seq}_86_run.log"
done

echo "===== Compare restart-aware labels against v5.5 ====="
"$PYTHON" "$ANALYZER" \
  --old-labels "$DATA/p2_3_v5_5_all_same_state_anchor_labels.csv" \
  --new-root "$OUT" \
  --out-dir "$OUT/comparison"

ZIP="outputs/p2_3_v5_15_restart_aware_review.zip"
rm -f "$ZIP"
zip -q -r "$ZIP" "$OUT" \
  "$SCRIPT" "$ANALYZER" \
  experiments/p1/run_p2_3_v5_15_restart_aware.sh \
  experiments/p1/p2_3_online_replay_v5_4_budget_sweep.py \
  experiments/p1/p1_multihorizon_counterfactual.py \
  experiments/p1/stream_state.py \
  experiments/p1/stream_state_snapshot.py \
  experiments/p1/velodepth_stream.py

echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
