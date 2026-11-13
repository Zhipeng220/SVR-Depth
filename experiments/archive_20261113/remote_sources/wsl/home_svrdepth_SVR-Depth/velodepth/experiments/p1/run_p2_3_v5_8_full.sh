#!/usr/bin/env bash
# P2.3 v5.8: three full 86-frame sequences, K=8, four temporal strata.
# Run only after the 32-frame smoke test passes.
# Run after: conda activate svrdepth
set -euo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth

SCRIPT="experiments/p1/p2_3_online_replay_v5_8_temporal_budget.py"
DATASET="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
PARENT="outputs/p2_3_v5_8_temporal_budget"
BASE="$PARENT/full"
ZIP="$PARENT/p2_3_v5_8_temporal_budget_review.zip"

test -s "$SCRIPT" || { echo "ERROR: missing $SCRIPT"; exit 2; }
test -s "$DATASET" || { echo "ERROR: missing $DATASET"; exit 2; }
python -m py_compile "$SCRIPT"
rm -rf "$BASE" "$ZIP"
mkdir -p "$BASE"

for seq in crowd2 balloon2 crowd3; do
  case "$seq" in
    crowd2) DATA_ROOT="/root/autodl-tmp/datasets/rgbd_bonn_crowd2" ;;
    balloon2) DATA_ROOT="/root/autodl-tmp/datasets/rgbd_bonn_balloon2" ;;
    crowd3) DATA_ROOT="/root/autodl-tmp/datasets/rgbd_bonn_crowd3" ;;
  esac
  if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: missing dataset path $DATA_ROOT"
    exit 2
  fi

  OUT="$BASE/${seq}_86"
  LOG="$BASE/${seq}_86_run.log"
  mkdir -p "$OUT"
  echo "===== P2.3 v5.8 FULL RUN: $seq (86 frames, K=8, 4 strata) ====="

  python "$SCRIPT" \
    --data-root "$DATA_ROOT" \
    --sequence "$seq" \
    --frames 86 \
    --train-dataset "$DATASET" \
    --policies auto,budget_fixed,causal_budget_state_age,causal_budget_motion_plus_learned,causal_budget_motion_plus_learned_stratified \
    --scheduled-action state_writeback \
    --fixed-interval 8 \
    --min-refresh-gap 8 \
    --candidate-stride 4 \
    --candidate-offset 1 \
    --budget-actions 8 \
    --time-strata 4 \
    --horizon 16 \
    --out-dir "$OUT" \
    2>&1 | tee "$LOG"
done

zip -q "$ZIP" "$SCRIPT" \
  experiments/p1/run_p2_3_v5_8_smoke.sh \
  experiments/p1/run_p2_3_v5_8_full.sh \
  experiments/p1/README_P2_3_v5_8.md
for seq in crowd2 balloon2 crowd3; do
  zip -q -r "$ZIP" "$BASE/${seq}_86"
  zip -q "$ZIP" "$BASE/${seq}_86_run.log"
done

echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
