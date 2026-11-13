#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/SVR-Depth/velodepth

DATASET="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
ROOT="outputs/p2_3_v5_15_restart_aware"
SCRIPT="experiments/p1/analyze_p2_3_v5_17_feature_recovery_audit.py"
OUT="outputs/p2_3_v5_17_feature_recovery_audit"
LOG="outputs/p2_3_v5_17_feature_recovery_audit_run.log"
ZIP="outputs/p2_3_v5_17_feature_recovery_audit_review.zip"

mkdir -p "$OUT" outputs

test -s "$DATASET" || { echo "FAIL: missing $DATASET"; exit 2; }
test -s "$SCRIPT" || { echo "FAIL: missing $SCRIPT"; exit 2; }
for seq in balloon2 crowd2 crowd3; do
  test -s "$ROOT/${seq}_86/p2_3_v5_15_same_state_anchor_labels.csv" || {
    echo "FAIL: missing v5.15 labels for $seq under $ROOT"; exit 2;
  }
done

python -m py_compile "$SCRIPT"

set -o pipefail
python "$SCRIPT" \
  --v515-root "$ROOT" \
  --dataset "$DATASET" \
  --out-dir "$OUT" \
  --alpha 10 \
  --budget-k 8 \
  --atol 1e-5 \
  --rtol 1e-4 \
  2>&1 | tee "$LOG"

test -s "$OUT/P2_3_v5_17_实验报告.md" || { echo "FAIL: report missing"; exit 3; }
test -s "$OUT/p2_3_v5_17_loso_predictions.csv" || { echo "FAIL: LOSO predictions missing"; exit 3; }

rm -f "$ZIP"
zip -qr "$ZIP" \
  "$OUT" \
  "$LOG" \
  "$SCRIPT" \
  "experiments/p1/run_p2_3_v5_17_feature_recovery_audit.sh" \
  "$DATASET" \
  "$ROOT/balloon2_86/p2_3_v5_15_same_state_anchor_labels.csv" \
  "$ROOT/crowd2_86/p2_3_v5_15_same_state_anchor_labels.csv" \
  "$ROOT/crowd3_86/p2_3_v5_15_same_state_anchor_labels.csv"

unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
