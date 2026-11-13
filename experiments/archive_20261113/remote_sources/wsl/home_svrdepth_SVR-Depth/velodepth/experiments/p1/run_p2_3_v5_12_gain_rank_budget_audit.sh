#!/usr/bin/env bash
# P2.3 v5.12 CPU-only same-state gain ranking and constrained-budget audit.
set -euo pipefail

cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

SCRIPT="experiments/p1/p2_3_v5_12_gain_rank_budget_audit.py"
LABELS="experiments/p1/data_p2_3_v5_12/P2_3_v5_5_完整序列同状态收益标签.csv"
OUT="outputs/p2_3_v5_12_gain_rank_budget_audit"

if [[ ! -s "$SCRIPT" || ! -s "$LABELS" ]]; then
  echo "ERROR: missing v5.12 script or bundled same-state labels."
  echo "Expected:"
  echo "  $SCRIPT"
  echo "  $LABELS"
  exit 2
fi

python -m py_compile "$SCRIPT"
python "$SCRIPT" --self-test
python "$SCRIPT" --labels-csv "$LABELS" --out-dir "$OUT"

ZIP="outputs/p2_3_v5_12_gain_rank_budget_audit_review.zip"
rm -f "$ZIP"
zip -q -r "$ZIP" "$OUT"
zip -q "$ZIP" \
  "$SCRIPT" \
  "experiments/p1/run_p2_3_v5_12_gain_rank_budget_audit.sh" \
  "experiments/p1/README_P2_3_v5_12.md" \
  "$LABELS"

echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
