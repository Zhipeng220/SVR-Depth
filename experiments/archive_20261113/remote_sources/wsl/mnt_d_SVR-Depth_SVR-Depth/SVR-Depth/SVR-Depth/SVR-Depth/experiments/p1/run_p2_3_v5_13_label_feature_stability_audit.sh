#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

SCRIPT="experiments/p1/p2_3_v5_13_label_feature_stability_audit.py"
DATA="experiments/p1/data_p2_3_v5_13"
OUT="outputs/p2_3_v5_13_label_feature_stability_audit"

if [[ ! -s "$SCRIPT" || ! -s "$DATA/p2_3_v5_5_all_same_state_anchor_labels.csv" || ! -s "$DATA/p2_2_2_valid_dataset.csv" ]]; then
  echo "ERROR: missing v5.13 script or bundled CSV inputs."
  echo "Expected: $SCRIPT and both CSV files under $DATA/"
  exit 2
fi

python -m py_compile "$SCRIPT"
python "$SCRIPT" --self-test
python "$SCRIPT" \
  --labels-csv "$DATA/p2_3_v5_5_all_same_state_anchor_labels.csv" \
  --p2-dataset-csv "$DATA/p2_2_2_valid_dataset.csv" \
  --out-dir "$OUT"

ZIP="outputs/p2_3_v5_13_label_feature_stability_audit_review.zip"
rm -f "$ZIP"
zip -q -r "$ZIP" "$OUT"
zip -q -r "$ZIP" \
  "$SCRIPT" \
  "experiments/p1/run_p2_3_v5_13_label_feature_stability_audit.sh" \
  "experiments/p1/README_P2_3_v5_13.md" \
  "$DATA" \
  "experiments/p1/reference_sources"

echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
