#!/usr/bin/env bash
# P2.3 v5.8: 32-frame smoke test, K=3, four temporal strata.
# Run after: conda activate svrdepth
set -euo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth

SCRIPT="experiments/p1/p2_3_online_replay_v5_8_temporal_budget.py"
DATASET="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
PARENT="outputs/p2_3_v5_8_temporal_budget"
OUT="$PARENT/crowd2_32_smoke"
LOG="$PARENT/crowd2_32_smoke_run.log"
ZIP="$PARENT/p2_3_v5_8_smoke_review.zip"

test -s "$SCRIPT" || { echo "ERROR: missing $SCRIPT"; exit 2; }
test -s "$DATASET" || { echo "ERROR: missing $DATASET"; exit 2; }
test -d /root/autodl-tmp/datasets/rgbd_bonn_crowd2 || {
  echo "ERROR: Crowd2 dataset path missing"; exit 2;
}
rm -rf "$OUT" "$LOG" "$ZIP"
mkdir -p "$OUT"

python -m py_compile "$SCRIPT"
python "$SCRIPT" \
  --data-root /root/autodl-tmp/datasets/rgbd_bonn_crowd2 \
  --sequence crowd2 \
  --frames 32 \
  --train-dataset "$DATASET" \
  --policies auto,budget_fixed,causal_budget_state_age,causal_budget_motion_plus_learned,causal_budget_motion_plus_learned_stratified \
  --scheduled-action state_writeback \
  --fixed-interval 8 \
  --min-refresh-gap 8 \
  --candidate-stride 4 \
  --candidate-offset 1 \
  --budget-actions 3 \
  --time-strata 4 \
  --horizon 16 \
  --out-dir "$OUT" \
  2>&1 | tee "$LOG"

python - <<'PY'
import pandas as pd
from pathlib import Path
out = Path("outputs/p2_3_v5_8_temporal_budget/crowd2_32_smoke")
summary_path = out / "p2_3_online_summary.csv"
strata_path = out / "p2_3_v5_8_temporal_stratum_summary.csv"
assert summary_path.is_file(), "summary CSV missing"
assert strata_path.is_file(), "temporal-stratum summary CSV missing"
df = pd.read_csv(summary_path)
required = {
    "auto", "budget_fixed", "causal_budget_state_age",
    "causal_budget_motion_plus_learned",
    "causal_budget_motion_plus_learned_stratified",
}
found = set(df["policy"].astype(str))
missing = sorted(required - found)
if missing:
    raise SystemExit(f"FAIL: missing policy summaries: {missing}")
r = df[df["policy"].isin(required - {"auto"})]
print("\n===== P2.3 v5.8 SMOKE BUDGET CHECK =====")
print(r[["policy", "frames_with_valid_metrics", "mean_absrel",
         "mean_rmse", "mean_delta1", "state_writeback_count",
         "causal_budget_met", "mean_decision_ms",
         "total_feature_extraction_ms", "total_predictor_inference_ms",
         "total_budget_control_ms"]].to_string(index=False))
if not (r["state_writeback_count"] == 3).all():
    raise SystemExit("FAIL: at least one scheduled policy did not reach K=3 writes")
if not r["causal_budget_met"].fillna(True).all():
    raise SystemExit("FAIL: a causal policy did not meet its budget")
print("PASS: all scheduled policies reached K=3; inspect temporal strata and logs before full run.")
PY

zip -q -r "$ZIP" "$OUT"
zip -q "$ZIP" "$LOG" "$SCRIPT" \
  experiments/p1/run_p2_3_v5_8_smoke.sh \
  experiments/p1/run_p2_3_v5_8_full.sh \
  experiments/p1/README_P2_3_v5_8.md
echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
