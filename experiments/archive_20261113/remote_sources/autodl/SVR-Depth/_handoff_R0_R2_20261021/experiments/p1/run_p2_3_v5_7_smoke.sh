#!/usr/bin/env bash
# P2.3 v5.7 32-frame smoke test. Run after: conda activate svrdepth
set -euo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth

SCRIPT="experiments/p1/p2_3_online_replay_v5_7_equal_budget.py"
DATASET="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
OUT="outputs/p2_3_v5_7_equal_budget/crowd2_32_smoke"
rm -rf "$OUT" "$OUT.run.log" "outputs/p2_3_v5_7_equal_budget/crowd2_32_smoke_review.zip"
mkdir -p "$OUT"

test -s "$SCRIPT" || { echo "ERROR: missing $SCRIPT"; exit 2; }
test -s "$DATASET" || { echo "ERROR: missing $DATASET"; exit 2; }
test -d /root/autodl-tmp/datasets/rgbd_bonn_crowd2 || {
  echo "ERROR: Crowd2 dataset path missing"; exit 2;
}

python -m py_compile "$SCRIPT"
python "$SCRIPT" \
  --data-root /root/autodl-tmp/datasets/rgbd_bonn_crowd2 \
  --sequence crowd2 \
  --frames 32 \
  --train-dataset "$DATASET" \
  --policies auto,budget_fixed,causal_budget_state_age,causal_budget_motion_plus_learned \
  --scheduled-action state_writeback \
  --fixed-interval 8 \
  --min-refresh-gap 8 \
  --candidate-stride 4 \
  --candidate-offset 1 \
  --budget-actions 3 \
  --horizon 16 \
  --out-dir "$OUT" \
  2>&1 | tee "$OUT.run.log"

python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("outputs/p2_3_v5_7_equal_budget/crowd2_32_smoke/p2_3_online_summary.csv")
if not p.is_file():
    raise SystemExit("FAIL: summary CSV was not generated")
df = pd.read_csv(p)
required = {"auto", "budget_fixed", "causal_budget_state_age", "causal_budget_motion_plus_learned"}
found = set(df["policy"].astype(str))
missing = sorted(required - found)
if missing:
    raise SystemExit(f"FAIL: missing policy summaries: {missing}")
cols = [c for c in [
    "policy", "frames_with_valid_metrics", "mean_absrel", "mean_rmse",
    "mean_delta1", "state_writeback_count", "auto_restart_count",
    "causal_budget_target", "causal_budget_forced_decisions"
] if c in df.columns]
print("\n===== P2.3 v5.7 SMOKE SUMMARY =====")
print(df[cols].to_string(index=False))
print("\nPASS: all requested policy summaries exist.")
print("NOTE: if auto-restart preemption prevents K=3 writes, inspect the logged action frames before proceeding.")
PY

ZIP="outputs/p2_3_v5_7_equal_budget/crowd2_32_smoke_review.zip"
rm -f "$ZIP"
zip -q -r "$ZIP" "$OUT"
zip -q "$ZIP" "$OUT.run.log" "$SCRIPT"
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
