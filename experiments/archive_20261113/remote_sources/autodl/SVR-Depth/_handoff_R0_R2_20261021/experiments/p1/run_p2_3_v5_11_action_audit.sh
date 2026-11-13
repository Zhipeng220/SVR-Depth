#!/usr/bin/env bash
# P2.3 v5.11: causal-budget action-selection audit.
# Run from the SVR-Depth repository after activating the svrdepth conda environment.
set -uo pipefail

cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

SCRIPT="experiments/p1/p2_3_online_replay_v5_11_action_audit.py"
TRAIN_CSV="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
OUT="outputs/p2_3_v5_11_action_audit"
TRAIN_SEQUENCES="crowd2,balloon2,crowd3"
POLICIES="auto,budget_fixed,causal_budget_state_age,causal_budget_state_age_stratified,causal_budget_motion_plus_learned,causal_budget_motion_plus_learned_stratified"

if [[ ! -s "$SCRIPT" ]]; then
  echo "ERROR: upload $SCRIPT into experiments/p1/ first."
  exit 2
fi
if [[ ! -s "$TRAIN_CSV" ]]; then
  echo "ERROR: missing training data $TRAIN_CSV"
  exit 2
fi

mkdir -p "$OUT"
python -m py_compile "$SCRIPT" || exit 2

echo "===== P2.3 v5.11 CONTROLLER SELF-TEST ====="
python "$SCRIPT" --controller-self-test 2>&1 | tee "$OUT/controller_self_test.log"
RC=${PIPESTATUS[0]}
if [[ "$RC" -ne 0 ]]; then
  echo "ERROR: controller self-test failed."
  exit "$RC"
fi

# Reuse the two external sequences already evaluated in v5.10.
# This is a mechanism-ablation rerun on the same holdout clips, not new independent validation.
STATUS="$OUT/p2_3_v5_11_run_status.tsv"
printf "sequence\tdata_root\tstatus\texit_code\n" > "$STATUS"
OVERALL=0

run_one() {
  local SEQ="$1"
  local DATA_ROOT="$2"
  local SEQ_OUT="$OUT/${SEQ}_86"
  local LOG="$OUT/${SEQ}_86_run.log"

  if [[ ! -s "$DATA_ROOT/rgb.txt" || ! -s "$DATA_ROOT/depth.txt" ]]; then
    echo "ERROR: missing rgb.txt or depth.txt in $DATA_ROOT"
    printf "%s\t%s\tFAIL\t2\n" "$SEQ" "$DATA_ROOT" >> "$STATUS"
    OVERALL=1
    return
  fi

  mkdir -p "$SEQ_OUT"
  echo "===== P2.3 v5.11 ACTION AUDIT: $SEQ ====="
  python "$SCRIPT" \
    --sequence "$SEQ" \
    --data-root "$DATA_ROOT" \
    --frames 86 \
    --max-time-diff 0.02 \
    --train-dataset "$TRAIN_CSV" \
    --training-sequences "$TRAIN_SEQUENCES" \
    --policies "$POLICIES" \
    --scheduled-action state_writeback \
    --fixed-interval 8 \
    --min-refresh-gap 8 \
    --budget-source explicit \
    --budget-actions 8 \
    --time-strata 4 \
    --candidate-stride 4 \
    --candidate-offset 1 \
    --threshold-mode train_oof_budget \
    --target-refresh-fraction 0.42857142857142855 \
    --horizon 16 \
    --ridge-alpha 10 \
    --out-dir "$SEQ_OUT" \
    2>&1 | tee "$LOG"
  local RC=${PIPESTATUS[0]}

  if [[ "$RC" -eq 0 && -s "$SEQ_OUT/p2_3_online_summary.csv" && -s "$SEQ_OUT/p2_3_online_frame_metrics.csv" && -s "$SEQ_OUT/p2_3_online_metadata.json" ]]; then
    printf "%s\t%s\tPASS\t0\n" "$SEQ" "$DATA_ROOT" >> "$STATUS"
  else
    printf "%s\t%s\tFAIL\t%s\n" "$SEQ" "$DATA_ROOT" "$RC" >> "$STATUS"
    OVERALL=1
    echo "FAIL: $SEQ; inspect $LOG"
  fi
}

run_one "person_tracking" "/root/autodl-tmp/datasets/rgbd_bonn_person_tracking"
run_one "synchronous" "/root/autodl-tmp/datasets/rgbd_bonn_synchronous"

# Create one compact action-decision audit table from the per-sequence summaries.
python - "$OUT" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
rows = []
for seq_dir in sorted(root.glob("*_86")):
    p = seq_dir / "p2_3_online_summary.csv"
    if not p.is_file():
        continue
    df = pd.read_csv(p, encoding="utf-8-sig")
    df.insert(0, "sequence_dir", seq_dir.name)
    rows.append(df)

if not rows:
    raise SystemExit("No summary CSVs found; cannot write action audit.")
out = pd.concat(rows, ignore_index=True)
preferred = [
    "sequence", "policy", "frames_with_valid_metrics",
    "mean_absrel", "mean_rmse", "mean_delta1",
    "state_writeback_count", "causal_budget_target",
    "causal_budget_successful_writebacks", "causal_budget_unused",
    "causal_budget_utilization", "causal_budget_met",
    "causal_score_decisions", "causal_score_rejections",
    "causal_score_selected_writebacks", "causal_forced_writebacks",
    "causal_writeback_frames", "causal_score_selected_writeback_frames",
    "causal_forced_writeback_frames", "causal_budget_forced_decisions",
    "auto_restart_count", "keyframe_candidate_compute_count",
    "mean_decision_ms", "mean_feature_extraction_ms",
    "mean_predictor_inference_ms", "mean_budget_control_ms",
]
cols = [c for c in preferred if c in out.columns]
out[cols].to_csv(root / "p2_3_v5_11_action_decision_audit.csv", index=False, encoding="utf-8-sig")
print("Wrote:", root / "p2_3_v5_11_action_decision_audit.csv")
print(out[cols].to_string(index=False))
PY
if [[ "$?" -ne 0 ]]; then
  OVERALL=1
fi

ZIP="outputs/p2_3_v5_11_action_audit_review.zip"
rm -f "$ZIP"
zip -q -r "$ZIP" "$OUT"
zip -q "$ZIP" "$SCRIPT" "experiments/p1/run_p2_3_v5_11_action_audit.sh" "experiments/p1/README_P2_3_v5_11_action_audit.md"
echo "===== ZIP INTEGRITY ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
exit "$OVERALL"
