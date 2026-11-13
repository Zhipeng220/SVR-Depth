#!/usr/bin/env bash
# P2.3 v5.10: discover unused Bonn sequences and run strict external-holdout replay.
# Run from the repository after: conda activate svrdepth
set -uo pipefail

cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

PREFLIGHT="experiments/p1/p2_3_v5_10_dataset_preflight.py"
SCRIPT="experiments/p1/p2_3_online_replay_v5_10_external.py"
TRAIN_CSV="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
DATASET_ROOT="/root/autodl-tmp/datasets"
OUT="outputs/p2_3_v5_10_external"
TRAIN_SEQUENCES="crowd2,balloon2,crowd3"
POLICIES="auto,budget_fixed,causal_budget_state_age,causal_budget_motion_plus_learned,causal_budget_motion_plus_learned_stratified"

if [[ ! -s "$PREFLIGHT" || ! -s "$SCRIPT" ]]; then
  echo "ERROR: upload $PREFLIGHT and $SCRIPT to experiments/p1/ first."
  exit 2
fi
if [[ ! -s "$TRAIN_CSV" ]]; then
  echo "ERROR: missing P2.2.2 predictor dataset: $TRAIN_CSV"
  exit 2
fi
mkdir -p "$OUT"
python -m py_compile "$PREFLIGHT" "$SCRIPT" || exit 2

# Inventory only: CPU and filesystem checks, no model loaded.
python "$PREFLIGHT" \
  --dataset-root "$DATASET_ROOT" \
  --frames 86 \
  --max-time-diff 0.02 \
  --min-gt-matches 80 \
  --used-sequences "$TRAIN_SEQUENCES" \
  --out-csv "$OUT/p2_3_v5_10_preflight.csv" \
  2>&1 | tee "$OUT/p2_3_v5_10_preflight.log"
rc=${PIPESTATUS[0]}
if [[ "$rc" -ne 0 ]]; then
  echo "ERROR: data preflight failed with code $rc"
  exit "$rc"
fi

# Export eligible candidates as TSV to avoid parsing paths through shell commas.
python - "$OUT/p2_3_v5_10_preflight.csv" > "$OUT/eligible_external_sequences.tsv" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if str(r.get("eligible_external_holdout", "")).strip().lower() in {"true", "1", "yes"}:
            print(f"{r['sequence']}\t{r['data_root']}")
PY

STATUS="$OUT/p2_3_v5_10_run_status.tsv"
printf "sequence\tdata_root\tstatus\texit_code\n" > "$STATUS"
OVERALL=0

if [[ ! -s "$OUT/eligible_external_sequences.tsv" ]]; then
  echo "NO ELIGIBLE EXTERNAL SEQUENCE: no GPU replay was started."
  printf "NO_ELIGIBLE_EXTERNAL_SEQUENCE\t-\tSKIPPED\t0\n" >> "$STATUS"
else
  while IFS=$'\t' read -r SEQ DATA_ROOT; do
    [[ -n "$SEQ" ]] || continue
    SEQ_OUT="$OUT/${SEQ}_86"
    LOG="$OUT/${SEQ}_86_run.log"
    mkdir -p "$SEQ_OUT"
    echo "===== P2.3 v5.10 EXTERNAL HOLDOUT: $SEQ ====="
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
    RC=${PIPESTATUS[0]}
    if [[ "$RC" -eq 0 && -s "$SEQ_OUT/p2_3_online_summary.csv" && -s "$SEQ_OUT/p2_3_online_metadata.json" ]]; then
      printf "%s\t%s\tPASS\t0\n" "$SEQ" "$DATA_ROOT" >> "$STATUS"
    else
      printf "%s\t%s\tFAIL\t%s\n" "$SEQ" "$DATA_ROOT" "$RC" >> "$STATUS"
      OVERALL=1
      echo "FAIL: $SEQ; inspect $LOG"
    fi
  done < "$OUT/eligible_external_sequences.tsv"
fi

# Keep the archive compact: metrics, metadata, preflight, and logs only.
ZIP="outputs/p2_3_v5_10_external_review.zip"
rm -f "$ZIP"
zip -q -r "$ZIP" "$OUT"
zip -q "$ZIP" "$PREFLIGHT" "$SCRIPT" \
  experiments/p1/run_p2_3_v5_10_external.sh \
  experiments/p1/README_P2_3_v5_10.md
echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
exit "$OVERALL"
