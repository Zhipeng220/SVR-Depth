#!/usr/bin/env bash
# P2.3 v5.3 causal online budget validation on all three Bonn sequences.
set -uo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

# Avoid non-interactive Conda activation errors when the caller is already in svrdepth.
if [[ "${CONDA_DEFAULT_ENV:-}" != "svrdepth" ]]; then
  [[ -f /root/miniconda3/etc/profile.d/conda.sh ]] && source /root/miniconda3/etc/profile.d/conda.sh
  conda activate svrdepth || exit 2
fi

SCRIPT="experiments/p1/p2_3_online_replay_v5_3_causal_budget.py"
DATASET="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
BASE="outputs/p2_3_v5_3_causal_budget"
if [[ ! -s "$SCRIPT" ]]; then echo "ERROR: missing $SCRIPT"; exit 2; fi
if [[ ! -s "$DATASET" ]]; then echo "ERROR: missing $DATASET"; exit 2; fi
python -m py_compile "$SCRIPT" || exit 2
mkdir -p "$BASE"

for seq in crowd2 balloon2 crowd3; do
  root="/root/autodl-tmp/datasets/rgbd_bonn_${seq}"
  out="$BASE/${seq}_86"
  log="$BASE/${seq}_86_run.log"
  if [[ ! -s "$root/rgb.txt" || ! -s "$root/depth.txt" ]]; then
    echo "ERROR: expected rgb.txt and depth.txt under $root"
    continue
  fi
  mkdir -p "$out"
  echo "===== P2.3 v5.3 CAUSAL BUDGET: $seq ====="
  if python "$SCRIPT" \
      --data-root "$root" --sequence "$seq" --frames 86 \
      --max-time-diff 0.02 --gt-scale 5000 --resolution-level 0 \
      --train-dataset "$DATASET" \
      --policies auto,candidate_fixed,causal_budget_state_only,causal_budget_motion_plus_learned \
      --scheduled-action state_writeback --budget-source fixed_baseline \
      --fixed-interval 8 --min-refresh-gap 4 \
      --candidate-stride 4 --candidate-offset 1 --horizon 16 \
      --out-dir "$out" 2>&1 | tee "$log"; then
    if [[ -s "$out/p2_3_online_summary.csv" && -s "$out/p2_3_online_frame_metrics.csv" && -s "$out/p2_3_online_metadata.json" ]]; then
      echo "PASS: complete v5.3 outputs for $seq"
    else
      echo "ERROR: $seq returned but result files are missing"
    fi
  else
    echo "FAIL: v5.3 replay failed for $seq; inspect $log"
  fi
done

ZIP="$BASE/p2_3_v5_3_causal_budget_review.zip"
rm -f "$ZIP"
zip -q "$ZIP" "$SCRIPT" "$DATASET"
# Add every output/log file except the ZIP being built (avoid self-inclusion).
find "$BASE" -type f ! -name "$(basename "$ZIP")" -print0 |
  while IFS= read -r -d '' file; do zip -q "$ZIP" "$file"; done
echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
