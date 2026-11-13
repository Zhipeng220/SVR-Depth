#!/usr/bin/env bash
# P2.3 v5.5: same-state counterfactual gain diagnosis on Bonn Crowd2/Balloon2/Crowd3.
set -uo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

if [[ "${CONDA_DEFAULT_ENV:-}" != "svrdepth" ]]; then
  if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
    source /root/miniconda3/etc/profile.d/conda.sh
  fi
  conda activate svrdepth || exit 2
fi

SCRIPT="experiments/p1/p2_3_v5_5_same_state_counterfactual.py"
HELPER="experiments/p1/p2_3_online_replay_v5_4_budget_sweep.py"
ANALYZER="experiments/p1/analyze_p2_3_v5_5_same_state.py"
DATASET="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
BASE="outputs/p2_3_v5_5_same_state"

for f in "$SCRIPT" "$HELPER" "$ANALYZER" "$DATASET"; do
  if [[ ! -s "$f" ]]; then echo "ERROR: missing required file $f"; exit 2; fi
done
python -m py_compile "$SCRIPT" "$HELPER" "$ANALYZER" || exit 2
mkdir -p "$BASE"

# Use --frames 32 and K=3 first for a smoke test:
# python "$SCRIPT" --data-root /root/autodl-tmp/datasets/rgbd_bonn_crowd2 \
#   --sequence crowd2 --frames 32 --train-dataset "$DATASET" \
#   --budget-actions 3 --out-dir "$BASE/crowd2_32_smoke"

for seq in crowd2 balloon2 crowd3; do
  root="/root/autodl-tmp/datasets/rgbd_bonn_${seq}"
  out="$BASE/${seq}_86"
  log="$BASE/${seq}_86_run.log"
  if [[ ! -s "$root/rgb.txt" || ! -s "$root/depth.txt" ]]; then
    echo "ERROR: expected rgb.txt and depth.txt under $root"
    continue
  fi
  mkdir -p "$out"
  echo "===== P2.3 v5.5 SAME-STATE COUNTERFACTUAL: $seq ====="
  if python "$SCRIPT" \
      --data-root "$root" \
      --sequence "$seq" \
      --frames 86 \
      --train-dataset "$DATASET" \
      --horizon 16 \
      --gamma 0.9 \
      --candidate-stride 4 \
      --candidate-offset 1 \
      --min-refresh-gap 4 \
      --fixed-interval 8 \
      --budget-actions 8 \
      --ridge-alpha 10 \
      --out-dir "$out" 2>&1 | tee "$log"; then
    if [[ -s "$out/p2_3_v5_5_same_state_anchor_labels.csv" \
       && -s "$out/p2_3_v5_5_gain_diagnostics.csv" \
       && -s "$out/p2_3_v5_5_trajectory_alignment.csv" \
       && -s "$out/p2_3_v5_5_metadata.json" ]]; then
      echo "PASS: v5.5 output files found for $seq"
    else
      echo "FAIL: $seq completed without all expected outputs; inspect $log"
    fi
  else
    echo "FAIL: v5.5 replay failed for $seq; inspect $log"
  fi
done

python "$ANALYZER" --root "$BASE" --out-dir "$BASE" || echo "WARNING: aggregate analysis failed; inspect outputs."

ZIP="$BASE/p2_3_v5_5_same_state_review.zip"
rm -f "$ZIP"
zip -q "$ZIP" "$SCRIPT" "$HELPER" "$ANALYZER" "$DATASET"
find "$BASE" -type f ! -name "$(basename "$ZIP")" -print0 |
  while IFS= read -r -d '' file; do zip -q "$ZIP" "$file"; done

echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
