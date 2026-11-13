#!/usr/bin/env bash
# P2.3 v5.4 pre-specified budget sweep: K=5,8,10 across three Bonn sequences.
set -uo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

if [[ "${CONDA_DEFAULT_ENV:-}" != "svrdepth" ]]; then
  [[ -f /root/miniconda3/etc/profile.d/conda.sh ]] && source /root/miniconda3/etc/profile.d/conda.sh
  conda activate svrdepth || exit 2
fi

SCRIPT="experiments/p1/p2_3_online_replay_v5_4_budget_sweep.py"
ANALYZER="experiments/p1/analyze_p2_3_v5_4_budget_sweep.py"
DATASET="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
BASE="outputs/p2_3_v5_4_budget_sweep"

for f in "$SCRIPT" "$ANALYZER" "$DATASET"; do
  if [[ ! -s "$f" ]]; then echo "ERROR: missing required file: $f"; exit 2; fi
done
python -m py_compile "$SCRIPT" "$ANALYZER" || exit 2
mkdir -p "$BASE"
failed=0

for seq in crowd2 balloon2 crowd3; do
  data_root="/root/autodl-tmp/datasets/rgbd_bonn_${seq}"
  if [[ ! -s "$data_root/rgb.txt" || ! -s "$data_root/depth.txt" ]]; then
    echo "ERROR: missing rgb.txt/depth.txt under $data_root"
    failed=1
    continue
  fi

  for K in 5 8 10; do
    out="$BASE/${seq}_K${K}"
    log="$BASE/${seq}_K${K}_run.log"
    mkdir -p "$out"
    echo "===== P2.3 v5.4 sequence=$seq, pre-specified K=$K ====="
    if python "$SCRIPT" \
        --data-root "$data_root" \
        --sequence "$seq" \
        --frames 86 \
        --max-time-diff 0.02 \
        --gt-scale 5000 \
        --resolution-level 0 \
        --train-dataset "$DATASET" \
        --policies auto,budget_fixed,causal_budget_state_only,causal_budget_motion_plus_learned \
        --scheduled-action state_writeback \
        --budget-source explicit \
        --budget-actions "$K" \
        --fixed-interval 8 \
        --min-refresh-gap 4 \
        --candidate-stride 4 \
        --candidate-offset 1 \
        --horizon 16 \
        --out-dir "$out" \
        2>&1 | tee "$log"; then
      if [[ -s "$out/p2_3_online_summary.csv" && -s "$out/p2_3_online_frame_metrics.csv" && -s "$out/p2_3_online_metadata.json" ]]; then
        echo "PASS: complete outputs for $seq K=$K"
      else
        echo "ERROR: output files missing for $seq K=$K"
        failed=1
      fi
    else
      echo "FAIL: replay failed for $seq K=$K; inspect $log"
      failed=1
    fi
  done
done

echo "===== Post-hoc predictor/selection diagnostics ====="
python "$ANALYZER" --root "$BASE" --dataset "$DATASET" --sequences crowd2,balloon2,crowd3 --budgets 5,8,10 || failed=1

ZIP="$BASE/p2_3_v5_4_budget_sweep_review.zip"
rm -f "$ZIP"
zip -q "$ZIP" "$SCRIPT" "$ANALYZER" "$DATASET"
find "$BASE" -type f ! -name "$(basename "$ZIP")" -print0 |
  while IFS= read -r -d '' file; do zip -q "$ZIP" "$file"; done

echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
if [[ "$failed" -ne 0 ]]; then
  echo "WARNING: at least one sequence/budget run failed; inspect the per-run logs."
  exit 1
fi
