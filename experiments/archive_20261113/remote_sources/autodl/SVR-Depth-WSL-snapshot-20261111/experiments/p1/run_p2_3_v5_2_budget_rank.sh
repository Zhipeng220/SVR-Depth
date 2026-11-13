#!/usr/bin/env bash
# P2.3 v5.2: label-supported, fixed-budget score-ranking diagnostic.
set -uo pipefail
cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2
conda activate svrdepth

SCRIPT="experiments/p1/p2_3_online_replay_v5_2_budget_rank.py"
SELECTOR="experiments/p1/p2_3_select_topk_budget_frames.py"
DATASET="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
BASE="outputs/p2_3_v5_2_budget_rank"

for f in "$SCRIPT" "$SELECTOR" "$DATASET"; do
  if [[ ! -s "$f" ]]; then echo "ERROR: missing $f"; exit 2; fi
done
python -m py_compile "$SCRIPT" "$SELECTOR" || exit 2
mkdir -p "$BASE"

for seq in balloon2 crowd3; do
  root="/root/autodl-tmp/datasets/rgbd_bonn_${seq}"
  if [[ ! -s "$root/rgb.txt" || ! -s "$root/depth.txt" ]]; then
    echo "ERROR: expected rgb.txt/depth.txt under $root"; exit 2
  fi

  # Build allowed anchors from eligible H=16 labels. No target values are used.
  allowed="$(python - "$DATASET" "$seq" <<'PY'
import sys, pandas as pd
p, seq = sys.argv[1], sys.argv[2]
d = pd.read_csv(p)
d["sequence"] = d["sequence"].astype(str).str.lower()
d["anchor_frame"] = pd.to_numeric(d["anchor_frame"], errors="coerce")
d["horizon_H"] = pd.to_numeric(d["horizon_H"], errors="coerce")
ok = d["eligible"].astype(str).str.lower().isin(["true", "1", "yes"])
d = d[(d.sequence == seq) & (d.horizon_H == 16) & ok]
a = sorted({int(x) for x in d.anchor_frame.dropna()
            if int(x) >= 5 and (int(x)-1) % 4 == 0})
print(",".join(map(str, a)))
PY
)"
  if [[ -z "$allowed" ]]; then echo "ERROR: no eligible candidate frames for $seq"; exit 2; fi

  echo "===== $seq eligible H=16 candidate frames: $allowed ====="
  score_out="$BASE/${seq}_score_only"
  mkdir -p "$score_out"

  # Pass 1: score the baseline state trajectory with no scheduled writes.
  # Huge threshold prevents policy-triggered writes; automatic restarts remain active.
  python "$SCRIPT" \
    --data-root "$root" --sequence "$seq" --frames 86 \
    --max-time-diff 0.02 --gt-scale 5000 --resolution-level 0 \
    --train-dataset "$DATASET" \
    --policies auto,learned_state_only,motion_plus_learned \
    --scheduled-action state_writeback \
    --fixed-interval 8 --min-refresh-gap 4 \
    --candidate-stride 4 --candidate-offset 1 \
    --allowed-action-frames "$allowed" \
    --threshold-mode fixed --threshold 1000000000 \
    --horizon 16 --out-dir "$score_out" \
    2>&1 | tee "$BASE/${seq}_score_only_run.log"
  if [[ ! -s "$score_out/p2_3_online_frame_metrics.csv" ]]; then
    echo "ERROR: score-only CSV missing for $seq"; exit 1
  fi

  for model in learned_state_only motion_plus_learned; do
    selection_json="$BASE/${seq}_${model}_topk_selection.json"
    frames="$(python "$SELECTOR" \
      --dataset "$DATASET" \
      --score-csv "$score_out/p2_3_online_frame_metrics.csv" \
      --sequence "$seq" --model "$model" --horizon 16 \
      --candidate-stride 4 --candidate-offset 1 --fixed-interval 8 \
      --out-json "$selection_json" --print-frames)"
    if [[ -z "$frames" ]]; then echo "ERROR: top-K frame selection failed: $seq/$model"; exit 1; fi
    echo "Top-K schedule $seq/$model: $frames"

    replay_out="$BASE/${seq}_budget_${model}"
    mkdir -p "$replay_out"
    python "$SCRIPT" \
      --data-root "$root" --sequence "$seq" --frames 86 \
      --max-time-diff 0.02 --gt-scale 5000 --resolution-level 0 \
      --train-dataset "$DATASET" \
      --policies auto,candidate_fixed,budget_topk \
      --scheduled-action state_writeback \
      --fixed-interval 8 --min-refresh-gap 4 \
      --candidate-stride 4 --candidate-offset 1 \
      --allowed-action-frames "$allowed" \
      --forced-action-frames "$frames" \
      --threshold-mode fixed --threshold 0 \
      --horizon 16 --out-dir "$replay_out" \
      2>&1 | tee "$BASE/${seq}_budget_${model}_run.log"
    if [[ ! -s "$replay_out/p2_3_online_summary.csv" ]]; then
      echo "ERROR: replay summary missing: $seq/$model"; exit 1
    fi
  done
done

ZIP="$BASE/p2_3_v5_2_budget_rank_review.zip"
rm -f "$ZIP"
zip -qr "$ZIP" "$SCRIPT" "$SELECTOR" "$DATASET" "$BASE"
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
