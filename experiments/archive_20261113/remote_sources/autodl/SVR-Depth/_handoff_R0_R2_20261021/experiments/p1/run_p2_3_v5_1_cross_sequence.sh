#!/usr/bin/env bash
# P2.3 v5.1: held-out Balloon2 and Crowd3 replay with B1-aligned state writeback.
# Run after: conda activate svrdepth
set -uo pipefail

cd /root/autodl-tmp/SVR-Depth/velodepth || exit 2

SCRIPT="experiments/p1/p2_3_online_replay_v5_1_writeback_gtmask.py"
DATASET="outputs/p2_2_2/p2_2_2_valid_dataset.csv"
BASE="outputs/p2_3_v5_cross_sequence"

if [[ ! -s "$SCRIPT" ]]; then
  echo "ERROR: upload $SCRIPT first"
  exit 2
fi
if [[ ! -s "$DATASET" ]]; then
  echo "ERROR: P2.2.2 dataset missing: $DATASET"
  exit 2
fi
python -m py_compile "$SCRIPT" || exit 2
mkdir -p "$BASE"

echo "===== Basic dataset path checks ====="
for seq in balloon2 crowd3; do
  root="/root/autodl-tmp/datasets/rgbd_bonn_${seq}"
  if [[ ! -s "$root/rgb.txt" || ! -s "$root/depth.txt" ]]; then
    echo "ERROR: expected rgb.txt/depth.txt under $root"
    echo "Inspect /root/autodl-tmp/datasets before running."
    exit 2
  fi
  echo "PASS: $seq -> $root"
done

# CPU-only preflight: fit Ridge and perform training-only temporal-block OOF
# threshold calibration; this does not load the VeloDepth model or require CUDA.
for seq in balloon2 crowd3; do
  out="$BASE/preflight_${seq}"
  log="$BASE/preflight_${seq}.log"
  mkdir -p "$out"
  echo "===== CPU PREFLIGHT: held-out $seq ====="
  if python "$SCRIPT" \
      --sequence "$seq" \
      --policies learned_state_only,motion_plus_learned \
      --horizon 16 \
      --scheduled-action state_writeback \
      --calibration-only \
      --train-dataset "$DATASET" \
      --out-dir "$out" \
      2>&1 | tee "$log"; then
    echo "PASS: CPU calibration preflight for $seq"
  else
    echo "FAIL: CPU calibration preflight for $seq; continuing to the other sequence"
  fi
done

# Full 86-frame replay. The local GT matcher preserves every RGB frame; any RGB
# without a GT within 20 ms remains in the temporal state stream but is excluded
# from that frame's metric aggregation.
for seq in balloon2 crowd3; do
  root="/root/autodl-tmp/datasets/rgbd_bonn_${seq}"
  out="$BASE/${seq}_86"
  log="$BASE/${seq}_86_run.log"
  mkdir -p "$out"
  echo "===== GPU REPLAY: held-out $seq ====="
  if python "$SCRIPT" \
      --data-root "$root" \
      --sequence "$seq" \
      --frames 86 \
      --max-time-diff 0.02 \
      --gt-scale 5000 \
      --resolution-level 0 \
      --train-dataset "$DATASET" \
      --policies auto,candidate_fixed,learned_state_only,motion_plus_learned \
      --scheduled-action state_writeback \
      --fixed-interval 8 \
      --min-refresh-gap 4 \
      --candidate-stride 4 \
      --candidate-offset 1 \
      --horizon 16 \
      --out-dir "$out" \
      2>&1 | tee "$log"; then
    if [[ -s "$out/p2_3_online_summary.csv" && -s "$out/p2_3_online_metadata.json" ]]; then
      echo "PASS: replay outputs exist for $seq"
    else
      echo "FAIL: replay returned but summary/metadata missing for $seq"
    fi
  else
    echo "FAIL: replay failed for $seq; inspect $log"
  fi
done

# Package any outputs that exist, even if one sequence failed, so diagnostics
# are not lost.
ZIP="$BASE/p2_3_v5_1_cross_sequence_review.zip"
rm -f "$ZIP"
zip -qr "$ZIP" "$SCRIPT" "$DATASET"
for seq in balloon2 crowd3; do
  [[ -d "$BASE/preflight_${seq}" ]] && zip -qr "$ZIP" "$BASE/preflight_${seq}"
  [[ -f "$BASE/preflight_${seq}.log" ]] && zip -q "$ZIP" "$BASE/preflight_${seq}.log"
  [[ -d "$BASE/${seq}_86" ]] && zip -qr "$ZIP" "$BASE/${seq}_86"
  [[ -f "$BASE/${seq}_86_run.log" ]] && zip -q "$ZIP" "$BASE/${seq}_86_run.log"
done

echo "===== ZIP integrity ====="
unzip -t "$ZIP" | tail -n 3
ls -lh "$ZIP"
echo "Review ZIP: $ZIP"
