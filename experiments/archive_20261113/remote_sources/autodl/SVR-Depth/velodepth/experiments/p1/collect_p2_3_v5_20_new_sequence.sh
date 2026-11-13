#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash collect_p2_3_v5_20_new_sequence.sh <new_sequence_name> <data_root> [frames=86] [out_dir]" >&2
  echo "Example: bash collect_p2_3_v5_20_new_sequence.sh person /root/autodl-tmp/datasets/rgbd_bonn_person 86" >&2
  exit 2
fi
SEQ_NAME="$1"
DATA_ROOT="$2"
FRAMES="${3:-86}"
OUT_DIR="${4:-outputs/p2_3_v5_20_external/${SEQ_NAME}}"

case "${SEQ_NAME,,}" in
  balloon2|crowd2|crowd3)
    echo "ERROR: ${SEQ_NAME} is part of the locked training set; it cannot be an external test sequence." >&2
    exit 2
    ;;
esac

if [[ ! -s "$DATA_ROOT/rgb.txt" || ! -s "$DATA_ROOT/depth.txt" ]]; then
  echo "ERROR: expected non-empty rgb.txt and depth.txt under: $DATA_ROOT" >&2
  exit 2
fi
if (( FRAMES <= 17 )); then
  echo "ERROR: frames must be > horizon(16) + offset(1). Use 86 for the registered protocol." >&2
  exit 2
fi
for f in \
  "$REPO_ROOT/experiments/p1/p2_3_online_replay_v5_4_budget_sweep.py" \
  "$REPO_ROOT/experiments/p1/p1_multihorizon_counterfactual.py" \
  "$REPO_ROOT/experiments/p1/stream_state.py" \
  "$REPO_ROOT/experiments/p1/stream_state_snapshot.py" \
  "$REPO_ROOT/experiments/p1/velodepth_stream.py"; do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: missing required project helper: $f" >&2
    echo "Compare/copy the reference dependencies bundled in this v5.20 package only if needed." >&2
    exit 2
  fi
done

cd "$REPO_ROOT"
export OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-/root/autodl-tmp/SVR-Depth/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

mkdir -p "$(dirname "$OUT_DIR")"
python -m py_compile "$SCRIPT_DIR/p2_3_v5_20_external_sequence_collector.py"
python "$SCRIPT_DIR/p2_3_v5_20_external_sequence_collector.py" \
  --data-root "$DATA_ROOT" \
  --sequence "$SEQ_NAME" \
  --frames "$FRAMES" \
  --horizon 16 \
  --gamma 0.9 \
  --candidate-stride 4 \
  --candidate-offset 1 \
  --min-refresh-gap 4 \
  --fixed-interval 8 \
  --budget-actions 8 \
  --policies auto \
  --out-dir "$OUT_DIR" \
  2>&1 | tee "${OUT_DIR}_collector.log"

LABEL_CSV="$OUT_DIR/p2_3_v5_15_same_state_anchor_labels.csv"
if [[ ! -s "$LABEL_CSV" ]]; then
  echo "ERROR: label CSV not produced: $LABEL_CSV" >&2
  exit 3
fi
python - "$LABEL_CSV" "$SEQ_NAME" <<'PY'
import sys, pandas as pd
p, seq = sys.argv[1:]
df = pd.read_csv(p)
need = {'sequence','trajectory_policy','anchor_frame','horizon_H','status','G_state_absrel_restart_aware_discounted_per_valid_frame','feature_scheduler_distance_kf','feature_state_features_teacher_2_mean','future_metric_pairs','future_frames_requested'}
missing = sorted(need-set(df.columns))
if missing: raise SystemExit(f'Missing label fields: {missing}')
df = df[(df.sequence.astype(str).str.lower()==seq.lower()) & (df.trajectory_policy.astype(str).str.lower()=='auto') & (pd.to_numeric(df.horizon_H,errors='coerce')==16) & (df.status.astype(str).str.lower()=='ok')]
if df.anchor_frame.duplicated().any(): raise SystemExit('Duplicate auto/H=16 anchor labels')
print(f'LABEL CHECK: sequence={seq}; valid_auto_H16_anchors={len(df)}; anchors={sorted(pd.to_numeric(df.anchor_frame).astype(int).tolist())}')
if len(df) < 12: raise SystemExit('Too few valid H=16 anchors (<12); do not proceed to external model evaluation')
PY

echo "External label collection complete: $LABEL_CSV"
echo "Next run the CPU evaluator with this exact CSV; do not modify the sequence name to disguise training footage."
