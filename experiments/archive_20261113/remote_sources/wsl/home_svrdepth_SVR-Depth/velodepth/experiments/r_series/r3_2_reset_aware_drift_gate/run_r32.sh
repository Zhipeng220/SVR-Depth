#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
output_root="$project_root/outputs/r_series/r3_2_reset_aware_drift_gate"
svr_python="${SVR_PYTHON:-/root/miniconda3/envs/svrdepth/bin/python}"
if [[ ! -x "$svr_python" ]]; then svr_python="$(command -v python3 || command -v python)"; fi
stage="${1:-preflight}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$project_root"

AGE_THRESHOLD="${R32_AGE_THRESHOLD:-16}"
BASELINE_RESIDUAL_THRESHOLD="${R32_BASELINE_RESIDUAL_THRESHOLD:-0.04}"
POLICIES_ALL="AlwaysCommit,NeverCommit,AgeGate,ResidualGate,DriftGate,AgeDriftGate"
POLICIES_SWEEP="ResidualGate,DriftGate,AgeDriftGate"
THRESHOLDS=(0.03 0.04 0.05 0.06 0.08 0.10)

run_tests() {
  PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "$svr_python" -m unittest discover -s "$script_dir" -p 'test_r3*.py' -v
}
run_root() {
  if [[ -n "${R32_ROOT:-}" ]]; then printf '%s\n' "$R32_ROOT"; return; fi
  [[ -f "$output_root/LATEST_RUN_PATH.txt" ]] || { echo "No R3.2 run root. Run 'start' first." >&2; return 2; }
  cat "$output_root/LATEST_RUN_PATH.txt"
}
latest_stage_path() {
  local marker="$1"
  [[ -f "$marker" ]] || { echo "Missing stage marker: $marker" >&2; return 2; }
  cat "$marker"
}

case "$stage" in
  start)
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    root="$output_root/run_$stamp"
    mkdir -p "$root"
    printf '%s\n' "$root" > "$output_root/LATEST_RUN_PATH.txt"
    cat > "$root/R3_2_PROTOCOL_LOCK.json" <<EOF
{
  "experiment": "R3.2 reset-aware drift commit gate",
  "created_utc": "$stamp",
  "frames": 86,
  "refresh_frames_zero_based": [9, 37, 65],
  "age_threshold": $AGE_THRESHOLD,
  "baseline_residual_threshold": $BASELINE_RESIDUAL_THRESHOLD,
  "sensitivity_thresholds": [0.03, 0.04, 0.05, 0.06, 0.08, 0.10],
  "sensitivity_policies": ["ResidualGate", "DriftGate", "AgeDriftGate"],
  "weights_frozen": true,
  "predictor_training": false,
  "gate_uses_gt": false,
  "refresh_candidate_compute_skipped": false,
  "threshold_selection_on_heldout": false
}
EOF
    echo "R3.2 run root: $root"
    ;;
  preflight)
    run_tests
    root="$(run_root)"
    stamp="$(date -u +%Y%m%dT%H%M%S_%NZ)"
    out="$root/preflight/$stamp"
    mkdir -p "$(dirname "$out")"
    "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" --preflight-only --out-dir "$out"
    ;;
  smoke)
    run_tests
    root="$(run_root)"
    stamp="$(date -u +%Y%m%dT%H%M%S_%NZ)"
    out="$root/smoke/$stamp"
    mkdir -p "$(dirname "$out")"
    "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" \
      --data-root /root/autodl-tmp/datasets/rgbd_bonn_crowd2 --sequence crowd2 \
      --frames 32 --refresh-frames 5,14,23 --age-threshold "$AGE_THRESHOLD" \
      --residual-threshold "$BASELINE_RESIDUAL_THRESHOLD" --policies "$POLICIES_ALL" --out-dir "$out"
    ;;
  baseline)
    run_tests
    root="$(run_root)"
    stamp="$(date -u +%Y%m%dT%H%M%S_%NZ)"
    base="$root/baseline/$stamp"
    mkdir -p "$base"
    for seq in crowd2 balloon2 crowd3; do
      echo "=== R3.2 baseline: $seq ==="
      "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" \
        --data-root "/root/autodl-tmp/datasets/rgbd_bonn_${seq}" --sequence "$seq" \
        --frames 86 --refresh-frames 9,37,65 --age-threshold "$AGE_THRESHOLD" \
        --residual-threshold "$BASELINE_RESIDUAL_THRESHOLD" --policies "$POLICIES_ALL" \
        --out-dir "$base/$seq"
    done
    printf '%s\n' "$base" > "$root/LATEST_BASELINE_PATH.txt"
    echo "Baseline root: $base"
    ;;
  sensitivity)
    run_tests
    root="$(run_root)"
    stamp="$(date -u +%Y%m%dT%H%M%S_%NZ)"
    base="$root/sensitivity/$stamp"
    mkdir -p "$base"
    for seq in crowd2 balloon2 crowd3; do
      for threshold in "${THRESHOLDS[@]}"; do
        slug="${threshold/./p}"
        echo "=== R3.2 sensitivity: sequence=$seq threshold=$threshold policies=$POLICIES_SWEEP ==="
        "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" \
          --data-root "/root/autodl-tmp/datasets/rgbd_bonn_${seq}" --sequence "$seq" \
          --frames 86 --refresh-frames 9,37,65 --age-threshold "$AGE_THRESHOLD" \
          --residual-threshold "$threshold" --policies "$POLICIES_SWEEP" \
          --out-dir "$base/$seq/thr_$slug"
      done
    done
    "$svr_python" "$script_dir/summarize_r32_sweep.py" "$base"
    printf '%s\n' "$base" > "$root/LATEST_SENSITIVITY_PATH.txt"
    echo "Sensitivity root: $base"
    ;;
  summarize)
    root="$(run_root)"
    base="$(latest_stage_path "$root/LATEST_SENSITIVITY_PATH.txt")"
    "$svr_python" "$script_dir/summarize_r32_sweep.py" "$base"
    ;;
  pack)
    root="$(run_root)"
    archive="${root}_review.zip"
    [[ ! -e "$archive" ]] || { echo "Archive already exists: $archive" >&2; exit 2; }
    "$svr_python" - "$root" "$archive" <<'PY'
import pathlib, sys, zipfile
root=pathlib.Path(sys.argv[1]).resolve(); archive=pathlib.Path(sys.argv[2]).resolve()
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(root.parent).as_posix())
    if z.testzip() is not None: raise RuntimeError('ZIP integrity failure')
print(archive)
PY
    ;;
  *)
    echo "Usage: bash experiments/r_series/r3_2_reset_aware_drift_gate/run_r32.sh {start|preflight|smoke|baseline|sensitivity|summarize|pack}" >&2
    echo "Optional env: SVR_PYTHON, R32_ROOT, R32_AGE_THRESHOLD (default 16), R32_BASELINE_RESIDUAL_THRESHOLD (default 0.04)" >&2
    exit 2
    ;;
esac
