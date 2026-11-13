#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
svr_python="${SVR_PYTHON:-/root/miniconda3/envs/svrdepth/bin/python}"
if [[ ! -x "$svr_python" ]]; then svr_python="$(command -v python3 || command -v python)"; fi
stage="${1:-preflight}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$project_root"

run_tests() {
  PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "$svr_python" -m unittest discover -s "$script_dir" -p 'test_r3*.py' -v
}
latest_sensitivity_root() {
  if [[ -n "${R3_SENSITIVITY_ROOT:-}" ]]; then
    printf '%s\n' "$R3_SENSITIVITY_ROOT"
    return
  fi
  local root="$project_root/outputs/r_series/r3_residual_sensitivity"
  [[ -d "$root" ]] || { echo "Missing sensitivity root: $root" >&2; return 2; }
  find "$root" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' | sort | tail -n 1
}
run_baseline() {
  local stamp base marker seq
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  base="$project_root/outputs/r_series/r3_gate_crossval/$stamp"
  marker="$project_root/outputs/r_series/r3_gate_crossval/LATEST_BASELINE_PATH.txt"
  mkdir -p "$base/baseline"
  printf '%s\n' "$base" > "$marker"
  local residual_threshold="${R3_RESIDUAL_THRESHOLD:-0.10}"
  local age_threshold="${R3_AGE_THRESHOLD:-16}"
  echo "R3 full baseline root: $base"
  for seq in crowd2 balloon2 crowd3; do
    echo "=== Full gate comparison: sequence=$seq ==="
    "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" \
      --data-root "/root/autodl-tmp/datasets/rgbd_bonn_${seq}" --sequence "$seq" \
      --frames 86 --refresh-frames 9,37,65 \
      --age-threshold "$age_threshold" --residual-threshold "$residual_threshold" \
      --policies AlwaysCommit,NeverCommit,AgeGate,ResidualGate \
      --out-dir "$base/baseline/$seq"
  done
  echo "$base"
}
run_crossval() {
  local base sensitivity
  if [[ -n "${R3_CROSSVAL_ROOT:-}" ]]; then
    base="$R3_CROSSVAL_ROOT"
  else
    local marker="$project_root/outputs/r_series/r3_gate_crossval/LATEST_BASELINE_PATH.txt"
    [[ -f "$marker" ]] || { echo "No baseline root found. Run 'baseline' or 'full' first." >&2; return 2; }
    base="$(cat "$marker")"
  fi
  sensitivity="$(latest_sensitivity_root)"
  [[ -f "$sensitivity/residual_threshold_sweep_summary.csv" ]] || {
    echo "Missing residual threshold sweep summary under $sensitivity" >&2; return 2;
  }
  [[ -d "$base/baseline" ]] || { echo "Missing baseline directory: $base/baseline" >&2; return 2; }
  echo "Using sensitivity root: $sensitivity"
  echo "Using baseline root: $base/baseline"
  "$svr_python" "$script_dir/r3_cross_sequence_validation.py" \
    --sensitivity-root "$sensitivity" --baseline-root "$base/baseline" \
    --out-dir "$base/cross_sequence_validation"
}
case "$stage" in
  preflight)
    run_tests
    "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" --preflight-only
    ;;
  smoke)
    run_tests
    "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" \
      --data-root /root/autodl-tmp/datasets/rgbd_bonn_crowd2 --sequence crowd2 \
      --frames 32 --refresh-frames 5,14,23
    ;;
  sensitivity)
    run_tests
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    base="$project_root/outputs/r_series/r3_residual_sensitivity/$stamp"
    mkdir -p "$base"
    for sequence in crowd2 balloon2 crowd3; do
      for threshold in 0.03 0.04 0.05 0.06 0.08 0.10; do
        slug="${threshold/./p}"
        echo "=== ResidualGate sensitivity: sequence=$sequence threshold=$threshold ==="
        "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" \
          --data-root "/root/autodl-tmp/datasets/rgbd_bonn_${sequence}" --sequence "$sequence" \
          --frames 86 --refresh-frames 9,37,65 --policies ResidualGate \
          --residual-threshold "$threshold" --out-dir "$base/$sequence/thr_$slug"
      done
    done
    "$svr_python" "$script_dir/summarize_residual_sweep.py" "$base"
    echo "Residual sensitivity outputs: $base"
    ;;
  baseline)
    run_tests
    run_baseline
    ;;
  crossval)
    run_tests
    run_crossval
    ;;
  full)
    run_tests
    sensitivity="$(latest_sensitivity_root)"
    [[ -f "$sensitivity/residual_threshold_sweep_summary.csv" ]] || {
      echo "Sensitivity results not found. Run 'sensitivity' first or set R3_SENSITIVITY_ROOT." >&2; exit 2;
    }
    run_baseline
    run_crossval
    ;;
  *)
    echo "Usage: bash experiments/r_series/r3_simple_commit_gate/run_r3.sh {preflight|smoke|sensitivity|baseline|crossval|full}" >&2
    echo "Environment: R3_SENSITIVITY_ROOT, R3_CROSSVAL_ROOT, R3_RESIDUAL_THRESHOLD, R3_AGE_THRESHOLD" >&2
    exit 2
    ;;
esac
