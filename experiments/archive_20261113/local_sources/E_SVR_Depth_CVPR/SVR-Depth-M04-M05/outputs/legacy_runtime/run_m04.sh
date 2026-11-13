#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
: "${VDEEP_SNAPSHOT:?Set VDEEP_SNAPSHOT to the locked VeloDepth snapshot directory}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON="${PYTHON:-python}"
MANIFEST="${M04_CHECKPOINT_MANIFEST:-experiments/p1/m04_checkpoint_manifest.csv}"
SCRIPT="experiments/p1/cvpr_m04_closed_loop.py"
mode="${1:-preflight}"
case "$mode" in
  preflight) "$PYTHON" -u "$SCRIPT" preflight --snapshot "$VDEEP_SNAPSHOT" --checkpoint-manifest "$MANIFEST" ;;
  canary) "$PYTHON" -u "$SCRIPT" canary --snapshot "$VDEEP_SNAPSHOT" --checkpoint-manifest "$MANIFEST" ;;
  run)
    : "${M04_CANARY_REPORT:?Set M04_CANARY_REPORT to a passed canary report}"
    "$PYTHON" -u "$SCRIPT" run --snapshot "$VDEEP_SNAPSHOT" --checkpoint-manifest "$MANIFEST" --canary-report "$M04_CANARY_REPORT" --timing-repeats "${M04_TIMING_REPEATS:-3}" ;;
  *) echo 'Usage: bash run_m04.sh [preflight|canary|run]' >&2; exit 2 ;;
esac
