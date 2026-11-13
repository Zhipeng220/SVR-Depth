#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
MODE="${1:-preflight}"
shift || true

: "${VDEEP_SNAPSHOT:?Set VDEEP_SNAPSHOT to the pinned VeloDepth snapshot directory}"
E4C_DEFAULT="${PROJECT_ROOT}/outputs/crosssource_v2_E4_20261104_C_s20261103_1000step/best_checkpoint.pt"
E4C_CHECKPOINT="${M04_E4C_CHECKPOINT:-${E4C_DEFAULT}}"

COMMON_ARGS=(
  --snapshot "${VDEEP_SNAPSHOT}"
  --e4c-checkpoint "${E4C_CHECKPOINT}"
  --timing-repeats "${M04_TIMING_REPEATS:-3}"
)

case "${MODE}" in
  preflight|canary)
    exec "${PYTHON}" -u "${PROJECT_ROOT}/experiments/p1/cvpr_m04_v31.py" \
      "${MODE}" "${COMMON_ARGS[@]}" "$@"
    ;;
  run)
    : "${M04_CANARY_REPORT:?Set M04_CANARY_REPORT to a passed M04 V3.1 canary summary}"
    exec "${PYTHON}" -u "${PROJECT_ROOT}/experiments/p1/cvpr_m04_v31.py" \
      run "${COMMON_ARGS[@]}" --canary-report "${M04_CANARY_REPORT}" "$@"
    ;;
  *)
    echo "Usage: bash scripts/run_m04_v31.sh [preflight|canary|run] [extra args]" >&2
    exit 2
    ;;
esac
