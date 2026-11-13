#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename -- "$SCRIPT_DIR")" == d13_package ]]; then
    cd -- "$SCRIPT_DIR/.."
else
    cd -- "$SCRIPT_DIR"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODULE=experiments.p1.d13_suite
if [[ $# -gt 0 ]]; then
    python -u -m "$MODULE.runner" "$@"
    exit $?
fi
RUN_ROOT="outputs/gt4a2_d13_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_ROOT"
python -m "$MODULE.tests" 2>&1 | tee "$RUN_ROOT/selftest.log"
python -u -m "$MODULE.runner" --smoke-only --out-dir "$RUN_ROOT/smoke" 2>&1 | tee "$RUN_ROOT/smoke.log"
python -u -m "$MODULE.runner" --out-dir "$RUN_ROOT/validation" 2>&1 | tee "$RUN_ROOT/validation.log"
