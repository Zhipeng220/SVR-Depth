#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"
python -m py_compile "$SCRIPT_DIR/analyze_p2_3_v5_20_external_validation.py"
python "$SCRIPT_DIR/analyze_p2_3_v5_20_external_validation.py" \
  --train-dataset "$SCRIPT_DIR/inputs/p2_3_v5_18_aligned_53_anchor_dataset.csv" \
  --out-dir "$SCRIPT_DIR/results" \
  "$@"
