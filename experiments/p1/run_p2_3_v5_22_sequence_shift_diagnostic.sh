#!/usr/bin/env bash
set -euo pipefail

REPO="${SVRDEPTH_ROOT:-/root/autodl-tmp/SVR-Depth/velodepth}"
SCRIPT="$REPO/experiments/p1/p2_3_v5_22_sequence_shift_diagnostic.py"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x /root/miniconda3/envs/svrdepth/bin/python ]]; then
  PY="/root/miniconda3/envs/svrdepth/bin/python"
else
  PY="python"
fi

if [[ ! -d "$REPO/experiments/p1" ]]; then
  echo "ERROR: VeloDepth project root not found: $REPO" >&2
  echo "Set SVRDEPTH_ROOT or edit REPO in this script." >&2
  exit 2
fi
if [[ ! -s "$SCRIPT" ]]; then
  echo "ERROR: missing Python script: $SCRIPT" >&2
  echo "Copy p2_3_v5_22_sequence_shift_diagnostic.py into experiments/p1/ first." >&2
  exit 2
fi

cd "$REPO"
echo "===== P2.3 v5.22: sequence-shift and decision-gain diagnostic ====="
echo "Python: $PY"
echo "Mode: CPU-only offline analysis"
echo "VeloDepth weights: frozen"
echo "GPU inference: not used"

"$PY" -m py_compile "$SCRIPT"
"$PY" "$SCRIPT" --self-test-only
"$PY" "$SCRIPT"

echo
echo "===== v5.22 completed ====="
echo "Review ZIP: $REPO/outputs/p2_3_v5_22_sequence_shift_diagnostic_review.zip"
echo "Report: $REPO/outputs/p2_3_v5_22_sequence_shift_diagnostic/P2_3_v5_22_实验报告.md"
