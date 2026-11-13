#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m py_compile \
  "$HERE/p2_3_v5_20_external_sequence_collector.py" \
  "$HERE/analyze_p2_3_v5_20_external_validation.py"
bash -n "$HERE/collect_p2_3_v5_20_new_sequence.sh"
bash -n "$HERE/run_p2_3_v5_20_external_validation.sh"
python "$HERE/p2_3_v5_20_external_sequence_collector.py" --self-test
python "$HERE/analyze_p2_3_v5_20_external_validation.py" --self-test
printf 'PASS: v5.20 package smoke tests. No external sequence result was generated.\n'
