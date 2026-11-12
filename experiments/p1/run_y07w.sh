#!/usr/bin/env bash
# Place this launcher in the VeloDepth repository root, then run with bash.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

SCRIPT="experiments/p1/gt4a2_y07w_same_head_no_writeback.py"
SCRIPT_SHA="e3ef2dae3bc387ddb9f9e11828d266bb14aaab9be1b17b9390666a8ac38ad358"
export HF_HOME="${HF_HOME:-/home/svrdepth/hf_cache}"
export VDEEP_SNAPSHOT="${VDEEP_SNAPSHOT:-$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c}"
Y04_DIR="${Y04_DIR:-outputs/gt4a2_y04_validation_fix3_20261025_225902}"
Y04A_DIR="${Y04A_DIR:-outputs/gt4a2_y04a_validation_20261026_200844}"
RUN_ROOT="outputs/gt4a2_y07w_$(date +%Y%m%d_%H%M%S)_$$"
phase="installation check"
on_exit() {
    local rc=$?
    if (( rc != 0 )); then
        printf '\n[STOPPED] %s failed (exit=%s). No completed formal result is being reported.\nLogs: %s\n' \
            "$phase" "$rc" "$RUN_ROOT" >&2
    fi
}
trap on_exit EXIT
printf '%s  %s\n' "$SCRIPT_SHA" "$SCRIPT" | sha256sum --check -
python -m py_compile "$SCRIPT"
python "$SCRIPT" --self-test
test -d "$VDEEP_SNAPSHOT"
test -d "$Y04_DIR"
test -f "$Y04A_DIR/y04a_summary.json"
mkdir -p "$RUN_ROOT"

verify_result() {
    python - "$1" "$2" "$SCRIPT_SHA" <<'PY'
import json
from pathlib import Path
import sys
import zipfile
out, mode, script_sha = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
s = json.loads((out / 'y07w_summary.json').read_text())
assert s['version'] == 'Y07-W-1.0' and s['script_sha256'] == script_sha, 'Unexpected script identity'
assert s['invariants_passed'] and not s['training_performed'] and not s['timing_claim'], s
assert s['new_inference_arms'] == ['Y07W_P_multi_current_output_no_writeback'], s
assert s['K'] == 32 and s['gain'] == 1.0, s
assert s['native_A_matched_frame_checks'] > 0 and s['P_N_age0_or_1_metric_checks'] > 0, s
if mode == 'smoke':
    assert s['status'] == 'SMOKE_PASS' and s['partial_smoke'] and s['new_rgb_records'] == 64, s
else:
    assert s['status'] == 'MECHANICS_PASS' and s['full_three_sequence_evaluation'] and not s['partial_smoke'], s
    assert s['new_rgb_records'] == 2218 and s['tail_comparison_records'] == 3752 and s['scored_feedback_pairs'] == 936, s
archive = out / 'y07w_review.zip'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None, 'Archive integrity failed'
    assert json.loads(z.read('y07w_summary.json')) == s, 'Archived summary differs'
print('[VERIFIED]', s['status'], archive.resolve(), flush=True)
PY
}

phase="preflight"
python -u "$SCRIPT" \
    --snapshot "$VDEEP_SNAPSHOT" --y04-dir "$Y04_DIR" --y04a-dir "$Y04A_DIR" \
    --preflight-only --out-dir "$RUN_ROOT/preflight" \
    2>&1 | tee "$RUN_ROOT/preflight.log"

phase="smoke"
python -u "$SCRIPT" \
    --snapshot "$VDEEP_SNAPSHOT" --y04-dir "$Y04_DIR" --y04a-dir "$Y04A_DIR" \
    --sequences crowd2 --smoke-only --out-dir "$RUN_ROOT/smoke" \
    2>&1 | tee "$RUN_ROOT/smoke.log"
verify_result "$RUN_ROOT/smoke" smoke

phase="formal evaluation"
python -u "$SCRIPT" \
    --snapshot "$VDEEP_SNAPSHOT" --y04-dir "$Y04_DIR" --y04a-dir "$Y04A_DIR" \
    --sequences crowd2,crowd3,balloon2 --out-dir "$RUN_ROOT/validation" \
    2>&1 | tee "$RUN_ROOT/validation.log"
verify_result "$RUN_ROOT/validation" formal
printf '\n回传正式结果包：%s/%s/validation/y07w_review.zip\n' "$PWD" "$RUN_ROOT"
