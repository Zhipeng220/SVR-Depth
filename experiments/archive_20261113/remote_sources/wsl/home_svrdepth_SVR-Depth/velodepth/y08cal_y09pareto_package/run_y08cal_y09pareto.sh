#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export HF_HOME="${HF_HOME:-/home/svrdepth/hf_cache}"
export VDEEP_SNAPSHOT="${VDEEP_SNAPSHOT:-$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c}"
Y04_DIR="${Y04_DIR:-outputs/gt4a2_y04_validation_fix3_20261025_225902}"
Y04A_DIR="${Y04A_DIR:-auto}"
SCRIPT="experiments/p1/gt4a2_y08cal_y09pareto.py"
if (( $# == 0 )); then
    RUN_ROOT="outputs/gt4a2_y08cal_y09pareto_$(date +%Y%m%d_%H%M%S)_$$"
    FORMAL="$RUN_ROOT/validation"
    resume=()
elif (( $# >= 2 )) && [[ "$1" == "--resume" ]]; then
    FORMAL="$2";RUN_ROOT="$(dirname -- "$FORMAL")";shift 2
    resume=(--resume "$@")
else
    printf 'Usage: bash run_y08cal_y09pareto.sh [--resume outputs/.../validation [--stage all|train|quality|timing|report]]\n' >&2
    exit 2
fi
phase="installation/self-test"
on_exit() {
    local rc=$?
    if (( rc != 0 )); then
        printf '\n[STOPPED] %s failed. No completed formal result is reported.\nLogs: %s\n' "$phase" "$RUN_ROOT" >&2
        if [[ -f "$FORMAL/protocol.json" ]]; then
            printf '续跑命令：bash run_y08cal_y09pareto.sh --resume %q\n' "$FORMAL" >&2
        else
            printf '正式任务尚未建立；修复当前错误后重新启动。\n' >&2
        fi
    fi
}
trap on_exit EXIT
sha256sum --check experiments/p1/y08y09_SHA256SUMS
python -m py_compile "$SCRIPT" experiments/p1/y08y09_common.py experiments/p1/y08y09_models.py
python experiments/p1/test_y08y09.py
mkdir -p "$RUN_ROOT"
verify_result() {
    python - "$1" "$2" <<'PY'
import json,sys,zipfile
from pathlib import Path
out,mode=Path(sys.argv[1]),sys.argv[2]
s=json.loads((out/'summary.json').read_text())
assert s['version']=='Y08CAL-Y09P-1.0' and s['auto_equivalence_passed']
assert s['all_timing_endpoints_verified'] and s['quality_and_timing_checked']
assert s['timing_trials']==s['expected_timing_trials']
if mode=='smoke':
    assert s['status']=='SMOKE_PASS' and s['partial_smoke']
    assert all(v['steps']==3 for v in s['newly_trained_heads'].values())
else:
    assert s['status']=='DEVELOPMENT_SUITE_COMPLETE' and not s['partial_smoke']
    assert s['raw_configurations']==36 and s['quality_variants']==72 and s['timing_trials']==648
    assert all(v['steps']==1000 and v['supervised_exposures']==30834 for v in s['newly_trained_heads'].values())
with zipfile.ZipFile(out/'y08cal_y09pareto_review.zip') as z:
    assert z.testzip() is None
    assert json.loads(z.read('summary.json'))==s
print('[VERIFIED]',mode,flush=True)
PY
}
if (( ${#resume[@]} == 0 )); then
    phase="preflight"
    python -u "$SCRIPT" --snapshot "$VDEEP_SNAPSHOT" --y04-dir "$Y04_DIR" --y04a-dir "$Y04A_DIR" \
        --preflight-only --out-dir "$RUN_ROOT/preflight" 2>&1 | tee "$RUN_ROOT/preflight.log"
    phase="smoke: real Auto audit, independent three-step training, small quality and timing"
    python -u "$SCRIPT" --snapshot "$VDEEP_SNAPSHOT" --y04-dir "$Y04_DIR" --y04a-dir "$Y04A_DIR" \
        --smoke-only --out-dir "$RUN_ROOT/smoke" 2>&1 | tee "$RUN_ROOT/smoke.log"
    verify_result "$RUN_ROOT/smoke" smoke
fi
phase="formal Y08-Cal and Y09-Pareto"
python -u "$SCRIPT" --snapshot "$VDEEP_SNAPSHOT" --y04-dir "$Y04_DIR" --y04a-dir "$Y04A_DIR" \
    --out-dir "$FORMAL" "${resume[@]}" 2>&1 | tee -a "$RUN_ROOT/validation.log"
# A deliberately requested intermediate stage is not reported as a finished suite.
if [[ " ${resume[*]} " == *" --stage train "* || " ${resume[*]} " == *" --stage quality "* || " ${resume[*]} " == *" --stage timing "* ]]; then
    if [[ ! -f "$FORMAL/summary.json" ]]; then
        printf '\n阶段完成；继续全流程：bash run_y08cal_y09pareto.sh --resume %q\n' "$FORMAL"
        exit 0
    fi
fi
verify_result "$FORMAL" formal
printf '\n回传正式结果包：%s/y08cal_y09pareto_review.zip\n' "$FORMAL"
