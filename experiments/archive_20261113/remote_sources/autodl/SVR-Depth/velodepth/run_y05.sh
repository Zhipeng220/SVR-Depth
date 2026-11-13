#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -d /root/autodl-tmp/hf_cache ]]; then
    DEFAULT_CACHE=/root/autodl-tmp/hf_cache
else
    DEFAULT_CACHE="$HOME/hf_cache"
fi
export HF_HOME="${HF_HOME:-$DEFAULT_CACHE}"
export VDEEP_SNAPSHOT="${VDEEP_SNAPSHOT:-$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c}"
export Y04_DIR="${Y04_DIR:-outputs/gt4a2_y04_validation_fix3_20261025_225902}"
export Y09_DIR="${Y09_DIR:-auto}"
export Y03_DIR="${Y03_DIR:-auto}"
export Y10_DIR="${Y10_DIR:-auto}"
SCRIPT=experiments/p1/gt4a2_y05_future_utility.py
if (( $# == 0 )); then
    RUN_ROOT="outputs/gt4a2_y05_$(date +%Y%m%d_%H%M%S)_$$"
    FORMAL="$RUN_ROOT/validation"
    extra=()
elif (( $# >= 2 )) && [[ "$1" == --resume ]]; then
    FORMAL="$2";RUN_ROOT="$(dirname -- "$FORMAL")";shift 2
    extra=(--resume "$@")
elif (( $# >= 1 )) && [[ "$1" == --resume-latest ]]; then
    FORMAL="$(python - <<'PY'
from pathlib import Path
files=list(Path('outputs').glob('gt4a2_y05_*/validation/protocol.json'))
if not files:raise SystemExit('未找到正式 Y05 断点；若停在预检查或 smoke，请执行 bash run_y05.sh。')
print(max(files,key=lambda p:p.stat().st_mtime_ns).parent)
PY
)"
    RUN_ROOT="$(dirname -- "$FORMAL")";shift
    extra=(--resume "$@")
else
    printf 'Usage: bash run_y05.sh [--resume DIR | --resume-latest] [--stage train|evaluate|timing|report]\n' >&2
    exit 2
fi
phase=self-test
on_exit() {
    local rc=$?
    if (( rc != 0 )); then
        printf '\n[STOPPED] Y05 %s failed. Logs: %s\n' "$phase" "$RUN_ROOT" >&2
        if [[ -f "$FORMAL/protocol.json" ]]; then
            printf '续跑命令：bash run_y05.sh --resume %q\n' "$FORMAL" >&2
        else
            printf '尚未进入正式训练。修复日志错误后重新执行 bash run_y05.sh。\n' >&2
        fi
    fi
}
trap on_exit EXIT
sha256sum --check experiments/p1/y05_SHA256SUMS
python -m py_compile experiments/p1/y05_common.py experiments/p1/y05_engine.py experiments/p1/y05_train.py experiments/p1/y05_eval.py "$SCRIPT"
python experiments/p1/test_y05.py
mkdir -p "$RUN_ROOT"
verify() {
    python - "$1" "$2" <<'PY'
import json,sys,zipfile
from pathlib import Path
p=Path(sys.argv[1]);smoke=sys.argv[2]=='smoke';s=json.loads((p/'summary.json').read_text())
assert s['experiment']=='Y05' and s['partial_smoke']==smoke
assert s['status']==('SMOKE_PASS' if smoke else 'Y05_DEVELOPMENT_SUITE_COMPLETE')
assert s['training_performed'] and s['gradient_audit_passed'] and s['no_automatic_method_pass']
assert s['Y11_pass'] is False and s['numerical_failure_branches']==0
with zipfile.ZipFile(p/'y05_review.zip') as z:
    assert z.testzip() is None and json.loads(z.read('summary.json'))==s
print('[VERIFIED] Y05',sys.argv[2],flush=True)
PY
}
if (( ${#extra[@]} == 0 )); then
    phase=preflight
    python -u "$SCRIPT" --preflight-only --out-dir "$RUN_ROOT/preflight" 2>&1 | tee "$RUN_ROOT/preflight.log"
    phase=CUDA-gradient-smoke
    python -u "$SCRIPT" --smoke-only --out-dir "$RUN_ROOT/smoke" 2>&1 | tee "$RUN_ROOT/smoke.log"
    verify "$RUN_ROOT/smoke" smoke
fi
phase=formal-Y05
python -u "$SCRIPT" --out-dir "$FORMAL" "${extra[@]}" 2>&1 | tee -a "$RUN_ROOT/validation.log"
if [[ ! -f "$FORMAL/summary.json" ]]; then
    printf '\n阶段完成。继续：bash run_y05.sh --resume %q\n' "$FORMAL"
    exit 0
fi
verify "$FORMAL" formal
printf '\n回传结果包：%s/y05_review.zip\n' "$FORMAL"
