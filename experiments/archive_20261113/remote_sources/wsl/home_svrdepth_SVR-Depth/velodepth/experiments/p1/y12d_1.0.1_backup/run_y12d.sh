#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export Y05R_DIR="${Y05R_DIR:-auto}"
if [[ -d /root/autodl-tmp/hf_cache ]]; then
    DEFAULT_CACHE=/root/autodl-tmp/hf_cache
else
    DEFAULT_CACHE="$HOME/hf_cache"
fi
export HF_HOME="${HF_HOME:-$DEFAULT_CACHE}"
SCRIPT=experiments/p1/gt4a2_y12d.py

sha256sum --check experiments/p1/y12d_SHA256SUMS
python -m py_compile experiments/p1/y12d_common.py experiments/p1/y12d_eval.py "$SCRIPT" experiments/p1/test_y12d.py
python experiments/p1/test_y12d.py

if (( $# == 1 )) && [[ "$1" == "--resume-latest" ]]; then
    FORMAL="$(python - <<'PY'
from pathlib import Path
import json
files=[]
for p in Path('outputs').glob('gt4a2_y12d_*/validation/protocol.json'):
    try:
        obj=json.loads(p.read_text())
        if obj.get('experiment')=='Y12-D' and obj.get('version')=='Y12-D-1.0.1' and not obj.get('partial_smoke'):
            files.append(p)
    except Exception:
        pass
if not files:
    raise SystemExit('未找到正式 Y12-D protocol；若此前停在 preflight/smoke，请先修复原因并重新执行 bash run_y12d.sh。')
print(max(files,key=lambda p:p.stat().st_mtime_ns).parent.resolve())
PY
)"
    LOGICAL_OUT="$FORMAL"
    python -u "$SCRIPT" --resume --out-dir "$LOGICAL_OUT"
    exit 0
fi
if (( $# != 0 )); then
    echo 'Usage: bash run_y12d.sh [--resume-latest]' >&2
    exit 2
fi

RUN_ROOT="outputs/gt4a2_y12d_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_ROOT"
phase=preflight
on_exit() {
    rc=$?
    if (( rc != 0 )); then
        echo "[STOPPED] Y12-D phase=$phase; inspect the log and outputs under $RUN_ROOT" >&2
        if [[ -f "$RUN_ROOT/validation/protocol.json" ]]; then
            echo 'Formal protocol exists. Resume completed anchor cases with: bash run_y12d.sh --resume-latest' >&2
        else
            echo 'No formal recovery protocol yet. Fix the preflight/smoke error, then start a fresh run.' >&2
        fi
    fi
}
trap on_exit EXIT

phase=preflight
python -u "$SCRIPT" --preflight-only --out-dir "$RUN_ROOT/preflight"
phase=real-CUDA-smoke
python -u "$SCRIPT" --smoke-only --out-dir "$RUN_ROOT/smoke"
phase=formal-27-anchor-diagnostic
python -u "$SCRIPT" --out-dir "$RUN_ROOT/validation"
echo "Y12-D 完成。正式结果目录：$PWD/$RUN_ROOT/validation"
echo "结果包：$PWD/$RUN_ROOT/validation/y12d_review.zip"
echo "判定：$PWD/$RUN_ROOT/validation/y12d_gate.json"
