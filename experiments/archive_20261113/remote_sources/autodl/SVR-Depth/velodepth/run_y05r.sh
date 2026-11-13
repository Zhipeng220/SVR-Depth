#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export Y05_DIR="${Y05_DIR:-auto}"
if [[ -d /root/autodl-tmp/hf_cache ]]; then
    DEFAULT_CACHE=/root/autodl-tmp/hf_cache
else
    DEFAULT_CACHE="$HOME/hf_cache"
fi
export HF_HOME="${HF_HOME:-$DEFAULT_CACHE}"
SCRIPT=experiments/p1/gt4a2_y05r.py

if (( $# == 1 )) && [[ "$1" == "--resume-latest" ]]; then
    FORMAL="$(python - <<'PY'
from pathlib import Path
import json
files=[]
for p in Path('outputs').glob('gt4a2_y05r_*/validation/protocol.json'):
    try:
        obj=json.loads(p.read_text())
        if obj.get('experiment')=='Y05-R' and not obj.get('partial_smoke'):
            files.append(p)
    except Exception:
        pass
if not files:
    raise SystemExit('未找到正式 Y05-R protocol；若此前停在 preflight/smoke，请修复原因后重新执行 bash run_y05r.sh。')
print(max(files,key=lambda p:p.stat().st_mtime_ns).parent.resolve())
PY
)"
    LOGICAL_OUT="$FORMAL"
    sha256sum --check experiments/p1/y05r_SHA256SUMS
    python -m py_compile experiments/p1/y05r_common.py experiments/p1/y05r_train.py experiments/p1/y05r_eval.py "$SCRIPT"
    python experiments/p1/test_y05r.py
    python -u "$SCRIPT" --resume --out-dir "$LOGICAL_OUT"
    exit 0
fi
if (( $# != 0 )); then
    echo 'Usage: bash run_y05r.sh [--resume-latest]' >&2
    exit 2
fi

sha256sum --check experiments/p1/y05r_SHA256SUMS
python -m py_compile experiments/p1/y05r_common.py experiments/p1/y05r_train.py experiments/p1/y05r_eval.py "$SCRIPT"
python experiments/p1/test_y05r.py
RUN_ROOT="outputs/gt4a2_y05r_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$RUN_ROOT"

phase=preflight
on_exit() {
    rc=$?
    if (( rc != 0 )); then
        echo "[STOPPED] Y05-R phase=$phase; inspect the log and outputs under $RUN_ROOT" >&2
        if [[ -f "$RUN_ROOT/validation/protocol.json" ]]; then
            echo 'Formal protocol exists. Resume with: bash run_y05r.sh --resume-latest' >&2
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
phase=formal-C-star-CF
python -u "$SCRIPT" --out-dir "$RUN_ROOT/validation"
echo "Y05-R 完成。正式结果目录：$PWD/$RUN_ROOT/validation"
echo "结果包：$PWD/$RUN_ROOT/validation/y05r_review.zip"
echo "最终检查点：$PWD/$RUN_ROOT/validation/training/C_star/final.pt"
echo "最终检查点：$PWD/$RUN_ROOT/validation/training/CF/final.pt"
