#!/usr/bin/env bash
set -uo pipefail

cd /home/svrdepth/SVR-Depth/velodepth || {
  echo "[FAIL] 工程目录不存在"
  exit 1
}
if [[ "${CONDA_DEFAULT_ENV:-}" == "svrdepth" ]]; then
  echo "[PASS] 已继承 svrdepth 环境，跳过重复激活"
else
  CONDA_BASE=""

  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    CONDA_BASE="$("${CONDA_EXE}" info --base 2>/dev/null || true)"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_CMD="$(command -v conda)"
    CONDA_BASE="$("$CONDA_CMD" info --base 2>/dev/null || true)"
  fi

  if [[ -n "$CONDA_BASE" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
  else
    for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "/opt/conda"; do
      if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
        source "$candidate/etc/profile.d/conda.sh"
        break
      fi
    done
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "[FAIL] 找不到 Conda 初始化脚本"
    exit 2
  fi

  conda activate svrdepth || {
    echo "[FAIL] 无法激活 svrdepth 环境"
    exit 2
  }
fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/svrdepth/hf_cache}"
export VDEEP_SNAPSHOT="${VDEEP_SNAPSHOT:-$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c}"

mkdir -p outputs/gt4a1_diagnostics
LOG="outputs/gt4a1_diagnostics/gt4a1_run.log"

echo "========== GT4-A1 syntax check =========="
python -m py_compile experiments/p1/gt4a1_failure_diagnosis.py || {
  echo "[FAIL] 语法检查失败"
  exit 1
}

echo "========== GT4-A1 frozen inference and diagnostics =========="
set +e
python -u experiments/p1/gt4a1_failure_diagnosis.py \
  --sequences crowd3,balloon2 \
  --gt4-dir outputs/gt4_validation \
  --gt2-dir outputs/gt2_validation \
  --snapshot "$VDEEP_SNAPSHOT" \
  --out-dir outputs/gt4a1_diagnostics \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo
echo "GT4-A1 exit code: $rc"
echo "Log: $LOG"
if [ "$rc" -ne 0 ]; then
  echo "[FAIL] 诊断中止；请保留日志并打包现有输出。"
  exit "$rc"
fi

echo
echo "========== GT4-A1 package review =========="
python -u experiments/p1/gt4a1_package_review.py
