#!/usr/bin/env bash
# GT4-A2-C2: no retraining; replay A/C/D continuously on Crowd2 RGB 0..767.
# Caller must activate the existing `svrdepth` environment first.
set +e
ROOT="/home/svrdepth/SVR-Depth/velodepth"
if [ ! -d "$ROOT" ]; then
  echo "[FAIL] 项目根目录不存在：$ROOT"
  echo "请修改本脚本中的 ROOT 为你的实际路径。"
  exit 2
fi
cd "$ROOT" || exit 2
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/svrdepth/hf_cache}"
export VDEEP_SNAPSHOT="${VDEEP_SNAPSHOT:-$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c}"
if [ -f outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv ]; then
  export GT4A2_CROWD2_PAIR_CSV="${GT4A2_CROWD2_PAIR_CSV:-outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv}"
else
  export GT4A2_CROWD2_PAIR_CSV="${GT4A2_CROWD2_PAIR_CSV:-outputs/gt1a1_validation/gt1a1_pairs.csv}"
fi
export GT4A2_C2_OUT="${GT4A2_C2_OUT:-outputs/gt4a2_crowd2_continuity}"
OUT="$GT4A2_C2_OUT"
mkdir -p "$OUT"
FAIL=0

echo "========== GT4-A2-C2 preflight (no training) =========="
echo "Project: $PWD"
echo "Environment: ${CONDA_DEFAULT_ENV:-unknown}"
echo "Snapshot: $VDEEP_SNAPSHOT"
echo "Crowd2 pairs: $GT4A2_CROWD2_PAIR_CSV"
echo "Checkpoint dir: outputs/gt4a2_exp1"
echo "Output dir: $OUT"

if [ "${CONDA_DEFAULT_ENV:-}" != "svrdepth" ]; then
  echo "[FAIL] 当前不是 svrdepth 环境。请先执行 conda activate svrdepth。"
  FAIL=1
fi

for f in \
  experiments/p1/gt4a2_c2_continuity_retest.py \
  experiments/p1/gt4a2_exp1_train_eval.py \
  experiments/p1/gt2_fixed_clip_repair_train.py \
  experiments/p1/gt4a2_continuity_package_review.py \
  velodepth/models/decoder_velo.py \
  velodepth/models/persistent_feature_repair.py \
  "$GT4A2_CROWD2_PAIR_CSV" \
  "$VDEEP_SNAPSHOT/config.json" "$VDEEP_SNAPSHOT/model.safetensors" \
  outputs/gt4a2_exp1/gt4a2_exp1_anchor_manifest.csv \
  outputs/gt4a2_exp1/gt4a2_exp1_config.json \
  outputs/gt4a2_exp1/gt4a2_exp1_C_final_step1000.pt \
  outputs/gt4a2_exp1/gt4a2_exp1_D_final_step1000.pt; do
  if [ -f "$f" ]; then echo "[PASS] $f"; else echo "[FAIL] 缺失：$f"; FAIL=1; fi
done

if [ "$FAIL" -eq 0 ]; then
  python -m py_compile \
    experiments/p1/gt4a2_c2_continuity_retest.py \
    experiments/p1/gt4a2_exp1_train_eval.py \
    experiments/p1/gt2_fixed_clip_repair_train.py \
    experiments/p1/gt4a2_continuity_package_review.py
  RC=$?
  if [ "$RC" -ne 0 ]; then echo "[FAIL] Python 语法检查"; FAIL=1; fi
  bash -n experiments/p1/run_gt4a2_c2_continuity_retest.sh
  if [ "$?" -ne 0 ]; then echo "[FAIL] Shell 语法检查"; FAIL=1; fi
fi

if [ "$FAIL" -eq 0 ]; then
  python -u experiments/p1/gt4a2_c2_continuity_retest.py \
    --snapshot "$VDEEP_SNAPSHOT" \
    --crowd2-pair-csv "$GT4A2_CROWD2_PAIR_CSV" \
    --checkpoint-dir outputs/gt4a2_exp1 \
    --out-dir "$OUT" --preflight-only \
    2>&1 | tee "$OUT/gt4a2_c2_preflight.log"
  RC=${PIPESTATUS[0]}
  echo "Preflight exit code: $RC"
  if [ "$RC" -ne 0 ]; then FAIL=1; fi
fi

if [ "$FAIL" -eq 0 ]; then
  echo
  echo "========== Start no-retraining continuous replay =========="
  python -u experiments/p1/gt4a2_c2_continuity_retest.py \
    --snapshot "$VDEEP_SNAPSHOT" \
    --crowd2-pair-csv "$GT4A2_CROWD2_PAIR_CSV" \
    --checkpoint-dir outputs/gt4a2_exp1 \
    --out-dir "$OUT" \
    2>&1 | tee "$OUT/gt4a2_c2_run.log"
  RC=${PIPESTATUS[0]}
  echo "Continuous replay exit code: $RC"
  if [ "$RC" -ne 0 ]; then FAIL=1; fi
fi

echo
echo "========== Package only required files =========="
python -u experiments/p1/gt4a2_continuity_package_review.py --out-dir "$OUT"
PKG_RC=$?
if [ "$PKG_RC" -ne 0 ]; then FAIL=1; fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo "[PASS] GT4-A2-C2 replay complete."
  echo "Report: $OUT/GT4_A2_Crowd2_连续状态复测报告.md"
else
  echo "[FAIL] Replay did not complete cleanly. Keep logs and upload outputs/GT4_A2_Crowd2_连续状态复测_review.zip."
fi
echo "Review ZIP: $PWD/outputs/GT4_A2_Crowd2_连续状态复测_review.zip"
echo "本脚本不训练、不修改模型权重；终端保持打开。"
