#!/usr/bin/env bash
# Run only GT4-A2 Experiment 1. Assumes the caller already activated conda env `svrdepth`.
# Does not call conda activate and does not close the caller's terminal.
set +e
ROOT="/home/svrdepth/SVR-Depth/velodepth"
if [ ! -d "$ROOT" ]; then
  echo "[FAIL] project root not found: $ROOT"
else
  cd "$ROOT" || true
  export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
  export HF_HOME="${HF_HOME:-/home/svrdepth/hf_cache}"
  export VDEEP_SNAPSHOT="${VDEEP_SNAPSHOT:-$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c}"
  export GT4A2_CROWD2_PAIR_CSV="${GT4A2_CROWD2_PAIR_CSV:-outputs/gt1a1_validation/gt1a1_pairs.csv}"
  export GT4A2_CROWD3_PAIR_CSV="${GT4A2_CROWD3_PAIR_CSV:-outputs/gt4_validation/crowd3/gt4_pairs.csv}"
  export GT4A2_BALLOON2_PAIR_CSV="${GT4A2_BALLOON2_PAIR_CSV:-outputs/gt4_validation/balloon2/gt4_pairs.csv}"
  OUT="outputs/gt4a2_exp1"
  mkdir -p "$OUT"
  FAIL=0

  echo "========== GT4-A2 Experiment 1: preflight =========="
  echo "Project: $PWD"
  echo "Environment: ${CONDA_DEFAULT_ENV:-unknown}"
  echo "Snapshot: $VDEEP_SNAPSHOT"
  echo "Crowd2 pairs: $GT4A2_CROWD2_PAIR_CSV"
  echo "Crowd3 pairs: $GT4A2_CROWD3_PAIR_CSV"
  echo "Balloon2 pairs: $GT4A2_BALLOON2_PAIR_CSV"

  if [ "${CONDA_DEFAULT_ENV:-}" != "svrdepth" ]; then
    echo "[FAIL] 当前不是 svrdepth conda 环境。请先在当前终端执行 conda activate svrdepth，再运行本脚本。"
    FAIL=1
  fi
  for f in \
    experiments/p1/gt4a2_exp1_train_eval.py \
    experiments/p1/gt4a2_exp1_package_review.py \
    experiments/p1/gt2_fixed_clip_repair_train.py \
    velodepth/models/decoder_velo.py \
    velodepth/models/persistent_feature_repair.py \
    "$GT4A2_CROWD2_PAIR_CSV" "$GT4A2_CROWD3_PAIR_CSV" "$GT4A2_BALLOON2_PAIR_CSV" \
    "$VDEEP_SNAPSHOT/config.json" "$VDEEP_SNAPSHOT/model.safetensors"; do
    if [ -f "$f" ]; then echo "[PASS] $f"; else echo "[FAIL] missing $f"; FAIL=1; fi
  done

  if [ "$FAIL" -eq 0 ]; then
    python -m py_compile experiments/p1/gt4a2_exp1_train_eval.py experiments/p1/gt4a2_exp1_package_review.py
    RC=$?
    if [ "$RC" -ne 0 ]; then echo "[FAIL] Python syntax"; FAIL=1; fi
  fi

  if [ "$FAIL" -eq 0 ]; then
    python -u experiments/p1/gt4a2_exp1_train_eval.py \
      --snapshot "$VDEEP_SNAPSHOT" \
      --crowd2-pair-csv "$GT4A2_CROWD2_PAIR_CSV" \
      --crowd3-pair-csv "$GT4A2_CROWD3_PAIR_CSV" \
      --balloon2-pair-csv "$GT4A2_BALLOON2_PAIR_CSV" \
      --out-dir "$OUT" --preflight-only \
      2>&1 | tee "$OUT/gt4a2_exp1_preflight.log"
    RC=${PIPESTATUS[0]}
    echo "Preflight exit code: $RC"
    if [ "$RC" -ne 0 ]; then FAIL=1; fi
  fi

  if [ "$FAIL" -eq 0 ]; then
    echo
    echo "========== Start fixed training and continuous evaluation =========="
    python -u experiments/p1/gt4a2_exp1_train_eval.py \
      --snapshot "$VDEEP_SNAPSHOT" \
      --crowd2-pair-csv "$GT4A2_CROWD2_PAIR_CSV" \
      --crowd3-pair-csv "$GT4A2_CROWD3_PAIR_CSV" \
      --balloon2-pair-csv "$GT4A2_BALLOON2_PAIR_CSV" \
      --out-dir "$OUT" \
      2>&1 | tee "$OUT/gt4a2_exp1_run.log"
    RC=${PIPESTATUS[0]}
    echo "Experiment 1 exit code: $RC"
    if [ "$RC" -ne 0 ]; then FAIL=1; fi
  fi

  echo
  echo "========== Package compact results, including failure logs if any =========="
  python -u experiments/p1/gt4a2_exp1_package_review.py --out-dir "$OUT"
  PKG_RC=$?
  echo "Package exit code: $PKG_RC"
  if [ "$PKG_RC" -ne 0 ]; then FAIL=1; fi

  echo
  if [ "$FAIL" -eq 0 ]; then
    echo "[PASS] Experiment 1 run completed. Review: $OUT/GT4_A2_Experiment1_report.md"
  else
    echo "[FAIL] Experiment 1 did not complete cleanly. Preserve the logs and upload outputs/GT4_A2_Experiment1_review.zip for diagnosis."
  fi
  echo "Review ZIP: $PWD/outputs/GT4_A2_Experiment1_review.zip"
  echo "Runner finished; this terminal should remain open."
fi
