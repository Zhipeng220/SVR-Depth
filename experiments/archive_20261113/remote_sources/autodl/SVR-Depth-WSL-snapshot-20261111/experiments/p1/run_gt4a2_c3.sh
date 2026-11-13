#!/usr/bin/env bash
# Caller must already be inside conda env svrdepth.
# This script never exits the user's parent shell; it runs smoke then formal C3.
set -u
set -o pipefail
ROOT="/home/svrdepth/SVR-Depth/velodepth"
cd "$ROOT" || { echo "[FAIL] Project root missing: $ROOT"; exit 2; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/svrdepth/hf_cache}"
export VDEEP_SNAPSHOT="${VDEEP_SNAPSHOT:-$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c}"

if [ "${CONDA_DEFAULT_ENV:-}" != "svrdepth" ]; then
    echo "[FAIL] Activate svrdepth in the current WSL terminal before running."
    exit 2
fi
if [ ! -f experiments/p1/gt4a2_c3_closed_loop.py ]; then
    echo "[FAIL] GT4-A2-C3 source not extracted into experiments/p1"
    exit 2
fi
mkdir -p outputs/gt4a2_c3_smoke outputs/gt4a2_c3
python -m py_compile experiments/p1/gt4a2_c3_closed_loop.py \
   experiments/p1/gt4a2_c3_package_review.py \
   experiments/p1/gt4a2_exp1_train_eval.py \
   experiments/p1/gt2_fixed_clip_repair_train.py || exit 2

PAIR_C2="outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv"
PAIR_C3="outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv"
PAIR_B2="outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv"
for p in "$PAIR_C2" "$PAIR_C3" "$PAIR_B2"; do
  if [ ! -f "$p" ]; then echo "[FAIL] Pair CSV missing: $p"; exit 2; fi
done
COMMON=(--snapshot "$VDEEP_SNAPSHOT" \
        --crowd2-pair-csv "$PAIR_C2" \
        --crowd3-pair-csv "$PAIR_C3" \
        --balloon2-pair-csv "$PAIR_B2" \
        --previous-config outputs/gt4a2_exp1/gt4a2_exp1_config.json)

run_logged() {
  local log="$1"
  shift
  "$@" 2>&1 | tee "$log"
  return "${PIPESTATUS[0]}"
}

echo "========== C3: preflight only =========="
run_logged outputs/gt4a2_c3/c3_preflight.log \
  python -u experiments/p1/gt4a2_c3_closed_loop.py \
     "${COMMON[@]}" --out-dir outputs/gt4a2_c3 --preflight-only
rc=$?
if [ "$rc" -ne 0 ]; then
   echo "[FAIL] Preflight failed ($rc); no training started."
   python -u experiments/p1/gt4a2_c3_package_review.py
   exit "$rc"
fi

if [ "${C3_RESUME:-0}" != "1" ] && \
   [ ! -f outputs/gt4a2_c3_smoke/gt4a2_c3_smoke_summary.json ]; then
    echo "========== C3: one complete cycle smoke test for C and D =========="
    run_logged outputs/gt4a2_c3_smoke/c3_smoke.log \
      python -u experiments/p1/gt4a2_c3_closed_loop.py \
      "${COMMON[@]}" --out-dir outputs/gt4a2_c3_smoke --smoke-only --steps 1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] Full-cycle smoke failed ($rc); formal 1000-step training NOT started."
        python -u experiments/p1/gt4a2_c3_package_review.py
        exit "$rc"
    fi
fi

echo "========== C3: formal C/D 1000 updates each =========="
EXTRA=()
if [ "${C3_RESUME:-0}" = "1" ]; then EXTRA+=(--resume); fi
run_logged outputs/gt4a2_c3/c3_train_eval.log \
    python -u experiments/p1/gt4a2_c3_closed_loop.py \
    "${COMMON[@]}" --out-dir outputs/gt4a2_c3 --steps 1000 "${EXTRA[@]}"
rc=$?

echo "========== C3: compact result archive =========="
python -u experiments/p1/gt4a2_c3_package_review.py
pkg_rc=$?
if [ "$rc" -ne 0 ] || [ "$pkg_rc" -ne 0 ]; then
    echo "[FAIL] C3 incomplete; upload outputs/GT4_A2_C3_review.zip for diagnosis."
    echo "Tip: if recovery checkpoints exist, run: C3_RESUME=1 bash experiments/p1/run_gt4a2_c3.sh"
    exit 1
fi
echo "[PASS] C3 completed; the research gate is recorded separately in gt4a2_c3_decision.json"
echo "Report: outputs/gt4a2_c3/GT4_A2_C3_report.md"
echo "Archive: outputs/GT4_A2_C3_review.zip"
