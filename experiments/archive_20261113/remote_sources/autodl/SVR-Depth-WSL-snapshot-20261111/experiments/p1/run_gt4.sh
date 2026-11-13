#!/usr/bin/env bash
# GT4 runner. Run from an already activated `svrdepth` conda environment.
# No `exit` is used so failures are reported without closing the caller's terminal.
set +e
ROOT="/home/svrdepth/SVR-Depth/velodepth"
if [ ! -d "$ROOT" ]; then
  echo "[FAIL] project root not found: $ROOT"
  echo "GT4 runner finished; terminal remains open."
else
cd "$ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/home/svrdepth/hf_cache}"
export VDEEP_SNAPSHOT="${VDEEP_SNAPSHOT:-$HF_HOME/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c}"
CROWD3_ROOT="${GT4_CROWD3_ROOT:-/home/svrdepth/datasets/rgbd_bonn_crowd3}"
BALLOON2_ROOT="${GT4_BALLOON2_ROOT:-/home/svrdepth/datasets/rgbd_bonn_balloon2}"
OUT="outputs/gt4_validation"
mkdir -p "$OUT/crowd3" "$OUT/balloon2"
FAIL=0

echo "========== GT4 preflight =========="
echo "Project: $PWD"
echo "Snapshot: $VDEEP_SNAPSHOT"
echo "Crowd3: $CROWD3_ROOT"
echo "Balloon2: $BALLOON2_ROOT"
for f in \
  experiments/p1/gt4_build_sequence_pairs.py \
  experiments/p1/gt4_cross_sequence_eval.py \
  experiments/p1/gt3_heldout_eval.py \
  experiments/p1/gt2_fixed_clip_repair_train.py \
  velodepth/models/decoder_velo.py \
  velodepth/models/persistent_feature_repair.py \
  outputs/gt2_validation/gt2_summary.json \
  outputs/gt2_validation/gt2_C_repair_head.pt \
  outputs/gt2_validation/gt2_D_repair_head.pt \
  outputs/gt1a1_validation/gt1a1_pairs.csv \
  "$VDEEP_SNAPSHOT/config.json" \
  "$VDEEP_SNAPSHOT/model.safetensors"; do
  if [ -f "$f" ]; then echo "[PASS] $f"; else echo "[FAIL] missing $f"; FAIL=1; fi
done
for r in "$CROWD3_ROOT" "$BALLOON2_ROOT"; do
  if [ -f "$r/rgb.txt" ] && [ -f "$r/depth.txt" ]; then
    echo "[PASS] sequence indices: $r/rgb.txt and depth.txt"
  else
    echo "[FAIL] sequence root missing rgb.txt/depth.txt: $r"
    FAIL=1
  fi
done

if [ "$FAIL" -eq 0 ]; then
  echo
  echo "========== syntax check =========="
  python -m py_compile \
    experiments/p1/gt4_build_sequence_pairs.py \
    experiments/p1/gt4_cross_sequence_eval.py \
    experiments/p1/gt3_heldout_eval.py \
    experiments/p1/gt2_fixed_clip_repair_train.py
  RC=$?
  echo "syntax exit code: $RC"
  if [ "$RC" -ne 0 ]; then FAIL=1; fi
fi

if [ "$FAIL" -eq 0 ]; then
  echo
  echo "========== pair manifest: Crowd3 =========="
  python -u experiments/p1/gt4_build_sequence_pairs.py \
    --sequence-root "$CROWD3_ROOT" --sequence-name Bonn_Crowd3 \
    --out-dir "$OUT/crowd3" --max-time-diff-ms 20.0 \
    2>&1 | tee "$OUT/crowd3/gt4_pair_build.log"
  RC=${PIPESTATUS[0]}
  echo "Crowd3 pair-builder exit code: $RC"
  if [ "$RC" -ne 0 ]; then FAIL=1; fi

  echo
  echo "========== pair manifest: Balloon2 =========="
  python -u experiments/p1/gt4_build_sequence_pairs.py \
    --sequence-root "$BALLOON2_ROOT" --sequence-name Bonn_Balloon2 \
    --out-dir "$OUT/balloon2" --max-time-diff-ms 20.0 \
    2>&1 | tee "$OUT/balloon2/gt4_pair_build.log"
  RC=${PIPESTATUS[0]}
  echo "Balloon2 pair-builder exit code: $RC"
  if [ "$RC" -ne 0 ]; then FAIL=1; fi
fi

if [ "$FAIL" -eq 0 ]; then
  echo
  echo "========== frozen cross-sequence eval: Crowd3 =========="
  python -u experiments/p1/gt4_cross_sequence_eval.py \
    --sequence-name Bonn_Crowd3 \
    --pair-csv "$OUT/crowd3/gt4_pairs.csv" \
    --snapshot "$VDEEP_SNAPSHOT" \
    --gt2-dir outputs/gt2_validation \
    --out-dir "$OUT/crowd3" \
    --max-time-diff-ms 20.0 --min-clips 3 \
    2>&1 | tee "$OUT/crowd3/gt4_run.log"
  RC=${PIPESTATUS[0]}
  echo "Crowd3 evaluation exit code: $RC"
  if [ "$RC" -ne 0 ]; then FAIL=1; fi

  echo
  echo "========== frozen cross-sequence eval: Balloon2 =========="
  python -u experiments/p1/gt4_cross_sequence_eval.py \
    --sequence-name Bonn_Balloon2 \
    --pair-csv "$OUT/balloon2/gt4_pairs.csv" \
    --snapshot "$VDEEP_SNAPSHOT" \
    --gt2-dir outputs/gt2_validation \
    --out-dir "$OUT/balloon2" \
    --max-time-diff-ms 20.0 --min-clips 3 \
    2>&1 | tee "$OUT/balloon2/gt4_run.log"
  RC=${PIPESTATUS[0]}
  echo "Balloon2 evaluation exit code: $RC"
  if [ "$RC" -ne 0 ]; then FAIL=1; fi
fi

echo
echo "========== GT4 runner summary =========="
if [ "$FAIL" -eq 0 ]; then
  echo "[PASS] Both sequence runs completed. Next: python -u experiments/p1/gt4_package_review.py"
else
  echo "[FAIL] One or more checks/runs failed. No training was performed. Preserve logs and run the packager to share diagnostics."
  echo "If sequence paths are wrong, search: find /home/svrdepth/datasets /mnt/d/SVR-Depth -type f \( -name rgb.txt -o -name depth.txt \) -print 2>/dev/null | head -100"
fi
echo "GT4 runner finished; current terminal should remain open."

fi
