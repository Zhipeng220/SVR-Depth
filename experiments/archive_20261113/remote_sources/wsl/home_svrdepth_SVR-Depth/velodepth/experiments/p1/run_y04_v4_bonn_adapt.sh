#!/usr/bin/env bash
# Y04-V4 Bonn13 in-domain training. No `set -euo pipefail`.
# Modes: gate | train | resume <run-directory>.
# Required: installed trainer + protocol + audit markdown in experiments/p1.

cd "$HOME/SVR-Depth/velodepth" || exit 1

# Avoid inheriting old prototype sitecustomize.py injected via PYTHONPATH.
if [ -n "${PYTHONPATH:-}" ]; then
  echo "[ENV] Clear inherited PYTHONPATH; determinism is embedded in V0.2."
  unset PYTHONPATH
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=20261025
export CUDA_VISIBLE_DEVICES=0

SCRIPT="$PWD/experiments/p1/gt4a2_y04_v4_bonn_train_v0_2.py"
PROTO="$PWD/experiments/p1/y04_v4_bonn_indomain_protocol_v1.json"
MANIFEST="$PWD/outputs/y04_v4_manifest_candidate_20261108_190940"
SNAPSHOT="$HOME/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c"
INIT="$PWD/outputs/gt4a2_c3/gt4a2_c3_initial_head.pt"
SEED=20261025
OUTROOT="$PWD/outputs"

for FILE in "$SCRIPT" "$PROTO" "$MANIFEST/manifest_identity.json" "$SNAPSHOT/model.safetensors" "$SNAPSHOT/config.json" "$INIT"; do
  if [ ! -f "$FILE" ]; then
    echo "[FAILED] Missing required file: $FILE"
    exit 1
  fi
done

python -m py_compile "$SCRIPT" || exit 1

run_logged() {
  local logfile="$1"; shift
  "$@" 2>&1 | tee "$logfile"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "[FAILED] $logfile (exit=$rc)"
    return "$rc"
  fi
  return 0
}

CURRENT_GATE="$OUTROOT/y04_v4_bonn13_indomain_gate_latest.json"

assert_gate() {
  if [ ! -f "$CURRENT_GATE" ]; then
    echo "[FAILED] Gate record missing; first run: bash $0 gate"
    return 1
  fi
  python - "$CURRENT_GATE" "$SCRIPT" "$PROTO" "$MANIFEST/manifest_identity.json" <<'PY'
import sys,json,hashlib
from pathlib import Path
state=json.loads(Path(sys.argv[1]).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for key,file in (("trainer_sha256",sys.argv[2]),("protocol_sha256",sys.argv[3]),("manifest_identity_sha256",sys.argv[4])):
    if sha(file)!=state.get(key):
        raise SystemExit(f"[FAILED] Gate stale: {key}; rerun gate")
if state.get("status")!="GATE_PASS": raise SystemExit("[FAILED] Gate not passed")
print("[GATE VERIFIED] Existing resume + 13-scene tests match code/protocol/manifest")
PY
}

mode="${1:-}"
if [ "$mode" = "gate" ]; then
  BASE="$OUTROOT/y04_v4_bonn13_indomain_gate_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$BASE" || exit 1
  echo "========== 0. 1000-STEP ADAPT PREFLIGHT (NO TRAINING) =========="
  run_logged "$BASE/adapt_preflight.log" python "$SCRIPT" \
    --mode adapt --preflight-only --profile bonn13 \
    --manifest-dir "$MANIFEST" --snapshot "$SNAPSHOT" --init-head "$INIT" \
    --protocol-file "$PROTO" --deterministic \
    --slots-per-seq 8 --steps 1000 --seed "$SEED" || exit 1

  COMMON=(
    --mode dev --profile bonn3 --manifest-dir "$MANIFEST"
    --snapshot "$SNAPSHOT" --init-head "$INIT"
    --protocol-file "$PROTO" --deterministic
    --slots-per-seq 1 --steps 6 --seed "$SEED"
    --cache-root "$BASE/resume_cache"
  )
  echo "========== 1. RESUMED 3/6 =========="
  run_logged "$BASE/a1.log" python "$SCRIPT" "${COMMON[@]}" \
    --run-dir "$BASE/resumed" --stop-after 3 || exit 1
  test -f "$BASE/resumed/recovery.pt" || { echo "[FAILED] Recovery missing"; exit 1; }

  echo "========== 2. RESUME TO 6/6 =========="
  run_logged "$BASE/a2.log" python "$SCRIPT" "${COMMON[@]}" \
    --run-dir "$BASE/resumed" --resume || exit 1
  test -f "$BASE/resumed/completion.json" || { echo "[FAILED] Resumed completion missing"; exit 1; }

  echo "========== 3. CONTINUOUS 6/6 =========="
  run_logged "$BASE/b.log" python "$SCRIPT" "${COMMON[@]}" \
    --run-dir "$BASE/continuous" || exit 1
  test -f "$BASE/continuous/completion.json" || { echo "[FAILED] Continuous completion missing"; exit 1; }

  python - "$BASE" <<'PY' || exit 1
import json,sys
from pathlib import Path
b=Path(sys.argv[1]); a=json.loads((b/'resumed/completion.json').read_text()); c=json.loads((b/'continuous/completion.json').read_text())
print('RESUMED:',a['head_sha256']); print('CONTINUOUS:',c['head_sha256'])
if a['head_sha256']!=c['head_sha256']: raise SystemExit('RESUME_EQUIVALENCE_FAILED')
print('RESUME_EQUIVALENCE_PASS')
PY

  echo "========== 4. BONN13 MULTI-SCENE 13-STEP DEV =========="
  run_logged "$BASE/bonn13_13steps.log" python "$SCRIPT" \
    --mode dev --profile bonn13 --manifest-dir "$MANIFEST" \
    --snapshot "$SNAPSHOT" --init-head "$INIT" \
    --protocol-file "$PROTO" --deterministic \
    --slots-per-seq 1 --steps 13 --seed "$SEED" \
    --cache-root "$BASE/bonn13_cache" --run-dir "$BASE/bonn13_dev" || exit 1
  test -f "$BASE/bonn13_dev/completion.json" || { echo "[FAILED] Bonn13 completion missing"; exit 1; }

  python - "$BASE" "$CURRENT_GATE" "$SCRIPT" "$PROTO" "$MANIFEST/manifest_identity.json" <<'PY' || exit 1
import sys,hashlib,json,os
from pathlib import Path
base,g,s,p,m=map(Path,sys.argv[1:])
def sha(x): return hashlib.sha256(x.read_bytes()).hexdigest()
obj={"status":"GATE_PASS","output_directory":str(base),"trainer_sha256":sha(s),"protocol_sha256":sha(p),"manifest_identity_sha256":sha(m),"resume_test":"EXACT_HEAD_MATCH","bonn13_short_dev":"13_UPDATES_COMPLETE","authorized_scope":"BONN_IN_DOMAIN_ONE_SEED_ONLY"}
tmp=Path(str(g)+'.tmp');tmp.write_text(json.dumps(obj,indent=2));os.replace(tmp,g)
print('GATE_PASS:',g)
PY
  echo "[GATE PASS] Only the single-seed Bonn in-domain 1000-update run is allowed next."
  exit 0
fi

if [ "$mode" = "train" ]; then
  assert_gate || exit 1
  RUN="$OUTROOT/y04_v4_bonn13_indomain_seed${SEED}_$(date +%Y%m%d_%H%M%S)"
  echo "========== ONE-SEED BONN13 IN-DOMAIN ADAPTATION =========="
  echo "RUN_DIR: $RUN"
  run_logged "$RUN.log" python "$SCRIPT" \
    --mode adapt --profile bonn13 --manifest-dir "$MANIFEST" \
    --snapshot "$SNAPSHOT" --init-head "$INIT" \
    --protocol-file "$PROTO" --deterministic \
    --slots-per-seq 8 --steps 1000 --seed "$SEED" --save-every 25 \
    --cache-root "$RUN/cache" --run-dir "$RUN" || exit 1
  test -f "$RUN/completion.json" || { echo "[FAILED] Adapt completion missing"; exit 1; }
  echo "[ADAPT_COMPLETE] $RUN"
  exit 0
fi

if [ "$mode" = "resume" ]; then
  RUN="${2:-}"
  if [ -z "$RUN" ] || [ ! -f "$RUN/run_identity.json" ]; then
    echo "Usage: bash $0 resume /absolute/path/to/unfinished/run"
    exit 1
  fi
  assert_gate || exit 1
  run_logged "$RUN.resume_$(date +%Y%m%d_%H%M%S).log" python "$SCRIPT" \
    --mode adapt --profile bonn13 --manifest-dir "$MANIFEST" \
    --snapshot "$SNAPSHOT" --init-head "$INIT" \
    --protocol-file "$PROTO" --deterministic \
    --slots-per-seq 8 --steps 1000 --seed "$SEED" --save-every 25 \
    --cache-root "$RUN/cache" --run-dir "$RUN" --resume || exit 1
  exit 0
fi

echo "Usage: bash $0 gate | train | resume /path/to/run"
exit 2
