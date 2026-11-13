#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
output_root="$project_root/outputs/r_series/r3_5_5_fixed_plan_commit_gate"
svr_python="${SVR_PYTHON:-/root/miniconda3/envs/svrdepth/bin/python}"
if [[ ! -x "$svr_python" ]]; then svr_python="$(command -v python3 || command -v python)"; fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$project_root"
script="$script_dir/r3_5_5_fixed_plan_commit_gate.py"

run_root() {
  [[ -f "$output_root/LATEST_RUN_PATH.txt" ]] || { echo "No R3.5.5 run root. Run 'start' first." >&2; return 2; }
  cat "$output_root/LATEST_RUN_PATH.txt"
}
run_tests() {
  PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "$svr_python" -m unittest discover -s "$script_dir" -p 'test_r3_5_5*.py' -v
  "$svr_python" -m py_compile "$script" "$script_dir/test_r3_5_5_fixed_plan_commit_gate.py"
}
case "${1:-preflight}" in
  start)
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    root="$output_root/run_$stamp"
    mkdir -p "$root"
    printf '%s\n' "$root" > "$output_root/LATEST_RUN_PATH.txt"
    cp "$script_dir/R3_5_5_PROTOCOL.md" "$root/"
    echo "R3.5.5 run root: $root"
    ;;
  preflight)
    run_tests
    "$svr_python" "$script" preflight --project-root "$project_root"
    echo "R3.5.5 preflight passed."
    ;;
  smoke)
    run_tests
    root="$(run_root)"
    stamp="$(date -u +%Y%m%dT%H%M%S_%NZ)"
    out="$root/smoke/$stamp"
    "$svr_python" "$script" collect --project-root "$project_root" \
      --data-root /root/autodl-tmp/datasets/rgbd_bonn_crowd2 --sequence crowd2 \
      --frames 32 --action-frames 5,17,25 --horizon 8 --gamma 0.9 --out-dir "$out"
    ;;
  full)
    run_tests
    root="$(run_root)"
    mkdir -p "$root/full/collect"
    for seq in crowd2 balloon2 crowd3; do
      echo "=== R3.5.5 AlwaysCommit collection: $seq ==="
      "$svr_python" "$script" collect --project-root "$project_root" \
        --data-root "/root/autodl-tmp/datasets/rgbd_bonn_${seq}" --sequence "$seq" \
        --frames 86 --action-frames 5,17,29,41,49,61 --horizon 16 --gamma 0.9 \
        --out-dir "$root/full/collect/$seq"
    done
    "$svr_python" "$script" calibrate --input-root "$root/full/collect" --out-dir "$root/full/calibration"
    if "$svr_python" "$script" should-run-gate --decision-json "$root/full/calibration/r355_gate_decision.json"; then
      mkdir -p "$root/full/never" "$root/full/gate"
      for seq in crowd2 balloon2 crowd3; do
        echo "=== R3.5.5 NeverCommit lower-bound replay: $seq ==="
        "$svr_python" "$script" never --project-root "$project_root" \
          --data-root "/root/autodl-tmp/datasets/rgbd_bonn_${seq}" --sequence "$seq" \
          --frames 86 --action-frames 5,17,29,41,49,61 --horizon 16 --gamma 0.9 \
          --out-dir "$root/full/never/$seq"
      done
      for seq in crowd2 balloon2 crowd3; do
        echo "=== R3.5.5 frozen GeometryGate replay: $seq ==="
        "$svr_python" "$script" gate --project-root "$project_root" \
          --data-root "/root/autodl-tmp/datasets/rgbd_bonn_${seq}" --sequence "$seq" \
          --frames 86 --action-frames 5,17,29,41,49,61 --horizon 16 --gamma 0.9 \
          --calibration-json "$root/full/calibration/r355_gate_decision.json" \
          --out-dir "$root/full/gate/$seq"
      done
      "$svr_python" "$script" compare --baseline-root "$root/full/collect" \
        --never-root "$root/full/never" --gate-root "$root/full/gate" \
        --calibration-json "$root/full/calibration/r355_gate_decision.json" \
        --out-dir "$root/full/comparison"
    else
      echo "OFFLINE STOP: no full gate replay. See $root/full/calibration/R3_5_5_离线门控筛选报告.md"
    fi
    printf '%s\n' "$root/full" > "$root/LATEST_FULL_PATH.txt"
    echo "R3.5.5 full outputs: $root/full"
    ;;
  pack)
    root="$(run_root)"
    archive="${root}_review.zip"
    [[ ! -e "$archive" ]] || { echo "Archive already exists: $archive" >&2; exit 2; }
    "$svr_python" - "$root" "$archive" "$script_dir" "$project_root" <<'PY'
import hashlib, pathlib, sys, zipfile
root=pathlib.Path(sys.argv[1]).resolve(); archive=pathlib.Path(sys.argv[2]).resolve()
script_dir=pathlib.Path(sys.argv[3]).resolve(); project=pathlib.Path(sys.argv[4]).resolve()
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(root.parent).as_posix())
    for p in sorted(script_dir.glob('*')):
        if p.is_file() and p.suffix in ('.py','.md','.sh'):
            z.write(p, 'runner/r3_5_5_fixed_plan_commit_gate/' + p.name)
    for rel, dest in [
        ('experiments/r_series/r3_3_geometry_gain_diagnostic', 'runner/dependencies/r3_3_geometry_gain_diagnostic'),
        ('experiments/r_series/r3_2_reset_aware_drift_gate', 'runner/dependencies/r3_2_reset_aware_drift_gate'),
        ('experiments/r_series/r2a_commit_enum', 'runner/dependencies/r2a_commit_enum'),
    ]:
        dep=project/rel
        if dep.exists():
            for p in sorted(dep.glob('*')):
                if p.is_file() and p.suffix in ('.py','.json','.md','.sh'):
                    z.write(p, dest+'/'+p.name)
    entries=sorted((n,hashlib.sha256(z.read(n)).hexdigest()) for n in z.namelist())
    z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in entries))
    if z.testzip() is not None: raise RuntimeError('ZIP integrity check failed')
print(archive)
PY
    ;;
  *)
    echo "Usage: $0 {start|preflight|smoke|full|pack}" >&2
    exit 2
    ;;
esac
