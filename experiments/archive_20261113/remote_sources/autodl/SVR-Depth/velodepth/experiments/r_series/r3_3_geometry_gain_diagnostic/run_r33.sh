#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
output_root="$project_root/outputs/r_series/r3_3_geometry_gain_diagnostic"
svr_python="${SVR_PYTHON:-/root/miniconda3/envs/svrdepth/bin/python}"
if [[ ! -x "$svr_python" ]]; then svr_python="$(command -v python3 || command -v python)"; fi
stage="${1:-preflight}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$project_root"

run_root() {
  if [[ -n "${R33_ROOT:-}" ]]; then printf '%s\n' "$R33_ROOT"; return; fi
  [[ -f "$output_root/LATEST_RUN_PATH.txt" ]] || { echo "No R3.3 run root. Run 'start' first." >&2; return 2; }
  cat "$output_root/LATEST_RUN_PATH.txt"
}
run_tests() {
  PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "$svr_python" -m unittest discover -s "$script_dir" -p 'test_r3_3*.py' -v
}

case "$stage" in
  start)
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    root="$output_root/run_$stamp"
    mkdir -p "$root"
    printf '%s\n' "$root" > "$output_root/LATEST_RUN_PATH.txt"
    cat > "$root/R3_3_PROTOCOL_LOCK.json" <<EOF
{
  "experiment": "R3.3 candidate geometry decomposition and writeback-gain diagnosis",
  "created_utc": "$stamp",
  "frames": 86,
  "actual_policy_action_frames_zero_based": [9, 37, 65],
  "diagnostic_anchor_offset": 1,
  "diagnostic_anchor_stride": 4,
  "horizon_H": 16,
  "gamma": 0.9,
  "policy": "AgeGate only at actual action frames; diagnostic-only candidates discarded",
  "geometry_fit": "robust Huber IRLS candidate_depth ~= scale * propagation_depth + shift",
  "weights_frozen": true,
  "predictor_training": false,
  "online_action_uses_gt": false,
  "future_gt_used_only_for_paired_counterfactual_evaluation": true,
  "diagnostic_candidates_add_compute": true,
  "claims_compute_savings": false
}
EOF
    echo "R3.3 run root: $root"
    ;;
  preflight)
    run_tests
    root="$(run_root)"
    stamp="$(date -u +%Y%m%dT%H%M%S_%NZ)"
    out="$root/preflight/$stamp"
    "$svr_python" "$script_dir/r3_3_geometry_gain_diagnostic.py" --project-root "$project_root" \
      --frames 86 --action-frames 9,37,65 --out-dir "$out" --preflight-only
    ;;
  smoke)
    run_tests
    root="$(run_root)"
    stamp="$(date -u +%Y%m%dT%H%M%S_%NZ)"
    out="$root/smoke/$stamp"
    "$svr_python" "$script_dir/r3_3_geometry_gain_diagnostic.py" --project-root "$project_root" \
      --data-root /root/autodl-tmp/datasets/rgbd_bonn_crowd2 --sequence crowd2 \
      --frames 32 --action-frames 5,9,13 --diagnostic-offset 1 --diagnostic-stride 4 \
      --horizon 8 --gamma 0.9 --out-dir "$out"
    ;;
  full)
    run_tests
    root="$(run_root)"
    for seq in crowd2 balloon2 crowd3; do
      echo "=== R3.3 geometry/gain diagnosis: $seq ==="
      "$svr_python" "$script_dir/r3_3_geometry_gain_diagnostic.py" --project-root "$project_root" \
        --data-root "/root/autodl-tmp/datasets/rgbd_bonn_${seq}" --sequence "$seq" \
        --frames 86 --action-frames 9,37,65 --diagnostic-offset 1 --diagnostic-stride 4 \
        --horizon "${R33_HORIZON:-16}" --gamma "${R33_GAMMA:-0.9}" \
        --age-threshold 16 --out-dir "$root/full/$seq"
    done
    printf '%s\n' "$root/full" > "$root/LATEST_FULL_PATH.txt"
    echo "R3.3 full results: $root/full"
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
            z.write(p, 'runner/r3_3_geometry_gain_diagnostic/' + p.name)
    dep_dir=project/'experiments/r_series/r3_2_reset_aware_drift_gate'
    for p in sorted(dep_dir.glob('*')):
        if p.is_file() and p.suffix in ('.py','.md','.sh'):
            z.write(p, 'runner/dependencies/r3_2_reset_aware_drift_gate/' + p.name)
    r2a_dir=project/'experiments/r_series/r2a_commit_enum'
    for p in sorted(r2a_dir.glob('*')):
        if p.is_file() and p.name in ('r2a_commit_enum.py','audited_source_hashes.json','test_r2a_commit_enum.py'):
            z.write(p, 'runner/dependencies/r2a_commit_enum/' + p.name)
    entries=sorted((n,hashlib.sha256(z.read(n)).hexdigest()) for n in z.namelist())
    z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in entries))
    if z.testzip() is not None: raise RuntimeError('ZIP integrity check failed')
print(archive)
PY
    ;;
  *)
    echo "Usage: bash experiments/r_series/r3_3_geometry_gain_diagnostic/run_r33.sh {start|preflight|smoke|full|pack}" >&2
    echo "Optional: SVR_PYTHON, R33_ROOT, R33_HORIZON (default 16), R33_GAMMA (default 0.9)" >&2
    exit 2
    ;;
esac
