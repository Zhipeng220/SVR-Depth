#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
svr_python="${SVR_PYTHON:-/root/miniconda3/envs/svrdepth/bin/python}"
if [[ ! -x "$svr_python" ]]; then svr_python="$(command -v python3 || command -v python)"; fi
stage="${1:-preflight}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$project_root"
PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "$svr_python" -m unittest discover -s "$script_dir" -p 'test_r3_gate_rules.py' -v
case "$stage" in
  preflight)
    "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" --preflight-only
    ;;
  smoke)
    "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" \
      --data-root /root/autodl-tmp/datasets/rgbd_bonn_crowd2 --sequence crowd2 \
      --frames 32 --refresh-frames 5,14,23
    ;;
  full)
    for sequence in crowd2 balloon2 crowd3; do
      "$svr_python" "$script_dir/r3_simple_commit_gate.py" --project-root "$project_root" \
        --data-root "/root/autodl-tmp/datasets/rgbd_bonn_${sequence}" --sequence "$sequence" \
        --frames 86 --refresh-frames 9,37,65
    done
    ;;
  *) echo "Usage: bash experiments/r_series/r3_simple_commit_gate/run_r3.sh {preflight|smoke|full}" >&2; exit 2 ;;
esac
