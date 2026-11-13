#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")"
python -u experiments/p1/cvpr_m05_formal.py "$@"
