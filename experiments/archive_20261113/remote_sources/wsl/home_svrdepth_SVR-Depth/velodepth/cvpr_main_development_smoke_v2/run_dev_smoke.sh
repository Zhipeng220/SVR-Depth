#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
PKG="$ROOT/cvpr_main_development_smoke_v2"
echo '[PRECHECK] standard-library branch tests'
python -m unittest discover -s "$PKG" -p 'test_*.py' -v
echo '[PRECHECK] source and checkpoint discovery; no model inference'
python -u "$PKG/cvpr_main_inventory_audit.py"
echo '[M01/M02] development prefix smoke, A/O_multi/P only'
python -u "$PKG/cvpr_m01_m02_smoke.py"
echo '[M04] same-head development prefix smoke'
python -u "$PKG/cvpr_m04_smoke.py"
echo '[COMPLETE] DEVELOPMENT SMOKE ONLY; formal M01/M02/M04 remain pending'
