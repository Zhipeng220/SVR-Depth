#!/usr/bin/env bash
# M05-derived, same-parent single-intervention diagnostic; NOT an online policy.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  echo "Usage: $0 {smoke|full} PLAN_JSON OUTPUT_DIRECTORY" >&2
  echo "Requires VDEEP_SNAPSHOT and M05_E4C_CHECKPOINT; optional PYTHON." >&2
  echo "smoke = first frozen cycle per recording; full = all frozen cycles." >&2
}
if [[ $# != 3 ]]; then usage; exit 2; fi
profile="$1"; plan="$2"; out="${3%/}"
[[ -n "$out" ]] || { echo 'Output directory must not be empty or /' >&2; exit 2; }
case "$profile" in smoke|full) ;; *) usage; exit 2;; esac
: "${VDEEP_SNAPSHOT:?Set VDEEP_SNAPSHOT to the pinned VeloDepth snapshot}"
: "${M05_E4C_CHECKPOINT:?Set M05_E4C_CHECKPOINT to the E4-C Best checkpoint}"
PYTHON="${PYTHON:-python}"
[[ -f "$plan" ]] || { echo "Missing prepared M05 plan: $plan" >&2; exit 2; }
[[ -f "$M05_E4C_CHECKPOINT" ]] || { echo "Missing E4-C checkpoint: $M05_E4C_CHECKPOINT" >&2; exit 2; }
[[ -f "$VDEEP_SNAPSHOT/model.safetensors" ]] || { echo 'Missing model.safetensors' >&2; exit 2; }
[[ ! -e "$out" ]] || { echo "Output directory already exists: $out" >&2; exit 2; }
[[ ! -e "${out}_oracle" ]] || { echo "Oracle output already exists: ${out}_oracle" >&2; exit 2; }
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; switch to GPU mode"'

cycles=()
if [[ "$profile" == smoke ]]; then cycles=(--development-max-cycles 1); fi
echo "[risk-update] profile=$profile; single-seed E4-C development only"
echo '[risk-update] O/T single intervention, replay repair off, H1-H4; not a closed-loop oracle'
echo '[risk-update] smoke uses one selected K16 cycle per recording, not 33 frames'
"$PYTHON" -u experiments/p1/cvpr_m05_formal.py run \
  --plan "$plan" --snapshot "$VDEEP_SNAPSHOT" \
  --development-e4c "$M05_E4C_CHECKPOINT" \
  --development-e4c-sha256 0447fbcc25d1ea93d5a80e7f8a7a3b38ac911a8465951009da53988e712754c0 \
  --development-e4c-seed 20261103 --out-dir "$out" "${cycles[@]}"
"$PYTHON" experiments/risk_update/oracle_probe.py --input-dir "$out" --out-dir "${out}_oracle"
