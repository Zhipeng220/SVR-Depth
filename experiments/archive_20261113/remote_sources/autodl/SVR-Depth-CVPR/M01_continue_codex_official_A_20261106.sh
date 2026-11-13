#!/usr/bin/env bash
# Continue the existing Codex M01 pipeline without modifying its project.
# Development-only upstream A smoke; NOT a VeloDepth paper reproduction.
set -Eeuo pipefail

PROJECT="${PROJECT:-/root/autodl-tmp/m01_20261106/M01_temporal_dev_20261106}"
if [[ ! -f "$PROJECT/scripts/m01_official_dev_smoke.py" ]]; then
  echo "ERROR: Codex project not found: $PROJECT" >&2
  echo "Set PROJECT to the *existing* M01_temporal_dev_20261106 directory." >&2
  exit 2
fi
PROJECT="$(cd "$PROJECT" && pwd -P)"
ORIGINAL="$PROJECT/reference/Velodepth-upstream"
CONFIG="$PROJECT/configs/eval/official_four_datasets.json"
DEV_CONFIG="$PROJECT/m01_dev_e4c.json"
if [[ ! -f "$ORIGINAL/velodepth/datasets/sequence_dataset.py" || ! -f "$ORIGINAL/scripts/eval.py" ]]; then
  echo 'ERROR: original upstream dataset Python sources are missing on this host.' >&2
  echo 'Do not restore an old local version over the server; obtain an intact server copy.' >&2
  exit 2
fi
if [[ ! -f "$CONFIG" || ! -f "$DEV_CONFIG" ]]; then
  echo 'ERROR: expected Codex configuration not found.' >&2
  exit 2
fi
# Reuse the four-HDF5 paths already frozen for development. Do not guess DATAROOT.
DATAROOT="${DATAROOT:-$(python - "$DEV_CONFIG" <<'PY'
import json, pathlib, sys
c=json.load(open(sys.argv[1],encoding='utf-8'))
paths={str(pathlib.Path(s['hdf5']).parent) for s in c['sequences'] if s.get('hdf5')}
if len(paths)!=1:
    raise SystemExit('ERROR: HDF5 files do not share a unique data root: '+repr(paths))
print(paths.pop())
PY
)}"
HF_DIR="${HF_HOME:-/root/autodl-tmp/hf_cache}"
if [[ ! -d "$DATAROOT" || ! -d "$HF_DIR" ]]; then
  echo "ERROR: DATAROOT or HF_HOME not found (DATAROOT=$DATAROOT HF_HOME=$HF_DIR)" >&2
  exit 2
fi
for dataset in Bonn TUM ScanNet Sintel; do
  if [[ ! -f "$DATAROOT/$dataset.hdf5" ]]; then
    echo "ERROR: missing $DATAROOT/$dataset.hdf5" >&2
    exit 2
  fi
done
# Prefer an explicitly provided compiled KNN extension directory. Otherwise
# auto-select only when exactly one directory can be proven to contain KNN*.so.
if [[ -z "${KNN_LIB_DIR:-}" ]]; then
  mapfile -t knn_dirs < <(find /root/autodl-tmp -type f -name 'KNN*.so' -printf '%h\n' 2>/dev/null | sort -u)
  if (( ${#knn_dirs[@]} != 1 )); then
    echo "ERROR: found ${#knn_dirs[@]} KNN extension directories; set KNN_LIB_DIR explicitly:" >&2
    printf '%s\n' "${knn_dirs[@]}" >&2
    exit 2
  fi
  KNN_LIB_DIR="${knn_dirs[0]}"
fi
if [[ ! -d "$KNN_LIB_DIR" ]] || ! compgen -G "$KNN_LIB_DIR/KNN*.so" >/dev/null; then
  echo "ERROR: no KNN*.so in $KNN_LIB_DIR" >&2
  exit 2
fi
python - <<'PY'
import numpy,torch
print('Python runtime numpy=',numpy.__version__, 'torch=',torch.__version__, 'cuda=',torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit('ERROR: GPU unavailable. Do not run official A smoke until the instance is in GPU mode.')
PY

WORKDIR="$(mktemp -d /root/autodl-tmp/m01_upstream_numpy2_XXXXXXXX)"
cleanup() { rm -rf -- "$WORKDIR"; }
trap cleanup EXIT
cp -a "$ORIGINAL" "$WORKDIR/upstream"

# Only the two verified legacy NumPy API calls in a TEMPORARY UPSTREAM COPY.
# Never edit the original upstream tree or any training / P_multi code.
python - "$ORIGINAL" "$WORKDIR/upstream" "$WORKDIR/compat_identity.json" <<'PY'
import hashlib,json,pathlib,sys
orig, temp, identity=map(pathlib.Path,sys.argv[1:])
items=[]
for relative in ('velodepth/datasets/sequence_dataset.py','velodepth/datasets/video_dataset.py'):
    source,target=orig/relative,temp/relative
    if not source.is_file() or not target.is_file():
        raise SystemExit(f'ERROR: missing upstream Python source: {relative}')
    before=source.read_bytes()
    text=target.read_text(encoding='utf-8')
    count=text.count('.tostring()')
    after=text.replace('.tostring()', '.tobytes()')
    if count:
        target.write_text(after, encoding='utf-8')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==hashlib.sha256(before).hexdigest()
    items.append({'relative_path':relative,'legacy_api_replacements_in_temporary_copy':count,
                  'original_sha256':hashlib.sha256(before).hexdigest(),
                  'temporary_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
identity.write_text(json.dumps({'scope':'development_only_upstream_numpy_compat',
    'original_upstream_source_unchanged': True,'files':items},indent=2)+'\n',encoding='utf-8')
print('Runtime compatibility copy prepared; original upstream files were not modified.', flush=True)
for item in items:print(' ',item['relative_path'], 'replacements=',item['legacy_api_replacements_in_temporary_copy'])
PY

OUT="${OUTPUT_DIR:-/root/autodl-tmp/m01_20261106/official_a_codex_resume_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ -e "$OUT" ]]; then
  echo "ERROR: result directory already exists: $OUT" >&2
  exit 2
fi
cd "$PROJECT"
set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/m01_official_dev_smoke.py \
  --official-repo "$WORKDIR/upstream" \
  --dataroot "$DATAROOT" \
  --config-file "$CONFIG" \
  --knn-lib "$KNN_LIB_DIR" \
  --hf-home "$HF_DIR" \
  --output-dir "$OUT" \
  --max-frames 2 --max-sequences 1
STATUS=$?
set -e
if [[ -d "$OUT" ]]; then
  cp "$WORKDIR/compat_identity.json" "$OUT/compat_identity.json"
  printf '\nOFFICIAL_A_SMOKE_EXIT_CODE=%s\n' "$STATUS"
  printf 'OUTPUT_DIR=%s\n' "$OUT"
  echo '--- Last 60 log lines ---'
  if [[ -f "$OUT/official_dev.log" ]]; then tail -n 60 "$OUT/official_dev.log"; fi
fi
exit "$STATUS"
