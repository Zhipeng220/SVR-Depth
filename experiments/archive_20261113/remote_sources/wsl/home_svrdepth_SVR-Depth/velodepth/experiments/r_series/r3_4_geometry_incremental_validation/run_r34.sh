#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
output_root="$project_root/outputs/r_series/r3_4_geometry_incremental_validation"
svr_python="${SVR_PYTHON:-/root/miniconda3/envs/svrdepth/bin/python}"
if [[ ! -x "$svr_python" ]]; then svr_python="$(command -v python3 || command -v python)"; fi
stage="${1:-preflight}"
cd "$project_root"
run_root() {
  [[ -f "$output_root/LATEST_RUN_PATH.txt" ]] || { echo "No R3.4 output root yet. Run analyze first." >&2; return 2; }
  cat "$output_root/LATEST_RUN_PATH.txt"
}
case "$stage" in
  preflight)
    PYTHONPATH="$script_dir${PYTHONPATH:+:$PYTHONPATH}" "$svr_python" -m unittest discover -s "$script_dir" -p 'test_r3_4*.py' -v
    "$svr_python" -m py_compile "$script_dir/r3_4_incremental_validation.py" "$script_dir/test_r3_4_incremental_validation.py"
    echo "R3.4 preflight passed."
    ;;
  analyze)
    [[ $# -ge 2 ]] || { echo "Usage: $0 analyze /path/to/R3_3_review.zip" >&2; exit 2; }
    input="$(realpath "$2")"
    [[ -f "$input" || -d "$input" ]] || { echo "Input not found: $input" >&2; exit 2; }
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    root="$output_root/run_$stamp"
    mkdir -p "$root"
    printf '%s\n' "$root" > "$output_root/LATEST_RUN_PATH.txt"
    cp "$input" "$root/input_R3_3_review.zip" 2>/dev/null || true
    "$svr_python" "$script_dir/r3_4_incremental_validation.py" --input "$input" --out-dir "$root"
    echo "R3.4 output root: $root"
    ;;
  pack)
    root="$(run_root)"
    archive="${root}_review.zip"
    [[ ! -e "$archive" ]] || { echo "Archive already exists: $archive" >&2; exit 2; }
    "$svr_python" - "$root" "$archive" "$script_dir" <<'PY'
import hashlib, pathlib, sys, zipfile
root=pathlib.Path(sys.argv[1]).resolve(); archive=pathlib.Path(sys.argv[2]).resolve(); src=pathlib.Path(sys.argv[3]).resolve()
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if p.is_file(): z.write(p, p.relative_to(root.parent).as_posix())
    for p in sorted(src.glob('*')):
        if p.is_file() and p.suffix in ('.py','.md','.sh'):
            z.write(p, 'runner/r3_4_geometry_incremental_validation/' + p.name)
    entries=sorted((n,hashlib.sha256(z.read(n)).hexdigest()) for n in z.namelist())
    z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in entries))
    if z.testzip() is not None: raise RuntimeError('ZIP integrity check failed')
print(archive)
PY
    ;;
  *)
    echo "Usage: $0 {preflight|analyze /path/to/R3_3_review.zip|pack}" >&2
    exit 2
    ;;
esac
