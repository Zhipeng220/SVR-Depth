"""Compatibility smoke for independently trained output-only adapters."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
from svr_depth.utils.legacy import materialize
from svr_depth.utils.provenance import sha256_file

def run_output_smoke(*, snapshot: str, pairs_dir: str, p_smoke: str, out_dir: str) -> int:
    source = Path(__file__).resolve().parents[1] / '_compat' / 'output_smoke'
    snapshot, pairs_dir, p_smoke, out_dir = [Path(x).expanduser().resolve() for x in (snapshot, pairs_dir, p_smoke, out_dir)]
    identity = json.loads((p_smoke / 'run_identity.json').read_text(encoding='utf8'))
    if identity.get('smoke') is not True:
        raise ValueError('Expected existing P smoke identity')
    names = ['smoke_o_fresh_v2.py', 'gt4a2_q0_2_full_cycle_fixed_gain.py',
             'gt4a2_q0_3_necessity_controls.py', 'y08y09_models.py', 'y08y09_common.py']
    # Validate all inputs before materializing or copying anything.
    pairs = {}
    for sequence in ('crowd2', 'crowd3', 'balloon2'):
        path = pairs_dir / f'gt4a2_exp1_{sequence}_pairs_used.csv'
        if sha256_file(path) != identity['pair_hashes'][sequence]:
            raise ValueError(f'{sequence} manifest differs from the P smoke manifest')
        pairs[sequence] = path
    runtime = materialize()
    installs = [(source / n, runtime / 'experiments/p1' / n) for n in names]
    installs += [(p, runtime / 'outputs/gt4a2_exp1' / p.name) for p in pairs.values()]
    for src, dst in installs:
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() and (not dst.is_file() or sha256_file(src) != sha256_file(dst)):
            raise RuntimeError(f'Refusing to replace a changed compatibility file: {dst}')
    for src, dst in installs:
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open('xb') as f:
                f.write(src.read_bytes())
    return subprocess.run([sys.executable, '-u', str(runtime / 'experiments/p1/smoke_o_fresh_v2.py'),
                           '--snapshot', str(snapshot), '--p-smoke', str(p_smoke), '--out-dir', str(out_dir)],
                          cwd=runtime, check=False).returncode
