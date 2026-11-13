"""Temporary bridge for the validated historical P_multi training routine."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path
from svr_depth.utils.legacy import materialize

def train_persistent(*, snapshot: str, pairs_dir: str, out_dir: str, seed: int, smoke: bool = False, resume: bool = False) -> int:
    if seed not in (20261025, 20261101, 20261102):
        raise ValueError('Historical protocol supports seeds 20261025, 20261101 and 20261102')
    paths = [Path(x).expanduser().resolve() for x in (snapshot, pairs_dir, out_dir)]
    runtime = materialize()
    if smoke and resume:
        raise ValueError('Cannot resume a smoke run')
    runner = runtime / 'experiments/p1/train_pmulti_fresh.py'
    if resume:
        # The resume helper was created after the historical archive. It is
        # kept separate so the original source lock remains byte-identical.
        source = Path(__file__).resolve().parents[1] / '_compat/train_pmulti_resume.py'
        runner = runtime / 'experiments/p1/train_pmulti_resume.py'
        data = source.read_bytes()
        if runner.exists() and runner.read_bytes() != data:
            raise RuntimeError(f'Refusing to overwrite changed resume helper: {runner}')
        if not runner.exists():
            runner.write_bytes(data)
    command = [sys.executable, '-u', str(runner),
               '--snapshot', str(paths[0]), '--pairs-dir', str(paths[1]), '--out-dir', str(paths[2]),
               '--seed', str(seed)]
    if smoke:
        command.append('--smoke')
    if resume:
        command.append('--resume')
    return subprocess.run(command, cwd=runtime, check=False).returncode
