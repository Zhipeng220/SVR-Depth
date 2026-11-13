#!/usr/bin/env python3
"""Run byte-preserved closed-loop or same-parent intervention protocols.

Arguments following the protocol name are passed to the historical runner.
"""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs-dir', help='Stage exact historically locked Bonn manifests before running')
    p.add_argument('protocol', choices=('closed_loop', 'intervention'))
    p.add_argument('arguments', nargs=argparse.REMAINDER)
    a = p.parse_args()
    from svr_depth.utils.legacy import materialize
    runtime = materialize()
    if a.pairs_dir:
        from svr_depth.evaluation.legacy_bridge import stage_locked_pairs
        stage_locked_pairs(runtime, Path(a.pairs_dir))
    name = 'cvpr_m04_closed_loop.py' if a.protocol == 'closed_loop' else 'cvpr_m05_formal.py'
    rest = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    return subprocess.run([sys.executable, '-u', str(runtime / 'experiments/p1' / name), *rest],
                          cwd=runtime, check=False).returncode

if __name__ == '__main__':
    raise SystemExit(main())
