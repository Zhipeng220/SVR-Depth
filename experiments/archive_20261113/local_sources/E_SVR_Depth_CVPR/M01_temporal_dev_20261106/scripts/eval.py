#!/usr/bin/env python3
"""Run historical closed-loop M04 or the integrated M05 V3.1 intervention.

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
    rest = a.arguments[1:] if a.arguments[:1] == ['--'] else a.arguments
    if a.protocol == 'intervention':
        if a.pairs_dir:
            p.error('M05 V3.1 accepts individual pair CSV options on its prepare command; --pairs-dir is for closed_loop')
        root = Path(__file__).resolve().parents[1]
        return subprocess.run([sys.executable, '-u', str(root / 'experiments/p1/cvpr_m05_formal.py'), *rest],
                              cwd=root, check=False).returncode
    from svr_depth.utils.legacy import materialize
    runtime = materialize()
    if a.pairs_dir:
        from svr_depth.evaluation.legacy_bridge import stage_locked_pairs
        stage_locked_pairs(runtime, Path(a.pairs_dir))
    return subprocess.run([sys.executable, '-u', str(runtime / 'experiments/p1/cvpr_m04_closed_loop.py'), *rest],
                          cwd=runtime, check=False).returncode

if __name__ == '__main__':
    raise SystemExit(main())
