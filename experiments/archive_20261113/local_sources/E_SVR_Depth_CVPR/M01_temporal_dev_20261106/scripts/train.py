#!/usr/bin/env python3
"""Run P_multi training or output-adapter smoke through compatibility code."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('persistent_feature', 'output_smoke'), required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--pairs-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--p-smoke', help='Existing P_multi smoke cache, required for output_smoke')
    a = p.parse_args()
    if a.method == 'output_smoke':
        if not a.smoke or not a.p_smoke or a.resume or a.seed != 20261025:
            p.error('output_smoke requires --smoke, --p-smoke and --seed 20261025; --resume is unsupported')
        from svr_depth.training.output_smoke import run_output_smoke
        return run_output_smoke(snapshot=a.snapshot, pairs_dir=a.pairs_dir, p_smoke=a.p_smoke, out_dir=a.out_dir)
    from svr_depth.training.trainer import train_persistent
    return train_persistent(snapshot=a.snapshot, pairs_dir=a.pairs_dir, out_dir=a.out_dir, seed=a.seed, smoke=a.smoke, resume=a.resume)

if __name__ == '__main__':
    raise SystemExit(main())
