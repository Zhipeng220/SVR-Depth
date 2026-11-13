#!/usr/bin/env python3
"""Run M01 quality or independently repeated timing from an explicit config."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svr_depth.evaluation.m01_runner import execute


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='M01 JSON configuration')
    p.add_argument('--mode', required=True, choices=['quality', 'timing'])
    p.add_argument('--out-dir', required=True, help='new output directory; never overwritten')
    p.add_argument('--development-frames', type=int, help='limit each sequence for development only; results are marked non-formal')
    args = p.parse_args()
    result = execute(args.config, args.out_dir, args.mode, development_frames=args.development_frames)
    label = "DEVELOPMENT " if args.development_frames is not None else ""
    print(f"[M01 {label}{'COMPLETE' if result['complete'] else 'INCOMPLETE'}] {result}")
    if not result['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
