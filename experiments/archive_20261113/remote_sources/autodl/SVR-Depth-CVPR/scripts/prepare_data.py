#!/usr/bin/env python3
"""Audit an existing raw Bonn RGB/depth pairing manifest."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svr_depth.datasets.bonn_raw import read_pairs
from svr_depth.utils.provenance import sha256_file

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', required=True, help='Existing RGB/GT CSV; original paths remain unchanged')
    p.add_argument('--max-delta-ms', type=float, default=20.0)
    args = p.parse_args()
    rows = read_pairs(args.pairs, max_delta_ms=args.max_delta_ms)
    print(json.dumps({'manifest': str(Path(args.pairs).resolve()), 'sha256': sha256_file(args.pairs),
        'frames': len(rows), 'matched': sum(r.status == 'MATCHED' for r in rows),
        'unmatched': sum(r.status == 'UNMATCHED' for r in rows),
        'max_delta_ms': args.max_delta_ms}, indent=2))

if __name__ == '__main__':
    main()
