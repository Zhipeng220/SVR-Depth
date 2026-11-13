#!/usr/bin/env python3
"""Compute descriptive risk from previously saved frame records."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svr_depth.evaluation.records import read_records
from svr_depth.analysis.risk import summarize_against_baseline
from svr_depth.analysis.propagation import summarize_by_age

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('analysis', choices=('risk', 'propagation'))
    p.add_argument('--records', required=True)
    p.add_argument('--baseline', default='A')
    p.add_argument('--baseline-seed', default=None)
    p.add_argument('--severe-delta', type=float, default=0.01)
    p.add_argument('--output', required=True, help='New output JSON; refuses to overwrite')
    a = p.parse_args()
    rows = read_records(a.records)
    results = (summarize_against_baseline(rows, a.baseline, a.severe_delta, a.baseline_seed)
               if a.analysis == 'risk' else summarize_by_age(rows))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf8') as f:
        json.dump({'analysis': a.analysis, 'results': results}, f, indent=2, allow_nan=False)
    print(out.resolve())

if __name__ == '__main__':
    main()
