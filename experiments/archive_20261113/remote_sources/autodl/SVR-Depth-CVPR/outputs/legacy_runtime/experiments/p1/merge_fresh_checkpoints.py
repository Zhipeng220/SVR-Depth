#!/usr/bin/env python3
"""Combine three completed fresh-training manifests without changing them."""
import argparse
import csv
import hashlib
from pathlib import Path

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows', nargs=3, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rows = []
    for name in a.rows:
        with Path(name).open(newline='', encoding='utf-8-sig') as f:
            data = list(csv.DictReader(f))
        if len(data) != 1:
            raise ValueError(f'Expected one row: {name}')
        row = data[0]
        checkpoint = Path(row['checkpoint_path'])
        if not checkpoint.is_file() or digest(checkpoint) != row['expected_sha256']:
            raise ValueError(f'Checkpoint missing or hash mismatch: {checkpoint}')
        if not Path(row['training_summary_path']).is_file():
            raise ValueError('Training summary missing')
        rows.append(row)
    if {int(r['seed']) for r in rows} != {20261025, 20261101, 20261102}:
        raise ValueError('Expected seeds 20261025, 20261101, 20261102')
    if len({r['expected_sha256'] for r in rows}) != 3:
        raise ValueError('Each seed requires a distinct checkpoint')
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['seed','checkpoint_path','expected_sha256','training_summary_path','training_log_path','notes'])
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: int(r['seed'])))
    print(out.resolve())

if __name__ == '__main__':
    main()
