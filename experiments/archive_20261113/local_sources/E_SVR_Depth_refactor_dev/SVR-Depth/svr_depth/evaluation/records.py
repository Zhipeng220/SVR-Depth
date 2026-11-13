"""Validation for portable, per-frame prediction records."""
from __future__ import annotations
import csv
import math
from pathlib import Path

REQUIRED = ('dataset', 'sequence', 'method', 'seed', 'index', 'k', 'status', 'absrel', 'mask_sha256')
STATUSES = {'ok', 'unmatched', 'invalid_gt', 'invalid_prediction'}

def read_records(path: str | Path) -> list[dict]:
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'Missing record columns: {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError('Frame records are empty')
    seen = set()
    for row in rows:
        if any(not str(row[k]).strip() for k in ('dataset', 'sequence', 'method', 'seed')):
            raise ValueError('Record identity fields must not be empty')
        row['index'] = int(row['index'])
        row['k'] = int(row['k'])
        if row['index'] < 0 or row['k'] < 1 or row['status'] not in STATUSES:
            raise ValueError('Invalid frame index or status')
        key = tuple(row[k] for k in ('dataset', 'sequence', 'method', 'seed', 'k', 'index'))
        if key in seen:
            raise ValueError(f'Duplicate frame record: {key}')
        seen.add(key)
        if row['status'] == 'ok':
            mask = row['mask_sha256'].lower()
            if len(mask) != 64 or any(c not in '0123456789abcdef' for c in mask):
                raise ValueError(f'Invalid GT mask hash for {key}')
            row['mask_sha256'] = mask
            value = float(row['absrel'])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid AbsRel for {key}')
            row['absrel'] = value
        else:
            row['absrel'] = None
    return rows
