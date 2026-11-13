"""Reproducible, monotonic one-to-one timestamp pairing for raw Bonn data."""
from __future__ import annotations
import csv
import math
from pathlib import Path

EPS = 1e-9

def read_index(root: Path, filename: str, subdir: str) -> list[dict]:
    rows = []
    for line_number, line in enumerate((root / filename).read_text(encoding='utf8', errors='replace').splitlines(), 1):
        fields = line.strip().split()
        if not fields or fields[0].startswith('#') or len(fields) < 2:
            continue
        try:
            stamp = float(fields[0])
        except ValueError:
            continue
        if not math.isfinite(stamp):
            raise ValueError(f'Nonfinite timestamp in {filename}:{line_number}')
        p = Path(fields[1])
        candidates = [p] if p.is_absolute() else [root/p, root/subdir/p, root/subdir/p.name, root/p.name]
        resolved = next((c.resolve() for c in candidates if c.is_file()), None)
        rows.append({'timestamp': stamp, 'path': resolved, 'raw_path': fields[1], 'line_number': line_number})
    if not rows:
        raise ValueError(f'No valid entries in {root / filename}')
    return rows

def ordered_match(rgb: list[dict], depth: list[dict], tolerance_ms: float) -> tuple[list[dict], dict[int, int]]:
    """Maximize matched count, then minimize total timestamp difference.

    Tie-breaking follows the historical GT4 pairing implementation. A depth
    image can only be used once, and assignment order must remain monotonic.
    """
    if not math.isfinite(tolerance_ms) or tolerance_ms <= 0:
        raise ValueError('tolerance_ms must be positive and finite')
    if any(rgb[i]['timestamp'] <= rgb[i-1]['timestamp'] for i in range(1, len(rgb))):
        raise ValueError('RGB timestamps must be strictly increasing')
    if any(row['path'] is None for row in rgb):
        raise FileNotFoundError('At least one indexed RGB image is missing')
    unique = {}
    for row in depth:
        if row['path'] is not None:
            unique.setdefault(str(row['path']), row)
    valid = sorted(unique.values(), key=lambda row: (row['timestamp'], row['line_number']))
    n, m = len(rgb), len(valid)
    count = [[0] * (m + 1) for _ in range(n + 1)]
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    action = [bytearray(m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_count, best_cost, best_action = count[i-1][j], cost[i-1][j], 1
            c2, d2 = count[i][j-1], cost[i][j-1]
            if c2 > best_count or (c2 == best_count and d2 < best_cost - EPS):
                best_count, best_cost, best_action = c2, d2, 2
            delta = abs(valid[j-1]['timestamp'] - rgb[i-1]['timestamp']) * 1000.0
            if delta <= tolerance_ms + EPS:
                c3, d3 = count[i-1][j-1] + 1, cost[i-1][j-1] + delta
                if c3 > best_count or (c3 == best_count and d3 < best_cost - EPS) or (c3 == best_count and abs(d3-best_cost) <= EPS):
                    best_count, best_cost, best_action = c3, d3, 3
            count[i][j], cost[i][j], action[i][j] = best_count, best_cost, best_action
    matches, i, j = {}, n, m
    while i > 0 and j > 0:
        a = action[i][j]
        if a == 3:
            matches[i-1] = j-1; i -= 1; j -= 1
        elif a == 1:
            i -= 1
        elif a == 2:
            j -= 1
        else:
            raise RuntimeError('Invalid timestamp pairing traceback')
    assigned = [matches[k] for k in sorted(matches)]
    if any(a >= b for a, b in zip(assigned, assigned[1:])):
        raise RuntimeError('Depth matches are not unique and monotonic')
    if any(valid[a]['timestamp'] >= valid[b]['timestamp'] for a, b in zip(assigned, assigned[1:])):
        raise RuntimeError('Matched depth timestamps are not strictly increasing')
    return valid, matches

def build_pairs(root: str | Path, output: str | Path, *, tolerance_ms: float = 20.0) -> dict:
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    rgb, depth = read_index(root, 'rgb.txt', 'rgb'), read_index(root, 'depth.txt', 'depth')
    valid, matches = ordered_match(rgb, depth, tolerance_ms)
    if not matches:
        raise ValueError('No RGB frames matched a valid depth image')
    fields = ['frame_index_zero_based', 'prediction_slot', 'rgb_timestamp', 'rgb_path', 'depth_timestamp',
              'depth_path', 'signed_delta_ms', 'abs_delta_ms', 'gt_match_status', 'gt_raw_index_line', 'gt_reused']
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, image in enumerate(rgb):
            d = valid[matches[idx]] if idx in matches else None
            signed = (d['timestamp'] - image['timestamp']) * 1000.0 if d else ''
            writer.writerow({'frame_index_zero_based': idx, 'prediction_slot': idx,
                'rgb_timestamp': image['timestamp'], 'rgb_path': str(image['path']),
                'depth_timestamp': d['timestamp'] if d else '', 'depth_path': str(d['path']) if d else '',
                'signed_delta_ms': signed, 'abs_delta_ms': abs(signed) if d else '',
                'gt_match_status': 'MATCHED' if d else 'NO_GT_WITHIN_TOLERANCE_OR_UNASSIGNED',
                'gt_raw_index_line': d['line_number'] if d else '', 'gt_reused': False})
    return {'sequence_root': str(root), 'manifest': str(output), 'rgb_frames': len(rgb),
            'matched': len(matches), 'unmatched': len(rgb)-len(matches),
            'missing_depth_index_files': sum(r['path'] is None for r in depth), 'tolerance_ms': tolerance_ms}

if __name__ == '__main__':
    import argparse
    import hashlib
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-delta-ms', type=float, default=20.0)
    args = parser.parse_args()
    result = build_pairs(args.sequence_root, args.output, tolerance_ms=args.max_delta_ms)
    def sha(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(b)
        return h.hexdigest()
    result['manifest_sha256'] = sha(args.output)
    result['rgb_index_sha256'] = sha(Path(args.sequence_root) / 'rgb.txt')
    result['depth_index_sha256'] = sha(Path(args.sequence_root) / 'depth.txt')
    result['note'] = 'New data pairing; compare exact frame identities with earlier runs before claiming matched inputs.'
    print(json.dumps(result, indent=2), flush=True)
