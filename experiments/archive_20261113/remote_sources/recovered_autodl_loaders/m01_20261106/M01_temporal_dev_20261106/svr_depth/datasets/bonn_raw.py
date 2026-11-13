"""Read an explicitly paired raw Bonn sequence without changing its protocol."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Frame:
    index: int
    rgb_path: Path
    rgb_timestamp: float
    depth_path: Path | None
    depth_timestamp: float | None
    status: str
    delta_ms: float | None


def read_pairs(path: str | Path, *, verify_files: bool = True, max_delta_ms: float = 20.0) -> list[Frame]:
    """Read all RGB frames, including unmatched frames, in chronological order.

    Existing manifest paths are preserved; this does not re-pair timestamps or
    claim that a dataset is independent of training data.
    """
    if max_delta_ms < 0 or not math.isfinite(max_delta_ms):
        raise ValueError('max_delta_ms must be nonnegative and finite')
    source = Path(path).expanduser().resolve()
    with source.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'Empty sequence manifest: {source}')
    frames, seen_depth = [], set()
    previous = -math.inf
    for expected, row in enumerate(rows):
        idx = int(row['frame_index_zero_based'])
        timestamp = float(row['rgb_timestamp'])
        if idx != expected or not math.isfinite(timestamp) or timestamp <= previous:
            raise ValueError(f'Invalid RGB index or timestamp at row {expected + 2}')
        previous = timestamp
        rgb_raw = row.get('rgb_path', '').strip()
        if not rgb_raw:
            raise ValueError(f'Missing RGB path at index {idx}')
        rgb = Path(rgb_raw).expanduser()
        rgb = rgb if rgb.is_absolute() else source.parent / rgb
        rgb = rgb.resolve()
        if verify_files and not rgb.is_file():
            raise FileNotFoundError(rgb)
        status = row.get('gt_match_status', '').strip().upper()
        if status not in {'MATCHED', 'UNMATCHED'}:
            raise ValueError(f'Invalid GT status at index {idx}: {status!r}')
        depth = None
        depth_timestamp = delta = None
        if status == 'MATCHED':
            raw = row.get('depth_path', '').strip()
            if not raw:
                raise ValueError(f'Matched frame {idx} has no depth path')
            depth = Path(raw).expanduser()
            depth = (depth if depth.is_absolute() else source.parent / depth).resolve()
            if verify_files and not depth.is_file():
                raise FileNotFoundError(depth)
            if depth in seen_depth or row.get('gt_reused', '').strip().lower() in {'true', '1', 'yes'}:
                raise ValueError(f'Depth reused at index {idx}: {depth}')
            seen_depth.add(depth)
            depth_timestamp = float(row['depth_timestamp'])
            delta = float(row['abs_delta_ms'])
            if not all(math.isfinite(x) for x in (depth_timestamp, delta)) or delta < 0 or delta > max_delta_ms:
                raise ValueError(f'Invalid RGB/depth timestamp delta at index {idx}: {delta}')
            if abs(abs(timestamp - depth_timestamp) * 1000 - delta) > 0.1:
                raise ValueError(f'RGB/depth timestamps do not match recorded delta at index {idx}')
        frames.append(Frame(idx, rgb, timestamp, depth, depth_timestamp, status, delta))
    return frames
