"""M01 frame sources: raw Bonn CSV and VeloDepth-schema HDF5 sequences.

The HDF5 RGB/depth byte and depth-decoding conventions follow VeloDepth's
SequenceDataset; unlike its official evaluation entry, this adapter processes
all manifest-selected frames and does not require reference optical flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .bonn_raw import read_pairs
from .native_tartanair import _h5_blob, _decode_h5_depth, CrossSourceDataError
from svr_depth.evaluation.m01_v31.hdf5_preflight import _h5_text, parse_split


@dataclass(frozen=True)
class FrameRecord:
    dataset: str
    sequence: str
    scene: str
    index: int
    rgb: np.ndarray
    gt: np.ndarray | None
    timestamp: float | None
    intrinsics: object = None
    pose: object = None
    gt_valid_mask: np.ndarray | None = None
    gt_error: str = ""


def _image(path_or_bytes):
    with Image.open(path_or_bytes) as im:
        return np.array(im.convert('RGB'), dtype=np.uint8, copy=True)


def _depth_png(path, scale):
    with Image.open(path) as im:
        raw = np.asarray(im)
    if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer) or raw.min() < 0 or raw.max() > 65535:
        raise ValueError(f'Expected unsigned 16-bit-compatible single-channel depth: {path}')
    return raw.astype(np.float32) / np.float32(scale)


def _h5_depth(blob, scale):
    try:
        return _decode_h5_depth(blob, scale_to_m=scale)
    except CrossSourceDataError:
        # Pillow versions may decode a 16-bit grayscale PNG into signed int32.
        with Image.open(BytesIO(blob)) as im:
            raw = np.asarray(im)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer) or raw.min() < 0 or raw.max() > 65535:
            raise
        return raw.astype(np.float32) / np.float32(scale)


def _record(spec, index, rgb, gt, timestamp, intrinsics=None, pose=None, error=''):
    if gt is not None and gt.shape != rgb.shape[:2]:
        raise ValueError(f"{spec['sequence']} frame {index}: GT and RGB shapes differ: {gt.shape}, {rgb.shape[:2]}")
    mask = None if gt is None else np.isfinite(gt) & (gt > float(spec['min_depth_m'])) & (gt <= float(spec['max_depth_m']))
    return FrameRecord(spec['dataset'], spec['sequence'], spec['scene'], index, rgb, gt,
                       timestamp, intrinsics, pose, mask, error)


def iter_bonn(spec, *, load_gt=True):
    rows = read_pairs(spec['pairs_csv'], max_delta_ms=float(spec['max_delta_ms']))
    for row in rows:
        rgb = _image(row.rgb_path)
        gt, error = None, ''
        if load_gt and row.depth_path is not None:
            try:
                gt = _depth_png(row.depth_path, float(spec['depth_scale']))
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        yield _record(spec, row.index, rgb, gt, row.rgb_timestamp,
                      spec.get('intrinsics'), spec.get('pose'), error)


def _at(sample, key, index):
    value = sample.get(key)
    return value[index] if isinstance(value, list) and index < len(value) else None


def iter_hdf5(spec, *, load_gt=True):
    if spec['dataset'] == 'ScanNet' and not spec.get('sequences_key'):
        raise ValueError('ScanNet requires an explicit sequences_key; ScanNetVid uses sequences_.json')
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError('HDF5 sources require h5py in the svrdepth environment') from exc
    path = Path(spec['hdf5']).expanduser().resolve(strict=True)
    with h5py.File(path, 'r') as h5:
        samples = json.loads(_h5_text(h5, spec.get('sequences_key', 'sequences.json')))
        split = dict(parse_split(_h5_text(h5, spec['split_key'])))
        name = spec['sequence']
        if name not in split or name not in samples:
            raise ValueError(f'HDF5 sequence {name!r} must exist exactly in split and metadata')
        sample = samples[name]
        images = sample['image']
        if not images:
            raise ValueError(f'{name}: no RGB frames in selected sequence')
        if len(images) != split[name]:
            raise ValueError(f'{name}: split frame count differs from sequences.json')
        depths = sample.get('depth', [None] * len(images))
        if len(depths) != len(images):
            raise ValueError(f'{name}: image/depth metadata lengths differ')
        previous_timestamp = -math.inf
        for index, image_ref in enumerate(images):
            rgb = _image(BytesIO(_h5_blob(h5, image_ref)))
            gt, error = None, ''
            if load_gt and depths[index] is not None:
                try:
                    gt = _h5_depth(_h5_blob(h5, depths[index]), float(spec['depth_scale']))
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
            timestamp = _at(sample, 'timestamp', index)
            if timestamp is not None:
                timestamp = float(timestamp)
                if not math.isfinite(timestamp) or timestamp <= previous_timestamp:
                    raise ValueError(f'{name}: non-increasing timestamp at {index}')
                previous_timestamp = timestamp
            camera = _at(sample, 'K', index)
            if camera is None:
                camera = {'model': _at(sample, 'camera_model', index),
                          'params': _at(sample, 'camera_params', index)}
            yield _record(spec, index, rgb, gt, timestamp, camera, _at(sample, 'cam2w', index), error)


def iter_sequence(spec, *, load_gt=True):
    if spec['dataset'] == 'Bonn' and 'pairs_csv' in spec:
        return iter_bonn(spec, load_gt=load_gt)
    if spec['dataset'] in ('Bonn', 'TUM', 'ScanNet', 'Sintel') and 'hdf5' in spec:
        return iter_hdf5(spec, load_gt=load_gt)
    raise ValueError(f"Unsupported M01 dataset: {spec['dataset']}")


# Backward compatibility for callers that imported the original name.
iter_tum = iter_hdf5
