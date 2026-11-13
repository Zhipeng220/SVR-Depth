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
from svr_depth.evaluation.m01_v31.hdf5_preflight import _h5_text, parse_split, decode_packed12_normalized


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
    # Values are only populated under an explicitly verified source contract.
    flow_fwd: np.ndarray | None = None
    flow_bwd: np.ndarray | None = None
    flow_fwd_mask: np.ndarray | None = None
    flow_bwd_mask: np.ndarray | None = None
    pose_valid: bool = False
    camera_valid: bool = False
    flow_valid: bool = False
    source_frame_id: str = ""
    source_frame_index: int | None = None
    flow_fwd_source: str = ""
    flow_bwd_source: str = ""
    flow_available_fields: tuple[str, ...] = ()
    flow_error: str = ""


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


def _record(spec, index, rgb, gt, timestamp, intrinsics=None, pose=None, error='', **temporal):
    if gt is not None and gt.shape != rgb.shape[:2]:
        raise ValueError(f"{spec['sequence']} frame {index}: GT and RGB shapes differ: {gt.shape}, {rgb.shape[:2]}")
    mask = None if gt is None else np.isfinite(gt) & (gt > float(spec['min_depth_m'])) & (gt <= float(spec['max_depth_m']))
    return FrameRecord(dataset=spec['dataset'], sequence=spec['sequence'], scene=spec['scene'],
                       index=index, rgb=rgb, gt=gt, timestamp=timestamp, intrinsics=intrinsics,
                       pose=pose, gt_valid_mask=mask, gt_error=error, **temporal)


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
                      spec.get('intrinsics'), spec.get('pose'), error,
                      source_frame_id=str(row.rgb_path), source_frame_index=row.index)


def _at(sample, key, index):
    value = sample.get(key)
    return value[index] if isinstance(value, list) and 0 <= index < len(value) else None


def decode_benchmark_flow(blob, *, encoding, geometry):
    """Decode under a caller-verified contract; never infer physical units."""
    if encoding != 'velodepth_packed12_rgb' or geometry != 'fraction_of_extent':
        raise ValueError('unsupported or unverified flow encoding/geometry')
    with Image.open(BytesIO(blob)) as im:
        raw = np.asarray(im)
    normalized = decode_packed12_normalized(raw)
    h, w = normalized.shape[:2]
    if min(h, w) < 2:
        raise ValueError('extent-based flow requires height and width >= 2')
    return normalized * np.asarray([w - 1, h - 1], dtype=np.float32)


def _flow_mask(blob, shape):
    with Image.open(BytesIO(blob)) as im:
        raw = np.asarray(im)
    if raw.ndim != 2 or raw.shape != shape:
        raise ValueError(f'flow mask shape {raw.shape} differs from RGB {shape}')
    return raw.astype(bool)


def _temporal_input(h5, sample, spec, index, shape, image_ref):
    fields = tuple(k for k in ('flow_fwd', 'flow_bwd', 'flow_fwd_mask',
                               'flow_bwd_mask', 'flow', 'flow_mask') if k in sample)
    options = spec.get('temporal_inputs') or {}
    result = {'source_frame_id': str(image_ref), 'source_frame_index': index,
              'flow_available_fields': fields,
              'pose_valid': bool(options.get('pose_provenance_verified') is True
                                 and options.get('pose_evidence')),
              'camera_valid': bool(options.get('camera_geometry_verified') is True
                                   and options.get('camera_evidence'))}
    if options.get('flow_contract_verified') is not True or not options.get('flow_evidence'):
        result['flow_error'] = 'unverified_flow_contract'
        return result
    result['flow_valid'] = True
    try:
        encoding, geometry = options['flow_encoding'], options['flow_geometry']
        for direction in ('fwd', 'bwd'):
            # Each offset and field is explicit; no assumption is made for N+1 flow lists.
            field = options[f'{direction}_field']
            mask_field = options[f'{direction}_mask_field']
            offset = options[f'{direction}_index_offset']
            if isinstance(offset, bool) or not isinstance(offset, int):
                raise ValueError(f'{direction} index offset must be an integer')
            ref = _at(sample, field, index + offset)
            mask_ref = _at(sample, mask_field, index + offset)
            if ref is not None:
                flow = decode_benchmark_flow(_h5_blob(h5, ref), encoding=encoding, geometry=geometry)
                if flow.shape != (*shape, 2):
                    raise ValueError(f'{direction} flow shape {flow.shape} differs from RGB {shape}')
                result[f'flow_{direction}'] = flow
                result[f'flow_{direction}_source'] = str(ref)
            if mask_ref is not None:
                result[f'flow_{direction}_mask'] = _flow_mask(_h5_blob(h5, mask_ref), shape)
    except Exception as exc:
        result['flow_error'] = f'{type(exc).__name__}: {exc}'
    return result


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
            temporal = _temporal_input(h5, sample, spec, index, rgb.shape[:2], image_ref) if load_gt else {
                'source_frame_id': str(image_ref), 'source_frame_index': index}
            yield _record(spec, index, rgb, gt, timestamp, camera, _at(sample, 'cam2w', index), error,
                          **temporal)


def iter_sequence(spec, *, load_gt=True):
    if spec['dataset'] == 'Bonn' and 'pairs_csv' in spec:
        return iter_bonn(spec, load_gt=load_gt)
    if spec['dataset'] in ('Bonn', 'TUM', 'ScanNet', 'Sintel') and 'hdf5' in spec:
        return iter_hdf5(spec, load_gt=load_gt)
    raise ValueError(f"Unsupported M01 dataset: {spec['dataset']}")


# Backward compatibility for callers that imported the original name.
iter_tum = iter_hdf5
