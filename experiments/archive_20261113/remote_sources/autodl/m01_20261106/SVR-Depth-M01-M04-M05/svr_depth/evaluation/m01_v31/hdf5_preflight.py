"""Inspect HDF5 sequence metadata without decoding images or scoring predictions.

This module intentionally does not choose a temporal flow index, flow unit,
visibility-mask convention, pose correction, or camera rectification policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np


def decode_packed12_normalized(rgb):
    """Reproduce upstream 12-bit RGB unpacking; physical flow units are unknown."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError('packed flow must be uint8 HxWx3 in RGB order')
    c = rgb.astype(np.uint16)
    x = (c[..., 0] | (c[..., 1] << 8)) & 0xFFF
    y = ((c[..., 1] >> 4) | (c[..., 2] << 4)) & 0xFFF
    packed = np.stack((x, y), axis=-1).astype(np.float32)
    return (packed + np.float32(0.5)) / np.float32(4095.0) * np.float32(2.0) - np.float32(1.0)


def upstream_last32_indices(frame_count):
    """Indices of the audited upstream entry when num_frames=-1, not a paper claim."""
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError('frame_count must be a positive integer')
    return list(range(max(0, frame_count - 32), frame_count))


def pose_diagnostics(value):
    if value is None:
        return {'status': 'missing', 'strict_valid': False}
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return {'status': 'invalid_type', 'strict_valid': False}
    if pose.shape != (4, 4):
        return {'status': 'invalid_shape', 'strict_valid': False}
    if not np.isfinite(pose).all():
        return {'status': 'nonfinite', 'strict_valid': False}
    rot = pose[:3, :3]
    with np.errstate(over='ignore', invalid='ignore'):
        gram = rot.T @ rot
        det = float(np.linalg.det(rot))
    if not np.isfinite(gram).all() or not np.isfinite(det):
        return {'status': 'numeric_overflow', 'strict_valid': False}
    error = float(np.max(np.abs(gram - np.eye(3))))
    good = (np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)
            and np.allclose(gram, np.eye(3), atol=1e-4)
            and np.isclose(det, 1, atol=1e-4))
    return {'status': 'finite_passes_strict_check' if good else 'finite_fails_strict_check',
            'strict_valid': bool(good), 'orthogonality_error_max': error,
            'determinant': det, 'identity': bool(np.allclose(pose, np.eye(4), atol=1e-8))}


def _at(values, index):
    return values[index] if isinstance(values, list) and index < len(values) else None


def camera_diagnostics(sequence, index):
    k = _at(sequence.get('K'), index)
    if k is not None:
        try:
            a = np.asarray(k, dtype=np.float64)
            good = (a.shape == (3, 3) and np.isfinite(a).all()
                    and a[0, 0] > 0 and a[1, 1] > 0
                    and np.allclose(a[2], [0, 0, 1], atol=1e-8))
            if good:
                det = np.linalg.det(a)
                good = np.isfinite(det) and abs(det) > 1e-12
        except (TypeError, ValueError, np.linalg.LinAlgError):
            good = False
        return {'model': 'K', 'status': 'pinhole_candidate' if good else 'invalid_K',
                'pixel_coordinate_convention_verified': False}
    model = _at(sequence.get('camera_model'), index)
    params = _at(sequence.get('camera_params'), index)
    if model == 'OPENCV' and params is not None:
        try:
            a = np.asarray(params, dtype=np.float64)
            if a.shape != (16,) or not np.isfinite(a).all():
                return {'model': model, 'status': 'invalid_camera_params'}
            magnitude = float(np.max(np.abs(a[4:])))
            return {'model': model,
                    'status': 'distortion_requires_handling' if magnitude > 1e-6 else 'zero_distortion_metadata_only',
                    'max_abs_distortion_parameter': magnitude,
                    'image_rectification_verified': False}
        except (TypeError, ValueError):
            return {'model': model, 'status': 'invalid_camera_params'}
    return {'model': model, 'status': 'camera_model_unavailable_or_unsupported'}


def _h5_text(h5, key):
    value = h5[key][()]
    if isinstance(value, str):
        return value
    raw = bytes(value) if isinstance(value, (bytes, np.bytes_, np.void)) else value.tobytes()
    return raw.decode('utf-8-sig')


def parse_split(text):
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = line.strip().rsplit(None, 1)
        if len(row) != 2 or not row[1].isdigit():
            raise ValueError(f'invalid split entry: {line!r}')
        result.append((row[0], int(row[1])))
    if len({name for name, _ in result}) != len(result):
        raise ValueError('duplicate split sequence')
    return result


def resolve_sequence(sequences, name):
    if name in sequences:
        return name
    matches = [k for k in sequences if PurePosixPath(k).name == PurePosixPath(name).name]
    if len(matches) != 1:
        raise ValueError(f'sequence must resolve uniquely: {name!r}')
    return matches[0]


def _candidates(sequence, field, previous_index):
    values = sequence.get(field)
    if not isinstance(values, list):
        return {'record_count': None, 'candidate_records': []}
    return {'record_count': len(values), 'candidate_records': [
        {'record_index': i, 'value': values[i]} for i in (previous_index, previous_index + 1) if i < len(values)
    ]}


def inspect_pair(sequence, previous_index):
    images = sequence.get('image')
    if not isinstance(images, list):
        raise ValueError('sequence image list missing')
    n = len(images)
    if isinstance(previous_index, bool) or not isinstance(previous_index, int) or not 0 <= previous_index < n - 1:
        raise ValueError('previous_index must identify a consecutive in-sequence pair')
    current_index = previous_index + 1
    previous_pose = pose_diagnostics(_at(sequence.get('cam2w'), previous_index))
    current_pose = pose_diagnostics(_at(sequence.get('cam2w'), current_index))
    return {
        'scope': 'METADATA_ONLY_NO_FRAME_DECODE_OR_SCORING',
        'frame_count': n,
        'previous_index': previous_index,
        'current_index': current_index,
        'image_paths': [images[previous_index], images[current_index]],
        'depth_paths': [_at(sequence.get('depth'), previous_index), _at(sequence.get('depth'), current_index)],
        'upstream_eval_last32_indices': upstream_last32_indices(n),
        'previous_pose': previous_pose,
        'current_pose': current_pose,
        'previous_camera': camera_diagnostics(sequence, previous_index),
        'current_camera': camera_diagnostics(sequence, current_index),
        'forward_flow_candidates_unverified': _candidates(sequence, 'flow_fwd', previous_index),
        'forward_mask_candidates_unverified': _candidates(sequence, 'flow_fwd_mask', previous_index),
        'backward_flow_metadata_present': isinstance(sequence.get('flow_bwd'), list),
        'backward_mask_metadata_present': isinstance(sequence.get('flow_bwd_mask'), list),
        'ready_for_temporal_scoring': False,
        'unresolved': [
            'Flow record indexing, direction, physical units and mask semantics require verification.',
            'Backward flow and an independent valid/occlusion mask must be supplied and verified.',
            'Pose provenance, timestamp pairing and camera pixel/rectification semantics require verification.',
            'Predictions, common GT mask and the authors temporal tau5 implementation are not supplied here.',
        ],
    }


def load_metadata(hdf5_path, sequence_key, split_key):
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError('h5py is required to read HDF5 metadata') from exc
    with h5py.File(hdf5_path, 'r') as h5:
        sequences = json.loads(_h5_text(h5, sequence_key))
        split = parse_split(_h5_text(h5, split_key))
    if not isinstance(sequences, dict):
        raise ValueError('sequence metadata must be a JSON object')
    return sequences, split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--sequences-key', default='sequences.json')
    parser.add_argument('--split-key', default='train.txt', help='metadata split; train.txt by default')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--list-sequences', action='store_true')
    group.add_argument('--sequence', help='sequence name in the requested split')
    parser.add_argument('--prev-index', type=int, default=0)
    args = parser.parse_args(argv)
    path = args.hdf5.expanduser().resolve(strict=True)
    sequences, split = load_metadata(path, args.sequences_key, args.split_key)
    if args.list_sequences:
        rows = []
        for name, declared in split:
            resolved = resolve_sequence(sequences, name)
            images = sequences[resolved].get('image', [])
            rows.append({'split_name': name, 'resolved_key': resolved, 'declared_frames': declared,
                         'metadata_image_count': len(images)})
        result = {'scope': 'SPLIT_METADATA_ONLY_NO_SCORING', 'hdf5': str(path),
                  'split_key': args.split_key, 'sequences': rows}
    else:
        selected = [name for name, _ in split if name == args.sequence]
        if not selected:
            raise ValueError('--sequence must match an entry of the selected split exactly')
        key = resolve_sequence(sequences, selected[0])
        result = inspect_pair(sequences[key], args.prev_index)
        result.update({'hdf5': str(path), 'split_key': args.split_key,
                       'split_name': selected[0], 'resolved_key': key})
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
