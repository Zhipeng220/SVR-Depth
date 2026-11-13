"""Read only VeloDepth HDF5 split metadata; never decode or score frames."""
import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath

import h5py
import numpy as np


DATASETS = (
    ('Bonn', 'sequences.json', 'val.txt'),
    ('TUM', 'sequences.json', 'val.txt'),
    ('ScanNet', 'sequences_.json', 'val.txt'),
    ('Sintel', 'sequences.json', 'training.txt'),
)


def read_text(h5, name):
    value = h5[name][()]
    raw = bytes(value) if isinstance(value, (bytes, np.bytes_)) else value.tobytes()
    return raw.decode('utf-8-sig')


def resolve_sequence(name, sequences):
    if name in sequences:
        return sequences[name]
    matches = [value for key, value in sequences.items()
               if PurePosixPath(key).name == PurePosixPath(name).name]
    if len(matches) != 1:
        raise ValueError(f'Split entry cannot be resolved uniquely: {name}')
    return matches[0]


def inspect_dataset(root, name, sequence_file, split_file):
    counts = Counter()
    camera_models = Counter()
    finite_ortho_errors = []
    samples = []
    with h5py.File(root / f'{name}.hdf5', 'r') as h5:
        sequences = json.loads(read_text(h5, sequence_file))
        lines = [line.rsplit(None, 1) for line in read_text(h5, split_file).splitlines() if line.strip()]
        for line in lines:
            if len(line) != 2 or not line[1].isdigit():
                raise ValueError(f'Invalid split row: {line}')
            seq = resolve_sequence(line[0], sequences)
            n = len(seq['image'])
            counts['sequences'] += 1
            counts['frames'] += n
            counts['declared_frames'] += int(line[1])
            for field in ('flow_fwd', 'flow_bwd', 'flow_fwd_mask', 'flow_bwd_mask'):
                counts[f'sequences_with_{field}'] += int(field in seq)
            poses = seq.get('cam2w', [])
            counts['pose_missing'] += max(0, n - len(poses))
            for pose in poses[:n]:
                if pose is None:
                    counts['pose_missing'] += 1
                    continue
                try:
                    matrix = np.asarray(pose, dtype=float)
                except (TypeError, ValueError):
                    counts['pose_bad_shape_or_type'] += 1
                    continue
                if matrix.shape != (4, 4):
                    counts['pose_bad_shape_or_type'] += 1
                    continue
                if not np.isfinite(matrix).all():
                    counts['pose_nonfinite'] += 1
                    continue
                rotation = matrix[:3, :3]
                ortho = rotation.T @ rotation
                finite_ortho_errors.append(float(np.max(np.abs(ortho - np.eye(3)))))
                valid = (np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
                         and np.allclose(ortho, np.eye(3), atol=1e-4)
                         and np.isclose(np.linalg.det(rotation), 1, atol=1e-4))
                counts['pose_finite_fails_strict_rigid_check'] += int(not valid)
                counts['pose_identity'] += int(np.allclose(matrix, np.eye(4), atol=1e-8))
            camera_models.update(seq.get('camera_model', [])[:n])
            for model, params in zip(seq.get('camera_model', [])[:n], seq.get('camera_params', [])[:n]):
                if model != 'OPENCV' or params is None:
                    continue
                arr = np.asarray(params, dtype=float)
                if arr.shape == (16,) and np.isfinite(arr).all() and np.max(np.abs(arr[4:])) > 1e-6:
                    counts['opencv_nonzero_distortion_metadata_frames'] += 1
            if len(samples) < 4:
                for field in ('flow_fwd', 'flow_fwd_mask'):
                    values = seq.get(field, [])
                    if values and isinstance(values[0], str):
                        samples.append({'field': field, 'node_exists': values[0].lstrip('/') in h5})
        counts['pose_strict_rejected_total'] = sum(counts[x] for x in (
            'pose_missing', 'pose_bad_shape_or_type', 'pose_nonfinite', 'pose_finite_fails_strict_rigid_check'))
    result = {key: counts[key] for key in (
        'sequences', 'frames', 'declared_frames', 'sequences_with_flow_fwd',
        'sequences_with_flow_bwd', 'sequences_with_flow_fwd_mask', 'sequences_with_flow_bwd_mask',
        'pose_missing', 'pose_bad_shape_or_type', 'pose_nonfinite',
        'pose_finite_fails_strict_rigid_check', 'pose_strict_rejected_total',
        'pose_identity', 'opencv_nonzero_distortion_metadata_frames')}
    result['camera_models'] = dict(camera_models)
    result['max_finite_rotation_orthogonality_error'] = max(finite_ortho_errors, default=None)
    result['sample_flow_nodes'] = samples
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve(strict=True)
    result = {'scope': 'SPLIT_METADATA_ONLY_NO_SCORING', 'data_root': str(root),
              'pose_check': 'provisional evaluator rigid checks: atol 1e-4 (rotation), 1e-6 (bottom row)',
              'limitations': ['Pose provenance and timestamp pairing are not established.',
                              'Field presence does not establish flow units, direction or full node coverage.',
                              'Nonzero distortion parameters do not establish whether images were rectified.'],
              'datasets': {name: inspect_dataset(root, name, seq, split) for name, seq, split in DATASETS}}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
