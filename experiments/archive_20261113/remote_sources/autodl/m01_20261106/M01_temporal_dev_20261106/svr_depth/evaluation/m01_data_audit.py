"""Read-only HDF5 metadata/pair inventory; makes no flow-index assumptions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from svr_depth.evaluation.m01_v31.hdf5_preflight import (
    _h5_text, parse_split, camera_diagnostics, pose_diagnostics,
)
from svr_depth.evaluation.m01_runner import load_config


def _at(sample, field, index):
    values = sample.get(field)
    return values[index] if isinstance(values, list) and 0 <= index < len(values) else None


def inspect_hdf5(spec, *, max_frames=None):
    """Inventory exact split and adjacent candidates; no images or metrics read."""
    import h5py
    if spec['dataset'] == 'ScanNet' and not spec.get('sequences_key'):
        raise ValueError('ScanNet requires an explicit sequences_key; ScanNetVid uses sequences_.json')
    with h5py.File(spec['hdf5'], 'r') as h5:
        split = dict(parse_split(_h5_text(h5, spec['split_key'])))
        sequence_key = spec.get('sequences_key', 'sequences.json')
        samples = json.loads(_h5_text(h5, sequence_key))
        name = spec['sequence']
        if name not in split or name not in samples:
            raise ValueError(f'{name!r} must exist exactly in split and metadata')
        sample = samples[name]
        images = sample.get('image')
        if not isinstance(images, list) or len(images) != split[name] or not images:
            raise ValueError(f'{name!r}: image count does not match the split')
        n = min(len(images), max_frames) if max_frames is not None else len(images)
        fields = ('image', 'depth', 'flow_fwd', 'flow_fwd_mask', 'flow_bwd',
                  'flow_bwd_mask', 'flow', 'flow_mask', 'cam2w', 'K',
                  'camera_model', 'camera_params', 'timestamp')
        lengths = {k: len(sample[k]) for k in fields if isinstance(sample.get(k), list)}
        pairs = []
        for index in range(n - 1):
            candidate_fields = ('flow_fwd', 'flow_fwd_mask', 'flow_bwd', 'flow_bwd_mask', 'flow', 'flow_mask')
            candidates = {field:[{'record_index':j, 'reference':_at(sample, field, j)}
                                 for j in (index, index + 1) if _at(sample, field, j) is not None]
                          for field in candidate_fields}
            pairs.append({'prev_index':index, 'curr_index':index + 1,
                          'prev_image_ref':images[index], 'curr_image_ref':images[index + 1],
                          'prev_depth_ref':_at(sample, 'depth', index),
                          'curr_depth_ref':_at(sample, 'depth', index + 1),
                          'prev_pose':pose_diagnostics(_at(sample, 'cam2w', index)),
                          'curr_pose':pose_diagnostics(_at(sample, 'cam2w', index + 1)),
                          'prev_camera':camera_diagnostics(sample, index),
                          'curr_camera':camera_diagnostics(sample, index + 1),
                          'flow_candidate_records_unverified':candidates,
                          'backward_flow_metadata_present':isinstance(sample.get('flow_bwd'), list),
                          'pairing_status':'unverified_metadata_candidates_only'})
        return {'dataset':spec['dataset'], 'sequence':name, 'scene':spec['scene'],
                'hdf5':str(Path(spec['hdf5']).resolve()), 'split_key':spec['split_key'],
                'sequences_key':sequence_key, 'declared_frames':split[name],
                'audited_frames':n, 'field_lengths':lengths, 'pairs':pairs}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output', type=Path, required=True, help='new JSON file; never overwritten')
    p.add_argument('--max-frames', type=int, help='limit metadata to first N frames per sequence')
    args = p.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        p.error('--max-frames must be positive')
    cfg = load_config(args.config)
    reports, failures = [], []
    for spec in cfg['sequences']:
        if not spec.get('hdf5'):
            failures.append({'dataset':spec['dataset'], 'sequence':spec['sequence'], 'reason':'non_hdf5_source_not_audited'})
            continue
        try:
            reports.append(inspect_hdf5(spec, max_frames=args.max_frames))
        except Exception as exc:
            failures.append({'dataset':spec['dataset'], 'sequence':spec['sequence'],
                             'reason':f'{type(exc).__name__}: {exc}'})
    result = {'scope':'metadata_only_no_frame_decode_inference_or_scoring',
              'flow_pairing_verified':False, 'camera_geometry_verified':False,
              'pose_provenance_verified':False, 'datasets':reports, 'failures':failures}
    with args.output.expanduser().open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(f'Wrote {args.output}; {len(reports)} sources, {len(failures)} failures')
    return 2 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
