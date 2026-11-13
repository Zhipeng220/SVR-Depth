#!/usr/bin/env python3
"""Bounded upstream A smoke: development splits only, never a paper result.

Runs the bundled upstream evaluator with runtime-only dataset caps. No upstream
source is edited. A fresh output directory is required and the manifest records
the altered splits and frame/sequence limits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace


SPLITS = {'Bonn':'train.txt', 'TUM':'train.txt', 'ScanNetVid':'train.txt', 'Sintel':'training.txt'}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _child(args):
    import torch
    import velodepth.datasets as datasets
    if not torch.cuda.is_available():
        raise RuntimeError('The upstream development smoke requires CUDA')
    for name, split in SPLITS.items():
        original = getattr(datasets, name)

        def make_development_class(base, chosen_split):
            class DevelopmentOnly(base):
                test_split = chosen_split

                def __init__(self, *pos, **kw):
                    super().__init__(*pos, **kw)
                    count = min(args.max_sequences, len(self.dataset))
                    self.dataset = [self.dataset[i] for i in range(count)]
                    self.sequences = [self.sequences[i] for i in range(count)]

                def get_test_idxs(self, count, keyframe_idx):
                    count_selected = min(args.max_frames, count)
                    return list(range(count - count_selected, count)), None

            return DevelopmentOnly

        setattr(datasets, name, make_development_class(original, split))
    upstream = runpy.run_path(str(args.official_repo / 'scripts/eval.py'), run_name='m01_upstream_development')
    worker_args = SimpleNamespace(distributed=False, rank=0, world_size=1, local_rank=0,
                                  master_port=29400, camera_gt=args.camera_gt,
                                  save_path=str(args.output_dir / 'upstream_metrics.json'))
    upstream['main_worker'](worker_args, str(args.config_file))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-repo', type=Path, required=True)
    parser.add_argument('--dataroot', type=Path, required=True)
    parser.add_argument('--config-file', type=Path, required=True)
    parser.add_argument('--knn-lib', type=Path, required=True)
    parser.add_argument('--hf-home', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=2)
    parser.add_argument('--max-sequences', type=int, default=1)
    parser.add_argument('--camera-gt', action='store_true')
    parser.add_argument('--_child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    for key in ('official_repo','dataroot','config_file','knn_lib','hf_home'):
        setattr(args, key, getattr(args, key).expanduser().resolve(strict=True))
    args.output_dir = args.output_dir.expanduser().resolve()
    if not 1 <= args.max_frames <= 8 or not 1 <= args.max_sequences <= 2:
        parser.error('development caps must be 1..8 frames and 1..2 sequences')
    if not (args.official_repo/'scripts/eval.py').is_file() or not any(args.knn_lib.glob('KNN*.so')):
        parser.error('upstream evaluator and isolated KNN extension are required')
    config = json.loads(args.config_file.read_text(encoding='utf-8'))
    if set(config['data']['val_datasets']) != set(SPLITS):
        parser.error('config must contain exactly ScanNetVid, Bonn, TUM and Sintel')
    if args._child:
        return _child(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(DATAROOT=str(args.dataroot), HF_HOME=str(args.hf_home), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled')
    env['PYTHONPATH'] = os.pathsep.join((str(args.official_repo), str(args.knn_lib), env.get('PYTHONPATH', '')))
    cmd = [sys.executable, str(Path(__file__).resolve()), '--_child',
           '--official-repo',str(args.official_repo), '--dataroot',str(args.dataroot),
           '--config-file',str(args.config_file), '--knn-lib',str(args.knn_lib),
           '--hf-home',str(args.hf_home), '--output-dir',str(args.output_dir),
           '--max-frames',str(args.max_frames), '--max-sequences',str(args.max_sequences)]
    if args.camera_gt:
        cmd.append('--camera-gt')
    manifest = {'scope':'development_smoke_not_paper_result', 'status':'running',
                'splits':SPLITS, 'max_frames_per_sequence':args.max_frames,
                'max_sequences_per_dataset':args.max_sequences, 'camera_gt':args.camera_gt,
                'frame_selection':'last N frames; runtime-only override',
                'official_repo':str(args.official_repo), 'dataroot':str(args.dataroot),
                'config_sha256':_sha(args.config_file),
                'upstream_eval_sha256':_sha(args.official_repo/'scripts/eval.py'),
                'knn_lib':str(args.knn_lib), 'hf_home':str(args.hf_home)}
    manifest_path = args.output_dir/'run_identity.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with (args.output_dir/'official_dev.log').open('x', encoding='utf-8') as log:
        completed = subprocess.run(cmd, cwd=args.official_repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(status='complete' if completed.returncode == 0 else 'failed',
                    process_returncode=completed.returncode)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Official A development smoke {manifest["status"]}: {args.output_dir}')
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
