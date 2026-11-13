#!/usr/bin/env python3
"""Fresh P_multi training for the M04/M05 development protocol.

The backbone is pretrained; the repair head is newly initialized. This runner
does not require old P_multi/C3 checkpoints. Historical C3 training primitives
are reused without changing their source files.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments' / 'p1')]
SEQUENCES = ('crowd2', 'crowd3', 'balloon2')
SEEDS = (20261025, 20261101, 20261102)
BASE_SHA = '615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e'
STEPS = 1000

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def balanced_order(seed, steps=STEPS):
    rng = random.Random(seed)
    order = []
    while len(order) < steps:
        block = list(range(24))
        rng.shuffle(block)
        order.extend(block)
    return order[:steps]

class SceneReader:
    def __init__(self, readers, slots):
        self.readers, self.slots = readers, slots
    def load(self, slot):
        seq, cycle = self.slots[slot]
        out = dict(self.readers[seq].load(cycle))
        out['cycle'] = slot
        out['y04_sequence'], out['y04_local_cycle'] = seq, cycle
        return out

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--pairs-dir', default=str(ROOT / 'outputs/gt4a2_exp1'))
    p.add_argument('--out-dir', required=True, help='New, seed-specific output directory')
    p.add_argument('--smoke', action='store_true', help='3 updates; does not save a formal checkpoint')
    args = p.parse_args()
    snapshot, pairs_dir, out = (Path(x).expanduser().resolve() for x in (args.snapshot, args.pairs_dir, args.out_dir))
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {out}')
    if not (snapshot / 'config.json').is_file() or sha(snapshot / 'model.safetensors') != BASE_SHA:
        raise RuntimeError('Missing snapshot or wrong pretrained model hash')
    # The untouched historical helpers import torch only after argument parsing.
    import numpy as np
    import torch
    import gt2_fixed_clip_repair_train as gt2
    import gt4a2_c3_closed_loop as c3
    import gt4a2_exp1_train_eval as exp1
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for training')
    paths = {s: pairs_dir / f'gt4a2_exp1_{s}_pairs_used.csv' for s in SEQUENCES}
    manifests = {s: exp1.read_manifest(paths[s], training=False) for s in SEQUENCES}
    ncycles = 1 if args.smoke else 8
    for seq, rows in manifests.items():
        if len(rows) < ncycles * 32:
            raise RuntimeError(f'{seq} requires {ncycles * 32} RGB frames')
    # New common initialization is deterministic and is saved with its hash.
    torch.manual_seed(20261024)
    model = gt2.load_model(snapshot, 'pre_readout', torch.device('cuda'))
    init_state = c3.initial_state(model.pixel_decoder.persistent_feature_repair)
    init_sha = exp1.state_sha256(init_state)
    del model
    torch.cuda.empty_cache()
    out.mkdir(parents=True, exist_ok=False)
    identity = {'seed': args.seed, 'smoke': args.smoke, 'backbone_sha256': BASE_SHA,
                'initial_head_sha256': init_sha, 'initialization_seed': 20261024,
                'pair_hashes': {s: sha(paths[s]) for s in SEQUENCES},
                'protocol': 'fresh initialization; P_multi; 32-frame cycles; pre_readout'}
    (out / 'run_identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
    readers = {}
    for seq in SEQUENCES:
        cache = c3.build_cycle_cache({'snapshot': snapshot, 'crowd2': paths[seq]}, manifests[seq],
                                     out / 'cache' / seq, torch.device('cuda'), limit=ncycles)
        readers[seq] = c3.CycleReader(cache, max_items=2)
    slots = [(s, i) for s in SEQUENCES for i in range(ncycles)]
    order = [0, 1, 2] if args.smoke else balanced_order(args.seed)
    reader = SceneReader(readers, slots)
    train_dir = out / 'training' / 'P_multi'
    train_dir.mkdir(parents=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = gt2.load_model(snapshot, 'pre_readout', torch.device('cuda'))
    c3.train_mode('C', model, reader, init_state, order, train_dir, torch.device('cuda'),
                  snapshot, paths['crowd2'], len(order), resume=False, smoke=args.smoke)
    if args.smoke:
        print('[PASS] smoke only; no formal checkpoint saved', flush=True)
        return
    cp = train_dir / 'gt4a2_c3_C_final_step1000.pt'
    payload = torch.load(cp, map_location='cpu', weights_only=False)
    # C3's single-scene metadata does not describe these multi-scene slots.
    payload.pop('training_original_rgb_indices', None)
    payload.update({'experiment': 'fresh P_multi', 'y04_label': 'Y04_P_multi',
        'y04_kind': 'persistent', 'training_sequences': list(SEQUENCES),
        'training_cycle_slots': [list(x) for x in slots], 'optimizer_steps': STEPS,
        'y04_outer_seed': args.seed, 'y04_sample_order_sha256': hashlib.sha256(','.join(map(str, order)).encode()).hexdigest(),
        'y04_initial_head_sha256': init_sha, 'fresh_initialization': True,
        'training_pair_sha256': identity['pair_hashes']})
    tmp = cp.with_suffix('.fresh.tmp')
    torch.save(payload, tmp); tmp.replace(cp)
    summary_path = train_dir / 'gt4a2_c3_C_training_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary.update({'checkpoint_sha256': sha(cp), 'fresh_initialization': True,
                    'y04_outer_seed': args.seed, 'pair_hashes': identity['pair_hashes']})
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (out / 'checkpoint_manifest_row.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['seed','checkpoint_path','expected_sha256','training_summary_path','training_log_path','notes'])
        w.writeheader(); w.writerow({'seed': args.seed, 'checkpoint_path': str(cp), 'expected_sha256': sha(cp),
            'training_summary_path': str(summary_path), 'training_log_path': '', 'notes': 'fresh P_multi; common initialization seed 20261024'})
    print(f'[PASS] fresh P_multi checkpoint: {cp}', flush=True)

if __name__ == '__main__':
    main()
