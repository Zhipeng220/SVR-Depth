#!/usr/bin/env python3
"""Train independent output-only controls on the CrossSource-v2 protocol.

The frozen backbone always propagates its native state. Adapters affect only
the supervised readout; their results are never written back to that state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import train_crosssource_v2 as common
from svr_depth.datasets.crosssource import CrossSourcePool, DatasetBalancedWindowSampler
from svr_depth.streaming.m01_engine import _imports


def save_atomic(obj, path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def readout(rt, model, adapter, method, window, frame, radius, features):
    rays = window['keyframe_state']['rays_map']
    if method == 'O_multi':
        adapted = adapter([x.detach() for x in features])
        output_radius = rt.models.readout_radius(model, adapted)
        return rt.gt2.make_z_depth(output_radius, rays, window['padded_shape'],
                                   window['paddings'], model.interpolation_mode)
    native = rt.gt2.make_z_depth(radius, rays, window['padded_shape'],
                                 window['paddings'], model.interpolation_mode)
    rgb = rt.models.rgb_for_output(frame, window)
    return adapter(native.detach(), rgb.detach())


def window_loss(rt, model, adapter, method, window, optimizer=None, scaler=None, weight=1.0):
    training = optimizer is not None
    if len(window['frames']) != 20 or len(window['statics']) != 19:
        raise RuntimeError('Expected a 20-frame window and 19 transitions')
    radius, confidence, features = common.move_state(window['keyframe_state'], next(adapter.parameters()).device)
    prev = window['frames'][0]
    count = int(window['n_supervised'])
    if count < 1:
        raise RuntimeError('Window has no supervised propagated frames')
    if training:
        optimizer.zero_grad(set_to_none=True)
    values = []
    for offset in range(1, 20):
        frame = window['frames'][offset]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            radius, confidence, features = rt.exp1.one_fast_step(
                model, prev, frame, window['statics'][offset - 1], radius, confidence, features)
        if frame['gt'] is not None:
            with torch.set_grad_enabled(training), torch.autocast('cuda', dtype=torch.float16):
                pred = readout(rt, model, adapter, method, window, frame, radius, features)
                loss, _ = rt.gt2.masked_log_depth_loss(pred, frame['gt'])
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite loss')
            if training:
                scaler.scale(loss * weight / count).backward()
            values.append(float(loss.detach().item()))
        prev = frame
    if len(values) != count:
        raise RuntimeError('Supervision count mismatch')
    if training:
        scaler.unscale_(optimizer)
        grads = [p.grad for p in adapter.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads) or not any(torch.count_nonzero(g).item() for g in grads):
            raise RuntimeError('Missing, zero or non-finite adapter gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0).item())
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            raise RuntimeError('Optimizer update skipped due to overflow')
    else:
        norm = None
    return {'mean_log_depth_loss': float(np.mean(values)), 'supervised_frames': count,
            'gradient_norm_before_clip': norm}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=('O_fast', 'O_multi'), required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--init-seed', type=int, default=20261103)
    p.add_argument('--expected-base-sha256', default=common.EXPECTED_BASE_SHA256)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--val-every', type=int, default=100)
    p.add_argument('--val-windows-per-dataset', type=int, default=8)
    p.add_argument('--sample-order', help='Existing P_multi sample_order.json to replay for paired training')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if min(a.steps, a.val_every, a.val_windows_per_dataset) < 1 or (a.smoke and a.resume):
        p.error('Counts must be positive; smoke cannot resume')
    if a.smoke:
        a.steps = 1
    config_path = Path(a.config).expanduser().resolve(strict=True)
    config = json.loads(config_path.read_text(encoding='utf-8'))
    tc = config.get('training', {})
    if int(tc.get('window_frames', 20)) != 20:
        raise ValueError('This protocol requires 20-frame windows')
    weights = tc['dataset_weights']
    loss_weights = tc.get('dataset_loss_weights', {x: 1.0 for x in weights})
    if set(loss_weights) != set(weights) or any(not math.isfinite(float(x)) or float(x) <= 0 for x in loss_weights.values()):
        raise ValueError('Invalid dataset loss weights')
    snapshot = Path(a.snapshot).expanduser().resolve(strict=True)
    base_sha = common.sha256_file(snapshot / 'model.safetensors')
    if base_sha.lower() != a.expected_base_sha256.lower():
        raise ValueError('Backbone SHA256 mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no training was run')
    train = CrossSourcePool.from_config(config_path, split='train', verify_files=True)
    val = CrossSourcePool.from_config(config_path, split='val', verify_files=True)
    train.validate_disjoint(val)
    sampler = DatasetBalancedWindowSampler(train, dataset_weights=weights, window_frames=20,
        min_supervised_frames=int(tc.get('min_supervised_frames', 1)), seed=a.seed)
    if a.sample_order:
        rows = json.loads(Path(a.sample_order).read_text(encoding='utf-8'))
        if len(rows) < a.steps:
            raise ValueError('Provided sample order is shorter than requested training')
        rows = rows[:a.steps]
    else:
        rows = [common.ref_to_dict(ref) for ref in sampler.sample_many(a.steps)]
    refs = [common.dict_to_ref(row) for row in rows]
    for ref in refs:
        adapter = train.adapter(ref.dataset_id, ref.sequence_id)
        if (Path(ref.manifest_path).resolve() != adapter.manifest.path.resolve() or ref.scene_id != adapter.manifest.scene_id
                or ref.length != 20 or ref.start_index < 0 or ref.start_index + 20 > len(adapter.manifest.frames)
                or not any(f.depth_path is not None for f in adapter.manifest.frames[ref.start_index + 1:ref.start_index + 20])):
            raise ValueError(f'Invalid training window: {ref}')
    val_refs = common.build_val_refs(val, 20, a.val_windows_per_dataset, a.seed + 1)
    if set(r.dataset_id for r in val_refs) != set(weights):
        raise ValueError('Validation must include every training dataset')
    manifest_paths = sorted({str(x.manifest.path) for pool in (train, val) for x in pool.sequences.values()})
    source_paths = [Path(__file__), Path(common.__file__), ROOT / 'svr_depth/datasets/crosssource.py',
                    ROOT / 'svr_depth/datasets/native_tartanair.py', ROOT / 'svr_depth/datasets/kitti_sceneflow.py',
                    ROOT / 'svr_depth/streaming/m01_engine.py', ROOT / 'velodepth/models/persistent_feature_repair.py',
                    ROOT / 'svr_depth/_compat/output_smoke/y08y09_models.py',
                    ROOT / 'svr_depth/_compat/output_smoke/y08y09_common.py',
                    common.HELPERS / 'gt2_fixed_clip_repair_train.py', common.HELPERS / 'gt4a2_exp1_train_eval.py']
    identity = {'protocol': 'SVR-Depth-CrossSource-v2-output-only', 'method': a.method,
        'seed': a.seed, 'initialization_seed': a.init_seed, 'smoke': a.smoke, 'steps': a.steps,
        'window_frames': 20, 'dataset_weights': weights, 'dataset_loss_weights': loss_weights,
        'val_every': a.val_every, 'val_windows_per_dataset': a.val_windows_per_dataset,
        'snapshot_sha256': base_sha, 'config_sha256': common.sha256_file(config_path),
        'manifest_sha256': {x: common.sha256_file(Path(x)) for x in manifest_paths},
        'source_sha256': {str(x.relative_to(ROOT)): common.sha256_file(x) for x in source_paths},
        'sample_order_sha256': hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        'validation_order_sha256': hashlib.sha256(json.dumps([common.ref_to_dict(x) for x in val_refs], sort_keys=True).encode()).hexdigest()}
    out = Path(a.out_dir).expanduser().resolve()
    if a.resume:
        if json.loads((out / 'run_identity.json').read_text(encoding='utf-8')) != identity:
            raise RuntimeError('Resume identity differs from the original run')
    else:
        out.mkdir(parents=True, exist_ok=False)
        (out / 'run_identity.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
        (out / 'sample_order.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        (out / 'validation_order.json').write_text(json.dumps([common.ref_to_dict(x) for x in val_refs], indent=2), encoding='utf-8')
    rt = _imports()
    device = torch.device('cuda')
    torch.manual_seed(a.init_seed)
    # Match P_multi's seeded head initialization before detaching the head.
    model = rt.gt2.load_model(snapshot, 'pre_readout' if a.method == 'O_multi' else 'off', device)
    if a.method == 'O_multi':
        adapter = model.pixel_decoder.persistent_feature_repair
        model.pixel_decoder.persistent_feature_repair = None
        model.pixel_decoder.feature_repair_mode = 'off'
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    torch.manual_seed(a.init_seed)
    if a.method == 'O_fast':
        adapter = rt.models.FastDepthAdapter().to(device)
    else:
        for param in adapter.parameters():
            param.requires_grad_(True)
    adapter.train()
    initial_sha = common.state_sha256(adapter.state_dict())
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    start, best, curve = 0, float('inf'), []
    if a.resume:
        ckpt = torch.load(out / 'recovery.pt', map_location='cpu', weights_only=True)
        if ckpt['identity'] != identity:
            raise RuntimeError('Recovery identity mismatch')
        adapter.load_state_dict(ckpt['adapter_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state']); scaler.load_state_dict(ckpt['scaler_state'])
        start, best, curve = int(ckpt['step']), float(ckpt['best_val_loss']), ckpt['curve']
        torch.set_rng_state(ckpt['torch_rng_state']); torch.cuda.set_rng_state_all(ckpt['cuda_rng_state'])
    for step in range(start + 1, a.steps + 1):
        ref = refs[step - 1]
        window = common.prepare_window(model, train, ref, device, rt.gt2, rt.exp1, rt.cv2)
        metrics = window_loss(rt, model, adapter, a.method, window, optimizer, scaler, float(loss_weights[ref.dataset_id]))
        row = {'step': step, **common.ref_to_dict(ref), **metrics}
        if step % a.val_every == 0 or step == a.steps:
            adapter.eval(); values = {}
            for vref in val_refs:
                window = common.prepare_window(model, val, vref, device, rt.gt2, rt.exp1, rt.cv2)
                result = window_loss(rt, model, adapter, a.method, window)
                values.setdefault(vref.dataset_id, []).extend([result['mean_log_depth_loss']] * result['supervised_frames'])
            scores = {k: float(np.mean(v)) for k, v in values.items()}
            score = float(np.mean(list(scores.values())))
            if not math.isfinite(score):
                raise RuntimeError('Non-finite validation score')
            row['val_macro_loss'], row['val_per_dataset'] = score, json.dumps(scores, sort_keys=True)
            if score < best:
                best = score
                if not a.smoke:
                    save_atomic({'identity': identity, 'step': step, 'adapter_state_dict': {k: v.detach().cpu() for k, v in adapter.state_dict().items()}, 'val': scores, 'initial_head_sha256': initial_sha}, out / 'best_checkpoint.pt')
            adapter.train()
        curve.append(row)
        common.write_curve(out / 'training_curve.csv', curve)
        if not a.smoke and (step % 25 == 0 or step == a.steps):
            save_atomic({'identity': identity, 'step': step, 'adapter_state_dict': {k: v.detach().cpu() for k, v in adapter.state_dict().items()}, 'optimizer_state': optimizer.state_dict(), 'scaler_state': scaler.state_dict(), 'best_val_loss': best, 'curve': curve, 'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all()}, out / 'recovery.pt')
        print(f"[{a.method}] {step}/{a.steps} {ref.dataset_id} loss={metrics['mean_log_depth_loss']:.6f}", flush=True)
    if not a.smoke:
        save_atomic({'identity': identity, 'step': a.steps, 'adapter_state_dict': {k: v.detach().cpu() for k, v in adapter.state_dict().items()}, 'best_val_loss': best, 'initial_head_sha256': initial_sha}, out / 'last_checkpoint.pt')
    print('[PASS] training completed' if not a.smoke else '[SMOKE PASS] no weights saved', flush=True)


if __name__ == '__main__':
    main()
