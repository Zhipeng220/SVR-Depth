#!/usr/bin/env python3
"""Development smoke only for O_multi and O_fast; no formal checkpoints."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments' / 'p1')]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--p-smoke', required=True, help='Completed fresh P_multi smoke directory')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import gt2_fixed_clip_repair_train as gt2
    import gt4a2_c3_closed_loop as c3
    import gt4a2_exp1_train_eval as exp1
    import gt4a2_q0_3_necessity_controls as q03
    from train_pmulti_fresh import BASE_SHA, SEQUENCES, SceneReader, sha
    from y08y09_models import FastDepthAdapter, rgb_for_output
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    snapshot = Path(args.snapshot).resolve()
    p_smoke, out = Path(args.p_smoke).resolve(), Path(args.out_dir).resolve()
    if out.exists():
        raise FileExistsError(f'Use a new output directory: {out}')
    if sha(snapshot / 'model.safetensors') != BASE_SHA:
        raise RuntimeError('Backbone does not match locked P_multi model')
    identity = json.loads((p_smoke / 'run_identity.json').read_text(encoding='utf8'))
    if identity.get('smoke') is not True or identity.get('seed') != 20261025 or identity.get('backbone_sha256') != BASE_SHA:
        raise RuntimeError('Expected the fresh 20261025 P smoke cache identity')
    pairs = ROOT / 'outputs' / 'gt4a2_exp1'
    for seq in SEQUENCES:
        pair = pairs / f'gt4a2_exp1_{seq}_pairs_used.csv'
        if sha(pair) != identity['pair_hashes'][seq]:
            raise RuntimeError(f'{seq}: pair manifest differs from P smoke')
    readers = {s: c3.CycleReader(p_smoke / 'cache' / s / 'cycle_cache', max_items=1) for s in SEQUENCES}
    slots = [(s, 0) for s in SEQUENCES]
    reader = SceneReader(readers, slots)
    for i in range(3):
        reader.load(i)  # verify each cache SHA before any training
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device('cuda')
    torch.manual_seed(20261024)
    model, head = q03.load_base_model(snapshot, device)
    init_state = c3.initial_state(head)
    if exp1.state_sha256(init_state) != identity['initial_head_sha256']:
        raise RuntimeError('O_multi initial head differs from fresh P initialization')
    radial = model.pixel_decoder.radial_module
    original_load = reader.load
    def load_with_shape(slot):
        cycle = original_load(slot)
        network = tuple(int(x) for x in cycle['network_shape'])
        if tuple(cycle['keyframe_state']['rays_map'].shape[-2:]) != network:
            raise RuntimeError('Cached rays and network canvas shapes differ')
        radial.set_original_shapes(network)
        return cycle
    reader.load = load_with_shape
    # A zero-initialized head must reproduce the native fast-step readout.
    cycle = reader.load(0)
    radius, conf, features = c3.move_state(cycle['keyframe_state'], device)
    first = c3.to_gpu_frame(cycle['frames'][0], device)
    second = c3.to_gpu_frame(cycle['frames'][1], device)
    shape = {'validity_mask': torch.ones((1,1,*cycle['network_shape']), device=device, dtype=torch.bool)}
    static = c3.to_gpu_static(cycle['statics'][0], device, shape)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        native_radius, _, native_features = exp1.one_fast_step(model, first, second, static, radius, conf, features)
        repaired = head(native_features)
        if any(not torch.equal(a, b) for a, b in zip(native_features, repaired)):
            raise RuntimeError('O_multi initial head is not an identity')
        readout_radius = q03.radius_from_features(model, repaired)
    if native_radius.shape != readout_radius.shape or not torch.allclose(native_radius.float(), readout_radius.float(), rtol=1e-3, atol=1e-3):
        raise RuntimeError(f'Initial O_multi readout differs from native radius: native={tuple(native_radius.shape)}, adapted={tuple(readout_radius.shape)}')
    print(f'[PASS] O_multi initial readout matches native: shape={tuple(native_radius.shape)}', flush=True)
    del cycle, radius, conf, features, first, second, shape, static, native_radius, native_features, repaired, readout_radius
    print('[O_MULTI] 3 updates on native, non-repaired state', flush=True)
    result_multi = q03.train_one_output_only_module('O_multi', model, head, reader, [0,1,2],
        out / 'O_multi', steps=3, initial_state=init_state, smoke=True)
    reader.load = original_load
    del head, model
    torch.cuda.empty_cache()
    print('[O_FAST] 3 updates using native depth and RGB', flush=True)
    torch.manual_seed(20261025); np.random.seed(20261025); random.seed(20261025)
    model = gt2.load_model(snapshot, 'off', device)
    adapter = FastDepthAdapter().to(device).train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    fast_rows = []
    for slot in range(3):
        cycle = reader.load(slot)
        model.pixel_decoder.radial_module.set_original_shapes(tuple(cycle['network_shape']))
        radius, conf, features = c3.move_state(cycle['keyframe_state'], device)
        rays = cycle['keyframe_state']['rays_map'].to(device)
        shape = {'validity_mask': torch.ones((1,1,*cycle['network_shape']), device=device, dtype=torch.bool),
                 'padded_shape': cycle['padded_shape'], 'paddings': cycle['paddings']}
        prev = c3.to_gpu_frame(cycle['frames'][0], device)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for offset in range(1,32):
            curr = c3.to_gpu_frame(cycle['frames'][offset], device)
            static = c3.to_gpu_static(cycle['statics'][offset-1], device, shape)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                radius, conf, features = exp1.one_fast_step(model, prev, curr, static, radius, conf, features)
                native = gt2.make_z_depth(radius, rays, shape['padded_shape'], shape['paddings'], model.interpolation_mode)
                rgb = rgb_for_output(curr, shape)
            if curr['gt'] is not None:
                with torch.autocast('cuda', dtype=torch.float16):
                    pred = adapter(native.detach(), rgb.detach())
                    loss, _ = gt2.masked_log_depth_loss(pred, curr['gt'])
                if not torch.isfinite(loss).item():
                    raise RuntimeError('O_fast non-finite loss')
                scaler.scale(loss / cycle['n_supervised']).backward()
                losses.append(float(loss.detach().item()))
            prev = curr
        if len(losses) != cycle['n_supervised']:
            raise RuntimeError('O_fast supervised frame count differs from cache')
        scaler.unscale_(optimizer)
        grads = [p.grad for p in adapter.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads) or not any(torch.count_nonzero(g).item() for g in grads):
            raise RuntimeError('O_fast gradients missing, zero, or non-finite')
        norm = float(torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0).item())
        old_scale = scaler.get_scale(); scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < old_scale:
            raise RuntimeError('O_fast optimizer update skipped')
        row = {'step': slot+1, 'sequence': SEQUENCES[slot], 'loss': sum(losses)/len(losses), 'gradient_norm': norm}
        fast_rows.append(row)
        print(f"[O_FAST] step={slot+1}/3 sequence={SEQUENCES[slot]} loss={row['loss']:.6f} grad={norm:.6f}", flush=True)
    result = {'status': 'SMOKE_PASS', 'formal_checkpoint_saved': False, 'p_smoke_identity': identity,
              'o_multi': result_multi, 'o_fast': fast_rows,
              'o_fast_peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024**3}
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(f'[PASS] O_multi and O_fast smoke; no formal checkpoints: {out}', flush=True)

if __name__ == '__main__':
    main()
