#!/usr/bin/env python3
"""SVR-Depth Y06 pilot: controlled history diagnostics and clean streaming costs.

Install under experiments/p1 in the same audited repository as Y02-A.
No training, Auto policy, K sweep, or target-GT-dependent online decisions.
--self-test runs CPU protocol tests without importing the model/repository.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

VERSION = 'Y06-pilot-1.0'
K = 32
EXPECTED = {
    'experiments/p1/gt4a2_y02a_cross_method_oracle.py':
        '575bc8bdf48027fc54b898975cc1737da17cf34d40bbdb8832424f32862dd521',
    'experiments/p1/gt4a2_q0_3_necessity_controls.py':
        '5ac9e648dbae36e17ebf64cd8891c226052f1a71171acf326f7dc6b591ff2390',
    'experiments/p1/gt4a2_y02_full_cycle_single_intervention.py':
        '7d03bbb0ca0ff686de044f96b4a4dacd583c796add6ce4b9bdeb813d24531f83',
    'experiments/p1/gt4a2_c3_closed_loop.py':
        '76327208acac4543b2a87ccba8b7af435e4c5ce65267b42052e51b435cbad7f5',
    'experiments/p1/gt4a2_exp1_train_eval.py':
        'c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f',
    'experiments/p1/gt2_fixed_clip_repair_train.py':
        'bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192',
    'velodepth/models/decoder_velo.py':
        '35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08',
    'velodepth/models/persistent_feature_repair.py':
        '6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c',
}
B1_SHA = '3eb52d56178a25092d9a3dc342b50aa81c6f8d5b218792a225e20ea2c6e01d1c'
METRICS = ('AbsRel', 'RMSE', 'delta1', 'signed_bias_mean_m', 'residual_mae_m')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def parse_ints(raw, low, high):
    values = [int(x.strip()) for x in raw.split(',') if x.strip()]
    if not values or len(values) != len(set(values)) or any(x < low or x > high for x in values):
        raise ValueError(f'Expected unique integers in {low}..{high}: {raw}')
    return sorted(values)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', default=os.environ.get('VDEEP_SNAPSHOT', ''))
    p.add_argument('--sequences', default='crowd2,crowd3,balloon2')
    p.add_argument('--checkpoint', choices=('C', 'D', 'both'), default='both')
    p.add_argument('--task', choices=('both', 'benchmark', 'history'), default='both')
    for seq in ('crowd2', 'crowd3', 'balloon2'):
        p.add_argument(f'--{seq}-pair-csv', default=f'outputs/gt4a2_exp1/gt4a2_exp1_{seq}_pairs_used.csv')
    for ck in ('c', 'd'):
        p.add_argument(f'--{ck}-checkpoint', default=f'outputs/gt4a2_c3/gt4a2_c3_{ck.upper()}_final_step1000.pt')
    p.add_argument('--b3-checkpoint', default='outputs/gt4a2_q0_3_validation/training/Q03_B3_output_adapter_final_step1000.pt')
    p.add_argument('--b1-fit', default='outputs/gt4a2_q0_3_validation/q0_3_global_scale_bias_fit.json')
    p.add_argument('--q03-review', default='outputs/gt4a2_q0_3_validation/q0_3_review.zip')
    p.add_argument('--y02-review', default='outputs/gt4a2_y02_validation/y02_review.zip')
    p.add_argument('--benchmark-frames', type=int, default=96, help='Prefix from frame0; must contain complete K32 cycles.')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup-cycles', type=int, default=1)
    p.add_argument('--once-age', type=int, choices=(8, 16, 24), default=8)
    p.add_argument('--timing-scope', choices=('rgb-resident', 'online'), default='rgb-resident')
    p.add_argument('--history-target-ages', default='8,16,24')
    p.add_argument('--history-ages', default='0,4,8,16,24')
    p.add_argument('--max-cycles-per-sequence', type=int, default=0, help='0 uses the five fixed Y02-A cycles.')
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--opencv-threads', type=int, default=1)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--monitor-gpu', default='auto', help='Physical nvidia-smi GPU index/UUID, independent of CUDA-visible index.')
    p.add_argument('--allow-shared-gpu', action='store_true', help='Diagnostic escape only; shared timings are marked unsuitable.')
    p.add_argument('--reference-tolerance', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=20261025)
    p.add_argument('--smoke-only', action='store_true')
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--no-examples', action='store_true')
    p.add_argument('--out-dir', default='outputs/gt4a2_y06_validation')
    a = p.parse_args()
    a.seqs = [x.strip().lower() for x in a.sequences.split(',') if x.strip()]
    a.ckpts = ('C', 'D') if a.checkpoint == 'both' else (a.checkpoint,)
    if not a.seqs or len(set(a.seqs)) != len(a.seqs) or set(a.seqs) - {'crowd2', 'crowd3', 'balloon2'}:
        p.error('Bad --sequences')
    if a.smoke_only:
        a.benchmark_frames, a.repeats, a.max_cycles_per_sequence = 64, 1, 1
        a.history_target_ages, a.history_ages = '16', '0,8,16'
    if a.benchmark_frames < K or a.benchmark_frames % K:
        p.error('--benchmark-frames must be a positive multiple of 32')
    if a.repeats < 1 or a.warmup_cycles < 1 or a.warmup_cycles * K > a.benchmark_frames:
        p.error('Invalid repeats/warmup; warmup must fit inside the benchmark prefix')
    if not 0 <= a.max_cycles_per_sequence <= 5 or a.cpu_threads < 1 or a.opencv_threads < 0:
        p.error('Invalid cycle/thread limits')
    if not math.isfinite(a.reference_tolerance) or not 0 < a.reference_tolerance <= 1e-3:
        p.error('--reference-tolerance must be in (0, 1e-3]')
    try:
        a.target_ages = parse_ints(a.history_target_ages, 1, 31)
        a.hist_ages = sorted(set(parse_ints(a.history_ages, 0, 31)) | {0} | set(a.target_ages))
    except ValueError as e:
        p.error(str(e))
    return a


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str = 'native'
    checkpoint: str = ''
    gain: float = 0.0
    once_age: int = 0


def arm_specs(checkpoints, once_age):
    out = [Arm('A'), Arm('B1', 'calibration'), Arm('B3', 'output')]
    for ck in checkpoints:
        out.extend([Arm(f'F_{ck}_once_age{once_age}', 'once', ck, 1.0, once_age),
                    Arm(f'S2_{ck}_gain0.5', 'every', ck, 0.5)])
    return out


def should_write(arm, age):
    return age > 0 and (arm.kind == 'every' or (arm.kind == 'once' and age == arm.once_age))


def expected_counts(arm, n):
    key = sum(i % K == 0 for i in range(n))
    prop = n - key
    return {'keyframe_calls': key, 'flow_calls': prop, 'native_transition_calls': prop,
            'repair_head_calls': sum(should_write(arm, i % K) for i in range(n)),
            'output_adapter_calls': prop if arm.kind == 'output' else 0,
            'additional_readout_passes': prop if arm.kind == 'output' else 0}


def rotated_orders(names, repeats, seed):
    initial = list(names)
    random.Random(seed).shuffle(initial)
    return [initial[r % len(initial):] + initial[:r % len(initial)] for r in range(repeats)]


def history_plan(rows, cycles, target_ages, hist_ages):
    result = []
    for cycle in cycles:
        for age in target_ages:
            target = cycle * K + age
            if target >= len(rows) or target - max(hist_ages) < 0:
                raise RuntimeError(f'Invalid history panel: cycle={cycle}, target={target}')
            matched = str(rows[target].get('gt_match_status', '')).upper() == 'MATCHED'
            result.append({'cycle': cycle, 'target_index': target, 'native_age': age,
                           'gt_matched': matched,
                           'histories': [{'state_age': h, 'anchor_index': target - h} for h in hist_ages]})
    return result


def timed_call(fn, synchronize, clock=time.perf_counter):
    synchronize()  # Pending work from before the sample is excluded.
    start = clock()
    result = fn()
    synchronize()  # Completion of this sample is included.
    return result, clock() - start


@dataclass
class State:
    radius: object
    confidence: object
    features: object
    rays_map: object


def state_copy(state):
    return State(state.radius.detach().clone(), state.confidence.detach().clone(),
                 [x.detach().clone() for x in state.features], state.rays_map.detach().clone())


def tensor_delta(a, b):
    return float((a.float() - b.float()).abs().max().item())


def advance_state(rt, model, shape, frame, prev, static, state, arm, head, index,
                  *, is_keyframe=None, debug=False):
    key = index % K == 0 if is_keyframe is None else is_keyframe
    counts = {key: 0 for key in expected_counts(Arm('A'), 0)}
    audit = {}
    if key:
        rays, radius, confidence, features = rt.exp1.keyframe_state(model, frame, shape)
        state = State(radius, confidence, features, rays)
        counts['keyframe_calls'] = 1
        return state, radius, counts, audit
    if state is None or prev is None or static is None:
        raise RuntimeError('Propagation requires a complete initial state and RGB transition')
    radius, confidence, features = rt.exp1.one_fast_step(
        model, prev, frame, static, state.radius, state.confidence, state.features)
    state = State(radius, confidence, features, state.rays_map)
    radius_out = radius
    counts['flow_calls'] = counts['native_transition_calls'] = 1
    write = should_write(arm, index % K)
    before = state_copy(state) if debug and (write or arm.kind == 'output') else None
    if arm.kind == 'output':
        adapted = head(state.features)
        radius_out = rt.q03.radius_from_features(model, adapted)
        counts['output_adapter_calls'] = counts['additional_readout_passes'] = 1
        del adapted
    elif write:
        if arm.gain == 1.0:
            # Match Y02-A one-shot head(features), including its execution cost.
            state.features = [x.detach() for x in head(state.features)]
        else:
            state.features = rt.q03.apply_repair_residual(state.features, head, arm.gain)
        counts['repair_head_calls'] = 1
        if radius_out is not state.radius:
            raise RuntimeError('F-only post-readout write changed current radius')
    if before is not None:
        audit = {'radius_mutation_max_m': tensor_delta(before.radius, state.radius),
                 'confidence_mutation_max': tensor_delta(before.confidence, state.confidence),
                 'rays_mutation_max': tensor_delta(before.rays_map, state.rays_map)}
        if arm.kind == 'output':
            audit['feature_mutation_max'] = max(tensor_delta(x, y) for x, y in zip(before.features, state.features))
        if max(audit.values(), default=0) != 0:
            raise RuntimeError(f'Unexpected state mutation in {arm.name}: {audit}')
    return state, radius_out, counts, audit


def score_prediction(gt, pred, np):
    mask = (gt > 0) & np.isfinite(gt)
    invalid = mask & (~np.isfinite(pred) | (pred <= 0))
    if int(invalid.sum()):
        raise RuntimeError(f'Invalid predictions on {int(invalid.sum())} GT pixels; cannot discard them')
    if int(mask.sum()) < 1000:
        raise RuntimeError('Too few GT-valid pixels')
    x = pred[mask].astype(np.float64)
    y = gt[mask].astype(np.float64)
    xm, ym = float(x.mean()), float(y.mean())
    var = float(np.mean((x - xm) ** 2))
    if var < 1e-12:
        scale, bias = 1.0, float(np.median(y - x))
    else:
        scale = float(np.mean((x - xm) * (y - ym)) / var)
        bias = ym - scale * xm
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError('Nonpositive/nonfinite GT-oracle scale; stop for diagnosis')
    cal = np.maximum(scale * pred.astype(np.float64) + bias, 1e-3).astype(np.float32)
    out = {'valid_pixels': int(mask.sum()), 'gt_valid_pixels': int(mask.sum()),
           'oracle_scale': scale, 'oracle_bias_m': bias,
           'oracle_nonpositive_fraction_before_clamp': float(np.mean(scale * x + bias <= 0))}
    for style, z in [('raw', x), ('oracle', cal[mask].astype(np.float64))]:
        residual = z - y
        values = (float(np.mean(np.abs(residual) / y)), float(np.sqrt(np.mean(residual ** 2))),
                  float(np.mean(np.maximum(y / z, z / y) < 1.25)), float(residual.mean()),
                  float(np.mean(np.abs(residual))))
        out.update({f'{style}_{key}': value for key, value in zip(METRICS, values)})
    if out['oracle_RMSE'] > out['raw_RMSE'] + 2e-6:
        raise RuntimeError('Unexpected oracle RMSE degradation')
    return out


def load_gt(rt, rec, shape):
    raw = rt.cv2.imread(str(rec['depth_path']), rt.cv2.IMREAD_UNCHANGED)
    if raw is None or raw.dtype != rt.np.uint16 or raw.shape != shape['original_shape']:
        raise RuntimeError(f'Bad GT: {rec["depth_path"]}')
    return raw.astype(rt.np.float32) / float(rt.exp1.DEPTH_SCALE)


def make_rgb_frame(rt, rec, shape, device, rgb=None):
    if rgb is None:
        return rt.exp1.load_frame(rec, shape, device, need_gt=False)
    if rgb.shape[:2] != shape['original_shape']:
        raise RuntimeError('RGB shape changed inside sequence')
    img, frame_u8 = rt.gt2.preprocess_rgb(rgb, device, shape['paddings'],
                                         shape['network_shape'], shape['mean'], shape['std'])
    return {'index': int(rec['frame_index_zero_based']), 'img': img, 'frame_u8': frame_u8,
            'gt_np': None, 'gt': None, 'gt_path': None, 'rgb_path': Path(rec['rgb_path'])}


class SequenceRunner:
    def __init__(self, rt, model, shape, device, arm, head, calibration):
        self.rt, self.model, self.shape, self.device = rt, model, shape, device
        self.arm, self.head, self.calibration = arm, head, calibration
        self.dis = rt.cv2.DISOpticalFlow_create(rt.cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.state = self.prev = None

    def reset(self):
        self.state = self.prev = None

    def step(self, index, rec, rgb=None, *, debug=False):
        rt = self.rt
        frame = make_rgb_frame(rt, rec, self.shape, self.device, rgb)
        key = index % K == 0
        static = None if key else rt.exp1.transition_static(self.model, self.prev, frame, self.shape, self.dis)
        with rt.torch.autocast(device_type='cuda', dtype=rt.torch.float16, enabled=True):
            state, radius_out, counts, audit = advance_state(
                rt, self.model, self.shape, frame, self.prev, static, self.state,
                self.arm, self.head, index, debug=debug)
            pred_t = rt.gt2.make_z_depth(radius_out, state.rays_map, self.shape['padded_shape'],
                                        self.shape['paddings'], self.model.interpolation_mode)
        pred = pred_t.detach().float().cpu().numpy()
        if self.arm.kind == 'calibration':
            pred = rt.np.maximum(float(self.calibration['scale']) * pred + float(self.calibration['bias_m']), 1e-4)
        self.state, self.prev = state, frame
        return pred, counts, audit


def runtime_imports(root):
    checks = {}
    for rel, expected in EXPECTED.items():
        path = root / rel
        actual = digest(path) if path.is_file() else None
        if actual != expected:
            raise RuntimeError(f'Audited source mismatch: {rel}: {actual} != {expected}')
        checks[rel] = {'sha256': actual, 'expected': expected, 'match': True}
    sys.path[:0] = [str(root / 'experiments/p1'), str(root)]
    y02a = importlib.import_module('gt4a2_y02a_cross_method_oracle')
    for mod in (y02a, y02a.q03, y02a.y02, y02a.y01, y02a.exp1, y02a.gt2):
        path = Path(mod.__file__).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f'Imported module outside expected repository: {path}')
        rel = str(path.relative_to(root)).replace('\\', '/')
        checks.setdefault(rel, {'sha256': digest(path), 'expected': None, 'match': None})
    return SimpleNamespace(y02a=y02a, exp1=y02a.exp1, gt2=y02a.gt2, q03=y02a.q03,
                           np=y02a.np, torch=y02a.torch, cv2=y02a.cv2), checks


def gpu_snapshot(rt, device, monitor):
    """Endpoint telemetry. It cannot certify that no transient process ran between checks."""
    result = {'available': False, 'mapping_verified': False, 'process_query_verified': False,
              'foreign_processes': []}
    try:
        props = rt.torch.cuda.get_device_properties(device)
        torch_uuid = str(getattr(props, 'uuid', '')).strip()
        identity = monitor
        if identity == 'auto':
            visible = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
            ordinal = device.index or 0
            if torch_uuid.startswith('GPU-'):
                identity = torch_uuid
            elif visible and ordinal < len(visible):
                identity = visible[ordinal]
            else:
                result['error'] = 'Set CUDA_VISIBLE_DEVICES and --monitor-gpu, or expose a PyTorch GPU UUID'
                return result
        def query(fields, apps=False):
            cmd = ['nvidia-smi', '--id=' + identity,
                   ('--query-compute-apps=' if apps else '--query-gpu=') + fields,
                   '--format=csv,noheader,nounits']
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=True)
            return list(csv.reader(io.StringIO(proc.stdout)))
        gpu = query('index,uuid,name,memory.used,temperature.gpu,utilization.gpu,clocks.sm,clocks.mem,power.draw')[0]
        result.update({'available': True, 'monitor_id': identity, 'physical_index': gpu[0].strip(),
                       'gpu_uuid': gpu[1].strip(), 'gpu_name': gpu[2].strip(),
                       'memory_used_mib': gpu[3].strip(), 'temperature_c': gpu[4].strip(),
                       'utilization_percent': gpu[5].strip(), 'sm_clock_mhz': gpu[6].strip(),
                       'memory_clock_mhz': gpu[7].strip(), 'power_w': gpu[8].strip()})
        if torch_uuid.startswith('GPU-'):
            result['mapping_verified'] = torch_uuid == result['gpu_uuid']
        else:
            visible = [x.strip() for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
            ordinal = device.index or 0
            selected = visible[ordinal] if ordinal < len(visible) else ''
            result['mapping_verified'] = selected in (result['physical_index'], result['gpu_uuid']) and (
                selected.startswith('GPU-') or os.environ.get('CUDA_DEVICE_ORDER') == 'PCI_BUS_ID')
        for row in query('gpu_uuid,pid,process_name,used_gpu_memory', apps=True):
            if len(row) < 4 or row[0].strip() != result['gpu_uuid']:
                continue
            pid = int(row[1].strip())
            if pid != os.getpid():
                result['foreign_processes'].append({'pid': pid, 'process': row[2].strip(),
                                                     'used_gpu_memory_mib': row[3].strip()})
        result['process_query_verified'] = True
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as e:
        result['error'] = str(e)
    return result


def telemetry_clean(snapshot):
    return (snapshot['available'] and snapshot['mapping_verified']
            and snapshot['process_query_verified'] and not snapshot['foreign_processes'])


def make_model_and_head(rt, a, arm, b3path, cpaths):
    device = rt.torch.device(a.device)
    model = rt.gt2.load_model(Path(a.snapshot).resolve(), 'off', device)
    head = None
    if arm.kind in ('output', 'once', 'every'):
        path = b3path if arm.kind == 'output' else cpaths[arm.checkpoint]
        expected = rt.y02a.B3_SHA if arm.kind == 'output' else rt.y02a.CKPT_SHA[arm.checkpoint]
        if digest(path) != expected:
            raise RuntimeError('Frozen checkpoint changed since preflight: ' + str(path))
        obj = rt.torch.load(path, map_location='cpu', weights_only=False)
        weights = obj['adapter_state_dict'] if arm.kind == 'output' else obj['repair_head_state_dict']
        head = rt.q03.make_frozen_head(weights, device)
        del obj, weights
    return model, head, device


def preflight(rt, a, root, sources):
    # Reuse the supplied Y02-A checks, training metadata and reference provenance.
    # Y02-A's preflight expects both audited C/D checkpoint files even for a C-only smoke.
    reference, manifests, _, qref, yref, b3path, cpaths = rt.y02a.preflight(a)
    fit_path = Path(a.b1_fit).resolve()
    if digest(fit_path) != B1_SHA:
        raise RuntimeError('B1 fit differs from the audited Q0-3 train-only fit')
    fit = json.loads(fit_path.read_text(encoding='utf-8'))
    if (fit.get('fit_sequence') != 'crowd2' or fit.get('fit_frame_indices_inclusive') != [0, 767]
            or fit.get('fit_uses_crowd3') is not False or fit.get('fit_uses_balloon2') is not False
            or fit.get('fit_uses_crowd2_temporal_holdout') is not False
            or not math.isfinite(float(fit['scale'])) or float(fit['scale']) <= 0
            or not math.isfinite(float(fit['bias_m']))):
        raise RuntimeError('B1 fit provenance is not train-only')
    arms = arm_specs(a.ckpts, a.once_age)
    plans = {}
    for seq, rows in manifests.items():
        if a.task != 'history' and len(rows) < a.benchmark_frames:
            raise RuntimeError(f'{seq}: benchmark prefix exceeds available RGB frames')
        bank = list(rt.y02a.CYCLES[seq])
        if a.max_cycles_per_sequence:
            bank = bank[:a.max_cycles_per_sequence]
        plans[seq] = history_plan(rows, bank, a.target_ages, a.hist_ages)
    env = {'python': sys.version, 'platform': sys.platform, 'torch': rt.torch.__version__,
           'numpy': rt.np.__version__, 'opencv': rt.cv2.__version__,
           'cuda_available': rt.torch.cuda.is_available(), 'torch_cuda_version': rt.torch.version.cuda,
           'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'CUDA_DEVICE_ORDER': os.environ.get('CUDA_DEVICE_ORDER'),
           'torch_threads': rt.torch.get_num_threads(), 'opencv_threads': rt.cv2.getNumThreads()}
    if rt.torch.cuda.is_available():
        dev = rt.torch.device(a.device)
        props = rt.torch.cuda.get_device_properties(dev)
        env.update({'gpu_name': props.name, 'gpu_total_memory_bytes': props.total_memory,
                    'cudnn_version': rt.torch.backends.cudnn.version(),
                    'cudnn_benchmark': rt.torch.backends.cudnn.benchmark,
                    'cudnn_deterministic': rt.torch.backends.cudnn.deterministic,
                    'cudnn_allow_tf32': rt.torch.backends.cudnn.allow_tf32,
                    'matmul_allow_tf32': rt.torch.backends.cuda.matmul.allow_tf32,
                    'gpu_endpoint': gpu_snapshot(rt, dev, a.monitor_gpu)})
    report = {'status': 'PREFLIGHT_PASS', 'version': VERSION, 'training_performed': False,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'script_sha256': digest(Path(__file__)), 'arguments': vars(a),
              'source_checks': sources, 'Y02A_reference_preflight': reference,
              'B1_fit': {'path': str(fit_path), 'sha256': digest(fit_path), 'metadata': fit},
              'environment': env, 'arms': [asdict(x) for x in arms],
              'benchmark_counts_per_trial': {x.name: expected_counts(x, a.benchmark_frames) for x in arms},
              'orders': {s: rotated_orders([x.name for x in arms], a.repeats, a.seed + n)
                         for n, s in enumerate(a.seqs)},
              'history_plan': plans,
              'timing_scope': {
                  'input': a.timing_scope,
                  'includes': ['RGB preprocessing/H2D', 'CPU DIS flow and transfer', 'native keyframe/propagation',
                               'required adapter/readout/writeback', 'z-depth postprocess', 'CPU depth output',
                               'per-frame CUDA completion synchronization'],
                  'rgb_decode_included': a.timing_scope == 'online',
                  'excludes': ['model/checkpoint loading', 'warmup', 'GT reads', 'metrics/oracle',
                               'diagnostic state clones', 'CSV/NPZ writes', 'telemetry calls'],
                  'schedule': 'serialized streaming with CPU depth output; not maximum batched GPU throughput'},
              'claim_limits': ['development pilot, not CVPR main results', 'K is fixed at32; no Auto or K sweep',
                               'GPU endpoint checks do not prove absence of transient interference',
                               'head-call savings alone do not establish quality-matched compute savings']}
    return report, manifests, plans, qref, yref, b3path, cpaths, fit, arms


def run_trial(rt, a, seq, rows, arm, repeat, position, rgb_cache, b3path, cpaths, fit):
    initial_gpu = gpu_snapshot(rt, rt.torch.device(a.device), a.monitor_gpu)
    if initial_gpu['foreign_processes'] and not a.allow_shared_gpu:
        raise RuntimeError('GPU_BUSY before model loading: use an idle GPU for cost measurements: ' + str(initial_gpu))
    model, head, device = make_model_and_head(rt, a, arm, b3path, cpaths)
    shape = rt.exp1.shape_info_for(model, Path(rows[0]['rgb_path']), device)
    runner = SequenceRunner(rt, model, shape, device, arm, head, fit)
    sync = lambda: rt.torch.cuda.synchronize(device)
    with rt.torch.no_grad():
        for idx in range(a.warmup_cycles * K):
            runner.step(idx, rows[idx], None if rgb_cache is None else rgb_cache[idx])
        runner.reset()
        gc.collect()
        sync()
        before = gpu_snapshot(rt, device, a.monitor_gpu)
        if before['foreign_processes'] and not a.allow_shared_gpu:
            raise RuntimeError('GPU_BUSY: another compute process is using the monitored GPU. '
                               'Pause same-GPU training or choose an idle GPU before benchmarking: ' + str(before))
        rt.torch.cuda.reset_peak_memory_stats(device)
        base_bytes = rt.torch.cuda.memory_allocated(device)
        times, counts = [], {key: 0 for key in expected_counts(arm, 0)}
        output_samples = {}
        sample_indices = {0, a.once_age + 1, K, a.benchmark_frames - 1}
        for idx, rec in enumerate(rows[:a.benchmark_frames]):
            rgb = None if rgb_cache is None else rgb_cache[idx]
            (pred, step_counts, _), elapsed = timed_call(lambda: runner.step(idx, rec, rgb), sync)
            for key, value in step_counts.items():
                counts[key] += value
            times.append({'sequence': seq, 'arm': arm.name, 'repeat': repeat, 'order_position': position,
                          'frame_index': idx, 'cycle': idx // K, 'age': idx % K,
                          'is_keyframe': idx % K == 0, 'latency_ms': elapsed * 1000,
                          'repair_applied': bool(step_counts['repair_head_calls']),
                          'output_adapter_called': bool(step_counts['output_adapter_calls']),
                          'timing_scope': a.timing_scope})
            if idx in sample_indices:
                output_samples[idx] = pred.copy()  # CPU-only audit copy, after the timer stopped.
            del pred
        sync()
        peak, reserved = rt.torch.cuda.max_memory_allocated(device), rt.torch.cuda.max_memory_reserved(device)
        after = gpu_snapshot(rt, device, a.monitor_gpu)
    if counts != expected_counts(arm, a.benchmark_frames):
        raise RuntimeError(f'Call-count mismatch in {arm.name}: {counts}')
    values = [x['latency_ms'] for x in times]
    key_values = [x['latency_ms'] for x in times if x['is_keyframe']]
    prop_values = [x['latency_ms'] for x in times if not x['is_keyframe']]
    meta = {'sequence': seq, 'arm': arm.name, 'repeat': repeat, 'order_position': position,
            'frames': len(times), 'timing_scope': a.timing_scope, **counts,
            'mean_ms': statistics.mean(values), 'p50_ms': float(rt.np.percentile(values, 50)),
            'p95_ms': float(rt.np.percentile(values, 95)), 'p99_ms': float(rt.np.percentile(values, 99)),
            'keyframe_mean_ms': statistics.mean(key_values),
            'propagation_mean_ms': statistics.mean(prop_values),
            'serialized_fps_excluding_logging': 1000.0 / statistics.mean(values),
            'torch_baseline_allocated_mib': base_bytes / 2**20,
            'torch_peak_allocated_mib': peak / 2**20, 'torch_peak_reserved_mib': reserved / 2**20,
            'torch_incremental_peak_mib': (peak - base_bytes) / 2**20,
            'head_parameters': 0 if head is None else sum(p.numel() for p in head.parameters()),
            'output_audit_samples': len(output_samples),
            'endpoint_checks_clean': telemetry_clean(before) and telemetry_clean(after),
            'foreign_process_detected': bool(before['foreign_processes'] or after['foreign_processes']),
            'gpu_before': before, 'gpu_after': after}
    del runner, shape, model, head
    gc.collect()
    rt.torch.cuda.empty_cache()
    return times, meta, output_samples


def check_reference(metrics, ref, tolerance, description):
    deviation = max(abs(metrics['raw_' + key] - float(ref[key])) for key in METRICS[:3])
    if deviation > tolerance:
        raise RuntimeError(f'Reference replay mismatch {description}: {deviation} > {tolerance}')
    if int(ref['valid_pixels']) != metrics['valid_pixels']:
        raise RuntimeError(f'Reference valid-pixel mismatch: {description}')
    return deviation


def quality_pass(rt, a, seq, rows, arms, qref, yref, b3path, cpaths, fit, out, *, timed_samples=None):
    result, invariants, a_metrics, a_audit_preds = [], [], {}, {}
    example_idx = a.once_age + 1
    examples = {}
    qmap = {'A': 'A_native', 'B1': 'B1_global_scale_bias', 'B3': 'B3_output_adapter_trained_output_only'}
    for arm in arms:  # A must be first. All GT/metrics here are outside benchmark passes.
        arm_samples = [s['predictions'] for s in (timed_samples or []) if s['arm'] == arm.name]
        model, head, device = make_model_and_head(rt, a, arm, b3path, cpaths)
        shape = rt.exp1.shape_info_for(model, Path(rows[0]['rgb_path']), device)
        runner = SequenceRunner(rt, model, shape, device, arm, head, fit)
        with rt.torch.no_grad():
            for idx, rec in enumerate(rows[:a.benchmark_frames]):
                pred, _, audit = runner.step(idx, rec, debug=True)
                timing_diffs = [float(rt.np.max(rt.np.abs(pred - sample[idx])))
                                for sample in arm_samples if idx in sample]
                timing_diff = max(timing_diffs) if timing_diffs else None
                if timing_diff is not None and timing_diff > 1e-3:
                    raise RuntimeError(f'Timed/quality output mismatch after warmup/reset: {seq}/{arm.name}/{idx}: {timing_diff}')
                if arm.name == 'A' and idx % K in (0, a.once_age):
                    a_audit_preds[idx] = pred.copy()
                if idx == example_idx and not a.no_examples:
                    examples[arm.name] = pred.copy()
                reset_diff = None
                if idx % K == 0 or (arm.kind == 'once' and idx % K == arm.once_age):
                    if arm.name == 'A':
                        a_audit_preds[idx] = pred.copy()
                    elif arm.kind != 'output' or idx % K == 0:
                        if idx % K == 0 or arm.kind == 'once':
                            expected = a_audit_preds[idx]
                            if arm.kind == 'calibration':
                                expected = rt.np.maximum(float(fit['scale']) * expected + float(fit['bias_m']), 1e-4)
                            reset_diff = float(rt.np.max(rt.np.abs(pred - expected)))
                            if reset_diff > 1e-3:
                                raise RuntimeError(f'Keyframe/once-current-output invariant failed: {seq}/{arm.name}/{idx}')
                if str(rec.get('gt_match_status', '')).upper() != 'MATCHED':
                    continue
                gt = load_gt(rt, rec, shape)
                metrics = score_prediction(gt, pred, rt.np)
                row = {'sequence': seq, 'arm': arm.name, 'frame_index': idx, 'cycle': idx // K,
                       'age': idx % K, 'is_keyframe': idx % K == 0, 'rgb_file': Path(rec['rgb_path']).name,
                       'gt_file': Path(rec['depth_path']).name, **metrics}
                if arm.name == 'A':
                    a_metrics[idx] = row
                base = a_metrics[idx]
                row.update({'raw_gain_AbsRel_vs_A': base['raw_AbsRel'] - row['raw_AbsRel'],
                            'oracle_gain_AbsRel_vs_A': base['oracle_AbsRel'] - row['oracle_AbsRel'],
                            'oracle_gain_RMSE_vs_A': base['oracle_RMSE'] - row['oracle_RMSE']})
                result.append(row)
                dev = None
                if arm.kind == 'every':
                    key = (seq, f'S2_{arm.checkpoint}_gain0.5_R1_F_only', idx)
                    dev = check_reference(metrics, qref[key], a.reference_tolerance, str(key))
                elif arm.name in qmap:
                    key = (seq, qmap[arm.name], idx)
                    dev = check_reference(metrics, qref[key], a.reference_tolerance, str(key))
                elif arm.kind == 'once' and idx % K > arm.once_age:
                    key = (seq, arm.checkpoint, idx // K, arm.once_age, 'F_only', idx)
                    if key not in yref:
                        raise RuntimeError('Missing Y02 reference for the benchmark once arm: ' + str(key))
                    dev = check_reference(metrics, yref[key], a.reference_tolerance, str(key))
                elif arm.kind == 'once':
                    dev = max(abs(metrics['raw_' + m] - base['raw_' + m]) for m in METRICS[:3])
                    if dev > a.reference_tolerance:
                        raise RuntimeError('Once arm differs from A before or at its write')
                invariants.append({'sequence': seq, 'arm': arm.name, 'frame_index': idx,
                                   'reference_max_metric_deviation': dev,
                                   'timed_output_sample_max_abs_m': timing_diff,
                                   'timed_output_sample_comparisons': len(timing_diffs),
                                   'keyframe_or_once_current_max_abs_m': reset_diff, **audit})
                if idx == example_idx and not a.no_examples:
                    examples['gt'], examples['mask'] = gt.copy(), ((gt > 0) & rt.np.isfinite(gt)).astype(rt.np.uint8)
        write_csv(out/'progress'/f'{seq}_{arm.name}_quality.csv', [r for r in result if r['arm'] == arm.name])
        del runner, model, head, shape
        gc.collect()
        rt.torch.cuda.empty_cache()
    if examples:
        (out/'selected_examples').mkdir(exist_ok=True)
        rt.np.savez_compressed(out/'selected_examples'/f'{seq}_benchmark_frame{example_idx}.npz', **examples)
    return result, invariants


def quality_summary(rows, np, n_rgb_frames):
    groups, cycle_groups = defaultdict(list), defaultdict(list)
    for r in rows:
        groups[(r['sequence'], r['arm'])].append(r)
        cycle_groups[(r['sequence'], r['arm'], r['cycle'])].append(r)
    cycles = []
    for (seq, arm, cycle), rr in sorted(cycle_groups.items()):
        cycles.append({'sequence': seq, 'arm': arm, 'cycle': cycle, 'scored_frames': len(rr),
                       'raw_mean_AbsRel': statistics.mean(r['raw_AbsRel'] for r in rr),
                       'raw_mean_gain_AbsRel_vs_A': statistics.mean(r['raw_gain_AbsRel_vs_A'] for r in rr),
                       'oracle_mean_gain_RMSE_vs_A': statistics.mean(r['oracle_gain_RMSE_vs_A'] for r in rr)})
    result = []
    for (seq, arm), rr in sorted(groups.items()):
        cc = [r for r in cycles if r['sequence'] == seq and r['arm'] == arm]
        row = {'sequence': seq, 'arm': arm, 'scored_frames': len(rr),
               'complete_rgb_cycles': n_rgb_frames // K, 'cycles_with_GT': len(cc),
               'raw_harmed_cycle_fraction_vs_A': statistics.mean(r['raw_mean_gain_AbsRel_vs_A'] < -1e-12 for r in cc),
               'raw_harmed_frame_fraction_vs_A': statistics.mean(r['raw_gain_AbsRel_vs_A'] < -1e-12 for r in rr),
               'raw_frame_AbsRel_p95': float(np.percentile([r['raw_AbsRel'] for r in rr], 95)),
               'raw_frame_delta_AbsRel_vs_A_p95': float(np.percentile([-r['raw_gain_AbsRel_vs_A'] for r in rr], 95)),
               'quality_scope': 'same prefix as timed runs; development diagnostic, not whole-video generalization'}
        for style in ('raw', 'oracle'):
            for metric in METRICS:
                row['mean_' + style + '_' + metric] = statistics.mean(r[style + '_' + metric] for r in rr)
        result.append(row)
    return result, cycles


def benchmark(rt, a, manifests, qref, yref, b3path, cpaths, fit, arms, report, out):
    all_frames, all_trials, all_quality, all_invariants = [], [], [], []
    by_name = {x.name: x for x in arms}
    for seq in a.seqs:
        rows = manifests[seq]
        timed_samples = []
        # Prime the same input files before every arm. Resident mode keeps RAW RGB on CPU only.
        raw_rgb = [rt.gt2.load_rgb(Path(rec['rgb_path'])) for rec in rows[:a.benchmark_frames]]
        rgb_cache = raw_rgb if a.timing_scope == 'rgb-resident' else None
        if rgb_cache is None:
            del raw_rgb
        for repeat, order in enumerate(report['orders'][seq]):
            for position, name in enumerate(order):
                print(f'[BENCH] {seq} repeat={repeat+1}/{a.repeats} pos={position+1} {name}', flush=True)
                frames, meta, samples = run_trial(rt, a, seq, rows, by_name[name], repeat, position,
                                                  rgb_cache, b3path, cpaths, fit)
                timed_samples.append({'arm': name, 'repeat': repeat, 'predictions': samples})
                all_frames.extend(frames)
                all_trials.append(meta)
                write_csv(out/'progress'/f'{seq}_r{repeat}_{name}_latencies.csv', frames)
                write_json(out/'progress'/f'{seq}_r{repeat}_{name}_trial.json', meta)
                print(f'[BENCH DONE] mean={meta["mean_ms"]:.3f}ms p95={meta["p95_ms"]:.3f}ms '
                      f'endpoint_clean={meta["endpoint_checks_clean"]}', flush=True)
        del rgb_cache
        if a.timing_scope == 'rgb-resident':
            del raw_rgb
        # Only after all timed arms for this sequence: load GT and evaluate quality.
        print(f'[QUALITY OFF TIMER] {seq}', flush=True)
        quality, invariants = quality_pass(rt, a, seq, rows, arms, qref, yref, b3path, cpaths, fit, out,
                                           timed_samples=timed_samples)
        all_quality.extend(quality)
        all_invariants.extend(invariants)
        del timed_samples, samples
    write_csv(out/'y06_latency_frames.csv', all_frames)
    write_json(out/'y06_latency_trials.json', all_trials)
    write_csv(out/'y06_benchmark_quality_frames.csv', all_quality)
    write_csv(out/'y06_benchmark_invariants.csv', all_invariants)
    qs, cycles = quality_summary(all_quality, rt.np, a.benchmark_frames)
    write_csv(out/'y06_benchmark_cycle_quality.csv', cycles)
    write_csv(out/'y06_benchmark_quality_summary.csv', qs)
    costs = []
    for seq in a.seqs:
        for arm in arms:
            trials = [x for x in all_trials if x['sequence'] == seq and x['arm'] == arm.name]
            samples = [x['latency_ms'] for x in all_frames if x['sequence'] == seq and x['arm'] == arm.name]
            row = {'sequence': seq, 'arm': arm.name, 'repeats': len(trials), 'timing_scope': a.timing_scope,
                   'trial_mean_ms_median': statistics.median(x['mean_ms'] for x in trials),
                   'trial_mean_ms_min': min(x['mean_ms'] for x in trials),
                   'trial_mean_ms_max': max(x['mean_ms'] for x in trials),
                   'pooled_frame_p50_ms': float(rt.np.percentile(samples, 50)),
                   'pooled_frame_p95_ms': float(rt.np.percentile(samples, 95)),
                   'pooled_frame_p99_ms': float(rt.np.percentile(samples, 99)),
                   'torch_peak_allocated_mib_max': max(x['torch_peak_allocated_mib'] for x in trials),
                   'all_endpoint_checks_clean': all(x['endpoint_checks_clean'] for x in trials),
                   **expected_counts(arm, a.benchmark_frames)}
            row.update(next(x for x in qs if x['sequence'] == seq and x['arm'] == arm.name))
            costs.append(row)
    for row in costs:
        for baseline in ('A', 'B1', 'B3'):
            base = next(x for x in costs if x['sequence'] == row['sequence'] and x['arm'] == baseline)
            row['latency_ratio_to_' + baseline] = row['trial_mean_ms_median'] / base['trial_mean_ms_median']
    write_csv(out/'y06_cost_quality_summary.csv', costs)
    return {'timed_trials': len(all_trials), 'timed_frames': len(all_frames),
            'quality_rows': len(all_quality), 'all_endpoint_checks_clean': all(x['endpoint_checks_clean'] for x in all_trials),
            'foreign_process_detected': any(x['foreign_process_detected'] for x in all_trials),
            'reference_max_metric_deviation': max((r['reference_max_metric_deviation'] or 0) for r in all_invariants),
            'timed_output_sample_max_abs_m': max((r['timed_output_sample_max_abs_m'] or 0) for r in all_invariants),
            'timed_output_sample_comparisons_with_GT': sum(r['timed_output_sample_comparisons'] for r in all_invariants),
            'note': 'Latency ratios are not quality-matched speedups. Raw quality/risk must be assessed jointly.'}


def history_diagnostics(rt, a, manifests, plans, qref, out):
    result, invariants = [], []
    native = Arm('A')
    device = rt.torch.device(a.device)
    for seq in a.seqs:
        model = rt.gt2.load_model(Path(a.snapshot).resolve(), 'off', device)
        rows = manifests[seq]
        shape = rt.exp1.shape_info_for(model, Path(rows[0]['rgb_path']), device)
        eligible = [x for x in plans[seq] if x['gt_matched']]
        if not eligible:
            raise RuntimeError(f'{seq}: none of the fixed history targets has matched GT')
        sample = next((x['target_index'] for x in eligible if x['native_age'] == 16), eligible[0]['target_index'])
        for target_plan in eligible:
            target = target_plan['target_index']
            print(f'[HISTORY OFF TIMER] {seq} target={target} histories={a.hist_ages}', flush=True)
            begin = min(h['anchor_index'] for h in target_plan['histories'])
            frames = {i: rt.exp1.load_frame(rows[i], shape, device, need_gt=False) for i in range(begin, target+1)}
            dis = rt.cv2.DISOpticalFlow_create(rt.cv2.DISOPTICAL_FLOW_PRESET_FAST)
            with rt.torch.no_grad():
                static = {i: rt.exp1.transition_static(model, frames[i-1], frames[i], shape, dis)
                          for i in range(begin+1, target+1)}
                gt = load_gt(rt, rows[target], shape)
                examples = {'gt': gt, 'mask': ((gt > 0) & rt.np.isfinite(gt)).astype(rt.np.uint8)}
                full = None
                for h in target_plan['histories']:
                    anchor, age = h['anchor_index'], h['state_age']
                    with rt.torch.autocast(device_type='cuda', dtype=rt.torch.float16, enabled=True):
                        state, radius, _, _ = advance_state(rt, model, shape, frames[anchor], None,
                                                           None, None, native, None, anchor, is_keyframe=True)
                        for idx in range(anchor+1, target+1):
                            # Deliberately no K32 refresh inside this counterfactual history.
                            state, radius, _, _ = advance_state(rt, model, shape, frames[idx], frames[idx-1],
                                                               static[idx], state, native, None, idx, is_keyframe=False)
                        pred_t = rt.gt2.make_z_depth(radius, state.rays_map, shape['padded_shape'],
                                                    shape['paddings'], model.interpolation_mode)
                    pred = pred_t.detach().float().cpu().numpy()
                    metrics = score_prediction(gt, pred, rt.np)
                    if age == 0:
                        full = metrics
                    if full is None:
                        raise RuntimeError('History0 full-path reference must be evaluated first')
                    row = {'sequence': seq, 'cycle': target_plan['cycle'], 'target_index': target,
                           'native_target_age': target_plan['native_age'], 'state_age': age, 'anchor_index': anchor,
                           'rgb_file': Path(rows[target]['rgb_path']).name, 'gt_file': Path(rows[target]['depth_path']).name,
                           **metrics}
                    for style in ('raw', 'oracle'):
                        for metric in ('AbsRel', 'RMSE'):
                            row[f'{style}_{metric}_full_reference'] = full[f'{style}_{metric}']
                            row[f'{style}_{metric}_excess_vs_full'] = metrics[f'{style}_{metric}'] - full[f'{style}_{metric}']
                    result.append(row)
                    if age == target_plan['native_age']:
                        key = (seq, 'A_native', target)
                        dev = check_reference(metrics, qref[key], a.reference_tolerance, 'native-history ' + str(key))
                        invariants.append({'sequence': seq, 'target_index': target, 'state_age': age,
                                           'native_history_reference_max_deviation': dev})
                    if target == sample and not a.no_examples:
                        examples[f'age{age}'] = pred.copy()
                    del state, radius, pred_t, pred
            if target == sample and not a.no_examples:
                (out/'selected_examples').mkdir(exist_ok=True)
                rt.np.savez_compressed(out/'selected_examples'/f'{seq}_history_target{target}.npz', **examples)
            write_csv(out/'progress'/f'{seq}_history.csv', [r for r in result if r['sequence'] == seq])
            del frames, static, dis, examples, gt
        del shape, model
        gc.collect()
        rt.torch.cuda.empty_cache()
    write_csv(out/'y06_history_frame_metrics.csv', result)
    write_csv(out/'y06_history_invariants.csv', invariants)
    grouped = defaultdict(list)
    for row in result:
        grouped[(row['sequence'], row['state_age'])].append(row)
    summary = []
    for (seq, age), rr in sorted(grouped.items()):
        row = {'sequence': seq, 'state_age': age, 'n_same_target_comparisons': len(rr),
               'cycles': len({r['cycle'] for r in rr}),
               'raw_RMSE_worse_than_full_fraction': statistics.mean(r['raw_RMSE_excess_vs_full'] > 1e-12 for r in rr),
               'oracle_RMSE_worse_than_full_fraction': statistics.mean(r['oracle_RMSE_excess_vs_full'] > 1e-12 for r in rr),
               'note': 'same-target controlled histories; full path is a comparator, not a GT oracle or guaranteed upper bound'}
        for style in ('raw', 'oracle'):
            for metric in ('AbsRel', 'RMSE'):
                col = f'{style}_{metric}'
                row['mean_' + col] = statistics.mean(r[col] for r in rr)
                row['mean_' + col + '_excess_vs_full'] = statistics.mean(r[col + '_excess_vs_full'] for r in rr)
        summary.append(row)
    write_csv(out/'y06_history_summary.csv', summary)
    return {'rows': len(result), 'scored_targets': len({(r['sequence'], r['target_index']) for r in result}),
            'skipped_unmatched_targets': sum(not x['gt_matched'] for pp in plans.values() for x in pp),
            'native_history_reference_max_deviation': max(r['native_history_reference_max_deviation'] for r in invariants),
            'note': 'offline diagnostic cost is excluded from benchmark; no artificial state corruption'}


def package_review(out, root, source_checks):
    destination = out/'y06_review.zip'
    temp = out/'y06_review.zip.tmp'
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path not in (destination, temp) and path.name != 'y06_run.lock':
                z.write(path, str(path.relative_to(out)).replace('\\', '/'))
        z.write(Path(__file__), Path(__file__).name)
        for rel in source_checks:
            z.write(root/rel, 'source_snapshots/' + rel)
    temp.replace(destination)
    return destination


def verify_frozen_inputs(report, root):
    for rel, rec in report['source_checks'].items():
        if digest(root/rel) != rec['sha256']:
            raise RuntimeError('Source changed during Y06; cannot bind final evidence to executed source: ' + rel)
    if digest(Path(__file__)) != report['script_sha256']:
        raise RuntimeError('Y06 script changed during execution')
    reference = report['Y02A_reference_preflight']
    records = [reference['b3_checkpoint'], *reference['c3_checkpoints'].values(),
               *reference['manifests'].values(), reference['q0_3_reference'], reference['y02_reference'],
               report['B1_fit']]
    for rec in records:
        if digest(Path(rec['path'])) != rec['sha256']:
            raise RuntimeError('Reference/checkpoint/manifest changed during Y06: ' + rec['path'])
    model = Path(report['arguments']['snapshot']).resolve()/'model.safetensors'
    if digest(model) != reference['base_model_sha256']:
        raise RuntimeError('Frozen base model changed during Y06')


def self_test():
    import unittest
    import numpy as np

    class Value(float):
        def detach(self): return self

    class FakeHead:
        def __call__(self, features): return [Value(x + 10) for x in features]
        def parameters(self): return []

    class FakeExp:
        @staticmethod
        def keyframe_state(*_): return 7, Value(1), 3, [Value(2)]
        @staticmethod
        def one_fast_step(_m, _p, _f, _st, r, c, fs): return Value(r+fs[0]), c, fs

    rt = SimpleNamespace(exp1=FakeExp, q03=SimpleNamespace(
        radius_from_features=lambda _m, fs: Value(fs[0]*2),
        apply_repair_residual=lambda fs, _h, g: [Value(x + 10*g) for x in fs]))

    class ProtocolTests(unittest.TestCase):
        def test_counts_three_cycles(self):
            arms = arm_specs(('C', 'D'), 8)
            for arm in arms:
                c = expected_counts(arm, 96)
                self.assertEqual(c['keyframe_calls'], 3)
                self.assertEqual(c['flow_calls'], 93)
            self.assertEqual(expected_counts(arms[2], 96)['additional_readout_passes'], 93)
            self.assertEqual(expected_counts(arms[3], 96)['repair_head_calls'], 3)
            self.assertEqual(expected_counts(arms[4], 96)['repair_head_calls'], 93)

        def test_partial_counts(self):
            self.assertEqual(expected_counts(Arm('F', 'once', 'C', 1, 8), 33)['repair_head_calls'], 1)
            self.assertEqual(expected_counts(Arm('A'), 33)['keyframe_calls'], 2)

        def test_keyframe_never_repaired(self):
            self.assertFalse(should_write(Arm('S', 'every', 'C', .5), 0))

        def test_output_only_state_and_future(self):
            b = Arm('B3', 'output')
            st, _, _, _ = advance_state(rt, None, {}, {}, None, None, None, b, FakeHead(), 0)
            st, output, calls, _ = advance_state(rt, None, {}, {}, {}, {}, st, b, FakeHead(), 1)
            self.assertEqual((st.radius, st.features[0], output), (3, 2, 24))
            self.assertEqual(calls['output_adapter_calls'], 1)
            st, _, _, _ = advance_state(rt, None, {}, {}, {}, {}, st, Arm('A'), None, 2)
            self.assertEqual(st.radius, 5)

        def test_once_future_effect_and_reset(self):
            arm = Arm('F', 'once', 'C', 1, 8)
            st = State(Value(4), 3, [Value(2)], 7)
            st, output, calls, _ = advance_state(rt, None, {}, {}, {}, {}, st, arm, FakeHead(), 8)
            self.assertEqual((output, st.radius, st.features[0]), (6, 6, 12))
            self.assertEqual(calls['repair_head_calls'], 1)
            st, output, calls, _ = advance_state(rt, None, {}, {}, {}, {}, st, arm, FakeHead(), 9)
            self.assertEqual((output, calls['repair_head_calls']), (18, 0))
            st, output, _, _ = advance_state(rt, None, {}, {}, {}, None, st, arm, FakeHead(), 32)
            self.assertEqual((output, st.features[0]), (1, 2))

        def test_history_crosses_global_boundary_without_reset(self):
            st = State(Value(4), 3, [Value(2)], 7)
            st, output, calls, _ = advance_state(rt, None, {}, {}, {}, {}, st, Arm('A'), None, 32, is_keyframe=False)
            self.assertEqual((output, calls['keyframe_calls']), (6, 0))

        def test_missing_state_rejected(self):
            with self.assertRaises(RuntimeError):
                advance_state(rt, None, {}, {}, {}, {}, None, Arm('A'), None, 1)

        def test_timing_wait_and_evaluation_boundary(self):
            value = [0.0]
            def sync(): value[0] += 2
            def work(): value[0] += 3; return 'depth'
            result, elapsed = timed_call(work, sync, lambda: value[0])
            value[0] += 10000  # GT/metric work happens outside the timed call.
            self.assertEqual((result, elapsed), ('depth', 5))

        def test_order_rotation(self):
            orders = rotated_orders(['A', 'B1', 'B3'], 3, 1)
            self.assertEqual(len({order[0] for order in orders}), 3)
            self.assertTrue(all(sorted(order) == ['A', 'B1', 'B3'] for order in orders))

        def test_history_same_targets(self):
            rows = [{'gt_match_status': 'MATCHED'} for _ in range(128)]
            plan = history_plan(rows, [1], [8, 16], [0, 4, 8, 16, 24])
            self.assertEqual([x['target_index'] for x in plan], [40, 48])
            for row in plan:
                self.assertTrue(all(h['anchor_index'] + h['state_age'] == row['target_index'] for h in row['histories']))
            self.assertEqual(plan[0]['histories'][-1]['anchor_index'], 16)

        def test_no_partial_history_panel(self):
            with self.assertRaises(RuntimeError):
                history_plan([{} for _ in range(64)], [0], [8], [0, 24])

        def test_oracle_removes_affine(self):
            pred = np.linspace(.5, 3, 1600).reshape(40, 40).astype(np.float32)
            gt = 1.1*pred + .15
            m = score_prediction(gt, pred, np)
            self.assertLess(m['oracle_RMSE'], 2e-7)
            self.assertGreater(m['raw_RMSE'], .1)

        def test_invalid_prediction_is_failure(self):
            gt = np.ones((40, 40), dtype=np.float32)
            pred = gt.copy(); pred[0, 0] = np.nan
            with self.assertRaises(RuntimeError): score_prediction(gt, pred, np)

        def test_gt_mask_and_nonaffine_residual(self):
            pred = np.linspace(.5, 3, 1600).reshape(40, 40).astype(np.float32)
            gt = pred + .02 * np.sin(pred*10)
            gt[0, 0] = 0
            m = score_prediction(gt, pred, np)
            self.assertEqual(m['valid_pixels'], 1599)
            self.assertGreater(m['oracle_RMSE'], .001)

        def test_unverified_telemetry_not_clean(self):
            self.assertFalse(telemetry_clean({'available': True, 'mapping_verified': True,
                                              'process_query_verified': False, 'foreign_processes': []}))

        def test_benchmark_pass_reset_counts_and_no_GT(self):
            calls = []
            class FakeRunner:
                def __init__(self, _rt, _m, _s, _d, arm, _h, _f):
                    self.arm, self.state = arm, None
                def reset(self): self.state = None
                def step(self, index, _rec, _rgb=None):
                    self.state, output, counts, audit = advance_state(
                        rt, None, {}, {}, {} if self.state else None,
                        {} if self.state else None, self.state, self.arm, FakeHead(), index)
                    calls.append((index, output))
                    return np.full((2,2), output, dtype=np.float32), counts, audit
            cuda = SimpleNamespace(synchronize=lambda *_: None, reset_peak_memory_stats=lambda *_: None,
                                   memory_allocated=lambda *_: 0, max_memory_allocated=lambda *_: 0,
                                   max_memory_reserved=lambda *_: 0, empty_cache=lambda: None)
            fake_rt = SimpleNamespace(np=np, torch=SimpleNamespace(device=lambda *_:'mock', cuda=cuda,
                                      no_grad=lambda:__import__('contextlib').nullcontext()),
                                      exp1=SimpleNamespace(shape_info_for=lambda *_:{}))
            args = SimpleNamespace(device='cuda:0', warmup_cycles=1, benchmark_frames=64, once_age=8,
                                   monitor_gpu='auto', allow_shared_gpu=False, timing_scope='rgb-resident')
            names = ('SequenceRunner', 'make_model_and_head', 'load_gt', 'gpu_snapshot')
            saved = {key: globals()[key] for key in names}
            try:
                globals()['SequenceRunner'] = FakeRunner
                globals()['make_model_and_head'] = lambda *_: (object(), FakeHead(), 'mock')
                def forbidden_GT(*_): raise AssertionError('GT was read during the latency trial')
                globals()['load_gt'] = forbidden_GT
                globals()['gpu_snapshot'] = lambda *_: {'available':True, 'mapping_verified':True,
                                                       'process_query_verified':True, 'foreign_processes':[]}
                arm = Arm('F', 'once', 'C', 1, 8)
                frames, meta, samples = run_trial(fake_rt, args, 'synthetic', [{'rgb_path':'unused'}]*64,
                                                  arm, 0, 0, [None]*64, None, {}, {})
                self.assertEqual(len(calls), 32+64)
                self.assertEqual((calls[0], calls[32]), ((0,1),(0,1)))
                self.assertEqual(len(frames), 64)
                self.assertEqual(meta['repair_head_calls'], 2)
                self.assertEqual(meta['native_transition_calls'], 62)
                self.assertEqual(set(samples), {0,9,32,63})
                self.assertTrue(meta['endpoint_checks_clean'])
            finally:
                globals().update(saved)

        def test_quality_pass_integration_with_missing_GT(self):
            # Exercise the complete offline loop and reference/reset checks; no model/GPU.
            gt = np.linspace(1, 2, 1600).reshape(40, 40).astype(np.float32)
            fit = {'scale': .9, 'bias_m': .05}
            arms = arm_specs(('C',), 8)
            def prediction(arm, index):
                age = index % K
                base = gt * (1 + .001 * age)
                if arm.kind == 'calibration': return np.maximum(.9*base + .05, 1e-4)
                if arm.kind == 'output' and age: return base*.99
                if arm.kind == 'every' and age: return base*.98
                if arm.kind == 'once' and age > 8: return base*.97
                return base
            class FakeRunner:
                def __init__(self, _rt, _model, _shape, _device, arm, _head, _fit): self.arm = arm
                def step(self, idx, _rec, **_): return prediction(self.arm, idx), {}, {}
            rows = [{'rgb_path': f'/rgb/{i}.png', 'depth_path': f'/gt/{i}.png',
                     'gt_match_status': 'UNMATCHED' if i == 7 else 'MATCHED'} for i in range(64)]
            qref, yref = {}, {}
            qmap = {'A':'A_native', 'B1':'B1_global_scale_bias', 'B3':'B3_output_adapter_trained_output_only',
                    'S2_C_gain0.5':'S2_C_gain0.5_R1_F_only'}
            for arm in arms:
                for idx in range(64):
                    m = score_prediction(gt, prediction(arm, idx), np)
                    record = {key: m['raw_' + key] for key in METRICS[:3]} | {'valid_pixels': m['valid_pixels']}
                    if arm.name in qmap: qref[('synthetic', qmap[arm.name], idx)] = record
                    if arm.kind == 'once' and idx % K > 8:
                        yref[('synthetic', 'C', idx//K, 8, 'F_only', idx)] = record
            fake_rt = SimpleNamespace(np=np, torch=SimpleNamespace(no_grad=lambda: __import__('contextlib').nullcontext(),
                                       cuda=SimpleNamespace(empty_cache=lambda: None)),
                                      exp1=SimpleNamespace(shape_info_for=lambda *_: {'original_shape':(40,40)}))
            args = SimpleNamespace(device='cuda:0', snapshot='unused', once_age=8, no_examples=True,
                                   benchmark_frames=64, reference_tolerance=2e-5)
            names = ('SequenceRunner', 'make_model_and_head', 'load_gt', 'write_csv')
            saved = {key: globals()[key] for key in names}
            output_writes = {}
            try:
                globals()['SequenceRunner'] = FakeRunner
                globals()['make_model_and_head'] = lambda *_: (object(), None, 'mock')
                globals()['load_gt'] = lambda *_: gt
                globals()['write_csv'] = lambda path, data: output_writes.update({str(path): list(data)})
                timed = [{'arm':arm.name, 'predictions':{9:prediction(arm,9).copy()}} for arm in arms]
                rr, inv = quality_pass(fake_rt, args, 'synthetic', rows, arms, qref, yref,
                                       None, {}, fit, Path('in_memory_test_output'), timed_samples=timed)
                self.assertEqual(len(rr), len(arms)*63)
                self.assertEqual(len(inv), len(rr))
                self.assertEqual(len(output_writes), len(arms))
                self.assertTrue(all((r['reference_max_metric_deviation'] or 0) < 1e-12 for r in inv))
                self.assertEqual(sum(r['timed_output_sample_comparisons'] for r in inv), len(arms))
                qs, cycles = quality_summary(rr, np, 64)
                self.assertTrue(all(r['complete_rgb_cycles'] == 2 for r in qs))
                self.assertEqual(len(cycles), len(arms)*2)
                timed[0]['predictions'][9][0,0] += .1
                with self.assertRaisesRegex(RuntimeError, 'Timed/quality output mismatch'):
                    quality_pass(fake_rt, args, 'synthetic', rows, arms, qref, yref,
                                 None, {}, fit, Path('in_memory_test_output'), timed_samples=timed)
            finally:
                globals().update(saved)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({'self_test_passed': result.wasSuccessful(), 'tests_run': result.testsRun,
                      'real_model_or_CUDA_executed': False}))
    return 0 if result.wasSuccessful() else 1


def main():
    a = parse_args()
    if a.self_test:
        return self_test()
    root = Path(__file__).resolve().parents[2]
    if not (root/'experiments/p1').is_dir() or not (root/'velodepth').is_dir():
        raise RuntimeError('Install this script under <repo>/experiments/p1 and run from the repository root')
    os.chdir(root)
    out = Path(a.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'y06_summary.json').exists() or (out/'y06_failure.json').exists():
        raise RuntimeError('Output directory already contains a completed/failed run; choose a new --out-dir')
    lock = out/'y06_run.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    sources = {}
    started = time.perf_counter()
    try:
        rt, sources = runtime_imports(root)
        rt.torch.set_num_threads(a.cpu_threads)
        rt.cv2.setNumThreads(a.opencv_threads)
        rt.torch.manual_seed(a.seed)
        rt.np.random.seed(a.seed)
        report, manifests, plans, qref, yref, b3path, cpaths, fit, arms = preflight(rt, a, root, sources)
        write_json(out/'y06_preflight.json', report)
        print('[PREFLIGHT PASS] fixed K32; no training; clean benchmark separated from GT diagnostics', flush=True)
        if a.preflight_only:
            return 0
        if not rt.torch.cuda.is_available() or not a.device.startswith('cuda'):
            raise RuntimeError('CUDA is required for the frozen VeloDepth runtime')
        rt.torch.cuda.set_device(rt.torch.device(a.device))
        results = {}
        if a.task in ('both', 'benchmark'):
            results['benchmark'] = benchmark(rt, a, manifests, qref, yref, b3path, cpaths, fit, arms, report, out)
        if a.task in ('both', 'history'):
            results['history'] = history_diagnostics(rt, a, manifests, plans, qref, out)
        verify_frozen_inputs(report, root)
        summary = {'status': 'PILOT_COMPLETE', 'mechanics_status': 'PASS', 'version': VERSION,
                   'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'frozen_inputs_verified_after_run': True,
                   'partial_smoke': a.smoke_only, 'training_performed': False, 'K': K,
                   'results': results, 'total_diagnostic_runtime_sec': time.perf_counter()-started,
                   'script_sha256': digest(Path(__file__)), 'preflight_sha256': digest(out/'y06_preflight.json'),
                   'claim_limits': report['claim_limits'],
                   'route_success': 'NOT_AUTOMATICALLY_DECIDED; assess quality, risk, measured cost and independent data'}
        write_json(out/'y06_summary.json', summary)
        package = package_review(out, root, sources)
        print(f'[DONE] {package}; total runtime is diagnostic runtime, not deployment latency', flush=True)
        return 0
    except Exception:
        failure = {'status': 'FAILED', 'version': VERSION, 'traceback': traceback.format_exc(),
                   'elapsed_sec': time.perf_counter()-started, 'scientific_route_passed': False}
        write_json(out/'y06_failure.json', failure)
        if sources:
            try:
                package_review(out, root, sources)
            except Exception:
                pass
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
