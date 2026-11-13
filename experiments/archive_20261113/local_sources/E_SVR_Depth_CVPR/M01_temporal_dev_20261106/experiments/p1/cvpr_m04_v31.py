#!/usr/bin/env python3
"""CVPR-M04 ClosedLoop-SameHead V3.1 (development-only implementation).

Uses exactly the E4-C Best checkpoint (CrossSource-v2 P_multi, step 700).
No legacy Y04 P_multi multi-seed checkpoint substitution is permitted.

Four continuous strategies from the same frozen E4-C P_multi checkpoint:
A_off          : D0; next=(r0,c0,F0)
O_same_head    : D1; next=(r0,c0,F0)
F_only         : D0; next=(r0,c0,F1)
P_full         : D1; next=(r1,c1,F1)
All retain the original rays; K16 natural refresh, no per-window resets.

These are independent continuous trajectories, not same-parent h0 interventions.
The O/F diagnostic arms intentionally execute both off and pre_readout
for each fast step. Their actual additional computation MUST be reported.

Model integration uses the repository's byte-pinned compatibility runtime and the E4-C
CrossSource-v2 helper provenance. Never modifies the base model
or checkpoint files, never trains, never accesses test data by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# M04 is a module of the unified SVR-Depth repository. For E4-C compatibility,
# inference imports the byte-pinned legacy VeloDepth runtime shipped INSIDE this
# repository, while the E4-C training helpers come from the current CrossSource-v2
# helper directory. No second repository or Conda environment is created.
ROOT = PROJECT_ROOT / "outputs" / "legacy_runtime"
E4C_HELPERS = PROJECT_ROOT / "experiments" / "crosssource_v2" / "legacy_helpers"
sys.path[:0] = [str(ROOT), str(E4C_HELPERS), str(ROOT / "experiments" / "p1")]
K = 16
GAIN = 1.0
EVALUATION_RNG_SEED = 20261103
FREQUENCY = 1
BRANCHES = ('A_off', 'O_same_head', 'F_only', 'P_full')
SEQUENCES = ('crowd2', 'crowd3', 'balloon2')
EXPECTED_BASE_SHA256 = '615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e'
EXPECTED_E4C_CHECKPOINT_SHA256 = '0447fbcc25d1ea93d5a80e7f8a7a3b38ac911a8465951009da53988e712754c0'
EXPECTED_E4C_INIT_HEAD_SHA256 = 'fbb01f8d81da722fc73c34dff7b833800d32270f1ca3d7616830ce869c77c16f'
EXPECTED_E4C_PARAMS = 170496
EXPECTED_E4C_STEP = 700
EXPECTED_E4C_SEED = 20261103
EXPECTED_E4C_TRAIN_STEPS = 1000
E4C_DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" /
    "crosssource_v2_E4_20261104_C_s20261103_1000step" /
    "best_checkpoint.pt"
)
E4C_DEFAULT_CONFIG = (
    PROJECT_ROOT / "configs" / "crosssource_v2" / "generated_20261103" /
    "C_tartanair_kitti.json"
)
PINNED_SOURCES = {
    "experiments/crosssource_v2/legacy_helpers/gt2_fixed_clip_repair_train.py":
        "50a8da294c87ff5ee43131d0c04fddd9227fa0ea76041a354d86c9f703a267b3",
    "experiments/crosssource_v2/legacy_helpers/gt4a2_exp1_train_eval.py":
        "911da6b1a8e0bfa21471d86816de823ababf561ba8a5b20df52d0d51a97dfd83",
    "velodepth/models/velodepth.py":
        "739c5f27a749174987b9a36edaac32fb98070e87dbef0a0a52f7430d61149479",
    "velodepth/models/encoder_velo.py":
        "3c5aa2a3a6dce03b68373953037fe8a0313a2cab36d66e5e3b88a56ec4a7ba8a",
    "velodepth/models/decoder_velo.py":
        "35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08",
    "velodepth/models/persistent_feature_repair.py":
        "6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c",
}
MANIFESTS = {
    'crowd2': ('outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv', 'f7bf9226edcedf6c05d63938e29f92bf9364878c9291dbd79172c973c296f350'),
    'crowd3': ('outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv', '9f4da8360b8f6b3039330dc7fe9dba6dd7219d9427e774b89006b09269fa17ad'),
    'balloon2': ('outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv', 'afbd46ebed192aaf69f580822993fb7c756c5656de0595f68143993859918b06'),
}


def resolved(raw: str | Path) -> Path:
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jdump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f'Refuse to write empty required CSV: {path}')
    cols = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)


def require_hash(path: Path, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f'{label}: missing {path}')
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f'{label}: SHA256 mismatch: {path}\nexpected={expected}\nactual={actual}')
    return actual


def parse_e4c_checkpoint(path: Path, *, with_torch: bool) -> list[dict]:
    """Validate the single, fixed E4-C Best checkpoint; never substitute Y04 seeds."""
    if not path.is_file():
        raise FileNotFoundError(f"E4-C Best checkpoint missing: {path}")
    actual = require_hash(path, EXPECTED_E4C_CHECKPOINT_SHA256, "E4-C Best checkpoint")
    rec = {
        "seed": EXPECTED_E4C_SEED,
        "checkpoint_kind": "E4-C_Best",
        "checkpoint_path": str(path.resolve()),
        "sha256": actual,
        "expected_sha256": EXPECTED_E4C_CHECKPOINT_SHA256,
        "step": EXPECTED_E4C_STEP,
    }
    if with_torch:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        identity = ckpt.get("identity", {})
        expected_identity = {
            "protocol": "SVR-Depth-CrossSource-v2",
            "method": "P_multi",
            "seed": EXPECTED_E4C_SEED,
            "initialization_seed": 20261103,
            "steps": 1000,
            "window_frames": 20,
            "tbptt_frames": 4,
            "snapshot_sha256": EXPECTED_BASE_SHA256,
        }
        mismatches = {
            key: {"actual": identity.get(key), "expected": value}
            for key, value in expected_identity.items()
            if identity.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"E4-C identity mismatch: {mismatches}")
        if int(ckpt.get("step", -1)) != EXPECTED_E4C_STEP:
            raise RuntimeError(f"Expected E4-C Best step {EXPECTED_E4C_STEP}, got {ckpt.get('step')}")
        if ckpt.get("init_head_sha256") != EXPECTED_E4C_INIT_HEAD_SHA256:
            raise RuntimeError("E4-C initialization-head SHA256 mismatch")
        head_state = ckpt.get("head_state")
        if not isinstance(head_state, dict) or not head_state:
            raise RuntimeError("E4-C checkpoint missing non-empty head_state")
        nparams = sum(
            int(v.numel()) for v in head_state.values()
            if isinstance(v, torch.Tensor)
        )
        if nparams != EXPECTED_E4C_PARAMS:
            raise RuntimeError(
                f"E4-C repair-head parameter mismatch: {nparams} != {EXPECTED_E4C_PARAMS}"
            )
        recorded_sources = identity.get("source_sha256", {})
        for rel, expected in PINNED_SOURCES.items():
            if rel.startswith("experiments/crosssource_v2/"):
                recorded = recorded_sources.get(rel)
                if recorded != expected:
                    raise RuntimeError(
                        f"E4-C helper provenance mismatch for {rel}: "
                        f"{recorded!r} != {expected}"
                    )
        # Cross-check the run identity and config that produced this checkpoint.
        run_dir = path.parent
        identity_path = run_dir / "run_identity.json"
        if not identity_path.is_file():
            raise FileNotFoundError(f"E4-C run_identity.json missing: {identity_path}")
        disk_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if disk_identity != identity:
            raise RuntimeError("E4-C checkpoint identity differs from run_identity.json")
        config_path = E4C_DEFAULT_CONFIG
        if not config_path.is_file():
            raise FileNotFoundError(f"E4-C training config missing: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if sha(config_path) != identity.get("config_sha256"):
            raise RuntimeError("E4-C config SHA256 differs from checkpoint identity")
        dataset_weights = config.get("training", {}).get("dataset_weights")
        if dataset_weights != {"tartanair": 0.5, "kitti_sceneflow": 0.5}:
            raise RuntimeError(f"Unexpected E4-C dataset weights: {dataset_weights}")
        rec.update({
            "identity": identity,
            "step": int(ckpt["step"]),
            "head_parameters": nparams,
            "init_head_sha256": ckpt["init_head_sha256"],
            "config_path": str(config_path.resolve()),
            "config_sha256": sha(config_path),
            "best_val_loss": ckpt.get("best_val_loss"),
        })
        del ckpt
    return [rec]


def source_hashes() -> dict:
    out = {}
    for rel, expected in PINNED_SOURCES.items():
        base = PROJECT_ROOT if rel.startswith("experiments/crosssource_v2/") else ROOT
        out[rel] = require_hash(base / rel, expected, rel)
    return out


def audit_manifests() -> tuple[dict[str, list[dict]], list[dict]]:
    data, audit = {}, []
    pair_root_raw = os.environ.get("M04_PAIR_ROOT", "").strip()
    for seq, (name, expected) in MANIFESTS.items():
        if pair_root_raw:
            path = (Path(pair_root_raw).expanduser().resolve() / Path(name).name)
        else:
            path = resolved(name)
        require_hash(path, expected, f'pair manifest: {seq}')
        with path.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise RuntimeError(f'Empty manifest: {seq}')
        rows.sort(key=lambda r: int(r['frame_index_zero_based']))
        ids = [int(r['frame_index_zero_based']) for r in rows]
        if ids != list(range(len(rows))):
            raise RuntimeError(f'Noncontiguous or nonzero start RGB index: {seq}')
        tprev = -float('inf')
        seen_gt = set()
        count_gt = 0
        for r in rows:
            ts = float(r['rgb_timestamp'])
            if not math.isfinite(ts) or ts <= tprev:
                raise RuntimeError(f'Invalid RGB timestamp: {seq} frame={r["frame_index_zero_based"]}')
            tprev = ts
            rgb = Path(r['rgb_path'])
            if not rgb.is_file():
                raise FileNotFoundError(rgb)
            if str(r['gt_match_status']).upper() == 'MATCHED':
                gt = Path(r['depth_path'])
                if not gt.is_file() or str(gt.resolve()) in seen_gt:
                    raise RuntimeError(f'Missing/duplicate GT in {seq}: {gt}')
                seen_gt.add(str(gt.resolve()))
                if float(r['abs_delta_ms']) > 20.0 or float(r['abs_delta_ms']) < 0.0:
                    raise RuntimeError(f'GT delta exceeds 20 ms: {seq}')
                if str(r['gt_reused']).strip().lower() in ('true', 'yes', '1'):
                    raise RuntimeError(f'GT reused: {seq}')
                count_gt += 1
        data[seq] = rows
        audit.append({'sequence': seq, 'path': str(path), 'sha256': expected,
                      'rgb_frames': len(rows), 'matched_gt': count_gt,
                      'unmatched_gt': len(rows)-count_gt, 'full_K16_cycles': len(rows)//K,
                      'tail_frames': len(rows)%K})
    return data, audit


def verify_import_origins() -> dict:
    """Ensure M04 loads the E4-C-compatible runtime and exact CrossSource helpers."""
    import importlib
    vd = importlib.import_module("velodepth.models.velodepth")
    gt2 = importlib.import_module("gt2_fixed_clip_repair_train")
    exp1 = importlib.import_module("gt4a2_exp1_train_eval")
    origins = {
        "velodepth": str(Path(vd.__file__).resolve()),
        "gt2_helper": str(Path(gt2.__file__).resolve()),
        "exp1_helper": str(Path(exp1.__file__).resolve()),
    }
    if not Path(vd.__file__).resolve().is_relative_to((ROOT / "velodepth").resolve()):
        raise RuntimeError(f"M04 loaded non-pinned VeloDepth module: {origins['velodepth']}")
    if Path(gt2.__file__).resolve() != (
        PROJECT_ROOT / "experiments/crosssource_v2/legacy_helpers/gt2_fixed_clip_repair_train.py"
    ).resolve():
        raise RuntimeError(f"M04 loaded unexpected E4-C gt2 helper: {origins['gt2_helper']}")
    if Path(exp1.__file__).resolve() != (
        PROJECT_ROOT / "experiments/crosssource_v2/legacy_helpers/gt4a2_exp1_train_eval.py"
    ).resolve():
        raise RuntimeError(f"M04 loaded unexpected E4-C exp1 helper: {origins['exp1_helper']}")
    return origins


def preflight(args) -> tuple[dict, dict[str, list[dict]]]:
    if not args.snapshot:
        raise RuntimeError('--snapshot or VDEEP_SNAPSHOT must be supplied')
    snap = resolved(args.snapshot)
    require_hash(snap / 'model.safetensors', EXPECTED_BASE_SHA256, 'VeloDepth backbone')
    if not (snap / 'config.json').is_file():
        raise FileNotFoundError(snap / 'config.json')
    src = source_hashes()
    import_origins = verify_import_origins()
    checkpoint_path = Path(args.e4c_checkpoint).expanduser().resolve()
    checkpoints = parse_e4c_checkpoint(checkpoint_path, with_torch=True)
    data, manifest_info = audit_manifests()
    evidence = {
        'experiment': 'CVPR-M04 ClosedLoop-SameHead V3.1',
        'role': 'development-only; no held-out test',
        'weight_source': 'E4-C Best checkpoint only',
        'K': K, 'gain': GAIN, 'frequency': FREQUENCY,
        'branches': list(BRANCHES),
        'model_snapshot': str(snap),
        'base_model_sha256': EXPECTED_BASE_SHA256,
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': checkpoints[0]['sha256'],
        'checkpoint_kind': 'E4-C_Best',
        'checkpoint_step': 700,
        'checkpoint_seed': EXPECTED_E4C_SEED,
        'evaluation_rng_seed': EVALUATION_RNG_SEED,
        'runner_sha256': sha(Path(__file__).resolve()),
        'checkpoints': checkpoints,
        'sources': src, 'import_origins': import_origins, 'data': manifest_info,
        'model_runtime': str(ROOT),
        'pair_root': os.environ.get('M04_PAIR_ROOT', str(ROOT / 'outputs/gt4a2_exp1')),
        'model_clock': 'frozen 16-frame refresh, no adaptive restart',
        'metric_contract': 'Bonn GT uint16 /5000, z-depth, GT-valid fixed mask, invalid pred is a failure',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'python': sys.version, 'platform': platform.platform(),
    }
    return evidence, data


def copy_state(s: dict) -> dict:
    return {'radius': s['radius'].detach().clone(),
            'confidence': s['confidence'].detach().clone(),
            'features': [x.detach().clone() for x in s['features']],
            'rays_map': s['rays_map'].detach().clone()}


def cpu_state(s: dict) -> dict:
    return {'radius':s['radius'].detach().cpu().clone(),
            'confidence':s['confidence'].detach().cpu().clone(),
            'features':[x.detach().cpu().clone() for x in s['features']],
            'rays_map':s['rays_map'].detach().cpu().clone()}


def maxdiff(a, b) -> float:
    import torch
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            return float('inf')
        if a.numel() == 0:
            return 0.0
        d = (a.float() - b.float()).abs()
        return float(d.max().item()) if bool(torch.isfinite(d).all()) else float('inf')
    if isinstance(a, dict):
        return max((maxdiff(a[k], b[k]) for k in a), default=0.0) if a.keys() == b.keys() else float('inf')
    if isinstance(a, (list, tuple)):
        return max((maxdiff(x,y) for x,y in zip(a,b)), default=0.0) if len(a) == len(b) else float('inf')
    return 0.0 if a == b else float('inf')


def make_rng(torch):
    return {'cpu': torch.get_rng_state().clone(),
            'cuda': [x.clone() for x in torch.cuda.get_rng_state_all()]}


def put_rng(torch, s):
    torch.set_rng_state(s['cpu'])
    torch.cuda.set_rng_state_all(s['cuda'])


def forward_step(model, exp1, prev, current, static, parent: dict, mode: str, torch) -> dict:
    model.pixel_decoder.feature_repair_mode = mode
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        r,c,f = exp1.one_fast_step(model, prev, current, static,
                                    parent['radius'], parent['confidence'], parent['features'])
    return {'radius': r.detach(), 'confidence': c.detach(),
            'features': [x.detach() for x in f], 'rays_map': parent['rays_map'].detach()}


def apply_branch(model, exp1, gt2, prev, current, static, state, branch, shape, torch):
    """Returns (new committed state, current depth on CPU, factual call counts)."""
    if branch in ('A_off', 'P_full'):
        which = 'off' if branch == 'A_off' else 'pre_readout'
        s = forward_step(model, exp1, prev, current, static, copy_state(state), which, torch)
        committed, visible = s, s
        steps = 1
    else:
        # Freeze RNG and independent tensor inputs for both candidate computations.
        seed_rng = make_rng(torch)
        put_rng(torch, seed_rng)
        s0 = forward_step(model, exp1, prev, current, static, copy_state(state), 'off', torch)
        put_rng(torch, seed_rng)
        s1 = forward_step(model, exp1, prev, current, static, copy_state(state), 'pre_readout', torch)
        if branch == 'O_same_head':
            committed, visible = s0, s1
        elif branch == 'F_only':
            committed, visible = copy_state(s0), s0
            committed['features'] = [x.detach().clone() for x in s1['features']]
        else:
            raise RuntimeError(f'Unknown branch: {branch}')
        steps = 2
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        z = gt2.make_z_depth(visible['radius'], visible['rays_map'], shape['padded_shape'],
                             shape['paddings'], model.interpolation_mode)
    return copy_state(committed), z.detach().float().cpu().numpy(), steps


def gt_metrics(row: dict, pred, shape: dict) -> dict:
    import cv2
    import numpy as np
    if str(row['gt_match_status']).strip().upper() != 'MATCHED':
        return {'metric_status': 'NO_GT', 'AbsRel': '', 'RMSE': '', 'delta1': '',
                'valid_ratio': '', 'gt_valid_ratio': '', 'valid_pixels': 0,
                'invalid_pred_gt_pixels': 0, 'total_pixels': ''}
    raw = cv2.imread(row['depth_path'], cv2.IMREAD_UNCHANGED)
    if raw is None or raw.ndim != 2 or raw.dtype != np.uint16:
        raise RuntimeError(f'Invalid GT PNG format: {row["depth_path"]}')
    if raw.shape != shape['original_shape'] or pred.shape != raw.shape:
        raise RuntimeError(f'RGB/GT/pred shape inconsistency at {row["frame_index_zero_based"]}')
    gt = raw.astype(np.float32)/5000.0
    m = np.isfinite(gt) & (gt > 0)
    n_gt = int(m.sum())
    if n_gt == 0:
        return {'metric_status': 'NO_VALID_GT', 'AbsRel': '', 'RMSE': '', 'delta1': '',
                'valid_ratio': 0, 'gt_valid_ratio': 0, 'valid_pixels': 0,
                'invalid_pred_gt_pixels': 0, 'total_pixels': int(gt.size)}
    ok = np.isfinite(pred) & (pred > 0)
    bad = int((m & ~ok).sum())
    if bad > 0:
        return {'metric_status': 'INVALID_PRED', 'AbsRel': '', 'RMSE': '', 'delta1': '',
                'valid_ratio': float((m & ok).sum()/gt.size),
                'gt_valid_ratio': n_gt/int(gt.size), 'valid_pixels': n_gt-bad,
                'invalid_pred_gt_pixels': bad, 'total_pixels': int(gt.size)}
    g = gt[m].astype(np.float64)
    p = pred[m].astype(np.float64)
    ratio = np.maximum(g/p, p/g)
    return {'metric_status': 'OK', 'AbsRel': float(np.mean(abs(g-p)/g)),
            'RMSE': float(np.sqrt(np.mean((p-g)**2))),
            'delta1': float(np.mean(ratio<1.25)),
            'valid_ratio': n_gt / gt.size,
            'gt_valid_ratio': n_gt / gt.size, 'valid_pixels': n_gt,
            'invalid_pred_gt_pixels': 0, 'total_pixels': int(gt.size)}


class ModuleLedger:
    """Counts true forward hooks. Modules are nested, never add counts as FLOPs."""
    def __init__(self, model):
        self.count = Counter()
        self.handles = []
        for name, module in (
            ('pixel_encoder',model.pixel_encoder),
            ('small_encoder',model.small_encoder),
            ('pixel_decoder',model.pixel_decoder),
            ('radial_forward_fast',model.pixel_decoder.radial_module),
            ('persistent_feature_repair',model.pixel_decoder.persistent_feature_repair),
        ):
            if module is not None:
                self.handles.append(module.register_forward_hook(
                    lambda _mod,_inp,_out,n=name: self.count.update((n,))
                ))
    def close(self):
        for handle in self.handles:handle.remove()


def setup_model(snapshot: Path, ck: dict, torch):
    import gt2_fixed_clip_repair_train as gt2
    model = gt2.load_model(snapshot,'pre_readout',torch.device('cuda'))
    payload = torch.load(ck['checkpoint_path'], map_location='cpu', weights_only=False)
    model.pixel_decoder.persistent_feature_repair.load_state_dict(
        payload['head_state'], strict=True)
    model.eval()
    model.pixel_decoder.persistent_feature_repair.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def stream_sequence(model, rows, branch, sequence, seed, timing_repeat, shape, torch,
                    *, canary: bool = False, max_frames: int | None = None):
    import cv2
    import gt4a2_exp1_train_eval as exp1
    import gt2_fixed_clip_repair_train as gt2
    count = ModuleLedger(model)
    frames_out, clocks, states_digest = [], [], []
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    torch.cuda.reset_peak_memory_stats()
    prev = None
    state = None
    n = min(max_frames, len(rows)) if max_frames is not None else len(rows)
    try:
        for idx in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            current = exp1.load_frame(rows[idx], shape, torch.device('cuda'), need_gt=False)
            if idx % K == 0:
                model.pixel_decoder.feature_repair_mode = 'off'
                rays, radius, conf, features = exp1.keyframe_state(model, current, shape)
                state = {'radius': radius.detach(), 'confidence': conf.detach(),
                         'features': [f.detach() for f in features], 'rays_map': rays.detach()}
                pred = None
                with torch.no_grad(), torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):
                    z = gt2.make_z_depth(state['radius'], state['rays_map'], shape['padded_shape'],
                                         shape['paddings'],model.interpolation_mode)
                pred = z.detach().float().cpu().numpy()
                forward_calls = 0
                stage = 'keyframe'
            else:
                static = exp1.transition_static(model, prev, current, shape, dis)
                state, pred, forward_calls = apply_branch(
                    model, exp1, gt2, prev, current, static, state, branch, shape, torch)
                stage = 'fast'
            torch.cuda.synchronize()
            elapsed = 1000*(time.perf_counter()-t0)
            # GT reading and metrics are explicitly excluded from timed region.
            clocks.append({'seed':seed,'sequence':sequence,'branch':branch,
                           'repeat':timing_repeat,'frame_index':idx,'age':idx%K,
                           'stage':stage,'total_wall_ms':elapsed,'fast_forward_calls':forward_calls})
            if timing_repeat == 0:
                metric = gt_metrics(rows[idx], pred, shape)
                frames_out.append({'seed':seed,'sequence':sequence,'branch':branch,
                    'frame_index':idx,'age':idx%K,'stage':stage,
                    'rgb_path':rows[idx]['rgb_path'],'gt_path':rows[idx].get('depth_path',''),
                    'gt_match_status':rows[idx]['gt_match_status'],
                    'gt_delta_ms':rows[idx].get('abs_delta_ms',''),
                    **metric})
            if canary:
                # Expensive snapshot serialization is only done for the tiny Canary run.
                states_digest.append({"state":cpu_state(state), "prediction":pred.copy()})
            prev = current
            if idx == 0 or (idx+1)%100==0 or idx==n-1:
                print(f'[M04] seed={seed} seq={sequence} branch={branch} '
                      f'rep={timing_repeat} frame={idx+1}/{n}', flush=True)
    finally:
        counts = dict(count.count)
        count.close()
    ledger=[{'seed':seed,'sequence':sequence,'branch':branch,'repeat':timing_repeat,
             'module':k,'calls':v,'nested_not_additive':True} for k,v in sorted(counts.items())]
    ledger.append({'seed':seed,'sequence':sequence,'branch':branch,'repeat':timing_repeat,
        'module':'CUDA_memory_peak','calls':'','nested_not_additive':'',
        'peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
        'peak_reserved_bytes':int(torch.cuda.max_memory_reserved())})
    return frames_out, clocks, ledger, states_digest


def summarize(frames: list[dict], clocks: list[dict]) -> tuple[list[dict],list[dict]]:
    from collections import defaultdict
    by = defaultdict(list)
    for row in frames:by[(row['seed'],row['sequence'],row['branch'])].append(row)
    result = []
    for (seed,seq,branch),grp in sorted(by.items()):
        valid = [x for x in grp if x['metric_status']=='OK']
        errors = [x for x in grp if x['metric_status'] in ('INVALID_PRED','NO_VALID_GT')]
        a = [float(x['AbsRel']) for x in valid]
        r = [float(x['RMSE']) for x in valid]
        d = [float(x['delta1']) for x in valid]
        complete = not errors and len(valid)>0
        result.append({'seed':seed,'sequence':seq,'branch':branch,'status': 'COMPLETE' if complete else 'INCOMPLETE',
            'all_frames':len(grp),'gt_evaluated_frames':len(valid),
            'frames_without_gt':sum(x['metric_status']=='NO_GT' for x in grp),
            'invalid_or_empty_gt_frames':len(errors),
            'AbsRel_macro_frames':statistics.mean(a) if complete else '',
            'RMSE_macro_frames':statistics.mean(r) if complete else '',
            'delta1_macro_frames':statistics.mean(d) if complete else '',
            'AbsRel_p95_frames':percentile(a,95) if complete else ''})
    bytime=defaultdict(list)
    for x in clocks:bytime[(x['seed'],x['sequence'],x['branch'],x['repeat'])].append(x)
    clock_summary=[]
    for (seed,seq,b,rep), rs in sorted(bytime.items()):
        v=[float(x['total_wall_ms']) for x in rs]
        clock_summary.append({'seed':seed,'sequence':seq,'branch':b,'repeat':rep,
            'frames':len(v),'mean_ms':statistics.mean(v),'p50_ms':percentile(v,50),
            'p95_ms':percentile(v,95),
            'keyframes':sum(x['stage']=='keyframe' for x in rs),
            'fast_forwards':sum(int(x['fast_forward_calls']) for x in rs)})
    return result,clock_summary


def percentile(values: list[float], q: float) -> float:
    if not values:raise ValueError('Empty data for percentile')
    v=sorted(values)
    pos=(len(v)-1)*q/100
    lo=int(pos); hi=min(lo+1,len(v)-1)
    return v[lo]+(v[hi]-v[lo])*(pos-lo)


def paired_effects(frames: list[dict]) -> list[dict]:
    groups=defaultdict(dict)
    for row in frames:
        groups[(row['seed'],row['sequence'],row['frame_index'])][row['branch']]=row
    out=[]
    for (seed,seq,idx),g in sorted(groups.items()):
        if set(g)!=set(BRANCHES):
            raise RuntimeError(f'Incomplete branch coverage: {(seed,seq,idx)} got={list(g)}')
        if len({x['gt_match_status'] for x in g.values()})!=1:
            raise RuntimeError(f'GT pairing mismatch: {seed},{seq},{idx}')
        rec={'seed':seed,'sequence':seq,'frame_index':idx,'age':idx%K,
             'metric_status': 'OK' if all(x['metric_status']=='OK' for x in g.values()) else 'INCOMPLETE',
             'P_minus_O_AbsRel':'','P_minus_A_AbsRel':'','F_minus_A_AbsRel':'',
             'P_harms_O': '', 'P_severe_harms_O':'', 'P_harms_A':'', 'P_severe_harms_A':''}
        if rec['metric_status']=='OK':
            a,o,f,p = (g[z] for z in BRANCHES)
            rec.update({'P_minus_O_AbsRel':float(p['AbsRel'])-float(o['AbsRel']),
                        'P_minus_A_AbsRel':float(p['AbsRel'])-float(a['AbsRel']),
                        'F_minus_A_AbsRel':float(f['AbsRel'])-float(a['AbsRel']),
                        'P_harms_O': int(float(p['AbsRel'])>float(o['AbsRel'])),
                        'P_severe_harms_O':int(float(p['AbsRel'])-float(o['AbsRel'])>0.01),
                        'P_harms_A':int(float(p['AbsRel'])>float(a['AbsRel'])),
                        'P_severe_harms_A':int(float(p['AbsRel'])-float(a['AbsRel'])>0.01)})
        out.append(rec)
    return out


def maxdiff_tensors_numpy(a,b) -> float:
    import numpy as np
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf')
    return float(np.max(np.abs(a.astype('float64')-b.astype('float64'))))


def summarize_paired_effects(paired: list[dict]) -> list[dict]:
    """Descriptive only: no frame/seed pseudoreplication or significance claims."""
    groups=defaultdict(list)
    for rec in paired:
        groups[(rec['seed'],rec['sequence'])].append(rec)
    output=[]
    for (seed,seq),records in sorted(groups.items()):
        ok=[r for r in records if r['metric_status']=='OK']
        # Expected absent GT is not an error: distinguish via raw metrics in summary.
        # Any true INVALID_PRED is caught by sequence_summary and invalidates this row.
        # Here we only write observed paired summary and expose N.
        row={'seed':seed,'sequence':seq,'paired_OK':len(ok),'unpaired_or_no_gt':len(records)-len(ok)}
        for key in ('P_minus_O_AbsRel','P_minus_A_AbsRel','F_minus_A_AbsRel',
                    'P_harms_O','P_severe_harms_O','P_harms_A','P_severe_harms_A'):
            row[key+'_mean']=statistics.mean(float(r[key]) for r in ok) if ok else ''
        output.append(row)
    return output


def canary(args, evidence, data) -> int:
    import torch
    import gt4a2_exp1_train_eval as exp1
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for Canary; this package is not GPU-tested here')
    ck=evidence['checkpoints'][0]
    model=setup_model(resolved(args.snapshot),ck,torch)
    seq='crowd2'
    shape=exp1.shape_info_for(model,Path(data[seq][0]['rgb_path']),torch.device('cuda'))
    outdir=make_dir(args.out_dir,'cvpr_m04_canary')
    jdump(outdir/'protocol_lock.json',evidence)
    by_branch={}
    records=[]
    for branch in BRANCHES:
        torch.manual_seed(EVALUATION_RNG_SEED)
        torch.cuda.manual_seed_all(EVALUATION_RNG_SEED)
        _,_,_,states=stream_sequence(model,data[seq],branch,seq,ck['seed'],0,shape,torch,
                                      canary=True,max_frames=32)
        by_branch[branch]=states
    assert len(by_branch['A_off'])==len(by_branch['O_same_head'])==32
    tolerance=args.tolerance
    for i in range(32):
        # Output-only must NEVER alter next-state trajectory.
        delta=maxdiff(by_branch['A_off'][i]['state'],by_branch['O_same_head'][i]['state'])
        records.append({'frame_index':i,'check':'O_committed_state_equal_A','max_abs':delta,
                        'pass':delta<=tolerance})
        if i%K==0:
            for other in ('F_only','P_full'):
                delta=maxdiff(by_branch['A_off'][i]['state'],by_branch[other][i]['state'])
                records.append({'frame_index':i,'check':f'keyframe_equal_{other}',
                                'max_abs':delta,'pass':delta<=tolerance})
    # At the first propagation after a shared keyframe, all four parents match.
    # This checks visible output choice, not just committed state equality.
    for i in (1, 17):
        for left,right,name in (("A_off","F_only","D0_A_eq_F"),
                                ("O_same_head","P_full","D1_O_eq_P")):
            delta=maxdiff_tensors_numpy(by_branch[left][i]["prediction"],
                                         by_branch[right][i]["prediction"])
            records.append({'frame_index':i,'check':name,'max_abs':delta,
                            'pass':delta<=tolerance})
    for i in (0,16):
        for branch in BRANCHES[1:]:
            delta=maxdiff_tensors_numpy(by_branch['A_off'][i]['prediction'],
                                         by_branch[branch][i]['prediction'])
            records.append({'frame_index':i,'check':f'keyframe_depth_equal_{branch}',
                            'max_abs':delta,'pass':delta<=tolerance})
    # Audit after same-head implementation: output-only state must equal A all frames;
    # at K keyframes every arm is identical by frozen refresh semantics.
    jdump(outdir/'m04_canary_summary.json',{
       'experiment':'M04 1.0 Canary', 'seed':ck['seed'], 'sequence':seq, 'frames':32,
       'tolerance':tolerance, 'checks':len(records),
       'all_pass':all(r['pass'] for r in records),
       'failures':[r for r in records if not r['pass']]})
    write_rows(outdir/'m04_canary_invariants.csv',records)
    print(f'[M04-CANARY] all_pass={all(r["pass"] for r in records)} report={outdir}',flush=True)
    return 0 if all(r['pass'] for r in records) else 2


def make_dir(raw: str, prefix: str) -> Path:
    p = resolved(raw) if raw else ROOT/'outputs'/f'{prefix}_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    if p.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment directory: {p}')
    p.mkdir(parents=True, exist_ok=False)
    return p


def full(args, evidence, data) -> int:
    import torch
    import gt4a2_exp1_train_eval as exp1
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for M04 full inference')
    report_path = resolved(args.canary_report)
    if not report_path.is_file():
        raise FileNotFoundError('Canary summary is required before full run')
    can = json.loads(report_path.read_text(encoding='utf-8'))
    if can.get('all_pass') is not True or can.get('tolerance') != args.tolerance:
        raise RuntimeError('Canary did not pass with identical tolerance; STOP')
    if can.get('seed') != evidence['checkpoints'][0]['seed']:
        raise RuntimeError('Canary seed identity mismatch')
    # Require matching source/backbone/manifest/3 checkpoint hashes from the original preflight.
    can_lock = json.loads((report_path.parent/'protocol_lock.json').read_text(encoding='utf-8'))
    for key in ('base_model_sha256','checkpoint_sha256','checkpoint_kind','checkpoint_step','checkpoint_seed','runner_sha256','sources','data','checkpoints','branches','K'):
        if can_lock.get(key)!=evidence.get(key):
            raise RuntimeError(f'Canary is stale / identity changed: {key}')
    outdir=make_dir(args.out_dir,'cvpr_m04_closed_loop')
    jdump(outdir/'protocol_lock.json',{
         **evidence,'canary_report':str(report_path), 'timing_repeats':args.timing_repeats,
         'runtime': {'GPU':torch.cuda.get_device_name(0), 'torch':torch.__version__}})
    all_frames=[]; all_time=[]; all_ledger=[]
    status='COMPLETE'
    try:
        for ck in evidence['checkpoints']:
            model=setup_model(resolved(args.snapshot),ck,torch)
            for seq in SEQUENCES:
                shape=exp1.shape_info_for(model,Path(data[seq][0]['rgb_path']),torch.device('cuda'))
                # Deterministic order permutation, not chosen from GT outcomes.
                for repeat in range(args.timing_repeats):
                    rot=(SEQUENCES.index(seq)+ck['seed']+repeat)%len(BRANCHES)
                    order=BRANCHES[rot:]+BRANCHES[:rot]
                    for branch in order:
                        torch.manual_seed(EVALUATION_RNG_SEED)
                        torch.cuda.manual_seed_all(EVALUATION_RNG_SEED)
                        f,t,l,_=stream_sequence(model,data[seq],branch,seq,ck['seed'],repeat,shape,torch)
                        all_frames.extend(f);all_time.extend(t);all_ledger.extend(l)
                        torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()
        summary,timing=summarize(all_frames,all_time)
        paired=paired_effects(all_frames)
        effect_summary=summarize_paired_effects(paired)
        if any(x['status']!='COMPLETE' for x in summary):status='INCOMPLETE_INVALID_PRED'
        write_rows(outdir/'m04_frame_metrics.csv',all_frames)
        write_rows(outdir/'m04_frame_timing.csv',all_time)
        write_rows(outdir/'m04_module_calls.csv',all_ledger)
        write_rows(outdir/'m04_sequence_summary.csv',summary)
        write_rows(outdir/'m04_timing_summary.csv',timing)
        write_rows(outdir/'m04_paired_effects.csv',paired)
        write_rows(outdir/'m04_effect_summary.csv',effect_summary)
        jdump(outdir/'m04_summary.json',{
            'experiment':'CVPR-M04 ClosedLoop-SameHead V3.1',
            'status':status,'training':False,'data_role':'development_only',
            'weight_source':'E4-C Best checkpoint only',
            'checkpoint_sha256':evidence['checkpoint_sha256'],
            'checkpoint_step':evidence['checkpoint_step'],
            'checkpoint_seed':evidence['checkpoint_seed'],
            'runner_sha256':evidence['runner_sha256'],
            'K':K,'seeds':[x['seed'] for x in evidence['checkpoints']],
            'sequence_count':len(SEQUENCES),'branch_count':len(BRANCHES),
            'timing_repeats':args.timing_repeats,
            'primary_contrast':'P_full minus O_same_head sequence-level AbsRel',
            'numeric_success_not_assumed':True,
            'warning':'Only 3 development recordings; no scene-level inferential CI; these closed-loop trajectories diverge',
            'sequence_summary':summary,'timing_summary':timing,
            'paired_effect_summary':effect_summary})
        (outdir/'REPORT.md').write_text(
            '# M04 V3.1 ClosedLoop-SameHead development run\n\n'
            f'Status: **{status}**; weight: E4-C Best step 700 (seed 20261103); Bonn recordings: 3; K16.\n\n'
            'Analyze the paired effects and per-video summaries; no test data used.\n'
            'Weights are exactly the E4-C Best checkpoint (step 700, seed 20261103); no Y04 multi-seed weights are used.\n'
            'P_full vs O_same_head uses the same trained P head; this is an independently evolving closed-loop strategy comparison.\n'
            'Output-only and F-only require two physical fast forwards per propagation frame in this reference implementation.\n'
            'All end-to-end measured latencies include both forward passes and exclude GT reading/metric/writing.\n'
            'Times include RGB disk access/preprocessing, optical flow, model work, z-depth and D2H synchronization.\n'
            'Nested module counts must not be added together as independent FLOPs.\n'
            'No assertion of cross-domain generalization or model novelty from this development run.\n',
            encoding='utf-8')
        pack(outdir)
    except Exception:
        # Write a failure marker and preserve the directory. No fake success package.
        (outdir/'FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    print(f'[M04] status={status} review_zip={outdir/"cvpr_m04_closed_loop_review.zip"}',flush=True)
    return 0 if status=='COMPLETE' else 2


def pack(outdir: Path) -> None:
    names=['protocol_lock.json','m04_frame_metrics.csv','m04_frame_timing.csv',
           'm04_module_calls.csv','m04_sequence_summary.csv','m04_timing_summary.csv',
           'm04_paired_effects.csv','m04_effect_summary.csv','m04_summary.json','REPORT.md']
    missing=[n for n in names if not (outdir/n).is_file()]
    if missing:raise RuntimeError(f'Missing mandatory review outputs: {missing}')
    target=outdir/'cvpr_m04_closed_loop_review.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in names:z.write(outdir/name,name)
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None or set(z.namelist())!=set(names):
            raise RuntimeError('Review ZIP failed CRC or whitelist')


def main():
    parser=argparse.ArgumentParser(description='CVPR M04 V3.1 closed-loop same-head using E4-C Best only (development only)')
    parser.add_argument('command',choices=['preflight','canary','run'])
    parser.add_argument('--snapshot',default=os.environ.get('VDEEP_SNAPSHOT',''))
    parser.add_argument(
        '--e4c-checkpoint',
        default=os.environ.get('M04_E4C_CHECKPOINT', str(E4C_DEFAULT_CHECKPOINT)),
        help='Exact E4-C Best checkpoint; SHA256 and provenance are hard-locked.',
    )
    parser.add_argument('--out-dir',default='')
    parser.add_argument('--canary-report',default='')
    parser.add_argument('--tolerance',type=float,default=1e-5)
    parser.add_argument('--timing-repeats',type=int,default=3)
    args=parser.parse_args()
    if args.timing_repeats not in (1,2,3,4,5):
        raise RuntimeError('--timing-repeats must be in 1..5, default 3')
    evidence,data=preflight(args)
    if args.command=='preflight':
        print(json.dumps({'status':'PREFLIGHT_PASS',**evidence},ensure_ascii=False,indent=2))
        return 0
    if args.command=='canary':return canary(args,evidence,data)
    if not args.canary_report:raise RuntimeError('--canary-report required for run')
    return full(args,evidence,data)


if __name__=='__main__':
    raise SystemExit(main())
