"""Fixed-K16 M01 engine based on the historical Y08/Y09 Stream.

The same FP32 flow boundary is used for every method. Input is a decoded RGB
array, so Bonn paths and TUM HDF5 use exactly the same preprocessing. No
training or model-code patching occurs here.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_sha(path, expected):
    path = Path(path).expanduser().resolve(strict=True)
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdefABCDEF' for c in expected):
        raise ValueError(f'Expected SHA256 must be supplied for {path}')
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(f'SHA256 mismatch for {path}: {actual} != {expected}')
    return str(path), actual


def _imports():
    for p in (ROOT, ROOT / 'experiments/crosssource_v2/legacy_helpers', ROOT / 'svr_depth/_compat/output_smoke'):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import cv2
    import numpy as np
    import torch
    import gt2_fixed_clip_repair_train as gt2
    import gt4a2_exp1_train_eval as exp1
    import y08y09_models as models
    return SimpleNamespace(cv2=cv2, np=np, torch=torch, gt2=gt2, exp1=exp1, models=models)


def _checkpoint(rt, spec, base_sha):
    path, actual = require_sha(spec['checkpoint'], spec['sha256'])
    try:
        obj = rt.torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise RuntimeError(f'Cannot safely load checkpoint {path} with weights_only=True: {exc}') from exc
    if not isinstance(obj, dict):
        raise ValueError('Checkpoint must be a state dictionary or a mapping containing one')
    meta = obj.get('identity', {})
    if meta and not isinstance(meta, dict):
        raise ValueError('Checkpoint identity must be a mapping')
    checkpoint_method = meta.get('method')
    if checkpoint_method and checkpoint_method != spec['method']:
        raise ValueError(f"Checkpoint method {checkpoint_method} does not match {spec['method']}")
    checkpoint_seed = meta.get('seed')
    if checkpoint_seed is not None and str(checkpoint_seed) != str(spec['seed']):
        raise ValueError(f"Checkpoint seed {checkpoint_seed} does not match {spec['seed']}")
    recorded_base = meta.get('snapshot_sha256') or meta.get('backbone_sha256') or obj.get('pretrained_weights_sha256')
    if recorded_base and recorded_base.lower() != base_sha.lower():
        raise ValueError('Checkpoint base snapshot SHA256 differs from configured backbone')
    if obj.get('smoke') is True or meta.get('smoke') is True:
        raise ValueError('Smoke checkpoint cannot be used as an M01 method checkpoint')
    key = spec.get('state_key')
    state = obj if key is None else obj.get(key)
    if not isinstance(state, dict) or not state:
        raise ValueError(f'Checkpoint state_key {key!r} missing or empty in {path}')
    return state, {'path': path, 'sha256': actual, 'state_key': key,
                   'checkpoint_identity': meta, 'recorded_backbone_sha256': recorded_base,
                   'training_provenance_verified': bool(recorded_base)}


class M01Engine:
    def __init__(self, snapshot, method, base_sha):
        self.rt = rt = _imports()
        torch = rt.torch
        if not torch.cuda.is_available():
            raise RuntimeError('M01 model inference requires an available CUDA GPU')
        self.device = torch.device('cuda')
        self.method = method
        self.identity = None
        family = method['method']
        state = None
        if family not in ('A', 'A_native'):
            state, self.identity = _checkpoint(rt, method, base_sha)
        mode = 'pre_readout' if family == 'P_multi' else 'off'
        self.model = rt.gt2.load_model(Path(snapshot), mode, self.device)
        if family == 'A_native':
            # load_model disables heuristics for fixed K16; restore exactly the
            # snapshot/config defaults used by the published model constructor.
            from velodepth.models.velodepth import FLOW_ALPHA, FLOW_BETA
            model_config = json.loads((Path(snapshot) / 'config.json').read_text(encoding='utf-8')).get('model', {})
            for key, default in (('alpha_flow', FLOW_ALPHA), ('beta_flow', FLOW_BETA),
                                 ('alpha_luma', FLOW_ALPHA), ('beta_luma', FLOW_BETA)):
                setattr(self.model, key, float(model_config.get(key, default)))
        self.model.eval()
        adapter = None
        if family == 'P_multi':
            head = self.model.pixel_decoder.persistent_feature_repair
            head.load_state_dict(state, strict=True)
            head.eval()
            self.model.pixel_decoder.persistent_feature_repair = rt.models.GainHead(head, 1.0)
        elif family == 'O_fast':
            adapter = rt.models.FastDepthAdapter().to(self.device)
            adapter.load_state_dict(state, strict=True)
            adapter.eval()
        elif family == 'O_multi':
            from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
            adapter = PersistentFeatureRepairHead(channels=int(self.model.pixel_decoder.radial_module.hidden_dim),
                                                  bottleneck=32, levels=4).to(self.device)
            adapter.load_state_dict(state, strict=True)
            adapter.eval()
        elif family not in ('A', 'A_native'):
            raise ValueError(f'Unknown method: {family}')
        for p in self.model.parameters():
            p.requires_grad_(False)
        if adapter is not None:
            for p in adapter.parameters():
                p.requires_grad_(False)
        self.adapter = adapter
        self.stream = None
        self.shape = None
        torch.cuda.reset_peak_memory_stats(self.device)

    def runtime_info(self):
        torch = self.rt.torch
        return {'torch':torch.__version__, 'cuda_runtime':torch.version.cuda,
                'cudnn':torch.backends.cudnn.version(),
                'gpu':torch.cuda.get_device_name(self.device),
                'gpu_total_bytes':int(torch.cuda.get_device_properties(self.device).total_memory),
                'opencv':self.rt.cv2.__version__, 'numpy':self.rt.np.__version__}

    def reset(self):
        self.stream = None
        self.shape = None
        self.rt.torch.cuda.reset_peak_memory_stats(self.device)

    def _shape_for(self, rgb):
        torch, gt2 = self.rt.torch, self.rt.gt2
        h, w = rgb.shape[:2]
        constraints = self.model.shape_constraints
        lo = float(constraints['pixels_min'])
        hi = lo + (float(constraints['pixels_max']) - lo) / 10.0
        pads, padded = gt2.get_paddings((h, w), constraints['ratio_bounds'])
        _, network = gt2.get_resize_factor(padded, (lo, hi))
        mean = torch.tensor(gt2.IMAGENET_DATASET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(gt2.IMAGENET_DATASET_STD, device=self.device).view(1, 3, 1, 1)
        return {'original_shape': (h, w), 'paddings': pads, 'padded_shape': padded,
                'network_shape': network, 'mean': mean, 'std': std,
                'validity_mask': torch.ones((1, 1, *network), device=self.device, dtype=torch.bool),
                'image_paddings': torch.zeros((1, 4), device=self.device, dtype=torch.long),
                'depth_paddings': torch.zeros((1, 4), device=self.device, dtype=torch.long)}

    def step(self, index, rgb):
        rt, torch = self.rt, self.rt.torch
        if self.stream is None:
            if index != 0:
                raise ValueError('A sequence must start at frame index 0')
            self.shape = self._shape_for(rgb)
            family = self.method['method']
            cfg = {'family': 'P' if family == 'P_multi' else ('A' if family == 'A_native' else family),
                   'schedule': 'Auto' if family == 'A_native' else 'K16', 'gain': 1.0, 'frequency': 1}
            self.stream = rt.models.Stream(rt, self.model, cfg, self.shape, adapter=self.adapter, debug=True)
        if rgb.shape[:2] != self.shape['original_shape']:
            raise ValueError('RGB image dimensions changed inside a sequence')
        img, frame_u8 = rt.gt2.preprocess_rgb(rgb, self.device, self.shape['paddings'],
                                               self.shape['network_shape'], self.shape['mean'], self.shape['std'])
        frame = {'img': img, 'frame_u8': frame_u8, 'index': index}
        with torch.no_grad():
            static = None
            if index > 0 and (self.method['method'] == 'A_native' or index % 16 != 0):
                # Deliberately outside CUDA autocast for A, P and both O methods.
                with torch.autocast('cuda', enabled=False):
                    static = rt.models.transition(rt, self.model, self.stream.previous, frame, self.shape, self.stream.dis)
            with torch.autocast('cuda', dtype=torch.float16):
                pred, meta = self.stream.advance(frame, index, static)
        return pred, meta

    def synchronize(self):
        self.rt.torch.cuda.synchronize(self.device)

    def to_numpy(self, prediction):
        return prediction.detach().float().cpu().numpy()

    def stats(self):
        return dict(self.stream.counts) if self.stream is not None else {}

    def peak_bytes(self):
        return int(self.rt.torch.cuda.max_memory_allocated(self.device))
