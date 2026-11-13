"""Use the bundled upstream functions for per-frame depth scoring only.

Calling the original SSI and delta functions does not reproduce the upstream
dataset transforms, temporal metric, frame selection or aggregation protocol.
"""
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def _upstream_functions():
    from velodepth.utils import evaluation_depth
    root = Path(__file__).resolve().parents[2]
    reference = root / 'reference/Velodepth-upstream/velodepth/utils/evaluation_depth.py'
    loaded = Path(evaluation_depth.__file__).resolve()
    if not reference.is_file():
        raise RuntimeError('Bundled upstream depth metric source is missing')
    if hashlib.sha256(reference.read_bytes()).digest() != hashlib.sha256(loaded.read_bytes()).digest():
        raise RuntimeError(f'Loaded metric source differs from bundled upstream: {loaded}')
    return evaluation_depth.ssi, evaluation_depth.delta


def score_upstream_depth(gt, pred, mask):
    """Return upstream per-frame d1 and d1-SSI on the supplied common mask."""
    import torch

    gt, pred, mask = np.asarray(gt), np.asarray(pred), np.asarray(mask)
    if gt.ndim != 2 or pred.shape != gt.shape or mask.shape != gt.shape or mask.dtype != np.bool_:
        raise ValueError('GT, prediction and boolean mask must have identical HxW shape')
    if not mask.any():
        return {'upstream_depth_status': 'no_valid_pixels'}
    g, p = gt[mask], pred[mask]
    if not np.all(np.isfinite(g) & (g > 0) & np.isfinite(p) & (p > 0)):
        return {'upstream_depth_status': 'invalid_input'}
    ssi, delta = _upstream_functions()
    g = torch.as_tensor(np.ascontiguousarray(g), dtype=torch.float32)
    p = torch.as_tensor(np.ascontiguousarray(p), dtype=torch.float32)
    with torch.no_grad():
        aligned = ssi(g, p)
        raw = float(delta(g, p, exponent=1.0))
        aligned_d1 = float(delta(g, aligned, exponent=1.0))
        fit_finite = bool(torch.isfinite(aligned).all())
        nonpositive_fraction = float((aligned <= 0).to(torch.float32).mean())
        nonfinite_fraction = float((~torch.isfinite(aligned)).to(torch.float32).mean())
    if not np.isfinite(raw) or not np.isfinite(aligned_d1):
        return {'upstream_depth_status': 'nonfinite_metric'}
    return {'upstream_depth_status': 'ok' if fit_finite else 'nonfinite_alignment',
            'upstream_depth_d1': raw, 'upstream_depth_d1_ssi': aligned_d1,
            'upstream_ssi_nonpositive_fraction': nonpositive_fraction,
            'upstream_ssi_nonfinite_fraction': nonfinite_fraction}
