"""Raw metric depth scoring on one common ground-truth-only mask.

Formulas match DICT_METRICS['arel'/'rmse'/'d1'] in upstream VeloDepth.
Never use evaluation_depth.arel: that separate helper median-scales predictions.
"""
from __future__ import annotations
import hashlib
import numpy as np


def array_digest(array):
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode('ascii'))
    h.update(str(a.shape).encode('ascii'))
    h.update(a.tobytes())
    return h.hexdigest()


def score_depth(pred, gt, mask):
    gt = np.asarray(gt)
    mask = np.asarray(mask)
    pred = np.asarray(pred)
    if gt.ndim != 2 or mask.shape != gt.shape or mask.dtype != np.bool_ or pred.shape != gt.shape:
        raise ValueError('prediction, GT and boolean mask must have identical HxW shape')
    n = int(mask.sum())
    if n == 0:
        return {'status': 'invalid_gt', 'valid_pixels': 0, 'mask_sha256': array_digest(mask)}
    if not np.all(np.isfinite(gt[mask]) & (gt[mask] > 0)):
        raise ValueError('GT mask includes non-finite or non-positive GT')
    if not np.all(np.isfinite(pred[mask]) & (pred[mask] > 0)):
        return {'status': 'invalid_prediction', 'valid_pixels': n, 'mask_sha256': array_digest(mask)}
    g, p = gt[mask].astype(np.float64), pred[mask].astype(np.float64)
    return {'status': 'ok', 'valid_pixels': n, 'mask_sha256': array_digest(mask),
            'absrel': float(np.mean(np.abs(g-p)/g)),
            'rmse': float(np.sqrt(np.mean((g-p)**2))),
            'delta1': float(np.mean(np.maximum(g/p, p/g) < 1.25))}


def score_ssi(pred, gt, mask):
    """Optional upstream-style per-frame scale/shift fit, never the raw score.

    Matches VeloDepth's error-trimmed (95th percentile, strict <) regularized
    least squares form. Returns unavailable instead of silently changing the
    mask or clipping non-positive fitted predictions.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    g, p = gt[mask], pred[mask]
    error = np.abs(g-p)
    fit = error < np.quantile(error, .95)
    if fit.sum() < 2:
        return {'ssi_status':'insufficient_fit_pixels'}
    X = np.stack((p[fit], np.ones(int(fit.sum()))), axis=1)
    coeff = np.linalg.solve(X.T @ X + 1e-4*np.eye(2), X.T @ g[fit])
    aligned = p*coeff[0]+coeff[1]
    if not np.all(np.isfinite(aligned) & (aligned > 0)):
        return {'ssi_status':'invalid_aligned_prediction'}
    ratio = np.maximum(g/aligned, aligned/g)
    return {'ssi_status':'ok', 'ssi_absrel':float(np.mean(np.abs(g-aligned)/g)),
            'ssi_rmse':float(np.sqrt(np.mean((g-aligned)**2))),
            'ssi_delta1':float(np.mean(ratio < 1.25)),
            'ssi_scale':float(coeff[0]), 'ssi_shift_m':float(coeff[1])}
