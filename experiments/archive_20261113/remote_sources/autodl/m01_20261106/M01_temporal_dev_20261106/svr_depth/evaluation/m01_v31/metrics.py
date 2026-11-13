"""Provisional M01 pair evaluator; see README before interpreting its output.

Arrays are NumPy HxW depths/masks and HxWx2 flows in pixel units. Pixel
centres have integer coordinates. A backward flow is on the current frame and
points to the previous frame; forward flow is on the previous frame.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Score:
    status: str
    valid_pixels: int = 0
    inliers: int = 0
    value: float | None = None
    detail: str = ""


def _array(x, shape, name, dtype=np.float64):
    a = np.asarray(x, dtype=dtype)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    return a


def _mask(x, shape, name):
    a = np.asarray(x)
    if a.shape != shape or a.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array of shape {shape}")
    return a


def pixel_grid(height, width):
    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    y, x = np.mgrid[:height, :width]
    return np.stack((x, y), axis=-1).astype(np.float64)


def bilinear_sample(image, xy):
    """Pull sample and return a full-support validity mask, including bounds.

    At fractional locations every nonzero-weight input pixel must be finite;
    coordinates outside the pixel-centre rectangle are invalid, never padded.
    """
    a = np.asarray(image, dtype=np.float64)
    coords = np.asarray(xy, dtype=np.float64)
    if a.ndim not in (2, 3) or coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError("image must be HxW[xC] and coordinates HxWx2")
    h, w = a.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("empty source image")
    finite_xy = np.isfinite(coords).all(axis=-1)
    inside = (finite_xy & (coords[..., 0] >= 0) & (coords[..., 0] <= w - 1)
              & (coords[..., 1] >= 0) & (coords[..., 1] <= h - 1))
    # Clamping here is only to make indexing safe. It does not make OOB valid.
    x = np.clip(np.where(finite_xy, coords[..., 0], 0), 0, w - 1)
    y = np.clip(np.where(finite_xy, coords[..., 1], 0), 0, h - 1)
    x0, y0 = np.floor(x).astype(np.intp), np.floor(y).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = x - x0, y - y0
    weights = ((1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy)
    corners = (a[y0, x0], a[y0, x1], a[y1, x0], a[y1, x1])
    result = np.zeros(coords.shape[:2] + a.shape[2:], dtype=np.float64)
    good = inside.copy()
    for weight, corner in zip(weights, corners):
        finite = np.isfinite(corner)
        if a.ndim == 3:
            good &= (weight == 0) | finite.all(axis=-1)
            result += weight[..., None] * np.where(finite, corner, 0)
        else:
            good &= (weight == 0) | finite
            result += weight * np.where(finite, corner, 0)
    return result, good


def _sample_mask(mask, xy):
    # Do not accept a bilinear mixture containing an invalid source pixel.
    _, valid = bilinear_sample(np.where(mask, 1.0, np.nan), xy)
    return valid


def backward_correspondence(flow_bwd_px, flow_fwd_px, valid_bwd, valid_fwd,
                            fb_max_error_px):
    """Current -> previous mapping and current-grid forward/backward mask."""
    bwd = np.asarray(flow_bwd_px, dtype=np.float64)
    if bwd.ndim != 3 or bwd.shape[-1] != 2:
        raise ValueError("backward flow must be HxWx2")
    shape = bwd.shape[:2]
    fwd = _array(flow_fwd_px, bwd.shape, "forward flow")
    vb = _mask(valid_bwd, shape, "backward flow validity")
    vf = _mask(valid_fwd, shape, "forward flow validity")
    if not math.isfinite(fb_max_error_px) or fb_max_error_px <= 0:
        raise ValueError("fb_max_error_px must be finite and > 0")
    coords = pixel_grid(*shape) + bwd
    fwd_warped, in_bounds = bilinear_sample(fwd, coords)
    residual = np.linalg.norm(np.where(np.isfinite(bwd), bwd, 0) + fwd_warped, axis=-1)
    valid = (vb & np.isfinite(bwd).all(axis=-1) & in_bounds
             & _sample_mask(vf, coords) & (residual < fb_max_error_px))
    return coords, valid


def resize_flow_pixels(flow_px, out_shape):
    """Resize a pixel displacement field using half-pixel centres, border mode.

    Both frames must undergo the same resize. Components are multiplied by
    new_width/old_width and new_height/old_height respectively.
    """
    flow = np.asarray(flow_px, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[-1] != 2 or not np.isfinite(flow).all():
        raise ValueError("flow must be finite HxWx2")
    oh, ow = flow.shape[:2]
    nh, nw = out_shape
    if min(oh, ow, nh, nw) < 1:
        raise ValueError("image dimensions must be positive")
    sx, sy = nw / ow, nh / oh
    xy = pixel_grid(nh, nw)
    xy[..., 0] = np.clip((xy[..., 0] + 0.5) / sx - 0.5, 0, ow - 1)
    xy[..., 1] = np.clip((xy[..., 1] + 0.5) / sy - 0.5, 0, oh - 1)
    result, _ = bilinear_sample(flow, xy)
    return result * np.array([sx, sy])


def crop_flow_pixels(flow_px, target_top_left, source_top_left, out_shape):
    """Crop target-domain flow and account for possibly different crop origins."""
    flow = np.asarray(flow_px, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError("flow must be HxWx2")
    ty, tx = target_top_left
    sy, sx = source_top_left
    h, w = out_shape
    if min(ty, tx, sy, sx) < 0 or h < 1 or w < 1 or ty+h > flow.shape[0] or tx+w > flow.shape[1]:
        raise ValueError("invalid crop")
    return flow[ty:ty+h, tx:tx+w].copy() + np.array([tx-sx, ty-sy])


def transform_intrinsics(K, scale_xy, crop_top_left=(0, 0)):
    """Apply half-pixel resize followed by crop to a pinhole intrinsic matrix."""
    k = _intrinsics(K, "intrinsics")
    sx, sy = scale_xy
    top, left = crop_top_left
    if not (math.isfinite(sx) and math.isfinite(sy) and sx > 0 and sy > 0):
        raise ValueError("resize scales must be positive and finite")
    affine = np.array([[sx, 0, (sx-1)/2-left], [0, sy, (sy-1)/2-top], [0, 0, 1]])
    return affine @ k


def fraction_of_extent_to_pixels(flow, shape):
    """Convert displacement divided by (W-1,H-1), not a grid in [-1,1]."""
    h, w = shape
    if h < 2 or w < 2:
        raise ValueError("extent-normalized flow needs height and width >= 2")
    return _array(flow, (h, w, 2), "normalized flow") * [w-1, h-1]


def _intrinsics(K, name):
    k = _array(K, (3, 3), name)
    if (not np.isfinite(k).all() or not np.allclose(k[2], [0, 0, 1], atol=1e-8)
            or k[0, 0] <= 0 or k[1, 1] <= 0 or abs(np.linalg.det(k)) < 1e-12):
        raise ValueError(f"{name} must be a finite, invertible pinhole K")
    return k


def _pose(c2w, valid, name):
    if valid is not True or c2w is None:
        return None
    p = _array(c2w, (4, 4), name)
    r = p[:3, :3]
    if (not np.isfinite(p).all() or not np.allclose(p[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-4)):
        raise ValueError(f"{name} is not a finite rigid camera-to-world transform")
    return p


def _score(inliers, valid, empty_status="no_valid_pixels"):
    n = int(valid.sum())
    if not n:
        return Score(empty_status)
    k = int((inliers & valid).sum())
    return Score("ok", n, k, k / n)


def _ssi(gt, pred, mask):
    """Approximate published code's per-frame 95% trimmed fit, with guards."""
    g, p = gt[mask].astype(np.float32), pred[mask].astype(np.float32)
    if not len(g):
        return None, "no GT pixels"
    error = np.abs(g-p)
    selected = error < np.float32(np.quantile(error, 0.95))
    if selected.sum() < 2:
        return None, "SSI strict 95% trim leaves fewer than two pixels"
    x = np.stack((p[selected], np.ones(selected.sum(), dtype=np.float32)), axis=1)
    a = x.T @ x + np.float32(1e-4) * np.eye(2, dtype=np.float32)
    try:
        scale, shift = np.linalg.solve(a, x.T @ g[selected])
    except np.linalg.LinAlgError:
        return None, "SSI fit is singular"
    aligned = pred * float(scale) + float(shift)
    if not np.isfinite(aligned[mask]).all() or np.any(aligned[mask] <= 0):
        return None, "SSI produces non-positive or non-finite depth"
    return aligned, ""


_USE_PROVISIONAL_SSI = object()


def _frame(gt, pred, mask, aligned_override=_USE_PROVISIONAL_SSI):
    if not mask.any():
        return {"delta1": Score("no_gt_pixels"), "delta1_ssi": Score("no_gt_pixels")}, None
    if not np.isfinite(gt[mask]).all() or np.any(gt[mask] <= 0):
        raise ValueError("GT mask includes invalid or non-positive GT")
    if not np.isfinite(pred[mask]).all() or np.any(pred[mask] <= 0):
        raise ValueError("prediction invalid on GT mask; do not silently remove pixels")
    ratio = np.maximum(gt[mask] / pred[mask], pred[mask] / gt[mask])
    n = int(mask.sum())
    k = int((ratio < 1.25).sum())
    delta = Score("ok", n, k, k/n)
    if aligned_override is _USE_PROVISIONAL_SSI:
        aligned, detail = _ssi(gt, pred, mask)
    elif aligned_override is None:
        aligned, detail = None, 'upstream SSI alignment unavailable'
    else:
        aligned = _array(aligned_override, gt.shape, 'provided SSI alignment')
        if not np.isfinite(aligned[mask]).all() or np.any(aligned[mask] <= 0):
            aligned, detail = None, 'provided SSI alignment invalid on GT mask'
        else:
            detail = ''
    if aligned is None:
        return {"delta1": delta, "delta1_ssi": Score("ssi_unavailable", detail=detail)}, None
    ratio_ssi = np.maximum(gt[mask] / aligned[mask], aligned[mask] / gt[mask])
    k = int((ratio_ssi < 1.25).sum())
    return {"delta1": delta, "delta1_ssi": Score("ok", n, k, k/n)}, aligned


def _temporal(prev, curr, previous_xy, valid, K_prev, prev_to_curr):
    h, w = prev.shape
    sampled_prev, finite = bilinear_sample(prev, previous_xy)
    # Pixels excluded by the common mask cannot poison the calculation.
    sample = np.where(valid & finite, sampled_prev, 0)
    xy = np.where(np.isfinite(previous_xy), previous_xy, 0)
    rays = np.concatenate((xy, np.ones((h, w, 1))), axis=-1) @ np.linalg.inv(K_prev).T
    points = rays * (sample / rays[..., 2])[..., None]
    transformed_z = (points @ prev_to_curr[:3, :3].T + prev_to_curr[:3, 3])[..., 2]
    if not finite[valid].all() or not np.isfinite(transformed_z[valid]).all() or np.any(transformed_z[valid] <= 0):
        return Score("invalid_prediction", int(valid.sum()), detail="invalid transformed depth on common mask")
    error = np.zeros_like(curr)
    error[valid] = np.abs(transformed_z[valid] - curr[valid]) / curr[valid]
    return _score(error < 0.05, valid)


def evaluate_pair(*, prev_pred, curr_pred, prev_gt, curr_gt, prev_gt_mask,
                  curr_gt_mask, K_prev, K_curr, prev_c2w, curr_c2w,
                  prev_pose_valid, curr_pose_valid, flow_bwd_px, flow_fwd_px,
                  flow_bwd_valid, flow_fwd_valid, fb_max_error_px,
                  prev_ssi_aligned=_USE_PROVISIONAL_SSI,
                  curr_ssi_aligned=_USE_PROVISIONAL_SSI):
    """Score a pair. Temporal output is provisional, not an official replica.

    Pose-valid booleans must represent verified original metadata, not the
    upstream loader's default True after an identity fallback. The two flow
    masks and GT masks must come from method-independent input data.
    """
    prev = np.asarray(prev_pred, dtype=np.float64)
    if prev.ndim != 2:
        raise ValueError("depths must be HxW")
    shape = prev.shape
    curr = _array(curr_pred, shape, "current prediction")
    gt0, gt1 = _array(prev_gt, shape, "previous GT"), _array(curr_gt, shape, "current GT")
    m0, m1 = _mask(prev_gt_mask, shape, "previous GT mask"), _mask(curr_gt_mask, shape, "current GT mask")
    frame0, ssi0 = _frame(gt0, prev, m0, prev_ssi_aligned)
    frame1, ssi1 = _frame(gt1, curr, m1, curr_ssi_aligned)
    k0, _ = _intrinsics(K_prev, "previous K"), _intrinsics(K_curr, "current K")
    p0, p1 = _pose(prev_c2w, prev_pose_valid, "previous pose"), _pose(curr_c2w, curr_pose_valid, "current pose")
    if p0 is None or p1 is None:
        tau = tau_ssi = Score("missing_or_unverified_pose")
    else:
        xy, common = backward_correspondence(flow_bwd_px, flow_fwd_px, flow_bwd_valid, flow_fwd_valid, fb_max_error_px)
        common &= m1 & _sample_mask(m0, xy)
        transform = np.linalg.inv(p1) @ p0
        tau = _temporal(prev, curr, xy, common, k0, transform)
        tau_ssi = (Score("ssi_unavailable", detail="one or both per-frame SSI fits failed")
                   if ssi0 is None or ssi1 is None else _temporal(ssi0, ssi1, xy, common, k0, transform))
    return {"previous": {k: asdict(v) for k, v in frame0.items()},
            "current": {k: asdict(v) for k, v in frame1.items()},
            "tau5": asdict(tau), "tau5_ssi": asdict(tau_ssi)}
