"""Candidate tau5 geometry, not yet verified against the paper authors.

All images share a pixel grid. Flow is current-to-previous, in pixels. The
camera intrinsics must use the explicitly selected pixel-centre convention.
There is no implicit forward/backward consistency filter or invented mask.
"""
from __future__ import annotations
import numpy as np
from .metrics import _intrinsics, _mask, _pose, _sample_mask, bilinear_sample, pixel_grid


def compute_paper_tau5(prev_depth, curr_depth, flow_bwd, flow_valid, K_prev,
                       pose_prev, pose_curr, mask_prev, mask_curr, *,
                       pose_prev_valid=False, pose_curr_valid=False,
                       camera_model='unverified', pixel_center='unverified',
                       flow_direction='unverified', flow_source='', pose_source=''):
    """Score candidates with audited counts; never infer flow/pose conventions."""
    result = {'tau5': None, 'valid_pixels': 0, 'inliers': 0, 'status': 'unavailable',
              'mask_counts': {}, 'flow_source': flow_source, 'pose_source': pose_source,
              'protocol': 'paper_tau5_candidate_v1', 'author_equivalent': False}
    if flow_bwd is None or flow_valid is None:
        result['status'] = 'missing_flow_or_validity_mask'
        return result
    if flow_direction != 'current_to_previous':
        result['status'] = 'unverified_flow_direction'
        return result
    if camera_model != 'pinhole_verified':
        result['status'] = 'unsupported_or_unverified_camera'
        return result
    if pixel_center not in ('integer', 'half'):
        result['status'] = 'unverified_pixel_center'
        return result
    if not pose_prev_valid or not pose_curr_valid or pose_prev is None or pose_curr is None:
        result['status'] = 'missing_or_unverified_pose'
        return result
    prev = np.asarray(prev_depth, dtype=np.float64)
    curr = np.asarray(curr_depth, dtype=np.float64)
    if prev.ndim != 2 or prev.shape != curr.shape:
        raise ValueError('Depth arrays must have identical HxW shapes')
    shape = prev.shape
    mp = _mask(mask_prev, shape, 'previous GT mask')
    mc = _mask(mask_curr, shape, 'current GT mask')
    fv = _mask(flow_valid, shape, 'flow validity')
    flow = np.asarray(flow_bwd, dtype=np.float64)
    if flow.shape != shape + (2,):
        raise ValueError('Backward flow must be HxWx2')
    k = _intrinsics(K_prev, 'previous intrinsics')
    p0 = _pose(pose_prev, True, 'previous pose')
    p1 = _pose(pose_curr, True, 'current pose')
    # Sampling is on integer array coordinates; only the camera ray convention
    # shifts. This prevents mixing half-centre K with array indexing.
    xy = pixel_grid(*shape) + flow
    sampled, sample_finite = bilinear_sample(prev, xy)
    _, inside = bilinear_sample(np.ones(shape), xy)
    source_mask = _sample_mask(mp, xy)
    finite_flow = np.isfinite(flow).all(axis=-1)
    valid = mc & fv & finite_flow & inside & source_mask
    n = int(valid.sum())
    result['mask_counts'] = {'current_gt_pixels': int(mc.sum()), 'previous_gt_pixels': int(mp.sum()),
        'flow_valid_pixels': int((fv & finite_flow).sum()), 'in_bounds_pixels': int(inside.sum()),
        'pose_valid': True, 'final_valid_pixels': n}
    result['valid_pixels'] = n
    if not n:
        result['status'] = 'no_valid_pixels'
        return result
    # Invalid predictions on the common mask invalidate the pair rather than
    # excluding difficult pixels and artificially inflating the result.
    if not (sample_finite[valid].all() and np.isfinite(curr[valid]).all() and np.all(curr[valid] > 0)
            and np.isfinite(sampled[valid]).all() and np.all(sampled[valid] > 0)):
        result['status'] = 'invalid_prediction'
        return result
    camera_xy = np.where(np.isfinite(xy), xy, 0) + (0.5 if pixel_center == 'half' else 0.0)
    rays = np.concatenate([camera_xy, np.ones(shape + (1,))], axis=-1) @ np.linalg.inv(k).T
    points = rays * np.where(valid, sampled, 0)[..., None] / rays[..., 2, None]
    transform = np.linalg.inv(p1) @ p0
    z = (points @ transform[:3, :3].T + transform[:3, 3])[..., 2]
    if not np.isfinite(z[valid]).all() or np.any(z[valid] <= 0):
        result['status'] = 'invalid_transformed_depth'
        return result
    error = np.abs(z[valid] - curr[valid]) / curr[valid]
    result['inliers'] = int((error < 0.05).sum())
    result['tau5'] = result['inliers'] / n
    result['status'] = 'ok'
    result['mask_counts']['tau5_inliers'] = result['inliers']
    return result
