"""Dual-anchor SA-OOF V1.0: tensor-only anchoring and output fusion.

Correspondence masks, image geometry and model adapters are supplied by caller.
All tensors are one HxW frame; no native propagation state is accepted.
"""
import math
import torch


def _valid_depth(d):
    return torch.isfinite(d) & (d > 0)


def _enough(mask):
    return int(mask.sum().item()) >= max(2048, math.ceil(mask.numel() * 0.05))


def keyframe_anchor(kf, ref, geometry_mask=None):
    if kf.ndim != 2 or kf.shape != ref.shape:
        raise ValueError('keyframe and reference must have matching HxW shape')
    valid = _valid_depth(kf) & _valid_depth(ref)
    if geometry_mask is not None:
        if geometry_mask.shape != kf.shape:
            raise ValueError('geometry mask must match depth shape')
        valid = valid & geometry_mask.bool()
    if not _enough(valid):
        return None
    return (kf[valid].log() - ref[valid].log()).median()


def decompose(prop, ref, anchor, corr, geometry_mask=None):
    if prop.ndim != 2 or prop.shape != ref.shape or corr.shape != prop.shape:
        raise ValueError('depths and correspondence mask must have matching HxW shape')
    valid = _valid_depth(prop) & _valid_depth(ref)
    if geometry_mask is not None:
        if geometry_mask.shape != prop.shape:
            raise ValueError('geometry mask must match depth shape')
        valid = valid & geometry_mask.bool()
    mask = valid & corr.bool()
    result = {'global_residual': None, 'local_residual': None,
              'valid_depth_mask': valid, 'correspondence_mask': mask,
              'status': None}
    if anchor is None or not bool(torch.isfinite(anchor).item()):
        result['status'] = 'invalid_anchor'
        return result
    if not _enough(mask):
        result['status'] = 'insufficient_corr_mask'
        return result
    residual = torch.zeros_like(prop)
    residual[valid] = ref[valid].log() + anchor - prop[valid].log()
    median = residual[mask].median()
    local = torch.zeros_like(prop)
    local[valid] = residual[valid] - median
    result.update(global_residual=median, local_residual=local, status='ok')
    return result


def fuse(prop, median=None, local=None, valid=None, gamma=None, eta=None):
    # Missing reference/invalid anchor/insufficient correspondences: exact fallback.
    if any(x is None for x in (median, local, valid, gamma, eta)):
        return prop.clone()
    if prop.ndim != 2 or local.shape != prop.shape or valid.shape != prop.shape or eta.shape != prop.shape:
        raise ValueError('spatial tensors must have matching HxW shape')
    gamma = torch.as_tensor(gamma, device=prop.device, dtype=prop.dtype)
    if gamma.numel() != 1 or not bool(torch.isfinite(gamma).all()) or not bool(((gamma >= 0) & (gamma <= 1)).all()):
        raise ValueError('gamma must be a finite scalar in [0,1]')
    if not bool(torch.isfinite(eta).all()) or not bool(((eta >= 0) & (eta <= 1)).all()):
        raise ValueError('eta must be finite and in [0,1]')
    out = prop.clone()
    mask = valid.bool() & _valid_depth(prop)
    out[mask] = (prop[mask].log() + gamma * median + eta[mask] * local[mask]).exp()
    return out
