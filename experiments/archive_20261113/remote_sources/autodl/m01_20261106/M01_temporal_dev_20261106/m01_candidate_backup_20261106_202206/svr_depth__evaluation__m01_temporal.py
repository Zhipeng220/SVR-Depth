"""Auditable provisional adjacent-frame scoring, never author-equivalent tau5."""
from __future__ import annotations

import math
import numpy as np

from svr_depth.evaluation.m01_v31.metrics import evaluate_pair
from svr_depth.evaluation.m01_metrics import array_digest


def score_temporal_pair(previous, current, protocol):
    """Score a pair or explain every unavailable input without changing inference.

    Forward flow belongs to the previous frame, backward flow to the current
    frame. A verified mapping is supplied by the dataset adapter, never inferred
    from file names or the number of flow entries.
    """
    if protocol.get('protocol', 'provisional_v1') != 'provisional_v1':
        raise ValueError('Only provisional_v1 is supported; author equivalence is not verified')
    prev, curr = previous['frame'], current['frame']
    result = {
        'prev_index': prev.index, 'curr_index': curr.index,
        'prev_source_frame_id': prev.source_frame_id, 'curr_source_frame_id': curr.source_frame_id,
        'prev_source_index': prev.source_frame_index, 'curr_source_index': curr.source_frame_index,
        'flow_fwd_source': prev.flow_fwd_source, 'flow_bwd_source': curr.flow_bwd_source,
        'flow_fwd_sha256':array_digest(prev.flow_fwd) if prev.flow_fwd is not None else '',
        'flow_bwd_sha256':array_digest(curr.flow_bwd) if curr.flow_bwd is not None else '',
        'flow_fwd_mask_sha256':array_digest(prev.flow_fwd_mask) if prev.flow_fwd_mask is not None else '',
        'flow_bwd_mask_sha256':array_digest(curr.flow_bwd_mask) if curr.flow_bwd_mask is not None else '',
        'flow_available_fields': ','.join(sorted(set(prev.flow_available_fields) | set(curr.flow_available_fields))),
        'temporal_protocol': 'provisional_v1', 'protocol_status': 'provisional',
        'tau5_provisional': None, 'tau5_ssi_provisional': None,
        'valid_pixels': 0, 'inliers': 0, 'ssi_valid_pixels': 0, 'ssi_inliers': 0,
        'ssi_status': 'not_evaluated',
    }
    issues = []
    if curr.index != prev.index + 1 or (prev.source_frame_index is not None and
            curr.source_frame_index is not None and curr.source_frame_index != prev.source_frame_index + 1):
        issues.append('nonconsecutive_frames')
    if previous.get('prediction') is None or current.get('prediction') is None:
        issues.append('missing_prediction')
    if prev.gt_error or curr.gt_error:
        issues.append('invalid_gt')
    elif prev.gt is None or curr.gt is None or prev.gt_valid_mask is None or curr.gt_valid_mask is None:
        issues.append('missing_gt')

    # Missing source data and an unverified mapping are distinct findings.
    if prev.flow_fwd is None:
        issues.append('missing_forward_flow' if not any(k in prev.flow_available_fields for k in ('flow_fwd', 'flow'))
                      else 'unverified_or_unavailable_forward_flow')
    if curr.flow_bwd is None:
        issues.append('missing_backward_flow' if 'flow_bwd' not in curr.flow_available_fields
                      else 'unverified_or_unavailable_backward_flow')
    if (prev.flow_fwd is not None and not prev.flow_valid) or (curr.flow_bwd is not None and not curr.flow_valid):
        issues.append('unverified_flow_contract')
    if prev.flow_fwd is not None and prev.flow_fwd_mask is None:
        issues.append('missing_forward_mask')
    if curr.flow_bwd is not None and curr.flow_bwd_mask is None:
        issues.append('missing_backward_mask')
    for frame in (prev, curr):
        if frame.flow_error and frame.flow_error != 'unverified_flow_contract':
            issues.append('flow_decode_error')
    for frame in (prev, curr):
        if frame.intrinsics is None:
            issues.append('missing_camera')
        elif isinstance(frame.intrinsics, dict):
            issues.append('unsupported_or_unverified_camera_model')
        elif not frame.camera_valid:
            issues.append('unverified_camera')
        if frame.pose is None:
            issues.append('missing_pose')
        elif not frame.pose_valid:
            issues.append('unverified_pose')
    threshold = protocol.get('fb_max_error_px')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
        issues.append('unconfigured_fb_threshold')
    issues = list(dict.fromkeys(issues))
    if issues:
        result.update(status=issues[0], issues=';'.join(issues),
                      detail=';'.join(x for x in (prev.flow_error, curr.flow_error) if x))
        return result
    try:
        scores = evaluate_pair(
            prev_pred=previous['prediction'], curr_pred=current['prediction'],
            prev_gt=prev.gt, curr_gt=curr.gt,
            prev_gt_mask=prev.gt_valid_mask, curr_gt_mask=curr.gt_valid_mask,
            K_prev=prev.intrinsics, K_curr=curr.intrinsics,
            prev_c2w=prev.pose, curr_c2w=curr.pose,
            prev_pose_valid=prev.pose_valid, curr_pose_valid=curr.pose_valid,
            flow_fwd_px=prev.flow_fwd, flow_bwd_px=curr.flow_bwd,
            flow_fwd_valid=prev.flow_fwd_mask, flow_bwd_valid=curr.flow_bwd_mask,
            fb_max_error_px=threshold,
            prev_ssi_aligned=previous.get('ssi_aligned'), curr_ssi_aligned=current.get('ssi_aligned'),
        )
        raw, ssi = scores['tau5'], scores['tau5_ssi']
        result.update(status=raw['status'], issues='', detail=raw.get('detail', ''),
                      tau5_provisional=raw['value'], valid_pixels=raw['valid_pixels'], inliers=raw['inliers'],
                      tau5_ssi_provisional=ssi['value'], ssi_valid_pixels=ssi['valid_pixels'],
                      ssi_inliers=ssi['inliers'], ssi_status=ssi['status'])
    except (ValueError, np.linalg.LinAlgError) as exc:
        result.update(status='invalid_temporal_input', issues='invalid_temporal_input',
                      detail=f'{type(exc).__name__}: {exc}')
    return result
