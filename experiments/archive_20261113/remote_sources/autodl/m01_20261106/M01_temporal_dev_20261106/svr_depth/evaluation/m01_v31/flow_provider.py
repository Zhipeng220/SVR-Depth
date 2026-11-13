"""Bidirectional DIS flow for explicitly labelled development diagnostics."""
from __future__ import annotations
import math
import numpy as np
from .metrics import bilinear_sample, pixel_grid


def forward_backward_masks(fwd, bwd, threshold_px):
    fwd=np.asarray(fwd,dtype=np.float32);bwd=np.asarray(bwd,dtype=np.float32)
    if fwd.ndim!=3 or fwd.shape[-1]!=2 or bwd.shape!=fwd.shape:
        raise ValueError('Both flows must have matching HxWx2 shapes')
    if not math.isfinite(threshold_px) or threshold_px<=0:raise ValueError('Explicit positive FB threshold required')
    grid=pixel_grid(*fwd.shape[:2]);finite_f=np.isfinite(fwd).all(-1);finite_b=np.isfinite(bwd).all(-1)
    sampled_b,inside_f=bilinear_sample(bwd,grid+fwd)
    sampled_f,inside_b=bilinear_sample(fwd,grid+bwd)
    err_f=np.linalg.norm(fwd+sampled_b,axis=-1);err_b=np.linalg.norm(bwd+sampled_f,axis=-1)
    return {'flow_fwd_valid':finite_f & inside_f & (err_f<threshold_px),
            'flow_bwd_valid':finite_b & inside_b & (err_b<threshold_px),
            'fb_error_fwd_px':err_f.astype(np.float32),'fb_error_bwd_px':err_b.astype(np.float32)}


def dis_pair(prev_rgb,curr_rgb,*,threshold_px):
    import cv2
    prev=np.asarray(prev_rgb);curr=np.asarray(curr_rgb)
    if prev.dtype!=np.uint8 or prev.ndim!=3 or prev.shape[-1]!=3 or curr.dtype!=np.uint8 or curr.shape!=prev.shape:
        raise ValueError('DIS inputs must be matching uint8 HxWx3 RGB frames')
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    a=cv2.cvtColor(prev,cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(curr,cv2.COLOR_RGB2GRAY)
    fwd=dis.calc(a,b,None);bwd=dis.calc(b,a,None)
    if not np.isfinite(fwd).all() or not np.isfinite(bwd).all():raise ValueError('Nonfinite DIS output')
    return {'flow_fwd_px':fwd,'flow_bwd_px':bwd,**forward_backward_masks(fwd,bwd,threshold_px),
            'source':'opencv_dis_fast_estimated_not_gt','opencv_version':cv2.__version__,'fb_threshold_px':float(threshold_px)}
