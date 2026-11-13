"""GT-depth temporal geometry diagnostic using explicit flow and pose sources."""
import numpy as np
from .camera_geometry import camera_rays
from .metrics import _pose, _sample_mask, bilinear_sample, pixel_grid


def evaluate_gt_oracle(prev_gt,curr_gt,prev_mask,curr_mask,flow_bwd,flow_valid,
                       camera_prev,pose_prev,pose_curr,*,pixel_center,device='cpu'):
    prev=np.asarray(prev_gt,dtype=np.float64);curr=np.asarray(curr_gt,dtype=np.float64)
    if prev.ndim!=2 or prev.shape!=curr.shape:raise ValueError('Depth shapes must match')
    shape=prev.shape
    masks=[np.asarray(m) for m in (prev_mask,curr_mask,flow_valid)]
    if any(m.shape!=shape or m.dtype!=np.bool_ for m in masks):raise ValueError('Boolean masks required')
    flow=np.asarray(flow_bwd,dtype=np.float64)
    if flow.shape!=shape+(2,):raise ValueError('Backward flow shape mismatch')
    p0=_pose(pose_prev,True,'previous pose');p1=_pose(pose_curr,True,'current pose')
    xy=pixel_grid(*shape)+flow
    sampled,finite=bilinear_sample(prev,xy);_,inside=bilinear_sample(np.ones(shape),xy)
    valid=masks[1]&masks[2]&np.isfinite(flow).all(-1)&inside&_sample_mask(masks[0],xy)
    valid &= finite & np.isfinite(sampled)&(sampled>0)&np.isfinite(curr)&(curr>0)
    count=int(valid.sum())
    if not count:return {'status':'no_valid_pixels','valid_pixels':0,'inliers':0,'tau5':None}
    safe_xy=np.where(np.isfinite(xy),xy,0)
    rays=camera_rays(camera_prev,safe_xy,pixel_center=pixel_center,device=device)
    points=rays*np.where(valid,sampled,0)[...,None]/rays[...,2,None]
    transform=np.linalg.inv(p1)@p0
    z=(points@transform[:3,:3].T+transform[:3,3])[...,2]
    if not np.isfinite(z[valid]).all() or np.any(z[valid]<=0):return {'status':'invalid_transformed_depth','valid_pixels':count,'inliers':0,'tau5':None}
    error=np.abs(z[valid]-curr[valid])/curr[valid]
    inliers=int((error<.05).sum())
    # Zero pose translation/rotation is a diagnostic control, not a ground truth assertion.
    no_comp=np.abs(sampled[valid]-curr[valid])/curr[valid]
    return {'status':'ok','valid_pixels':count,'inliers':inliers,'tau5':inliers/count,
            'tau5_no_ego_motion':float((no_comp<.05).mean()),
            'flow_valid_pixels':int(masks[2].sum()),'current_gt_pixels':int(masks[1].sum()),
            'previous_gt_pixels':int(masks[0].sum()),'in_bounds_pixels':int(inside.sum())}
