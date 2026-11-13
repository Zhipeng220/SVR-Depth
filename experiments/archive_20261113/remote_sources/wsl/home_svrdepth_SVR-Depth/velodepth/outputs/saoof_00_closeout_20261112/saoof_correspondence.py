"""Current-frame correspondence mask using backward sampling of forward flow.

Pixel-center backward mapping q -> p=q+F_b(q); forward field is bilinearly
sampled at p. This discretization needs to be recorded in formal runs.
"""
import numpy as np
import torch
import torch.nn.functional as F

def correspondence_mask(forward, backward):
    if forward.ndim != 3 or forward.shape[-1] != 2 or backward.shape != forward.shape:
        raise ValueError('flows must be matching HxWx2 arrays')
    h,w=forward.shape[:2]
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    bx,by=backward[...,0],backward[...,1]
    px=xx+bx; py=yy+by
    inside=np.isfinite(px)&np.isfinite(py)&(px>=0)&(px<=w-1)&(py>=0)&(py<=h-1)
    safe_x=np.where(inside,px,-1).astype(np.float32); safe_y=np.where(inside,py,-1).astype(np.float32)
    grid=np.stack((2*(safe_x+.5)/w-1,2*(safe_y+.5)/h-1),axis=-1)
    flow=torch.from_numpy(np.ascontiguousarray(forward,dtype=np.float32)).permute(2,0,1).unsqueeze(0)
    sampled=F.grid_sample(flow,torch.from_numpy(grid).unsqueeze(0),mode='bilinear',padding_mode='border',align_corners=False)[0].permute(1,2,0).numpy()
    error=np.linalg.norm(sampled+backward,axis=2)
    threshold=1.0+0.05*(np.linalg.norm(sampled,axis=2)+np.linalg.norm(backward,axis=2))
    mask=inside & np.isfinite(sampled).all(axis=2) & np.isfinite(backward).all(axis=2) & np.isfinite(error) & (error<=threshold)
    return mask,error
