"""Current-frame correspondence mask using backward sampling of forward flow.

Pixel-center backward mapping q -> p=q+F_b(q); forward field is bilinearly
sampled at p. This discretization needs to be recorded in formal runs.
"""
import cv2
import numpy as np

def correspondence_mask(forward, backward, valid_depth):
    if forward.ndim != 3 or forward.shape[-1] != 2 or backward.shape != forward.shape or valid_depth.shape != forward.shape[:2]:
        raise ValueError('flows must be HxWx2 and depth mask HxW')
    h,w=valid_depth.shape
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    bx,by=backward[...,0],backward[...,1]
    px=xx+bx; py=yy+by
    inside=np.isfinite(px)&np.isfinite(py)&(px>=0)&(px<=w-1)&(py>=0)&(py<=h-1)
    safe_x=np.where(inside,px,-1).astype(np.float32); safe_y=np.where(inside,py,-1).astype(np.float32)
    sampled=cv2.remap(forward.astype(np.float32),safe_x,safe_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    error=np.linalg.norm(sampled+backward,axis=2)
    threshold=1.0+0.05*(np.linalg.norm(sampled,axis=2)+np.linalg.norm(backward,axis=2))
    mask=inside & np.isfinite(sampled).all(axis=2) & np.isfinite(backward).all(axis=2) & np.isfinite(error) & (error<=threshold) & valid_depth.astype(bool)
    return mask,error
