"""Camera rays for verified pinhole or upstream OPENCV metadata; diagnostic only."""
import numpy as np
from .metrics import _intrinsics


def camera_rays(camera, xy, *, pixel_center, device='cpu'):
    xy=np.asarray(xy,dtype=np.float64)
    if xy.ndim!=3 or xy.shape[-1]!=2:raise ValueError('pixel coordinates must be HxWx2')
    if pixel_center not in ('integer','half'):raise ValueError('Pixel centre must be specified')
    coordinates=xy+(0.5 if pixel_center=='half' else 0.0)
    if not isinstance(camera,dict):
        k=_intrinsics(camera,'camera intrinsics')
        return np.concatenate((coordinates,np.ones((*xy.shape[:2],1))),-1) @ np.linalg.inv(k).T
    if camera.get('model')!='OPENCV':raise ValueError('Unsupported camera model')
    params=np.asarray(camera.get('params'),dtype=np.float64)
    if params.shape!=(16,) or not np.isfinite(params).all() or np.any(params[:2]<=0):raise ValueError('Invalid OPENCV parameters')
    if np.any(np.abs(params[7:10])>0):raise ValueError('Upstream OPENCV does not support rational denominator')
    import torch
    from velodepth.utils.camera import OPENCV
    with torch.no_grad():
        obj=OPENCV(torch.as_tensor(params,dtype=torch.float32,device=device))
        uv=torch.as_tensor(coordinates,dtype=torch.float32,device=device).permute(2,0,1).unsqueeze(0)
        rays=obj.unproject(uv).squeeze(0).permute(1,2,0).cpu().numpy()
    if not np.isfinite(rays).all():raise ValueError('Camera unprojection returned nonfinite rays')
    return rays
