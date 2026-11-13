"""V1.1: unified RGB reference input derived from official native canvas."""
import torch
import torch.nn.functional as F

NATIVE_SIZE=(434,574)
REFERENCE_SIZES=((224,294),(294,392))
OFFICIAL_CANVAS_TAG='bonn_official_native_canvas_v1.1'

def official_frame_from_batch(batch, index, device):
    """Extract the exact native tensor produced by the official Bonn Loader."""
    image=batch['data']['image'][0,index:index+1].to(device)
    if tuple(image.shape)!=(1,3,*NATIVE_SIZE):
        raise ValueError('official Bonn batch must use the frozen native canvas')
    mean=torch.tensor([.485,.456,.406],device=image.device)[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device=image.device)[None,:,None,None]
    u8=((image.float()*std+mean).clamp(0,1)*255).byte()[0].cpu()
    return {'img':image,'frame_u8':u8,'preprocessing':OFFICIAL_CANVAS_TAG,'index':index}

def reference_input(native_rgb_normalized, size):
    if native_rgb_normalized.ndim != 4 or native_rgb_normalized.shape[-2:] != NATIVE_SIZE:
        raise ValueError('expected Bx3x434x574 native input')
    size=tuple(size)
    if size not in REFERENCE_SIZES:
        raise ValueError('unsupported V1.1 reference size')
    return F.interpolate(native_rgb_normalized,size=size,mode='bilinear',align_corners=False,antialias=False)
