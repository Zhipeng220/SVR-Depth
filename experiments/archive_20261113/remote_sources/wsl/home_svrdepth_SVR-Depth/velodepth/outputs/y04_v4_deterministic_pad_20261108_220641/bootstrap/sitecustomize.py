import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, os.getcwd())
import velodepth.ops.losses
_pad=F.pad
_grid=F.grid_sample
class _Reflection(torch.autograd.Function):
 @staticmethod
 def forward(ctx, x, pad):
  ctx.pad=pad; ctx.shape=x.shape; ctx.device=x.device; ctx.dtype=x.dtype
  return _pad(x,pad,mode='reflect')
 @staticmethod
 def backward(ctx, grad):
  with torch.enable_grad():
   x=torch.zeros(ctx.shape,device='cpu',dtype=ctx.dtype,requires_grad=True)
   y=_pad(x,ctx.pad,mode='reflect')
   g=torch.autograd.grad(y,x,grad.to('cpu'))[0]
  return g.to(ctx.device),None
def patched(x,pad,mode='constant',value=None):
 if mode=='reflect' and x.is_cuda and x.requires_grad and len(pad)==4 and x.ndim==4:
  return _Reflection.apply(x,tuple(pad))
 return _pad(x,pad,mode,value)
F.pad=patched
class _Grid(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,grid,mode,padding_mode,align_corners):
  ctx.save_for_backward(x,grid); ctx.options=(mode,padding_mode,align_corners)
  return _grid(x,grid,mode=mode,padding_mode=padding_mode,align_corners=align_corners)
 @staticmethod
 def backward(ctx,grad):
  x,grid=ctx.saved_tensors; mode,padding,align=ctx.options
  dtype=torch.float32 if x.dtype in (torch.float16,torch.bfloat16) else x.dtype
  with torch.enable_grad():
   xc=x.detach().to('cpu',dtype=dtype).requires_grad_(True); gc=grid.detach().to('cpu',dtype=dtype).requires_grad_(True)
   y=_grid(xc,gc,mode=mode,padding_mode=padding,align_corners=align)
   gx,gg=torch.autograd.grad(y,(xc,gc),grad.to('cpu',dtype=dtype))
  return gx.to(x.device,dtype=x.dtype),gg.to(grid.device,dtype=grid.dtype),None,None,None
def patched_grid(x,grid,mode='bilinear',padding_mode='zeros',align_corners=None):
 if x.is_cuda and (x.requires_grad or grid.requires_grad): return _Grid.apply(x,grid,mode,padding_mode,align_corners)
 return _grid(x,grid,mode=mode,padding_mode=padding_mode,align_corners=align_corners)
F.grid_sample=patched_grid
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
print('[DETERMINISTIC REFLECT CPU BACKWARD PROTOTYPE]',flush=True)
