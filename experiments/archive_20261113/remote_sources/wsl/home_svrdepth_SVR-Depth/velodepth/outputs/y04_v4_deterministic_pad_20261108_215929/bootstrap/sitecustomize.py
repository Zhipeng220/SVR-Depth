import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, os.getcwd())
import velodepth.ops.losses
_pad=F.pad
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
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.set_float32_matmul_precision('highest')
print('[DETERMINISTIC REFLECT CPU BACKWARD PROTOTYPE]',flush=True)
