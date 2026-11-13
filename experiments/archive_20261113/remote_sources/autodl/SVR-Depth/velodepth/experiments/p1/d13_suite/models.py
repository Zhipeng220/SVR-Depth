"""Real four-scale residual operators. No GT, future frame, or sequence ID enters forward."""
import torch
from torch import nn
from torch.nn import functional as F

class ResidualPaths(nn.Module):
    def __init__(self,policy,arm):
        super().__init__();self.policy=policy;self.arm=arm;c=policy['feature_channels'];h=policy['hidden'];levels=policy['levels']
        self.trunks=nn.ModuleList([nn.Sequential(nn.Conv2d(c+4,h,1,bias=False),nn.GroupNorm(8,h),nn.GELU(),
            nn.Conv2d(h,h,3,padding=1,bias=False),nn.GroupNorm(8,h),nn.GELU()) for _ in range(levels)])
        self.read_heads=nn.ModuleList([nn.Conv2d(h,c,1) for _ in range(levels)])
        self.trans_heads=nn.ModuleList([nn.Conv2d(h,c,1) for _ in range(levels)])
        self.gates=nn.ModuleList([nn.Conv2d(h,1,1) for _ in range(levels)]) if policy['experiment'].startswith('D') else None
        for layer in [*self.read_heads,*self.trans_heads,*list(self.gates or [])]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def envelope(self,delta,base):
        k=self.policy['envelope_block'];eta=self.policy['envelope_eta'];x=base.float();d=delta.float()
        # Ceil pooling counts actual pixels, including partial boundary blocks.
        mean=F.avg_pool2d(x,k,k,ceil_mode=True,count_include_pad=False)
        var=(F.avg_pool2d(x.square(),k,k,ceil_mode=True,count_include_pad=False)-mean.square()).clamp_min(0)
        # Positive epsilon avoids sqrt'(0) NaN at the zero-initialized residual.
        std=(var.mean(1,keepdim=True)+1e-12).sqrt()
        rms=(F.avg_pool2d(d.square().mean(1,keepdim=True),k,k,ceil_mode=True,count_include_pad=False)+1e-12).sqrt()
        scale=F.interpolate(torch.tanh(eta*std/(rms+1e-6)),size=d.shape[-2:],mode='nearest')
        return (d*scale).to(delta.dtype)

    def forward(self,features,causal,force_alpha=None):
        assert len(features)==self.policy['levels']
        reads=[];commits=[];deltas=[];alphas=[];ungated=[]
        for i,x in enumerate(features):
            z=F.interpolate(causal.float(),size=x.shape[-2:],mode='bilinear',align_corners=False).to(x.dtype)
            h=self.trunks[i](torch.cat([x,z],1));dr=self.read_heads[i](h);dt=self.trans_heads[i](h)
            if self.gates is not None:
                dt=self.envelope(dt,x)
                alpha=torch.sigmoid(self.gates[i](h))
                if self.arm.endswith('_A') or force_alpha is not None:alpha=torch.full_like(alpha,.5 if force_alpha is None else float(force_alpha))
                read_delta=alpha*dt+(1-alpha)*dr;commit_delta=alpha*dt;alphas.append(alpha)
            else:
                if self.arm.endswith('_A'):dr=dt=(dr+dt)*.5
                read_delta,commit_delta=dr,dt
            reads.append(x+read_delta);commits.append(x+commit_delta);ungated.append(x+dt);deltas.append(commit_delta)
        norm=sum(d.float().square().sum() for d in deltas)/sum(x.detach().float().square().sum() for x in features).clamp_min(1e-12)
        return dict(read=reads,commit=commits,ungated=ungated,alpha=alphas,regularizer=norm)

def gate_loss(alphas,delta,tau=0.):
    # Frame-level counterfactual target, detached; NOT a certified per-pixel risk label.
    target=(delta.detach() < -tau).float()
    return torch.stack([F.binary_cross_entropy(a.float().clamp(1e-6,1-1e-6),target.expand_as(a)) for a in alphas]).mean()
