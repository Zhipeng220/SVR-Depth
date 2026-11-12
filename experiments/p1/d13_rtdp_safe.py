import torch
from torch import nn
import torch.nn.functional as F

class D13RTDPSafe(nn.Module):
    """Gemini D-series: dual-path residual with envelope and causal harm gate.
    Input tensors are feature maps [B,C,H,W]; z is a causal feature map [B,Z,H,W].
    """
    def __init__(self, channels:int, z_channels:int, hidden:int=64, envelope_eta:float=1.0):
        super().__init__()
        self.envelope_eta=float(envelope_eta)
        self.trunk=nn.Sequential(
            nn.Conv2d(channels+z_channels,hidden,3,padding=1),
            nn.GELU(),
            nn.Conv2d(hidden,hidden,3,padding=1),
            nn.GELU(),
        )
        self.trans_head=nn.Conv2d(hidden,channels,1)
        self.read_head=nn.Conv2d(hidden,channels,1)
        self.gate_head=nn.Conv2d(hidden,1,1)
    def envelope(self, delta, state, eps=1e-6):
        # Local 8x8 statistics; pad/ceil pooling supports non-multiple sizes.
        local_var=F.avg_pool2d(state.square(),8,8,ceil_mode=True) - F.avg_pool2d(state,8,8,ceil_mode=True).square()
        local_std=local_var.clamp_min(0).mean(1,keepdim=True).sqrt()
        bound=self.envelope_eta*local_std
        mag=F.avg_pool2d(delta.square().mean(1,keepdim=True),8,8,ceil_mode=True).sqrt()
        scale=torch.tanh(bound/(mag+eps))
        scale=F.interpolate(scale,size=delta.shape[-2:],mode="nearest")
        return delta*scale
    def forward(self, state, z, force_alpha=None):
        h=self.trunk(torch.cat([state,z],dim=1))
        dt=self.envelope(self.trans_head(h),state)
        dr=self.read_head(h)
        alpha=torch.sigmoid(self.gate_head(h))
        if force_alpha is not None:
            alpha=torch.full_like(alpha,float(force_alpha))
        # Gemini's prescribed readout interpolation and persistent commit.
        read_feature=state + alpha*dt + (1-alpha)*dr
        committed_state=state + alpha*dt
        return {"read_feature":read_feature,"committed_state":committed_state,
                "delta_trans":dt,"delta_read":dr,"alpha":alpha}

def harm_gate_loss(alpha, delta_future, tau=0.0, mu=1.0):
    """delta_future = loss(commit)-loss(no-commit), computed from causal training pair."""
    target=(delta_future < -tau).to(alpha.dtype)
    if target.ndim==3: target=target.unsqueeze(1)
    if target.shape[-2:] != alpha.shape[-2:]:
        target=F.interpolate(target.float(),size=alpha.shape[-2:],mode="nearest")
    bce=F.binary_cross_entropy(alpha.clamp(1e-6,1-1e-6),target)
    harm=F.relu(delta_future).mean()
    return bce+mu*harm
