"""Output-only depth adapter used by the O_fast comparison."""
import torch
from torch import nn
from torch.nn import functional as F

class FastDepthAdapter(nn.Module):
    """Apply a bounded residual in log-depth without modifying streaming state."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, depth, rgb_small, gain=1.0):
        shape = depth.shape[-2:]
        small = F.interpolate(depth[None, None].float().clamp_min(1e-4).log(),
                              size=rgb_small.shape[-2:], mode='bilinear', align_corners=False)
        delta = 0.5 * torch.tanh(self.net(torch.cat([small, rgb_small], 1)))
        delta = F.interpolate(delta.float(), size=shape, mode='bilinear', align_corners=False)[0, 0]
        return depth.float() * torch.exp(float(gain) * delta)
