from __future__ import annotations

from typing import Sequence
import torch
from torch import nn


class PersistentFeatureRepairHead(nn.Module):
    """Zero-initialized residual repair for four BCHW persistent feature maps."""
    def __init__(self, channels: int = 256, bottleneck: int = 32, levels: int = 4):
        super().__init__()
        if min(channels, bottleneck, levels) < 1:
            raise ValueError("channels, bottleneck, and levels must be positive")
        groups = 8 if bottleneck % 8 == 0 else 1
        self.channels, self.bottleneck, self.levels = channels, bottleneck, levels
        blocks = []
        for _ in range(levels):
            block = nn.Sequential(
                nn.Conv2d(channels, bottleneck, 1, bias=False),
                nn.GroupNorm(groups, bottleneck),
                nn.GELU(),
                nn.Conv2d(bottleneck, bottleneck, 3, padding=1, bias=False),
                nn.GroupNorm(groups, bottleneck),
                nn.GELU(),
                nn.Conv2d(bottleneck, channels, 1, bias=True),
            )
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if not isinstance(features, (list, tuple)) or len(features) != self.levels:
            raise ValueError(f"Expected a list/tuple of {self.levels} feature maps")
        out = []
        for i, (x, block) in enumerate(zip(features, self.blocks)):
            if not isinstance(x, torch.Tensor) or x.ndim != 4:
                raise ValueError(f"Feature level {i} must be a BCHW tensor")
            if x.shape[1] != self.channels:
                raise ValueError(
                    f"Feature level {i}: expected {self.channels} channels, got {x.shape[1]}"
                )
            out.append(x + block(x))
        return out
