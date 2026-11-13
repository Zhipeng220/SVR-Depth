import pytest
import torch

from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead


def test_zero_initialized_head_is_exact_identity_and_differentiable():
    torch.manual_seed(7)
    head = PersistentFeatureRepairHead(channels=32, bottleneck=8, levels=4)
    xs = [torch.randn(2, 32, 8, 12, requires_grad=True) for _ in range(4)]
    ys = head(xs)
    assert len(ys) == 4
    assert all(torch.equal(x, y) for x, y in zip(xs, ys))
    sum(y.square().mean() for y in ys).backward()
    assert any(p.grad is not None for p in head.parameters())


def test_rejects_wrong_level_count():
    head = PersistentFeatureRepairHead(channels=32, bottleneck=8, levels=4)
    with pytest.raises(ValueError, match="Expected a list/tuple"):
        head([torch.randn(1, 32, 8, 8)])


def test_rejects_wrong_channel_count():
    head = PersistentFeatureRepairHead(channels=32, bottleneck=8, levels=4)
    xs = [torch.randn(1, 32, 8, 8) for _ in range(3)] + [torch.randn(1, 16, 8, 8)]
    with pytest.raises(ValueError, match="expected 32 channels"):
        head(xs)
