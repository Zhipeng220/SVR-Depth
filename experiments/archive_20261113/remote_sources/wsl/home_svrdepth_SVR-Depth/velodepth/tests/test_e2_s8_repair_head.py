import torch
from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead


def main():
    torch.manual_seed(7)
    head = PersistentFeatureRepairHead(channels=64, bottleneck=16, levels=4)
    x = [torch.randn(2, 64, 12, 16) for _ in range(4)]
    y = head(x)

    assert len(y) == 4
    assert all(a.shape == b.shape for a, b in zip(x, y))
    max_diff = max((a - b).abs().max().item() for a, b in zip(x, y))
    assert max_diff == 0.0, f"zero-init head must start as exact identity, max diff={max_diff}"

    loss = sum(t.square().mean() for t in y)
    loss.backward()
    assert any(p.grad is not None for p in head.parameters()), "head received no gradients"

    nparams = sum(p.numel() for p in head.parameters())
    print("E2-S8 repair-head unit test: PASS")
    print("levels=4; shape-preserving=True; initial_max_abs_diff=", max_diff)
    print("parameters=", nparams)
    print("gradient_path=True")


if __name__ == "__main__":
    main()
