#!/usr/bin/env python3
"""Check the runtime dependencies and exercise the repair head on CPU."""
from __future__ import annotations

import importlib
import sys

REQUIRED = [
    "torch", "torchvision", "einops", "timm", "huggingface_hub",
    "safetensors", "cv2", "PIL", "numpy",
]

def main() -> int:
    missing = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "available")
            print(f"[OK] {name}: {version}")
        except Exception as exc:
            missing.append((name, exc))
            print(f"[MISSING] {name}: {type(exc).__name__}: {exc}")
    if missing:
        print("\nInstall the project requirements, then rerun this check.")
        return 1

    import torch
    from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead
    head = PersistentFeatureRepairHead(channels=32, bottleneck=8, levels=4).eval()
    xs = [torch.randn(1, 32, 8, 8) for _ in range(4)]
    with torch.no_grad():
        ys = head(xs)
    assert all(torch.equal(x, y) for x, y in zip(xs, ys))
    print("[PASS] persistent repair head identity smoke test")
    print(f"CUDA available: {torch.cuda.is_available()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
