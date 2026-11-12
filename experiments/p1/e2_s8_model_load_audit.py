import copy
import json
import os
import sys
import traceback
from pathlib import Path

import torch
from safetensors.torch import load_file
from velodepth.models.velodepth import VeloDepth
from velodepth.utils.misc import get_params


def main():
    snapshot = Path(os.environ["VDEEP_SNAPSHOT"])
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"

    assert config_path.is_file(), f"配置不存在：{config_path}"
    assert weights_path.is_file(), f"权重不存在：{weights_path}"

    print("STAGE 1: read config", flush=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)

    config["model"]["persistent_feature_repair"] = {
        "enabled": True,
        "mode": "post_readout",
        "bottleneck": 32,
    }

    print("STAGE 2: construct model on CPU", flush=True)
    model = VeloDepth(config)
    model.cpu()

    decoder = model.pixel_decoder
    head = decoder.persistent_feature_repair

    assert head is not None, "修复头没有成功注册"
    assert decoder.feature_repair_mode == "post_readout"

    head_params = list(head.parameters())
    print("修复模式：", decoder.feature_repair_mode)
    print("修复头参数量：", sum(p.numel() for p in head_params))
    print("STAGE 3: load original checkpoint on CPU", flush=True)

    state = load_file(str(weights_path), device="cpu")
    result = model.load_state_dict(state, strict=False)

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    expected_prefix = "pixel_decoder.persistent_feature_repair."
    bad_missing = [k for k in missing if not k.startswith(expected_prefix)]

    print("缺失键数量：", len(missing))
    print("意外键数量：", len(unexpected))
    print("新增修复头缺失键数量：",
          sum(k.startswith(expected_prefix) for k in missing))

    if bad_missing:
        print("非预期缺失键示例：", bad_missing[:30])
    if unexpected:
        print("意外键示例：", unexpected[:30])

    assert not bad_missing, "存在非预期缺失键"
    assert not unexpected, "存在意外权重键"

    del state

    print("STAGE 4: verify optimizer parameter registration", flush=True)
    groups, _ = get_params(model.pixel_decoder, lr=1e-4, wd=0.01)
    registered_ids = {
        id(p)
        for group in groups
        for p in group["params"]
    }

    trainable = [p for p in head_params if p.requires_grad]
    assert trainable, "修复头没有可训练参数"
    unregistered = [p for p in trainable if id(p) not in registered_ids]

    print("修复头可训练张量数：", len(trainable))
    print("未注册张量数：", len(unregistered))
    assert not unregistered, "部分修复头参数未进入优化器参数组"

    print("STAGE 5: verify shape preservation and identity", flush=True)
    head.eval()
    features = [
        torch.randn(1, 512, 32, 32),
        torch.randn(1, 512, 16, 16),
        torch.randn(1, 512, 8, 8),
        torch.randn(1, 512, 4, 4),
    ]

    with torch.no_grad():
        repaired = head(features)

    assert len(repaired) == len(features)
    assert all(a.shape == b.shape for a, b in zip(features, repaired))

    max_diff = max(
        (a - b).abs().max().item()
        for a, b in zip(features, repaired)
    )
    print("初始最大绝对差：", max_diff)
    assert max_diff == 0.0, "修复头初始输出不是严格恒等"

    print("\nE2-S8 MODEL LOAD AUDIT: PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
