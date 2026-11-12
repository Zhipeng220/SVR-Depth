import copy
import hashlib
import inspect
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from velodepth.models.velodepth import VeloDepth


def read_rgb_index(root, start=8, count=3):
    index = root / "rgb.txt"
    if not index.is_file():
        raise FileNotFoundError(f"RGB index not found: {index}")

    records = []
    for line in index.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        try:
            timestamp = float(parts[0])
        except ValueError:
            continue

        rel = Path(parts[1])
        candidates = (
            [rel] if rel.is_absolute() else [
                root / rel,
                root / "rgb" / rel,
                root / "rgb" / rel.name,
                root / rel.name,
            ]
        )
        image_path = next(
            (p for p in candidates if p.is_file()), None
        )
        if image_path is None:
            raise FileNotFoundError(
                f"Cannot resolve RGB index entry: {parts[1]}"
            )

        records.append({
            "timestamp": timestamp,
            "path": image_path,
        })

    selected = records[start:start + count]
    if len(selected) != count:
        raise RuntimeError(
            f"需要 {count} 帧，实际找到 {len(selected)} 帧"
        )
    return selected


def summarize_grads(grads):
    present = [g.detach() for g in grads if g is not None]
    if not present:
        return {
            "gradient_tensors": 0,
            "nonzero_tensors": 0,
            "max_abs": 0.0,
            "l1": 0.0,
            "all_finite": True,
        }

    return {
        "gradient_tensors": len(present),
        "nonzero_tensors": sum(
            int(torch.count_nonzero(g).item() > 0)
            for g in present
        ),
        "max_abs": max(g.abs().max().item() for g in present),
        "l1": sum(g.abs().sum().item() for g in present),
        "all_finite": all(
            torch.isfinite(g).all().item() for g in present
        ),
    }


def main():
    torch.manual_seed(20261024)
    torch.set_num_threads(4)

    if not torch.cuda.is_available():
        raise RuntimeError("S4 要求 CUDA，拒绝退化为 CPU 测试")

    root = Path(os.environ["BONN_SEQUENCE_ROOT"])
    snapshot = Path(os.environ["VDEEP_SNAPSHOT"])
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"

    for p in (root / "rgb.txt", config_path, weights_path):
        if not p.is_file():
            raise FileNotFoundError(f"必要文件不存在：{p}")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    records = read_rgb_index(root, start=8, count=3)
    frames = []

    for i, record in enumerate(records):
        with Image.open(record["path"]) as im:
            frames.append(np.asarray(im.convert("RGB")).copy())

        print(
            f"frame={i+1}, timestamp={record['timestamp']}, "
            f"file={record['path'].name}, "
            f"shape={frames[-1].shape}",
            flush=True,
        )

    if len({x.shape for x in frames}) != 1:
        raise RuntimeError("三帧尺寸不一致")

    rgbs = torch.from_numpy(
        np.stack(frames)
    ).permute(0, 3, 1, 2).contiguous()

    print("STAGE 1: construct and load original checkpoint", flush=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["model"]["persistent_feature_repair"] = {
        "enabled": True,
        "mode": "post_readout",
        "bottleneck": 32,
    }

    model = VeloDepth(config)

    state = load_file(str(weights_path), device="cpu")
    load_result = model.load_state_dict(state, strict=False)

    prefix = "pixel_decoder.persistent_feature_repair."
    bad_missing = [
        k for k in load_result.missing_keys
        if not k.startswith(prefix)
    ]

    print("Missing keys:", len(load_result.missing_keys), flush=True)
    print("Unexpected keys:", len(load_result.unexpected_keys), flush=True)

    if bad_missing or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={bad_missing[:10]}, "
            f"unexpected={load_result.unexpected_keys[:10]}"
        )

    del state

    model.to(device)
    model.eval()  # 单独调用：此项目的 train() 不返回 self

    # 冻结原模型，只允许修复头参数参与梯度计算。
    for p in model.parameters():
        p.requires_grad_(False)

    decoder = model.pixel_decoder
    head = decoder.persistent_feature_repair

    if head is None:
        raise RuntimeError("修复头没有注册")

    for p in head.parameters():
        p.requires_grad_(True)

    decoder.feature_repair_mode = "post_readout"

    # 与 S3 使用相同的三帧传播条件。
    model.resolution_level = 0
    model.interpolation_mode = "bilinear"
    model.alpha_flow = 1e9
    model.beta_flow = 0.0
    model.alpha_luma = 1e9
    model.beta_luma = 0.0

    # 关键：解除测试方法上的 no_grad 装饰器。
    # infer 的输入预处理、DIS 光流和真实传播流程仍使用项目原实现。
    raw_encode_test = inspect.unwrap(
        type(model).encode_decode_test
    )
    model.encode_decode_test = MethodType(raw_encode_test, model)

    raw_infer = inspect.unwrap(type(model).infer)

    # 统计实际关键帧与快速传播调用。
    calls = {"keyframe": 0, "fast": 0}

    original_keyframe = decoder.forward_keyframe
    original_fast = decoder.forward_fast

    def counted_keyframe(*args, **kwargs):
        calls["keyframe"] += 1
        return original_keyframe(*args, **kwargs)

    def counted_fast(*args, **kwargs):
        calls["fast"] += 1
        return original_fast(*args, **kwargs)

    decoder.forward_keyframe = counted_keyframe
    decoder.forward_fast = counted_fast

    print(
        "STAGE 2: run actual three-frame inference graph",
        flush=True,
    )
    print(
        "注意：本轮不人为扰动修复头；保持零初始化。",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats(device)

    # 去掉 infer/encode_decode_test 上的 no_grad，
    # 但保留 FP16 autocast 以控制显存。
    with torch.enable_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = raw_infer(
                model, rgbs=rgbs, normalize=True
            )

    if "depth" not in outputs:
        raise RuntimeError(
            f"缺少 depth 输出，现有键：{list(outputs.keys())}"
        )

    depth = outputs["depth"]

    if depth.ndim != 4 or depth.shape[0] != 3:
        raise RuntimeError(
            f"预期 depth 为 [3,C,H,W]，实际 {tuple(depth.shape)}"
        )

    if not torch.isfinite(depth).all():
        raise RuntimeError("预测深度包含 NaN 或 Inf")

    if calls != {"keyframe": 1, "fast": 2}:
        raise RuntimeError(
            f"未得到预期的 1 次关键帧 + 2 次快速传播：{calls}"
        )

    print("实际调用次数：", calls, flush=True)
    print("深度输出形状：", tuple(depth.shape), flush=True)
    print(
        "STAGE 3: compare frame-2 and frame-3 gradient paths",
        flush=True,
    )

    params = list(head.parameters())
    if not params or not all(p.requires_grad for p in params):
        raise RuntimeError("修复头参数未全部设为可训练")

    # D 模式：第二帧当前输出应与修复头无梯度关联；
    # 第三帧输出则应通过第二帧写回的状态关联到修复头。
    loss_frame2 = depth[1].float().square().mean()
    loss_frame3 = depth[2].float().square().mean()

    if not loss_frame2.requires_grad or not loss_frame3.requires_grad:
        raise RuntimeError(
            "输出计算图未保留；无法执行真实梯度路径验收"
        )

    grads_frame2 = torch.autograd.grad(
        loss_frame2,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    stats_frame2 = summarize_grads(grads_frame2)

    grads_frame3 = torch.autograd.grad(
        loss_frame3,
        params,
        retain_graph=False,
        allow_unused=True,
    )
    stats_frame3 = summarize_grads(grads_frame3)

    print("frame-2 loss:", loss_frame2.item(), flush=True)
    print("frame-2 gradient:", stats_frame2, flush=True)
    print("frame-3 loss:", loss_frame3.item(), flush=True)
    print("frame-3 gradient:", stats_frame3, flush=True)

    report = {
        "experiment": "E2-S8-S4 real propagation graph gradient audit",
        "scope": (
            "actual VeloDepth three-frame inference graph; "
            "zero-initialized repair head; synthetic scalar loss; no GT"
        ),
        "timestamp": datetime.now().astimezone().isoformat(),
        "sequence_root": str(root),
        "frames": [
            {
                "timestamp": r["timestamp"],
                "filename": r["path"].name,
            }
            for r in records
        ],
        "checkpoint": {
            "missing_keys": len(load_result.missing_keys),
            "unexpected_keys": len(load_result.unexpected_keys),
            "non_head_missing_keys": bad_missing,
        },
        "repair_head": {
            "parameters": sum(p.numel() for p in params),
            "initialization": "zero-initialized residual; no synthetic perturbation",
            "mode": decoder.feature_repair_mode,
        },
        "call_counts": calls,
        "depth_shape": list(depth.shape),
        "frame2_only_loss_gradient": stats_frame2,
        "frame3_only_loss_gradient": stats_frame3,
        "gpu": torch.cuda.get_device_name(0),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in (
                "velodepth/models/decoder_velo.py",
                "velodepth/models/persistent_feature_repair.py",
                "experiments/p1/e2_s8_real_graph_gradient_audit.py",
            )
        },
    }

    report_path = Path(
        "outputs/e2_s8_validation/real_graph_gradient_s4.json"
    )
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    # 验收条件：第二帧无修复头梯度，第三帧有有限、非零梯度。
    if not stats_frame2["all_finite"] or not stats_frame3["all_finite"]:
        raise AssertionError("梯度出现 NaN 或 Inf")

    if stats_frame2["max_abs"] > 1e-12:
        raise AssertionError(
            "D 模式异常：第二帧当前输出对修复头产生了非零梯度"
        )

    if stats_frame3["nonzero_tensors"] == 0 or stats_frame3["l1"] <= 0:
        raise AssertionError(
            "第三帧损失未对修复头产生非零梯度"
        )

    print("\nE2-S8-S4 REAL GRAPH GRADIENT: PASS", flush=True)
    print("报告：", report_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
