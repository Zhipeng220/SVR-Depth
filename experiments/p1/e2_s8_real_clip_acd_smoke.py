import copy
import json
import os
import sys
import traceback
from pathlib import Path

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
        if rel.is_absolute():
            candidates = [rel]
        else:
            candidates = [
                root / rel,
                root / "rgb" / rel,
                root / "rgb" / rel.name,
                root / rel.name,
            ]

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
            f"需要 {count} 帧，但索引从 {start} 开始只找到 "
            f"{len(selected)} 帧；总有效条目数={len(records)}"
        )

    return selected


def max_diff(a, b, frame_index):
    keys = ("depth", "distance", "confidence")
    values = {}

    for key in keys:
        if key not in a or key not in b:
            continue

        delta = (a[key][frame_index] - b[key][frame_index]).abs()
        values[key] = {
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
        }

    if not values:
        raise RuntimeError("没有可用于比较的输出张量")

    combined = max(v["max_abs"] for v in values.values())
    return {"combined_max_abs": combined, "per_output": values}


def main():
    torch.manual_seed(20261024)
    torch.set_num_threads(4)

    if not torch.cuda.is_available():
        raise RuntimeError("本轮要求使用 CUDA，不在 CPU 上启动完整推理")

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    root = Path(os.environ["BONN_SEQUENCE_ROOT"])
    snapshot = Path(os.environ["VDEEP_SNAPSHOT"])

    records = read_rgb_index(root, start=8, count=3)
    print("STAGE 1: selected real RGB frames", flush=True)

    frames = []
    for i, record in enumerate(records):
        with Image.open(record["path"]) as image:
            frames.append(np.asarray(image.convert("RGB")).copy())

        print(
            f"frame={i + 1}, "
            f"timestamp={record['timestamp']}, "
            f"file={record['path'].name}, "
            f"shape={frames[-1].shape}",
            flush=True,
        )

    if len({x.shape for x in frames}) != 1:
        raise RuntimeError("三帧图像尺寸不一致")

    rgbs = torch.from_numpy(np.stack(frames)).permute(
        0, 3, 1, 2
    ).contiguous()

    print("STAGE 2: construct model and load pretrained weights", flush=True)

    config = json.loads(
        (snapshot / "config.json").read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)
    config["model"]["persistent_feature_repair"] = {
        "enabled": True,
        "mode": "post_readout",
        "bottleneck": 32,
    }

    model = VeloDepth(config)

    state = load_file(
        str(snapshot / "model.safetensors"),
        device="cpu",
    )
    result = model.load_state_dict(state, strict=False)

    expected_prefix = "pixel_decoder.persistent_feature_repair."
    bad_missing = [
        k for k in result.missing_keys
        if not k.startswith(expected_prefix)
    ]

    print("Missing keys:", len(result.missing_keys), flush=True)
    print("Unexpected keys:", len(result.unexpected_keys), flush=True)

    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            f"权重加载异常：missing={bad_missing[:10]}, "
            f"unexpected={result.unexpected_keys[:10]}"
        )

    del state

    decoder = model.pixel_decoder
    head = decoder.persistent_feature_repair
    if head is None:
        raise RuntimeError("修复头未注册")

    # 仅用于可控的推理路径诊断，不代表训练所得参数。
    with torch.no_grad():
        for block in head.blocks:
            block[-1].weight.normal_(mean=0.0, std=1e-3)
            block[-1].bias.zero_()

    model.to(device)
    model.eval()

    # 固定三帧都走传播路径，避免刷新打断状态递推。
    model.resolution_level = 0
    model.interpolation_mode = "bilinear"
    model.alpha_flow = 1e9
    model.beta_flow = 0.0
    model.alpha_luma = 1e9
    model.beta_luma = 0.0

    # 记录实际关键帧与快速传播调用次数。
    stats = {"keyframe": 0, "fast": 0}
    original_keyframe = decoder.forward_keyframe
    original_fast = decoder.forward_fast

    def counted_keyframe(*args, **kwargs):
        stats["keyframe"] += 1
        return original_keyframe(*args, **kwargs)

    def counted_fast(*args, **kwargs):
        stats["fast"] += 1
        return original_fast(*args, **kwargs)

    decoder.forward_keyframe = counted_keyframe
    decoder.forward_fast = counted_fast

    results = {}
    call_counts = {}

    for label, mode in (
        ("A", "off"),
        ("C", "pre_readout"),
        ("D", "post_readout"),
    ):
        print(f"STAGE 3: run mode {label} ({mode})", flush=True)

        stats["keyframe"] = 0
        stats["fast"] = 0
        decoder.feature_repair_mode = mode

        with torch.inference_mode():
            output = model.infer(rgbs=rgbs, normalize=True)

        results[label] = {
            k: v.detach().float().cpu()
            for k, v in output.items()
            if isinstance(v, torch.Tensor)
        }
        call_counts[label] = dict(stats)

        print(
            f"mode={label}, "
            f"keyframes={stats['keyframe']}, "
            f"fast_calls={stats['fast']}",
            flush=True,
        )

        for key in ("depth", "distance", "confidence"):
            if key in results[label]:
                tensor = results[label][key]
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(
                        f"{label}/{key} 出现非有限值"
                    )
                print(
                    f"  {key}: shape={tuple(tensor.shape)}",
                    flush=True,
                )

        if stats["keyframe"] != 1 or stats["fast"] < 2:
            raise RuntimeError(
                f"{label} 未形成预期的三帧递推：{stats}"
            )

        del output
        torch.cuda.empty_cache()

    print("STAGE 4: compare actual model outputs", flush=True)

    differences = {
        "C_vs_A": {
            f"frame_{i + 1}": max_diff(
                results["C"], results["A"], i
            )
            for i in range(3)
        },
        "D_vs_A": {
            f"frame_{i + 1}": max_diff(
                results["D"], results["A"], i
            )
            for i in range(3)
        },
    }

    report = {
        "experiment": "E2-S8-S3 real clip A/C/D smoke",
        "scope": "real RGB inference; synthetic repair perturbation",
        "sequence_root": str(root),
        "frames": [
            {
                "timestamp": r["timestamp"],
                "filename": r["path"].name,
            }
            for r in records
        ],
        "call_counts": call_counts,
        "differences": differences,
    }

    output_path = Path(
        "outputs/e2_s8_validation/real_clip_acd_smoke.json"
    )
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    tol = 1e-6
    min_effect = 1e-8

    # Frame 1 is a keyframe, so the fast-path repair must not alter it.
    for label in ("C_vs_A", "D_vs_A"):
        if differences[label]["frame_1"]["combined_max_abs"] > tol:
            raise AssertionError(
                f"{label}: 首帧关键帧输出发生了非预期变化"
            )

    # D must preserve the current output of frame 2.
    if differences["D_vs_A"]["frame_2"]["combined_max_abs"] > tol:
        raise AssertionError(
            "D: 第二帧当前输出被意外改变"
        )

    # C repairs before readout, so frame 2 should respond.
    if differences["C_vs_A"]["frame_2"]["combined_max_abs"] <= min_effect:
        raise AssertionError(
            "C: 第二帧没有观察到修复对当前输出的影响"
        )

    # D repairs only the carried state, so frame 3 should respond.
    if differences["D_vs_A"]["frame_3"]["combined_max_abs"] <= min_effect:
        raise AssertionError(
            "D: 第三帧没有观察到持久状态修复的后续影响"
        )

    print("\nE2-S8-S3 REAL CLIP A/C/D: PASS", flush=True)
    print("报告：", output_path, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
