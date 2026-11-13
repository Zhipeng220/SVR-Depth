#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.3 v5.25a — Minimal Tile/Halo feasibility preflight for VeloDepth.

This experiment:
  1) captures the exact packed inputs used by VeloDepth.infer(),
  2) profiles the existing full-frame streaming path on a short sequence,
  3) benchmarks full-frame pixel_encoder vs a central crop and a crop with halo,
  4) compares alignable encoder feature regions and estimates pixel-area reduction.

It does NOT implement local decoder refresh, alter weights, or claim end-to-end
speedup. All crop/feature alignment checks are diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

from velodepth.models.velodepth import VeloDepth
from stream_state import StreamState
from velodepth_stream import VeloDepthStream

VERSION = "P2.3 v5.25a Tile/Halo feasibility preflight"


def clone_payload(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: clone_payload(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clone_payload(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone_payload(v) for v in x)
    return copy.deepcopy(x)


def load_rgb(folder: str, count: int) -> torch.Tensor:
    paths = sorted(Path(folder).glob("*.png"))[:count]
    if len(paths) != count:
        raise RuntimeError(f"Expected at least {count} PNG images in {folder}, found {len(paths)}")
    frames = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read {p}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_shapes(x: Any, prefix: str = "out") -> dict[str, list[int]]:
    found = {}
    if isinstance(x, torch.Tensor):
        found[prefix] = list(x.shape)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            found.update(tensor_shapes(v, f"{prefix}.{i}"))
    elif isinstance(x, dict):
        for k, v in x.items():
            found.update(tensor_shapes(v, f"{prefix}.{k}"))
    return found


def flatten_tensors(x: Any, prefix: str = "out") -> dict[str, torch.Tensor]:
    found = {}
    if isinstance(x, torch.Tensor):
        found[prefix] = x
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            found.update(flatten_tensors(v, f"{prefix}.{i}"))
    elif isinstance(x, dict):
        for k, v in x.items():
            found.update(flatten_tensors(v, f"{prefix}.{k}"))
    return found


def discover_patch_stride(module: nn.Module) -> tuple[int | None, list[dict]]:
    candidates = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            stride = child.stride
            kernel = child.kernel_size
            if stride[0] == stride[1] and kernel[0] == kernel[1] and stride[0] > 1 and kernel[0] == stride[0]:
                candidates.append({
                    "module": name,
                    "stride": int(stride[0]),
                    "kernel": int(kernel[0]),
                    "reason": "Conv2d kernel_size == stride",
                })
    # Prefer the largest patchifying convolution; report all candidates for audit.
    if candidates:
        candidates.sort(key=lambda x: (x["stride"], x["kernel"]), reverse=True)
        return candidates[0]["stride"], candidates
    # Try explicit encoder attributes without assuming a particular implementation.
    for obj_name, obj in [("pixel_encoder", module)] + [
        (n, m) for n, m in module.named_modules()
    ]:
        for attr in ("patch_size", "patch_stride", "stride"):
            val = getattr(obj, attr, None)
            if isinstance(val, int) and val > 1:
                return int(val), [{"module": obj_name, "stride": int(val), "reason": f"attribute {attr}"}]
            if isinstance(val, (tuple, list)) and len(val) == 2 and val[0] == val[1] and isinstance(val[0], int) and val[0] > 1:
                return int(val[0]), [{"module": obj_name, "stride": int(val[0]), "reason": f"attribute {attr}"}]
    return None, candidates


def align_down(x: int, stride: int) -> int:
    return max(0, (x // stride) * stride)


def align_up(x: int, stride: int, limit: int) -> int:
    return min(limit, int(math.ceil(x / stride) * stride))


def choose_core_and_tile(H: int, W: int, stride: int, halo: int):
    # Central core covers approximately half height x half width (25% area).
    core_h = max(stride, (max(stride, H // 2) // stride) * stride)
    core_w = max(stride, (max(stride, W // 2) // stride) * stride)
    core_y0 = align_down((H - core_h) // 2, stride)
    core_x0 = align_down((W - core_w) // 2, stride)
    core_y1 = min(H, core_y0 + core_h)
    core_x1 = min(W, core_x0 + core_w)
    tile_y0 = align_down(max(0, core_y0 - halo), stride)
    tile_x0 = align_down(max(0, core_x0 - halo), stride)
    tile_y1 = align_up(min(H, core_y1 + halo), stride, H)
    tile_x1 = align_up(min(W, core_x1 + halo), stride, W)
    return {
        "core": (core_y0, core_y1, core_x0, core_x1),
        "tile_halo": (tile_y0, tile_y1, tile_x0, tile_x1),
        "tile_no_halo": (core_y0, core_y1, core_x0, core_x1),
    }


def run_encoder_bench(model, image: torch.Tensor, warmup: int, repeats: int, device: torch.device):
    def one():
        with torch.no_grad():
            return model.pixel_encoder(image)

    for _ in range(warmup):
        _ = one()
    sync(device)
    times, last = [], None
    for _ in range(repeats):
        sync(device)
        t0 = time.perf_counter()
        last = one()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return last, {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "repeats": int(repeats),
    }


def tensor_to_grid(t: torch.Tensor, input_hw: tuple[int, int], patch_stride: int | None):
    """Return B,Hf,Wf,C grid and a layout label if the tensor can be spatially aligned."""
    H, W = input_hw
    if t.ndim == 4:
        # Most feature maps are BCHW. Accept BHWC only when last dim looks like channels.
        if t.shape[-2] > 1 and t.shape[-1] > 1:
            return t.permute(0, 2, 3, 1), "BCHW"
    if t.ndim == 3 and patch_stride:
        gh = int(round(H / patch_stride))
        gw = int(round(W / patch_stride))
        if t.shape[1] == gh * gw:
            return t.reshape(t.shape[0], gh, gw, t.shape[-1]), "BNC->BHWC"
    return None, "not_spatially_aligned"


def crop_grid_region(grid: torch.Tensor, input_hw: tuple[int, int], box: tuple[int, int, int, int]):
    H, W = input_hw
    y0, y1, x0, x1 = box
    gh, gw = grid.shape[1], grid.shape[2]
    fy0 = int(round(y0 / H * gh))
    fy1 = int(round(y1 / H * gh))
    fx0 = int(round(x0 / W * gw))
    fx1 = int(round(x1 / W * gw))
    return grid[:, fy0:fy1, fx0:fx1, :]


def feature_compare(full_out: Any, crop_out: Any, full_hw: tuple[int, int],
                    crop_hw: tuple[int, int], full_core_box: tuple[int, int, int, int],
                    crop_core_box: tuple[int, int, int, int], patch_stride: int | None):
    full_leaves = flatten_tensors(full_out)
    crop_leaves = flatten_tensors(crop_out)
    rows = []
    for key in sorted(set(full_leaves) & set(crop_leaves)):
        a, b = full_leaves[key], crop_leaves[key]
        ga, layout_a = tensor_to_grid(a, full_hw, patch_stride)
        gb, layout_b = tensor_to_grid(b, crop_hw, patch_stride)
        row = {
            "tensor_path": key,
            "full_shape": str(list(a.shape)),
            "crop_shape": str(list(b.shape)),
            "layout": layout_a if layout_a == layout_b else f"{layout_a}/{layout_b}",
            "comparable": False,
            "core_mae": None,
            "core_max_abs": None,
            "core_relative_l2": None,
            "core_cosine": None,
            "note": "",
        }
        if ga is None or gb is None:
            row["note"] = "Could not infer a common spatial feature grid."
            rows.append(row)
            continue
        # Map the same full-frame core to the full and cropped encoder outputs.
        a_core = crop_grid_region(ga, full_hw, full_core_box)
        b_core = crop_grid_region(gb, crop_hw, crop_core_box)
        if a_core.shape != b_core.shape or a_core.numel() == 0:
            row["note"] = f"Core grid shape mismatch: {list(a_core.shape)} vs {list(b_core.shape)}"
            rows.append(row)
            continue
        af = a_core.detach().float()
        bf = b_core.detach().float()
        diff = (af - bf).abs()
        denom = torch.linalg.vector_norm(af).item()
        rel_l2 = torch.linalg.vector_norm(af - bf).item() / max(denom, 1e-12)
        cos = torch.nn.functional.cosine_similarity(af.reshape(1, -1), bf.reshape(1, -1), dim=1).item()
        row.update({
            "comparable": True,
            "core_mae": float(diff.mean().item()),
            "core_max_abs": float(diff.max().item()),
            "core_relative_l2": float(rel_l2),
            "core_cosine": float(cos),
            "note": "Compared corresponding central-core feature region; crop may alter global context/positional encoding.",
        })
        rows.append(row)
    return rows


def profile_stream(model, packed_inputs, metas, device):
    stream = VeloDepthStream(model)
    timings = []
    handles = []

    def make_hooks(name):
        def pre(module, inputs):
            sync(device)
            module.__dict__["_v525_start"] = time.perf_counter()
        def post(module, inputs, output):
            sync(device)
            t0 = module.__dict__.pop("_v525_start", None)
            if t0 is not None:
                first = inputs[0] if inputs else None
                timings.append({
                    "component": name,
                    "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                    "input_shape": str(list(first.shape)) if isinstance(first, torch.Tensor) else "",
                    "output_shapes": json.dumps(tensor_shapes(output), ensure_ascii=False),
                })
        return pre, post

    for name in ["pixel_encoder", "small_encoder"]:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            pre, post = make_hooks(name)
            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(post))

    decoder = model.pixel_decoder
    method_originals = {}
    for method_name in ["forward_keyframe", "forward_fast"]:
        original = getattr(decoder, method_name)
        method_originals[method_name] = original
        def make_wrapper(method, name):
            def wrapped(*args, **kwargs):
                sync(device)
                t0 = time.perf_counter()
                result = method(*args, **kwargs)
                sync(device)
                timings.append({
                    "component": f"pixel_decoder.{name}",
                    "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                    "input_shape": "",
                    "output_shapes": json.dumps(tensor_shapes(result), ensure_ascii=False),
                })
                return result
            return wrapped
        setattr(decoder, method_name, make_wrapper(original, method_name))

    try:
        with torch.no_grad():
            _ = stream.run(clone_payload(packed_inputs), clone_payload(metas))
    finally:
        for h in handles:
            h.remove()
        for name, original in method_originals.items():
            setattr(decoder, name, original)

    df = pd.DataFrame(timings)
    summary = []
    if not df.empty:
        for component, group in df.groupby("component", sort=False):
            summary.append({
                "component": component,
                "calls": int(len(group)),
                "total_ms": float(group.elapsed_ms.sum()),
                "mean_ms_per_call": float(group.elapsed_ms.mean()),
                "median_ms_per_call": float(group.elapsed_ms.median()),
            })
    return df, pd.DataFrame(summary), stream.trace


def main():
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--rgb", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--resolution_level", type=int, default=0)
    parser.add_argument("--halo", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    if args.frames < 3:
        parser.error("--frames must be >= 3 to profile both keyframe and propagation.")
    if args.halo < 0 or args.repeats < 1 or args.warmup < 0:
        parser.error("halo must be >= 0, repeats >= 1, warmup >= 0.")
    if not torch.cuda.is_available():
        raise RuntimeError("This is a GPU profiling experiment; run on an AutoDL GPU instance.")

    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else ROOT
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        root / "outputs" / "p2_3_v5_25_tile_halo_feasibility"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    print(f"[1/5] Loading VeloDepth; root={root}")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(device)
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    print(f"[2/5] Capturing exact packed inputs from {args.frames} frames")
    rgbs = load_rgb(args.rgb, args.frames)
    captured = {}
    original = model.encode_decode_test

    def capture(inputs, image_metas):
        captured["inputs"] = clone_payload(inputs)
        captured["metas"] = clone_payload(image_metas)
        return original(inputs, image_metas)

    model.encode_decode_test = capture
    try:
        with torch.no_grad():
            _ = model.infer(rgbs, normalize=True)
    finally:
        model.encode_decode_test = original

    if "inputs" not in captured:
        raise RuntimeError("Failed to capture the exact packed payload used by encode_decode_test().")
    raw_payload = clone_payload(captured["inputs"])
    metas = clone_payload(captured["metas"])
    T = int(metas[0]["T"])
    if T < 3:
        raise RuntimeError(f"Captured T={T}; expected >= 3.")

    print("[3/5] Profiling the existing full-frame streaming path")
    component_detail, component_summary, trace = profile_stream(model, raw_payload, metas, device)
    component_detail.to_csv(out_dir / "p2_3_v5_25_component_profile_detail.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(out_dir / "p2_3_v5_25_component_profile_summary.csv", index=False, encoding="utf-8-sig")

    print("[4/5] Benchmarking full-frame, tile-only, and tile+halo pixel_encoder")
    prepared = clone_payload(raw_payload)
    model._set_image_paddings(prepared)
    image_bt = model._bt(prepared["image"], 1, T)
    image = image_bt[:, 0].contiguous()
    H, W = int(image.shape[-2]), int(image.shape[-1])

    stride, stride_candidates = discover_patch_stride(model.pixel_encoder)
    if stride is None:
        # Keep the crop test runnable, but explicitly mark alignment as unknown.
        stride = 1
        stride_note = "Could not infer patch stride; crops are pixel-aligned only and token-grid comparisons may be skipped."
    else:
        stride_note = f"Detected patch/alignment stride {stride}."
    boxes = choose_core_and_tile(H, W, stride, args.halo)
    core = boxes["core"]
    tile_halo = boxes["tile_halo"]
    tile_no_halo = boxes["tile_no_halo"]

    full_out, full_time = run_encoder_bench(model, image, args.warmup, args.repeats, device)

    def crop_image(box):
        y0, y1, x0, x1 = box
        return image[..., y0:y1, x0:x1].contiguous()

    no_halo_image = crop_image(tile_no_halo)
    halo_image = crop_image(tile_halo)
    no_halo_out = None
    halo_out = None
    no_halo_time = {"status": "not_run"}
    halo_time = {"status": "not_run"}
    no_halo_error = ""
    halo_error = ""
    try:
        no_halo_out, no_halo_time = run_encoder_bench(model, no_halo_image, args.warmup, args.repeats, device)
        no_halo_time["status"] = "ok"
    except Exception as e:
        no_halo_error = f"{type(e).__name__}: {e}"
        no_halo_time = {"status": "error", "error": no_halo_error}
    try:
        halo_out, halo_time = run_encoder_bench(model, halo_image, args.warmup, args.repeats, device)
        halo_time["status"] = "ok"
    except Exception as e:
        halo_error = f"{type(e).__name__}: {e}"
        halo_time = {"status": "error", "error": halo_error}

    full_area = H * W
    core_y0, core_y1, core_x0, core_x1 = core
    hy0, hy1, hx0, hx1 = tile_halo
    ny0, ny1, nx0, nx1 = tile_no_halo
    core_hw = (core_y1 - core_y0, core_x1 - core_x0)
    halo_hw = (hy1 - hy0, hx1 - hx0)
    nohalo_hw = (ny1 - ny0, nx1 - nx0)

    feature_rows = []
    if no_halo_out is not None:
        feature_rows.extend(feature_compare(
            full_out, no_halo_out, (H, W), nohalo_hw, core, (0, core_hw[0], 0, core_hw[1]), stride
        ))
        for r in feature_rows[-len(flatten_tensors(full_out)):]:
            r["variant"] = "tile_no_halo"
    if halo_out is not None:
        halo_core_box = (core_y0 - hy0, core_y1 - hy0, core_x0 - hx0, core_x1 - hx0)
        halo_rows = feature_compare(
            full_out, halo_out, (H, W), halo_hw, core, halo_core_box, stride
        )
        for r in halo_rows:
            r["variant"] = "tile_halo"
        feature_rows.extend(halo_rows)
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(out_dir / "p2_3_v5_25_feature_equivalence.csv", index=False, encoding="utf-8-sig")

    full_shapes = tensor_shapes(full_out)
    tile_shapes = {
        "full": full_shapes,
        "tile_no_halo": tensor_shapes(no_halo_out) if no_halo_out is not None else {},
        "tile_halo": tensor_shapes(halo_out) if halo_out is not None else {},
    }
    benchmark = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(0),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "rgb_dir": str(Path(args.rgb).resolve()),
        "frames": args.frames,
        "resolution_level": args.resolution_level,
        "input_hw_after_padding": [H, W],
        "patch_stride": stride if stride != 1 or stride_candidates else None,
        "patch_stride_candidates": stride_candidates,
        "patch_stride_note": stride_note,
        "halo_pixels": args.halo,
        "core_box_y0_y1_x0_x1": list(core),
        "tile_no_halo_box": list(tile_no_halo),
        "tile_halo_box": list(tile_halo),
        "full_area_pixels": int(full_area),
        "core_area_pixels": int(core_hw[0] * core_hw[1]),
        "tile_no_halo_area_pixels": int(nohalo_hw[0] * nohalo_hw[1]),
        "tile_halo_area_pixels": int(halo_hw[0] * halo_hw[1]),
        "core_area_ratio_to_full": float(core_hw[0] * core_hw[1] / full_area),
        "tile_no_halo_area_ratio_to_full": float(nohalo_hw[0] * nohalo_hw[1] / full_area),
        "tile_halo_area_ratio_to_full": float(halo_hw[0] * halo_hw[1] / full_area),
        "full_encoder_timing": full_time,
        "tile_no_halo_timing": no_halo_time,
        "tile_halo_timing": halo_time,
        "no_halo_error": no_halo_error,
        "halo_error": halo_error,
        "feature_shapes": tile_shapes,
        "stream_trace": trace,
        "interpretation_boundary": (
            "Encoder-only local crop feasibility. Does not implement or prove local pixel_decoder refresh, "
            "end-to-end speedup, output quality, or RK3588 deployment."
        ),
    }
    (out_dir / "p2_3_v5_25_metadata.json").write_text(
        json.dumps(benchmark, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_rows = []
    for variant, timing, shape, box in [
        ("full", full_time, full_shapes, (0, H, 0, W)),
        ("tile_no_halo", no_halo_time, tile_shapes["tile_no_halo"], tile_no_halo),
        ("tile_halo", halo_time, tile_shapes["tile_halo"], tile_halo),
    ]:
        y0, y1, x0, x1 = box
        summary_rows.append({
            "variant": variant,
            "input_h": y1 - y0,
            "input_w": x1 - x0,
            "input_area_pixels": int((y1 - y0) * (x1 - x0)),
            "input_area_ratio_to_full": float((y1 - y0) * (x1 - x0) / full_area),
            "mean_ms": timing.get("mean_ms"),
            "median_ms": timing.get("median_ms"),
            "min_ms": timing.get("min_ms"),
            "status": timing.get("status", "ok"),
            "error": timing.get("error", ""),
            "tensor_leaf_count": len(shape),
        })
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "p2_3_v5_25_encoder_crop_benchmark.csv", index=False, encoding="utf-8-sig"
    )

    report_lines = [
        "# P2.3 v5.25a — 最小 Tile/Halo 可行性预检",
        "",
        "## 实验定位",
        "",
        "- 本轮只测试 pixel_encoder 的整帧与局部 crop 调用，并 profile 原始流式执行路径。",
        "- 不实现局部 pixel_decoder，不改变状态、不训练权重，不宣称端到端省算。",
        "- 中心 core 约为整帧面积的 25%；Halo 默认 32 像素。",
        "",
        "## 运行环境",
        "",
        f"- GPU：{benchmark['gpu_name']}",
        f"- 输入尺寸（padding 后）：{H} × {W}",
        f"- 编码对齐步长：{benchmark['patch_stride']}; {stride_note}",
        f"- 帧数：{args.frames}",
        "",
        "## 输入面积与编码时间",
        "",
        "| Variant | H×W | 输入面积比例 | Mean ms | Median ms | 状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in summary_rows:
        report_lines.append(
            f"| {r['variant']} | {r['input_h']}×{r['input_w']} | {r['input_area_ratio_to_full']:.3f} "
            f"| {r['mean_ms'] if r['mean_ms'] is not None else 'NA'} "
            f"| {r['median_ms'] if r['median_ms'] is not None else 'NA'} "
            f"| {r['status']} |"
        )
    report_lines += [
        "",
        "## 原始流式组件 profile",
        "",
        "| Component | Calls | Total ms | Mean ms/call |",
        "|---|---:|---:|---:|",
    ]
    for _, r in component_summary.iterrows():
        report_lines.append(
            f"| {r['component']} | {int(r['calls'])} | {r['total_ms']:.3f} | {r['mean_ms_per_call']:.3f} |"
        )
    report_lines += [
        "",
        "## 特征对应区域比较",
        "",
        "详细指标见 `p2_3_v5_25_feature_equivalence.csv`。只有 `comparable=True` 的行才具有对应的空间网格比较。",
        "即使可比较，差异也可能来自全局注意力上下文、位置编码、padding 或动态形状行为；Halo 不保证恢复整帧特征。",
        "",
        "## 初步 Go/No-Go 规则",
        "",
        "- **接口 No-Go：** crop 调用报错，或输出结构/空间网格无法对应。先查明 patch/position embedding 与动态形状限制，不进入局部解码。",
        "- **语义 No-Go：** 带 Halo 的核心区域特征仍与整帧特征严重不一致。不能把局部编码输出直接作为原模型等价刷新。",
        "- **成本 No-Go：** crop 编码延迟没有下降，或后续 Halo/拼接/解码成本预计会抵消节省。暂不开发完整 Tile/Halo。",
        "- **条件 Go：** crop 可运行、核心特征差异可解释、编码时间明显下降。下一步才实现局部 decoder/融合原型，并单独测端到端质量和耗时。",
        "",
        "## 关键限制",
        "",
        "本报告是短序列预检。它不等于完整 Tile/Halo 系统，也不证明局部刷新后的深度误差可接受；需要下一轮局部解码与边界融合实验。",
        "",
    ]
    (out_dir / "P2_3_v5_25_实验报告.md").write_text("\n".join(report_lines), encoding="utf-8")

    bundle = out_dir / "p2_3_v5_25_tile_halo_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            if p.is_file() and p.name != bundle.name:
                z.write(p, arcname=p.name)

    print("\nPASS: v5.25a preflight finished")
    print(f"Output directory: {out_dir}")
    print(f"Report: {out_dir / 'P2_3_v5_25_实验报告.md'}")
    print(f"Review ZIP: {bundle}")


if __name__ == "__main__":
    main()
