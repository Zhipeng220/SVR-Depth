#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.3 v5.25b — Multi-frame Halo sensitivity and stable stream latency audit.

This experiment:
  1) captures exact packed inputs from VeloDepth.infer(),
  2) compares full-frame encoding against crop variants with configured Halo 0/32/64,
  3) measures per-frame/per-layer spatial feature differences,
  4) profiles repeated streaming runs with warmup separated from measured runs.

It does NOT implement local decoder refresh, alter weights, or claim end-to-end
speedup. Feature and timing measurements are diagnostic only.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
import zipfile
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

VERSION = "P2.3 v5.25b Halo sensitivity and stable latency audit"


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


def load_rgb(folder: str, count: int, start_index: int = 0):
    all_paths = sorted(Path(folder).glob("*.png"))
    paths = all_paths[start_index:start_index + count]
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
    """Return a B,Hf,Wf,C grid, detecting BHWC vs BCHW using the expected patch grid."""
    H, W = input_hw
    if t.ndim == 4:
        if patch_stride:
            gh = int(round(H / patch_stride))
            gw = int(round(W / patch_stride))
            # VeloDepth pixel_encoder emits per-layer tokens as BHWC:
            # [B, grid_h, grid_w, channels], e.g. [1, 31, 41, 1024].
            if tuple(t.shape[1:3]) == (gh, gw):
                return t, "BHWC"
            # Support conventional BCHW feature maps when spatial dimensions match.
            if tuple(t.shape[2:4]) == (gh, gw):
                return t.permute(0, 2, 3, 1), "BCHW"
        # Conservative fallback: identify a large channel dimension at the end.
        if t.shape[-1] >= max(t.shape[1], t.shape[2]):
            return t, "BHWC_heuristic"
        if t.shape[1] >= max(t.shape[2], t.shape[3]):
            return t.permute(0, 2, 3, 1), "BCHW_heuristic"
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



def profile_stream_repeated(model, packed_inputs, metas, device, warmup_runs: int, measured_runs: int):
    """Profile full stream repeatedly; retain warmup and measured runs separately."""
    event_records = []
    run_records = []
    current = {"run_index": -1, "phase": "setup"}
    handles = []
    starts = {}

    def event_start():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def event_end():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def make_hooks(name):
        def pre(module, inputs):
            stack = starts.setdefault(id(module), [])
            stack.append(event_start())

        def post(module, inputs, output):
            stack = starts.setdefault(id(module), [])
            if not stack:
                return
            start_event = stack.pop()
            end_event = event_end()
            first = inputs[0] if inputs else None
            event_records.append({
                "run_index": current["run_index"],
                "phase": current["phase"],
                "component": name,
                "input_shape": str(list(first.shape)) if isinstance(first, torch.Tensor) else "",
                "output_shapes": json.dumps(tensor_shapes(output), ensure_ascii=False),
                "_start_event": start_event,
                "_end_event": end_event,
            })
        return pre, post

    for name in ["pixel_encoder", "small_encoder"]:
        module = getattr(model, name, None)
        if isinstance(module, nn.Module):
            pre, post = make_hooks(name)
            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(post))

    decoder = model.pixel_decoder
    originals = {}
    for method_name in ["forward_keyframe", "forward_fast"]:
        original = getattr(decoder, method_name)
        originals[method_name] = original

        def make_wrapper(method, name):
            def wrapped(*args, **kwargs):
                start_event = event_start()
                result = method(*args, **kwargs)
                end_event = event_end()
                event_records.append({
                    "run_index": current["run_index"],
                    "phase": current["phase"],
                    "component": f"pixel_decoder.{name}",
                    "input_shape": "",
                    "output_shapes": json.dumps(tensor_shapes(result), ensure_ascii=False),
                    "_start_event": start_event,
                    "_end_event": end_event,
                })
                return result
            return wrapped
        setattr(decoder, method_name, make_wrapper(original, method_name))

    try:
        total_runs = warmup_runs + measured_runs
        for run_idx in range(total_runs):
            phase = "warmup" if run_idx < warmup_runs else "measured"
            current["run_index"] = run_idx
            current["phase"] = phase
            sync(device)
            t0 = time.perf_counter()
            stream = VeloDepthStream(model)
            with torch.no_grad():
                _ = stream.run(clone_payload(packed_inputs), clone_payload(metas))
            sync(device)
            elapsed = (time.perf_counter() - t0) * 1000.0
            trace = stream.trace
            run_records.append({
                "run_index": run_idx,
                "phase": phase,
                "stream_total_ms": elapsed,
                "frame_count": int(metas[0]["T"]),
                "keyframe_count": int(sum(x.get("action") == "keyframe" for x in trace)),
                "propagation_count": int(sum(x.get("action") == "propagation" for x in trace)),
                "rejected_propagation_count": int(sum(x.get("action") == "propagation_rejected" for x in trace)),
                "trace_json": json.dumps(trace, ensure_ascii=False),
            })
    finally:
        for h in handles:
            h.remove()
        for name, original in originals.items():
            setattr(decoder, name, original)

    # Synchronization after each run ensures all event pairs are complete.
    detail_rows = []
    call_counter = {}
    for rec in event_records:
        start_event = rec.pop("_start_event")
        end_event = rec.pop("_end_event")
        try:
            elapsed_ms = float(start_event.elapsed_time(end_event))
        except Exception:
            elapsed_ms = float("nan")
        component = rec["component"]
        key = (rec["run_index"], component)
        call_counter[key] = call_counter.get(key, 0) + 1
        detail_rows.append({
            **rec,
            "call_index_within_component_run": call_counter[key],
            "elapsed_ms": elapsed_ms,
        })
    detail = pd.DataFrame(detail_rows)
    runs = pd.DataFrame(run_records)
    if detail.empty:
        summary = pd.DataFrame(columns=[
            "phase", "component", "call_count", "mean_ms", "median_ms", "p90_ms", "min_ms", "max_ms"
        ])
    else:
        measured = detail[detail["phase"] == "measured"].copy()
        summary = (
            measured.groupby("component", as_index=False)
            .agg(
                call_count=("elapsed_ms", "count"),
                mean_ms=("elapsed_ms", "mean"),
                median_ms=("elapsed_ms", "median"),
                p90_ms=("elapsed_ms", lambda x: float(np.nanpercentile(x, 90))),
                min_ms=("elapsed_ms", "min"),
                max_ms=("elapsed_ms", "max"),
            )
        )
        summary.insert(0, "phase", "measured_only")
    return detail, summary, runs


def summarize_halo_results(benchmark_df: pd.DataFrame, feature_df: pd.DataFrame):
    # Use only comparable spatial feature rows. Each row is one layer on one frame.
    comp = feature_df[feature_df["comparable"] == True].copy() if not feature_df.empty else pd.DataFrame()
    if not comp.empty:
        halo_summary = (
            comp.groupby("halo_config_px", as_index=False)
            .agg(
                frame_layer_pairs=("tensor_path", "count"),
                unique_frames=("frame_index", "nunique"),
                layer_count=("tensor_path", "nunique"),
                mean_core_mae=("core_mae", "mean"),
                median_core_mae=("core_mae", "median"),
                mean_relative_l2=("core_relative_l2", "mean"),
                median_relative_l2=("core_relative_l2", "median"),
                p90_relative_l2=("core_relative_l2", lambda x: float(np.nanpercentile(x, 90))),
                mean_cosine=("core_cosine", "mean"),
                median_cosine=("core_cosine", "median"),
            )
        )
        layer_summary = (
            comp.groupby(["halo_config_px", "tensor_path"], as_index=False)
            .agg(
                frame_count=("frame_index", "nunique"),
                mean_core_mae=("core_mae", "mean"),
                mean_relative_l2=("core_relative_l2", "mean"),
                mean_cosine=("core_cosine", "mean"),
            )
        )
    else:
        halo_summary = pd.DataFrame()
        layer_summary = pd.DataFrame()

    bench_summary = (
        benchmark_df.groupby(["variant", "halo_config_px"], dropna=False, as_index=False)
        .agg(
            frame_count=("frame_index", "nunique"),
            input_area_ratio=("input_area_ratio_to_full", "mean"),
            mean_latency_ms=("mean_ms", "mean"),
            median_latency_ms=("median_ms", "median"),
            min_latency_ms=("min_ms", "min"),
            max_latency_ms=("max_ms", "max"),
        )
    )
    # Compare each crop against the same frame's full-frame timing.
    full = benchmark_df[benchmark_df["variant"] == "full"][["frame_index", "mean_ms"]].rename(
        columns={"mean_ms": "full_mean_ms"}
    )
    bench_summary_detail = benchmark_df.merge(full, on="frame_index", how="left")
    bench_summary_detail["speedup_vs_full"] = (
        bench_summary_detail["full_mean_ms"] / bench_summary_detail["mean_ms"]
    )
    speedup = (
        bench_summary_detail.groupby(["variant", "halo_config_px"], dropna=False, as_index=False)
        .agg(
            mean_speedup_vs_full=("speedup_vs_full", "mean"),
            median_speedup_vs_full=("speedup_vs_full", "median"),
        )
    )
    bench_summary = bench_summary.merge(speedup, on=["variant", "halo_config_px"], how="left")
    return halo_summary, layer_summary, bench_summary, bench_summary_detail


def main():
    parser = argparse.ArgumentParser(description="P2.3 v5.25b Halo sensitivity and stable latency audit")
    parser.add_argument("--rgb", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resolution_level", type=int, default=0)
    parser.add_argument("--halos", default="0,32,64")
    parser.add_argument("--warmup", type=int, default=2, help="warmup runs per encoder shape")
    parser.add_argument("--repeats", type=int, default=5, help="measured repeats per encoder shape")
    parser.add_argument("--stream-warmup", type=int, default=2)
    parser.add_argument("--stream-repeats", type=int, default=5)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    halos = sorted({int(x.strip()) for x in args.halos.split(",") if x.strip()})
    if args.frames < 5:
        parser.error("--frames must be >= 5 for the multi-frame sensitivity audit.")
    if args.start_index < 0:
        parser.error("--start-index must be >= 0.")
    if not halos or any(h < 0 for h in halos):
        parser.error("--halos must be non-negative comma-separated integers.")
    if args.warmup < 1 or args.repeats < 3:
        parser.error("Use --warmup >= 1 and --repeats >= 3.")
    if args.stream_warmup < 1 or args.stream_repeats < 3:
        parser.error("Use --stream-warmup >= 1 and --stream-repeats >= 3.")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this GPU profiling experiment on an AutoDL GPU instance.")

    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else ROOT
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        root / "outputs" / "p2_3_v5_25b_halo_sensitivity"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    print(f"[1/6] Loading VeloDepth from {root}")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(device)
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    print(f"[2/6] Loading {args.frames} RGB frames")
    all_paths = sorted(Path(args.rgb).glob("*.png"))
    paths = all_paths[args.start_index:args.start_index + args.frames]
    if len(paths) != args.frames:
        raise RuntimeError(
            f"Expected {args.frames} PNG frames from start-index {args.start_index}, found {len(paths)}."
        )
    rgbs = load_rgb(args.rgb, args.frames, args.start_index)
    frame_names = [p.name for p in paths]

    print("[3/6] Capturing exact packed inputs (the original inference runs once here)")
    captured = {}
    original_encode_decode_test = model.encode_decode_test

    def capture(inputs, image_metas):
        captured["inputs"] = clone_payload(inputs)
        captured["metas"] = clone_payload(image_metas)
        return original_encode_decode_test(inputs, image_metas)

    model.encode_decode_test = capture
    try:
        with torch.no_grad():
            _ = model.infer(rgbs, normalize=True)
    finally:
        model.encode_decode_test = original_encode_decode_test

    if "inputs" not in captured:
        raise RuntimeError("Failed to capture packed inputs from encode_decode_test.")
    raw_payload = clone_payload(captured["inputs"])
    metas = clone_payload(captured["metas"])
    T = int(metas[0]["T"])
    if T != args.frames:
        raise RuntimeError(f"Captured T={T}, but requested frames={args.frames}.")

    prepared = clone_payload(raw_payload)
    model._set_image_paddings(prepared)
    image_bt = model._bt(prepared["image"], 1, T)
    H, W = int(image_bt.shape[-2]), int(image_bt.shape[-1])
    stride, stride_candidates = discover_patch_stride(model.pixel_encoder)
    if stride is None:
        raise RuntimeError("Could not infer patch stride; refusing to compare misaligned crops.")
    stride_note = f"Detected patch/alignment stride {stride}."

    print("[4/6] Halo sensitivity: per-frame full vs crop encoder benchmarks and feature comparisons")
    benchmark_rows = []
    feature_rows = []
    geometry_rows = []
    for frame_idx in range(T):
        image = image_bt[:, frame_idx].contiguous()
        full_out, full_time = run_encoder_bench(
            model, image, args.warmup, args.repeats, device
        )
        full_shapes = tensor_shapes(full_out)
        full_row = {
            "frame_index": frame_idx,
            "frame_name": frame_names[frame_idx],
            "variant": "full",
            "halo_config_px": None,
            "input_h": H,
            "input_w": W,
            "input_area_pixels": H * W,
            "input_area_ratio_to_full": 1.0,
            **full_time,
            "status": "ok",
        }
        benchmark_rows.append(full_row)

        boxes = {halo: choose_core_and_tile(H, W, stride, halo) for halo in halos}
        # Rotate order across frames to reduce a fixed ordering/thermal bias.
        rotated = halos[frame_idx % len(halos):] + halos[:frame_idx % len(halos)]
        for halo in rotated:
            box = boxes[halo]["tile_halo"]
            core = boxes[halo]["core"]
            y0, y1, x0, x1 = box
            cy0, cy1, cx0, cx1 = core
            crop = image[..., y0:y1, x0:x1].contiguous()
            crop_out, crop_time = run_encoder_bench(
                model, crop, args.warmup, args.repeats, device
            )
            crop_h, crop_w = int(crop.shape[-2]), int(crop.shape[-1])
            effective = {
                "top": cy0 - y0,
                "bottom": y1 - cy1,
                "left": cx0 - x0,
                "right": x1 - cx1,
            }
            geometry_rows.append({
                "frame_index": frame_idx,
                "frame_name": frame_names[frame_idx],
                "halo_config_px": halo,
                "patch_stride": stride,
                "input_h": crop_h,
                "input_w": crop_w,
                "core_y0": cy0,
                "core_y1": cy1,
                "core_x0": cx0,
                "core_x1": cx1,
                "tile_y0": y0,
                "tile_y1": y1,
                "tile_x0": x0,
                "tile_x1": x1,
                "effective_halo_top": effective["top"],
                "effective_halo_bottom": effective["bottom"],
                "effective_halo_left": effective["left"],
                "effective_halo_right": effective["right"],
                "input_area_ratio_to_full": float(crop_h * crop_w / (H * W)),
            })
            benchmark_rows.append({
                "frame_index": frame_idx,
                "frame_name": frame_names[frame_idx],
                "variant": f"crop_halo_{halo}",
                "halo_config_px": halo,
                "input_h": crop_h,
                "input_w": crop_w,
                "input_area_pixels": crop_h * crop_w,
                "input_area_ratio_to_full": float(crop_h * crop_w / (H * W)),
                **crop_time,
                "status": "ok",
            })
            crop_core = (cy0 - y0, cy1 - y0, cx0 - x0, cx1 - x0)
            rows = feature_compare(
                full_out, crop_out, (H, W), (crop_h, crop_w),
                core, crop_core, stride
            )
            for row in rows:
                row.update({
                    "frame_index": frame_idx,
                    "frame_name": frame_names[frame_idx],
                    "halo_config_px": halo,
                })
                feature_rows.append(row)

    benchmark_df = pd.DataFrame(benchmark_rows)
    feature_df = pd.DataFrame(feature_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    benchmark_df.to_csv(out_dir / "p2_3_v5_25b_encoder_benchmark_per_frame.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(out_dir / "p2_3_v5_25b_feature_comparison_per_frame_layer.csv", index=False, encoding="utf-8-sig")
    geometry_df.to_csv(out_dir / "p2_3_v5_25b_crop_geometry.csv", index=False, encoding="utf-8-sig")

    halo_summary, layer_summary, bench_summary, bench_detail = summarize_halo_results(
        benchmark_df, feature_df
    )
    halo_summary.to_csv(out_dir / "p2_3_v5_25b_halo_summary.csv", index=False, encoding="utf-8-sig")
    layer_summary.to_csv(out_dir / "p2_3_v5_25b_halo_by_layer.csv", index=False, encoding="utf-8-sig")
    bench_summary.to_csv(out_dir / "p2_3_v5_25b_encoder_benchmark_summary.csv", index=False, encoding="utf-8-sig")
    bench_detail.to_csv(out_dir / "p2_3_v5_25b_encoder_benchmark_with_speedup.csv", index=False, encoding="utf-8-sig")

    print("[5/6] Repeated stream profile with warmup excluded from steady-state summary")
    component_detail, component_summary, stream_runs = profile_stream_repeated(
        model, raw_payload, metas, device, args.stream_warmup, args.stream_repeats
    )
    component_detail.to_csv(out_dir / "p2_3_v5_25b_component_profile_detail.csv", index=False, encoding="utf-8-sig")
    component_summary.to_csv(out_dir / "p2_3_v5_25b_component_profile_steady_state.csv", index=False, encoding="utf-8-sig")
    stream_runs.to_csv(out_dir / "p2_3_v5_25b_stream_repeat_summary.csv", index=False, encoding="utf-8-sig")

    measured_runs = stream_runs[stream_runs["phase"] == "measured"]
    stable_stream_summary = {
        "measured_runs": int(len(measured_runs)),
        "mean_total_ms": float(measured_runs["stream_total_ms"].mean()),
        "median_total_ms": float(measured_runs["stream_total_ms"].median()),
        "p90_total_ms": float(np.nanpercentile(measured_runs["stream_total_ms"], 90)),
        "min_total_ms": float(measured_runs["stream_total_ms"].min()),
        "max_total_ms": float(measured_runs["stream_total_ms"].max()),
    }

    print("[6/6] Writing metadata, report, and review ZIP")
    metadata = {
        "version": "P2.3 v5.25b Halo sensitivity and stable latency audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "rgb_dir": str(Path(args.rgb).resolve()),
        "frame_names": frame_names,
        "frame_count": args.frames,
        "start_index": args.start_index,
        "resolution_level": args.resolution_level,
        "halo_configs_requested_px": halos,
        "encoder_warmup_per_shape": args.warmup,
        "encoder_measured_repeats_per_shape": args.repeats,
        "stream_warmup_runs": args.stream_warmup,
        "stream_measured_runs": args.stream_repeats,
        "padded_input_hw": [H, W],
        "patch_stride": stride,
        "patch_stride_candidates": stride_candidates,
        "patch_stride_note": stride_note,
        "stable_stream_summary": stable_stream_summary,
        "feature_rows": int(len(feature_df)),
        "comparable_feature_rows": int((feature_df["comparable"] == True).sum()) if not feature_df.empty else 0,
        "noncomparable_feature_rows": int((feature_df["comparable"] != True).sum()) if not feature_df.empty else 0,
        "interpretation_boundary": (
            "Multi-frame encoder crop sensitivity and warmed streaming component profiling. "
            "No local decoder or blending is implemented; no end-to-end quality/speedup claim."
        ),
    }
    (out_dir / "p2_3_v5_25b_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# P2.3 v5.25b — Halo 敏感性与稳定耗时验证",
        "",
        "## 1. 实验范围",
        "",
        f"- 序列：`{Path(args.rgb).resolve()}`；帧数：{args.frames}；帧范围索引：{args.start_index}–{args.start_index + args.frames - 1}。",
        f"- 帧文件：{', '.join(frame_names)}",
        f"- Halo 配置：{', '.join(map(str, halos))} px；patch stride：{stride}。",
        f"- 编码器：每种形状预热 {args.warmup} 次，测量 {args.repeats} 次。",
        f"- 流式 profile：预热 {args.stream_warmup} 次，稳定测量 {args.stream_repeats} 次。",
        "- 不训练模型、不修改权重、不实现局部解码；稳定组件 profile 只用于诊断。",
        "",
        "## 2. 编码延迟与面积",
        "",
        "| Variant | 帧数 | 平均输入面积比例 | 平均延迟 ms | 中位延迟 ms | 相对整帧加速 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in bench_summary.sort_values(["variant", "halo_config_px"], na_position="first").iterrows():
        speedup = r.get("median_speedup_vs_full", float("nan"))
        report.append(
            f"| {r['variant']} | {int(r['frame_count'])} | {r['input_area_ratio']:.3f} "
            f"| {r['mean_latency_ms']:.3f} | {r['median_latency_ms']:.3f} "
            f"| {speedup:.2f}× |"
        )
    report += [
        "",
        "## 3. 特征一致性（空间特征层）",
        "",
        "| Halo 配置 px | 帧-层样本数 | 特征层数 | 平均 MAE | 平均相对 L2 | 中位相对 L2 | P90 相对 L2 | 平均余弦相似度 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in halo_summary.sort_values("halo_config_px").iterrows():
        report.append(
            f"| {int(r['halo_config_px'])} | {int(r['frame_layer_pairs'])} | {int(r['layer_count'])} "
            f"| {r['mean_core_mae']:.6f} | {r['mean_relative_l2']:.6f} "
            f"| {r['median_relative_l2']:.6f} | {r['p90_relative_l2']:.6f} "
            f"| {r['mean_cosine']:.6f} |"
        )
    report += [
        "",
        "## 4. 稳定状态流式耗时",
        "",
        f"- 稳定测量总运行次数：{stable_stream_summary['measured_runs']}。",
        f"- 总耗时均值：{stable_stream_summary['mean_total_ms']:.3f} ms。",
        f"- 总耗时中位数：{stable_stream_summary['median_total_ms']:.3f} ms。",
        f"- 总耗时 P90：{stable_stream_summary['p90_total_ms']:.3f} ms。",
        "",
        "| Component | 测量调用数 | 均值 ms | 中位数 ms | P90 ms | 最小 ms | 最大 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in component_summary.sort_values("component").iterrows():
        report.append(
            f"| {r['component']} | {int(r['call_count'])} | {r['mean_ms']:.3f} "
            f"| {r['median_ms']:.3f} | {r['p90_ms']:.3f} | {r['min_ms']:.3f} | {r['max_ms']:.3f} |"
        )
    report += [
        "",
        "## 5. 判读边界",
        "",
        "- Halo 配置会按 patch stride 向外对齐；实际每边 Halo 以 `p2_3_v5_25b_crop_geometry.csv` 为准，不将配置值误当成实际有效值。",
        "- 特征指标是编码器中间特征一致性，不是最终深度误差；各层等权汇总只是诊断摘要。",
        "- 编码延迟使用每帧相同输入，先预热后测量；不同 Halo 仍可能受 GPU 时钟/顺序影响，采用轮换顺序与多次重复降低偏差。",
        "- 稳定组件耗时用 CUDA events 记录 GPU 时间；完整流式耗时是同步后的 wall-clock 时间，两者口径不同，不能简单相加。",
        "- 本实验不证明局部解码可用、最终深度质量可接受或端到端加速。",
        "",
        "## 6. 下一步决策规则",
        "",
        "1. 若 Halo=64 相对 Halo=32 的误差改善很小、但编码耗时/面积明显增加，优先保留 32px 配置进入局部解码原型。",
        "2. 若 Halo=64 持续显著降低多帧多层误差且仍有明确编码收益，可选择 64px 作为后续原型配置。",
        "3. 若 Halo 增大后误差仍高或层间差异极大，先检查位置编码、全局上下文和 decoder 对 token 的依赖，不进入局部刷新实现。",
        "4. 若稳定 profile 显示传播路径占主导，则将局部关键帧编码的系统收益预期下调，并先确定可优化的真实成本项。",
        "",
    ]
    (out_dir / "P2_3_v5_25b_实验报告.md").write_text("\n".join(report), encoding="utf-8")

    bundle = out_dir / "p2_3_v5_25b_halo_sensitivity_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            if p.is_file() and p.name != bundle.name:
                z.write(p, arcname=p.name)

    print("PASS: P2.3 v5.25b Halo sensitivity and stable latency audit")
    print(f"Output directory: {out_dir}")
    print(f"Report: {out_dir / 'P2_3_v5_25b_实验报告.md'}")
    print(f"Review ZIP: {bundle}")


if __name__ == "__main__":
    main()
