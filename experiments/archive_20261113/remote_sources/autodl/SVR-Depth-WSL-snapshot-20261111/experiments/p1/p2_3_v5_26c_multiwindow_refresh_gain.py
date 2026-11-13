#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.3 v5.26c — Multi-window local refresh gain validation.

Purpose:
  * Re-evaluate Halo 32/64 local refresh gain across multiple temporal windows.
  * Measure propagation, local keyframe encoding/decoding, fusion, and boundary
    consistency at propagation ages 1, 2, 4, and 8.
  * Audit confidence output distributions across propagation, local decode,
    fused output, and full-frame keyframe reference.

Scope limits:
  * Full-frame keyframe outputs are offline references only.
  * Automatic refresh requests are logged but ignored to construct a
    counterfactual no-refresh rollout.
  * No ground-truth depth metric is computed.
  * No local teacher-feature state writeback or post-refresh rollout is tested.
"""
from __future__ import annotations

import argparse
import copy
import gc
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


VERSION = "P2.3 v5.26c multi-window refresh gain + confidence audit"
DEFAULT_HALOS = [32, 64]


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


def load_rgb(folder: str, count: int, start_index: int) -> torch.Tensor:
    paths = sorted(Path(folder).glob("*.png"))[start_index:start_index + count]
    if len(paths) != count:
        raise RuntimeError(
            f"Expected {count} PNG frames from index {start_index} in {folder}; found {len(paths)}."
        )
    frames = []
    for p in paths:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read image: {p}")
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


def discover_patch_stride(module: nn.Module) -> tuple[int | None, list[dict]]:
    candidates = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Conv2d):
            stride = child.stride
            kernel = child.kernel_size
            if (stride[0] == stride[1] and kernel[0] == kernel[1]
                    and stride[0] > 1 and kernel[0] == stride[0]):
                candidates.append({
                    "module": name, "stride": int(stride[0]), "kernel": int(kernel[0]),
                    "reason": "Conv2d kernel_size == stride",
                })
    if candidates:
        candidates.sort(key=lambda x: (x["stride"], x["kernel"]), reverse=True)
        return candidates[0]["stride"], candidates
    for obj_name, obj in [("pixel_encoder", module)] + [(n, m) for n, m in module.named_modules()]:
        for attr in ("patch_size", "patch_stride", "stride"):
            val = getattr(obj, attr, None)
            if isinstance(val, int) and val > 1:
                return int(val), [{"module": obj_name, "stride": int(val), "reason": f"attribute {attr}"}]
            if (isinstance(val, (tuple, list)) and len(val) == 2
                    and val[0] == val[1] and isinstance(val[0], int) and val[0] > 1):
                return int(val[0]), [{"module": obj_name, "stride": int(val[0]), "reason": f"attribute {attr}"}]
    return None, candidates


def align_down(x: int, stride: int) -> int:
    return max(0, (x // stride) * stride)


def align_up(x: int, stride: int, limit: int) -> int:
    return min(limit, int(math.ceil(x / stride) * stride))


def choose_core_and_tile(H: int, W: int, stride: int, halo: int):
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
        "tile": (tile_y0, tile_y1, tile_x0, tile_x1),
    }


def crop_spatial_tensor(t: torch.Tensor, full_hw: tuple[int, int],
                        box: tuple[int, int, int, int]) -> torch.Tensor:
    """Crop BCHW, BHWC, or flattened Bx(H*W)xC tensors while preserving layout."""
    H, W = full_hw
    y0, y1, x0, x1 = box
    if t.ndim >= 4 and tuple(t.shape[-2:]) == (H, W):
        return t[..., y0:y1, x0:x1].contiguous()
    if t.ndim >= 4 and tuple(t.shape[1:3]) == (H, W):
        return t[:, y0:y1, x0:x1, ...].contiguous()
    if t.ndim == 3 and t.shape[1] == H * W:
        b, _, c = t.shape
        grid = t.reshape(b, H, W, c)
        return grid[:, y0:y1, x0:x1, :].reshape(b, (y1-y0)*(x1-x0), c).contiguous()
    raise ValueError(
        f"Cannot crop tensor shape {tuple(t.shape)} as spatial {H}x{W}; "
        "expected BCHW/BHWC or Bx(H*W)xC."
    )


def crop_frame_inputs(frame_inputs: dict, full_hw: tuple[int, int],
                      box: tuple[int, int, int, int]) -> dict:
    result = {}
    for key, value in frame_inputs.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
        elif key in {"image", "image_original", "validity_mask", "rays"}:
            result[key] = crop_spatial_tensor(value, full_hw, box)
        else:
            # The keyframe adapter only passes the four spatial fields above.
            result[key] = value
    if "image" in result:
        result["image_original"] = result["image"]
    return result


def frame_inputs_at(model, payload: dict, frame_idx: int, T: int) -> dict:
    image = model._bt(payload["image"], 1, T)[:, frame_idx].contiguous()
    if "validity_mask" not in payload:
        raise KeyError("Packed input is missing validity_mask.")
    validity = model._bt(payload["validity_mask"], 1, T)[:, frame_idx].contiguous()
    result = {"image": image, "image_original": image, "validity_mask": validity}
    if "rays" in payload:
        result["rays"] = model._bt(payload["rays"], 1, T)[:, frame_idx].contiguous()
    return result


def keyframe_once(model, frame_inputs: dict, device: torch.device):
    """Execute the same encoder grouping + forward_keyframe path as VeloDepthStream."""
    inputs = clone_payload(frame_inputs)
    metas = [{"T": 1, "B": 1}]
    sync(device)
    total_start = time.perf_counter()

    enc_start = time.perf_counter()
    features, tokens = model.pixel_encoder(inputs["image"])
    sync(device)
    encoder_ms = (time.perf_counter() - enc_start) * 1000.0

    features = [
        model.stacking_fn(features[i:j]).contiguous()
        for i, j in model.slices_encoder_range
    ]
    tokens = [
        model.stacking_fn(tokens[i:j]).contiguous()
        for i, j in model.slices_encoder_range
    ]
    inputs["features"], inputs["tokens"] = features, tokens

    decoder_start = time.perf_counter()
    model.pixel_decoder.set_shapes(inputs, metas)
    outputs = model.pixel_decoder.forward_keyframe(inputs, metas)
    sync(device)
    decoder_ms = (time.perf_counter() - decoder_start) * 1000.0
    total_ms = (time.perf_counter() - total_start) * 1000.0
    shapes = tensor_shapes(outputs)
    return outputs, {
        "encoder_ms": float(encoder_ms),
        "decoder_setshape_forward_ms": float(decoder_ms),
        "keyframe_pipeline_ms": float(total_ms),
        "output_shapes": json.dumps(shapes, ensure_ascii=False),
    }


def benchmark_keyframe(model, frame_inputs: dict, device: torch.device,
                       warmup: int, repeats: int, target_idx: int, variant: str,
                       halo: int | None, input_area_ratio: float, crop_hw: tuple[int, int]):
    rows = []
    last_output = None
    last_shapes = {}
    for i in range(warmup + repeats):
        phase = "warmup" if i < warmup else "measured"
        try:
            with torch.no_grad():
                outputs, timing = keyframe_once(model, frame_inputs, device)
            last_output = outputs
            last_shapes = tensor_shapes(outputs)
            rows.append({
                "target_frame_index": target_idx,
                "variant": variant,
                "halo_config_px": halo,
                "phase": phase,
                "repeat_index": i,
                "input_h": crop_hw[0],
                "input_w": crop_hw[1],
                "input_area_ratio_to_full": input_area_ratio,
                "encoder_ms": timing["encoder_ms"],
                "decoder_setshape_forward_ms": timing["decoder_setshape_forward_ms"],
                "keyframe_pipeline_ms": timing["keyframe_pipeline_ms"],
                "status": "ok",
                "error": "",
                "output_shapes": timing["output_shapes"],
            })
        except Exception as exc:
            rows.append({
                "target_frame_index": target_idx,
                "variant": variant,
                "halo_config_px": halo,
                "phase": phase,
                "repeat_index": i,
                "input_h": crop_hw[0],
                "input_w": crop_hw[1],
                "input_area_ratio_to_full": input_area_ratio,
                "encoder_ms": np.nan,
                "decoder_setshape_forward_ms": np.nan,
                "keyframe_pipeline_ms": np.nan,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "output_shapes": "",
            })
            # If any run fails, later timing summaries must not silently pretend
            # the variant is stable. Continue to collect the error trace only.
            break
    if not any(r["phase"] == "measured" and r["status"] == "ok" for r in rows):
        last_output = None
    return last_output, rows, last_shapes


def to_hw(t: torch.Tensor | None, H: int, W: int) -> torch.Tensor | None:
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        return None
    t = t.detach().float()
    if t.ndim < 2 or tuple(t.shape[-2:]) != (H, W):
        raise ValueError(f"Expected output ending in {(H, W)}, got {tuple(t.shape)}.")
    return t.reshape(-1, H, W)[0]


def validity_hw(frame_inputs: dict, H: int, W: int, device: torch.device) -> torch.Tensor:
    v = frame_inputs.get("validity_mask")
    if isinstance(v, torch.Tensor):
        if tuple(v.shape[-2:]) == (H, W):
            return v.detach().reshape(-1, H, W)[0] > 0.5
        if v.ndim >= 4 and tuple(v.shape[1:3]) == (H, W):
            grid = v.detach().reshape(-1, H, W, v.shape[-1])[0]
            return grid[..., 0] > 0.5
    return torch.ones((H, W), dtype=torch.bool, device=device)


def box_mask(H: int, W: int, box: tuple[int, int, int, int], device) -> torch.Tensor:
    y0, y1, x0, x1 = box
    mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    mask[y0:y1, x0:x1] = True
    return mask


def boundary_mask(H: int, W: int, box: tuple[int, int, int, int], band: int, device):
    y0, y1, x0, x1 = box
    tile = box_mask(H, W, box, device)
    inner = torch.zeros_like(tile)
    if y1 - y0 > 2 * band and x1 - x0 > 2 * band:
        inner[y0+band:y1-band, x0+band:x1-band] = True
    return tile & ~inner


def feather_mask(tile_box: tuple[int, int, int, int],
                 core_box: tuple[int, int, int, int], device) -> torch.Tensor:
    """Separable feather: 1 in core, linearly fades to 0 at the outer tile edge."""
    y0, y1, x0, x1 = tile_box
    cy0, cy1, cx0, cx1 = core_box
    th, tw = y1-y0, x1-x0
    wy = torch.ones(th, dtype=torch.float32, device=device)
    wx = torch.ones(tw, dtype=torch.float32, device=device)

    top = cy0 - y0
    bottom = y1 - cy1
    left = cx0 - x0
    right = x1 - cx1
    if top > 0:
        wy[:top] = torch.arange(top, dtype=torch.float32, device=device) / float(top)
    if bottom > 0:
        vals = torch.arange(bottom-1, -1, -1, dtype=torch.float32, device=device) / float(bottom)
        wy[th-bottom:] = vals
    if left > 0:
        wx[:left] = torch.arange(left, dtype=torch.float32, device=device) / float(left)
    if right > 0:
        vals = torch.arange(right-1, -1, -1, dtype=torch.float32, device=device) / float(right)
        wx[tw-right:] = vals
    return wy[:, None] * wx[None, :]


def metric_pair(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> dict:
    valid = mask & torch.isfinite(pred) & torch.isfinite(ref)
    p = pred[valid].float()
    r = ref[valid].float()
    if p.numel() == 0:
        return {
            "n_pixels": 0, "mae": np.nan, "rmse": np.nan,
            "relative_l2": np.nan, "max_abs": np.nan,
            "cosine": np.nan, "gradient_mae": np.nan,
        }
    d = p - r
    denom = torch.linalg.vector_norm(r).item()
    pn = torch.linalg.vector_norm(p).item()
    rn = torch.linalg.vector_norm(r).item()
    cosine = float(torch.dot(p, r).item() / max(pn * rn, 1e-12))

    # First-order spatial-gradient discrepancy, evaluated over the same region.
    # Gradients are padded to the original HxW grid for consistent masking.
    pgx = torch.zeros_like(pred, dtype=torch.float32)
    rgx = torch.zeros_like(ref, dtype=torch.float32)
    pgy = torch.zeros_like(pred, dtype=torch.float32)
    rgy = torch.zeros_like(ref, dtype=torch.float32)
    pgx[:, 1:] = pred[:, 1:].float() - pred[:, :-1].float()
    rgx[:, 1:] = ref[:, 1:].float() - ref[:, :-1].float()
    pgy[1:, :] = pred[1:, :].float() - pred[:-1, :].float()
    rgy[1:, :] = ref[1:, :].float() - ref[:-1, :].float()
    grad_valid = valid & torch.isfinite(pgx) & torch.isfinite(rgx) & torch.isfinite(pgy) & torch.isfinite(rgy)
    if grad_valid.any():
        grad_mae = float((
            (pgx[grad_valid] - rgx[grad_valid]).abs()
            + (pgy[grad_valid] - rgy[grad_valid]).abs()
        ).mean().item() / 2.0)
    else:
        grad_mae = np.nan

    return {
        "n_pixels": int(p.numel()),
        "mae": float(d.abs().mean().item()),
        "rmse": float(torch.sqrt((d*d).mean()).item()),
        "relative_l2": float(torch.linalg.vector_norm(d).item() / max(denom, 1e-12)),
        "max_abs": float(d.abs().max().item()),
        "cosine": cosine,
        "gradient_mae": grad_mae,
    }


def core_edge_mask(H: int, W: int, core_box: tuple[int, int, int, int],
                   band: int, device) -> torch.Tensor:
    """Ring spanning both sides of the core boundary, clipped to the full frame."""
    y0, y1, x0, x1 = core_box
    expanded = (
        max(0, y0 - band), min(H, y1 + band),
        max(0, x0 - band), min(W, x1 + band),
    )
    outer = box_mask(H, W, expanded, device)
    inner = torch.zeros((H, W), dtype=torch.bool, device=device)
    iy0, iy1 = min(H, y0 + band), max(0, y1 - band)
    ix0, ix1 = min(W, x0 + band), max(0, x1 - band)
    if iy1 > iy0 and ix1 > ix0:
        inner[iy0:iy1, ix0:ix1] = True
    return outer & ~inner


def output_triplet(outputs, H: int, W: int):
    if outputs is None or len(outputs) < 3:
        raise RuntimeError("forward_keyframe did not return (rays_base, radius, confidence, ...).")
    radius = to_hw(outputs[1], H, W)
    confidence = to_hw(outputs[2], H, W)
    return radius, confidence


def summarize_timings(timing_df: pd.DataFrame) -> pd.DataFrame:
    if timing_df.empty:
        return pd.DataFrame()
    measured = timing_df[(timing_df["phase"] == "measured") & (timing_df["status"] == "ok")].copy()
    if measured.empty:
        return pd.DataFrame()
    return measured.groupby(["variant", "halo_config_px"], dropna=False, as_index=False).agg(
        target_frames=("target_frame_index", "nunique"),
        measured_runs=("keyframe_pipeline_ms", "count"),
        mean_encoder_ms=("encoder_ms", "mean"),
        median_encoder_ms=("encoder_ms", "median"),
        mean_decoder_ms=("decoder_setshape_forward_ms", "mean"),
        median_decoder_ms=("decoder_setshape_forward_ms", "median"),
        mean_keyframe_pipeline_ms=("keyframe_pipeline_ms", "mean"),
        median_keyframe_pipeline_ms=("keyframe_pipeline_ms", "median"),
        p90_keyframe_pipeline_ms=("keyframe_pipeline_ms", lambda x: float(np.nanpercentile(x, 90))),
        mean_area_ratio=("input_area_ratio_to_full", "mean"),
    )




def resolve_window_starts(total_frames: int, window_length: int,
                          max_windows: int, starts_arg: str) -> list[int]:
    """Resolve explicit starts or select up to max_windows evenly spread windows."""
    max_start = total_frames - window_length
    if max_start < 0:
        raise ValueError(
            f"Dataset has {total_frames} PNG frames, fewer than window length {window_length}."
        )
    if starts_arg.strip().lower() != "auto":
        try:
            starts = [int(x.strip()) for x in starts_arg.split(",") if x.strip()]
        except ValueError as exc:
            raise ValueError("--window-starts must be 'auto' or comma-separated integers.") from exc
        if not starts:
            raise ValueError("--window-starts resolved to an empty list.")
        if len(set(starts)) != len(starts):
            raise ValueError(f"Duplicate explicit window starts are not allowed: {starts}")
        if any(x < 0 or x > max_start for x in starts):
            raise ValueError(
                f"Every window start must be between 0 and {max_start}; received {starts}."
            )
        return sorted(starts)

    if max_windows < 1:
        raise ValueError("--max-windows must be >= 1.")
    if max_windows == 1:
        return [0]
    if total_frames >= window_length * max_windows:
        starts = np.linspace(0, max_start, max_windows).round().astype(int).tolist()
    elif total_frames >= window_length * 2:
        starts = [0, max_start]
    elif total_frames > window_length:
        n = min(max_windows, 3)
        starts = np.linspace(0, max_start, n).round().astype(int).tolist()
    else:
        starts = [0]
    return sorted(set(starts))


def window_overlap_map(starts: list[int], length: int) -> dict[int, list[str]]:
    result = {}
    for i, start in enumerate(starts):
        overlaps = []
        end = start + length - 1
        for j, other in enumerate(starts):
            if i == j:
                continue
            other_end = other + length - 1
            n_overlap = max(0, min(end, other_end) - max(start, other) + 1)
            if n_overlap > 0:
                overlaps.append(f"W{j+1:02d}:{n_overlap}frames")
        result[start] = overlaps
    return result


def capture_window_payload(model, rgb_tensor: torch.Tensor) -> tuple[dict, list]:
    """Capture the exact packed tensors produced by model.infer for one window."""
    captured = {}
    original = model.encode_decode_test

    def capture(inputs, image_metas):
        captured["inputs"] = clone_payload(inputs)
        captured["metas"] = clone_payload(image_metas)
        return original(inputs, image_metas)

    model.encode_decode_test = capture
    try:
        with torch.no_grad():
            inference_output = model.infer(rgb_tensor, normalize=True)
        del inference_output
    finally:
        model.encode_decode_test = original

    if "inputs" not in captured or "metas" not in captured:
        raise RuntimeError("Failed to capture packed inputs from model.infer().")
    return clone_payload(captured["inputs"]), clone_payload(captured["metas"])


def confidence_audit_row(window_id: str, window_start_index: int, target_age: int,
                         frame_name: str, halo: int, source_name: str, region_name: str,
                         tensor: torch.Tensor | None, mask: torch.Tensor,
                         reference: torch.Tensor | None) -> dict | None:
    """Summarize confidence distribution and its agreement with full-frame reference."""
    if tensor is None:
        return None
    selected = mask & torch.isfinite(tensor)
    total = int(mask.sum().item())
    values = tensor[selected].float()
    row = {
        "window_id": window_id,
        "window_start_index": int(window_start_index),
        "target_age": int(target_age),
        "frame_name": frame_name,
        "halo_config_px": int(halo),
        "source": source_name,
        "region": region_name,
        "n_pixels": int(values.numel()),
        "finite_ratio": float(values.numel() / max(total, 1)),
        "mean": np.nan,
        "std": np.nan,
        "min": np.nan,
        "p01": np.nan,
        "p10": np.nan,
        "p50": np.nan,
        "p90": np.nan,
        "p99": np.nan,
        "max": np.nan,
        "mae_to_reference": np.nan,
        "rmse_to_reference": np.nan,
        "relative_l2_to_reference": np.nan,
        "pearson_to_reference": np.nan,
    }
    if values.numel() == 0:
        return row
    q = torch.quantile(
        values,
        torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], device=values.device, dtype=values.dtype),
    )
    row.update({
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "p01": float(q[0].item()),
        "p10": float(q[1].item()),
        "p50": float(q[2].item()),
        "p90": float(q[3].item()),
        "p99": float(q[4].item()),
        "max": float(values.max().item()),
    })
    if reference is not None:
        common = mask & torch.isfinite(tensor) & torch.isfinite(reference)
        x = tensor[common].float()
        y = reference[common].float()
        if x.numel() > 0:
            d = x - y
            row["mae_to_reference"] = float(d.abs().mean().item())
            row["rmse_to_reference"] = float(torch.sqrt((d * d).mean()).item())
            row["relative_l2_to_reference"] = float(
                torch.linalg.vector_norm(d).item()
                / max(torch.linalg.vector_norm(y).item(), 1e-12)
            )
            xc = x - x.mean()
            yc = y - y.mean()
            denom = torch.linalg.vector_norm(xc) * torch.linalg.vector_norm(yc)
            if float(denom.item()) > 1e-12:
                row["pearson_to_reference"] = float((torch.dot(xc, yc) / denom).item())
    return row


def annotate_timing_rows(rows: list[dict], window_id: str, start_index: int,
                         target_age: int, frame_name: str) -> list[dict]:
    for row in rows:
        row["window_id"] = window_id
        row["window_start_index"] = int(start_index)
        row["target_age"] = int(target_age)
        row["target_frame_name"] = frame_name
    return rows


def build_latency_summaries(timing_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if timing_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    measured = timing_df[
        (timing_df["phase"] == "measured") & (timing_df["status"] == "ok")
    ].copy()
    if measured.empty:
        return pd.DataFrame(), pd.DataFrame()

    def aggregate(keys):
        return measured.groupby(keys, dropna=False, as_index=False).agg(
            windows=("window_id", "nunique"),
            target_observations=("target_age", "nunique"),
            measured_runs=("keyframe_pipeline_ms", "count"),
            mean_encoder_ms=("encoder_ms", "mean"),
            median_encoder_ms=("encoder_ms", "median"),
            mean_decoder_ms=("decoder_setshape_forward_ms", "mean"),
            median_decoder_ms=("decoder_setshape_forward_ms", "median"),
            mean_keyframe_pipeline_ms=("keyframe_pipeline_ms", "mean"),
            median_keyframe_pipeline_ms=("keyframe_pipeline_ms", "median"),
            p90_keyframe_pipeline_ms=("keyframe_pipeline_ms", lambda x: float(np.nanpercentile(x, 90))),
            mean_area_ratio=("input_area_ratio_to_full", "mean"),
        )
    by_window = aggregate(["window_id", "window_start_index", "variant", "halo_config_px"])
    overall = aggregate(["variant", "halo_config_px"])
    return by_window, overall
def main():
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--rgb", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb")
    parser.add_argument("--frames", type=int, default=9, help="Frames per window; must cover max target age.")
    parser.add_argument("--window-starts", default="auto",
                        help="Use 'auto' or comma-separated zero-based frame indices.")
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--target-ages", default="1,2,4,8")
    parser.add_argument("--resolution_level", type=int, default=0)
    parser.add_argument("--halos", default="32,64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--stream-warmup-rollouts", type=int, default=1)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--boundary-band", type=int, default=4)
    args = parser.parse_args()

    halos = sorted({int(x.strip()) for x in args.halos.split(",") if x.strip()})
    ages = sorted({int(x.strip()) for x in args.target_ages.split(",") if x.strip()})
    if args.frames < 2:
        parser.error("--frames must be >= 2.")
    if not ages or any(a < 1 for a in ages):
        parser.error("--target-ages must contain positive comma-separated integers.")
    if max(ages) >= args.frames:
        parser.error(f"max target age ({max(ages)}) must be < --frames ({args.frames}).")
    if not halos or any(h < 0 for h in halos):
        parser.error("--halos must be non-negative comma-separated integers.")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("Use --warmup >= 1 and --repeats >= 2.")
    if args.stream_warmup_rollouts < 1:
        parser.error("--stream-warmup-rollouts must be >= 1.")
    if args.boundary_band < 1:
        parser.error("--boundary-band must be >= 1.")
    if args.max_windows < 1:
        parser.error("--max-windows must be >= 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this GPU experiment on an AutoDL GPU instance.")

    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else ROOT
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        root / "outputs" / "p2_3_v5_26c_multiwindow_refresh_gain"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    print(f"[1/8] Loading VeloDepth; root={root}")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(device)
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    print("[2/8] Resolving multi-window sample plan")
    all_paths = sorted(Path(args.rgb).glob("*.png"))
    if not all_paths:
        raise RuntimeError(f"No PNG frames found in {args.rgb}")
    starts = resolve_window_starts(len(all_paths), args.frames, args.max_windows, args.window_starts)
    if len(starts) < 2:
        print(
            "[WARNING] Only one distinct window is available. This run can smoke-test the pipeline, "
            "but cannot establish multi-window reproducibility. Use a longer sequence or explicit starts."
        )
    if args.window_starts.strip().lower() == "auto" and len(starts) < args.max_windows:
        print(
            f"[INFO] Dataset supports {len(starts)} selected window(s) under the auto non-overlap policy "
            f"(or short-sequence fallback); requested max_windows={args.max_windows}."
        )
    overlap_map = window_overlap_map(starts, args.frames)
    window_manifest_rows = []
    for i, start in enumerate(starts):
        end = start + args.frames - 1
        window_manifest_rows.append({
            "window_id": f"W{i+1:02d}",
            "window_start_index": start,
            "window_end_index": end,
            "window_length": args.frames,
            "first_frame_name": all_paths[start].name,
            "last_frame_name": all_paths[end].name,
            "overlap_with_other_windows": ";".join(overlap_map[start]) if overlap_map[start] else "none",
            "overlap_frame_count_total": int(sum(
                max(0, min(end, other + args.frames - 1) - max(start, other) + 1)
                for other in starts if other != start
            )),
        })
    manifest_df = pd.DataFrame(window_manifest_rows)
    manifest_df.to_csv(out_dir / "p2_3_v5_26c_window_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"  PNG frames available: {len(all_paths)}")
    print(f"  Selected window starts: {starts}")
    print(f"  Window length: {args.frames}; target ages: {ages}; Halo: {halos}")

    timing_rows = []
    metric_rows = []
    gain_context_rows = []
    confidence_rows = []
    propagation_rows = []
    rollout_summary_rows = []
    geometry_rows = []
    shape_rows = []
    map_arrays = {}
    metadata_windows = []
    # Different VeloDepth releases expose different Decoder attributes.
    # Only preserve/change this optional flag when it already exists.
    camera_flag_exists = hasattr(model.pixel_decoder, "test_gt_camera")
    original_test_gt_camera = getattr(model.pixel_decoder, "test_gt_camera", None)
    if not camera_flag_exists:
        print(
            "[INFO] Decoder has no 'test_gt_camera' attribute; "
            "leaving decoder flags unchanged for this VeloDepth version."
        )

    try:
        for wi, start_index in enumerate(starts):
            window_id = f"W{wi+1:02d}"
            end_index = start_index + args.frames - 1
            window_paths = all_paths[start_index:end_index + 1]
            frame_names = [p.name for p in window_paths]
            print(f"\n[Window {window_id}] indices {start_index}..{end_index}: "
                  f"{frame_names[0]} -> {frame_names[-1]}")

            print(f"[{window_id}] Capturing exact packed inputs from model.infer()")
            rgb_tensor = load_rgb(args.rgb, args.frames, start_index)
            raw_payload, metas = capture_window_payload(model, rgb_tensor)
            del rgb_tensor
            T = int(metas[0]["T"])
            if T != args.frames:
                raise RuntimeError(f"{window_id}: captured T={T}; requested {args.frames}.")

            prepared = clone_payload(raw_payload)
            model._set_image_paddings(prepared)
            if "flow_bwd" not in prepared:
                raise KeyError(f"{window_id}: packed input is missing flow_bwd.")
            prepared["flow_bwd"], prepared["grid_sample"] = model._normalize_and_grid_from_flow(
                prepared["flow_bwd"], prepared["image"],
                prepared["image_paddings"], prepared["depth_paddings"]
            )
            image_bt = model._bt(prepared["image"], 1, T)
            H, W = int(image_bt.shape[-2]), int(image_bt.shape[-1])
            stride, stride_candidates = discover_patch_stride(model.pixel_encoder)
            if stride is None:
                raise RuntimeError("Could not infer pixel patch stride; refusing misaligned crop decode.")
            geometries = {halo: choose_core_and_tile(H, W, stride, halo) for halo in halos}
            core = geometries[halos[0]]["core"]
            if any(geo["core"] != core for geo in geometries.values()):
                raise RuntimeError("Core geometry unexpectedly changed across Halo settings.")

            had_rays = "rays" in prepared
            if camera_flag_exists:
                model.pixel_decoder.test_gt_camera = bool(had_rays)

            def run_no_refresh_rollout(collect: bool):
                """One full keyframe at local age 0, then a forced no-refresh rollout."""
                stream = VeloDepthStream(model)
                state = StreamState()
                sync(device)
                keyframe_start = time.perf_counter()
                _, state = stream.keyframe_step(prepared, 0, state, T)
                sync(device)
                keyframe_ms = (time.perf_counter() - keyframe_start) * 1000.0
                local_rows = []
                local_snapshots = {}
                for t in range(1, T):
                    sync(device)
                    t0 = time.perf_counter()
                    result, state, restart_requested = stream.propagate_step(prepared, t, state, T)
                    sync(device)
                    step_ms = (time.perf_counter() - t0) * 1000.0

                    # Counterfactual no-refresh policy: log the original trigger,
                    # but continue the current propagated state instead of refreshing.
                    if restart_requested:
                        state.distance_kf += 1
                    state.first_keyframe = True
                    state.current_t = t + 1

                    local_rows.append({
                        "window_id": window_id,
                        "window_start_index": int(start_index),
                        "frame_index_in_window": int(t),
                        "global_frame_index": int(start_index + t),
                        "frame_name": frame_names[t],
                        "propagation_age": int(t),
                        "propagate_step_wall_ms": float(step_ms),
                        "original_heuristic_requested_refresh": bool(restart_requested),
                        "forced_no_refresh": True,
                        "state_distance_kf_after_step": int(state.distance_kf),
                    })
                    if collect and t in ages:
                        local_snapshots[t] = {
                            "radius": result.radius.detach().clone(),
                            "confidence": result.confidence.detach().clone() if result.confidence is not None else None,
                        }
                return float(keyframe_ms), local_rows, local_snapshots

            print(f"[{window_id}] Warming up propagation rollout")
            for warm_idx in range(args.stream_warmup_rollouts):
                _, _, _ = run_no_refresh_rollout(collect=False)
                print(f"  warmup rollout {warm_idx+1}/{args.stream_warmup_rollouts} complete")

            print(f"[{window_id}] Measuring no-refresh rollout")
            rollout_keyframe_ms, this_prop_rows, prop_snapshots = run_no_refresh_rollout(collect=True)
            propagation_rows.extend(this_prop_rows)
            if set(prop_snapshots) != set(ages):
                raise RuntimeError(
                    f"{window_id}: missing target-age snapshots {sorted(prop_snapshots)}; expected {ages}."
                )
            prop_df_window = pd.DataFrame(this_prop_rows)
            trigger_ages = prop_df_window.loc[
                prop_df_window["original_heuristic_requested_refresh"], "propagation_age"
            ].astype(int).tolist()
            rollout_summary_rows.append({
                "window_id": window_id,
                "window_start_index": int(start_index),
                "window_end_index": int(end_index),
                "initial_keyframe_wall_ms": float(rollout_keyframe_ms),
                "mean_propagation_step_ms": float(prop_df_window["propagate_step_wall_ms"].mean()),
                "median_propagation_step_ms": float(prop_df_window["propagate_step_wall_ms"].median()),
                "total_propagation_wall_ms": float(prop_df_window["propagate_step_wall_ms"].sum()),
                "original_heuristic_trigger_count": len(trigger_ages),
                "original_heuristic_trigger_ages": ",".join(map(str, trigger_ages)) if trigger_ages else "none",
                "forced_no_refresh": True,
            })

            print(f"[{window_id}] Evaluating target ages and local refresh variants")
            for age in ages:
                frame_name = frame_names[age]
                current_inputs = frame_inputs_at(model, prepared, age, T)
                valid_mask = validity_hw(current_inputs, H, W, device)
                prop_radius = to_hw(prop_snapshots[age]["radius"], H, W)
                prop_conf = to_hw(prop_snapshots[age]["confidence"], H, W)
                core_mask = box_mask(H, W, core, device) & valid_mask
                full_mask = torch.ones((H, W), dtype=torch.bool, device=device) & valid_mask

                full_out, full_rows, full_shapes = benchmark_keyframe(
                    model, current_inputs, device, args.warmup, args.repeats,
                    age, "full_reference", None, 1.0, (H, W)
                )
                timing_rows.extend(annotate_timing_rows(
                    full_rows, window_id, start_index, age, frame_name
                ))
                if full_out is None:
                    raise RuntimeError(
                        f"{window_id} age {age}: full-frame reference failed; inspect latency CSV."
                    )
                ref_radius, ref_conf = output_triplet(full_out, H, W)
                shape_rows.append({
                    "window_id": window_id, "window_start_index": int(start_index),
                    "target_age": int(age), "frame_name": frame_name,
                    "variant": "full_reference", "halo_config_px": None,
                    "input_hw": [H, W], "output_shapes": full_shapes,
                })

                map_prefix = f"{window_id.lower()}_age{age}"
                map_arrays[f"{map_prefix}_prop_radius"] = prop_radius.detach().cpu().numpy()
                map_arrays[f"{map_prefix}_reference_radius"] = ref_radius.detach().cpu().numpy()

                for halo in halos:
                    geo = geometries[halo]
                    core_box = geo["core"]
                    tile_box = geo["tile"]
                    ty0, ty1, tx0, tx1 = tile_box
                    tile_h, tile_w = ty1-ty0, tx1-tx0
                    tile_area_ratio = float(tile_h * tile_w / (H * W))
                    crop_inputs = crop_frame_inputs(current_inputs, (H, W), tile_box)
                    crop_out, crop_rows, crop_shapes = benchmark_keyframe(
                        model, crop_inputs, device, args.warmup, args.repeats,
                        age, f"crop_halo_{halo}", halo, tile_area_ratio, (tile_h, tile_w)
                    )
                    timing_rows.extend(annotate_timing_rows(
                        crop_rows, window_id, start_index, age, frame_name
                    ))
                    effective_halo = {
                        "top": core_box[0] - ty0,
                        "bottom": ty1 - core_box[1],
                        "left": core_box[2] - tx0,
                        "right": tx1 - core_box[3],
                    }
                    geometry_rows.append({
                        "window_id": window_id, "window_start_index": int(start_index),
                        "target_age": int(age), "frame_name": frame_name,
                        "halo_config_px": int(halo), "patch_stride": int(stride),
                        "full_h": H, "full_w": W,
                        "core_y0": core_box[0], "core_y1": core_box[1],
                        "core_x0": core_box[2], "core_x1": core_box[3],
                        "tile_y0": ty0, "tile_y1": ty1,
                        "tile_x0": tx0, "tile_x1": tx1,
                        "tile_h": tile_h, "tile_w": tile_w,
                        "effective_halo_top": effective_halo["top"],
                        "effective_halo_bottom": effective_halo["bottom"],
                        "effective_halo_left": effective_halo["left"],
                        "effective_halo_right": effective_halo["right"],
                        "tile_area_ratio_to_full": tile_area_ratio,
                        "status": "ok" if crop_out is not None else "decoder_error",
                    })
                    shape_rows.append({
                        "window_id": window_id, "window_start_index": int(start_index),
                        "target_age": int(age), "frame_name": frame_name,
                        "variant": f"crop_halo_{halo}", "halo_config_px": int(halo),
                        "input_hw": [tile_h, tile_w], "output_shapes": crop_shapes,
                    })
                    if crop_out is None:
                        continue

                    local_radius, local_conf = output_triplet(crop_out, tile_h, tile_w)
                    local_full = torch.zeros((H, W), dtype=local_radius.dtype, device=device)
                    local_full[ty0:ty1, tx0:tx1] = local_radius
                    local_conf_full = None
                    if local_conf is not None:
                        local_conf_full = torch.zeros((H, W), dtype=local_conf.dtype, device=device)
                        local_conf_full[ty0:ty1, tx0:tx1] = local_conf

                    weight = feather_mask(tile_box, core_box, device)
                    fused_radius = prop_radius.clone()
                    fused_radius[ty0:ty1, tx0:tx1] = (
                        (1.0 - weight) * prop_radius[ty0:ty1, tx0:tx1]
                        + weight * local_radius
                    )
                    fused_conf = None
                    if prop_conf is not None and local_conf is not None:
                        fused_conf = prop_conf.clone()
                        fused_conf[ty0:ty1, tx0:tx1] = (
                            (1.0 - weight) * prop_conf[ty0:ty1, tx0:tx1]
                            + weight * local_conf
                        )

                    tile_mask = box_mask(H, W, tile_box, device) & valid_mask
                    halo_band_mask = tile_mask & ~box_mask(H, W, core_box, device)
                    outer_boundary = boundary_mask(H, W, tile_box, args.boundary_band, device) & valid_mask
                    core_edge = core_edge_mask(H, W, core_box, args.boundary_band, device) & tile_mask & valid_mask
                    regions = [
                        ("core", core_mask),
                        ("tile", tile_mask),
                        ("halo_band", halo_band_mask),
                        ("core_edge_band", core_edge),
                        ("tile_outer_boundary", outer_boundary),
                        ("full", full_mask),
                    ]

                    radius_comparisons = [
                        ("prop_vs_full", prop_radius, ref_radius),
                        ("local_vs_full", local_full, ref_radius),
                        ("fused_vs_full", fused_radius, ref_radius),
                        ("fused_vs_prop", fused_radius, prop_radius),
                    ]
                    for comparison, pred, ref in radius_comparisons:
                        for region_name, region_mask in regions:
                            if comparison == "local_vs_full" and region_name == "full":
                                continue
                            effective_mask = (region_mask & tile_mask) if comparison == "local_vs_full" else region_mask
                            m = metric_pair(pred, ref, effective_mask)
                            metric_rows.append({
                                "window_id": window_id,
                                "window_start_index": int(start_index),
                                "target_age": int(age),
                                "frame_name": frame_name,
                                "halo_config_px": int(halo),
                                "quantity": "radius",
                                "comparison": comparison,
                                "region": region_name,
                                **m,
                            })

                    if ref_conf is not None:
                        audit_sources = [
                            ("propagation", prop_conf, ref_conf),
                            ("full_reference", ref_conf, None),
                            ("local_crop", local_conf_full, ref_conf),
                            ("fused", fused_conf, ref_conf),
                        ]
                        for source_name, conf_tensor, compare_ref in audit_sources:
                            if conf_tensor is None:
                                continue
                            for region_name, region_mask in regions:
                                if source_name == "local_crop":
                                    if region_name == "full":
                                        continue
                                    effective_mask = region_mask & tile_mask
                                else:
                                    effective_mask = region_mask
                                row = confidence_audit_row(
                                    window_id, start_index, age, frame_name, halo,
                                    source_name, region_name, conf_tensor,
                                    effective_mask, compare_ref
                                )
                                if row is not None:
                                    confidence_rows.append(row)

                    map_arrays[f"{map_prefix}_local_radius_halo{halo}"] = local_full.detach().cpu().numpy()
                    map_arrays[f"{map_prefix}_fused_radius_halo{halo}"] = fused_radius.detach().cpu().numpy()
                    map_arrays[f"{map_prefix}_blend_weight_halo{halo}"] = torch.nn.functional.pad(
                        weight, (tx0, W-tx1, ty0, H-ty1), value=0.0
                    ).detach().cpu().numpy()

                    del crop_out, local_radius, local_full, weight, fused_radius
                    if local_conf_full is not None:
                        del local_conf_full
                    if fused_conf is not None:
                        del fused_conf

                del full_out, ref_radius, current_inputs
                if ref_conf is not None:
                    del ref_conf

            metadata_windows.append({
                "window_id": window_id,
                "window_start_index": int(start_index),
                "window_end_index": int(end_index),
                "frame_names": frame_names,
                "padded_input_hw": [H, W],
                "patch_stride": int(stride),
                "patch_stride_candidates": stride_candidates,
                "core_box": list(core),
                "has_rays_input": bool(had_rays),
            })

            del prepared, raw_payload, metas, prop_snapshots, this_prop_rows, prop_df_window
            gc.collect()
            torch.cuda.empty_cache()

    finally:
        if camera_flag_exists:
            model.pixel_decoder.test_gt_camera = original_test_gt_camera

    print("\n[8/8] Aggregating multi-window results and writing review bundle")
    timing_df = pd.DataFrame(timing_rows)
    metrics_df = pd.DataFrame(metric_rows)
    propagation_df = pd.DataFrame(propagation_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    confidence_df = pd.DataFrame(confidence_rows)
    rollout_df = pd.DataFrame(rollout_summary_rows)

    timing_df.to_csv(out_dir / "p2_3_v5_26c_keyframe_latency_per_window.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "p2_3_v5_26c_local_refresh_metrics.csv", index=False, encoding="utf-8-sig")
    propagation_df.to_csv(out_dir / "p2_3_v5_26c_propagation_age_trace.csv", index=False, encoding="utf-8-sig")
    geometry_df.to_csv(out_dir / "p2_3_v5_26c_crop_geometry.csv", index=False, encoding="utf-8-sig")
    confidence_df.to_csv(out_dir / "p2_3_v5_26c_confidence_semantic_audit.csv", index=False, encoding="utf-8-sig")
    rollout_df.to_csv(out_dir / "p2_3_v5_26c_rollout_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "p2_3_v5_26c_decoder_shapes.json").write_text(
        json.dumps(shape_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    latency_by_window, latency_overall = build_latency_summaries(timing_df)
    latency_by_window.to_csv(out_dir / "p2_3_v5_26c_keyframe_latency_summary_per_window.csv", index=False, encoding="utf-8-sig")
    latency_overall.to_csv(out_dir / "p2_3_v5_26c_keyframe_latency_summary_overall.csv", index=False, encoding="utf-8-sig")

    # Match propagation and fused comparisons for each window, age, Halo and region.
    if not metrics_df.empty:
        radius_df = metrics_df[metrics_df["quantity"] == "radius"].copy()
        join_keys = [
            "window_id", "window_start_index", "target_age",
            "frame_name", "halo_config_px", "region",
        ]
        prop_df = radius_df[radius_df["comparison"] == "prop_vs_full"].copy()
        fused_df = radius_df[radius_df["comparison"] == "fused_vs_full"].copy()
        prop_df = prop_df[join_keys + ["relative_l2", "mae", "gradient_mae"]].rename(columns={
            "relative_l2": "prop_relative_l2",
            "mae": "prop_mae",
            "gradient_mae": "prop_gradient_mae",
        })
        fused_df = fused_df[join_keys + ["relative_l2", "mae", "gradient_mae"]].rename(columns={
            "relative_l2": "fused_relative_l2",
            "mae": "fused_mae",
            "gradient_mae": "fused_gradient_mae",
        })
        gain_df = prop_df.merge(fused_df, on=join_keys, how="inner", validate="one_to_one")
        gain_df["refresh_gain_relative_l2"] = gain_df["prop_relative_l2"] - gain_df["fused_relative_l2"]
        gain_df["refresh_gain_mae"] = gain_df["prop_mae"] - gain_df["fused_mae"]
        gain_df["refresh_gain_gradient_mae"] = gain_df["prop_gradient_mae"] - gain_df["fused_gradient_mae"]
        gain_df["positive_refresh_gain"] = gain_df["refresh_gain_relative_l2"] > 0
        trigger_lookup = propagation_df[[
            "window_id", "propagation_age", "original_heuristic_requested_refresh"
        ]].rename(columns={
            "propagation_age": "target_age",
            "original_heuristic_requested_refresh": "original_heuristic_requested_refresh",
        })
        gain_df = gain_df.merge(
            trigger_lookup, on=["window_id", "target_age"], how="left", validate="many_to_one"
        )
        gain_df.to_csv(out_dir / "p2_3_v5_26c_refresh_gain_per_window.csv", index=False, encoding="utf-8-sig")

        summary_keys = ["target_age", "halo_config_px", "region"]
        multi_summary = gain_df.groupby(summary_keys, as_index=False).agg(
            n_window_observations=("window_id", "nunique"),
            mean_refresh_gain=("refresh_gain_relative_l2", "mean"),
            median_refresh_gain=("refresh_gain_relative_l2", "median"),
            std_refresh_gain=("refresh_gain_relative_l2", "std"),
            min_refresh_gain=("refresh_gain_relative_l2", "min"),
            max_refresh_gain=("refresh_gain_relative_l2", "max"),
            positive_windows=("positive_refresh_gain", "sum"),
            positive_rate=("positive_refresh_gain", "mean"),
            mean_prop_relative_l2=("prop_relative_l2", "mean"),
            mean_fused_relative_l2=("fused_relative_l2", "mean"),
            mean_refresh_gain_mae=("refresh_gain_mae", "mean"),
            mean_refresh_gain_gradient_mae=("refresh_gain_gradient_mae", "mean"),
        )
        multi_summary.to_csv(out_dir / "p2_3_v5_26c_multiwindow_summary.csv", index=False, encoding="utf-8-sig")

        overall_summary = gain_df.groupby(["halo_config_px", "region"], as_index=False).agg(
            n_window_age_observations=("window_id", "count"),
            unique_windows=("window_id", "nunique"),
            mean_refresh_gain=("refresh_gain_relative_l2", "mean"),
            median_refresh_gain=("refresh_gain_relative_l2", "median"),
            std_refresh_gain=("refresh_gain_relative_l2", "std"),
            min_refresh_gain=("refresh_gain_relative_l2", "min"),
            max_refresh_gain=("refresh_gain_relative_l2", "max"),
            positive_observations=("positive_refresh_gain", "sum"),
            positive_rate=("positive_refresh_gain", "mean"),
            mean_prop_relative_l2=("prop_relative_l2", "mean"),
            mean_fused_relative_l2=("fused_relative_l2", "mean"),
            mean_refresh_gain_mae=("refresh_gain_mae", "mean"),
            mean_refresh_gain_gradient_mae=("refresh_gain_gradient_mae", "mean"),
        )
        overall_summary.to_csv(out_dir / "p2_3_v5_26c_overall_summary.csv", index=False, encoding="utf-8-sig")
    else:
        gain_df = pd.DataFrame()
        multi_summary = pd.DataFrame()
        overall_summary = pd.DataFrame()
        gain_df.to_csv(out_dir / "p2_3_v5_26c_refresh_gain_per_window.csv", index=False, encoding="utf-8-sig")
        multi_summary.to_csv(out_dir / "p2_3_v5_26c_multiwindow_summary.csv", index=False, encoding="utf-8-sig")
        overall_summary.to_csv(out_dir / "p2_3_v5_26c_overall_summary.csv", index=False, encoding="utf-8-sig")

    if not confidence_df.empty:
        confidence_summary = confidence_df.groupby(
            ["source", "halo_config_px", "region"], dropna=False, as_index=False
        ).agg(
            n_window_age_observations=("window_id", "count"),
            unique_windows=("window_id", "nunique"),
            mean_confidence=("mean", "mean"),
            mean_within_map_std=("std", "mean"),
            mean_p01=("p01", "mean"),
            mean_p50=("p50", "mean"),
            mean_p99=("p99", "mean"),
            mean_mae_to_reference=("mae_to_reference", "mean"),
            mean_relative_l2_to_reference=("relative_l2_to_reference", "mean"),
            mean_pearson_to_reference=("pearson_to_reference", "mean"),
        )
    else:
        confidence_summary = pd.DataFrame()
    confidence_summary.to_csv(out_dir / "p2_3_v5_26c_confidence_audit_summary.csv", index=False, encoding="utf-8-sig")

    trigger_summary = propagation_df.groupby("window_id", as_index=False).agg(
        window_start_index=("window_start_index", "first"),
        propagation_steps=("propagation_age", "count"),
        trigger_count=("original_heuristic_requested_refresh", "sum"),
    )
    trigger_ages_by_window = {}
    for wid, group in propagation_df.groupby("window_id"):
        trigger_ages = group.loc[
            group["original_heuristic_requested_refresh"], "propagation_age"
        ].astype(int).tolist()
        trigger_ages_by_window[wid] = ",".join(map(str, trigger_ages)) if trigger_ages else "none"
    trigger_summary["trigger_ages"] = trigger_summary["window_id"].map(trigger_ages_by_window)
    trigger_summary.to_csv(out_dir / "p2_3_v5_26c_refresh_trigger_summary.csv", index=False, encoding="utf-8-sig")

    map_path = out_dir / "p2_3_v5_26c_prediction_maps.npz"
    if map_arrays:
        np.savez_compressed(map_path, **map_arrays)

    metadata = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "rgb_dir": str(Path(args.rgb).resolve()),
        "available_png_frames": len(all_paths),
        "window_length": args.frames,
        "window_starts": starts,
        "window_count": len(starts),
        "max_windows_requested": args.max_windows,
        "target_ages": ages,
        "halo_configs_px": halos,
        "warmup_per_keyframe_variant": args.warmup,
        "measured_repeats_per_keyframe_variant": args.repeats,
        "stream_warmup_rollouts_per_window": args.stream_warmup_rollouts,
        "boundary_band_px": args.boundary_band,
        "windows": metadata_windows,
        "overlap_policy": "Auto mode uses non-overlapping windows when dataset length permits; short-sequence fallback may overlap and is explicitly recorded in manifest.",
        "forced_no_refresh": True,
        "automatic_restart_requests_logged_but_ignored": True,
        "ground_truth_depth_used": False,
        "local_feature_state_writeback": False,
        "interpretation_boundary": (
            "Multi-window counterfactual open-loop rollout. Radius/confidence are compared to same-frame "
            "full-frame keyframe predictions, not ground-truth depth. Automatic refresh requests are logged "
            "but deliberately ignored for the no-refresh trajectory. No local teacher-feature state writeback "
            "or post-refresh propagation rollout is tested."
        ),
    }
    (out_dir / "p2_3_v5_26c_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# P2.3 v5.26c — 多窗口刷新收益验证",
        "",
        "## 1. 实验设置",
        "",
        f"- GPU：{metadata['gpu_name']}；数据目录：`{args.rgb}`；可用 PNG 帧数：{len(all_paths)}。",
        f"- 窗口长度：{args.frames} 帧；实际窗口数：{len(starts)}；目标年龄：{', '.join(map(str, ages))}。",
        f"- Halo：{', '.join(map(str, halos))} px；每种关键帧输入预热 {args.warmup} 次、测量 {args.repeats} 次。",
        "- 每个窗口从本窗口第 0 帧执行一次完整关键帧，然后构造强制不刷新的连续传播轨迹。",
        "- 自动刷新启发式请求被记录但在反事实轨迹中忽略；整帧关键帧仅作为离线参照。",
        "- 本轮不使用 GT 深度、不写回局部 `features_teacher`、不测试刷新后的后续传播。",
        "",
        "## 2. 窗口清单与独立性",
        "",
        "| Window | 起始索引 | 结束索引 | 首帧 | 末帧 | 与其他窗口重叠 |",
        "|---|---:|---:|---|---|---|",
    ]
    for _, r in manifest_df.iterrows():
        report.append(
            f"| {r['window_id']} | {int(r['window_start_index'])} | {int(r['window_end_index'])} "
            f"| {r['first_frame_name']} | {r['last_frame_name']} | {r['overlap_with_other_windows']} |"
        )
    if len(starts) < 3:
        report += [
            "",
            f"**样本量提醒：** 本次只得到 {len(starts)} 个窗口，不能视为充分的三窗口复现。"
            "如果结论对研究方向有决定性影响，应增加独立片段或序列。",
        ]
    report += [
        "",
        "窗口不重叠只代表没有重复使用同一帧，不等于统计独立；同一视频序列中的不同时间片段仍可能相关。",
        "",
        "## 3. 原始刷新启发式触发情况",
        "",
        "| Window | 传播步数 | 原始策略请求刷新次数 | 请求年龄 |",
        "|---|---:|---:|---|",
    ]
    for _, r in trigger_summary.iterrows():
        report.append(
            f"| {r['window_id']} | {int(r['propagation_steps'])} | {int(r['trigger_count'])} | {r['trigger_ages']} |"
        )
    report += [
        "",
        "注意：触发请求在本轮被忽略，以形成 no-refresh 反事实轨迹。若所有窗口均无请求，说明当前采样没有覆盖原触发器会激活的状态。",
        "",
        "## 4. 各传播年龄的刷新收益",
        "",
        "正收益表示融合后更接近同帧整帧关键帧参照；该指标不是 GT 深度精度。",
        "",
        "| Age | Halo | Region | 窗口数 | 平均收益 | 中位数 | 正收益率 | 最小值 | 最大值 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not multi_summary.empty:
        chosen = multi_summary[multi_summary["region"].isin(["core", "full", "core_edge_band"])]
        for _, r in chosen.sort_values(["target_age", "halo_config_px", "region"]).iterrows():
            report.append(
                f"| {int(r['target_age'])} | {int(r['halo_config_px'])} | {r['region']} "
                f"| {int(r['n_window_observations'])} | {r['mean_refresh_gain']:.6f} "
                f"| {r['median_refresh_gain']:.6f} | {r['positive_rate']:.3f} "
                f"| {r['min_refresh_gain']:.6f} | {r['max_refresh_gain']:.6f} |"
            )
    else:
        report.append("| No metric rows | — | — | — | — | — | — | — | — |")
    report += [
        "",
        "## 5. 跨年龄整体收益",
        "",
        "| Halo | Region | 窗口×年龄观测数 | 平均收益 | 中位数 | 正收益率 | 平均 Prop rel L2 | 平均 Fused rel L2 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not overall_summary.empty:
        chosen = overall_summary[overall_summary["region"].isin(["core", "full", "core_edge_band"])]
        for _, r in chosen.sort_values(["halo_config_px", "region"]).iterrows():
            report.append(
                f"| {int(r['halo_config_px'])} | {r['region']} "
                f"| {int(r['n_window_age_observations'])} | {r['mean_refresh_gain']:.6f} "
                f"| {r['median_refresh_gain']:.6f} | {r['positive_rate']:.3f} "
                f"| {r['mean_prop_relative_l2']:.6f} | {r['mean_fused_relative_l2']:.6f} |"
            )
    else:
        report.append("| No summary rows | — | — | — | — | — | — | — |")
    report += [
        "",
        "## 6. Confidence 输出分布审计",
        "",
        "此表用于检查传播、局部解码、融合与整帧参照的 confidence 数值范围是否可比；不假定它们天然具有相同语义。",
        "",
        "| Source | Halo | Region | 观测数 | 平均值 | 平均图内标准差 | 平均 P01 | 平均 P50 | 平均 P99 | 平均 MAE to ref | 平均 Pearson |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if not confidence_summary.empty:
        selected_conf = confidence_summary[
            confidence_summary["region"].isin(["core", "full", "tile"])
            & confidence_summary["source"].isin(["propagation", "full_reference", "local_crop", "fused"])
        ]
        for _, r in selected_conf.sort_values(["source", "halo_config_px", "region"]).iterrows():
            report.append(
                f"| {r['source']} | {int(r['halo_config_px'])} | {r['region']} "
                f"| {int(r['n_window_age_observations'])} | {r['mean_confidence']:.6f} "
                f"| {r['mean_within_map_std']:.6f} | {r['mean_p01']:.6f} "
                f"| {r['mean_p50']:.6f} | {r['mean_p99']:.6f} "
                f"| {r['mean_mae_to_reference']:.6f} | {r['mean_pearson_to_reference']:.6f} |"
            )
    else:
        report.append("| Confidence output unavailable | — | — | — | — | — | — | — | — | — | — |")
    report += [
        "",
        "## 7. 关键帧处理耗时",
        "",
        "| Variant | 窗口数 | 测量次数 | 平均 Encoder ms | 平均 Decoder ms | 平均 Keyframe pipeline ms | 平均面积占比 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not latency_overall.empty:
        for _, r in latency_overall.sort_values(["variant", "halo_config_px"], na_position="first").iterrows():
            report.append(
                f"| {r['variant']} | {int(r['windows'])} | {int(r['measured_runs'])} "
                f"| {r['mean_encoder_ms']:.3f} | {r['mean_decoder_ms']:.3f} "
                f"| {r['mean_keyframe_pipeline_ms']:.3f} | {r['mean_area_ratio']:.3f} |"
            )
    else:
        report.append("| No successful timing rows | — | — | — | — | — | — |")
    report += [
        "",
        "## 8. Go/No-Go 判据",
        "",
        "1. **重复性 Go：** 核心区在多个窗口/年龄获得正收益，且正收益率和收益中位数不依赖单个窗口。",
        "2. **空间质量 No-Go：** 核心区收益伴随核心边界带系统性负收益，需先改融合权重或 tile/Halo。",
        "3. **Confidence 暂停条件：** 若传播与整帧参照的分布/相关性明显不一致，不把 confidence 直接用于在线刷新触发器。",
        "4. **采样 No-Go：** 若只获得一个窗口，或窗口大量重叠，不将其视为多窗口复现证据。",
        "5. **触发器检查：** 若所有窗口均没有原有启发式请求刷新，应额外选取更具挑战性的片段；不能据此判断触发机制无效。",
        "6. 即使多窗口正收益成立，本轮仍未验证局部 `features_teacher` 写回和刷新后的连续传播，不等于在线闭环已完成。",
        "",
    ]
    report_path = out_dir / "P2_3_v5_26c_实验报告.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    bundle = out_dir / "p2_3_v5_26c_multiwindow_refresh_gain_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            if p.is_file() and p.name != bundle.name:
                z.write(p, arcname=p.name)

    print("\n[DONE] P2.3 v5.26c multi-window refresh gain validation completed")
    print(f"Output directory: {out_dir}")
    print(f"Report: {report_path}")
    print(f"Review ZIP: {bundle}")


if __name__ == "__main__":
    main()
