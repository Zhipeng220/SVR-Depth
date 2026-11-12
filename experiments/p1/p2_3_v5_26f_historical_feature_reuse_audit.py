#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.3 v5.26f — Historical keyframe feature reuse audit.

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


VERSION = "P2.3 v5.26f historical keyframe feature reuse audit"
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


def encode_grouped_features(model, frame_inputs: dict, device: torch.device):
    """Run the actual VeloDepth pixel encoder and apply the model's own grouping."""
    x = clone_payload(frame_inputs)
    sync(device)
    t0 = time.perf_counter()
    raw_features, raw_tokens = model.pixel_encoder(x["image"])
    sync(device)
    encoder_ms = (time.perf_counter() - t0) * 1000.0

    features = [
        model.stacking_fn(raw_features[i:j]).contiguous()
        for i, j in model.slices_encoder_range
    ]
    tokens = [
        model.stacking_fn(raw_tokens[i:j]).contiguous()
        for i, j in model.slices_encoder_range
    ]
    return features, tokens, float(encoder_ms)


def benchmark_encoder_cache(model, frame_inputs: dict, device: torch.device,
                            warmup: int, repeats: int, variant: str):
    """Benchmark encoder-only cache creation and return the last valid feature/token cache."""
    rows = []
    last_features = None
    last_tokens = None
    for i in range(warmup + repeats):
        phase = "warmup" if i < warmup else "measured"
        try:
            features, tokens, encoder_ms = encode_grouped_features(
                model, frame_inputs, device
            )
            last_features, last_tokens = features, tokens
            rows.append({
                "variant": variant,
                "phase": phase,
                "repeat_index": i,
                "encoder_ms": float(encoder_ms),
                "decoder_ms": 0.0,
                "total_measured_path_ms": float(encoder_ms),
                "status": "ok",
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "variant": variant,
                "phase": phase,
                "repeat_index": i,
                "encoder_ms": float("nan"),
                "decoder_ms": float("nan"),
                "total_measured_path_ms": float("nan"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            break
    if not any(r["phase"] == "measured" and r["status"] == "ok" for r in rows):
        last_features, last_tokens = None, None
    return last_features, last_tokens, rows


def collection_drift_metrics(source_obj, target_obj, window_id: str,
                             start_index: int, target_age: int, frame_name: str,
                             halo: int, collection_name: str) -> list[dict]:
    """Compare cached historical features/tokens against current-frame encoded tensors."""
    source = flatten_tensor_leaves(source_obj, collection_name)
    target = flatten_tensor_leaves(target_obj, collection_name)
    rows = []
    for key in sorted(set(source) | set(target)):
        a = source.get(key)
        b = target.get(key)
        row = {
            "window_id": window_id,
            "window_start_index": int(start_index),
            "source_frame_age": 0,
            "target_age": int(target_age),
            "source_frame_name": "",
            "target_frame_name": frame_name,
            "halo_config_px": int(halo),
            "collection": collection_name,
            "tensor_key": key,
            "source_shape": json.dumps(list(a.shape)) if a is not None else "",
            "target_shape": json.dumps(list(b.shape)) if b is not None else "",
            "status": "ok",
            "relative_l2_source_vs_current": float("nan"),
            "mae_source_vs_current": float("nan"),
            "cosine_source_vs_current": float("nan"),
            "source_mean": float("nan"),
            "current_mean": float("nan"),
            "source_std": float("nan"),
            "current_std": float("nan"),
        }
        if a is None or b is None:
            row["status"] = "missing_tensor"
        elif tuple(a.shape) != tuple(b.shape):
            row["status"] = "shape_mismatch"
        else:
            af = a.detach().float().reshape(-1)
            bf = b.detach().float().reshape(-1)
            an = torch.linalg.vector_norm(af)
            bn = torch.linalg.vector_norm(bf)
            row.update({
                "relative_l2_source_vs_current": float(
                    torch.linalg.vector_norm(af - bf).item() / max(bn.item(), 1e-12)
                ),
                "mae_source_vs_current": float((af - bf).abs().mean().item()),
                "cosine_source_vs_current": float(
                    torch.dot(af, bf).item() / max(an.item() * bn.item(), 1e-12)
                ),
                "source_mean": float(af.mean().item()),
                "current_mean": float(bf.mean().item()),
                "source_std": float(af.std(unbiased=False).item()),
                "current_std": float(bf.std(unbiased=False).item()),
            })
        rows.append(row)
    return rows


def decode_with_features(model, frame_inputs: dict, features, tokens,
                         device: torch.device):
    """Decode given cached encoder features using the normal keyframe decoder path."""
    inputs = clone_payload(frame_inputs)
    inputs["features"] = clone_payload(features)
    inputs["tokens"] = clone_payload(tokens)
    metas = [{"T": 1, "B": 1}]

    sync(device)
    t0 = time.perf_counter()
    model.pixel_decoder.set_shapes(inputs, metas)
    outputs = model.pixel_decoder.forward_keyframe(inputs, metas)
    sync(device)
    decoder_ms = (time.perf_counter() - t0) * 1000.0
    return outputs, float(decoder_ms)


def infer_token_grid(token_count: int, full_hw: tuple[int, int],
                     stride: int) -> tuple[int, int, int] | None:
    """Infer (grid_h, grid_w, prefix_tokens) for flattened spatial tokens."""
    H, W = full_hw
    candidates = []
    for prefix in (0, 1):
        n = token_count - prefix
        if n < 4:
            continue
        for h in range(2, int(math.sqrt(n)) + 1):
            if n % h:
                continue
            w = n // h
            for gh, gw in ((h, w), (w, h)):
                aspect_err = abs(math.log(max((gh / gw) / (H / W), 1e-12)))
                if aspect_err > 0.08:
                    continue
                scale_h = H / gh
                scale_w = W / gw
                scale_err = abs(math.log(max(scale_h / scale_w, 1e-12)))
                stride_err = abs(math.log(max(math.sqrt(scale_h * scale_w) / stride, 1e-12)))
                candidates.append((aspect_err + scale_err + 0.05 * stride_err,
                                   gh, gw, prefix))
    if not candidates:
        return None
    _, gh, gw, prefix = min(candidates, key=lambda x: x[0])
    return int(gh), int(gw), int(prefix)


def crop_encoded_tensor(t: torch.Tensor, full_hw: tuple[int, int],
                        box: tuple[int, int, int, int], stride: int):
    """
    Crop a spatial encoder tensor using its own grid resolution.
    Supports BCHW/BHWC feature maps and BxNxC spatial tokens, optionally with
    a leading CLS token. Non-spatial/global tensors are retained unchanged.
    Returns (tensor, audit metadata).
    """
    H, W = full_hw
    y0, y1, x0, x1 = box
    info = {
        "input_shape": list(t.shape),
        "method": "unchanged_nonspatial",
        "grid_hw": None,
        "crop_grid_box": None,
        "output_shape": list(t.shape),
    }

    if t.ndim >= 4:
        # Determine whether the spatial axes are BCHW (last two) or BHWC (1:3)
        # by matching each candidate grid's aspect ratio to the input frame.
        candidates = []
        if t.shape[-2] > 1 and t.shape[-1] > 1:
            gh, gw = int(t.shape[-2]), int(t.shape[-1])
            err = abs(math.log(max((gh / gw) / (H / W), 1e-12)))
            candidates.append((err, "BCHW", gh, gw))
        if t.shape[1] > 1 and t.shape[2] > 1:
            gh, gw = int(t.shape[1]), int(t.shape[2])
            err = abs(math.log(max((gh / gw) / (H / W), 1e-12)))
            candidates.append((err, "BHWC", gh, gw))
        if candidates:
            err, layout, gh, gw = min(candidates, key=lambda x: x[0])
            if err < 0.15:
                fy0 = max(0, min(gh, int(round(y0 * gh / H))))
                fy1 = max(fy0 + 1, min(gh, int(round(y1 * gh / H))))
                fx0 = max(0, min(gw, int(round(x0 * gw / W))))
                fx1 = max(fx0 + 1, min(gw, int(round(x1 * gw / W))))
                if layout == "BCHW":
                    cropped = t[..., fy0:fy1, fx0:fx1].contiguous()
                else:
                    cropped = t[:, fy0:fy1, fx0:fx1, ...].contiguous()
                info.update({
                    "method": f"{layout}_spatial_crop",
                    "grid_hw": [gh, gw],
                    "crop_grid_box": [fy0, fy1, fx0, fx1],
                    "output_shape": list(cropped.shape),
                })
                return cropped, info

    if t.ndim == 3:
        b, n, c = t.shape
        grid = infer_token_grid(int(n), full_hw, stride)
        if grid is not None:
            gh, gw, prefix = grid
            spatial_n = n - prefix
            patches = t[:, prefix:, :].reshape(b, gh, gw, c)
            fy0 = max(0, min(gh, int(round(y0 * gh / H))))
            fy1 = max(fy0 + 1, min(gh, int(round(y1 * gh / H))))
            fx0 = max(0, min(gw, int(round(x0 * gw / W))))
            fx1 = max(fx0 + 1, min(gw, int(round(x1 * gw / W))))
            cropped_grid = patches[:, fy0:fy1, fx0:fx1, :].contiguous()
            cropped_seq = cropped_grid.reshape(b, -1, c)
            if prefix:
                cropped = torch.cat([t[:, :prefix, :], cropped_seq], dim=1).contiguous()
            else:
                cropped = cropped_seq
            info.update({
                "method": "flattened_spatial_tokens_crop",
                "grid_hw": [gh, gw],
                "prefix_tokens": prefix,
                "crop_grid_box": [fy0, fy1, fx0, fx1],
                "output_shape": list(cropped.shape),
            })
            return cropped, info

    return t, info


def crop_encoded_collection(obj, full_hw: tuple[int, int],
                            box: tuple[int, int, int, int], stride: int,
                            prefix: str = "tensor"):
    """Recursively crop feature/token collections and retain a per-tensor audit."""
    audit = []
    if isinstance(obj, torch.Tensor):
        cropped, info = crop_encoded_tensor(obj, full_hw, box, stride)
        info["tensor_key"] = prefix
        audit.append(info)
        return cropped, audit
    if isinstance(obj, list):
        out = []
        for i, value in enumerate(obj):
            cropped, rows = crop_encoded_collection(
                value, full_hw, box, stride, f"{prefix}[{i}]"
            )
            out.append(cropped)
            audit.extend(rows)
        return out, audit
    if isinstance(obj, tuple):
        out = []
        for i, value in enumerate(obj):
            cropped, rows = crop_encoded_collection(
                value, full_hw, box, stride, f"{prefix}[{i}]"
            )
            out.append(cropped)
            audit.extend(rows)
        return tuple(out), audit
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            cropped, rows = crop_encoded_collection(
                value, full_hw, box, stride, f"{prefix}.{key}"
            )
            out[key] = cropped
            audit.extend(rows)
        return out, audit
    return obj, audit


def flatten_tensor_leaves(obj, prefix="tensor"):
    result = {}
    if isinstance(obj, torch.Tensor):
        result[prefix] = obj
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            result.update(flatten_tensor_leaves(value, f"{prefix}[{i}]"))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            result.update(flatten_tensor_leaves(value, f"{prefix}.{key}"))
    return result


def collection_shape_compatibility(a_obj, b_obj):
    """Check whether two nested tensor collections have identical tensor keys and shapes."""
    a = flatten_tensor_leaves(a_obj, "tensor")
    b = flatten_tensor_leaves(b_obj, "tensor")
    a_keys, b_keys = set(a), set(b)
    problems = []
    if a_keys != b_keys:
        problems.append({
            "issue": "tensor_key_set_mismatch",
            "a_only": sorted(a_keys - b_keys),
            "b_only": sorted(b_keys - a_keys),
        })
    for key in sorted(a_keys & b_keys):
        if tuple(a[key].shape) != tuple(b[key].shape):
            problems.append({
                "issue": "tensor_shape_mismatch",
                "key": key,
                "a_shape": list(a[key].shape),
                "b_shape": list(b[key].shape),
            })
    return len(problems) == 0, problems


def feature_pair_metrics(local_obj, global_crop_obj, window_id: str,
                         start_index: int, age: int, frame_name: str,
                         halo: int, collection_name: str) -> list[dict]:
    local = flatten_tensor_leaves(local_obj, collection_name)
    global_crop = flatten_tensor_leaves(global_crop_obj, collection_name)
    rows = []
    for key in sorted(set(local) | set(global_crop)):
        a = local.get(key)
        b = global_crop.get(key)
        row = {
            "window_id": window_id,
            "window_start_index": int(start_index),
            "target_age": int(age),
            "frame_name": frame_name,
            "halo_config_px": int(halo),
            "collection": collection_name,
            "tensor_key": key,
            "local_crop_shape": json.dumps(list(a.shape)) if a is not None else "",
            "full_encode_cropped_shape": json.dumps(list(b.shape)) if b is not None else "",
            "status": "ok",
            "relative_l2": np.nan,
            "mae": np.nan,
            "cosine": np.nan,
            "local_mean": np.nan,
            "global_crop_mean": np.nan,
            "local_std": np.nan,
            "global_crop_std": np.nan,
            "std_ratio_local_over_global_crop": np.nan,
        }
        if a is None or b is None:
            row["status"] = "missing_tensor"
        elif tuple(a.shape) != tuple(b.shape):
            row["status"] = "shape_mismatch"
        else:
            af = a.detach().float().reshape(-1)
            bf = b.detach().float().reshape(-1)
            d = af - bf
            an = torch.linalg.vector_norm(af)
            bn = torch.linalg.vector_norm(bf)
            row.update({
                "relative_l2": float(torch.linalg.vector_norm(d).item() / max(bn.item(), 1e-12)),
                "mae": float(d.abs().mean().item()),
                "cosine": float(torch.dot(af, bf).item() / max(an.item() * bn.item(), 1e-12)),
                "local_mean": float(af.mean().item()),
                "global_crop_mean": float(bf.mean().item()),
                "local_std": float(af.std(unbiased=False).item()),
                "global_crop_std": float(bf.std(unbiased=False).item()),
                "std_ratio_local_over_global_crop": float(
                    af.std(unbiased=False).item() / max(bf.std(unbiased=False).item(), 1e-12)
                ),
            })
        rows.append(row)
    return rows


def benchmark_variant(model, frame_inputs: dict, device: torch.device,
                      variant: str, warmup: int, repeats: int,
                      feature_cache=None, token_cache=None):
    rows = []
    last_outputs = None
    last_features = None
    last_tokens = None
    for i in range(warmup + repeats):
        phase = "warmup" if i < warmup else "measured"
        try:
            if feature_cache is None:
                features, tokens, encoder_ms = encode_grouped_features(model, frame_inputs, device)
            else:
                features, tokens = feature_cache, token_cache
                encoder_ms = 0.0
            outputs, decoder_ms = decode_with_features(model, frame_inputs, features, tokens, device)
            last_outputs = outputs
            last_features = features
            last_tokens = tokens
            rows.append({
                "variant": variant,
                "phase": phase,
                "repeat_index": i,
                "encoder_ms": float(encoder_ms),
                "decoder_ms": float(decoder_ms),
                "total_measured_path_ms": float(encoder_ms + decoder_ms),
                "status": "ok",
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "variant": variant,
                "phase": phase,
                "repeat_index": i,
                "encoder_ms": np.nan,
                "decoder_ms": np.nan,
                "total_measured_path_ms": np.nan,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            break
    if not any(r["phase"] == "measured" and r["status"] == "ok" for r in rows):
        last_outputs = None
    return last_outputs, last_features, last_tokens, rows


def crop_region_masks(tile_hw: tuple[int, int], core_box_local: tuple[int, int, int, int],
                      boundary_band: int, valid_mask: torch.Tensor, device):
    th, tw = tile_hw
    tile_box_local = (0, th, 0, tw)
    full_mask = torch.ones((th, tw), dtype=torch.bool, device=device)
    core_mask = box_mask(th, tw, core_box_local, device)
    halo_mask = full_mask & ~core_mask
    core_edge = core_edge_mask(th, tw, core_box_local, boundary_band, device)
    outer = boundary_mask(th, tw, tile_box_local, boundary_band, device)
    return {
        "core": core_mask & valid_mask,
        "tile": full_mask & valid_mask,
        "halo_band": halo_mask & valid_mask,
        "core_edge_band": core_edge & valid_mask,
        "tile_outer_boundary": outer & valid_mask,
    }
def main():
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--rgb", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb")
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--window-starts", default="auto")
    parser.add_argument("--max-windows", type=int, default=3)
    parser.add_argument("--target-ages", default="1,2,4,8")
    parser.add_argument("--resolution_level", type=int, default=0)
    parser.add_argument("--halos", default="32,64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--boundary-band", type=int, default=4)
    args = parser.parse_args()

    ages = sorted({int(x.strip()) for x in args.target_ages.split(",") if x.strip()})
    halos = sorted({int(x.strip()) for x in args.halos.split(",") if x.strip()})
    if args.frames < 2:
        parser.error("--frames must be >= 2.")
    if not ages or any(a < 1 for a in ages) or max(ages) >= args.frames:
        parser.error("Target ages must be positive and smaller than --frames.")
    if not halos or any(h < 1 for h in halos):
        parser.error("--halos must contain positive integers, e.g. 32,64.")
    if args.warmup < 1 or args.repeats < 1:
        parser.error("--warmup >= 1 and --repeats >= 1 are required.")
    if args.max_windows < 1 or args.boundary_band < 1:
        parser.error("--max-windows and --boundary-band must be >= 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this experiment on the AutoDL GPU instance.")

    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else ROOT
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        root / "outputs" / "p2_3_v5_26f_encoder_decoder_bias_audit"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    print(f"[1/7] Loading VeloDepth; root={root}")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(device)
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    print("[2/7] Resolving sample windows")
    all_paths = sorted(Path(args.rgb).glob("*.png"))
    if not all_paths:
        raise RuntimeError(f"No PNG frames found in {args.rgb}")
    starts = resolve_window_starts(len(all_paths), args.frames, args.max_windows, args.window_starts)
    overlaps = window_overlap_map(starts, args.frames)
    manifest = []
    for wi, start in enumerate(starts):
        end = start + args.frames - 1
        manifest.append({
            "window_id": f"W{wi+1:02d}",
            "window_start_index": start,
            "window_end_index": end,
            "window_length": args.frames,
            "first_frame_name": all_paths[start].name,
            "last_frame_name": all_paths[end].name,
            "overlap_with_other_windows": ";".join(overlaps[start]) if overlaps[start] else "none",
        })
    pd.DataFrame(manifest).to_csv(
        out_dir / "p2_3_v5_26f_window_manifest.csv", index=False, encoding="utf-8-sig"
    )
    print(f"  PNG frames: {len(all_paths)}")
    print(f"  Window starts: {starts}; target ages: {ages}; halos: {halos}")

    timing_rows = []
    output_metric_rows = []
    feature_metric_rows = []
    historical_drift_rows = []
    crop_audit_rows = []
    compatibility_rows = []
    shape_rows = []
    map_arrays = {}
    metadata_windows = []

    for wi, start_index in enumerate(starts):
        window_id = f"W{wi+1:02d}"
        frame_names = [p.name for p in all_paths[start_index:start_index + args.frames]]
        print(f"\n[Window {window_id}] start={start_index}: {frame_names[0]} -> {frame_names[-1]}")

        rgb_tensor = load_rgb(args.rgb, args.frames, start_index)
        packed, metas = capture_window_payload(model, rgb_tensor)
        del rgb_tensor
        T = int(metas[0]["T"])
        if T != args.frames:
            raise RuntimeError(f"{window_id}: packed T={T}, expected {args.frames}.")
        prepared = clone_payload(packed)
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
            raise RuntimeError("Could not infer pixel encoder patch stride.")
        geometry = {halo: choose_core_and_tile(H, W, stride, halo) for halo in halos}
        core = geometry[halos[0]]["core"]
        if any(g["core"] != core for g in geometry.values()):
            raise RuntimeError("Core geometry differs between Halo settings.")

        metadata_windows.append({
            "window_id": window_id,
            "window_start_index": start_index,
            "window_end_index": start_index + args.frames - 1,
            "frame_names": frame_names,
            "input_hw": [H, W],
            "patch_stride": stride,
            "patch_stride_candidates": stride_candidates,
            "core_box": list(core),
        })

        # Build a real historical keyframe feature/token cache once per window.
        # This cache comes from frame age 0 and is reused against later target frames.
        source_age = 0
        source_frame_name = frame_names[source_age]
        source_inputs = frame_inputs_at(model, prepared, source_age, T)
        print(f"[{window_id}] Building historical keyframe cache from age=0 ({source_frame_name})")
        historical_features, historical_tokens, cache_rows = benchmark_encoder_cache(
            model, source_inputs, device, args.warmup, args.repeats,
            "historical_keyframe_cache_build"
        )
        for r in cache_rows:
            r.update({
                "window_id": window_id,
                "window_start_index": start_index,
                "target_age": source_age,
                "frame_name": source_frame_name,
                "source_frame_name": source_frame_name,
                "source_frame_age": source_age,
                "halo_config_px": None,
                "input_area_ratio": 1.0,
            })
        timing_rows.extend(cache_rows)
        if historical_features is None or historical_tokens is None:
            raise RuntimeError(
                f"{window_id}: failed to build historical keyframe feature cache; inspect timing CSV."
            )
        shape_rows.append({
            "window_id": window_id,
            "target_age": source_age,
            "frame_name": source_frame_name,
            "variant": "historical_keyframe_cache_build",
            "halo_config_px": None,
            "input_hw": [H, W],
            "feature_shapes": tensor_shapes({
                "features": historical_features,
                "tokens": historical_tokens,
            }),
            "output_shapes": {},
        })
        del source_inputs

        for age in ages:
            frame_name = frame_names[age]
            full_inputs = frame_inputs_at(model, prepared, age, T)
            full_valid = validity_hw(full_inputs, H, W, device)
            print(f"[{window_id}] age={age}, frame={frame_name}: full reference encode/decode")

            ref_out, full_features, full_tokens, rows = benchmark_variant(
                model, full_inputs, device, "full_frame_reference",
                args.warmup, args.repeats
            )
            for r in rows:
                r.update({
                    "window_id": window_id, "window_start_index": start_index,
                    "target_age": age, "frame_name": frame_name,
                    "halo_config_px": None,
                })
            timing_rows.extend(rows)
            if ref_out is None:
                raise RuntimeError(
                    f"{window_id} age {age}: full-frame reference failed; inspect timing CSV."
                )
            ref_radius, ref_conf = output_triplet(ref_out, H, W)
            map_key = f"{window_id.lower()}_age{age}"
            map_arrays[f"{map_key}_reference_radius"] = ref_radius.detach().cpu().numpy()
            if ref_conf is not None:
                map_arrays[f"{map_key}_reference_confidence"] = ref_conf.detach().cpu().numpy()
            shape_rows.append({
                "window_id": window_id, "target_age": age, "frame_name": frame_name,
                "variant": "full_frame_reference", "halo_config_px": None,
                "input_hw": [H, W], "feature_shapes": tensor_shapes({"features": full_features, "tokens": full_tokens}),
                "output_shapes": tensor_shapes(ref_out),
            })

            for halo in halos:
                core_y0, core_y1, core_x0, core_x1 = core
                ty0, ty1, tx0, tx1 = geometry[halo]["tile"]
                tile_box = geometry[halo]["tile"]
                tile_h, tile_w = ty1 - ty0, tx1 - tx0
                crop_inputs = crop_frame_inputs(full_inputs, (H, W), tile_box)
                crop_valid = validity_hw(crop_inputs, tile_h, tile_w, device)
                core_local = (
                    core_y0 - ty0, core_y1 - ty0,
                    core_x0 - tx0, core_x1 - tx0,
                )
                masks = crop_region_masks(
                    (tile_h, tile_w), core_local, args.boundary_band, crop_valid, device
                )
                ref_radius_crop = ref_radius[ty0:ty1, tx0:tx1]
                ref_conf_crop = ref_conf[ty0:ty1, tx0:tx1] if ref_conf is not None else None
                crop_ratio = float(tile_h * tile_w / (H * W))

                print(f"  Halo {halo}: direct crop encoder + local decoder")
                local_out, local_features, local_tokens, local_rows = benchmark_variant(
                    model, crop_inputs, device, f"direct_crop_encode_decode_halo{halo}",
                    args.warmup, args.repeats
                )
                for r in local_rows:
                    r.update({
                        "window_id": window_id, "window_start_index": start_index,
                        "target_age": age, "frame_name": frame_name,
                        "halo_config_px": halo, "input_area_ratio": crop_ratio,
                    })
                timing_rows.extend(local_rows)
                shape_rows.append({
                    "window_id": window_id, "target_age": age, "frame_name": frame_name,
                    "variant": "direct_crop_encode_decode", "halo_config_px": halo,
                    "input_hw": [tile_h, tile_w],
                    "feature_shapes": tensor_shapes({"features": local_features, "tokens": local_tokens}),
                    "output_shapes": tensor_shapes(local_out) if local_out is not None else {},
                })

                crop_features, audit_f = crop_encoded_collection(
                    full_features, (H, W), tile_box, stride, "features"
                )
                crop_tokens, audit_t = crop_encoded_collection(
                    full_tokens, (H, W), tile_box, stride, "tokens"
                )
                historical_crop_features, audit_hf = crop_encoded_collection(
                    historical_features, (H, W), tile_box, stride, "historical_features"
                )
                historical_crop_tokens, audit_ht = crop_encoded_collection(
                    historical_tokens, (H, W), tile_box, stride, "historical_tokens"
                )
                for a in audit_hf + audit_ht:
                    a.update({
                        "window_id": window_id, "window_start_index": start_index,
                        "target_age": age, "frame_name": frame_name,
                        "source_frame_age": 0, "source_frame_name": source_frame_name,
                        "halo_config_px": halo, "input_hw": [H, W],
                        "tile_box": list(tile_box),
                    })
                crop_audit_rows.extend(audit_hf + audit_ht)

                for a in audit_f + audit_t:
                    a.update({
                        "window_id": window_id, "window_start_index": start_index,
                        "target_age": age, "frame_name": frame_name,
                        "halo_config_px": halo, "input_hw": [H, W],
                        "tile_box": list(tile_box),
                    })
                crop_audit_rows.extend(audit_f + audit_t)

                print(f"  Halo {halo}: full-frame encoder features cropped + local decoder")
                cached_out, _, _, cached_rows = benchmark_variant(
                    model, crop_inputs, device,
                    f"full_encode_crop_features_local_decode_halo{halo}",
                    args.warmup, args.repeats, crop_features, crop_tokens
                )
                for r in cached_rows:
                    r.update({
                        "window_id": window_id, "window_start_index": start_index,
                        "target_age": age, "frame_name": frame_name,
                        "halo_config_px": halo, "input_area_ratio": crop_ratio,
                        "diagnostic_only": True,
                    })
                timing_rows.extend(cached_rows)
                shape_rows.append({
                    "window_id": window_id, "target_age": age, "frame_name": frame_name,
                    "variant": "full_encode_crop_features_local_decode", "halo_config_px": halo,
                    "input_hw": [tile_h, tile_w],
                    "feature_shapes": tensor_shapes({"features": crop_features, "tokens": crop_tokens}),
                    "output_shapes": tensor_shapes(cached_out) if cached_out is not None else {},
                })

                # 2x2 factorial crossover:
                # A = local spatial features + local global-token source (direct crop baseline)
                # B = full-frame spatial features cropped to tile + local tokens
                # C = local spatial features + full-frame tokens
                # D = full-frame spatial features cropped to tile + full-frame tokens (existing hybrid)
                feature_compatible, feature_shape_problems = collection_shape_compatibility(
                    local_features, crop_features
                ) if local_features is not None else (False, [{"issue": "local_features_unavailable"}])
                token_compatible, token_shape_problems = collection_shape_compatibility(
                    local_tokens, crop_tokens
                ) if local_tokens is not None else (False, [{"issue": "local_tokens_unavailable"}])

                compatibility_rows.extend([
                    {
                        "window_id": window_id, "window_start_index": start_index,
                        "target_age": age, "frame_name": frame_name,
                        "halo_config_px": halo, "factor": "spatial_features",
                        "compatible": bool(feature_compatible),
                        "problems_json": json.dumps(feature_shape_problems, ensure_ascii=False),
                    },
                    {
                        "window_id": window_id, "window_start_index": start_index,
                        "target_age": age, "frame_name": frame_name,
                        "halo_config_px": halo, "factor": "global_tokens",
                        "compatible": bool(token_compatible),
                        "problems_json": json.dumps(token_shape_problems, ensure_ascii=False),
                    },
                ])

                cross_global_spatial_local_token_name = "full_spatial_local_token_local_decode"
                cross_local_spatial_global_token_name = "local_spatial_full_token_local_decode"

                if feature_compatible and local_tokens is not None:
                    cross_global_spatial_local_token_out, _, _, cross_rows = benchmark_variant(
                        model, crop_inputs, device,
                        cross_global_spatial_local_token_name,
                        args.warmup, args.repeats,
                        crop_features, local_tokens
                    )
                else:
                    cross_global_spatial_local_token_out = None
                    cross_rows = [{
                        "variant": cross_global_spatial_local_token_name,
                        "phase": "measured", "repeat_index": 0,
                        "encoder_ms": np.nan, "decoder_ms": np.nan,
                        "total_measured_path_ms": np.nan,
                        "status": "skipped_shape_mismatch",
                        "error": json.dumps({
                            "feature_compatible": feature_compatible,
                            "local_tokens_available": local_tokens is not None,
                            "feature_problems": feature_shape_problems,
                        }, ensure_ascii=False),
                    }]
                for r in cross_rows:
                    r.update({
                        "window_id": window_id, "window_start_index": start_index,
                        "target_age": age, "frame_name": frame_name,
                        "halo_config_px": halo, "input_area_ratio": crop_ratio,
                        "diagnostic_only": True,
                    })
                timing_rows.extend(cross_rows)
                shape_rows.append({
                    "window_id": window_id, "target_age": age, "frame_name": frame_name,
                    "variant": cross_global_spatial_local_token_name, "halo_config_px": halo,
                    "input_hw": [tile_h, tile_w],
                    "feature_shapes": tensor_shapes({"features": crop_features, "tokens": local_tokens}),
                    "output_shapes": tensor_shapes(cross_global_spatial_local_token_out)
                    if cross_global_spatial_local_token_out is not None else {},
                })

                if local_features is not None and token_compatible:
                    cross_local_spatial_global_token_out, _, _, cross_rows = benchmark_variant(
                        model, crop_inputs, device,
                        cross_local_spatial_global_token_name,
                        args.warmup, args.repeats,
                        local_features, crop_tokens
                    )
                else:
                    cross_local_spatial_global_token_out = None
                    cross_rows = [{
                        "variant": cross_local_spatial_global_token_name,
                        "phase": "measured", "repeat_index": 0,
                        "encoder_ms": np.nan, "decoder_ms": np.nan,
                        "total_measured_path_ms": np.nan,
                        "status": "skipped_shape_mismatch",
                        "error": json.dumps({
                            "local_features_available": local_features is not None,
                            "token_compatible": token_compatible,
                            "token_problems": token_shape_problems,
                        }, ensure_ascii=False),
                    }]
                for r in cross_rows:
                    r.update({
                        "window_id": window_id, "window_start_index": start_index,
                        "target_age": age, "frame_name": frame_name,
                        "halo_config_px": halo, "input_area_ratio": crop_ratio,
                        "diagnostic_only": True,
                    })
                timing_rows.extend(cross_rows)
                shape_rows.append({
                    "window_id": window_id, "target_age": age, "frame_name": frame_name,
                    "variant": cross_local_spatial_global_token_name, "halo_config_px": halo,
                    "input_hw": [tile_h, tile_w],
                    "feature_shapes": tensor_shapes({"features": local_features, "tokens": crop_tokens}),
                    "output_shapes": tensor_shapes(cross_local_spatial_global_token_out)
                    if cross_local_spatial_global_token_out is not None else {},
                })

                if local_features is not None:
                    feature_metric_rows.extend(feature_pair_metrics(
                        local_features, crop_features, window_id, start_index, age,
                        frame_name, halo, "features"
                    ))
                if local_tokens is not None:
                    feature_metric_rows.extend(feature_pair_metrics(
                        local_tokens, crop_tokens, window_id, start_index, age,
                        frame_name, halo, "tokens"
                    ))

                drift_rows = collection_drift_metrics(
                    historical_crop_features, crop_features, window_id, start_index,
                    age, frame_name, halo, "spatial_features"
                )
                drift_rows += collection_drift_metrics(
                    historical_crop_tokens, crop_tokens, window_id, start_index,
                    age, frame_name, halo, "global_tokens"
                )
                for drift_row in drift_rows:
                    drift_row["source_frame_name"] = source_frame_name
                historical_drift_rows.extend(drift_rows)

                # Reuse frame-age-0 full-frame encoder cache while decoding the current target crop.
                print(f"  Halo {halo}: reuse historical keyframe cache (source age 0 -> target age {age})")
                history_out, _, _, history_rows = benchmark_variant(
                    model, crop_inputs, device,
                    "historical_keyframe_feature_reuse_local_decode",
                    args.warmup, args.repeats,
                    historical_crop_features, historical_crop_tokens
                )
                for r in history_rows:
                    r.update({
                        "window_id": window_id, "window_start_index": start_index,
                        "target_age": age, "frame_name": frame_name,
                        "source_frame_age": 0, "source_frame_name": source_frame_name,
                        "halo_config_px": halo, "input_area_ratio": crop_ratio,
                        "diagnostic_only": True,
                    })
                timing_rows.extend(history_rows)
                shape_rows.append({
                    "window_id": window_id, "target_age": age, "frame_name": frame_name,
                    "variant": "historical_keyframe_feature_reuse_local_decode",
                    "halo_config_px": halo, "input_hw": [tile_h, tile_w],
                    "feature_shapes": tensor_shapes({
                        "features": historical_crop_features,
                        "tokens": historical_crop_tokens,
                    }),
                    "output_shapes": tensor_shapes(history_out) if history_out is not None else {},
                })

                variant_outputs = [
                    ("direct_crop_encode_decode", local_out),
                    ("full_spatial_local_token_local_decode", cross_global_spatial_local_token_out),
                    ("local_spatial_full_token_local_decode", cross_local_spatial_global_token_out),
                    ("full_encode_crop_features_local_decode", cached_out),
                    ("historical_keyframe_feature_reuse_local_decode", history_out),
                ]
                for variant_name, out in variant_outputs:
                    if out is None:
                        output_metric_rows.append({
                            "window_id": window_id, "window_start_index": start_index,
                            "target_age": age, "frame_name": frame_name,
                            "halo_config_px": halo, "variant": variant_name,
                            "quantity": "radius", "region": "core",
                            "status": "variant_failed", "error": "No successful measured output",
                        })
                        continue
                    local_radius, local_conf = output_triplet(out, tile_h, tile_w)
                    for quantity, pred, ref in [
                        ("radius", local_radius, ref_radius_crop),
                        ("confidence", local_conf, ref_conf_crop),
                    ]:
                        if pred is None or ref is None:
                            continue
                        for region_name, region_mask in masks.items():
                            m = metric_pair(pred, ref, region_mask)
                            output_metric_rows.append({
                                "window_id": window_id,
                                "window_start_index": start_index,
                                "target_age": age,
                                "frame_name": frame_name,
                                "halo_config_px": halo,
                                "variant": variant_name,
                                "quantity": quantity,
                                "region": region_name,
                                "status": "ok",
                                **m,
                            })
                    map_arrays[f"{map_key}_halo{halo}_{variant_name}_radius"] = local_radius.detach().cpu().numpy()
                    if local_conf is not None:
                        map_arrays[f"{map_key}_halo{halo}_{variant_name}_confidence"] = local_conf.detach().cpu().numpy()

                if local_out is not None and cached_out is not None:
                    local_radius, local_conf = output_triplet(local_out, tile_h, tile_w)
                    cached_radius, cached_conf = output_triplet(cached_out, tile_h, tile_w)
                    for quantity, a, b in [
                        ("radius", local_radius, cached_radius),
                        ("confidence", local_conf, cached_conf),
                    ]:
                        if a is None or b is None:
                            continue
                        m = metric_pair(a, b, masks["tile"])
                        output_metric_rows.append({
                            "window_id": window_id, "window_start_index": start_index,
                            "target_age": age, "frame_name": frame_name,
                            "halo_config_px": halo,
                            "variant": "direct_crop_vs_full_encode_crop_features",
                            "quantity": quantity, "region": "tile",
                            "status": "ok", **m,
                        })

                del crop_features, crop_tokens, local_out, cached_out
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            del ref_out, full_features, full_tokens, full_inputs
            gc.collect()
            torch.cuda.empty_cache()

        del prepared, packed, metas
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[7/7] Writing v5.26f diagnostic tables and report")
    timing_df = pd.DataFrame(timing_rows)
    output_df = pd.DataFrame(output_metric_rows)
    feature_df = pd.DataFrame(feature_metric_rows)
    crop_audit_df = pd.DataFrame(crop_audit_rows)
    timing_df.to_csv(out_dir / "p2_3_v5_26f_timing.csv", index=False, encoding="utf-8-sig")
    output_df.to_csv(out_dir / "p2_3_v5_26f_output_bias_metrics.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(out_dir / "p2_3_v5_26f_encoder_feature_metrics.csv", index=False, encoding="utf-8-sig")
    crop_audit_df.to_csv(out_dir / "p2_3_v5_26f_feature_crop_audit.csv", index=False, encoding="utf-8-sig")
    compatibility_df = pd.DataFrame(compatibility_rows)
    compatibility_df.to_csv(
        out_dir / "p2_3_v5_26f_factor_shape_compatibility.csv",
        index=False, encoding="utf-8-sig"
    )

    # Per-case 2x2 factorial effects on radius error relative to the full-frame reference.
    factor_case_rows = []
    factor_summary = pd.DataFrame()
    if not output_df.empty:
        radius_good = output_df[
            (output_df["status"] == "ok") & (output_df["quantity"] == "radius")
            & output_df["variant"].isin([
                "direct_crop_encode_decode",
                "full_spatial_local_token_local_decode",
                "local_spatial_full_token_local_decode",
                "full_encode_crop_features_local_decode",
            ])
        ].copy()
        factor_keys = [
            "window_id", "window_start_index", "target_age",
            "frame_name", "halo_config_px", "region",
        ]
        pivot = radius_good.pivot_table(
            index=factor_keys, columns="variant", values="relative_l2", aggfunc="mean"
        ).reset_index()
        A = "direct_crop_encode_decode"
        B = "full_spatial_local_token_local_decode"
        C = "local_spatial_full_token_local_decode"
        D = "full_encode_crop_features_local_decode"
        for _, r in pivot.iterrows():
            if not all(name in pivot.columns and pd.notna(r[name]) for name in (A, B, C, D)):
                continue
            local_spatial_mean = (float(r[A]) + float(r[C])) / 2.0
            full_spatial_mean = (float(r[B]) + float(r[D])) / 2.0
            local_token_mean = (float(r[A]) + float(r[B])) / 2.0
            full_token_mean = (float(r[C]) + float(r[D])) / 2.0
            factor_case_rows.append({
                **{k: r[k] for k in factor_keys},
                "error_A_local_spatial_local_token": float(r[A]),
                "error_B_full_spatial_local_token": float(r[B]),
                "error_C_local_spatial_full_token": float(r[C]),
                "error_D_full_spatial_full_token": float(r[D]),
                "global_spatial_feature_benefit": local_spatial_mean - full_spatial_mean,
                "global_token_benefit": local_token_mean - full_token_mean,
                "interaction_difference_in_differences": (float(r[A]) - float(r[B])) - (float(r[C]) - float(r[D])),
            })
        factor_cases_df = pd.DataFrame(factor_case_rows)
        factor_cases_df.to_csv(
            out_dir / "p2_3_v5_26f_factorial_effects_per_case.csv",
            index=False, encoding="utf-8-sig"
        )
        if not factor_cases_df.empty:
            factor_summary = factor_cases_df.groupby(
                ["target_age", "halo_config_px", "region"], as_index=False
            ).agg(
                observations=("global_spatial_feature_benefit", "count"),
                mean_global_spatial_feature_benefit=("global_spatial_feature_benefit", "mean"),
                median_global_spatial_feature_benefit=("global_spatial_feature_benefit", "median"),
                spatial_benefit_positive_rate=("global_spatial_feature_benefit", lambda x: float((x > 0).mean())),
                mean_global_token_benefit=("global_token_benefit", "mean"),
                median_global_token_benefit=("global_token_benefit", "median"),
                token_benefit_positive_rate=("global_token_benefit", lambda x: float((x > 0).mean())),
                mean_interaction=("interaction_difference_in_differences", "mean"),
            )
    else:
        pd.DataFrame().to_csv(
            out_dir / "p2_3_v5_26f_factorial_effects_per_case.csv",
            index=False, encoding="utf-8-sig"
        )
    factor_summary.to_csv(
        out_dir / "p2_3_v5_26f_factorial_effect_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    (out_dir / "p2_3_v5_26f_shapes.json").write_text(
        json.dumps(shape_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if map_arrays:
        np.savez_compressed(out_dir / "p2_3_v5_26f_prediction_maps.npz", **map_arrays)

    successful = timing_df[
        (timing_df["phase"] == "measured") & (timing_df["status"] == "ok")
    ].copy()
    if not successful.empty:
        timing_summary = successful.groupby(
            ["variant", "halo_config_px"], dropna=False, as_index=False
        ).agg(
            observations=("window_id", "count"),
            mean_encoder_ms=("encoder_ms", "mean"),
            mean_decoder_ms=("decoder_ms", "mean"),
            mean_total_path_ms=("total_measured_path_ms", "mean"),
            median_total_path_ms=("total_measured_path_ms", "median"),
        )
    else:
        timing_summary = pd.DataFrame()
    timing_summary.to_csv(out_dir / "p2_3_v5_26f_timing_summary.csv", index=False, encoding="utf-8-sig")

    if not output_df.empty and "relative_l2" in output_df.columns:
        good = output_df[(output_df["status"] == "ok") & (output_df["quantity"] == "radius")].copy()
        out_summary = good.groupby(
            ["variant", "halo_config_px", "region"], dropna=False, as_index=False
        ).agg(
            observations=("relative_l2", "count"),
            mean_relative_l2=("relative_l2", "mean"),
            median_relative_l2=("relative_l2", "median"),
            std_relative_l2=("relative_l2", "std"),
            mean_mae=("mae", "mean"),
            mean_cosine=("cosine", "mean"),
            mean_gradient_mae=("gradient_mae", "mean"),
        )
    else:
        out_summary = pd.DataFrame()
    out_summary.to_csv(out_dir / "p2_3_v5_26f_output_bias_summary.csv", index=False, encoding="utf-8-sig")

    if not output_df.empty:
        historical_output_summary = output_df[
            (output_df["status"] == "ok")
            & (output_df["quantity"] == "radius")
            & (output_df["variant"] == "historical_keyframe_feature_reuse_local_decode")
        ].groupby(
            ["target_age", "halo_config_px", "region"], as_index=False
        ).agg(
            observations=("relative_l2", "count"),
            mean_relative_l2=("relative_l2", "mean"),
            median_relative_l2=("relative_l2", "median"),
        )
    else:
        historical_output_summary = pd.DataFrame()
    historical_output_summary.to_csv(
        out_dir / "p2_3_v5_26f_historical_reuse_output_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    if not feature_df.empty:
        feature_summary = feature_df.groupby(
            ["collection", "tensor_key", "status"], dropna=False, as_index=False
        ).agg(
            observations=("status", "count"),
            mean_relative_l2=("relative_l2", "mean"),
            median_relative_l2=("relative_l2", "median"),
            mean_mae=("mae", "mean"),
            mean_cosine=("cosine", "mean"),
            mean_std_ratio=("std_ratio_local_over_global_crop", "mean"),
        )
    else:
        feature_summary = pd.DataFrame()
    feature_summary.to_csv(out_dir / "p2_3_v5_26f_encoder_feature_summary.csv", index=False, encoding="utf-8-sig")

    historical_drift_df = pd.DataFrame(historical_drift_rows)
    historical_drift_df.to_csv(
        out_dir / "p2_3_v5_26f_historical_feature_drift.csv",
        index=False, encoding="utf-8-sig"
    )
    if not historical_drift_df.empty:
        historical_drift_summary = historical_drift_df[
            historical_drift_df["status"] == "ok"
        ].groupby(
            ["collection", "target_age", "halo_config_px"], as_index=False
        ).agg(
            observations=("relative_l2_source_vs_current", "count"),
            mean_relative_l2_source_vs_current=("relative_l2_source_vs_current", "mean"),
            median_relative_l2_source_vs_current=("relative_l2_source_vs_current", "median"),
            mean_cosine_source_vs_current=("cosine_source_vs_current", "mean"),
            mean_mae_source_vs_current=("mae_source_vs_current", "mean"),
        )
    else:
        historical_drift_summary = pd.DataFrame()
    historical_drift_summary.to_csv(
        out_dir / "p2_3_v5_26f_historical_feature_drift_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    # Compare the encoder-only keyframe cache build amortized over the selected target ages
    # against current-frame full encoding and local-decode paths. This is an estimate,
    # not a full stream benchmark; the denominator is the selected ages only.
    measured_timing = timing_df[
        (timing_df["phase"] == "measured") & (timing_df["status"] == "ok")
    ].copy()
    cache_build_summary = measured_timing[
        measured_timing["variant"] == "historical_keyframe_cache_build"
    ].groupby("window_id", as_index=False).agg(
        cache_encoder_ms=("encoder_ms", "mean")
    )
    historical_decode_summary = measured_timing[
        measured_timing["variant"] == "historical_keyframe_feature_reuse_local_decode"
    ].groupby(["window_id", "halo_config_px"], as_index=False).agg(
        historical_reuse_decoder_ms=("decoder_ms", "mean")
    )
    current_ref_summary = measured_timing[
        measured_timing["variant"] == "full_frame_reference"
    ].groupby("window_id", as_index=False).agg(
        current_full_frame_ms=("total_measured_path_ms", "mean"),
        current_full_encoder_ms=("encoder_ms", "mean"),
    )
    direct_summary = measured_timing[
        measured_timing["variant"].astype(str).str.startswith("direct_crop_encode_decode")
    ].groupby(["window_id", "halo_config_px"], as_index=False).agg(
        direct_crop_path_ms=("total_measured_path_ms", "mean")
    )
    reuse_cost_summary = historical_decode_summary.merge(
        cache_build_summary, on="window_id", how="left"
    ).merge(current_ref_summary, on="window_id", how="left").merge(
        direct_summary, on=["window_id", "halo_config_px"], how="left"
    )
    reuse_cost_summary["selected_target_age_count_for_amortization"] = len(ages)
    reuse_cost_summary["amortized_cache_encoder_ms_per_selected_target"] = (
        reuse_cost_summary["cache_encoder_ms"] / max(len(ages), 1)
    )
    reuse_cost_summary["historical_reuse_amortized_path_ms"] = (
        reuse_cost_summary["historical_reuse_decoder_ms"]
        + reuse_cost_summary["amortized_cache_encoder_ms_per_selected_target"]
    )
    reuse_cost_summary["saving_vs_current_full_frame_ms"] = (
        reuse_cost_summary["current_full_frame_ms"]
        - reuse_cost_summary["historical_reuse_amortized_path_ms"]
    )
    reuse_cost_summary["saving_vs_current_full_frame_pct"] = (
        100.0 * reuse_cost_summary["saving_vs_current_full_frame_ms"]
        / reuse_cost_summary["current_full_frame_ms"].clip(lower=1e-12)
    )
    reuse_cost_summary.to_csv(
        out_dir / "p2_3_v5_26f_historical_reuse_cost_summary.csv",
        index=False, encoding="utf-8-sig"
    )

    metadata = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "rgb_dir": str(Path(args.rgb).resolve()),
        "available_png_frames": len(all_paths),
        "window_starts": starts,
        "window_count": len(starts),
        "window_length": args.frames,
        "target_ages": ages,
        "halo_configs_px": halos,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "patch_stride_by_window": {x["window_id"]: x["patch_stride"] for x in metadata_windows},
        "variant_definitions": {
            "full_frame_reference": "Full image encoded and decoded; offline reference only.",
            "direct_crop_encode_decode": "A: crop image, then encode and decode; local spatial features + local global tokens.",
            "full_spatial_local_token_local_decode": "B: crop spatial features from full-frame encoding + local-crop global tokens.",
            "local_spatial_full_token_local_decode": "C: local-crop spatial features + global tokens from full-frame encoding.",
            "full_encode_crop_features_local_decode": "D: crop spatial features and retain global tokens from the current target frame's full-frame encoding; same-frame diagnostic oracle.",
            "historical_keyframe_cache_build": "Encoder-only cache build from age-0 keyframe, measured separately; the cache is reused for all selected target ages in the same window.",
            "historical_keyframe_feature_reuse_local_decode": "E: current target crop RGB decoded using spatial features and global tokens cached from the age-0 keyframe; stale-cache diagnostic, not the native VeloDepth temporal propagation path.",
            "factorial_effects": "2x2 diagnostic estimates of spatial-feature-source effect, global-token-source effect, and their interaction.",
        },
        "ground_truth_depth_used": False,
        "feature_state_writeback": False,
        "historical_cache_source_age": 0,
        "historical_cache_reuse_target_ages": ages,
        "interpretation_limit": "Output errors are relative to the same-frame full-frame model prediction, not ground-truth depth. Historical age-0 features/tokens decoded with later-frame RGB are a stale-cache counterfactual, not the native VeloDepth temporal propagation path. Amortized timing divides one encoder cache-build estimate by the selected target ages only and is not a full online stream benchmark.",
        "windows": metadata_windows,
    }
    (out_dir / "p2_3_v5_26f_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# P2.3 v5.26f — 历史关键帧特征复用审计",
        "",
        "## 1. 目的",
        "",
        "将局部裁剪输出偏差拆分为：局部输入编码造成的差异，以及在全帧编码特征已知时局部解码造成的差异。",
        "",
        "## 2. 变体定义",
        "",
        "- **full_frame_reference**：整帧编码 + 整帧解码，作为同帧模型输出参照，不是真值。",
        "- **A / direct_crop_encode_decode**：局部空间特征 + 局部编码产生的全局 Token。",
        "- **B / full_spatial_local_token_local_decode**：整帧空间特征裁剪 + 局部编码产生的全局 Token。",
        "- **C / local_spatial_full_token_local_decode**：局部空间特征 + 整帧编码产生的全局 Token。",
        "- **D / full_encode_crop_features_local_decode**：整帧空间特征裁剪 + 整帧编码产生的全局 Token。",
        "- 四组共享相同裁剪 RGB 输入、解码器和 Halo；只有空间特征来源与全局 Token 来源改变。",
        "",
        "## 3. 样本窗口",
        "",
        "| Window | 起点 | 终点 | 首帧 | 末帧 | 重叠 |",
        "|---|---:|---:|---|---|---|",
    ]
    for r in manifest:
        report.append(
            f"| {r['window_id']} | {r['window_start_index']} | {r['window_end_index']} "
            f"| {r['first_frame_name']} | {r['last_frame_name']} | {r['overlap_with_other_windows']} |"
        )
    report += [
        "",
        "## 4. 输出误差摘要",
        "",
        "以下指标衡量与同帧整帧模型输出的差异，不代表 GT 深度精度。",
        "",
        "| Variant | Halo | Region | 观测数 | Mean rel L2 | Median rel L2 | Mean cosine | Mean gradient MAE |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    if not out_summary.empty:
        for _, r in out_summary[
            out_summary["region"].isin(["core", "tile", "core_edge_band", "tile_outer_boundary"])
        ].sort_values(["variant", "halo_config_px", "region"], na_position="first").iterrows():
            halo_text = "—" if pd.isna(r["halo_config_px"]) else str(int(r["halo_config_px"]))
            report.append(
                f"| {r['variant']} | {halo_text} | {r['region']} | {int(r['observations'])} "
                f"| {r['mean_relative_l2']:.6f} | {r['median_relative_l2']:.6f} "
                f"| {r['mean_cosine']:.6f} | {r['mean_gradient_mae']:.6f} |"
            )
    else:
        report.append("| No valid output metrics | — | — | — | — | — | — | — |")
    report += [
        "",
        "## 5. 编码特征差异摘要",
        "",
        "| Collection | Tensor | Status | Observations | Mean rel L2 | Mean cosine | Mean std ratio |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    if not feature_summary.empty:
        for _, r in feature_summary.sort_values(["collection", "tensor_key", "status"]).iterrows():
            report.append(
                f"| {r['collection']} | `{r['tensor_key']}` | {r['status']} "
                f"| {int(r['observations'])} | {r['mean_relative_l2']:.6f} "
                f"| {r['mean_cosine']:.6f} | {r['mean_std_ratio']:.6f} |"
            )
    else:
        report.append("| No feature pairs | — | — | — | — | — | — |")
    report += [
        "",
        "## 6. 关键判读规则",
        "",
        "1. 若局部编码特征与整帧编码特征裁剪后差异很大，且 full-encode/crop-feature/local-decode 输出明显优于 direct-crop 输出，则局部编码上下文是主要嫌疑。",
        "2. 若两种局部输出都明显偏离整帧参照，则局部解码上下文、解码器形状/坐标假设或特征裁剪本身仍可能有问题。",
        "3. 若 feature crop audit 出现 shape mismatch、非空间 token 被保留或 decoder variant 失败，不应直接把输出差异归因于编码器；需先修正特征裁剪映射。",
        "4. 若 full-encode/crop-feature/local-decode 接近整帧参照，而 direct-crop 仍退化，下一步才考虑使用整帧编码特征上的局部解码或共享编码特征机制。",
        "5. 本轮不使用 GT 深度，不修改权重，不做局部状态写回；不能据此宣称真实深度精度提升。",
        "",
        "## 7. 计时解释",
        "",
        "direct-crop 的总耗时包含局部编码与解码；full-encode/crop-feature/local-decode 的计时只包含局部解码，整帧编码成本已在参考路径中产生，属于诊断对照，不可直接当作部署时延。",
        "",
    ]
    report += [
        "",
        "## 6. 二因素消融效应（Radius 相对 L2）",
        "",
        "定义：正的 `global_spatial_feature_benefit` 表示整帧编码空间特征（裁剪后）平均降低误差；正的 `global_token_benefit` 表示使用整帧全局 Token 平均降低误差。交互项为 `(A-B)-(C-D)`。这些都是相对整帧模型输出的误差，不是真实深度误差。",
        "",
        "| Age | Halo | Region | N | 全局空间特征收益 | 正收益率 | 全局 Token 收益 | 正收益率 | 平均交互项 |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not factor_summary.empty:
        for _, r in factor_summary[
            factor_summary["region"].isin(["core", "core_edge_band", "tile_outer_boundary", "tile"])
        ].sort_values(["target_age", "halo_config_px", "region"]).iterrows():
            report.append(
                f"| {int(r['target_age'])} | {int(r['halo_config_px'])} | {r['region']} "
                f"| {int(r['observations'])} | {r['mean_global_spatial_feature_benefit']:.6f} "
                f"| {r['spatial_benefit_positive_rate']:.3f} "
                f"| {r['mean_global_token_benefit']:.6f} "
                f"| {r['token_benefit_positive_rate']:.3f} "
                f"| {r['mean_interaction']:.6f} |"
            )
    else:
        report.append("| No complete A/B/C/D cells | — | — | — | — | — | — | — | — |")
    report += [
        "",
        "## 7. Shape compatibility gate",
        "",
        "`p2_3_v5_26f_factor_shape_compatibility.csv` 必须显示各窗口/年龄/Halo 的空间特征与 Token 张量键、形状兼容。若有不兼容或 crossover 变体失败，不应解释为因素效应；应先检查特征裁剪映射与 Decoder 输入约束。",
        "",
        "## 8. 解释边界",
        "",
        "A/B/C/D 的 factorial 对比用于分离空间特征来源和全局 Token 来源的贡献；整帧输出只是模型参照，不是真实深度真值。整帧编码特征裁剪是离线诊断控制，不代表可直接部署的计算路径。无权重修改、无 GT 深度评估、无状态写回。",
        "",
    ]

    report += [
        "",
        "## 9. 历史关键帧特征复用（核心新增实验）",
        "",
        "E 使用窗口第 0 帧的整帧编码特征与全局 Token，按同一空间 tile 裁剪后，解码传播年龄为 1/2/4/8 的当前帧 RGB。它是陈旧缓存反事实诊断，不是 VeloDepth 原生时序传播路径，也不代表可直接部署的算法。",
        "",
        "| Age | Halo | 区域 | 观测数 | 平均相对 L2 | 中位相对 L2 |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    if not historical_output_summary.empty:
        for _, r in historical_output_summary[
            historical_output_summary["region"].isin(
                ["core", "core_edge_band", "tile_outer_boundary", "tile"]
            )
        ].sort_values(["target_age", "halo_config_px", "region"]).iterrows():
            report.append(
                f"| {int(r['target_age'])} | {int(r['halo_config_px'])} | {r['region']} "
                f"| {int(r['observations'])} | {r['mean_relative_l2']:.6f} "
                f"| {r['median_relative_l2']:.6f} |"
            )
    else:
        report.append("| No valid historical-reuse output metrics | — | — | — | — | — |")

    report += [
        "",
        "### 历史特征随年龄的漂移",
        "",
        "| Collection | Age | Halo | 张量观测数 | 平均相对 L2（历史对当前） | 平均 Cosine |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not historical_drift_summary.empty:
        for _, r in historical_drift_summary.sort_values(
            ["collection", "target_age", "halo_config_px"]
        ).iterrows():
            report.append(
                f"| {r['collection']} | {int(r['target_age'])} | {int(r['halo_config_px'])} "
                f"| {int(r['observations'])} | {r['mean_relative_l2_source_vs_current']:.6f} "
                f"| {r['mean_cosine_source_vs_current']:.6f} |"
            )
    else:
        report.append("| No valid historical feature drift metrics | — | — | — | — |")

    report += [
        "",
        "### 复用路径的估算计算成本",
        "",
        "以下把窗口 age-0 编码缓存的平均 Encoder 时间，均摊到本次指定的 target ages 数量上，再加上 E 的局部 Decoder 时间。这个估算只均摊到选定年龄，不包含完整时序流的其他帧与传播路径开销。",
        "",
        "| Window | Halo | 缓存 Encoder ms | 均摊 Encoder ms/目标 | 历史复用路径估算 ms | 当前整帧路径 ms | 相对节省 % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    if not reuse_cost_summary.empty:
        for _, r in reuse_cost_summary.sort_values(["window_id", "halo_config_px"]).iterrows():
            report.append(
                f"| {r['window_id']} | {int(r['halo_config_px'])} "
                f"| {r['cache_encoder_ms']:.3f} | {r['amortized_cache_encoder_ms_per_selected_target']:.3f} "
                f"| {r['historical_reuse_amortized_path_ms']:.3f} "
                f"| {r['current_full_frame_ms']:.3f} | {r['saving_vs_current_full_frame_pct']:.2f}% |"
            )
    else:
        report.append("| No valid timing rows | — | — | — | — | — | — |")

    report += [
        "",
        "## 10. 决策限制",
        "",
        "1. 如果 E 的输出误差随 target age 快速增加，历史特征直接替代当前帧特征不成立；这不否定 VeloDepth 原生传播路径，只否定这种简化的 stale-cache 对照。",
        "2. 如果 E 保持较低相对误差且估算成本下降，也只能支持继续做原生 temporal state / feature reuse 实验，不能直接声称真实深度精度或完整在线速度已经改善。",
        "3. 全部输出误差都相对同帧整帧模型预测，不是 GT 深度指标；本实验不写回 `features_teacher`，不修改权重。",
        "",
    ]

    report_path = out_dir / "P2_3_v5_26f_实验报告.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    bundle = out_dir / "p2_3_v5_26f_global_token_spatial_feature_ablation_review.zip"
    excluded_npz = []
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            if not p.is_file() or p.name == bundle.name:
                continue
            # NPZ prediction maps are optional pixel-level diagnostics.
            # CSV metrics + shape audit + report are sufficient for routine
            # factorial-effect analysis, so keep NPZ on disk but omit from review ZIP.
            if p.suffix.lower() == ".npz":
                excluded_npz.append(p.name)
                continue
            z.write(p, arcname=p.name)

    print("\n[DONE] v5.26f historical keyframe feature reuse audit finished.")
    print(f"Report: {report_path}")
    print(f"Review ZIP: {bundle}")
    if excluded_npz:
        print("Optional NPZ retained locally but excluded from review ZIP: "
              + ", ".join(excluded_npz))


if __name__ == "__main__":
    main()
