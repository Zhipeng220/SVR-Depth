#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.3 v5.26 — Single-step local keyframe decode + boundary fusion prototype.

Purpose:
  * Use a prior full-frame keyframe followed by one real propagation step to
    obtain a propagated canvas for the current frame.
  * Decode a full-frame keyframe output only as an offline reference.
  * Decode the current frame on local crops with halo=0/32/64, feather-blend
    each local result into the propagated canvas, and compare against the
    full-frame keyframe reference.
  * Measure full-frame vs local keyframe encoder+decoder latency.

Scope limits:
  * This is a single-step prototype. It does NOT write cropped teacher features
    back into the full-frame temporal state and does NOT test subsequent
    propagation after a local refresh.
  * The full-frame keyframe output is an evaluation reference only. It is not
    part of the local deployment path.
  * No ground-truth depth metrics are computed. Metrics compare radius and
    confidence outputs against the full-frame keyframe reference.
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


VERSION = "P2.3 v5.26b long-interval propagation + local refresh gain"
DEFAULT_HALOS = [0, 32, 64]


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


def main():
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--rgb", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb")
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--start-index", type=int, default=0)
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
    if args.start_index < 0:
        parser.error("--start-index must be >= 0.")
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
    if not torch.cuda.is_available():
        raise RuntimeError("Run this GPU experiment on an AutoDL GPU instance.")

    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else ROOT
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        root / "outputs" / "p2_3_v5_26b_long_interval_refresh"
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

    print(f"[2/8] Loading {args.frames} RGB frames")
    all_paths = sorted(Path(args.rgb).glob("*.png"))
    frame_paths = all_paths[args.start_index:args.start_index + args.frames]
    if len(frame_paths) != args.frames:
        raise RuntimeError(
            f"Expected {args.frames} PNG frames starting at index {args.start_index}; got {len(frame_paths)}."
        )
    rgbs = load_rgb(args.rgb, args.frames, args.start_index)
    frame_names = [p.name for p in frame_paths]

    print("[3/8] Capturing exact packed inputs from VeloDepth.infer()")
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
        raise RuntimeError(f"Captured T={T}; requested {args.frames}.")

    print("[4/8] Preparing normalized flow and crop geometry")
    prepared = clone_payload(raw_payload)
    model._set_image_paddings(prepared)
    if "flow_bwd" not in prepared:
        raise KeyError("Packed input is missing flow_bwd; cannot construct a propagation rollout.")
    prepared["flow_bwd"], prepared["grid_sample"] = model._normalize_and_grid_from_flow(
        prepared["flow_bwd"], prepared["image"], prepared["image_paddings"], prepared["depth_paddings"]
    )
    image_bt = model._bt(prepared["image"], 1, T)
    H, W = int(image_bt.shape[-2]), int(image_bt.shape[-1])
    stride, stride_candidates = discover_patch_stride(model.pixel_encoder)
    if stride is None:
        raise RuntimeError("Could not infer pixel patch stride; refusing to run misaligned crop decode.")
    geometries = {halo: choose_core_and_tile(H, W, stride, halo) for halo in halos}
    core = geometries[halos[0]]["core"]
    if any(geo["core"] != core for geo in geometries.values()):
        raise RuntimeError("Core geometry unexpectedly changed across Halo settings.")

    had_rays = "rays" in prepared
    old_camera_flag = model.pixel_decoder.test_gt_camera
    model.pixel_decoder.test_gt_camera = bool(had_rays)

    propagation_rows = []
    prop_snapshots = {}
    timing_rows = []
    metric_rows = []
    geometry_rows = []
    shape_rows = []
    map_arrays = {}

    def run_no_refresh_rollout(collect: bool):
        """One keyframe at age 0, then forced propagation-only through the clip."""
        stream = VeloDepthStream(model)
        state = StreamState()
        kf_start = time.perf_counter()
        _, state = stream.keyframe_step(prepared, 0, state, T)
        sync(device)
        kf_ms = (time.perf_counter() - kf_start) * 1000.0
        local_rows = []
        local_snapshots = {}
        for t in range(1, T):
            sync(device)
            t0 = time.perf_counter()
            result, state, restart_requested = stream.propagate_step(prepared, t, state, T)
            sync(device)
            step_ms = (time.perf_counter() - t0) * 1000.0

            # Counterfactual: ignore the automatic refresh request so that the
            # same propagation state is followed continuously. propagate_step()
            # already produced and installed the candidate state. If it signaled
            # restart, manually advance age/current_t as a no-refresh policy.
            if restart_requested:
                state.distance_kf += 1
            state.first_keyframe = True
            state.current_t = t + 1

            local_rows.append({
                "frame_index": t,
                "frame_name": frame_names[t],
                "propagation_age": t,
                "propagate_step_wall_ms": float(step_ms),
                "original_heuristic_requested_refresh": bool(restart_requested),
                "forced_no_refresh": True,
                "state_distance_kf_after_step": int(state.distance_kf),
            })
            if collect and t in ages:
                local_snapshots[t] = {
                    "radius": result.radius.detach().clone(),
                    "confidence": result.confidence.detach().clone(),
                    "rays": result.rays.detach().clone(),
                }
        return float(kf_ms), local_rows, local_snapshots

    print("[5/8] Warming up the forced no-refresh propagation rollout")
    for i in range(args.stream_warmup_rollouts):
        _, _, _ = run_no_refresh_rollout(collect=False)
        print(f"  warmup rollout {i+1}/{args.stream_warmup_rollouts} complete")

    print("[6/8] Measuring a continuous no-refresh rollout")
    try:
        rollout_kf_ms, propagation_rows, prop_snapshots = run_no_refresh_rollout(collect=True)
        if set(prop_snapshots) != set(ages):
            raise RuntimeError(f"Missing target-age snapshots: got {sorted(prop_snapshots)}, expected {ages}.")

        print("[7/8] Evaluating full references and local refresh at target ages")
        for age in ages:
            frame_name = frame_names[age]
            current_inputs = frame_inputs_at(model, prepared, age, T)
            valid_mask = validity_hw(current_inputs, H, W, device)
            prop_radius = to_hw(prop_snapshots[age]["radius"], H, W)
            prop_conf = to_hw(prop_snapshots[age]["confidence"], H, W)
            core_mask = box_mask(H, W, core, device) & valid_mask
            full_mask = torch.ones((H, W), dtype=torch.bool, device=device) & valid_mask

            # Full-frame current keyframe is an offline reference only.
            full_out, full_rows, full_shapes = benchmark_keyframe(
                model, current_inputs, device, args.warmup, args.repeats,
                age, "full_reference", None, 1.0, (H, W)
            )
            timing_rows.extend(full_rows)
            if full_out is None:
                raise RuntimeError(f"Full-frame reference failed at target age {age}; inspect timing CSV.")
            ref_radius, ref_conf = output_triplet(full_out, H, W)
            shape_rows.append({
                "target_age": age, "frame_name": frame_name, "variant": "full_reference",
                "halo_config_px": None, "input_hw": [H, W], "output_shapes": full_shapes,
            })

            map_arrays[f"age{age}_prop_radius"] = prop_radius.detach().cpu().numpy()
            map_arrays[f"age{age}_reference_radius"] = ref_radius.detach().cpu().numpy()

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
                timing_rows.extend(crop_rows)
                effective_halo = {
                    "top": core_box[0] - ty0, "bottom": ty1 - core_box[1],
                    "left": core_box[2] - tx0, "right": tx1 - core_box[3],
                }
                geometry_rows.append({
                    "target_age": age, "frame_name": frame_name,
                    "halo_config_px": halo, "patch_stride": stride,
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
                    "target_age": age, "frame_name": frame_name,
                    "variant": f"crop_halo_{halo}", "halo_config_px": halo,
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
                        if comparison == "local_vs_full":
                            effective_mask = region_mask & tile_mask
                        else:
                            effective_mask = region_mask
                        m = metric_pair(pred, ref, effective_mask)
                        metric_rows.append({
                            "target_age": age, "frame_name": frame_name,
                            "halo_config_px": halo, "quantity": "radius",
                            "comparison": comparison, "region": region_name, **m,
                        })

                if ref_conf is not None and local_conf is not None and prop_conf is not None and fused_conf is not None:
                    confidence_comparisons = [
                        ("prop_vs_full", prop_conf, ref_conf),
                        ("local_vs_full", local_conf_full, ref_conf),
                        ("fused_vs_full", fused_conf, ref_conf),
                        ("fused_vs_prop", fused_conf, prop_conf),
                    ]
                    for comparison, pred, ref in confidence_comparisons:
                        for region_name, region_mask in regions:
                            if comparison == "local_vs_full" and region_name == "full":
                                continue
                            effective_mask = (region_mask & tile_mask) if comparison == "local_vs_full" else region_mask
                            m = metric_pair(pred, ref, effective_mask)
                            metric_rows.append({
                                "target_age": age, "frame_name": frame_name,
                                "halo_config_px": halo, "quantity": "confidence",
                                "comparison": comparison, "region": region_name, **m,
                            })

                map_arrays[f"age{age}_local_radius_halo{halo}"] = local_full.detach().cpu().numpy()
                map_arrays[f"age{age}_fused_radius_halo{halo}"] = fused_radius.detach().cpu().numpy()
                map_arrays[f"age{age}_fused_conf_halo{halo}"] = (
                    fused_conf.detach().cpu().numpy() if fused_conf is not None
                    else np.full((H, W), np.nan, dtype=np.float32)
                )
                map_arrays[f"age{age}_blend_weight_halo{halo}"] = torch.nn.functional.pad(
                    weight, (tx0, W-tx1, ty0, H-ty1), value=0.0
                ).detach().cpu().numpy()

    finally:
        model.pixel_decoder.test_gt_camera = old_camera_flag

    print("[8/8] Writing reports and review ZIP")
    timing_df = pd.DataFrame(timing_rows)
    metrics_df = pd.DataFrame(metric_rows)
    propagation_df = pd.DataFrame(propagation_rows)
    geometry_df = pd.DataFrame(geometry_rows)

    timing_df.to_csv(out_dir / "p2_3_v5_26b_keyframe_latency_per_frame.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "p2_3_v5_26b_local_refresh_metrics.csv", index=False, encoding="utf-8-sig")
    propagation_df.to_csv(out_dir / "p2_3_v5_26b_propagation_age_trace.csv", index=False, encoding="utf-8-sig")
    geometry_df.to_csv(out_dir / "p2_3_v5_26b_crop_geometry.csv", index=False, encoding="utf-8-sig")
    latency_summary = summarize_timings(timing_df)
    latency_summary.to_csv(out_dir / "p2_3_v5_26b_keyframe_latency_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "p2_3_v5_26b_decoder_shapes.json").write_text(
        json.dumps(shape_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if map_arrays:
        np.savez_compressed(out_dir / "p2_3_v5_26b_prediction_maps.npz", **map_arrays)

    # Refresh gain: positive means fusion is closer to the same-frame full keyframe reference.
    gain_rows = []
    if not metrics_df.empty:
        rm = metrics_df[metrics_df["quantity"] == "radius"]
        idx_cols = ["target_age", "frame_name", "halo_config_px", "region"]
        prop = rm[rm["comparison"] == "prop_vs_full"].set_index(idx_cols)
        fused = rm[rm["comparison"] == "fused_vs_full"].set_index(idx_cols)
        common = prop.index.intersection(fused.index)
        for key in common:
            pr = prop.loc[key]
            fu = fused.loc[key]
            # There should be one row per key; guard against accidental duplicates.
            if isinstance(pr, pd.DataFrame) or isinstance(fu, pd.DataFrame):
                continue
            gain_rows.append({
                "target_age": key[0], "frame_name": key[1],
                "halo_config_px": key[2], "region": key[3],
                "prop_relative_l2": float(pr["relative_l2"]),
                "fused_relative_l2": float(fu["relative_l2"]),
                "refresh_gain_relative_l2": float(pr["relative_l2"] - fu["relative_l2"]),
                "prop_mae": float(pr["mae"]),
                "fused_mae": float(fu["mae"]),
                "refresh_gain_mae": float(pr["mae"] - fu["mae"]),
                "prop_gradient_mae": float(pr["gradient_mae"]),
                "fused_gradient_mae": float(fu["gradient_mae"]),
                "refresh_gain_gradient_mae": float(pr["gradient_mae"] - fu["gradient_mae"]),
                "positive_refresh_gain": bool(pr["relative_l2"] > fu["relative_l2"]),
            })
    gain_df = pd.DataFrame(gain_rows)
    gain_df.to_csv(out_dir / "p2_3_v5_26b_refresh_gain.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "rgb_dir": str(Path(args.rgb).resolve()),
        "frame_names": frame_names,
        "frame_count": args.frames,
        "start_index": args.start_index,
        "target_ages": ages,
        "halo_configs_requested_px": halos,
        "warmup_per_keyframe_variant": args.warmup,
        "measured_repeats_per_keyframe_variant": args.repeats,
        "stream_warmup_rollouts": args.stream_warmup_rollouts,
        "padded_input_hw": [H, W],
        "patch_stride": stride,
        "patch_stride_candidates": stride_candidates,
        "core_box": list(core),
        "boundary_band_px": args.boundary_band,
        "has_rays_input": had_rays,
        "forced_no_refresh": True,
        "automatic_restart_requests_are_logged_but_ignored": True,
        "rollout_keyframe_wall_ms_measured": rollout_kf_ms,
        "local_decoder_errors": int((geometry_df["status"] != "ok").sum()) if not geometry_df.empty else 0,
        "interpretation_boundary": (
            "Counterfactual open-loop rollout: one full keyframe at age 0, then continuous propagation "
            "with automatic restart requests logged but deliberately ignored. Local/full keyframe outputs "
            "are evaluated offline at target ages. Metrics compare radius/confidence to same-frame full "
            "keyframe predictions, not ground-truth depth. No local feature-state writeback or post-refresh "
            "propagation rollout is tested."
        ),
    }
    (out_dir / "p2_3_v5_26b_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# P2.3 v5.26b — 长间隔传播下的局部刷新收益验证",
        "",
        "## 1. 实验范围",
        "",
        f"- GPU：{metadata['gpu_name']}；输入帧数：{args.frames}；初始关键帧：age 0。",
        f"- 目标传播年龄：{', '.join(map(str, ages))}；Halo：{', '.join(map(str, halos))} px；patch stride：{stride}。",
        f"- 每个关键帧输入形状预热 {args.warmup} 次、正式测量 {args.repeats} 次；完整传播轨迹预热 {args.stream_warmup_rollouts} 次。",
        "- 从 age 0 的完整关键帧状态开始，连续执行 `forward_fast` 传播；在目标年龄记录预测。",
        "- 原始自动刷新启发式的请求会被记录，但为构造强制不刷新的反事实轨迹而忽略。",
        "- 同帧整帧关键帧输出仅作为离线参照；不使用 GT 深度，不写回局部 `features_teacher`，不测试刷新后的后续传播。",
        "",
        "## 2. 原始传播误差随年龄变化",
        "",
        "| Age | 原始启发式请求刷新 | 传播耗时 ms | Core 相对 L2 | Full 相对 L2 |",
        "|---:|---|---:|---:|---:|",
    ]
    if not propagation_df.empty and not metrics_df.empty:
        for _, pr in propagation_df[propagation_df["frame_index"].isin(ages)].sort_values("frame_index").iterrows():
            age = int(pr["propagation_age"])
            subset = metrics_df[
                (metrics_df["target_age"] == age)
                & (metrics_df["halo_config_px"] == halos[0])
                & (metrics_df["quantity"] == "radius")
                & (metrics_df["comparison"] == "prop_vs_full")
            ]
            core_rows = subset[subset["region"] == "core"]
            full_rows = subset[subset["region"] == "full"]
            core_err = float(core_rows["relative_l2"].mean()) if not core_rows.empty else np.nan
            full_err = float(full_rows["relative_l2"].mean()) if not full_rows.empty else np.nan
            report.append(
                f"| {age} | {bool(pr['original_heuristic_requested_refresh'])} | "
                f"{pr['propagate_step_wall_ms']:.3f} | {core_err:.6f} | {full_err:.6f} |"
            )
    else:
        report.append("| No data | — | — | — | — |")
    report += [
        "",
        "## 3. 局部关键帧处理耗时",
        "",
        "| Variant | 目标帧数 | 输入面积比例 | Encoder ms | Decoder ms | Keyframe pipeline ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not latency_summary.empty:
        for _, r in latency_summary.sort_values(["variant", "halo_config_px"], na_position="first").iterrows():
            report.append(
                f"| {r['variant']} | {int(r['target_frames'])} | {r['mean_area_ratio']:.3f} "
                f"| {r['mean_encoder_ms']:.3f} | {r['mean_decoder_ms']:.3f} "
                f"| {r['mean_keyframe_pipeline_ms']:.3f} |"
            )
    else:
        report.append("| No successful timing rows | — | — | — | — | — |")
    report += [
        "",
        "## 4. 局部刷新收益（正值代表更接近整帧参照）",
        "",
        "| Age | Halo | Region | Prop rel L2 | Fused rel L2 | Refresh gain | Fused gradient MAE |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    if not gain_df.empty:
        selected = gain_df[gain_df["region"].isin(["core", "full", "core_edge_band", "tile_outer_boundary"])]
        for _, r in selected.sort_values(["target_age", "halo_config_px", "region"]).iterrows():
            report.append(
                f"| {int(r['target_age'])} | {int(r['halo_config_px'])} | {r['region']} "
                f"| {r['prop_relative_l2']:.6f} | {r['fused_relative_l2']:.6f} "
                f"| {r['refresh_gain_relative_l2']:.6f} | {r['fused_gradient_mae']:.6f} |"
            )
    else:
        report.append("| No gain rows | — | — | — | — | — | — |")
    report += [
        "",
        "## 5. 判读与限制",
        "",
        "- `refresh_gain_relative_l2 > 0`：局部融合在该区域更接近同帧整帧关键帧参照；小于等于 0 则未获益。",
        "- `original_heuristic_requested_refresh` 记录原策略在强制无刷新轨迹上会请求刷新的时刻；这些请求在本轮被忽略。",
        "- `core_edge_band` 覆盖核心区边界两侧；`tile_outer_boundary` 检查裁剪外缘。`gradient_mae` 用于观察局部空间梯度与整帧参照的差异。",
        "- 所有质量指标均为模型输出之间的一致性，不是真实深度准确率；没有 GT 深度指标。",
        "- 即使某年龄获得正收益，本轮也没有把局部 `features_teacher` 写回全局状态，不代表刷新后的连续传播闭环已经成立。",
        "",
        "## 6. 下一阶段 Go/No-Go",
        "",
        "1. 若传播误差随 age 增长且 Halo 32/64 在多个年龄、核心区上有稳定正收益：进入状态写回原型。",
        "2. 若只在个别帧获益：先结合原始启发式请求和刷新收益，扩大帧样本验证，不立即开发全局调度器。",
        "3. 若局部刷新在所有年龄都无正收益：检查局部裁剪解码与全帧语义差异，暂缓状态写回。",
        "4. 若关键帧成本节省被边界或状态管理成本抵消：暂缓系统化实现。",
        "",
    ]
    report_path = out_dir / "P2_3_v5_26b_实验报告.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    bundle = out_dir / "p2_3_v5_26b_long_interval_refresh_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            if p.is_file() and p.name != bundle.name:
                z.write(p, arcname=p.name)

    print("[DONE] P2.3 v5.26b completed")
    print(f"Output directory: {out_dir}")
    print(f"Report: {report_path}")
    print(f"Review ZIP: {bundle}")


if __name__ == "__main__":
    main()
