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


VERSION = "P2.3 v5.26 single-step local decode + boundary fusion"
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
        return {"n_pixels": 0, "mae": np.nan, "rmse": np.nan, "relative_l2": np.nan,
                "max_abs": np.nan, "cosine": np.nan}
    d = p-r
    denom = torch.linalg.vector_norm(r).item()
    pn = torch.linalg.vector_norm(p).item()
    rn = torch.linalg.vector_norm(r).item()
    cosine = float(torch.dot(p, r).item() / max(pn * rn, 1e-12))
    return {
        "n_pixels": int(p.numel()),
        "mae": float(d.abs().mean().item()),
        "rmse": float(torch.sqrt((d*d).mean()).item()),
        "relative_l2": float(torch.linalg.vector_norm(d).item() / max(denom, 1e-12)),
        "max_abs": float(d.abs().max().item()),
        "cosine": cosine,
    }


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
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resolution_level", type=int, default=0)
    parser.add_argument("--halos", default="0,32,64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--boundary-band", type=int, default=4)
    args = parser.parse_args()

    halos = sorted({int(x.strip()) for x in args.halos.split(",") if x.strip()})
    if args.frames < 2:
        parser.error("--frames must be >= 2.")
    if args.start_index < 0:
        parser.error("--start-index must be >= 0.")
    if not halos or any(h < 0 for h in halos):
        parser.error("--halos must be non-negative comma-separated integers.")
    if args.warmup < 1 or args.repeats < 2:
        parser.error("Use --warmup >= 1 and --repeats >= 2.")
    if args.boundary_band < 1:
        parser.error("--boundary-band must be >= 1.")
    if not torch.cuda.is_available():
        raise RuntimeError("Run this GPU experiment on an AutoDL GPU instance.")

    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else ROOT
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        root / "outputs" / "p2_3_v5_26_local_decode_fusion"
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

    print(f"[2/7] Loading {args.frames} RGB frames")
    all_paths = sorted(Path(args.rgb).glob("*.png"))
    frame_paths = all_paths[args.start_index:args.start_index + args.frames]
    if len(frame_paths) != args.frames:
        raise RuntimeError(
            f"Expected {args.frames} PNG frames starting at index {args.start_index}; got {len(frame_paths)}."
        )
    rgbs = load_rgb(args.rgb, args.frames, args.start_index)
    frame_names = [p.name for p in frame_paths]

    print("[3/7] Capturing exact packed inputs from VeloDepth.infer()")
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

    print("[4/7] Preparing inputs and checking crop alignment")
    prepared = clone_payload(raw_payload)
    model._set_image_paddings(prepared)
    if "flow_bwd" not in prepared:
        raise KeyError("Packed input is missing flow_bwd; cannot construct a propagation base.")
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
    for halo, geo in geometries.items():
        if geo["core"] != core:
            raise RuntimeError("Core geometry unexpectedly changed across halo settings.")

    had_rays = "rays" in prepared
    old_camera_flag = model.pixel_decoder.test_gt_camera
    model.pixel_decoder.test_gt_camera = bool(had_rays)

    timing_rows = []
    metric_rows = []
    geometry_rows = []
    propagation_rows = []
    shape_rows = []
    map_arrays = {}

    print("[5/7] Running one-step propagation + full reference + local crop keyframe decodes")
    try:
        for target_idx in range(1, T):
            frame_name = frame_names[target_idx]
            current_full_inputs = frame_inputs_at(model, prepared, target_idx, T)
            valid_mask = validity_hw(current_full_inputs, H, W, device)
            core_mask = box_mask(H, W, core, device) & valid_mask

            # Obtain the existing propagation prediction from the immediately
            # preceding frame's full keyframe state. This is the canvas used by
            # the prototype; the current-frame full keyframe is reference only.
            stream = VeloDepthStream(model)
            state = StreamState()
            prev_result, state = stream.keyframe_step(prepared, target_idx - 1, state, T)
            sync(device)
            prop_start = time.perf_counter()
            prop_result, state, restart_requested = stream.propagate_step(
                prepared, target_idx, state, T
            )
            sync(device)
            propagation_ms = (time.perf_counter() - prop_start) * 1000.0
            prop_radius = to_hw(prop_result.radius, H, W)
            prop_conf = to_hw(prop_result.confidence, H, W)
            propagation_rows.append({
                "target_frame_index": target_idx,
                "frame_name": frame_name,
                "restart_requested_by_original_heuristic": bool(restart_requested),
                "propagate_step_wall_ms": float(propagation_ms),
                "base_source": "one-step forward_fast from previous full-frame keyframe",
            })

            # Full-frame current keyframe is an offline reference, not part of
            # the local-refresh deployment path.
            full_inputs = current_full_inputs
            full_out, rows, full_shapes = benchmark_keyframe(
                model, full_inputs, device, args.warmup, args.repeats,
                target_idx, "full_reference", None, 1.0, (H, W)
            )
            timing_rows.extend(rows)
            if full_out is None:
                raise RuntimeError(
                    f"Full-frame reference keyframe failed at frame {target_idx}; see timing CSV."
                )
            ref_radius, ref_conf = output_triplet(full_out, H, W)
            shape_rows.append({
                "target_frame_index": target_idx, "variant": "full_reference",
                "halo_config_px": None, "input_hw": [H, W], "output_shapes": full_shapes,
            })

            prop_all = torch.ones((H, W), dtype=torch.bool, device=device) & valid_mask
            tile_metrics_cache = {}
            map_arrays[f"frame{target_idx}_reference_radius"] = ref_radius.detach().cpu().numpy()
            map_arrays[f"frame{target_idx}_propagation_radius"] = prop_radius.detach().cpu().numpy()

            # Reference comparisons for the unrefreshed propagation base.
            for region_name, region_mask in [
                ("full", prop_all),
                ("core", core_mask),
            ]:
                for pred_name, pred, ref, quantity in [
                    ("prop_vs_full", prop_radius, ref_radius, "radius"),
                ]:
                    m = metric_pair(pred, ref, region_mask)
                    metric_rows.append({
                        "target_frame_index": target_idx, "frame_name": frame_name,
                        "halo_config_px": None, "quantity": quantity,
                        "comparison": pred_name, "region": region_name, **m,
                    })
                if prop_conf is not None and ref_conf is not None:
                    m = metric_pair(prop_conf, ref_conf, region_mask)
                    metric_rows.append({
                        "target_frame_index": target_idx, "frame_name": frame_name,
                        "halo_config_px": None, "quantity": "confidence",
                        "comparison": "prop_vs_full", "region": region_name, **m,
                    })

            for halo in halos:
                geo = geometries[halo]
                core_box = geo["core"]
                tile_box = geo["tile"]
                ty0, ty1, tx0, tx1 = tile_box
                tile_h, tile_w = ty1-ty0, tx1-tx0
                tile_area_ratio = float(tile_h * tile_w / (H * W))
                crop_inputs = crop_frame_inputs(current_full_inputs, (H, W), tile_box)
                crop_out, crop_rows, crop_shapes = benchmark_keyframe(
                    model, crop_inputs, device, args.warmup, args.repeats,
                    target_idx, f"crop_halo_{halo}", halo, tile_area_ratio, (tile_h, tile_w)
                )
                timing_rows.extend(crop_rows)
                effective_halo = {
                    "top": core_box[0] - ty0, "bottom": ty1 - core_box[1],
                    "left": core_box[2] - tx0, "right": tx1 - core_box[3],
                }
                geometry_rows.append({
                    "target_frame_index": target_idx, "frame_name": frame_name,
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
                    "target_frame_index": target_idx, "variant": f"crop_halo_{halo}",
                    "halo_config_px": halo, "input_hw": [tile_h, tile_w],
                    "output_shapes": crop_shapes,
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
                edge_mask = boundary_mask(H, W, tile_box, args.boundary_band, device) & valid_mask

                comparisons = [
                    ("local_vs_full", local_full, ref_radius, "radius"),
                    ("prop_vs_full", prop_radius, ref_radius, "radius"),
                    ("fused_vs_full", fused_radius, ref_radius, "radius"),
                    ("fused_vs_prop", fused_radius, prop_radius, "radius"),
                ]
                region_masks = [
                    ("core", core_mask), ("tile", tile_mask),
                    ("halo_band", halo_band_mask), ("tile_boundary", edge_mask),
                    ("full", prop_all),
                ]
                for comparison, pred, ref, quantity in comparisons:
                    for region_name, region_mask in region_masks:
                        # Local crop values outside its tile are zero placeholders;
                        # only compare local_vs_full within core/tile/halo regions.
                        if comparison == "local_vs_full" and region_name in {"full"}:
                            continue
                        if comparison == "local_vs_full" and region_name == "core":
                            pred_region = local_full
                        elif comparison == "local_vs_full" and region_name == "tile":
                            pred_region = local_full
                        elif comparison == "local_vs_full" and region_name == "halo_band":
                            pred_region = local_full
                        else:
                            pred_region = pred
                        m = metric_pair(pred_region, ref, region_mask)
                        metric_rows.append({
                            "target_frame_index": target_idx, "frame_name": frame_name,
                            "halo_config_px": halo, "quantity": quantity,
                            "comparison": comparison, "region": region_name, **m,
                        })

                if ref_conf is not None and local_conf is not None and prop_conf is not None:
                    for comparison, pred, ref in [
                        ("local_vs_full", local_conf_full, ref_conf),
                        ("fused_vs_full", fused_conf, ref_conf),
                        ("fused_vs_prop", fused_conf, prop_conf),
                    ]:
                        for region_name, region_mask in region_masks:
                            if comparison == "local_vs_full" and region_name == "full":
                                continue
                            m = metric_pair(pred, ref, region_mask)
                            metric_rows.append({
                                "target_frame_index": target_idx, "frame_name": frame_name,
                                "halo_config_px": halo, "quantity": "confidence",
                                "comparison": comparison, "region": region_name, **m,
                            })

                # Store fused maps for offline plotting/review. The full-frame
                # reference is not used to construct the fused prediction.
                map_arrays[f"frame{target_idx}_fused_radius_halo{halo}"] = fused_radius.detach().cpu().numpy()
                map_arrays[f"frame{target_idx}_fused_conf_halo{halo}"] = (
                    fused_conf.detach().cpu().numpy() if fused_conf is not None
                    else np.full((H, W), np.nan, dtype=np.float32)
                )
                map_arrays[f"frame{target_idx}_local_radius_halo{halo}"] = local_full.detach().cpu().numpy()
                map_arrays[f"frame{target_idx}_blend_weight_halo{halo}"] = torch.nn.functional.pad(
                    weight, (tx0, W-tx1, ty0, H-ty1), value=0.0
                ).detach().cpu().numpy()

    finally:
        model.pixel_decoder.test_gt_camera = old_camera_flag

    print("[6/7] Writing CSV, NPZ, metadata, report, and review ZIP")
    timing_df = pd.DataFrame(timing_rows)
    metrics_df = pd.DataFrame(metric_rows)
    geometry_df = pd.DataFrame(geometry_rows)
    propagation_df = pd.DataFrame(propagation_rows)
    shapes_path = out_dir / "p2_3_v5_26_decoder_shapes.json"
    shapes_path.write_text(json.dumps(shape_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    timing_df.to_csv(out_dir / "p2_3_v5_26_keyframe_latency_per_frame.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "p2_3_v5_26_local_refresh_metrics.csv", index=False, encoding="utf-8-sig")
    geometry_df.to_csv(out_dir / "p2_3_v5_26_crop_geometry.csv", index=False, encoding="utf-8-sig")
    propagation_df.to_csv(out_dir / "p2_3_v5_26_propagation_base_timing.csv", index=False, encoding="utf-8-sig")
    latency_summary = summarize_timings(timing_df)
    latency_summary.to_csv(out_dir / "p2_3_v5_26_keyframe_latency_summary.csv", index=False, encoding="utf-8-sig")
    if map_arrays:
        np.savez_compressed(out_dir / "p2_3_v5_26_radius_confidence_maps.npz", **map_arrays)

    metadata = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "rgb_dir": str(Path(args.rgb).resolve()),
        "frame_names": frame_names,
        "frame_count": args.frames,
        "start_index": args.start_index,
        "target_frame_indices": list(range(1, T)),
        "halo_configs_requested_px": halos,
        "warmup_per_keyframe_variant": args.warmup,
        "measured_repeats_per_keyframe_variant": args.repeats,
        "padded_input_hw": [H, W],
        "patch_stride": stride,
        "patch_stride_candidates": stride_candidates,
        "core_box": list(core),
        "boundary_band_px": args.boundary_band,
        "has_rays_input": had_rays,
        "local_decoder_errors": int((geometry_df["status"] != "ok").sum()) if not geometry_df.empty else 0,
        "interpretation_boundary": (
            "Single-step local keyframe decode and feather fusion into a one-step propagation canvas. "
            "Full-frame keyframe outputs are references only and are excluded from local deployment-path cost. "
            "No local teacher-feature state writeback, subsequent propagation rollout, or ground-truth depth metric."
        ),
    }
    (out_dir / "p2_3_v5_26_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# P2.3 v5.26 — 局部解码与边界融合原型",
        "",
        "## 1. 实验目的与范围",
        "",
        f"- GPU：{metadata['gpu_name']}；输入帧数：{args.frames}；处理目标帧：1–{T-1}。",
        f"- Halo 配置：{', '.join(map(str, halos))} px；patch stride：{stride}。",
        f"- 每种关键帧输入形状预热 {args.warmup} 次、测量 {args.repeats} 次。",
        "- 每个目标帧先以完整上一帧关键帧状态执行一次真实传播，得到融合底图；当前帧的整帧关键帧输出仅作为离线参照。",
        "- 局部关键帧输出通过 separable feather mask 融合到传播底图：core 权重为 1，Halo 区域向 tile 外缘线性衰减到 0。",
        "- 本轮不把局部 `features_teacher` 写回全局状态，不评估刷新后的后续传播，也没有使用 GT 深度。",
        "",
        "## 2. 关键帧编码 + 解码耗时",
        "",
        "| Variant | 目标帧数 | 平均输入面积比例 | Encoder ms | Decoder set_shapes + forward ms | 完整 keyframe pipeline ms |",
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
        report.append("| No successful keyframe timing rows | — | — | — | — | — |")
    report += [
        "",
        "说明：`full_reference` 的耗时用于比较，不属于局部刷新部署路径。若在实际触发帧上用局部关键帧替代整帧关键帧，传播尝试是两种方案共有的成本；此表比较的是替换的关键帧编码+解码成本。",
        "",
        "## 3. Radius 输出相对于整帧关键帧参照的误差",
        "",
        "| Halo | 比较方式 | 区域 | 平均 MAE | 平均 RMSE | 平均相对 L2 | 平均余弦 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    radius_metrics = metrics_df[metrics_df["quantity"] == "radius"] if not metrics_df.empty else pd.DataFrame()
    if not radius_metrics.empty:
        display = radius_metrics.groupby(["halo_config_px", "comparison", "region"], dropna=False, as_index=False).agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_relative_l2=("relative_l2", "mean"),
            mean_cosine=("cosine", "mean"),
            target_frames=("target_frame_index", "nunique"),
        )
        chosen = display[
            display["comparison"].isin(["prop_vs_full", "local_vs_full", "fused_vs_full", "fused_vs_prop"])
            & display["region"].isin(["core", "tile", "halo_band", "tile_boundary", "full"])
        ]
        for _, r in chosen.sort_values(["halo_config_px", "comparison", "region"], na_position="first").iterrows():
            halo_label = "baseline" if pd.isna(r["halo_config_px"]) else str(int(r["halo_config_px"]))
            report.append(
                f"| {halo_label} | {r['comparison']} | {r['region']} "
                f"| {r['mean_mae']:.6f} | {r['mean_rmse']:.6f} "
                f"| {r['mean_relative_l2']:.6f} | {r['mean_cosine']:.6f} |"
            )
    else:
        report.append("| No metrics were produced; inspect run.log and decoder shape/errors. | — | — | — | — | — |")
    report += [
        "",
        "## 4. 如何判读边界融合",
        "",
        "- `prop_vs_full`：当前传播底图相对于当前帧整帧关键帧参照的差异。",
        "- `local_vs_full`：局部关键帧解码结果相对于整帧参照的差异，只在 core/tile/Halo 区域计算。",
        "- `fused_vs_full`：局部结果融合进传播底图后，相对于整帧参照的差异。",
        "- `fused_vs_prop`：融合相对于原传播底图改变了多少。",
        "- `tile_boundary`：tile 外缘指定宽度的区域，用于观察融合是否在边界附近引入明显偏差；它不是感知质量指标。",
        "- 若 `fused_vs_full` 在 core 优于 `prop_vs_full`，说明局部刷新对核心区域有帮助；若 Halo 区域/边界明显变差，则需调整 feather 宽度或融合规则。",
        "",
        "## 5. 判读边界与后续 Go/No-Go",
        "",
        "1. **接口 No-Go：** 局部 `forward_keyframe` 报错、输出空间尺寸不匹配或特征/解码形状不兼容。先解决动态形状接口。",
        "2. **质量 No-Go：** 融合后 core 相对整帧参照没有改善，或边界带出现明显误差。先调整 tile/Halo/权重函数。",
        "3. **成本 No-Go：** 局部 keyframe pipeline 耗时未明显低于整帧 pipeline。暂不继续系统化开发。",
        "4. **条件 Go：** 局部解码可运行、核心区域相对传播底图更接近整帧参照、边界误差可控且局部 keyframe 成本较低。下一步再做全序列状态写回与后续传播闭环。",
        "5. 本轮没有写回局部 `features_teacher`。因此即使单帧融合通过，也不能宣称已完成在线局部刷新策略或端到端加速。",
        "",
    ]
    report_path = out_dir / "P2_3_v5_26_实验报告.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    bundle = out_dir / "p2_3_v5_26_local_decode_fusion_review.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.iterdir()):
            if p.is_file() and p.name != bundle.name:
                z.write(p, arcname=p.name)

    print("[7/7] PASS: P2.3 v5.26 local decode + boundary fusion prototype")
    print(f"Output directory: {out_dir}")
    print(f"Report: {report_path}")
    print(f"Review ZIP: {bundle}")


if __name__ == "__main__":
    main()
