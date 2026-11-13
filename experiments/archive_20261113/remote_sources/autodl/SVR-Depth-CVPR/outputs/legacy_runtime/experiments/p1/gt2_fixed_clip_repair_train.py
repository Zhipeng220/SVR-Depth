#!/usr/bin/env python3
"""GT2: matched C/D fixed-clip real-GT repair-head training for VeloDepth.

Scope: three consecutive Bonn Crowd2 RGB frames (default global indices 8,9,10),
using the already audited GT1-A1 one-to-one pairing CSV. Only the persistent
feature repair head is optimized. The original VeloDepth weights stay frozen.
This is a fixed-clip optimizability test, NOT a generalization evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file
from torchvision.transforms.v2 import functional as TF

from velodepth.models.velodepth import (
    VeloDepth,
    _postprocess,
    get_paddings,
    get_resize_factor,
)
from velodepth.models.encoder_velo import rgb2luma
from velodepth.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD

DEPTH_SCALE = 5000.0
EXPECTED_HEAD_PREFIX = "pixel_decoder.persistent_feature_repair."


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_pair_rows(pair_csv: Path, indices: list[int], max_delta_ms: float) -> list[dict]:
    if not pair_csv.is_file():
        raise FileNotFoundError(
            f"GT1-A1 配对清单不存在：{pair_csv}\n"
            "请确认 GT1-A1 已运行，或通过 --pair-csv 指定实际路径。"
        )
    by_index = {}
    with pair_csv.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            by_index[int(row["frame_index_zero_based"])] = row

    if indices != list(range(indices[0], indices[0] + 3)):
        raise ValueError("GT2 第一版要求恰好 3 个连续全局 RGB 索引，例如 8,9,10。")

    selected = []
    for idx in indices:
        if idx not in by_index:
            raise RuntimeError(f"GT1-A1 配对清单缺少 RGB 全局索引 {idx}")
        row = by_index[idx]
        if row.get("gt_match_status") != "MATCHED":
            raise RuntimeError(f"索引 {idx} 没有有效 GT 配对：{row}")
        rgb_path = Path(row["rgb_path"])
        gt_path = Path(row["depth_path"])
        if not rgb_path.is_file():
            raise FileNotFoundError(f"RGB 文件不存在：{rgb_path}")
        if not gt_path.is_file():
            raise FileNotFoundError(f"GT 文件不存在：{gt_path}")
        dt = float(row["abs_delta_ms"])
        if dt > max_delta_ms:
            raise RuntimeError(
                f"索引 {idx} 的 RGB-GT 时间差 {dt:.3f} ms 超过阈值 {max_delta_ms:.3f} ms"
            )
        reused = str(row.get("gt_reused", "False")).strip().lower() in {"true", "1", "yes"}
        if reused:
            raise RuntimeError(f"GT1-A1 配对清单显示索引 {idx} 重用了 GT：{gt_path}")
        selected.append({
            "index": idx,
            "rgb_timestamp": float(row["rgb_timestamp"]),
            "gt_timestamp": float(row["depth_timestamp"]),
            "rgb_path": rgb_path.resolve(),
            "gt_path": gt_path.resolve(),
            "abs_delta_ms": dt,
        })

    if len({str(x["gt_path"]) for x in selected}) != 3:
        raise RuntimeError("GT2 三帧的 GT 路径不唯一，停止训练。")
    if any(selected[i]["rgb_timestamp"] <= selected[i - 1]["rgb_timestamp"] for i in range(1, 3)):
        raise RuntimeError("RGB 时间戳非严格递增，停止训练。")
    return selected


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


def calculate_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if not valid.any():
        raise RuntimeError("GT 与预测没有共同有效像素。")
    g = gt[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)
    ratio = np.maximum(g / p, p / g)
    return {
        "AbsRel": float(np.mean(np.abs(g - p) / g)),
        "RMSE": float(np.sqrt(np.mean((g - p) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "valid_ratio": float(valid.mean()),
        "gt_valid_ratio": float((gt > 0).mean()),
        "valid_pixels": int(valid.sum()),
        "total_pixels": int(valid.size),
    }


def masked_log_depth_loss(pred: torch.Tensor, gt: torch.Tensor) -> tuple[torch.Tensor, int]:
    pred32 = pred.float()
    gt32 = gt.float()
    mask = (gt32 > 0) & torch.isfinite(gt32) & torch.isfinite(pred32.detach()) & (pred32.detach() > 0)
    count = int(mask.sum().item())
    if count < 1000:
        raise RuntimeError(f"有效监督像素过少：{count}")
    delta = torch.log(pred32[mask].clamp_min(1e-4)) - torch.log(gt32[mask].clamp_min(1e-4))
    # Smooth robust log-depth L1. This is the explicitly logged GT objective,
    # not the Dummy loss from the evaluation config.
    loss = torch.sqrt(delta.square() + 1e-6).mean()
    return loss, count


def load_model(snapshot: Path, mode: str, device: torch.device) -> torch.nn.Module:
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"预训练快照不完整：{snapshot}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = json.loads(json.dumps(config))  # independent deep copy
    if mode == "off":
        config["model"].pop("persistent_feature_repair", None)
    else:
        config["model"]["persistent_feature_repair"] = {
            "enabled": True,
            "mode": mode,
            "bottleneck": 32,
        }

    model = VeloDepth(config)
    model.cpu()
    model.eval()
    state = load_file(str(weights_path), device="cpu")
    result = model.load_state_dict(state, strict=(mode == "off"))
    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    del state

    if mode == "off":
        if missing or unexpected:
            raise RuntimeError(f"A 基线权重不严格匹配：missing={missing[:10]}, unexpected={unexpected[:10]}")
    else:
        bad_missing = [k for k in missing if not k.startswith(EXPECTED_HEAD_PREFIX)]
        if bad_missing or unexpected:
            raise RuntimeError(
                "原始预训练权重与修复头模型不匹配："
                f"bad_missing={bad_missing[:10]}, unexpected={unexpected[:10]}"
            )
        if not missing:
            raise RuntimeError("预期的新修复头缺失键为 0；请检查当前模型/权重是否与审计版本一致。")

    model.to(device)
    model.eval()
    # Avoid chained assignment: this project checkout has a custom eval/cpu
    # behavior that was previously observed to return None in a chained expression.
    model.resolution_level = 0
    model.interpolation_mode = "bilinear"
    model.alpha_flow = 1e9
    model.beta_flow = 0.0
    model.alpha_luma = 1e9
    model.beta_luma = 0.0

    for p in model.parameters():
        p.requires_grad_(False)

    if mode != "off":
        head = model.pixel_decoder.persistent_feature_repair
        if head is None:
            raise RuntimeError("修复头未注册。请先确认 E2-S8 当前源码补丁已应用。")
        configured_mode = getattr(model.pixel_decoder, "feature_repair_mode", None)
        if configured_mode != mode:
            raise RuntimeError(
                f"修复头配置/分支接线不一致：requested={mode}, configured={configured_mode}。"
                "停止训练，不要在脚本里强行覆盖 mode。"
            )
        for p in head.parameters():
            p.requires_grad_(True)
        head.train()  # model stays eval; GroupNorm has no running-stat updates
    else:
        if getattr(model.pixel_decoder, "persistent_feature_repair", None) is not None:
            raise RuntimeError("A 组不应注册修复头。")
    return model


def preprocess_rgb(rgb_np: np.ndarray, device, paddings, new_shape, mean, std):
    x = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device)
    x = TF.normalize(x.float() / 255.0, mean=IMAGENET_DATASET_MEAN, std=IMAGENET_DATASET_STD)
    x = F.pad(x, paddings, value=0.0)
    x = F.interpolate(x, size=new_shape, mode="bilinear", align_corners=False)
    frame_u8 = ((x * std + mean).clamp(0, 1) * 255.0).byte().squeeze(0).cpu()
    return x, frame_u8


def make_z_depth(radius_state, rays_map, padded_shape, paddings, interpolation_mode):
    points_canvas = radius_state * rays_map
    points_original = _postprocess(
        points_canvas,
        padded_shape,
        paddings,
        interpolation_mode=interpolation_mode,
    )
    # VeloDepth returns 3-D points here; channel -1 is camera-coordinate z-depth,
    # matching the already audited GT0/GT1 scripts (not radial distance).
    return points_original[0, -1].float()


def build_clip_context(model, selected: list[dict], device: torch.device) -> dict:
    root_img = load_rgb(selected[0]["rgb_path"])
    original_h, original_w = root_img.shape[:2]
    for rec in selected:
        with Image.open(rec["rgb_path"]) as im:
            if im.size != (original_w, original_h):
                raise RuntimeError(f"RGB 尺寸不一致：{rec['rgb_path']} size={im.size}")

    ratio_bounds = model.shape_constraints["ratio_bounds"]
    pixels_min = float(model.shape_constraints["pixels_min"])
    pixels_max = float(model.shape_constraints["pixels_max"])
    interval = (pixels_max - pixels_min) / 10.0
    pixels_bounds = (pixels_min, pixels_min + interval)
    paddings, (padded_h, padded_w) = get_paddings((original_h, original_w), ratio_bounds)
    _resize_factor, (new_h, new_w) = get_resize_factor((padded_h, padded_w), pixels_bounds)

    mean = torch.tensor(IMAGENET_DATASET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DATASET_STD, device=device).view(1, 3, 1, 1)
    frames = []
    for rec in selected:
        rgb_np = load_rgb(rec["rgb_path"])
        img, frame_u8 = preprocess_rgb(rgb_np, device, paddings, (new_h, new_w), mean, std)
        del rgb_np
        gt_raw = cv2.imread(str(rec["gt_path"]), cv2.IMREAD_UNCHANGED)
        if gt_raw is None:
            raise RuntimeError(f"无法读取 GT：{rec['gt_path']}")
        if gt_raw.dtype != np.uint16 or gt_raw.ndim != 2:
            raise RuntimeError(f"GT 格式错误：{rec['gt_path']} dtype={gt_raw.dtype}, shape={gt_raw.shape}")
        if gt_raw.shape != (original_h, original_w):
            raise RuntimeError(
                f"GT/RGB 尺寸不一致：GT={gt_raw.shape}, RGB={(original_h, original_w)}, path={rec['gt_path']}"
            )
        gt_np = gt_raw.astype(np.float32) / DEPTH_SCALE
        frames.append({
            **rec,
            "img": img,
            "frame_u8": frame_u8,
            "gt_np": gt_np,
            "gt": torch.from_numpy(gt_np).to(device),
        })

    validity_mask = torch.ones((1, 1, new_h, new_w), device=device, dtype=torch.bool)
    image_paddings = torch.zeros((1, 4), device=device, dtype=torch.long)
    depth_paddings = torch.zeros((1, 4), device=device, dtype=torch.long)
    fast_static = []
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    for t in (1, 2):
        curr_rgb = frames[t]["frame_u8"].permute(1, 2, 0).numpy()
        prev_rgb = frames[t - 1]["frame_u8"].permute(1, 2, 0).numpy()
        flow_np = dis.calc(
            cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY),
            None,
        )
        flow_px = torch.from_numpy(flow_np).to(device=device).permute(2, 0, 1).unsqueeze(0)
        flow_norm, grid_sample = model._normalize_and_grid_from_flow(
            flow_px, frames[t]["img"], image_paddings, depth_paddings
        )
        image_residual = rgb2luma(frames[t]["img"]).unsqueeze(1) - F.grid_sample(
            rgb2luma(frames[t - 1]["img"]).unsqueeze(1), grid_sample, mode="bilinear"
        )
        fast_static.append({
            "flow_norm": flow_norm,
            "grid_sample": grid_sample,
            "image_residual": image_residual,
            "validity_mask": validity_mask,
        })
        del flow_np, flow_px, curr_rgb, prev_rgb

    base_inputs = {
        "image": frames[0]["img"],
        "image_original": frames[0]["img"],
        "validity_mask": validity_mask,
    }
    base_metas = [{"T": 1, "B": 1}]
    # Use no_grad, NOT inference_mode: these cached tensors will be consumed by
    # differentiable fast steps, and inference tensors can break autograd saves.
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        base_features, base_tokens = model.pixel_encoder(base_inputs["image"])
        base_features = [
            model.stacking_fn(base_features[i:j]).contiguous()
            for i, j in model.slices_encoder_range
        ]
        base_tokens = [
            model.stacking_fn(base_tokens[i:j]).contiguous()
            for i, j in model.slices_encoder_range
        ]
        base_inputs["features"], base_inputs["tokens"] = base_features, base_tokens
        model.pixel_decoder.set_shapes(base_inputs, base_metas)
        rays_base, radius_state, confidence_state, features_state = model.pixel_decoder.forward_keyframe(
            base_inputs, base_metas
        )

    rays_map = rays_base.reshape(1, new_h, new_w, 3).permute(0, 3, 1, 2).contiguous().detach()
    context = {
        "frames": frames,
        "fast_static": fast_static,
        "rays_map": rays_map,
        "keyframe_radius": radius_state.detach(),
        "keyframe_confidence": confidence_state.detach(),
        "keyframe_features": [x.detach() for x in features_state],
        "paddings": paddings,
        "padded_shape": (padded_h, padded_w),
        "original_shape": (original_h, original_w),
        "network_shape": (new_h, new_w),
        "validity_mask": validity_mask,
        "image_paddings": image_paddings,
        "depth_paddings": depth_paddings,
    }
    del base_inputs, base_metas, base_features, base_tokens, rays_base, radius_state, confidence_state, features_state
    return context


def fast_step(model, context: dict, t: int, radius_state, features_state):
    frame = context["frames"][t]
    prev = context["frames"][t - 1]
    static = context["fast_static"][t - 1]
    flow_t_tm1 = static["flow_norm"].clone()
    residuals_t_tm1 = static["image_residual"].clone()
    features_residual, flow_refined, flow_features = model.small_encoder(
        rgb_t=frame["img"],
        depth_t=radius_state,
        flow_t=flow_t_tm1,
        raft_t=None,
        residuals_t=residuals_t_tm1,
    )
    if len(features_residual) > 4:
        features_residual = [
            x.permute(0, 3, 1, 2)
            for i, x in enumerate(features_residual)
            if i in [2, 5, 8, 11]
        ]
    fast_in = {
        "grid_sample": static["grid_sample"],
        "image_residual": static["image_residual"],
        "image": frame["img"],
        "image_original": frame["img"],
        "validity_mask": static["validity_mask"],
        "flow_bwd": static["flow_norm"],
        "features_teacher": features_state,
        "features_residual": features_residual,
        "flow_features": flow_features,
        "flow_bwd_refined": flow_refined,
    }
    fast_metas = [{"T": 1, "B": 1}]
    radius_next, confidence_next, features_next = model.pixel_decoder.forward_fast(fast_in, fast_metas)
    del prev, flow_t_tm1, residuals_t_tm1, features_residual, flow_refined, flow_features, fast_in
    return radius_next, confidence_next, features_next


def forward_clip(model, context: dict) -> list[torch.Tensor]:
    radius_state = context["keyframe_radius"]
    confidence_state = context["keyframe_confidence"]
    features_state = context["keyframe_features"]
    preds = []
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        preds.append(make_z_depth(
            radius_state, context["rays_map"], context["padded_shape"],
            context["paddings"], model.interpolation_mode
        ))
    for t in (1, 2):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            radius_state, confidence_state, features_state = fast_step(
                model, context, t, radius_state, features_state
            )
            pred = make_z_depth(
                radius_state, context["rays_map"], context["padded_shape"],
                context["paddings"], model.interpolation_mode
            )
        preds.append(pred)
    return preds


def evaluate_clip(model, context: dict, mode: str, stage: str) -> tuple[list[np.ndarray], list[dict], float, int]:
    with torch.no_grad():
        preds_t = forward_clip(model, context)
        pred_np = [x.detach().float().cpu().numpy() for x in preds_t]
    metric_rows = []
    target_loss = float("nan")
    target_valid = 0
    with torch.no_grad():
        loss_t, target_valid = masked_log_depth_loss(
            torch.from_numpy(pred_np[-1]).to(context["frames"][-1]["gt"].device),
            context["frames"][-1]["gt"],
        )
        target_loss = float(loss_t.item())
    for i, (frame, pred) in enumerate(zip(context["frames"], pred_np)):
        m = calculate_metrics(frame["gt_np"], pred)
        metric_rows.append({
            "mode": mode,
            "stage": stage,
            "frame_index_zero_based": frame["index"],
            "rgb_file": frame["rgb_path"].name,
            "gt_file": frame["gt_path"].name,
            "abs_timestamp_diff_ms": frame["abs_delta_ms"],
            **m,
            "target_frame_log_depth_loss": target_loss if i == 2 else "",
        })
    return pred_np, metric_rows, target_loss, target_valid


def clone_state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def state_delta_l2(module: torch.nn.Module, initial: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for name, param in module.state_dict().items():
        ref = initial[name].to(device=param.device, dtype=param.dtype)
        total += float((param.detach() - ref).float().square().sum().item())
    return math.sqrt(total)


def train_mode(
    model: torch.nn.Module,
    context: dict,
    mode: str,
    steps: int,
    lr: float,
    out_dir: Path,
    initial_head_state: dict[str, torch.Tensor],
) -> tuple[list[dict], dict, float, float, list[np.ndarray], list[dict]]:
    head = model.pixel_decoder.persistent_feature_repair
    if head is None:
        raise RuntimeError(f"{mode}: 修复头不存在")
    head.load_state_dict(initial_head_state, strict=True)
    head.train()
    trainable = [p for p in head.parameters() if p.requires_grad]
    expected = sum(p.numel() for p in head.parameters())
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if expected != 170496:
        raise RuntimeError(f"修复头参数量与已审计版本不一致：expected 170496, actual {expected}")
    if not trainable or actual != expected:
        raise RuntimeError(f"训练参数泄漏或为空：head={expected}, all_trainable={actual}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    curve_rows = []
    initial_params = clone_state_cpu(head)
    initial_preds, initial_metrics, initial_loss, initial_valid = evaluate_clip(
        model, context, mode, "before_training"
    )
    print(
        f"[{mode}] before: target_log_loss={initial_loss:.6f}, "
        f"AbsRel(frame {context['frames'][-1]['index']})={initial_metrics[-1]['AbsRel']:.6f}, "
        f"valid_pixels={initial_valid}", flush=True
    )

    for step in range(1, steps + 1):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        preds = forward_clip(model, context)
        loss, valid_count = masked_log_depth_loss(preds[-1], context["frames"][-1]["gt"])
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"{mode} step {step}: loss 非有限")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in trainable if p.grad is not None]
        if not grads:
            raise RuntimeError(f"{mode} step {step}: 修复头没有梯度")
        finite = all(torch.isfinite(g).all().item() for g in grads)
        nonzero_tensors = sum(bool(torch.count_nonzero(g).item()) for g in grads)
        grad_l1 = sum(float(g.detach().abs().sum().item()) for g in grads)
        if not finite:
            raise RuntimeError(f"{mode} step {step}: 梯度出现 NaN/Inf")
        if nonzero_tensors == 0 or grad_l1 <= 0:
            raise RuntimeError(f"{mode} step {step}: 梯度全零")
        # Clip only the trainable head to protect the short prototype run.
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0).item())
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        param_delta = state_delta_l2(head, initial_params)
        row = {
            "mode": mode,
            "step": step,
            "target_log_depth_loss_before_update": float(loss.detach().item()),
            "valid_supervision_pixels": valid_count,
            "grad_tensors": len(grads),
            "nonzero_grad_tensors": nonzero_tensors,
            "grad_l1_unclipped": grad_l1,
            "grad_norm_before_clip": grad_norm,
            "param_delta_l2_from_init": param_delta,
            "step_wall_sec": elapsed,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "scale": float(scaler.get_scale()),
        }
        curve_rows.append(row)
        if step == 1 or step % 10 == 0 or step == steps:
            print(
                f"[{mode}] step={step:03d}/{steps} loss={row['target_log_depth_loss_before_update']:.6f} "
                f"grad_l1={grad_l1:.4g} delta={param_delta:.4g} "
                f"sec={elapsed:.2f} peak={row['cuda_peak_allocated_gib']:.2f}GiB",
                flush=True,
            )
        del preds, loss, grads

    final_preds, final_metrics, final_loss, final_valid = evaluate_clip(
        model, context, mode, "after_training"
    )
    final_delta = state_delta_l2(head, initial_params)
    if final_delta <= 0:
        raise RuntimeError(f"{mode}: optimizer 执行后参数未发生变化")

    checkpoint_path = out_dir / f"gt2_{mode}_repair_head.pt"
    torch.save({
        "experiment": "GT2 fixed-clip real-GT repair-head training",
        "mode": mode,
        "step_count": steps,
        "learning_rate": lr,
        "seed": int(os.environ.get("GT2_SEED", "20261024")),
        "clip_indices": [x["index"] for x in context["frames"]],
        "target_index": context["frames"][-1]["index"],
        "depth_scale": DEPTH_SCALE,
        "loss": "masked robust absolute log-depth error: mean(sqrt((log(pred)-log(gt))^2 + 1e-6))",
        "repair_head_state_dict": clone_state_cpu(head),
        "initial_head_state_sha256": hashlib.sha256(
            b"".join(v.contiguous().numpy().tobytes() for _, v in sorted(initial_params.items()))
        ).hexdigest(),
        "pretrained_weights_sha256": sha256_file(Path(os.environ["VDEEP_SNAPSHOT"]) / "model.safetensors"),
        "pair_csv_sha256": sha256_file(Path(os.environ["GT2_PAIR_CSV"])),
    }, checkpoint_path)
    # Check checkpoint can be loaded back into the actual registered head.
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    head.load_state_dict(loaded["repair_head_state_dict"], strict=True)
    reload_ok = all(
        torch.equal(v.detach().cpu(), loaded["repair_head_state_dict"][k])
        for k, v in head.state_dict().items()
    )
    if not reload_ok:
        raise RuntimeError(f"{mode}: 修复头检查点保存/重载不一致")

    print(
        f"[{mode}] after: target_log_loss={final_loss:.6f}, "
        f"AbsRel={final_metrics[-1]['AbsRel']:.6f}, "
        f"param_delta_l2={final_delta:.6g}, checkpoint_reload=PASS",
        flush=True,
    )
    return curve_rows, {
        "mode": mode,
        "steps": steps,
        "lr": lr,
        "initial_target_log_loss": initial_loss,
        "final_target_log_loss": final_loss,
        "initial_target_absrel": initial_metrics[-1]["AbsRel"],
        "final_target_absrel": final_metrics[-1]["AbsRel"],
        "initial_target_rmse": initial_metrics[-1]["RMSE"],
        "final_target_rmse": final_metrics[-1]["RMSE"],
        "initial_target_delta1": initial_metrics[-1]["delta1"],
        "final_target_delta1": final_metrics[-1]["delta1"],
        "initial_valid_supervision_pixels": initial_valid,
        "final_valid_supervision_pixels": final_valid,
        "trainable_parameter_count": expected,
        "final_param_delta_l2": final_delta,
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload": "PASS",
        "fixed_clip_fit_improved": bool(final_loss < initial_loss),
        "generalization_claim": False,
    }, initial_loss, final_loss, final_preds, initial_metrics + final_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="GT2 C/D 100-300 step fixed-clip training")
    parser.add_argument("--pair-csv", default="outputs/gt1a1_validation/gt1a1_pairs.csv")
    parser.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    parser.add_argument("--out-dir", default="outputs/gt2_validation")
    parser.add_argument("--indices", default="8,9,10")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-time-diff-ms", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()

    if args.steps < 1 or args.steps > 300:
        parser.error("GT2 首轮 steps 必须在 1 到 300 之间；默认 100。")
    if args.lr <= 0:
        parser.error("--lr 必须大于 0")
    if not torch.cuda.is_available():
        raise RuntimeError("GT2 训练要求 CUDA；当前 torch.cuda.is_available() 为 False")
    if not args.snapshot:
        raise RuntimeError("未设置 --snapshot 或 VDEEP_SNAPSHOT")

    pair_csv = Path(args.pair_csv).resolve()
    snapshot = Path(args.snapshot).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
    if len(indices) != 3:
        raise ValueError("--indices 必须恰好包含 3 个索引，例如 8,9,10")

    os.environ["GT2_PAIR_CSV"] = str(pair_csv)
    os.environ["GT2_SEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)

    selected = read_pair_rows(pair_csv, indices, args.max_time_diff_ms)
    weights_path = snapshot / "model.safetensors"
    config_path = snapshot / "config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"快照缺少 config.json/model.safetensors：{snapshot}")

    print("GT2: fixed-clip real-GT repair-head training", flush=True)
    print("device:", torch.cuda.get_device_name(0), flush=True)
    print("clip indices:", indices, flush=True)
    print("pair CSV:", pair_csv, flush=True)
    print("snapshot:", snapshot, flush=True)
    print("steps per mode:", args.steps, "lr:", args.lr, flush=True)
    print("GT unit: uint16 / 5000 meters; target: final-frame camera z-depth", flush=True)

    # Save experiment protocol/config before expensive model work.
    exp_cfg = {
        "experiment": "GT2",
        "description": "100-300 step fixed-clip real-GT optimizability test",
        "clip_indices": indices,
        "target_index": indices[-1],
        "pair_csv": str(pair_csv),
        "pair_csv_sha256": sha256_file(pair_csv),
        "snapshot": str(snapshot),
        "pretrained_weights_sha256": sha256_file(weights_path),
        "depth_scale": DEPTH_SCALE,
        "max_time_diff_ms": args.max_time_diff_ms,
        "loss": "masked robust absolute log-depth error",
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "steps_per_mode": args.steps,
        "modes": ["C:pre_readout", "D:post_readout"],
        "seed": args.seed,
        "amp": "CUDA autocast float16 + GradScaler",
        "main_model_frozen": True,
        "only_repair_head_trainable": True,
        "generalization_claim": False,
    }
    (out_dir / "gt2_config.json").write_text(json.dumps(exp_cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(pair_csv, out_dir / "gt2_gt1a1_pairs_used.csv")
    shutil.copy2(Path(__file__).resolve(), out_dir / Path(__file__).name)

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # A baseline: same exact three-frame clip and GT1-A1 pairs, no repair head.
    print("\n========== A: same-clip frozen baseline =========", flush=True)
    model_a = load_model(snapshot, "off", device)
    context_a = build_clip_context(model_a, selected, device)
    a_preds, a_rows, a_loss, a_valid = evaluate_clip(model_a, context_a, "A", "repair_off")
    print(
        f"[A] target log loss={a_loss:.6f}, AbsRel={a_rows[-1]['AbsRel']:.6f}, "
        f"RMSE={a_rows[-1]['RMSE']:.6f}, delta1={a_rows[-1]['delta1']:.6f}", flush=True
    )
    all_metric_rows = list(a_rows)
    del context_a, model_a
    torch.cuda.empty_cache()

    # C first: record the initial head and reuse its exact weights for D.
    summaries = []
    curves = []
    trained_preds = {}
    initial_preds_by_mode = {}
    initial_metrics_by_mode = {}
    common_initial_head = None

    for mode in ("pre_readout", "post_readout"):
        short = "C" if mode == "pre_readout" else "D"
        print(f"\n========== {short}: {mode} =========", flush=True)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = load_model(snapshot, mode, device)
        head = model.pixel_decoder.persistent_feature_repair
        if common_initial_head is None:
            common_initial_head = clone_state_cpu(head)
            torch.save(common_initial_head, out_dir / "gt2_initial_repair_head.pt")
        else:
            head.load_state_dict(common_initial_head, strict=True)

        context = build_clip_context(model, selected, device)
        before_preds, before_rows, before_loss, before_valid = evaluate_clip(
            model, context, short, "before_training"
        )
        initial_preds_by_mode[short] = before_preds
        initial_metrics_by_mode[short] = before_rows
        all_metric_rows.extend(before_rows)
        print(
            f"[{short}] initial: target log loss={before_loss:.6f}, "
            f"AbsRel={before_rows[-1]['AbsRel']:.6f}", flush=True
        )

        curve_rows, summary, _initial_loss, _final_loss, after_preds, returned_metric_rows = train_mode(
            model, context, short, args.steps, args.lr, out_dir, common_initial_head
        )
        curves.extend(curve_rows)
        summaries.append(summary)
        trained_preds[short] = after_preds
        # train_mode returns before+after rows; before rows are already included above.
        after_rows = [r for r in returned_metric_rows if r.get("stage") == "after_training"]
        all_metric_rows.extend(after_rows)
        del context, model
        torch.cuda.empty_cache()

    # Regression check: zero-initialized C/D should initially match A on this clip.
    initial_diff = {}
    for short in ("C", "D"):
        diffs = []
        for p_a, p_m in zip(a_preds, initial_preds_by_mode[short]):
            diffs.append(float(np.max(np.abs(p_a.astype(np.float32) - p_m.astype(np.float32)))))
        initial_diff[short] = {"max_abs_depth_diff_m_by_frame": diffs, "max_abs_depth_diff_m": max(diffs)}
        print(f"[{short}] init-vs-A max abs depth diff: {max(diffs):.8g} m", flush=True)
        if max(diffs) > 1e-4:
            raise RuntimeError(f"{short} 零初始化回归超限：max abs diff={max(diffs)} m")

    write_csv(out_dir / "gt2_training_curve.csv", curves)
    write_csv(out_dir / "gt2_clip_metrics.csv", all_metric_rows)

    summary = {
        "experiment": "GT2 fixed-clip real-GT repair-head training",
        "engineering_status": "PASS",
        "scope": "fixed-clip optimizability only; no generalization claim",
        "sequence": "Bonn Crowd2",
        "clip_indices": indices,
        "target_index": indices[-1],
        "pair_csv": str(pair_csv),
        "pair_csv_sha256": sha256_file(pair_csv),
        "pretrained_weights_sha256": sha256_file(weights_path),
        "source_sha256": {
            str(Path("velodepth/models/decoder_velo.py")): sha256_file(Path("velodepth/models/decoder_velo.py")),
            str(Path("velodepth/models/persistent_feature_repair.py")): sha256_file(Path("velodepth/models/persistent_feature_repair.py")),
            str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        },
        "depth_scale": DEPTH_SCALE,
        "gpu": torch.cuda.get_device_name(0),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "peak_cuda_allocated_gib_over_run": torch.cuda.max_memory_allocated(device) / 1024**3,
        "steps_per_mode": args.steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "initial_C_D_max_abs_depth_diff_m": initial_diff,
        "A_target_frame_metrics": {k: a_rows[-1][k] for k in ("AbsRel", "RMSE", "delta1", "valid_ratio")},
        "modes": summaries,
        "interpretation": "A lower fixed-clip loss is only evidence of optimizability; it is not evidence of generalization or method necessity.",
    }
    (out_dir / "gt2_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# GT2 实验报告：固定片段真实 GT 修复头训练",
        "",
        "## 1. 实验目的与范围",
        "",
        "验证 C（读出前修复）和 D（读出后修复持久状态）是否能通过真实 Bonn Crowd2 深度 GT 执行稳定的修复头参数更新。",
        "本实验仅为固定三帧片段可优化性测试，不是跨片段泛化或方法有效性结论。",
        "",
        "## 2. 协议",
        "",
        f"- 全局 RGB 索引：{indices}",
        f"- GT 配对：复用 GT1-A1 一对一配对清单 `{pair_csv}`",
        f"- 最大 RGB-GT 时间差：{args.max_time_diff_ms:.3f} ms",
        f"- GT 单位：原始 uint16 / {DEPTH_SCALE:.0f} 米；零值无效；不额外截断深度范围",
        "- 预测：VeloDepth 相机坐标 z-depth，不是三维半径/距离",
        "- 监督：第三帧有效像素的 masked robust log-depth loss",
        "- 冻结：预训练模型所有参数；只训练 170,496 参数修复头（若当前配置参数量一致）",
        f"- 优化：AdamW，lr={args.lr:g}，weight_decay=1e-4，梯度裁剪范数=1.0",
        f"- 训练预算：C、D 各 {args.steps} 步；CUDA autocast FP16 + GradScaler",
        "",
        "## 3. 同片段 A/C/D 指标",
        "",
        "详细逐帧数据见 `gt2_clip_metrics.csv`；训练曲线见 `gt2_training_curve.csv`。",
        "",
        "| 组别 | 训练前第三帧 AbsRel | 训练后第三帧 AbsRel | 训练前 log-depth loss | 训练后 log-depth loss | 参数变化 L2 | 检查点重载 |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| A（repair off） | — | {a_rows[-1]['AbsRel']:.6f} | — | {a_loss:.6f} | 0 | 不适用 |",
    ]
    for s in summaries:
        report.append(
            f"| {s['mode']} | {s['initial_target_absrel']:.6f} | {s['final_target_absrel']:.6f} | "
            f"{s['initial_target_log_loss']:.6f} | {s['final_target_log_loss']:.6f} | "
            f"{s['final_param_delta_l2']:.6g} | {s['checkpoint_reload']} |"
        )
    report += [
        "",
        "## 4. 工程验收",
        "",
        "- 原始预训练权重：除新增修复头外严格匹配。",
        "- C/D 零初始化：初始预测与 A 的最大深度差必须不超过 1e-4 m。",
        "- 优化器：只注册修复头参数。",
        "- 每一步：检查 loss、梯度有限性、非零梯度、参数变化和耗时。",
        "- 检查点：保存后重新载入修复头，并核验张量一致。",
        "",
        "## 5. 结果判读",
        "",
    ]
    for s in summaries:
        if s["fixed_clip_fit_improved"]:
            report.append(f"- {s['mode']}：固定片段训练后 log-depth loss 下降，支持该模式在此片段上可优化。")
        else:
            report.append(f"- {s['mode']}：固定片段训练后 log-depth loss 未下降；先排查学习率、梯度与状态路径，不扩大预算。")
    report += [
        "- 若 AbsRel/RMSE 没有同步改善，即使训练 loss 下降，也不能称为几何质量改善。",
        "- C/D 只在固定三帧上训练与评估，不能把训练后指标当成验证集或独立测试集。",
        "- D 的必要性必须在后续留出状态、C/B 对照中验证；GT2 本身不回答 D 是否优于 C/B。",
        "",
        "## 6. 文件清单",
        "",
        "- `gt2_config.json`：固定实验协议与哈希",
        "- `gt2_training_curve.csv`：逐步 loss、梯度、参数变化、耗时与显存",
        "- `gt2_clip_metrics.csv`：A/C/D 训练前后逐帧指标",
        "- `gt2_summary.json`：汇总与解释边界",
        "- `gt2_C_repair_head.pt`、`gt2_D_repair_head.pt`：训练后修复头检查点",
        "- `gt2_initial_repair_head.pt`：C/D 共用初始修复头权重",
        "- `gt2_gt1a1_pairs_used.csv`：本轮固定使用的配对清单副本",
        "",
        "## 7. 决策",
        "",
        "工程 PASS 仅代表真实 GT 更新、梯度、参数保存/重载链路通过。只有在训练 loss 下降且 GT 指标有合理变化时，才进入 GT3 的早期/晚期状态验证；若训练 loss 不下降，先定位实现，不租更大 GPU。",
        "",
    ]
    (out_dir / "GT2_report.md").write_text("\n".join(report), encoding="utf-8")

    # Bundle outputs with stdlib zipfile (does not require system zip/unzip).
    archive = out_dir / "GT2_review.zip"
    included = [
        "GT2_report.md", "gt2_config.json", "gt2_training_curve.csv",
        "gt2_clip_metrics.csv", "gt2_summary.json", "gt2_C_repair_head.pt",
        "gt2_D_repair_head.pt", "gt2_initial_repair_head.pt",
        "gt2_gt1a1_pairs_used.csv", Path(__file__).name, "gt2_run.log",
    ]
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in included:
            p = out_dir / name
            if p.is_file():
                z.write(p, arcname=f"outputs/gt2_validation/{name}")
    with zipfile.ZipFile(archive, "r") as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"GT2 ZIP 完整性失败：{bad}")

    print("\n========== GT2 SUMMARY =========", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print("报告：", out_dir / "GT2_report.md", flush=True)
    print("结果包：", archive, flush=True)
    print("GT2 TRAINING RUN: PASS (engineering checks; not a generalization claim)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
