#!/usr/bin/env python3
"""GT4-A2 Experiment 1: multi-state, four-future-frame repair-head training.

Fixed protocol:
- Crowd2 A trajectory with a keyframe refresh at RGB indices 0,32,64,...
- First 24 complete 32-frame cycles; one anchor per cycle, offsets 4,12,20 repeated.
- Cache the A state after anchor-1; unroll anchor plus four future frames.
- Supervise the four future frames with equal-weight valid-pixel robust log-depth loss.
- Train C (pre_readout) and D (post_readout) for 1000 steps each, LR 1e-4,
  identical initialization and deterministic shared sample order; frozen backbone.
- Use final step only; evaluate A/C/D continuously on full Crowd3 and Balloon2,
  refreshing keyframes every 32 original RGB frames and never resetting every 3 frames.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))
import gt2_fixed_clip_repair_train as gt2  # noqa: E402

DEPTH_SCALE = 5000.0
MAX_TIME_DIFF_MS = 20.0
REFRESH_INTERVAL = 32
ANCHOR_OFFSETS = (4, 12, 20)
N_TRAIN_CYCLES = 24
STEPS_PER_MODE = 1000
LEARNING_RATE = 1e-4
SEED = 20261024
MODES = {"C": "pre_readout", "D": "post_readout"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(path: Path, *, training: bool) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"配对清单不存在：{path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"配对清单为空：{path}")
    rows.sort(key=lambda r: int(r["frame_index_zero_based"]))
    indices = [int(r["frame_index_zero_based"]) for r in rows]
    if indices != list(range(len(rows))):
        raise RuntimeError(
            f"{path}: RGB 索引必须完整、连续且从 0 开始；"
            f"首尾={indices[0]}..{indices[-1]}, 行数={len(rows)}"
        )
    seen_gt = set()
    prev_ts = None
    for i, row in enumerate(rows):
        row["_index"] = i
        rgb = Path(row.get("rgb_path", ""))
        if not rgb.is_file():
            raise FileNotFoundError(f"RGB 缺失，停止：index={i}, path={rgb}")
        try:
            ts = float(row["rgb_timestamp"])
        except Exception as e:
            raise RuntimeError(f"RGB 时间戳无效：index={i}, row={row}") from e
        if prev_ts is not None and ts <= prev_ts:
            raise RuntimeError(f"RGB 时间戳非严格递增：index={i}")
        prev_ts = ts
        status = str(row.get("gt_match_status", "")).strip().upper()
        depth_raw = str(row.get("depth_path", "")).strip()
        if status == "MATCHED":
            if not depth_raw:
                raise RuntimeError(f"MATCHED 行缺少 depth_path：index={i}")
            depth = Path(depth_raw)
            if not depth.is_file():
                raise FileNotFoundError(f"MATCHED GT 文件缺失：index={i}, path={depth}")
            if str(row.get("gt_reused", "False")).strip().lower() in {"true", "1", "yes"}:
                raise RuntimeError(f"配对清单标记 GT 被重复使用：index={i}")
            dt = float(row.get("abs_delta_ms") or "nan")
            if not np.isfinite(dt) or dt > MAX_TIME_DIFF_MS:
                raise RuntimeError(f"RGB-GT 时间差超限：index={i}, dt={dt}")
            resolved = str(depth.resolve())
            if resolved in seen_gt:
                raise RuntimeError(f"GT 重复分配：index={i}, path={resolved}")
            seen_gt.add(resolved)
        elif training:
            raise RuntimeError(f"训练清单存在未匹配 GT：index={i}, status={status}")
    return rows


def build_anchor_manifest(rows: list[dict]) -> list[dict]:
    anchors = []
    all_target_gt = set()
    for cycle in range(N_TRAIN_CYCLES):
        cycle_start = cycle * REFRESH_INTERVAL
        cycle_end = cycle_start + REFRESH_INTERVAL - 1
        if cycle_end >= len(rows):
            raise RuntimeError(
                f"Crowd2 不足 {N_TRAIN_CYCLES} 个完整刷新周期："
                f"周期 {cycle} 需要索引至 {cycle_end}，总帧数={len(rows)}"
            )
        offset = ANCHOR_OFFSETS[cycle % len(ANCHOR_OFFSETS)]
        anchor = cycle_start + offset
        future = list(range(anchor + 1, anchor + 5))
        if future[-1] > cycle_end:
            raise RuntimeError(f"锚点未来监督越过刷新周期：cycle={cycle}, future={future}")
        for idx in future:
            row = rows[idx]
            if str(row.get("gt_match_status", "")).strip().upper() != "MATCHED":
                raise RuntimeError(
                    f"固定训练窗口缺少 GT，不能悄悄跳过：cycle={cycle}, anchor={anchor}, target={idx}"
                )
            p = str(Path(row["depth_path"]).resolve())
            if p in all_target_gt:
                raise RuntimeError(f"24 个窗口的监督 GT 不唯一：index={idx}, path={p}")
            all_target_gt.add(p)
        anchors.append({
            "sample_id": cycle,
            "refresh_cycle": cycle,
            "cycle_start_index": cycle_start,
            "cycle_end_index": cycle_end,
            "anchor_offset_after_keyframe": offset,
            "anchor_index": anchor,
            "cached_state_index": anchor - 1,
            "supervised_future_indices": ",".join(map(str, future)),
            "rgb_paths_exist_for_complete_cycle": True,
            "all_four_future_gt_valid": True,
        })
    if len(anchors) != N_TRAIN_CYCLES:
        raise RuntimeError(f"锚点数量错误：{len(anchors)}")
    return anchors


def shape_info_for(model, first_rgb_path: Path, device: torch.device) -> dict:
    rgb = gt2.load_rgb(first_rgb_path)
    original_h, original_w = rgb.shape[:2]
    ratio_bounds = model.shape_constraints["ratio_bounds"]
    pixels_min = float(model.shape_constraints["pixels_min"])
    pixels_max = float(model.shape_constraints["pixels_max"])
    interval = (pixels_max - pixels_min) / 10.0
    pixels_bounds = (pixels_min, pixels_min + interval)
    paddings, (padded_h, padded_w) = gt2.get_paddings((original_h, original_w), ratio_bounds)
    _resize_factor, (new_h, new_w) = gt2.get_resize_factor((padded_h, padded_w), pixels_bounds)
    mean = torch.tensor(gt2.IMAGENET_DATASET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(gt2.IMAGENET_DATASET_STD, device=device).view(1, 3, 1, 1)
    return {
        "original_shape": (original_h, original_w),
        "paddings": paddings,
        "padded_shape": (padded_h, padded_w),
        "network_shape": (new_h, new_w),
        "mean": mean,
        "std": std,
        "validity_mask": torch.ones((1, 1, new_h, new_w), device=device, dtype=torch.bool),
        "image_paddings": torch.zeros((1, 4), device=device, dtype=torch.long),
        "depth_paddings": torch.zeros((1, 4), device=device, dtype=torch.long),
    }


def load_frame(row: dict, shape: dict, device: torch.device, *, need_gt: bool) -> dict:
    idx = int(row["frame_index_zero_based"])
    rgb_path = Path(row["rgb_path"])
    rgb_np = gt2.load_rgb(rgb_path)
    if rgb_np.shape[:2] != shape["original_shape"]:
        raise RuntimeError(f"RGB 尺寸在序列中发生变化：index={idx}, shape={rgb_np.shape}")
    img, frame_u8 = gt2.preprocess_rgb(
        rgb_np, device, shape["paddings"], shape["network_shape"], shape["mean"], shape["std"]
    )
    item = {
        "index": idx,
        "rgb_path": rgb_path,
        "rgb_file": rgb_path.name,
        "rgb_timestamp": float(row["rgb_timestamp"]),
        "img": img,
        "frame_u8": frame_u8,
        "gt_np": None,
        "gt": None,
        "gt_path": None,
        "abs_delta_ms": row.get("abs_delta_ms", ""),
        "gt_match_status": row.get("gt_match_status", ""),
    }
    if need_gt:
        status = str(row.get("gt_match_status", "")).strip().upper()
        if status != "MATCHED":
            raise RuntimeError(f"需要监督/评测 GT 但该帧未匹配：index={idx}, status={status}")
        gt_path = Path(row["depth_path"])
        raw = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"无法读取 GT：index={idx}, path={gt_path}")
        if raw.dtype != np.uint16 or raw.ndim != 2:
            raise RuntimeError(f"GT 格式不符合 uint16 单通道：index={idx}, dtype={raw.dtype}, shape={raw.shape}")
        if raw.shape != shape["original_shape"]:
            raise RuntimeError(f"GT/RGB 尺寸不一致：index={idx}, GT={raw.shape}, RGB={shape['original_shape']}")
        gt_np = raw.astype(np.float32) / DEPTH_SCALE
        item["gt_np"] = gt_np
        item["gt"] = torch.from_numpy(gt_np).to(device)
        item["gt_path"] = gt_path
        item["abs_delta_ms"] = float(row.get("abs_delta_ms") or 0.0)
    return item


def keyframe_state(model, frame: dict, shape: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    base_inputs = {
        "image": frame["img"],
        "image_original": frame["img"],
        "validity_mask": shape["validity_mask"],
    }
    metas = [{"T": 1, "B": 1}]
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        features, tokens = model.pixel_encoder(base_inputs["image"])
        base_inputs["features"] = [
            model.stacking_fn(features[i:j]).contiguous()
            for i, j in model.slices_encoder_range
        ]
        base_inputs["tokens"] = [
            model.stacking_fn(tokens[i:j]).contiguous()
            for i, j in model.slices_encoder_range
        ]
        model.pixel_decoder.set_shapes(base_inputs, metas)
        rays, radius, confidence, lateral = model.pixel_decoder.forward_keyframe(base_inputs, metas)
    rays_map = rays.reshape(1, *shape["network_shape"], 3).permute(0, 3, 1, 2).contiguous().detach()
    return rays_map, radius.detach(), confidence.detach(), [x.detach() for x in lateral]


def transition_static(model, prev_frame: dict, curr_frame: dict, shape: dict, dis) -> dict:
    curr_rgb = curr_frame["frame_u8"].permute(1, 2, 0).numpy()
    prev_rgb = prev_frame["frame_u8"].permute(1, 2, 0).numpy()
    flow_np = dis.calc(
        cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY), None
    )
    flow_px = torch.from_numpy(flow_np).to(device=curr_frame["img"].device).permute(2, 0, 1).unsqueeze(0)
    flow_norm, grid_sample = model._normalize_and_grid_from_flow(
        flow_px, curr_frame["img"], shape["image_paddings"], shape["depth_paddings"]
    )
    image_residual = gt2.rgb2luma(curr_frame["img"]).unsqueeze(1) - F.grid_sample(
        gt2.rgb2luma(prev_frame["img"]).unsqueeze(1), grid_sample, mode="bilinear"
    )
    del curr_rgb, prev_rgb, flow_np, flow_px
    return {
        "flow_norm": flow_norm,
        "grid_sample": grid_sample,
        "image_residual": image_residual,
        "validity_mask": shape["validity_mask"],
    }


def one_fast_step(model, prev_frame: dict, curr_frame: dict, static: dict,
                  radius: torch.Tensor, confidence: torch.Tensor,
                  features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    context = {"frames": [prev_frame, curr_frame], "fast_static": [static]}
    radius, confidence, features = gt2.fast_step(model, context, 1, radius, features)
    return radius, confidence, features


def state_to_cpu(radius, confidence, features, rays_map) -> dict:
    return {
        "radius": radius.detach().cpu().clone(),
        "confidence": confidence.detach().cpu().clone(),
        "features": [x.detach().cpu().clone() for x in features],
        "rays_map": rays_map.detach().cpu().clone(),
    }


def precompute_a_states(model, rows: list[dict], anchors: list[dict], shape: dict,
                        device: torch.device) -> dict[int, dict]:
    anchor_set = {int(a["anchor_index"]) for a in anchors}
    max_prev_idx = max(anchor_set) - 1
    cache = {}
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev_frame = None
    radius = confidence = features = rays_map = None
    print(f"[A-cache] 连续运行 A：frame 0..{max_prev_idx}，refresh={REFRESH_INTERVAL}", flush=True)
    for idx in range(max_prev_idx + 1):
        frame = load_frame(rows[idx], shape, device, need_gt=False)
        if idx % REFRESH_INTERVAL == 0:
            rays_map, radius, confidence, features = keyframe_state(model, frame, shape)
        else:
            static = transition_static(model, prev_frame, frame, shape, dis)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                radius, confidence, features = one_fast_step(
                    model, prev_frame, frame, static, radius, confidence, features
                )
        if idx + 1 in anchor_set:
            anchor = idx + 1
            cache[anchor] = state_to_cpu(radius, confidence, features, rays_map)
            print(f"[A-cache] cached anchor={anchor} state_at={idx}", flush=True)
        prev_frame = frame
        if idx == 0 or (idx + 1) % 100 == 0 or idx == max_prev_idx:
            print(f"[A-cache] processed {idx + 1}/{max_prev_idx + 1}", flush=True)
    if set(cache) != anchor_set:
        raise RuntimeError(f"A 状态缓存不完整：expected={sorted(anchor_set)}, actual={sorted(cache)}")
    return cache


def build_training_contexts(model_a, rows: list[dict], anchors: list[dict], state_cache: dict,
                            shape: dict, device: torch.device) -> list[dict]:
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    contexts = []
    for a in anchors:
        anchor = int(a["anchor_index"])
        indices = list(range(anchor - 1, anchor + 5))  # prev, anchor, four future frames
        target_indices = set(range(anchor + 1, anchor + 5))
        frames = [load_frame(rows[i], shape, device, need_gt=(i in target_indices)) for i in indices]
        statics = [transition_static(model_a, frames[t - 1], frames[t], shape, dis)
                   for t in range(1, len(frames))]
        state = state_cache[anchor]
        contexts.append({
            "sample_id": int(a["sample_id"]),
            "anchor_index": anchor,
            "refresh_cycle": int(a["refresh_cycle"]),
            "frames": frames,
            "fast_static": statics,
            "initial_state": {
                "radius": state["radius"],
                "confidence": state["confidence"],
                "features": state["features"],
            },
            "rays_map": state["rays_map"].to(device),
            "paddings": shape["paddings"],
            "padded_shape": shape["padded_shape"],
            "network_shape": shape["network_shape"],
        })
        print(f"[windows] ready sample={a['sample_id']} anchor={anchor} targets={sorted(target_indices)}", flush=True)
    return contexts


def move_initial_state(context: dict, device: torch.device):
    s = context["initial_state"]
    return (
        s["radius"].to(device),
        s["confidence"].to(device),
        [x.to(device) for x in s["features"]],
    )


def forward_train_window(model, context: dict, *, collect_frame_losses: bool = False):
    device = context["frames"][0]["img"].device
    radius, confidence, features = move_initial_state(context, device)
    frame_losses = []
    valid_counts = []
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        for t in range(1, 6):
            radius, confidence, features = one_fast_step(
                model, context["frames"][t - 1], context["frames"][t],
                context["fast_static"][t - 1], radius, confidence, features
            )
            if t >= 2:  # anchor frame t=1 is not supervised; t=2..5 are four future frames
                pred = gt2.make_z_depth(
                    radius, context["rays_map"], context["padded_shape"],
                    context["paddings"], model.interpolation_mode
                )
                loss, count = gt2.masked_log_depth_loss(pred, context["frames"][t]["gt"])
                frame_losses.append(loss)
                valid_counts.append(count)
    if len(frame_losses) != 4:
        raise RuntimeError(f"未来监督帧数错误：{len(frame_losses)}")
    mean_loss = torch.stack(frame_losses).mean()
    if collect_frame_losses:
        return mean_loss, [float(x.detach().item()) for x in frame_losses], valid_counts
    return mean_loss, valid_counts


def evaluate_train_objective(model, contexts: list[dict]) -> float:
    losses = []
    model.eval()
    model.pixel_decoder.persistent_feature_repair.eval()
    with torch.no_grad():
        for context in contexts:
            loss, _counts = forward_train_window(model, context)
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def generate_sample_order(n_samples: int, steps: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < steps:
        order.extend(rng.permutation(n_samples).tolist())
    return order[:steps]


def train_one_mode(mode_label: str, mode: str, contexts: list[dict], snapshot: Path,
                   out_dir: Path, initial_head_state: dict[str, torch.Tensor],
                   sample_order: list[int], anchor_rows: list[dict], device: torch.device) -> dict:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = gt2.load_model(snapshot, mode, device)
    head = model.pixel_decoder.persistent_feature_repair
    if head is None:
        raise RuntimeError(f"{mode_label}: 修复头未注册")
    head.load_state_dict(initial_head_state, strict=True)
    model.eval()
    head.train()
    trainable = [p for p in head.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_count = sum(p.numel() for p in head.parameters())
    if head_count != 170496 or total_trainable != head_count:
        raise RuntimeError(f"训练参数不符合冻结协议：head={head_count}, all_trainable={total_trainable}")

    # Warm up model shape metadata; this is a keyframe forward only, its state is discarded.
    _warm = keyframe_state(model, contexts[0]["frames"][0], {
        "validity_mask": torch.ones((1, 1, *contexts[0]["network_shape"]), device=device, dtype=torch.bool),
        "network_shape": contexts[0]["network_shape"],
    })
    del _warm
    model.eval()
    head.train()

    initial_objective = evaluate_train_objective(model, contexts)
    head.train()
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    curve_rows = []
    start_all = time.perf_counter()
    print(f"[{mode_label}] initial mean four-future-frame objective={initial_objective:.7f}", flush=True)

    for step, sample_id in enumerate(sample_order, start=1):
        context = contexts[sample_id]
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            loss, frame_losses, valid_counts = forward_train_window(
                model, context, collect_frame_losses=True
            )
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"{mode_label} step={step}: loss 非有限")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in trainable if p.grad is not None]
        if not grads:
            raise RuntimeError(f"{mode_label} step={step}: 修复头没有梯度")
        if not all(torch.isfinite(g).all().item() for g in grads):
            raise RuntimeError(f"{mode_label} step={step}: 梯度出现 NaN/Inf")
        nonzero = sum(bool(torch.count_nonzero(g).item()) for g in grads)
        grad_l1 = sum(float(g.detach().abs().sum().item()) for g in grads)
        if nonzero == 0 or grad_l1 <= 0:
            raise RuntimeError(f"{mode_label} step={step}: 梯度全零")
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0).item())
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        row = {
            "mode": mode_label,
            "step": step,
            "sample_id": context["sample_id"],
            "refresh_cycle": context["refresh_cycle"],
            "anchor_index": context["anchor_index"],
            "future1_log_depth_loss": frame_losses[0],
            "future2_log_depth_loss": frame_losses[1],
            "future3_log_depth_loss": frame_losses[2],
            "future4_log_depth_loss": frame_losses[3],
            "equal_weight_mean_loss_before_update": float(loss.detach().item()),
            "future1_valid_pixels": valid_counts[0],
            "future2_valid_pixels": valid_counts[1],
            "future3_valid_pixels": valid_counts[2],
            "future4_valid_pixels": valid_counts[3],
            "grad_tensors": len(grads),
            "nonzero_grad_tensors": nonzero,
            "grad_l1_unclipped": grad_l1,
            "grad_norm_before_clip": grad_norm,
            "step_wall_sec": elapsed,
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "amp_scale": float(scaler.get_scale()),
        }
        curve_rows.append(row)
        if step == 1 or step % 10 == 0 or step == STEPS_PER_MODE:
            print(
                f"[{mode_label}] step={step:04d}/{STEPS_PER_MODE} sample={sample_id:02d} "
                f"anchor={context['anchor_index']} loss={row['equal_weight_mean_loss_before_update']:.6f} "
                f"grad_l1={grad_l1:.4g} sec={elapsed:.2f} "
                f"peak={row['cuda_peak_allocated_gib']:.2f}GiB",
                flush=True,
            )
    final_objective = evaluate_train_objective(model, contexts)
    head_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    ckpt_path = out_dir / f"gt4a2_exp1_{mode_label}_final_step1000.pt"
    ckpt = {
        "experiment": "GT4-A2 Experiment 1 multi-state training",
        "mode_label": mode_label,
        "mode": mode,
        "step_count": STEPS_PER_MODE,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "refresh_interval": REFRESH_INTERVAL,
        "anchor_manifest_sha256": sha256_file(out_dir / "gt4a2_exp1_anchor_manifest.csv"),
        "sample_order_sha256": hashlib.sha256(",".join(map(str, sample_order)).encode()).hexdigest(),
        "initial_head_state_sha256": state_sha256(initial_head_state),
        "repair_head_state_dict": head_state,
        "pretrained_weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "training_pair_csv_sha256": sha256_file(Path(os.environ["GT4A2_CROWD2_PAIR_CSV"])),
        "initial_mean_training_objective": initial_objective,
        "final_mean_training_objective": final_objective,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "loss": "equal mean over four future frames; each frame uses valid-pixel robust absolute log-depth loss",
        "backbone_frozen": True,
        "checkpoint_policy": "last step only; no validation-based selection",
    }
    torch.save(ckpt, ckpt_path)
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    head.load_state_dict(loaded["repair_head_state_dict"], strict=True)
    reload_ok = all(torch.equal(v.detach().cpu(), loaded["repair_head_state_dict"][k])
                    for k, v in head.state_dict().items())
    if not reload_ok:
        raise RuntimeError(f"{mode_label}: 最终检查点重载不一致")
    elapsed_all = time.perf_counter() - start_all
    summary = {
        "mode_label": mode_label,
        "mode": mode,
        "steps": STEPS_PER_MODE,
        "learning_rate": LEARNING_RATE,
        "initial_mean_training_objective": initial_objective,
        "final_mean_training_objective": final_objective,
        "training_objective_relative_reduction_pct": (initial_objective-final_objective)/max(initial_objective,1e-12)*100.0,
        "trainable_parameters": head_count,
        "initial_head_state_sha256": state_sha256(initial_head_state),
        "final_head_state_sha256": state_sha256(head_state),
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_reload": "PASS" if reload_ok else "FAIL",
        "elapsed_sec": elapsed_all,
        "mean_step_sec": float(np.mean([r["step_wall_sec"] for r in curve_rows])),
    }
    write_csv(out_dir / f"gt4a2_exp1_{mode_label}_training_curve.csv", curve_rows)
    write_json(out_dir / f"gt4a2_exp1_{mode_label}_training_summary.json", summary)
    print(f"[{mode_label}] FINAL objective={final_objective:.7f}; checkpoint={ckpt_path}; reload={reload_ok}", flush=True)
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return summary


def evaluate_sequence_mode(sequence_name: str, pair_csv: Path, mode_label: str, mode: str | None,
                           checkpoint_path: Path | None, snapshot: Path, device: torch.device,
                           a_keyframes: dict | None = None) -> tuple[list[dict], dict, dict]:
    rows = read_manifest(pair_csv, training=False)
    if mode_label == "A":
        model = gt2.load_model(snapshot, "off", device)
    else:
        model = gt2.load_model(snapshot, mode, device)
        head = model.pixel_decoder.persistent_feature_repair
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("mode_label") != mode_label or ckpt.get("mode") != mode or ckpt.get("step_count") != STEPS_PER_MODE:
            raise RuntimeError(f"{mode_label}: 最终检查点元数据不匹配")
        head.load_state_dict(ckpt["repair_head_state_dict"], strict=True)
        head.eval()
    model.eval()
    shape = shape_info_for(model, Path(rows[0]["rgb_path"]), device)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev_frame = None
    radius = confidence = features = rays_map = None
    frame_rows = []
    keyframe_preds = {}
    keyframe_max_diff = 0.0
    t0 = time.perf_counter()
    print(f"[eval {sequence_name}/{mode_label}] full continuous run: {len(rows)} RGB frames", flush=True)

    with torch.no_grad():
        for idx, row in enumerate(rows):
            matched = str(row.get("gt_match_status", "")).strip().upper() == "MATCHED"
            frame = load_frame(row, shape, device, need_gt=matched)
            if idx % REFRESH_INTERVAL == 0:
                rays_map, radius, confidence, features = keyframe_state(model, frame, shape)
            else:
                static = transition_static(model, prev_frame, frame, shape, dis)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    radius, confidence, features = one_fast_step(
                        model, prev_frame, frame, static, radius, confidence, features
                    )
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                pred_t = gt2.make_z_depth(radius, rays_map, shape["padded_shape"],
                                          shape["paddings"], model.interpolation_mode)
            pred_np = pred_t.detach().float().cpu().numpy()
            is_keyframe = idx % REFRESH_INTERVAL == 0
            if is_keyframe:
                keyframe_preds[idx] = pred_np.copy()
                if mode_label != "A" and a_keyframes is not None:
                    if idx not in a_keyframes:
                        raise RuntimeError(f"{sequence_name}: A 缺少 keyframe {idx}")
                    keyframe_max_diff = max(
                        keyframe_max_diff,
                        float(np.max(np.abs(pred_np.astype(np.float32)-a_keyframes[idx].astype(np.float32))))
                    )
            if matched:
                metrics = gt2.calculate_metrics(frame["gt_np"], pred_np)
                row_out = {
                    "sequence": sequence_name,
                    "mode": mode_label,
                    "frame_index_zero_based": idx,
                    "refresh_cycle": idx // REFRESH_INTERVAL,
                    "frame_offset_in_cycle": idx % REFRESH_INTERVAL,
                    "is_keyframe": str(is_keyframe).lower(),
                    "rgb_file": frame["rgb_file"],
                    "gt_file": Path(row["depth_path"]).name,
                    "abs_timestamp_diff_ms": float(row.get("abs_delta_ms") or 0.0),
                    **metrics,
                }
            else:
                row_out = {
                    "sequence": sequence_name,
                    "mode": mode_label,
                    "frame_index_zero_based": idx,
                    "refresh_cycle": idx // REFRESH_INTERVAL,
                    "frame_offset_in_cycle": idx % REFRESH_INTERVAL,
                    "is_keyframe": str(is_keyframe).lower(),
                    "rgb_file": frame["rgb_file"],
                    "gt_file": "",
                    "abs_timestamp_diff_ms": "",
                    "AbsRel": "", "RMSE": "", "delta1": "", "valid_ratio": "",
                    "gt_valid_ratio": "", "valid_pixels": 0, "total_pixels": int(np.prod(shape["original_shape"])),
                }
            frame_rows.append(row_out)
            prev_frame = frame
            if idx == 0 or (idx + 1) % 100 == 0 or idx == len(rows)-1:
                print(f"[eval {sequence_name}/{mode_label}] {idx+1}/{len(rows)} frames", flush=True)
    elapsed = time.perf_counter() - t0
    matched_rows = [r for r in frame_rows if r["AbsRel"] != ""]
    if not matched_rows:
        raise RuntimeError(f"{sequence_name}/{mode_label}: 无有效 GT 评测帧")
    def means(selected):
        return {m: float(np.mean([float(r[m]) for r in selected])) for m in ("AbsRel", "RMSE", "delta1")}
    summary = {
        "sequence": sequence_name,
        "mode": mode_label,
        "rgb_frames_processed": len(rows),
        "matched_gt_frames_scored": len(matched_rows),
        "keyframes": sum(1 for r in rows if int(r["frame_index_zero_based"]) % REFRESH_INTERVAL == 0),
        "all_matched_frame_macro_metrics": means(matched_rows),
        "propagated_matched_frame_macro_metrics": means([r for r in matched_rows if r["is_keyframe"] == "false"]),
        "elapsed_sec": elapsed,
        "keyframe_max_abs_depth_diff_vs_A_m": keyframe_max_diff if mode_label != "A" else 0.0,
    }
    print(f"[eval {sequence_name}/{mode_label}] complete in {elapsed:.1f}s; matched={len(matched_rows)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return frame_rows, summary, keyframe_preds


def aggregate_results(frame_rows: list[dict], mode_summaries: list[dict]) -> tuple[list[dict], list[dict], dict]:
    seq_summary_rows = []
    cycle_rows = []
    decisions = {}
    for sequence in ("crowd3", "balloon2"):
        by_mode = {}
        for mode in ("A", "C", "D"):
            mr = [r for r in frame_rows if r["sequence"] == sequence and r["mode"] == mode and r["AbsRel"] != ""]
            if not mr:
                raise RuntimeError(f"缺少评测指标：{sequence}/{mode}")
            def mean_metric(rows, metric):
                return float(np.mean([float(r[metric]) for r in rows]))
            all_metrics = {m: mean_metric(mr, m) for m in ("AbsRel", "RMSE", "delta1")}
            prop = [r for r in mr if r["is_keyframe"] == "false"]
            prop_metrics = {m: mean_metric(prop, m) for m in ("AbsRel", "RMSE", "delta1")}
            by_mode[mode] = {"all": all_metrics, "propagated": prop_metrics, "frames": mr}
            seq_summary_rows.append({
                "sequence": sequence,
                "mode": mode,
                "scored_frames_all": len(mr),
                "scored_propagated_frames": len(prop),
                "AbsRel_all": all_metrics["AbsRel"],
                "RMSE_all": all_metrics["RMSE"],
                "delta1_all": all_metrics["delta1"],
                "AbsRel_propagated_only": prop_metrics["AbsRel"],
                "RMSE_propagated_only": prop_metrics["RMSE"],
                "delta1_propagated_only": prop_metrics["delta1"],
                "relative_AbsRel_reduction_vs_A_pct": "" if mode == "A" else (by_mode["A"]["all"]["AbsRel"]-all_metrics["AbsRel"])/max(by_mode["A"]["all"]["AbsRel"],1e-12)*100.0,
                "relative_RMSE_reduction_vs_A_pct": "" if mode == "A" else (by_mode["A"]["all"]["RMSE"]-all_metrics["RMSE"])/max(by_mode["A"]["all"]["RMSE"],1e-12)*100.0,
                "delta1_change_vs_A_percentage_points": "" if mode == "A" else (all_metrics["delta1"]-by_mode["A"]["all"]["delta1"])*100.0,
            })
        # Per-cycle metrics use identical matched frames and include the keyframe for end-to-end cycle scoring.
        cycle_ids = sorted({int(r["refresh_cycle"]) for r in by_mode["A"]["frames"]})
        for cyc in cycle_ids:
            row = {"sequence": sequence, "refresh_cycle": cyc}
            cycle_means = {}
            for mode in ("A", "C", "D"):
                rr = [r for r in by_mode[mode]["frames"] if int(r["refresh_cycle"]) == cyc]
                if not rr:
                    raise RuntimeError(f"周期缺少帧：{sequence} cycle={cyc} mode={mode}")
                cycle_means[mode] = {m: float(np.mean([float(r[m]) for r in rr])) for m in ("AbsRel", "RMSE", "delta1")}
                row[f"{mode}_matched_frames"] = len(rr)
                for metric in ("AbsRel", "RMSE", "delta1"):
                    row[f"{mode}_{metric}"] = cycle_means[mode][metric]
            row["C_absrel_better_than_A"] = str(cycle_means["C"]["AbsRel"] < cycle_means["A"]["AbsRel"]).lower()
            row["D_absrel_better_than_A"] = str(cycle_means["D"]["AbsRel"] < cycle_means["A"]["AbsRel"]).lower()
            cycle_rows.append(row)
        base = by_mode["A"]["all"]
        decisions[sequence] = {}
        for mode in ("C", "D"):
            cand = by_mode[mode]["all"]
            rel_abs = (base["AbsRel"] - cand["AbsRel"]) / max(base["AbsRel"], 1e-12) * 100.0
            delta_change = cand["delta1"] - base["delta1"]
            passed = (rel_abs >= 5.0 and cand["RMSE"] <= base["RMSE"] and delta_change >= -0.005)
            decisions[sequence][mode] = {
                "absrel_reduction_pct": rel_abs,
                "rmse_nonworsening": bool(cand["RMSE"] <= base["RMSE"]),
                "delta1_change": delta_change,
                "delta1_drop_within_0_5_pp": bool(delta_change >= -0.005),
                "pass": bool(passed),
            }
    pass_modes = [m for m in ("C", "D") if all(decisions[s][m]["pass"] for s in ("crowd3", "balloon2"))]
    if pass_modes:
        verdict = "PASS_EXPERIMENT_1_ENTER_EXPERIMENT_2"
        next_action = "C only: experiment 2 centers on C" if pass_modes == ["C"] else ("D only: experiment 2 centers on D" if pass_modes == ["D"] else "Both C and D pass: run experiment 2 for both modes")
    else:
        verdict = "FAIL_EXPERIMENT_1_STOP_CURRENT_UNCONDITIONAL_REPAIR_BUDGET"
        next_action = "C and D both fail the predeclared gate; do not start experiment 2. Modify the unconditional residual-repair mechanism."
    decision = {
        "gate_scope": "all matched frames in each complete continuous sequence, including keyframes; propagated-only metrics are also reported diagnostically",
        "criteria": {
            "AbsRel": "at least 5% relative reduction versus A on both Crowd3 and Balloon2",
            "RMSE": "must not worsen versus A on either sequence",
            "delta1": "absolute decline no more than 0.005 (0.5 percentage points) versus A on either sequence",
        },
        "per_sequence": decisions,
        "passing_modes": pass_modes,
        "verdict": verdict,
        "next_action": next_action,
    }
    return seq_summary_rows, cycle_rows, decision


def preflight(args) -> dict:
    snapshot = Path(args.snapshot).resolve()
    crowd2_csv = Path(args.crowd2_pair_csv).resolve()
    crowd3_csv = Path(args.crowd3_pair_csv).resolve()
    balloon2_csv = Path(args.balloon2_pair_csv).resolve()
    for p in (snapshot / "config.json", snapshot / "model.safetensors", crowd2_csv, crowd3_csv, balloon2_csv,
              ROOT / "velodepth/models/decoder_velo.py", ROOT / "velodepth/models/persistent_feature_repair.py",
              ROOT / "experiments/p1/gt2_fixed_clip_repair_train.py"):
        if not p.is_file():
            raise FileNotFoundError(f"前置文件不存在：{p}")
    train_rows = read_manifest(crowd2_csv, training=True)
    anchors = build_anchor_manifest(train_rows)
    eval_rows = {
        "crowd3": read_manifest(crowd3_csv, training=False),
        "balloon2": read_manifest(balloon2_csv, training=False),
    }
    train_rgb = {str(Path(r["rgb_path"]).resolve()) for r in train_rows}
    train_gt = {str(Path(r["depth_path"]).resolve()) for r in train_rows if str(r.get("depth_path", "")).strip()}
    disjoint = {}
    for seq, rows in eval_rows.items():
        eval_rgb = {str(Path(r["rgb_path"]).resolve()) for r in rows}
        eval_gt = {str(Path(r["depth_path"]).resolve()) for r in rows if str(r.get("depth_path", "")).strip() and str(r.get("gt_match_status", "")).upper() == "MATCHED"}
        overlap = {"rgb": len(train_rgb & eval_rgb), "gt": len(train_gt & eval_gt)}
        if overlap["rgb"] or overlap["gt"]:
            raise RuntimeError(f"训练/测试路径重叠：{seq}: {overlap}")
        disjoint[seq] = overlap
    return {
        "status": "PASS",
        "snapshot": str(snapshot),
        "pretrained_weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "crowd2_pair_csv": str(crowd2_csv),
        "crowd2_pair_csv_sha256": sha256_file(crowd2_csv),
        "crowd2_rgb_rows": len(train_rows),
        "crowd2_training_anchors": anchors,
        "crowd3_pair_csv": str(crowd3_csv),
        "crowd3_rgb_rows": len(eval_rows["crowd3"]),
        "balloon2_pair_csv": str(balloon2_csv),
        "balloon2_rgb_rows": len(eval_rows["balloon2"]),
        "train_test_path_overlap": disjoint,
        "fixed_protocol": {
            "refresh_interval": REFRESH_INTERVAL,
            "anchor_offsets_cycle_repeating": list(ANCHOR_OFFSETS),
            "train_cycles": N_TRAIN_CYCLES,
            "future_frames_per_anchor": 4,
            "steps_per_mode": STEPS_PER_MODE,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "depth_scale": DEPTH_SCALE,
            "max_rgb_gt_delta_ms": MAX_TIME_DIFF_MS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    parser.add_argument("--crowd2-pair-csv", default="outputs/gt1a1_validation/gt1a1_pairs.csv")
    parser.add_argument("--crowd3-pair-csv", default="outputs/gt4_validation/crowd3/gt4_pairs.csv")
    parser.add_argument("--balloon2-pair-csv", default="outputs/gt4_validation/balloon2/gt4_pairs.csv")
    parser.add_argument("--out-dir", default="outputs/gt4a2_exp1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if not args.snapshot:
        raise RuntimeError("未设置 VDEEP_SNAPSHOT")
    if not torch.cuda.is_available() and not args.preflight_only:
        raise RuntimeError("实验 1 训练/评估需要 CUDA；torch.cuda.is_available() 为 False")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pre = preflight(args)
    write_json(out_dir / "gt4a2_exp1_preflight.json", pre)
    print(f"[PASS] preflight: Crowd2 rows={pre['crowd2_rgb_rows']}; anchors={len(pre['crowd2_training_anchors'])}; Crowd3={pre['crowd3_rgb_rows']}; Balloon2={pre['balloon2_rgb_rows']}", flush=True)
    if args.preflight_only:
        print("GT4-A2 Experiment 1 preflight: PASS (no training started)", flush=True)
        return 0
    final_summary_path = out_dir / "gt4a2_exp1_summary.json"
    if final_summary_path.exists():
        raise RuntimeError(f"已存在完成标记 {final_summary_path}；为防止覆盖已完成实验，请先归档旧输出后再运行。")

    snapshot = Path(args.snapshot).resolve()
    crowd2_csv = Path(args.crowd2_pair_csv).resolve()
    crowd3_csv = Path(args.crowd3_pair_csv).resolve()
    balloon2_csv = Path(args.balloon2_pair_csv).resolve()
    anchors = pre["crowd2_training_anchors"]
    train_rows = read_manifest(crowd2_csv, training=True)
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Save the immutable protocol and input manifests before expensive work.
    write_csv(out_dir / "gt4a2_exp1_anchor_manifest.csv", anchors)
    for src, name in ((crowd2_csv, "gt4a2_exp1_crowd2_pairs_used.csv"),
                      (crowd3_csv, "gt4a2_exp1_crowd3_pairs_used.csv"),
                      (balloon2_csv, "gt4a2_exp1_balloon2_pairs_used.csv")):
        (out_dir / name).write_bytes(src.read_bytes())
    sample_order = generate_sample_order(len(anchors), STEPS_PER_MODE, SEED)
    order_rows = [{"step": i+1, "sample_id": sid, "anchor_index": anchors[sid]["anchor_index"],
                   "refresh_cycle": anchors[sid]["refresh_cycle"]} for i, sid in enumerate(sample_order)]
    write_csv(out_dir / "gt4a2_exp1_sample_order.csv", order_rows)
    cfg = {
        "experiment": "GT4-A2 Experiment 1: multi-state training + continuous full-sequence evaluation",
        "protocol": pre["fixed_protocol"],
        "loss": "equal mean of four future-frame masked robust absolute log-depth losses; each frame independently averages valid pixels",
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "backbone_frozen": True,
        "checkpoint_selection": "final step only",
        "evaluation_refresh_interval": REFRESH_INTERVAL,
        "evaluation_reset_every_three_frames": False,
        "evaluation_metrics_primary": "macro average of per-frame metrics over all matched frames, including keyframes",
        "secondary_metrics": "propagated frames only; per-refresh-cycle A/C/D AbsRel comparisons",
        "pair_hashes": {"crowd2": sha256_file(crowd2_csv), "crowd3": sha256_file(crowd3_csv), "balloon2": sha256_file(balloon2_csv)},
        "source_hashes": {str(p.relative_to(ROOT)): sha256_file(p) for p in [
            ROOT / "velodepth/models/decoder_velo.py",
            ROOT / "velodepth/models/persistent_feature_repair.py",
            ROOT / "experiments/p1/gt2_fixed_clip_repair_train.py",
            Path(__file__).resolve(),
        ]},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    }
    write_json(out_dir / "gt4a2_exp1_config.json", cfg)

    print("\n========== 1. Cache A states from sequence start =========", flush=True)
    model_a = gt2.load_model(snapshot, "off", device)
    shape_train = shape_info_for(model_a, Path(train_rows[0]["rgb_path"]), device)
    state_cache = precompute_a_states(model_a, train_rows, anchors, shape_train, device)
    print("\n========== 2. Prepare 24 fixed training windows =========", flush=True)
    contexts = build_training_contexts(model_a, train_rows, anchors, state_cache, shape_train, device)
    del model_a, state_cache
    torch.cuda.empty_cache()

    print("\n========== 3. Construct shared initialization =========", flush=True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model_c = gt2.load_model(snapshot, MODES["C"], device)
    initial_head_state = {k: v.detach().cpu().clone() for k, v in model_c.pixel_decoder.persistent_feature_repair.state_dict().items()}
    initial_hash = state_sha256(initial_head_state)
    torch.save(initial_head_state, out_dir / "gt4a2_exp1_initial_head.pt")
    if sum(p.numel() for p in model_c.pixel_decoder.persistent_feature_repair.parameters()) != 170496:
        raise RuntimeError("修复头参数量与已审计版本 170496 不一致")
    del model_c
    torch.cuda.empty_cache()

    print("\n========== 4. Train C: pre_readout, 1000 steps =========", flush=True)
    c_summary = train_one_mode("C", MODES["C"], contexts, snapshot, out_dir,
                               initial_head_state, sample_order, anchors, device)
    print("\n========== 5. Train D: post_readout, 1000 steps =========", flush=True)
    d_summary = train_one_mode("D", MODES["D"], contexts, snapshot, out_dir,
                               initial_head_state, sample_order, anchors, device)
    if c_summary["initial_head_state_sha256"] != d_summary["initial_head_state_sha256"] or c_summary["initial_head_state_sha256"] != initial_hash:
        raise RuntimeError("C/D 未使用完全相同的初始修复头状态")
    # Release cached windows before full-sequence evaluation.
    del contexts, shape_train
    torch.cuda.empty_cache()

    print("\n========== 6. Continuous full-sequence evaluation =========", flush=True)
    all_frame_rows = []
    eval_summaries = []
    keyframe_checks = []
    checkpoint_paths = {
        "C": out_dir / "gt4a2_exp1_C_final_step1000.pt",
        "D": out_dir / "gt4a2_exp1_D_final_step1000.pt",
    }
    for seq, pair_csv in (("crowd3", crowd3_csv), ("balloon2", balloon2_csv)):
        a_rows_eval, a_summary, a_keyframes = evaluate_sequence_mode(
            seq, pair_csv, "A", None, None, snapshot, device
        )
        all_frame_rows.extend(a_rows_eval)
        eval_summaries.append(a_summary)
        keyframe_checks.append({"sequence": seq, "mode": "A", "max_abs_depth_diff_vs_A_m": 0.0,
                                "pass": True, "keyframes_compared": len(a_keyframes)})
        for label in ("C", "D"):
            rows_eval, summary_eval, _ = evaluate_sequence_mode(
                seq, pair_csv, label, MODES[label], checkpoint_paths[label], snapshot, device,
                a_keyframes=a_keyframes
            )
            all_frame_rows.extend(rows_eval)
            eval_summaries.append(summary_eval)
            diff = float(summary_eval["keyframe_max_abs_depth_diff_vs_A_m"])
            keyframe_checks.append({"sequence": seq, "mode": label, "max_abs_depth_diff_vs_A_m": diff,
                                    "pass": diff <= 1e-4, "keyframes_compared": len(a_keyframes)})
            if diff > 1e-4:
                raise RuntimeError(f"{seq}/{label}: 关键帧输出与 A 不一致，max_diff={diff}")
    write_csv(out_dir / "gt4a2_exp1_full_sequence_frame_metrics.csv", all_frame_rows)
    write_csv(out_dir / "gt4a2_exp1_keyframe_checks.csv", keyframe_checks)
    seq_rows, cycle_rows, decision = aggregate_results(all_frame_rows, eval_summaries)
    write_csv(out_dir / "gt4a2_exp1_sequence_summary.csv", seq_rows)
    write_csv(out_dir / "gt4a2_exp1_cycle_summary.csv", cycle_rows)
    write_json(out_dir / "gt4a2_exp1_decision.json", decision)

    summary = {
        "experiment": "GT4-A2 Experiment 1",
        "engineering_status": "PASS",
        "training": {"C": c_summary, "D": d_summary},
        "evaluation_summaries": eval_summaries,
        "keyframe_checks": keyframe_checks,
        "decision": decision,
        "runtime_peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "preflight": pre,
        "limitations": [
            "Only one fixed training seed was used, as preregistered for this decision experiment.",
            "Crowd2 is the training sequence; Crowd3 and Balloon2 are the held-out evaluation sequences.",
            "No error maps, depth-bin analysis, refresh-interval sweep, Auto comparison, or latency study was run.",
            "Passing this gate is a project development decision, not a CVPR acceptance criterion.",
        ],
    }
    write_json(final_summary_path, summary)

    lines = [
        "# GT4-A2 实验 1：多状态训练与完整序列连续评估",
        "",
        "## 1. 实验目的",
        "",
        "检验增加 Crowd2 真实历史状态和四帧未来监督后，现有 C/D 无条件持久特征修复头是否能在 Crowd3、Balloon2 两条独立序列上同时稳定改善。",
        "本轮没有新增误差图、深度分桶或刷新间隔扫描；没有执行实验 2。",
        "",
        "## 2. 固定协议",
        "",
        f"- 关键帧刷新间隔：每 {REFRESH_INTERVAL} 个 RGB 索引；从索引 0 开始连续运行。",
        f"- 训练状态：A 从 Crowd2 序列开头连续运行，缓存锚点前一帧状态。",
        f"- 训练锚点：前 {N_TRAIN_CYCLES} 个完整刷新周期，每周期 1 个锚点；偏移按 {ANCHOR_OFFSETS} 循环。",
        "- 训练窗口：锚点前一帧状态 → 锚点帧 → 后续 4 帧；只监督锚点后的 4 帧。",
        "- 损失：每个未来帧单独计算有效像素 robust log-depth loss，再对四帧等权平均。",
        f"- C/D：各 {STEPS_PER_MODE} 步，AdamW，lr={LEARNING_RATE:g}，主干冻结，共用完全相同初始头和样本顺序。",
        "- 检查点：第 1000 步最终权重；没有按测试结果挑选中间权重。",
        "- 评估：Crowd3、Balloon2 全序列连续运行，每 32 个原始 RGB 索引刷新关键帧；没有三帧重置。",
        "- 主要准入指标：全序列所有有效配对帧的逐帧宏平均，包含关键帧；另报告仅传播帧指标。",
        "",
        "## 3. 训练结果",
        "",
        "| 模式 | 训练前 24 窗口平均 loss | 第 1000 步后平均 loss | 相对下降 | 平均 step 时间 (s) | 检查点重载 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for s in (c_summary, d_summary):
        lines.append(f"| {s['mode_label']} | {s['initial_mean_training_objective']:.6f} | {s['final_mean_training_objective']:.6f} | {s['training_objective_relative_reduction_pct']:.2f}% | {s['mean_step_sec']:.3f} | {s['checkpoint_reload']} |")
    lines += ["", "## 4. 完整序列指标（准入使用此表）", "",
              "| 序列 | 模式 | AbsRel ↓ | RMSE ↓ (m) | δ1 ↑ | AbsRel 相对下降 | RMSE 不恶化 | δ1 下降≤0.5pp | 单序列通过 |",
              "|---|---|---:|---:|---:|---:|---|---|---|"]
    row_map = {(r["sequence"], r["mode"]): r for r in seq_rows}
    for seq in ("crowd3", "balloon2"):
        for mode_label in ("A", "C", "D"):
            r = row_map[(seq, mode_label)]
            if mode_label == "A":
                lines.append(f"| {seq} | A | {r['AbsRel_all']:.6f} | {r['RMSE_all']:.6f} | {r['delta1_all']:.6f} | — | — | — | — |")
            else:
                gate = decision["per_sequence"][seq][mode_label]
                lines.append(f"| {seq} | {mode_label} | {r['AbsRel_all']:.6f} | {r['RMSE_all']:.6f} | {r['delta1_all']:.6f} | {gate['absrel_reduction_pct']:.2f}% | {'是' if gate['rmse_nonworsening'] else '否'} | {'是' if gate['delta1_drop_within_0_5_pp'] else '否'} | {'PASS' if gate['pass'] else 'FAIL'} |")
    lines += ["", "## 5. 传播帧单独指标（诊断，不替代准入表）", "",
              "| 序列 | 模式 | AbsRel ↓ | RMSE ↓ (m) | δ1 ↑ |",
              "|---|---|---:|---:|---:|"]
    for r in seq_rows:
        lines.append(f"| {r['sequence']} | {r['mode']} | {r['AbsRel_propagated_only']:.6f} | {r['RMSE_propagated_only']:.6f} | {r['delta1_propagated_only']:.6f} |")
    lines += ["", "## 6. 刷新周期胜负", "",
              "`gt4a2_exp1_cycle_summary.csv` 给出每个 32 帧周期 A/C/D 的平均 AbsRel、RMSE、δ1，以及 C/D 是否优于 A。周期均值包含该周期关键帧，并使用三组共同的有效 GT 帧。",
              "", "## 7. 关键帧不变量", "",
              "| 序列 | 模式 | 最大深度差 vs A (m) | 检查 |", "|---|---|---:|---|"]
    for k in keyframe_checks:
        lines.append(f"| {k['sequence']} | {k['mode']} | {k['max_abs_depth_diff_vs_A_m']:.8g} | {'PASS' if k['pass'] else 'FAIL'} |")
    lines += ["", "## 8. 预先声明的最终决策", "",
              f"- 结果：**{decision['verdict']}**",
              f"- 通过模式：{', '.join(decision['passing_modes']) if decision['passing_modes'] else '无'}",
              f"- 下一步：{decision['next_action']}",
              "- 准入线：两条序列的 AbsRel 均至少相对下降 5%；RMSE 不恶化；δ1 绝对下降不超过 0.005。",
              "- 如果 C/D 均未通过，不启动实验 2；按预定规则停止扩大当前无条件修复结构的训练预算。",
              "", "## 9. 输出文件", "",
              "- `gt4a2_exp1_preflight.json`：前置审计与固定锚点清单",
              "- `gt4a2_exp1_anchor_manifest.csv`：24 个锚点和四帧监督清单",
              "- `gt4a2_exp1_sample_order.csv`：C/D 共用的 1000 步样本顺序",
              "- `gt4a2_exp1_C_training_curve.csv`、`gt4a2_exp1_D_training_curve.csv`：逐步训练日志",
              "- `gt4a2_exp1_full_sequence_frame_metrics.csv`：全序列逐帧 A/C/D 指标",
              "- `gt4a2_exp1_cycle_summary.csv`：每个刷新周期的胜负",
              "- `gt4a2_exp1_sequence_summary.csv`：序列级指标和相对变化",
              "- `gt4a2_exp1_decision.json`：准入决策",
              "- `gt4a2_exp1_C_final_step1000.pt`、`gt4a2_exp1_D_final_step1000.pt`：最终检查点",
              "", "## 10. 解释边界", "",
              "训练目标下降仅说明训练窗口可优化；研究结论由预先声明的 Crowd3/Balloon2 完整序列准入线决定。此结果不是统计显著性声明，也不是 CVPR 接收保证。", ""]
    (out_dir / "GT4_A2_Experiment1_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n========== EXPERIMENT 1 DECISION =========", flush=True)
    print(json.dumps(decision, indent=2, ensure_ascii=False), flush=True)
    print("Report:", out_dir / "GT4_A2_Experiment1_report.md", flush=True)
    print("Summary:", final_summary_path, flush=True)
    print("GT4-A2 EXPERIMENT 1: PASS (run complete; research gate reported separately)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
