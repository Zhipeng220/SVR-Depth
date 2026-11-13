#!/usr/bin/env python3
"""GT1: memory-bounded full-sequence native VeloDepth A baseline for Bonn Crowd2.

The streaming loop mirrors VeloDepth.infer/encode_decode_test for the fixed GT0
protocol: resolution_level=0, one initial keyframe, then fast propagation only
(alpha_flow/alpha_luma set to 1e9, beta terms set to 0). It keeps only current
and previous preprocessed frames on device and writes predictions to a memmap.
"""
import argparse
import bisect
import csv
import hashlib
import json
import os
import platform
import sys
import time
import traceback
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
from velodepth.models.decoder_velo import flow2grid_sample
from velodepth.models.encoder_velo import rgb2luma
from velodepth.utils.constants import IMAGENET_DATASET_MEAN, IMAGENET_DATASET_STD

DEPTH_SCALE = 5000.0
MAX_TIME_DIFF_SEC = 0.020


def resolve_path(root: Path, raw_path: str, subdir: str):
    p = Path(raw_path)
    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [root / p, root / subdir / p, root / subdir / p.name, root / p.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def read_index(root: Path, filename: str, subdir: str):
    index_path = root / filename
    if not index_path.is_file():
        raise FileNotFoundError(f"索引不存在：{index_path}")
    records = []
    for line_number, line in enumerate(index_path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            timestamp = float(fields[0])
        except ValueError:
            continue
        records.append({
            "timestamp": timestamp,
            "raw_path": fields[1],
            "path": resolve_path(root, fields[1], subdir),
            "line_number": line_number,
        })
    if not records:
        raise RuntimeError(f"索引没有有效记录：{index_path}")
    return records


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def calculate_metrics(gt, pred):
    valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if not valid.any():
        raise RuntimeError("GT 与预测没有共同有效像素")
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
        "median_gt_m": float(np.median(g)),
        "median_pred_m": float(np.median(p)),
        "gt_min_m": float(np.min(g)),
        "gt_max_m": float(np.max(g)),
        "pred_min_m": float(np.min(p)),
        "pred_max_m": float(np.max(p)),
    }


def prepare_pairs(rgb_records, depth_records, max_delta_sec):
    valid_depths = [r for r in depth_records if r["path"] is not None]
    valid_depths.sort(key=lambda x: x["timestamp"])
    depth_times = [r["timestamp"] for r in valid_depths]
    pairs = []
    for frame_idx, rgb in enumerate(rgb_records):
        chosen = None
        delta = None
        if valid_depths:
            pos = bisect.bisect_left(depth_times, rgb["timestamp"])
            candidates = []
            if pos > 0:
                candidates.append(valid_depths[pos - 1])
            if pos < len(valid_depths):
                candidates.append(valid_depths[pos])
            if candidates:
                chosen = min(candidates, key=lambda d: abs(d["timestamp"] - rgb["timestamp"]))
                delta = abs(chosen["timestamp"] - rgb["timestamp"])
                if delta > max_delta_sec:
                    chosen, delta = None, None
        pairs.append({
            "frame_index_zero_based": frame_idx,
            "rgb_timestamp": rgb["timestamp"],
            "rgb_path": str(rgb["path"]) if rgb["path"] else "",
            "depth_timestamp": chosen["timestamp"] if chosen else "",
            "depth_path": str(chosen["path"]) if chosen else "",
            "signed_delta_ms": ((chosen["timestamp"] - rgb["timestamp"]) * 1000.0) if chosen else "",
            "abs_delta_ms": (delta * 1000.0) if delta is not None else "",
            "gt_match_status": "MATCHED" if chosen else "NO_GT_WITHIN_TOLERANCE",
            "gt_raw_index_line": chosen["line_number"] if chosen else "",
        })
    # Mark whether the same GT file was assigned to multiple RGB frames.
    counts = {}
    for p in pairs:
        if p["depth_path"]:
            counts[p["depth_path"]] = counts.get(p["depth_path"], 0) + 1
    for p in pairs:
        p["gt_reused"] = bool(p["depth_path"] and counts[p["depth_path"]] > 1)
    return pairs, valid_depths


def load_rgb(path: Path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB")).copy()


def preprocess_rgb(rgb_np, device, paddings, new_shape, mean, std):
    x = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device)
    x = TF.normalize(x.float() / 255.0, mean=IMAGENET_DATASET_MEAN, std=IMAGENET_DATASET_STD)
    x = F.pad(x, paddings, value=0.0)
    x = F.interpolate(x, size=new_shape, mode="bilinear", align_corners=False)
    # Match VeloDepth.infer's denormalize -> uint8 path used for DIS flow.
    frame_u8 = ((x * std + mean).clamp(0, 1) * 255.0).byte().squeeze(0).cpu()
    return x, frame_u8


def make_model(snapshot: Path, device):
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    if not config_path.is_file():
        raise FileNotFoundError(f"模型配置不存在：{config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"预训练权重不存在：{weights_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model", {}).get("persistent_feature_repair"):
        raise RuntimeError("预训练配置意外启用了 persistent_feature_repair")
    model = VeloDepth(config)
    model.cpu()
    model.eval()
    decoder = model.pixel_decoder
    if getattr(decoder, "persistent_feature_repair", None) is not None:
        raise RuntimeError("GT1 A 基线错误地注册了修复头")
    if getattr(decoder, "feature_repair_mode", "off") != "off":
        raise RuntimeError("GT1 A 基线的 feature_repair_mode 不是 off")
    state = load_file(str(weights_path), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    model.to(device)
    model.eval()
    model.resolution_level = 0
    model.interpolation_mode = "bilinear"
    # Match the GT0 protocol: one keyframe, no heuristic refreshes.
    model.alpha_flow = 1e9
    model.beta_flow = 0.0
    model.alpha_luma = 1e9
    model.beta_luma = 0.0
    return model, weights_path


def run_stream(model, records, pairs_by_index, out_dir, snapshot, args):
    if not torch.cuda.is_available():
        raise RuntimeError("GT1 当前脚本要求 CUDA，与 GT0 运行环境对齐")
    device = torch.device("cuda")
    root = Path(os.environ["BONN_SEQUENCE_ROOT"]).resolve()
    rgb_records = read_index(root, "rgb.txt", "rgb")
    depth_records = read_index(root, "depth.txt", "depth")

    # Check timestamps and image sizes before any GPU work.
    rgb_times = [r["timestamp"] for r in rgb_records]
    if any(rgb_times[i] <= rgb_times[i - 1] for i in range(1, len(rgb_times))):
        raise RuntimeError("rgb.txt 时间戳非严格递增，停止以避免传播顺序不明确")
    missing_rgb = [r for r in rgb_records if r["path"] is None]
    if missing_rgb:
        details = [(r["line_number"], r["timestamp"], r["raw_path"]) for r in missing_rgb[:20]]
        raise RuntimeError(f"存在 {len(missing_rgb)} 条 RGB 文件缺失，不能构建连续传播基线。示例：{details}")

    with Image.open(rgb_records[0]["path"]) as im:
        original_W, original_H = im.size
    for r in rgb_records:
        with Image.open(r["path"]) as im:
            if im.size != (original_W, original_H):
                raise RuntimeError(f"RGB 尺寸不一致：{r['path']} size={im.size}, expected={(original_W, original_H)}")

    all_pairs, valid_depths = prepare_pairs(rgb_records, depth_records, MAX_TIME_DIFF_SEC)
    selected = records
    selected_indices = [i for i, _ in selected]
    selected_pair_rows = [all_pairs[i] for i in selected_indices]
    write_csv(out_dir / "gt1_rgb_depth_pairs.csv", selected_pair_rows)
    missing_gt_rows = [{
        "depth_index_line": r["line_number"],
        "timestamp": r["timestamp"],
        "raw_path": r["raw_path"],
    } for r in depth_records if r["path"] is None]
    write_csv(out_dir / "gt1_missing_depth_files.csv", missing_gt_rows)

    # Preprocessing settings copied from VeloDepth.infer.
    ratio_bounds = model.shape_constraints["ratio_bounds"]
    pixels_min = model.shape_constraints["pixels_min"]
    pixels_max = model.shape_constraints["pixels_max"]
    interval = (pixels_max - pixels_min) / 10.0
    pixels_bounds = (0 * interval + pixels_min, 1 * interval + pixels_min)
    paddings, (padded_H, padded_W) = get_paddings((original_H, original_W), ratio_bounds)
    resize_factor, (new_H, new_W) = get_resize_factor((padded_H, padded_W), pixels_bounds)
    print(f"原始分辨率：{original_H}x{original_W}; 网络画布：{new_H}x{new_W}; resize_factor={resize_factor:.6f}", flush=True)

    mean = torch.tensor(IMAGENET_DATASET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_DATASET_STD, device=device).view(1, 3, 1, 1)
    validity_mask = torch.ones((1, 1, new_H, new_W), device=device, dtype=torch.bool)
    # The normal infer path finds zero image padding for an all-valid mask.
    image_paddings = torch.zeros((1, 4), device=device, dtype=torch.long)
    depth_paddings = torch.zeros((1, 4), device=device, dtype=torch.long)

    tmp_pred_path = out_dir / "gt1_a_predictions.partial.npy"
    final_pred_path = out_dir / "gt1_a_predictions.npy"
    pred_mm = np.lib.format.open_memmap(
        tmp_pred_path, mode="w+", dtype=np.float32,
        shape=(len(selected), original_H, original_W),
    )

    frame_rows = []
    metric_rows = []
    keyframe_calls = 0
    fast_calls = 0
    previous_img = None
    previous_frame_u8 = None
    rays_base = None
    radius_state = None
    confidence_state = None
    features_state = None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        for slot, (global_idx, rgb_record) in enumerate(selected):
            frame_start = time.perf_counter()
            rgb_np = load_rgb(rgb_record["path"])
            img, frame_u8 = preprocess_rgb(rgb_np, device, paddings, (new_H, new_W), mean, std)
            del rgb_np

            if slot == 0:
                base_inputs = {
                    "image": img,
                    "image_original": img,
                    "validity_mask": validity_mask,
                }
                base_metas = [{"T": 1, "B": 1}]
                base_features, base_tokens = model.pixel_encoder(base_inputs["image"])
                base_features = [model.stacking_fn(base_features[i:j]).contiguous()
                                 for i, j in model.slices_encoder_range]
                base_tokens = [model.stacking_fn(base_tokens[i:j]).contiguous()
                               for i, j in model.slices_encoder_range]
                base_inputs["features"], base_inputs["tokens"] = base_features, base_tokens
                model.pixel_decoder.set_shapes(base_inputs, base_metas)
                rays_base, radius_state, confidence_state, features_state = model.pixel_decoder.forward_keyframe(
                    base_inputs, base_metas
                )
                keyframe_calls += 1
                # Release encoder tensors after the persistent keyframe state is built.
                del base_inputs, base_metas, base_features, base_tokens
            else:
                curr_rgb = frame_u8.permute(1, 2, 0).numpy()
                prev_rgb = previous_frame_u8.permute(1, 2, 0).numpy()
                flow_np = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST).calc(
                    cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY),
                    None,
                )
                flow_px = torch.from_numpy(flow_np).to(device=device).permute(2, 0, 1).unsqueeze(0)
                flow_norm, grid_sample = model._normalize_and_grid_from_flow(
                    flow_px, img, image_paddings, depth_paddings
                )
                image_residual = rgb2luma(img).unsqueeze(1) - F.grid_sample(
                    rgb2luma(previous_img).unsqueeze(1), grid_sample, mode="bilinear"
                )
                fast_in = {
                    "grid_sample": grid_sample,
                    "image_residual": image_residual,
                    "image": img,
                    "image_original": img,
                    "validity_mask": validity_mask,
                    "flow_bwd": flow_norm,
                    "features_teacher": features_state,
                }
                fast_metas = [{"T": 1, "B": 1}]
                flow_t_tm1 = fast_in["flow_bwd"].clone()
                residuals_t_tm1 = fast_in["image_residual"].clone()
                features_residual, flow_refined, flow_features = model.small_encoder(
                    rgb_t=img,
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
                fast_in["features_residual"] = features_residual
                fast_in["flow_features"] = flow_features
                fast_in["flow_bwd_refined"] = flow_refined
                radius_state, confidence_state, features_state = model.pixel_decoder.forward_fast(
                    fast_in, fast_metas
                )
                fast_calls += 1
                del flow_np, flow_px, flow_norm, grid_sample, image_residual
                del fast_in, features_residual, flow_refined, flow_features, flow_t_tm1, residuals_t_tm1

            # Same as infer(): radius * fixed keyframe rays, then postprocess to original HxW.
            rays_map = rays_base.reshape(1, new_H, new_W, 3).permute(0, 3, 1, 2).contiguous()
            points_canvas = radius_state * rays_map
            points_original = _postprocess(
                points_canvas, (padded_H, padded_W), paddings,
                interpolation_mode=model.interpolation_mode,
            )
            pred = points_original[0, -1].float().cpu().numpy()
            if pred.shape != (original_H, original_W):
                raise RuntimeError(f"预测尺寸错误：{pred.shape}")
            if not np.isfinite(pred).all():
                raise RuntimeError(f"预测包含 NaN/Inf：RGB index={global_idx}")
            pred_mm[slot] = pred

            pair = all_pairs[global_idx]
            row = {
                "frame_index_zero_based": global_idx,
                "prediction_slot": slot,
                "rgb_timestamp": rgb_record["timestamp"],
                "rgb_file": Path(rgb_record["path"]).name,
                "gt_file": Path(pair["depth_path"]).name if pair["depth_path"] else "",
                "timestamp_diff_ms": pair["abs_delta_ms"],
                "gt_match_status": pair["gt_match_status"],
                "gt_reused": pair["gt_reused"],
                "inference_sec": time.perf_counter() - frame_start,
            }
            if pair["depth_path"]:
                gt_raw = cv2.imread(pair["depth_path"], cv2.IMREAD_UNCHANGED)
                if gt_raw is None:
                    raise RuntimeError(f"无法读取 GT：{pair['depth_path']}")
                if gt_raw.dtype != np.uint16 or gt_raw.ndim != 2:
                    raise RuntimeError(f"GT 格式错误：{pair['depth_path']} dtype={gt_raw.dtype} shape={gt_raw.shape}")
                if gt_raw.shape != pred.shape:
                    raise RuntimeError(f"GT/预测尺寸不一致：{gt_raw.shape} vs {pred.shape}")
                gt = gt_raw.astype(np.float32) / DEPTH_SCALE
                metrics = calculate_metrics(gt, pred)
                row.update(metrics)
                row["metrics_status"] = "EVALUATED"
                metric_rows.append(row.copy())
            else:
                row.update({
                    "AbsRel": "", "RMSE": "", "delta1": "", "valid_ratio": "",
                    "gt_valid_ratio": "", "valid_pixels": "", "total_pixels": "",
                    "median_gt_m": "", "median_pred_m": "", "gt_min_m": "", "gt_max_m": "",
                    "pred_min_m": "", "pred_max_m": "", "metrics_status": "NO_GT_WITHIN_TOLERANCE",
                })
            frame_rows.append(row)

            previous_img = img
            previous_frame_u8 = frame_u8
            if slot % 25 == 0 or slot == len(selected) - 1:
                print(
                    f"frame {slot + 1}/{len(selected)} (index={global_idx}) "
                    f"KF={keyframe_calls} fast={fast_calls} "
                    f"matched_GT={len(metric_rows)} "
                    f"elapsed={time.perf_counter() - total_start:.1f}s",
                    flush=True,
                )
            del points_canvas, points_original, pred, frame_u8

    torch.cuda.synchronize()
    elapsed_total = time.perf_counter() - total_start
    pred_mm.flush()
    del pred_mm
    tmp_pred_path.replace(final_pred_path)

    write_csv(out_dir / "gt1_a_baseline_frames.csv", frame_rows)
    evaluated = [r for r in frame_rows if r.get("metrics_status") == "EVALUATED"]
    if not evaluated:
        raise RuntimeError("全序列没有任何可评测帧")

    names = ["AbsRel", "RMSE", "delta1", "valid_ratio", "gt_valid_ratio"]
    macro = {name: float(np.mean([float(r[name]) for r in evaluated])) for name in names}
    total_valid = sum(int(r["valid_pixels"]) for r in evaluated)
    total_pixels = sum(int(r["total_pixels"]) for r in evaluated)
    pixel_weighted = {
        "AbsRel": float(sum(float(r["AbsRel"]) * int(r["valid_pixels"]) for r in evaluated) / total_valid),
        "RMSE": float(np.sqrt(sum(float(r["RMSE"]) ** 2 * int(r["valid_pixels"]) for r in evaluated) / total_valid)),
        "delta1": float(sum(float(r["delta1"]) * int(r["valid_pixels"]) for r in evaluated) / total_valid),
        "valid_ratio": float(total_valid / total_pixels),
    }
    matched_deltas = [float(r["timestamp_diff_ms"]) for r in evaluated]
    duplicate_gt_count = sum(bool(r["gt_reused"]) for r in frame_rows)
    missing_rgb_count = 0  # missing RGB files are fatal in the preflight stage.
    missing_depth_count = sum(r["path"] is None for r in depth_records)
    unmatched_count = len(frame_rows) - len(evaluated)
    weights_path = snapshot / "model.safetensors"
    summary = {
        "experiment": "GT1 Bonn Crowd2 full-sequence native A baseline",
        "inference_status": "PASS",
        "evaluation_status": "PASS" if unmatched_count == 0 else "PASS_WITH_UNMATCHED_GT_FRAMES",
        "sequence_root": str(root),
        "snapshot": str(snapshot),
        "checkpoint_sha256": sha256_file(weights_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "depth_scale": DEPTH_SCALE,
        "max_time_diff_ms": MAX_TIME_DIFF_SEC * 1000,
        "original_resolution": [original_H, original_W],
        "network_resolution": [new_H, new_W],
        "resolution_level": 0,
        "keyframe_policy": "one initial keyframe; fast propagation for all remaining frames; heuristic refresh disabled with alpha=1e9",
        "rgb_index_records": len(rgb_records),
        "processed_rgb_frames": len(selected),
        "evaluated_frames": len(evaluated),
        "unmatched_gt_frames": unmatched_count,
        "missing_rgb_files": missing_rgb_count,
        "missing_depth_files_in_index": missing_depth_count,
        "rgb_gt_pairs_reusing_gt": duplicate_gt_count,
        "keyframe_calls": keyframe_calls,
        "fast_calls": fast_calls,
        "max_timestamp_diff_ms": max(matched_deltas) if matched_deltas else None,
        "mean_per_frame_metrics": macro,
        "pixel_weighted_metrics": pixel_weighted,
        "inference_wall_time_sec": elapsed_total,
        "mean_sec_per_rgb_frame": elapsed_total / max(1, len(selected)),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "frames": len(frame_rows),
    }
    (out_dir / "gt1_a_baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# GT1 实验报告：Bonn Crowd2 全序列原生 A 基线",
        "",
        "## 1. 实验目的与协议",
        "",
        "- 模型：原生 VeloDepth 预训练权重，修复头关闭。",
        f"- 原始 GT PNG：uint16 / {DEPTH_SCALE:.0f} 米。",
        f"- RGB-GT 最近邻时间戳匹配，最大时间差 {MAX_TIME_DIFF_SEC * 1000:.1f} ms。",
        "- 推理：全序列按 RGB 时间顺序处理；第一帧关键帧，其余帧连续 fast 传播。",
        "- 为复现 GT0 协议，关键帧启发式刷新关闭；不得把该结果解释为默认刷新策略下的表现。",
        "- 不使用额外裁剪或 10 m 深度上限；GT 零值、非有限值和非正预测不参与指标。",
        "",
        "## 2. 序列覆盖与运行情况",
        "",
        f"- RGB 索引记录：{len(rgb_records)}",
        f"- 本次处理 RGB 帧：{len(selected)}",
        f"- 有效 GT 评测帧：{len(evaluated)}",
        f"- 无 GT 匹配帧：{unmatched_count}",
        f"- depth.txt 中缺失文件：{missing_depth_count}",
        f"- 重复使用 GT 的 RGB 帧数：{duplicate_gt_count}",
        f"- 关键帧调用：{keyframe_calls}",
        f"- fast 调用：{fast_calls}",
        f"- 最大时间戳差：{summary['max_timestamp_diff_ms'] if summary['max_timestamp_diff_ms'] is not None else 'N/A'} ms",
        f"- 总推理墙钟时间：{elapsed_total:.3f} s",
        f"- 平均每个 RGB 帧耗时：{summary['mean_sec_per_rgb_frame']:.3f} s",
        f"- 峰值 CUDA 已分配显存：{summary['peak_cuda_allocated_gib']:.3f} GiB",
        "",
        "## 3. 指标汇总",
        "",
        "### 逐帧宏平均（每帧权重相同）",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ]
    for name, value in macro.items():
        report.append(f"| {name} | {value:.6f} |")
    report += [
        "",
        "### 像素加权汇总",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ]
    for name, value in pixel_weighted.items():
        report.append(f"| {name} | {value:.6f} |")
    report += [
        "",
        "## 4. 结论边界",
        "",
        "本实验建立的是与 GT0 一致的、关闭启发式刷新的一次关键帧连续传播基线。",
        "它不代表默认 VeloDepth 关键帧刷新策略，也不证明修复头的增益。",
        "后续 A/C/D 对照必须复用相同帧、相同 GT 配对和相同指标协议。",
        "",
        "## 5. 产物",
        "",
        "- `gt1_a_baseline_frames.csv`：全 RGB 帧逐帧指标；未匹配 GT 帧也保留行。",
        "- `gt1_a_baseline_summary.json`：协议、权重哈希、覆盖率、指标和资源统计。",
        "- `gt1_rgb_depth_pairs.csv`：逐 RGB 帧的 GT 配对清单。",
        "- `gt1_missing_depth_files.csv`：索引中不存在的深度文件。",
        "- `gt1_a_predictions.npy`：预测数组，shape=(处理帧数,H,W)，按 RGB 索引顺序保存。",
        "",
    ]
    (out_dir / ("GT1_PRECHECK_report.md" if args.num_frames is not None else "GT1_report.md")).write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\n========== GT1 SUMMARY ==========")
    print(json.dumps({k: summary[k] for k in [
        "inference_status", "evaluation_status", "processed_rgb_frames", "evaluated_frames",
        "unmatched_gt_frames", "keyframe_calls", "fast_calls", "mean_per_frame_metrics",
        "pixel_weighted_metrics", "inference_wall_time_sec", "peak_cuda_allocated_gib"
    ]}, indent=2, ensure_ascii=False), flush=True)
    print("预测文件：", final_pred_path, flush=True)
    print("逐帧 CSV：", out_dir / "gt1_a_baseline_frames.csv", flush=True)
    print("报告目录：", out_dir, flush=True)

    if args.compare_gt0:
        gt0_dir = Path(args.gt0_dir).resolve()
        comparisons = []
        for slot, (global_idx, _) in enumerate(selected):
            old_path = gt0_dir / f"pred_depth_frame_{global_idx:03d}.npy"
            if not old_path.is_file():
                raise FileNotFoundError(f"GT0 参考预测不存在：{old_path}")
            old = np.load(old_path).astype(np.float32)
            new = np.asarray(np.load(final_pred_path, mmap_mode="r")[slot], dtype=np.float32)
            if old.shape != new.shape:
                raise RuntimeError(f"GT0/stream shape 不一致：{old.shape} vs {new.shape}")
            diff = np.abs(old - new)
            comparisons.append({
                "frame_index_zero_based": global_idx,
                "max_abs_diff_m": float(diff.max()),
                "mean_abs_diff_m": float(diff.mean()),
                "p99_abs_diff_m": float(np.quantile(diff, 0.99)),
            })
        compare_summary = {
            "experiment": "GT1 streaming preflight vs GT0 three-frame infer",
            "frames": comparisons,
            "threshold_max_abs_diff_m": 0.02,
            "threshold_mean_abs_diff_m": 0.002,
            "pass": all(x["max_abs_diff_m"] <= 0.02 and x["mean_abs_diff_m"] <= 0.002 for x in comparisons),
        }
        (out_dir / "gt1_preflight_equivalence.json").write_text(
            json.dumps(compare_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("GT0 等价性复核：", json.dumps(compare_summary, indent=2, ensure_ascii=False), flush=True)
        if not compare_summary["pass"]:
            raise RuntimeError("GT1 流式实现与 GT0 三帧参考差异超限；停止，不要运行全序列")
        print("GT1 PRECHECK EQUIVALENCE: PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=0, help="rgb.txt 中的零基索引")
    parser.add_argument("--num-frames", type=int, default=None, help="仅处理指定帧数；省略则处理剩余全部帧")
    parser.add_argument("--out-dir", default="outputs/gt1_validation")
    parser.add_argument("--compare-gt0", action="store_true", help="将流式预测与 GT0 已保存的三帧预测比较")
    parser.add_argument("--gt0-dir", default="outputs/gt0_validation")
    args = parser.parse_args()
    if args.start_index < 0 or (args.num_frames is not None and args.num_frames < 1):
        parser.error("start-index 必须 >=0，num-frames 必须 >=1")
    root_env = os.environ.get("BONN_SEQUENCE_ROOT")
    snapshot_env = os.environ.get("VDEEP_SNAPSHOT")
    if not root_env:
        raise RuntimeError("请先设置 BONN_SEQUENCE_ROOT")
    if not snapshot_env:
        raise RuntimeError("请先设置 VDEEP_SNAPSHOT")
    root = Path(root_env).resolve()
    snapshot = Path(snapshot_env).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_records = read_index(root, "rgb.txt", "rgb")
    if args.start_index >= len(rgb_records):
        raise RuntimeError(f"start-index={args.start_index} 超出 RGB 记录数量 {len(rgb_records)}")
    end = len(rgb_records) if args.num_frames is None else min(len(rgb_records), args.start_index + args.num_frames)
    selected = [(i, rgb_records[i]) for i in range(args.start_index, end)]
    if args.compare_gt0 and args.num_frames is None:
        raise RuntimeError("--compare-gt0 仅用于指定的小规模预检，不要对全序列使用")
    if args.compare_gt0 and args.num_frames != 3:
        raise RuntimeError("--compare-gt0 预检要求 --num-frames 3")
    print("GT1 Bonn Crowd2 streaming A baseline", flush=True)
    print("sequence root:", root, flush=True)
    print("snapshot:", snapshot, flush=True)
    print("selected RGB frames:", len(selected), "from", args.start_index, "to", end - 1, flush=True)
    model, _weights_path = make_model(snapshot, torch.device("cuda"))
    run_stream(model, selected, None, out_dir, snapshot, args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
