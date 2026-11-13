#!/usr/bin/env python3
"""E2-S8-S5: minimal GT-supervised optimizer-step smoke test for C/D repair modes.

This script is intentionally a bounded feasibility test, not a benchmark.
It uses the raw Bonn/TUM RGB-D folder protocol:
  - nearest timestamp pairing from rgb.txt to depth.txt (max delta 20 ms)
  - uint16 depth PNG / 5000 meters; zero is invalid
  - VeloDepth output `depth` (camera z-depth), not `distance`
  - supervise the third frame so D's frame-2 persistent-state repair can affect it

Required environment variables:
  BONN_SEQUENCE_ROOT=/path/to/rgbd_bonn_crowd2
  VDEEP_SNAPSHOT=/path/to/huggingface/velodepth/snapshot

Example:
  python experiments/p1/e2_s8_s5_gt_train_smoke.py --mode D --steps 1
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from velodepth.models.velodepth import VeloDepth


def parse_index(root: Path, filename: str, subdir: str) -> list[dict[str, Any]]:
    index = root / filename
    if not index.is_file():
        raise FileNotFoundError(f"索引文件不存在：{index}")
    records: list[dict[str, Any]] = []
    missing_depth = []
    for line_no, line in enumerate(index.read_text(errors="ignore").splitlines(), 1):
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
                root / subdir / rel,
                root / subdir / rel.name,
                root / rel.name,
            ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            if filename == "depth.txt":
                missing_depth.append((line_no, parts[1]))
                continue
            raise FileNotFoundError(
                f"{filename}:{line_no} 无法定位文件 {parts[1]!r}；尝试过："
                + ", ".join(str(p) for p in candidates)
            )
        records.append({"timestamp": timestamp, "path": path, "index_line": line_no})
    if missing_depth:
        print(
            f"[INDEX WARNING] {filename}: skipped "
            f"{len(missing_depth)} missing depth files",
            flush=True
        )
        for line_no, name in missing_depth:
            print(f"  missing line={line_no}: {name}", flush=True)

    if not records:
        raise RuntimeError(f"索引没有有效条目：{index}")
    return records


def pair_rgb_depth(root: Path, start: int, count: int, max_delta_ms: float):
    rgbs = parse_index(root, "rgb.txt", "rgb")
    depths = parse_index(root, "depth.txt", "depth")
    selected = rgbs[start:start + count]
    if len(selected) != count:
        raise RuntimeError(
            f"RGB 索引从零起始索引 {start} 取 {count} 帧失败；"
            f"有效 RGB 总数={len(rgbs)}"
        )
    depth_ts = np.asarray([r["timestamp"] for r in depths], dtype=np.float64)
    pairs = []
    for rgb in selected:
        j = int(np.argmin(np.abs(depth_ts - rgb["timestamp"])))
        dep = depths[j]
        delta_ms = abs(float(rgb["timestamp"]) - float(dep["timestamp"])) * 1000.0
        if delta_ms > max_delta_ms:
            raise RuntimeError(
                f"RGB {rgb['path'].name} 最近深度时间差 {delta_ms:.3f} ms "
                f"> {max_delta_ms:.3f} ms"
            )
        pairs.append({
            "rgb_timestamp": float(rgb["timestamp"]),
            "rgb_path": rgb["path"],
            "depth_timestamp": float(dep["timestamp"]),
            "depth_path": dep["path"],
            "delta_ms": float(delta_ms),
            "rgb_global_index_zero_based": int(start + len(pairs)),
        })
    return pairs, len(rgbs), len(depths)


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"RGB 读取失败：{path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_raw_depth(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"深度 PNG 读取失败：{path}")
    if raw.ndim != 2:
        raise RuntimeError(f"深度 GT 必须是单通道，实际 shape={raw.shape}, path={path}")
    if raw.dtype != np.uint16:
        raise RuntimeError(
            f"原始 GT 预期 uint16 PNG，实际 dtype={raw.dtype}, path={path}。"
            "不要在未经确认的情况下自动缩放/转换标签。"
        )
    gt_m = raw.astype(np.float32) / 5000.0
    valid = (raw > 0) & np.isfinite(gt_m)
    meta = {
        "path": str(path),
        "dtype": str(raw.dtype),
        "shape": list(raw.shape),
        "raw_min_nonzero": int(raw[raw > 0].min()) if np.any(raw > 0) else None,
        "raw_max": int(raw.max()),
        "valid_pixels": int(valid.sum()),
        "valid_ratio": float(valid.mean()),
        "gt_median_m": float(np.median(gt_m[valid])) if np.any(valid) else None,
        "gt_scale_divisor": 5000,
        "zero_is_invalid": True,
    }
    return gt_m, meta


def summarize_gradients(named_params) -> dict[str, Any]:
    present = [(name, p.grad.detach()) for name, p in named_params if p.grad is not None]
    grads = [g for _, g in present]
    nonzero_names = [name for name, g in present if torch.count_nonzero(g).item() > 0]
    front_names = [name for name in nonzero_names if not name.endswith((".6.weight", ".6.bias"))]
    return {
        "gradient_tensors": len(grads),
        "nonzero_tensors": len(nonzero_names),
        "nonzero_parameter_names": nonzero_names,
        "nonzero_front_layer_tensors": len(front_names),
        "max_abs": max((float(g.abs().max().item()) for g in grads), default=0.0),
        "l1": sum(float(g.abs().sum().item()) for g in grads),
        "all_finite": all(bool(torch.isfinite(g).all().item()) for g in grads),
    }


def depth_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> dict[str, Any]:
    pred = pred.float()
    gt = gt.float()
    if pred.shape != gt.shape:
        raise RuntimeError(f"预测/GT 尺寸不一致：pred={tuple(pred.shape)}, gt={tuple(gt.shape)}")
    if not torch.isfinite(pred).all():
        raise RuntimeError("预测深度出现 NaN/Inf")
    p = pred.clamp(min=1e-4)
    g = gt.clamp(min=1e-4)
    m = valid.bool()
    if int(m.sum().item()) == 0:
        raise RuntimeError("GT 有效掩码为空")
    diff = p[m] - g[m]
    ratio = torch.maximum(p[m] / g[m], g[m] / p[m])
    return {
        "valid_pixels": int(m.sum().item()),
        "absrel": float((diff.abs() / g[m]).mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "delta1": float((ratio < 1.25).float().mean().item()),
        "pred_nonpositive_on_gt_valid": int(((pred <= 0) & m).sum().item()),
    }


def log_depth_l1(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    gt = gt.float()
    m = valid.bool() & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0)
    if int(m.sum().item()) == 0:
        raise RuntimeError("监督损失没有有效像素")
    # z-depth output is supervised against registered camera z-depth GT.
    return (torch.log(pred[m].clamp(min=1e-4)) - torch.log(gt[m].clamp(min=1e-4))).abs().mean()


def run_infer(raw_infer, model, rgbs: torch.Tensor, grad: bool):
    if grad:
        with torch.enable_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return raw_infer(model, rgbs=rgbs, normalize=True)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return raw_infer(model, rgbs=rgbs, normalize=True)


def load_model(snapshot: Path, mode: str, device: torch.device):
    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"
    for p in (config_path, weights_path):
        if not p.is_file():
            raise FileNotFoundError(f"模型文件不存在：{p}")
    config = copy.deepcopy(json.loads(config_path.read_text(encoding="utf-8")))
    config["model"]["persistent_feature_repair"] = {
        "enabled": True,
        "mode": "post_readout" if mode == "D" else "pre_readout",
        "bottleneck": 32,
    }
    model = VeloDepth(config)
    state = load_file(str(weights_path), device="cpu")
    result = model.load_state_dict(state, strict=False)
    expected_prefix = "pixel_decoder.persistent_feature_repair."
    bad_missing = [k for k in result.missing_keys if not k.startswith(expected_prefix)]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint 不匹配：非修复头缺失键={bad_missing[:10]}, "
            f"unexpected={result.unexpected_keys[:10]}"
        )
    del state
    model.to(device)
    model.eval()
    # Keep all pretrained weights frozen; only the repair head is optimized.
    for p in model.parameters():
        p.requires_grad_(False)
    decoder = model.pixel_decoder
    head = decoder.persistent_feature_repair
    if head is None:
        raise RuntimeError("修复头未注册")
    for p in head.parameters():
        p.requires_grad_(True)
    decoder.feature_repair_mode = "off"

    # Match S8 gradient-audit conditions: force one keyframe + fast propagation.
    model.resolution_level = 0
    model.interpolation_mode = "bilinear"
    model.alpha_flow = 1e9
    model.beta_flow = 0.0
    model.alpha_luma = 1e9
    model.beta_luma = 0.0

    # Unwrap inference decorators to allow gradients through the real propagation graph.
    raw_encode_test = inspect.unwrap(type(model).encode_decode_test)
    model.encode_decode_test = MethodType(raw_encode_test, model)
    raw_infer = inspect.unwrap(type(model).infer)
    return model, decoder, head, raw_infer, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["C", "D"], required=True,
                        help="C=pre_readout；D=post_readout 持久状态写回")
    parser.add_argument("--steps", type=int, default=1,
                        help="本次只建议先跑 1 步；通过后再单独跑 10-20 步小试")
    parser.add_argument("--start-index", type=int, default=8,
                        help="rgb.txt 去掉注释/无效行后的零起始索引，默认选第 8-10 号帧")
    parser.add_argument("--num-frames", type=int, default=3,
                        help="必须至少 3 帧；本脚本以最后一帧做未来监督")
    parser.add_argument("--max-delta-ms", type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--outdir", default="outputs/e2_s8_s5_gt_smoke")
    args = parser.parse_args()

    if args.steps < 1:
        raise ValueError("--steps 必须 >= 1")
    if args.num_frames < 3:
        raise ValueError("D 模式至少需要 1 次关键帧 + 2 次 fast，即 3 帧")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验要求 CUDA；拒绝静默退化到 CPU")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.set_num_threads(4)
    device = torch.device("cuda")
    root = Path(os.environ["BONN_SEQUENCE_ROOT"]).resolve()
    snapshot = Path(os.environ["VDEEP_SNAPSHOT"]).resolve()
    outdir = Path(args.outdir) / f"mode_{args.mode}_start{args.start_index}_steps{args.steps}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("sequence root:", root, flush=True)
    print("snapshot:", snapshot, flush=True)
    print("mode:", args.mode, "steps:", args.steps, flush=True)

    pairs, rgb_count, depth_count = pair_rgb_depth(
        root, args.start_index, args.num_frames, args.max_delta_ms
    )
    if len({str(p["depth_path"]) for p in pairs}) != len(pairs):
        raise RuntimeError("所选监督窗口中出现重复深度文件；本轮拒绝继续，先修正配对协议")

    rgb_arrays, gt_arrays, gt_metas = [], [], []
    for i, pair in enumerate(pairs):
        rgb = read_rgb(pair["rgb_path"])
        gt_m, gt_meta = read_raw_depth(pair["depth_path"])
        if rgb.shape[:2] != gt_m.shape:
            raise RuntimeError(
                f"RGB/GT 分辨率不一致：{pair['rgb_path'].name} {rgb.shape[:2]} vs "
                f"{pair['depth_path'].name} {gt_m.shape}；本轮不自动 resize GT"
            )
        rgb_arrays.append(rgb)
        gt_arrays.append(gt_m)
        gt_metas.append(gt_meta)
        pair["rgb_path"] = str(pair["rgb_path"])
        pair["depth_path"] = str(pair["depth_path"])
        pair["pair_id"] = i
        pair["gt"] = gt_meta
        print(
            f"pair[{i}] rgb={Path(pair['rgb_path']).name} depth={Path(pair['depth_path']).name} "
            f"dt={pair['delta_ms']:.3f}ms valid={gt_meta['valid_pixels']} "
            f"median={gt_meta['gt_median_m']}", flush=True
        )

    rgbs = torch.from_numpy(np.stack(rgb_arrays)).permute(0, 3, 1, 2).contiguous()
    gts = torch.from_numpy(np.stack(gt_arrays)).to(device=device, dtype=torch.float32)
    valid_masks = torch.isfinite(gts) & (gts > 0)
    if not bool(valid_masks[-1].any().item()):
        raise RuntimeError("最后一帧 GT 没有有效像素")

    model, decoder, head, raw_infer, load_result = load_model(snapshot, args.mode, device)
    call_counter = {"keyframe": 0, "fast": 0}
    call_counts: dict[str, dict[str, int]] = {}
    original_keyframe = decoder.forward_keyframe
    original_fast = decoder.forward_fast

    def counted_keyframe(*a, **kw):
        call_counter["keyframe"] += 1
        return original_keyframe(*a, **kw)

    def counted_fast(*a, **kw):
        call_counter["fast"] += 1
        return original_fast(*a, **kw)

    decoder.forward_keyframe = counted_keyframe
    decoder.forward_fast = counted_fast

    def checked_infer(label: str, grad: bool):
        call_counter["keyframe"] = 0
        call_counter["fast"] = 0
        result = run_infer(raw_infer, model, rgbs, grad=grad)
        expected = {"keyframe": 1, "fast": args.num_frames - 1}
        call_counts[label] = dict(call_counter)
        if call_counter != expected:
            raise RuntimeError(
                f"{label}: 关键帧/快速传播调用数不符合固定递推协议："
                f"actual={call_counter}, expected={expected}"
            )
        return result

    # A baseline: same pretrained weights, repair disabled.
    decoder.feature_repair_mode = "off"
    base_out = checked_infer("A_baseline", grad=False)
    base_depth = base_out["depth"].detach().float()
    if base_depth.shape[0] != args.num_frames:
        raise RuntimeError(f"输出帧数不符：{tuple(base_depth.shape)}")
    if tuple(base_depth.shape[-2:]) != tuple(gts.shape[-2:]):
        raise RuntimeError(
            f"预测/GT 输出分辨率不一致：pred={tuple(base_depth.shape[-2:])}, "
            f"gt={tuple(gts.shape[-2:])}"
        )
    baseline_metrics = [
        depth_metrics(base_depth[i, 0], gts[i], valid_masks[i])
        for i in range(args.num_frames)
    ]
    del base_out

    # Before training, C/D must be identity because the repair head is zero-initialized.
    decoder.feature_repair_mode = "pre_readout" if args.mode == "C" else "post_readout"
    with torch.no_grad():
        pre_out = checked_infer("pre_update_" + args.mode, grad=False)
    pre_depth = pre_out["depth"].detach().float()
    del pre_out
    initial_max_diff = float((pre_depth - base_depth).abs().max().item())
    if initial_max_diff > 1e-6:
        raise RuntimeError(
            f"初始恒等性失败：mode={args.mode}, depth max abs diff={initial_max_diff}"
        )
    pre_metrics = [
        depth_metrics(pre_depth[i, 0], gts[i], valid_masks[i])
        for i in range(args.num_frames)
    ]
    initial_params = [p.detach().clone() for p in head.parameters()]
    params = list(head.parameters())
    named_params = list(head.named_parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    step_rows = []
    t0 = time.time()
    last_loss = None
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = checked_infer(f"train_step_{step}", grad=True)
        pred_last = outputs["depth"][-1, 0]
        loss = log_depth_l1(pred_last, gts[-1], valid_masks[-1])
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"step {step}: loss 非有限")
        loss.backward()
        grad_stats = summarize_gradients(named_params)
        if not grad_stats["all_finite"]:
            raise RuntimeError(f"step {step}: 梯度包含 NaN/Inf")
        if grad_stats["nonzero_tensors"] == 0 or grad_stats["l1"] <= 0:
            raise RuntimeError(f"step {step}: GT 损失未产生非零修复头梯度：{grad_stats}")
        optimizer.step()
        last_loss = float(loss.detach().item())
        step_rows.append({
            "step": step,
            "loss_log_depth_l1_before_update": last_loss,
            "gradient_tensors": grad_stats["gradient_tensors"],
            "nonzero_gradient_tensors": grad_stats["nonzero_tensors"],
            "nonzero_front_layer_tensors": grad_stats["nonzero_front_layer_tensors"],
            "nonzero_parameter_names": ";".join(grad_stats["nonzero_parameter_names"]),
            "gradient_max_abs": grad_stats["max_abs"],
            "gradient_l1": grad_stats["l1"],
            "gradient_all_finite": grad_stats["all_finite"],
        })
        print(
            f"step={step}/{args.steps} loss={last_loss:.8f} "
            f"nonzero_grad_tensors={grad_stats['nonzero_tensors']} "
            f"grad_l1={grad_stats['l1']:.6g}", flush=True
        )
        del outputs, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Verify that optimizer.step really changed repair-head parameters.
    with torch.no_grad():
        delta_max = max(float((p.detach() - p0).abs().max().item())
                        for p, p0 in zip(params, initial_params))
        delta_l2 = float(torch.sqrt(sum(((p.detach() - p0) ** 2).sum()
                                       for p, p0 in zip(params, initial_params))).item())

    with torch.no_grad():
        post_out = checked_infer("post_update_" + args.mode, grad=False)
    post_depth = post_out["depth"].detach().float()
    del post_out
    post_metrics = [
        depth_metrics(post_depth[i, 0], gts[i], valid_masks[i])
        for i in range(args.num_frames)
    ]
    post_loss = float(log_depth_l1(post_depth[-1, 0], gts[-1], valid_masks[-1]).item())
    if delta_max <= 0.0 or delta_l2 <= 0.0:
        raise RuntimeError("optimizer.step 后修复头参数没有发生变化")

    # Difference checks make the causal role observable, not an accuracy claim.
    frame_diffs = [float((post_depth[i] - base_depth[i]).abs().max().item())
                   for i in range(args.num_frames)]
    if args.mode == "D" and frame_diffs[1] > 1e-6:
        raise RuntimeError(
            f"D 模式当前帧读出应保持不变，但第 2 帧 max diff={frame_diffs[1]}"
        )

    report = {
        "experiment": "E2-S8-S5 GT-supervised optimizer-step smoke",
        "status_scope": "learnability smoke only; not generalization or final method validation",
        "timestamp": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "mode_name": "pre_readout" if args.mode == "C" else "post_readout",
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "sequence_root": str(root),
        "snapshot": str(snapshot),
        "gpu": torch.cuda.get_device_name(0),
        "source_sha256": {
            "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "decoder_velo.py": hashlib.sha256(Path("velodepth/models/decoder_velo.py").read_bytes()).hexdigest(),
            "persistent_feature_repair.py": hashlib.sha256(Path("velodepth/models/persistent_feature_repair.py").read_bytes()).hexdigest(),
            "velodepth.py": hashlib.sha256(Path("velodepth/models/velodepth.py").read_bytes()).hexdigest(),
        },
        "index_counts": {"rgb_records": rgb_count, "depth_records": depth_count},
        "pairing": pairs,
        "gt_contract": {
            "raw_png_dtype_required": "uint16",
            "depth_scale_divisor": 5000,
            "zero_invalid": True,
            "compared_prediction": "VeloDepth output depth (z component), not distance",
            "gt_metas": gt_metas,
            "valid_pixel_counts": [int(x.sum().item()) for x in valid_masks],
        },
        "checkpoint_load": {
            "missing_keys": len(load_result.missing_keys),
            "unexpected_keys": len(load_result.unexpected_keys),
        },
        "call_counts_by_pass": call_counts,
        "repair_head": {
            "parameters": sum(p.numel() for p in params),
            "parameter_tensors": len(params),
            "initial_identity_depth_max_abs_diff": initial_max_diff,
            "parameter_delta_max_abs": delta_max,
            "parameter_delta_l2": delta_l2,
        },
        "baseline_A_metrics_by_frame": baseline_metrics,
        "pre_update_C_or_D_metrics_by_frame": pre_metrics,
        "post_update_metrics_by_frame": post_metrics,
        "training": {
            "supervision_frame_index_within_clip": args.num_frames - 1,
            "loss": "masked mean absolute log-depth error",
            "loss_before_last_update": last_loss,
            "loss_after_update_eval": post_loss,
            "steps": step_rows,
            "wall_seconds": time.time() - t0,
        },
        "post_update_depth_max_abs_diff_vs_A_by_frame": frame_diffs,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "interpretation_guardrail": (
            "This repeats the same short clip for optimization. Its post-update metrics are training-clip metrics, "
            "not held-out results. A successful update proves only GT loss/gradient/optimizer connectivity."
        ),
    }
    report_path = outdir / "s5_gt_train_smoke.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = outdir / "s5_train_steps.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)
    torch.save({k: v.detach().cpu() for k, v in head.state_dict().items()}, outdir / "repair_head_state.pt")

    print("\nRESULT", json.dumps({
        "mode": args.mode,
        "steps": args.steps,
        "initial_identity_max_diff": initial_max_diff,
        "parameter_delta_max_abs": delta_max,
        "loss_before_last_update": last_loss,
        "loss_after_update_eval": post_loss,
        "baseline_frame3": baseline_metrics[-1],
        "post_update_frame3": post_metrics[-1],
        "frame_max_abs_diff_vs_A": frame_diffs,
        "peak_memory_bytes": report["peak_memory_bytes"],
        "report": str(report_path),
    }, indent=2, ensure_ascii=False), flush=True)
    print("E2-S8-S5 GT TRAIN SMOKE: PASS (learnability checks only)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
