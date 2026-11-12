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
from PIL import Image
from safetensors.torch import load_file
from velodepth.models.velodepth import VeloDepth


DEPTH_SCALE = 5000.0
MAX_TIME_DIFF_SEC = 0.020
START_INDEX = 8
FRAME_COUNT = 3


def resolve_path(root, raw_path, subdir):
    p = Path(raw_path)

    if p.is_absolute():
        candidates = [p]
    else:
        candidates = [
            root / p,
            root / subdir / p,
            root / subdir / p.name,
            root / p.name,
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return None


def read_index(root, filename, subdir):
    index_path = root / filename

    if not index_path.is_file():
        raise FileNotFoundError(f"索引不存在：{index_path}")

    records = []

    for line_number, line in enumerate(
        index_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
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

        path = resolve_path(root, fields[1], subdir)

        records.append({
            "timestamp": timestamp,
            "raw_path": fields[1],
            "path": path,
            "line_number": line_number,
        })

    if not records:
        raise RuntimeError(f"索引没有有效记录：{index_path}")

    return records


def calculate_metrics(gt, pred):
    valid = (
        (gt > 0)
        & np.isfinite(gt)
        & np.isfinite(pred)
        & (pred > 0)
    )

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


def write_csv(path, rows):
    if not rows:
        raise RuntimeError(f"拒绝写出空 CSV：{path}")

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.manual_seed(20261023)
    torch.set_num_threads(4)

    root = Path(os.environ["BONN_SEQUENCE_ROOT"]).resolve()
    snapshot = Path(os.environ["VDEEP_SNAPSHOT"]).resolve()
    out_dir = Path("outputs/gt0_validation").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = snapshot / "config.json"
    weights_path = snapshot / "model.safetensors"

    if not config_path.is_file():
        raise FileNotFoundError(f"模型配置不存在：{config_path}")

    if not weights_path.is_file():
        raise FileNotFoundError(f"预训练权重不存在：{weights_path}")

    print("GT0: Bonn Crowd2 three-frame calibration", flush=True)
    print("数据目录：", root, flush=True)
    print("预训练快照：", snapshot, flush=True)

    # ---------------------------------------------------------
    # 1. Load the exact three RGB records and nearest GT records.
    # ---------------------------------------------------------
    rgb_records = read_index(root, "rgb.txt", "rgb")
    depth_records = read_index(root, "depth.txt", "depth")

    depth_records.sort(key=lambda x: x["timestamp"])

    selected_rgb = rgb_records[
        START_INDEX:START_INDEX + FRAME_COUNT
    ]

    if len(selected_rgb) != FRAME_COUNT:
        raise RuntimeError(
            f"RGB 索引不足：期望 {FRAME_COUNT} 帧，"
            f"实际取得 {len(selected_rgb)} 帧"
        )

    pairs = []

    for offset, rgb in enumerate(selected_rgb):
        if rgb["path"] is None:
            raise FileNotFoundError(
                f"RGB 文件不存在：{rgb['raw_path']}"
            )

        depth = min(
            depth_records,
            key=lambda item: abs(
                item["timestamp"] - rgb["timestamp"]
            ),
        )

        if depth["path"] is None:
            raise FileNotFoundError(
                "最近邻深度文件不存在："
                f"RGB={rgb['path'].name}, GT={depth['raw_path']}"
            )

        delta = abs(depth["timestamp"] - rgb["timestamp"])

        if delta > MAX_TIME_DIFF_SEC:
            raise RuntimeError(
                f"RGB 索引 {START_INDEX + offset} 超过时间差上限："
                f"{delta * 1000:.3f} ms"
            )

        pairs.append({
            "frame_index_zero_based": START_INDEX + offset,
            "rgb_timestamp": rgb["timestamp"],
            "rgb_path": str(rgb["path"]),
            "depth_timestamp": depth["timestamp"],
            "depth_path": str(depth["path"]),
            "signed_delta_ms": (
                depth["timestamp"] - rgb["timestamp"]
            ) * 1000.0,
            "abs_delta_ms": delta * 1000.0,
        })

    print("\nSTAGE 1: RGB-GT association", flush=True)

    for pair in pairs:
        print(
            f"frame={pair['frame_index_zero_based']} "
            f"rgb={Path(pair['rgb_path']).name} "
            f"gt={Path(pair['depth_path']).name} "
            f"delta={pair['abs_delta_ms']:.3f} ms",
            flush=True,
        )

    # ---------------------------------------------------------
    # 2. Read RGB and raw GT; validate raw depth encoding.
    # ---------------------------------------------------------
    rgb_arrays = []
    gt_arrays = []
    pair_rows = []

    for pair in pairs:
        with Image.open(pair["rgb_path"]) as image:
            rgb = np.asarray(image.convert("RGB")).copy()

        gt_raw = cv2.imread(
            pair["depth_path"], cv2.IMREAD_UNCHANGED
        )

        if gt_raw is None:
            raise RuntimeError(
                f"OpenCV 无法读取 GT：{pair['depth_path']}"
            )

        if gt_raw.dtype != np.uint16:
            raise TypeError(
                f"GT 必须是 uint16 PNG，实际为 {gt_raw.dtype}："
                f"{pair['depth_path']}"
            )

        if gt_raw.ndim != 2:
            raise ValueError(
                f"GT 应为单通道深度图，实际 shape={gt_raw.shape}"
            )

        if rgb.shape[:2] != gt_raw.shape:
            raise ValueError(
                f"RGB/GT 尺寸不一致："
                f"RGB={rgb.shape[:2]}, GT={gt_raw.shape}"
            )

        gt = gt_raw.astype(np.float32) / DEPTH_SCALE

        positive_raw = gt_raw[gt_raw > 0]

        if positive_raw.size == 0:
            raise RuntimeError(f"GT 没有有效深度：{pair['depth_path']}")

        rgb_arrays.append(rgb)
        gt_arrays.append(gt)

        detail = dict(pair)
        detail.update({
            "rgb_shape": str(tuple(rgb.shape)),
            "gt_shape": str(tuple(gt_raw.shape)),
            "gt_dtype": str(gt_raw.dtype),
            "depth_scale": DEPTH_SCALE,
            "gt_zero_ratio": float((gt_raw == 0).mean()),
            "gt_valid_ratio": float((gt_raw > 0).mean()),
            "gt_raw_min_positive": int(positive_raw.min()),
            "gt_raw_median_positive": float(
                np.median(positive_raw)
            ),
            "gt_raw_max": int(gt_raw.max()),
        })

        pair_rows.append(detail)

    shapes = {tuple(rgb.shape) for rgb in rgb_arrays}

    if len(shapes) != 1:
        raise RuntimeError(f"三帧 RGB 尺寸不一致：{shapes}")

    write_csv(
        out_dir / "gt0_rgb_depth_pairs.csv",
        pair_rows,
    )

    print("\nSTAGE 2: GT encoding", flush=True)

    for row in pair_rows:
        print(
            f"frame={row['frame_index_zero_based']} "
            f"dtype={row['gt_dtype']} "
            f"shape={row['gt_shape']} "
            f"zero_ratio={row['gt_zero_ratio']:.4f} "
            f"median_raw={row['gt_raw_median_positive']:.1f}",
            flush=True,
        )

    # ---------------------------------------------------------
    # 3. Construct original model without the repair head.
    # ---------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GT0 本轮要求使用 CUDA，以与此前 S8 推理环境对齐"
        )

    device = torch.device("cuda")
    print("\nGPU：", torch.cuda.get_device_name(0), flush=True)

    config = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    if "persistent_feature_repair" in config.get("model", {}):
        raise RuntimeError(
            "原始配置不应包含修复头配置，请检查快照是否被修改"
        )

    print("STAGE 3: construct original VeloDepth", flush=True)

    model = VeloDepth(config)
    model.cpu()

    decoder = model.pixel_decoder

    if getattr(decoder, "persistent_feature_repair", None) is not None:
        raise RuntimeError("A 基线错误地注册了修复头")

    if getattr(decoder, "feature_repair_mode", "off") != "off":
        raise RuntimeError("A 基线的 feature_repair_mode 不是 off")

    state = load_file(str(weights_path), device="cpu")
    incompatible = model.load_state_dict(state, strict=True)
    del state

    print("原始权重加载：PASS", flush=True)
    print("修复头：未注册", flush=True)
    print("加载结果：", incompatible, flush=True)

    model.to(device)
    model.eval()

    # Match the already validated S8 real-clip inference settings.
    model.resolution_level = 0
    model.interpolation_mode = "bilinear"
    model.alpha_flow = 1e9
    model.beta_flow = 0.0
    model.alpha_luma = 1e9
    model.beta_luma = 0.0

    stats = {"keyframe": 0, "fast": 0}

    original_keyframe = decoder.forward_keyframe
    original_fast = decoder.forward_fast

    def counted_keyframe(*args, **kwargs):
        stats["keyframe"] += 1
        return original_keyframe(*args, **kwargs)

    def counted_fast(*args, **kwargs):
        stats["fast"] += 1
        return original_fast(*args, **kwargs)

    decoder.forward_keyframe = counted_keyframe
    decoder.forward_fast = counted_fast

    rgbs = torch.from_numpy(
        np.stack(rgb_arrays)
    ).permute(0, 3, 1, 2).contiguous()

    print("\nSTAGE 4: original A inference", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        output = model.infer(rgbs=rgbs, normalize=True)

    torch.cuda.synchronize()
    elapsed_sec = time.perf_counter() - start

    if stats["keyframe"] != 1 or stats["fast"] != 2:
        raise RuntimeError(
            f"不是预期的 1 keyframe + 2 fast：{stats}"
        )

    if "depth" not in output:
        raise KeyError(f"模型没有 depth 输出，现有键：{list(output)}")

    predictions = output["depth"].detach().float().cpu().numpy()

    if predictions.shape[0] != FRAME_COUNT:
        raise RuntimeError(
            f"预测帧数错误：shape={predictions.shape}"
        )

    print("调用次数：", stats, flush=True)
    print("预测 shape：", predictions.shape, flush=True)

    # ---------------------------------------------------------
    # 4. Per-frame metrics using the existing Bonn P0 protocol.
    # ---------------------------------------------------------
    print("\nSTAGE 5: calculate GT metrics", flush=True)

    metric_rows = []

    for i, pair in enumerate(pairs):
        pred = np.squeeze(predictions[i]).astype(np.float32)
        gt = gt_arrays[i]

        if pred.shape != gt.shape:
            raise RuntimeError(
                f"预测/GT shape 不一致：{pred.shape} vs {gt.shape}"
            )

        if not np.isfinite(pred).all():
            raise RuntimeError(
                f"预测包含 NaN/Inf：frame={pair['frame_index_zero_based']}"
            )

        metrics = calculate_metrics(gt, pred)

        pred_path = out_dir / (
            f"pred_depth_frame_{pair['frame_index_zero_based']:03d}.npy"
        )
        np.save(pred_path, pred)

        row = {
            "frame_index_zero_based": pair["frame_index_zero_based"],
            "rgb_file": Path(pair["rgb_path"]).name,
            "depth_file": Path(pair["depth_path"]).name,
            "rgb_timestamp": pair["rgb_timestamp"],
            "depth_timestamp": pair["depth_timestamp"],
            "timestamp_diff_ms": pair["abs_delta_ms"],
            "gt_scale": DEPTH_SCALE,
            **metrics,
        }

        metric_rows.append(row)

        print(
            f"frame={row['frame_index_zero_based']} "
            f"AbsRel={metrics['AbsRel']:.6f} "
            f"RMSE={metrics['RMSE']:.6f} "
            f"delta1={metrics['delta1']:.6f} "
            f"valid_ratio={metrics['valid_ratio']:.6f}",
            flush=True,
        )

    write_csv(
        out_dir / "gt0_a_baseline_frames.csv",
        metric_rows,
    )

    metric_names = [
        "AbsRel", "RMSE", "delta1", "valid_ratio",
    ]

    mean_metrics = {
        name: float(np.mean([row[name] for row in metric_rows]))
        for name in metric_names
    }

    summary = {
        "experiment": "GT0 Bonn Crowd2 three-frame A baseline",
        "status": "PASS",
        "scope": (
            "three-frame GT calibration only; "
            "not a dataset-level performance claim"
        ),
        "sequence_root": str(root),
        "snapshot": str(snapshot),
        "checkpoint_sha256": hashlib.sha256(
            weights_path.read_bytes()
        ).hexdigest(),
        "depth_scale": DEPTH_SCALE,
        "max_time_diff_ms": MAX_TIME_DIFF_SEC * 1000,
        "rgb_start_index_zero_based": START_INDEX,
        "frame_count": FRAME_COUNT,
        "call_counts": stats,
        "mean_per_frame_metrics": mean_metrics,
        "max_timestamp_diff_ms": float(max(
            row["timestamp_diff_ms"] for row in metric_rows
        )),
        "inference_wall_time_sec_single_run": elapsed_sec,
        "peak_cuda_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 1024**3
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "frames": metric_rows,
    }

    summary_path = out_dir / "gt0_a_baseline_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Generate a human-readable report automatically.
    report = [
        "# GT0 实验报告：Bonn Crowd2 三帧 GT 校准与 A 基线",
        "",
        "## 1. 实验目的",
        "",
        "校验原始深度 PNG、RGB-GT 时间戳配对及 VeloDepth z-depth 输出，",
        "为后续 C/D 真实监督实验建立可复现的 A 基线。",
        "",
        "## 2. 协议",
        "",
        f"- GT 转换：`uint16 / {DEPTH_SCALE:.0f}` 米。",
        "- GT 零值视为无效；预测必须为有限正值。",
        "- RGB-GT：最近邻时间戳配对。",
        f"- 最大允许时间差：{MAX_TIME_DIFF_SEC * 1000:.1f} ms。",
        "- 指标按帧计算后取算术平均，与既有 Bonn P0 脚本一致。",
        "- 输出使用模型的 `depth`（相机 z-depth），不是 `distance`。",
        "- 模型：原始预训练权重，修复头未注册，repair off。",
        "- 推理：1 次关键帧、2 次 fast。",
        "",
        "## 3. A 基线结果",
        "",
        "| 指标 | 三帧逐帧均值 |",
        "|---|---:|",
    ]

    for name, value in mean_metrics.items():
        report.append(f"| {name} | {value:.6f} |")

    report += [
        "",
        "## 4. 逐帧结果",
        "",
        "| RGB 索引 | RGB 文件 | GT 文件 | 时间差 ms | AbsRel | RMSE | δ1 | 有效比例 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]

    for row in metric_rows:
        report.append(
            f"| {row['frame_index_zero_based']} "
            f"| {row['rgb_file']} "
            f"| {row['depth_file']} "
            f"| {row['timestamp_diff_ms']:.3f} "
            f"| {row['AbsRel']:.6f} "
            f"| {row['RMSE']:.6f} "
            f"| {row['delta1']:.6f} "
            f"| {row['valid_ratio']:.6f} |"
        )

    report += [
        "",
        "## 5. 运行与资源",
        "",
        f"- GPU：{summary['gpu']}",
        f"- 单次推理墙钟时间：{elapsed_sec:.3f} 秒（仅记录，不作为正式速度基准）",
        f"- 峰值已分配 CUDA 显存：{summary['peak_cuda_allocated_gib']:.3f} GiB",
        f"- 最大 RGB-GT 时间差：{summary['max_timestamp_diff_ms']:.3f} ms",
        "- 预训练权重 SHA256 已写入 JSON。",
        "",
        "## 6. 结论边界",
        "",
        "本报告仅验证三帧数据契约与原生 A 基线，不代表全序列性能，",
        "也不证明 C/D 修复头能提高精度。三帧校准通过后才进入 GT1。",
        "",
        "## 7. 输出文件",
        "",
        "- `gt0_rgb_depth_pairs.csv`：配对、路径、时间差及 GT 原始统计。",
        "- `gt0_a_baseline_frames.csv`：逐帧深度指标。",
        "- `gt0_a_baseline_summary.json`：运行环境、权重哈希及汇总。",
        "- `pred_depth_frame_*.npy`：三帧原始 A 预测 z-depth。",
    ]

    report_path = out_dir / "GT0_report.md"
    report_path.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n========== GT0 SUMMARY ==========", flush=True)
    print(json.dumps(mean_metrics, indent=2), flush=True)
    print("最大时间差(ms)：", summary["max_timestamp_diff_ms"], flush=True)
    print("峰值显存(GiB)：", summary["peak_cuda_allocated_gib"], flush=True)
    print("报告：", report_path, flush=True)
    print("GT0: PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
