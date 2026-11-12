import csv
from pathlib import Path

import cv2
import numpy as np

out = Path("outputs/gt0_validation")
pairs_file = out / "gt0_rgb_depth_pairs.csv"
rows_out = out / "GT0_P1_protocol_sensitivity.csv"
report_out = out / "GT0_P1_report.md"

with pairs_file.open(encoding="utf-8-sig", newline="") as f:
    pairs = list(csv.DictReader(f))

results = []

for row in pairs:
    idx = int(row["frame_index_zero_based"])
    raw = cv2.imread(
        row["depth_path"], cv2.IMREAD_UNCHANGED
    )
    pred_path = out / f"pred_depth_frame_{idx:03d}.npy"

    if raw is None:
        raise FileNotFoundError(row["depth_path"])
    if raw.dtype != np.uint16 or raw.ndim != 2:
        raise ValueError(
            f"GT 格式错误：{row['depth_path']} "
            f"{raw.dtype} {raw.shape}"
        )

    pred = np.load(pred_path).astype(np.float64)
    if pred.shape != raw.shape:
        raise ValueError(
            f"尺寸不匹配：GT={raw.shape}, pred={pred.shape}"
        )

    raw_valid = raw > 0
    base = raw_valid & np.isfinite(pred) & (pred > 0)

    for scale in (5000.0, 1000.0):
        gt = raw.astype(np.float64) / scale

        # 两种掩码：不设最大深度、限制在 10 m 内。
        for max_depth in (None, 10.0):
            mask = base & (gt >= 0.001)

            if max_depth is not None:
                mask &= gt <= max_depth

            if not mask.any():
                raise RuntimeError(
                    f"没有有效像素：frame={idx}, "
                    f"scale={scale}, max_depth={max_depth}"
                )

            g = gt[mask]
            p = pred[mask]
            ratio = np.maximum(g / p, p / g)

            result = {
                "frame_index": idx,
                "depth_scale": int(scale),
                "max_depth_m": (
                    "none" if max_depth is None
                    else max_depth
                ),
                "raw_gt_median_m": float(
                    np.median(gt[raw_valid])
                ),
                "raw_gt_max_m": float(
                    np.max(gt[raw_valid])
                ),
                "fraction_gt_within_10m": float(
                    ((gt <= 10.0) & raw_valid).sum()
                    / raw_valid.sum()
                ),
                "valid_pixel_ratio": float(mask.mean()),
                "valid_pixel_count": int(mask.sum()),
                "AbsRel": float(
                    np.mean(np.abs(g - p) / g)
                ),
                "RMSE_m": float(
                    np.sqrt(np.mean((g - p) ** 2))
                ),
                "delta1": float(np.mean(ratio < 1.25)),
            }
            results.append(result)

            print(
                f"frame={idx} scale={int(scale)} "
                f"max_depth={max_depth} "
                f"median_gt={result['raw_gt_median_m']:.3f}m "
                f"max_gt={result['raw_gt_max_m']:.3f}m "
                f"AbsRel={result['AbsRel']:.6f} "
                f"RMSE={result['RMSE_m']:.6f} "
                f"delta1={result['delta1']:.6f} "
                f"valid={result['valid_pixel_ratio']:.4f}"
            )

with rows_out.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

lines = [
    "# GT0-P1：深度缩放与有效像素掩码敏感性审计",
    "",
    "本实验复用 GT0 原始预测，不重新运行模型。",
    "不同缩放因子的指标仅用于敏感性分析，不能单凭",
    "哪个指标更好来决定正确的深度单位。",
    "",
    "## 各帧结果",
    "",
    "| 帧 | Scale | 最大深度 | GT 中位数(m) | GT 最大值(m) | AbsRel | RMSE(m) | δ1 | 有效比例 |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]

for r in results:
    lines.append(
        f"| {r['frame_index']} | {r['depth_scale']} "
        f"| {r['max_depth_m']} "
        f"| {r['raw_gt_median_m']:.4f} "
        f"| {r['raw_gt_max_m']:.4f} "
        f"| {r['AbsRel']:.6f} "
        f"| {r['RMSE_m']:.6f} "
        f"| {r['delta1']:.6f} "
        f"| {r['valid_pixel_ratio']:.6f} |"
    )

lines += [
    "",
    "## 结论边界",
    "",
    "- 本实验不负责决定原始 PNG 的正确编码。",
    "- 需要结合数据集原始编码或转换源码确定最终协议。",
    "- 正式评测前必须固定缩放、裁剪、深度范围与掩码。",
]

report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n输出：", rows_out)
print("报告：", report_out)
