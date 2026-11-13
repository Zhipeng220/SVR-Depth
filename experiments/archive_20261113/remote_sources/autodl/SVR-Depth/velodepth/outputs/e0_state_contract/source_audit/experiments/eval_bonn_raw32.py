from pathlib import Path
import csv
import cv2
import numpy as np

DATA = Path("/root/autodl-tmp/datasets/rgbd_bonn_crowd2")
PRED = Path("outputs/p0_bonn_crowd2_32")
SAVE = Path("outputs/baseline/bonn_crowd2_32.csv")
MAX_DIFF = 0.02  # RGB 与深度时间戳最大偏差：20ms

def read_index(filename):
    records = []
    for line in (DATA / filename).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        records.append((float(parts[0]), DATA / parts[1]))
    return records

rgb_index = read_index("rgb.txt")
depth_index = sorted(read_index("depth.txt"))
rgb_times = {p.name: t for t, p in rgb_index}
depth_times = np.array([t for t, p in depth_index])

# 与 VeloDepth demo.py 的文件夹读取顺序一致
rgb_files = sorted((DATA / "rgb").glob("*.png"))
pred_files = sorted(PRED.glob("*_depth.npy"))

assert len(pred_files) == 32, f"预测文件数异常: {len(pred_files)}"
assert len(rgb_files) >= 32

rows = []
unmatched = []

for i, (rgb_path, pred_path) in enumerate(
    zip(rgb_files[:32], pred_files)
):
    assert rgb_path.name in rgb_times, rgb_path.name
    rgb_t = rgb_times[rgb_path.name]

    # 匹配时间戳最近的深度真值
    j = int(np.argmin(np.abs(depth_times - rgb_t)))
    depth_t, depth_path = depth_index[j]
    dt = abs(depth_t - rgb_t)

    if dt > MAX_DIFF:
        unmatched.append((i, dt))
        continue

    gt_raw = cv2.imread(
        str(depth_path), cv2.IMREAD_UNCHANGED
    )
    assert gt_raw is not None
    assert gt_raw.dtype == np.uint16

    gt = gt_raw.astype(np.float32) / 5000.0
    pred = np.load(pred_path).squeeze().astype(np.float32)

    assert pred.shape == gt.shape, (
        f"Frame {i}: pred={pred.shape}, gt={gt.shape}"
    )

    valid = (
        (gt > 0) &
        np.isfinite(gt) &
        np.isfinite(pred) &
        (pred > 0)
    )

    if valid.sum() == 0:
        unmatched.append((i, "no valid pixels"))
        continue

    g = gt[valid]
    p = pred[valid]

    abs_rel = np.mean(np.abs(g - p) / g)
    rmse = np.sqrt(np.mean((g - p) ** 2))
    ratio = np.maximum(g / p, p / g)
    delta1 = np.mean(ratio < 1.25)

    rows.append({
        "frame": i,
        "rgb_file": rgb_path.name,
        "depth_file": depth_path.name,
        "timestamp_diff_ms": round(dt * 1000, 3),
        "valid_ratio": float(valid.mean()),
        "AbsRel": float(abs_rel),
        "RMSE": float(rmse),
        "delta1": float(delta1),
        "median_gt": float(np.median(g)),
        "median_pred": float(np.median(p)),
    })

assert rows, "没有成功匹配的有效帧，请检查时间戳"

SAVE.parent.mkdir(parents=True, exist_ok=True)

with SAVE.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n===== Bonn crowd2 P0 Diagnostic =====")
print("预测帧数:", len(pred_files))
print("成功匹配:", len(rows))
print("未匹配:", unmatched[:10])

for metric in ["AbsRel", "RMSE", "delta1", "valid_ratio"]:
    values = [r[metric] for r in rows]
    print(f"{metric}: {np.mean(values):.6f}")

print("最大匹配时间差(ms):",
      max(r["timestamp_diff_ms"] for r in rows))
print("GT深度中位数(米):",
      np.median([r["median_gt"] for r in rows]))
print("预测深度中位数(米):",
      np.median([r["median_pred"] for r in rows]))
print("结果已保存:", SAVE)
