import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".png", ".tif", ".tiff", ".exr", ".jpg", ".jpeg"}


def resolve(root, raw, kind):
    p = Path(raw)
    candidates = [p] if p.is_absolute() else [
        root / p, root / kind / p,
        root / kind / p.name, root / p.name
    ]
    return next((x.resolve() for x in candidates if x.is_file()), None)


def parse_index(path, root, kind):
    records = []
    if not path.is_file():
        return records
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            ts = float(fields[0])
        except ValueError:
            continue
        if Path(fields[1]).suffix.lower() not in EXTS:
            continue
        records.append({
            "timestamp": ts,
            "raw_path": fields[1],
            "path": resolve(root, fields[1], kind)
        })
    return records


def parse_associations(path, root):
    rgb, depth = [], []
    if not path.is_file():
        return rgb, depth
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if len(f) < 4:
            continue
        try:
            rt, dt = float(f[0]), float(f[2])
        except ValueError:
            continue
        if (Path(f[1]).suffix.lower() not in EXTS or
                Path(f[3]).suffix.lower() not in EXTS):
            continue
        rgb.append({
            "timestamp": rt, "raw_path": f[1],
            "path": resolve(root, f[1], "rgb")
        })
        depth.append({
            "timestamp": dt, "raw_path": f[3],
            "path": resolve(root, f[3], "depth")
        })
    return rgb, depth


def image_stats(path):
    with Image.open(path) as im:
        a = np.asarray(im)

    result = {
        "path": str(path),
        "shape": list(a.shape),
        "dtype": str(a.dtype)
    }
    if a.ndim == 2 and np.issubdtype(a.dtype, np.number):
        x = a.astype(np.float64)
        finite = np.isfinite(x)
        valid = finite & (x > 0)
        v = x[valid]
        result.update({
            "raw_min": float(x[finite].min()) if finite.any() else None,
            "raw_max": float(x[finite].max()) if finite.any() else None,
            "zero_fraction": float(np.mean(x == 0)),
            "positive_pixels": int(valid.sum()),
            "positive_min": float(v.min()) if v.size else None,
            "positive_median": float(np.median(v)) if v.size else None,
            "positive_p95": float(np.percentile(v, 95)) if v.size else None,
            "positive_max": float(v.max()) if v.size else None
        })
    return result


def main():
    root = Path(os.environ["BONN_SEQUENCE_ROOT"]).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"数据目录不存在：{root}")

    out = Path("outputs/e2_s8_validation")
    out.mkdir(parents=True, exist_ok=True)

    print("E2-S8-S5-A GT AUDIT")
    print("时间：", datetime.now().astimezone().isoformat())
    print("数据目录：", root.resolve())
    print("目录内容：", sorted(p.name for p in root.iterdir())[:80])

    rgb = parse_index(root / "rgb.txt", root, "rgb")
    if not rgb:
        raise RuntimeError("rgb.txt 不存在或没有可解析的图像记录")

    candidates = {}
    for name in ("depth.txt", "groundtruth.txt"):
        records = parse_index(root / name, root, "depth")
        if records:
            candidates[name] = records

    assoc_rgb, assoc_depth = parse_associations(
        root / "associations.txt", root
    )

    print(f"RGB 记录：{len(rgb)}")
    print(f"RGB 路径存在：{sum(r['path'] is not None for r in rgb)}")

    for name, records in candidates.items():
        print(
            f"{name}：记录 {len(records)}，"
            f"路径存在 {sum(r['path'] is not None for r in records)}"
        )

    if "depth.txt" in candidates:
        depth_name, depth = "depth.txt", candidates["depth.txt"]
    elif any(r["path"] is not None for r in assoc_depth):
        depth_name, depth = "associations.txt", assoc_depth
        if not any(r["path"] is not None for r in rgb):
            rgb = assoc_rgb
    else:
        usable = [
            (name, records) for name, records in candidates.items()
            if any(r["path"] is not None for r in records)
        ]
        if not usable:
            raise RuntimeError(
                "没有可用的深度图索引。请检查 depth.txt、"
                "associations.txt 及实际目录结构。"
            )
        depth_name, depth = usable[0]

    rgb = sorted(
        [r for r in rgb if r["path"] is not None],
        key=lambda r: r["timestamp"]
    )
    depth = sorted(
        [r for r in depth if r["path"] is not None],
        key=lambda r: r["timestamp"]
    )

    if not rgb or not depth:
        raise RuntimeError("RGB 或深度图没有可解析的实际文件")

    dts = np.array([r["timestamp"] for r in depth])
    matches = []

    for r in rgb:
        j = int(np.searchsorted(dts, r["timestamp"]))
        choices = [i for i in (j - 1, j) if 0 <= i < len(depth)]
        i = min(choices, key=lambda k: abs(dts[k] - r["timestamp"]))
        d = depth[i]
        matches.append({
            "rgb_timestamp": r["timestamp"],
            "rgb_path": str(r["path"]),
            "depth_timestamp": d["timestamp"],
            "depth_path": str(d["path"]),
            "delta_seconds": abs(d["timestamp"] - r["timestamp"]),
            "depth_index": i
        })

    delta = np.array([m["delta_seconds"] for m in matches])
    reused = sum(
        n > 1 for n in Counter(m["depth_index"] for m in matches).values()
    )

    print("\n========== 时间戳匹配统计 ==========")
    print("RGB / 深度记录：", len(rgb), "/", len(depth))
    print("匹配数量：", len(matches))
    print("时间差 min / median / p95 / max（秒）：",
          float(delta.min()), float(np.median(delta)),
          float(np.percentile(delta, 95)), float(delta.max()))
    print("时间差不超过 5 ms：", int((delta <= .005).sum()), "/", len(delta))
    print("时间差不超过 20 ms：", int((delta <= .020).sum()), "/", len(delta))
    print("被多个 RGB 帧重复匹配的深度帧数：", reused)

    # 检查首帧、中间帧、末帧及 S4 使用的第 9–11 帧。
    indices = sorted(set([0, 8, 9, 10, len(matches)//2, len(matches)-1]))
    samples = []
    mismatches = 0

    print("\n========== 样本图像检查 ==========")
    for i in indices:
        if i >= len(matches):
            continue
        m = matches[i]
        rs = image_stats(Path(m["rgb_path"]))
        ds = image_stats(Path(m["depth_path"]))
        same_hw = rs["shape"][:2] == ds["shape"][:2]
        mismatches += int(not same_hw)

        sample = {
            **m, "rgb_stats": rs, "depth_stats": ds,
            "same_height_width": same_hw
        }
        samples.append(sample)

        print(
            f"sample={i}, dt={m['delta_seconds']:.6f}s, "
            f"RGB={Path(m['rgb_path']).name}, "
            f"depth={Path(m['depth_path']).name}"
        )
        print("  RGB:", rs["shape"], rs["dtype"])
        print("  Depth:", ds)
        print("  高宽一致：", same_hw)

    report = {
        "experiment": "E2-S8-S5-A GT index audit",
        "timestamp": datetime.now().astimezone().isoformat(),
        "sequence_root": str(root.resolve()),
        "rgb_records": len(rgb),
        "depth_index_used": depth_name,
        "depth_records": len(depth),
        "matched_frames": len(matches),
        "timestamp_delta_seconds": {
            "min": float(delta.min()),
            "median": float(np.median(delta)),
            "p95": float(np.percentile(delta, 95)),
            "max": float(delta.max()),
            "within_5ms": int((delta <= .005).sum()),
            "within_20ms": int((delta <= .020).sum())
        },
        "reused_depth_frames": reused,
        "sample_shape_mismatches": mismatches,
        "samples": samples,
        "depth_units": "NOT CONFIRMED; raw values only",
        "warning": (
            "Nearest timestamp matches are diagnostic only. "
            "Confirm synchronization, geometric alignment, depth scale "
            "and invalid-pixel semantics before using GT for training."
        )
    }

    jp = out / "s5_gt_index_audit.json"
    tp = out / "s5_gt_index_audit.txt"
    jp.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                  encoding="utf-8")

    tp.write_text(
        "\n".join([
            "E2-S8-S5-A GT INDEX AUDIT",
            f"Root: {root.resolve()}",
            f"RGB records: {len(rgb)}",
            f"Depth records: {len(depth)} ({depth_name})",
            f"Matched frames: {len(matches)}",
            f"Median / p95 / max delta: {np.median(delta):.6f} / "
            f"{np.percentile(delta, 95):.6f} / {delta.max():.6f} sec",
            f"Within 20 ms: {int((delta <= .020).sum())}/{len(delta)}",
            f"Sample shape mismatches: {mismatches}",
            "Depth units: NOT CONFIRMED",
            f"JSON: {jp}"
        ]) + "\n",
        encoding="utf-8"
    )

    print("\n报告已保存：", jp)
    print("摘要已保存：", tp)
    print("S5-A AUDIT: PASS (index audit completed; GT validity not yet certified)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"S5-A AUDIT FAILED: {exc}", file=sys.stderr)
        raise
