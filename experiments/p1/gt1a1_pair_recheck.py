#!/usr/bin/env python3
"""GT1-A1: ordered one-to-one RGB/GT timestamp matching and metric recheck.

No model inference is performed. Existing GT1 predictions are reused in RGB index order.
Objective: maximize the number of valid matches within tolerance, then minimize total
absolute timestamp difference, with monotonic one-to-one matching.
"""
import csv
import json
import os
import sys
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np

DEPTH_SCALE = 5000.0
DEFAULT_TOLERANCE_MS = 20.0
EPS = 1e-9


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
    rows = []
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
        rows.append({
            "timestamp": timestamp,
            "raw_path": fields[1],
            "path": resolve_path(root, fields[1], subdir),
            "line_number": line_number,
        })
    if not rows:
        raise RuntimeError(f"索引没有有效记录：{index_path}")
    return rows


def write_csv(path: Path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        if not fieldnames:
            f.write("\n")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def ordered_one_to_one_match(rgb_records, depth_records, tolerance_ms):
    """Max-cardinality, min-total-|delta| ordered matching via dynamic programming."""
    # A physical GT image may only be used once, even if its path appears more than once in depth.txt.
    path_to_record = {}
    duplicate_index_path_rows = 0
    for rec in depth_records:
        if rec["path"] is None:
            continue
        key = str(rec["path"])
        if key in path_to_record:
            duplicate_index_path_rows += 1
            # Keep the first index occurrence deterministically.
            continue
        path_to_record[key] = rec
    valid_depths = sorted(path_to_record.values(), key=lambda r: (r["timestamp"], r["line_number"]))

    rgb_times = np.asarray([r["timestamp"] for r in rgb_records], dtype=np.float64)
    depth_times = np.asarray([r["timestamp"] for r in valid_depths], dtype=np.float64)
    n, m = len(rgb_records), len(valid_depths)
    # Count is the primary objective; total absolute timestamp error in ms is secondary.
    count = np.zeros((n + 1, m + 1), dtype=np.int16)
    cost = np.zeros((n + 1, m + 1), dtype=np.float64)
    # action: 1=skip RGB, 2=skip depth, 3=match
    action = np.zeros((n + 1, m + 1), dtype=np.uint8)

    for i in range(1, n + 1):
        rt = rgb_times[i - 1]
        for j in range(1, m + 1):
            best_count = int(count[i - 1, j])
            best_cost = float(cost[i - 1, j])
            best_action = 1

            c2 = int(count[i, j - 1])
            d2 = float(cost[i, j - 1])
            if c2 > best_count or (c2 == best_count and d2 < best_cost - EPS):
                best_count, best_cost, best_action = c2, d2, 2

            delta_ms = abs(float(depth_times[j - 1] - rt)) * 1000.0
            if delta_ms <= tolerance_ms + EPS:
                c3 = int(count[i - 1, j - 1]) + 1
                d3 = float(cost[i - 1, j - 1]) + delta_ms
                if (c3 > best_count or
                    (c3 == best_count and d3 < best_cost - EPS) or
                    (c3 == best_count and abs(d3 - best_cost) <= EPS)):
                    best_count, best_cost, best_action = c3, d3, 3

            count[i, j] = best_count
            cost[i, j] = best_cost
            action[i, j] = best_action

    matches = {}
    i, j = n, m
    while i > 0 and j > 0:
        a = int(action[i, j])
        if a == 3:
            matches[i - 1] = j - 1
            i -= 1
            j -= 1
        elif a == 1:
            i -= 1
        elif a == 2:
            j -= 1
        else:
            raise RuntimeError(f"DP 回溯动作异常：i={i}, j={j}, action={a}")

    # Explicit invariants: unique GT, monotonic assignment, within tolerance.
    assigned = [matches[k] for k in sorted(matches)]
    if len(assigned) != len(set(assigned)):
        raise AssertionError("一对一约束失败：GT 被重复分配")
    if any(assigned[k] >= assigned[k + 1] for k in range(len(assigned) - 1)):
        raise AssertionError("单调约束失败：GT 时间顺序发生逆序")
    for rgb_idx, depth_idx in matches.items():
        delta_ms = abs(rgb_times[rgb_idx] - depth_times[depth_idx]) * 1000.0
        if delta_ms > tolerance_ms + 1e-6:
            raise AssertionError(f"时间差超容差：RGB {rgb_idx}, delta={delta_ms:.6f} ms")

    return valid_depths, matches, duplicate_index_path_rows, float(cost[n, m])


def calculate_metrics(gt_raw, pred):
    if gt_raw.dtype != np.uint16 or gt_raw.ndim != 2:
        raise RuntimeError(f"GT 格式错误：dtype={gt_raw.dtype}, shape={gt_raw.shape}")
    if gt_raw.shape != pred.shape:
        raise RuntimeError(f"GT/预测尺寸不一致：{gt_raw.shape} vs {pred.shape}")
    gt = gt_raw.astype(np.float32) / DEPTH_SCALE
    valid = (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    if not valid.any():
        raise RuntimeError("GT 与预测没有共同有效像素")
    g = gt[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)
    absrel_values = np.abs(g - p) / g
    sqerr_values = (g - p) ** 2
    ratio = np.maximum(g / p, p / g)
    return {
        "AbsRel": float(absrel_values.mean()),
        "RMSE": float(np.sqrt(sqerr_values.mean())),
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
        "_absrel_sum": float(absrel_values.sum()),
        "_sqerr_sum": float(sqerr_values.sum()),
        "_delta1_count": int(np.count_nonzero(ratio < 1.25)),
        "_gt_valid_pixels": int(np.count_nonzero(gt > 0)),
    }


def fnum(x, digits=6):
    return "N/A" if x is None else f"{float(x):.{digits}f}"


def main():
    started = time.time()
    root = Path(os.environ.get(
        "BONN_SEQUENCE_ROOT",
        "/mnt/d/SVR-Depth/SVR-Depth/data/bonn/rgbd_bonn_crowd2/rgbd_bonn_crowd2",
    )).resolve()
    baseline_dir = Path(os.environ.get("GT1_OUT_DIR", "outputs/gt1_validation")).resolve()
    out_dir = Path(os.environ.get("GT1A1_OUT_DIR", "outputs/gt1a1_validation")).resolve()
    pred_path = Path(os.environ.get("GT1_PRED_PATH", str(baseline_dir / "gt1_a_predictions.npy"))).resolve()
    tolerance_ms = float(os.environ.get("GT1A1_TOLERANCE_MS", str(DEFAULT_TOLERANCE_MS)))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("GT1-A1: one-to-one timestamp pairing + metric recheck", flush=True)
    print("Sequence root:", root, flush=True)
    print("Baseline dir:", baseline_dir, flush=True)
    print("Prediction array:", pred_path, flush=True)
    print("Tolerance (ms):", tolerance_ms, flush=True)

    required = [baseline_dir / "gt1_a_baseline_summary.json",
                baseline_dir / "gt1_rgb_depth_pairs.csv",
                baseline_dir / "gt1_a_baseline_frames.csv",
                root / "rgb.txt", root / "depth.txt", pred_path]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("缺少 GT1-A1 必需输入文件：\n" + "\n".join(missing))

    with (baseline_dir / "gt1_a_baseline_summary.json").open(encoding="utf-8") as f:
        old_summary = json.load(f)
    old_pairs = read_csv(baseline_dir / "gt1_rgb_depth_pairs.csv")
    old_pair_by_idx = {int(r["frame_index_zero_based"]): r for r in old_pairs}

    rgb_records = read_index(root, "rgb.txt", "rgb")
    depth_records = read_index(root, "depth.txt", "depth")
    if any(rgb_records[i]["timestamp"] <= rgb_records[i-1]["timestamp"] for i in range(1, len(rgb_records))):
        raise RuntimeError("rgb.txt 时间戳非严格递增，停止匹配")
    missing_rgb = [r for r in rgb_records if r["path"] is None]
    if missing_rgb:
        raise RuntimeError(f"存在 {len(missing_rgb)} 个 RGB 文件缺失，停止复核")

    preds = np.load(pred_path, mmap_mode="r")
    if preds.ndim != 3:
        raise RuntimeError(f"预测数组应为 (N,H,W)，实际 shape={preds.shape}")
    if preds.shape[0] != len(rgb_records):
        raise RuntimeError(
            f"预测帧数与 RGB 索引不一致：pred={preds.shape[0]}, rgb={len(rgb_records)}；"
            "禁止猜测预测槽位映射"
        )

    valid_depths, matches, duplicate_index_path_rows, total_delta_ms = ordered_one_to_one_match(
        rgb_records, depth_records, tolerance_ms
    )
    used_depth_indices = set(matches.values())

    pair_rows = []
    unmatched_rgb_rows = []
    frame_rows = []
    changed_pair_rows = []
    macro_metric_rows = []
    total_valid = total_pixels = total_gt_valid = 0
    sum_absrel = sum_sqerr = 0.0
    delta1_count = 0
    abs_deltas = []

    for rgb_idx, rgb in enumerate(rgb_records):
        depth_idx = matches.get(rgb_idx)
        gt_rec = valid_depths[depth_idx] if depth_idx is not None else None
        delta_signed = (gt_rec["timestamp"] - rgb["timestamp"]) * 1000.0 if gt_rec else None
        delta_abs = abs(delta_signed) if delta_signed is not None else None
        pair = {
            "frame_index_zero_based": rgb_idx,
            "prediction_slot": rgb_idx,
            "rgb_timestamp": rgb["timestamp"],
            "rgb_path": str(rgb["path"]),
            "depth_timestamp": gt_rec["timestamp"] if gt_rec else "",
            "depth_path": str(gt_rec["path"]) if gt_rec else "",
            "signed_delta_ms": delta_signed if delta_signed is not None else "",
            "abs_delta_ms": delta_abs if delta_abs is not None else "",
            "gt_match_status": "MATCHED" if gt_rec else "NO_GT_WITHIN_TOLERANCE_OR_UNASSIGNED",
            "gt_raw_index_line": gt_rec["line_number"] if gt_rec else "",
            "gt_reused": False,
        }
        pair_rows.append(pair)
        if gt_rec:
            abs_deltas.append(delta_abs)
            gt_raw = cv2.imread(str(gt_rec["path"]), cv2.IMREAD_UNCHANGED)
            if gt_raw is None:
                raise RuntimeError(f"无法读取 GT：{gt_rec['path']}")
            pred = np.asarray(preds[rgb_idx], dtype=np.float32)
            met = calculate_metrics(gt_raw, pred)
            row = {
                "frame_index_zero_based": rgb_idx,
                "prediction_slot": rgb_idx,
                "rgb_timestamp": rgb["timestamp"],
                "rgb_file": rgb["path"].name,
                "gt_file": gt_rec["path"].name,
                "timestamp_diff_ms": delta_abs,
                "gt_match_status": "MATCHED",
                "gt_reused": False,
                "AbsRel": met["AbsRel"], "RMSE": met["RMSE"], "delta1": met["delta1"],
                "valid_ratio": met["valid_ratio"], "gt_valid_ratio": met["gt_valid_ratio"],
                "valid_pixels": met["valid_pixels"], "total_pixels": met["total_pixels"],
                "median_gt_m": met["median_gt_m"], "median_pred_m": met["median_pred_m"],
                "gt_min_m": met["gt_min_m"], "gt_max_m": met["gt_max_m"],
                "pred_min_m": met["pred_min_m"], "pred_max_m": met["pred_max_m"],
                "metrics_status": "EVALUATED",
            }
            macro_metric_rows.append(row)
            total_valid += met["valid_pixels"]
            total_pixels += met["total_pixels"]
            total_gt_valid += met["_gt_valid_pixels"]
            sum_absrel += met["_absrel_sum"]
            sum_sqerr += met["_sqerr_sum"]
            delta1_count += met["_delta1_count"]
        else:
            row = {
                "frame_index_zero_based": rgb_idx,
                "prediction_slot": rgb_idx,
                "rgb_timestamp": rgb["timestamp"],
                "rgb_file": rgb["path"].name,
                "gt_file": "", "timestamp_diff_ms": "",
                "gt_match_status": "NO_GT_WITHIN_TOLERANCE_OR_UNASSIGNED",
                "gt_reused": False,
                "AbsRel": "", "RMSE": "", "delta1": "", "valid_ratio": "", "gt_valid_ratio": "",
                "valid_pixels": "", "total_pixels": "", "median_gt_m": "", "median_pred_m": "",
                "gt_min_m": "", "gt_max_m": "", "pred_min_m": "", "pred_max_m": "",
                "metrics_status": "NO_UNIQUE_GT_ASSIGNMENT",
            }
            unmatched_rgb_rows.append({
                "frame_index_zero_based": rgb_idx,
                "rgb_timestamp": rgb["timestamp"],
                "rgb_path": str(rgb["path"]),
                "reason": "No unique monotonic GT assignment within tolerance",
            })
        frame_rows.append(row)

        old = old_pair_by_idx.get(rgb_idx, {})
        old_gt = old.get("depth_path", "")
        new_gt = pair["depth_path"]
        if old_gt != new_gt:
            changed_pair_rows.append({
                "frame_index_zero_based": rgb_idx,
                "rgb_timestamp": rgb["timestamp"],
                "old_depth_path": old_gt,
                "new_depth_path": new_gt,
                "old_abs_delta_ms": old.get("abs_delta_ms", ""),
                "new_abs_delta_ms": pair["abs_delta_ms"],
                "change_type": "PAIR_CHANGED" if old_gt and new_gt else ("BECAME_UNMATCHED" if old_gt else "NEW_MATCH"),
            })

    unused_gt_rows = []
    for j, gt_rec in enumerate(valid_depths):
        if j not in used_depth_indices:
            unused_gt_rows.append({
                "depth_timestamp": gt_rec["timestamp"],
                "depth_path": str(gt_rec["path"]),
                "depth_index_line": gt_rec["line_number"],
                "reason": "Not selected by maximum-cardinality monotonic matching",
            })

    if not macro_metric_rows:
        raise RuntimeError("一对一匹配后没有可评测帧")
    unique_assigned_paths = [r["depth_path"] for r in pair_rows if r["depth_path"]]
    if len(unique_assigned_paths) != len(set(unique_assigned_paths)):
        raise AssertionError("最终配对仍存在重复 GT 文件")

    metric_names = ["AbsRel", "RMSE", "delta1", "valid_ratio", "gt_valid_ratio"]
    macro = {name: float(np.mean([float(r[name]) for r in macro_metric_rows])) for name in metric_names}
    pixel_weighted = {
        "AbsRel": float(sum_absrel / total_valid),
        "RMSE": float(np.sqrt(sum_sqerr / total_valid)),
        "delta1": float(delta1_count / total_valid),
        "valid_ratio": float(total_valid / total_pixels),
    }

    old_macro = old_summary.get("mean_per_frame_metrics", {})
    old_pixel = old_summary.get("pixel_weighted_metrics", {})
    def diff_block(old_block, new_block):
        out = {}
        for k, v in new_block.items():
            if k in old_block:
                out[k] = float(v) - float(old_block[k])
        return out

    summary = {
        "experiment": "GT1-A1 ordered one-to-one timestamp pairing and metric recheck",
        "status": "PASS",
        "inference_reused": True,
        "prediction_path": str(pred_path),
        "sequence_root": str(root),
        "depth_scale": DEPTH_SCALE,
        "tolerance_ms": tolerance_ms,
        "matching_objective": "maximize match count, then minimize total absolute timestamp error; monotonic one-to-one",
        "rgb_index_records": len(rgb_records),
        "depth_index_records": len(depth_records),
        "valid_unique_gt_images": len(valid_depths),
        "missing_depth_files_in_index": sum(r["path"] is None for r in depth_records),
        "duplicate_depth_index_path_rows_collapsed": duplicate_index_path_rows,
        "matched_pairs": len(matches),
        "unmatched_rgb_frames": len(unmatched_rgb_rows),
        "unused_valid_gt_images": len(unused_gt_rows),
        "unique_gt_files_assigned": len(unique_assigned_paths),
        "duplicate_gt_assignments": len(unique_assigned_paths) - len(set(unique_assigned_paths)),
        "changed_pairs_vs_gt1": len(changed_pair_rows),
        "max_abs_timestamp_delta_ms": float(max(abs_deltas)) if abs_deltas else None,
        "mean_abs_timestamp_delta_ms": float(np.mean(abs_deltas)) if abs_deltas else None,
        "median_abs_timestamp_delta_ms": float(np.median(abs_deltas)) if abs_deltas else None,
        "total_abs_timestamp_delta_ms": total_delta_ms,
        "mean_per_frame_metrics": macro,
        "pixel_weighted_metrics": pixel_weighted,
        "global_gt_valid_ratio": float(total_gt_valid / total_pixels),
        "original_gt1_mean_per_frame_metrics": old_macro,
        "original_gt1_pixel_weighted_metrics": old_pixel,
        "delta_mean_per_frame_metrics_new_minus_old": diff_block(old_macro, macro),
        "delta_pixel_weighted_metrics_new_minus_old": diff_block(old_pixel, pixel_weighted),
        "runtime_sec": time.time() - started,
    }

    write_csv(out_dir / "gt1a1_pairs.csv", pair_rows)
    write_csv(out_dir / "gt1a1_frames.csv", frame_rows)
    write_csv(out_dir / "gt1a1_unmatched_rgb.csv", unmatched_rgb_rows,
              ["frame_index_zero_based", "rgb_timestamp", "rgb_path", "reason"])
    write_csv(out_dir / "gt1a1_unused_gt.csv", unused_gt_rows,
              ["depth_timestamp", "depth_path", "depth_index_line", "reason"])
    write_csv(out_dir / "gt1a1_changed_pairs.csv", changed_pair_rows,
              ["frame_index_zero_based", "rgb_timestamp", "old_depth_path", "new_depth_path", "old_abs_delta_ms", "new_abs_delta_ms", "change_type"])
    missing_depth_rows = [{"depth_index_line": r["line_number"], "timestamp": r["timestamp"], "raw_path": r["raw_path"]}
                          for r in depth_records if r["path"] is None]
    write_csv(out_dir / "gt1a1_missing_depth_files.csv", missing_depth_rows,
              ["depth_index_line", "timestamp", "raw_path"])
    (out_dir / "gt1a1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# GT1-A1 实验报告：一对一时间戳配对与指标复核", "",
        "## 1. 目的与方法", "",
        "本实验不重新运行模型、不重新训练，直接复用 GT1 的预测数组。RGB 与实际存在的 GT 图像按时间戳进行单调一对一匹配。优化目标是先最大化 20 ms 容差内的匹配数量，再最小化所有匹配的绝对时间差总和。每张 GT 图像最多使用一次。", "",
        f"- 序列：`{root}`",
        f"- 预测数组：`{pred_path}`",
        f"- GT 深度尺度：uint16 / {DEPTH_SCALE:.0f} 米",
        f"- 最大时间差：{tolerance_ms:.3f} ms",
        "- 评测掩码和指标定义沿用 GT1：GT>0、GT/预测有限且预测>0；无额外裁剪和深度上限。", "",
        "## 2. 配对审计", "",
        f"- RGB 记录：{len(rgb_records)}",
        f"- depth.txt 记录：{len(depth_records)}",
        f"- 实际存在且去重后的 GT 图像：{len(valid_depths)}",
        f"- 一对一匹配：{len(matches)}",
        f"- 未匹配 RGB：{len(unmatched_rgb_rows)}",
        f"- 未被使用的有效 GT：{len(unused_gt_rows)}",
        f"- 相对 GT1 原始配对发生变化的 RGB 行：{len(changed_pair_rows)}",
        f"- GT 重复分配数：{summary['duplicate_gt_assignments']}（必须为 0）",
        f"- 最大 / 平均 / 中位绝对时间差：{fnum(summary['max_abs_timestamp_delta_ms'], 3)} / {fnum(summary['mean_abs_timestamp_delta_ms'], 3)} / {fnum(summary['median_abs_timestamp_delta_ms'], 3)} ms", "",
        "## 3. 指标对比", "",
        "### 逐帧宏平均（每个已匹配帧等权）", "",
        "| 指标 | GT1 原始 | GT1-A1 一对一 | 新值 - 原值 |", "|---|---:|---:|---:|",
    ]
    for k in metric_names:
        ov = old_macro.get(k)
        nv = macro.get(k)
        dv = float(nv) - float(ov) if ov is not None else None
        report.append(f"| {k} | {fnum(ov)} | {fnum(nv)} | {fnum(dv)} |")
    report += ["", "### 像素加权指标", "", "| 指标 | GT1 原始 | GT1-A1 一对一 | 新值 - 原值 |", "|---|---:|---:|---:|"]
    for k in ["AbsRel", "RMSE", "delta1", "valid_ratio"]:
        ov = old_pixel.get(k)
        nv = pixel_weighted.get(k)
        dv = float(nv) - float(ov) if ov is not None else None
        report.append(f"| {k} | {fnum(ov)} | {fnum(nv)} | {fnum(dv)} |")
    report += [
        "", "## 4. 结论判定", "",
        "- 配对唯一性：PASS，所有被分配的 GT 路径唯一。",
        "- 时间顺序：PASS，配对随 RGB 顺序单调递增。",
        f"- 帧覆盖：{'PASS（所有 RGB 均匹配）' if not unmatched_rgb_rows else '需报告未匹配帧；不得静默删除'}。",
        "- 指标变化只反映 GT 配对修正与有效帧集合变化；预测本身没有改变。",
        "- 若存在未匹配 RGB，后续 A/C/D 对照必须使用相同的 GT1-A1 匹配帧集合。", "",
        "## 5. 输出文件", "",
        "- `gt1a1_summary.json`：协议、覆盖率和指标对比。",
        "- `gt1a1_pairs.csv`：所有 RGB 的新配对。",
        "- `gt1a1_frames.csv`：逐帧指标。",
        "- `gt1a1_unmatched_rgb.csv`：未匹配 RGB。",
        "- `gt1a1_unused_gt.csv`：未使用的有效 GT。",
        "- `gt1a1_changed_pairs.csv`：相对 GT1 原始配对发生变化的帧。",
        "- `gt1a1_missing_depth_files.csv`：depth.txt 中文件缺失记录。", "",
        "## 6. 运行边界", "",
        "本脚本没有重新推理或改变预测；结果有效性依赖 GT1 预测数组与原始 RGB 索引顺序一致。脚本已检查预测帧数与 RGB 记录数相等。",
    ]
    (out_dir / "GT1-A1_实验报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n========== GT1-A1 SUMMARY ==========")
    print(json.dumps({
        "matched_pairs": summary["matched_pairs"],
        "unmatched_rgb_frames": summary["unmatched_rgb_frames"],
        "unused_valid_gt_images": summary["unused_valid_gt_images"],
        "changed_pairs_vs_gt1": summary["changed_pairs_vs_gt1"],
        "duplicate_gt_assignments": summary["duplicate_gt_assignments"],
        "mean_per_frame_metrics": macro,
        "pixel_weighted_metrics": pixel_weighted,
    }, indent=2, ensure_ascii=False), flush=True)
    print("报告：", out_dir / "GT1-A1_实验报告.md", flush=True)
    print("输出目录：", out_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GT1-A1 FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise
