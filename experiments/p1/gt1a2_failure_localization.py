#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT1-A2: diagnose temporal degradation from GT1-A1 per-frame metrics.

No model inference, no GPU, no training. Uses the frozen GT1-A1 one-to-one pairing
and per-frame metrics CSV. The thresholds below are transparent screening rules,
not statistical significance tests and not proof of causality.
"""
import argparse
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

REQUIRED = [
    "frame_index_zero_based", "rgb_timestamp", "AbsRel", "RMSE", "delta1",
    "valid_ratio", "median_gt_m", "median_pred_m", "metrics_status",
]
DEFAULT_INPUT = Path("outputs/gt1a1_validation/gt1a1_frames.csv")
DEFAULT_OUTPUT = Path("outputs/gt1a2_validation")
WINDOWS = (25, 50)
PERSISTENCE = 10
BASELINE_FRAMES = 25


def finite_float(value, field, row_idx):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"第 {row_idx} 行字段 {field} 不是数值：{value!r}")
    if not math.isfinite(x):
        raise ValueError(f"第 {row_idx} 行字段 {field} 非有限值：{value!r}")
    return x


def read_frames(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到 GT1-A1 逐帧结果：{path}\n"
            "请先运行 GT1-A1，并确认 outputs/gt1a1_validation/gt1a1_frames.csv 存在。"
        )
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in REQUIRED if c not in fields]
        if missing:
            raise ValueError(f"输入 CSV 缺少必要列：{missing}; 实际列：{fields}")
        raw_rows = list(reader)

    rows = []
    for line_no, raw in enumerate(raw_rows, 2):
        status = (raw.get("metrics_status") or "").strip().upper()
        match_status = (raw.get("gt_match_status") or "").strip().upper()
        if status != "EVALUATED":
            raise ValueError(
                f"第 {line_no} 行 metrics_status={status!r}，"
                "GT1-A2 要求所有连续帧均有有效指标，不允许跨缺失帧计算窗口。"
            )
        if match_status and match_status != "MATCHED":
            raise ValueError(f"第 {line_no} 行 GT 匹配状态不是 MATCHED：{match_status}")
        idx = int(raw["frame_index_zero_based"])
        row = {
            "frame_index_zero_based": idx,
            "rgb_timestamp": finite_float(raw["rgb_timestamp"], "rgb_timestamp", line_no),
            "AbsRel": finite_float(raw["AbsRel"], "AbsRel", line_no),
            "RMSE": finite_float(raw["RMSE"], "RMSE", line_no),
            "delta1": finite_float(raw["delta1"], "delta1", line_no),
            "valid_ratio": finite_float(raw["valid_ratio"], "valid_ratio", line_no),
            "median_gt_m": finite_float(raw["median_gt_m"], "median_gt_m", line_no),
            "median_pred_m": finite_float(raw["median_pred_m"], "median_pred_m", line_no),
            "gt_file": raw.get("gt_file", ""),
            "rgb_file": raw.get("rgb_file", ""),
            "timestamp_diff_ms": raw.get("timestamp_diff_ms", ""),
        }
        if row["median_gt_m"] <= 0 or row["median_pred_m"] <= 0:
            raise ValueError(
                f"第 {line_no} 行深度中位数非正，不能计算深度比值："
                f"GT={row['median_gt_m']}, pred={row['median_pred_m']}"
            )
        row["pred_gt_median_ratio"] = row["median_pred_m"] / row["median_gt_m"]
        rows.append(row)

    if not rows:
        raise ValueError("输入 CSV 没有可评测帧。")
    rows.sort(key=lambda r: r["frame_index_zero_based"])
    indices = [r["frame_index_zero_based"] for r in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("输入 CSV 存在重复 RGB 帧索引。")
    expected = list(range(indices[0], indices[0] + len(indices)))
    if indices != expected:
        raise ValueError(
            "帧索引存在缺口或非连续区间，拒绝跨缺口计算滑动窗口。"
            f"前几个索引={indices[:10]}，最后几个索引={indices[-10:]}"
        )
    if indices[0] != 0:
        raise ValueError(f"预期从 RGB 索引 0 开始，实际从 {indices[0]} 开始。")
    if len(rows) < max(WINDOWS) + PERSISTENCE - 1:
        raise ValueError(f"有效帧数不足以计算 50 帧窗口和持续告警：{len(rows)}")
    return rows, fields


def mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


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


def rolling_rows(rows, window):
    result = []
    for end_pos in range(window - 1, len(rows)):
        subset = rows[end_pos - window + 1:end_pos + 1]
        ratios = [r["pred_gt_median_ratio"] for r in subset]
        gt_medians = [r["median_gt_m"] for r in subset]
        pred_medians = [r["median_pred_m"] for r in subset]
        result.append({
            "window_size": window,
            "window_start_index": subset[0]["frame_index_zero_based"],
            "window_end_index": subset[-1]["frame_index_zero_based"],
            "window_start_timestamp": subset[0]["rgb_timestamp"],
            "window_end_timestamp": subset[-1]["rgb_timestamp"],
            "n_frames": len(subset),
            "AbsRel_mean": mean([r["AbsRel"] for r in subset]),
            "RMSE_frame_mean_m": mean([r["RMSE"] for r in subset]),
            "delta1_mean": mean([r["delta1"] for r in subset]),
            "valid_ratio_mean": mean([r["valid_ratio"] for r in subset]),
            "median_frame_pred_gt_ratio": median(ratios),
            "mean_frame_pred_gt_ratio": mean(ratios),
            "ratio_of_window_medians": median(pred_medians) / median(gt_medians),
            "median_gt_m": median(gt_medians),
            "median_pred_m": median(pred_medians),
        })
    return result


def first_sustained(rows, field, predicate, persistence=PERSISTENCE):
    flags = [bool(predicate(float(r[field]))) for r in rows]
    for i in range(0, len(flags) - persistence + 1):
        if all(flags[i:i + persistence]):
            first = rows[i]
            confirm = rows[i + persistence - 1]
            return {
                "first_alert_window_end_index": int(first["window_end_index"]),
                "first_alert_window_start_index": int(first["window_start_index"]),
                "confirmation_window_end_index": int(confirm["window_end_index"]),
                "confirmation_window_start_index": int(confirm["window_start_index"]),
                "persistence_windows": persistence,
                "first_alert_value": float(first[field]),
                "confirmation_value": float(confirm[field]),
            }
    return None


def fmt(x, digits=4):
    if x is None:
        return "未触发"
    return f"{x:.{digits}f}"


def find_evidence(project_root: Path, input_fields):
    names = [x.lower() for x in input_fields]
    diagnostic_terms = ("flow", "state", "confidence", "feature", "warp", "propagat", "motion", "age")
    per_frame_diagnostic_columns = [
        field for field in input_fields
        if any(term in field.lower() for term in diagnostic_terms)
    ]
    log_path = project_root / "outputs/gt1_validation/gt1_run.log"
    log_excerpt = []
    keyframe_calls = None
    fast_calls = None
    log_exists = log_path.is_file()
    if log_exists:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for pattern, key in [
            (r"keyframe_calls['\"]?\s*:\s*(\d+)", "keyframe_calls"),
            (r"fast_calls['\"]?\s*:\s*(\d+)", "fast_calls"),
        ]:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if m:
                if key == "keyframe_calls":
                    keyframe_calls = int(m.group(1))
                else:
                    fast_calls = int(m.group(1))
        for line in text.splitlines():
            low = line.lower()
            if any(x in low for x in ("keyframe_calls", "fast_calls", "frame ", "inference_wall_time_sec")):
                log_excerpt.append(line.strip()[:300])
    return {
        "gt1_run_log_exists": log_exists,
        "keyframe_calls_from_log": keyframe_calls,
        "fast_calls_from_log": fast_calls,
        "per_frame_flow_or_state_diagnostic_columns": per_frame_diagnostic_columns,
        "conclusion": (
            "现有逐帧 CSV 提供深度误差、有效像素比例和预测/GT 深度中位数，"
            "但没有逐帧光流质量、传播状态差异或状态年龄等诊断量。"
            "GT1 日志主要记录关键帧/fast 调用、时间和汇总结果；不能仅凭这些日志将退化归因于某个内部机制。"
        ),
        "log_excerpt": log_excerpt[:20],
    }


def save_plots(out_dir: Path, all_rolling, abs_threshold, ratio_threshold):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"plots_created": False, "reason": f"matplotlib 不可用：{exc!r}"}

    by_window = {w: [r for r in all_rolling if r["window_size"] == w] for w in WINDOWS}
    try:
        plt.figure(figsize=(12, 5))
        for w in WINDOWS:
            rows = by_window[w]
            plt.plot(
                [r["window_end_index"] for r in rows],
                [r["AbsRel_mean"] for r in rows],
                label=f"{w}-frame rolling AbsRel",
            )
        plt.axhline(abs_threshold, linestyle="--", label=f"screening threshold={abs_threshold:.3f}")
        plt.xlabel("RGB frame index (window end)")
        plt.ylabel("Mean per-frame AbsRel")
        plt.title("GT1-A2: rolling AbsRel")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "gt1a2_rolling_absrel.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 5))
        for w in WINDOWS:
            rows = by_window[w]
            plt.plot(
                [r["window_end_index"] for r in rows],
                [r["median_frame_pred_gt_ratio"] for r in rows],
                label=f"{w}-frame rolling median ratio",
            )
        plt.axhline(1.0, linestyle=":", label="ideal ratio=1.0")
        plt.axhline(ratio_threshold, linestyle="--", label=f"20% drop threshold={ratio_threshold:.3f}")
        plt.axhline(0.75, linestyle="-.", label="severe underestimation threshold=0.75")
        plt.xlabel("RGB frame index (window end)")
        plt.ylabel("Median of per-frame median(pred)/median(GT)")
        plt.title("GT1-A2: rolling depth-scale ratio")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "gt1a2_depth_ratio.png", dpi=150)
        plt.close()
        return {"plots_created": True, "files": ["gt1a2_rolling_absrel.png", "gt1a2_depth_ratio.png"]}
    except Exception as exc:
        return {"plots_created": False, "reason": f"绘图失败：{exc!r}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="GT1-A1 逐帧 CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="GT1-A2 输出目录")
    parser.add_argument("--persistence", type=int, default=PERSISTENCE, help="持续窗口数，默认 10")
    parser.add_argument("--baseline-frames", type=int, default=BASELINE_FRAMES, help="基线帧数，默认前 25 帧")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path.cwd()

    print("GT1-A2: rolling-window failure localization", flush=True)
    print("Input:", input_path, flush=True)
    print("Output:", out_dir, flush=True)
    print("No inference / no training / no GPU computation.", flush=True)

    rows, fields = read_frames(input_path)
    if args.baseline_frames < 5 or args.baseline_frames >= len(rows):
        raise ValueError(f"baseline-frames 参数不合理：{args.baseline_frames}")
    if args.persistence < 3:
        raise ValueError("persistence 至少为 3，避免单个波动触发告警。")

    # Export frame-level ratio alongside the frozen per-frame metrics.
    frame_out = []
    for r in rows:
        frame_out.append({
            "frame_index_zero_based": r["frame_index_zero_based"],
            "rgb_timestamp": r["rgb_timestamp"],
            "AbsRel": r["AbsRel"],
            "RMSE": r["RMSE"],
            "delta1": r["delta1"],
            "valid_ratio": r["valid_ratio"],
            "median_gt_m": r["median_gt_m"],
            "median_pred_m": r["median_pred_m"],
            "pred_gt_median_ratio": r["pred_gt_median_ratio"],
            "gt_file": r["gt_file"],
            "rgb_file": r["rgb_file"],
        })
    write_csv(out_dir / "gt1a2_frame_diagnostics.csv", frame_out)

    baseline = rows[:args.baseline_frames]
    base_absrel = mean([r["AbsRel"] for r in baseline])
    base_rmse = mean([r["RMSE"] for r in baseline])
    base_delta1 = mean([r["delta1"] for r in baseline])
    base_ratio = median([r["pred_gt_median_ratio"] for r in baseline])
    abs_threshold = max(1.5 * base_absrel, base_absrel + 0.05)
    ratio_drop_threshold = 0.8 * base_ratio
    severe_ratio_threshold = 0.75

    all_rolling = []
    onset = {}
    for window in WINDOWS:
        rr = rolling_rows(rows, window)
        all_rolling.extend(rr)
        onset[str(window)] = {
            "absrel_sustained_degradation": first_sustained(
                rr, "AbsRel_mean", lambda x: x >= abs_threshold, args.persistence
            ),
            "depth_ratio_20pct_drop": first_sustained(
                rr, "median_frame_pred_gt_ratio",
                lambda x: x <= ratio_drop_threshold, args.persistence
            ),
            "severe_underestimation_ratio_le_0_75": first_sustained(
                rr, "median_frame_pred_gt_ratio",
                lambda x: x <= severe_ratio_threshold, args.persistence
            ),
        }

    write_csv(out_dir / "gt1a2_rolling_windows.csv", all_rolling)

    # Non-overlapping 100-frame blocks, to expose whether the sequence has phases.
    segments = []
    for start in range(0, len(rows), 100):
        subset = rows[start:start + 100]
        segments.append({
            "start_index": subset[0]["frame_index_zero_based"],
            "end_index": subset[-1]["frame_index_zero_based"],
            "n_frames": len(subset),
            "AbsRel_mean": mean([r["AbsRel"] for r in subset]),
            "RMSE_frame_mean_m": mean([r["RMSE"] for r in subset]),
            "delta1_mean": mean([r["delta1"] for r in subset]),
            "valid_ratio_mean": mean([r["valid_ratio"] for r in subset]),
            "median_frame_pred_gt_ratio": median([r["pred_gt_median_ratio"] for r in subset]),
            "median_gt_m": median([r["median_gt_m"] for r in subset]),
            "median_pred_m": median([r["median_pred_m"] for r in subset]),
        })
    write_csv(out_dir / "gt1a2_100frame_segments.csv", segments)

    evidence = find_evidence(project_root, fields)
    (out_dir / "gt1a2_evidence_audit.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "experiment": "GT1-A2 rolling-window temporal degradation localization",
        "status": "PASS",
        "inference_reused": True,
        "training_performed": False,
        "gpu_required": False,
        "input_csv": str(input_path),
        "evaluated_frames": len(rows),
        "frame_index_start": rows[0]["frame_index_zero_based"],
        "frame_index_end": rows[-1]["frame_index_zero_based"],
        "baseline_definition": f"first {args.baseline_frames} evaluated RGB frames",
        "persistence_windows": args.persistence,
        "window_sizes": list(WINDOWS),
        "baseline_metrics": {
            "AbsRel_mean": base_absrel,
            "RMSE_frame_mean_m": base_rmse,
            "delta1_mean": base_delta1,
            "median_frame_pred_gt_ratio": base_ratio,
        },
        "screening_thresholds": {
            "AbsRel_threshold": abs_threshold,
            "AbsRel_rule": "max(1.5 * first-window baseline mean AbsRel, baseline mean AbsRel + 0.05)",
            "depth_ratio_20pct_drop_threshold": ratio_drop_threshold,
            "depth_ratio_rule": "median of per-frame median(pred)/median(GT) <= 0.8 * first-window baseline median ratio",
            "severe_underestimation_threshold": severe_ratio_threshold,
            "persistence_rule": f"threshold condition holds for {args.persistence} consecutive rolling-window endpoints",
            "warning": "These are exploratory screening rules, not statistical significance tests or causal proof.",
        },
        "onset_candidates": onset,
        "non_overlapping_100_frame_segments": segments,
        "evidence_audit": evidence,
    }
    (out_dir / "gt1a2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plots = save_plots(out_dir, all_rolling, abs_threshold, ratio_drop_threshold)
    summary["plots"] = plots
    (out_dir / "gt1a2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Human-readable report.
    def onset_line(label, result):
        if result is None:
            return f"- {label}：在本序列内未满足连续 {args.persistence} 个窗口的筛查条件。"
        return (
            f"- {label}：首次告警窗口结束索引 {result['first_alert_window_end_index']}，"
            f"连续条件确认索引 {result['confirmation_window_end_index']}；"
            f"告警窗口覆盖索引 {result['first_alert_window_start_index']}–"
            f"{result['first_alert_window_end_index']}，"
            f"确认时指标值 {result['confirmation_value']:.4f}。"
        )

    report = [
        "# GT1-A2 实验报告：滑动窗口传播退化定位",
        "",
        "## 1. 实验目的与边界",
        "",
        "本实验复用 GT1-A1 的一对一 RGB–GT 配对及逐帧预测指标，不重新推理、不训练模型、不需要 GPU。",
        "滑动窗口均为按 RGB 索引递增的尾随窗口；RMSE 为窗口内逐帧 RMSE 的算术平均，不是全像素池化 RMSE。",
        "深度比值是每帧 `median_pred_m / median_gt_m` 的滚动中位数，不是整张深度图的像素级尺度估计。",
        "",
        "## 2. 基线与筛查阈值",
        "",
        f"- 基线：前 {args.baseline_frames} 帧。",
        f"- 基线 AbsRel：{base_absrel:.6f}；基线帧均 RMSE：{base_rmse:.6f} m；基线 δ1：{base_delta1:.6f}。",
        f"- 基线逐帧深度中位数比值的中位数：{base_ratio:.6f}。",
        f"- AbsRel 告警阈值：{abs_threshold:.6f}，规则为 `max(1.5 × 基线 AbsRel, 基线 AbsRel + 0.05)`。",
        f"- 深度比值偏移阈值：{ratio_drop_threshold:.6f}，即基线比值下降 20%。",
        f"- 严重低估筛查阈值：深度比值 ≤ {severe_ratio_threshold:.2f}。",
        f"- 持续性规则：条件需连续满足 {args.persistence} 个相邻窗口端点。",
        "",
        "这些阈值是透明的探索性筛查规则，不是经过统计校准的显著性检验，也不证明因果关系。",
        "",
        "## 3. 首次持续告警候选",
        "",
    ]
    for window in WINDOWS:
        report.append(f"### {window} 帧窗口")
        report.append("")
        o = onset[str(window)]
        report.append(onset_line("AbsRel 持续升高", o["absrel_sustained_degradation"]))
        report.append(onset_line("深度比值较基线下降 20%", o["depth_ratio_20pct_drop"]))
        report.append(onset_line("严重低估（比值 ≤ 0.75）", o["severe_underestimation_ratio_le_0_75"]))
        report.append("")
    report.extend([
        "## 4. 非重叠 100 帧阶段统计",
        "",
        "| RGB 索引区间 | 帧数 | AbsRel ↓ | 帧均 RMSE ↓ | δ1 ↑ | 深度中位数比值中位数 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for s in segments:
        report.append(
            f"| {s['start_index']}–{s['end_index']} | {s['n_frames']} | "
            f"{s['AbsRel_mean']:.4f} | {s['RMSE_frame_mean_m']:.4f} m | "
            f"{s['delta1_mean']:.2%} | {s['median_frame_pred_gt_ratio']:.3f} |"
        )
    report.extend([
        "",
        "## 5. 光流/状态证据覆盖审计",
        "",
        f"- 逐帧光流或传播状态诊断列：{evidence['per_frame_flow_or_state_diagnostic_columns'] or '无'}。",
        f"- GT1 日志是否存在：{'是' if evidence['gt1_run_log_exists'] else '否'}。",
        f"- 日志记录的关键帧调用：{evidence['keyframe_calls_from_log']}；fast 调用：{evidence['fast_calls_from_log']}。",
        "- 现有证据只足以描述误差、有效像素比例和深度中位数随时间的变化；不能单凭这些结果断言是状态累积、光流失败或场景变化导致。",
        "",
        "## 6. 结论边界与下一步",
        "",
        "GT1-A2 的目标是定位可重复的性能退化区间，不是证明某种失效机制。若 AbsRel 告警与深度比值下跌出现在不同区间，应分别报告，不能合并成单一原因。",
        "如果现有逐帧文件没有光流质量、传播状态差异或状态年龄记录，下一步应仅在识别出的候选区间补充这些诊断量，或做受控的刷新/状态干预对照；不要直接将相关性解释为因果。",
        "",
        "## 7. 输出文件",
        "",
        "- `gt1a2_frame_diagnostics.csv`：逐帧深度中位数比值。",
        "- `gt1a2_rolling_windows.csv`：25/50 帧滚动指标。",
        "- `gt1a2_100frame_segments.csv`：非重叠 100 帧阶段统计。",
        "- `gt1a2_summary.json`：阈值、告警候选、阶段统计和证据审计。",
        "- `gt1a2_evidence_audit.json`：现有日志/逐帧列可用性审计。",
        "- `gt1a2_rolling_absrel.png`、`gt1a2_depth_ratio.png`：若 matplotlib 可用则生成。",
        "",
        "**状态：GT1-A2 分析脚本执行通过，仅代表诊断计算完成；退化原因仍需后续对照验证。**",
        "",
    ])
    (out_dir / "GT1-A2_实验报告.md").write_text("\n".join(report), encoding="utf-8")

    print("\n========== GT1-A2 SUMMARY ==========", flush=True)
    print(json.dumps({
        "status": summary["status"],
        "evaluated_frames": len(rows),
        "baseline_absrel": base_absrel,
        "absrel_threshold": abs_threshold,
        "baseline_depth_ratio": base_ratio,
        "depth_ratio_20pct_drop_threshold": ratio_drop_threshold,
        "onset_candidates": onset,
        "plots": plots,
        "output_dir": str(out_dir),
    }, ensure_ascii=False, indent=2), flush=True)
    print("\nGT1-A2 PASS", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GT1-A2 FAIL: {exc}", file=sys.stderr, flush=True)
        raise
