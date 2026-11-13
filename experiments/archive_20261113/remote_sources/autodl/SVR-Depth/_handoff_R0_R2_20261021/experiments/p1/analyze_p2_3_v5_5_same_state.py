#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate and audit P2.3 v5.5 same-state counterfactual outputs."""
import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


def read_if_exists(path):
    return pd.read_csv(path, encoding="utf-8-sig") if path.is_file() else pd.DataFrame()


def safe_bool_count(series):
    if series.empty:
        return 0
    return int(series.astype(str).str.lower().isin(["true", "1", "yes"]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/p2_3_v5_5_same_state")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else root
    out.mkdir(parents=True, exist_ok=True)

    label_parts, diag_parts, align_parts, summary_parts = [], [], [], []
    logs = sorted(root.glob("*_run.log"))
    run_rows = []
    for seq_dir in sorted(root.iterdir()) if root.exists() else []:
        if not seq_dir.is_dir():
            continue
        labels = read_if_exists(seq_dir / "p2_3_v5_5_same_state_anchor_labels.csv")
        diags = read_if_exists(seq_dir / "p2_3_v5_5_gain_diagnostics.csv")
        align = read_if_exists(seq_dir / "p2_3_v5_5_trajectory_alignment.csv")
        summary = read_if_exists(seq_dir / "p2_3_v5_5_policy_summary.csv")
        if not labels.empty:
            label_parts.append(labels)
        if not diags.empty:
            diag_parts.append(diags)
        if not align.empty:
            align_parts.append(align)
        if not summary.empty:
            summary_parts.append(summary)

    for log in logs:
        txt = log.read_text(encoding="utf-8", errors="replace")
        run_rows.append({
            "log": log.name,
            "bytes": log.stat().st_size,
            "traceback": "Traceback" in txt,
            "error_marker": "ERROR" in txt,
            "fail_marker": "FAIL:" in txt,
            "finished_marker": "P2.3 v5.5 SAME-STATE GAIN DIAGNOSTICS" in txt,
        })

    labels = pd.concat(label_parts, ignore_index=True) if label_parts else pd.DataFrame()
    diags = pd.concat(diag_parts, ignore_index=True) if diag_parts else pd.DataFrame()
    align = pd.concat(align_parts, ignore_index=True) if align_parts else pd.DataFrame()
    summaries = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    run_df = pd.DataFrame(run_rows)

    if not labels.empty:
        labels.to_csv(out / "p2_3_v5_5_all_same_state_anchor_labels.csv", index=False, encoding="utf-8-sig")
    if not diags.empty:
        diags.to_csv(out / "p2_3_v5_5_all_gain_diagnostics.csv", index=False, encoding="utf-8-sig")
    if not align.empty:
        align.to_csv(out / "p2_3_v5_5_all_trajectory_alignment.csv", index=False, encoding="utf-8-sig")
    if not summaries.empty:
        summaries.to_csv(out / "p2_3_v5_5_all_policy_summary.csv", index=False, encoding="utf-8-sig")
    if not run_df.empty:
        run_df.to_csv(out / "p2_3_v5_5_run_integrity.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# P2.3 v5.5 同状态反事实收益诊断报告",
        "",
        "## 1. 运行完整性",
        "",
        f"- 序列结果目录中读取到的日志：{len(run_df)}。",
        f"- 有 Traceback 的日志：{int(run_df['traceback'].sum()) if not run_df.empty else 0}。",
        f"- 含 ERROR 标记的日志：{int(run_df['error_marker'].sum()) if not run_df.empty else 0}。",
        f"- 含 FAIL 标记的日志：{int(run_df['fail_marker'].sum()) if not run_df.empty else 0}。",
    ]
    if not labels.empty:
        status_counts = labels["status"].astype(str).value_counts().to_dict() if "status" in labels else {}
        valid_labels = labels[
            (labels.get("status", pd.Series([], dtype=str)).astype(str) == "ok")
            & np.isfinite(pd.to_numeric(labels.get("G_state_absrel_discounted_per_valid_frame"), errors="coerce"))
        ] if "G_state_absrel_discounted_per_valid_frame" in labels else pd.DataFrame()
        lines += [
            f"- 同状态锚点记录：{len(labels)}。",
            f"- 有效收益标签：{len(valid_labels)}。",
            f"- 标签状态分布：`{json.dumps(status_counts, ensure_ascii=False)}`。",
            "",
            "正收益标签表示：从同一个刷新前状态出发，B1 持久几何状态写回降低了未来 H=16 折扣 AbsRel。",
        ]
    else:
        valid_labels = pd.DataFrame()
        lines += ["- 尚未找到同状态锚点 CSV。"]

    if not align.empty:
        mismatch = align[~align["action_match"].astype(str).str.lower().isin(["true", "1", "yes"])]
        finite_diffs = pd.to_numeric(align["absrel_diff_replayed_minus_logged"], errors="coerce").dropna().abs()
        max_diff = float(finite_diffs.max()) if len(finite_diffs) else float("nan")
        lines += [
            "",
            "## 2. 在线轨迹复现核验（解释标签前必须先看）",
            "",
            f"- 逐帧动作不一致数：{len(mismatch)} / {len(align)}。",
            f"- 有效帧 AbsRel 最大绝对差：{max_diff:.8g}。",
            "",
        ]
        if len(mismatch) == 0 and (not math.isfinite(max_diff) or max_diff <= 1e-4):
            lines.append("**轨迹复现检查通过**（动作完全一致，且 AbsRel 最大绝对差不超过 1e-4；若无有效指标差则需人工确认覆盖率）。")
        else:
            lines.append("**轨迹复现检查未通过或有数值偏差。** 在定位差异前，不应把这些标签当作精确对应原在线轨迹的证据。")
    else:
        lines += ["", "## 2. 在线轨迹复现核验", "", "未找到轨迹对齐 CSV；无法确认标签生成是否保持在线轨迹。"]

    if not diags.empty:
        lines += [
            "",
            "## 3. 预测分数与同状态收益的关系",
            "",
            "Pearson/Spearman 基于每个测试序列、每种在线轨迹和预测器的有效锚点。样本来自短序列且未来窗口重叠，结果为诊断性证据，不是独立样本统计检验。",
            "",
            "| 序列 | 在线轨迹 | 预测器 | n | Pearson | Spearman | Top-25% 收益 lift | 正收益率 lift |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
        for _, r in diags.iterrows():
            lines.append(
                f"| {r.get('sequence','')} | {r.get('trajectory_policy','')} | {r.get('predictor','')} | "
                f"{int(r.get('valid_anchor_count',0))} | {float(r.get('pearson_score_vs_same_state_gain',float('nan'))):+.3f} | "
                f"{float(r.get('spearman_score_vs_same_state_gain',float('nan'))):+.3f} | "
                f"{float(r.get('top_quartile_gain_lift',float('nan'))):+.6f} | "
                f"{float(r.get('positive_gain_rate_lift',float('nan'))):+.3f} |"
            )
    else:
        lines += ["", "## 3. 预测分数诊断", "", "尚未找到 gain diagnostics CSV。"]

    if not summaries.empty:
        lines += [
            "",
            "## 4. 在线策略深度指标（背景信息）",
            "",
            "| 序列 | 策略 | 有效帧 | AbsRel | RMSE | δ1 | 写回次数 | 自动重启 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in summaries.iterrows():
            lines.append(
                f"| {r.get('sequence','')} | {r.get('policy','')} | {int(r.get('frames_with_valid_metrics',0))} | "
                f"{float(r.get('mean_absrel',float('nan'))):.6f} | {float(r.get('mean_rmse',float('nan'))):.6f} | "
                f"{float(r.get('mean_delta1',float('nan'))):.6f} | {int(r.get('state_writeback_count',0))} | "
                f"{int(r.get('auto_restart_count',0))} |"
            )

    lines += [
        "",
        "## 5. 方法边界",
        "",
        "1. 每个锚点的 B0/B1 分支从该在线策略轨迹的同一个 pre-action state snapshot 出发，修复了按帧编号拼接不同状态轨迹标签的问题。",
        "2. B1 保留锚点当前帧传播输出，只提交 `features_teacher`、`radius_teacher`、`confidence_teacher`、`rays_base`。",
        "3. 未来 H=16 窗口采用 P1.5 受控协议：强制传播，记录但忽略未来 restart 请求；锚点本身若请求 restart，则跳过该锚点。",
        "4. 预测分数来自 P2.2.2 训练集上、排除当前测试序列后的 Ridge 模型；未来收益标签只用于离线诊断，不进入在线动作决策。",
        "5. 该实验仍计算完整 keyframe candidate，不证明计算节省、端到端 FPS 或 RK3588 实时部署。",
        "",
        "## 6. 判读顺序",
        "",
        "先确认轨迹复现核验通过，再看有效标签覆盖率，最后看同状态 score-target 的排序关系。若动作/指标复现不一致，先修复状态恢复与重放逻辑；若复现通过但相关性和 Top-25% lift 仍不稳定，则当前预测器未获得足够证据支持进入大规模训练。",
    ]
    report_path = out / "P2_3_v5_5_结果分析.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote:", report_path)
    for name in [
        "p2_3_v5_5_all_same_state_anchor_labels.csv",
        "p2_3_v5_5_all_gain_diagnostics.csv",
        "p2_3_v5_5_all_trajectory_alignment.csv",
        "p2_3_v5_5_all_policy_summary.csv",
        "p2_3_v5_5_run_integrity.csv",
    ]:
        p = out / name
        if p.exists():
            print("Wrote:", p)


if __name__ == "__main__":
    main()
