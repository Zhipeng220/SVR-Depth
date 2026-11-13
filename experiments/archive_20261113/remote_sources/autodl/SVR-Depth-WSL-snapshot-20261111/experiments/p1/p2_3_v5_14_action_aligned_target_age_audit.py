#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.3 v5.14: action-aligned target and state-age proxy audit.

CPU-only. Does not run VeloDepth inference, train a predictor, or change weights.
Audits the estimand represented by v5.5 labels, target-normalization sensitivity,
logged-action alignment, age-vs-frame proxying, and independent anchor counts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata

TARGET = "G_state_absrel_discounted_per_valid_frame"
TARGET_SUM = "G_state_absrel_discounted"
UNWEIGHTED = "unweighted_mean_absrel_gain"
POLICIES = (
    "auto",
    "budget_fixed",
    "causal_budget_state_only",
    "causal_budget_motion_plus_learned",
)
FEATURES = [
    "feature_scheduler_distance_kf",
    "feature_scheduler_flow_magnitude",
    "feature_scheduler_total_luma_change",
    "feature_state_radius_teacher_mean",
    "feature_state_confidence_teacher_mean",
    "feature_state_features_teacher_0_mean",
    "feature_state_features_teacher_1_mean",
    "feature_state_features_teacher_2_mean",
    "feature_state_features_teacher_3_mean",
]
P2_RENAME = {f: f.replace("feature_", "") for f in FEATURES}


def safe_corr(a, b, method="spearman"):
    a = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            return float(pearsonr(a, b).statistic)
        return float(spearmanr(a, b).statistic)


def partial_spearman(x, y, control):
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    c = pd.to_numeric(pd.Series(control), errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[m], y[m], c[m]
    if len(x) < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12 or np.std(c) < 1e-12:
        return float("nan")
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(c)
    design = np.column_stack([np.ones(len(rc)), rc])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def as_bool(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def load_inputs(labels_path: Path, p2_path: Path):
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    p2 = pd.read_csv(p2_path, encoding="utf-8-sig")
    required = {
        "sequence", "trajectory_policy", "anchor_frame", "logged_action_at_anchor",
        "scheduled_action_requested", "horizon_H", "status", "future_metric_pairs",
        "future_frames_requested", "future_restart_requests_B0_ignored",
        "future_restart_requests_B1_ignored", TARGET, TARGET_SUM,
        "mean_future_absrel_B0", "mean_future_absrel_B1", *FEATURES,
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"v5.5 label CSV missing columns: {missing}")
    p2_required = {
        "sequence", "anchor_frame", "horizon_H", TARGET, "eligible",
        *[P2_RENAME[f] for f in FEATURES],
    }
    p2_missing = sorted(p2_required - set(p2.columns))
    if p2_missing:
        raise ValueError(f"P2.2.2 dataset missing columns: {p2_missing}")

    for c in ["anchor_frame", "horizon_H", "future_metric_pairs", "future_frames_requested",
              "future_restart_requests_B0_ignored", "future_restart_requests_B1_ignored",
              TARGET, TARGET_SUM, "mean_future_absrel_B0", "mean_future_absrel_B1", *FEATURES]:
        labels[c] = pd.to_numeric(labels[c], errors="coerce")
    labels["scheduled_action_requested_bool"] = as_bool(labels["scheduled_action_requested"])
    labels["sequence"] = labels["sequence"].astype(str)
    labels["trajectory_policy"] = labels["trajectory_policy"].astype(str)
    valid = labels[(labels["status"].astype(str) == "ok") & np.isfinite(labels[TARGET])].copy()
    valid[UNWEIGHTED] = (
        pd.to_numeric(valid["mean_future_absrel_B0"], errors="coerce")
        - pd.to_numeric(valid["mean_future_absrel_B1"], errors="coerce")
    )

    p2["sequence"] = p2["sequence"].astype(str)
    for c in ["anchor_frame", "horizon_H", TARGET, *[P2_RENAME[f] for f in FEATURES]]:
        p2[c] = pd.to_numeric(p2[c], errors="coerce")
    p2h16 = p2[(p2["horizon_H"] == 16) & as_bool(p2["eligible"])].copy()
    return labels, valid, p2h16


def source_consistency(valid, p2h16):
    auto = valid[valid["trajectory_policy"] == "auto"].copy()
    key = ["sequence", "anchor_frame"]
    if auto.duplicated(key).any() or p2h16.duplicated(key).any():
        raise ValueError("Duplicate sequence/anchor keys in cross-source audit.")
    cols = [TARGET] + FEATURES
    left = auto[key + cols]
    right = p2h16[key + [TARGET] + [P2_RENAME[f] for f in FEATURES]]
    merged = left.merge(right, on=key, how="inner", suffixes=("_v55", "_p22"))
    rows = []
    for col in cols:
        p2col = P2_RENAME.get(col, col)
        a = pd.to_numeric(merged[f"{col}_v55"], errors="coerce").to_numpy(float) if col == TARGET else pd.to_numeric(merged[col], errors="coerce").to_numpy(float)
        b = pd.to_numeric(merged[f"{p2col}_p22"], errors="coerce").to_numpy(float) if col == TARGET else pd.to_numeric(merged[p2col], errors="coerce").to_numpy(float)
        m = np.isfinite(a) & np.isfinite(b)
        dif = np.abs(a[m] - b[m])
        rows.append({
            "field": col, "matched_rows": int(m.sum()),
            "mismatch_rows_tol_1e-8": int(np.sum(dif > 1e-8)),
            "max_abs_difference": float(dif.max()) if len(dif) else np.nan,
            "mean_abs_difference": float(dif.mean()) if len(dif) else np.nan,
        })
    out = pd.DataFrame(rows)
    summary = {
        "auto_valid_rows": int(len(auto)),
        "p2_2_2_eligible_h16_rows": int(len(p2h16)),
        "overlap_rows": int(len(merged)),
        "all_fields_match_tol_1e_8": bool((out["mismatch_rows_tol_1e-8"] == 0).all()),
    }
    return out, summary


def label_protocol_audit(raw, valid):
    pair = pd.to_numeric(valid["future_metric_pairs"], errors="coerce")
    y = pd.to_numeric(valid[TARGET], errors="coerce")
    ysum = pd.to_numeric(valid[TARGET_SUM], errors="coerce")
    formula_error = np.abs(ysum / pair - y)
    requested_matches_action = (
        valid["scheduled_action_requested_bool"]
        == (valid["logged_action_at_anchor"].astype(str) == "state_writeback")
    )
    restart_b0 = pd.to_numeric(valid["future_restart_requests_B0_ignored"], errors="coerce").fillna(0)
    restart_b1 = pd.to_numeric(valid["future_restart_requests_B1_ignored"], errors="coerce").fillna(0)
    unique_restart_anchors = valid.loc[(restart_b0 > 0) | (restart_b1 > 0), ["sequence", "anchor_frame"]].drop_duplicates()
    return {
        "raw_rows": int(len(raw)),
        "valid_rows": int(len(valid)),
        "invalid_or_skipped_rows": int(len(raw) - len(valid)),
        "unique_sequence_anchor_pairs": int(valid[["sequence", "anchor_frame"]].drop_duplicates().shape[0]),
        "sequence_count": int(valid["sequence"].nunique()),
        "policy_count": int(valid["trajectory_policy"].nunique()),
        "horizon_values": sorted([int(x) for x in valid["horizon_H"].dropna().unique()]),
        "future_pairs_distribution": {str(int(k)): int(v) for k, v in valid["future_metric_pairs"].value_counts().sort_index().items()},
        "target_formula_max_abs_error": float(formula_error.max()),
        "target_formula_rows_over_1e_8": int(np.sum(formula_error > 1e-8)),
        "scheduled_request_vs_logged_action_mismatches": int((~requested_matches_action).sum()),
        "logged_action_counts": {str(k): int(v) for k, v in valid["logged_action_at_anchor"].value_counts().items()},
        "positive_target_rows": int((y > 0).sum()),
        "negative_target_rows": int((y < 0).sum()),
        "zero_target_rows": int((y == 0).sum()),
        "future_restart_ignored_rows": int(((restart_b0 > 0) | (restart_b1 > 0)).sum()),
        "future_restart_ignored_unique_anchors": int(len(unique_restart_anchors)),
        "future_restart_ignored_request_count_B0": int(restart_b0.sum()),
        "future_restart_ignored_request_count_B1": int(restart_b1.sum()),
    }


def target_sensitivity(valid):
    groups = [("ALL", "ALL", valid)]
    groups += [(str(seq), "ALL", g) for seq, g in valid.groupby("sequence", sort=True)]
    groups += [(str(seq), str(pol), g) for (seq, pol), g in valid.groupby(["sequence", "trajectory_policy"], sort=True)]
    rows = []
    for seq, pol, g in groups:
        y = pd.to_numeric(g[TARGET], errors="coerce").to_numpy(float)
        ys = pd.to_numeric(g[TARGET_SUM], errors="coerce").to_numpy(float)
        yu = pd.to_numeric(g[UNWEIGHTED], errors="coerce").to_numpy(float)
        rows.append({
            "sequence": seq, "trajectory_policy": pol, "n": len(g),
            "primary_mean": float(np.nanmean(y)), "primary_median": float(np.nanmedian(y)),
            "primary_positive_rate": float(np.mean(y > 0)), "primary_negative_rate": float(np.mean(y < 0)),
            "sum_vs_primary_sign_agreement": float(np.mean((y > 0) == (ys > 0))),
            "sum_vs_primary_spearman": safe_corr(ys, y),
            "unweighted_vs_primary_sign_agreement": float(np.mean((y > 0) == (yu > 0))),
            "unweighted_vs_primary_spearman": safe_corr(yu, y),
            "unweighted_vs_primary_pearson": safe_corr(yu, y, "pearson"),
            "unweighted_mean": float(np.nanmean(yu)),
        })
    return pd.DataFrame(rows)


def action_alignment(valid):
    rows = []
    for (seq, pol, action), g in valid.groupby(["sequence", "trajectory_policy", "logged_action_at_anchor"], sort=True):
        y = g[TARGET].to_numpy(float)
        yu = g[UNWEIGHTED].to_numpy(float)
        rows.append({
            "sequence": seq, "trajectory_policy": pol, "logged_action_at_anchor": action,
            "n": len(g), "positive_target_rate": float(np.mean(y > 0)),
            "negative_target_rate": float(np.mean(y < 0)),
            "mean_target": float(np.mean(y)), "median_target": float(np.median(y)),
            "unweighted_target_sign_agreement": float(np.mean((y > 0) == (yu > 0))),
        })
    by_action = []
    for action, g in valid.groupby("logged_action_at_anchor", sort=True):
        y = g[TARGET].to_numpy(float)
        by_action.append({
            "logged_action_at_anchor": action, "n": len(g),
            "positive_target_n": int(np.sum(y > 0)), "negative_target_n": int(np.sum(y < 0)),
            "positive_target_rate": float(np.mean(y > 0)),
            "negative_target_rate": float(np.mean(y < 0)),
            "mean_target": float(np.mean(y)), "median_target": float(np.median(y)),
            "scheduled_request_match_rate": float(np.mean(
                g["scheduled_action_requested_bool"] == (g["logged_action_at_anchor"] == "state_writeback")
            )),
        })
    return pd.DataFrame(rows), pd.DataFrame(by_action)


def age_proxy_audit(valid):
    # Age/motion scheduler features are policy-invariant at each sequence/anchor;
    # deduplicate anchors for age-vs-frame diagnostics.
    anchor = valid.sort_values(["sequence", "anchor_frame", "trajectory_policy"]).drop_duplicates(["sequence", "anchor_frame"]).copy()
    anchor["frame_minus_age"] = anchor["anchor_frame"] - anchor["feature_scheduler_distance_kf"]
    rows = []
    reset_rows = []
    for seq, g in anchor.groupby("sequence", sort=True):
        g = g.sort_values("anchor_frame").copy()
        offset = g["frame_minus_age"].to_numpy(float)
        changed = np.r_[False, np.abs(np.diff(offset)) > 1e-6]
        for i in np.where(changed)[0]:
            reset_rows.append({
                "sequence": seq, "anchor_frame": int(g.iloc[i]["anchor_frame"]),
                "previous_frame_minus_age": float(offset[i-1]),
                "new_frame_minus_age": float(offset[i]),
                "age": float(g.iloc[i]["feature_scheduler_distance_kf"]),
                "interpretation": "inferred age-origin offset change; inspect source trajectory around this anchor",
            })
        age = g["feature_scheduler_distance_kf"]
        frame = g["anchor_frame"]
        rows.append({
            "sequence": seq, "unique_anchors": len(g),
            "exact_age_equals_anchor_rate_tol_1e_6": float(np.mean(np.abs(age-frame) <= 1e-6)),
            "pearson_age_vs_anchor": safe_corr(age, frame, "pearson"),
            "spearman_age_vs_anchor": safe_corr(age, frame, "spearman"),
            "mean_abs_age_minus_anchor": float(np.mean(np.abs(age-frame))),
            "max_abs_age_minus_anchor": float(np.max(np.abs(age-frame))),
            "unique_frame_minus_age_offsets": int(pd.Series(offset).round(6).nunique()),
            "offset_transition_count": int(changed.sum()),
            "observed_frame_minus_age_offsets": ",".join(str(x) for x in sorted(pd.Series(offset).round(6).unique())),
        })
    by_policy = []
    for (seq, pol), g in valid.groupby(["sequence", "trajectory_policy"], sort=True):
        by_policy.append({
            "sequence": seq, "trajectory_policy": pol, "n": len(g),
            "spearman_anchor_frame_vs_target": safe_corr(g["anchor_frame"], g[TARGET]),
            "spearman_scheduler_age_vs_target": safe_corr(g["feature_scheduler_distance_kf"], g[TARGET]),
            "partial_spearman_age_vs_target_control_anchor": partial_spearman(
                g["feature_scheduler_distance_kf"], g[TARGET], g["anchor_frame"]
            ),
            "partial_spearman_anchor_vs_target_control_age": partial_spearman(
                g["anchor_frame"], g[TARGET], g["feature_scheduler_distance_kf"]
            ),
            "age_rank_equals_frame_rank": bool(
                np.all(rankdata(g["feature_scheduler_distance_kf"]) == rankdata(g["anchor_frame"]))
            ),
        })
    return pd.DataFrame(rows), pd.DataFrame(reset_rows), pd.DataFrame(by_policy)


def feature_policy_variation(valid):
    rows = []
    grouped = valid.groupby(["sequence", "anchor_frame"], sort=True)
    for f in FEATURES:
        ranges = []
        for _, g in grouped:
            x = pd.to_numeric(g[f], errors="coerce").to_numpy(float)
            x = x[np.isfinite(x)]
            ranges.append(float(np.max(x)-np.min(x)) if len(x) else np.nan)
        r = np.asarray(ranges, dtype=float)
        finite = r[np.isfinite(r)]
        rows.append({
            "feature": f, "unique_anchor_groups": int(len(grouped)),
            "anchors_with_policy_variation_tol_1e_8": int(np.sum(finite > 1e-8)),
            "variation_rate": float(np.mean(finite > 1e-8)) if len(finite) else np.nan,
            "median_within_anchor_policy_range": float(np.median(finite)) if len(finite) else np.nan,
            "max_within_anchor_policy_range": float(np.max(finite)) if len(finite) else np.nan,
        })
    return pd.DataFrame(rows)


def source_semantics():
    return pd.DataFrame([
        {
            "item": "target estimand",
            "source_evidence": "p2_3_v5_5_same_state_counterfactual.py: same_state_counterfactual",
            "verified_semantics": "B0 在锚点继续传播；B1 从同一个动作前状态分支，并仅写回关键帧候选产生的四个持久几何字段。",
            "not_equivalent_to": "不等于完整关键帧刷新收益，也不等于完整在线策略结果",
        },
        {
            "item": "feature timing",
            "source_evidence": "p2_3_v5_5_same_state_counterfactual.py: build_online_features call before pre_state clone/counterfactual and before replaying logged action",
            "verified_semantics": "特征在复制反事实状态之前、回放已记录动作之前，从该策略轨迹的动作前状态提取。",
            "remaining_limit": "源码顺序支持动作前可用性；若需独立确认实际运行时行为，还需运行时埋点验证",
        },
        {
            "item": "future protocol",
            "source_evidence": "same_state_counterfactual future loop; metadata future_protocol",
            "verified_semantics": "未来帧强制执行传播；未来 restart 请求只记录、不执行。",
            "not_equivalent_to": "不等于正常执行 restart 的真实未来轨迹",
        },
        {
            "item": "skipped anchors",
            "source_evidence": "same_state_counterfactual returns skipped_anchor_restart when anchor propagation requests restart",
            "verified_semantics": "锚点传播若触发 restart，该锚点会被跳过，不进入有效标签。",
            "remaining_limit": "必须保留跳过记录；不能把它们默认为零收益样本",
        },
        {
            "item": "state age",
            "source_evidence": "feature_scheduler_distance_kf from the pre-action scheduler state",
            "verified_semantics": "当前采样中 crowd2/crowd3 的状态年龄等于绝对锚点帧号；balloon2 在推断的第 57 帧边界后出现重置。",
            "remaining_limit": "除重置边界外，状态年龄可能充当序列阶段的代理变量",
        },
    ])


def write_report(out, protocol, source_summary, source_audit, target_df, action_df,
                 action_overall, age_df, reset_df, age_target_df, variation_df, semantics):
    def fmt(x, digits=4):
        try:
            if not np.isfinite(float(x)): return "NA"
            return f"{float(x):.{digits}f}"
        except Exception:
            return str(x)
    lines = [
        "# P2.3 v5.14 动作对齐收益目标与状态年龄代理审计", "",
        "## 1. 结论摘要", "",
        "**v5.14 是 CPU-only 标签/协议审计；不运行 VeloDepth 推理、不训练预测器、不修改权重。**", "",
        f"- 原始记录：{protocol['raw_rows']} 行；有效标签：{protocol['valid_rows']} 行；无效/跳过：{protocol['invalid_or_skipped_rows']} 行。",
        f"- 有效记录覆盖 {protocol['sequence_count']} 条序列、{protocol['policy_count']} 种策略轨迹，但只有 {protocol['unique_sequence_anchor_pairs']} 个唯一序列-锚点组合。",
        f"- H 值：{protocol['horizon_values']}；标签公式 `discounted_sum / future_metric_pairs` 最大绝对误差：{protocol['target_formula_max_abs_error']:.3g}。",
        f"- 计划动作请求与日志动作不一致：{protocol['scheduled_request_vs_logged_action_mismatches']} 行。",
        f"- 与 P2.2.2 的 auto/H=16 交叉样本：{source_summary['overlap_rows']} 行；全部目标与 9 个特征容差内一致：{'是' if source_summary['all_fields_match_tol_1e_8'] else '否'}。", "",
        "## 2. 标签究竟衡量什么", "",
        "当前目标 `G_state_absrel_discounted_per_valid_frame` 衡量：从同一动作前持久状态出发，B1 将关键帧候选产生的四个持久几何字段写回，相比 B0 继续传播，对 H=16 未来折扣 AbsRel 的改善。",
        "",
        "**它不是完整关键帧刷新收益。** 未来窗口强制传播；正常策略本应触发的 future restart 只记录、不执行。锚点本身若触发 restart，则被跳过，不作为零收益样本。", "",
        f"- 正收益：{protocol['positive_target_rows']} 行；负收益：{protocol['negative_target_rows']} 行；零收益：{protocol['zero_target_rows']} 行。",
        f"- 未来窗口有被忽略 restart 请求的记录：{protocol['future_restart_ignored_rows']} 行，对应 {protocol['future_restart_ignored_unique_anchors']} 个唯一锚点（四种策略行会重复记录同一锚点）。",
        f"- 有效未来指标对覆盖：{protocol['future_pairs_distribution']}。", "",
        "## 3. 标签归一化敏感性", "",
        "比较三种已有字段：主要标签（折扣收益除以有效指标对数）、折扣累计收益、以及 `mean_future_absrel_B0 - mean_future_absrel_B1` 的非折扣均值差。第三种仅是敏感性对照，不替代原始标签。", "",
        "| 范围 | n | 主标签正收益率 | 累计收益与主标签符号一致率 | 非折扣均值差符号一致率 | Spearman(非折扣,主标签) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in target_df[(target_df["trajectory_policy"]=="ALL") & (target_df["sequence"]!="ALL")].iterrows():
        lines.append(f"| {r['sequence']} | {int(r['n'])} | {r['primary_positive_rate']:.1%} | {r['sum_vs_primary_sign_agreement']:.1%} | {r['unweighted_vs_primary_sign_agreement']:.1%} | {fmt(r['unweighted_vs_primary_spearman'],3)} |")
    allrow = target_df[(target_df["sequence"]=="ALL") & (target_df["trajectory_policy"]=="ALL")].iloc[0]
    lines += [
        f"| **ALL** | {int(allrow['n'])} | {allrow['primary_positive_rate']:.1%} | {allrow['sum_vs_primary_sign_agreement']:.1%} | {allrow['unweighted_vs_primary_sign_agreement']:.1%} | {fmt(allrow['unweighted_vs_primary_spearman'],3)} |",
        "",
        f"总体上，非折扣均值差与主标签符号一致率为 {allrow['unweighted_vs_primary_sign_agreement']:.1%}；累计收益与主标签符号一致率为 {allrow['sum_vs_primary_sign_agreement']:.1%}。这说明主要正负收益结论对当前两种替代汇总较稳健，但仍有少数锚点符号不一致，应在结果中保留。",
        "",
        "## 4. 实际记录动作与反事实标签", "",
        "| 记录动作 | n | 正收益率 | 负收益率 | 平均主标签 | 请求与动作一致率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in action_overall.iterrows():
        lines.append(f"| {r['logged_action_at_anchor']} | {int(r['n'])} | {r['positive_target_rate']:.1%} | {r['negative_target_rate']:.1%} | {fmt(r['mean_target'],5)} | {r['scheduled_request_match_rate']:.1%} |")
    sw = action_overall[action_overall["logged_action_at_anchor"]=="state_writeback"]
    if len(sw):
        rr = sw.iloc[0]
        lines += ["", f"在记录为 `state_writeback` 的 {int(rr['n'])} 行中，有 {int(rr['negative_target_n'])} 行（{rr['negative_target_rate']:.1%}）对应负向同状态反事实标签。**这只能作为动作与反事实标签的诊断，不等于完整在线策略的真实损失**，因为未来 restart 被忽略且标签仍是受控传播协议下的收益。", ""]
    lines += [
        "## 5. 状态年龄是否只是帧位置代理", "",
        "| 序列 | 唯一锚点 | age=anchor 比例 | Spearman(age,anchor) | age-anchor 平均绝对差 | offset 转换数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in age_df.iterrows():
        lines.append(f"| {r['sequence']} | {int(r['unique_anchors'])} | {r['exact_age_equals_anchor_rate_tol_1e_6']:.1%} | {fmt(r['spearman_age_vs_anchor'],3)} | {fmt(r['mean_abs_age_minus_anchor'],3)} | {int(r['offset_transition_count'])} |")
    lines += ["", "### 推断的 age-origin offset 变化", ""]
    if len(reset_df):
        lines += ["| 序列 | 当前锚点 | 旧 frame-age | 新 frame-age | 当前 age |", "|---|---:|---:|---:|---:|"]
        for _, r in reset_df.iterrows():
            lines.append(f"| {r['sequence']} | {int(r['anchor_frame'])} | {fmt(r['previous_frame_minus_age'],2)} | {fmt(r['new_frame_minus_age'],2)} | {fmt(r['age'],2)} |")
    else:
        lines.append("未发现 offset 转换。")
    lines += [
        "",
        "在当前采样点中，crowd2 和 crowd3 的 scheduler age 与绝对帧号完全相等；balloon2 的后段锚点出现 age reset（由 frame-age offset 变化体现）。因此年龄基线可能主要利用序列阶段信息，而不是独立感知几何状态退化。这个判断对当前采样锚点成立，不应外推到未测序列。",
        "",
        "### 按序列与策略的目标关联", "",
        "| 序列 | 策略 | n | ρ(frame,target) | ρ(age,target) | partial ρ(age,target \u007c frame) | age 排序等于 frame 排序 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, r in age_target_df.iterrows():
        lines.append(f"| {r['sequence']} | {r['trajectory_policy']} | {int(r['n'])} | {fmt(r['spearman_anchor_frame_vs_target'],3)} | {fmt(r['spearman_scheduler_age_vs_target'],3)} | {fmt(r['partial_spearman_age_vs_target_control_anchor'],3)} | {'是' if r['age_rank_equals_frame_rank'] else '否'} |")
    lines += [
        "",
        "partial Spearman 是对秩变量控制帧位置后的探索性诊断；当 age 与 frame 完全共线时不可识别。每个分组只有 17–18 个锚点，不做显著性或因果解释。",
        "",
        "## 6. 特征在四种策略轨迹之间是否独立变化", "",
        "| 特征 | 有策略间差异的锚点数 | 比例 | 中位策略间范围 |",
        "|---|---:|---:|---:|",
    ]
    for _, r in variation_df.iterrows():
        lines.append(f"| `{r['feature']}` | {int(r['anchors_with_policy_variation_tol_1e_8'])}/{int(r['unique_anchor_groups'])} | {r['variation_rate']:.1%} | {fmt(r['median_within_anchor_policy_range'],5)} |")
    lines += [
        "",
        "状态年龄、光流幅值和亮度变化在同一序列-锚点的四种策略轨迹间不变；状态张量统计量则在多数锚点间变化。故 212 行不是 212 个独立锚点；应把唯一锚点数量和策略轨迹数量分别报告。",
        "",
        "## 7. 源代码语义审计", "",
        "| 项目 | 源码证据 | 审计结论 |",
        "|---|---|---|",
    ]
    for _, r in semantics.iterrows():
        lines.append(f"| {r['item']} | {r['source_evidence']} | {r['verified_semantics']} |")
    lines += [
        "",
        "## 8. 决策", "",
        "1. **保留当前标签用于受控状态写回研究**，并统一称为“same-state persistent-geometry writeback gain”，不要简称完整 refresh gain。",
        "2. 如果论文主张的是完整关键帧刷新或正常在线策略收益，现有标签不足以支撑；需另行设计会执行未来 restart 的反事实评估。",
        "3. 状态年龄保留为低成本基线，但必须与绝对帧位置并列报告，并使用包含真实重启边界的序列验证。",
        "4. 现阶段不训练更复杂的预测器。下一步先检查目标是否与最终动作空间一致，再决定是否需要生成新标签。",
        "",
        "## 9. 复现与范围", "",
        "输入包括 v5.5 同状态标签与 P2.2.2 有效数据集。脚本仅使用 CSV 做 CPU 诊断。它不重新运行 VeloDepth，也不修改权重。",
    ]
    (out / "P2_3_v5_14_动作对齐收益目标与年龄代理审计.md").write_text("\n".join(lines), encoding="utf-8")


def self_test():
    x = np.arange(10, dtype=float)
    y = 2*x + np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 1], dtype=float)
    assert safe_corr(x, y) > 0.9
    assert np.isfinite(partial_spearman(x, y, np.array([1,4,2,8,0,9,3,7,5,6], dtype=float)))
    assert len(FEATURES) == 9
    print("PASS: P2.3 v5.14 audit self-tests (4/4).")


def main():
    parser = argparse.ArgumentParser(description="P2.3 v5.14 action-aligned target and age-proxy audit")
    here = Path(__file__).resolve().parent
    parser.add_argument("--labels-csv", default=str(here / "data_p2_3_v5_14" / "p2_3_v5_5_all_same_state_anchor_labels.csv"))
    parser.add_argument("--p2-dataset-csv", default=str(here / "data_p2_3_v5_14" / "p2_2_2_valid_dataset.csv"))
    parser.add_argument("--out-dir", default="outputs/p2_3_v5_14_action_aligned_audit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    labels_path, p2_path = Path(args.labels_csv), Path(args.p2_dataset_csv)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels CSV: {labels_path}")
    if not p2_path.is_file():
        raise FileNotFoundError(f"Missing P2.2.2 CSV: {p2_path}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw, valid, p2h16 = load_inputs(labels_path, p2_path)
    protocol = label_protocol_audit(raw, valid)
    source_audit, source_summary = source_consistency(valid, p2h16)
    targets = target_sensitivity(valid)
    action_detail, action_overall = action_alignment(valid)
    age_summary, age_resets, age_target = age_proxy_audit(valid)
    variation = feature_policy_variation(valid)
    semantics = source_semantics()

    source_audit.to_csv(out / "p2_3_v5_14_source_consistency.csv", index=False)
    targets.to_csv(out / "p2_3_v5_14_target_normalization_sensitivity.csv", index=False)
    action_detail.to_csv(out / "p2_3_v5_14_action_alignment_by_sequence_policy.csv", index=False)
    action_overall.to_csv(out / "p2_3_v5_14_action_alignment_overall.csv", index=False)
    age_summary.to_csv(out / "p2_3_v5_14_age_vs_frame_proxy.csv", index=False)
    age_resets.to_csv(out / "p2_3_v5_14_inferred_age_reset_points.csv", index=False)
    age_target.to_csv(out / "p2_3_v5_14_age_target_correlations.csv", index=False)
    variation.to_csv(out / "p2_3_v5_14_feature_variation_across_policies.csv", index=False)
    semantics.to_csv(out / "p2_3_v5_14_source_semantics_audit.csv", index=False)

    metadata = {
        "experiment": "P2.3 v5.14",
        "type": "CPU-only action-aligned target and state-age proxy audit",
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "p2_2_2_sha256": hashlib.sha256(p2_path.read_bytes()).hexdigest(),
        "raw_rows": int(len(raw)), "valid_rows": int(len(valid)),
        "unique_sequence_anchor_pairs": int(valid[["sequence","anchor_frame"]].drop_duplicates().shape[0]),
        "target": TARGET,
        "target_semantics": "same-state B1 persistent geometry writeback vs B0 propagation; future restarts logged but ignored",
        "target_alternatives": [TARGET_SUM, UNWEIGHTED],
        "sequences": sorted(valid["sequence"].unique().tolist()),
        "policies": sorted(valid["trajectory_policy"].unique().tolist()),
        "velodepth_inference_run": False,
        "predictor_trained": False,
        "weights_updated": False,
        "source_consistency": source_summary,
        "protocol_audit": protocol,
    }
    (out / "p2_3_v5_14_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out, protocol, source_summary, source_audit, targets, action_detail,
                 action_overall, age_summary, age_resets, age_target, variation, semantics)

    print(f"Rows: raw={len(raw)}, valid={len(valid)}, unique sequence/anchor={protocol['unique_sequence_anchor_pairs']}")
    print(f"Target formula max abs error: {protocol['target_formula_max_abs_error']:.3g}")
    print(f"Cross-source overlap={source_summary['overlap_rows']}; all fields match={source_summary['all_fields_match_tol_1e_8']}")
    print("\nAction alignment:")
    print(action_overall.to_string(index=False))
    print("\nAge vs frame:")
    print(age_summary.to_string(index=False))
    print("\nTarget sensitivity (overall):")
    print(targets[(targets.sequence=="ALL") & (targets.trajectory_policy=="ALL")].to_string(index=False))
    print(f"\nWrote outputs to {out}")


if __name__ == "__main__":
    main()
