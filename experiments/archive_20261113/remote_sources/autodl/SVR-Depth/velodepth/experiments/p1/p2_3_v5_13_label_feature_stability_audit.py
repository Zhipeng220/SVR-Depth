#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.3 v5.13: same-state label / feature stability audit.

CPU-only. Does not run VeloDepth inference and does not update model weights.
Uses cleaned v5.5 same-state labels and cross-checks overlapping auto H=16 rows
against the P2.2.2 valid dataset.

Target G_state_absrel_discounted_per_valid_frame is a protocol-specific B1
persistent-state writeback gain relative to B0 propagation, not full-keyframe
refresh gain. v5.5 future counterfactual branches force propagation and log but
do not execute future restart requests.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "G_state_absrel_discounted_per_valid_frame"
SEQUENCES = ("crowd2", "balloon2", "crowd3")
POLICIES = ("auto", "budget_fixed", "causal_budget_state_only", "causal_budget_motion_plus_learned")
FEATURE_MAP = {
    "age": ["feature_scheduler_distance_kf"],
    "motion": ["feature_scheduler_flow_magnitude", "feature_scheduler_total_luma_change"],
    "state": [
        "feature_state_radius_teacher_mean",
        "feature_state_confidence_teacher_mean",
        "feature_state_features_teacher_0_mean",
        "feature_state_features_teacher_1_mean",
        "feature_state_features_teacher_2_mean",
        "feature_state_features_teacher_3_mean",
    ],
}
ALL_FEATURES = FEATURE_MAP["age"] + FEATURE_MAP["motion"] + FEATURE_MAP["state"]
FEATURE_GROUPS = {
    "age_only": FEATURE_MAP["age"],
    "motion_only": FEATURE_MAP["motion"],
    "state_only": FEATURE_MAP["state"],
    "age_plus_motion": FEATURE_MAP["age"] + FEATURE_MAP["motion"],
    "age_plus_state": FEATURE_MAP["age"] + FEATURE_MAP["state"],
    "motion_plus_state_no_age": FEATURE_MAP["motion"] + FEATURE_MAP["state"],
    "all_9_features": ALL_FEATURES,
}
P2_FEATURE_RENAME = {name: name.replace("feature_", "") for name in ALL_FEATURES}
ALPHAS = (1.0, 10.0, 100.0)


def finite_corr(a, b, method="spearman"):
    a = pd.to_numeric(pd.Series(a), errors="coerce").to_numpy(float)
    b = pd.to_numeric(pd.Series(b), errors="coerce").to_numpy(float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            return float(pearsonr(a, b).statistic)
        return float(spearmanr(a, b).statistic)


def partial_spearman(x, y, control):
    """Rank both variables and control for ranked anchor frame via residual correlation."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    c = pd.to_numeric(pd.Series(control), errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[mask], y[mask], c[mask]
    if len(x) < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12 or np.std(c) < 1e-12:
        return float("nan")
    rx, ry, rc = rankdata(x), rankdata(y), rankdata(c)
    design = np.column_stack([np.ones(len(rc)), rc])
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(("true", "1", "yes"))


def read_inputs(labels_path: Path, p2_path: Path):
    labels = pd.read_csv(labels_path, encoding="utf-8-sig")
    p2 = pd.read_csv(p2_path, encoding="utf-8-sig")
    required = {"sequence", "trajectory_policy", "anchor_frame", "status", "horizon_H", TARGET, *ALL_FEATURES}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"v5.5 label CSV missing required columns: {missing}")
    p2_required = {"sequence", "anchor_frame", "horizon_H", TARGET, "eligible", *[P2_FEATURE_RENAME[f] for f in ALL_FEATURES]}
    p2_missing = sorted(p2_required - set(p2.columns))
    if p2_missing:
        raise ValueError(f"P2.2.2 CSV missing required columns: {p2_missing}")
    labels["sequence"] = labels["sequence"].astype(str)
    labels["trajectory_policy"] = labels["trajectory_policy"].astype(str)
    labels["anchor_frame"] = pd.to_numeric(labels["anchor_frame"], errors="coerce")
    labels[TARGET] = pd.to_numeric(labels[TARGET], errors="coerce")
    for col in ALL_FEATURES:
        labels[col] = pd.to_numeric(labels[col], errors="coerce")
    if not (pd.to_numeric(labels["horizon_H"], errors="coerce") == 16).all():
        raise ValueError("Expected v5.5 label file to contain H=16 only.")
    valid = labels[(labels["status"].astype(str) == "ok") & np.isfinite(labels[TARGET])].copy()
    p2["sequence"] = p2["sequence"].astype(str)
    p2["anchor_frame"] = pd.to_numeric(p2["anchor_frame"], errors="coerce")
    p2["horizon_H"] = pd.to_numeric(p2["horizon_H"], errors="coerce")
    p2[TARGET] = pd.to_numeric(p2[TARGET], errors="coerce")
    for f in ALL_FEATURES:
        p2[P2_FEATURE_RENAME[f]] = pd.to_numeric(p2[P2_FEATURE_RENAME[f]], errors="coerce")
    p2_h16 = p2[(p2["horizon_H"] == 16) & bool_series(p2["eligible"])].copy()
    return labels, valid, p2_h16


def source_consistency_audit(valid: pd.DataFrame, p2_h16: pd.DataFrame):
    auto = valid[valid["trajectory_policy"] == "auto"].copy()
    key = ["sequence", "anchor_frame"]
    if auto.duplicated(key).any():
        raise ValueError("Cleaned v5.5 auto rows are not unique by sequence/anchor.")
    if p2_h16.duplicated(key).any():
        raise ValueError("P2.2.2 eligible H=16 rows are not unique by sequence/anchor.")
    merged = auto[key + [TARGET] + ALL_FEATURES].merge(
        p2_h16[key + [TARGET] + [P2_FEATURE_RENAME[f] for f in ALL_FEATURES]],
        on=key, how="inner", suffixes=("_v55", "_p22")
    )
    rows = []
    for col in [TARGET] + ALL_FEATURES:
        p2_col = P2_FEATURE_RENAME.get(col, col)
        if col == TARGET:
            a = pd.to_numeric(merged[f"{TARGET}_v55"], errors="coerce").to_numpy(float)
            b = pd.to_numeric(merged[f"{TARGET}_p22"], errors="coerce").to_numpy(float)
        else:
            a = pd.to_numeric(merged[col], errors="coerce").to_numpy(float)
            b = pd.to_numeric(merged[p2_col], errors="coerce").to_numpy(float)
        mask = np.isfinite(a) & np.isfinite(b)
        diffs = np.abs(a[mask] - b[mask])
        rows.append({
            "field": col, "matched_rows": int(mask.sum()),
            "exact_match_rows_tol_1e-8": int(np.sum(diffs <= 1e-8)),
            "mismatch_rows_tol_1e-8": int(np.sum(diffs > 1e-8)),
            "max_abs_difference": float(np.max(diffs)) if len(diffs) else np.nan,
            "mean_abs_difference": float(np.mean(diffs)) if len(diffs) else np.nan,
            "interpretation": "cross-source exact agreement" if len(diffs) and np.all(diffs <= 1e-8) else "review mismatches",
        })
    audit = pd.DataFrame(rows)
    summary = {
        "v55_auto_valid_rows": int(len(auto)),
        "p2_2_2_eligible_h16_rows": int(len(p2_h16)),
        "overlap_rows": int(len(merged)),
        "all_fields_exact_match": bool((audit["mismatch_rows_tol_1e-8"] == 0).all()),
        "key_unique_on_both_sides": True,
    }
    return audit, summary


def label_summaries(valid: pd.DataFrame):
    rows = []
    for (seq, policy), g in valid.groupby(["sequence", "trajectory_policy"], sort=True):
        y = g[TARGET].to_numpy(float)
        a = g["anchor_frame"].to_numpy(float)
        d = g["feature_scheduler_distance_kf"].to_numpy(float)
        rows.append({
            "sequence": seq, "trajectory_policy": policy, "n": len(g),
            "gain_mean": float(np.mean(y)), "gain_median": float(np.median(y)),
            "gain_std": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
            "gain_q25": float(np.quantile(y, .25)), "gain_q75": float(np.quantile(y, .75)),
            "gain_min": float(np.min(y)), "gain_max": float(np.max(y)),
            "positive_rate": float(np.mean(y > 0)), "negative_rate": float(np.mean(y < 0)),
            "near_zero_abs_le_1e3_rate": float(np.mean(np.abs(y) <= 1e-3)),
            "near_zero_abs_le_5e3_rate": float(np.mean(np.abs(y) <= 5e-3)),
            "spearman_anchor_vs_gain": finite_corr(a, y, "spearman"),
            "spearman_distance_vs_anchor": finite_corr(d, a, "spearman"),
            "distance_equals_anchor_rate_tol_1e_6": float(np.mean(np.isfinite(d) & np.isfinite(a) & (np.abs(d-a) <= 1e-6))),
        })
    return pd.DataFrame(rows)


def temporal_summaries(valid: pd.DataFrame):
    g = valid.copy()
    g["temporal_block"] = np.select(
        [g["anchor_frame"] < 25, (g["anchor_frame"] >= 25) & (g["anchor_frame"] < 49), g["anchor_frame"] >= 49],
        ["early_<25", "middle_25_48", "late_>=49"], default="unknown",
    )
    rows = []
    for (seq, policy, block), s in g.groupby(["sequence", "trajectory_policy", "temporal_block"], sort=True):
        y = s[TARGET].to_numpy(float)
        rows.append({
            "sequence": seq, "trajectory_policy": policy, "temporal_block": block, "n": len(s),
            "anchor_min": float(s["anchor_frame"].min()), "anchor_max": float(s["anchor_frame"].max()),
            "gain_mean": float(np.mean(y)), "gain_median": float(np.median(y)),
            "positive_rate": float(np.mean(y > 0)), "negative_rate": float(np.mean(y < 0)),
            "near_zero_abs_le_1e3_rate": float(np.mean(np.abs(y) <= 1e-3)),
        })
    return pd.DataFrame(rows)


def feature_distributions_and_correlations(valid: pd.DataFrame):
    dist_rows, corr_rows, pair_rows = [], [], []
    for (seq, policy), g in valid.groupby(["sequence", "trajectory_policy"], sort=True):
        for f in ALL_FEATURES:
            x = pd.to_numeric(g[f], errors="coerce").to_numpy(float)
            finite = x[np.isfinite(x)]
            y = g[TARGET].to_numpy(float)
            dist_rows.append({
                "sequence": seq, "trajectory_policy": policy, "feature": f,
                "n": len(g), "missing_n": int(np.sum(~np.isfinite(x))),
                "missing_rate": float(np.mean(~np.isfinite(x))),
                "unique_finite_values": int(len(np.unique(finite))),
                "feature_mean": float(np.mean(finite)) if len(finite) else np.nan,
                "feature_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan,
                "q01": float(np.quantile(finite, .01)) if len(finite) else np.nan,
                "q25": float(np.quantile(finite, .25)) if len(finite) else np.nan,
                "feature_median": float(np.median(finite)) if len(finite) else np.nan,
                "q75": float(np.quantile(finite, .75)) if len(finite) else np.nan,
                "q99": float(np.quantile(finite, .99)) if len(finite) else np.nan,
                "spearman_anchor": finite_corr(x, g["anchor_frame"], "spearman"),
            })
            corr_rows.append({
                "sequence": seq, "trajectory_policy": policy, "feature": f, "n": len(g),
                "spearman_feature_gain": finite_corr(x, y, "spearman"),
                "pearson_feature_gain": finite_corr(x, y, "pearson"),
                "partial_spearman_control_anchor": partial_spearman(x, y, g["anchor_frame"]),
                "spearman_feature_anchor": finite_corr(x, g["anchor_frame"], "spearman"),
            })
        for i, f1 in enumerate(ALL_FEATURES):
            for f2 in ALL_FEATURES[i+1:]:
                pair_rows.append({
                    "sequence": seq, "trajectory_policy": policy, "feature_1": f1, "feature_2": f2,
                    "pearson": finite_corr(g[f1], g[f2], "pearson"),
                    "spearman": finite_corr(g[f1], g[f2], "spearman"),
                })
    return pd.DataFrame(dist_rows), pd.DataFrame(corr_rows), pd.DataFrame(pair_rows)


def model_metrics(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    if len(y) == 0:
        return {"n": 0, "spearman": np.nan, "pearson": np.nan, "mae": np.nan, "rmse": np.nan,
                "r2": np.nan, "top_quartile_gain_lift": np.nan, "positive_rate_lift": np.nan}
    top_n = max(1, int(math.ceil(.25 * len(y))))
    idx = np.argsort(-pred, kind="mergesort")[:top_n]
    return {
        "n": int(len(y)), "spearman": finite_corr(pred, y, "spearman"),
        "pearson": finite_corr(pred, y, "pearson"),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "top_quartile_gain_lift": float(np.mean(y[idx]) - np.mean(y)),
        "positive_rate_lift": float(np.mean(y[idx] > 0) - np.mean(y > 0)),
    }


def loso_ablation(valid: pd.DataFrame):
    rows = []
    for heldout in SEQUENCES:
        train = valid[valid["sequence"] != heldout].copy()
        test = valid[valid["sequence"] == heldout].copy()
        for group_name, columns in FEATURE_GROUPS.items():
            for alpha in ALPHAS:
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=alpha)),
                ])
                pipe.fit(train[columns], train[TARGET].to_numpy(float))
                pred = pipe.predict(test[columns])
                test_copy = test.copy()
                test_copy["prediction"] = pred
                for policy, gp in test_copy.groupby("trajectory_policy", sort=True):
                    rows.append({
                        "heldout_sequence": heldout, "trajectory_policy": policy,
                        "feature_group": group_name, "feature_count": len(columns),
                        "ridge_alpha": alpha,
                        "train_sequences": ",".join(s for s in SEQUENCES if s != heldout),
                        "train_rows": len(train), "test_rows": len(gp),
                        "preprocessing_fit_on_train_only": True,
                        **model_metrics(gp[TARGET], gp["prediction"]),
                    })
    out = pd.DataFrame(rows)
    macro = out.groupby(["feature_group", "ridge_alpha"], as_index=False).agg(
        group_count=("spearman", "count"),
        mean_spearman=("spearman", "mean"),
        median_spearman=("spearman", "median"),
        mean_top_quartile_gain_lift=("top_quartile_gain_lift", "mean"),
        mean_positive_rate_lift=("positive_rate_lift", "mean"),
        mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_r2=("r2", "mean"),
    )
    return out, macro


def provenance_table():
    rows = [
        ("feature_scheduler_distance_kf", "state.distance_kf", "v5.4 build_online_features reads scheduler state; v5.5 calls feature extraction before the same-state counterfactual and before replaying the logged action.", "action-pre by source-code order", "High temporal-proxy risk; restart/keyframe reset changes semantics."),
        ("feature_scheduler_flow_magnitude", "state.flow_magnitude", "Same pre-action scheduler-state snapshot; feature function docstring says current input and current pre-action persistent state.", "action-pre by source-code order", "Verify scalar update semantics; cross-sequence scale may differ."),
        ("feature_scheduler_total_luma_change", "state.total_luma_change", "Same pre-action scheduler-state snapshot.", "action-pre by source-code order", "May correlate with time/scene phase; inspect scale and near-constant groups."),
        ("feature_state_radius_teacher_mean", "mean(state.radius_teacher)", "Tensor summary extracted from the current persistent state before action.", "action-pre by source-code order", "Distribution and gain association may reverse across sequences."),
        ("feature_state_confidence_teacher_mean", "mean(state.confidence_teacher)", "Tensor summary extracted from the current persistent state before action.", "action-pre by source-code order", "Check comparability of confidence semantics across sequences."),
        ("feature_state_features_teacher_0_mean", "mean(state.features_teacher[0])", "Tensor summary extracted from the current persistent state before action.", "action-pre by source-code order", "Learned-state channel scale/meaning may shift."),
        ("feature_state_features_teacher_1_mean", "mean(state.features_teacher[1])", "Tensor summary extracted from the current persistent state before action.", "action-pre by source-code order", "Learned-state channel scale/meaning may shift."),
        ("feature_state_features_teacher_2_mean", "mean(state.features_teacher[2])", "Tensor summary extracted from the current persistent state before action.", "action-pre by source-code order", "Learned-state channel scale/meaning may shift."),
        ("feature_state_features_teacher_3_mean", "mean(state.features_teacher[3])", "Tensor summary extracted from the current persistent state before action.", "action-pre by source-code order", "Learned-state channel scale/meaning may shift."),
        (TARGET, "same_state_counterfactual B0 vs B1", "v5.5 clones the exact pre-action state; future branches force propagation and log restart requests but do not execute future restarts.", "counterfactual target; not a feature", "Not full-keyframe refresh gain and not actual future auto-policy outcome; state the estimand explicitly."),
    ]
    return pd.DataFrame(rows, columns=["field", "source_or_definition", "source_code_evidence", "temporal_status", "audit_note"])


def build_report(out_dir, labels, valid, source_audit, source_summary,
                 label_summary, temporal, feature_corr, ablation_macro, provenance):
    lines = [
        "# P2.3 v5.13 收益标签与特征有效性审计", "",
        "## 1. 总结判定", "",
        "**v5.13 是 CPU-only 数据与特征诊断，不运行 VeloDepth 推理、不更新权重，也不重新生成反事实标签。**", "",
        f"- v5.5 标签文件：{len(labels)} 行；有效标签 {len(valid)} 行；非有效/跳过 {len(labels)-len(valid)} 行。",
        f"- 有效标签覆盖 {valid['sequence'].nunique()} 条序列、{valid['trajectory_policy'].nunique()} 种轨迹策略。",
        f"- 与 P2.2.2 数据中 eligible H=16 行的交集为 {source_summary['overlap_rows']} 行。",
        f"- 重叠 auto 轨迹记录的收益标签和 9 个特征全部在 1e-8 容差内一致：**{'是' if source_summary['all_fields_exact_match'] else '否，需检查 mismatch 表'}**。",
        "- 交叉核验说明两份归档在重叠 auto/H=16 样本上可追溯一致；不证明其他轨迹标签等同于实际部署的未来收益。", "",
        "## 2. 数据来源与标签语义", "",
        "- v5.5 标签来自同一动作前持久状态的 B0/B1 反事实分支：B0 继续传播，B1 更新指定持久几何状态。",
        "- 目标 `G_state_absrel_discounted_per_valid_frame` 为 H=16 折扣 AbsRel 改善方向；正值表示 B1 相对 B0 改善。",
        "- v5.5 反事实未来分支强制传播，记录但不执行未来 restart 请求。因此标签不是完整 keyframe refresh gain，也不是实际 auto 策略未来轨迹的直接收益。",
        "- 特征由 v5.4 `build_online_features` 从当前输入与动作前持久状态读取；v5.5 在反事实分支前调用它。这支持其动作前可用性，但不能代替真实部署时的运行时验证。", "",
        "### 来源一致性核验", "",
        "| 字段 | 重叠行 | 容差内一致 | 不一致 | 最大绝对差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in source_audit.iterrows():
        lines.append(f"| `{r['field']}` | {r['matched_rows']} | {r['exact_match_rows_tol_1e-8']} | {r['mismatch_rows_tol_1e-8']} | {r['max_abs_difference']:.3g} |")
    lines += ["", "## 3. 标签分布与时间非平稳性", "",
        "逐序列/轨迹的完整数值见 `p2_3_v5_13_label_summary_by_sequence_policy.csv`；时间段结果见 `p2_3_v5_13_label_summary_by_temporal_block.csv`。时间段固定为锚点 `<25`、`25–48`、`>=49`。", "",
        "| 序列 | 轨迹 | n | 均值 | 中位数 | 正收益率 | 负收益率 | |G|<=0.001 | ρ(anchor,gain) | distance=anchor 比例 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in label_summary.itertuples():
        lines.append(f"| {r.sequence} | {r.trajectory_policy} | {r.n} | {r.gain_mean:.5f} | {r.gain_median:.5f} | {r.positive_rate:.1%} | {r.negative_rate:.1%} | {r.near_zero_abs_le_1e3_rate:.1%} | {r.spearman_anchor_vs_gain:.3f} | {r.distance_equals_anchor_rate_tol_1e_6:.1%} |")
    lines += ["", "解释时应优先看同一序列不同时间段的标签变化，以及同一特征在不同序列中的关联方向；不能将相关性解释为因果。", "",
        "## 4. 特征—收益关系是否跨序列稳定", "",
        "完整的每个序列 × 轨迹 × 特征结果见 `p2_3_v5_13_feature_target_correlations.csv`。partial Spearman 通过对特征、收益和锚点秩变换后控制锚点秩趋势计算，仅用于诊断时间代理关系。", "",
        "| 序列 | 轨迹 | 特征 | Spearman(特征,收益) | 控锚点 partial Spearman | Spearman(特征,锚点) |",
        "|---|---|---|---:|---:|---:|"]
    for r in feature_corr.itertuples():
        if r.feature in ("feature_scheduler_distance_kf", "feature_scheduler_flow_magnitude",
                         "feature_state_radius_teacher_mean", "feature_state_features_teacher_2_mean"):
            lines.append(f"| {r.sequence} | {r.trajectory_policy} | `{r.feature}` | {r.spearman_feature_gain:.3f} | {r.partial_spearman_control_anchor:.3f} | {r.spearman_feature_anchor:.3f} |")
    lines += ["", "### 本轮必须关注的三类风险", "",
        "1. **时间代理：** 如果 `scheduler_distance_kf` 与 `anchor_frame` 高度一致，它可能主要编码序列进度；keyframe/restart 重置又会改变其语义。",
        "2. **跨序列关系反转：** 同一特征的收益相关性若在不同序列中改变符号，单一全局 Ridge 可能无法稳定泛化。",
        "3. **标签估计对象限制：** 反事实分支未来忽略 restart 请求，标签衡量指定传播协议下的状态写回收益，而不是实际在线策略全路径收益。", "",
        "## 5. 特征组消融与正则化敏感性", "",
        "采用完整序列留一验证；每折的缺失值填补、标准化和 Ridge 拟合只使用另外两条序列。alpha=1/10/100 全部报告，不根据留出序列选择最优 alpha。", "",
        "| 特征组 | alpha | 分组数 | 平均 Spearman | 平均 Top-25% gain lift | 平均正收益率提升 | 平均 RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in ablation_macro.sort_values(["feature_group", "ridge_alpha"]).itertuples():
        lines.append(f"| {r.feature_group} | {r.ridge_alpha:g} | {r.group_count} | {r.mean_spearman:.3f} | {r.mean_top_quartile_gain_lift:.5f} | {r.mean_positive_rate_lift:+.1%} | {r.mean_rmse:.5f} |")
    lines += ["", "这里的 12 个测试分组（3 条留出序列 × 4 种轨迹策略）不是 12 条独立序列。结果是探索性消融，不应做过度显著性推断。", "",
        "## 6. 特征来源与可用性审计", "",
        "| 字段 | 来源/定义 | 时间状态 | 主要风险 |",
        "|---|---|---|---|"]
    for r in provenance.itertuples():
        lines.append(f"| `{r.field}` | {r.source_or_definition} | {r.temporal_status} | {r.audit_note} |")
    lines += ["", "## 7. 下一步决策", "",
        "**本轮不直接决定重训预测器。** 按以下规则决策：", "",
        "- 如果 source consistency 出现不一致，先修复数据对齐与导出流程，再做任何模型比较。",
        "- 如果状态年龄与锚点高度共线，保留状态年龄作为强基线，并明确区分时间进度和真实状态退化。",
        "- 如果特征—收益关系在序列间反转，优先分析序列阶段、重启和反事实目标定义，不应直接增加模型复杂度。",
        "- 如果所有特征组在 LOSO 下都弱于简单基线，下一步应重新审查标签的研究目标和动作语义，而不是投入 VeloDepth 权重训练。",
        "- 只有标签对齐可信、特征在动作前可用、跨序列排序稳定超过简单基线后，才恢复因果在线影子策略验证。", "",
        "## 8. 输出与可复现性", "",
        "本目录同时保存输入 SHA-256、数据来源交叉核验、逐序列/轨迹标签统计、逐特征统计、相关性、特征组 LOSO 消融、源代码语义审计和运行配置。该包只使用 CPU。"]
    (out_dir / "P2_3_v5_13_收益标签与特征有效性审计.md").write_text("\n".join(lines), encoding="utf-8")


def self_test():
    x = np.arange(10, dtype=float)
    y = 2*x + np.array([0,1,0,-1,0,1,0,-1,0,1], dtype=float)
    assert finite_corr(x, y, "spearman") > 0.9
    assert np.isfinite(partial_spearman(x, y, np.array([1,4,2,8,0,9,3,7,5,6], dtype=float)))
    assert set(FEATURE_GROUPS["all_9_features"]) == set(ALL_FEATURES)
    assert len(set(ALL_FEATURES)) == 9
    print("PASS: P2.3 v5.13 audit self-tests (4/4).")


def main():
    parser = argparse.ArgumentParser(description="P2.3 v5.13 label/feature stability audit")
    here = Path(__file__).resolve().parent
    parser.add_argument("--labels-csv", default=str(here / "data_p2_3_v5_13" / "p2_3_v5_5_all_same_state_anchor_labels.csv"))
    parser.add_argument("--p2-dataset-csv", default=str(here / "data_p2_3_v5_13" / "p2_2_2_valid_dataset.csv"))
    parser.add_argument("--out-dir", default="outputs/p2_3_v5_13_label_feature_stability_audit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    labels_path, p2_path = Path(args.labels_csv), Path(args.p2_dataset_csv)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing v5.5 label CSV: {labels_path}")
    if not p2_path.is_file():
        raise FileNotFoundError(f"Missing P2.2.2 dataset CSV: {p2_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, valid, p2_h16 = read_inputs(labels_path, p2_path)
    source_audit, source_summary = source_consistency_audit(valid, p2_h16)
    label_summary = label_summaries(valid)
    temporal = temporal_summaries(valid)
    feature_dist, feature_corr, feature_pairs = feature_distributions_and_correlations(valid)
    ablation, ablation_macro = loso_ablation(valid)
    provenance = provenance_table()

    source_audit.to_csv(out_dir / "p2_3_v5_13_source_consistency.csv", index=False)
    label_summary.to_csv(out_dir / "p2_3_v5_13_label_summary_by_sequence_policy.csv", index=False)
    temporal.to_csv(out_dir / "p2_3_v5_13_label_summary_by_temporal_block.csv", index=False)
    feature_dist.to_csv(out_dir / "p2_3_v5_13_feature_distribution_by_sequence_policy.csv", index=False)
    feature_corr.to_csv(out_dir / "p2_3_v5_13_feature_target_correlations.csv", index=False)
    feature_pairs.to_csv(out_dir / "p2_3_v5_13_feature_pair_correlations.csv", index=False)
    ablation.to_csv(out_dir / "p2_3_v5_13_loso_feature_ablation.csv", index=False)
    ablation_macro.to_csv(out_dir / "p2_3_v5_13_loso_feature_ablation_macro.csv", index=False)
    provenance.to_csv(out_dir / "p2_3_v5_13_feature_provenance_audit.csv", index=False)

    metadata = {
        "experiment": "P2.3 v5.13",
        "type": "CPU-only same-state label and feature stability audit",
        "labels_csv": str(labels_path),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "p2_2_2_csv": str(p2_path),
        "p2_2_2_sha256": hashlib.sha256(p2_path.read_bytes()).hexdigest(),
        "raw_label_rows": int(len(labels)),
        "valid_label_rows": int(len(valid)),
        "p2_2_2_h16_eligible_rows": int(len(p2_h16)),
        "cross_source_overlap_rows": int(source_summary["overlap_rows"]),
        "cross_source_all_fields_match_tol_1e8": bool(source_summary["all_fields_exact_match"]),
        "sequences": list(SEQUENCES), "trajectory_policies": list(POLICIES),
        "target": TARGET,
        "target_semantics": "same-state B1 persistent geometry writeback vs B0 propagation; H=16 discounted AbsRel; future restart requests are logged but not executed",
        "feature_groups": FEATURE_GROUPS,
        "loso": "whole sequence held out; preprocessing and Ridge fitted on other two sequences only",
        "ridge_alphas_reported_not_selected": list(ALPHAS),
        "velodepth_inference_run": False, "velodepth_weights_updated": False,
    }
    (out_dir / "p2_3_v5_13_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    build_report(out_dir, labels, valid, source_audit, source_summary,
                 label_summary, temporal, feature_corr, ablation_macro, provenance)

    print(f"Rows: raw={len(labels)}, valid={len(valid)}, P2.2.2 eligible H16={len(p2_h16)}")
    print(f"Cross-source overlap={source_summary['overlap_rows']}; all fields exact={source_summary['all_fields_exact_match']}")
    print("\nLabel summary (sequence/policy):")
    print(label_summary[["sequence","trajectory_policy","n","gain_mean","positive_rate","spearman_anchor_vs_gain","distance_equals_anchor_rate_tol_1e_6"]].to_string(index=False))
    print("\nLOSO feature-ablation macro:")
    print(ablation_macro.sort_values(["feature_group","ridge_alpha"]).to_string(index=False))
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
