#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare v5.15 restart-aware labels against legacy v5.5 forced-propagation labels."""
import argparse
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np

OLD_TARGET = "G_state_absrel_discounted_per_valid_frame"
NEW_TARGET = "G_state_absrel_restart_aware_discounted_per_valid_frame"
KEY = ["sequence", "trajectory_policy", "anchor_frame"]


def corr(a, b, method="spearman"):
    x = pd.to_numeric(pd.Series(a), errors="coerce")
    y = pd.to_numeric(pd.Series(b), errors="coerce")
    m = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return float("nan")
    return float(x[m].corr(y[m], method=method))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-labels", required=True)
    ap.add_argument("--new-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    old = pd.read_csv(args.old_labels)
    old = old[old["status"].astype(str).eq("ok")].copy()
    if not set(KEY + [OLD_TARGET]).issubset(old.columns):
        raise ValueError("Old v5.5 labels missing expected columns.")
    paths = sorted(Path(args.new_root).rglob("p2_3_v5_15_same_state_anchor_labels.csv"))
    if not paths:
        raise FileNotFoundError(f"No v5.15 label CSV found under {args.new_root}")
    new_parts = [pd.read_csv(p).assign(_source_file=str(p)) for p in paths]
    new = pd.concat(new_parts, ignore_index=True)
    new = new[new["status"].astype(str).eq("ok")].copy()
    required = KEY + [NEW_TARGET, "future_auto_restarts_executed_B0", "future_auto_restarts_executed_B1"]
    if not set(required).issubset(new.columns):
        raise ValueError("New v5.15 labels missing expected columns.")
    merged = old[KEY + [OLD_TARGET]].merge(new[required + ["future_metric_pairs"]], on=KEY, how="inner", validate="one_to_one")
    if merged.empty:
        raise RuntimeError("No overlapping valid labels between v5.5 and v5.15.")
    x = pd.to_numeric(merged[OLD_TARGET], errors="coerce")
    y = pd.to_numeric(merged[NEW_TARGET], errors="coerce")
    valid = np.isfinite(x) & np.isfinite(y)
    merged = merged.loc[valid].copy()
    x = pd.to_numeric(merged[OLD_TARGET], errors="coerce").to_numpy(float)
    y = pd.to_numeric(merged[NEW_TARGET], errors="coerce").to_numpy(float)
    merged["old_gain_sign"] = np.where(x > 0, "positive", np.where(x < 0, "negative", "zero"))
    merged["new_gain_sign"] = np.where(y > 0, "positive", np.where(y < 0, "negative", "zero"))
    merged["sign_agreement"] = ((x > 0) == (y > 0)) & ((x < 0) == (y < 0))
    merged["absolute_target_difference"] = np.abs(x - y)
    merged["relative_target_difference_abs"] = np.abs(x-y) / np.maximum(np.abs(x), 1e-12)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "p2_3_v5_15_vs_v5_5_paired_labels.csv", index=False)

    rows = []
    groups = [("ALL", "ALL", merged)] + [
        (str(s), "ALL", g) for s, g in merged.groupby("sequence", sort=True)
    ] + [
        (str(s), str(p), g) for (s, p), g in merged.groupby(["sequence", "trajectory_policy"], sort=True)
    ]
    for seq, pol, g in groups:
        ox = pd.to_numeric(g[OLD_TARGET], errors="coerce")
        ny = pd.to_numeric(g[NEW_TARGET], errors="coerce")
        rows.append({
            "sequence": seq, "trajectory_policy": pol, "n": int(len(g)),
            "old_mean": float(ox.mean()), "new_mean": float(ny.mean()),
            "old_median": float(ox.median()), "new_median": float(ny.median()),
            "old_positive_rate": float((ox > 0).mean()), "new_positive_rate": float((ny > 0).mean()),
            "sign_agreement_rate": float(g["sign_agreement"].mean()),
            "spearman_old_vs_new": corr(ox, ny, "spearman"),
            "pearson_old_vs_new": corr(ox, ny, "pearson"),
            "mean_abs_target_difference": float(g["absolute_target_difference"].mean()),
            "median_B0_auto_restarts": float(pd.to_numeric(g["future_auto_restarts_executed_B0"]).median()),
            "median_B1_auto_restarts": float(pd.to_numeric(g["future_auto_restarts_executed_B1"]).median()),
            "anchors_with_any_restart_rate": float(((pd.to_numeric(g["future_auto_restarts_executed_B0"]).fillna(0) + pd.to_numeric(g["future_auto_restarts_executed_B1"]).fillna(0)) > 0).mean()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "p2_3_v5_15_vs_v5_5_comparison_summary.csv", index=False)
    overall = summary[(summary.sequence == "ALL") & (summary.trajectory_policy == "ALL")].iloc[0]
    report = f"""# P2.3 v5.15：重启感知标签对照报告

## 1. 运行范围

本报告对比 v5.5 的强制传播标签与 v5.15 的分支独立自动重启标签。两者均从同一个动作前状态分支，B1 只写回四个持久几何字段；差别在未来 H=16 窗口：v5.5 记录 restart 请求但不执行，v5.15 在各分支独立执行同帧 keyframe restart。

**限制：** v5.15 没有重放或重新计算未来的策略主动刷新动作，因此它仍不是完整在线策略收益。当前预测器仍由旧的 P2.2.2 强制传播目标训练，本轮相关性只衡量旧分数对新标签的迁移能力。

## 2. 总体对照

- 配对有效标签数：{int(overall.n)}
- v5.5 标签平均值：{overall.old_mean:.6f}
- v5.15 标签平均值：{overall.new_mean:.6f}
- 正收益率：v5.5 {overall.old_positive_rate:.1%}；v5.15 {overall.new_positive_rate:.1%}
- 正负符号一致率：{overall.sign_agreement_rate:.1%}
- Spearman(v5.5, v5.15)：{overall.spearman_old_vs_new:.3f}
- Pearson(v5.5, v5.15)：{overall.pearson_old_vs_new:.3f}
- 平均绝对标签差：{overall.mean_abs_target_difference:.6f}
- 至少一个分支发生自动重启的配对记录比例：{overall.anchors_with_any_restart_rate:.1%}

## 3. 解释规则

- 如果新旧标签排序和符号大体一致：强制传播协议可能足以近似自动重启感知收益，但仍需检查每条序列/策略分组。
- 如果排序明显改变：旧标签不能直接用于训练目标与实际 restart-aware 行为之间的迁移；后续需要决定采用新标签重新训练预测器，或明确把论文主张收窄为受控状态写回价值。
- 如果重启触发很少：不能由此宣称 restart 对长期收益不重要；需要检查 H、采样锚点和序列覆盖是否足以触发。

## 4. 不能据此宣称的结论

本轮不证明完整策略的在线收益、不证明相同预算下的性能优势、不证明端到端延迟/FPS，也不修改 VeloDepth 权重。下一步仍需在固定计算预算下比较实际在线策略。
"""
    (out / "P2_3_v5_15_重启感知标签对照报告.md").write_text(report, encoding="utf-8")
    print(summary[summary.sequence.isin(["ALL"])].to_string(index=False))
    print(f"Wrote comparison outputs to {out}")


if __name__ == "__main__":
    main()
