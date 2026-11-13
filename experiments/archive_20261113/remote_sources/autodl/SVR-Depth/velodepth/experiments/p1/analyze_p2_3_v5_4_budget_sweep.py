#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc diagnostics for P2.3 v5.4.

This script does not feed H=16 target labels into the online controller. It joins
completed replay scores/actions with the P2.2.2 H=16 target only after inference,
to audit predictor validity and selected-action gain.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

CAUSAL_POLICIES = (
    "causal_budget_state_only",
    "causal_budget_motion_plus_learned",
)
POLICIES = ("auto", "budget_fixed", *CAUSAL_POLICIES)
TARGET = "G_state_absrel_discounted_per_valid_frame"


def safe_num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def corr(a, b, method="pearson"):
    x = pd.to_numeric(pd.Series(a), errors="coerce")
    y = pd.to_numeric(pd.Series(b), errors="coerce")
    d = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 3 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        return float("nan")
    if method == "spearman":
        return float(d["x"].rank(method="average").corr(d["y"].rank(method="average")))
    return float(d["x"].corr(d["y"]))


def parse_bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/p2_3_v5_4_budget_sweep")
    ap.add_argument("--dataset", default="outputs/p2_2_2/p2_2_2_valid_dataset.csv")
    ap.add_argument("--sequences", default="crowd2,balloon2,crowd3")
    ap.add_argument("--budgets", default="5,8,10")
    args = ap.parse_args()

    root = Path(args.root)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing P2.2.2 dataset: {dataset_path}")
    ds = pd.read_csv(dataset_path, encoding="utf-8-sig")
    ds["sequence"] = ds["sequence"].astype(str)
    ds["horizon_H"] = pd.to_numeric(ds["horizon_H"], errors="coerce")
    ds["anchor_frame"] = pd.to_numeric(ds["anchor_frame"], errors="coerce")
    ds[TARGET] = pd.to_numeric(ds[TARGET], errors="coerce")
    if "eligible" in ds:
        ds = ds[parse_bool_series(ds["eligible"])]
    target = ds[(ds["horizon_H"] == 16) & np.isfinite(ds[TARGET])][
        ["sequence", "anchor_frame", TARGET, "future_gt_coverage_ratio", "future_metric_pair_coverage_ratio"]
    ].drop_duplicates(["sequence", "anchor_frame"])

    sequences = [x.strip() for x in args.sequences.split(",") if x.strip()]
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    comparison_rows = []
    gain_rows = []
    missing = []

    for seq in sequences:
        for k in budgets:
            out = root / f"{seq}_K{k}"
            summary_path = out / "p2_3_online_summary.csv"
            frame_path = out / "p2_3_online_frame_metrics.csv"
            meta_path = out / "p2_3_online_metadata.json"
            if not (summary_path.exists() and frame_path.exists() and meta_path.exists()):
                missing.append(f"{seq}_K{k}")
                continue
            summary = pd.read_csv(summary_path)
            frames = pd.read_csv(frame_path)
            meta = __import__("json").loads(meta_path.read_text(encoding="utf-8"))
            frames["frame_index"] = pd.to_numeric(frames["frame_index"], errors="coerce")
            frames["predictor_score"] = pd.to_numeric(frames["predictor_score"], errors="coerce")
            frames["sequence"] = frames["sequence"].astype(str)

            summary_by_policy = summary.set_index("policy")
            fixed_row = summary_by_policy.loc["budget_fixed"] if "budget_fixed" in summary_by_policy.index else None
            auto_row = summary_by_policy.loc["auto"] if "auto" in summary_by_policy.index else None

            for policy in POLICIES:
                if policy not in summary_by_policy.index:
                    continue
                r = summary_by_policy.loc[policy]
                n_writebacks = int(r.get("state_writeback_count", 0))
                fixed_absrel = safe_num(fixed_row.get("mean_absrel")) if fixed_row is not None else float("nan")
                fixed_rmse = safe_num(fixed_row.get("mean_rmse")) if fixed_row is not None else float("nan")
                fixed_delta1 = safe_num(fixed_row.get("mean_delta1")) if fixed_row is not None else float("nan")
                absrel = safe_num(r.get("mean_absrel"))
                rmse = safe_num(r.get("mean_rmse"))
                delta1 = safe_num(r.get("mean_delta1"))
                is_causal = policy in CAUSAL_POLICIES
                cap = k if is_causal or policy == "budget_fixed" else float("nan")
                utilization = n_writebacks / cap if isinstance(cap, (int, float)) and math.isfinite(float(cap)) and cap > 0 else float("nan")
                comparison_rows.append({
                    "sequence": seq,
                    "budget_cap_K": k,
                    "policy": policy,
                    "valid_metric_frames": int(r.get("frames_with_valid_metrics", 0)),
                    "writeback_cap": cap,
                    "successful_writebacks": n_writebacks,
                    "budget_utilization": utilization,
                    "budget_respected": (n_writebacks <= cap) if math.isfinite(float(cap)) else True,
                    "forced_budget_actions": int(r.get("causal_budget_forced_decisions", 0)) if is_causal else 0,
                    "mean_absrel": absrel,
                    "absrel_pct_vs_budget_fixed": (absrel / fixed_absrel - 1) * 100 if math.isfinite(absrel) and math.isfinite(fixed_absrel) and fixed_absrel != 0 else float("nan"),
                    "mean_rmse": rmse,
                    "rmse_pct_vs_budget_fixed": (rmse / fixed_rmse - 1) * 100 if math.isfinite(rmse) and math.isfinite(fixed_rmse) and fixed_rmse != 0 else float("nan"),
                    "mean_delta1": delta1,
                    "delta1_pp_vs_budget_fixed": (delta1 - fixed_delta1) * 100 if math.isfinite(delta1) and math.isfinite(fixed_delta1) else float("nan"),
                    "auto_restarts": int(r.get("auto_restart_count", 0)),
                    "mean_decision_ms": safe_num(r.get("mean_decision_ms")),
                    "mean_model_action_ms": safe_num(r.get("mean_model_action_ms")),
                    "fixed_budget_frames": ";".join(map(str, meta.get("budget_fixed_action_frames", []))),
                })

                if policy == "budget_fixed":
                    selected = frames[(frames["policy"] == policy) & (frames["action"] == "state_writeback")].copy()
                    selected["frame_index"] = pd.to_numeric(selected["frame_index"], errors="coerce")
                    selected = selected.merge(target, left_on=["sequence", "frame_index"], right_on=["sequence", "anchor_frame"], how="left")
                    all_candidate = frames[(frames["policy"] == policy) & frames["frame_index"].isin(target["anchor_frame"])].copy()
                    all_candidate = all_candidate.merge(target, left_on=["sequence", "frame_index"], right_on=["sequence", "anchor_frame"], how="inner")
                    gain_rows.append({
                        "sequence": seq, "budget_cap_K": k, "policy": policy,
                        "candidate_score_target_pearson": float("nan"),
                        "candidate_score_target_spearman": float("nan"),
                        "score_target_n": 0,
                        "successful_writebacks": n_writebacks,
                        "selected_actions_with_H16_target": int(np.isfinite(pd.to_numeric(selected[TARGET], errors="coerce")).sum()),
                        "selected_target_mean": safe_num(pd.to_numeric(selected[TARGET], errors="coerce").mean()),
                        "all_candidate_target_mean": safe_num(pd.to_numeric(all_candidate[TARGET], errors="coerce").mean()),
                        "selected_target_lift": safe_num(pd.to_numeric(selected[TARGET], errors="coerce").mean() - pd.to_numeric(all_candidate[TARGET], errors="coerce").mean()),
                        "selected_top_quartile_target_rate": float("nan"),
                    })
                    continue

                if not is_causal:
                    continue
                policy_frames = frames[frames["policy"] == policy].copy()
                candidate = policy_frames[np.isfinite(policy_frames["predictor_score"])].copy()
                candidate = candidate.merge(
                    target, left_on=["sequence", "frame_index"],
                    right_on=["sequence", "anchor_frame"], how="inner"
                )
                candidate = candidate.replace([np.inf, -np.inf], np.nan).dropna(subset=["predictor_score", TARGET])
                selected = policy_frames[policy_frames["action"] == "state_writeback"].copy()
                selected = selected.merge(
                    target, left_on=["sequence", "frame_index"],
                    right_on=["sequence", "anchor_frame"], how="left"
                )
                candidate_mean = safe_num(candidate[TARGET].mean())
                selected_targets = pd.to_numeric(selected[TARGET], errors="coerce")
                selected_mean = safe_num(selected_targets.mean())
                q75 = safe_num(candidate[TARGET].quantile(0.75)) if len(candidate) else float("nan")
                top_rate = float((selected_targets.dropna() >= q75).mean()) if math.isfinite(q75) and selected_targets.notna().any() else float("nan")
                gain_rows.append({
                    "sequence": seq,
                    "budget_cap_K": k,
                    "policy": policy,
                    "candidate_score_target_pearson": corr(candidate["predictor_score"], candidate[TARGET], "pearson"),
                    "candidate_score_target_spearman": corr(candidate["predictor_score"], candidate[TARGET], "spearman"),
                    "score_target_n": int(len(candidate)),
                    "successful_writebacks": n_writebacks,
                    "selected_actions_with_H16_target": int(selected_targets.notna().sum()),
                    "selected_target_mean": selected_mean,
                    "all_candidate_target_mean": candidate_mean,
                    "selected_target_lift": selected_mean - candidate_mean if math.isfinite(selected_mean) and math.isfinite(candidate_mean) else float("nan"),
                    "selected_top_quartile_target_rate": top_rate,
                })

    outdir = root
    outdir.mkdir(parents=True, exist_ok=True)
    comp = pd.DataFrame(comparison_rows)
    gain = pd.DataFrame(gain_rows)
    comp_path = outdir / "p2_3_v5_4_budget_comparison.csv"
    gain_path = outdir / "p2_3_v5_4_gain_diagnostics.csv"
    comp.to_csv(comp_path, index=False)
    gain.to_csv(gain_path, index=False)

    report = [
        "# P2.3 v5.4 预设预算实验：自动汇总与收益诊断",
        "",
        "## 解释规则",
        "- K 是预先指定的成功写回上限，不是必须完成的目标；不允许末尾强制补足。",
        "- `budget_fixed` 是只依赖帧数、候选锚点和 K 的均匀间隔动作基线，不读取预测分数或未来收益标签。",
        "- 预测器分数与 H=16 目标的相关性、所选动作的目标值均在回放完成后计算；这些标签没有进入在线决策。",
        "- `*_pct_vs_budget_fixed` 为正表示比均匀固定预算基线更差；负表示更好。若实际写回次数不同，不能视为严格等动作数比较。",
        "",
        "## 深度指标与实际动作数",
        "",
        "| 序列 | K 上限 | 策略 | 成功写回 | 利用率 | AbsRel | 相对 budget_fixed | RMSE | 相对 budget_fixed | δ1 | δ1 变化 | 强制动作 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if len(comp):
        for _, r in comp.iterrows():
            absrel_pct = f"{r['absrel_pct_vs_budget_fixed']:+.2f}%" if math.isfinite(safe_num(r["absrel_pct_vs_budget_fixed"])) else "—"
            rmse_pct = f"{r['rmse_pct_vs_budget_fixed']:+.2f}%" if math.isfinite(safe_num(r["rmse_pct_vs_budget_fixed"])) else "—"
            delta_pp = f"{r['delta1_pp_vs_budget_fixed']:+.3f} pp" if math.isfinite(safe_num(r["delta1_pp_vs_budget_fixed"])) else "—"
            util = f"{r['budget_utilization']:.0%}" if math.isfinite(safe_num(r["budget_utilization"])) else "—"
            report.append(
                f"| {r['sequence']} | {int(r['budget_cap_K'])} | {r['policy']} | "
                f"{int(r['successful_writebacks'])} | {util} | {r['mean_absrel']:.6f} | {absrel_pct} | "
                f"{r['mean_rmse']:.6f} | {rmse_pct} | {r['mean_delta1']:.6f} | {delta_pp} | {int(r['forced_budget_actions'])} |"
            )
    report += [
        "",
        "## 预测分数是否识别了高收益帧？",
        "",
        "| 序列 | K 上限 | 策略 | 分数-目标 Pearson | Spearman | 有效配对数 | 成功写回 | 有 H=16 标签的写回 | 选中帧目标均值 | 候选目标均值 | 选择提升 | 选中帧落入候选目标前 25% 比例 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if len(gain):
        for _, r in gain.iterrows():
            def fmt(x, spec=".4f"):
                return format(x, spec) if math.isfinite(safe_num(x)) else "—"
            report.append(
                f"| {r['sequence']} | {int(r['budget_cap_K'])} | {r['policy']} | "
                f"{fmt(r['candidate_score_target_pearson'])} | {fmt(r['candidate_score_target_spearman'])} | "
                f"{int(r['score_target_n'])} | {int(r['successful_writebacks'])} | "
                f"{int(r['selected_actions_with_H16_target'])} | {fmt(r['selected_target_mean'])} | "
                f"{fmt(r['all_candidate_target_mean'])} | {fmt(r['selected_target_lift'])} | "
                f"{fmt(r['selected_top_quartile_target_rate'], '.1%')} |"
            )
    report += [
        "",
        "## 判断边界",
        "",
        "- 相关性接近 0 或为负，表示当前分数未能稳定排序 H=16 目标；需结合每序列小样本数解释。",
        "- 正的 `selected_target_lift` 表示被选中的写回锚点平均目标值高于可匹配候选锚点；这仍是离线诊断，不是独立因果证明。",
        "- K 是上限，若学习策略使用的成功写回少于 `budget_fixed`，质量对比也反映不同动作数量，必须同时报告。",
        "- 当前每个序列只有 86 帧、三个场景，不能据此声称跨数据集泛化、节省计算或 RK3588 实时性。",
        "",
        f"缺少结果目录：{', '.join(missing) if missing else '无'}",
        "",
    ]
    report_path = outdir / "P2_3_v5_4_结果分析.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote: {comp_path}")
    print(f"Wrote: {gain_path}")
    print(f"Wrote: {report_path}")
    print(f"Completed result directories: {len(set((r['sequence'], r['budget_cap_K']) for r in comparison_rows))}/{len(sequences)*len(budgets)}")


if __name__ == "__main__":
    main()
