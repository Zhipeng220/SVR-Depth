#!/usr/bin/env python3
"""Offline R3.4: leave-one-sequence-out validation of geometry-feature increment."""
from __future__ import annotations
import argparse
import io
import json
import math
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

SEQUENCES = ("crowd2", "balloon2", "crowd3")
TARGET = "gain_absrel_discounted_per_valid_frame"
FEATURE_SETS = {
    "MeanOnly": [],
    "AgeOnly": ["distance_kf"],
    "RawMedianOnly": ["raw_rel_median"],
    "FitMedianOnly": ["fit_residual_norm_median"],
    "AgePlusRawMedian": ["distance_kf", "raw_rel_median"],
    "AgePlusFitMedian": ["distance_kf", "fit_residual_norm_median"],
    "AgePlusBothMedians": ["distance_kf", "raw_rel_median", "fit_residual_norm_median"],
    "AgePlusRawMeanControl": ["distance_kf", "raw_rel_mean"],
    "AgePlusFlowControl": ["distance_kf", "flow_magnitude"],
}
ALPHA = 1.0  # fixed before evaluation; no tuning on held-out sequence


def load_diagnostics(source: Path) -> pd.DataFrame:
    frames = []
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            for name in z.namelist():
                m = re.search(r"/full/(crowd2|balloon2|crowd3)/r33_geometry_gain_diagnostics\.csv$", name)
                if m:
                    df = pd.read_csv(io.BytesIO(z.read(name)))
                    if "sequence" not in df:
                        df["sequence"] = m.group(1)
                    frames.append(df)
    elif source.is_dir():
        for seq in SEQUENCES:
            matches = list(source.glob(f"**/full/{seq}/r33_geometry_gain_diagnostics.csv"))
            if not matches:
                matches = list(source.glob(f"**/{seq}/r33_geometry_gain_diagnostics.csv"))
            if matches:
                df = pd.read_csv(matches[0])
                if "sequence" not in df:
                    df["sequence"] = seq
                frames.append(df)
    else:
        raise FileNotFoundError(f"R3.3 source not found: {source}")
    if not frames:
        raise ValueError("No R3.3 geometry CSVs found. Expected full/{crowd2,balloon2,crowd3}/r33_geometry_gain_diagnostics.csv")
    data = pd.concat(frames, ignore_index=True)
    required = {"sequence", "anchor_frame_zero_based", "is_policy_action_anchor",
                "counterfactual_status", TARGET, *[f for fs in FEATURE_SETS.values() for f in fs],
                "raw_rel_mean", "flow_magnitude"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data[data["sequence"].isin(SEQUENCES)].copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data[(data["counterfactual_status"] == "ok") & np.isfinite(data[TARGET])].copy()
    data = data.sort_values(["sequence", "anchor_frame_zero_based"]).reset_index(drop=True)
    data["row_id"] = data["sequence"].astype(str) + ":" + data["anchor_frame_zero_based"].astype(int).astype(str)
    return data


def fit_predict_ridge(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    y = train[TARGET].to_numpy(dtype=float)
    if not features:
        return np.full(len(test), float(np.mean(y)))
    xtr = train[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    xte = test[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    # Imputation values are learned on the training split only.
    med = np.nanmedian(xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    xtr = np.where(np.isfinite(xtr), xtr, med)
    xte = np.where(np.isfinite(xte), xte, med)
    xm = xtr.mean(axis=0)
    xs = xtr.std(axis=0)
    xs = np.where(xs > 1e-12, xs, 1.0)
    xtr = (xtr - xm) / xs
    xte = (xte - xm) / xs
    ym = float(y.mean())
    ys = float(y.std())
    if ys <= 1e-12:
        return np.full(len(test), ym)
    ytr = (y - ym) / ys
    # Intercept is represented by centering; alpha is fixed and features are standardized.
    gram = xtr.T @ xtr + ALPHA * np.eye(xtr.shape[1])
    beta = np.linalg.solve(gram, xtr.T @ ytr)
    return (xte @ beta) * ys + ym


def corr(a, b, method="pearson"):
    aa = pd.Series(np.asarray(a, dtype=float))
    bb = pd.Series(np.asarray(b, dtype=float))
    if aa.nunique(dropna=True) < 2 or bb.nunique(dropna=True) < 2:
        return float("nan")
    return float(aa.corr(bb, method=method))


def score_predictions(df: pd.DataFrame, pred_col: str, scope: str) -> dict:
    d = df.copy()
    target_col = TARGET if TARGET in d.columns else "actual_gain"
    y = d[target_col].to_numpy(dtype=float)
    pred = d[pred_col].to_numpy(dtype=float)
    commit_model = pred > 0.0  # fixed zero threshold from gain definition; no test-fold tuning
    age_commit = pd.to_numeric(d["distance_kf"], errors="coerce").to_numpy(dtype=float) >= 16.0
    actual_action = d["is_policy_action_anchor"].astype(int).to_numpy() == 1
    utility_model = np.where(commit_model, y, 0.0)
    utility_age = np.where(age_commit, y, 0.0)
    utility_all = y
    negative = y < 0.0
    predicted_negative = ~commit_model
    committed_n = int(commit_model.sum())
    return {
        "scope": scope,
        "n": int(len(d)),
        "mae": float(np.mean(np.abs(y - pred))) if len(d) else float("nan"),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))) if len(d) else float("nan"),
        "pearson_gain_prediction": corr(y, pred, "pearson"),
        "spearman_gain_prediction": corr(y, pred, "spearman"),
        "sign_accuracy": float(np.mean((pred > 0.0) == (y > 0.0))) if len(d) else float("nan"),
        "negative_gain_recall_discarded": float(np.sum(predicted_negative & negative) / np.sum(negative)) if np.sum(negative) else float("nan"),
        "negative_gain_precision_discarded": float(np.sum(predicted_negative & negative) / np.sum(predicted_negative)) if np.sum(predicted_negative) else float("nan"),
        "commit_rate_model": float(np.mean(commit_model)) if len(d) else float("nan"),
        "negative_rate_among_model_commits": float(np.mean(negative[commit_model])) if committed_n else float("nan"),
        "local_utility_model_mean": float(np.mean(utility_model)) if len(d) else float("nan"),
        "local_utility_agegate_mean": float(np.mean(utility_age)) if len(d) else float("nan"),
        "local_utility_always_commit_mean": float(np.mean(utility_all)) if len(d) else float("nan"),
        "local_utility_never_commit_mean": 0.0,
        "agegate_commit_rate": float(np.mean(age_commit)) if len(d) else float("nan"),
        "agegate_negative_gain_recall_discarded": float(np.sum((~age_commit) & negative) / np.sum(negative)) if np.sum(negative) else float("nan"),
        "agegate_positive_gain_recall_committed": float(np.sum(age_commit & (~negative) & (y > 0.0)) / np.sum(y > 0.0)) if np.sum(y > 0.0) else float("nan"),
        "agegate_sign_accuracy": float(np.mean(age_commit == (y > 0.0))) if len(d) else float("nan"),
        "agegate_negative_rate_among_commits": float(np.mean(negative[age_commit])) if np.any(age_commit) else float("nan"),
        "actual_action_anchor_count": int(actual_action.sum()),
    }


def evaluate(data: pd.DataFrame, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    for heldout in SEQUENCES:
        train = data[data["sequence"] != heldout].copy()
        test = data[data["sequence"] == heldout].copy()
        if train.empty or test.empty:
            continue
        for name, features in FEATURE_SETS.items():
            preds = fit_predict_ridge(train, test, features)
            for (_, row), pred in zip(test.iterrows(), preds):
                pred_rows.append({
                    "row_id": row["row_id"], "sequence": row["sequence"],
                    "anchor_frame_zero_based": int(row["anchor_frame_zero_based"]),
                    "is_policy_action_anchor": int(row["is_policy_action_anchor"]),
                    "heldout_sequence": heldout, "model": name, "features": "+".join(features) if features else "(training mean)",
                    "actual_gain": float(row[TARGET]), "predicted_gain": float(pred),
                    "distance_kf": float(row["distance_kf"]) if pd.notna(row["distance_kf"]) else np.nan,
                    "raw_rel_median": float(row["raw_rel_median"]) if pd.notna(row["raw_rel_median"]) else np.nan,
                    "fit_residual_norm_median": float(row["fit_residual_norm_median"]) if pd.notna(row["fit_residual_norm_median"]) else np.nan,
                    "raw_rel_mean": float(row["raw_rel_mean"]) if pd.notna(row["raw_rel_mean"]) else np.nan,
                    "flow_magnitude": float(row["flow_magnitude"]) if pd.notna(row["flow_magnitude"]) else np.nan,
                })
    pred = pd.DataFrame(pred_rows)
    if pred.empty:
        raise ValueError("No leave-one-sequence-out predictions could be generated.")
    pred.to_csv(out_dir / "r34_loso_predictions.csv", index=False, encoding="utf-8-sig")

    metric_rows = []
    for model, dmodel in pred.groupby("model", sort=False):
        for scope, d in [
            ("all_diagnostic_anchors", dmodel),
            ("actual_policy_action_anchors", dmodel[dmodel["is_policy_action_anchor"] == 1]),
        ]:
            if d.empty:
                continue
            # Aggregate out-of-sequence predictions. Each row was predicted by a model
            # trained only on the other two sequences.
            row = score_predictions(d, "predicted_gain", scope)
            row["model"] = model
            row["heldout_sequence"] = "OOF_ALL"
            metric_rows.append(row)
            for seq, ds in d.groupby("sequence"):
                r = score_predictions(ds, "predicted_gain", scope)
                r["model"] = model
                r["heldout_sequence"] = seq
                metric_rows.append(r)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "r34_loso_metrics.csv", index=False, encoding="utf-8-sig")

    # Compare only predeclared candidate models against AgeOnly; also compare their
    # action-selection utility against the explicit AgeGate age>=16 rule.
    comparisons = []
    for scope in ["all_diagnostic_anchors", "actual_policy_action_anchors"]:
        block = metrics[(metrics.scope == scope) & (metrics.heldout_sequence == "OOF_ALL")].set_index("model")
        if "AgeOnly" not in block.index:
            continue
        for model in ["AgePlusRawMedian", "AgePlusFitMedian", "AgePlusBothMedians",
                      "AgePlusRawMeanControl", "AgePlusFlowControl"]:
            if model not in block.index:
                continue
            m, a = block.loc[model], block.loc["AgeOnly"]
            comparisons.append({
                "scope": scope, "model": model,
                "n": int(m["n"]),
                "mae_model": m["mae"], "mae_age_only": a["mae"],
                "delta_mae_vs_age_only_negative_is_better": m["mae"] - a["mae"],
                "rmse_model": m["rmse"], "rmse_age_only": a["rmse"],
                "delta_rmse_vs_age_only_negative_is_better": m["rmse"] - a["rmse"],
                "model_local_utility_mean": m["local_utility_model_mean"],
                "agegate_rule_local_utility_mean": m["local_utility_agegate_mean"],
                "delta_model_utility_vs_agegate": m["local_utility_model_mean"] - m["local_utility_agegate_mean"],
                "model_negative_gain_recall": m["negative_gain_recall_discarded"],
                "age_only_negative_gain_recall": a["negative_gain_recall_discarded"],
                "agegate_rule_negative_gain_recall": m["agegate_negative_gain_recall_discarded"],
                "note": "OOF local counterfactual utility; not a closed-loop policy result"
            })
    comp = pd.DataFrame(comparisons)
    comp.to_csv(out_dir / "r34_incremental_comparisons.csv", index=False, encoding="utf-8-sig")

    # Audit dataset and actual action anchors, independent of model.
    audit = []
    for seq, d in data.groupby("sequence"):
        y = d[TARGET].to_numpy(float)
        action = d[d["is_policy_action_anchor"].astype(int) == 1]
        audit.append({
            "sequence": seq, "valid_diagnostic_anchors": int(len(d)),
            "positive_gain": int(np.sum(y > 0)), "negative_gain": int(np.sum(y < 0)),
            "zero_gain": int(np.sum(y == 0)),
            "actual_policy_action_anchors_with_valid_gain": int(len(action)),
            "actual_action_positive_gain": int(np.sum(action[TARGET].to_numpy(float) > 0)),
            "actual_action_negative_gain": int(np.sum(action[TARGET].to_numpy(float) < 0)),
        })
    audit_df = pd.DataFrame(audit)
    audit_df.to_csv(out_dir / "r34_dataset_audit.csv", index=False, encoding="utf-8-sig")

    # Report with pooled OOF metrics; no p-values because only 3 sequences and overlapping windows.
    allm = metrics[(metrics.scope == "all_diagnostic_anchors") & (metrics.heldout_sequence == "OOF_ALL")].set_index("model")
    actm = metrics[(metrics.scope == "actual_policy_action_anchors") & (metrics.heldout_sequence == "OOF_ALL")].set_index("model")
    def f(v, n=5):
        return "NA" if pd.isna(v) else f"{float(v):.{n}f}"
    report = []
    report.append("# R3.4 候选几何特征增量验证报告")
    report.append("")
    report.append("## 1. 研究问题与协议")
    report.append("")
    report.append("本轮完全离线，使用 R3.3 已有的同状态反事实收益标签，不运行 GPU、不重算候选、不训练神经网络。按序列留一法（LOSO）评估：每折用另外两条序列的所有有效锚点训练固定 L2 Ridge 模型，在未参与训练的序列上预测。特征标准化、缺失值填补和目标标准化均只使用训练折统计量。正负收益决策阈值固定为 0，不在测试序列上调阈值。")
    report.append("")
    report.append("主要比较：AgeOnly 与 AgePlusRawMedian、AgePlusFitMedian、AgePlusBothMedians；AgePlusRawMeanControl 和 AgePlusFlowControl 为预先指定的负面对照。Ridge alpha=1.0 固定，不做超参数搜索。")
    report.append("")
    report.append("## 2. 数据审计")
    report.append("")
    report.append(audit_df.to_markdown(index=False))
    report.append("")
    report.append(f"有效反事实标签总数：{len(data)}；正收益 {int((data[TARGET] > 0).sum())}；负收益 {int((data[TARGET] < 0).sum())}。这些 H=16 未来窗口存在重叠，53 个锚点不是独立样本。")
    report.append("")
    report.append("## 3. LOSO 连续收益预测：所有诊断锚点")
    report.append("")
    report.append("| 模型 | N | MAE ↓ | RMSE ↓ | 相关系数（Pearson） | 模型符号准确率 ↑ | 模型负收益召回率 ↑ | AgeGate负收益召回率 ↑ | 模型局部效用/锚点 ↑ | AgeGate局部效用/锚点 |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in FEATURE_SETS:
        if model not in allm.index: continue
        r=allm.loc[model]
        report.append(f"| {model} | {int(r['n'])} | {f(r['mae'])} | {f(r['rmse'])} | {f(r['pearson_gain_prediction'])} | {f(r['sign_accuracy'])} | {f(r['negative_gain_recall_discarded'])} | {f(r['agegate_negative_gain_recall_discarded'])} | {f(r['local_utility_model_mean'])} | {f(r['local_utility_agegate_mean'])} |")
    report.append("")
    report.append("## 4. 真正动作锚点的留一序列检查")
    report.append("")
    report.append("该子集只有 9 个锚点（每条序列 3 个），仅作为极小样本的方向性检查，不能作为统计结论。")
    report.append("")
    report.append("| 模型 | N | MAE ↓ | RMSE ↓ | 模型符号准确率 ↑ | 模型负收益召回率 ↑ | AgeGate负收益召回率 ↑ | 模型局部效用/锚点 ↑ | AgeGate局部效用/锚点 |")
    report.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in FEATURE_SETS:
        if model not in actm.index: continue
        r=actm.loc[model]
        report.append(f"| {model} | {int(r['n'])} | {f(r['mae'])} | {f(r['rmse'])} | {f(r['sign_accuracy'])} | {f(r['negative_gain_recall_discarded'])} | {f(r['agegate_negative_gain_recall_discarded'])} | {f(r['local_utility_model_mean'])} | {f(r['local_utility_agegate_mean'])} |")
    report.append("")
    report.append("## 5. 增量比较")
    report.append("")
    report.append("见 `r34_incremental_comparisons.csv`。MAE/RMSE 的 delta 为候选模型减 AgeOnly，负数代表连续收益预测改善；utility delta 为模型按预测收益正负选择 COMMIT/DISCARD 的局部反事实效用减显式 AgeGate（distance_kf >= 16）规则。")
    report.append("")
    report.append("这里的 utility 是对诊断锚点逐点套用规则的局部反事实模拟，不是重新运行的完整在线策略。非实际动作锚点原本没有触发真实候选动作，因此这项汇总不能证明完整闭环提升。")
    report.append("")
    report.append("## 6. 判读限制")
    report.append("")
    report.append("- 仅 3 条开发序列；LOSO 比随机拆分更严格，但不能替代独立外部测试集。")
    report.append("- 未来 H=16 窗口重叠，按锚点计算的指标存在相关性；不报告 p 值。")
    report.append("- 正负收益阈值固定为 0，未按测试折调阈值；L2 alpha 固定为 1.0。")
    report.append("- 9 个真实动作锚点不足以训练或确认通用门控；逐序列的 3 点指标只作描述。")
    report.append("- 即使某模型的离线 utility 高于 AgeGate，也只能说明该特征在当前局部反事实标签上值得进入下一步闭环测试，不能声称在线性能改善或省算。")
    report.append("")
    report.append("## 7. 决策门槛")
    report.append("")
    report.append("只有当至少一个 Age+几何模型在 LOSO 上同时满足以下条件，才建议进入 R3.5 小规模闭环验证：")
    report.append("1. 相对 AgeOnly 的连续收益预测 MAE 不恶化，且负收益召回率有明确增量；")
    report.append("2. 局部效用高于 AgeGate，不是仅在合并总体上改善、而某条序列严重退化；")
    report.append("3. 真实动作锚点的方向性结果没有出现明显反转。")
    report.append("")
    report.append("否则停止门控特征堆叠，保留 R3.3 的机制诊断结论。")
    report.append("")
    report.append("## 8. 结论")
    report.append("")
    report.append("本轮是离线特征增量筛查，不是闭环算法验证。最终结论应由上述 LOSO 数值决定：若几何特征未能稳定改善 AgeOnly 的预测误差、负收益识别和 AgeGate 局部效用，就不进入新规则开发。")
    (out_dir / "R3_4_候选几何特征增量验证报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    summary = {
        "experiment": "R3.4 candidate geometry feature incremental validation",
        "source_rows": int(len(data)),
        "positive_gain": int((data[TARGET] > 0).sum()),
        "negative_gain": int((data[TARGET] < 0).sum()),
        "actual_action_anchors": int((data["is_policy_action_anchor"].astype(int) == 1).sum()),
        "models": list(FEATURE_SETS),
        "ridge_alpha_fixed": ALPHA,
        "gain_sign_threshold_fixed": 0.0,
        "evaluation": "leave-one-sequence-out",
        "uses_gpu": False,
        "trains_neural_predictor": False,
        "claims_closed_loop_improvement": False
    }
    (out_dir / "r34_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="R3.3 *_review.zip or extracted run directory")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    summary = evaluate(load_diagnostics(Path(args.input)), Path(args.out_dir))
    print(json.dumps(summary, indent=2))
    print(f"R3.4 report: {Path(args.out_dir) / 'R3_4_候选几何特征增量验证报告.md'}")

if __name__ == "__main__":
    main()
