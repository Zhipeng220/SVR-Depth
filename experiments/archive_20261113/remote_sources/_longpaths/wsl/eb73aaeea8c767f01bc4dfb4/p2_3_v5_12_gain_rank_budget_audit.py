#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.3 v5.12: leakage-safe same-state gain ranking and constrained budget audit.

CPU-only diagnostic. It does not run VeloDepth inference and does not update VeloDepth weights.
The target is the P1.5-compatible B1 state-writeback gain:
G_state_absrel_discounted_per_valid_frame.
This is NOT a full-keyframe refresh gain.

The experiment:
1) Leave one complete sequence out when fitting a Ridge predictor.
2) Measure per-trajectory rank quality on the held-out sequence.
3) Simulate an offline, constrained K=8 selection using the held-out scores,
   a state-age heuristic, a fixed schedule, and an oracle upper bound.
4) Enforce candidate-anchor spacing >= 8 frames and per-bin quotas [3,3,2]
   over label-covered anchors (0:29, 29:57, 57:86).
The constrained top-K selection sees all candidate scores from the clip, so it is
an OFFLINE ranking diagnostic, not a causal online controller. Oracle selection
uses labels and is an upper bound only. Because H=16 target windows overlap, sums
of anchor gains are not additive sequence-level gains.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEQUENCES = ("crowd2", "balloon2", "crowd3")
TARGET = "G_state_absrel_discounted_per_valid_frame"
STATUS_OK = "ok"
MOTION_FEATURES = [
    "feature_scheduler_distance_kf",
    "feature_scheduler_flow_magnitude",
    "feature_scheduler_total_luma_change",
]
STATE_FEATURES = [
    "feature_state_radius_teacher_mean",
    "feature_state_confidence_teacher_mean",
    "feature_state_features_teacher_0_mean",
    "feature_state_features_teacher_1_mean",
    "feature_state_features_teacher_2_mean",
    "feature_state_features_teacher_3_mean",
]
FEATURES = MOTION_FEATURES + STATE_FEATURES
SCORE_COLUMNS = {
    "loso_ridge_motion_state": "score_loso_ridge_motion_state",
    "state_age": "score_state_age",
    "flow_magnitude": "score_flow_magnitude",
    "luma_change": "score_luma_change",
}
# H=16 labels only exist for anchors through frame 69 in the 86-frame clips.
# Three bins preserve full label coverage while allowing an exact K=8 / min-gap=8 set.
BIN_EDGES = (0, 29, 57, 86)
BIN_QUOTAS = (3, 3, 2)
K = 8
MIN_GAP = 8
FIXED_TEMPLATE = (5, 13, 21, 33, 41, 49, 61, 69)


def safe_corr(a, b, method="spearman"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    try:
        if method == "pearson":
            return float(pearsonr(a, b).statistic)
        return float(spearmanr(a, b).statistic)
    except Exception:
        return float("nan")


def ranking_metrics(frame: pd.DataFrame, score_col: str, model: str):
    score = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(float)
    gain = pd.to_numeric(frame[TARGET], errors="coerce").to_numpy(float)
    ok = np.isfinite(score) & np.isfinite(gain)
    score, gain = score[ok], gain[ok]
    n = len(gain)
    if n == 0:
        return {"model": model, "n": 0, "pearson": np.nan, "spearman": np.nan,
                "top_quartile_gain_lift": np.nan, "positive_gain_rate": np.nan,
                "top_quartile_positive_rate": np.nan, "positive_rate_lift": np.nan}
    top_n = max(1, int(math.ceil(0.25 * n)))
    if np.std(score) <= 1e-12:
        top_gain_lift = top_positive = positive_lift = np.nan
    else:
        idx = np.argsort(-score, kind="mergesort")[:top_n]
        top_gain_lift = float(np.mean(gain[idx]) - np.mean(gain))
        top_positive = float(np.mean(gain[idx] > 0))
        positive_lift = top_positive - float(np.mean(gain > 0))
    return {
        "model": model, "n": int(n),
        "pearson": safe_corr(score, gain, "pearson"),
        "spearman": safe_corr(score, gain, "spearman"),
        "mean_gain_all": float(np.mean(gain)),
        "top_quartile_n": int(top_n),
        "top_quartile_gain_lift": top_gain_lift,
        "positive_gain_rate": float(np.mean(gain > 0)),
        "top_quartile_positive_rate": top_positive,
        "positive_rate_lift": positive_lift,
    }


def add_bin(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    anchors = pd.to_numeric(out["anchor_frame"], errors="coerce").to_numpy(float)
    out["temporal_bin"] = np.select(
        [
            (anchors >= BIN_EDGES[0]) & (anchors < BIN_EDGES[1]),
            (anchors >= BIN_EDGES[1]) & (anchors < BIN_EDGES[2]),
            (anchors >= BIN_EDGES[2]) & (anchors < BIN_EDGES[3]),
        ],
        [0, 1, 2],
        default=-1,
    ).astype(int)
    return out


def enumerate_feasible_subsets(frame: pd.DataFrame):
    """Return all exact K subsets with per-bin quotas and min-gap constraint."""
    frame = frame.sort_values("anchor_frame", kind="mergesort").copy()
    options = []
    for bin_id, quota in enumerate(BIN_QUOTAS):
        group = frame[frame["temporal_bin"] == bin_id]
        rows = [row for _, row in group.iterrows()]
        if len(rows) < quota:
            return []
        options.append(list(itertools.combinations(rows, quota)))
    feasible = []
    for parts in itertools.product(*options):
        rows = [r for part in parts for r in part]
        rows.sort(key=lambda r: int(r["anchor_frame"]))
        anchors = [int(r["anchor_frame"]) for r in rows]
        if len(anchors) != K:
            continue
        if any((anchors[i + 1] - anchors[i]) < MIN_GAP for i in range(len(anchors) - 1)):
            continue
        feasible.append(rows)
    return feasible


def select_subset(frame: pd.DataFrame, method: str, score_col: str | None):
    """Select a feasible subset. The fixed schedule never reads target labels."""
    feasible = enumerate_feasible_subsets(frame)
    if not feasible:
        return None, 0
    if method == "fixed_spaced":
        def key(rows):
            anchors = [int(r["anchor_frame"]) for r in rows]
            cost = sum((a - t) ** 2 for a, t in zip(anchors, FIXED_TEMPLATE))
            return (cost, tuple(anchors))
        return min(feasible, key=key), len(feasible)

    if method == "oracle_gain":
        score_col = TARGET
    if score_col is None:
        raise ValueError(f"Missing score column for method={method}")

    def key(rows):
        scores = np.asarray([float(r[score_col]) for r in rows], dtype=float)
        anchors = tuple(int(r["anchor_frame"]) for r in rows)
        score_sum = float(np.sum(scores))
        # Deterministic secondary tie-break uses only frame geometry, not the target,
        # except for oracle_gain where the target is itself the declared oracle score.
        fixed_cost = sum((a - t) ** 2 for a, t in zip(anchors, FIXED_TEMPLATE))
        return (-score_sum, fixed_cost, anchors)
    return min(feasible, key=key), len(feasible)


def load_labels(csv_path: Path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = {
        "sequence", "trajectory_policy", "anchor_frame", "status", "horizon_H",
        TARGET, *FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Label CSV missing required columns: {missing}")
    df["sequence"] = df["sequence"].astype(str)
    if not set(df["sequence"].unique()).issubset(set(SEQUENCES)):
        raise ValueError(f"Unexpected sequences: {sorted(set(df.sequence)-set(SEQUENCES))}")
    if not (pd.to_numeric(df["horizon_H"], errors="coerce") == 16).all():
        raise ValueError("Expected H=16 labels only.")
    keys = ["sequence", "trajectory_policy", "anchor_frame"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate sequence/trajectory_policy/anchor_frame rows.")
    raw_n = len(df)
    df["anchor_frame"] = pd.to_numeric(df["anchor_frame"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = df[(df["status"].astype(str) == STATUS_OK) & np.isfinite(df[TARGET])].copy()
    valid = add_bin(valid)
    if (valid["temporal_bin"] < 0).any():
        raise ValueError("Some anchor frames fall outside the declared temporal bins.")
    valid["score_state_age"] = valid["feature_scheduler_distance_kf"]
    valid["score_flow_magnitude"] = valid["feature_scheduler_flow_magnitude"]
    valid["score_luma_change"] = valid["feature_scheduler_total_luma_change"]
    return valid, raw_n, len(valid)


def fit_loso_predict(valid: pd.DataFrame, alpha: float):
    outputs = []
    folds = []
    for heldout in SEQUENCES:
        train = valid[valid.sequence != heldout].copy()
        test = valid[valid.sequence == heldout].copy()
        if train.empty or test.empty:
            raise ValueError(f"Empty LOSO fold for heldout={heldout}")
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])
        model.fit(train[FEATURES], train[TARGET])
        pred = model.predict(test[FEATURES])
        test["score_loso_ridge_motion_state"] = pred
        test["loso_fold"] = f"leave_out_{heldout}"
        outputs.append(test)
        folds.append({
            "heldout_sequence": heldout,
            "training_sequences": ",".join(s for s in SEQUENCES if s != heldout),
            "train_rows": int(len(train)), "test_rows": int(len(test)),
            "feature_count": len(FEATURES), "features": ",".join(FEATURES),
            "ridge_alpha": float(alpha),
            "preprocessing_fit_on_training_only": True,
            "heldout_sequence_excluded_from_fit": heldout not in set(train.sequence),
        })
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(folds)


def selection_audit(predictions: pd.DataFrame):
    selection_rows = []
    group_rows = []
    feasibility_rows = []
    methods = [
        ("fixed_spaced", None),
        ("state_age_topk", "score_state_age"),
        ("flow_topk", "score_flow_magnitude"),
        ("luma_topk", "score_luma_change"),
        ("loso_ridge_topk", "score_loso_ridge_motion_state"),
        ("oracle_gain", TARGET),
    ]
    for (seq, trajectory), group in predictions.groupby(["sequence", "trajectory_policy"], sort=True):
        group = add_bin(group)
        group = group.sort_values("anchor_frame", kind="mergesort").copy()
        oracle_rows, oracle_count = select_subset(group, "oracle_gain", TARGET)
        oracle_sum = float(sum(float(r[TARGET]) for r in oracle_rows)) if oracle_rows else np.nan
        for method, score_col in methods:
            chosen, feasible_count = select_subset(group, method, score_col)
            feasible = chosen is not None
            feasibility_rows.append({
                "sequence": seq, "trajectory_policy": trajectory, "method": method,
                "candidate_rows": int(len(group)), "feasible_subsets_count": int(feasible_count),
                "budget_K": K, "min_gap_frames": MIN_GAP,
                "bin_quotas": ",".join(map(str, BIN_QUOTAS)),
                "exact_budget_feasible": bool(feasible),
            })
            if not feasible:
                group_rows.append({
                    "sequence": seq, "trajectory_policy": trajectory, "method": method,
                    "selected_count": 0, "selected_frames": "", "selected_gain_sum": np.nan,
                    "selected_gain_mean": np.nan, "selected_positive_gain_rate": np.nan,
                    "oracle_gain_sum": oracle_sum, "oracle_regret_gain_sum": np.nan,
                    "bin0_count": 0, "bin1_count": 0, "bin2_count": 0,
                    "exact_budget_feasible": False,
                })
                continue
            chosen = sorted(chosen, key=lambda r: int(r["anchor_frame"]))
            anchors = [int(r["anchor_frame"]) for r in chosen]
            gains = [float(r[TARGET]) for r in chosen]
            bin_counts = [sum(int(r["temporal_bin"]) == b for r in chosen) for b in range(3)]
            chosen_score_col = score_col if score_col is not None else None
            for rank, row in enumerate(chosen, 1):
                selection_rows.append({
                    "sequence": seq, "trajectory_policy": trajectory, "method": method,
                    "selection_rank_by_frame": rank, "anchor_frame": int(row["anchor_frame"]),
                    "temporal_bin": int(row["temporal_bin"]),
                    "score_used_for_selection": float(row[chosen_score_col]) if chosen_score_col else np.nan,
                    "same_state_gain_label": float(row[TARGET]),
                    "gain_positive": bool(float(row[TARGET]) > 0),
                    "oracle_gain_sum_for_group": oracle_sum,
                    "feasible_subsets_count": int(feasible_count),
                })
            selected_sum = float(np.sum(gains))
            group_rows.append({
                "sequence": seq, "trajectory_policy": trajectory, "method": method,
                "selected_count": len(chosen), "selected_frames": ",".join(map(str, anchors)),
                "selected_gain_sum": selected_sum,
                "selected_gain_mean": float(np.mean(gains)),
                "selected_positive_gain_rate": float(np.mean(np.asarray(gains) > 0)),
                "oracle_gain_sum": oracle_sum,
                "oracle_regret_gain_sum": float(oracle_sum - selected_sum) if np.isfinite(oracle_sum) else np.nan,
                "bin0_count": bin_counts[0], "bin1_count": bin_counts[1], "bin2_count": bin_counts[2],
                "exact_budget_feasible": True,
            })
    selected = pd.DataFrame(selection_rows)
    group_summary = pd.DataFrame(group_rows)
    feasibility = pd.DataFrame(feasibility_rows)
    if not selected.empty:
        stratum = selected.groupby(["sequence", "trajectory_policy", "method", "temporal_bin"], as_index=False).agg(
            selected_count=("anchor_frame", "size"),
            selected_gain_mean=("same_state_gain_label", "mean"),
            selected_gain_sum=("same_state_gain_label", "sum"),
            selected_positive_gain_rate=("gain_positive", "mean"),
        )
    else:
        stratum = pd.DataFrame()
    return selected, group_summary, feasibility, stratum


def build_macro(selection_groups: pd.DataFrame):
    # Average trajectories within sequence first; then macro-average sequences equally.
    numeric = [
        "selected_gain_sum", "selected_gain_mean", "selected_positive_gain_rate",
        "oracle_gain_sum", "oracle_regret_gain_sum",
    ]
    per_sequence = selection_groups[selection_groups.exact_budget_feasible].groupby(
        ["sequence", "method"], as_index=False
    )[numeric].mean()
    macro = per_sequence.groupby("method", as_index=False)[numeric].mean()
    sequence_counts = per_sequence.groupby("method")["sequence"].nunique().rename("sequences_in_macro")
    macro = macro.merge(sequence_counts, on="method", how="left")
    return per_sequence, macro


def build_report(out_dir: Path, valid: pd.DataFrame, raw_n: int, valid_n: int,
                 rank: pd.DataFrame, group_summary: pd.DataFrame, macro: pd.DataFrame,
                 folds: pd.DataFrame, feasibility: pd.DataFrame):
    lines = [
        "# P2.3 v5.12 同状态收益排序与约束预算审计",
        "",
        "## 1. 实验边界",
        "",
        "- CPU-only 离线分析；没有运行 VeloDepth 推理，也没有更新 VeloDepth 权重。",
        "- 目标为 P1.5 兼容的 `G_state_absrel_discounted_per_valid_frame`，正值表示 B1 持久几何状态写回相对 B0 的未来 H=16 折扣 AbsRel 改善；不是完整 keyframe 刷新收益。",
        "- 使用按序列留一验证（LOSO），每折的插补器、标准化器与 Ridge 均只在另外两条训练序列上拟合。",
        "- K=8、最小帧间隔 8、三个时间段 `[0,29)`, `[29,57)`, `[57,86)`，配额为 3/3/2。使用三个时间段是因为现有 H=16 标签只覆盖到锚点 69；不能把没有标签的 73/77/81/85 帧用于收益排序。",
        "- Top-K 选择使用整段候选的分数，是离线排序诊断，不是因果在线策略。Oracle 使用真实标签，仅是同一约束下的上界。",
        "- 不同锚点的 H=16 未来窗口重叠，所选标签之和不等于真实在线累计收益；这里只比较选中锚点的标签质量。",
        "",
        "## 2. 输入与切分审计",
        "",
        f"- 原始行数：{raw_n}；status=ok 且收益标签有效的行数：{valid_n}。",
        f"- 有效序列：{', '.join(sorted(valid.sequence.unique()))}。",
        "- 标签行按 sequence / trajectory_policy / anchor_frame 唯一；所有模型的测试序列完全排除在拟合之外。",
        "",
        "| 留出序列 | 训练序列 | 训练行 | 测试行 |",
        "|---|---|---:|---:|",
    ]
    for r in folds.itertuples():
        lines.append(f"| {r.heldout_sequence} | {r.training_sequences} | {r.train_rows} | {r.test_rows} |")
    lines += [
        "",
        "## 3. 候选级排序指标",
        "",
        "按留出序列与轨迹分组，Spearman 越高越好；Top-quartile gain lift 为预测分数最高四分之一候选的平均标签减去全候选平均标签，越大越好。",
        "",
        "| 序列 | 轨迹 | 模型/启发式 | n | Spearman | Top-25% gain lift | 正收益率 lift |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rank.itertuples():
        lines.append(
            f"| {r.sequence} | {r.trajectory_policy} | {r.model} | {r.n} | "
            f"{r.spearman:.4f} | {r.top_quartile_gain_lift:.6f} | {r.positive_rate_lift:.4f} |"
        )
    lines += [
        "",
        "## 4. 约束预算 K=8 的离线选择",
        "",
        "每个 sequence × trajectory_policy 组内，所有方法使用同一批有效标签候选、同一 K、同一最小间隔和相同的 3/3/2 时间段配额。",
        "",
        "| 方法 | 覆盖序列数 | 平均选中锚点标签 | 平均正收益率 | 平均 oracle regret（标签和差） |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in macro.itertuples():
        lines.append(
            f"| {r.method} | {r.sequences_in_macro} | {r.selected_gain_mean:.6f} | "
            f"{r.selected_positive_gain_rate:.4f} | {r.oracle_regret_gain_sum:.6f} |"
        )
    lines += [
        "",
        "## 5. 判读限制",
        "",
        "- `oracle_gain` 利用真实标签挑选动作，只提供理想上界，不是可部署方法。",
        "- `fixed_spaced` 只按预先定义的帧位置模板选择，不读取收益标签。",
        "- `state_age_topk`、`flow_topk`、`luma_topk` 与 `loso_ridge_topk` 均为全片候选分数排序，不使用标签做动作选择，但使用了未来候选的特征分数，因此不能视为因果在线策略。",
        "- 由于只有三条序列且标签窗口重叠，结果属于研究决策用的探索性证据，不是强统计显著性或广泛泛化声明。",
        "- 如果 Ridge 在候选级相关性或受约束选择上没有稳定超过 state-age 与 fixed-spaced，不建议继续扩大模型训练；应先检查标签、特征覆盖与动作语义。",
        "",
        "## 6. 下一步决策",
        "",
        "只有当 LOSO Ridge 在多个留出序列/轨迹上表现出正向排序能力，并在相同预算与间隔约束下稳定优于固定位置与状态年龄基线，才进入下一阶段的因果在线影子策略验证。否则保留状态年龄作为低开销基线，不训练 VeloDepth 主模型。",
    ]
    (out_dir / "P2_3_v5_12_同状态收益排序与约束预算审计.md").write_text("\n".join(lines), encoding="utf-8")


def run_self_test():
    toy = pd.DataFrame({
        "anchor_frame": [1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69],
        "same_state_gain": np.arange(18, dtype=float),
    })
    toy[TARGET] = toy["same_state_gain"]
    toy = add_bin(toy)
    chosen, count = select_subset(toy, "oracle_gain", TARGET)
    assert chosen is not None, "oracle subset should be feasible"
    anchors = [int(r["anchor_frame"]) for r in chosen]
    assert len(anchors) == K
    assert [sum(int(r["temporal_bin"]) == b for r in chosen) for b in range(3)] == list(BIN_QUOTAS)
    assert all(anchors[i+1]-anchors[i] >= MIN_GAP for i in range(K-1))
    chosen_fixed, _ = select_subset(toy, "fixed_spaced", None)
    assert chosen_fixed is not None
    fixed_anchors = [int(r["anchor_frame"]) for r in chosen_fixed]
    assert [sum(BIN_EDGES[b] <= a < BIN_EDGES[b+1] for a in fixed_anchors) for b in range(3)] == list(BIN_QUOTAS)
    assert all(fixed_anchors[i+1]-fixed_anchors[i] >= MIN_GAP for i in range(K-1))
    # Simulate a missing anchor (as occurs in one sequence around a restart) and ensure
    # the selector remains feasible without ever consulting target labels for fixed selection.
    missing = toy[toy.anchor_frame != 61].copy()
    chosen_missing, _ = select_subset(missing, "fixed_spaced", None)
    assert chosen_missing is not None
    print("PASS: P2.3 v5.12 selection self-tests (K=8, 3/3/2 bins, min-gap=8, missing-anchor case).")


def main():
    parser = argparse.ArgumentParser(description="P2.3 v5.12 offline same-state gain ranking and constrained-budget audit")
    default_csv = Path(__file__).resolve().parent / "data_p2_3_v5_12" / "P2_3_v5_5_完整序列同状态收益标签.csv"
    parser.add_argument("--labels-csv", default=str(default_csv))
    parser.add_argument("--out-dir", default="outputs/p2_3_v5_12_gain_rank_budget_audit")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return

    labels_path = Path(args.labels_csv)
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Missing same-state label CSV: {labels_path}\n"
            "Upload the package with data_p2_3_v5_12/P2_3_v5_5_完整序列同状态收益标签.csv, "
            "or pass --labels-csv to the existing full-run v5.5 label file."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    valid, raw_n, valid_n = load_labels(labels_path)
    predictions, folds = fit_loso_predict(valid, args.ridge_alpha)

    ranking_rows = []
    for (seq, trajectory), group in predictions.groupby(["sequence", "trajectory_policy"], sort=True):
        for model, score_col in SCORE_COLUMNS.items():
            ranking_rows.append({
                "sequence": seq, "trajectory_policy": trajectory,
                **ranking_metrics(group, score_col, model),
            })
    ranking = pd.DataFrame(ranking_rows)

    selected, selection_groups, feasibility, stratum = selection_audit(predictions)
    per_seq, macro = build_macro(selection_groups)
    predictions.to_csv(out_dir / "p2_3_v5_12_loso_predictions.csv", index=False)
    folds.to_csv(out_dir / "p2_3_v5_12_loso_fold_audit.csv", index=False)
    ranking.to_csv(out_dir / "p2_3_v5_12_candidate_ranking_metrics.csv", index=False)
    selected.to_csv(out_dir / "p2_3_v5_12_selected_anchor_rows.csv", index=False)
    selection_groups.to_csv(out_dir / "p2_3_v5_12_budget_selection_by_trajectory.csv", index=False)
    feasibility.to_csv(out_dir / "p2_3_v5_12_budget_feasibility_audit.csv", index=False)
    stratum.to_csv(out_dir / "p2_3_v5_12_selected_gain_by_stratum.csv", index=False)
    per_seq.to_csv(out_dir / "p2_3_v5_12_budget_selection_per_sequence.csv", index=False)
    macro.to_csv(out_dir / "p2_3_v5_12_budget_selection_macro.csv", index=False)

    metadata = {
        "experiment": "P2.3 v5.12",
        "experiment_type": "CPU-only offline same-state gain ranking and constrained budget selection",
        "source_label_csv": str(labels_path),
        "source_label_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "raw_rows": int(raw_n), "valid_rows": int(valid_n),
        "sequences": list(SEQUENCES), "target": TARGET,
        "target_semantics": "P1.5-compatible B1 persistent geometry state writeback gain vs B0 propagation; not full-keyframe refresh gain",
        "features": FEATURES, "ridge_alpha": float(args.ridge_alpha),
        "split": "leave-one-sequence-out; preprocessing and Ridge fitted only on the other two sequences",
        "budget_K": K, "minimum_gap_frames": MIN_GAP,
        "temporal_bin_edges": list(BIN_EDGES), "temporal_bin_quotas": list(BIN_QUOTAS),
        "selection_semantics": "offline whole-clip score ranking; not causal online; oracle uses target labels as an upper bound only",
        "overlap_warning": "H=16 target windows overlap; selected gain sums are not additive sequence-level gains",
        "velodepth_weights_updated": False, "gpu_inference_run": False,
    }
    (out_dir / "p2_3_v5_12_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    build_report(out_dir, valid, raw_n, valid_n, ranking, selection_groups, macro, folds, feasibility)
    print(f"Input rows: {raw_n}; valid rows: {valid_n}")
    print("LOSO folds:")
    print(folds[["heldout_sequence", "training_sequences", "train_rows", "test_rows"]].to_string(index=False))
    print("\nRanking macro by model (average over sequence × trajectory rows):")
    print(ranking.groupby("model")[["spearman", "top_quartile_gain_lift", "positive_rate_lift"]].mean().to_string())
    print("\nBudget selection macro:")
    print(macro.to_string(index=False))
    print(f"\nWrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
