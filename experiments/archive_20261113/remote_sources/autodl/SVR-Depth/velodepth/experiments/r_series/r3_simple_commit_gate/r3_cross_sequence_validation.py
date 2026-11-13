#!/usr/bin/env python3
"""Leave-one-sequence-out validation for R3 residual thresholds.

Threshold selection uses only the two training sequences in each fold.
The held-out sequence is read only after a threshold has been selected.
"""
from __future__ import annotations

import argparse
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SEQUENCES = ("crowd2", "balloon2", "crowd3")
REQUIRED_POLICIES = ("AlwaysCommit", "NeverCommit", "AgeGate", "ResidualGate")
METRICS = ("mean_absrel", "mean_rmse", "mean_delta1")



def attach_action_patterns(sensitivity: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Derive action pattern and commit count from the per-anchor action CSV.

    The sweep summary intentionally contains trajectory metrics only; actions
    are recorded in residual_threshold_sweep_actions.csv.
    """
    required = {"sequence", "residual_threshold", *METRICS}
    missing = required - set(sensitivity.columns)
    if missing:
        raise ValueError(f"Sensitivity summary missing columns: {sorted(missing)}")
    action_required = {"sequence", "residual_threshold", "anchor_frame_zero_based", "selected_action"}
    action_missing = action_required - set(actions.columns)
    if action_missing:
        raise ValueError(f"Sensitivity actions missing columns: {sorted(action_missing)}")

    metrics = sensitivity.copy()
    metrics["sequence"] = metrics["sequence"].astype(str).str.lower()
    metrics["residual_threshold"] = pd.to_numeric(metrics["residual_threshold"], errors="raise").astype(float)

    decisions = actions.copy()
    decisions["sequence"] = decisions["sequence"].astype(str).str.lower()
    decisions["residual_threshold"] = pd.to_numeric(decisions["residual_threshold"], errors="raise").astype(float)
    decisions["anchor_frame_zero_based"] = pd.to_numeric(decisions["anchor_frame_zero_based"], errors="raise").astype(int)
    if "strategy" in decisions.columns:
        decisions = decisions[decisions["strategy"].astype(str) == "ResidualGate"]

    expected_anchors = [9, 37, 65]
    pattern_rows = []
    for (sequence, threshold), group in decisions.groupby(["sequence", "residual_threshold"], sort=True):
        group = group.sort_values("anchor_frame_zero_based")
        anchors = group["anchor_frame_zero_based"].tolist()
        if anchors != expected_anchors:
            raise ValueError(
                f"Expected anchors {expected_anchors} for {sequence}/threshold={threshold}, got {anchors}"
            )
        symbols = []
        for action in group["selected_action"].astype(str):
            action = action.strip().upper()
            if action == "COMMIT":
                symbols.append("C")
            elif action == "DISCARD":
                symbols.append("D")
            else:
                raise ValueError(f"Unknown selected_action {action!r} for {sequence}/threshold={threshold}")
        pattern = "".join(symbols)
        pattern_rows.append({
            "sequence": sequence,
            "residual_threshold": float(threshold),
            "pattern": pattern,
            "commits": pattern.count("C"),
        })
    pattern_df = pd.DataFrame(pattern_rows)
    if pattern_df.empty:
        raise ValueError("No valid per-anchor action rows found in residual_threshold_sweep_actions.csv")
    merged = metrics.merge(pattern_df, on=["sequence", "residual_threshold"], how="left", validate="one_to_one")
    if merged[["pattern", "commits"]].isna().any().any():
        missing_pairs = merged.loc[merged["pattern"].isna(), ["sequence", "residual_threshold"]].to_dict("records")
        raise ValueError(f"Missing action patterns for sensitivity rows: {missing_pairs}")
    merged["commits"] = merged["commits"].astype(int)
    return merged

def _metric_row(df: pd.DataFrame, sequence: str, threshold: float):
    rows = df[(df["sequence"].astype(str).str.lower() == sequence) &
              (pd.to_numeric(df["residual_threshold"], errors="coerce") - threshold).abs().lt(1e-9)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one sensitivity row for {sequence}, threshold={threshold}; got {len(rows)}")
    return rows.iloc[0]


def _baseline_row(baselines: dict, sequence: str, policy: str):
    rows = baselines[sequence]
    rows = rows[rows["strategy"].astype(str) == policy]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one baseline row for {sequence}/{policy}; got {len(rows)}")
    return rows.iloc[0]


def select_threshold_for_fold(sensitivity: pd.DataFrame, baselines: dict,
                              heldout_sequence: str, thresholds: list[float]):
    """Select threshold using training sequences only, with predeclared score.

    Primary: equal-weight mean of per-sequence AbsRel ratios relative to that
    sequence's NeverCommit baseline. Secondary: equal-weight mean RMSE ratio.
    Tertiary: maximize mean absolute delta1 gain relative to NeverCommit.
    Final deterministic tie-break: smaller threshold (more conservative).
    """
    train_sequences = [s for s in SEQUENCES if s != heldout_sequence]
    score_rows = []
    for tau in thresholds:
        absrel_ratios, rmse_ratios, delta1_gains = [], [], []
        for seq in train_sequences:
            r = _metric_row(sensitivity, seq, tau)
            never = _baseline_row(baselines, seq, "NeverCommit")
            never_absrel = float(never["mean_absrel"])
            never_rmse = float(never["mean_rmse"])
            if not (math.isfinite(never_absrel) and never_absrel > 0 and
                    math.isfinite(never_rmse) and never_rmse > 0):
                raise ValueError(f"Invalid NeverCommit normalization baseline for {seq}")
            absrel_ratios.append(float(r["mean_absrel"]) / never_absrel)
            rmse_ratios.append(float(r["mean_rmse"]) / never_rmse)
            delta1_gains.append(float(r["mean_delta1"]) - float(never["mean_delta1"]))
        score_rows.append({
            "heldout_sequence": heldout_sequence,
            "training_sequences": ";".join(train_sequences),
            "residual_threshold": float(tau),
            "train_mean_absrel_ratio_vs_never": sum(absrel_ratios) / len(absrel_ratios),
            "train_mean_rmse_ratio_vs_never": sum(rmse_ratios) / len(rmse_ratios),
            "train_mean_delta1_gain_vs_never": sum(delta1_gains) / len(delta1_gains),
            "selection_uses_heldout_metrics": False,
        })
    # Lexicographic predeclared objective; held-out data is never referenced here.
    ranked = sorted(
        score_rows,
        key=lambda x: (
            x["train_mean_absrel_ratio_vs_never"],
            x["train_mean_rmse_ratio_vs_never"],
            -x["train_mean_delta1_gain_vs_never"],
            x["residual_threshold"],
        ),
    )
    chosen = ranked[0]["residual_threshold"]
    for row in score_rows:
        row["selected_for_fold"] = abs(row["residual_threshold"] - chosen) < 1e-9
    return chosen, score_rows


def load_baselines(baseline_root: Path):
    baselines = {}
    for seq in SEQUENCES:
        path = baseline_root / seq / "trajectory_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline summary: {path}")
        df = pd.read_csv(path)
        missing = set(("strategy", *METRICS)) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        absent = set(REQUIRED_POLICIES) - set(df["strategy"].astype(str))
        if absent:
            raise ValueError(f"{path} missing required policies: {sorted(absent)}")
        baselines[seq] = df
    return baselines


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sensitivity-root", type=Path, required=True)
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    sensitivity_root = args.sensitivity_root.resolve()
    baseline_root = args.baseline_root.resolve()
    out_dir = args.out_dir.resolve()
    summary_path = sensitivity_root / "residual_threshold_sweep_summary.csv"
    actions_path = sensitivity_root / "residual_threshold_sweep_actions.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing sensitivity summary: {summary_path}")
    if not actions_path.is_file():
        raise FileNotFoundError(f"Missing sensitivity action table: {actions_path}")
    sensitivity = attach_action_patterns(pd.read_csv(summary_path), pd.read_csv(actions_path))
    sensitivity["sequence"] = sensitivity["sequence"].astype(str).str.lower()
    thresholds = sorted(float(x) for x in sensitivity["residual_threshold"].unique())
    if len(thresholds) < 2:
        raise ValueError(f"Need at least two predeclared thresholds, got {thresholds}")
    for seq in SEQUENCES:
        got = set(round(float(x), 9) for x in sensitivity.loc[sensitivity["sequence"] == seq, "residual_threshold"])
        expected = set(round(x, 9) for x in thresholds)
        if got != expected:
            raise ValueError(f"Threshold grid mismatch for {seq}: got {sorted(got)}, expected {sorted(expected)}")
    baselines = load_baselines(baseline_root)

    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    selection_rows, heldout_rows, full_rows, fold_summary = [], [], [], []
    for seq in SEQUENCES:
        # Full same-protocol comparison from the baseline run.
        for _, row in baselines[seq].iterrows():
            rec = {"sequence": seq, **row.to_dict(), "comparison_source": "same_protocol_full_baseline"}
            full_rows.append(rec)

        selected_tau, scores = select_threshold_for_fold(sensitivity, baselines, seq, thresholds)
        selection_rows.extend(scores)

        # Held-out residual metrics are looked up only after threshold selection.
        residual = _metric_row(sensitivity, seq, selected_tau)
        candidates = []
        for policy in ("AlwaysCommit", "NeverCommit", "AgeGate"):
            row = _baseline_row(baselines, seq, policy)
            candidates.append({
                "heldout_sequence": seq,
                "policy": policy,
                "residual_threshold": "",
                "mean_absrel": float(row["mean_absrel"]),
                "mean_rmse": float(row["mean_rmse"]),
                "mean_delta1": float(row["mean_delta1"]),
                "geom_commits_optional": int(row.get("geom_commits_optional", -1)),
                "comparison_source": "full_baseline_run",
            })
        candidates.append({
            "heldout_sequence": seq,
            "policy": "ResidualGate_LOSO",
            "residual_threshold": selected_tau,
            "mean_absrel": float(residual["mean_absrel"]),
            "mean_rmse": float(residual["mean_rmse"]),
            "mean_delta1": float(residual["mean_delta1"]),
            "geom_commits_optional": int(residual["commits"]),
            "comparison_source": "predeclared_threshold_sweep_selected_on_other_two_sequences",
        })
        never_absrel = next(x["mean_absrel"] for x in candidates if x["policy"] == "NeverCommit")
        never_rmse = next(x["mean_rmse"] for x in candidates if x["policy"] == "NeverCommit")
        never_delta1 = next(x["mean_delta1"] for x in candidates if x["policy"] == "NeverCommit")
        for x in candidates:
            x["absrel_ratio_vs_never"] = x["mean_absrel"] / never_absrel
            x["rmse_ratio_vs_never"] = x["mean_rmse"] / never_rmse
            x["delta1_gain_vs_never"] = x["mean_delta1"] - never_delta1
            heldout_rows.append(x)

        selected_row = next(x for x in candidates if x["policy"] == "ResidualGate_LOSO")
        age_row = next(x for x in candidates if x["policy"] == "AgeGate")
        always_row = next(x for x in candidates if x["policy"] == "AlwaysCommit")
        never_row = next(x for x in candidates if x["policy"] == "NeverCommit")
        fold_summary.append({
            "heldout_sequence": seq,
            "training_sequences": ";".join(s for s in SEQUENCES if s != seq),
            "selected_residual_threshold": selected_tau,
            "selected_pattern": str(residual["pattern"]),
            "selected_commit_count": int(residual["commits"]),
            "residual_absrel": selected_row["mean_absrel"],
            "age_absrel": age_row["mean_absrel"],
            "always_absrel": always_row["mean_absrel"],
            "never_absrel": never_row["mean_absrel"],
            "residual_absrel_better_than_age": selected_row["mean_absrel"] < age_row["mean_absrel"],
            "residual_absrel_better_than_always": selected_row["mean_absrel"] < always_row["mean_absrel"],
            "residual_absrel_better_than_never": selected_row["mean_absrel"] < never_row["mean_absrel"],
            "heldout_metrics_used_for_selection": False,
        })

    pd.DataFrame(selection_rows).to_csv(out_dir / "loco_threshold_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(heldout_rows).to_csv(out_dir / "loco_heldout_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(full_rows).to_csv(out_dir / "full_gate_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_summary).to_csv(out_dir / "loco_fold_summary.csv", index=False, encoding="utf-8-sig")

    mean_rows = pd.DataFrame(heldout_rows)
    residual_rows = mean_rows[mean_rows["policy"] == "ResidualGate_LOSO"]
    never_rows = mean_rows[mean_rows["policy"] == "NeverCommit"]
    age_rows = mean_rows[mean_rows["policy"] == "AgeGate"]
    always_rows = mean_rows[mean_rows["policy"] == "AlwaysCommit"]
    meta = {
        "status": "completed",
        "experiment": "R3 full gate comparison and leave-one-sequence-out threshold validation",
        "sequences": list(SEQUENCES),
        "threshold_grid": thresholds,
        "baseline_policies": list(REQUIRED_POLICIES),
        "selection_rule": {
            "primary": "equal-weight mean of per-training-sequence AbsRel ratio to that sequence's NeverCommit baseline; lower is better",
            "secondary": "equal-weight mean of per-training-sequence RMSE ratio to NeverCommit; lower is better",
            "tertiary": "maximize mean delta1 gain to NeverCommit",
            "final_tie_break": "smaller threshold",
            "heldout_sequence_metrics_used_for_selection": False,
        },
        "sensitivity_root": str(sensitivity_root),
        "baseline_root": str(baseline_root),
        "folds": fold_summary,
        "loco_residual_absrel_better_than_age_count": int(sum(f["residual_absrel_better_than_age"] for f in fold_summary)),
        "loco_residual_absrel_better_than_always_count": int(sum(f["residual_absrel_better_than_always"] for f in fold_summary)),
        "loco_residual_absrel_better_than_never_count": int(sum(f["residual_absrel_better_than_never"] for f in fold_summary)),
        "mean_loco_residual_absrel_ratio_vs_never": float((residual_rows["absrel_ratio_vs_never"]).mean()),
        "mean_age_absrel_ratio_vs_never": float((age_rows["absrel_ratio_vs_never"]).mean()),
        "mean_always_absrel_ratio_vs_never": float((always_rows["absrel_ratio_vs_never"]).mean()),
        "mean_loco_residual_rmse_ratio_vs_never": float((residual_rows["rmse_ratio_vs_never"]).mean()),
        "warning": "Only three development sequences are available. LOSO is exploratory cross-validation, not independent external test-set evidence. Post-refresh commit gating still computes every refresh candidate and therefore does not demonstrate refresh-compute savings.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "crossval_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")

    lines = [
        "# R3 完整门控对照与留一序列验证报告",
        "",
        f"- 状态：`{meta['status']}`",
        f"- 残差阈值网格：`{', '.join(f'{x:.2f}' for x in thresholds)}`",
        "- 完整基线：AlwaysCommit、NeverCommit、AgeGate、ResidualGate@0.10",
        "- 选择方式：每折只用另外两条序列的指标选阈值，留出序列不参与选择。",
        "",
        "## 1. 留一序列选择结果",
        "",
        "| 留出序列 | 训练序列 | 选择阈值 | 动作模式 | 提交数 | AbsRel | RMSE | δ₁ |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for f in fold_summary:
        chosen = next(x for x in heldout_rows if x["heldout_sequence"] == f["heldout_sequence"] and x["policy"] == "ResidualGate_LOSO")
        lines.append(f"| {f['heldout_sequence']} | {f['training_sequences']} | {f['selected_residual_threshold']:.2f} | `{f['selected_pattern']}` | {f['selected_commit_count']} | {chosen['mean_absrel']:.6f} | {chosen['mean_rmse']:.6f} | {chosen['mean_delta1']:.6f} |")
    lines += [
        "",
        "## 2. 留出序列主指标对照",
        "",
        "| 留出序列 | Always AbsRel | Never AbsRel | Age AbsRel | LOSO Residual AbsRel | Residual 优于 Age? |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for f in fold_summary:
        lines.append(f"| {f['heldout_sequence']} | {f['always_absrel']:.6f} | {f['never_absrel']:.6f} | {f['age_absrel']:.6f} | {f['residual_absrel']:.6f} | {'是' if f['residual_absrel_better_than_age'] else '否'} |")
    lines += [
        "",
        "## 3. 汇总",
        "",
        f"- LOSO ResidualGate 的平均 AbsRel / NeverCommit 比值：{meta['mean_loco_residual_absrel_ratio_vs_never']:.4f}。",
        f"- AgeGate 的平均 AbsRel / NeverCommit 比值：{meta['mean_age_absrel_ratio_vs_never']:.4f}。",
        f"- AlwaysCommit 的平均 AbsRel / NeverCommit 比值：{meta['mean_always_absrel_ratio_vs_never']:.4f}。",
        f"- LOSO ResidualGate 在 {meta['loco_residual_absrel_better_than_age_count']}/3 条留出序列上优于 AgeGate 的 AbsRel。",
        f"- LOSO ResidualGate 在 {meta['loco_residual_absrel_better_than_always_count']}/3 条留出序列上优于 AlwaysCommit 的 AbsRel。",
        f"- LOSO ResidualGate 在 {meta['loco_residual_absrel_better_than_never_count']}/3 条留出序列上优于 NeverCommit 的 AbsRel。",
        "",
        "## 4. 解释边界",
        "",
        "本报告的阈值选择只读取训练折的两条序列；留出序列指标只在阈值锁定后用于评估。主指标是按训练序列等权平均的 AbsRel 比值（相对于各自 NeverCommit），以减少序列绝对误差尺度不同造成的偏置；RMSE 比值和 delta1 增益仅作预先规定的次级判据。",
        "",
        "只有三条开发序列，因此这是一项探索性 LOSO 交叉验证，不能替代独立外部测试。所有策略仍在三个锚点计算刷新候选，Post-Refresh Commit Gate 不等于省掉刷新推理。",
        "",
        "## 5. 文件说明",
        "",
        "- `full_gate_comparison.csv`：每条序列的四种同协议基线结果。",
        "- `loco_threshold_selection.csv`：每折所有候选阈值的训练折评分与选择标记。",
        "- `loco_heldout_comparison.csv`：锁定阈值后，留出序列与 Always/Never/Age 的指标比较。",
        "- `loco_fold_summary.csv`：每折选择结果和主指标胜负。",
        "- `crossval_metadata.json`：协议、数据源和选择规则审计。",
    ]
    (out_dir / "R3_完整门控对照与LOSO验证报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Create a review ZIP next to the cross-validation root, containing all baseline
    # and validation artifacts plus the exact runner/source used.
    package_root = baseline_root.parent
    package_path = package_root.parent / f"{package_root.name}_review.zip"
    if package_path.exists():
        raise FileExistsError(f"Review ZIP already exists: {package_path}")
    script_dir = Path(__file__).resolve().parent
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(package_root.rglob("*")):
            if p.is_file():
                z.write(p, f"{package_root.name}/{p.relative_to(package_root).as_posix()}")
        for name in ("run_r3.sh", "r3_simple_commit_gate.py", "r3_gate_rules.py",
                     "r3_cross_sequence_validation.py", "test_r3_cross_sequence_validation.py",
                     "test_r3_gate_rules.py", "R3_CROSSVAL_PROTOCOL.md", "README.md"):
            p = script_dir / name
            if p.is_file():
                z.write(p, f"runner/{name}")
        if z.testzip() is not None:
            raise RuntimeError(f"Review ZIP integrity check failed: {package_path}")
    print(f"LOSO validation complete: {out_dir}")
    print(f"Review ZIP: {package_path}")


if __name__ == "__main__":
    main()
