#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P2.3 v5.23 — CPU-only multi-budget decision-headroom audit.

Uses the locked v5.22 LOSO predictions and its synchronous diagnostic predictions.
It does NOT run VeloDepth inference, train a predictor, tune hyperparameters, or
modify model weights. It extends the v5.22 Top-8-only comparison to K=3,5,8,10
and adds a deterministic evenly-spaced temporal baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "P2.3 v5.23 budget-headroom audit"
DEFAULT_K = [3, 5, 8, 10]
AGE_MODEL = "state_age_ols"
CANDIDATE_MODEL = "age_resid_state2_mean"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def find_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "experiments" / "p1").is_dir():
            raise FileNotFoundError(f"Not a VeloDepth project root: {root}")
        return root
    for p in Path(__file__).resolve().parents:
        if (p / "experiments" / "p1").is_dir() and (p / "outputs").is_dir():
            return p
    raise FileNotFoundError(
        "Cannot find project root. Pass --repo-root /root/autodl-tmp/SVR-Depth/velodepth"
    )


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    loso_path = input_dir / "p2_3_v5_22_loso_predictions.csv"
    external_path = input_dir / "p2_3_v5_22_external_predictions.csv"
    for p in (loso_path, external_path):
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Required v5.22 output is missing or empty: {p}")

    loso_raw = pd.read_csv(loso_path)
    external_raw = pd.read_csv(external_path)

    required_loso = {"heldout_sequence", "anchor_frame", "model", "prediction", "target"}
    required_external = {
        "sequence", "anchor_frame", "target_gain",
        "age_baseline_prediction", "candidate_prediction",
    }
    if not required_loso.issubset(loso_raw.columns):
        raise ValueError(f"LOSO CSV missing columns: {sorted(required_loso - set(loso_raw.columns))}")
    if not required_external.issubset(external_raw.columns):
        raise ValueError(
            f"External CSV missing columns: {sorted(required_external - set(external_raw.columns))}"
        )

    loso = loso_raw[loso_raw["model"].isin([AGE_MODEL, CANDIDATE_MODEL])].copy()
    loso = loso.rename(columns={
        "heldout_sequence": "sequence",
        "target": "target_gain",
        "prediction": "score",
    })
    loso["split"] = "LOSO"

    age = external_raw[[
        "sequence", "anchor_frame", "target_gain", "age_baseline_prediction"
    ]].copy()
    age = age.rename(columns={"age_baseline_prediction": "score"})
    age["model"] = AGE_MODEL
    age["split"] = "external_diagnostic"

    candidate = external_raw[[
        "sequence", "anchor_frame", "target_gain", "candidate_prediction"
    ]].copy()
    candidate = candidate.rename(columns={"candidate_prediction": "score"})
    candidate["model"] = CANDIDATE_MODEL
    candidate["split"] = "external_diagnostic"

    external = pd.concat([age, candidate], ignore_index=True)
    use_cols = ["split", "sequence", "anchor_frame", "model", "score", "target_gain"]
    loso = loso[use_cols].copy()
    external = external[use_cols].copy()

    for label, df in [("LOSO", loso), ("external", external)]:
        df["anchor_frame"] = pd.to_numeric(df["anchor_frame"], errors="coerce")
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["target_gain"] = pd.to_numeric(df["target_gain"], errors="coerce")
        if df[["anchor_frame", "score", "target_gain"]].isna().any().any():
            raise ValueError(f"{label} input contains NaN/non-numeric frame, score, or target values.")
        if not np.isfinite(df[["anchor_frame", "score", "target_gain"]].to_numpy(float)).all():
            raise ValueError(f"{label} input contains non-finite values.")
        if df.duplicated(["split", "sequence", "anchor_frame", "model"]).any():
            raise ValueError(f"{label} input has duplicate sequence/frame/model rows.")

    # Target labels should match across the two models for each sequence/frame.
    for split, df in [("LOSO", loso), ("external_diagnostic", external)]:
        counts = df.groupby(["sequence", "anchor_frame"])["target_gain"].nunique()
        if (counts > 1).any():
            raise ValueError(f"Target values differ between models in {split} input.")

    meta = {
        "loso_input": str(loso_path),
        "loso_sha256": sha256(loso_path),
        "external_input": str(external_path),
        "external_sha256": sha256(external_path),
        "loso_rows": int(len(loso)),
        "external_rows": int(len(external)),
    }
    return loso, external, meta


def topk_rows(group: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    # Stable deterministic tie break: higher score first, then earlier anchor.
    return group.sort_values(
        [score_col, "anchor_frame"], ascending=[False, True], kind="mergesort"
    ).head(k)


def evenly_spaced_rows(group: pd.DataFrame, k: int) -> pd.DataFrame:
    ordered = group.sort_values("anchor_frame", kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    idx = np.rint(np.linspace(0, n - 1, k)).astype(int)
    if len(set(idx.tolist())) != k:
        raise RuntimeError(f"Could not construct {k} unique temporal anchors from n={n}.")
    return ordered.iloc[idx]


def build_metrics(data: pd.DataFrame, budgets: list[int]) -> pd.DataFrame:
    rows = []
    for (split, sequence, model), group in data.groupby(
        ["split", "sequence", "model"], sort=True
    ):
        group = group.sort_values("anchor_frame", kind="mergesort").reset_index(drop=True)
        n = len(group)
        for k in budgets:
            if k > n:
                continue
            selected = topk_rows(group, "score", k)
            oracle = topk_rows(group, "target_gain", k)
            fixed = evenly_spaced_rows(group, k)
            selected_mean = float(selected["target_gain"].mean())
            oracle_mean = float(oracle["target_gain"].mean())
            fixed_mean = float(fixed["target_gain"].mean())
            regret = oracle_mean - selected_mean
            rows.append({
                "split": split,
                "sequence": sequence,
                "model": model,
                "n_candidates": int(n),
                "K": int(k),
                "selection_fraction": float(k / n),
                "selected_mean_gain": selected_mean,
                "oracle_mean_gain": oracle_mean,
                "fixed_temporal_mean_gain": fixed_mean,
                "all_candidate_mean_gain": float(group["target_gain"].mean()),
                "regret_to_oracle": float(regret),
                "selected_positive_rate": float((selected["target_gain"] > 0).mean()),
                "oracle_overlap_count": int(
                    len(set(selected["anchor_frame"]) & set(oracle["anchor_frame"]))
                ),
                "oracle_overlap_fraction": float(
                    len(set(selected["anchor_frame"]) & set(oracle["anchor_frame"])) / k
                ),
                "selected_anchor_frames": ",".join(
                    str(int(x)) for x in sorted(selected["anchor_frame"].tolist())
                ),
                "oracle_anchor_frames": ",".join(
                    str(int(x)) for x in sorted(oracle["anchor_frame"].tolist())
                ),
                "fixed_temporal_anchor_frames": ",".join(
                    str(int(x)) for x in sorted(fixed["anchor_frame"].tolist())
                ),
            })
    return pd.DataFrame(rows)


def build_summaries(per_sequence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro = (
        per_sequence.groupby(["split", "K", "model"], as_index=False)
        .agg(
            sequence_count=("sequence", "nunique"),
            macro_selected_mean_gain=("selected_mean_gain", "mean"),
            macro_oracle_mean_gain=("oracle_mean_gain", "mean"),
            macro_fixed_temporal_mean_gain=("fixed_temporal_mean_gain", "mean"),
            macro_all_candidate_mean_gain=("all_candidate_mean_gain", "mean"),
            macro_regret_to_oracle=("regret_to_oracle", "mean"),
            macro_selected_positive_rate=("selected_positive_rate", "mean"),
            macro_oracle_overlap_fraction=("oracle_overlap_fraction", "mean"),
        )
    )
    pivot_gain = per_sequence.pivot_table(
        index=["split", "sequence", "K"],
        columns="model",
        values="selected_mean_gain",
        aggfunc="first",
    ).reset_index()
    pivot_regret = per_sequence.pivot_table(
        index=["split", "sequence", "K"],
        columns="model",
        values="regret_to_oracle",
        aggfunc="first",
    ).reset_index()
    delta = pivot_gain.merge(
        pivot_regret,
        on=["split", "sequence", "K"],
        suffixes=("_selected_gain", "_regret"),
    )
    # Pandas names are stable for the model names, but guard missing columns.
    age_gain_col = f"{AGE_MODEL}_selected_gain"
    cand_gain_col = f"{CANDIDATE_MODEL}_selected_gain"
    age_regret_col = f"{AGE_MODEL}_regret"
    cand_regret_col = f"{CANDIDATE_MODEL}_regret"
    if age_gain_col in delta.columns and cand_gain_col in delta.columns:
        delta["candidate_minus_age_selected_gain"] = delta[cand_gain_col] - delta[age_gain_col]
    if age_regret_col in delta.columns and cand_regret_col in delta.columns:
        delta["candidate_minus_age_regret"] = delta[cand_regret_col] - delta[age_regret_col]
    return macro, delta


def write_report(
    report_path: Path,
    per_sequence: pd.DataFrame,
    macro: pd.DataFrame,
    delta: pd.DataFrame,
    budgets: list[int],
    input_meta: dict,
) -> None:
    lines = [
        "# P2.3 v5.23 — Multi-budget Decision-Headroom Audit",
        "",
        "## 1. 实验定位",
        "",
        "CPU-only 离线审计。复用 v5.22 已保存的 LOSO 预测和 synchronous 诊断预测，",
        "将固定预算从 Top-8 扩展到 K=3/5/8/10，并加入按时间均匀分布的固定时序基线。",
        "不执行 VeloDepth 推理、不训练/校准模型、不修改权重、不需要 GPU。",
        "",
        "## 2. 输入完整性",
        "",
        f"- LOSO 预测行数：{input_meta['loso_rows']}",
        f"- 外部诊断预测行数：{input_meta['external_rows']}",
        f"- 预算：{', '.join(map(str, budgets))}",
        f"- LOSO 输入 SHA256：`{input_meta['loso_sha256']}`",
        f"- 外部输入 SHA256：`{input_meta['external_sha256']}`",
        "",
        "## 3. 宏平均结果",
        "",
        "| Split | K | Model | Sequences | Selected mean gain | Oracle mean gain | Fixed temporal mean gain | Mean regret |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in macro.sort_values(["split", "K", "model"]).iterrows():
        lines.append(
            f"| {r['split']} | {int(r['K'])} | {r['model']} | {int(r['sequence_count'])} "
            f"| {r['macro_selected_mean_gain']:.6f} | {r['macro_oracle_mean_gain']:.6f} "
            f"| {r['macro_fixed_temporal_mean_gain']:.6f} | {r['macro_regret_to_oracle']:.6f} |"
        )

    lines += [
        "",
        "## 4. Candidate 相对年龄基线",
        "",
        "| Split | Sequence | K | Candidate − age selected gain | Candidate − age regret |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in delta.sort_values(["split", "sequence", "K"]).iterrows():
        dg = r.get("candidate_minus_age_selected_gain", np.nan)
        dr = r.get("candidate_minus_age_regret", np.nan)
        lines.append(
            f"| {r['split']} | {r['sequence']} | {int(r['K'])} "
            f"| {dg:.6f} | {dr:.6f} |"
        )

    lines += [
        "",
        "## 5. 判读与限制",
        "",
        "- `oracle_mean_gain` 是在同一组已观测标签上按真实目标选出的 Top-K 上界，不是可部署策略。",
        "- `fixed_temporal_mean_gain` 是从候选锚点中均匀选 K 个锚点的确定性基线；它不是完整在线固定间隔回放。",
        "- LOSO 指标是三个序列的留一序列预测诊断；锚点未来窗口重叠，不能把每个锚点视为独立样本。",
        "- synchronous 结果只标记为 `external_diagnostic`：它有项目历史暴露，不应声称是全项目范围的全新独立测试。",
        "- 本轮只回答不同预算下的离线排序头room，不证明闭环策略收益、真实省算、端到端延迟或 RK3588 部署。",
        "",
        "## 6. 后续决策建议",
        "",
        "1. 若候选只在某个预算或某条训练序列上改善，而在 synchronous 诊断上持续退化，不要继续调参或用 synchronous 标签校准。",
        "2. 若年龄基线与 Oracle 的差距在多数预算下很小，学习型全局价值排序的增量空间有限；优先验证空间局部误差是否存在可利用的分配空间。",
        "3. 若严格预算 K=3/5 在 LOSO 中显示明显排序 headroom，但外部诊断不稳定，下一步应是新增真正未参与项目开发的序列做锁定验证，而不是扩大模型复杂度。",
        "4. 即使离线排序通过，仍须单独验证真实局部执行是否减少昂贵算子调用，并在同实际成本下改善完整序列质量。",
        "",
        "## 7. 输入路径",
        "",
        f"- `{input_meta['loso_input']}`",
        f"- `{input_meta['external_input']}`",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--input-dir", default=None,
                        help="Directory containing v5.22 LOSO/external prediction CSVs.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--budgets", default="3,5,8,10")
    args = parser.parse_args()

    root = find_root(args.repo_root)
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else (
        root / "outputs" / "p2_3_v5_22_sequence_shift_diagnostic"
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        root / "outputs" / "p2_3_v5_23_budget_headroom"
    )
    budgets = sorted({int(x.strip()) for x in args.budgets.split(",") if x.strip()})
    if not budgets or any(k < 1 for k in budgets):
        parser.error("--budgets must be comma-separated positive integers")

    out_dir.mkdir(parents=True, exist_ok=True)
    loso, external, input_meta = load_inputs(input_dir)
    data = pd.concat([loso, external], ignore_index=True)
    per_sequence = build_metrics(data, budgets)
    macro, delta = build_summaries(per_sequence)

    per_path = out_dir / "p2_3_v5_23_budget_headroom_per_sequence.csv"
    macro_path = out_dir / "p2_3_v5_23_budget_headroom_macro.csv"
    delta_path = out_dir / "p2_3_v5_23_candidate_delta_vs_age.csv"
    report_path = out_dir / "P2_3_v5_23_实验报告.md"
    meta_path = out_dir / "p2_3_v5_23_metadata.json"

    per_sequence.to_csv(per_path, index=False, encoding="utf-8-sig")
    macro.to_csv(macro_path, index=False, encoding="utf-8-sig")
    delta.to_csv(delta_path, index=False, encoding="utf-8-sig")
    write_report(report_path, per_sequence, macro, delta, budgets, input_meta)

    metadata = {
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "budgets": budgets,
        "method": "offline ranking headroom audit; no inference/training/weight update",
        **input_meta,
        "outputs": [per_path.name, macro_path.name, delta_path.name, report_path.name],
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Keep a portable review bundle with exact inputs and outputs.
    bundle_path = out_dir / "p2_3_v5_23_budget_headroom_review.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__).resolve(), arcname="source/p2_3_v5_23_budget_headroom_audit.py")
        for p in [per_path, macro_path, delta_path, report_path, meta_path]:
            z.write(p, arcname=f"results/{p.name}")
        for p in [
            input_dir / "p2_3_v5_22_loso_predictions.csv",
            input_dir / "p2_3_v5_22_external_predictions.csv",
        ]:
            z.write(p, arcname=f"inputs/{p.name}")

    print(f"PASS: {VERSION}")
    print(f"Output directory: {out_dir}")
    print(f"Per-sequence results: {per_path}")
    print(f"Macro results: {macro_path}")
    print(f"Candidate-vs-age deltas: {delta_path}")
    print(f"Report: {report_path}")
    print(f"Review ZIP: {bundle_path}")


if __name__ == "__main__":
    main()
