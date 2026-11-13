#!/usr/bin/env python3
"""R3.5.5: fixed-plan post-candidate commit gate with one frozen geometry feature.

The collection pass uses the same six fixed action slots per sequence and ALWAYS
commits each candidate. At each action slot it records the pre-decision raw geometry
feature and a same-state COMMIT/DISCARD counterfactual gain. A leave-one-sequence-out
threshold is selected using only the other two sequences. If the OOF local-utility
screen fails, full gate replays are skipped. If it passes, each held-out sequence is
replayed with its training-fold threshold and compared with its AlwaysCommit baseline.

This experiment does not save candidate inference compute: the candidate is already
computed before the gate decides whether to commit its persistent geometry state.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT_DEFAULT = HERE.parents[2]
R33_DIR = HERE.parent / "r3_3_geometry_gain_diagnostic"
if not R33_DIR.exists():
    raise FileNotFoundError(f"Required R3.3 dependency not found: {R33_DIR}")
sys.path.insert(0, str(R33_DIR))
import r3_3_geometry_gain_diagnostic as r33
from r3_3_geometry import decompose_depth_difference

SEQUENCES = ("crowd2", "balloon2", "crowd3")
ACTION_FRAMES_DEFAULT = (5, 17, 29, 41, 49, 61)
FEATURE = "raw_rel_median"
GAIN = "gain_absrel_discounted_per_valid_frame"
ALPHA_EPS = 1e-12


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


def write_csv(path: Path, rows):
    r33.write_csv(path, rows)


def choose_threshold(train: pd.DataFrame) -> dict:
    """Choose a one-sided threshold using training sequences only.

    Decision is fixed to COMMIT iff raw_rel_median >= threshold. Among thresholds,
    maximize sum of local counterfactual gain on training anchors. Ties prefer the
    lowest threshold (i.e. the least intervention / closest to AlwaysCommit).
    """
    d = train[[FEATURE, GAIN]].copy()
    d[FEATURE] = pd.to_numeric(d[FEATURE], errors="coerce")
    d[GAIN] = pd.to_numeric(d[GAIN], errors="coerce")
    d = d[np.isfinite(d[FEATURE]) & np.isfinite(d[GAIN])]
    if d.empty:
        return {"threshold": None, "active": False, "reason": "no_valid_training_rows",
                "train_n": 0, "train_utility_always": float("nan"),
                "train_utility_gate": float("nan"), "train_delta_utility": float("nan")}
    x = d[FEATURE].to_numpy(float)
    g = d[GAIN].to_numpy(float)
    ux = np.unique(x)
    if len(ux) == 1:
        candidates = [float(ux[0]), float(np.nextafter(ux[0], math.inf))]
    else:
        mids = ((ux[:-1] + ux[1:]) / 2.0).tolist()
        candidates = [float(ux[0] - max(abs(ux[0]), 1.0) * 1e-9), *map(float, mids),
                      float(np.nextafter(ux[-1], math.inf))]
    always = float(np.sum(g))
    scored = []
    for threshold in candidates:
        committed = x >= threshold
        utility = float(np.sum(g[committed]))
        scored.append((utility, int(committed.sum()), float(threshold)))
    # Highest utility, then more commits to prefer the simpler action, then lowest threshold.
    best_utility = max(r[0] for r in scored)
    tied = [r for r in scored if abs(r[0] - best_utility) <= ALPHA_EPS]
    best = sorted(tied, key=lambda r: (-r[1], r[2]))[0]
    delta = best[0] - always
    active = bool(delta > 1e-10 and best[1] < len(d))
    return {
        "threshold": best[2] if active else None, "active": active,
        "reason": "training_utility_improved" if active else "no_training_improvement_over_always_commit",
        "train_n": int(len(d)), "train_commit_n": int(best[1]),
        "train_utility_always": always, "train_utility_gate": float(best[0]),
        "train_delta_utility": float(delta),
    }


def evaluate_threshold(test: pd.DataFrame, threshold: float | None) -> dict:
    d = test[[FEATURE, GAIN]].copy()
    d[FEATURE] = pd.to_numeric(d[FEATURE], errors="coerce")
    d[GAIN] = pd.to_numeric(d[GAIN], errors="coerce")
    d = d[np.isfinite(d[FEATURE]) & np.isfinite(d[GAIN])]
    if d.empty:
        return {"n": 0, "always_utility": float("nan"), "gate_utility": float("nan"),
                "delta_utility": float("nan"), "gate_commit_n": 0,
                "negative_gain_n": 0, "negative_gain_recall_discarded": float("nan")}
    x, g = d[FEATURE].to_numpy(float), d[GAIN].to_numpy(float)
    commit = np.ones(len(d), dtype=bool) if threshold is None else x >= float(threshold)
    neg = g < 0
    return {
        "n": int(len(d)), "always_utility": float(g.sum()),
        "gate_utility": float(g[commit].sum()),
        "delta_utility": float(g[commit].sum() - g.sum()),
        "gate_commit_n": int(commit.sum()),
        "negative_gain_n": int(neg.sum()),
        "negative_gain_recall_discarded": float(np.sum((~commit) & neg) / np.sum(neg)) if np.sum(neg) else float("nan"),
        "gate_negative_commits": int(np.sum(commit & neg)),
        "threshold": float(threshold) if threshold is not None else float("nan"),
    }


def calibrate_loso(input_root: Path, out_dir: Path) -> dict:
    """Fit thresholds on two sequences and evaluate the third; labels are baseline-trajectory local gains."""
    frames = []
    for seq in SEQUENCES:
        p = input_root / seq / "r355_action_geometry_gain.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing collection output: {p}")
        d = pd.read_csv(p)
        d["sequence"] = seq
        frames.append(d)
    data = pd.concat(frames, ignore_index=True)
    data = data[(data["counterfactual_status"] == "ok") &
                np.isfinite(pd.to_numeric(data[FEATURE], errors="coerce")) &
                np.isfinite(pd.to_numeric(data[GAIN], errors="coerce"))].copy()
    if data.empty:
        raise RuntimeError("No valid same-state feature/gain rows for LOSO calibration")
    fold_rows, oof_rows = [], []
    thresholds = {}
    for heldout in SEQUENCES:
        train = data[data.sequence != heldout]
        test = data[data.sequence == heldout]
        selected = choose_threshold(train)
        threshold = selected["threshold"] if selected["active"] else None
        thresholds[heldout] = {
            "threshold": threshold,
            "active": bool(selected["active"]),
            "train_sequences": [s for s in SEQUENCES if s != heldout],
            "heldout_sequence": heldout,
            **selected,
        }
        metrics = evaluate_threshold(test, threshold)
        fold_rows.append({"heldout_sequence": heldout, **metrics, **thresholds[heldout]})
        for _, row in test.iterrows():
            commit = (True if threshold is None else float(row[FEATURE]) >= float(threshold))
            oof_rows.append({
                "sequence": heldout, "anchor_frame_zero_based": int(row["anchor_frame_zero_based"]),
                "feature_raw_rel_median": float(row[FEATURE]), "actual_gain": float(row[GAIN]),
                "threshold": float(threshold) if threshold is not None else float("nan"),
                "threshold_active": int(threshold is not None),
                "gate_action": "COMMIT" if commit else "DISCARD",
                "gate_local_utility": float(row[GAIN]) if commit else 0.0,
                "always_commit_local_utility": float(row[GAIN]),
            })
    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows)
    delta = pd.to_numeric(fold_df["delta_utility"], errors="coerce")
    wins = int((delta > 1e-10).sum())
    pooled_delta = float(delta.sum())
    passed = bool(wins >= 2 and pooled_delta > 1e-10)
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(out_dir / "r355_loso_thresholds_and_utility.csv", index=False, encoding="utf-8-sig")
    oof_df.to_csv(out_dir / "r355_oof_action_decisions.csv", index=False, encoding="utf-8-sig")
    decision = {
        "experiment": "R3.5.5 fixed-plan post-candidate geometry commit gate",
        "feature": FEATURE, "decision_rule": "COMMIT iff raw_rel_median >= threshold",
        "threshold_selection": "maximize sum of same-state local gain on training sequences; ties prefer more commits",
        "evaluation": "leave-one-sequence-out across crowd2, balloon2, crowd3",
        "valid_action_rows": int(len(data)),
        "positive_gain_rows": int((pd.to_numeric(data[GAIN]) > 0).sum()),
        "negative_gain_rows": int((pd.to_numeric(data[GAIN]) < 0).sum()),
        "folds_improved_over_always_commit": wins,
        "pooled_oof_delta_utility": pooled_delta,
        "offline_gate_pass": passed,
        "gate_replay_allowed": passed,
        "thresholds_by_heldout_sequence": thresholds,
        "stop_rule": "Proceed to full gate replays only if OOF local utility exceeds AlwaysCommit on >=2/3 held-out sequences and pooled delta utility > 0. Otherwise stop this branch.",
        "limitations": [
            "Only three development sequences; not an independent external test.",
            "H=16 counterfactual windows overlap.",
            "Local utility is not a complete closed-loop result.",
            "Candidate inference is performed before the gate; no compute savings are claimed."
        ],
    }
    write_json(out_dir / "r355_gate_decision.json", decision)
    lines = [
        "# R3.5.5 离线门控筛选报告", "",
        f"- 有效真实计划动作锚点：{len(data)}",
        f"- 正收益标签：{decision['positive_gain_rows']}",
        f"- 负收益标签：{decision['negative_gain_rows']}",
        f"- LOSO 中优于 AlwaysCommit 的折数：{wins}/3",
        f"- 汇总 OOF 局部效用增量：{pooled_delta:.8f}",
        f"- 离线筛选：{'通过，可运行冻结门控闭环' if passed else '未通过，停止，不运行额外门控闭环'}",
        "", "## 按留出序列结果", "",
        "| 留出序列 | 阈值是否启用 | 阈值 | N | AlwaysCommit 局部效用 | Gate 局部效用 | 增量 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in fold_df.iterrows():
        thr = "AlwaysCommit（训练折未获益）" if pd.isna(r.get("threshold")) else f"{float(r['threshold']):.8g}"
        lines.append(f"| {r['heldout_sequence']} | {bool(r['active'])} | {thr} | {int(r['n'])} | {float(r['always_utility']):.8f} | {float(r['gate_utility']):.8f} | {float(r['delta_utility']):.8f} |")
    lines += [
        "", "## 解释边界", "",
        "该表使用在 AlwaysCommit 参考轨迹上采集的特征与同状态反事实标签，属于阈值的离线筛选。只有通过预先声明的止损条件，才执行使用折外阈值的完整状态闭环重放。",
        "即使离线筛选通过，仍需以闭环逐帧指标作最终判断。三条序列属于开发集，不能作为独立外部泛化证明。",
    ]
    (out_dir / "R3_5_5_离线门控筛选报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


class R355Experiment(r33.R33Experiment):
    def run_fixed_plan(self, action_frames, horizon, gamma, mode, threshold, compute_gain):
        self.seed(self.args.seed)
        state = self.restore(self.base)
        clock = r33.r32.r2a.Clocks()
        slots = set(action_frames)
        anchor_rows, frame_rows, future_rows = [], [], []
        last_kf = last_geom = 0
        self.sync()
        for t in range(self.T):
            self._last_prop_output = None
            need_candidate = t in slots
            out, state, ev, candidate = self.prepare_step(state, t, need_candidate)
            prop_out = self._last_prop_output
            action = "NO_OPTIONAL_ACTION"
            did_commit = False
            geom = {}
            features = {}
            gain = {"counterfactual_status": "not_computed_gate_replay" if not compute_gain else "candidate_unavailable"}
            if need_candidate and candidate is not None:
                features = self.causal_features(state, candidate, out, prop_out)
                cand_depth = self.cf.output_depth(out, self.model, self.paddings, self.padded_shape)
                prop_depth = self.cf.output_depth(prop_out, self.model, self.paddings, self.padded_shape)
                geom = decompose_depth_difference(cand_depth, prop_depth)
                if compute_gain:
                    gain, traces, restored_state = self.paired_future_gain(state, candidate, t, horizon, gamma)
                    state = restored_state
                    future_rows.extend(traces)
                if mode == "always_commit":
                    do_commit = True
                elif mode == "never_commit":
                    do_commit = False
                elif mode == "geometry_gate":
                    value = float(geom.get(FEATURE, float("nan")))
                    do_commit = math.isfinite(value) and threshold is not None and value >= float(threshold)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                action = "COMMIT" if do_commit else "DISCARD"
                if do_commit:
                    self.commit(state, candidate)
                    did_commit = True
                anchor_rows.append({
                    "sequence": self.args.sequence,
                    "anchor_frame_zero_based": int(t),
                    "is_policy_action_anchor": 1,
                    "trajectory_policy": mode,
                    "mode": mode, "threshold_raw_rel_median": threshold,
                    "gate_feature": float(geom.get(FEATURE, float("nan"))),
                    **features, **geom, **gain,
                })
                current_out = out
            else:
                current_out = out
                if need_candidate:
                    action = "NATIVE_RESTART_NO_OPTIONAL_ACTION" if ev.get("native_restart") else "CANDIDATE_UNAVAILABLE"
                    anchor_rows.append({
                        "sequence": self.args.sequence, "anchor_frame_zero_based": int(t),
                        "is_policy_action_anchor": 1, "trajectory_policy": mode,
                        "mode": mode, "threshold_raw_rel_median": threshold,
                        "gate_action": action, "counterfactual_status": "skipped_anchor_restart_or_no_candidate",
                        "skip_reason": "native restart takes priority or candidate unavailable",
                    })
            native = bool(t == 0 or ev.get("native_restart", 0))
            if native:
                last_kf = last_geom = t
            elif did_commit:
                last_geom = t
            clock.advance(t, native=native, commit=did_commit)
            if clock.steps_since_kf_reset != t - last_kf or clock.steps_since_geom_commit != t - last_geom:
                raise RuntimeError(f"Clock transition mismatch at frame {t}")
            metric = self.cf.eval_output(current_out, self.model, self.paddings, self.padded_shape, self.gt[t])
            frame_rows.append({
                "sequence": self.args.sequence, "frame_zero_based": int(t),
                "policy_action_anchor": int(t in slots), "gate_action": action,
                "geom_commit": int(did_commit), "native_restart": int(bool(ev.get("native_restart", 0))),
                "output_source": "REFRESH" if (t in slots and candidate is not None) or native or t == 0 else "PROP",
                "gt_available": int(bool(self.records[t]["gt_available"])),
                "diagnostic_candidate_computed": int(bool(need_candidate and candidate is not None)),
                **asdict(clock), **ev,
                **{m: float(metric.get(m, float("nan"))) for m in ("absrel", "rmse", "delta1")},
            })
            del candidate, out
        return anchor_rows, frame_rows, future_rows


def make_args(ns):
    return argparse.Namespace(
        project_root=Path(ns.project_root).resolve(),
        data_root=Path(ns.data_root).resolve(),
        sequence=ns.sequence, frames=ns.frames,
        resolution_level=ns.resolution_level, gt_scale=ns.gt_scale,
        max_time_diff=ns.max_time_diff, seed=ns.seed, tolerance=ns.tolerance,
        out_dir=Path(ns.out_dir).resolve(),
    )


def run_gpu(args, mode, threshold, compute_gain):
    root = Path(args.project_root).resolve()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    action_frames = [int(x) for x in args.action_frames.split(",") if x.strip()]
    if len(action_frames) != len(set(action_frames)) or any(x < 1 or x >= args.frames for x in action_frames):
        raise ValueError("Action frames must be unique zero-based indices within the requested sequence")
    meta = {
        "experiment": "R3.5.5 fixed-plan post-candidate commit gate",
        "sequence": args.sequence, "frames": args.frames,
        "action_frames_zero_based": action_frames, "mode": mode,
        "feature": FEATURE, "threshold": threshold, "compute_gain": compute_gain,
        "horizon_H": args.horizon, "gamma": args.gamma,
        "weights_frozen": True, "predictor_training": False,
        "online_gate_uses_gt": False,
        "cost_scope": "candidate refresh inference is performed before gate; no candidate compute savings claimed",
        "status": "started",
    }
    try:
        audit = r33.r32.r2a.static_preflight(root)
        write_json(out / "source_audit.json", audit)
        meta["source_audit_passed"] = True
        exp_args = make_args(args)
        exp = R355Experiment(root, exp_args)
        with exp.torch.inference_mode():
            write_json(out / "r0_runtime_audit.json", exp.r0_audit(action_frames))
            anchors, frames, future = exp.run_fixed_plan(
                action_frames, args.horizon, args.gamma, mode, threshold, compute_gain
            )
        write_csv(out / "r355_action_geometry_gain.csv", anchors)
        write_csv(out / "r355_online_frame_metrics.csv", frames)
        write_csv(out / "r355_future_branch_trace.csv", future)
        valid = [r for r in anchors if r.get("counterfactual_status") == "ok"]
        meta.update({
            "status": "completed", "gpu": exp.torch.cuda.get_device_name(0),
            "torch_version": str(exp.torch.__version__),
            "gt_matches": int(sum(int(r["gt_available"]) for r in exp.records)),
            "action_anchors_requested": len(action_frames),
            "action_anchors_with_candidate": sum(1 for r in anchors if r.get("gate_feature") is not None),
            "valid_same_state_gain_labels": len(valid),
            "skipped_anchors": sum(1 for r in anchors if r.get("counterfactual_status") != "ok") if compute_gain else None,
            "source_audit_passed": True,
        })
        write_json(out / "metadata.json", meta)
        print(f"R3.5.5 {mode} complete: {out}")
        print(f"action anchors={len(anchors)}; valid gains={len(valid) if compute_gain else 'not computed'}")
        return 0
    except Exception:
        meta["status"] = "failed"
        write_json(out / "metadata.json", meta)
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        return 1


def compare_closed_loop(baseline_root: Path, never_root: Path, gate_root: Path,
                        calibration_json: Path, out_dir: Path):
    decision = json.loads(calibration_json.read_text(encoding="utf-8"))
    rows = []
    for seq in SEQUENCES:
        paths = {
            "AlwaysCommit": baseline_root / seq / "r355_online_frame_metrics.csv",
            "NeverCommit": never_root / seq / "r355_online_frame_metrics.csv",
            "GeometryGate": gate_root / seq / "r355_online_frame_metrics.csv",
        }
        dfs = {name: pd.read_csv(path) for name, path in paths.items()}
        lengths = {name: len(df) for name, df in dfs.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Frame count mismatch for {seq}: {lengths}")
        row = {"sequence": seq, "frames": next(iter(lengths.values()))}
        for metric in ("absrel", "rmse", "delta1"):
            arrays = {
                name: pd.to_numeric(df[metric], errors="coerce").to_numpy(float)
                for name, df in dfs.items()
            }
            valid_masks = {name: np.isfinite(vals) for name, vals in arrays.items()}
            for name, mask in valid_masks.items():
                row[f"valid_{metric}_{name}"] = int(mask.sum())
            paired_mask = np.logical_and.reduce(list(valid_masks.values()))
            row[f"paired_valid_{metric}_frames"] = int(paired_mask.sum())
            means = {
                name: float(np.mean(vals[paired_mask])) if paired_mask.any() else float("nan")
                for name, vals in arrays.items()
            }
            for name, value in means.items():
                row[f"{name}_{metric}"] = value
            row[f"GeometryGate_minus_AlwaysCommit_{metric}"] = means["GeometryGate"] - means["AlwaysCommit"]
        b_abs = row["AlwaysCommit_absrel"]
        g_abs = row["GeometryGate_absrel"]
        row["GeometryGate_relative_absrel_change"] = (g_abs - b_abs) / b_abs if b_abs else float("nan")
        for name, df in dfs.items():
            row[f"{name}_successful_commits"] = int(pd.to_numeric(df.geom_commit, errors="coerce").fillna(0).sum())
            row[f"candidate_compute_count_{name}"] = int(pd.to_numeric(df.diagnostic_candidate_computed, errors="coerce").fillna(0).sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "r355_closed_loop_comparison.csv", index=False, encoding="utf-8-sig")
    improved = int((df["GeometryGate_minus_AlwaysCommit_absrel"] < 0).sum())
    worst_rel_regression = float(df.GeometryGate_relative_absrel_change.max())
    pooled_base = float(df.AlwaysCommit_absrel.mean())
    pooled_gate = float(df.GeometryGate_absrel.mean())
    candidate_counts_match = bool(
        (df.candidate_compute_count_AlwaysCommit == df.candidate_compute_count_NeverCommit).all()
        and (df.candidate_compute_count_AlwaysCommit == df.candidate_compute_count_GeometryGate).all()
    )
    closed_pass = bool(improved >= 2 and pooled_gate < pooled_base
                       and worst_rel_regression <= 0.02 and candidate_counts_match)
    summary = {
        "offline_gate_pass": bool(decision["offline_gate_pass"]),
        "sequences_absrel_improved_vs_alwayscommit": improved,
        "mean_sequence_absrel_alwayscommit": pooled_base,
        "mean_sequence_absrel_nevercommit": float(df.NeverCommit_absrel.mean()),
        "mean_sequence_absrel_geometry_gate": pooled_gate,
        "mean_sequence_absrel_delta_gate_minus_always": pooled_gate - pooled_base,
        "worst_relative_absrel_regression": worst_rel_regression,
        "candidate_compute_counts_match": candidate_counts_match,
        "closed_loop_gate_pass": closed_pass,
        "rule": "pass iff >=2/3 sequences improve AbsRel vs AlwaysCommit, macro mean improves, no sequence regresses >2% relative, and candidate compute counts match",
        "compute_savings_claimed": False,
        "thresholds_by_heldout_sequence": decision["thresholds_by_heldout_sequence"],
    }
    write_json(out_dir / "r355_closed_loop_decision.json", summary)
    report = [
        "# R3.5.5 闭环对照报告", "",
        "| 序列 | AlwaysCommit AbsRel | NeverCommit AbsRel | GeometryGate AbsRel | Gate−Always | 相对变化 | Gate提交数 | 候选计算数（A/N/G） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        report.append(
            f"| {r.sequence} | {r.AlwaysCommit_absrel:.6f} | {r.NeverCommit_absrel:.6f} | "
            f"{r.GeometryGate_absrel:.6f} | {r.GeometryGate_minus_AlwaysCommit_absrel:+.6f} | "
            f"{r.GeometryGate_relative_absrel_change:+.2%} | {int(r.GeometryGate_successful_commits)} | "
            f"{int(r.candidate_compute_count_AlwaysCommit)}/{int(r.candidate_compute_count_NeverCommit)}/{int(r.candidate_compute_count_GeometryGate)} |"
        )
    report += [
        "", f"- 至少改善的序列数：{improved}/3",
        f"- 三序列宏平均 AbsRel：AlwaysCommit={pooled_base:.6f}；NeverCommit={df.NeverCommit_absrel.mean():.6f}；GeometryGate={pooled_gate:.6f}",
        f"- 最差单序列相对退化：{worst_rel_regression:+.2%}",
        f"- 候选计算次数是否一致：{candidate_counts_match}",
        f"- 闭环判定：{'通过' if closed_pass else '未通过'}",
        "", "## 解释边界", "",
        "候选仍在门控前计算，因此这项实验只检验同候选计算预算下的质量变化，不证明真实推理省算。",
        "三条序列属于开发集；即使通过，也只支持继续进行独立序列验证。",
    ]
    (out_dir / "R3_5_5_闭环对照报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    for name in ("collect", "gate", "never"):
        p = sub.add_parser(name)
        p.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
        p.add_argument("--data-root", type=Path, required=True)
        p.add_argument("--sequence", choices=SEQUENCES, required=True)
        p.add_argument("--frames", type=int, default=86)
        p.add_argument("--action-frames", default=",".join(map(str, ACTION_FRAMES_DEFAULT)))
        p.add_argument("--horizon", type=int, default=16)
        p.add_argument("--gamma", type=float, default=0.9)
        p.add_argument("--out-dir", type=Path, required=True)
        p.add_argument("--resolution-level", type=int, default=0)
        p.add_argument("--gt-scale", type=float, default=5000.0)
        p.add_argument("--max-time-diff", type=float, default=0.02)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--tolerance", type=float, default=1e-6)
        if name == "gate":
            p.add_argument("--calibration-json", type=Path, required=True)
    p = sub.add_parser("calibrate")
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--never-root", type=Path, required=True)
    p.add_argument("--gate-root", type=Path, required=True)
    p.add_argument("--calibration-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p = sub.add_parser("should-run-gate")
    p.add_argument("--decision-json", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "preflight":
        root = Path(args.project_root).resolve()
        required = [
            R33_DIR / "r3_3_geometry_gain_diagnostic.py",
            R33_DIR / "r3_3_geometry.py",
            R33_DIR.parent / "r3_2_reset_aware_drift_gate" / "r3_simple_commit_gate.py",
        ]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            raise SystemExit("Missing dependencies:\n" + "\n".join(missing))
        audit = r33.r32.r2a.static_preflight(root)
        print(json.dumps({"dependencies": "ok", "source_audit": audit}, ensure_ascii=False, indent=2, default=str))
    elif args.cmd == "collect":
        return run_gpu(args, "always_commit", None, True)
    elif args.cmd == "gate":
        decision = json.loads(args.calibration_json.read_text(encoding="utf-8"))
        fold = decision["thresholds_by_heldout_sequence"][args.sequence]
        threshold = fold["threshold"]
        if threshold is None:
            # Training fold did not beat AlwaysCommit; preserve the no-op baseline.
            return run_gpu(args, "always_commit", None, False)
        return run_gpu(args, "geometry_gate", float(threshold), False)
    elif args.cmd == "never":
        return run_gpu(args, "never_commit", None, False)
    elif args.cmd == "calibrate":
        decision = calibrate_loso(args.input_root, args.out_dir)
        print(json.dumps(decision, ensure_ascii=False, indent=2))
        return 0
    elif args.cmd == "compare":
        summary = compare_closed_loop(args.baseline_root, args.never_root, args.gate_root,
                                      args.calibration_json, args.out_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    elif args.cmd == "should-run-gate":
        decision = json.loads(args.decision_json.read_text(encoding="utf-8"))
        print("PASS" if decision["gate_replay_allowed"] else "STOP")
        return 0 if decision["gate_replay_allowed"] else 1
    return 0

if __name__ == "__main__":
    result = main()
    if isinstance(result, int):
        raise SystemExit(result)
