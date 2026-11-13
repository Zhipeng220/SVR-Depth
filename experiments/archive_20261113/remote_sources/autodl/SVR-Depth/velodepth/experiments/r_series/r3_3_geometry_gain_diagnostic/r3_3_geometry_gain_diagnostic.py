#!/usr/bin/env python3
"""R3.3: decompose candidate-vs-propagation geometry and pair it with writeback gain.

Diagnostic-only: frozen VeloDepth, no learned predictor, no GT in feature construction
or online action choice. A single AgeGate trajectory (action slots only at 9/37/65 by
default) is replayed. Extra diagnostic candidates at every 4th frame are computed from
restored pre-action state and discarded; they do not alter the online trajectory.
At each eligible diagnostic anchor, paired future branches are forked from the same
post-propagation state: B0=discard candidate geometry, B1=commit candidate geometry.
Future branches execute their own native restarts and do not replay later optional actions.
This is a local counterfactual diagnostic, not full-policy gain and not a latency benchmark.
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

HERE = Path(__file__).resolve().parent
R32_DIR = HERE.parent / "r3_2_reset_aware_drift_gate"
sys.path.insert(0, str(R32_DIR))
import r3_simple_commit_gate as r32
from r3_3_geometry import decompose_depth_difference, discounted_gain, diagnostic_anchors, association_rows


def write_csv(path: Path, rows):
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")


class R33Experiment(r32.R3Experiment):
    def paired_future_gain(self, state_after_prop, candidate, anchor, horizon, gamma):
        """Compare candidate writeback/discard from exactly the same post-prop state."""
        actual_snapshot = self.capture(state_after_prop)
        result = {
            "counterfactual_status": "pending", "gain_horizon_H": int(horizon),
            "gain_gamma": float(gamma), "gain_future_metric_pairs": 0,
            "gain_future_gt_frames": 0, "gain_native_restarts_B0": 0,
            "gain_native_restarts_B1": 0, "gain_absrel_discounted_sum": float("nan"),
            "gain_absrel_discounted_per_valid_frame": float("nan"),
            "gain_mean_future_absrel_B0": float("nan"), "gain_mean_future_absrel_B1": float("nan"),
            "gain_mean_future_rmse_B0": float("nan"), "gain_mean_future_rmse_B1": float("nan"),
            "gain_mean_future_delta1_B0": float("nan"), "gain_mean_future_delta1_B1": float("nan"),
        }
        traces = []
        try:
            b0_state = self.restore(actual_snapshot)
            b1_state = self.restore(actual_snapshot)
            self.commit(b1_state, candidate)
            b0_runtime = self.capture(b0_state)
            b1_runtime = self.capture(b1_state)
            rows_b0, rows_b1 = [], []
            restart_b0 = restart_b1 = gt_count = 0
            for k in range(1, horizon + 1):
                frame_idx = anchor + k
                if frame_idx >= self.T:
                    break

                # B0 branch: restore its own state and decoder runtime, then advance.
                b0_state = self.restore(b0_runtime)
                payload0 = self.h.clone_payload(self.inputs)
                out0, b0_state, native0 = self.stream.propagate_step(payload0, frame_idx, b0_state, self.T)
                if native0:
                    payload0_kf = self.h.clone_payload(self.inputs)
                    out0, b0_state = self.stream.keyframe_step(payload0_kf, frame_idx, b0_state, self.T)
                m0 = self.cf.eval_output(out0, self.model, self.paddings, self.padded_shape, self.gt[frame_idx])
                b0_runtime = self.capture(b0_state)

                # B1 branch: same input/frame, but persistent geometry fields were committed at anchor.
                b1_state = self.restore(b1_runtime)
                payload1 = self.h.clone_payload(self.inputs)
                out1, b1_state, native1 = self.stream.propagate_step(payload1, frame_idx, b1_state, self.T)
                if native1:
                    payload1_kf = self.h.clone_payload(self.inputs)
                    out1, b1_state = self.stream.keyframe_step(payload1_kf, frame_idx, b1_state, self.T)
                m1 = self.cf.eval_output(out1, self.model, self.paddings, self.padded_shape, self.gt[frame_idx])
                b1_runtime = self.capture(b1_state)

                rows_b0.append(m0); rows_b1.append(m1)
                restart_b0 += int(bool(native0)); restart_b1 += int(bool(native1))
                gt_count += int(bool(self.records[frame_idx]["gt_available"]))
                traces.append({
                    "sequence": self.args.sequence, "anchor_frame_zero_based": int(anchor),
                    "future_frame_zero_based": int(frame_idx), "future_step_k": int(k),
                    "branch_B0": "DISCARD", "branch_B1": "COMMIT",
                    "native_restart_B0": int(bool(native0)), "native_restart_B1": int(bool(native1)),
                    "gt_available": int(bool(self.records[frame_idx]["gt_available"])),
                    "absrel_B0": float(m0.get("absrel", float("nan"))),
                    "absrel_B1": float(m1.get("absrel", float("nan"))),
                    "rmse_B0": float(m0.get("rmse", float("nan"))),
                    "rmse_B1": float(m1.get("rmse", float("nan"))),
                    "delta1_B0": float(m0.get("delta1", float("nan"))),
                    "delta1_B1": float(m1.get("delta1", float("nan"))),
                })
            discounted_sum, gain_per_pair, pair_count = discounted_gain(rows_b0, rows_b1, gamma)
            def mean_metric(rows, key):
                vals = [float(r.get(key, float("nan"))) for r in rows]
                vals = [v for v in vals if math.isfinite(v)]
                return sum(vals) / len(vals) if vals else float("nan")
            result.update({
                "counterfactual_status": "ok" if pair_count else "no_valid_metric_pairs",
                "gain_future_metric_pairs": int(pair_count), "gain_future_gt_frames": int(gt_count),
                "gain_native_restarts_B0": int(restart_b0), "gain_native_restarts_B1": int(restart_b1),
                "gain_absrel_discounted_sum": float(discounted_sum) if pair_count else float("nan"),
                "gain_absrel_discounted_per_valid_frame": float(gain_per_pair),
                "gain_mean_future_absrel_B0": mean_metric(rows_b0, "absrel"),
                "gain_mean_future_absrel_B1": mean_metric(rows_b1, "absrel"),
                "gain_mean_future_rmse_B0": mean_metric(rows_b0, "rmse"),
                "gain_mean_future_rmse_B1": mean_metric(rows_b1, "rmse"),
                "gain_mean_future_delta1_B0": mean_metric(rows_b0, "delta1"),
                "gain_mean_future_delta1_B1": mean_metric(rows_b1, "delta1"),
                "gain_positive_means": "candidate persistent geometry writeback lowers future AbsRel",
                "gain_protocol": "same post-propagation state; B0 discard vs B1 commit four geometry fields; branch-specific native restarts; no future optional actions",
            })
        finally:
            # Restore both the decoder runtime and a fresh copy of the actual state.
            restored_actual_state = self.restore(actual_snapshot)
        return result, traces, restored_actual_state

    def run_r33(self, action_frames, diag_frames, horizon, gamma, age_threshold):
        self.seed(self.args.seed)
        state = self.restore(self.base)
        clock = r32.r2a.Clocks()
        action_slots, diag_slots = set(action_frames), set(diag_frames)
        anchor_rows, frame_rows, future_rows = [], [], []
        last_kf = last_geom = 0
        self.sync()
        for t in range(self.T):
            self._last_prop_output = None
            need_candidate = t in diag_slots
            out, state, ev, candidate = self.prepare_step(state, t, need_candidate)
            prop_out = self._last_prop_output
            action = "NO_OPTIONAL_ACTION"
            did_commit = False
            features = {}
            if need_candidate and candidate is not None:
                features = self.causal_features(state, candidate, out, prop_out)
                candidate_depth = self.cf.output_depth(out, self.model, self.paddings, self.padded_shape)
                prop_depth = self.cf.output_depth(prop_out, self.model, self.paddings, self.padded_shape)
                geom = decompose_depth_difference(candidate_depth, prop_depth)
                # Features and paired gain are measured at the same exact pre-action trajectory state.
                gain, traces, restored_state = self.paired_future_gain(
                    state, candidate, t, horizon, gamma
                )
                # The paired counterfactual restores the model runtime and returns a
                # fresh copy of the actual state snapshot. Continue the main trajectory
                # from that restored state, not the pre-branch object.
                state = restored_state
                future_rows.extend(traces)
                row = {
                    "sequence": self.args.sequence, "anchor_frame_zero_based": int(t),
                    "is_policy_action_anchor": int(t in action_slots),
                    "trajectory_policy": "AgeGate_only_at_action_frames",
                    "logged_action_before_this_anchor": "not_applicable_new_replay",
                    "native_restart_at_anchor": 0,
                    **features, **geom, **gain,
                }
                anchor_rows.append(row)
                # paired_future_gain restores the decoder runtime to the actual propagated branch;
                # the live `state` object was never mutated by either counterfactual branch.
                if t in action_slots:
                    do_commit = r32.decide("AgeGate", features, age_threshold, 0.04)
                    action = "COMMIT" if do_commit else "DISCARD"
                    if do_commit:
                        self.commit(state, candidate)
                        did_commit = True
                    current_out = out  # refresh candidate output is used at the action anchor either way
                else:
                    action = "DIAGNOSTIC_CANDIDATE_DISCARDED_NO_POLICY_ACTION"
                    current_out = prop_out  # diagnostic-only candidate must not change current output
            else:
                current_out = out
                if t in action_slots and ev.get("native_restart"):
                    action = "NATIVE_RESTART_NO_OPTIONAL_ACTION"
                elif need_candidate:
                    action = "DIAGNOSTIC_CANDIDATE_UNAVAILABLE"
                if need_candidate:
                    anchor_rows.append({
                        "sequence": self.args.sequence, "anchor_frame_zero_based": int(t),
                        "is_policy_action_anchor": int(t in action_slots),
                        "trajectory_policy": "AgeGate_only_at_action_frames",
                        "counterfactual_status": "skipped_anchor_restart_or_no_candidate",
                        "skip_reason": "native restart takes priority or candidate unavailable",
                        "native_restart_at_anchor": int(bool(ev.get("native_restart", 0))),
                        "gain_horizon_H": int(horizon), "gain_gamma": float(gamma),
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
                "diagnostic_candidate_computed": int(need_candidate and candidate is not None),
                "policy_action_anchor": int(t in action_slots), "gate_action": action,
                "geom_commit": int(did_commit), "native_restart": int(bool(ev.get("native_restart", 0))),
                "output_source": "REFRESH" if (t in action_slots and candidate is not None) or native or t == 0 else "PROP",
                "gt_available": int(bool(self.records[t]["gt_available"])),
                **asdict(clock), **ev,
                **{m: float(metric.get(m, float("nan"))) for m in ("absrel", "rmse", "delta1")},
            })
            del candidate, out
        feature_names = [
            "depth_disagreement_rel_mean", "raw_rel_mean", "raw_rel_median", "raw_rel_p90",
            "scale_shift_scale", "scale_shift_shift_over_median_prop", "scale_shift_abs_scale_deviation",
            "fit_residual_norm_mean", "fit_residual_norm_median", "fit_residual_norm_p90",
            "fit_residual_fraction_gt_0p05", "fit_residual_grid_cv", "fit_residual_grid_max",
            "distance_kf", "flow_magnitude", "total_luma_change",
        ]
        associations = association_rows(anchor_rows, feature_names)
        return anchor_rows, frame_rows, future_rows, associations


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", type=Path, default=HERE.parents[2])
    ap.add_argument("--data-root", type=Path, default=Path("/root/autodl-tmp/datasets/rgbd_bonn_crowd2"))
    ap.add_argument("--sequence", default="crowd2")
    ap.add_argument("--frames", type=int, default=86)
    ap.add_argument("--action-frames", default="9,37,65")
    ap.add_argument("--diagnostic-offset", type=int, default=1)
    ap.add_argument("--diagnostic-stride", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--age-threshold", type=float, default=16.0)
    args = ap.parse_args()
    root = args.project_root.resolve()
    action_frames = [int(x) for x in args.action_frames.split(",") if x.strip()]
    diag_frames = diagnostic_anchors(args.frames, args.diagnostic_offset, args.diagnostic_stride, args.horizon)
    if len(action_frames) != len(set(action_frames)) or not set(action_frames).issubset(set(diag_frames)):
        ap.error("Every action frame must be a unique member of the diagnostic anchor schedule")
    if args.horizon < 1 or not (0 < args.gamma <= 1):
        ap.error("horizon must be positive and gamma in (0,1]")
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        ap.error(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    meta = {
        "experiment": "R3.3 candidate geometry decomposition and same-state writeback-gain diagnosis",
        "created_utc": stamp, "sequence": args.sequence, "frames": args.frames,
        "action_frames_zero_based": action_frames, "diagnostic_frames_zero_based": diag_frames,
        "diagnostic_anchor_count_requested": len(diag_frames), "horizon_H": args.horizon,
        "gamma": args.gamma, "age_threshold": args.age_threshold,
        "trajectory": "AgeGate actions only at action_frames; diagnostic-only candidates at other anchors are discarded and propagation output remains current output",
        "weights_frozen": True, "predictor_training": False,
        "online_action_uses_gt": False, "GT_use": "future GT is used only to evaluate paired counterfactual branches after feature capture; never used for action selection",
        "geometry_fit": "robust Huber IRLS fit candidate_depth ~= scale * propagation_depth + shift; fit residual measures local difference left after global affine alignment",
        "counterfactual_protocol": "same post-propagation state; B0 discard vs B1 commit four geometry fields; branch-specific native restarts; no future optional actions replayed",
        "not_a_full_policy_gain": True,
        "cost_scope": "diagnostic-only candidates and paired branches add compute; timings are not deployment latency; no candidate inference savings claimed",
        "status": "started",
    }
    try:
        audit = r32.r2a.static_preflight(root)
        write_json(out / "source_audit.json", audit)
        meta["source_audit_passed"] = True
        if args.preflight_only:
            meta["status"] = "preflight_passed"
            write_json(out / "metadata.json", meta)
            print(f"R3.3 preflight passed: {out}")
            return 0
        exp = R33Experiment(root, args)
        with exp.torch.inference_mode():
            write_json(out / "r0_runtime_audit.json", exp.r0_audit(diag_frames))
            anchors, frames, future, assoc = exp.run_r33(
                action_frames, diag_frames, args.horizon, args.gamma, args.age_threshold
            )
        write_csv(out / "r33_geometry_gain_diagnostics.csv", anchors)
        write_csv(out / "r33_online_frame_metrics.csv", frames)
        write_csv(out / "r33_future_branch_trace.csv", future)
        write_csv(out / "r33_feature_gain_association.csv", assoc)
        meta.update({
            "status": "completed", "gpu": exp.torch.cuda.get_device_name(0),
            "torch_version": str(exp.torch.__version__),
            "gt_matches": sum(int(r["gt_available"]) for r in exp.records),
            "anchors_with_geometry_and_gain": sum(1 for r in anchors if r.get("counterfactual_status") == "ok"),
            "anchors_skipped": sum(1 for r in anchors if r.get("counterfactual_status") != "ok"),
            "source_audit_passed": True,
            "all_diagnostic_candidates_are_not_online_actions": True,
            "gain_target": "mean over valid paired future frames of gamma**k * (AbsRel_B0 - AbsRel_B1); missing pairs keep original k exponent",
            "outputs": {
                "geometry_gain_diagnostics": str(out / "r33_geometry_gain_diagnostics.csv"),
                "online_frame_metrics": str(out / "r33_online_frame_metrics.csv"),
                "future_branch_trace": str(out / "r33_future_branch_trace.csv"),
                "feature_gain_association": str(out / "r33_feature_gain_association.csv"),
            },
        })
        good = [r for r in anchors if r.get("counterfactual_status") == "ok"]
        report_lines = [
            "# R3.3 候选几何差异分解与写回收益诊断",
            "", f"- 序列：{args.sequence}", f"- 帧数：{args.frames}",
            f"- 有效 GT 帧：{meta['gt_matches']}", f"- 诊断锚点：{len(diag_frames)}",
            f"- 有效配对收益锚点：{len(good)}", f"- 跳过锚点：{meta['anchors_skipped']}",
            f"- 动作锚点：{action_frames}", f"- H={args.horizon}, gamma={args.gamma}",
            "", "## 解释边界", "",
            "- 几何特征只使用当前传播深度、当前刷新候选和当前状态，不使用 GT。",
            "- 配对未来收益是同一锚点的局部反事实，不等于完整在线策略收益。",
            "- H=16 的相邻未来窗口重叠，锚点不是独立样本；相关系数仅作描述性诊断，不报告显著性或据此选阈值。",
            "- 诊断锚点上额外计算候选并运行分支，因此本实验不衡量部署延迟，也不声称节省计算。",
        ]
        (out / "R3_3诊断摘要.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    except Exception:
        meta["status"] = "failed"
        (out / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(out / "metadata.json", meta)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    write_json(out / "metadata.json", meta)
    print(f"R3.3 results: {out}")
    print(f"anchors={len(anchors)}; valid paired gains={meta['anchors_with_geometry_and_gain']}; skipped={meta['anchors_skipped']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
