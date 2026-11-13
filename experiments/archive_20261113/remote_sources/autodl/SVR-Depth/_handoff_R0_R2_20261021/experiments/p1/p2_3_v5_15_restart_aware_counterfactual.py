#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.3 v5.15 same-state counterfactual gain diagnosis.

This experiment captures pre-action states on each actual online policy trajectory,
then forks B0 (propagation without writeback) and B1-aligned state-writeback branches
from the exact same state snapshot. It does not train/update VeloDepth weights.
The future branch protocol executes automatic restarts independently in each branch.
Future scheduled policy actions are not replayed or recomputed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

def load_replay_dependencies():
    """Load project-specific replay helpers after the standalone self-test path."""
    global replay, cf, StreamState, VeloDepthStream, VeloDepth
    global capture_state, restore_state, clone_payload, build_online_features
    global capture_packed_inputs_without_full_inference, match_gt_tolerant
    global OnlineRidge, write_csv, sync_cuda, warmup_policy, summarize
    here = Path(__file__).resolve().parent
    base_script = here / "p2_3_online_replay_v5_4_budget_sweep.py"
    if not base_script.is_file():
        raise FileNotFoundError(
            f"Required helper script is missing: {base_script}. "
            "Upload p2_3_online_replay_v5_4_budget_sweep.py beside this script."
        )
    spec = importlib.util.spec_from_file_location("svrdepth_v54_replay", base_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper script: {base_script}")
    replay = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = replay
    spec.loader.exec_module(replay)
    cf = replay.cf
    StreamState = replay.StreamState
    VeloDepthStream = replay.VeloDepthStream
    VeloDepth = replay.VeloDepth
    capture_state = replay.capture_state
    restore_state = replay.restore_state
    clone_payload = replay.clone_payload
    build_online_features = replay.build_online_features
    capture_packed_inputs_without_full_inference = replay.capture_packed_inputs_without_full_inference
    match_gt_tolerant = replay.match_gt_tolerant
    OnlineRidge = replay.OnlineRidge
    write_csv = replay.write_csv
    sync_cuda = replay.sync_cuda
    warmup_policy = replay.warmup_policy
    summarize = replay.summarize

PREDICTOR_NAMES = (
    "causal_budget_state_only",
    "causal_budget_motion_plus_learned",
)
POLICIES = (
    "auto",
    "budget_fixed",
    "causal_budget_state_only",
    "causal_budget_motion_plus_learned",
)
LEGACY_TARGET = "G_state_absrel_discounted_per_valid_frame"
TARGET = "G_state_absrel_restart_aware_discounted_per_valid_frame"


def make_fixed_budget_frames(frames, offset, stride, min_gap, k):
    eligible = [
        t for t in range(offset, frames, stride)
        if t >= min_gap
    ]
    if k < 0 or k > len(eligible):
        raise ValueError(f"K={k} outside [0,{len(eligible)}] eligible anchors")
    if k == 0:
        return set(), eligible
    if k == len(eligible):
        return set(eligible), eligible
    indices = np.rint(np.linspace(0, len(eligible) - 1, k)).astype(int).tolist()
    selected = {eligible[i] for i in indices}
    if len(selected) != k:
        raise RuntimeError("Unable to construct unique fixed-budget anchors")
    return selected, eligible


def finite_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def compute_restart_aware_gain(rows_b0, rows_b1, gamma):
    """Discount paired AbsRel gains using original future-step indices."""
    discounted_sum = 0.0
    pair_count = 0
    for k, (m0, m1) in enumerate(zip(rows_b0, rows_b1), 1):
        a0 = m0.get("absrel", float("nan"))
        a1 = m1.get("absrel", float("nan"))
        if math.isfinite(float(a0)) and math.isfinite(float(a1)):
            discounted_sum += (gamma ** k) * (float(a0) - float(a1))
            pair_count += 1
    return discounted_sum, pair_count


def self_test():
    rows_b0 = [
        {"absrel": 0.30},
        {"absrel": float("nan")},
        {"absrel": 0.20},
    ]
    rows_b1 = [
        {"absrel": 0.20},
        {"absrel": 0.10},
        {"absrel": 0.30},
    ]
    total, n = compute_restart_aware_gain(rows_b0, rows_b1, 0.9)
    expected = (0.9 * 0.1) + ((0.9 ** 3) * (-0.1))
    assert n == 2 and abs(total - expected) < 1e-12
    assert abs(total / n - expected / 2) < 1e-12
    assert TARGET == "G_state_absrel_restart_aware_discounted_per_valid_frame"
    print("PASS: P2.3 v5.15 self-tests (discount indices, valid-pair denominator, target semantics).")


def same_state_counterfactual(
    model, stream, inputs_reference, total_frames, anchor, state_before,
    records, gt, paddings, padded_shape, horizon, gamma
):
    """B0 vs B1 from one pre-action state; execute each branch's future auto restarts."""
    state_snapshot_source = cf.clone_state(state_before)
    pre_snapshot = capture_state(model, state_snapshot_source)
    result = {
        "anchor_frame": int(anchor),
        "horizon_H": int(horizon),
        "gamma": float(gamma),
        "status": "pending",
        "future_frames_requested": int(horizon),
        "future_metric_pairs": 0,
        "future_frames_with_gt": 0,
        "future_restart_requests_B0": 0,
        "future_restart_requests_B1": 0,
        "future_auto_restarts_executed_B0": 0,
        "future_auto_restarts_executed_B1": 0,
        "G_state_absrel_restart_aware_discounted_sum": float("nan"),
        TARGET: float("nan"),
        "mean_future_absrel_B0": float("nan"),
        "mean_future_absrel_B1": float("nan"),
        "mean_future_rmse_B0": float("nan"),
        "mean_future_rmse_B1": float("nan"),
        "mean_future_delta1_B0": float("nan"),
        "mean_future_delta1_B1": float("nan"),
        "future_protocol": (
            "branch-specific propagation; if a branch requests automatic restart, "
            "execute keyframe_step on the same frame; no future scheduled policy actions "
            "are replayed or recomputed"
        ),
        "positive_gain_means": "B1 persistent geometry writeback lowers restart-aware future AbsRel",
    }
    trace_rows = []
    try:
        # B0 anchor: accept propagation output and propagated state.
        b0_state = cf.clone_state(state_snapshot_source)
        restore_state(model, b0_state, pre_snapshot)
        anchor_inputs = clone_payload(inputs_reference)
        _out_b0_anchor, b0_state, anchor_restart = stream.propagate_step(
            anchor_inputs, anchor, b0_state, total_frames
        )
        if anchor_restart:
            result["status"] = "skipped_anchor_restart"
            result["reason"] = "Anchor propagation requested restart; same as v5.5, this anchor is excluded."
            result["_future_trace_rows"] = trace_rows
            return result

        # Runtime immediately after the shared B0 propagation at the anchor.
        post_prop_snapshot = capture_state(model, b0_state)

        # Compute B1 keyframe candidate from the exact same pre-action state.
        refresh_state = cf.clone_state(state_snapshot_source)
        restore_state(model, refresh_state, pre_snapshot)
        refresh_inputs = clone_payload(inputs_reference)
        _refresh_output, refresh_state = stream.keyframe_step(
            refresh_inputs, anchor, refresh_state, total_frames
        )

        # B1 starts from the same propagated state as B0 and commits only persistent
        # geometry fields; scheduler and fast-tail state remain shared at the anchor.
        b1_state = cf.clone_state(b0_state)
        for field_name in cf.COMMIT_FIELDS:
            value = getattr(refresh_state, field_name)
            if value is None:
                raise RuntimeError(f"Keyframe candidate field {field_name} is None at anchor {anchor}")
            setattr(b1_state, field_name, cf.clone_payload(value))

        # Restore the common post-propagation decoder runtime before snapshotting each
        # branch. From this point onward, each branch keeps its own runtime and state.
        cf.restore_runtime(model, post_prop_snapshot)
        b0_runtime = capture_state(model, b0_state)
        b1_runtime = capture_state(model, b1_state)

        rows_b0, rows_b1 = [], []
        restart_requests_b0 = restart_requests_b1 = 0
        restarts_executed_b0 = restarts_executed_b1 = 0
        gt_count = 0

        def advance_branch(branch_name, branch_state, runtime_snapshot, frame_idx, step_k):
            # Restore this branch's own decoder runtime, propagate one frame, and
            # execute an automatic restart on this same frame when requested.
            cf.restore_runtime(model, runtime_snapshot)
            branch_input = clone_payload(inputs_reference)
            out, next_state, restart_requested = stream.propagate_step(
                branch_input, frame_idx, branch_state, total_frames
            )
            action = "propagation"
            restart_executed = False
            if restart_requested:
                restart_input = clone_payload(inputs_reference)
                out, next_state = stream.keyframe_step(
                    restart_input, frame_idx, next_state, total_frames
                )
                action = "auto_restart"
                restart_executed = True
            metric = cf.eval_output(out, model, paddings, padded_shape, gt[frame_idx])
            next_runtime = capture_state(model, next_state)
            trace_rows.append({
                "anchor_frame": int(anchor),
                "future_frame": int(frame_idx),
                "step_k": int(step_k),
                "branch": branch_name,
                "restart_requested": bool(restart_requested),
                "restart_executed": bool(restart_executed),
                "executed_action": action,
                "gt_available": bool(records[frame_idx]["gt_available"]),
                "absrel": metric.get("absrel", float("nan")),
                "rmse": metric.get("rmse", float("nan")),
                "delta1": metric.get("delta1", float("nan")),
                "valid_ratio": metric.get("valid_ratio", float("nan")),
            })
            return next_state, next_runtime, metric, bool(restart_requested), bool(restart_executed)

        for k in range(1, horizon + 1):
            frame_idx = anchor + k
            if frame_idx >= total_frames:
                raise RuntimeError(
                    f"Anchor {anchor} lacks complete H={horizon} future window in T={total_frames}"
                )

            b1_state, b1_runtime, m1, req1, exec1 = advance_branch(
                "B1_state_writeback", b1_state, b1_runtime, frame_idx, k
            )
            b0_state, b0_runtime, m0, req0, exec0 = advance_branch(
                "B0_propagation", b0_state, b0_runtime, frame_idx, k
            )
            rows_b0.append(m0)
            rows_b1.append(m1)
            gt_count += int(bool(records[frame_idx]["gt_available"]))
            restart_requests_b0 += int(req0)
            restart_requests_b1 += int(req1)
            restarts_executed_b0 += int(exec0)
            restarts_executed_b1 += int(exec1)

        discounted_sum, pair_count = compute_restart_aware_gain(rows_b0, rows_b1, gamma)
        result.update({
            "status": "ok" if pair_count else "no_valid_metric_pairs",
            "future_metric_pairs": int(pair_count),
            "future_frames_with_gt": int(gt_count),
            "future_restart_requests_B0": int(restart_requests_b0),
            "future_restart_requests_B1": int(restart_requests_b1),
            "future_auto_restarts_executed_B0": int(restarts_executed_b0),
            "future_auto_restarts_executed_B1": int(restarts_executed_b1),
            "G_state_absrel_restart_aware_discounted_sum": float(discounted_sum) if pair_count else float("nan"),
            TARGET: float(discounted_sum / pair_count) if pair_count else float("nan"),
            "mean_future_absrel_B0": finite_mean([r["absrel"] for r in rows_b0]),
            "mean_future_absrel_B1": finite_mean([r["absrel"] for r in rows_b1]),
            "mean_future_rmse_B0": finite_mean([r["rmse"] for r in rows_b0]),
            "mean_future_rmse_B1": finite_mean([r["rmse"] for r in rows_b1]),
            "mean_future_delta1_B0": finite_mean([r["delta1"] for r in rows_b0]),
            "mean_future_delta1_B1": finite_mean([r["delta1"] for r in rows_b1]),
            "_future_trace_rows": trace_rows,
        })
        return result
    finally:
        # Restore the real policy trajectory before its recorded action is executed.
        restore_state(model, state_before, pre_snapshot)


def execute_logged_action(model, stream, inputs_reference, state, t, total_frames, row, scheduled_action):
    """Replay the action recorded by v5.4, without recalculating its decision."""
    requested = bool(row.get("scheduled_action_requested", False))
    if t == 0:
        out, state = stream.keyframe_step(clone_payload(inputs_reference), t, state, total_frames)
        action = "initial_keyframe"
        return out, state, action

    if requested and scheduled_action == "state_writeback":
        pre_state = cf.clone_state(state)
        pre_snapshot = capture_state(model, pre_state)

        propagated_state = cf.clone_state(pre_state)
        restore_state(model, propagated_state, pre_snapshot)
        out, propagated_state, restart = stream.propagate_step(
            clone_payload(inputs_reference), t, propagated_state, total_frames
        )
        if restart:
            out, state = stream.keyframe_step(
                clone_payload(inputs_reference), t, propagated_state, total_frames
            )
            return out, state, "auto_restart"

        post_prop_snapshot = capture_state(model, propagated_state)
        refresh_state = cf.clone_state(pre_state)
        restore_state(model, refresh_state, pre_snapshot)
        _refresh_out, refresh_state = stream.keyframe_step(
            clone_payload(inputs_reference), t, refresh_state, total_frames
        )
        cf.restore_runtime(model, post_prop_snapshot)
        for field_name in cf.COMMIT_FIELDS:
            value = getattr(refresh_state, field_name)
            if value is None:
                raise RuntimeError(f"Keyframe candidate field {field_name} is None at frame {t}")
            setattr(propagated_state, field_name, cf.clone_payload(value))
        return out, propagated_state, "state_writeback"

    if requested:
        state.first_keyframe = True
        out, state = stream.keyframe_step(
            clone_payload(inputs_reference), t, state, total_frames
        )
        state.fast_tail_radius = state.radius_teacher.detach().clone()
        state.fast_tail_confidence = state.confidence_teacher.detach().clone()
        state.fast_tail_features = [x.detach().clone() for x in state.features_teacher]
        state.first_keyframe = False
        return out, state, "policy_refresh"

    out, state, restart = stream.propagate_step(
        clone_payload(inputs_reference), t, state, total_frames
    )
    if restart:
        out, state = stream.keyframe_step(
            clone_payload(inputs_reference), t, state, total_frames
        )
        return out, state, "auto_restart"
    return out, state, "propagation"


def replay_trajectory_with_counterfactuals(
    policy, model, base_snapshot, inputs, metas, frames, records, gt,
    paddings, padded_shape, args, logged_rows, predictors
):
    """Replay one policy's recorded actions while evaluating same-state forks."""
    inputs_reference = clone_payload(inputs)
    total_frames = int(metas[0]["T"])
    warmup_policy(model, base_snapshot, inputs_reference, total_frames)

    state = StreamState()
    restore_state(model, state, base_snapshot)
    if "rays" in inputs_reference:
        model.pixel_decoder.test_gt_camera = True
    stream = VeloDepthStream(model)

    anchor_rows = []
    alignment_rows = []
    actual_rows = []
    future_trace_rows = []
    for t, logged in enumerate(logged_rows):
        if t >= args.frames:
            break
        is_candidate = (
            t >= args.candidate_offset
            and (t - args.candidate_offset) % args.candidate_stride == 0
        )
        full_horizon = t + args.horizon < total_frames
        if is_candidate and full_horizon:
            # Build features and scores from the exact pre-action state of this
            # policy trajectory, not from a separately generated P2.2.2 trajectory.
            features = build_online_features(
                state, frames[t], frames[t - 1] if t > 0 else None
            )
            scores = {
                name: predictor.predict(features)
                for name, predictor in predictors.items()
            }
            pre_state_for_label = cf.clone_state(state)
            cf_result = same_state_counterfactual(
                model=model,
                stream=stream,
                inputs_reference=inputs_reference,
                total_frames=total_frames,
                anchor=t,
                state_before=pre_state_for_label,
                records=records,
                gt=gt,
                paddings=paddings,
                padded_shape=padded_shape,
                horizon=args.horizon,
                gamma=args.gamma,
            )
            # Remove per-future-frame trace from the aggregate anchor row.
            trace = cf_result.pop("_future_trace_rows", [])
            for trace_row in trace:
                trace_row["sequence"] = args.sequence
                trace_row["trajectory_policy"] = policy
                future_trace_rows.append(trace_row)
            # same_state_counterfactual restores the actual online state/runtime.
            item = {
                "sequence": args.sequence,
                "trajectory_policy": policy,
                "anchor_frame": int(t),
                "logged_action_at_anchor": str(logged.get("action", "")),
                "scheduled_action_requested": bool(logged.get("scheduled_action_requested", False)),
                "gt_available_at_anchor": bool(records[t]["gt_available"]),
                "anchor_gt_time_diff_ms": (
                    records[t]["time_diff"] * 1000.0
                    if math.isfinite(records[t]["time_diff"]) else float("nan")
                ),
                "feature_scheduler_distance_kf": features.get("scheduler_distance_kf", float("nan")),
                "feature_scheduler_flow_magnitude": features.get("scheduler_flow_magnitude", float("nan")),
                "feature_scheduler_total_luma_change": features.get("scheduler_total_luma_change", float("nan")),
                "feature_state_radius_teacher_mean": features.get("state_radius_teacher_mean", float("nan")),
                "feature_state_confidence_teacher_mean": features.get("state_confidence_teacher_mean", float("nan")),
                "feature_state_features_teacher_0_mean": features.get("state_features_teacher_0_mean", float("nan")),
                "feature_state_features_teacher_1_mean": features.get("state_features_teacher_1_mean", float("nan")),
                "feature_state_features_teacher_2_mean": features.get("state_features_teacher_2_mean", float("nan")),
                "feature_state_features_teacher_3_mean": features.get("state_features_teacher_3_mean", float("nan")),
                "score_state_only": scores.get("causal_budget_state_only", float("nan")),
                "score_motion_plus_learned": scores.get("causal_budget_motion_plus_learned", float("nan")),
            }
            item.update(cf_result)
            anchor_rows.append(item)

        # Replay the actual logged action after the counterfactual fork. This action
        # is the same action chosen by the first v5.4 replay; no label enters policy.
        sync_cuda()
        out, state, replayed_action = execute_logged_action(
            model, stream, inputs_reference, state, t, total_frames, logged,
            args.scheduled_action
        )
        m = cf.eval_output(out, model, paddings, padded_shape, gt[t])
        logged_absrel = float(logged.get("absrel", float("nan")))
        replay_absrel = float(m.get("absrel", float("nan")))
        absrel_diff = (
            replay_absrel - logged_absrel
            if math.isfinite(replay_absrel) and math.isfinite(logged_absrel)
            else float("nan")
        )
        logged_action = str(logged.get("action", ""))
        alignment_rows.append({
            "sequence": args.sequence,
            "policy": policy,
            "frame_index": int(t),
            "logged_action": logged_action,
            "replayed_action": replayed_action,
            "action_match": bool(logged_action == replayed_action),
            "logged_absrel": logged_absrel,
            "replayed_absrel": replay_absrel,
            "absrel_diff_replayed_minus_logged": absrel_diff,
            "logged_gt_available": bool(logged.get("gt_available", False)),
            "replayed_gt_available": bool(records[t]["gt_available"]),
        })
        actual_rows.append({
            "sequence": args.sequence,
            "policy": policy,
            "frame_index": int(t),
            "action": replayed_action,
            "absrel": replay_absrel,
            "rmse": m.get("rmse", float("nan")),
            "delta1": m.get("delta1", float("nan")),
            "valid_ratio": m.get("valid_ratio", float("nan")),
        })
    return anchor_rows, alignment_rows, actual_rows, future_trace_rows


def make_diagnostics(anchor_rows):
    rows = []
    df = pd.DataFrame(anchor_rows)
    if df.empty:
        return rows
    for (seq, trajectory), sub in df.groupby(["sequence", "trajectory_policy"], dropna=False):
        for predictor_name, score_col in [
            ("causal_budget_state_only", "score_state_only"),
            ("causal_budget_motion_plus_learned", "score_motion_plus_learned"),
        ]:
            valid = sub[
                np.isfinite(pd.to_numeric(sub[score_col], errors="coerce"))
                & np.isfinite(pd.to_numeric(sub[TARGET], errors="coerce"))
                & (sub["status"].astype(str) == "ok")
            ].copy()
            n = len(valid)
            if n:
                scores = pd.to_numeric(valid[score_col], errors="coerce").to_numpy(float)
                targets = pd.to_numeric(valid[TARGET], errors="coerce").to_numpy(float)
                pearson = float(pd.Series(scores).corr(pd.Series(targets), method="pearson")) if n >= 2 else float("nan")
                spearman = float(pd.Series(scores).corr(pd.Series(targets), method="spearman")) if n >= 2 else float("nan")
                top_n = max(1, int(math.ceil(0.25 * n)))
                order = np.argsort(scores)[::-1]
                top_targets = targets[order[:top_n]]
                overall_mean = float(np.mean(targets))
                selected_mean = float(np.mean(top_targets))
                top_positive = float(np.mean(top_targets > 0))
                overall_positive = float(np.mean(targets > 0))
                lift = selected_mean - overall_mean
                positive_rate_lift = top_positive - overall_positive
            else:
                pearson = spearman = overall_mean = selected_mean = lift = float("nan")
                top_n = 0
                top_positive = overall_positive = positive_rate_lift = float("nan")
            rows.append({
                "sequence": seq,
                "trajectory_policy": trajectory,
                "predictor": predictor_name,
                "valid_anchor_count": int(n),
                "pearson_score_vs_same_state_gain": pearson,
                "spearman_score_vs_same_state_gain": spearman,
                "mean_gain_all_anchors": overall_mean,
                "mean_gain_top_score_quartile": selected_mean,
                "top_quartile_gain_lift": lift,
                "positive_gain_rate_all": overall_positive,
                "positive_gain_rate_top_quartile": top_positive,
                "positive_gain_rate_lift": positive_rate_lift,
                "top_quartile_n": int(top_n),
                "target_definition": TARGET,
            })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="P2.3 v5.15 same-state counterfactual gain diagnostics"
    )
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--sequence", choices=("crowd2", "balloon2", "crowd3"), default=None)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--train-dataset", default="outputs/p2_2_2/p2_2_2_valid_dataset.csv")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--candidate-stride", type=int, default=4)
    ap.add_argument("--candidate-offset", type=int, default=1)
    ap.add_argument("--min-refresh-gap", type=int, default=4)
    ap.add_argument("--fixed-interval", type=int, default=8)
    ap.add_argument("--budget-actions", type=int, default=8)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--target-refresh-fraction", type=float, default=3.0 / 7.0)
    ap.add_argument("--policies", default="auto,budget_fixed,causal_budget_state_only,causal_budget_motion_plus_learned")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.data_root or not args.sequence or not args.out_dir:
        ap.error("--data-root, --sequence, and --out-dir are required unless --self-test is used")

    load_replay_dependencies()

    if args.frames <= args.horizon + args.candidate_offset:
        ap.error("frames must exceed horizon + candidate offset to provide a full future window")
    if args.horizon != 16:
        ap.error("P2.3 v5.15 currently fixes H=16 to match the project's primary target")
    if not (0 < args.gamma <= 1):
        ap.error("--gamma must be in (0,1]")
    if args.candidate_stride < args.min_refresh_gap:
        ap.error("candidate-stride must be >= min-refresh-gap for budget controller compatibility")
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if policies != [p for p in POLICIES if p in policies] or set(policies) - set(POLICIES):
        ap.error(f"--policies must be an ordered subset of {POLICIES}")
    if not set(PREDICTOR_NAMES).issubset(policies):
        ap.error("Both causal predictor policies must be included to produce same-state score diagnostics")

    fixed_frames, eligible_frames = make_fixed_budget_frames(
        args.frames, args.candidate_offset, args.candidate_stride,
        args.min_refresh_gap, args.budget_actions
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run on the AutoDL GPU instance, not the no-card instance.")

    root = Path(args.data_root)
    rgb_index = cf.read_index(root, "rgb.txt", "rgb")
    depth_index = cf.read_index(root, "depth.txt", "depth")
    selected = rgb_index[:args.frames]
    if len(selected) != args.frames:
        raise RuntimeError(f"Requested {args.frames} RGB frames but found {len(selected)} in {root}")
    records = match_gt_tolerant(selected, depth_index, args.max_time_diff)
    frames = cf.load_frames(records)
    gt = [
        cf.load_gt(r["gt_path"], args.gt_scale) if r["gt_available"] else None
        for r in records
    ]
    matched = sum(bool(r["gt_available"]) for r in records)
    print(f"Sequence={args.sequence}; frames={len(frames)}; GT matches={matched}/{len(records)}")

    torch.set_grad_enabled(False)
    print("Loading frozen VeloDepth weights...")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(torch.device("cuda"))
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    print("Capturing packed inputs/flow; aborting neural inference at encode_decode_test boundary...")
    inputs, metas = capture_packed_inputs_without_full_inference(model, frames)
    total_frames = int(metas[0]["T"])
    if total_frames != args.frames:
        raise RuntimeError(f"Packed frame count {total_frames} != requested {args.frames}")
    H, W = int(frames.shape[-2]), int(frames.shape[-1])
    paddings, (padded_h, padded_w) = replay.get_paddings(
        (H, W), model.shape_constraints["ratio_bounds"]
    )
    padded_shape = (padded_h, padded_w)
    base_snapshot = capture_state(model, StreamState())
    if "rays" in inputs:
        model.pixel_decoder.test_gt_camera = True

    predictors = {}
    for name in PREDICTOR_NAMES:
        predictors[name] = OnlineRidge(
            args.train_dataset, args.sequence, name,
            horizon=args.horizon, alpha=args.ridge_alpha,
            target_refresh_fraction=args.target_refresh_fraction,
        )
        p = predictors[name]
        print(
            f"Predictor {name}: train_n={p.train_n}; OOF_n={p.oof_n}; "
            f"threshold={p.calibrated_threshold:.8f}; OOF_rate={p.oof_achieved_fraction:.3f}"
        )

    # Namespace fields consumed by the existing v5.4 replay harness.
    replay_args = argparse.Namespace(
        sequence=args.sequence,
        frames=args.frames,
        fixed_interval=args.fixed_interval,
        min_refresh_gap=args.min_refresh_gap,
        candidate_stride=args.candidate_stride,
        candidate_offset=args.candidate_offset,
        allowed_action_frame_set=None,
        forced_action_frame_set=set(),
        scheduled_action="state_writeback",
        threshold_mode="train_oof_budget",
        threshold=0.0,
        target_refresh_fraction=args.target_refresh_fraction,
        horizon=args.horizon,
        gamma=args.gamma,
        ridge_alpha=args.ridge_alpha,
        budget_actions=args.budget_actions,
        fixed_budget_frame_set=fixed_frames,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_anchor_rows, all_alignment_rows, all_policy_rows, all_policy_summaries = [], [], [], []
    all_future_trace_rows = []
    calibration_rows, calibration_fold_rows = [], []

    for policy in policies:
        predictor = predictors.get(policy)
        print(f"\n===== P2.3 v5.15 ONLINE TRAJECTORY: {policy} =====")
        logged_rows = replay.run_policy(
            policy, model, base_snapshot, inputs, metas, frames, records, gt,
            paddings, padded_shape, args.sequence, replay_args, predictor
        )
        all_policy_rows.extend(logged_rows)
        all_policy_summaries.extend(summarize(logged_rows))
        if predictor is not None:
            calibration_rows.extend(predictor.oof_rows)
            calibration_fold_rows.extend(predictor.fold_rows)

        print(f"===== P2.3 v5.15 SAME-STATE COUNTERFACTUALS: {policy} =====")
        anchors, alignment, actual, future_trace = replay_trajectory_with_counterfactuals(
            policy, model, base_snapshot, inputs, metas, frames, records, gt,
            paddings, padded_shape, replay_args, logged_rows, predictors
        )
        all_anchor_rows.extend(anchors)
        all_alignment_rows.extend(alignment)
        all_future_trace_rows.extend(future_trace)
        all_policy_rows.extend([])

        action_mismatches = sum(not r["action_match"] for r in alignment)
        finite_diffs = [
            abs(float(r["absrel_diff_replayed_minus_logged"]))
            for r in alignment if math.isfinite(float(r["absrel_diff_replayed_minus_logged"]))
        ]
        max_absrel_diff = max(finite_diffs) if finite_diffs else float("nan")
        print(
            f"ALIGNMENT {policy}: action_mismatches={action_mismatches}/{len(alignment)}; "
            f"max_absrel_diff={max_absrel_diff:.8g}"
        )

    diagnostics = make_diagnostics(all_anchor_rows)
    frame_csv = out_dir / "p2_3_v5_15_online_frame_metrics.csv"
    summary_csv = out_dir / "p2_3_v5_15_policy_summary.csv"
    anchor_csv = out_dir / "p2_3_v5_15_same_state_anchor_labels.csv"
    diagnostic_csv = out_dir / "p2_3_v5_15_gain_diagnostics.csv"
    alignment_csv = out_dir / "p2_3_v5_15_trajectory_alignment.csv"
    calibration_csv = out_dir / "p2_3_v5_15_threshold_calibration_oof.csv"
    calibration_folds_csv = out_dir / "p2_3_v5_15_threshold_calibration_folds.csv"
    future_trace_csv = out_dir / "p2_3_v5_15_future_restart_trace.csv"
    metadata_json = out_dir / "p2_3_v5_15_metadata.json"

    write_csv(frame_csv, all_policy_rows)
    write_csv(summary_csv, all_policy_summaries)
    write_csv(anchor_csv, all_anchor_rows)
    write_csv(diagnostic_csv, diagnostics)
    write_csv(alignment_csv, all_alignment_rows)
    write_csv(calibration_csv, calibration_rows)
    write_csv(calibration_folds_csv, calibration_fold_rows)
    write_csv(future_trace_csv, all_future_trace_rows)

    metadata = {
        "experiment": "P2.3 v5.15 same-state counterfactual gain diagnosis",
        "sequence": args.sequence,
        "data_root": str(root),
        "frames": int(args.frames),
        "gt_matches": int(matched),
        "horizon_H": int(args.horizon),
        "gamma": float(args.gamma),
        "target": TARGET,
        "positive_target_means": "B1 persistent geometry state writeback lowers restart-aware future AbsRel relative to B0 propagation",
        "model_weights_frozen": True,
        "predictor_training": "Ridge predictors trained on P2.2.2 H=16 rows from the other two sequences; VeloDepth weights are not trained",
        "state_alignment": "At each candidate anchor, save the pre-action state from the exact policy trajectory; fork B0 and B1 from that same state snapshot; restore the actual trajectory before executing its logged action",
        "B0": "propagate at anchor; no persistent geometry fields written back",
        "B1": "propagate at anchor and retain its output; compute full keyframe candidate from same pre-action snapshot; commit only COMMIT_FIELDS",
        "B1_commit_fields": list(cf.COMMIT_FIELDS),
        "future_protocol": "branch-specific future propagation; each branch executes its own automatic restart on the same frame when requested; future scheduled policy actions are not replayed or recomputed; anchor restart requests cause anchor skip",
        "target_formula": "sum_{k=1..H, finite paired AbsRel} gamma**k*(AbsRel_B0[k]-AbsRel_B1[k]) / number_of_finite_metric_pairs; missing pairs do not renumber discount exponent",
        "diagnostic_only_future_labels": True,
        "future_labels_enter_online_action_decisions": False,
        "same_state_labels_are_per_policy_trajectory": True,
        "policies": policies,
        "pre_specified_budget_K": int(args.budget_actions),
        "budget_fixed_scheduled_frames": sorted(int(x) for x in fixed_frames),
        "eligible_budget_anchors": [int(x) for x in eligible_frames],
        "candidate_stride": int(args.candidate_stride),
        "candidate_offset_zero_based": int(args.candidate_offset),
        "min_refresh_gap": int(args.min_refresh_gap),
        "trajectory_replay_validation": "action sequence and frame AbsRel are compared against original v5.4 replay; inspect trajectory_alignment CSV before interpreting labels",
        "timing_scope": "diagnostic branch replay is not a latency benchmark and does not claim compute savings",
        "comparison_target": "legacy v5.5 forced-propagation labels; old predictors remain trained on P2.2.2 forced-propagation targets",
        "restart_aware_scope_limit": "automatic restarts are executed independently in B0/B1 branches; future scheduled policy actions are not replayed or recomputed, so this is not full online-policy gain",
        "outputs": {
            "online_frame_metrics": str(frame_csv),
            "policy_summary": str(summary_csv),
            "same_state_anchor_labels": str(anchor_csv),
            "gain_diagnostics": str(diagnostic_csv),
            "trajectory_alignment": str(alignment_csv),
            "threshold_calibration_oof": str(calibration_csv),
            "threshold_calibration_folds": str(calibration_folds_csv),
            "future_restart_trace": str(future_trace_csv),
        },
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== P2.3 v5.15 POLICY SUMMARY =====")
    print(pd.DataFrame(all_policy_summaries)[[
        "policy", "frames_with_valid_metrics", "mean_absrel", "mean_rmse",
        "mean_delta1", "state_writeback_count", "auto_restart_count",
        "mean_model_action_ms", "mean_decision_ms"
    ]].to_string(index=False))
    print("\n===== P2.3 v5.15 SAME-STATE GAIN DIAGNOSTICS =====")
    if diagnostics:
        print(pd.DataFrame(diagnostics)[[
            "trajectory_policy", "predictor", "valid_anchor_count",
            "pearson_score_vs_same_state_gain", "spearman_score_vs_same_state_gain",
            "top_quartile_gain_lift", "positive_gain_rate_lift"
        ]].to_string(index=False))
    print("\nWrote:", frame_csv)
    print("Wrote:", summary_csv)
    print("Wrote:", anchor_csv)
    print("Wrote:", diagnostic_csv)
    print("Wrote:", alignment_csv)
    print("Wrote:", metadata_json)
    print("IMPORTANT: verify trajectory_alignment.csv; no model weights were updated.")


if __name__ == "__main__":
    main()
