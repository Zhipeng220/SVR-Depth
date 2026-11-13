#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2-S2: persistent-state field ablation for SVR-Depth.

Frozen weights. This is a mechanism diagnostic, not a learned method.

At each fixed anchor:
  PROP / FULL_DONOR_FAST / NATIVE_FULL_REFRESH: current-frame references.
  OUTPUT_ONLY: same donor current output; persistent state remains propagated.
  NOOP_COMMIT: explicitly reassign the original six state/cache fields unchanged.
  FEATURE_ONLY: commit donor features_teacher + fast_tail_features.
  RADIUS_ONLY: commit donor radius_teacher + fast_tail_radius.
  CONFIDENCE_ONLY: commit donor confidence_teacher + fast_tail_confidence.
  FULL_COMMIT: commit all six donor fields.

All future branches start from the same propagated anchor state, use the same
current donor output conceptually, restore the same decoder runtime snapshot
before each frame, and synchronize scheduler fields to OUTPUT_ONLY. Restart
requests are recorded but ignored so every branch uses a fixed propagation
schedule. This does NOT claim that the full-frame teacher donor is cheap.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

import p1_multihorizon_counterfactual as mh
from stream_state import StreamState, clone_state
from stream_state_snapshot import capture_state, restore_state
from velodepth_stream import VeloDepthStream, StepOutput
from velodepth.models.velodepth import VeloDepth, get_paddings


SCHEDULER_FIELDS = (
    "current_t", "distance_kf", "flow_magnitude",
    "total_luma_change", "first_keyframe",
)
STATE_CACHE_FIELDS = (
    "features_teacher", "radius_teacher", "confidence_teacher",
    "fast_tail_features", "fast_tail_radius", "fast_tail_confidence",
)
FIELD_GROUPS = {
    "FEATURE_ONLY": ("features_teacher", "fast_tail_features"),
    "RADIUS_ONLY": ("radius_teacher", "fast_tail_radius"),
    "CONFIDENCE_ONLY": ("confidence_teacher", "fast_tail_confidence"),
    "FULL_COMMIT": STATE_CACHE_FIELDS,
}
BRANCHES = (
    "OUTPUT_ONLY", "NOOP_COMMIT", "FEATURE_ONLY",
    "RADIUS_ONLY", "CONFIDENCE_ONLY", "FULL_COMMIT",
)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clone_value(v):
    if isinstance(v, torch.Tensor):
        return v.detach().clone()
    if isinstance(v, list):
        return [clone_value(x) for x in v]
    if isinstance(v, tuple):
        return tuple(clone_value(x) for x in v)
    if isinstance(v, dict):
        return {k: clone_value(x) for k, x in v.items()}
    return v


def max_tensor_diff(a, b):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return 0.0 if a is b else float("inf")
    if tuple(a.shape) != tuple(b.shape):
        return float("inf")
    aa, bb = a.detach().float(), b.detach().float()
    nan_a, nan_b = torch.isnan(aa), torch.isnan(bb)
    if not torch.equal(nan_a, nan_b):
        return float("inf")
    inf_a, inf_b = torch.isinf(aa), torch.isinf(bb)
    if not torch.equal(inf_a, inf_b):
        return float("inf")
    if bool((inf_a & (aa != bb)).any()):
        return float("inf")
    finite = torch.isfinite(aa) & torch.isfinite(bb)
    if not bool(finite.any()):
        return 0.0
    return float((aa[finite] - bb[finite]).abs().max().item())


def max_nested_diff(a, b):
    if a is None or b is None:
        return 0.0 if a is None and b is None else float("inf")
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return max_tensor_diff(a, b)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((max_nested_diff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max((max_nested_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (bool, int, float, str)) and isinstance(b, (bool, int, float, str)):
        if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
            return 0.0 if a == b else float("inf")
        return float(abs(float(a) - float(b)))
    return 0.0 if a == b else float("inf")


def max_state_diff(a, b):
    return max(
        (max_nested_diff(getattr(a, f.name), getattr(b, f.name))
         for f in fields(StreamState)),
        default=0.0,
    )


def max_output_diff(a, b):
    return max(
        max_tensor_diff(a.radius, b.radius),
        max_tensor_diff(a.rays, b.rays),
        max_tensor_diff(a.confidence, b.confidence),
    )


def restore_runtime(model, snapshot):
    # Restore decoder runtime attributes only; the caller's branch state remains independent.
    temp = clone_state(snapshot.stream_state)
    restore_state(model, temp, snapshot)


def sync_scheduler(dst, src):
    for name in SCHEDULER_FIELDS:
        setattr(dst, name, clone_value(getattr(src, name)))


def build_teacher_conditioned_features(model, inputs, stream, frame_idx, total_frames, rays_base):
    """Generate a full-frame teacher-conditioned donor in the current ray field."""
    image = stream._frame(inputs["image"], frame_idx, total_frames)
    features, tokens = model.pixel_encoder(image)
    features = [
        model.stacking_fn(features[i:j]).contiguous()
        for i, j in model.slices_encoder_range
    ]
    tokens = [
        model.stacking_fn(tokens[i:j]).contiguous()
        for i, j in model.slices_encoder_range
    ]
    metas = [{"T": 1, "B": 1}]
    base_inputs = {"image": image, "features": features, "tokens": tokens}
    model.pixel_decoder.set_shapes(base_inputs, metas)
    adapted_features = model.pixel_decoder.input_adapter(features)
    radial = model.pixel_decoder.radial_module
    ray_embedding = radial.embed_rays(rays_base)
    conditioned = radial.condition(adapted_features, ray_embedding)
    h_common, w_common = radial.shapes
    return [
        x.reshape(x.shape[0], h_common, w_common, x.shape[-1])
         .permute(0, 3, 1, 2).contiguous()
        for x in conditioned
    ]


def metrics_row(anchor, frame_idx, branch, out, model, paddings, padded_shape, gt, records):
    m = mh.eval_output(out, model, paddings, padded_shape, gt)
    dt = records[frame_idx]["time_diff"]
    return {
        "anchor_frame": anchor,
        "frame_index": frame_idx,
        "step_from_anchor": frame_idx - anchor,
        "branch": branch,
        "rgb_timestamp": records[frame_idx]["rgb_ts"],
        "gt_available": bool(records[frame_idx]["gt_available"]),
        "gt_time_diff_ms": dt * 1000.0 if math.isfinite(dt) else float("nan"),
        "absrel": m["absrel"],
        "rmse": m["rmse"],
        "delta1": m["delta1"],
        "valid_ratio": m["valid_ratio"],
    }


def save_prediction(out_dir, anchor, frame_idx, branch, out, model, paddings, padded_shape, gt):
    if gt is None:
        return
    pred = mh.output_depth(out, model, paddings, padded_shape).astype(np.float32)
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    path = Path(out_dir) / "predictions" / f"a{anchor:03d}_f{frame_idx:03d}_{branch}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pred=pred, gt=gt.astype(np.float32), valid=valid)


def forced_propagation_step(stream, inputs, frame_idx, state, total_frames, runtime_snapshot, model):
    """Force a common propagation schedule; log restart signals but do not run keyframes."""
    restore_runtime(model, runtime_snapshot)
    out, state, restart = stream.propagate_step(inputs, frame_idx, state, total_frames)
    if restart:
        state.first_keyframe = False
        state.distance_kf += 1
        state.current_t = frame_idx + 1
    return out, state, bool(restart)


def mean_metric(rows, branch, key, max_step):
    vals = [
        float(r[key]) for r in rows
        if r["branch"] == branch
        and 1 <= int(r["step_from_anchor"]) <= max_step
        and math.isfinite(float(r[key]))
    ]
    return sum(vals) / len(vals) if vals else float("nan")


def paired_gain(rows, branch, key, max_step, higher_is_better=False):
    ref_by_frame = {
        int(r["frame_index"]): r for r in rows
        if r["branch"] == "OUTPUT_ONLY"
        and 1 <= int(r["step_from_anchor"]) <= max_step
    }
    branch_by_frame = {
        int(r["frame_index"]): r for r in rows
        if r["branch"] == branch
        and 1 <= int(r["step_from_anchor"]) <= max_step
    }
    vals = []
    for frame in sorted(set(ref_by_frame) & set(branch_by_frame)):
        a = float(ref_by_frame[frame][key])
        b = float(branch_by_frame[frame][key])
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        vals.append((b - a) if higher_is_better else (a - b))
    return (sum(vals) / len(vals), len(vals)) if vals else (float("nan"), 0)


def assign_group(state, baseline_state, donor_values, group):
    """Apply a predeclared field group to a branch state."""
    if group == "NOOP_COMMIT":
        fields_to_write = STATE_CACHE_FIELDS
        source = {name: getattr(baseline_state, name) for name in STATE_CACHE_FIELDS}
    elif group == "OUTPUT_ONLY":
        return
    else:
        fields_to_write = FIELD_GROUPS[group]
        source = donor_values
    for name in fields_to_write:
        setattr(state, name, clone_value(source[name]))


def main():
    ap = argparse.ArgumentParser(description="E2-S2 persistent-state field ablation")
    ap.add_argument("--data-root", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2")
    ap.add_argument("--frames", type=int, default=46)
    ap.add_argument("--anchors", default="9,37", help="Zero-based fixed development anchors")
    ap.add_argument("--horizons", default="1,4,8")
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument(
        "--out-dir",
        default="outputs/e2_s2_field_ablation/crowd2_a9_a37_run01",
    )
    args = ap.parse_args()

    anchors = mh.parse_ints(args.anchors, "--anchors", 1)
    horizons = mh.parse_ints(args.horizons, "--horizons", 1)
    if args.frames < max(anchors) + max(horizons) + 1:
        ap.error(f"--frames must include anchor {max(anchors)} plus horizon {max(horizons)}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. Run E2-S2 on the GPU instance, not no-card mode.")

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {out_dir}. Choose a new --out-dir to preserve prior runs."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    root = Path(args.data_root)
    rgb_rows = mh.read_index(root, "rgb.txt", "rgb")
    depth_rows = mh.read_index(root, "depth.txt", "depth")
    selected = rgb_rows[:args.frames]
    if len(selected) != args.frames:
        raise RuntimeError(f"Requested {args.frames} RGB frames, found {len(selected)}")
    records = mh.match_gt(selected, depth_rows, args.max_time_diff)
    gt = [
        mh.load_gt(row["gt_path"], args.gt_scale) if row["gt_available"] else None
        for row in records
    ]

    print("=== E2-S2 persistent-state field ablation ===")
    print(f"Frames={args.frames}; anchors={anchors}; horizons={horizons}")
    print(f"GT matched={sum(bool(r['gt_available']) for r in records)}/{len(records)}")
    print("Weights frozen. No training. Full-frame donor is an oracle diagnostic.")
    print("Field groups: NOOP_COMMIT, FEATURE_ONLY, RADIUS_ONLY, CONFIDENCE_ONLY, FULL_COMMIT.")

    frames = mh.load_frames(records)
    raw_h, raw_w = int(frames.shape[-2]), int(frames.shape[-1])
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(torch.device("cuda"))
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"
    inputs, metas = mh.capture_inputs(model, frames)
    total_frames = int(metas[0]["T"])
    paddings, padded_shape = get_paddings(
        (raw_h, raw_w), model.shape_constraints["ratio_bounds"]
    )
    stream = VeloDepthStream(model)
    initial_snapshot = capture_state(model, StreamState())

    current_rows, future_rows, noop_rows, status_rows, timing_rows = [], [], [], [], []
    anchor_summaries = []

    for anchor in anchors:
        print(f"\n--- anchor A{anchor} ---")
        state_before = mh.prefix_state(
            model, stream, inputs, total_frames, anchor, initial_snapshot
        )
        pre_anchor_snapshot = capture_state(model, state_before)

        prop_state = clone_state(state_before)
        restore_state(model, prop_state, pre_anchor_snapshot)
        with torch.inference_mode():
            prop_out, prop_state, restart_anchor = stream.propagate_step(
                inputs, anchor, prop_state, total_frames
            )
        if restart_anchor:
            status_rows.append({
                "anchor_frame": anchor,
                "status": "skipped_anchor_restart",
                "reason": "Native propagation requests a restart at the anchor; predeclared rule skips it.",
            })
            print(f"SKIP A{anchor}: propagation requests restart at anchor")
            continue
        post_prop_runtime = capture_state(model, prop_state)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            donor_features = build_teacher_conditioned_features(
                model, inputs, stream, anchor, total_frames, prop_state.rays_base
            )
        torch.cuda.synchronize()
        donor_generation_ms = (time.perf_counter() - t0) * 1000.0

        for level, (base, donor) in enumerate(zip(prop_state.features_teacher, donor_features)):
            if tuple(base.shape) != tuple(donor.shape):
                raise RuntimeError(
                    f"A{anchor}: donor shape mismatch at level {level}: {tuple(base.shape)} vs {tuple(donor.shape)}"
                )
        if len(prop_state.features_teacher) != len(donor_features):
            raise RuntimeError(f"A{anchor}: donor feature-level count mismatch")

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            donor_radius, donor_confidence = model.pixel_decoder.readout_fast_features(donor_features)
        torch.cuda.synchronize()
        donor_readout_ms = (time.perf_counter() - t0) * 1000.0

        donor_out = StepOutput(
            radius=donor_radius,
            rays=prop_out.rays.detach().clone(),
            confidence=donor_confidence,
            kind="full_donor_fast",
            frame=anchor,
        )

        # Native full refresh is a current-frame reference only; it is not paired as the same state trajectory.
        native_state = clone_state(state_before)
        restore_state(model, native_state, pre_anchor_snapshot)
        with torch.inference_mode():
            native_out, native_state = stream.keyframe_step(
                inputs, anchor, native_state, total_frames
            )

        for branch, out in (
            ("PROP", prop_out),
            ("FULL_DONOR_FAST", donor_out),
            ("NATIVE_FULL_REFRESH", native_out),
        ):
            current_rows.append(metrics_row(
                anchor, anchor, branch, out, model, paddings, padded_shape, gt[anchor], records
            ))
            save_prediction(out_dir, anchor, anchor, branch, out, model,
                            paddings, padded_shape, gt[anchor])

        baseline_state = clone_state(prop_state)
        donor_values = {
            "features_teacher": donor_features,
            "radius_teacher": donor_radius,
            "confidence_teacher": donor_confidence,
            "fast_tail_features": donor_features,
            "fast_tail_radius": donor_radius,
            "fast_tail_confidence": donor_confidence,
        }

        branch_states = {name: clone_state(baseline_state) for name in BRANCHES}
        # This is a real assignment path: rewrite the six fields with identical baseline values.
        assign_group(branch_states["NOOP_COMMIT"], baseline_state, donor_values, "NOOP_COMMIT")
        anchor_noop_state_diff = max_state_diff(branch_states["NOOP_COMMIT"], baseline_state)
        if anchor_noop_state_diff > args.tolerance:
            raise AssertionError(
                f"A{anchor}: NOOP_COMMIT changed state at anchor: diff={anchor_noop_state_diff}"
            )

        for name in ("FEATURE_ONLY", "RADIUS_ONLY", "CONFIDENCE_ONLY", "FULL_COMMIT"):
            assign_group(branch_states[name], baseline_state, donor_values, name)

        anchor_rows = []
        max_h = max(horizons)
        max_noop_output_diff = 0.0
        max_noop_state_diff = 0.0
        noop_pass = True

        for step in range(1, max_h + 1):
            frame_idx = anchor + step
            outs, restarts = {}, {}

            # OUTPUT_ONLY is the fixed-control trajectory. Every other branch restores
            # the same decoder runtime snapshot and is forced through the same frame index.
            for branch in BRANCHES:
                state = branch_states[branch]
                out, state, restart = forced_propagation_step(
                    stream, inputs, frame_idx, state, total_frames,
                    post_prop_runtime, model
                )
                if branch != "OUTPUT_ONLY":
                    sync_scheduler(state, branch_states["OUTPUT_ONLY"])
                branch_states[branch] = state
                outs[branch] = out
                restarts[branch] = restart

            noop_out_diff = max_output_diff(outs["OUTPUT_ONLY"], outs["NOOP_COMMIT"])
            noop_state_diff = max_state_diff(
                branch_states["OUTPUT_ONLY"], branch_states["NOOP_COMMIT"]
            )
            noop_restart_match = restarts["OUTPUT_ONLY"] == restarts["NOOP_COMMIT"]
            noop_step_pass = (
                noop_out_diff <= args.tolerance
                and noop_state_diff <= args.tolerance
                and noop_restart_match
            )
            noop_pass = noop_pass and noop_step_pass
            max_noop_output_diff = max(max_noop_output_diff, noop_out_diff)
            max_noop_state_diff = max(max_noop_state_diff, noop_state_diff)
            noop_rows.append({
                "anchor_frame": anchor,
                "future_frame": frame_idx,
                "step_from_anchor": step,
                "output_max_abs_diff_OUTPUT_ONLY_vs_NOOP_COMMIT": noop_out_diff,
                "state_max_abs_diff_OUTPUT_ONLY_vs_NOOP_COMMIT": noop_state_diff,
                "restart_OUTPUT_ONLY": restarts["OUTPUT_ONLY"],
                "restart_NOOP_COMMIT": restarts["NOOP_COMMIT"],
                "restart_match": noop_restart_match,
                "pass_at_tolerance": noop_step_pass,
                "tolerance": args.tolerance,
            })

            for branch in BRANCHES:
                row = metrics_row(
                    anchor, frame_idx, branch, outs[branch], model,
                    paddings, padded_shape, gt[frame_idx], records
                )
                row["restart_requested"] = restarts[branch]
                row["scheduler_synced_to_output_only"] = branch != "OUTPUT_ONLY"
                future_rows.append(row)
                anchor_rows.append(row)
                if step in set(horizons):
                    save_prediction(
                        out_dir, anchor, frame_idx, branch, outs[branch],
                        model, paddings, padded_shape, gt[frame_idx]
                    )

            print(
                f"A{anchor} H={step}: "
                f"NOOP out_diff={noop_out_diff:.3g}, state_diff={noop_state_diff:.3g}, "
                f"pass={noop_step_pass}; "
                f"AbsRel OUTPUT={next(r['absrel'] for r in anchor_rows if r['step_from_anchor']==step and r['branch']=='OUTPUT_ONLY'):.6f}, "
                f"FEATURE={next(r['absrel'] for r in anchor_rows if r['step_from_anchor']==step and r['branch']=='FEATURE_ONLY'):.6f}, "
                f"RADIUS={next(r['absrel'] for r in anchor_rows if r['step_from_anchor']==step and r['branch']=='RADIUS_ONLY'):.6f}, "
                f"CONF={next(r['absrel'] for r in anchor_rows if r['step_from_anchor']==step and r['branch']=='CONFIDENCE_ONLY'):.6f}, "
                f"FULL={next(r['absrel'] for r in anchor_rows if r['step_from_anchor']==step and r['branch']=='FULL_COMMIT'):.6f}"
            )

        timing_rows.append({
            "anchor_frame": anchor,
            "full_frame_donor_generation_ms": donor_generation_ms,
            "donor_fast_readout_ms": donor_readout_ms,
            "noop_anchor_state_diff": anchor_noop_state_diff,
            "noop_max_future_output_abs_diff": max_noop_output_diff,
            "noop_max_future_state_abs_diff": max_noop_state_diff,
            "noop_pass": noop_pass,
            "tolerance": args.tolerance,
            "restart_at_anchor": False,
        })
        status_rows.append({
            "anchor_frame": anchor,
            "status": "completed",
            "noop_pass": noop_pass,
            "noop_max_output_diff": max_noop_output_diff,
            "noop_max_state_diff": max_noop_state_diff,
        })
        print(
            f"A{anchor}: NOOP pass={noop_pass}; max output diff={max_noop_output_diff:.3g}; "
            f"max state diff={max_noop_state_diff:.3g}; "
            f"donor_generation={donor_generation_ms:.2f}ms; readout={donor_readout_ms:.2f}ms"
        )

        for horizon in horizons:
            row = {
                "anchor_frame": anchor,
                "horizon_H": horizon,
                "future_frames_with_gt": sum(
                    bool(records[anchor + k]["gt_available"]) for k in range(1, horizon + 1)
                ),
                "noop_pass": noop_pass,
                "noop_max_output_abs_diff": max_noop_output_diff,
                "noop_max_state_abs_diff": max_noop_state_diff,
            }
            for branch in BRANCHES:
                row[f"{branch}_mean_absrel"] = mean_metric(anchor_rows, branch, "absrel", horizon)
                row[f"{branch}_mean_rmse"] = mean_metric(anchor_rows, branch, "rmse", horizon)
                row[f"{branch}_mean_delta1"] = mean_metric(anchor_rows, branch, "delta1", horizon)
            for branch in ("NOOP_COMMIT", "FEATURE_ONLY", "RADIUS_ONLY", "CONFIDENCE_ONLY", "FULL_COMMIT"):
                g_abs, n_abs = paired_gain(anchor_rows, branch, "absrel", horizon, higher_is_better=False)
                g_rmse, n_rmse = paired_gain(anchor_rows, branch, "rmse", horizon, higher_is_better=False)
                g_delta, n_delta = paired_gain(anchor_rows, branch, "delta1", horizon, higher_is_better=True)
                row[f"G_absrel_OUTPUT_ONLY_minus_{branch}"] = g_abs
                row[f"G_rmse_OUTPUT_ONLY_minus_{branch}"] = g_rmse
                row[f"G_delta1_{branch}_minus_OUTPUT_ONLY"] = g_delta
                row[f"paired_frames_{branch}_absrel"] = n_abs
                row[f"paired_frames_{branch}_rmse"] = n_rmse
                row[f"paired_frames_{branch}_delta1"] = n_delta
            anchor_summaries.append(row)

    write_csv(out_dir / "e2_s2_current_metrics.csv", current_rows)
    write_csv(out_dir / "e2_s2_future_metrics.csv", future_rows)
    write_csv(out_dir / "e2_s2_noop_commit_audit.csv", noop_rows)
    write_csv(out_dir / "e2_s2_horizon_summary.csv", anchor_summaries)
    write_csv(out_dir / "e2_s2_anchor_status.csv", status_rows)
    write_csv(out_dir / "e2_s2_timing_and_gate.csv", timing_rows)

    metadata = {
        "experiment": "E2-S2_persistent_state_field_ablation",
        "model": "lpiccinelli/velodepth",
        "weights_frozen": True,
        "data_root": str(root),
        "frames": args.frames,
        "anchors_zero_based": anchors,
        "horizons": horizons,
        "gt_match_tolerance_seconds": args.max_time_diff,
        "gt_scale": args.gt_scale,
        "resolution_level": args.resolution_level,
        "noop_tolerance": args.tolerance,
        "branches": {
            "OUTPUT_ONLY": "same full donor current output; persistent state stays native propagated",
            "NOOP_COMMIT": "explicitly assigns all six baseline feature/geometry/cache fields unchanged",
            "FEATURE_ONLY": "donor features_teacher and fast_tail_features",
            "RADIUS_ONLY": "donor radius_teacher and fast_tail_radius",
            "CONFIDENCE_ONLY": "donor confidence_teacher and fast_tail_confidence",
            "FULL_COMMIT": "all six donor feature/geometry/cache fields",
            "PROP": "native propagated current-frame reference",
            "FULL_DONOR_FAST": "full-frame teacher-conditioned donor through fast readout",
            "NATIVE_FULL_REFRESH": "native keyframe output at anchor; separate current-frame reference",
        },
        "state_cache_fields": list(STATE_CACHE_FIELDS),
        "scheduler_fields_synchronized_to_output_only": list(SCHEDULER_FIELDS),
        "future_policy": "forced propagation; restart signals recorded but ignored; scheduler fields synchronized to OUTPUT_ONLY",
        "current_output_shared_between_future_branches": True,
        "limitations": [
            "A9 and A37 are preselected development anchors, not independent validation",
            "full-frame teacher donor is an oracle diagnostic, not a low-cost online executor",
            "field groups include paired fast-tail cache fields to preserve cache coherence",
            "future forced-propagation results do not estimate policy performance when keyframe restarts are executed",
            "no training is performed",
        ],
        "outputs": {
            "current_metrics": str(out_dir / "e2_s2_current_metrics.csv"),
            "future_metrics": str(out_dir / "e2_s2_future_metrics.csv"),
            "noop_audit": str(out_dir / "e2_s2_noop_commit_audit.csv"),
            "horizon_summary": str(out_dir / "e2_s2_horizon_summary.csv"),
            "anchor_status": str(out_dir / "e2_s2_anchor_status.csv"),
            "timing_and_gate": str(out_dir / "e2_s2_timing_and_gate.csv"),
            "predictions": str(out_dir / "predictions"),
        },
    }
    (out_dir / "e2_s2_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== E2-S2 finished ===")
    print("Current metrics:", out_dir / "e2_s2_current_metrics.csv")
    print("Future metrics:", out_dir / "e2_s2_future_metrics.csv")
    print("NOOP commit audit:", out_dir / "e2_s2_noop_commit_audit.csv")
    print("Horizon summary:", out_dir / "e2_s2_horizon_summary.csv")
    print("Metadata:", out_dir / "e2_s2_metadata.json")
    print("Positive G_absrel means the field-commit branch lowered future AbsRel.")
    print("Do not expand to local-tile experiments unless NOOP passes and a field group shows reproducible AbsRel gain.")


if __name__ == "__main__":
    main()
