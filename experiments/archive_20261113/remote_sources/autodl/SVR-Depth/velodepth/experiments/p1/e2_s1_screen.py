#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2-S1 screen: NOOP audit + full-donor/ native-refresh positive controls.

Frozen-weight diagnostic only. This is deliberately a screening phase, not the
local-tile ablation or a training script.

Branches at each anchor:
  PROP: native propagation candidate at anchor (current-frame reference only)
  DONOR_OUTPUT_ONLY: full-frame teacher-conditioned neck -> fast readout at
                     anchor, but retain propagated persistent state for future
  DONOR_FULL_COMMIT: same exact donor current output, commit donor state fields
  NATIVE_FULL_REFRESH: native keyframe path at anchor, then future propagation
  NOOP_DUPLICATE: duplicate of DONOR_OUTPUT_ONLY future state to test that two
                  identical branches remain tensor-equivalent.

For paired DONOR_OUTPUT_ONLY vs DONOR_FULL_COMMIT future comparisons, scheduler
fields are synchronized to DONOR_OUTPUT_ONLY after each future step and restart
requests are logged but ignored; the future action sequence is fixed. Native
refresh is reported as a separate reference because its state/rays/scheduler
may differ from propagation.
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
    "current_t", "distance_kf", "flow_magnitude", "total_luma_change",
    "first_keyframe",
)
COMMIT_FIELDS = (
    "features_teacher", "radius_teacher", "confidence_teacher",
    "fast_tail_features", "fast_tail_radius", "fast_tail_confidence",
)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields_out = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields_out, extrasaction="ignore")
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
    if a.shape != b.shape:
        return float("inf")
    aa = a.detach().float()
    bb = b.detach().float()
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
    return max((max_nested_diff(getattr(a, f.name), getattr(b, f.name))
                for f in fields(StreamState)), default=0.0)


def max_output_diff(a, b):
    return max(
        max_tensor_diff(a.radius, b.radius),
        max_tensor_diff(a.rays, b.rays),
        max_tensor_diff(a.confidence, b.confidence),
    )


def restore_runtime(model, snapshot):
    temp = clone_state(snapshot.stream_state)
    restore_state(model, temp, snapshot)


def sync_scheduler(dst, src):
    for name in SCHEDULER_FIELDS:
        setattr(dst, name, clone_value(getattr(src, name)))


def build_teacher_conditioned_features(model, inputs, stream, frame_idx, total_frames, rays_base):
    """Same donor construction as E2-S0; donor remains a diagnostic candidate."""
    image = stream._frame(inputs["image"], frame_idx, total_frames)
    features, tokens = model.pixel_encoder(image)
    features = [model.stacking_fn(features[i:j]).contiguous()
                for i, j in model.slices_encoder_range]
    tokens = [model.stacking_fn(tokens[i:j]).contiguous()
              for i, j in model.slices_encoder_range]
    metas = [{"T": 1, "B": 1}]
    base_inputs = {"image": image, "features": features, "tokens": tokens}
    model.pixel_decoder.set_shapes(base_inputs, metas)
    adapted_features = model.pixel_decoder.input_adapter(features)
    radial = model.pixel_decoder.radial_module
    ray_embedding = radial.embed_rays(rays_base)
    conditioned = radial.condition(adapted_features, ray_embedding)
    h_common, w_common = radial.shapes
    return [x.reshape(x.shape[0], h_common, w_common, x.shape[-1])
             .permute(0, 3, 1, 2).contiguous() for x in conditioned]


def save_prediction(out_dir, anchor, frame_idx, branch, out, model, paddings, padded_shape, gt):
    if gt is None:
        return
    pred = mh.output_depth(out, model, paddings, padded_shape)
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    p = Path(out_dir) / "predictions" / f"a{anchor:03d}_f{frame_idx:03d}_{branch}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, pred=pred.astype(np.float32), gt=gt.astype(np.float32), valid=valid)


def metrics_row(anchor, frame_idx, branch, out, model, paddings, padded_shape, gt, records):
    m = mh.eval_output(out, model, paddings, padded_shape, gt)
    return {
        "anchor_frame": anchor,
        "frame_index": frame_idx,
        "step_from_anchor": frame_idx - anchor,
        "branch": branch,
        "rgb_timestamp": records[frame_idx]["rgb_ts"],
        "gt_available": bool(records[frame_idx]["gt_available"]),
        "gt_time_diff_ms": (records[frame_idx]["time_diff"] * 1000.0
                            if math.isfinite(records[frame_idx]["time_diff"]) else float("nan")),
        "absrel": m["absrel"], "rmse": m["rmse"],
        "delta1": m["delta1"], "valid_ratio": m["valid_ratio"],
    }


def forced_propagation_step(stream, inputs, frame_idx, state, total_frames, runtime_snapshot, model):
    """Run propagation while keeping a fixed future schedule even on restart requests."""
    restore_runtime(model, runtime_snapshot)
    out, state, restart = stream.propagate_step(inputs, frame_idx, state, total_frames)
    if restart:
        # Counterfactual protocol: reject native restart but advance to the next
        # frame uniformly instead of leaving current_t/distance_kf stale.
        state.first_keyframe = False
        state.distance_kf += 1
        state.current_t = frame_idx + 1
    return out, state, bool(restart)


def mean_finite(rows, branch, key, max_step):
    vals = [float(r[key]) for r in rows
            if r["branch"] == branch and 1 <= int(r["step_from_anchor"]) <= max_step
            and math.isfinite(float(r[key]))]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description="E2-S1 donor and state-commit screening experiment")
    ap.add_argument("--data-root", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2")
    ap.add_argument("--frames", type=int, default=46)
    ap.add_argument("--anchors", default="9,37", help="Zero-based anchor indices; fixed before run")
    ap.add_argument("--horizons", default="1,4,8")
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("--out-dir", default="outputs/e2_s1_screen/crowd2_a9_a37")
    args = ap.parse_args()

    anchors = mh.parse_ints(args.anchors, "--anchors", 1)
    horizons = mh.parse_ints(args.horizons, "--horizons", 1)
    if args.frames < max(anchors) + max(horizons) + 1:
        ap.error(f"--frames must include anchor {max(anchors)} plus horizon {max(horizons)}")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. Run this experiment on the GPU instance, not no-card mode.")

    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = False
    root = Path(args.data_root)
    rgb_rows = mh.read_index(root, "rgb.txt", "rgb")
    depth_rows = mh.read_index(root, "depth.txt", "depth")
    selected = rgb_rows[:args.frames]
    if len(selected) != args.frames:
        raise RuntimeError(f"Requested {args.frames} RGB frames, found {len(selected)}")
    records = mh.match_gt(selected, depth_rows, args.max_time_diff)
    gt = [mh.load_gt(row["gt_path"], args.gt_scale) if row["gt_available"] else None
          for row in records]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== E2-S1 screen: NOOP + full donor + native refresh ===")
    print(f"Frames={args.frames}; anchors={anchors}; horizons={horizons}")
    print(f"GT matched={sum(bool(r['gt_available']) for r in records)}/{len(records)}")
    print("No training. Local tile ablation is gated on this screen's results.")

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

    current_rows, future_rows, noop_rows, anchor_status_rows = [], [], [], []
    timing_rows = []
    for anchor in anchors:
        print(f"\n--- anchor A{anchor} ---")
        state_before = mh.prefix_state(model, stream, inputs, total_frames, anchor, initial_snapshot)
        pre_anchor_snapshot = capture_state(model, state_before)

        prop_state = clone_state(state_before)
        restore_state(model, prop_state, pre_anchor_snapshot)
        with torch.inference_mode():
            prop_out, prop_state, restart_anchor = stream.propagate_step(
                inputs, anchor, prop_state, total_frames
            )
        if restart_anchor:
            row = {"anchor_frame": anchor, "status": "skipped_anchor_restart",
                   "reason": "Propagation requests a restart at the anchor; predeclared rule skips it."}
            anchor_status_rows.append(row)
            print("SKIP:", row["reason"])
            continue
        post_prop_runtime = capture_state(model, prop_state)

        # Full-frame teacher donor in the current propagated ray field.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            donor_features = build_teacher_conditioned_features(
                model, inputs, stream, anchor, total_frames, prop_state.rays_base
            )
        torch.cuda.synchronize()
        donor_encoder_ms = (time.perf_counter() - t0) * 1000.0
        for i, (a, b) in enumerate(zip(prop_state.features_teacher, donor_features)):
            if tuple(a.shape) != tuple(b.shape):
                raise RuntimeError(f"A{anchor}: donor shape mismatch at level {i}: {a.shape} vs {b.shape}")

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

        # Native full refresh from the same pre-anchor state (separate reference).
        native_state = clone_state(state_before)
        restore_state(model, native_state, pre_anchor_snapshot)
        with torch.inference_mode():
            native_out, native_state = stream.keyframe_step(inputs, anchor, native_state, total_frames)
        native_runtime = capture_state(model, native_state)

        for branch, out in (("PROP", prop_out), ("FULL_DONOR_FAST", donor_out),
                            ("NATIVE_FULL_REFRESH", native_out)):
            current_rows.append(metrics_row(anchor, anchor, branch, out, model,
                                            paddings, padded_shape, gt[anchor], records))
            save_prediction(out_dir, anchor, anchor, branch, out, model,
                            paddings, padded_shape, gt[anchor])

        # The output-only branch's current output is donor_out, but its persistent
        # future state remains the unmodified propagated state.
        output_only_state = clone_state(prop_state)
        noop_state = clone_state(prop_state)
        donor_commit_state = clone_state(prop_state)
        for name in COMMIT_FIELDS:
            if name == "features_teacher":
                value = donor_features
            elif name == "radius_teacher":
                value = donor_radius
            elif name == "confidence_teacher":
                value = donor_confidence
            elif name == "fast_tail_features":
                value = donor_features
            elif name == "fast_tail_radius":
                value = donor_radius
            elif name == "fast_tail_confidence":
                value = donor_confidence
            setattr(donor_commit_state, name, clone_value(value))

        # Runtime snapshot for donor branch uses the same propagated ray/shapes.
        donor_runtime = post_prop_runtime
        native_state_future = clone_state(native_state)
        native_runtime_future = native_runtime

        max_h = max(horizons)
        noop_pass = True
        max_noop_output = 0.0
        max_noop_state = 0.0
        for step in range(1, max_h + 1):
            frame_idx = anchor + step
            # Output-only branch is the reference fixed-control trajectory.
            out_only, output_only_state, restart_only = forced_propagation_step(
                stream, inputs, frame_idx, output_only_state, total_frames,
                post_prop_runtime, model
            )
            # A duplicate branch proves that a no-op intervention itself is inert.
            out_noop, noop_state, restart_noop = forced_propagation_step(
                stream, inputs, frame_idx, noop_state, total_frames,
                post_prop_runtime, model
            )
            noop_out_diff = max_output_diff(out_only, out_noop)
            noop_state_diff = max_state_diff(output_only_state, noop_state)
            max_noop_output = max(max_noop_output, noop_out_diff)
            max_noop_state = max(max_noop_state, noop_state_diff)
            noop_step_pass = (noop_out_diff <= args.tolerance and noop_state_diff <= args.tolerance
                              and restart_only == restart_noop)
            noop_pass = noop_pass and noop_step_pass
            noop_rows.append({
                "anchor_frame": anchor, "future_frame": frame_idx, "step": step,
                "output_max_abs_diff": noop_out_diff,
                "state_max_abs_diff": noop_state_diff,
                "restart_requested_output_only": restart_only,
                "restart_requested_noop_duplicate": restart_noop,
                "pass_at_tolerance": noop_step_pass,
                "tolerance": args.tolerance,
            })

            # Donor-commit branch shares scheduler control with output-only branch.
            sync_scheduler(donor_commit_state, output_only_state)
            out_commit, donor_commit_state, restart_commit = forced_propagation_step(
                stream, inputs, frame_idx, donor_commit_state, total_frames,
                donor_runtime, model
            )
            sync_scheduler(donor_commit_state, output_only_state)

            # Native refresh is a separate reference trajectory; retain its own state.
            out_native, native_state_future, restart_native = forced_propagation_step(
                stream, inputs, frame_idx, native_state_future, total_frames,
                native_runtime_future, model
            )

            branch_outs = (
                ("DONOR_OUTPUT_ONLY", out_only),
                ("DONOR_FULL_COMMIT", out_commit),
                ("NATIVE_FULL_REFRESH", out_native),
            )
            for branch, out in branch_outs:
                future_rows.append(metrics_row(anchor, frame_idx, branch, out, model,
                                               paddings, padded_shape, gt[frame_idx], records))
                save_prediction(out_dir, anchor, frame_idx, branch, out, model,
                                paddings, padded_shape, gt[frame_idx])
            # Also record the duplicate no-op branch's metrics for independent auditing.
            future_rows.append(metrics_row(anchor, frame_idx, "NOOP_DUPLICATE", out_noop, model,
                                           paddings, padded_shape, gt[frame_idx], records))
            print(
                f"A{anchor} H{step}: donor-only AbsRel={future_rows[-4]['absrel'] if len(future_rows)>=4 else float('nan'):.6f}; "
                f"NOOP out_diff={noop_out_diff:.3g}, state_diff={noop_state_diff:.3g}; "
                f"commit_restart={restart_commit}, native_restart={restart_native}"
            )

        timing_rows.append({
            "anchor_frame": anchor,
            "full_frame_donor_generation_ms": donor_encoder_ms,
            "donor_fast_readout_ms": donor_readout_ms,
            "noop_pass": noop_pass,
            "noop_max_output_abs_diff": max_noop_output,
            "noop_max_state_abs_diff": max_noop_state,
            "tolerance": args.tolerance,
            "restart_at_anchor": False,
        })
        print(f"A{anchor}: NOOP pass={noop_pass}; max output diff={max_noop_output:.3g}; "
              f"max state diff={max_noop_state:.3g}")

    write_csv(out_dir / "e2_s1_current_metrics.csv", current_rows)
    write_csv(out_dir / "e2_s1_future_metrics.csv", future_rows)
    write_csv(out_dir / "e2_s1_noop_audit.csv", noop_rows)
    write_csv(out_dir / "e2_s1_timing_and_gate.csv", timing_rows)
    write_csv(out_dir / "e2_s1_anchor_status.csv", anchor_status_rows)

    summary_rows = []
    for anchor in anchors:
        ar = [r for r in future_rows if int(r["anchor_frame"]) == anchor]
        for h in horizons:
            row = {"anchor_frame": anchor, "horizon_H": h}
            for branch in ("DONOR_OUTPUT_ONLY", "DONOR_FULL_COMMIT", "NATIVE_FULL_REFRESH", "NOOP_DUPLICATE"):
                row[f"{branch}_mean_absrel"] = mean_finite(ar, branch, "absrel", h)
                row[f"{branch}_mean_rmse"] = mean_finite(ar, branch, "rmse", h)
                row[f"{branch}_mean_delta1"] = mean_finite(ar, branch, "delta1", h)
            out_only = row["DONOR_OUTPUT_ONLY_mean_absrel"]
            commit = row["DONOR_FULL_COMMIT_mean_absrel"]
            native = row["NATIVE_FULL_REFRESH_mean_absrel"]
            row["G_writeback_absrel"] = (out_only - commit
                                          if math.isfinite(out_only) and math.isfinite(commit) else float("nan"))
            row["G_native_vs_output_only_absrel"] = (out_only - native
                                                      if math.isfinite(out_only) and math.isfinite(native) else float("nan"))
            row["positive_gain_means_lower_absrel"] = True
            summary_rows.append(row)
    write_csv(out_dir / "e2_s1_horizon_summary.csv", summary_rows)

    metadata = {
        "experiment": "E2-S1_screen_noop_full_donor_native_refresh",
        "model": "lpiccinelli/velodepth",
        "weights_frozen": True,
        "data_root": str(root),
        "frames": args.frames,
        "anchors_zero_based": anchors,
        "horizons": horizons,
        "gt_match_tolerance_seconds": args.max_time_diff,
        "gt_scale": args.gt_scale,
        "resolution_level": args.resolution_level,
        "no_op_tolerance": args.tolerance,
        "branches": {
            "PROP": "native propagation output at anchor; current-frame reference",
            "DONOR_OUTPUT_ONLY": "full teacher donor current output; future state remains propagated",
            "DONOR_FULL_COMMIT": "same donor current output; commits features/radius/confidence and fast-tail caches",
            "NATIVE_FULL_REFRESH": "native keyframe refresh at anchor and its own future state trajectory",
            "NOOP_DUPLICATE": "duplicate of donor output-only future state for output/state equivalence audit",
        },
        "future_control": "fixed propagation; restart requests logged and ignored; scheduler of donor commit synced to output-only branch",
        "donor_source": "full-frame pixel encoder + input adapter + camera conditioning using propagated rays_base",
        "commit_fields": list(COMMIT_FIELDS),
        "scheduler_fields": list(SCHEDULER_FIELDS),
        "limitations": [
            "development screening only; A9/A37 are not independent validation anchors",
            "native full refresh has a distinct rays/state/scheduler trajectory and is a separate reference",
            "teacher donor is full-frame and not a deployable local repair executor",
            "local tile ablation is intentionally gated on these results",
            "fixed propagation future is a causal counterfactual, not end-to-end native-policy latency evaluation",
        ],
        "outputs": {
            "current_metrics": str(out_dir / "e2_s1_current_metrics.csv"),
            "future_metrics": str(out_dir / "e2_s1_future_metrics.csv"),
            "noop_audit": str(out_dir / "e2_s1_noop_audit.csv"),
            "horizon_summary": str(out_dir / "e2_s1_horizon_summary.csv"),
            "timing_and_gate": str(out_dir / "e2_s1_timing_and_gate.csv"),
            "anchor_status": str(out_dir / "e2_s1_anchor_status.csv"),
            "predictions": str(out_dir / "predictions"),
        },
    }
    (out_dir / "e2_s1_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== E2-S1 finished ===")
    print(f"Results: {out_dir}")
    print("Review e2_s1_noop_audit.csv first. If any NOOP row fails, do not interpret method gains.")
    print("Then inspect e2_s1_horizon_summary.csv; positive G_writeback_absrel means commit improves AbsRel.")


if __name__ == "__main__":
    main()
