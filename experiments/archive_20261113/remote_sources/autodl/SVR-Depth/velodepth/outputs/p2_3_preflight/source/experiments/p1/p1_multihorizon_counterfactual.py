#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-horizon B1/B2 state-writeback counterfactual evaluation for SVR-Depth.

B0: accept propagation output at anchor t.
B1: share the full-refresh output at t and commit refreshed persistent geometry.
B2: share the same full-refresh output at t but keep the propagated state.
For t+1..t+H, both branches are forced through propagate_step. Restart signals
are logged but not executed as keyframes, making the future action sequence fixed.

This is a full-frame refresh oracle/causal diagnostic, NOT local-tile performance.
Primary label: G_state(H)=sum_k gamma**k * (AbsRel_B2[t+k]-AbsRel_B1[t+k]).
Positive G_state means writeback lowers future AbsRel.
"""
import os
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import copy
import csv
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

from velodepth.models.velodepth import VeloDepth, _postprocess, get_paddings
from stream_state import StreamState, clone_state
from stream_state_snapshot import capture_state, restore_state
from velodepth_stream import VeloDepthStream

COMMIT_FIELDS = ("features_teacher", "radius_teacher", "confidence_teacher", "rays_base")
SCHEDULER_FIELDS = ("current_t", "distance_kf", "flow_magnitude", "total_luma_change", "first_keyframe")


def clone_payload(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: clone_payload(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clone_payload(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone_payload(v) for v in x)
    return copy.deepcopy(x)


def parse_ints(text, name, minimum):
    try:
        values = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    except ValueError as e:
        raise ValueError(f"{name} must be comma-separated integers") from e
    if not values or any(x < minimum for x in values):
        raise ValueError(f"{name} values must be >= {minimum}")
    return values


def read_index(root, filename, subdir):
    index = root / filename
    if not index.is_file():
        raise FileNotFoundError(index)
    rows = []
    for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            ts = float(fields[0])
        except ValueError:
            continue
        rel = Path(fields[1])
        candidates = [rel] if rel.is_absolute() else [root / rel, root / subdir / rel.name]
        path = next((p for p in candidates if p.is_file()), None)
        if path is not None:
            rows.append((ts, path))
    rows.sort(key=lambda x: x[0])
    if not rows:
        raise RuntimeError(f"No valid image entries found in {index}")
    return rows


def match_gt(rgb_rows, depth_rows, max_diff):
    """Attach GT when the nearest depth timestamp is within tolerance.

    IMPORTANT: never drop an RGB frame from the sequence. A frame without a
    valid GT match still participates in temporal propagation; it is excluded
    only from metric computation. This preserves frame indices and discount
    exponents for multi-horizon counterfactuals.
    """
    depth_ts = [x[0] for x in depth_rows]
    result = []
    for rgb_index, (rgb_ts, rgb_path) in enumerate(rgb_rows):
        i = bisect_left(depth_ts, rgb_ts)
        candidates = []
        if i < len(depth_rows):
            candidates.append(depth_rows[i])
        if i > 0:
            candidates.append(depth_rows[i - 1])

        nearest = min(candidates, key=lambda x: abs(x[0] - rgb_ts)) if candidates else None
        if nearest is None:
            nearest_ts, nearest_path, dt = float("nan"), None, float("nan")
        else:
            nearest_ts, nearest_path = nearest
            dt = abs(nearest_ts - rgb_ts)

        matched = nearest is not None and dt <= max_diff
        result.append({
            "frame_index": rgb_index,
            "rgb_ts": rgb_ts,
            "rgb_path": rgb_path,
            "gt_ts": nearest_ts if matched else float("nan"),
            "gt_path": nearest_path if matched else None,
            "time_diff": dt if matched else float("nan"),
            "gt_available": bool(matched),
            "nearest_depth_ts": nearest_ts,
            "nearest_depth_path": str(nearest_path) if nearest_path is not None else "",
            "nearest_time_diff": dt,
            "gt_match_reason": "matched" if matched else (
                "no_depth_records" if nearest is None else "outside_tolerance"
            ),
        })
    return result


def load_frames(records):
    imgs = []
    for row in records:
        bgr = cv2.imread(str(row["rgb_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read {row['rgb_path']}")
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()


def load_gt(path, scale):
    """Load a matched GT map; return None for RGB frames without matched GT."""
    if path is None:
        return None
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None or d.ndim != 2:
        raise RuntimeError(f"Invalid GT depth image: {path}")
    return d.astype(np.float32) / scale


def capture_inputs(model, frames):
    captured = {}
    original = model.encode_decode_test

    def wrapper(inputs, image_metas):
        if "inputs" not in captured:
            captured["inputs"] = clone_payload(inputs)
            captured["metas"] = clone_payload(image_metas)
        return original(inputs, image_metas)

    model.encode_decode_test = wrapper
    try:
        with torch.inference_mode():
            model.infer(frames, normalize=True)
    finally:
        model.encode_decode_test = original
    if "inputs" not in captured:
        raise RuntimeError("Failed to capture model.infer() packed inputs")

    inputs, metas = clone_payload(captured["inputs"]), clone_payload(captured["metas"])
    b, t = int(metas[0]["B"]), int(metas[0]["T"])
    if b != 1 or t != len(frames):
        raise RuntimeError(f"Expected B=1,T={len(frames)}; captured B={b},T={t}")
    # Same input packing/flow normalization used by VeloDepthStream.run().
    model._set_image_paddings(inputs)
    inputs["flow_bwd"], inputs["grid_sample"] = model._normalize_and_grid_from_flow(
        inputs["flow_bwd"], inputs["image"], inputs["image_paddings"], inputs["depth_paddings"]
    )
    return inputs, metas


def restore_runtime(model, snapshot):
    # Restore model runtime attributes while leaving actual branch states intact.
    temp = clone_state(snapshot.stream_state)
    restore_state(model, temp, snapshot)


def sync_scheduler(dst, src):
    for name in SCHEDULER_FIELDS:
        setattr(dst, name, clone_payload(getattr(src, name)))


def output_depth(output, model, paddings, padded_shape):
    points = output.radius * output.rays
    points = _postprocess(points, padded_shape, paddings=paddings,
                          interpolation_mode=model.interpolation_mode)
    return points[0, -1].detach().float().cpu().numpy()


def metrics(pred, gt):
    if pred.shape != gt.shape:
        raise RuntimeError(f"Pred/GT size mismatch: {pred.shape} vs {gt.shape}")
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    ratio_valid = float(valid.mean())
    if not valid.any():
        return {"absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan"),
                "valid_ratio": ratio_valid}
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    absrel = float(np.mean(np.abs(p - g) / np.maximum(g, 1e-8)))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    delta = np.maximum(p / np.maximum(g, 1e-8), g / np.maximum(p, 1e-8))
    return {"absrel": absrel, "rmse": rmse, "delta1": float(np.mean(delta < 1.25)),
            "valid_ratio": ratio_valid}


def eval_output(out, model, paddings, padded_shape, gt):
    # A missing timestamp-matched GT excludes this frame from accuracy metrics,
    # but does not remove it from the RGB stream or temporal state transitions.
    if gt is None:
        return {"absrel": float("nan"), "rmse": float("nan"),
                "delta1": float("nan"), "valid_ratio": float("nan")}
    return metrics(output_depth(out, model, paddings, padded_shape), gt)


def mean_metric(rows, key):
    x = np.asarray([r[key] for r in rows], dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def discounted_gain(b1, b2, key, gamma, higher_is_better=False):
    """Discount paired gains by ORIGINAL future-step index, skipping invalid GT.

    Rows are retained for every future RGB frame. If step k has no usable GT,
    its gain is omitted, but later steps still use gamma**k rather than being
    renumbered. No coverage renormalization is applied to the discounted sum.
    """
    total = 0.0
    valid_pairs = 0
    for k, (m1, m2) in enumerate(zip(b1, b2), 1):
        v1, v2 = m1.get(key, float("nan")), m2.get(key, float("nan"))
        if not (math.isfinite(v1) and math.isfinite(v2)):
            continue
        gain = v1 - v2 if higher_is_better else v2 - v1
        total += (gamma ** k) * gain
        valid_pairs += 1
    return float(total) if valid_pairs else float("nan")


def count_finite_pairs(rows1, rows2, key):
    """Count time steps with finite metrics in both counterfactual branches."""
    return sum(
        math.isfinite(a.get(key, float("nan"))) and math.isfinite(b.get(key, float("nan")))
        for a, b in zip(rows1, rows2)
    )


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def prefix_state(model, stream, inputs, t_total, anchor, initial_snapshot):
    state = StreamState()
    restore_state(model, state, initial_snapshot)
    _, state = stream.keyframe_step(inputs, 0, state, t_total)
    for t in range(1, anchor):
        _, state, restart = stream.propagate_step(inputs, t, state, t_total)
        if restart:
            # Match the original policy: discard propagation candidate and keyframe same frame.
            _, state = stream.keyframe_step(inputs, t, state, t_total)
    return state


def evaluate_anchor(model, stream, inputs, t_total, anchor, records, gt, initial_snapshot,
                    paddings, padded_shape, horizons, gamma):
    state_before = prefix_state(model, stream, inputs, t_total, anchor, initial_snapshot)
    pre = capture_state(model, state_before)

    # B0: real propagation candidate at anchor.
    b0_state = clone_state(state_before)
    restore_state(model, b0_state, pre)
    out_b0, b0_state, restart = stream.propagate_step(inputs, anchor, b0_state, t_total)
    if restart:
        return None, {
            "anchor_frame": anchor,
            "anchor_rgb_timestamp": records[anchor]["rgb_ts"],
            "status": "skipped_anchor_restart",
            "reason": "B0 propagation requests restart; normal policy would keyframe this frame.",
        }
    post_prop_snapshot = capture_state(model, b0_state)

    # Compute a full-frame refresh candidate from the SAME pre-anchor state.
    refresh_state = clone_state(state_before)
    restore_state(model, refresh_state, pre)
    out_refresh, refresh_state = stream.keyframe_step(inputs, anchor, refresh_state, t_total)

    # B1 commits only the persistent geometric fields. Scheduler and fast-tail
    # state stay on the propagated trajectory; B2 does not commit refresh state.
    b1 = clone_state(b0_state)
    for name in COMMIT_FIELDS:
        value = getattr(refresh_state, name)
        if value is None:
            raise RuntimeError(f"Refresh state field {name} is None")
        setattr(b1, name, clone_payload(value))
    b2 = clone_state(b0_state)

    m0 = eval_output(out_b0, model, paddings, padded_shape, gt[anchor])
    mr = eval_output(out_refresh, model, paddings, padded_shape, gt[anchor])
    anchor_record = records[anchor]
    anchor_info = {
        "anchor_frame": anchor,
        "anchor_rgb_timestamp": anchor_record["rgb_ts"],
        "anchor_gt_available": bool(anchor_record["gt_available"]),
        "anchor_gt_timestamp": anchor_record["gt_ts"],
        "anchor_gt_time_diff_ms": anchor_record["time_diff"] * 1000.0
            if math.isfinite(anchor_record["time_diff"]) else float("nan"),
        "anchor_nearest_gt_time_diff_ms": anchor_record["nearest_time_diff"] * 1000.0
            if math.isfinite(anchor_record["nearest_time_diff"]) else float("nan"),
        "absrel_B0": m0["absrel"], "rmse_B0": m0["rmse"], "delta1_B0": m0["delta1"],
        "absrel_shared_refresh_B1_B2": mr["absrel"],
        "rmse_shared_refresh_B1_B2": mr["rmse"],
        "delta1_shared_refresh_B1_B2": mr["delta1"],
        "G_inst_absrel": m0["absrel"] - mr["absrel"],
        "G_inst_rmse": m0["rmse"] - mr["rmse"],
        "G_inst_delta1": mr["delta1"] - m0["delta1"],
        "current_B1_B2_output_shared": True,
        "refresh_type": "full_frame_oracle",
    }

    b1_rows, b2_rows, frame_rows = [], [], []
    for k in range(1, max(horizons) + 1):
        frame_idx = anchor + k
        sync_scheduler(b1, b2)
        restore_runtime(model, post_prop_snapshot)
        out1, b1, restart1 = stream.propagate_step(inputs, frame_idx, b1, t_total)
        restore_runtime(model, post_prop_snapshot)
        out2, b2, restart2 = stream.propagate_step(inputs, frame_idx, b2, t_total)
        # The B2 scheduler defines a shared scheduler path for the next frame.
        sync_scheduler(b1, b2)

        m1 = eval_output(out1, model, paddings, padded_shape, gt[frame_idx])
        m2 = eval_output(out2, model, paddings, padded_shape, gt[frame_idx])
        b1_rows.append(m1)
        b2_rows.append(m2)
        future_record = records[frame_idx]
        frame_rows.append({
            "anchor_frame": anchor, "future_step": k, "future_frame": frame_idx,
            "future_rgb_timestamp": future_record["rgb_ts"],
            "future_gt_available": bool(future_record["gt_available"]),
            "future_gt_timestamp": future_record["gt_ts"],
            "future_gt_time_diff_ms": future_record["time_diff"] * 1000.0
                if math.isfinite(future_record["time_diff"]) else float("nan"),
            "future_nearest_gt_time_diff_ms": future_record["nearest_time_diff"] * 1000.0
                if math.isfinite(future_record["nearest_time_diff"]) else float("nan"),
            "future_gt_match_reason": future_record["gt_match_reason"],
            "restart_requested_B1_ignored": bool(restart1),
            "restart_requested_B2_ignored": bool(restart2),
            "absrel_B1_writeback": m1["absrel"], "absrel_B2_no_writeback": m2["absrel"],
            "G_state_step_absrel": m2["absrel"] - m1["absrel"],
            "rmse_B1_writeback": m1["rmse"], "rmse_B2_no_writeback": m2["rmse"],
            "G_state_step_rmse": m2["rmse"] - m1["rmse"],
            "delta1_B1_writeback": m1["delta1"], "delta1_B2_no_writeback": m2["delta1"],
            "G_state_step_delta1": m1["delta1"] - m2["delta1"],
            "valid_ratio_B1": m1["valid_ratio"], "valid_ratio_B2": m2["valid_ratio"],
        })

    summaries = []
    for h in horizons:
        r1, r2 = b1_rows[:h], b2_rows[:h]
        g_absrel = discounted_gain(r1, r2, "absrel", gamma)
        evaluated_pairs = count_finite_pairs(r1, r2, "absrel")
        gt_frames = sum(bool(records[anchor + step]["gt_available"]) for step in range(1, h + 1))
        summaries.append({
            **anchor_info, "horizon_H": h, "gamma": gamma,
            "mean_future_absrel_B1": mean_metric(r1, "absrel"),
            "mean_future_absrel_B2": mean_metric(r2, "absrel"),
            "mean_future_rmse_B1": mean_metric(r1, "rmse"),
            "mean_future_rmse_B2": mean_metric(r2, "rmse"),
            "mean_future_delta1_B1": mean_metric(r1, "delta1"),
            "mean_future_delta1_B2": mean_metric(r2, "delta1"),
            "G_state_absrel_discounted": g_absrel,
            "G_state_absrel_discounted_per_step": g_absrel / h,
            "G_state_absrel_discounted_per_valid_frame":
                g_absrel / evaluated_pairs if evaluated_pairs else float("nan"),
            "G_state_rmse_discounted": discounted_gain(r1, r2, "rmse", gamma),
            "G_state_delta1_discounted": discounted_gain(r1, r2, "delta1", gamma, higher_is_better=True),
            "future_frames_requested": h,
            "future_frames_with_gt": int(gt_frames),
            "future_gt_coverage_ratio": float(gt_frames / h),
            "future_frames_evaluated": int(evaluated_pairs),
            "future_metric_pair_coverage_ratio": float(evaluated_pairs / h),
            "future_policy": "forced_propagation; restart requests logged but ignored",
            "discount_policy": "original future_step exponent retained; invalid metric pairs skipped",
        })
    return (frame_rows, summaries), None


def main():
    ap = argparse.ArgumentParser(description="Multi-horizon B1/B2 state-writeback evaluation")
    ap.add_argument("--data-root", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--anchors", default="1", help="Zero-based anchor indices; must be >=1")
    ap.add_argument("--horizons", default="1,4,8,16")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/p1_counterfactual/multihorizon")
    args = ap.parse_args()

    anchors = parse_ints(args.anchors, "--anchors", 1)
    horizons = parse_ints(args.horizons, "--horizons", 1)
    if args.frames < 2 or not (0 < args.gamma <= 1) or args.max_time_diff <= 0 or args.gt_scale <= 0:
        ap.error("Invalid frames/gamma/max-time-diff/gt-scale value")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. Use the AutoDL RTX 4080 SUPER instance, not no-card mode.")

    root = Path(args.data_root)
    rgb_index = read_index(root, "rgb.txt", "rgb")
    depth_index = read_index(root, "depth.txt", "depth")
    selected = rgb_index[:args.frames]
    if len(selected) != args.frames:
        raise RuntimeError(f"Requested {args.frames} frames but found {len(selected)} valid RGB records")
    records = match_gt(selected, depth_index, args.max_time_diff)
    matched_count = sum(bool(row["gt_available"]) for row in records)
    unmatched_count = len(records) - matched_count
    max_needed = max(anchors) + max(horizons)
    if max_needed >= args.frames:
        raise ValueError(f"Need frame {max_needed}; --frames={args.frames} only covers 0..{args.frames-1}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== SVR-Depth multi-horizon counterfactual evaluation ===")
    print("Frames:", args.frames, "anchors:", anchors, "horizons:", horizons, "gamma:", args.gamma)
    print("GT timestamp tolerance (ms):", args.max_time_diff * 1000.0)
    print(f"GT matches: {matched_count}/{len(records)}; unmatched RGB frames retained: {unmatched_count}")
    print("Unmatched RGB frames remain in propagation; only their GT metrics are omitted.")
    print("Refresh: full-frame oracle; not local-tile refresh")
    print("Future policy: forced propagation; restart requests are logged but not executed")

    frames = load_frames(records)
    gt = [load_gt(x["gt_path"], args.gt_scale) if x["gt_available"] else None for x in records]
    H, W = int(frames.shape[-2]), int(frames.shape[-1])
    print("RGB tensor:", tuple(frames.shape), "first matched GT shape:",
          next((x.shape for x in gt if x is not None), None))
    matched_diffs = [x["time_diff"] for x in records if math.isfinite(x["time_diff"])]
    print("Max accepted RGB/GT timestamp difference (ms):",
          max(matched_diffs) * 1000.0 if matched_diffs else float("nan"))

    print("Loading frozen VeloDepth weights...")
    torch.set_grad_enabled(False)
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(torch.device("cuda"))
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    inputs, metas = capture_inputs(model, frames)
    t_total = int(metas[0]["T"])
    print(f"PASS: captured official packed inputs and flow (T={t_total})")
    paddings, (padded_h, padded_w) = get_paddings((H, W), model.shape_constraints["ratio_bounds"])
    padded_shape = (padded_h, padded_w)
    initial_snapshot = capture_state(model, StreamState())
    stream = VeloDepthStream(model)

    all_frames, all_summaries, skipped = [], [], []
    for anchor in anchors:
        print(f"\n--- Anchor {anchor} ---")
        result, skip = evaluate_anchor(
            model, stream, inputs, t_total, anchor, records, gt, initial_snapshot,
            paddings, padded_shape, horizons, args.gamma,
        )
        if skip is not None:
            print("SKIP:", skip["reason"])
            skipped.append(skip)
            continue
        frame_rows, summary_rows = result
        all_frames.extend(frame_rows)
        all_summaries.extend(summary_rows)
        print(f"G_inst(AbsRel)={summary_rows[0]['G_inst_absrel']:.6f}")
        for row in summary_rows:
            print(f"  H={row['horizon_H']:>2}: G_state(AbsRel)={row['G_state_absrel_discounted']:.6f}; "
                  f"mean B1={row['mean_future_absrel_B1']:.6f}, B2={row['mean_future_absrel_B2']:.6f}; "
                  f"GT={row['future_frames_with_gt']}/{row['future_frames_requested']}, "
                  f"metric_pairs={row['future_frames_evaluated']}/{row['future_frames_requested']}")

    frame_csv = out_dir / "multihorizon_frame_metrics.csv"
    summary_csv = out_dir / "multihorizon_summary.csv"
    skipped_csv = out_dir / "skipped_anchors.csv"
    metadata_json = out_dir / "run_metadata.json"
    write_csv(frame_csv, all_frames)
    write_csv(summary_csv, all_summaries)
    write_csv(skipped_csv, skipped, ["anchor_frame", "anchor_rgb_timestamp", "status", "reason"])
    metadata = {
        "experiment": "multi_horizon_counterfactual_state_writeback",
        "model": "lpiccinelli/velodepth", "weights_frozen": True,
        "device": str(next(model.parameters()).device), "torch_version": torch.__version__,
        "data_root": str(root), "frames": args.frames, "rgb_frames_loaded": len(records),
        "matched_gt_frames": matched_count, "unmatched_gt_frames": unmatched_count,
        "gt_match_ratio": float(matched_count / len(records)) if records else 0.0,
        "gt_match_policy": "nearest depth timestamp accepted only when abs(dt) <= max_time_diff; unmatched RGB frames retained",
        "anchors_zero_based": anchors,
        "horizons": horizons, "gamma": args.gamma, "gt_scale": args.gt_scale,
        "max_time_diff_seconds": args.max_time_diff, "resolution_level": args.resolution_level,
        "primary_loss": "AbsRel",
        "G_inst": "AbsRel(B0 propagation)-AbsRel(shared full-frame refresh)",
        "G_state": "sum_{k=1..H} gamma^k*(AbsRel(B2 no-writeback)-AbsRel(B1 writeback))",
        "B1_commit_fields": list(COMMIT_FIELDS), "scheduler_path": "synchronized from B2",
        "future_policy": "forced propagation; restart requests logged but ignored",
        "discount_policy": "missing GT metrics skipped without renumbering future-step discount exponents",
        "coverage_fields": ["future_frames_requested", "future_frames_with_gt", "future_gt_coverage_ratio", "future_frames_evaluated", "future_metric_pair_coverage_ratio"],
        "refresh_type": "full-frame oracle, not local-tile refresh",
        "outputs": {"frame_metrics": str(frame_csv), "summary": str(summary_csv), "skipped": str(skipped_csv)},
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Finished ===")
    print("Frame metrics:", frame_csv)
    print("Horizon summary:", summary_csv)
    print("Skipped anchors:", skipped_csv)
    print("Metadata:", metadata_json)
    print("Interpretation: positive G_inst means refresh lowers current AbsRel; positive G_state means B1 writeback lowers discounted future AbsRel.")


if __name__ == "__main__":
    main()
