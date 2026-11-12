#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SVR-Depth P1 error-distribution audit for real B1/B2 counterfactuals.

Runs the real frozen VeloDepth model on Bonn RGB-D, reconstructs each anchor's
prefix, then compares:
  anchor frame: B0 propagation vs shared full-frame refresh output;
  future frames: B1 (refresh state writeback) vs B2 (no refresh-state writeback).

Outputs per-frame/shared-mask metrics, GT-depth-bin metrics, spatial-grid metrics,
error-map PNGs, prediction NPZs, and run metadata. This is an audit/diagnostic,
not a training script and not a local-tile refresh or runtime benchmark.
"""
import os
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

from velodepth.models.velodepth import VeloDepth, get_paddings
from stream_state import StreamState, clone_state
from stream_state_snapshot import capture_state, restore_state
from velodepth_stream import VeloDepthStream
import p1_multihorizon_counterfactual as cf

COMMIT_FIELDS = ("features_teacher", "radius_teacher", "confidence_teacher", "rays_base")
SCHEDULER_FIELDS = ("current_t", "distance_kf", "flow_magnitude", "total_luma_change", "first_keyframe")


def parse_ints(text, name, minimum=1):
    try:
        values = sorted(set(int(x.strip()) for x in text.split(",") if x.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers") from exc
    if not values or any(x < minimum for x in values):
        raise argparse.ArgumentTypeError(f"{name} values must be >= {minimum}")
    return values


def clone_payload(x):
    return cf.clone_payload(x)


def sync_scheduler(dst, src):
    for name in SCHEDULER_FIELDS:
        setattr(dst, name, clone_payload(getattr(src, name)))


def prediction_depth(output, model, paddings, padded_shape):
    return cf.output_depth(output, model, paddings, padded_shape).astype(np.float32)


def valid_masks(pred_a, pred_b, gt):
    gt_valid = np.isfinite(gt) & (gt > 0)
    a_valid = np.isfinite(pred_a) & (pred_a > 0)
    b_valid = np.isfinite(pred_b) & (pred_b > 0)
    shared = gt_valid & a_valid & b_valid
    return gt_valid, a_valid, b_valid, shared


def metric_values(pred, gt, mask):
    n = int(mask.sum())
    if n == 0:
        return {"count": 0, "absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan"), "mae": float("nan")}
    p = pred[mask].astype(np.float64)
    g = gt[mask].astype(np.float64)
    e = p - g
    ratio = np.maximum(p / np.maximum(g, 1e-8), g / np.maximum(p, 1e-8))
    return {
        "count": n,
        "absrel": float(np.mean(np.abs(e) / np.maximum(g, 1e-8))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "mae": float(np.mean(np.abs(e))),
    }


def pair_metrics(pred_a, pred_b, gt):
    gt_valid, a_valid, b_valid, shared = valid_masks(pred_a, pred_b, gt)
    # The shared-mask metrics ensure A/B are compared on exactly the same pixels.
    ma = metric_values(pred_a, gt, shared)
    mb = metric_values(pred_b, gt, shared)
    # Independent coverage is reported separately; it must not silently alter A/B metrics.
    return {
        "gt_valid_pixels": int(gt_valid.sum()),
        "gt_valid_ratio": float(gt_valid.mean()),
        "a_valid_pixels": int((gt_valid & a_valid).sum()),
        "a_valid_ratio_on_gt": float((gt_valid & a_valid).sum() / max(int(gt_valid.sum()), 1)),
        "b_valid_pixels": int((gt_valid & b_valid).sum()),
        "b_valid_ratio_on_gt": float((gt_valid & b_valid).sum() / max(int(gt_valid.sum()), 1)),
        "shared_valid_pixels": int(shared.sum()),
        "shared_valid_ratio_on_gt": float(shared.sum() / max(int(gt_valid.sum()), 1)),
        "a": ma, "b": mb,
        "gain_absrel_b_minus_a": mb["absrel"] - ma["absrel"],
        "gain_rmse_b_minus_a": mb["rmse"] - ma["rmse"],
        "gain_delta1_a_minus_b": ma["delta1"] - mb["delta1"],
        "shared_mask": shared,
        "gt_valid_mask": gt_valid,
    }


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def flatten_pair_row(base, pair, a_name, b_name):
    row = dict(base)
    for label, side in ((a_name, pair["a"]), (b_name, pair["b"])):
        for metric in ("count", "absrel", "rmse", "delta1", "mae"):
            row[f"{label}_{metric}"] = side[metric]
    row.update({k: v for k, v in pair.items() if k not in ("a", "b", "shared_mask", "gt_valid_mask")})
    row[f"gain_absrel_{b_name}_minus_{a_name}"] = pair["gain_absrel_b_minus_a"]
    row[f"gain_rmse_{b_name}_minus_{a_name}"] = pair["gain_rmse_b_minus_a"]
    row[f"gain_delta1_{a_name}_minus_{b_name}"] = pair["gain_delta1_a_minus_b"]
    return row


def make_error_map(pred_a, pred_b, gt, shared, title_a, title_b, out_path):
    """Write a compact three-panel OpenCV visualization without matplotlib."""
    gt_valid = np.isfinite(gt) & (gt > 0)
    absrel_a = np.full(gt.shape, np.nan, dtype=np.float32)
    absrel_b = np.full(gt.shape, np.nan, dtype=np.float32)
    absrel_a[shared] = np.abs(pred_a[shared] - gt[shared]) / np.maximum(gt[shared], 1e-6)
    absrel_b[shared] = np.abs(pred_b[shared] - gt[shared]) / np.maximum(gt[shared], 1e-6)
    gain = absrel_b - absrel_a  # positive => A has lower error / is better
    vals = np.concatenate([absrel_a[shared], absrel_b[shared]]) if shared.any() else np.array([0.0])
    vmax = float(np.percentile(vals, 98)) if vals.size else 1.0
    vmax = max(vmax, 1e-6)

    def heat(arr, lo, hi, cmap=cv2.COLORMAP_TURBO):
        scaled = np.zeros(arr.shape, dtype=np.uint8)
        valid = np.isfinite(arr)
        scaled[valid] = np.clip((arr[valid] - lo) / max(hi - lo, 1e-8) * 255.0, 0, 255).astype(np.uint8)
        image = cv2.applyColorMap(scaled, cmap)
        image[~valid] = (40, 40, 40)
        return image

    a_img = heat(absrel_a, 0.0, vmax)
    b_img = heat(absrel_b, 0.0, vmax)
    # Signed gain: blue-ish negative, red-ish positive; zero at midpoint.
    gmax = float(np.percentile(np.abs(gain[shared]), 98)) if shared.any() else 1.0
    gmax = max(gmax, 1e-6)
    g_scaled = np.zeros(gt.shape, dtype=np.uint8)
    g_scaled[shared] = np.clip((gain[shared] / gmax * 127.0) + 128.0, 0, 255).astype(np.uint8)
    g_img = cv2.applyColorMap(g_scaled, cv2.COLORMAP_JET)
    g_img[~shared] = (40, 40, 40)

    panels = [(a_img, f"{title_a} AbsRel error"), (b_img, f"{title_b} AbsRel error"), (g_img, "B minus A error (positive=A better)")]
    rendered = []
    for image, label in panels:
        cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        rendered.append(image)
    canvas = cv2.hconcat(rendered)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"Could not write error map: {out_path}")


def add_depth_bin_rows(rows, base, pred_a, pred_b, gt, shared, edges, a_name, b_name):
    gt_valid = np.isfinite(gt) & (gt > 0)
    intervals = list(zip(edges[:-1], edges[1:])) + [(edges[-1], float("inf"))]
    for lo, hi in intervals:
        in_bin = gt_valid & (gt >= lo) & (gt < hi)
        mask = in_bin & shared
        ma = metric_values(pred_a, gt, mask)
        mb = metric_values(pred_b, gt, mask)
        denom = int(in_bin.sum())
        row = {
            **base,
            "depth_bin_m": f"[{lo:g},{hi:g})" if np.isfinite(hi) else f"[{lo:g},inf)",
            "depth_min_m": float(lo), "depth_max_m": float(hi) if np.isfinite(hi) else "inf",
            "gt_bin_pixels": denom,
            "shared_valid_pixels": int(mask.sum()),
            "shared_coverage_in_bin": float(mask.sum() / max(denom, 1)),
        }
        for name, m in ((a_name, ma), (b_name, mb)):
            for metric in ("count", "absrel", "rmse", "delta1", "mae"):
                row[f"{name}_{metric}"] = m[metric]
        row[f"gain_absrel_{b_name}_minus_{a_name}"] = mb["absrel"] - ma["absrel"]
        row[f"gain_rmse_{b_name}_minus_{a_name}"] = mb["rmse"] - ma["rmse"]
        row[f"gain_delta1_{a_name}_minus_{b_name}"] = ma["delta1"] - mb["delta1"]
        rows.append(row)


def add_spatial_rows(rows, base, pred_a, pred_b, gt, shared, grid):
    h, w = gt.shape
    err_a = np.abs(pred_a - gt) / np.maximum(gt, 1e-6)
    err_b = np.abs(pred_b - gt) / np.maximum(gt, 1e-6)
    for gy in range(grid):
        y0, y1 = round(gy * h / grid), round((gy + 1) * h / grid)
        for gx in range(grid):
            x0, x1 = round(gx * w / grid), round((gx + 1) * w / grid)
            mask = shared[y0:y1, x0:x1]
            if int(mask.sum()) == 0:
                a_mean = b_mean = gain = float("nan")
            else:
                a_mean = float(np.mean(err_a[y0:y1, x0:x1][mask]))
                b_mean = float(np.mean(err_b[y0:y1, x0:x1][mask]))
                gain = b_mean - a_mean
            rows.append({**base, "grid_rows": grid, "grid_cols": grid,
                         "grid_y": gy, "grid_x": gx, "x0": x0, "y0": y0,
                         "x1": x1, "y1": y1, "shared_valid_pixels": int(mask.sum()),
                         "mean_absrel_error_A": a_mean, "mean_absrel_error_B": b_mean,
                         "gain_B_minus_A": gain})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--anchors", default="1,8,15", help="Zero-based anchor indices")
    ap.add_argument("--horizon", type=int, default=16, help="Number of forced-propagation future frames")
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--depth-edges", default="0,1,1.5,2,2.5,3,3.5,4,5,10",
                    help="GT depth-bin edges in meters; last bin includes values >= last edge")
    ap.add_argument("--grid", type=int, default=8, help="Spatial grid resolution, e.g. 8 means 8x8")
    ap.add_argument("--out-dir", default="outputs/p1_counterfactual/error_audit")
    args = ap.parse_args()

    anchors = parse_ints(args.anchors, "--anchors", 1)
    try:
        edges = [float(x.strip()) for x in args.depth_edges.split(",")]
    except ValueError as exc:
        ap.error(f"Invalid --depth-edges: {exc}")
    if len(edges) < 2 or any(not np.isfinite(x) for x in edges) or any(b <= a for a, b in zip(edges, edges[1:])):
        ap.error("--depth-edges must be strictly increasing finite numbers, with at least two edges")
    if args.frames < 2 or args.horizon < 1 or args.grid < 1 or args.gt_scale <= 0 or args.max_time_diff <= 0:
        ap.error("Invalid frames/horizon/grid/scale/time-diff")
    if max(anchors) + args.horizon >= args.frames:
        ap.error(f"Need max(anchor)+horizon < frames; got {max(anchors)}+{args.horizon} >= {args.frames}")
    if not torch.cuda.is_available():
        raise RuntimeError("Real-model audit requires CUDA. Use the RTX 4080 SUPER instance, not no-card mode.")

    root = Path(args.data_root)
    rgb_rows = cf.read_index(root, "rgb.txt", "rgb")
    depth_rows = cf.read_index(root, "depth.txt", "depth")
    selected = rgb_rows[:args.frames]
    if len(selected) != args.frames:
        raise RuntimeError(f"Requested {args.frames} RGB frames, found {len(selected)}")
    records = cf.match_gt(selected, depth_rows, args.max_time_diff)
    frames = cf.load_frames(records)
    gt = [cf.load_gt(r["gt_path"], args.gt_scale) for r in records]
    h_img, w_img = gt[0].shape

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    map_dir = out_dir / "error_maps"
    pred_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    print("=== SVR-Depth P1 error-distribution audit ===")
    print(f"device=cuda; frames={args.frames}; anchors={anchors}; horizon={args.horizon}; grid={args.grid}x{args.grid}")
    print(f"depth bins={edges}; max RGB/GT time diff={max(r['time_diff'] for r in records)*1000:.3f} ms")
    print("metrics use a shared valid-pixel mask for A/B comparison; coverage is reported separately")

    torch.set_grad_enabled(False)
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(torch.device("cuda")); model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"
    inputs, metas = cf.capture_inputs(model, frames)
    t_total = int(metas[0]["T"])
    paddings, (padded_h, padded_w) = get_paddings((int(frames.shape[-2]), int(frames.shape[-1])), model.shape_constraints["ratio_bounds"])
    padded_shape = (padded_h, padded_w)
    initial_snapshot = capture_state(model, StreamState())
    stream = VeloDepthStream(model)

    frame_rows, depth_rows_out, spatial_rows, anchor_rows, skipped = [], [], [], [], []
    for anchor in anchors:
        print(f"\n--- anchor {anchor} ---")
        state_before = cf.prefix_state(model, stream, inputs, t_total, anchor, initial_snapshot)
        pre = capture_state(model, state_before)
        b0_state = clone_state(state_before)
        restore_state(model, b0_state, pre)
        out_b0, b0_state, restart = stream.propagate_step(inputs, anchor, b0_state, t_total)
        if restart:
            skipped.append({"anchor_frame": anchor, "status": "skipped_anchor_restart",
                            "reason": "B0 propagation requested restart; normal policy would keyframe this frame"})
            print("SKIP: B0 requested keyframe restart at anchor")
            continue
        post_prop_snapshot = capture_state(model, b0_state)
        refresh_state = clone_state(state_before)
        restore_state(model, refresh_state, pre)
        out_refresh, refresh_state = stream.keyframe_step(inputs, anchor, refresh_state, t_total)
        b1_state = clone_state(b0_state)
        for name in COMMIT_FIELDS:
            value = getattr(refresh_state, name)
            if value is None:
                raise RuntimeError(f"Refresh state field {name} is None")
            setattr(b1_state, name, clone_payload(value))
        b2_state = clone_state(b0_state)

        # Audit anchor-frame B0 vs shared full-frame refresh.
        pred_b0 = prediction_depth(out_b0, model, paddings, padded_shape)
        pred_ref = prediction_depth(out_refresh, model, paddings, padded_shape)
        gt_anchor = gt[anchor]
        anchor_pair = pair_metrics(pred_b0, pred_ref, gt_anchor)
        anchor_base = {"phase": "anchor", "anchor_frame": anchor, "future_step": 0,
                       "frame": anchor, "rgb_timestamp": records[anchor]["rgb_ts"],
                       "gt_timestamp": records[anchor]["gt_ts"],
                       "time_diff_ms": records[anchor]["time_diff"] * 1000.0}
        anchor_rows.append(flatten_pair_row(anchor_base, anchor_pair, "B0_propagation", "shared_refresh"))
        np.savez_compressed(pred_dir / f"anchor_{anchor:03d}_frame_{anchor:03d}.npz",
                            pred_a=pred_b0, pred_b=pred_ref, gt=gt_anchor)
        make_error_map(pred_b0, pred_ref, gt_anchor, anchor_pair["shared_mask"], "B0", "Refresh",
                       map_dir / f"anchor_{anchor:03d}_frame_{anchor:03d}_B0_vs_refresh.png")
        add_depth_bin_rows(depth_rows_out, {**anchor_base, "comparison": "B0_vs_refresh"}, pred_b0, pred_ref,
                           gt_anchor, anchor_pair["shared_mask"], edges, "B0", "refresh")
        add_spatial_rows(spatial_rows, {**anchor_base, "comparison": "B0_vs_refresh"}, pred_b0, pred_ref,
                         gt_anchor, anchor_pair["shared_mask"], args.grid)

        for k in range(1, args.horizon + 1):
            frame_idx = anchor + k
            sync_scheduler(b1_state, b2_state)
            tmp = clone_state(post_prop_snapshot.stream_state)
            restore_state(model, tmp, post_prop_snapshot)
            out1, b1_state, restart1 = stream.propagate_step(inputs, frame_idx, b1_state, t_total)
            tmp = clone_state(post_prop_snapshot.stream_state)
            restore_state(model, tmp, post_prop_snapshot)
            out2, b2_state, restart2 = stream.propagate_step(inputs, frame_idx, b2_state, t_total)
            sync_scheduler(b1_state, b2_state)

            pred1 = prediction_depth(out1, model, paddings, padded_shape)
            pred2 = prediction_depth(out2, model, paddings, padded_shape)
            gt_frame = gt[frame_idx]
            pair = pair_metrics(pred1, pred2, gt_frame)
            base = {"phase": "future", "anchor_frame": anchor, "future_step": k,
                    "frame": frame_idx, "rgb_timestamp": records[frame_idx]["rgb_ts"],
                    "gt_timestamp": records[frame_idx]["gt_ts"],
                    "time_diff_ms": records[frame_idx]["time_diff"] * 1000.0,
                    "restart_requested_B1_ignored": bool(restart1),
                    "restart_requested_B2_ignored": bool(restart2)}
            frame_rows.append(flatten_pair_row(base, pair, "B1_writeback", "B2_no_writeback"))
            add_depth_bin_rows(depth_rows_out, {**base, "comparison": "B1_vs_B2"}, pred1, pred2,
                               gt_frame, pair["shared_mask"], edges, "B1", "B2")
            add_spatial_rows(spatial_rows, {**base, "comparison": "B1_vs_B2"}, pred1, pred2,
                             gt_frame, pair["shared_mask"], args.grid)
            np.savez_compressed(pred_dir / f"anchor_{anchor:03d}_frame_{frame_idx:03d}.npz",
                                pred_b1=pred1, pred_b2=pred2, gt=gt_frame,
                                shared_valid=pair["shared_mask"].astype(np.uint8))
            make_error_map(pred1, pred2, gt_frame, pair["shared_mask"], "B1 writeback", "B2 no-writeback",
                           map_dir / f"anchor_{anchor:03d}_frame_{frame_idx:03d}_B1_vs_B2.png")
        print(f"anchor {anchor}: anchor and {args.horizon} future frames audited")

    csv_write(out_dir / "anchor_metrics.csv", anchor_rows)
    csv_write(out_dir / "future_frame_metrics.csv", frame_rows)
    csv_write(out_dir / "depth_bin_metrics.csv", depth_rows_out)
    csv_write(out_dir / "spatial_grid_metrics.csv", spatial_rows)
    csv_write(out_dir / "skipped_anchors.csv", skipped)
    metadata = {
        "experiment": "p1_error_distribution_audit", "model": "lpiccinelli/velodepth",
        "weights_frozen": True, "device": "cuda", "frames": args.frames,
        "anchors_zero_based": anchors, "horizon": args.horizon, "depth_edges_m": edges,
        "spatial_grid": [args.grid, args.grid], "gt_scale": args.gt_scale,
        "max_time_diff_seconds": args.max_time_diff, "resolution_level": args.resolution_level,
        "comparison_mask": "shared GT-valid and positive finite predictions in both compared branches",
        "coverage_reported_separately": True,
        "anchor_comparison": "B0 propagation vs shared full-frame refresh output",
        "future_comparison": "B1 refresh state writeback vs B2 no refresh-state writeback; forced propagation; restart requests ignored",
        "refresh_type": "full-frame oracle; not local-tile refresh",
        "outputs": ["anchor_metrics.csv", "future_frame_metrics.csv", "depth_bin_metrics.csv",
                    "spatial_grid_metrics.csv", "predictions/*.npz", "error_maps/*.png", "skipped_anchors.csv"]
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Audit complete ===")
    print("Output directory:", out_dir)
    print("Anchor rows:", len(anchor_rows), "future rows:", len(frame_rows),
          "depth-bin rows:", len(depth_rows_out), "spatial rows:", len(spatial_rows))
    print("Key files: anchor_metrics.csv, future_frame_metrics.csv, depth_bin_metrics.csv, spatial_grid_metrics.csv")


if __name__ == "__main__":
    main()
