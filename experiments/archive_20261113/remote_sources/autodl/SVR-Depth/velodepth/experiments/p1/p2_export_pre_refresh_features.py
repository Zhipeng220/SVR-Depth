#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export pre-refresh, online-observable features for SVR-Depth P2.1.

This script:
- reuses the already-tested VeloDepth streaming helpers from
  experiments/p1/p1_multihorizon_counterfactual.py;
- replays the stream to each anchor and summarizes the persistent state
  BEFORE any refresh is performed at that anchor;
- never reads future GT or refresh outputs as model inputs;
- writes one row per anchor. Join labels from the existing P1.5
  multihorizon_summary.csv by sequence + anchor_frame + horizon_H.

Run on the GPU AutoDL instance, not the 0.5-core / 2-GB no-card instance.
"""
import os
if not os.environ.get("OMP_NUM_THREADS", "").isdigit() or int(os.environ.get("OMP_NUM_THREADS", "0")) < 1:
    os.environ["OMP_NUM_THREADS"] = "4"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

import p1_multihorizon_counterfactual as cf
from stream_state import StreamState
from stream_state_snapshot import capture_state, restore_state
from velodepth_stream import VeloDepthStream
from velodepth.models.velodepth import VeloDepth


def scalar_value(x):
    if x is None:
        return float("nan")
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        if x.numel() == 1:
            return float(x.detach().float().cpu().item())
        return float(x.detach().float().mean().cpu().item())
    if isinstance(x, (bool, int, float, np.number)):
        return float(x)
    return float("nan")


def finite_tensor_sample(x, max_samples=12000):
    """Small deterministic sample for robust distribution summaries."""
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return None
    y = x.detach().reshape(-1)
    # Convert only the sampled values to float to limit memory traffic.
    stride = max(1, int(math.ceil(y.numel() / max_samples)))
    y = y[::stride].float()
    y = y[torch.isfinite(y)]
    if y.numel() == 0:
        return None
    return y


def write_tensor_stats(row, prefix, value):
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            write_tensor_stats(row, f"{prefix}_{i}", item)
        return
    if not isinstance(value, torch.Tensor):
        row[f"{prefix}_mean"] = scalar_value(value)
        return
    sample = finite_tensor_sample(value)
    if sample is None:
        for suffix in ("mean", "std", "min", "max", "q10", "q50", "q90"):
            row[f"{prefix}_{suffix}"] = float("nan")
        return
    q = torch.quantile(sample, torch.tensor([0.10, 0.50, 0.90], device=sample.device))
    row[f"{prefix}_mean"] = float(sample.mean().cpu())
    row[f"{prefix}_std"] = float(sample.std(unbiased=False).cpu())
    row[f"{prefix}_min"] = float(sample.min().cpu())
    row[f"{prefix}_max"] = float(sample.max().cpu())
    row[f"{prefix}_q10"] = float(q[0].cpu())
    row[f"{prefix}_q50"] = float(q[1].cpu())
    row[f"{prefix}_q90"] = float(q[2].cpu())


def image_stats(row, frame_rgb, prev_rgb):
    # These are inexpensive current-frame observations, not GT or future data.
    arr = frame_rgb.detach().cpu().numpy().astype(np.float32)
    for c, name in enumerate(("r", "g", "b")):
        row[f"image_{name}_mean"] = float(arr[c].mean())
        row[f"image_{name}_std"] = float(arr[c].std())
    if prev_rgb is not None:
        prev = prev_rgb.detach().cpu().numpy().astype(np.float32)
        diff = np.abs(arr - prev)
        row["image_absdiff_mean_prev"] = float(diff.mean())
        row["image_absdiff_p90_prev"] = float(np.quantile(diff, 0.90))
    else:
        row["image_absdiff_mean_prev"] = float("nan")
        row["image_absdiff_p90_prev"] = float("nan")


def extract_state_features(state, row):
    # Scheduler / temporal-age features available before the anchor action.
    for name in (
        "current_t", "distance_kf", "flow_magnitude",
        "total_luma_change", "first_keyframe"
    ):
        row[f"scheduler_{name}"] = scalar_value(getattr(state, name, None))

    # Persistent geometry / confidence state. These are pre-refresh values.
    for name in ("radius_teacher", "confidence_teacher", "rays_base", "features_teacher"):
        write_tensor_stats(row, f"state_{name}", getattr(state, name, None))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No feature rows were produced.")
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Export pre-refresh features for SVR-Depth P2.1")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--sequence", required=True, help="e.g. crowd2, balloon2, crowd3")
    ap.add_argument("--frames", type=int, default=86)
    ap.add_argument("--anchors", default="1,5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69")
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for real VeloDepth feature extraction. Use the GPU AutoDL instance.")

    anchors = cf.parse_ints(args.anchors, "--anchors", 1)
    root = Path(args.data_root)
    rgb_index = cf.read_index(root, "rgb.txt", "rgb")
    selected = rgb_index[:args.frames]
    if len(selected) != args.frames:
        raise RuntimeError(f"Requested {args.frames} frames, found {len(selected)} RGB entries.")
    # Labels/GT are deliberately not loaded here: this script exports inputs only.
    records = [
        {"frame_index": i, "rgb_ts": ts, "rgb_path": path}
        for i, (ts, path) in enumerate(selected)
    ]
    max_anchor = max(anchors)
    if max_anchor >= args.frames:
        raise ValueError(f"Anchor {max_anchor} exceeds selected frame range 0..{args.frames-1}")

    frames = cf.load_frames(records)
    print("RGB frames:", tuple(frames.shape))
    print("Loading frozen VeloDepth weights...")
    torch.set_grad_enabled(False)
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(torch.device("cuda"))
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    inputs, metas = cf.capture_inputs(model, frames)
    t_total = int(metas[0]["T"])
    initial_snapshot = capture_state(model, StreamState())
    stream = VeloDepthStream(model)

    rows = []
    for anchor in anchors:
        if anchor >= args.frames:
            print(f"SKIP anchor {anchor}: out of frame range")
            continue
        print(f"Extracting pre-refresh features at anchor {anchor}...")
        state_before = cf.prefix_state(
            model, stream, inputs, t_total, anchor, initial_snapshot
        )
        row = {
            "sequence": args.sequence,
            "anchor_frame": int(anchor),
            "anchor_rgb_timestamp": float(records[anchor]["rgb_ts"]),
            "feature_timing": "before_anchor_action_and_before_refresh",
            "label_source": "join_from_existing_P1.5_summary_csv",
            "uses_future_gt_as_input": False,
            "uses_refresh_output_as_input": False,
        }
        extract_state_features(state_before, row)
        image_stats(row, frames[anchor], frames[anchor - 1] if anchor > 0 else None)
        rows.append(row)

    out_path = Path(args.out) if args.out else (
        ROOT / "outputs" / "p2_prediction" / f"pre_refresh_features_{args.sequence}.csv"
    )
    write_rows(out_path, rows)
    print("\n=== Feature export finished ===")
    print("Sequence:", args.sequence)
    print("Rows:", len(rows))
    print("Output:", out_path)
    print("Reminder: these are features only; no predictor has been trained.")
    print("Join labels by sequence + anchor_frame, keeping horizon_H as a separate target dimension.")


if __name__ == "__main__":
    main()
