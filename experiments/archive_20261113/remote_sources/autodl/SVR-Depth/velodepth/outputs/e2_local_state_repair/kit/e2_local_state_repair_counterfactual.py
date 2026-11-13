#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2-S0: one-anchor local persistent-state repair counterfactual.

Diagnostic only:
- frozen VeloDepth weights
- one full-frame teacher encoding at the anchor, conditioned on the current
  propagation ray field
- replace one fixed tile of the propagated neck with teacher-conditioned
  features
- share the same repaired current output between output-only and coherent
  state-commit branches
- force propagation for the future window and evaluate paired GT metrics

This is not the final learned/low-cost method. The full-frame teacher is an
oracle donor used to test whether local persistent repair has exploitable value.
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
from pathlib import Path

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
    "current_t",
    "distance_kf",
    "flow_magnitude",
    "total_luma_change",
    "first_keyframe",
)


def sync_scheduler(dst, src):
    for name in SCHEDULER_FIELDS:
        setattr(dst, name, getattr(src, name))


def restore_runtime(model, snapshot):
    # Restore only model runtime attributes via a temporary state object;
    # the live branch state remains independent.
    temp = clone_state(snapshot.stream_state)
    restore_state(model, temp, snapshot)


def tile_name(tile_id):
    return ("top_left", "top_right", "bottom_left", "bottom_right")[tile_id]


def build_teacher_conditioned_features(model, inputs, stream, frame_idx, total_frames, rays_base):
    """Encode the current RGB frame and condition the neck on the existing stream rays."""
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
    teacher_features = [
        x.reshape(x.shape[0], h_common, w_common, x.shape[-1])
         .permute(0, 3, 1, 2).contiguous()
        for x in conditioned
    ]
    return teacher_features


def replace_one_tile(propagated_features, teacher_features, tile_id):
    """Replace the same normalized 2x2-grid tile at every feature level."""
    if len(propagated_features) != len(teacher_features):
        raise RuntimeError(
            f"Feature-level count mismatch: propagated={len(propagated_features)}, "
            f"teacher={len(teacher_features)}"
        )
    repaired = []
    tile_rows = []
    for level, (base, donor) in enumerate(zip(propagated_features, teacher_features)):
        if tuple(base.shape) != tuple(donor.shape):
            raise RuntimeError(
                f"Level {level} shape mismatch: propagated={tuple(base.shape)}, "
                f"teacher={tuple(donor.shape)}"
            )
        h, w = base.shape[-2:]
        row = tile_id // 2
        col = tile_id % 2
        y0, y1 = row * h // 2, (row + 1) * h // 2
        x0, x1 = col * w // 2, (col + 1) * w // 2
        if y1 <= y0 or x1 <= x0:
            raise RuntimeError(f"Empty tile at level {level}: {(y0, y1, x0, x1)}")
        out = base.detach().clone()
        out[..., y0:y1, x0:x1] = donor[..., y0:y1, x0:x1].to(
            device=out.device, dtype=out.dtype
        )
        repaired.append(out)
        tile_rows.append({
            "level": level, "height": h, "width": w,
            "y0": y0, "y1": y1, "x0": x0, "x1": x1,
            "replaced_fraction": float(((y1-y0) * (x1-x0)) / (h*w)),
        })
    return repaired, tile_rows


def max_abs_diff_list(a, b):
    if len(a) != len(b):
        return float("inf")
    vals = []
    for x, y in zip(a, b):
        if x.shape != y.shape:
            return float("inf")
        vals.append(float((x.detach().float() - y.detach().float()).abs().max().item()))
    return max(vals, default=0.0)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="E2-S0 local state repair counterfactual smoke")
    ap.add_argument("--data-root", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2")
    ap.add_argument("--frames", type=int, default=18)
    ap.add_argument("--anchor", type=int, default=9, help="Zero-based frame index")
    ap.add_argument("--horizons", default="8", help="Comma-separated future horizons")
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/e2_local_state_repair/smoke_crowd2_a9_h8")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    args = ap.parse_args()

    horizons = mh.parse_ints(args.horizons, "--horizons", 1)
    if args.anchor < 1 or args.frames < 3:
        ap.error("--anchor must be >=1 and --frames must be >=3")
    if args.anchor + max(horizons) >= args.frames:
        ap.error(
            f"Need frame {args.anchor + max(horizons)}; --frames={args.frames} "
            f"covers only 0..{args.frames-1}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required. Do not run E2-S0 in no-card mode.")

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
    matched = sum(row["gt_available"] for row in records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== E2-S0 local state repair counterfactual ===")
    print(f"Frames={args.frames}; anchor={args.anchor}; horizons={horizons}")
    print(f"GT matched={matched}/{len(records)}; tolerance={args.max_time_diff*1000:.2f} ms")
    print("Repair source: full-frame teacher-conditioned features; one 2x2 tile is copied.")
    print("This is an oracle mechanism test, not a low-cost online method.")

    frames = mh.load_frames(records)
    height, width = int(frames.shape[-2]), int(frames.shape[-1])
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(torch.device("cuda"))
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    inputs, metas = mh.capture_inputs(model, frames)
    total_frames = int(metas[0]["T"])
    paddings, (padded_h, padded_w) = get_paddings(
        (height, width), model.shape_constraints["ratio_bounds"]
    )
    padded_shape = (padded_h, padded_w)
    stream = VeloDepthStream(model)
    initial_snapshot = capture_state(model, StreamState())

    state_before = mh.prefix_state(
        model, stream, inputs, total_frames, args.anchor, initial_snapshot
    )
    pre_anchor_snapshot = capture_state(model, state_before)
    prop_state = clone_state(state_before)
    restore_state(model, prop_state, pre_anchor_snapshot)
    with torch.inference_mode():
        prop_out, prop_state, restart_at_anchor = stream.propagate_step(
            inputs, args.anchor, prop_state, total_frames
        )
    if restart_at_anchor:
        row = {
            "anchor_frame": args.anchor,
            "status": "skipped_anchor_restart",
            "reason": "Native propagation requests a keyframe at the selected anchor.",
        }
        write_csv(out_dir / "e2_skipped_anchors.csv", [row])
        print("SKIP:", row["reason"])
        return

    tile_id = args.anchor % 4
    torch.cuda.synchronize()
    teacher_start = time.perf_counter()
    with torch.inference_mode():
        teacher_features = build_teacher_conditioned_features(
            model, inputs, stream, args.anchor, total_frames, prop_state.rays_base
        )
    torch.cuda.synchronize()
    teacher_ms = (time.perf_counter() - teacher_start) * 1000.0

    repaired_features, tile_rows = replace_one_tile(
        prop_state.features_teacher, teacher_features, tile_id
    )
    feature_delta = max_abs_diff_list(prop_state.features_teacher, repaired_features)

    torch.cuda.synchronize()
    readout_start = time.perf_counter()
    with torch.inference_mode():
        repair_radius, repair_confidence = model.pixel_decoder.readout_fast_features(
            repaired_features
        )
    torch.cuda.synchronize()
    readout_ms = (time.perf_counter() - readout_start) * 1000.0

    rays = stream._rays_bchw(prop_state, height, width)
    local_out = StepOutput(
        radius=repair_radius,
        rays=rays.detach().clone(),
        confidence=repair_confidence,
        kind="local_repair_diagnostic",
        frame=args.anchor,
    )
    metric_prop = mh.eval_output(prop_out, model, paddings, padded_shape, gt[args.anchor])
    metric_local = mh.eval_output(local_out, model, paddings, padded_shape, gt[args.anchor])

    # Both branches emit this exact same current-frame local-repair output.
    output_only_state = clone_state(prop_state)
    coherent_state = clone_state(prop_state)
    coherent_state.features_teacher = [x.detach().clone() for x in repaired_features]
    coherent_state.radius_teacher = repair_radius.detach().clone()
    coherent_state.confidence_teacher = repair_confidence.detach().clone()
    # Keep next-keyframe overlap cache consistent with the committed geometry.
    coherent_state.fast_tail_features = [x.detach().clone() for x in repaired_features]
    coherent_state.fast_tail_radius = repair_radius.detach().clone()
    coherent_state.fast_tail_confidence = repair_confidence.detach().clone()
    # rays_base and scheduler fields deliberately retain the native propagated path.

    # Snapshot mutable decoder attributes after the anchor operation; future branches
    # restore these attributes before every step while keeping branch states separate.
    post_anchor_runtime = capture_state(model, prop_state)

    anchor_row = {
        "anchor_frame": args.anchor,
        "anchor_rgb_timestamp": records[args.anchor]["rgb_ts"],
        "anchor_gt_available": bool(records[args.anchor]["gt_available"]),
        "anchor_gt_time_diff_ms": records[args.anchor]["time_diff"] * 1000.0
            if math.isfinite(records[args.anchor]["time_diff"]) else float("nan"),
        "tile_id": tile_id,
        "tile_name": tile_name(tile_id),
        "repair_levels": len(repaired_features),
        "feature_max_abs_change": feature_delta,
        "absrel_PROP": metric_prop["absrel"],
        "absrel_LOCAL_REPAIR_CURRENT": metric_local["absrel"],
        "G_inst_absrel_PROP_minus_LOCAL": metric_prop["absrel"] - metric_local["absrel"],
        "rmse_PROP": metric_prop["rmse"],
        "rmse_LOCAL_REPAIR_CURRENT": metric_local["rmse"],
        "delta1_PROP": metric_prop["delta1"],
        "delta1_LOCAL_REPAIR_CURRENT": metric_local["delta1"],
        "full_teacher_feature_generation_ms": teacher_ms,
        "fast_readout_ms": readout_ms,
        "current_output_shared_between_future_branches": True,
        "restart_at_anchor": False,
    }
    write_csv(out_dir / "e2_anchor_effects.csv", [anchor_row])
    write_csv(out_dir / "e2_tile_geometry.csv", tile_rows)

    future_rows = []
    max_h = max(horizons)
    for step in range(1, max_h + 1):
        frame_idx = args.anchor + step
        sync_scheduler(coherent_state, output_only_state)

        restore_runtime(model, post_anchor_runtime)
        with torch.inference_mode():
            out_only, output_only_state, restart_only = stream.propagate_step(
                inputs, frame_idx, output_only_state, total_frames
            )

        restore_runtime(model, post_anchor_runtime)
        with torch.inference_mode():
            out_coherent, coherent_state, restart_coherent = stream.propagate_step(
                inputs, frame_idx, coherent_state, total_frames
            )
        sync_scheduler(coherent_state, output_only_state)

        metric_only = mh.eval_output(
            out_only, model, paddings, padded_shape, gt[frame_idx]
        )
        metric_coherent = mh.eval_output(
            out_coherent, model, paddings, padded_shape, gt[frame_idx]
        )
        future_rows.append({
            "anchor_frame": args.anchor,
            "future_step": step,
            "future_frame": frame_idx,
            "future_rgb_timestamp": records[frame_idx]["rgb_ts"],
            "future_gt_available": bool(records[frame_idx]["gt_available"]),
            "future_gt_time_diff_ms": records[frame_idx]["time_diff"] * 1000.0
                if math.isfinite(records[frame_idx]["time_diff"]) else float("nan"),
            "absrel_OUTPUT_ONLY": metric_only["absrel"],
            "absrel_COHERENT_COMMIT": metric_coherent["absrel"],
            "G_state_absrel_OUTPUT_ONLY_minus_COMMIT":
                metric_only["absrel"] - metric_coherent["absrel"],
            "rmse_OUTPUT_ONLY": metric_only["rmse"],
            "rmse_COHERENT_COMMIT": metric_coherent["rmse"],
            "G_state_rmse_OUTPUT_ONLY_minus_COMMIT":
                metric_only["rmse"] - metric_coherent["rmse"],
            "delta1_OUTPUT_ONLY": metric_only["delta1"],
            "delta1_COHERENT_COMMIT": metric_coherent["delta1"],
            "G_state_delta1_COMMIT_minus_OUTPUT_ONLY":
                metric_coherent["delta1"] - metric_only["delta1"],
            "valid_ratio_OUTPUT_ONLY": metric_only["valid_ratio"],
            "valid_ratio_COHERENT_COMMIT": metric_coherent["valid_ratio"],
            "restart_requested_OUTPUT_ONLY_ignored": bool(restart_only),
            "restart_requested_COHERENT_ignored": bool(restart_coherent),
            "future_policy": "forced propagation; restart requests logged but ignored",
        })
        print(
            f"future={frame_idx} H={step}: "
            f"AbsRel output-only={metric_only['absrel']:.6f}, "
            f"coherent={metric_coherent['absrel']:.6f}, "
            f"gain={metric_only['absrel']-metric_coherent['absrel']:+.6f}, "
            f"GT={records[frame_idx]['gt_available']}"
        )

    summaries = []
    for h in horizons:
        rows = future_rows[:h]
        pairs = [
            row for row in rows
            if math.isfinite(row["absrel_OUTPUT_ONLY"])
            and math.isfinite(row["absrel_COHERENT_COMMIT"])
        ]
        absrel_only = [row["absrel_OUTPUT_ONLY"] for row in pairs]
        absrel_commit = [row["absrel_COHERENT_COMMIT"] for row in pairs]
        rmse_only = [row["rmse_OUTPUT_ONLY"] for row in pairs]
        rmse_commit = [row["rmse_COHERENT_COMMIT"] for row in pairs]
        delta_only = [row["delta1_OUTPUT_ONLY"] for row in pairs]
        delta_commit = [row["delta1_COHERENT_COMMIT"] for row in pairs]
        summaries.append({
            "anchor_frame": args.anchor,
            "tile_id": tile_id,
            "tile_name": tile_name(tile_id),
            "horizon_H": h,
            "future_frames_requested": h,
            "future_frames_with_gt": sum(bool(records[args.anchor+k]["gt_available"])
                                         for k in range(1, h+1)),
            "future_metric_pairs": len(pairs),
            "mean_future_absrel_OUTPUT_ONLY": sum(absrel_only)/len(absrel_only)
                if absrel_only else float("nan"),
            "mean_future_absrel_COHERENT_COMMIT": sum(absrel_commit)/len(absrel_commit)
                if absrel_commit else float("nan"),
            "G_state_absrel_mean": sum(
                row["absrel_OUTPUT_ONLY"] - row["absrel_COHERENT_COMMIT"]
                for row in pairs
            ) / len(pairs) if pairs else float("nan"),
            "mean_future_rmse_OUTPUT_ONLY": sum(rmse_only)/len(rmse_only)
                if rmse_only else float("nan"),
            "mean_future_rmse_COHERENT_COMMIT": sum(rmse_commit)/len(rmse_commit)
                if rmse_commit else float("nan"),
            "G_state_rmse_mean": sum(
                row["rmse_OUTPUT_ONLY"] - row["rmse_COHERENT_COMMIT"]
                for row in pairs
            ) / len(pairs) if pairs else float("nan"),
            "mean_future_delta1_OUTPUT_ONLY": sum(delta_only)/len(delta_only)
                if delta_only else float("nan"),
            "mean_future_delta1_COHERENT_COMMIT": sum(delta_commit)/len(delta_commit)
                if delta_commit else float("nan"),
            "G_state_delta1_mean": sum(
                row["delta1_COHERENT_COMMIT"] - row["delta1_OUTPUT_ONLY"]
                for row in pairs
            ) / len(pairs) if pairs else float("nan"),
            "full_teacher_feature_generation_ms": teacher_ms,
            "fast_readout_ms": readout_ms,
            "interpretation": "positive G_state means coherent commit improves future GT metric",
        })

    write_csv(out_dir / "e2_future_frame_metrics.csv", future_rows)
    write_csv(out_dir / "e2_horizon_summary.csv", summaries)
    metadata = {
        "experiment": "E2-S0_local_state_repair_counterfactual",
        "model": "lpiccinelli/velodepth",
        "weights_frozen": True,
        "data_root": str(root),
        "frames": args.frames,
        "anchor_zero_based": args.anchor,
        "horizons": horizons,
        "tile_id": tile_id,
        "tile_name": tile_name(tile_id),
        "tile_policy": "tile_id = anchor % 4; normalized 2x2 grid copied at every feature level",
        "teacher_policy": "full-frame pixel encoder + camera conditioning using the current propagation rays_base",
        "state_commit_fields": [
            "features_teacher", "radius_teacher", "confidence_teacher",
            "fast_tail_features", "fast_tail_radius", "fast_tail_confidence"
        ],
        "unchanged_fields": ["rays_base", *SCHEDULER_FIELDS],
        "future_policy": "forced propagation; restart requests logged but ignored",
        "gt_match_tolerance_seconds": args.max_time_diff,
        "gt_scale": args.gt_scale,
        "resolution_level": args.resolution_level,
        "teacher_feature_generation_ms": teacher_ms,
        "fast_readout_ms": readout_ms,
        "limitations": [
            "single anchor and one tile; smoke test only",
            "teacher donor uses full-frame encoding and is not a low-cost online executor",
            "future-window restart requests are logged but not executed",
            "positive teacher-reference alignment is not a substitute for GT metric improvement"
        ],
        "outputs": {
            "anchor_effects": str(out_dir / "e2_anchor_effects.csv"),
            "tile_geometry": str(out_dir / "e2_tile_geometry.csv"),
            "future_frame_metrics": str(out_dir / "e2_future_frame_metrics.csv"),
            "horizon_summary": str(out_dir / "e2_horizon_summary.csv"),
        },
    }
    (out_dir / "e2_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== E2-S0 finished ===")
    print("Anchor effects:", out_dir / "e2_anchor_effects.csv")
    print("Future metrics:", out_dir / "e2_future_frame_metrics.csv")
    print("Horizon summary:", out_dir / "e2_horizon_summary.csv")
    print("Metadata:", out_dir / "e2_metadata.json")
    print("Positive G_state means coherent state commit lowered future GT error.")
    print("This smoke is not sufficient evidence for generalization or deployment cost.")


if __name__ == "__main__":
    main()
