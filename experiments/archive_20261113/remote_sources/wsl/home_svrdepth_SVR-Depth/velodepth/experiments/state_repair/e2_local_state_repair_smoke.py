#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2 single-anchor local state-repair counterfactual smoke.

Frozen-model diagnostic only. Uses a full-frame teacher neck as an ORACLE donor,
then replaces one fixed 2x2 tile in the propagated neck. It compares future GT
metrics under forced propagation; restart requests are logged but not executed.
This is not a deployable local-compute method and does not train any weights.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import csv
import copy
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path
from types import SimpleNamespace

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
        raise RuntimeError(f"No valid entries in {index}")
    return rows


def match_gt(rgb_rows, depth_rows, max_diff):
    depth_ts = [x[0] for x in depth_rows]
    result = []
    for idx, (ts, path) in enumerate(rgb_rows):
        j = bisect_left(depth_ts, ts)
        candidates = []
        if j < len(depth_rows):
            candidates.append(depth_rows[j])
        if j > 0:
            candidates.append(depth_rows[j - 1])
        nearest = min(candidates, key=lambda x: abs(x[0] - ts)) if candidates else None
        dt = abs(nearest[0] - ts) if nearest else float("inf")
        matched = nearest is not None and dt <= max_diff
        result.append({
            "frame_index": idx, "rgb_ts": ts, "rgb_path": path,
            "gt_ts": nearest[0] if matched else float("nan"),
            "gt_path": nearest[1] if matched else None,
            "time_diff": dt if matched else float("nan"),
            "gt_available": bool(matched),
            "nearest_time_diff": dt,
            "gt_match_reason": "matched" if matched else "outside_tolerance",
        })
    return result


def load_frames(records):
    imgs = []
    for row in records:
        bgr = cv2.imread(str(row["rgb_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Cannot read RGB: {row['rgb_path']}")
        imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()


def load_gt(path, scale):
    if path is None:
        return None
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None or d.ndim != 2:
        raise RuntimeError(f"Invalid depth GT: {path}")
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
        raise RuntimeError("Could not capture packed inputs from model.infer")
    inputs, metas = clone_payload(captured["inputs"]), clone_payload(captured["metas"])
    b, t = int(metas[0]["B"]), int(metas[0]["T"])
    if b != 1 or t != len(frames):
        raise RuntimeError(f"Expected B=1,T={len(frames)}; got B={b},T={t}")
    model._set_image_paddings(inputs)
    inputs["flow_bwd"], inputs["grid_sample"] = model._normalize_and_grid_from_flow(
        inputs["flow_bwd"], inputs["image"], inputs["image_paddings"], inputs["depth_paddings"]
    )
    return inputs, metas


def restore_runtime(model, snapshot):
    temp = clone_state(snapshot.stream_state)
    restore_state(model, temp, snapshot)


def sync_scheduler(dst, src):
    for name in SCHEDULER_FIELDS:
        setattr(dst, name, clone_payload(getattr(src, name)))


def prefix_state(model, stream, inputs, total, anchor, initial_snapshot):
    state = StreamState()
    restore_state(model, state, initial_snapshot)
    _, state = stream.keyframe_step(inputs, 0, state, total)
    for t in range(1, anchor):
        _, state, restart = stream.propagate_step(inputs, t, state, total)
        if restart:
            _, state = stream.keyframe_step(inputs, t, state, total)
    return state


def build_teacher_neck_same_rays(model, inputs, frame_idx, total, rays_base):
    """Run keyframe encoder + radial neck using the propagation state's ray field."""
    image = model._bt(inputs["image"], 1, total)[:, frame_idx]
    base = {
        "image": image,
        "image_original": image,
        "validity_mask": model._bt(inputs["validity_mask"], 1, total)[:, frame_idx],
    }
    if "rays" in inputs:
        base["rays"] = model._bt(inputs["rays"], 1, total)[:, frame_idx]
    metas = [{"T": 1, "B": 1}]
    feats, tokens = model.pixel_encoder(base["image"])
    feats = [model.stacking_fn(feats[i:j]).contiguous() for i, j in model.slices_encoder_range]
    tokens = [model.stacking_fn(tokens[i:j]).contiguous() for i, j in model.slices_encoder_range]
    base["features"], base["tokens"] = feats, tokens
    model.pixel_decoder.set_shapes(base, metas)
    adapted = model.pixel_decoder.input_adapter(base["features"])
    radial = model.pixel_decoder.radial_module
    with torch.inference_mode():
        _, _, teacher_neck = radial.forward_keyframe(adapted, rays_base)
    return [x.detach().clone() for x in teacher_neck]


def patch_tile(base_features, donor_features, tile_index):
    if len(base_features) != len(donor_features):
        raise RuntimeError(f"Neck level count mismatch: {len(base_features)} vs {len(donor_features)}")
    patched, tile_rows = [], []
    tile_row, tile_col = divmod(tile_index, 2)
    for level, (base, donor) in enumerate(zip(base_features, donor_features)):
        if base.shape != donor.shape:
            raise RuntimeError(f"Neck shape mismatch at level {level}: {tuple(base.shape)} vs {tuple(donor.shape)}")
        if base.ndim != 4:
            raise RuntimeError(f"Expected B,C,H,W neck tensor, got {tuple(base.shape)}")
        _, _, h, w = base.shape
        y0, y1 = tile_row * h // 2, (tile_row + 1) * h // 2
        x0, x1 = tile_col * w // 2, (tile_col + 1) * w // 2
        if y1 <= y0 or x1 <= x0:
            raise RuntimeError(f"Empty tile at level {level}")
        out = base.detach().clone()
        out[..., y0:y1, x0:x1] = donor[..., y0:y1, x0:x1]
        patched.append(out)
        tile_rows.append({"level": level, "h": h, "w": w, "y0": y0, "y1": y1, "x0": x0, "x1": x1})
    return patched, tile_rows


def native_fast_readout_from_neck(model, neck):
    """Run the native fast decoder's post-fusion readout on an existing neck."""
    radial = model.pixel_decoder.radial_module
    lateral = [neck[0]] + [radial.process_features[i](neck[i + 1]) for i in range(len(radial.ups))]
    latents = lateral[0]
    decoded = []
    for i, up in enumerate(radial.ups):
        latents = latents + lateral[i + 1]
        latents = up(latents)
        decoded.append(latents)
    lograd = radial.depth_proj(decoded)
    radius = torch.exp(lograd.clamp(min=-8.0, max=8.0) + 2.0)
    # Native forward_fast currently aliases log-confidence to log-depth.
    confidence = torch.exp(lograd.clamp(min=-8.0, max=10.0))
    return radius.detach().clone(), confidence.detach().clone()


def output_depth(radius, rays_bchw, model, paddings, padded_shape):
    points = radius * rays_bchw  # B,C,H,W; _postprocess expects a 4-D tensor here
    points = _postprocess(points, padded_shape, paddings=paddings, interpolation_mode=model.interpolation_mode)
    return points[0, 2].detach().float().cpu().numpy()


def metrics(pred, gt):
    if gt is None:
        return {"absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan"), "valid_ratio": float("nan")}
    if pred.shape != gt.shape:
        raise RuntimeError(f"Prediction/GT shape mismatch: {pred.shape} vs {gt.shape}")
    valid = np.isfinite(gt) & (gt > 0) & np.isfinite(pred) & (pred > 0)
    if not valid.any():
        return {"absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan"), "valid_ratio": 0.0}
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    ratio = np.maximum(p / np.maximum(g, 1e-8), g / np.maximum(p, 1e-8))
    return {
        "absrel": float(np.mean(np.abs(p - g) / np.maximum(g, 1e-8))),
        "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "valid_ratio": float(valid.mean()),
    }


def eval_step(output, model, paddings, padded_shape, gt):
    return metrics(output_depth(output.radius, output.rays, model, paddings, padded_shape), gt)


def write_csv(path, rows):
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2")
    ap.add_argument("--frames", type=int, default=18, help="Need at least anchor+horizon+1 frames")
    ap.add_argument("--anchor", type=int, default=9, help="Zero-based frame index")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.anchor < 1 or args.horizon < 1 or args.frames < args.anchor + args.horizon + 1:
        ap.error("Require anchor>=1, horizon>=1, frames>=anchor+horizon+1")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for E2 smoke")

    root = Path(args.data_root)
    rgb_rows = read_index(root, "rgb.txt", "rgb")[:args.frames]
    if len(rgb_rows) < args.anchor + args.horizon + 1:
        raise RuntimeError("Not enough valid RGB index records for requested anchor+horizon")
    depth_rows = read_index(root, "depth.txt", "depth")
    records = match_gt(rgb_rows, depth_rows, args.max_time_diff)
    gt = [load_gt(r["gt_path"], args.gt_scale) if r["gt_available"] else None for r in records]
    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs/state_repair") / f"e2_crowd2_a{args.anchor}_h{args.horizon}_{__import__('datetime').datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=False)

    print(f"E2 smoke: root={root}; frames={len(records)}; anchor={args.anchor}; horizon={args.horizon}")
    print(f"Timestamp GT matches: {sum(x is not None for x in gt)}/{len(gt)} (max delta={args.max_time_diff*1000:.1f} ms)")
    frames = load_frames(records)
    torch.set_grad_enabled(False)
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to("cuda").eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"
    inputs, metas = capture_inputs(model, frames)
    total = int(metas[0]["T"])

    H0, W0 = int(frames.shape[-2]), int(frames.shape[-1])
    paddings, padded_shape = get_paddings((H0, W0), model.shape_constraints["ratio_bounds"])
    stream = VeloDepthStream(model)
    initial = capture_state(model, StreamState())
    state_before = prefix_state(model, stream, inputs, total, args.anchor, initial)
    pre_anchor = capture_state(model, state_before)

    state_prop = clone_state(state_before)
    restore_state(model, state_prop, pre_anchor)
    out_prop, state_prop, restart = stream.propagate_step(inputs, args.anchor, state_prop, total)
    if restart:
        row = {"anchor": args.anchor, "status": "skipped_anchor_restart", "reason": "Native propagation requested keyframe restart at anchor; no forced local intervention run."}
        write_csv(out_dir / "e2_skipped_anchors.csv", [row])
        (out_dir / "REPORT.md").write_text("# E2 smoke skipped\n\nThe anchor's native propagation requested a keyframe restart. No intervention was evaluated.\n", encoding="utf-8")
        print("E2 SKIPPED: anchor requested restart. Output:", out_dir)
        return

    post_anchor_snapshot = capture_state(model, state_prop)
    # Teacher donor is generated for the same frame using the propagation state's ray field.
    donor = build_teacher_neck_same_rays(model, inputs, args.anchor, total, state_prop.rays_base)
    tile_index = args.anchor % 4  # deterministic row-major 2x2 rotation
    local_neck, tile_manifest = patch_tile(state_prop.features_teacher, donor, tile_index)
    full_neck = [x.detach().clone() for x in donor]
    radius_local, conf_local = native_fast_readout_from_neck(model, local_neck)
    radius_full, conf_full = native_fast_readout_from_neck(model, full_neck)
    rays_flat = state_prop.rays_base
    _, _, h_img, w_img = inputs["image"].shape[0], inputs["image"].shape[1], inputs["image"].shape[-2], inputs["image"].shape[-1]
    rays_bchw = rays_flat.reshape(1, h_img, w_img, 3).permute(0, 3, 1, 2).contiguous()
    local_out = SimpleNamespace(radius=radius_local, rays=rays_bchw)
    full_out = SimpleNamespace(radius=radius_full, rays=rays_bchw)

    # Every repair branch emits the SAME local repaired output at the anchor;
    # only persistent state differs. PROP emits the native propagation output.
    branches = {"PROP": clone_state(state_prop), "LOCAL_OUTPUT_ONLY": clone_state(state_prop),
                "LOCAL_DEPTH_COMMIT": clone_state(state_prop), "LOCAL_FEATURE_COMMIT": clone_state(state_prop),
                "LOCAL_COHERENT_COMMIT": clone_state(state_prop), "FULL_STATE_READOUT_REFERENCE": clone_state(state_prop)}
    branches["LOCAL_DEPTH_COMMIT"].radius_teacher = radius_local.detach().clone()
    branches["LOCAL_DEPTH_COMMIT"].fast_tail_radius = radius_local.detach().clone()
    branches["LOCAL_FEATURE_COMMIT"].features_teacher = [x.detach().clone() for x in local_neck]
    branches["LOCAL_FEATURE_COMMIT"].fast_tail_features = [x.detach().clone() for x in local_neck]
    branches["LOCAL_COHERENT_COMMIT"].features_teacher = [x.detach().clone() for x in local_neck]
    branches["LOCAL_COHERENT_COMMIT"].radius_teacher = radius_local.detach().clone()
    branches["LOCAL_COHERENT_COMMIT"].confidence_teacher = conf_local.detach().clone()
    branches["LOCAL_COHERENT_COMMIT"].fast_tail_features = [x.detach().clone() for x in local_neck]
    branches["LOCAL_COHERENT_COMMIT"].fast_tail_radius = radius_local.detach().clone()
    branches["LOCAL_COHERENT_COMMIT"].fast_tail_confidence = conf_local.detach().clone()
    branches["FULL_STATE_READOUT_REFERENCE"].features_teacher = [x.detach().clone() for x in full_neck]
    branches["FULL_STATE_READOUT_REFERENCE"].radius_teacher = radius_full.detach().clone()
    branches["FULL_STATE_READOUT_REFERENCE"].confidence_teacher = conf_full.detach().clone()
    branches["FULL_STATE_READOUT_REFERENCE"].fast_tail_features = [x.detach().clone() for x in full_neck]
    branches["FULL_STATE_READOUT_REFERENCE"].fast_tail_radius = radius_full.detach().clone()
    branches["FULL_STATE_READOUT_REFERENCE"].fast_tail_confidence = conf_full.detach().clone()

    current_rows = []
    prop_metric = eval_step(out_prop, model, paddings, padded_shape, gt[args.anchor])
    current_rows.append({"branch": "PROP", "anchor": args.anchor, **prop_metric})
    for name in ("LOCAL_OUTPUT_ONLY", "LOCAL_DEPTH_COMMIT", "LOCAL_FEATURE_COMMIT", "LOCAL_COHERENT_COMMIT"):
        current_rows.append({"branch": name, "anchor": args.anchor, **eval_step(local_out, model, paddings, padded_shape, gt[args.anchor])})
    current_rows.append({"branch": "FULL_STATE_READOUT_REFERENCE", "anchor": args.anchor, **eval_step(full_out, model, paddings, padded_shape, gt[args.anchor])})
    write_csv(out_dir / "e2_anchor_metrics.csv", current_rows)

    # All branches share the same scheduler path. Restart signals are recorded but ignored.
    branch_outputs = {"PROP": out_prop, "LOCAL_OUTPUT_ONLY": local_out,
                      "LOCAL_DEPTH_COMMIT": local_out, "LOCAL_FEATURE_COMMIT": local_out,
                      "LOCAL_COHERENT_COMMIT": local_out, "FULL_STATE_READOUT_REFERENCE": full_out}
    future_rows = []
    restart_rows = []
    branch_names = list(branches)
    for step in range(1, args.horizon + 1):
        frame_idx = args.anchor + step
        scheduler_before = {k: clone_payload(getattr(branches["PROP"], k)) for k in SCHEDULER_FIELDS}
        step_results = {}
        for name in branch_names:
            state = branches[name]
            for k, v in scheduler_before.items():
                setattr(state, k, clone_payload(v))
            restore_runtime(model, post_anchor_snapshot)
            out, state, requested_restart = stream.propagate_step(inputs, frame_idx, state, total)
            branches[name] = state
            step_results[name] = (clone_payload(out), bool(requested_restart))
        # Propagation branch supplies the common post-step scheduler for all branches.
        for name in branch_names:
            sync_scheduler(branches[name], branches["PROP"])
        for name, (out, requested_restart) in step_results.items():
            m = eval_step(out, model, paddings, padded_shape, gt[frame_idx])
            future_rows.append({"branch": name, "anchor": args.anchor, "future_step": step,
                                "future_frame": frame_idx, "gt_available": gt[frame_idx] is not None,
                                "gt_time_diff_ms": records[frame_idx]["time_diff"] * 1000 if records[frame_idx]["gt_available"] else float("nan"),
                                "restart_requested_ignored": requested_restart, **m})
            restart_rows.append({"branch": name, "future_step": step, "future_frame": frame_idx,
                                 "restart_requested_ignored": requested_restart})
    write_csv(out_dir / "e2_future_frame_metrics.csv", future_rows)
    write_csv(out_dir / "e2_restart_requests.csv", restart_rows)

    summary_rows = []
    for name in branch_names:
        rows = [r for r in future_rows if r["branch"] == name and math.isfinite(float(r["absrel"]))]
        mean_absrel = float(np.mean([r["absrel"] for r in rows])) if rows else float("nan")
        mean_rmse = float(np.mean([r["rmse"] for r in rows])) if rows else float("nan")
        summary_rows.append({"branch": name, "anchor": args.anchor, "horizon": args.horizon,
                             "future_gt_frames_evaluated": len(rows), "future_gt_frames_requested": args.horizon,
                             "mean_future_absrel": mean_absrel, "mean_future_rmse": mean_rmse,
                             "gain_vs_local_output_only_absrel": (
                                 float(np.mean([r["absrel"] for r in future_rows if r["branch"] == "LOCAL_OUTPUT_ONLY" and math.isfinite(float(r["absrel"]))])) - mean_absrel
                                 if rows and any(r["branch"] == "LOCAL_OUTPUT_ONLY" and math.isfinite(float(r["absrel"])) for r in future_rows) else float("nan")),
                             "interpretation": "positive gain means lower future AbsRel than LOCAL_OUTPUT_ONLY"})
    write_csv(out_dir / "e2_summary.csv", summary_rows)
    metadata = {
        "experiment": "E2_local_state_repair_single_anchor_smoke", "status": "completed",
        "weights_frozen": True, "device": str(next(model.parameters()).device), "torch": torch.__version__,
        "data_root": str(root), "frames": len(records), "anchor_zero_based": args.anchor,
        "horizon": args.horizon, "tile_grid": "2x2", "tile_index_row_major": tile_index,
        "teacher_donor": "full-frame keyframe neck computed under anchor propagation rays; diagnostic oracle only",
        "gt_match_policy": f"nearest depth timestamp within {args.max_time_diff}s; unmatched RGB remains in stream and is omitted from metrics",
        "gt_scale": args.gt_scale, "resolution_level": args.resolution_level,
        "future_policy": "forced propagation; restart requests logged but not executed; scheduler synchronized from PROP",
        "important_limit": "single anchor smoke only; no statistical claim; donor computation is not local-compute cost",
        "tile_manifest": tile_manifest,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# E2 single-anchor local state repair smoke", "",
              f"- Status: completed (this is not evidence of method success)",
              f"- Sequence: {root.name}; anchor={args.anchor}; horizon={args.horizon}",
              f"- GT matches: {sum(x is not None for x in gt)}/{len(gt)}",
              "- Teacher donor: full-frame oracle neck under the propagated ray field; its cost is not local-method cost.",
              "- Future rollouts: forced propagation; restart requests logged but ignored; shared scheduler path.",
              "- Read `e2_summary.csv` and `e2_future_frame_metrics.csv`; this single anchor is a smoke, not a go/no-go statistical result.", ""]
    (out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("E2 output directory:", out_dir.resolve())
    print("E2 smoke completed. Review e2_anchor_metrics.csv, e2_future_frame_metrics.csv, e2_summary.csv.")


if __name__ == "__main__":
    main()
