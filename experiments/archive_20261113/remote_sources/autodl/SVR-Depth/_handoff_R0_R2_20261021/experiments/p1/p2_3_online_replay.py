#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.3 minimal online replay prototype for SVR-Depth.

This is an exploratory, full-frame online replay harness, NOT a local-refiner
implementation and NOT an end-to-end latency benchmark. It uses the real
VeloDepthStream propagation/keyframe methods and computes each policy's action
from information available at the current frame before that frame's action.

Policies:
  auto               original automatic keyframe/restart behavior (B0 baseline)
  fixedN             force a full keyframe every N source frames (e.g. fixed8)
  motion_only        Ridge score from P2.2.2 motion features, trigger score > threshold
  learned_state_only Ridge score from P2.2.2 learned-state features
  motion_plus_learned Ridge score from both feature families

Predictors are fitted at runtime on P2.2.2 H=16 rows from the OTHER TWO
sequences (leave-one-sequence-out). No VeloDepth weights are trained/changed.

Important semantics:
- A policy-triggered full keyframe refresh bypasses propagation for that frame
  and disables overlap blending for that scheduled refresh. This is a new
  scheduled-refresh action semantics; it is logged explicitly and should be
  reviewed before treating results as final.
- An automatic restart preserves VeloDepthStream's existing behavior: run the
  propagation candidate, discard it when restart=True, then keyframe the SAME
  frame.
- Flow/preprocessing tensors are captured by running model.infer() only up to
  encode_decode_test(), then aborting at that boundary. Therefore action timing
  excludes preprocessing/optical-flow construction and must NOT be reported as
  end-to-end latency/FPS.
"""

import argparse
import copy
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

import p1_multihorizon_counterfactual as cf
from stream_state import StreamState
from stream_state_snapshot import capture_state, restore_state
from velodepth_stream import VeloDepthStream
from velodepth.models.velodepth import VeloDepth, get_paddings

SEQUENCES = ("crowd2", "balloon2", "crowd3")
MOTION = [
    "scheduler_distance_kf",
    "scheduler_flow_magnitude",
    "scheduler_total_luma_change",
    "image_absdiff_mean_prev",
    "image_absdiff_p90_prev",
]
LEARNED = [
    f"state_features_teacher_{i}_{stat}"
    for i in range(4) for stat in ("mean", "std")
]
MODEL_FEATURES = {
    "motion_only": MOTION,
    "learned_state_only": LEARNED,
    "motion_plus_learned": MOTION + LEARNED,
}
TARGET = "G_state_absrel_discounted_per_valid_frame"


class _CapturedPackedInputs(Exception):
    """Private sentinel: stop model.infer immediately after packed-input capture."""


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


def capture_packed_inputs_without_full_inference(model, frames):
    """Capture infer()'s exact packed payload, then abort before neural inference."""
    captured = {}
    original = model.encode_decode_test

    def capture_and_abort(inputs, image_metas):
        captured["inputs"] = clone_payload(inputs)
        captured["metas"] = clone_payload(image_metas)
        raise _CapturedPackedInputs()

    model.encode_decode_test = capture_and_abort
    try:
        with torch.inference_mode():
            try:
                model.infer(frames, normalize=True)
            except _CapturedPackedInputs:
                pass
    finally:
        model.encode_decode_test = original

    if "inputs" not in captured or "metas" not in captured:
        raise RuntimeError("Failed to capture packed inputs at encode_decode_test().")

    inputs = clone_payload(captured["inputs"])
    metas = clone_payload(captured["metas"])
    b, t = int(metas[0]["B"]), int(metas[0]["T"])
    if b != 1 or t != len(frames):
        raise RuntimeError(f"Expected packed B=1,T={len(frames)}; got B={b},T={t}")

    model._set_image_paddings(inputs)
    inputs["flow_bwd"], inputs["grid_sample"] = model._normalize_and_grid_from_flow(
        inputs["flow_bwd"], inputs["image"], inputs["image_paddings"], inputs["depth_paddings"]
    )
    return inputs, metas


def scalar_value(x):
    if x is None:
        return float("nan")
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float("nan")
        return float(x.detach().float().mean().cpu().item())
    if isinstance(x, (bool, int, float, np.number)):
        return float(x)
    return float("nan")


def finite_tensor_sample(x, max_samples=12000):
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return None
    y = x.detach().reshape(-1)
    stride = max(1, int(math.ceil(y.numel() / max_samples)))
    y = y[::stride].float()
    y = y[torch.isfinite(y)]
    return y if y.numel() else None


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


def build_online_features(state, frame_rgb, prev_rgb):
    """Features computed from current input and current pre-action persistent state."""
    row = {}
    for name in ("current_t", "distance_kf", "flow_magnitude", "total_luma_change", "first_keyframe"):
        row[f"scheduler_{name}"] = scalar_value(getattr(state, name, None))
    for name in ("radius_teacher", "confidence_teacher", "rays_base", "features_teacher"):
        write_tensor_stats(row, f"state_{name}", getattr(state, name, None))

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
    return row


class OnlineRidge:
    def __init__(self, train_csv, heldout_sequence, model_name, horizon=16, alpha=10.0):
        if model_name not in MODEL_FEATURES:
            raise ValueError(f"Unknown predictor: {model_name}")
        path = Path(train_csv)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing P2.2.2 training data: {path}. Pass --train-dataset to the valid_dataset.csv."
            )
        df = pd.read_csv(path, encoding="utf-8-sig")
        required = {"sequence", "horizon_H", TARGET, *MODEL_FEATURES[model_name]}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Training CSV missing columns: {sorted(missing)}")
        df["horizon_H"] = pd.to_numeric(df["horizon_H"], errors="coerce")
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        train = df[(df["horizon_H"] == horizon) & (df["sequence"] != heldout_sequence)].copy()
        train = train[np.isfinite(train[TARGET].to_numpy(float))]
        self.columns = MODEL_FEATURES[model_name]
        X = train[self.columns].apply(pd.to_numeric, errors="coerce").astype(float)
        y = train[TARGET].to_numpy(float)
        if len(train) < 8:
            raise RuntimeError(f"Only {len(train)} training rows for {model_name}; need at least 8.")
        self.medians = X.median(axis=0).fillna(0.0)
        X = X.fillna(self.medians)
        varying = X.nunique(dropna=False) > 1
        self.columns = list(X.columns[varying])
        if not self.columns:
            raise RuntimeError(f"All training features are constant for {model_name}.")
        X = X[self.columns]
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = Ridge(alpha=alpha)
        self.model.fit(Xs, y)
        self.train_n = int(len(train))
        self.heldout_sequence = heldout_sequence
        self.model_name = model_name
        self.horizon = horizon

    def predict(self, features):
        X = pd.DataFrame([{c: features.get(c, np.nan) for c in self.columns}])
        X = X.apply(pd.to_numeric, errors="coerce").astype(float).fillna(self.medians[self.columns])
        return float(self.model.predict(self.scaler.transform(X))[0])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nno_rows\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def should_refresh(policy, t, last_keyframe, fixed_interval, min_gap, score, threshold):
    if policy == "auto":
        return False
    if policy.startswith("fixed"):
        interval = int(policy[len("fixed"):]) if policy[len("fixed"):].isdigit() else fixed_interval
        return (t - last_keyframe) >= interval
    if policy in MODEL_FEATURES:
        return (t - last_keyframe) >= min_gap and math.isfinite(score) and score > threshold
    raise ValueError(f"Unknown policy {policy}")


def run_policy(policy, model, base_snapshot, inputs, metas, frames, records, gt,
               paddings, padded_shape, sequence, args, predictor=None):
    # Restore decoder runtime attributes between policies; each policy gets a fresh state.
    state = StreamState()
    restore_state(model, state, base_snapshot)
    if "rays" in inputs:
        model.pixel_decoder.test_gt_camera = True
    stream = VeloDepthStream(model)
    total_frames = int(metas[0]["T"])
    last_keyframe = 0
    rows = []
    for t in range(total_frames):
        decision_ms = 0.0
        score = float("nan")
        distance_before_action = scalar_value(getattr(state, "distance_kf", 0))
        decision_start = time.perf_counter()
        if t == 0:
            do_refresh = False
            action = "initial_keyframe"
        else:
            feat = build_online_features(state, frames[t], frames[t - 1])
            if predictor is not None:
                score = predictor.predict(feat)
            do_refresh = should_refresh(
                policy, t, last_keyframe, args.fixed_interval,
                args.min_refresh_gap, score, args.threshold
            )
            action = "policy_refresh" if do_refresh else "propagation"
        decision_ms = (time.perf_counter() - decision_start) * 1000.0

        sync_cuda()
        action_start = time.perf_counter()
        restart = False
        if t == 0:
            out, state = stream.keyframe_step(inputs, t, state, total_frames)
            last_keyframe = t
        elif do_refresh:
            # A policy-triggered scheduled refresh is defined as a fresh keyframe:
            # skip propagation and disable overlap blending for this frame.
            state.first_keyframe = True
            out, state = stream.keyframe_step(inputs, t, state, total_frames)
            # Seed overlap-tail fields with the new keyframe state for safe future restarts.
            state.fast_tail_radius = state.radius_teacher.detach().clone()
            state.fast_tail_confidence = state.confidence_teacher.detach().clone()
            state.fast_tail_features = [x.detach().clone() for x in state.features_teacher]
            state.first_keyframe = False
            last_keyframe = t
        else:
            out, state, restart = stream.propagate_step(inputs, t, state, total_frames)
            if restart:
                # Preserve the original VeloDepth behavior: reject propagation and keyframe same frame.
                out, state = stream.keyframe_step(inputs, t, state, total_frames)
                action = "auto_restart"
                last_keyframe = t
        sync_cuda()
        action_ms = (time.perf_counter() - action_start) * 1000.0

        metric_start = time.perf_counter()
        m = cf.eval_output(out, model, paddings, padded_shape, gt[t])
        metric_ms = (time.perf_counter() - metric_start) * 1000.0
        rows.append({
            "sequence": sequence,
            "policy": policy,
            "frame_index": t,
            "action": action,
            "predictor_score": score,
            "threshold": args.threshold if predictor is not None else float("nan"),
            "distance_kf_before_action": int(max(0, round(distance_before_action))) if math.isfinite(distance_before_action) else -1,
            "last_keyframe_frame": last_keyframe,
            "auto_restart_flag": bool(restart),
            "gt_available": bool(records[t]["gt_available"]),
            "gt_time_diff_ms": records[t]["time_diff"] * 1000.0 if math.isfinite(records[t]["time_diff"]) else float("nan"),
            "absrel": m["absrel"], "rmse": m["rmse"], "delta1": m["delta1"],
            "valid_ratio": m["valid_ratio"],
            "decision_ms": decision_ms,
            "model_action_ms": action_ms,
            "metric_eval_ms": metric_ms,
        })
        if t == 0 or t == total_frames - 1 or action != "propagation":
            print(f"[{policy}] t={t:03d} action={action:16s} score={score: .6f} "
                  f"AbsRel={m['absrel']:.5f} action_ms={action_ms:.2f}")

    return rows


def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def summarize(rows):
    out = []
    for policy in sorted(set(r["policy"] for r in rows)):
        sub = [r for r in rows if r["policy"] == policy]
        valid = [r for r in sub if math.isfinite(float(r["absrel"]))]
        actions = [r["action"] for r in sub]
        action_ms = [r["model_action_ms"] for r in sub]
        decisions = [r["decision_ms"] for r in sub]
        out.append({
            "sequence": sub[0]["sequence"],
            "policy": policy,
            "frames": len(sub),
            "frames_with_valid_metrics": len(valid),
            "mean_absrel": safe_mean([r["absrel"] for r in valid]),
            "mean_rmse": safe_mean([r["rmse"] for r in valid]),
            "mean_delta1": safe_mean([r["delta1"] for r in valid]),
            "mean_valid_ratio": safe_mean([r["valid_ratio"] for r in valid]),
            "policy_refresh_count": sum(a == "policy_refresh" for a in actions),
            "auto_restart_count": sum(a == "auto_restart" for a in actions),
            "keyframe_count_including_initial": sum(a in ("initial_keyframe", "policy_refresh", "auto_restart") for a in actions),
            "mean_model_action_ms": safe_mean(action_ms),
            "p95_model_action_ms": float(np.percentile(action_ms, 95)) if action_ms else float("nan"),
            "mean_decision_ms": safe_mean(decisions),
            "estimated_action_only_fps_not_end_to_end": 1000.0 / safe_mean(action_ms) if safe_mean(action_ms) > 0 else float("nan"),
            "timing_scope": "model action replay only; excludes preprocessing/optical flow and metric evaluation",
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="P2.3 minimal online replay prototype")
    ap.add_argument("--data-root", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2")
    ap.add_argument("--sequence", choices=SEQUENCES, default="crowd2")
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--max-time-diff", type=float, default=0.02)
    ap.add_argument("--gt-scale", type=float, default=5000.0)
    ap.add_argument("--resolution-level", type=int, default=0)
    ap.add_argument("--train-dataset", default="outputs/p2_2_2/p2_2_2_valid_dataset.csv")
    ap.add_argument("--policies", default="auto,fixed8,motion_only,learned_state_only,motion_plus_learned")
    ap.add_argument("--fixed-interval", type=int, default=8)
    ap.add_argument("--min-refresh-gap", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--out-dir", default="outputs/p2_3_online_replay/crowd2_32")
    args = ap.parse_args()

    if args.frames < 3 or args.fixed_interval < 1 or args.min_refresh_gap < 1:
        ap.error("frames must be >=3 and refresh intervals must be >=1")
    if args.horizon != 16:
        ap.error("This first P2.3 pilot currently uses H=16, matching the planned long-horizon target.")
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    allowed = {"auto", "fixed8", "fixed16", *MODEL_FEATURES.keys()}
    unknown = set(policies) - allowed
    if unknown:
        ap.error(f"Unknown policies {sorted(unknown)}; allowed: {sorted(allowed)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check nvidia-smi and torch.cuda.is_available() before P2.3.")

    root = Path(args.data_root)
    rgb_index = cf.read_index(root, "rgb.txt", "rgb")
    depth_index = cf.read_index(root, "depth.txt", "depth")
    selected = rgb_index[:args.frames]
    if len(selected) != args.frames:
        raise RuntimeError(f"Requested {args.frames} RGB frames but found {len(selected)} in {root}")
    records = cf.match_gt(selected, depth_index, args.max_time_diff)
    frames = cf.load_frames(records)
    gt = [cf.load_gt(r["gt_path"], args.gt_scale) if r["gt_available"] else None for r in records]
    matched = sum(bool(r["gt_available"]) for r in records)
    print(f"Sequence={args.sequence}; frames={len(frames)}; GT matches={matched}/{len(records)}")

    torch.set_grad_enabled(False)
    print("Loading frozen VeloDepth weights...")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(torch.device("cuda"))
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    print("Capturing packed inputs/flow; neural inference is aborted at encode_decode_test boundary...")
    inputs, metas = capture_packed_inputs_without_full_inference(model, frames)
    total_frames = int(metas[0]["T"])
    print(f"PASS: packed inputs captured, T={total_frames}; no full reference inference was run by capture.")

    H, W = int(frames.shape[-2]), int(frames.shape[-1])
    paddings, (padded_h, padded_w) = get_paddings((H, W), model.shape_constraints["ratio_bounds"])
    padded_shape = (padded_h, padded_w)
    base_snapshot = capture_state(model, StreamState())

    predictors = {}
    for p in policies:
        if p in MODEL_FEATURES:
            predictors[p] = OnlineRidge(
                args.train_dataset, args.sequence, p, horizon=args.horizon, alpha=args.ridge_alpha
            )
            print(f"Predictor {p}: trained on {predictors[p].train_n} H={args.horizon} samples from other sequences; alpha={args.ridge_alpha}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for policy in policies:
        print(f"\n===== RUN POLICY: {policy} =====")
        rows = run_policy(
            policy, model, base_snapshot, inputs, metas, frames, records, gt,
            paddings, padded_shape, args.sequence, args, predictors.get(policy)
        )
        all_rows.extend(rows)

    # Keep requested policy order in summary rather than alphabetic order.
    summary = []
    all_summaries = summarize(all_rows)
    for policy in policies:
        summary.extend([r for r in all_summaries if r["policy"] == policy])
    frame_csv = out_dir / "p2_3_online_frame_metrics.csv"
    summary_csv = out_dir / "p2_3_online_summary.csv"
    metadata_json = out_dir / "p2_3_online_metadata.json"
    write_csv(frame_csv, all_rows)
    write_csv(summary_csv, summary)
    metadata = {
        "experiment": "P2.3 minimal full-frame online replay",
        "sequence": args.sequence,
        "data_root": str(root),
        "frames": args.frames,
        "gt_matches": matched,
        "gt_match_tolerance_seconds": args.max_time_diff,
        "gt_scale": args.gt_scale,
        "resolution_level": args.resolution_level,
        "weights_frozen": True,
        "predictor_target": TARGET,
        "predictor_horizon": args.horizon,
        "predictor_training": "Ridge fitted at runtime on P2.2.2 H=16 rows from the other two sequences; no VeloDepth weights updated",
        "policies": policies,
        "fixed_interval": args.fixed_interval,
        "min_refresh_gap": args.min_refresh_gap,
        "score_threshold": args.threshold,
        "scheduled_refresh_semantics": "direct full keyframe; skip propagation on that frame; disable overlap blending for that scheduled frame; seed fast-tail fields from refreshed state",
        "automatic_restart_semantics": "original adapter behavior: propagate candidate, discard if restart=True, then keyframe same frame",
        "timing_scope": "model action replay only; excludes RGB preprocessing, optical-flow construction, packed-input capture, and metric evaluation; not end-to-end latency/FPS",
        "outputs": {"frame_metrics": str(frame_csv), "summary": str(summary_csv), "metadata": str(metadata_json)},
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== P2.3 PILOT SUMMARY =====")
    print(pd.DataFrame(summary)[[
        "policy", "frames_with_valid_metrics", "mean_absrel", "mean_rmse", "mean_delta1",
        "policy_refresh_count", "auto_restart_count", "mean_model_action_ms", "p95_model_action_ms", "mean_decision_ms"
    ]].to_string(index=False))
    print("\nWrote:", frame_csv)
    print("Wrote:", summary_csv)
    print("Wrote:", metadata_json)
    print("WARNING: this is a 32-frame exploratory replay, not a final same-budget or end-to-end latency result.")


if __name__ == "__main__":
    main()
