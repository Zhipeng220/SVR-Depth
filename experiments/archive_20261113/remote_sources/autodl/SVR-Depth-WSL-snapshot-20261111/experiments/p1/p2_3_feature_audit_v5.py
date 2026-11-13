#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2.3 v5 feature-consistency audit + v4 online replay.

This is an exploratory, full-frame online replay harness, NOT a local-refiner
implementation and NOT an end-to-end latency benchmark. It uses the real
VeloDepthStream propagation/keyframe methods and computes each policy's action
from information available at the current frame before that frame's action.

Policies:
  auto               original automatic keyframe/restart behavior (B0 baseline)
  fixedN             force a full keyframe every N source frames (e.g. fixed8)
  candidate_fixed    candidate-anchor-aligned fixed baseline (e.g. t=9,17,25)
  motion_only        Ridge score from P2.2.2 motion features, trigger score > threshold
  learned_state_only Ridge score from P2.2.2 learned-state features
  motion_plus_learned Ridge score from both feature families

Predictors are fitted at runtime on P2.2.2 H=16 rows from the OTHER TWO
sequences (leave-one-sequence-out). Threshold calibration uses 3 contiguous
anchor-rank blocks per training sequence: every OOF fold validates on a block
from BOTH training sequences and trains on the remaining blocks from BOTH.
No VeloDepth weights are trained/changed.

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


def fit_ridge_fold(train_df, valid_df, columns, target, alpha):
    """Fit preprocessing/model on one sequence fold and score the held-out fold."""
    X_train = train_df[columns].apply(pd.to_numeric, errors="coerce").astype(float)
    y_train = pd.to_numeric(train_df[target], errors="coerce").to_numpy(float)
    medians = X_train.median(axis=0).fillna(0.0)
    X_train = X_train.fillna(medians)
    varying = X_train.nunique(dropna=False) > 1
    fold_columns = list(X_train.columns[varying])
    if not fold_columns:
        return np.asarray([], dtype=float)
    X_train = X_train[fold_columns]
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    model = Ridge(alpha=alpha)
    model.fit(Xs_train, y_train)

    X_valid = valid_df[fold_columns].apply(pd.to_numeric, errors="coerce").astype(float)
    X_valid = X_valid.fillna(medians[fold_columns])
    return model.predict(scaler.transform(X_valid))


def threshold_for_top_fraction(scores, fraction):
    """Choose a score threshold targeting a top-fraction of OOF scores."""
    vals = np.asarray(scores, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise RuntimeError("Cannot calibrate threshold: no finite out-of-fold scores.")
    k = int(math.ceil(float(fraction) * vals.size))
    k = max(1, min(vals.size, k))
    ordered = np.sort(vals)
    if k >= vals.size:
        return float(np.nextafter(ordered[0], -np.inf))
    lower = ordered[vals.size - k - 1]
    upper = ordered[vals.size - k]
    # Midpoint selects top-k exactly if boundary scores differ.
    if upper > lower:
        return float(lower + (upper - lower) / 2.0)
    return float(lower)


def make_temporal_block_folds(train_df, n_splits=3):
    """Return temporal-block folds where each fold's train/valid contain both sequences.

    Within each training sequence, unique anchor_frame values are sorted and split
    into contiguous rank blocks. The same block rank is held out from every sequence
    in a fold. This avoids the v3 failure mode where a fold trained on Balloon2 was
    used to calibrate scores for Crowd3 (and vice versa).
    """
    if "anchor_frame" not in train_df.columns:
        raise ValueError("Training CSV must contain anchor_frame for temporal-block OOF calibration.")
    seqs = sorted(train_df["sequence"].astype(str).unique().tolist())
    if len(seqs) < 2:
        raise RuntimeError(f"Temporal-block calibration requires at least two training sequences; got {seqs}")

    fold_by_index = {}
    per_sequence_blocks = {}
    for seq in seqs:
        sub = train_df[train_df["sequence"].astype(str) == seq].copy()
        anchors = pd.to_numeric(sub["anchor_frame"], errors="coerce").dropna().unique()
        anchors = np.sort(anchors)
        if len(anchors) < n_splits:
            raise RuntimeError(
                f"Sequence {seq} has only {len(anchors)} unique anchors; cannot make {n_splits} temporal folds."
            )
        chunks = [chunk for chunk in np.array_split(anchors, n_splits) if len(chunk)]
        if len(chunks) != n_splits:
            raise RuntimeError(f"Could not create {n_splits} temporal blocks for sequence {seq}.")
        per_sequence_blocks[seq] = []
        for fold_id, chunk in enumerate(chunks):
            anchor_set = set(float(x) for x in chunk.tolist())
            idxs = sub.index[pd.to_numeric(sub["anchor_frame"], errors="coerce").isin(anchor_set)].tolist()
            for idx in idxs:
                fold_by_index[idx] = fold_id
            per_sequence_blocks[seq].append({
                "fold_id": fold_id,
                "anchor_min": float(np.min(chunk)),
                "anchor_max": float(np.max(chunk)),
                "n_rows": len(idxs),
            })

    fold_ids = pd.Series(fold_by_index, name="_calibration_fold")
    folds = []
    for fold_id in range(n_splits):
        valid_idx = fold_ids.index[fold_ids == fold_id]
        valid = train_df.loc[valid_idx].copy()
        fit = train_df.loc[~train_df.index.isin(valid_idx)].copy()
        fit_sequences = sorted(fit["sequence"].astype(str).unique().tolist())
        valid_sequences = sorted(valid["sequence"].astype(str).unique().tolist())
        if fit_sequences != seqs or valid_sequences != seqs:
            raise RuntimeError(
                f"Calibration fold {fold_id} does not contain both training sequences: "
                f"train={fit_sequences}, valid={valid_sequences}, expected={seqs}."
            )
        folds.append((fold_id, fit, valid, per_sequence_blocks))

    # Every training row must be held out exactly once.
    all_valid = [idx for _, _, valid, _ in folds for idx in valid.index.tolist()]
    if len(all_valid) != len(set(all_valid)) or set(all_valid) != set(train_df.index):
        raise RuntimeError("Temporal-block OOF split is not a one-to-one partition of training rows.")
    return folds


class OnlineRidge:
    def __init__(self, train_csv, heldout_sequence, model_name, horizon=16, alpha=10.0,
                 target_refresh_fraction=3.0/7.0):
        if model_name not in MODEL_FEATURES:
            raise ValueError(f"Unknown predictor: {model_name}")
        path = Path(train_csv)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing P2.2.2 training data: {path}. Pass --train-dataset to the valid_dataset.csv."
            )
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["sequence"] = df["sequence"].astype(str)
        available_sequences = set(df["sequence"].dropna().tolist())
        if heldout_sequence not in available_sequences:
            raise RuntimeError(
                f"Held-out sequence '{heldout_sequence}' is not present in {path}. "
                f"Available sequence values: {sorted(available_sequences)}. "
                "Refusing to continue because the leave-one-sequence-out split could leak."
            )
        required = {"sequence", "anchor_frame", "horizon_H", TARGET, *MODEL_FEATURES[model_name]}
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
        self.target_refresh_fraction = float(target_refresh_fraction)

        # Training-only temporal-block OOF predictions. The held-out test sequence
        # is excluded before this stage. Each fold trains and validates on both of
        # the remaining sequences, preventing the v3 one-sequence-per-fold score-scale
        # mismatch. Temporal blocks are contiguous in anchor order within each sequence.
        self.oof_rows = []
        self.fold_rows = []
        oof_scores = []
        train_sequences = sorted(train["sequence"].astype(str).unique().tolist())
        folds = make_temporal_block_folds(train, n_splits=3)
        for fold_id, fold_train, fold_valid, block_info in folds:
            fold_scores = fit_ridge_fold(
                fold_train, fold_valid, MODEL_FEATURES[model_name], TARGET, alpha
            )
            if len(fold_scores) != len(fold_valid):
                raise RuntimeError(
                    f"Fold {fold_id} produced {len(fold_scores)} scores for {len(fold_valid)} validation rows."
                )
            fold_scores = np.asarray(fold_scores, dtype=float)
            if not np.isfinite(fold_scores).all():
                raise RuntimeError(f"Non-finite OOF prediction found in fold {fold_id} for {model_name}.")
            self.fold_rows.append({
                "model": model_name,
                "heldout_test_sequence": heldout_sequence,
                "fold_id": fold_id,
                "fold_method": "within-sequence contiguous anchor-rank blocks; same block held out in both train sequences",
                "fold_train_rows": int(len(fold_train)),
                "fold_valid_rows": int(len(fold_valid)),
                "fold_train_sequences": ";".join(sorted(fold_train["sequence"].astype(str).unique())),
                "fold_valid_sequences": ";".join(sorted(fold_valid["sequence"].astype(str).unique())),
                "oof_score_mean": float(np.mean(fold_scores)),
                "oof_score_std": float(np.std(fold_scores)),
                "oof_score_min": float(np.min(fold_scores)),
                "oof_score_max": float(np.max(fold_scores)),
                "oof_true_gain_mean": float(pd.to_numeric(fold_valid[TARGET], errors="coerce").mean()),
                "fold_valid_anchor_min": float(pd.to_numeric(fold_valid["anchor_frame"], errors="coerce").min()),
                "fold_valid_anchor_max": float(pd.to_numeric(fold_valid["anchor_frame"], errors="coerce").max()),
            })
            for (idx, sample), pred in zip(fold_valid.iterrows(), fold_scores):
                item = {
                    "model": model_name,
                    "heldout_test_sequence": heldout_sequence,
                    "oof_fold_id": int(fold_id),
                    "oof_sequence": str(sample["sequence"]),
                    "oof_anchor_frame": float(sample["anchor_frame"]),
                    "sample_index": int(idx),
                    "fold_train_rows": int(len(fold_train)),
                    "fold_valid_rows": int(len(fold_valid)),
                    "fold_train_sequences": ";".join(sorted(fold_train["sequence"].astype(str).unique())),
                    "oof_predicted_gain": float(pred),
                    "oof_true_gain": float(sample[TARGET]),
                }
                self.oof_rows.append(item)
                oof_scores.append(float(pred))
        if len(oof_scores) != len(train):
            raise RuntimeError(
                f"Expected one OOF score per training row for {model_name}; got {len(oof_scores)} vs {len(train)}."
            )
        if not oof_scores:
            raise RuntimeError(
                f"Could not generate training-only OOF scores for threshold calibration ({model_name})."
            )
        self.calibrated_threshold = threshold_for_top_fraction(
            oof_scores, self.target_refresh_fraction
        )
        self.oof_n = len(oof_scores)
        self.oof_achieved_fraction = float(
            np.mean(np.asarray(oof_scores, dtype=float) > self.calibrated_threshold)
        )
        self.oof_score_mean = float(np.mean(oof_scores))
        self.oof_score_std = float(np.std(oof_scores))
        self.oof_score_min = float(np.min(oof_scores))
        self.oof_score_max = float(np.max(oof_scores))

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


def warmup_policy(model, base_snapshot, inputs_reference, total_frames):
    """Warm CUDA/model kernels on a disposable state before collecting timings."""
    state = StreamState()
    restore_state(model, state, base_snapshot)
    if "rays" in inputs_reference:
        model.pixel_decoder.test_gt_camera = True
    stream = VeloDepthStream(model)
    with torch.inference_mode():
        warm_inputs = clone_payload(inputs_reference)
        _, state = stream.keyframe_step(warm_inputs, 0, state, total_frames)
        if total_frames > 1:
            warm_inputs = clone_payload(inputs_reference)
            _, state, restart = stream.propagate_step(
                warm_inputs, 1, state, total_frames
            )
            if restart:
                warm_inputs = clone_payload(inputs_reference)
                _, state = stream.keyframe_step(
                    warm_inputs, 1, state, total_frames
                )
    sync_cuda()
    # Reset decoder/runtime state after warm-up; measured replay starts clean.
    reset_state = StreamState()
    restore_state(model, reset_state, base_snapshot)


def run_policy(policy, model, base_snapshot, inputs, metas, frames, records, gt,
               paddings, padded_shape, sequence, args, predictor=None,
               feature_audit_rows=None):
    # Treat the packed tensors as immutable reference data. The stream methods may
    # mutate input payloads, so each action receives a fresh copy.
    inputs_reference = clone_payload(inputs)
    total_frames = int(metas[0]["T"])
    warmup_policy(model, base_snapshot, inputs_reference, total_frames)

    # Each policy starts from a clean state after its own warm-up pass.
    state = StreamState()
    restore_state(model, state, base_snapshot)
    if "rays" in inputs_reference:
        model.pixel_decoder.test_gt_camera = True
    stream = VeloDepthStream(model)
    last_keyframe = 0
    rows = []
    if feature_audit_rows is None:
        feature_audit_rows = []
    for t in range(total_frames):
        decision_ms = 0.0
        score = float("nan")
        feat = None
        candidate_due = False
        gap_eligible = False
        distance_before_action = scalar_value(getattr(state, "distance_kf", 0))
        last_keyframe_before_action = last_keyframe
        decision_start = time.perf_counter()
        if t == 0:
            do_refresh = False
            action = "initial_keyframe"
        else:
            # P2.2 features were sampled at anchors 1,5,9,...; make learned
            # decisions only at that same cadence to reduce train/test mismatch.
            candidate_due = (
                (t - args.candidate_offset) % args.candidate_stride == 0
            )
            gap_eligible = (t - last_keyframe) >= args.min_refresh_gap
            need_predictor_features = predictor is not None and candidate_due and gap_eligible
            need_audit_features = (
                args.dump_online_features and policy == "auto" and candidate_due
            )
            if need_predictor_features or need_audit_features:
                feat = build_online_features(state, frames[t], frames[t - 1])
            if need_predictor_features:
                score = predictor.predict(feat)
            if policy == "candidate_fixed":
                # Candidate-aligned fixed baseline: with stride=4, offset=1,
                # interval=8 and min-gap=4, refresh at frames 9,17,25,... .
                do_refresh = (
                    candidate_due
                    and gap_eligible
                    and ((t - args.candidate_offset) % args.fixed_interval == 0)
                )
            else:
                effective_threshold = (
                    predictor.calibrated_threshold
                    if predictor is not None and args.threshold_mode == "train_oof_budget"
                    else args.threshold
                )
                do_refresh = should_refresh(
                    policy, t, last_keyframe, args.fixed_interval,
                    args.min_refresh_gap, score, effective_threshold
                )
            action = "policy_refresh" if do_refresh else "propagation"
        decision_ms = (time.perf_counter() - decision_start) * 1000.0

        # Clone before timing the model action. This prevents a previous action or
        # policy from contaminating the shared packed flow/input tensors.
        frame_inputs = clone_payload(inputs_reference)
        sync_cuda()
        action_start = time.perf_counter()
        restart = False
        if t == 0:
            out, state = stream.keyframe_step(frame_inputs, t, state, total_frames)
            last_keyframe = t
        elif do_refresh:
            # A policy-triggered scheduled refresh is defined as a fresh keyframe:
            # skip propagation and disable overlap blending for this frame.
            state.first_keyframe = True
            out, state = stream.keyframe_step(frame_inputs, t, state, total_frames)
            # Seed overlap-tail fields with the new keyframe state for safe future restarts.
            state.fast_tail_radius = state.radius_teacher.detach().clone()
            state.fast_tail_confidence = state.confidence_teacher.detach().clone()
            state.fast_tail_features = [x.detach().clone() for x in state.features_teacher]
            state.first_keyframe = False
            last_keyframe = t
        else:
            out, state, restart = stream.propagate_step(
                frame_inputs, t, state, total_frames
            )
            if restart:
                # Restore the same frame's original packed inputs before the restart keyframe.
                restart_inputs = clone_payload(inputs_reference)
                out, state = stream.keyframe_step(
                    restart_inputs, t, state, total_frames
                )
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
            "threshold": (
                predictor.calibrated_threshold
                if predictor is not None and args.threshold_mode == "train_oof_budget"
                else (args.threshold if predictor is not None else float("nan"))
            ),
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
        if (
            args.dump_online_features and policy == "auto"
            and candidate_due and feat is not None
        ):
            audit_row = {
                "sequence": sequence,
                "anchor_frame": int(t),
                "policy": policy,
                "feature_timing": "before_anchor_action_and_before_refresh",
                "last_keyframe_before_action": int(last_keyframe_before_action),
                "distance_kf_before_action": distance_before_action,
            }
            audit_row.update(feat)
            feature_audit_rows.append(audit_row)
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
            "median_model_action_ms": float(np.median(action_ms)) if action_ms else float("nan"),
            "p95_model_action_ms": float(np.percentile(action_ms, 95)) if action_ms else float("nan"),
            "max_model_action_ms": float(np.max(action_ms)) if action_ms else float("nan"),
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
    ap.add_argument("--policies", default="auto,fixed8,candidate_fixed,motion_only,learned_state_only,motion_plus_learned")
    ap.add_argument("--fixed-interval", type=int, default=8)
    ap.add_argument("--min-refresh-gap", type=int, default=8)
    ap.add_argument("--candidate-stride", type=int, default=4,
                    help="Run learned-policy decisions only every N frames; default matches P2.2 anchor cadence.")
    ap.add_argument("--candidate-offset", type=int, default=1,
                    help="Zero-based frame-index offset for candidate anchors; default matches anchors 1,5,9,...")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Used only when --threshold-mode=fixed.")
    ap.add_argument("--threshold-mode", choices=("fixed", "train_oof_budget"),
                    default="train_oof_budget",
                    help="Calibrate threshold using OOF predictions on the two training sequences only.")
    ap.add_argument("--target-refresh-fraction", type=float, default=3.0/7.0,
                    help="Target fraction of training-only OOF scores above the calibrated threshold.")
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--ridge-alpha", type=float, default=10.0)
    ap.add_argument("--calibration-only", action="store_true",
                    help="Run CPU-only predictor/OOF calibration audit and exit before loading VeloDepth or requiring CUDA.")
    ap.add_argument("--dump-online-features", action="store_true",
                    help="Dump pre-action online features at every candidate anchor for feature consistency audit. Intended with --policies auto.")
    ap.add_argument("--out-dir", default="outputs/p2_3_online_replay/crowd2_32")
    args = ap.parse_args()

    if (args.frames < 3 or args.fixed_interval < 1 or args.min_refresh_gap < 1
            or args.candidate_stride < 1
            or not 0 <= args.candidate_offset < args.candidate_stride):
        ap.error("frames >=3; intervals/stride >=1; candidate offset must be in [0, stride)")
    if not 0.0 < args.target_refresh_fraction < 1.0:
        ap.error("target-refresh-fraction must be strictly between 0 and 1")
    requested_policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    if "candidate_fixed" in requested_policies and args.fixed_interval % args.candidate_stride != 0:
        ap.error("candidate_fixed requires fixed-interval to be divisible by candidate-stride")
    if args.horizon != 16:
        ap.error("This first P2.3 pilot currently uses H=16, matching the planned long-horizon target.")
    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    allowed = {"auto", "fixed8", "fixed16", "candidate_fixed", *MODEL_FEATURES.keys()}
    unknown = set(policies) - allowed
    if unknown:
        ap.error(f"Unknown policies {sorted(unknown)}; allowed: {sorted(allowed)}")
    if args.calibration_only:
        calibration_policies = [p for p in policies if p in MODEL_FEATURES]
        if not calibration_policies:
            ap.error("--calibration-only requires at least one predictor policy: " + ", ".join(MODEL_FEATURES))
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        calibration_rows = []
        calibration_fold_rows = []
        calibration_meta = {
            "experiment": "P2.3 v4 CPU-only temporal-block OOF calibration audit",
            "heldout_test_sequence": args.sequence,
            "horizon": args.horizon,
            "target_refresh_fraction": args.target_refresh_fraction,
            "ridge_alpha": args.ridge_alpha,
            "heldout_test_sequence_excluded_from_training_and_calibration": True,
            "method": "3-fold temporal-block OOF within each of the two training sequences; every fold's train and validation split must include both sequences",
            "predictors": {},
        }
        for p in calibration_policies:
            pred = OnlineRidge(
                args.train_dataset, args.sequence, p, horizon=args.horizon,
                alpha=args.ridge_alpha,
                target_refresh_fraction=args.target_refresh_fraction,
            )
            calibration_rows.extend(pred.oof_rows)
            calibration_fold_rows.extend(pred.fold_rows)
            calibration_meta["predictors"][p] = {
                "training_rows": pred.train_n,
                "oof_rows": pred.oof_n,
                "threshold": pred.calibrated_threshold,
                "oof_fraction_above_threshold": pred.oof_achieved_fraction,
                "oof_score_mean": pred.oof_score_mean,
                "oof_score_std": pred.oof_score_std,
                "oof_score_min": pred.oof_score_min,
                "oof_score_max": pred.oof_score_max,
                "fold_count": len(pred.fold_rows),
            }
            print(
                f"CALIBRATION {p}: train_n={pred.train_n}; OOF_n={pred.oof_n}; "
                f"folds={len(pred.fold_rows)}; threshold={pred.calibrated_threshold:.8f}; "
                f"OOF fraction={pred.oof_achieved_fraction:.3f}; "
                f"score mean/std={pred.oof_score_mean:.6f}/{pred.oof_score_std:.6f}"
            )
        oof_csv = out_dir / "p2_3_threshold_calibration_oof.csv"
        folds_csv = out_dir / "p2_3_threshold_calibration_folds.csv"
        meta_json = out_dir / "p2_3_calibration_only_metadata.json"
        write_csv(oof_csv, calibration_rows)
        write_csv(folds_csv, calibration_fold_rows)
        meta_json.write_text(json.dumps(calibration_meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Wrote:", oof_csv)
        print("Wrote:", folds_csv)
        print("Wrote:", meta_json)
        print("PASS: CPU-only calibration audit complete; no VeloDepth model was loaded and no CUDA was required.")
        return

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
    calibration_rows = []
    calibration_fold_rows = []
    for p in policies:
        if p in MODEL_FEATURES:
            predictors[p] = OnlineRidge(
                args.train_dataset, args.sequence, p, horizon=args.horizon,
                alpha=args.ridge_alpha,
                target_refresh_fraction=args.target_refresh_fraction,
            )
            pred = predictors[p]
            calibration_rows.extend(pred.oof_rows)
            calibration_fold_rows.extend(pred.fold_rows)
            print(
                f"Predictor {p}: train_n={pred.train_n} H={args.horizon}; "
                f"OOF_n={pred.oof_n}; temporal-block folds=3; "
                f"OOF threshold={pred.calibrated_threshold:.8f}; "
                f"OOF rate={pred.oof_achieved_fraction:.3f}; "
                f"OOF score mean/std={pred.oof_score_mean:.6f}/{pred.oof_score_std:.6f}; "
                f"target rate={args.target_refresh_fraction:.3f}; alpha={args.ridge_alpha}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_feature_audit_rows = []
    for policy in policies:
        print(f"\n===== RUN POLICY: {policy} =====")
        rows = run_policy(
            policy, model, base_snapshot, inputs, metas, frames, records, gt,
            paddings, padded_shape, args.sequence, args, predictors.get(policy),
            all_feature_audit_rows
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
    if args.dump_online_features:
        feature_audit_csv = out_dir / "p2_3_online_feature_audit.csv"
        write_csv(feature_audit_csv, all_feature_audit_rows)
        print("Wrote:", feature_audit_csv)
    calibration_csv = out_dir / "p2_3_threshold_calibration_oof.csv"
    calibration_fold_csv = out_dir / "p2_3_threshold_calibration_folds.csv"
    write_csv(calibration_csv, calibration_rows)
    write_csv(calibration_fold_csv, calibration_fold_rows)
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
        "dump_online_features": args.dump_online_features,
        "feature_audit_semantics": (
            "pre-action features captured on the auto policy at candidate anchors; "
            "compare against P2.2.2 dataset rows by sequence+anchor_frame+horizon_H=16"
            if args.dump_online_features else None
        ),
        "predictor_target": TARGET,
        "predictor_horizon": args.horizon,
        "predictor_training": "Ridge fitted at runtime on P2.2.2 H=16 rows from the other two sequences; no VeloDepth weights updated",
        "threshold_calibration_method": "3-fold temporal-block OOF within each of the two training sequences; each fold validates on a contiguous anchor-rank block from both sequences and trains on the remaining blocks from both sequences; held-out test sequence excluded",
        "threshold_calibration_caveat": "Only two training sequences are available for leave-one-sequence-out testing. This is an internal threshold-calibration repair, not sufficient evidence of cross-scene generalization.",
        "policies": policies,
        "fixed_interval": args.fixed_interval,
        "min_refresh_gap": args.min_refresh_gap,
        "candidate_stride": args.candidate_stride,
        "candidate_offset_zero_based": args.candidate_offset,
        "input_isolation": "each action receives a fresh clone of immutable packed inputs",
        "timing_warmup": "each policy gets an unrecorded keyframe+propagation warm-up before measured replay",
        "threshold_mode": args.threshold_mode,
        "fixed_score_threshold": args.threshold,
        "target_refresh_fraction": args.target_refresh_fraction,
        "predictor_calibration": {
            p: {
                "threshold": predictors[p].calibrated_threshold,
                "training_rows": predictors[p].train_n,
                "oof_rows": predictors[p].oof_n,
                "oof_achieved_fraction_above_threshold": predictors[p].oof_achieved_fraction,
                "oof_score_mean": predictors[p].oof_score_mean,
                "oof_score_std": predictors[p].oof_score_std,
                "oof_score_min": predictors[p].oof_score_min,
                "oof_score_max": predictors[p].oof_score_max,
                "oof_sequences": sorted(set(r["oof_sequence"] for r in predictors[p].oof_rows)),
                "heldout_test_sequence_excluded_from_calibration": args.sequence,
            }
            for p in predictors
        },
        "candidate_fixed_semantics": "refresh at candidate-anchor-aligned frames every fixed_interval frames (offset=candidate_offset), subject to min-refresh-gap; e.g. t=9,17,25 for stride=4, offset=1, interval=8",
        "scheduled_refresh_semantics": "direct full keyframe; skip propagation on that frame; disable overlap blending for that scheduled frame; seed fast-tail fields from refreshed state",
        "automatic_restart_semantics": "original adapter behavior: propagate candidate, discard if restart=True, then keyframe same frame",
        "timing_scope": "model action replay only; excludes RGB preprocessing, optical-flow construction, packed-input capture, and metric evaluation; not end-to-end latency/FPS",
        "outputs": {
            "frame_metrics": str(frame_csv),
            "summary": str(summary_csv),
            "metadata": str(metadata_json),
            "threshold_calibration_oof": str(calibration_csv),
            "threshold_calibration_folds": str(calibration_fold_csv),
        },
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== P2.3 PILOT SUMMARY =====")
    print(pd.DataFrame(summary)[[
        "policy", "frames_with_valid_metrics", "mean_absrel", "mean_rmse", "mean_delta1",
        "policy_refresh_count", "auto_restart_count", "mean_model_action_ms",
        "median_model_action_ms", "p95_model_action_ms", "max_model_action_ms", "mean_decision_ms"
    ]].to_string(index=False))
    print("\nWrote:", frame_csv)
    print("Wrote:", summary_csv)
    print("Wrote:", metadata_json)
    print("Wrote:", calibration_csv)
    print("Wrote:", calibration_fold_csv)
    print("WARNING: this is an exploratory replay, not a final same-budget or end-to-end latency result.")


if __name__ == "__main__":
    main()
