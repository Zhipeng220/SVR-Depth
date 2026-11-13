#!/usr/bin/env python3
"""SVR-Depth Y01: natural propagation error diagnostics.

Runs the audited native VeloDepth A trajectory only (no repair, no training),
then diagnoses scale/bias error, post-oracle-calibration spatial residuals, and
error conditioned on state age, motion magnitude, and a forward/backward-flow
inconsistency proxy for possible occlusion.

The oracle affine calibration is strictly offline diagnostic analysis; it is not
inserted into the online inference path and is not a deployable result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

import gt4a2_exp1_train_eval as exp1  # noqa: E402
import gt2_fixed_clip_repair_train as gt2  # noqa: E402

REFRESH_INTERVAL = 32
TILE_GRID = 8
VALID_SEQUENCES = {"crowd2", "crowd3", "balloon2"}
AGE_BANDS = (
    (0, 0, "keyframe_age0"),
    (1, 7, "early_age1_7"),
    (8, 15, "mid1_age8_15"),
    (16, 23, "mid2_age16_23"),
    (24, 31, "late_age24_31"),
)

# Lock to the source state previously audited for E0/Y02.
EXPECTED_SOURCE_HASHES = {
    "experiments/p1/gt4a2_c3_closed_loop.py":
        "76327208acac4543b2a87ccba8b7af435e4c5ce65267b42052e51b435cbad7f5",
    "experiments/p1/gt4a2_exp1_train_eval.py":
        "c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f",
    "experiments/p1/gt2_fixed_clip_repair_train.py":
        "bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192",
    "velodepth/models/decoder_velo.py":
        "35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08",
    "velodepth/models/persistent_feature_repair.py":
        "6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="Y01 natural propagation error diagnostics; native A only, no training.")
    ap.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    ap.add_argument("--crowd2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv")
    ap.add_argument("--crowd3-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv")
    ap.add_argument("--balloon2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv")
    ap.add_argument("--sequences", default="crowd2,crowd3,balloon2",
                    help="Comma-separated subset of crowd2,crowd3,balloon2.")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="0=all frames; otherwise use the first N frames per selected sequence.")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--smoke-only", action="store_true",
                    help="Run a short mechanics smoke (defaulting to Crowd2, 64 frames).")
    ap.add_argument("--out-dir", default="outputs/gt4a2_y01_validation")
    args = ap.parse_args()
    if args.max_frames < 0:
        ap.error("--max-frames must be >= 0")
    seqs = [x.strip().lower() for x in args.sequences.split(",") if x.strip()]
    if not seqs or any(x not in VALID_SEQUENCES for x in seqs) or len(set(seqs)) != len(seqs):
        ap.error("--sequences must be a unique comma-separated subset of crowd2,crowd3,balloon2")
    if args.smoke_only:
        if args.sequences == "crowd2,crowd3,balloon2":
            seqs = ["crowd2"]
        args.max_frames = min(args.max_frames, 64) if args.max_frames else 64
    args.sequences_list = seqs
    return args


def verify_source_hashes() -> dict:
    result = {}
    for rel, expected in EXPECTED_SOURCE_HASHES.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Required audited source is missing: {path}")
        actual = sha256_file(path)
        result[rel] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            raise RuntimeError(f"Source changed since E0; refusing Y01: {rel}: {actual} != {expected}")
    return result


def pair_paths_for(args) -> dict[str, Path]:
    all_paths = {
        "crowd2": Path(args.crowd2_pair_csv).resolve(),
        "crowd3": Path(args.crowd3_pair_csv).resolve(),
        "balloon2": Path(args.balloon2_pair_csv).resolve(),
    }
    return {name: all_paths[name] for name in args.sequences_list}


def build_preflight(args) -> dict:
    if not args.snapshot:
        raise RuntimeError("VDEEP_SNAPSHOT is unset; pass --snapshot or export VDEEP_SNAPSHOT.")
    snapshot = Path(args.snapshot).resolve()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pretrained snapshot incomplete: {snapshot}")
    source_hashes = verify_source_hashes()
    manifests = {}
    for name, path in pair_paths_for(args).items():
        rows = exp1.read_manifest(path, training=False)
        selected = rows[:args.max_frames] if args.max_frames else rows
        matched = sum(str(r.get("gt_match_status", "")).strip().upper() == "MATCHED" for r in selected)
        manifests[name] = {
            "path": str(path), "sha256": sha256_file(path),
            "rgb_rows_total": len(rows), "rgb_rows_selected": len(selected),
            "gt_matched_selected": matched,
            "keyframe_rows_selected": sum(int(r["frame_index_zero_based"]) % REFRESH_INTERVAL == 0 for r in selected),
            "state_age_definition": "frame_index_zero_based % 32; age 0 is keyframe refresh",
        }
    return {
        "status": "PREFLIGHT_PASS",
        "experiment": "Y01 natural propagation error diagnostics",
        "root": str(ROOT), "snapshot": str(snapshot),
        "pretrained_weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "source_hashes": source_hashes,
        "manifests": manifests,
        "protocol": {
            "native_A_only": True, "training_performed": False,
            "refresh_interval": REFRESH_INTERVAL, "tile_grid": TILE_GRID,
            "age_bands": [{"min_age": lo, "max_age": hi, "label": label} for lo, hi, label in AGE_BANDS],
            "motion_proxy": "DIS optical-flow magnitude on original RGB resolution",
            "occlusion_proxy": "forward/backward flow inconsistency; not ground-truth occlusion",
            "oracle_calibration": "per-frame least-squares GT = scale * prediction + bias; offline diagnostic only",
            "no_oracle_in_online_path": True,
        },
    }


def age_band(age: int) -> str:
    for lo, hi, label in AGE_BANDS:
        if lo <= age <= hi:
            return label
    raise ValueError(f"Invalid state age: {age}")


def prediction_numpy(radius, rays_map, shape, model) -> np.ndarray:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        pred = gt2.make_z_depth(radius, rays_map, shape["padded_shape"], shape["paddings"], model.interpolation_mode)
    return pred.detach().float().cpu().numpy()


def valid_depth_mask(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)


def calculate_masked_metrics(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    mask = mask & (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    n = int(mask.sum())
    if n == 0:
        return {"n_pixels": 0, "AbsRel": "", "RMSE": "", "delta1": "", "signed_bias_mean_m": "", "residual_mae_m": ""}
    g = gt[mask].astype(np.float64)
    p = pred[mask].astype(np.float64)
    residual = p - g
    ratio = np.maximum(g / p, p / g)
    return {
        "n_pixels": n,
        "AbsRel": float(np.mean(np.abs(residual) / g)),
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "signed_bias_mean_m": float(np.mean(residual)),
        "residual_mae_m": float(np.mean(np.abs(residual))),
    }


def fit_frame_scale_bias(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """Fit the deliberately optimistic per-frame oracle relation GT ~= scale * pred + bias."""
    x = pred[mask].astype(np.float64)
    y = gt[mask].astype(np.float64)
    if x.size < 1000:
        raise RuntimeError(f"Too few valid pixels for per-frame scale/bias fit: {x.size}")
    xm, ym = float(x.mean()), float(y.mean())
    var = float(np.mean((x - xm) ** 2))
    if not np.isfinite(var) or var < 1e-12:
        scale = 1.0
        bias = float(np.median(y - x))
    else:
        scale = float(np.mean((x - xm) * (y - ym)) / var)
        bias = float(ym - scale * xm)
    fitted = scale * x + bias
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    ratio = y / np.maximum(x, 1e-8)
    return {
        "oracle_scale_gt_from_pred": scale,
        "oracle_bias_m": bias,
        "oracle_fit_r2": float(r2),
        "median_gt_over_pred_ratio": float(np.median(ratio)),
        "mean_gt_over_pred_ratio": float(np.mean(ratio)),
        "oracle_scale_positive": bool(scale > 0),
    }


def flow_motion_occlusion(prev_rgb: np.ndarray, curr_rgb: np.ndarray, dis) -> tuple[np.ndarray, np.ndarray]:
    """Return current-pixel backward-flow magnitude and FB-inconsistency proxy mask."""
    prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    flow_fwd = dis.calc(prev_gray, curr_gray, None)  # previous -> current
    flow_bwd = dis.calc(curr_gray, prev_gray, None)  # current -> previous; aligned to current pixels
    h, w = curr_gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xx + flow_bwd[..., 0]
    map_y = yy + flow_bwd[..., 1]
    in_bounds = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
    fwd_at_prev = cv2.remap(
        flow_fwd, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0.0, 0.0)
    )
    fb = flow_bwd + fwd_at_prev
    fb_err = np.sqrt(np.sum(fb.astype(np.float32) ** 2, axis=2))
    mag_bwd = np.sqrt(np.sum(flow_bwd.astype(np.float32) ** 2, axis=2))
    mag_fwd = np.sqrt(np.sum(fwd_at_prev.astype(np.float32) ** 2, axis=2))
    threshold = 1.0 + 0.05 * (mag_bwd + mag_fwd)
    inconsistent = (~in_bounds) | (fb_err > threshold) | (~np.isfinite(mag_bwd))
    return mag_bwd.astype(np.float32), inconsistent.astype(bool)


def metric_fields(prefix: str, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    m = calculate_masked_metrics(gt, pred, mask)
    return {f"{prefix}_{k}": v for k, v in m.items()}


def residual_diagnostics(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, scale: float, bias: float) -> dict:
    raw = calculate_masked_metrics(gt, pred, mask)
    cal_unclipped = scale * pred.astype(np.float64) + bias
    nonpositive_fraction = float(np.mean(cal_unclipped[mask] <= 0)) if mask.any() else 1.0
    cal = np.maximum(cal_unclipped, 1e-3).astype(np.float32)
    oracle = calculate_masked_metrics(gt, cal, mask)
    r = cal.astype(np.float64) - gt.astype(np.float64)
    vals = r[mask]
    # Gradient residual only where both neighboring pixels are valid.
    mx = mask[:, 1:] & mask[:, :-1]
    my = mask[1:, :] & mask[:-1, :]
    gx = np.abs(r[:, 1:] - r[:, :-1])[mx]
    gy = np.abs(r[1:, :] - r[:-1, :])[my]
    grad_values = np.concatenate([gx, gy]) if gx.size + gy.size else np.asarray([], dtype=np.float64)
    return {
        "raw_AbsRel": raw["AbsRel"], "raw_RMSE": raw["RMSE"], "raw_delta1": raw["delta1"],
        "raw_signed_bias_mean_m": raw["signed_bias_mean_m"],
        "oracle_AbsRel": oracle["AbsRel"], "oracle_RMSE": oracle["RMSE"], "oracle_delta1": oracle["delta1"],
        "oracle_signed_residual_mean_m": float(np.mean(vals)) if vals.size else "",
        "oracle_signed_residual_median_m": float(np.median(vals)) if vals.size else "",
        "oracle_residual_mae_m": float(np.mean(np.abs(vals))) if vals.size else "",
        "oracle_residual_p95_abs_m": float(np.percentile(np.abs(vals), 95)) if vals.size else "",
        "oracle_residual_gradient_l1_m": float(np.mean(grad_values)) if grad_values.size else "",
        "oracle_nonpositive_before_clamp_fraction": nonpositive_fraction,
    }, cal


def synthetic_perturbation_rows(sequence: str, idx: int, age: int, band: str,
                                gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> list[dict]:
    """Auxiliary sanity checks only; never used to fit or alter the online model."""
    rng_seed = 20261025 + idx + sum(ord(c) for c in sequence) * 100003
    rng = np.random.default_rng(rng_seed)
    noise = rng.normal(0.0, 0.01 * np.maximum(pred.astype(np.float64), 1e-3), size=pred.shape)
    variants = {
        "scale_plus_10pct": pred.astype(np.float32) * 1.10,
        "scale_minus_10pct": pred.astype(np.float32) * 0.90,
        "bias_plus_0p05m": pred.astype(np.float32) + 0.05,
        "bias_minus_0p05m": np.maximum(pred.astype(np.float32) - 0.05, 1e-3),
        "gaussian_noise_sigma_1pct_depth": np.maximum(pred.astype(np.float64) + noise, 1e-3).astype(np.float32),
    }
    out = []
    for name, variant in variants.items():
        metrics = calculate_masked_metrics(gt, variant, valid)
        out.append({
            "sequence": sequence, "frame_index": idx, "state_age": age, "age_band": band,
            "synthetic_variant": name, "n_pixels": metrics["n_pixels"],
            "AbsRel": metrics["AbsRel"], "RMSE": metrics["RMSE"],
            "delta1": metrics["delta1"], "signed_bias_mean_m": metrics["signed_bias_mean_m"],
            "note": "Artificial perturbation is an auxiliary metric-sensitivity check only; not a model branch or causal evidence.",
        })
    return out


def make_tile_rows(sequence: str, idx: int, age: int, band: str, gt: np.ndarray, pred: np.ndarray,
                   cal_pred: np.ndarray, valid: np.ndarray, flow_mag: np.ndarray | None,
                   occ_proxy: np.ndarray | None) -> list[dict]:
    h, w = gt.shape
    y_edges = np.linspace(0, h, TILE_GRID + 1, dtype=int)
    x_edges = np.linspace(0, w, TILE_GRID + 1, dtype=int)
    rows = []
    for gy in range(TILE_GRID):
        for gx in range(TILE_GRID):
            y0, y1 = int(y_edges[gy]), int(y_edges[gy + 1])
            x0, x1 = int(x_edges[gx]), int(x_edges[gx + 1])
            m = valid[y0:y1, x0:x1]
            if int(m.sum()) == 0:
                continue
            g = gt[y0:y1, x0:x1]
            p = pred[y0:y1, x0:x1]
            cp = cal_pred[y0:y1, x0:x1]
            raw = calculate_masked_metrics(g, p, m)
            cal = calculate_masked_metrics(g, cp, m)
            raw_res = p[m].astype(np.float64) - g[m].astype(np.float64)
            cal_res = cp[m].astype(np.float64) - g[m].astype(np.float64)
            row = {
                "sequence": sequence, "frame_index": idx, "state_age": age, "age_band": band,
                "tile_y": gy, "tile_x": gx, "n_valid_pixels": int(m.sum()),
                "raw_AbsRel": raw["AbsRel"], "raw_RMSE": raw["RMSE"],
                "raw_signed_bias_mean_m": float(np.mean(raw_res)),
                "oracle_AbsRel": cal["AbsRel"], "oracle_RMSE": cal["RMSE"],
                "oracle_signed_residual_mean_m": float(np.mean(cal_res)),
                "oracle_residual_mae_m": float(np.mean(np.abs(cal_res))),
            }
            if flow_mag is not None:
                fm = flow_mag[y0:y1, x0:x1]
                row["mean_flow_mag_px"] = float(np.mean(fm[np.isfinite(fm)])) if np.isfinite(fm).any() else ""
            else:
                row["mean_flow_mag_px"] = ""
            if occ_proxy is not None:
                row["occlusion_proxy_fraction"] = float(np.mean(occ_proxy[y0:y1, x0:x1]))
            else:
                row["occlusion_proxy_fraction"] = ""
            rows.append(row)
    return rows


def pixel_group_rows(sequence: str, idx: int, age: int, band: str, gt: np.ndarray, pred: np.ndarray,
                     cal_pred: np.ndarray, valid: np.ndarray, flow_mag: np.ndarray | None,
                     occ_proxy: np.ndarray | None) -> list[dict]:
    if flow_mag is None or occ_proxy is None:
        return []
    vals = flow_mag[valid & np.isfinite(flow_mag)]
    if vals.size < 1000:
        return []
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    motion_bins = {
        "motion_low_within_frame": valid & np.isfinite(flow_mag) & (flow_mag <= q1),
        "motion_mid_within_frame": valid & np.isfinite(flow_mag) & (flow_mag > q1) & (flow_mag <= q2),
        "motion_high_within_frame": valid & np.isfinite(flow_mag) & (flow_mag > q2),
    }
    masks = dict(motion_bins)
    masks["fb_consistent_proxy"] = valid & (~occ_proxy)
    masks["fb_inconsistent_possible_occlusion"] = valid & occ_proxy
    rows = []
    for group, mask in masks.items():
        raw = calculate_masked_metrics(gt, pred, mask)
        cal = calculate_masked_metrics(gt, cal_pred, mask)
        rows.append({
            "sequence": sequence, "frame_index": idx, "state_age": age, "age_band": band,
            "pixel_group": group, "n_pixels": raw["n_pixels"],
            "raw_AbsRel": raw["AbsRel"], "raw_RMSE": raw["RMSE"],
            "raw_delta1": raw["delta1"], "raw_signed_bias_mean_m": raw["signed_bias_mean_m"],
            "oracle_AbsRel": cal["AbsRel"], "oracle_RMSE": cal["RMSE"],
            "oracle_delta1": cal["delta1"], "oracle_signed_residual_mean_m": cal["signed_bias_mean_m"],
        })
    return rows


def safe_mean(values):
    vals = [float(x) for x in values if x not in (None, "") and np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else ""


def aggregate_frame_rows(frame_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    by_age = defaultdict(list)
    by_motion = defaultdict(list)
    for row in frame_rows:
        by_age[(row["sequence"], row["age_band"])].append(row)
    age_summary = []
    for (seq, band), rr in sorted(by_age.items()):
        age_summary.append({
            "sequence": seq, "age_band": band, "n_gt_frames": len(rr),
            "mean_raw_AbsRel": safe_mean([r["raw_AbsRel"] for r in rr]),
            "mean_oracle_AbsRel": safe_mean([r["oracle_AbsRel"] for r in rr]),
            "mean_raw_RMSE": safe_mean([r["raw_RMSE"] for r in rr]),
            "mean_oracle_RMSE": safe_mean([r["oracle_RMSE"] for r in rr]),
            "mean_raw_delta1": safe_mean([r["raw_delta1"] for r in rr]),
            "mean_oracle_delta1": safe_mean([r["oracle_delta1"] for r in rr]),
            "mean_oracle_scale": safe_mean([r["oracle_scale_gt_from_pred"] for r in rr]),
            "mean_oracle_bias_m": safe_mean([r["oracle_bias_m"] for r in rr]),
            "mean_flow_mag_px": safe_mean([r["mean_flow_mag_px"] for r in rr]),
            "mean_occlusion_proxy_fraction": safe_mean([r["occlusion_proxy_fraction"] for r in rr]),
            "note": "Macro average over matched frames; per-frame oracle uses each frame GT and is diagnostic only.",
        })
    # Define frame motion strata by per-sequence terciles of mean flow magnitude.
    seq_motion = defaultdict(list)
    for row in frame_rows:
        if row.get("mean_flow_mag_px") not in (None, "") and np.isfinite(float(row["mean_flow_mag_px"])):
            seq_motion[row["sequence"]].append(float(row["mean_flow_mag_px"]))
    thresholds = {seq: np.quantile(v, [1 / 3, 2 / 3]).tolist() for seq, v in seq_motion.items() if v}
    for row in frame_rows:
        v = row.get("mean_flow_mag_px")
        if v in (None, "") or row["sequence"] not in thresholds:
            row["frame_motion_group"] = "unavailable"
            continue
        q1, q2 = thresholds[row["sequence"]]
        v = float(v)
        row["frame_motion_group"] = "low_motion_frame" if v <= q1 else ("mid_motion_frame" if v <= q2 else "high_motion_frame")
        by_motion[(row["sequence"], row["frame_motion_group"])].append(row)
    motion_summary = []
    for (seq, group), rr in sorted(by_motion.items()):
        motion_summary.append({
            "sequence": seq, "frame_motion_group": group, "n_gt_frames": len(rr),
            "motion_mean_px": safe_mean([r["mean_flow_mag_px"] for r in rr]),
            "raw_AbsRel": safe_mean([r["raw_AbsRel"] for r in rr]),
            "oracle_AbsRel": safe_mean([r["oracle_AbsRel"] for r in rr]),
            "raw_RMSE": safe_mean([r["raw_RMSE"] for r in rr]),
            "oracle_RMSE": safe_mean([r["oracle_RMSE"] for r in rr]),
            "occlusion_proxy_fraction": safe_mean([r["occlusion_proxy_fraction"] for r in rr]),
            "note": "Frame strata use per-sequence terciles of mean original-resolution DIS flow magnitude.",
        })
    # Per-sequence overall summary.
    seq_summary = []
    by_seq = defaultdict(list)
    for row in frame_rows:
        by_seq[row["sequence"]].append(row)
    for seq, rr in sorted(by_seq.items()):
        raw = safe_mean([r["raw_AbsRel"] for r in rr])
        oracle = safe_mean([r["oracle_AbsRel"] for r in rr])
        seq_summary.append({
            "sequence": seq, "n_gt_frames": len(rr),
            "mean_raw_AbsRel": raw, "mean_oracle_AbsRel": oracle,
            "relative_AbsRel_reduction_after_per_frame_oracle_pct":
                (float(raw) - float(oracle)) / max(float(raw), 1e-12) * 100.0 if raw != "" and oracle != "" else "",
            "mean_raw_RMSE": safe_mean([r["raw_RMSE"] for r in rr]),
            "mean_oracle_RMSE": safe_mean([r["oracle_RMSE"] for r in rr]),
            "mean_raw_delta1": safe_mean([r["raw_delta1"] for r in rr]),
            "mean_oracle_delta1": safe_mean([r["oracle_delta1"] for r in rr]),
            "mean_oracle_scale": safe_mean([r["oracle_scale_gt_from_pred"] for r in rr]),
            "median_oracle_scale": float(np.median([float(r["oracle_scale_gt_from_pred"]) for r in rr])),
            "mean_oracle_bias_m": safe_mean([r["oracle_bias_m"] for r in rr]),
            "mean_oracle_fit_r2": safe_mean([r["oracle_fit_r2"] for r in rr]),
            "mean_oracle_residual_gradient_l1_m": safe_mean([r["oracle_residual_gradient_l1_m"] for r in rr]),
            "mean_flow_mag_px": safe_mean([r["mean_flow_mag_px"] for r in rr]),
            "mean_occlusion_proxy_fraction": safe_mean([r["occlusion_proxy_fraction"] for r in rr]),
            "note": "Per-frame oracle affine fit is deliberately optimistic and uses each frame GT; diagnostic only, not a deployable baseline.",
        })
    return age_summary, motion_summary, seq_summary


def aggregate_pixel_groups(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["sequence"], r["age_band"], r["pixel_group"])].append(r)
    out = []
    for (seq, band, group), rr in sorted(groups.items()):
        out.append({
            "sequence": seq, "age_band": band, "pixel_group": group,
            "n_frame_group_rows": len(rr),
            "mean_pixels_per_frame": safe_mean([r["n_pixels"] for r in rr]),
            "mean_raw_AbsRel": safe_mean([r["raw_AbsRel"] for r in rr]),
            "mean_oracle_AbsRel": safe_mean([r["oracle_AbsRel"] for r in rr]),
            "mean_raw_RMSE": safe_mean([r["raw_RMSE"] for r in rr]),
            "mean_oracle_RMSE": safe_mean([r["oracle_RMSE"] for r in rr]),
            "mean_raw_signed_bias_m": safe_mean([r["raw_signed_bias_mean_m"] for r in rr]),
            "mean_oracle_signed_residual_m": safe_mean([r["oracle_signed_residual_mean_m"] for r in rr]),
            "note": "Macro average over per-frame pixel groups; flow-consistency groups are only an occlusion proxy.",
        })
    return out


def aggregate_spatial_tiles(tile_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in tile_rows:
        groups[(r["sequence"], r["age_band"], int(r["tile_y"]), int(r["tile_x"]))].append(r)
    out = []
    for (seq, band, gy, gx), rr in sorted(groups.items()):
        out.append({
            "sequence": seq, "age_band": band, "tile_y": gy, "tile_x": gx,
            "n_frame_tiles": len(rr),
            "mean_raw_AbsRel": safe_mean([r["raw_AbsRel"] for r in rr]),
            "mean_oracle_AbsRel": safe_mean([r["oracle_AbsRel"] for r in rr]),
            "mean_raw_RMSE": safe_mean([r["raw_RMSE"] for r in rr]),
            "mean_oracle_RMSE": safe_mean([r["oracle_RMSE"] for r in rr]),
            "mean_raw_signed_bias_m": safe_mean([r["raw_signed_bias_mean_m"] for r in rr]),
            "mean_oracle_signed_residual_m": safe_mean([r["oracle_signed_residual_mean_m"] for r in rr]),
            "mean_oracle_residual_mae_m": safe_mean([r["oracle_residual_mae_m"] for r in rr]),
            "mean_flow_mag_px": safe_mean([r["mean_flow_mag_px"] for r in rr]),
            "mean_occlusion_proxy_fraction": safe_mean([r["occlusion_proxy_fraction"] for r in rr]),
            "note": "8x8 image tile macro averages; calibrated residual uses per-frame GT-oracle scale/bias.",
        })
    return out


def build_failure_frames(frame_rows: list[dict], top_k: int = 25) -> list[dict]:
    out = []
    by_seq = defaultdict(list)
    for r in frame_rows:
        by_seq[r["sequence"]].append(r)
    for seq, rr in sorted(by_seq.items()):
        raw_sorted = sorted(rr, key=lambda x: float(x["raw_AbsRel"]), reverse=True)
        oracle_sorted = sorted(rr, key=lambda x: float(x["oracle_AbsRel"]), reverse=True)
        rank_raw = {int(r["frame_index"]): i + 1 for i, r in enumerate(raw_sorted[:top_k])}
        rank_oracle = {int(r["frame_index"]): i + 1 for i, r in enumerate(oracle_sorted[:top_k])}
        chosen = set(rank_raw) | set(rank_oracle)
        for r in rr:
            idx = int(r["frame_index"])
            if idx in chosen:
                out.append({**r, "rank_worst_raw_AbsRel": rank_raw.get(idx, ""),
                            "rank_worst_oracle_AbsRel": rank_oracle.get(idx, "")})
    return out


def zip_review(out_dir: Path, filenames: list[str]) -> Path:
    path = out_dir / "y01_review.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in filenames:
            p = out_dir / name
            if p.is_file():
                zf.write(p, arcname=name)
        script_path = Path(__file__).resolve()
        if script_path.is_file():
            zf.write(script_path, arcname=script_path.name)
    return path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "y01_summary.json"
    if summary_path.exists() and not args.preflight_only:
        raise RuntimeError(f"Refusing to overwrite existing result: {summary_path}; choose a new --out-dir")

    pre = build_preflight(args)
    write_json(out_dir / "y01_preflight.json", pre)
    print(f"[PASS] Y01 preflight: sequences={args.sequences_list}, native A only, K={REFRESH_INTERVAL}", flush=True)
    if args.preflight_only:
        print("Y01 preflight complete; no inference or training performed.", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Y01 inference requires CUDA; no training is performed.")

    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    np.random.seed(20261025)
    torch.cuda.manual_seed_all(20261025)
    snapshot = Path(args.snapshot).resolve()
    pair_paths = pair_paths_for(args)

    frame_rows: list[dict] = []
    tile_rows: list[dict] = []
    pixel_rows: list[dict] = []
    synthetic_rows: list[dict] = []
    runtime_rows: list[dict] = []
    start_all = time.perf_counter()

    for seq, pair_path in pair_paths.items():
        rows_all = exp1.read_manifest(pair_path, training=False)
        rows = rows_all[:args.max_frames] if args.max_frames else rows_all
        model = gt2.load_model(snapshot, "off", device)
        model.eval()
        shape = exp1.shape_info_for(model, Path(rows[0]["rgb_path"]), device)
        dis_model = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        dis_diag = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        prev_frame = None
        prev_rgb = None
        radius = confidence = features = rays_map = None
        seq_start = time.perf_counter()
        print(f"[{seq}] native A trajectory: frames={len(rows)}, matched_GT="
              f"{sum(str(r.get('gt_match_status','')).strip().upper() == 'MATCHED' for r in rows)}", flush=True)

        for idx, row in enumerate(rows):
            matched = str(row.get("gt_match_status", "")).strip().upper() == "MATCHED"
            frame = exp1.load_frame(row, shape, device, need_gt=matched)
            rgb_original = gt2.load_rgb(Path(row["rgb_path"]))
            if rgb_original.shape[:2] != shape["original_shape"]:
                raise RuntimeError(f"RGB size changed at {seq}/{idx}: {rgb_original.shape}")

            if idx % REFRESH_INTERVAL == 0:
                rays_map, radius, confidence, features = exp1.keyframe_state(model, frame, shape)
            else:
                static = exp1.transition_static(model, prev_frame, frame, shape, dis_model)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    radius, confidence, features = exp1.one_fast_step(
                        model, prev_frame, frame, static, radius, confidence, features
                    )

            pred_np = prediction_numpy(radius, rays_map, shape, model)
            if pred_np.shape != shape["original_shape"]:
                raise RuntimeError(f"Prediction/GT resolution mismatch at {seq}/{idx}: {pred_np.shape} vs {shape['original_shape']}")

            age = idx % REFRESH_INTERVAL
            band = age_band(age)
            flow_mag = occ_proxy = None
            if prev_rgb is not None:
                flow_mag, occ_proxy = flow_motion_occlusion(prev_rgb, rgb_original, dis_diag)
                mean_flow = float(np.mean(flow_mag[np.isfinite(flow_mag)]))
                median_flow = float(np.median(flow_mag[np.isfinite(flow_mag)]))
                p90_flow = float(np.percentile(flow_mag[np.isfinite(flow_mag)], 90))
                occ_fraction = float(np.mean(occ_proxy))
                flow_available = True
            else:
                mean_flow = median_flow = p90_flow = occ_fraction = ""
                flow_available = False

            if matched:
                gt_np = frame["gt_np"]
                if gt_np is None or gt_np.shape != pred_np.shape:
                    raise RuntimeError(f"Missing or mismatched GT at {seq}/{idx}")
                valid = valid_depth_mask(gt_np, pred_np)
                if int(valid.sum()) < 1000:
                    raise RuntimeError(f"Too few valid pixels at {seq}/{idx}: {int(valid.sum())}")
                scale_bias = fit_frame_scale_bias(gt_np, pred_np, valid)
                residual, cal_pred = residual_diagnostics(
                    gt_np, pred_np, valid,
                    float(scale_bias["oracle_scale_gt_from_pred"]), float(scale_bias["oracle_bias_m"])
                )
                row_out = {
                    "sequence": seq, "frame_index": idx, "rgb_file": Path(row["rgb_path"]).name,
                    "rgb_path": str(row["rgb_path"]), "gt_path": str(row.get("depth_path", "")),
                    "rgb_timestamp": row.get("rgb_timestamp", ""), "abs_delta_ms": row.get("abs_delta_ms", ""),
                    "state_age": age, "age_band": band, "is_keyframe": bool(age == 0),
                    "gt_valid_ratio": float(np.mean(gt_np > 0)), "valid_pixels": int(valid.sum()),
                    "mean_flow_mag_px": mean_flow, "median_flow_mag_px": median_flow,
                    "p90_flow_mag_px": p90_flow, "occlusion_proxy_fraction": occ_fraction,
                    "flow_available": flow_available,
                    **scale_bias, **residual,
                }
                frame_rows.append(row_out)
                tile_rows.extend(make_tile_rows(seq, idx, age, band, gt_np, pred_np, cal_pred, valid, flow_mag, occ_proxy))
                pixel_rows.extend(pixel_group_rows(seq, idx, age, band, gt_np, pred_np, cal_pred, valid, flow_mag, occ_proxy))
                synthetic_rows.extend(synthetic_perturbation_rows(seq, idx, age, band, gt_np, pred_np, valid))

            prev_frame = frame
            prev_rgb = rgb_original
            if idx == 0 or (idx + 1) % 100 == 0 or idx == len(rows) - 1:
                print(f"[{seq}] processed {idx + 1}/{len(rows)} frames", flush=True)

        runtime_rows.append({
            "sequence": seq, "frames_processed": len(rows),
            "gt_matched_frames": sum(r["sequence"] == seq for r in frame_rows),
            "wall_sec_including_load_flow_inference_metrics": time.perf_counter() - seq_start,
        })
        del model, dis_model, dis_diag, prev_frame, prev_rgb, radius, confidence, features, rays_map
        torch.cuda.empty_cache()

    age_summary, motion_summary, seq_summary = aggregate_frame_rows(frame_rows)
    pixel_summary = aggregate_pixel_groups(pixel_rows)
    spatial_summary = aggregate_spatial_tiles(tile_rows)
    failure_rows = build_failure_frames(frame_rows)

    write_csv(out_dir / "y01_frame_metrics.csv", frame_rows)
    write_csv(out_dir / "y01_age_summary.csv", age_summary)
    write_csv(out_dir / "y01_frame_motion_summary.csv", motion_summary)
    write_csv(out_dir / "y01_pixel_group_metrics.csv", pixel_rows)
    write_csv(out_dir / "y01_synthetic_perturbation.csv", synthetic_rows)
    write_csv(out_dir / "y01_pixel_group_summary.csv", pixel_summary)
    write_csv(out_dir / "y01_spatial_residual_tiles.csv", tile_rows)
    write_csv(out_dir / "y01_spatial_residual_summary.csv", spatial_summary)
    write_csv(out_dir / "y01_failure_frames.csv", failure_rows)
    write_csv(out_dir / "y01_runtime.csv", runtime_rows)

    summary = {
        "status": "DIAGNOSTICS_PASS",
        "experiment": "Y01 natural propagation error diagnostics",
        "training_performed": False,
        "is_smoke_or_partial": bool(args.max_frames),
        "max_frames_per_sequence": args.max_frames,
        "runtime_sec": time.perf_counter() - start_all,
        "selected_sequences": args.sequences_list,
        "refresh_interval": REFRESH_INTERVAL,
        "tile_grid": f"{TILE_GRID}x{TILE_GRID}",
        "counts": {
            "frame_metric_rows": len(frame_rows), "age_summary_rows": len(age_summary),
            "frame_motion_summary_rows": len(motion_summary), "pixel_group_metric_rows": len(pixel_rows),
            "pixel_group_summary_rows": len(pixel_summary), "synthetic_perturbation_rows": len(synthetic_rows),
            "spatial_tile_rows": len(tile_rows),
            "spatial_summary_rows": len(spatial_summary), "failure_frame_rows": len(failure_rows),
        },
        "sequence_summary": seq_summary,
        "mechanics": {
            "all_selected_frames_follow_native_A": True,
            "repair_head_loaded_or_applied": False,
            "training_performed": False,
            "per_frame_oracle_uses_GT": True,
            "oracle_used_in_online_path": False,
            "occlusion_is_flow_consistency_proxy_not_GT_label": True,
            "all_frame_predictions_shape_checked": True,
        },
        "interpretation_limits": [
            "Y01 evaluates native A only; no repair checkpoint is loaded and no training is performed.",
            "Per-frame scale/bias calibration uses the same frame GT and is an optimistic oracle diagnostic, not a deployable baseline.",
            "The flow-based possible-occlusion mask is a forward/backward consistency proxy, not ground-truth occlusion.",
            "Motion bins are relative within-frame pixel flow terciles; frame-motion strata use per-sequence mean-flow terciles.",
            "Frames/snapshots from the same video are correlated; summaries are descriptive and not independent-sample significance tests.",
            "Crowd2, Crowd3 and Balloon2 are development sequences, not final unbiased test domains.",
            "This experiment does not establish latency gains or prove that a specific residual mechanism causes the error.",
        ],
    }
    write_json(out_dir / "y01_summary.json", summary)
    write_json(out_dir / "y01_preflight.json", pre)
    review_files = [
        "y01_summary.json", "y01_preflight.json", "y01_frame_metrics.csv", "y01_age_summary.csv",
        "y01_frame_motion_summary.csv", "y01_pixel_group_metrics.csv", "y01_pixel_group_summary.csv",
        "y01_synthetic_perturbation.csv",
        "y01_spatial_residual_tiles.csv", "y01_spatial_residual_summary.csv", "y01_failure_frames.csv", "y01_runtime.csv",
    ]
    review_zip = zip_review(out_dir, review_files)
    print(f"[DIAGNOSTICS_PASS] Y01 review package: {review_zip}", flush=True)
    print(json.dumps({"sequence_summary": seq_summary, "counts": summary["counts"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
