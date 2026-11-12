#!/usr/bin/env python3
"""SVR-Depth Q0-3: merged necessity controls for persistent feature writeback.

One locked experiment compares:
A native VeloDepth; B1 global depth scale/bias; B2 channel-wise affine output-only;
B3 a trained output-only adapter plus frozen C/D output-only diagnostics;
S1 sparse once-per-cycle C-head F-only writeback; S2/S3 per-propagation-frame
C/D F-only writeback at gain 0.5/1.0.

Output-only arms never commit adapted features or calibrated depth/radius to the
recursive state. The native radius and native feature state are always carried
forward. Adapter supervision uses the same Crowd2 cycle order and GT-bearing
propagation frames as the audited C3 training protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

import gt4a2_q0_2_full_cycle_fixed_gain as q02  # noqa: E402
import gt4a2_c3_closed_loop as c3  # noqa: E402
import gt4a2_exp1_train_eval as exp1  # noqa: E402
import gt2_fixed_clip_repair_train as gt2  # noqa: E402
from velodepth.models.persistent_feature_repair import PersistentFeatureRepairHead  # noqa: E402

REFRESH = 32
TRAIN_END_EXCLUSIVE = 768
FORMAL_STEPS = 1000
SEED = 20261024
LR = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
TBPTT = 4
METRICS = ("AbsRel", "RMSE", "delta1")
SEQUENCES = ("crowd2", "crowd3", "balloon2")
OUTPUT_ONLY_ARMS = {
    "B2_channel_affine_output_only",
    "B3_output_adapter_trained_output_only",
    "B3_frozen_C_output_only",
    "B3_frozen_D_output_only",
}
STATE_ARMS = {
    "S1_C_once_per_cycle_F_only",
    "S2_C_gain0.5_R1_F_only",
    "S2_D_gain0.5_R1_F_only",
    "S3_C_gain1.0_R1_F_only",
    "S3_D_gain1.0_R1_F_only",
}
ARM_GROUP = {
    "A_native": "A",
    "B1_global_scale_bias": "B1",
    "B2_channel_affine_output_only": "B2",
    "B3_output_adapter_trained_output_only": "B3",
    "B3_frozen_C_output_only": "B3_frozen_diagnostic",
    "B3_frozen_D_output_only": "B3_frozen_diagnostic",
    "S1_C_once_per_cycle_F_only": "S1",
    "S2_C_gain0.5_R1_F_only": "S2",
    "S2_D_gain0.5_R1_F_only": "S2",
    "S3_C_gain1.0_R1_F_only": "S3",
    "S3_D_gain1.0_R1_F_only": "S3",
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
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="Q0-3 merged necessity controls; frozen VeloDepth backbone.")
    ap.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    ap.add_argument("--crowd2-pair-csv", default=q02.SEQUENCE_PATHS["crowd2"])
    ap.add_argument("--crowd3-pair-csv", default=q02.SEQUENCE_PATHS["crowd3"])
    ap.add_argument("--balloon2-pair-csv", default=q02.SEQUENCE_PATHS["balloon2"])
    ap.add_argument("--previous-config", default="outputs/gt4a2_exp1/gt4a2_exp1_config.json")
    ap.add_argument("--c3-frame-csv", default="outputs/gt4a2_c3/gt4a2_c3_full_sequence_frame_metrics.csv")
    ap.add_argument("--c3-output-dir", default="outputs/gt4a2_c3")
    ap.add_argument("--c-checkpoint", default=q02.CHECKPOINT_PATHS["C"])
    ap.add_argument("--d-checkpoint", default=q02.CHECKPOINT_PATHS["D"])
    ap.add_argument("--out-dir", default="outputs/gt4a2_q0_3_validation")
    ap.add_argument("--sequences", default="crowd2,crowd3,balloon2")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Smoke only: 0 means full sequences; otherwise >=33.")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--smoke-only", action="store_true",
                    help="One training update per adapter and 64-frame Crowd2 mechanics test; no formal checkpoints.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume adapter training from recovery checkpoints; evaluation is rerun.")
    ap.add_argument("--metric-tolerance", type=float, default=1e-5)
    ap.add_argument("--keyframe-pixel-tolerance-m", type=float, default=1e-3)
    return ap.parse_args()


def infer_repair_head_spec(state: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Infer (channels, bottleneck, levels) from the audited residual-head state dict."""
    required = ("blocks.0.0.weight", "blocks.0.6.weight")
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f"Cannot infer repair-head dimensions; missing state keys: {missing}")
    first = state["blocks.0.0.weight"]
    last = state["blocks.0.6.weight"]
    if first.ndim != 4 or last.ndim != 4:
        raise RuntimeError("Unexpected repair-head convolution weight ranks")
    bottleneck = int(first.shape[0])
    channels = int(first.shape[1])
    if int(last.shape[0]) != channels or int(last.shape[1]) != bottleneck:
        raise RuntimeError(
            "Repair-head state dict has inconsistent channel/bottleneck dimensions: "
            f"first={tuple(first.shape)}, last={tuple(last.shape)}"
        )
    level_ids = set()
    for key in state:
        match = re.match(r"^blocks\.(\d+)\.0\.weight$", key)
        if match:
            level_ids.add(int(match.group(1)))
    levels = len(level_ids)
    if levels < 1 or level_ids != set(range(levels)):
        raise RuntimeError(f"Repair-head block indices are not contiguous: {sorted(level_ids)}")
    return channels, bottleneck, levels


class ChannelAffine4(torch.nn.Module):
    """Identity-initialized per-channel affine over the actual persistent-state channels."""
    def __init__(self, channels: int, levels: int = 4):
        super().__init__()
        self.channels = int(channels)
        self.levels = int(levels)
        self.scale = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(1, channels, 1, 1)) for _ in range(levels)
        ])
        self.bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(1, channels, 1, 1)) for _ in range(levels)
        ])

    def forward(self, features):
        if len(features) != self.levels:
            raise ValueError(f"ChannelAffine4 expected {self.levels} levels, got {len(features)}")
        out = []
        for i, x in enumerate(features):
            if x.shape[1] != self.channels:
                raise ValueError(f"ChannelAffine4 level {i}: expected {self.channels}, got {x.shape[1]}")
            out.append(x * self.scale[i] + self.bias[i])
        return out


def parse_sequences(raw: str) -> list[str]:
    seqs = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not seqs or len(seqs) != len(set(seqs)):
        raise ValueError("--sequences must contain unique sequence names")
    unknown = sorted(set(seqs) - set(SEQUENCES))
    if unknown:
        raise ValueError(f"Unknown sequences: {unknown}")
    return seqs


def read_sample_order(c3_dir: Path, checkpoints: dict[str, Path]) -> tuple[list[int], str, dict]:
    order_path = c3_dir / "gt4a2_c3_sample_order.csv"
    if not order_path.is_file():
        raise FileNotFoundError(f"C3 sample order missing: {order_path}")
    with order_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != FORMAL_STEPS:
        raise RuntimeError(f"Expected {FORMAL_STEPS} C3 training samples, got {len(rows)}")
    rows = sorted(rows, key=lambda r: int(r["step"]))
    if [int(r["step"]) for r in rows] != list(range(1, FORMAL_STEPS + 1)):
        raise RuntimeError("C3 sample order steps are not contiguous 1..1000")
    order = [int(r["cycle"]) for r in rows]
    if any(x < 0 or x >= 24 for x in order):
        raise RuntimeError("C3 sample order contains a cycle outside 0..23")
    digest = hashlib.sha256(",".join(map(str, order)).encode()).hexdigest()

    ckpt_meta = {}
    for label, path in checkpoints.items():
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if obj.get("sample_order_sha256") != digest:
            raise RuntimeError(f"{label} checkpoint sample order does not match C3 sample order")
        if int(obj.get("step_count", -1)) != FORMAL_STEPS:
            raise RuntimeError(f"{label} checkpoint is not final step 1000")
        ckpt_meta[label] = {
            "sample_order_sha256": obj.get("sample_order_sha256"),
            "initial_head_state_sha256": obj.get("initial_head_state_sha256"),
            "training_rgb_indices": obj.get("training_original_rgb_indices", []),
            "training_cycle_count": obj.get("training_cycle_count"),
            "learning_rate": obj.get("learning_rate"),
            "checkpoint_sha256": sha256_file(path),
        }
        del obj
    if ckpt_meta["C"]["initial_head_state_sha256"] != ckpt_meta["D"]["initial_head_state_sha256"]:
        raise RuntimeError("C/D repair heads did not share the same initialization")
    for label, meta in ckpt_meta.items():
        if meta["training_rgb_indices"] != [0, TRAIN_END_EXCLUSIVE - 1] or int(meta["training_cycle_count"] or -1) != 24:
            raise RuntimeError(f"{label} checkpoint training split is not the audited Crowd2 0..767 / 24-cycle split")
    return order, digest, ckpt_meta


def build_preflight(args, sequences: list[str]) -> tuple[dict, dict, dict, list[int], str, dict]:
    if not args.snapshot:
        raise RuntimeError("VDEEP_SNAPSHOT is unset; pass --snapshot or export it.")
    if args.max_frames < 0 or (args.max_frames > 0 and args.max_frames < 33):
        raise ValueError("--max-frames must be 0 (full) or >=33")
    if not args.smoke_only and sequences != list(SEQUENCES):
        raise ValueError("Formal Q0-3 must run all three development sequences: crowd2,crowd3,balloon2")
    if args.smoke_only:
        if args.resume:
            raise ValueError("--resume is not used with --smoke-only")
        if args.max_frames == 0:
            args.max_frames = 64
        if args.max_frames > 128:
            raise ValueError("Smoke is capped at 128 frames")
        if sequences != ["crowd2"]:
            raise ValueError("Smoke-only must use --sequences crowd2")
    elif args.max_frames != 0:
        raise ValueError("--max-frames is only allowed with --smoke-only")
    snapshot = Path(args.snapshot).resolve()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pretrained snapshot incomplete: {snapshot}")

    # Reuse the audited Q0-2 source/checkpoint/manifest checks without changing them.
    q02_args = SimpleNamespace(
        snapshot=str(snapshot), max_frames=args.max_frames,
        crowd2_pair_csv=args.crowd2_pair_csv,
        crowd3_pair_csv=args.crowd3_pair_csv,
        balloon2_pair_csv=args.balloon2_pair_csv,
        c3_frame_csv=args.c3_frame_csv,
        checkpoint="both", c_checkpoint=args.c_checkpoint, d_checkpoint=args.d_checkpoint,
    )
    q02_pre, manifests, historical_refs = q02.build_preflight(q02_args, sequences)

    # Reuse C3's exact split/source/manifest checks.
    c3_args = SimpleNamespace(
        snapshot=str(snapshot),
        crowd2_pair_csv=args.crowd2_pair_csv,
        crowd3_pair_csv=args.crowd3_pair_csv,
        balloon2_pair_csv=args.balloon2_pair_csv,
        previous_config=args.previous_config,
        steps=FORMAL_STEPS,
        smoke_only=False,
    )
    c3_paths, train_rows, train_cycles, c3_pre = c3.preflight(c3_args)

    c3_dir = Path(args.c3_output_dir).resolve()
    for label in ("C", "D"):
        for suffix in ("training_summary.json", "training_curve.csv"):
            required = c3_dir / f"gt4a2_c3_{label}_{suffix}"
            if not required.is_file():
                raise FileNotFoundError(
                    f"Historical C3 training compute record required for fair cost comparison: {required}"
                )
    init_path = c3_dir / "gt4a2_c3_initial_head.pt"
    if not init_path.is_file():
        raise FileNotFoundError(f"C3 shared initial head state missing: {init_path}")
    initial_head_state = torch.load(init_path, map_location="cpu", weights_only=False)
    init_hash = exp1.state_sha256(initial_head_state)
    checkpoints = {"C": Path(args.c_checkpoint).resolve(), "D": Path(args.d_checkpoint).resolve()}
    sample_order, order_sha, ckpt_meta = read_sample_order(c3_dir, checkpoints)
    if init_hash != ckpt_meta["C"]["initial_head_state_sha256"]:
        raise RuntimeError("Saved C3 initial head does not match the initial hash recorded in C/D checkpoints")
    if len(train_rows) < TRAIN_END_EXCLUSIVE:
        raise RuntimeError("Crowd2 manifest does not cover training indices 0..767")
    if len(train_cycles) != 24:
        raise RuntimeError(f"Expected 24 C3 training cycles, got {len(train_cycles)}")

    # Explicit split manifest: only Crowd2 indices 0..767 may be used for fitting.
    split_rows = []
    for seq, rows in manifests.items():
        for idx, row in enumerate(rows):
            if seq == "crowd2" and idx < TRAIN_END_EXCLUSIVE:
                split = "train_fit"
            elif seq == "crowd2":
                split = "crowd2_temporal_holdout"
            else:
                split = "cross_sequence_development_only"
            split_rows.append({
                "sequence": seq, "frame_index_zero_based": idx,
                "rgb_file": Path(row["rgb_path"]).name,
                "gt_file": Path(row["depth_path"]).name if str(row.get("gt_match_status", "")).upper() == "MATCHED" else "",
                "gt_matched": str(str(row.get("gt_match_status", "")).upper() == "MATCHED").lower(),
                "split": split,
                "used_for_calibration_or_adapter_fit": str(split == "train_fit").lower(),
            })

    pre = {
        "status": "PREFLIGHT_PASS",
        "experiment": "Q0-3 merged necessity controls",
        "root": str(ROOT), "snapshot": str(snapshot),
        "pretrained_weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "q0_2_source_and_checkpoint_preflight": q02_pre,
        "c3_training_split_preflight": c3_pre,
        "c3_initial_head": {"path": str(init_path), "sha256_state": init_hash},
        "c3_sample_order": {"path": str(c3_dir / "gt4a2_c3_sample_order.csv"),
                            "sha256": order_sha, "steps": len(sample_order),
                            "checkpoint_metadata": ckpt_meta},
        "training_split": {"fit_sequence": "crowd2", "fit_indices": [0, 767],
                           "heldout_temporal_indices": [768, len(manifests["crowd2"]) - 1],
                           "crowd3_balloon2_used_for_fit": False},
        "protocol": {
            "refresh_interval": REFRESH, "formal_adapter_steps": FORMAL_STEPS,
            "optimizer": "AdamW", "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP, "tbptt_frames": TBPTT,
            "checkpoint_selection": "final step only; no held-out selection",
            "output_only_state_contract": "preserve native radius, confidence, and all four native feature maps; temporary adapter output is read only for current-frame metrics",
            "persistent_state_contract": "F-only post-readout writeback; native current-frame radius/confidence are not recomputed",
            "B3_fairness": "same common initial head as C/D, same C3 sample order, same matched propagation-frame GTs, same 1000 optimizer updates and final-step-only selection; wall time and peak memory measured separately",
            "calibration": "global linear z-depth scale/bias fitted only on native-A predictions and GT pixels from Crowd2 indices 0..767",
            "development_limit": "Crowd2/Crowd3/Balloon2 are development data; only Crowd2 768+ is temporal holdout, not a new independent domain",
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    return pre, manifests, historical_refs, sample_order, order_sha, {
        "paths": c3_paths, "train_rows": train_rows, "train_cycles": train_cycles,
        "initial_head_state": initial_head_state, "initial_head_hash": init_hash,
        "checkpoints": checkpoints, "checkpoint_meta": ckpt_meta, "split_rows": split_rows,
    }


def readout_from_features(radial, features: list[torch.Tensor]) -> torch.Tensor:
    """Mirror RadialModule.forward_fast readout without committing these features."""
    if len(features) != 4:
        raise RuntimeError(f"Expected four feature levels, got {len(features)}")
    lateral_features = [features[0]] + [
        radial.process_features[i](features[i + 1]) for i in range(len(radial.ups))
    ]
    latents = lateral_features[0]
    out_features = []
    for i, up in enumerate(radial.ups):
        latents = latents + lateral_features[i + 1]
        latents = up(latents)
        out_features.append(latents)
    return radial.depth_proj(out_features)


def radius_from_features(model, features: list[torch.Tensor]) -> torch.Tensor:
    radial = model.pixel_decoder.radial_module
    logdepth = readout_from_features(radial, features)
    return torch.exp(logdepth.clip(min=-8.0, max=8.0) + 2.0)


def load_base_model(snapshot: Path, device: torch.device):
    """Load the exact VeloDepth weights, detach the new repair module, freeze the base."""
    model = gt2.load_model(snapshot, "pre_readout", device)
    template = model.pixel_decoder.persistent_feature_repair
    if template is None:
        raise RuntimeError("Expected audited persistent feature repair module is not registered")
    model.pixel_decoder.persistent_feature_repair = None
    model.pixel_decoder.feature_repair_mode = "off"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    template.eval()
    return model, template


def make_frozen_head(state: dict, device: torch.device) -> torch.nn.Module:
    channels, bottleneck, levels = infer_repair_head_spec(state)
    head = PersistentFeatureRepairHead(
        channels=channels, bottleneck=bottleneck, levels=levels
    ).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def apply_repair_residual(features, head, gain: float):
    if len(features) != len(head.blocks):
        raise RuntimeError(f"Repair level mismatch: features={len(features)} blocks={len(head.blocks)}")
    return [(x + gain * block(x)).detach() for x, block in zip(features, head.blocks)]


def output_split(seq: str, idx: int) -> str:
    if seq == "crowd2" and idx < TRAIN_END_EXCLUSIVE:
        return "train_fit"
    if seq == "crowd2":
        return "crowd2_temporal_holdout"
    return "cross_sequence_development_only"


def fit_stats_add(stats: dict, pred: np.ndarray, gt: np.ndarray) -> None:
    """Merge per-frame centered moments to avoid cancellation in pixel-level OLS."""
    p = pred.astype(np.float64, copy=False)
    g = gt.astype(np.float64, copy=False)
    mask = np.isfinite(p) & np.isfinite(g) & (p > 0) & (g > 0)
    if not np.any(mask):
        return
    pp = p[mask]
    gg = g[mask]
    bn = int(pp.size)
    bmp = float(pp.mean())
    bmg = float(gg.mean())
    centered_p = pp - bmp
    centered_g = gg - bmg
    bm2p = float(np.dot(centered_p, centered_p))
    bcov = float(np.dot(centered_p, centered_g))
    old_n = int(stats["n"])
    if old_n == 0:
        stats["n"] = bn
        stats["mean_p"] = bmp
        stats["mean_g"] = bmg
        stats["m2_p"] = bm2p
        stats["cov_pg"] = bcov
        return
    new_n = old_n + bn
    dp = bmp - float(stats["mean_p"])
    dg = bmg - float(stats["mean_g"])
    cross_weight = old_n * bn / float(new_n)
    stats["m2_p"] += bm2p + dp * dp * cross_weight
    stats["cov_pg"] += bcov + dp * dg * cross_weight
    stats["mean_p"] += dp * bn / float(new_n)
    stats["mean_g"] += dg * bn / float(new_n)
    stats["n"] = new_n


def fit_scale_bias(stats: dict) -> dict:
    n = int(stats["n"])
    if n < 1000:
        raise RuntimeError(f"Too few pixels for global scale/bias fit: n={n}")
    denom = float(stats["m2_p"])
    if not np.isfinite(denom) or denom <= 1e-12:
        raise RuntimeError("Global scale/bias regression is ill-conditioned")
    scale = float(stats["cov_pg"]) / denom
    bias = float(stats["mean_g"]) - scale * float(stats["mean_p"])
    if not np.isfinite(scale) or not np.isfinite(bias) or scale <= 0:
        raise RuntimeError(f"Invalid fitted scale/bias: scale={scale}, bias={bias}")
    return {
        "fit_method": "ordinary_least_squares_gt_z_equals_scale_times_native_pred_z_plus_bias",
        "scale": float(scale), "bias_m": float(bias), "valid_pixel_pairs": n,
        "fit_sequence": "crowd2", "fit_frame_indices_inclusive": [0, TRAIN_END_EXCLUSIVE - 1],
        "fit_uses_crowd3": False, "fit_uses_balloon2": False,
        "fit_uses_crowd2_temporal_holdout": False,
        "fit_sufficient_statistics": {
            "n": n, "mean_native_pred_z": float(stats["mean_p"]),
            "mean_gt_z": float(stats["mean_g"]), "centered_sum_sq_pred": float(stats["m2_p"]),
            "centered_cross_sum_pred_gt": float(stats["cov_pg"]),
        },
    }


def make_frame_row(seq, arm, idx, rec, metrics, frame_wall, age, is_keyframe,
                   repair_applied, repair_count_cycle, repair_count_total,
                   output_adapter_call, extra_readout, run_source="Q0-3_new_inference"):
    matched = metrics is not None
    return {
        "sequence": seq, "split": output_split(seq, idx), "group": ARM_GROUP[arm],
        "arm": arm, "frame_index_zero_based": idx,
        "refresh_cycle": idx // REFRESH, "frame_offset_in_cycle": age,
        "is_keyframe": str(is_keyframe).lower(),
        "gt_matched": str(matched).lower(),
        "rgb_file": Path(rec["rgb_path"]).name,
        "gt_file": Path(rec["depth_path"]).name if matched else "",
        "AbsRel": metrics["AbsRel"] if matched else "",
        "RMSE": metrics["RMSE"] if matched else "",
        "delta1": metrics["delta1"] if matched else "",
        "valid_ratio": metrics.get("valid_ratio", "") if matched else "",
        "gt_valid_ratio": metrics.get("gt_valid_ratio", "") if matched else "",
        "valid_pixels": metrics.get("valid_pixels", "") if matched else "",
        "total_pixels": metrics.get("total_pixels", "") if matched else "",
        "repair_applied_this_frame": str(repair_applied).lower(),
        "repair_count_cycle": repair_count_cycle,
        "repair_count_cumulative": repair_count_total,
        "output_adapter_called": str(output_adapter_call).lower(),
        "additional_readout_pass": str(extra_readout).lower(),
        "frame_wall_sec": frame_wall,
        "run_source": run_source,
    }



def verify_readout_equivalence(model, rows: list[dict], shape: dict, device: torch.device,
                               tolerance_m: float = 1e-3) -> dict:
    """Verify the manual readout reproduces the native *propagation* readout.

    Keyframe decoding is a different path (RadialModule.forward_keyframe uses
    camera-conditioned encoder features and rays). It must not be compared with
    readout_from_features(), which mirrors RadialModule.forward_fast only.
    Output-only adapters are applied to propagated states, never to keyframes.
    """
    if len(rows) < 2:
        raise RuntimeError("Need at least two frames to verify output-only readout wiring")

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev = exp1.load_frame(rows[0], shape, device, need_gt=False)
    rays, radius_key_native, confidence_key, features_key = exp1.keyframe_state(
        model, prev, shape
    )

    # The keyframe output is intentionally kept native. Check that it is finite,
    # but do not compare it with the propagated-feature readout implementation.
    keyframe_finite = bool(torch.isfinite(radius_key_native).all().item())
    if not keyframe_finite:
        raise RuntimeError("Native keyframe radius contains non-finite values")

    curr = exp1.load_frame(rows[1], shape, device, need_gt=False)
    static = exp1.transition_static(model, prev, curr, shape, dis)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        radius_native, confidence_native, features_native = exp1.one_fast_step(
            model, prev, curr, static, radius_key_native, confidence_key, features_key
        )
        radius_manual = radius_from_features(model, features_native)
        diff = float((radius_manual.float() - radius_native.float()).abs().max().item())

    result = {
        "keyframe_native_radius_finite": keyframe_finite,
        "keyframe_native_radius_shape": list(radius_key_native.shape),
        "propagated_max_abs_radius_difference": diff,
        "max_abs_radius_difference": diff,
        "tolerance_m": tolerance_m,
        "passed": diff <= tolerance_m,
        "tested_frame_index": 1,
        "comparison_scope": "native fast-step propagated radius vs manual readout from the same propagated feature state",
        "keyframe_comparison": "not applicable: native keyframe decoder uses a distinct camera-conditioned readout path",
    }
    if diff > tolerance_m:
        raise RuntimeError(
            f"Propagated output-only readout does not reproduce native fast-step radius: "
            f"{diff} > {tolerance_m}"
        )

    del prev, curr, rays, radius_key_native, confidence_key, features_key, static
    del radius_native, confidence_native, features_native, radius_manual
    return result

def run_arm(seq: str, arm: str, model, rows: list[dict], shape: dict,
            device: torch.device, *, adapter=None, gain=1.0, schedule="none",
            scale_bias=None, fit_stats=None) -> tuple[list[dict], dict, dict[int, np.ndarray]]:
    """Run one arm from sequence start; output-only changes never enter native state."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    prev_frame = None
    radius = confidence = features = rays_map = None
    out_rows = []
    keyframes = {}
    repair_total = repair_cycle = 0
    frame_times = []
    extra_readout_count = 0
    adapter_calls = 0
    state_mutation_max = 0.0
    radius_mutation_max = 0.0
    confidence_mutation_max = 0.0
    fit_stats_wall_sec = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    start_run = time.perf_counter()
    radial = model.pixel_decoder.radial_module

    with torch.no_grad():
        for idx, rec in enumerate(rows):
            t0 = time.perf_counter()
            matched = str(rec.get("gt_match_status", "")).strip().upper() == "MATCHED"
            frame = exp1.load_frame(rec, shape, device, need_gt=matched)
            age = idx % REFRESH
            cycle = idx // REFRESH
            is_keyframe = age == 0
            repair_applied = False
            output_adapter_call = False
            extra_readout = False
            radius_out = None

            if is_keyframe:
                rays_map, radius, confidence, features = exp1.keyframe_state(model, frame, shape)
                repair_cycle = 0
                radius_out = radius
            else:
                static = exp1.transition_static(model, prev_frame, frame, shape, dis)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    # This is always the native A transition/readout. Its radius,
                    # confidence and features are the only state passed forward.
                    radius, confidence, features = exp1.one_fast_step(
                        model, prev_frame, frame, static, radius, confidence, features
                    )
                    radius_out = radius

                    if arm in OUTPUT_ONLY_ARMS:
                        if adapter is None:
                            raise RuntimeError(f"{arm} requires an output-only adapter")
                        if idx == 1:
                            native_radius_before = radius.detach().clone()
                            native_confidence_before = confidence.detach().clone()
                            native_features_before = [x.detach().clone() for x in features]
                        adapted_features = adapter(features)
                        radius_out = radius_from_features(model, adapted_features)
                        adapter_calls += 1
                        extra_readout_count += 1
                        output_adapter_call = True
                        extra_readout = True
                        if idx == 1:
                            state_mutation = max(
                                [float((x.float() - x0.float()).abs().max().item())
                                 for x, x0 in zip(features, native_features_before)] + [0.0]
                            )
                            radius_mutation = float((radius.float() - native_radius_before.float()).abs().max().item())
                            confidence_mutation = float((confidence.float() - native_confidence_before.float()).abs().max().item())
                            state_mutation_max = max(state_mutation_max, state_mutation)
                            radius_mutation_max = max(radius_mutation_max, radius_mutation)
                            confidence_mutation_max = max(confidence_mutation_max, confidence_mutation)
                            if state_mutation != 0.0 or radius_mutation != 0.0 or confidence_mutation != 0.0:
                                raise RuntimeError(
                                    f"{arm} mutated native state: feature={state_mutation}, "
                                    f"radius={radius_mutation}, confidence={confidence_mutation}"
                                )
                            del native_radius_before, native_confidence_before, native_features_before
                        # Intentionally do not assign adapted_features to `features`,
                        # and never assign radius_out to the native `radius`.
                        del adapted_features

                    if arm in STATE_ARMS and schedule != "none":
                        should_write = (schedule == "once_age1" and age == 1) or (schedule == "every_prop")
                        if should_write:
                            if adapter is None:
                                raise RuntimeError(f"{arm} requires a repair head")
                            before = features
                            alpha = float(gain)
                            features = apply_repair_residual(features, adapter, alpha)
                            repair_applied = True
                            repair_total += 1
                            repair_cycle += 1
                            # F-only post-readout semantics: current radius/output stays native.
                            if radius_out is not radius:
                                raise RuntimeError("F-only writeback must not alter current output radius")
                            del before
                del static

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                pred_t = gt2.make_z_depth(
                    radius_out, rays_map, shape["padded_shape"], shape["paddings"],
                    model.interpolation_mode
                )
            pred_np = pred_t.detach().float().cpu().numpy()

            if arm == "B1_global_scale_bias":
                if scale_bias is None:
                    raise RuntimeError("B1 requires fitted scale/bias")
                pred_np = np.maximum(
                    float(scale_bias["scale"]) * pred_np + float(scale_bias["bias_m"]),
                    1e-4,
                )

            if fit_stats is not None and seq == "crowd2" and idx < TRAIN_END_EXCLUSIVE and matched:
                # Fit only from native-A predictions; no calibrated outputs are used.
                fit_t0 = time.perf_counter()
                fit_stats_add(fit_stats, pred_np, frame["gt_np"])
                fit_stats_wall_sec += time.perf_counter() - fit_t0

            if is_keyframe:
                keyframes[idx] = pred_np.copy()

            if matched:
                metrics = gt2.calculate_metrics(frame["gt_np"], pred_np)
            else:
                metrics = None
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elapsed_frame = time.perf_counter() - t0
            frame_times.append(elapsed_frame)
            out_rows.append(make_frame_row(
                seq, arm, idx, rec, metrics, elapsed_frame, age, is_keyframe,
                repair_applied, repair_cycle, repair_total,
                output_adapter_call, extra_readout,
            ))
            prev_frame = frame
            if idx == 0 or (idx + 1) % 100 == 0 or idx == len(rows) - 1:
                print(f"[{seq}/{arm}] {idx+1}/{len(rows)} frames", flush=True)

    elapsed = time.perf_counter() - start_run
    scored = [r for r in out_rows if r["gt_matched"] == "true"]
    run_meta = {
        "sequence": seq, "group": ARM_GROUP[arm], "arm": arm,
        "frames_processed": len(rows), "scored_frames": len(scored),
        "keyframe_calls": sum(1 for i in range(len(rows)) if i % REFRESH == 0),
        "native_transition_calls": max(0, len(rows) - sum(1 for i in range(len(rows)) if i % REFRESH == 0)),
        "repair_head_calls": repair_total,
        "output_adapter_calls": adapter_calls,
        "additional_readout_passes": extra_readout_count,
        "total_readout_passes_estimated": len(rows) + extra_readout_count,
        "output_only_state_feature_mutation_max_abs": state_mutation_max,
        "output_only_state_radius_mutation_max_abs": radius_mutation_max,
        "output_only_state_confidence_mutation_max_abs": confidence_mutation_max,
        "run_wall_sec": elapsed,
        "fit_stats_accumulation_sec": fit_stats_wall_sec,
        "run_wall_sec_excluding_fit_stats_accumulation": max(0.0, elapsed - fit_stats_wall_sec),
        "ms_per_rgb_frame": elapsed / max(len(rows), 1) * 1000,
        "ms_per_rgb_frame_excluding_fit_stats_accumulation": max(0.0, elapsed - fit_stats_wall_sec) / max(len(rows), 1) * 1000,
        "frame_wall_p50_ms": float(np.percentile(frame_times, 50) * 1000) if frame_times else "",
        "frame_wall_p95_ms": float(np.percentile(frame_times, 95) * 1000) if frame_times else "",
        "repair_schedule": schedule, "gain": float(gain) if schedule != "none" else "",
        "state_semantics": (
            "native_A_state_preserved_output_only" if arm in OUTPUT_ONLY_ARMS or arm == "B1_global_scale_bias"
            else "F_only_post_readout_writeback" if arm in STATE_ARMS else "native_A"
        ),
        "timing_scope": "wall time includes RGB/GT I/O, preprocessing, DIS flow, model transitions, readout, adaptation, and metrics; model load excluded",
    }
    return out_rows, run_meta, keyframes


def train_one_output_only_module(
    label: str, model, module, reader, sample_order: list[int],
    training_dir: Path, *, steps: int, initial_state: dict[str, torch.Tensor],
    lr: float = LR, smoke: bool = False, resume: bool = False,
) -> dict:
    """Train on C3's exact cycle order/GT-bearing propagation frames, without state writes."""
    training_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = training_dir / f"{label}_final_step{steps}.pt"
    curve_path = training_dir / f"{label}_training_curve.csv"
    module.load_state_dict(initial_state, strict=True)
    module.train()
    trainable = [p for p in module.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError(f"{label}: no trainable parameters")
    expected_trainable = sum(p.numel() for p in module.parameters())
    if sum(p.numel() for p in trainable) != expected_trainable:
        raise RuntimeError(f"{label}: unexpected frozen/trainable parameter mix")

    if ckpt_path.exists():
        if not resume or smoke:
            raise RuntimeError(f"Checkpoint already exists; use --resume or a new output directory: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        expected_init_hash = exp1.state_sha256(initial_state)
        expected_order_hash = hashlib.sha256(",".join(map(str, sample_order)).encode()).hexdigest()
        expected_weights_hash = sha256_file(Path(os.environ["VDEEP_SNAPSHOT"]) / "model.safetensors")
        if (ckpt.get("mode") != label or int(ckpt.get("step_count", -1)) != steps
                or ckpt.get("initial_adapter_state_sha256") != expected_init_hash
                or ckpt.get("sample_order_sha256") != expected_order_hash
                or ckpt.get("pretrained_weights_sha256") != expected_weights_hash):
            raise RuntimeError(f"{label}: existing final checkpoint provenance mismatch: {ckpt_path}")
        module.load_state_dict(ckpt["adapter_state_dict"], strict=True)
        summary_path = training_dir / f"{label}_training_summary.json"
        if summary_path.is_file():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"Final checkpoint exists but summary is missing: {summary_path}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    init_hash = exp1.state_sha256(initial_state)
    order_sha = hashlib.sha256(",".join(map(str, sample_order)).encode()).hexdigest()
    requested = steps
    start_step = 0
    if resume and not smoke:
        start_step = c3.resume_states(
            training_dir, label, optimizer, scaler, module, order_sha, init_hash, requested
        )
    if start_step > requested:
        raise RuntimeError(f"{label}: recovery step exceeds requested steps")
    if start_step and not curve_path.is_file():
        raise RuntimeError(f"{label}: recovery exists but training curve is missing")
    if resume and not start_step and curve_path.is_file():
        raise RuntimeError(f"{label}: curve exists but no recovery checkpoint; refusing silent restart")
    if not resume and curve_path.exists() and not smoke:
        raise RuntimeError(f"{label}: training curve already exists; use --resume or a new output directory")

    curve_rows = []
    if start_step:
        with curve_path.open("r", newline="", encoding="utf-8-sig") as f:
            curve_rows = list(csv.DictReader(f))

    device = next(model.parameters()).device
    radial = model.pixel_decoder.radial_module
    torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()
    total_gt_frames = 0
    total_rgb_frames = 0
    total_valid_pixels = 0
    max_peak_gib = 0.0

    for step in range(start_step + 1, requested + 1):
        cycle_id = int(sample_order[step - 1])
        cycle = reader.load(cycle_id)
        n_supervised = int(cycle["n_supervised"])
        if n_supervised < 1:
            raise RuntimeError(f"{label}: cycle {cycle_id} has no supervised propagation frames")
        radius, confidence, features = c3.move_state(cycle["keyframe_state"], device)
        rays = cycle["keyframe_state"]["rays_map"].to(device)
        local_shape = {"validity_mask": torch.ones(
            (1, 1, *cycle["network_shape"]), device=device, dtype=torch.bool
        )}
        prev = c3.to_gpu_frame(cycle["frames"][0], device)
        optimizer.zero_grad(set_to_none=True)
        gt_frames_this_step = 0
        valid_pixels_this_step = 0
        frame_loss_values = []
        segment_lengths = []
        step_start = time.perf_counter()

        for segment_begin in range(1, REFRESH, TBPTT):
            segment_end = min(REFRESH - 1, segment_begin + TBPTT - 1)
            loss_terms = []
            held = [prev]
            for offset in range(segment_begin, segment_end + 1):
                curr = c3.to_gpu_frame(cycle["frames"][offset], device)
                static = c3.to_gpu_static(cycle["statics"][offset - 1], device, local_shape)
                # The complete recursive state follows native A; no adapter result is
                # assigned to radius/confidence/features used by the next frame.
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    radius, confidence, features = exp1.one_fast_step(
                        model, prev, curr, static, radius, confidence, features
                    )
                if curr["gt"] is not None:
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        adapted = module(features)
                        logdepth = readout_from_features(radial, adapted)
                        radius_output = torch.exp(logdepth.clip(min=-8.0, max=8.0) + 2.0)
                        pred = gt2.make_z_depth(
                            radius_output, rays, cycle["padded_shape"], cycle["paddings"],
                            model.interpolation_mode
                        )
                        loss, valid_count = gt2.masked_log_depth_loss(pred, curr["gt"])
                    if not torch.isfinite(loss.detach()).all():
                        raise RuntimeError(f"{label}: non-finite loss at step={step}, cycle={cycle_id}, offset={offset}")
                    loss_terms.append(loss)
                    frame_loss_values.append(float(loss.detach().item()))
                    gt_frames_this_step += 1
                    valid_pixels_this_step += valid_count
                    del adapted, logdepth, radius_output, pred
                else:
                    # Still execute the output-only path on unsupervised propagation
                    # frames; it cannot affect state and receives no loss/gradient.
                    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                        adapted = module(features)
                        logdepth = readout_from_features(radial, adapted)
                        radius_output = torch.exp(logdepth.clip(min=-8.0, max=8.0) + 2.0)
                        pred = gt2.make_z_depth(
                            radius_output, rays, cycle["padded_shape"], cycle["paddings"],
                            model.interpolation_mode
                        )
                    del adapted, logdepth, radius_output, pred
                held.append((curr, static))
                prev = curr
            if loss_terms:
                segment_loss = torch.stack(loss_terms).sum() / float(n_supervised)
                scaler.scale(segment_loss).backward()
                del segment_loss
            radius, confidence, features = c3.detach_state(radius, confidence, features)
            prev = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in prev.items()}
            segment_lengths.append(segment_end - segment_begin + 1)
            del loss_terms, held

        if gt_frames_this_step != n_supervised:
            raise RuntimeError(f"{label}: GT coverage mismatch {gt_frames_this_step}!={n_supervised}")
        scaler.unscale_(optimizer)
        grads = [p.grad for p in trainable if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all().item() for g in grads):
            raise RuntimeError(f"{label}: missing or non-finite gradients")
        nonzero = sum(bool(torch.count_nonzero(g).item()) for g in grads)
        if nonzero == 0:
            raise RuntimeError(f"{label}: all gradients are zero")
        grad_l1 = sum(float(g.detach().abs().sum().item()) for g in grads)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP).item())
        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        if float(scaler.get_scale()) < old_scale:
            raise RuntimeError(f"{label}: GradScaler skipped optimizer update")
        torch.cuda.synchronize()
        step_wall = time.perf_counter() - step_start
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        max_peak_gib = max(max_peak_gib, peak_gib)
        total_gt_frames += gt_frames_this_step
        total_rgb_frames += REFRESH
        total_valid_pixels += valid_pixels_this_step
        row = {
            "mode": label, "step": step, "cycle": cycle_id,
            "optimizer_updates_this_cycle": 1, "rgb_frames_exposed": REFRESH,
            "output_adapter_forward_calls": REFRESH - 1,
            "additional_readout_passes": REFRESH - 1,
            "mean_log_depth_loss_before_update": float(np.mean(frame_loss_values)),
            "supervised_propagation_frames": gt_frames_this_step,
            "valid_supervision_pixels": valid_pixels_this_step,
            "gradient_segments": sum(1 for x in segment_lengths if x > 0),
            "gradient_nonzero_tensors": nonzero,
            "gradient_l1_unclipped": grad_l1,
            "gradient_norm_before_clip": grad_norm,
            "param_delta_l2_from_init": float("nan"),
            "step_wall_sec": step_wall, "cuda_peak_allocated_gib": peak_gib,
            "grad_scaler_scale": float(scaler.get_scale()),
            "segment_lengths": ",".join(map(str, segment_lengths)),
        }
        curve_rows.append(row)
        if smoke or step == 1 or step % 10 == 0 or step == requested:
            print(f"[TRAIN {label}] step={step}/{requested} cycle={cycle_id} "
                  f"GTframes={gt_frames_this_step} loss-gradient-L1={grad_l1:.4g} "
                  f"sec={step_wall:.2f} peak={peak_gib:.2f}GiB", flush=True)
        if not smoke and (step % 25 == 0 or step == requested):
            write_csv(curve_path, curve_rows)
        if not smoke and (step % 25 == 0 or step == requested):
            c3.save_recovery(training_dir, label, step, module, optimizer, scaler,
                             order_sha, init_hash, requested)
        del cycle, prev, radius, confidence, features, rays

    elapsed_current = time.perf_counter() - total_start
    if smoke:
        return {
            "mode": label, "smoke": True, "steps": requested,
            "trainable_parameter_count": expected_trainable,
            "optimizer_updates": requested, "training_wall_sec_current_session": elapsed_current,
            "mean_step_wall_sec": float(np.mean([r["step_wall_sec"] for r in curve_rows])),
            "first_step_mean_log_depth_loss": float(curve_rows[0]["mean_log_depth_loss_before_update"]),
            "last_step_mean_log_depth_loss": float(curve_rows[-1]["mean_log_depth_loss_before_update"]),
            "peak_cuda_allocated_gib": max_peak_gib,
            "supervised_propagation_frame_exposures": total_gt_frames,
            "rgb_frame_exposures": total_rgb_frames,
            "output_adapter_forward_calls": int(sum(int(r["output_adapter_forward_calls"]) for r in curve_rows)),
            "additional_readout_passes": int(sum(int(r["additional_readout_passes"]) for r in curve_rows)),
            "valid_supervision_pixel_exposures": total_valid_pixels,
            "checkpoint_saved": False, "checkpoint_selection": "smoke only; not a formal checkpoint",
        }

    state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
    torch.save({
        "experiment": "Q0-3 output-only necessity control",
        "mode": label, "step_count": requested, "learning_rate": lr,
        "weight_decay": WEIGHT_DECAY, "grad_clip_norm": GRAD_CLIP,
        "trainable_parameter_count": expected_trainable,
        "initial_adapter_state_sha256": init_hash,
        "sample_order_sha256": order_sha,
        "training_rgb_indices_inclusive": [0, TRAIN_END_EXCLUSIVE - 1],
        "training_cycle_count": 24,
        "training_gt_policy": "same C3 GT-bearing propagation frames in sampled 32-frame cycles; keyframe GT is not supervised",
        "state_contract": "native radius/confidence/features carried forward; adapted features are used only for current-frame readout",
        "adapter_state_dict": state,
        "checkpoint_selection": "final step only; no validation-based selection",
        "pretrained_weights_sha256": sha256_file(Path(os.environ["VDEEP_SNAPSHOT"]) / "model.safetensors"),
    }, ckpt_path)
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    module.load_state_dict(loaded["adapter_state_dict"], strict=True)
    if exp1.state_sha256(loaded["adapter_state_dict"]) != exp1.state_sha256(state):
        raise RuntimeError(f"{label}: checkpoint reload mismatch")
    first_loss_window = [float(r["mean_log_depth_loss_before_update"]) for r in curve_rows[:min(24, len(curve_rows))]]
    last_loss_window = [float(r["mean_log_depth_loss_before_update"]) for r in curve_rows[-min(24, len(curve_rows)):]]
    summary = {
        "mode": label, "steps": requested, "closed_loop_training": False,
        "trainable_parameter_count": expected_trainable,
        "optimizer_updates": requested,
        "training_wall_sec_current_session": elapsed_current,
        "training_wall_sec_sum_logged_steps": float(sum(float(r["step_wall_sec"]) for r in curve_rows)),
        "mean_step_wall_sec": float(np.mean([float(r["step_wall_sec"]) for r in curve_rows])),
        "first_24_step_mean_log_depth_loss": float(np.mean(first_loss_window)),
        "last_24_step_mean_log_depth_loss": float(np.mean(last_loss_window)),
        "peak_cuda_allocated_gib": max(max_peak_gib, max(float(r["cuda_peak_allocated_gib"]) for r in curve_rows)),
        "supervised_propagation_frame_exposures": int(sum(int(r["supervised_propagation_frames"]) for r in curve_rows)),
        "rgb_frame_exposures": int(sum(int(r["rgb_frames_exposed"]) for r in curve_rows)),
        "output_adapter_forward_calls": int(sum(int(r["output_adapter_forward_calls"]) for r in curve_rows)),
        "additional_readout_passes": int(sum(int(r["additional_readout_passes"]) for r in curve_rows)),
        "valid_supervision_pixel_exposures": int(sum(int(r["valid_supervision_pixels"]) for r in curve_rows)),
        "initial_adapter_state_sha256": init_hash,
        "sample_order_sha256": order_sha,
        "checkpoint": str(ckpt_path), "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_reload": "PASS", "checkpoint_selection": "final step only",
        "same_c3_cycle_order": True, "same_gt_bearing_propagation_frames": True,
        "generalization_claim": False,
    }
    write_json(training_dir / f"{label}_training_summary.json", summary)
    return summary


def load_trained_module(label: str, template, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained Q0-3 adapter missing: {checkpoint_path}")
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if obj.get("mode") != label or int(obj.get("step_count", -1)) != FORMAL_STEPS:
        raise RuntimeError(f"Trained adapter metadata mismatch: {checkpoint_path}")
    expected_weights_hash = sha256_file(Path(os.environ["VDEEP_SNAPSHOT"]) / "model.safetensors")
    if obj.get("pretrained_weights_sha256") != expected_weights_hash:
        raise RuntimeError(f"Trained adapter was produced from a different pretrained snapshot: {checkpoint_path}")
    template.load_state_dict(obj["adapter_state_dict"], strict=True)
    template.to(device).eval()
    for p in template.parameters():
        p.requires_grad_(False)
    meta = {
        "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
        "mode": label, "step_count": int(obj["step_count"]),
        "trainable_parameter_count": int(obj["trainable_parameter_count"]),
        "sample_order_sha256": obj.get("sample_order_sha256"),
        "initial_adapter_state_sha256": obj.get("initial_adapter_state_sha256"),
        "pretrained_weights_sha256": obj.get("pretrained_weights_sha256"),
        "checkpoint_selection": obj.get("checkpoint_selection"),
    }
    del obj
    return template, meta


def make_arm_adapters(device, initial_head_state, checkpoint_paths):
    # Derive dimensions from the exact shared C3 initialization; never hardcode width.
    channels, bottleneck, levels = infer_repair_head_spec(initial_head_state)
    # B2: identity-initialized channel affine, trained output-only.
    affine = ChannelAffine4(channels=channels, levels=levels).to(device)
    affine_init = {k: v.detach().cpu().clone() for k, v in affine.state_dict().items()}
    # B3: same head architecture/common initialization as C/D, trained output-only.
    output_head = PersistentFeatureRepairHead(
        channels=channels, bottleneck=bottleneck, levels=levels
    ).to(device)
    output_head.load_state_dict(initial_head_state, strict=True)
    output_init = {k: v.detach().cpu().clone() for k, v in output_head.state_dict().items()}
    frozen_heads = {}
    for label in ("C", "D"):
        obj = torch.load(checkpoint_paths[label], map_location="cpu", weights_only=False)
        frozen_heads[label] = make_frozen_head(obj["repair_head_state_dict"], device)
        del obj
    return affine, affine_init, output_head, output_init, frozen_heads


def summarize(all_rows: list[dict], run_meta: list[dict]) -> tuple[list[dict], list[dict]]:
    baseline = {
        (r["sequence"], int(r["frame_index_zero_based"])): r
        for r in all_rows if r["arm"] == "A_native"
    }
    seq_rows = []
    cycle_rows = []
    grouped = defaultdict(list)
    for r in all_rows:
        if r.get("run_source") != "Q0-3_new_inference":
            continue
        grouped[(r["sequence"], r["arm"], r["split"])].append(r)

    for (seq, arm, split), rows in sorted(grouped.items()):
        scored = [r for r in rows if r["gt_matched"] == "true" and r["AbsRel"] != ""]
        prop = [r for r in scored if r["is_keyframe"] == "false"]
        diffs = []
        worse = []
        for r in scored:
            a = baseline.get((seq, int(r["frame_index_zero_based"])))
            if a and a["AbsRel"] != "":
                d = float(r["AbsRel"]) - float(a["AbsRel"])
                diffs.append(d)
                worse.append(d > 0)
        seq_rows.append({
            "sequence": seq, "split": split, "group": ARM_GROUP[arm], "arm": arm,
            "rgb_frames": len(rows), "scored_frames": len(scored),
            "scored_propagated_frames": len(prop),
            "AbsRel_macro": float(np.mean([float(r["AbsRel"]) for r in scored])) if scored else "",
            "RMSE_macro": float(np.mean([float(r["RMSE"]) for r in scored])) if scored else "",
            "delta1_macro": float(np.mean([float(r["delta1"]) for r in scored])) if scored else "",
            "AbsRel_propagated_macro": float(np.mean([float(r["AbsRel"]) for r in prop])) if prop else "",
            "RMSE_propagated_macro": float(np.mean([float(r["RMSE"]) for r in prop])) if prop else "",
            "delta1_propagated_macro": float(np.mean([float(r["delta1"]) for r in prop])) if prop else "",
            "AbsRel_p95_frame": float(np.percentile([float(r["AbsRel"]) for r in scored], 95)) if scored else "",
            "AbsRel_max_frame": float(max(float(r["AbsRel"]) for r in scored)) if scored else "",
            "mean_frame_delta_AbsRel_vs_A": float(np.mean(diffs)) if diffs else "",
            "fraction_frames_worse_than_A": float(np.mean(worse)) if worse else "",
            "n_frames_worse_than_A": int(sum(worse)),
            "n_frames_compared_to_A": len(worse),
        })

    by_cycle = defaultdict(list)
    for r in all_rows:
        if r.get("run_source") != "Q0-3_new_inference" or r["gt_matched"] != "true" or r["AbsRel"] == "":
            continue
        by_cycle[(r["sequence"], r["arm"], int(r["refresh_cycle"]))].append(r)
    a_cycles = defaultdict(list)
    for (seq, arm, cyc), rows in by_cycle.items():
        if arm == "A_native":
            a_cycles[(seq, cyc)] = rows
    for (seq, arm, cyc), rows in sorted(by_cycle.items()):
        if len(rows) != REFRESH:
            # A cycle is complete in RGB terms only if all 32 RGB rows exist.
            n_rgb = sum(1 for r in all_rows if r.get("run_source") == "Q0-3_new_inference"
                        and r["sequence"] == seq and r["arm"] == arm
                        and int(r["refresh_cycle"]) == cyc)
            complete = n_rgb == REFRESH
        else:
            n_rgb = REFRESH
            complete = True
        a_rows = a_cycles.get((seq, cyc), [])
        if not a_rows:
            continue
        cur_abs = float(np.mean([float(r["AbsRel"]) for r in rows]))
        a_abs = float(np.mean([float(r["AbsRel"]) for r in a_rows]))
        cur_rmse = float(np.mean([float(r["RMSE"]) for r in rows]))
        a_rmse = float(np.mean([float(r["RMSE"]) for r in a_rows]))
        cur_delta = float(np.mean([float(r["delta1"]) for r in rows]))
        a_delta = float(np.mean([float(r["delta1"]) for r in a_rows]))
        frame_diffs = []
        for r in rows:
            a = baseline.get((seq, int(r["frame_index_zero_based"])))
            if a and a["AbsRel"] != "":
                frame_diffs.append(float(r["AbsRel"]) - float(a["AbsRel"]))
        cycle_rows.append({
            "sequence": seq, "split": output_split(seq, cyc * REFRESH),
            "group": ARM_GROUP[arm], "arm": arm, "refresh_cycle": cyc,
            "rgb_frames_in_cycle": n_rgb, "cycle_complete_32_frames": str(complete).lower(),
            "scored_frames": len(rows),
            "AbsRel_cycle_macro": cur_abs, "RMSE_cycle_macro": cur_rmse,
            "delta1_cycle_macro": cur_delta,
            "A_AbsRel_cycle_macro": a_abs, "A_RMSE_cycle_macro": a_rmse,
            "A_delta1_cycle_macro": a_delta,
            "delta_cycle_AbsRel_policy_minus_A": cur_abs - a_abs,
            "delta_cycle_RMSE_policy_minus_A": cur_rmse - a_rmse,
            "delta_cycle_delta1_policy_minus_A": cur_delta - a_delta,
            "fraction_frames_worse_than_A": float(np.mean([d > 0 for d in frame_diffs])) if frame_diffs else "",
            "cycle_worse_than_A": str(cur_abs > a_abs).lower(),
        })
    return seq_rows, cycle_rows


def build_cost_rows(run_meta: list[dict]) -> list[dict]:
    return sorted(run_meta, key=lambda r: (r["sequence"], r["arm"]))


def build_invariants(all_rows, run_meta, keyframes_by_arm, historical_refs, sequences, tolerance, pixel_tol):
    out = []
    for seq in sequences:
        a_key = keyframes_by_arm.get((seq, "A_native"), {})
        for (s, arm), kframes in keyframes_by_arm.items():
            if s != seq or arm == "A_native" or arm == "B1_global_scale_bias":
                continue
            diffs = []
            for idx, pred in kframes.items():
                if idx in a_key:
                    diffs.append(float(np.max(np.abs(pred.astype(np.float32) - a_key[idx].astype(np.float32)))))
            maximum = max(diffs, default=0.0)
            out.append({
                "sequence": seq, "arm": arm,
                "invariant": "keyframe_output_matches_A_for_noncalibration_arms",
                "observed_deviation_m": maximum, "threshold_m": pixel_tol,
                "passed": maximum <= pixel_tol,
            })

    # The output-only code path must reconstruct the unmodified native readout
    # before adapter application. Identity is checked in the run-time helper too.
    # Verify D/G1.0 exactly reproduces historical native C3-D.
    ref_by_seq_idx = {}
    for seq in sequences:
        for idx, ref in enumerate(historical_refs[seq]["D"]):
            ref_by_seq_idx[(seq, idx)] = ref
    for seq in sequences:
        rows = [r for r in all_rows if r["sequence"] == seq and r["arm"] == "S3_D_gain1.0_R1_F_only"
                and r["gt_matched"] == "true" and r["AbsRel"] != ""]
        diffs = []
        for r in rows:
            ref = ref_by_seq_idx.get((seq, int(r["frame_index_zero_based"])))
            if not ref or str(ref.get("AbsRel", "")).strip() == "":
                continue
            for metric in METRICS:
                diffs.append(abs(float(r[metric]) - float(ref[metric])))
        maximum = max(diffs, default=0.0)
        out.append({
            "sequence": seq, "arm": "S3_D_gain1.0_R1_F_only",
            "invariant": "D_gain1_F_only_reproduces_historical_C3_native_D",
            "observed_deviation": maximum, "threshold": tolerance,
            "passed": maximum <= tolerance,
        })

    # Output-only state contract: adapted features and output radius never mutate native state.
    meta_by_arm_seq = {(m["sequence"], m["arm"]): m for m in run_meta}
    for arm in sorted(OUTPUT_ONLY_ARMS):
        arm_meta = [m for m in run_meta if m["arm"] == arm]
        if not arm_meta:
            continue
        max_feature_mutation = max(float(m.get("output_only_state_feature_mutation_max_abs", 0.0)) for m in arm_meta)
        max_radius_mutation = max(float(m.get("output_only_state_radius_mutation_max_abs", 0.0)) for m in arm_meta)
        max_confidence_mutation = max(float(m.get("output_only_state_confidence_mutation_max_abs", 0.0)) for m in arm_meta)
        out.append({
            "sequence": "all", "arm": arm,
            "invariant": "output_only_preserves_native_feature_and_radius_state",
            "observed_deviation": max(max_feature_mutation, max_radius_mutation, max_confidence_mutation), "threshold": 0.0,
            "passed": max_feature_mutation == 0.0 and max_radius_mutation == 0.0 and max_confidence_mutation == 0.0,
            "detail": f"feature_mutation={max_feature_mutation}; radius_mutation={max_radius_mutation}; confidence_mutation={max_confidence_mutation}",
        })
    expected_arms = {
        "A_native", "B1_global_scale_bias", "B2_channel_affine_output_only",
        "B3_output_adapter_trained_output_only", "B3_frozen_C_output_only",
        "B3_frozen_D_output_only", "S1_C_once_per_cycle_F_only",
        "S2_C_gain0.5_R1_F_only", "S2_D_gain0.5_R1_F_only",
        "S3_C_gain1.0_R1_F_only", "S3_D_gain1.0_R1_F_only",
    }
    for seq in sequences:
        present = {m["arm"] for m in run_meta if m["sequence"] == seq}
        missing = sorted(expected_arms - present)
        out.append({
            "sequence": seq, "arm": "all",
            "invariant": "all_expected_arms_completed",
            "observed_deviation": len(missing), "threshold": 0,
            "passed": len(missing) == 0, "detail": ",".join(missing),
        })
        seq_rows = [r for r in all_rows if r["sequence"] == seq and r.get("run_source") == "Q0-3_new_inference"]
        for arm in sorted(OUTPUT_ONLY_ARMS):
            arm_rows = [r for r in seq_rows if r["arm"] == arm]
            if not arm_rows:
                continue
            expected_indices = {int(r["frame_index_zero_based"]) for r in arm_rows
                                if int(r["frame_offset_in_cycle"]) > 0}
            observed_indices = {int(r["frame_index_zero_based"]) for r in arm_rows
                                if r.get("output_adapter_called") == "true"}
            mismatch = sorted(expected_indices ^ observed_indices)
            out.append({
                "sequence": seq, "arm": arm,
                "invariant": "output_only_adapter_called_on_every_propagation_frame",
                "observed_deviation": len(mismatch), "threshold": 0,
                "passed": len(mismatch) == 0, "detail": ",".join(map(str, mismatch[:20])),
            })
        for arm in sorted(STATE_ARMS):
            arm_rows = [r for r in seq_rows if r["arm"] == arm]
            if not arm_rows:
                continue
            if arm == "S1_C_once_per_cycle_F_only":
                expected_indices = {int(r["frame_index_zero_based"]) for r in arm_rows
                                    if int(r["frame_offset_in_cycle"]) == 1}
            else:
                expected_indices = {int(r["frame_index_zero_based"]) for r in arm_rows
                                    if int(r["frame_offset_in_cycle"]) > 0}
            observed_indices = {int(r["frame_index_zero_based"]) for r in arm_rows
                                if r.get("repair_applied_this_frame") == "true"}
            mismatch = sorted(expected_indices ^ observed_indices)
            out.append({
                "sequence": seq, "arm": arm,
                "invariant": "repair_schedule_exact",
                "observed_deviation": len(mismatch), "threshold": 0,
                "passed": len(mismatch) == 0, "detail": ",".join(map(str, mismatch[:20])),
            })
    return out


def historical_reference_rows(historical_refs, manifests, sequences):
    out = []
    for seq in sequences:
        rows = manifests[seq]
        for mode in ("A", "C", "D"):
            refs = historical_refs[seq][mode]
            for idx, ref in enumerate(refs):
                matched = str(ref.get("AbsRel", "")).strip() != ""
                out.append({
                    "sequence": seq, "mode": mode,
                    "frame_index_zero_based": idx,
                    "rgb_file": Path(rows[idx]["rgb_path"]).name,
                    "gt_file": Path(rows[idx]["depth_path"]).name if matched else "",
                    "AbsRel": ref.get("AbsRel", ""), "RMSE": ref.get("RMSE", ""),
                    "delta1": ref.get("delta1", ""), "run_source": "historical_C3_reference",
                })
    return out


def main() -> int:
    args = parse_args()
    sequences = parse_sequences(args.sequences)
    if args.smoke_only and sequences != ["crowd2"]:
        raise ValueError("Run smoke with --sequences crowd2")
    out_dir = Path(args.out_dir).resolve()
    final_marker = out_dir / "q0_3_summary.json"
    if final_marker.exists() and not args.preflight_only:
        raise RuntimeError(f"Q0-3 completion marker exists; refuse overwrite: {final_marker}")

    pre, manifests, historical_refs, sample_order, order_sha, context = build_preflight(args, sequences)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "q0_3_preflight.json", pre)
    write_csv(out_dir / "q0_3_split_manifest.csv", context["split_rows"])
    print(f"[PASS] Q0-3 preflight: sequences={sequences}, K={REFRESH}, C/D checkpoints verified", flush=True)
    if args.preflight_only:
        print("Q0-3 preflight complete; no training or inference performed.")
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("Q0-3 requires CUDA for the audited VeloDepth inference/training path")
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    snapshot = Path(args.snapshot).resolve()
    os.environ["VDEEP_SNAPSHOT"] = str(snapshot)
    training_dir = out_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)

    # Build/reuse the exact C3 24-cycle native-A cache. Smoke caches only cycle 0.
    cache_build_start = time.perf_counter()
    cache_dir = c3.build_cycle_cache(
        context["paths"], context["train_rows"], training_dir, device,
        limit=1 if args.smoke_only else None,
    )
    cache_build_wall_sec = time.perf_counter() - cache_build_start
    print(f"[CACHE] build/reuse wall={cache_build_wall_sec:.1f}s", flush=True)
    reader = c3.CycleReader(cache_dir, max_items=2)
    train_order = [0] if args.smoke_only else sample_order
    requested_steps = 1 if args.smoke_only else FORMAL_STEPS

    model_train, template_head = load_base_model(snapshot, device)
    cached_keyframe_warmup_maxdiff = c3.warmup_shapes(model_train, reader, device)
    train_shape = exp1.shape_info_for(
        model_train, Path(context["train_rows"][0]["rgb_path"]), device
    )
    training_readout_check = verify_readout_equivalence(
        model_train, context["train_rows"], train_shape, device
    )
    print(
        f"[PASS] output-only readout wiring; cached keyframe diff={cached_keyframe_warmup_maxdiff:g}, "
        f"manual readout max diff={training_readout_check['max_abs_radius_difference']:g}",
        flush=True,
    )
    # The registered template is detached from the model so only the adapter trains.
    template_head.load_state_dict(context["initial_head_state"], strict=True)
    template_head.to(device)
    for p in template_head.parameters():
        p.requires_grad_(True)
    head_channels = int(template_head.channels)
    head_bottleneck = int(template_head.bottleneck)
    head_levels = int(template_head.levels)
    affine = ChannelAffine4(channels=head_channels, levels=head_levels).to(device)
    affine_init = {k: v.detach().cpu().clone() for k, v in affine.state_dict().items()}
    output_head = template_head
    output_init = {k: v.detach().cpu().clone() for k, v in output_head.state_dict().items()}

    b2_summary = train_one_output_only_module(
        "Q03_B2_channel_affine", model_train, affine, reader, train_order, training_dir,
        steps=requested_steps, initial_state=affine_init, lr=LR,
        smoke=args.smoke_only, resume=args.resume,
    )
    b3_summary = train_one_output_only_module(
        "Q03_B3_output_adapter", model_train, output_head, reader, train_order, training_dir,
        steps=requested_steps, initial_state=output_init, lr=LR,
        smoke=args.smoke_only, resume=args.resume,
    )
    del model_train, reader, template_head
    torch.cuda.empty_cache()

    if args.smoke_only:
        eval_sequences = ["crowd2"]
        max_frames = args.max_frames
        output_head.eval()
        for p in output_head.parameters():
            p.requires_grad_(False)
        affine.eval()
        for p in affine.parameters():
            p.requires_grad_(False)
        ckpt_paths = context["checkpoints"]
        frozen_heads = {}
        for label in ("C", "D"):
            obj = torch.load(ckpt_paths[label], map_location="cpu", weights_only=False)
            frozen_heads[label] = make_frozen_head(obj["repair_head_state_dict"], device)
            del obj
        eval_arms = [
            ("A_native", None, 1.0, "none"),
            ("B1_global_scale_bias", None, 1.0, "none"),
            ("B2_channel_affine_output_only", affine, 1.0, "none"),
            ("B3_output_adapter_trained_output_only", output_head, 1.0, "none"),
            ("B3_frozen_C_output_only", frozen_heads["C"], 1.0, "none"),
            ("B3_frozen_D_output_only", frozen_heads["D"], 1.0, "none"),
            ("S1_C_once_per_cycle_F_only", frozen_heads["C"], 1.0, "once_age1"),
            ("S2_C_gain0.5_R1_F_only", frozen_heads["C"], 0.5, "every_prop"),
            ("S2_D_gain0.5_R1_F_only", frozen_heads["D"], 0.5, "every_prop"),
            ("S3_C_gain1.0_R1_F_only", frozen_heads["C"], 1.0, "every_prop"),
            ("S3_D_gain1.0_R1_F_only", frozen_heads["D"], 1.0, "every_prop"),
        ]
        smoke_rows = []
        smoke_runs = []
        smoke_readout_checks = []
        smoke_keyframes = {}
        fit_stats = {"n": 0, "mean_p": 0.0, "mean_g": 0.0, "m2_p": 0.0, "cov_pg": 0.0}
        # Fit B1 from the first smoke frames only; this is mechanics-only and is not reused formally.
        for seq in eval_sequences:
            seq_rows = manifests[seq][:max_frames]
            model, _tmp = load_base_model(snapshot, device)
            shape = exp1.shape_info_for(model, Path(seq_rows[0]["rgb_path"]), device)
            smoke_readout_checks.append({
                "sequence": seq,
                **verify_readout_equivalence(model, seq_rows, shape, device),
            })
            a_rows, a_meta, a_keys = run_arm(
                seq, "A_native", model, seq_rows, shape, device, fit_stats=fit_stats
            )
            smoke_rows.extend(a_rows); smoke_runs.append(a_meta)
            smoke_keyframes[(seq, "A_native")] = a_keys
            del model, _tmp
        smoke_calibration = fit_scale_bias(fit_stats)
        for seq in eval_sequences:
            seq_rows = manifests[seq][:max_frames]
            model, _tmp = load_base_model(snapshot, device)
            shape = exp1.shape_info_for(model, Path(seq_rows[0]["rgb_path"]), device)
            for arm, adapter, gain, schedule in eval_arms:
                if arm == "A_native":
                    continue
                rows_out, meta, keys = run_arm(
                    seq, arm, model, seq_rows, shape, device, adapter=adapter,
                    gain=gain, schedule=schedule,
                    scale_bias=smoke_calibration if arm == "B1_global_scale_bias" else None,
                )
                smoke_rows.extend(rows_out); smoke_runs.append(meta)
                smoke_keyframes[(seq, arm)] = keys
            del model, _tmp
        seq_summary, cycle_summary = summarize(smoke_rows, smoke_runs)
        smoke_invariants = build_invariants(
            smoke_rows, smoke_runs, smoke_keyframes, historical_refs, eval_sequences,
            args.metric_tolerance, args.keyframe_pixel_tolerance_m,
        )
        for check in smoke_readout_checks:
            smoke_invariants.append({
                "sequence": check["sequence"], "arm": "output_only_readout_helper",
                "invariant": "manual_readout_matches_native_radius",
                "observed_deviation": check["max_abs_radius_difference"],
                "threshold": check["tolerance_m"], "passed": bool(check["passed"]),
            })
        smoke_all_pass = all(bool(r["passed"]) for r in smoke_invariants)
        write_csv(out_dir / "q0_3_smoke_frame_metrics.csv", smoke_rows)
        write_csv(out_dir / "q0_3_smoke_sequence_summary.csv", seq_summary)
        write_csv(out_dir / "q0_3_smoke_cycle_summary.csv", cycle_summary)
        write_csv(out_dir / "q0_3_smoke_invariants.csv", smoke_invariants)
        smoke_summary = {
            "status": "MECHANICS_PASS" if smoke_all_pass else "MECHANICS_FAIL_DO_NOT_INTERPRET",
            "experiment": "Q0-3 merged necessity controls",
            "formal_1000_step_training_started": False,
            "smoke_optimizer_updates_per_adapter": 1,
            "formal_checkpoints_saved": False,
            "all_invariants_pass": smoke_all_pass,
            "sequences": eval_sequences, "max_frames": max_frames,
            "cycle_cache_build_wall_sec": cache_build_wall_sec,
            "training_smoke": {"B2": b2_summary, "B3": b3_summary},
            "temporary_smoke_scale_bias": smoke_calibration,
            "runs": smoke_runs,
            "invariants": smoke_invariants,
            "output_only_readout_equivalence_checks": smoke_readout_checks,
            "interpretation": "Smoke checks wiring only. Do not interpret its fitted calibration or one-step adapters as scientific results.",
        }
        write_json(out_dir / "q0_3_smoke_summary.json", smoke_summary)
        with zipfile.ZipFile(out_dir / "q0_3_smoke_review.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            for name in ["q0_3_preflight.json", "q0_3_split_manifest.csv",
                         "q0_3_smoke_frame_metrics.csv", "q0_3_smoke_sequence_summary.csv",
                         "q0_3_smoke_cycle_summary.csv", "q0_3_smoke_invariants.csv",
                         "q0_3_smoke_summary.json"]:
                p = out_dir / name
                if p.is_file():
                    zf.write(p, name)
            zf.write(Path(__file__).resolve(), Path(__file__).name)
        print(f"[{'SMOKE PASS' if smoke_all_pass else 'SMOKE FAIL'}] Q0-3 smoke package: {out_dir / 'q0_3_smoke_review.zip'}", flush=True)
        return 0 if smoke_all_pass else 2

    # Formal adapter training completed. Load final checkpoints and historical C/D heads.
    affine_ckpt = training_dir / "Q03_B2_channel_affine_final_step1000.pt"
    output_ckpt = training_dir / "Q03_B3_output_adapter_final_step1000.pt"
    head_channels, head_bottleneck, head_levels = infer_repair_head_spec(
        context["initial_head_state"]
    )
    affine, affine_meta = load_trained_module(
        "Q03_B2_channel_affine",
        ChannelAffine4(channels=head_channels, levels=head_levels).to(device),
        affine_ckpt, device
    )
    output_head, output_meta = load_trained_module(
        "Q03_B3_output_adapter",
        PersistentFeatureRepairHead(
            channels=head_channels, bottleneck=head_bottleneck, levels=head_levels
        ).to(device),
        output_ckpt, device
    )
    if affine_meta["sample_order_sha256"] != order_sha or output_meta["sample_order_sha256"] != order_sha:
        raise RuntimeError("B2/B3 adapter sample-order provenance does not match C3")
    if output_meta["initial_adapter_state_sha256"] != context["initial_head_hash"]:
        raise RuntimeError("B3 output-only adapter did not start from the shared C3 initialization")
    expected_affine_params = 2 * head_channels * head_levels
    expected_output_params = sum(int(t.numel()) for t in context["initial_head_state"].values())
    if (affine_meta["trainable_parameter_count"] != expected_affine_params
            or output_meta["trainable_parameter_count"] != expected_output_params):
        raise RuntimeError(
            "B2/B3 adapter parameter counts differ from the dimensions of the shared C3 initialization: "
            f"B2={affine_meta['trainable_parameter_count']}/{expected_affine_params}, "
            f"B3={output_meta['trainable_parameter_count']}/{expected_output_params}"
        )
    frozen_heads = {}
    for label in ("C", "D"):
        obj = torch.load(context["checkpoints"][label], map_location="cpu", weights_only=False)
        frozen_heads[label] = make_frozen_head(obj["repair_head_state_dict"], device)
        del obj

    # Run native A first. Fit B1 only from native-A predictions on Crowd2 0..767.
    fit_stats = {"n": 0, "mean_p": 0.0, "mean_g": 0.0, "m2_p": 0.0, "cov_pg": 0.0}
    all_rows = []
    run_meta = []
    keyframes_by_arm = {}
    model_loads = []
    readout_checks = []
    for seq in sequences:
        seq_rows = manifests[seq]
        model_load_start = time.perf_counter()
        model, _tmp = load_base_model(snapshot, device)
        model_load_sec = time.perf_counter() - model_load_start
        shape = exp1.shape_info_for(model, Path(seq_rows[0]["rgb_path"]), device)
        readout_checks.append({
            "sequence": seq, "phase": "A_pass",
            **verify_readout_equivalence(model, seq_rows, shape, device),
        })
        rows_out, meta, keys = run_arm(
            seq, "A_native", model, seq_rows, shape, device,
            fit_stats=fit_stats if seq == "crowd2" else None,
        )
        meta["model_load_sec_shared_for_sequence"] = model_load_sec
        all_rows.extend(rows_out); run_meta.append(meta)
        keyframes_by_arm[(seq, "A_native")] = keys
        model_loads.append({"sequence": seq, "phase": "A_pass", "model_load_sec": model_load_sec})
        write_csv(out_dir / "q0_3_progress" / f"{seq}_A_native.csv", rows_out)
        del model, _tmp
        torch.cuda.empty_cache()

    calibration_fit_start = time.perf_counter()
    scale_bias = fit_scale_bias(fit_stats)
    scale_bias["final_ols_fit_compute_sec"] = time.perf_counter() - calibration_fit_start
    write_json(out_dir / "q0_3_global_scale_bias_fit.json", scale_bias)
    print(f"[B1 fit] scale={scale_bias['scale']:.8g}, bias={scale_bias['bias_m']:.8g}m, "
          f"pixels={scale_bias['valid_pixel_pairs']}", flush=True)

    eval_arms = [
        ("B1_global_scale_bias", None, 1.0, "none"),
        ("B2_channel_affine_output_only", affine, 1.0, "none"),
        ("B3_output_adapter_trained_output_only", output_head, 1.0, "none"),
        ("B3_frozen_C_output_only", frozen_heads["C"], 1.0, "none"),
        ("B3_frozen_D_output_only", frozen_heads["D"], 1.0, "none"),
        ("S1_C_once_per_cycle_F_only", frozen_heads["C"], 1.0, "once_age1"),
        ("S2_C_gain0.5_R1_F_only", frozen_heads["C"], 0.5, "every_prop"),
        ("S2_D_gain0.5_R1_F_only", frozen_heads["D"], 0.5, "every_prop"),
        ("S3_C_gain1.0_R1_F_only", frozen_heads["C"], 1.0, "every_prop"),
        ("S3_D_gain1.0_R1_F_only", frozen_heads["D"], 1.0, "every_prop"),
    ]
    for seq in sequences:
        seq_rows = manifests[seq]
        model_load_start = time.perf_counter()
        model, _tmp = load_base_model(snapshot, device)
        model_load_sec = time.perf_counter() - model_load_start
        model_loads.append({"sequence": seq, "phase": "intervention_pass", "model_load_sec": model_load_sec})
        shape = exp1.shape_info_for(model, Path(seq_rows[0]["rgb_path"]), device)
        readout_checks.append({
            "sequence": seq, "phase": "intervention_pass",
            **verify_readout_equivalence(model, seq_rows, shape, device),
        })
        for arm, adapter, gain, schedule in eval_arms:
            rows_out, meta, keys = run_arm(
                seq, arm, model, seq_rows, shape, device,
                adapter=adapter, gain=gain, schedule=schedule,
                scale_bias=scale_bias if arm == "B1_global_scale_bias" else None,
            )
            meta["model_load_sec_shared_for_sequence"] = model_load_sec
            all_rows.extend(rows_out); run_meta.append(meta)
            keyframes_by_arm[(seq, arm)] = keys
            write_csv(out_dir / "q0_3_progress" / f"{seq}_{arm}.csv", rows_out)
            write_json(out_dir / "q0_3_progress" / f"{seq}_{arm}_runtime.json", meta)
            print(f"[DONE] {seq}/{arm}: wall={meta['run_wall_sec']:.1f}s, "
                  f"repair={meta['repair_head_calls']}, output_adapter={meta['output_adapter_calls']}, "
                  f"extra_readout={meta['additional_readout_passes']}", flush=True)
        del model, _tmp
        torch.cuda.empty_cache()

    seq_summary, cycle_summary = summarize(all_rows, run_meta)
    cycle_groups = defaultdict(list)
    for cr in cycle_summary:
        if cr["cycle_complete_32_frames"] == "true":
            cycle_groups[(cr["sequence"], cr["arm"], cr["split"])].append(cr)
    for sr in seq_summary:
        cc = cycle_groups.get((sr["sequence"], sr["arm"], sr["split"]), [])
        if cc:
            deltas = [float(x["delta_cycle_AbsRel_policy_minus_A"]) for x in cc]
            damaged = [d > 0 for d in deltas]
            sr["complete_cycles"] = len(cc)
            sr["complete_cycles_worse_than_A"] = int(sum(damaged))
            sr["fraction_complete_cycles_worse_than_A"] = float(np.mean(damaged))
            sr["p95_complete_cycle_delta_AbsRel"] = float(np.percentile(deltas, 95))
            worst = max(cc, key=lambda x: float(x["delta_cycle_AbsRel_policy_minus_A"]))
            sr["worst_complete_cycle_index"] = int(worst["refresh_cycle"])
            sr["worst_complete_cycle_delta_AbsRel"] = float(worst["delta_cycle_AbsRel_policy_minus_A"])
        else:
            sr["complete_cycles"] = 0
            sr["complete_cycles_worse_than_A"] = ""
            sr["fraction_complete_cycles_worse_than_A"] = ""
            sr["p95_complete_cycle_delta_AbsRel"] = ""
            sr["worst_complete_cycle_index"] = ""
            sr["worst_complete_cycle_delta_AbsRel"] = ""
    invariant_rows = build_invariants(
        all_rows, run_meta, keyframes_by_arm, historical_refs, sequences,
        args.metric_tolerance, args.keyframe_pixel_tolerance_m,
    )
    for check in readout_checks:
        invariant_rows.append({
            "sequence": check["sequence"], "arm": "output_only_readout_helper",
            "invariant": "manual_readout_matches_native_radius",
            "observed_deviation": check["max_abs_radius_difference"],
            "threshold": check["tolerance_m"], "passed": bool(check["passed"]),
        })
    training_cost_rows = []
    # Historical C3 compute is read from the actual C3 summaries/curves.
    c3_dir = Path(args.c3_output_dir).resolve()
    for label in ("C", "D"):
        summary_path = c3_dir / f"gt4a2_c3_{label}_training_summary.json"
        curve_path = c3_dir / f"gt4a2_c3_{label}_training_curve.csv"
        if summary_path.is_file() and curve_path.is_file():
            old_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            with curve_path.open("r", newline="", encoding="utf-8-sig") as f:
                curve = list(csv.DictReader(f))
            training_cost_rows.append({
                "method": f"C3_persistent_{label}", "trainable_parameters": sum(int(t.numel()) for t in context["initial_head_state"].values()),
                "optimizer_updates": int(old_summary.get("steps", FORMAL_STEPS)),
                "repair_head_forward_calls": FORMAL_STEPS * (REFRESH - 1),
                "output_adapter_forward_calls": 0,
                "additional_readout_passes": 0,
                "training_wall_sec": old_summary.get("elapsed_sec", ""),
                "training_step_wall_sec_sum": float(sum(float(r.get("wall_sec", 0) or 0) for r in curve)),
                "mean_step_wall_sec": old_summary.get("mean_logged_step_wall_sec", ""),
                "mean_training_loss": old_summary.get("mean_logged_train_loss", ""),
                "training_loss_first_window": "",
                "training_loss_last_window": "",
                "peak_cuda_allocated_gib": max([float(r.get("peak_cuda_allocated_gib", 0) or 0) for r in curve] or [0]),
                "checkpoint_selection": "final step only",
                "same_c3_cycle_order": True,
                "supervised_propagation_frame_exposures": int(sum(
                    int(next(cy["supervised_propagated_frames"] for cy in context["train_cycles"]
                            if int(cy["cycle"]) == int(cycle_id)))
                    for cycle_id in sample_order
                )),
                "valid_supervision_pixel_exposures": "not logged in historical C3 curve",
                "source": "historical_C3_training_summary_and_curve",
            })
        else:
            training_cost_rows.append({
                "method": f"C3_persistent_{label}", "trainable_parameters": sum(int(t.numel()) for t in context["initial_head_state"].values()),
                "optimizer_updates": FORMAL_STEPS,
                "repair_head_forward_calls": FORMAL_STEPS * (REFRESH - 1),
                "output_adapter_forward_calls": 0,
                "additional_readout_passes": 0,
                "training_wall_sec": "",
                "training_step_wall_sec_sum": "",
                "mean_step_wall_sec": "", "mean_training_loss": "",
                "training_loss_first_window": "", "training_loss_last_window": "",
                "peak_cuda_allocated_gib": "",
                "checkpoint_selection": "final step only",
                "same_c3_cycle_order": True,
                "supervised_propagation_frame_exposures": "",
                "valid_supervision_pixel_exposures": "",
                "source": "historical_C3_training_compute_unavailable",
            })
    for label, summary in (("Q03_B2_channel_affine", b2_summary), ("Q03_B3_output_adapter", b3_summary)):
        training_cost_rows.append({
            "method": label, "trainable_parameters": summary.get("trainable_parameter_count", ""),
            "optimizer_updates": summary.get("optimizer_updates", summary.get("steps", "")),
            "repair_head_forward_calls": 0,
            "output_adapter_forward_calls": summary.get("output_adapter_forward_calls", ""),
            "additional_readout_passes": summary.get("additional_readout_passes", ""),
            "training_wall_sec": summary.get("training_wall_sec_current_session", ""),
            "training_step_wall_sec_sum": summary.get("training_wall_sec_sum_logged_steps", ""),
            "mean_step_wall_sec": summary.get("mean_step_wall_sec", ""),
            "mean_training_loss": "",
            "training_loss_first_window": summary.get("first_24_step_mean_log_depth_loss", summary.get("first_step_mean_log_depth_loss", "")),
            "training_loss_last_window": summary.get("last_24_step_mean_log_depth_loss", summary.get("last_step_mean_log_depth_loss", "")),
            "peak_cuda_allocated_gib": summary.get("peak_cuda_allocated_gib", ""),
            "checkpoint_selection": summary.get("checkpoint_selection", ""),
            "same_c3_cycle_order": True,
            "supervised_propagation_frame_exposures": summary.get("supervised_propagation_frame_exposures", ""),
            "valid_supervision_pixel_exposures": summary.get("valid_supervision_pixel_exposures", ""),
            "source": "Q0-3 measured training",
        })

    all_pass = all(bool(r["passed"]) for r in invariant_rows)
    historical_rows = historical_reference_rows(historical_refs, manifests, sequences)
    measured_model_load_sec = float(sum(float(x["model_load_sec"]) for x in model_loads))
    measured_adapter_training_sec = float(
        b2_summary.get("training_wall_sec_current_session", 0.0)
        + b3_summary.get("training_wall_sec_current_session", 0.0)
    )
    measured_inference_sec = float(sum(float(r["run_wall_sec"]) for r in run_meta))
    measured_total_sec = cache_build_wall_sec + measured_model_load_sec + measured_adapter_training_sec + measured_inference_sec
    summary = {
        "status": "MECHANICS_PASS" if all_pass else "MECHANICS_FAIL_DO_NOT_INTERPRET",
        "experiment": "Q0-3 merged necessity controls",
        "training_performed": True,
        "runtime_sec": float(sum(float(r["run_wall_sec"]) for r in run_meta)),
        "cycle_cache_build_wall_sec": cache_build_wall_sec,
        "model_load_wall_sec_sum": measured_model_load_sec,
        "adapter_training_wall_sec_current_sessions": measured_adapter_training_sec,
        "adapter_training_step_wall_sec_sum": float(
            b2_summary.get("training_wall_sec_sum_logged_steps", 0.0)
            + b3_summary.get("training_wall_sec_sum_logged_steps", 0.0)
        ),
        "total_measured_wall_sec_current_session_basis": measured_total_sec,
        "sequences": sequences, "refresh_interval": REFRESH,
        "training_split": {"fit_sequence": "crowd2", "fit_indices_inclusive": [0, 767],
                           "crowd2_temporal_holdout": [768, len(manifests["crowd2"]) - 1],
                           "crowd3_balloon2_used_for_fit": False},
        "calibration": scale_bias,
        "trained_adapters": {"B2": affine_meta, "B3": output_meta},
        "counts": {
            "new_frame_rows": len(all_rows),
            "new_inference_runs": len(run_meta),
            "sequence_summary_rows": len(seq_summary),
            "cycle_summary_rows": len(cycle_summary),
            "invariant_rows": len(invariant_rows),
        },
        "mechanics": {"all_invariants_pass": all_pass,
                      "metric_tolerance": args.metric_tolerance,
                      "keyframe_pixel_tolerance_m": args.keyframe_pixel_tolerance_m},
        "protocol_limits": [
            "B1 scale/bias is fit only on native-A Crowd2 pixels from frames 0..767; no Crowd3/Balloon2 or Crowd2 768+ GT is used for fitting.",
            "B2 and B3 output-only adapters are applied to every non-keyframe propagation readout (ages 1..31); keyframe outputs remain native because C3 supervision uses GT-bearing propagation frames only. Native radius, confidence and feature state are always carried forward.",
            "B3 includes a newly trained same-architecture adapter from the C3 common initialization, using the exact C3 cycle order, GT-bearing propagation frames, 1000 updates and final-step-only selection.",
            "Frozen C/D output-only arms are diagnostics only and cannot alone prove writeback necessity because those heads were trained under state-feedback semantics.",
            "S1 uses the C head once at age 1 per K=32 cycle, post-readout F-only. This is a cross-semantics intervention, not native C reproduction.",
            "S2/S3 cross both C/D heads under F-only post-readout semantics; D/G1.0 must reproduce historical native C3-D.",
            "Timing is measured per arm and includes I/O, preprocessing, optical flow, inference, adaptation and metrics; model load is reported separately. Single-run timings are descriptive, not proof of speedup.",
            "Crowd2/Crowd3/Balloon2 are development data; Crowd2 768+ is a temporal holdout but not an independent scene/domain.",
        ],
        "model_loads": model_loads,
        "training_compute": training_cost_rows,
        "cached_keyframe_state_warmup_max_abs_diff": cached_keyframe_warmup_maxdiff,
        "training_output_only_readout_equivalence_check": training_readout_check,
        "evaluation_readout_equivalence_checks": readout_checks,
    }
    write_csv(out_dir / "q0_3_frame_metrics.csv", all_rows)
    write_csv(out_dir / "q0_3_sequence_summary.csv", seq_summary)
    write_csv(out_dir / "q0_3_cycle_summary.csv", cycle_summary)
    write_csv(out_dir / "q0_3_runtime_costs.csv", build_cost_rows(run_meta))
    write_csv(out_dir / "q0_3_invariants.csv", invariant_rows)
    write_csv(out_dir / "q0_3_historical_C3_reference.csv", historical_rows)
    write_csv(out_dir / "q0_3_training_compute.csv", training_cost_rows)
    write_json(out_dir / "q0_3_summary.json", summary)

    readme = f"""# Q0-3 Merged Necessity Controls

Status: **{summary['status']}**

- Sequences: {', '.join(sequences)}
- K: {REFRESH}
- Adapter training: B2 and B3 each use the C3 1000-step cycle order and final-step-only checkpoint selection.
- Calibration fit: Crowd2 frame indices 0..767 only.
- Global scale/bias: scale={scale_bias['scale']:.8g}, bias={scale_bias['bias_m']:.8g} m.
- New inference runs: {len(run_meta)}
- Total measured arm wall time (sum, model loads excluded): {summary['runtime_sec']:.1f} sec.
- Mechanics invariants: {'PASS' if all_pass else 'FAIL'}.

## Arms

A native VeloDepth; B1 global z-depth scale/bias; B2 trained per-channel affine output-only; B3 trained same-architecture output-only adapter; frozen C/D output-only diagnostic controls; S1 one C-head F-only writeback at age 1 per cycle; S2 C/D gain 0.5 every propagation frame; S3 C/D gain 1.0 every propagation frame.

## State isolation

B1 calibrates only the emitted z-depth. B2/B3 compute a temporary adapted readout on every propagation frame (ages 1..31); keyframe outputs remain native to match the C3 supervision set. The native radius, confidence and feature maps are the only state carried forward. The adapted output is never written to the next frame. S1/S2/S3 commit only the repaired feature maps after current-frame readout; current radius/confidence are unchanged.

## Training fairness

The trained B3 adapter starts from the exact shared C3 initialization. B2 and B3 use the exact C3 sampled 32-frame cycle order, the same GT-bearing propagation frames, 1000 optimizer updates, AdamW, the same loss, and final-step-only checkpoint selection. Actual wall time, per-step time, trainable parameters, frame/GT exposure counts and peak CUDA memory are in `q0_3_training_compute.csv`. Historical C/D compute is read from their recorded summaries/curves.

## Interpretation limits

Frozen C/D output-only runs are diagnostics, not the fair B3 comparator. Crowd2 0..767 is training/fitting data; Crowd2 768+ is temporal holdout; Crowd3 and Balloon2 are development cross-sequence evaluations. Single-run runtime is descriptive and does not establish speedup.
"""
    (out_dir / "Q0_3_README.md").write_text(readme, encoding="utf-8")

    include = [
        "q0_3_preflight.json", "q0_3_split_manifest.csv", "q0_3_global_scale_bias_fit.json",
        "q0_3_frame_metrics.csv", "q0_3_sequence_summary.csv", "q0_3_cycle_summary.csv",
        "q0_3_runtime_costs.csv", "q0_3_invariants.csv", "q0_3_historical_C3_reference.csv",
        "q0_3_training_compute.csv", "q0_3_summary.json", "Q0_3_README.md",
        "training/Q03_B2_channel_affine_final_step1000.pt",
        "training/Q03_B3_output_adapter_final_step1000.pt",
        "training/Q03_B2_channel_affine_training_summary.json",
        "training/Q03_B3_output_adapter_training_summary.json",
        "training/Q03_B2_channel_affine_training_curve.csv",
        "training/Q03_B3_output_adapter_training_curve.csv",
    ]
    review_zip = out_dir / "q0_3_review.zip"
    with zipfile.ZipFile(review_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            p = out_dir / rel
            if p.is_file():
                zf.write(p, rel)
        zf.write(Path(__file__).resolve(), Path(__file__).name)
    print(f"[{summary['status']}] Q0-3 review package: {review_zip}", flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise
