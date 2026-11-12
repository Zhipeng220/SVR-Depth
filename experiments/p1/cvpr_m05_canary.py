#!/usr/bin/env python3
"""CVPR M05-Canary: five-branch persistent-state contract check.

Development-only, one Crowd2 window, K=16, age=4, H=4 by default.
No training, optimizer, checkpoint writes, or test-set claims.

The five branches follow CVPR主实验执行计划 v2.1:
  A: D0 output; commit (r0,c0,F0)
  O: D1 output; commit (r0,c0,F0)
  F: D0 output; commit (r0,c0,F1)
  T: D1 output; commit (r0,c0,F1)
  P_native: D1 output; commit (r1,c1,F1)

A-native and P-maintained parent histories are run separately. After the
intervention, repair is disabled for all future replay frames until the next
scheduled refresh. The default anchor is in the Crowd2 temporal development
tail (cycle 48 of K=16; frame 772), not a blind test set.
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
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

import gt2_fixed_clip_repair_train as gt2
import gt4a2_exp1_train_eval as exp1

K = 16
DEFAULT_CYCLE = 48
DEFAULT_AGE = 4
DEFAULT_HORIZON = 4
P_CHECKPOINT_SHA256 = "77a86cdf523f776608b6b591d6c6552002b90398e85d00d3a3fe29afe14a0a24"
BASE_MODEL_SHA256 = "615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e"
# The base-model digest above is checked against the audited Y02/Y04 provenance.
# A mismatch is a hard stop; never silently accept a different base snapshot.

SOURCE_HASHES = {
    "experiments/p1/gt4a2_y04_fair_multiscene_fix2.py":
        "4972e676e8f4b7c867615546eac34dcebce96c6b3e8cc0b5ad51e91fce14d291",
    "experiments/p1/gt4a2_exp1_train_eval.py":
        "c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f",
    "experiments/p1/gt2_fixed_clip_repair_train.py":
        "bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192",
    "experiments/p1/gt4a2_c3_closed_loop.py":
        "76327208acac4543b2a87ccba8b7af435e4c5ce65267b42052e51b435cbad7f5",
    "velodepth/models/velodepth.py":
        "739c5f27a749174987b9a36edaac32fb98070e87dbef0a0a52f7430d61149479",
    "velodepth/models/encoder_velo.py":
        "3c5aa2a3a6dce03b68373953037fe8a0313a2cab36d66e5e3b88a56ec4a7ba8a",
    "velodepth/models/decoder_velo.py":
        "35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08",
    "velodepth/models/persistent_feature_repair.py":
        "6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c",
}

PAIR_CSV_DEFAULT = "outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_sources() -> dict:
    result = {}
    for rel, expected in SOURCE_HASHES.items():
        p = ROOT / rel
        if not p.is_file():
            raise FileNotFoundError(f"Required pinned source missing: {p}")
        actual = sha256_file(p)
        result[rel] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            raise RuntimeError(
                f"Source hash mismatch; refusing M05-Canary: {rel}: {actual} != {expected}"
            )
    return result


def find_p_checkpoint(explicit: str) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        if not p.is_file():
            raise FileNotFoundError(f"P_multi checkpoint not found: {p}")
        actual = sha256_file(p)
        if actual != P_CHECKPOINT_SHA256:
            raise RuntimeError(f"P_multi checkpoint SHA mismatch: {actual}")
        return p.resolve()

    candidates = []
    preferred = (
        ROOT / "outputs/gt4a2_y04_validation_fix3_20261025_225902/"
        "training/P_multi/gt4a2_c3_C_final_step1000.pt"
    )
    if preferred.is_file():
        candidates.append(preferred)
    outputs = ROOT / "outputs"
    if outputs.is_dir():
        for p in outputs.rglob("gt4a2_c3_C_final_step1000.pt"):
            if p.parent.name == "P_multi" and p not in candidates:
                candidates.append(p)

    matches = []
    for p in candidates:
        if p.is_file() and sha256_file(p) == P_CHECKPOINT_SHA256:
            matches.append(p.resolve())
    unique = sorted({str(p): p for p in matches}.values(), key=lambda p: str(p))
    if len(unique) != 1:
        raise RuntimeError(
            "Could not identify exactly one P_multi checkpoint by the M00-verified "
            f"SHA256. Matching candidates={len(unique)}. Set SVRDEPTH_P_CHECKPOINT "
            "to the exact path from M00 checkpoint_candidates.csv."
        )
    return unique[0]


def load_checkpoint_metadata(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "mode_label": "C",
        "mode": "pre_readout",
        "y04_label": "Y04_P_multi",
        "y04_kind": "persistent",
        "pretrained_weights_sha256": BASE_MODEL_SHA256,
    }
    for key, expected in required.items():
        actual = ckpt.get(key)
        if actual != expected:
            raise RuntimeError(
                f"P_multi checkpoint metadata mismatch: {key}={actual!r}, expected={expected!r}"
            )
    if int(ckpt.get("step_count", -1)) != 1000:
        raise RuntimeError("P_multi checkpoint is not the frozen final 1000-step checkpoint.")
    state = ckpt.get("repair_head_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("P_multi checkpoint has no repair_head_state_dict.")
    return {
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "mode_label": ckpt["mode_label"],
        "mode": ckpt["mode"],
        "y04_label": ckpt["y04_label"],
        "y04_kind": ckpt["y04_kind"],
        "step_count": int(ckpt["step_count"]),
        "pretrained_weights_sha256": ckpt["pretrained_weights_sha256"],
        "repair_head_state_dict": state,
        "repair_head_keys": len(state),
    }


def clone_state(s: dict) -> dict:
    return {
        "radius": s["radius"].detach().clone(),
        "confidence": s["confidence"].detach().clone(),
        "features": [x.detach().clone() for x in s["features"]],
        "rays_map": s["rays_map"].detach().clone(),
    }


def max_abs(a, b) -> float:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return float("inf")
        if a.numel() == 0:
            return 0.0
        d = (a.detach().float() - b.detach().float()).abs()
        if not bool(torch.isfinite(d).all()):
            return float("inf")
        return float(d.max().item())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((max_abs(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return float("inf")
        return max((max_abs(a[k], b[k]) for k in a), default=0.0)
    return 0.0 if a == b else float("inf")


def state_diffs(a: dict, b: dict) -> dict:
    return {
        "radius": max_abs(a["radius"], b["radius"]),
        "confidence": max_abs(a["confidence"], b["confidence"]),
        "features": max_abs(a["features"], b["features"]),
        "rays_map": max_abs(a["rays_map"], b["rays_map"]),
    }


def state_hash(s: dict) -> str:
    h = hashlib.sha256()
    for name in ("radius", "confidence", "features", "rays_map"):
        values = s[name] if isinstance(s[name], list) else [s[name]]
        h.update(name.encode("utf-8"))
        for x in values:
            arr = x.detach().cpu().contiguous().numpy()
            h.update(str(arr.dtype).encode("ascii"))
            h.update(str(tuple(arr.shape)).encode("ascii"))
            h.update(arr.tobytes())
    return h.hexdigest()


def set_repair_mode(model, mode: str) -> None:
    if mode not in {"off", "pre_readout"}:
        raise ValueError(mode)
    model.pixel_decoder.feature_repair_mode = mode


def run_step(model, prev, curr, static, state):
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        r, c, f = exp1.one_fast_step(
            model, prev, curr, static,
            state["radius"], state["confidence"], state["features"]
        )
    return {
        "radius": r.detach(),
        "confidence": c.detach(),
        "features": [x.detach() for x in f],
        "rays_map": state["rays_map"].detach(),
    }


def build_parent(model, frames, static_by_idx, shape, cycle_start: int,
                 intervention_idx: int, mode: str) -> dict:
    set_repair_mode(model, mode)
    rays, radius, confidence, features = exp1.keyframe_state(
        model, frames[cycle_start], shape
    )
    state = {
        "radius": radius.detach(),
        "confidence": confidence.detach(),
        "features": [x.detach() for x in features],
        "rays_map": rays.detach(),
    }
    for idx in range(cycle_start + 1, intervention_idx):
        state = run_step(model, frames[idx - 1], frames[idx], static_by_idx[idx], state)
    return clone_state(state)


def make_prediction(model, radius, rays_map, shape) -> np.ndarray:
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        pred = gt2.make_z_depth(
            radius, rays_map, shape["padded_shape"], shape["paddings"],
            model.interpolation_mode
        )
    return pred.detach().float().cpu().numpy()


def metrics_or_empty(gt_np, pred_np) -> dict:
    if gt_np is None:
        return {"gt_matched": False, "AbsRel": "", "RMSE": "", "delta1": "",
                "valid_ratio": "", "valid_pixels": 0}
    return {"gt_matched": True, **gt2.calculate_metrics(gt_np, pred_np)}


def replay_branch(model, frames, static_by_idx, shape, intervention_idx: int,
                  horizon: int, initial_state: dict, parent_label: str,
                  branch_label: str, tolerance: float) -> tuple[list[dict], dict[int, np.ndarray]]:
    set_repair_mode(model, "off")
    state = clone_state(initial_state)
    rows = []
    predictions = {}
    for h in range(1, horizon + 1):
        idx = intervention_idx + h
        state = run_step(model, frames[idx - 1], frames[idx], static_by_idx[idx], state)
        pred = make_prediction(model, state["radius"], state["rays_map"], shape)
        predictions[h] = pred
        metric = metrics_or_empty(frames[idx]["gt_np"], pred)
        rows.append({
            "parent": parent_label,
            "branch": branch_label,
            "horizon": h,
            "frame_index": idx,
            "repair_mode_after_intervention": "off",
            **metric,
        })
        if not np.isfinite(pred).all():
            raise RuntimeError(f"Non-finite future prediction: {parent_label}/{branch_label}/H{h}")
    return rows, predictions


def build_branches(model, frames, static_by_idx, shape, parent_state: dict,
                   parent_label: str, intervention_idx: int, horizon: int,
                   tolerance: float) -> tuple[list[dict], list[dict], dict]:
    prev = frames[intervention_idx - 1]
    curr = frames[intervention_idx]

    set_repair_mode(model, "off")
    s0 = run_step(model, prev, curr, static_by_idx[intervention_idx], clone_state(parent_state))
    set_repair_mode(model, "pre_readout")
    s1 = run_step(model, prev, curr, static_by_idx[intervention_idx], clone_state(parent_state))

    d0 = make_prediction(model, s0["radius"], parent_state["rays_map"], shape)
    d1 = make_prediction(model, s1["radius"], parent_state["rays_map"], shape)

    # A/O share the native state; F/T share repaired features but retain r0,c0.
    states = {
        "A": {"radius": s0["radius"], "confidence": s0["confidence"],
              "features": s0["features"], "rays_map": parent_state["rays_map"]},
        "O": {"radius": s0["radius"], "confidence": s0["confidence"],
              "features": s0["features"], "rays_map": parent_state["rays_map"]},
        "F": {"radius": s0["radius"], "confidence": s0["confidence"],
              "features": s1["features"], "rays_map": parent_state["rays_map"]},
        "T": {"radius": s0["radius"], "confidence": s0["confidence"],
              "features": s1["features"], "rays_map": parent_state["rays_map"]},
        "P_native": {"radius": s1["radius"], "confidence": s1["confidence"],
                     "features": s1["features"], "rays_map": parent_state["rays_map"]},
    }
    current_pred = {
        "A": d0, "F": d0, "O": d1, "T": d1, "P_native": d1,
    }

    current_rows = []
    for branch in ("A", "O", "F", "T", "P_native"):
        metric = metrics_or_empty(curr["gt_np"], current_pred[branch])
        current_rows.append({
            "parent": parent_label,
            "branch": branch,
            "frame_index": intervention_idx,
            "frame_age": intervention_idx % K,
            "current_output_source": "D0" if branch in {"A", "F"} else "D1",
            "committed_radius_source": "r1" if branch == "P_native" else "r0",
            "committed_confidence_source": "c1" if branch == "P_native" else "c0",
            "committed_feature_source": "F0" if branch in {"A", "O"} else "F1",
            "committed_state_sha256": state_hash(states[branch]),
            **metric,
        })

    future_rows = []
    future_predictions = {}
    for branch in ("A", "O", "F", "T", "P_native"):
        rows, preds = replay_branch(
            model, frames, static_by_idx, shape, intervention_idx, horizon,
            states[branch], parent_label, branch, tolerance
        )
        future_rows.extend(rows)
        future_predictions[branch] = preds

    diffs = {
        "A_vs_O_committed_state": state_diffs(states["A"], states["O"]),
        "F_vs_T_committed_state": state_diffs(states["F"], states["T"]),
        "F_vs_P_native_committed_state": state_diffs(states["F"], states["P_native"]),
        "current_D0_A_vs_F_max_abs": float(np.max(np.abs(current_pred["A"] - current_pred["F"]))),
        "current_D1_O_vs_T_max_abs": float(np.max(np.abs(current_pred["O"] - current_pred["T"]))),
        "current_D1_T_vs_P_native_max_abs": float(np.max(np.abs(current_pred["T"] - current_pred["P_native"]))),
    }

    future_checks = {}
    for h in range(1, horizon + 1):
        a = future_predictions["A"][h]
        o = future_predictions["O"][h]
        f = future_predictions["F"][h]
        t = future_predictions["T"][h]
        p = future_predictions["P_native"][h]
        future_checks[str(h)] = {
            "A_vs_O_max_abs": float(np.max(np.abs(a - o))),
            "F_vs_T_max_abs": float(np.max(np.abs(f - t))),
            "T_minus_O_equals_F_minus_A_max_abs": float(
                np.max(np.abs((t - o) - (f - a)))
            ),
            "P_native_vs_T_max_abs": float(np.max(np.abs(p - t))),
        }

    # Fixed-K propagation API control: incoming confidence is not an input to
    # gt2.fast_step; perturbing only that stored value must not change the next step.
    set_repair_mode(model, "off")
    confidence_state = clone_state(states["A"])
    confidence_perturbed = clone_state(states["A"])
    confidence_perturbed["confidence"] = (
        confidence_perturbed["confidence"] * 1.25 + 0.01
    )
    next_idx = intervention_idx + 1
    out_c0 = run_step(model, frames[intervention_idx], frames[next_idx],
                      static_by_idx[next_idx], confidence_state)
    out_c1 = run_step(model, frames[intervention_idx], frames[next_idx],
                      static_by_idx[next_idx], confidence_perturbed)
    confidence_only = {
        "input_confidence_max_abs_change": max_abs(
            confidence_state["confidence"], confidence_perturbed["confidence"]
        ),
        "next_radius_max_abs_change": max_abs(out_c0["radius"], out_c1["radius"]),
        "next_confidence_output_max_abs_change": max_abs(
            out_c0["confidence"], out_c1["confidence"]
        ),
        "next_features_max_abs_change": max_abs(out_c0["features"], out_c1["features"]),
        "scope": "fixed-K fast_step path only; does not test keyframe overlap blending",
    }

    invariants = {
        "A_O_state_equal": max_abs(states["A"], states["O"]) <= tolerance,
        "F_T_state_equal": max_abs(states["F"], states["T"]) <= tolerance,
        "F_T_Pnative_feature_equal": (
            max_abs(states["F"]["features"], states["T"]["features"]) <= tolerance
            and max_abs(states["F"]["features"], states["P_native"]["features"]) <= tolerance
        ),
        "current_D0_A_F_equal": diffs["current_D0_A_vs_F_max_abs"] <= tolerance,
        "current_D1_O_T_equal": diffs["current_D1_O_vs_T_max_abs"] <= tolerance,
        "current_D1_T_Pnative_equal": diffs["current_D1_T_vs_P_native_max_abs"] <= tolerance,
        "future_A_O_equal": all(v["A_vs_O_max_abs"] <= tolerance for v in future_checks.values()),
        "future_F_T_equal": all(v["F_vs_T_max_abs"] <= tolerance for v in future_checks.values()),
        "future_T_O_equals_F_A": all(
            v["T_minus_O_equals_F_minus_A_max_abs"] <= tolerance
            for v in future_checks.values()
        ),
        "confidence_not_consumed_by_fixed_fast_step": (
            confidence_only["next_radius_max_abs_change"] <= tolerance
            and confidence_only["next_confidence_output_max_abs_change"] <= tolerance
            and confidence_only["next_features_max_abs_change"] <= tolerance
        ),
    }

    detail = {
        "parent": parent_label,
        "state_diffs": diffs,
        "future_checks": future_checks,
        "confidence_only_control": confidence_only,
        "invariants": invariants,
        "passed": all(invariants.values()),
    }
    return current_rows, future_rows, detail


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    ap.add_argument("--checkpoint", default=os.environ.get("SVRDEPTH_P_CHECKPOINT", ""))
    ap.add_argument("--sequence", default="crowd2", choices=("crowd2",))
    ap.add_argument("--pair-csv", default=PAIR_CSV_DEFAULT)
    ap.add_argument("--cycle", type=int, default=DEFAULT_CYCLE)
    ap.add_argument("--age", type=int, default=DEFAULT_AGE)
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--tolerance", type=float, default=1e-5)
    ap.add_argument("--out-dir", default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M05-Canary requires CUDA; no CPU fallback is allowed.")
    if not args.snapshot:
        raise RuntimeError("VDEEP_SNAPSHOT is empty.")
    snapshot = Path(args.snapshot).expanduser().resolve()
    for name in ("config.json", "model.safetensors"):
        if not (snapshot / name).is_file():
            raise FileNotFoundError(snapshot / name)
    base_sha = sha256_file(snapshot / "model.safetensors")
    if base_sha != BASE_MODEL_SHA256:
        raise RuntimeError(f"VeloDepth model SHA mismatch: {base_sha}")

    source_meta = verify_sources()
    checkpoint_path = find_p_checkpoint(args.checkpoint)
    checkpoint_meta = load_checkpoint_metadata(checkpoint_path)

    pair_path = Path(args.pair_csv)
    if not pair_path.is_absolute():
        pair_path = ROOT / pair_path
    rows = exp1.read_manifest(pair_path, training=False)
    cycle_start = args.cycle * K
    intervention_idx = cycle_start + args.age
    last_idx = intervention_idx + args.horizon
    next_refresh_idx = cycle_start + K
    if args.age < 1 or args.age >= K:
        raise ValueError(f"age must be in 1..{K-1}")
    if args.horizon < 1 or intervention_idx + args.horizon >= next_refresh_idx:
        raise ValueError(
            f"Horizon crosses K={K} refresh boundary: anchor={intervention_idx}, "
            f"H={args.horizon}, next_refresh={next_refresh_idx}"
        )
    if last_idx >= len(rows):
        raise RuntimeError(f"Manifest has {len(rows)} frames; required through index {last_idx}.")
    target_indices = [intervention_idx] + list(range(intervention_idx + 1, last_idx + 1))
    for idx in target_indices:
        if str(rows[idx].get("gt_match_status", "")).strip().upper() != "MATCHED":
            raise RuntimeError(f"Canary requires matched GT at frame {idx}.")
    if args.tolerance <= 0:
        raise ValueError("--tolerance must be positive.")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        out_dir = ROOT / "outputs" / f"cvpr_m05_canary_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda")
    print("[M05-CANARY] source hashes PASS", flush=True)
    print(f"[M05-CANARY] P_multi checkpoint={checkpoint_path}", flush=True)
    print(f"[M05-CANARY] checkpoint sha256={checkpoint_meta['sha256']}", flush=True)
    print(f"[M05-CANARY] data={pair_path} sha256={sha256_file(pair_path)}", flush=True)
    print(
        f"[M05-CANARY] K={K} cycle={args.cycle} age={args.age} "
        f"anchor={intervention_idx} H={args.horizon} next_refresh={next_refresh_idx}",
        flush=True,
    )
    print("[M05-CANARY] loading frozen VeloDepth + P_multi head", flush=True)

    model = gt2.load_model(snapshot, "pre_readout", device)
    head = model.pixel_decoder.persistent_feature_repair
    if head is None:
        raise RuntimeError("P_multi repair head is not registered.")
    head.load_state_dict(checkpoint_meta.pop("repair_head_state_dict"), strict=True)
    head.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.pixel_decoder.feature_repair_mode = "off"

    shape = exp1.shape_info_for(model, Path(rows[cycle_start]["rgb_path"]), device)
    frames = {}
    for idx in range(cycle_start, last_idx + 1):
        need_gt = idx in target_indices
        frames[idx] = exp1.load_frame(rows[idx], shape, device, need_gt=need_gt)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    static_by_idx = {}
    for idx in range(cycle_start + 1, last_idx + 1):
        static_by_idx[idx] = exp1.transition_static(
            model, frames[idx - 1], frames[idx], shape, dis
        )

    # Generate both parent histories from the same keyframe and same RGB/flow inputs.
    parent_a = build_parent(
        model, frames, static_by_idx, shape, cycle_start, intervention_idx, "off"
    )
    parent_p = build_parent(
        model, frames, static_by_idx, shape, cycle_start, intervention_idx, "pre_readout"
    )

    current_rows = []
    future_rows = []
    details = []
    for label, parent in (("A_native", parent_a), ("P_maintained", parent_p)):
        crows, frows, detail = build_branches(
            model, frames, static_by_idx, shape, parent, label,
            intervention_idx, args.horizon, args.tolerance
        )
        current_rows.extend(crows)
        future_rows.extend(frows)
        details.append(detail)
        print(
            f"[M05-CANARY] parent={label} invariants="
            f"{'PASS' if detail['passed'] else 'FAIL'}",
            flush=True,
        )

    invariants_pass = all(x["passed"] for x in details)
    summary = {
        "experiment": "CVPR-M05-Canary",
        "status": "CANARY_PASS" if invariants_pass else "CANARY_INVARIANT_FAIL",
        "training_performed": False,
        "model_inference_performed": True,
        "test_set_used": False,
        "development_only": True,
        "sequence": args.sequence,
        "pair_manifest": str(pair_path.resolve()),
        "pair_manifest_sha256": sha256_file(pair_path),
        "cycle": args.cycle,
        "K": K,
        "age": args.age,
        "intervention_frame_index": intervention_idx,
        "horizon": args.horizon,
        "future_frame_indices": list(range(intervention_idx + 1, last_idx + 1)),
        "next_refresh_index": next_refresh_idx,
        "parent_states": ["A_native", "P_maintained"],
        "branches": ["A", "O", "F", "T", "P_native"],
        "future_repair_mode": "off",
        "anchor_role": "Crowd2 temporal development tail; previously exposed development data, not blind test",
        "checkpoint": checkpoint_meta,
        "base_model_sha256": base_sha,
        "source_hashes": source_meta,
        "details": details,
        "interpretation_limit": (
            "One anchor per parent is a mechanics/state-contract canary only. "
            "It is not a formal causal estimate, statistical result, independent test, "
            "or evidence of generalization. Formal M05 must use pre-frozen multiple "
            "anchors/recordings and the v2.1 sampling/statistics protocol."
        ),
    }
    (out_dir / "m05_canary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(out_dir / "m05_canary_current_frame.csv", current_rows)
    write_csv(out_dir / "m05_canary_future_frames.csv", future_rows)
    (out_dir / "REPORT.md").write_text(
        "\n".join([
            "# CVPR M05-Canary",
            "",
            f"- Status: **{summary['status']}**",
            "- Training: NOT RUN",
            "- Test set: NOT USED",
            f"- Sequence: {args.sequence}",
            f"- K / age / H: {K} / {args.age} / {args.horizon}",
            f"- Intervention frame: {intervention_idx}",
            f"- Next refresh: {next_refresh_idx}",
            f"- P_multi checkpoint SHA256: `{checkpoint_meta['sha256']}`",
            "",
            "## Parent-state invariant checks",
            "",
            *[
                f"- {d['parent']}: {'PASS' if d['passed'] else 'FAIL'}"
                for d in details
            ],
            "",
            "## Interpretation",
            "",
            "This is a one-window development mechanics canary. It does not establish "
            "a causal benefit, statistical significance, independent-test performance, "
            "or generalization. A failed invariant invalidates the corresponding branch "
            "comparison and must be debugged before formal M05.",
            "",
            "## Files",
            "",
            "- `m05_canary.json`",
            "- `m05_canary_current_frame.csv`",
            "- `m05_canary_future_frames.csv`",
            "- `cvpr_m05_canary_review.zip`",
        ]) + "\n",
        encoding="utf-8",
    )
    review = out_dir / "cvpr_m05_canary_review.zip"
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in (
            "m05_canary.json",
            "m05_canary_current_frame.csv",
            "m05_canary_future_frames.csv",
            "REPORT.md",
        ):
            z.write(out_dir / name, name)
    with zipfile.ZipFile(review) as z:
        if z.testzip() is not None:
            raise RuntimeError("M05 review ZIP CRC verification failed")

    print(f"[M05-CANARY] status={summary['status']}", flush=True)
    print(f"[M05-CANARY] report={out_dir / 'REPORT.md'}", flush=True)
    print(f"[M05-CANARY] review={review}", flush=True)
    return 0 if invariants_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
