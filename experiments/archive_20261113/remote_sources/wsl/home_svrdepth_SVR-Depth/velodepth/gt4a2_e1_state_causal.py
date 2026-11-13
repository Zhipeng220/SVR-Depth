#!/usr/bin/env python3
"""GT4-A2 E1: single persistent-state intervention causal audit.

No training is performed. The script:
  1) caches natural A states at fixed, predeclared propagation ages;
  2) applies the same repair-head weights once to each identical snapshot;
  3) separates current readout from the committed future state;
  4) disables repair during future replay and uses identical inputs/refresh times;
  5) writes compact CSV/JSON evidence and a review ZIP.

Run from the repository root or pass absolute paths.
"""
from __future__ import annotations

import argparse
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
# h=31 intentionally lands on the next scheduled keyframe for offset-1 snapshots.
# h=30 is the final propagation frame before that refresh and is the main long-horizon read.
HORIZONS = (1, 2, 4, 8, 16, 30, 31)
MODES = {"C": "pre_readout", "D": "post_readout"}

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
    import csv
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="E1 single-state-intervention causal audit; no training.")
    ap.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    ap.add_argument("--crowd2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv")
    ap.add_argument("--crowd3-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv")
    ap.add_argument("--balloon2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv")
    ap.add_argument("--c-checkpoint", default="outputs/gt4a2_c3/gt4a2_c3_C_final_step1000.pt")
    ap.add_argument("--d-checkpoint", default="outputs/gt4a2_c3/gt4a2_c3_D_final_step1000.pt")
    ap.add_argument("--checkpoint", choices=("C", "D", "both"), default="both")
    ap.add_argument("--max-snapshots-per-sequence", type=int, default=0,
                    help="0=all eligible cycles; otherwise evenly sample this many cycles per sequence.")
    ap.add_argument("--readout-tolerance", type=float, default=0.02,
                    help="Maximum absolute radius mismatch (meters) allowed when reconstructing Q(F) from cached features.")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--out-dir", default="outputs/gt4a2_e1_validation")
    return ap.parse_args()


def verify_source_hashes() -> dict:
    result = {}
    for rel, expected in EXPECTED_SOURCE_HASHES.items():
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Required audited source is missing: {path}")
        actual = sha256_file(path)
        result[rel] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            raise RuntimeError(
                f"Source changed since E0; refusing E1: {rel}: {actual} != {expected}"
            )
    return result


def selected_checkpoints(args) -> dict[str, Path]:
    paths = {"C": Path(args.c_checkpoint).resolve(), "D": Path(args.d_checkpoint).resolve()}
    selected = ("C", "D") if args.checkpoint == "both" else (args.checkpoint,)
    return {k: paths[k] for k in selected}


def load_checkpoint_metadata(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Repair checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    expected_mode = MODES[label]
    if ckpt.get("mode_label") != label or ckpt.get("mode") != expected_mode:
        raise RuntimeError(
            f"{label} checkpoint metadata mismatch: mode_label={ckpt.get('mode_label')}, "
            f"mode={ckpt.get('mode')}, expected={expected_mode}"
        )
    if int(ckpt.get("step_count", -1)) != 1000:
        raise RuntimeError(f"{label} checkpoint is not the final 1000-step C3 checkpoint.")
    if "repair_head_state_dict" not in ckpt:
        raise RuntimeError(f"{label} checkpoint has no repair_head_state_dict.")
    meta = {
        "label": label,
        "path": str(path),
        "sha256": sha256_file(path),
        "mode": expected_mode,
        "step_count": int(ckpt["step_count"]),
        "repair_head_state_keys": len(ckpt["repair_head_state_dict"]),
    }
    del ckpt
    return meta


def build_snapshots(rows: list[dict], max_snapshots: int) -> list[dict]:
    """Snapshot after frame cycle*32+1, i.e. one natural propagation write after a keyframe."""
    candidates = []
    cycle = 0
    while True:
        state_index = cycle * REFRESH_INTERVAL + 1
        if state_index + max(HORIZONS) >= len(rows):
            break
        candidates.append({
            "cycle": cycle,
            "snapshot_index": state_index,
            # exp1.precompute_a_states caches state at anchor_index-1.
            "anchor_index": state_index + 1,
        })
        cycle += 1
    if not candidates:
        raise RuntimeError(f"No eligible snapshots in sequence with {len(rows)} frames.")
    if max_snapshots > 0 and len(candidates) > max_snapshots:
        indices = np.linspace(0, len(candidates) - 1, num=max_snapshots, dtype=int)
        candidates = [candidates[int(i)] for i in sorted(set(indices.tolist()))]
    return candidates


def decode_radius_from_state_features(model, features: list[torch.Tensor]):
    """Reproduce RadialModule.forward_fast's readout from its pre-readout feature state."""
    radial = model.pixel_decoder.radial_module
    lateral_features = [features[0]] + [
        radial.process_features[i](features[i + 1]) for i in range(len(radial.ups))
    ]
    latents = lateral_features[0]
    features_student = []
    for i, up in enumerate(radial.ups):
        latents = latents + lateral_features[i + 1]
        latents = up(latents)
        features_student.append(latents)
    logdepth = radial.depth_proj(features_student)
    # Exact behavior in audited decoder_velo.py: logconf_student = logdepth_student.
    radius = torch.exp(logdepth.clip(min=-8.0, max=8.0) + 2.0)
    confidence = torch.exp(logdepth.clip(min=-8.0, max=10.0))
    return radius, confidence


def max_abs_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().item())


def feature_delta_summary(features_a, features_prime) -> dict:
    sum_delta_sq = 0.0
    sum_a_sq = 0.0
    max_abs = 0.0
    for a, b in zip(features_a, features_prime):
        delta = b.detach().float() - a.detach().float()
        af = a.detach().float()
        max_abs = max(max_abs, float(delta.abs().max().item()))
        sum_delta_sq += float(delta.square().sum().item())
        sum_a_sq += float(af.square().sum().item())
    return {
        "feature_delta_max_abs": max_abs,
        "feature_delta_rel_l2": float(np.sqrt(sum_delta_sq) / max(np.sqrt(sum_a_sq), 1e-12)),
    }


def state_to_device(cache_state: dict, device: torch.device) -> dict:
    return {
        "radius": cache_state["radius"].to(device).detach().clone(),
        "confidence": cache_state["confidence"].to(device).detach().clone(),
        "features": [x.to(device).detach().clone() for x in cache_state["features"]],
        "rays_map": cache_state["rays_map"].to(device).detach().clone(),
    }


def pred_numpy(radius, rays_map, shape, model) -> np.ndarray:
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        pred = gt2.make_z_depth(
            radius, rays_map, shape["padded_shape"], shape["paddings"],
            model.interpolation_mode
        )
    return pred.detach().float().cpu().numpy()


def score_prediction(frame: dict, pred_np: np.ndarray) -> dict:
    if frame.get("gt_np") is None:
        return {
            "gt_matched": False, "AbsRel": "", "RMSE": "", "delta1": "",
            "valid_ratio": "", "gt_valid_ratio": "", "valid_pixels": 0,
            "total_pixels": int(np.prod(pred_np.shape[-2:])),
        }
    return {"gt_matched": True, **gt2.calculate_metrics(frame["gt_np"], pred_np)}


def make_window_frames(rows, state_index, shape, device):
    frames = {}
    for idx in range(state_index, state_index + max(HORIZONS) + 1):
        row = rows[idx]
        matched = str(row.get("gt_match_status", "")).strip().upper() == "MATCHED"
        frames[idx] = exp1.load_frame(row, shape, device, need_gt=matched)
    return frames


def precompute_window_static(model, frames, state_index, shape):
    """Compute each transition once; every branch reuses the exact same flow tensors."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    static_by_index = {}
    with torch.no_grad():
        for idx in range(state_index + 1, state_index + max(HORIZONS) + 1):
            if idx % REFRESH_INTERVAL == 0:
                continue
            static_by_index[idx] = exp1.transition_static(
                model, frames[idx - 1], frames[idx], shape, dis
            )
    return static_by_index


def replay_future(model, rows, frames, static_by_index, shape, state_index,
                  initial_state: dict, sequence: str, checkpoint_label: str,
                  snapshot_cycle: int, branch: str) -> tuple[list[dict], dict[int, np.ndarray]]:
    radius = initial_state["radius"].detach().clone()
    confidence = initial_state["confidence"].detach().clone()
    features = [x.detach().clone() for x in initial_state["features"]]
    rays_map = initial_state["rays_map"].detach().clone()
    prev_frame = frames[state_index]
    rows_out = []
    predictions = {}

    with torch.no_grad():
        for h in range(1, max(HORIZONS) + 1):
            idx = state_index + h
            frame = frames[idx]
            is_keyframe = (idx % REFRESH_INTERVAL == 0)
            if is_keyframe:
                rays_map, radius, confidence, features = exp1.keyframe_state(model, frame, shape)
            else:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    radius, confidence, features = exp1.one_fast_step(
                        model, prev_frame, frame, static_by_index[idx],
                        radius, confidence, features
                    )
            if h in HORIZONS:
                pred_np = pred_numpy(radius, rays_map, shape, model)
                predictions[h] = pred_np
                metrics = score_prediction(frame, pred_np)
                rows_out.append({
                    "sequence": sequence,
                    "checkpoint": checkpoint_label,
                    "snapshot_cycle": snapshot_cycle,
                    "snapshot_index": state_index,
                    "branch": branch,
                    "horizon": h,
                    "target_index": idx,
                    "refresh_frame": is_keyframe,
                    **metrics,
                })
            prev_frame = frame
    return rows_out, predictions


def aggregate_future_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("gt_matched") and row.get("AbsRel") != "":
            key = (row["sequence"], row["checkpoint"], row["branch"], int(row["horizon"]))
            groups[key].append(row)
    summary = []
    for (seq, ckpt, branch, horizon), rr in sorted(groups.items()):
        summary.append({
            "sequence": seq, "checkpoint": ckpt, "branch": branch, "horizon": horizon,
            "n_snapshot_target_pairs": len(rr),
            "mean_AbsRel": float(np.mean([float(x["AbsRel"]) for x in rr])),
            "median_AbsRel": float(np.median([float(x["AbsRel"]) for x in rr])),
            "mean_RMSE": float(np.mean([float(x["RMSE"]) for x in rr])),
            "mean_delta1": float(np.mean([float(x["delta1"]) for x in rr])),
            "note": "Overlapping snapshots are correlated; descriptive mechanism diagnostics only.",
        })
    paired = []
    branch_names = ("O_output_only", "C_consistent", "D_current", "D_consistent")
    keys = sorted({(r["sequence"], r["checkpoint"], int(r["snapshot_cycle"]), int(r["horizon"]))
                   for r in rows if r.get("gt_matched") and r.get("AbsRel") != ""})
    for key in keys:
        # Branch is deliberately excluded from this key so each effect is paired to A
        # at the same snapshot and future horizon.
        seq, ckpt, cycle, horizon = key
        rr = [r for r in rows if r["sequence"] == seq and r["checkpoint"] == ckpt
              and int(r["snapshot_cycle"]) == cycle and int(r["horizon"]) == horizon
              and r.get("gt_matched") and r.get("AbsRel") != ""]
        by_branch = {r["branch"]: r for r in rr}
        if "A" not in by_branch:
            continue
        base = by_branch["A"]
        for branch in branch_names:
            if branch not in by_branch:
                continue
            cur = by_branch[branch]
            a_abs = float(base["AbsRel"])
            c_abs = float(cur["AbsRel"])
            paired.append({
                "sequence": seq, "checkpoint": ckpt, "snapshot_cycle": cycle,
                "horizon": horizon, "branch": branch,
                "A_AbsRel": a_abs, "branch_AbsRel": c_abs,
                "delta_AbsRel": c_abs - a_abs,
                "relative_AbsRel_reduction_pct": (a_abs - c_abs) / max(a_abs, 1e-12) * 100.0,
                "delta_RMSE": float(cur["RMSE"]) - float(base["RMSE"]),
                "delta_delta1": float(cur["delta1"]) - float(base["delta1"]),
            })
    return summary, paired


def build_parser_preflight(args):
    if not args.snapshot:
        raise RuntimeError("VDEEP_SNAPSHOT is unset; pass --snapshot or export VDEEP_SNAPSHOT.")
    snapshot = Path(args.snapshot).resolve()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pretrained snapshot incomplete: {snapshot}")
    source_hashes = verify_source_hashes()
    pair_paths = {
        "crowd2": Path(args.crowd2_pair_csv).resolve(),
        "crowd3": Path(args.crowd3_pair_csv).resolve(),
        "balloon2": Path(args.balloon2_pair_csv).resolve(),
    }
    manifests = {}
    for name, path in pair_paths.items():
        rows = exp1.read_manifest(path, training=False)
        snapshots = build_snapshots(rows, args.max_snapshots_per_sequence)
        manifests[name] = {
            "path": str(path), "sha256": sha256_file(path),
            "rgb_rows": len(rows), "eligible_snapshots": len(snapshots),
            "snapshot_indices": [x["snapshot_index"] for x in snapshots],
        }
    ckpts = {}
    for label, path in selected_checkpoints(args).items():
        ckpts[label] = load_checkpoint_metadata(path, label)
    return {
        "status": "PREFLIGHT_PASS",
        "root": str(ROOT),
        "snapshot": str(snapshot),
        "pretrained_weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "source_hashes": source_hashes,
        "manifests": manifests,
        "checkpoints": ckpts,
        "protocol": {
            "refresh_interval": REFRESH_INTERVAL,
            "snapshots": "A trajectory, state immediately after propagation at cycle_start+1",
            "one_time_intervention": True,
            "repair_disabled_during_future_replay": True,
            "horizons": list(HORIZONS),
            "h30": "last propagation frame before the scheduled keyframe",
            "h31": "scheduled keyframe reset control; do not interpret as persistent repair gain",
            "branches": ["A", "O_output_only", "C_consistent", "D_current", "D_consistent"],
        },
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "e1_summary.json").exists() and not args.preflight_only:
        raise RuntimeError(f"Refusing to overwrite existing result: {out_dir / 'e1_summary.json'}")

    pre = build_parser_preflight(args)
    write_json(out_dir / "e1_preflight.json", pre)
    print(f"[PASS] E1 preflight: source hashes={len(pre['source_hashes'])}; "
          f"manifests={len(pre['manifests'])}; checkpoints={list(pre['checkpoints'])}", flush=True)
    if args.preflight_only:
        print("E1 preflight complete; no model inference or training was run.", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("E1 inference requires CUDA; no training is performed.")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    np.random.seed(20261024)
    torch.cuda.manual_seed_all(20261024)

    pair_paths = {
        "crowd2": Path(args.crowd2_pair_csv).resolve(),
        "crowd3": Path(args.crowd3_pair_csv).resolve(),
        "balloon2": Path(args.balloon2_pair_csv).resolve(),
    }
    checkpoint_paths = selected_checkpoints(args)
    current_rows, future_rows, state_rows = [], [], []
    invariant_rows = []
    start_all = time.perf_counter()

    for seq, pair_path in pair_paths.items():
        rows = exp1.read_manifest(pair_path, training=False)
        targets = build_snapshots(rows, args.max_snapshots_per_sequence)
        model_a = gt2.load_model(Path(args.snapshot).resolve(), "off", device)
        shape = exp1.shape_info_for(model_a, Path(rows[0]["rgb_path"]), device)
        print(f"[{seq}] caching A states: {len(rows)} frames, {len(targets)} snapshots", flush=True)
        anchors = [{"anchor_index": x["anchor_index"], "refresh_cycle": x["cycle"]}
                   for x in targets]
        state_cache = exp1.precompute_a_states(model_a, rows, anchors, shape, device)
        print(f"[{seq}] A state cache complete", flush=True)

        for checkpoint_label, checkpoint_path in checkpoint_paths.items():
            model_repair = gt2.load_model(Path(args.snapshot).resolve(), "pre_readout", device)
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            head = model_repair.pixel_decoder.persistent_feature_repair
            head.load_state_dict(ckpt["repair_head_state_dict"], strict=True)
            model_repair.eval()
            head.eval()
            # Initialize decoder runtime shapes on the repair model. The output is discarded.
            first_frame = exp1.load_frame(rows[0], shape, device, need_gt=False)
            with torch.no_grad():
                exp1.keyframe_state(model_repair, first_frame, shape)
            del first_frame

            for target in targets:
                cycle = int(target["cycle"])
                t = int(target["snapshot_index"])
                cache_key = int(target["anchor_index"])
                cached = state_cache[cache_key]
                base_state = state_to_device(cached, device)
                features_a = base_state["features"]
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    features_prime = [x.detach().clone() for x in head(features_a)]
                    radius_prime, confidence_prime = decode_radius_from_state_features(
                        model_repair, features_prime
                    )
                    radius_read_a, confidence_read_a = decode_radius_from_state_features(
                        model_repair, features_a
                    )

                q_reconstruction_diff = max_abs_tensor(radius_read_a, base_state["radius"])
                q_conf_reconstruction_diff = max_abs_tensor(confidence_read_a, base_state["confidence"])
                fdelta = feature_delta_summary(features_a, features_prime)
                rdelta = {
                    "radius_delta_max_abs": max_abs_tensor(radius_prime, base_state["radius"]),
                    "radius_delta_rel_l2": float(
                        (radius_prime.detach().float() - base_state["radius"].detach().float()).norm().item()
                        / max(base_state["radius"].detach().float().norm().item(), 1e-12)
                    ),
                }
                state_rows.append({
                    "sequence": seq, "checkpoint": checkpoint_label, "snapshot_cycle": cycle,
                    "snapshot_index": t, "snapshot_offset": t % REFRESH_INTERVAL,
                    "q_reconstruction_max_abs_m": q_reconstruction_diff,
                    "confidence_reconstruction_max_abs": q_conf_reconstruction_diff,
                    **fdelta, **rdelta,
                })
                if q_reconstruction_diff > args.readout_tolerance:
                    raise RuntimeError(
                        f"Q(F) reconstruction mismatch at {seq}/cycle{cycle}: "
                        f"{q_reconstruction_diff:.6g} m > tolerance {args.readout_tolerance}. "
                        "Do not interpret E1; check autocast/readout/source."
                    )

                # Distinguish displayed output from committed future state.
                # A: output Q(F), commit (F,r)
                # O: output Q(F'), commit exactly A state
                # C: output Q(F'), commit (F',r')
                # D: output Q(F), commit (F',r)
                # D-consistent: output Q(F), commit (F',r')
                current_outputs = {
                    "A": (base_state["radius"], base_state["confidence"]),
                    "O_output_only": (radius_prime, confidence_prime),
                    "C_consistent": (radius_prime, confidence_prime),
                    "D_current": (base_state["radius"], base_state["confidence"]),
                    "D_consistent": (base_state["radius"], base_state["confidence"]),
                }
                future_states = {
                    "A": base_state,
                    "O_output_only": base_state,
                    "C_consistent": {
                        "radius": radius_prime, "confidence": confidence_prime,
                        "features": features_prime, "rays_map": base_state["rays_map"],
                    },
                    "D_current": {
                        "radius": base_state["radius"], "confidence": base_state["confidence"],
                        "features": features_prime, "rays_map": base_state["rays_map"],
                    },
                    "D_consistent": {
                        "radius": radius_prime, "confidence": confidence_prime,
                        "features": features_prime, "rays_map": base_state["rays_map"],
                    },
                }

                # Current snapshot-frame readout metrics; state itself is not advanced here.
                frame_t = exp1.load_frame(
                    rows[t], shape, device,
                    need_gt=(str(rows[t].get("gt_match_status", "")).strip().upper() == "MATCHED")
                )
                for branch, (out_radius, _out_conf) in current_outputs.items():
                    pred = pred_numpy(out_radius, base_state["rays_map"], shape, model_a)
                    metric = score_prediction(frame_t, pred)
                    current_rows.append({
                        "sequence": seq, "checkpoint": checkpoint_label, "snapshot_cycle": cycle,
                        "snapshot_index": t, "snapshot_offset": t % REFRESH_INTERVAL,
                        "branch": branch, "output_source": "Q(F_prime)" if branch in
                        ("O_output_only", "C_consistent") else "Q(F)",
                        **metric,
                    })

                # Load the identical future RGB/GT window and compute each transition once.
                frames = make_window_frames(rows, t, shape, device)
                # Use the already-loaded current frame with GT for current scoring and replay.
                frames[t] = frame_t
                static_by_index = precompute_window_static(model_a, frames, t, shape)

                predictions_by_branch = {}
                rows_by_branch = {}
                for branch in ("A", "O_output_only", "C_consistent", "D_current", "D_consistent"):
                    rr, pp = replay_future(
                        model_a, rows, frames, static_by_index, shape, t,
                        future_states[branch], seq, checkpoint_label, cycle, branch
                    )
                    rows_by_branch[branch] = rr
                    predictions_by_branch[branch] = pp

                for branch, rr in rows_by_branch.items():
                    for row in rr:
                        h = int(row["horizon"])
                        pred = predictions_by_branch[branch][h]
                        pred_a = predictions_by_branch["A"][h]
                        pred_c = predictions_by_branch["C_consistent"][h]
                        row["pred_max_abs_diff_vs_A_m"] = float(
                            np.max(np.abs(pred.astype(np.float32) - pred_a.astype(np.float32)))
                        )
                        row["pred_max_abs_diff_vs_C_m"] = float(
                            np.max(np.abs(pred.astype(np.float32) - pred_c.astype(np.float32)))
                        )
                        future_rows.append(row)

                # Mechanics checks: output-only future == A; C-consistent future == D-consistent.
                for h in HORIZONS:
                    pa = predictions_by_branch["A"][h]
                    po = predictions_by_branch["O_output_only"][h]
                    pc = predictions_by_branch["C_consistent"][h]
                    pd = predictions_by_branch["D_consistent"][h]
                    da_o = float(np.max(np.abs(pa.astype(np.float32) - po.astype(np.float32))))
                    dc_dd = float(np.max(np.abs(pc.astype(np.float32) - pd.astype(np.float32))))
                    max_branch_diff = max(
                        float(np.max(np.abs(pa.astype(np.float32) -
                                            predictions_by_branch[b][h].astype(np.float32))))
                        for b in predictions_by_branch
                    )
                    invariant_rows.append({
                        "sequence": seq, "checkpoint": checkpoint_label, "snapshot_cycle": cycle,
                        "snapshot_index": t, "horizon": h,
                        "A_vs_O_max_abs_m": da_o,
                        "C_vs_D_consistent_max_abs_m": dc_dd,
                        "max_all_branches_vs_A_m": max_branch_diff,
                        "is_scheduled_keyframe": bool((t + h) % REFRESH_INTERVAL == 0),
                    })

                del frames, static_by_index, predictions_by_branch, rows_by_branch
                del features_prime, radius_prime, confidence_prime, radius_read_a, confidence_read_a
                del base_state, future_states, current_outputs, frame_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[{seq}/{checkpoint_label}] cycle={cycle} done", flush=True)

            del model_repair, ckpt
            torch.cuda.empty_cache()
        del model_a, state_cache
        torch.cuda.empty_cache()

    future_summary, paired_effects = aggregate_future_rows(future_rows)
    write_csv(out_dir / "e1_current_snapshot_metrics.csv", current_rows)
    write_csv(out_dir / "e1_future_horizon_metrics.csv", future_rows)
    write_csv(out_dir / "e1_future_horizon_summary.csv", future_summary)
    write_csv(out_dir / "e1_paired_effects.csv", paired_effects)
    write_csv(out_dir / "e1_state_diagnostics.csv", state_rows)
    write_csv(out_dir / "e1_branch_invariants.csv", invariant_rows)

    readout_max = max((float(r["q_reconstruction_max_abs_m"]) for r in state_rows), default=0.0)
    ao_max = max((float(r["A_vs_O_max_abs_m"]) for r in invariant_rows), default=0.0)
    cdc_max = max((float(r["C_vs_D_consistent_max_abs_m"]) for r in invariant_rows), default=0.0)
    refresh_rows = [r for r in invariant_rows if r["is_scheduled_keyframe"]]
    refresh_max = max((float(r["max_all_branches_vs_A_m"]) for r in refresh_rows), default=0.0)
    invariants_pass = (
        readout_max <= args.readout_tolerance and ao_max <= 1e-3 and
        cdc_max <= 1e-3 and refresh_max <= 1e-3
    )
    summary = {
        "status": "MECHANICS_PASS" if invariants_pass else "MECHANICS_FAIL_DO_NOT_INTERPRET",
        "training_performed": False,
        "runtime_sec": time.perf_counter() - start_all,
        "source_hashes": pre["source_hashes"],
        "pretrained_weights_sha256": pre["pretrained_weights_sha256"],
        "checkpoint_metadata": pre["checkpoints"],
        "protocol": pre["protocol"],
        "counts": {
            "current_snapshot_rows": len(current_rows),
            "future_horizon_rows": len(future_rows),
            "state_diagnostic_rows": len(state_rows),
            "branch_invariant_rows": len(invariant_rows),
        },
        "mechanics": {
            "max_QF_reconstruction_abs_error_m": readout_max,
            "max_future_A_vs_O_abs_error_m": ao_max,
            "max_future_C_vs_D_consistent_abs_error_m": cdc_max,
            "max_all_branch_difference_at_scheduled_refresh_m": refresh_max,
            "thresholds": {
                "QF_reconstruction_m": args.readout_tolerance,
                "trajectory_identity_m": 1e-3,
                "scheduled_refresh_identity_m": 1e-3,
            },
        },
        "interpretation_limits": [
            "This is a one-time state intervention on A-trajectory snapshots; it does not train a model.",
            "The fixed-refresh h=31 point is the next scheduled keyframe reset, so h=30 is the final pre-refresh propagation point.",
            "Repeated snapshot windows overlap and are correlated; summary rows are descriptive diagnostics, not independent-sample significance tests.",
            "Crowd2, Crowd3 and Balloon2 are development sequences, not an unbiased final test set.",
            "Confidence is not an independent head in this source version: logconf_student is assigned logdepth_student.",
            "No latency or compute-saving claim is established by this experiment.",
        ],
    }
    write_json(out_dir / "e1_summary.json", summary)
    report = f"""# E1 单次持久状态干预结果

状态：**{summary['status']}**

- 是否训练：否
- 源码哈希：与 E0 锁定版本核对
- Q(F) 重读最大绝对误差：{readout_max:.8g} m
- A 与 output-only 的未来轨迹最大差：{ao_max:.8g} m
- C-consistent 与 D-consistent 的未来轨迹最大差：{cdc_max:.8g} m
- 预定关键帧刷新时所有分支相对 A 的最大差：{refresh_max:.8g} m

## 解释规则

1. `A` 与 `O_output_only` 的未来状态相同；两者未来轨迹必须一致。它们在快照帧的输出可不同。
2. `C_consistent` 与 `D_consistent` 的提交状态均为 `(F', r')`；未来轨迹必须一致。它们的快照帧输出不同。
3. `D_current` 提交 `(F', r)`；与 `D_consistent` 的未来差异检验保留原 radius 是否重要。
4. `h=30` 是下次刷新前最后一个传播帧；`h=31` 是固定调度的关键帧刷新控制点。
5. 若状态重读或轨迹恒等性不变量失败，先修复接线，不能解释指标优劣。
6. 该实验只诊断单次干预对自然 A 状态的影响；它不是重复修复实验，也不证明跨场景泛化或节省计算。

详细结果见 `e1_current_snapshot_metrics.csv`、`e1_future_horizon_metrics.csv`、
`e1_future_horizon_summary.csv`、`e1_paired_effects.csv`、
`e1_state_diagnostics.csv` 和 `e1_branch_invariants.csv`。
"""
    (out_dir / "E1_README.md").write_text(report, encoding="utf-8")

    review_zip = out_dir / "e1_review.zip"
    include = [
        "e1_preflight.json", "e1_current_snapshot_metrics.csv", "e1_future_horizon_metrics.csv",
        "e1_future_horizon_summary.csv", "e1_paired_effects.csv", "e1_state_diagnostics.csv",
        "e1_branch_invariants.csv", "e1_summary.json", "E1_README.md",
    ]
    script_path = Path(__file__).resolve()
    with zipfile.ZipFile(review_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            p = out_dir / rel
            if p.is_file():
                zf.write(p, rel)
        if script_path.is_file():
            zf.write(script_path, "gt4a2_e1_state_causal.py")
    print(f"[{summary['status']}] E1 review package: {review_zip}", flush=True)
    return 0 if invariants_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise
