#!/usr/bin/env python3
"""SVR-Depth Y02: full-cycle single-intervention causal experiment.

No training is performed. For natural A-trajectory snapshots at ages 8, 16 and 24:
  1) fork identical full states into A, output-only and F-only branches;
  2) apply the selected C/D repair head once;
  3) disable all repair during future replay;
  4) replay every propagation frame until the next scheduled keyframe refresh;
  5) record per-frame paired effects, future cumulative gain, damage risk and invariants.

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
SNAPSHOT_AGES = (8, 16, 24)
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
    ap = argparse.ArgumentParser(description="Y02 full-cycle single-intervention causal experiment; no training.")
    ap.add_argument("--snapshot", default=os.environ.get("VDEEP_SNAPSHOT", ""))
    ap.add_argument("--crowd2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd2_pairs_used.csv")
    ap.add_argument("--crowd3-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_crowd3_pairs_used.csv")
    ap.add_argument("--balloon2-pair-csv", default="outputs/gt4a2_exp1/gt4a2_exp1_balloon2_pairs_used.csv")
    ap.add_argument("--c-checkpoint", default="outputs/gt4a2_c3/gt4a2_c3_C_final_step1000.pt")
    ap.add_argument("--d-checkpoint", default="outputs/gt4a2_c3/gt4a2_c3_D_final_step1000.pt")
    ap.add_argument("--checkpoint", choices=("C", "D", "both"), default="both")
    ap.add_argument("--sequences", default="crowd2,crowd3,balloon2",
                    help="Comma-separated subset of crowd2,crowd3,balloon2.")
    ap.add_argument("--max-cycles-per-sequence", type=int, default=0,
                    help="0=all eligible cycles; otherwise evenly sample cycles, keeping ages 8/16/24 in each selected cycle.")
    ap.add_argument("--readout-tolerance", type=float, default=0.02,
                    help="Maximum absolute radius mismatch (meters) allowed when reconstructing Q(F) from cached features.")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--smoke-only", action="store_true",
                    help="Run a one-cycle mechanism smoke if max-cycles-per-sequence is not explicitly limited.")
    ap.add_argument("--out-dir", default="outputs/gt4a2_y02_validation")
    args = ap.parse_args()
    if args.max_cycles_per_sequence < 0:
        ap.error("--max-cycles-per-sequence must be >= 0")
    valid_sequences = {"crowd2", "crowd3", "balloon2"}
    seqs = [x.strip().lower() for x in args.sequences.split(",") if x.strip()]
    if not seqs or any(x not in valid_sequences for x in seqs) or len(set(seqs)) != len(seqs):
        ap.error("--sequences must be a unique comma-separated subset of crowd2,crowd3,balloon2")
    args.sequences_list = seqs
    if args.smoke_only and args.max_cycles_per_sequence == 0:
        args.max_cycles_per_sequence = 1
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
            raise RuntimeError(
                f"Source changed since E0; refusing Y02: {rel}: {actual} != {expected}"
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


def build_snapshots(rows: list[dict], max_cycles: int) -> list[dict]:
    """Build natural A-state snapshots at offsets 8/16/24 with a complete future cycle.

    A candidate is eligible only when the next scheduled keyframe row exists, so
    every selected snapshot has all propagation frames through refresh_index-1 and
    a real refresh frame for the reset invariant.
    """
    candidates = []
    cycle = 0
    while True:
        refresh_index = (cycle + 1) * REFRESH_INTERVAL
        if refresh_index >= len(rows):
            break
        cycle_start = cycle * REFRESH_INTERVAL
        for age in SNAPSHOT_AGES:
            state_index = cycle_start + age
            if state_index >= refresh_index:
                continue
            candidates.append({
                "cycle": cycle,
                "snapshot_age": age,
                "snapshot_index": state_index,
                # precompute_a_states caches state at anchor_index-1.
                "anchor_index": state_index + 1,
                "refresh_index": refresh_index,
                "future_propagation_frames": refresh_index - state_index - 1,
            })
        cycle += 1
    if not candidates:
        raise RuntimeError(f"No eligible Y02 snapshots in sequence with {len(rows)} frames.")
    if max_cycles > 0:
        cycle_ids = sorted({int(x["cycle"]) for x in candidates})
        if len(cycle_ids) > max_cycles:
            chosen_idx = np.linspace(0, len(cycle_ids) - 1, num=max_cycles, dtype=int)
            chosen_cycles = {cycle_ids[int(i)] for i in sorted(set(chosen_idx.tolist()))}
            candidates = [x for x in candidates if int(x["cycle"]) in chosen_cycles]
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


def make_window_frames(rows, state_index, refresh_index, shape, device):
    """Load snapshot, every future propagation frame, and the next keyframe reset frame."""
    frames = {}
    for idx in range(state_index, refresh_index + 1):
        row = rows[idx]
        matched = str(row.get("gt_match_status", "")).strip().upper() == "MATCHED"
        frames[idx] = exp1.load_frame(row, shape, device, need_gt=matched)
    return frames



def precompute_window_static(model, frames, state_index, refresh_index, shape):
    """Compute each transition once; all branches reuse the exact same flow tensors."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    static_by_index = {}
    with torch.no_grad():
        for idx in range(state_index + 1, refresh_index + 1):
            if idx % REFRESH_INTERVAL == 0:
                continue
            static_by_index[idx] = exp1.transition_static(
                model, frames[idx - 1], frames[idx], shape, dis
            )
    return static_by_index



def replay_future(model, frames, static_by_index, shape, state_index, refresh_index,
                  initial_state: dict, sequence: str, checkpoint_label: str,
                  snapshot_cycle: int, snapshot_age: int, branch: str):
    """Replay every future propagation frame and the next keyframe reset; no repair is applied."""
    radius = initial_state["radius"].detach().clone()
    confidence = initial_state["confidence"].detach().clone()
    features = [x.detach().clone() for x in initial_state["features"]]
    rays_map = initial_state["rays_map"].detach().clone()
    prev_frame = frames[state_index]
    rows_out = []
    predictions = {}
    with torch.no_grad():
        for h in range(1, refresh_index - state_index + 1):
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
            pred_np = pred_numpy(radius, rays_map, shape, model)
            predictions[h] = pred_np
            metrics = score_prediction(frame, pred_np)
            rows_out.append({
                "sequence": sequence,
                "checkpoint": checkpoint_label,
                "snapshot_cycle": snapshot_cycle,
                "snapshot_age": snapshot_age,
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
    """Aggregate every-frame future metrics and exact same-snapshot paired effects."""
    groups = defaultdict(list)
    for row in rows:
        if (not row.get("refresh_frame") and row.get("gt_matched")
                and row.get("AbsRel") != ""):
            key = (row["sequence"], row["checkpoint"], int(row["snapshot_age"]),
                   row["branch"], int(row["horizon"]))
            groups[key].append(row)
    summary = []
    for (seq, ckpt, age, branch, horizon), rr in sorted(groups.items()):
        summary.append({
            "sequence": seq, "checkpoint": ckpt, "snapshot_age": age,
            "branch": branch, "horizon": horizon,
            "n_snapshot_target_pairs": len(rr),
            "mean_AbsRel": float(np.mean([float(x["AbsRel"]) for x in rr])),
            "median_AbsRel": float(np.median([float(x["AbsRel"]) for x in rr])),
            "mean_RMSE": float(np.mean([float(x["RMSE"]) for x in rr])),
            "mean_delta1": float(np.mean([float(x["delta1"]) for x in rr])),
            "note": "Overlapping snapshots/frames are correlated; descriptive mechanism diagnostics only.",
        })
    paired = []
    branch_names = ("O_output_only", "F_only")
    keys = sorted({
        (r["sequence"], r["checkpoint"], int(r["snapshot_cycle"]),
         int(r["snapshot_age"]), int(r["horizon"]))
        for r in rows if (not r.get("refresh_frame") and r.get("gt_matched")
                          and r.get("AbsRel") != "")
    })
    row_lookup = {
        (r["sequence"], r["checkpoint"], int(r["snapshot_cycle"]),
         int(r["snapshot_age"]), int(r["horizon"]), r["branch"]): r
        for r in rows if not r.get("refresh_frame")
    }
    for seq, ckpt, cycle, age, horizon in keys:
        base = row_lookup.get((seq, ckpt, cycle, age, horizon, "A"))
        if base is None:
            continue
        for branch in branch_names:
            cur = row_lookup.get((seq, ckpt, cycle, age, horizon, branch))
            if cur is None or not cur.get("gt_matched") or cur.get("AbsRel") == "":
                continue
            a_abs = float(base["AbsRel"])
            c_abs = float(cur["AbsRel"])
            paired.append({
                "sequence": seq, "checkpoint": ckpt, "snapshot_cycle": cycle,
                "snapshot_age": age, "horizon": horizon,
                "target_index": int(cur["target_index"]), "branch": branch,
                "A_AbsRel": a_abs, "branch_AbsRel": c_abs,
                "delta_AbsRel_branch_minus_A": c_abs - a_abs,
                "gain_AbsRel_A_minus_branch": a_abs - c_abs,
                "relative_AbsRel_reduction_pct": (a_abs - c_abs) / max(a_abs, 1e-12) * 100.0,
                "delta_RMSE": float(cur["RMSE"]) - float(base["RMSE"]),
                "delta_delta1": float(cur["delta1"]) - float(base["delta1"]),
            })
    return summary, paired


def summarize_snapshot_gains(paired_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Report summed future gain and per-frame gain for each snapshot, then by age."""
    per_snapshot = defaultdict(list)
    for row in paired_rows:
        key = (row["sequence"], row["checkpoint"], int(row["snapshot_cycle"]),
               int(row["snapshot_age"]), row["branch"])
        per_snapshot[key].append(row)
    snapshot_summary = []
    for key, rr in sorted(per_snapshot.items()):
        seq, ckpt, cycle, age, branch = key
        gains = np.asarray([float(x["gain_AbsRel_A_minus_branch"]) for x in rr], dtype=float)
        harms = np.maximum(-gains, 0.0)
        snapshot_summary.append({
            "sequence": seq, "checkpoint": ckpt, "snapshot_cycle": cycle,
            "snapshot_age": age, "branch": branch,
            "n_future_gt_frames": int(len(gains)),
            "future_cumulative_gain_sum_AbsRel": float(gains.sum()),
            "future_mean_gain_per_frame_AbsRel": float(gains.mean()) if len(gains) else "",
            "positive_gain_frame_fraction": float(np.mean(gains > 0)) if len(gains) else "",
            "harmed_frame_fraction": float(np.mean(gains < 0)) if len(gains) else "",
            "p95_harm_AbsRel": float(np.percentile(harms, 95)) if len(harms) else "",
            "worst_frame_gain_AbsRel": float(gains.min()) if len(gains) else "",
            "note": "Sum is over matched future propagation frames until refresh; overlapping snapshots are correlated.",
        })
    age_groups = defaultdict(list)
    for row in snapshot_summary:
        key = (row["sequence"], row["checkpoint"], int(row["snapshot_age"]), row["branch"])
        age_groups[key].append(row)
    age_summary = []
    for key, rr in sorted(age_groups.items()):
        seq, ckpt, age, branch = key
        cum = np.asarray([float(x["future_cumulative_gain_sum_AbsRel"]) for x in rr], dtype=float)
        mean_gain = np.asarray([float(x["future_mean_gain_per_frame_AbsRel"]) for x in rr], dtype=float)
        age_summary.append({
            "sequence": seq, "checkpoint": ckpt, "snapshot_age": age, "branch": branch,
            "n_snapshots": len(rr),
            "mean_future_cumulative_gain_sum_AbsRel": float(cum.mean()),
            "median_future_cumulative_gain_sum_AbsRel": float(np.median(cum)),
            "positive_cumulative_gain_snapshot_fraction": float(np.mean(cum > 0)),
            "mean_per_frame_gain_AbsRel": float(mean_gain.mean()),
            "mean_harmed_frame_fraction": float(np.mean([float(x["harmed_frame_fraction"]) for x in rr])),
            "p95_snapshot_harm_AbsRel": float(np.percentile(
                [float(x["p95_harm_AbsRel"]) for x in rr], 95
            )),
            "note": "Descriptive across overlapping snapshots; not an independent-sample significance test.",
        })
    return snapshot_summary, age_summary



def build_parser_preflight(args):
    if not args.snapshot:
        raise RuntimeError("VDEEP_SNAPSHOT is unset; pass --snapshot or export VDEEP_SNAPSHOT.")
    snapshot = Path(args.snapshot).resolve()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Pretrained snapshot incomplete: {snapshot}")
    source_hashes = verify_source_hashes()
    all_pair_paths = {
        "crowd2": Path(args.crowd2_pair_csv).resolve(),
        "crowd3": Path(args.crowd3_pair_csv).resolve(),
        "balloon2": Path(args.balloon2_pair_csv).resolve(),
    }
    pair_paths = {name: all_pair_paths[name] for name in args.sequences_list}
    manifests = {}
    for name, path in pair_paths.items():
        rows = exp1.read_manifest(path, training=False)
        snapshots = build_snapshots(rows, args.max_cycles_per_sequence)
        manifests[name] = {
            "path": str(path), "sha256": sha256_file(path),
            "rgb_rows": len(rows), "eligible_snapshots": len(snapshots),
            "eligible_cycles": sorted({int(x["cycle"]) for x in snapshots}),
            "snapshot_age_counts": {
                str(age): sum(int(x["snapshot_age"]) == age for x in snapshots)
                for age in SNAPSHOT_AGES
            },
            "snapshot_indices": [x["snapshot_index"] for x in snapshots],
        }
    ckpts = {}
    for label, path in selected_checkpoints(args).items():
        ckpts[label] = load_checkpoint_metadata(path, label)
    return {
        "status": "PREFLIGHT_PASS",
        "experiment": "Y02 full-cycle single-intervention causal experiment",
        "root": str(ROOT),
        "snapshot": str(snapshot),
        "pretrained_weights_sha256": sha256_file(snapshot / "model.safetensors"),
        "source_hashes": source_hashes,
        "manifests": manifests,
        "checkpoints": ckpts,
        "protocol": {
            "refresh_interval": REFRESH_INTERVAL,
            "snapshot_ages": list(SNAPSHOT_AGES),
            "snapshots": "Natural A-trajectory state immediately after propagation at cycle_start+age",
            "branches": {
                "A": "native output Q(F); future state (F,r,confidence)",
                "O_output_only": "current output Q(F_prime); future state exactly A",
                "F_only": "current output Q(F); commit F_prime only, retain original radius/confidence",
            },
            "one_time_intervention": True,
            "repair_disabled_during_future_replay": True,
            "record_every_propagation_frame_until_refresh": True,
            "refresh_frame": "recorded separately as reset invariant; excluded from future cumulative gain",
            "training_performed": False,
        },
    }



def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "y02_summary.json"
    if summary_path.exists() and not args.preflight_only:
        raise RuntimeError(f"Refusing to overwrite existing result: {summary_path}")

    pre = build_parser_preflight(args)
    write_json(out_dir / "y02_preflight.json", pre)
    print(f"[PASS] Y02 preflight: sequences={args.sequences_list}, ages={SNAPSHOT_AGES}, "
          f"checkpoints={list(pre['checkpoints'])}", flush=True)
    if args.preflight_only:
        print("Y02 preflight complete; no model inference or training was run.", flush=True)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("Y02 inference requires CUDA; no training is performed.")

    device = torch.device("cuda")
    torch.set_num_threads(4)
    torch.manual_seed(20261025)
    np.random.seed(20261025)
    torch.cuda.manual_seed_all(20261025)

    all_pair_paths = {
        "crowd2": Path(args.crowd2_pair_csv).resolve(),
        "crowd3": Path(args.crowd3_pair_csv).resolve(),
        "balloon2": Path(args.balloon2_pair_csv).resolve(),
    }
    pair_paths = {name: all_pair_paths[name] for name in args.sequences_list}
    checkpoint_paths = selected_checkpoints(args)

    current_rows = []
    future_rows = []
    refresh_rows = []
    state_rows = []
    invariant_rows = []
    start_all = time.perf_counter()

    for seq, pair_path in pair_paths.items():
        rows = exp1.read_manifest(pair_path, training=False)
        targets = build_snapshots(rows, args.max_cycles_per_sequence)
        model_a = gt2.load_model(Path(args.snapshot).resolve(), "off", device)
        shape = exp1.shape_info_for(model_a, Path(rows[0]["rgb_path"]), device)
        print(f"[{seq}] caching natural A states: frames={len(rows)}, snapshots={len(targets)}, "
              f"cycles={len({int(x['cycle']) for x in targets})}", flush=True)
        anchors = [{"anchor_index": x["anchor_index"], "refresh_cycle": x["cycle"]}
                   for x in targets]
        state_cache = exp1.precompute_a_states(model_a, rows, anchors, shape, device)
        print(f"[{seq}] A-state cache complete", flush=True)

        for checkpoint_label, checkpoint_path in checkpoint_paths.items():
            model_repair = gt2.load_model(Path(args.snapshot).resolve(), "pre_readout", device)
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            head = model_repair.pixel_decoder.persistent_feature_repair
            head.load_state_dict(ckpt["repair_head_state_dict"], strict=True)
            model_repair.eval()
            head.eval()

            # Initialize decoder runtime shapes; discard this output.
            first_frame = exp1.load_frame(rows[0], shape, device, need_gt=False)
            with torch.no_grad():
                exp1.keyframe_state(model_repair, first_frame, shape)
            del first_frame

            for target in targets:
                cycle = int(target["cycle"])
                age = int(target["snapshot_age"])
                t = int(target["snapshot_index"])
                refresh_index = int(target["refresh_index"])
                cache_key = int(target["anchor_index"])
                base_state = state_to_device(state_cache[cache_key], device)
                features_a = base_state["features"]

                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=True
                ):
                    features_prime = [x.detach().clone() for x in head(features_a)]
                    radius_prime, confidence_prime = decode_radius_from_state_features(
                        model_repair, features_prime
                    )
                    radius_read_a, confidence_read_a = decode_radius_from_state_features(
                        model_repair, features_a
                    )

                q_reconstruction_diff = max_abs_tensor(radius_read_a, base_state["radius"])
                q_conf_reconstruction_diff = max_abs_tensor(
                    confidence_read_a, base_state["confidence"]
                )
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
                    "snapshot_age": age, "snapshot_index": t, "refresh_index": refresh_index,
                    "q_reconstruction_max_abs_m": q_reconstruction_diff,
                    "confidence_reconstruction_max_abs": q_conf_reconstruction_diff,
                    **fdelta, **rdelta,
                })
                if q_reconstruction_diff > args.readout_tolerance:
                    raise RuntimeError(
                        f"Q(F) reconstruction mismatch at {seq}/cycle{cycle}/age{age}: "
                        f"{q_reconstruction_diff:.6g} m > tolerance {args.readout_tolerance}. "
                        "Do not interpret Y02; check autocast/readout/source."
                    )

                # A: output Q(F), commit original full state.
                # O: output Q(F'), commit exactly the original A state.
                # F-only: output Q(F), commit F' but retain original radius/confidence.
                current_outputs = {
                    "A": (base_state["radius"], base_state["confidence"]),
                    "O_output_only": (radius_prime, confidence_prime),
                    "F_only": (base_state["radius"], base_state["confidence"]),
                }
                future_states = {
                    "A": base_state,
                    "O_output_only": base_state,
                    "F_only": {
                        "radius": base_state["radius"],
                        "confidence": base_state["confidence"],
                        "features": features_prime,
                        "rays_map": base_state["rays_map"],
                    },
                }

                frames = make_window_frames(rows, t, refresh_index, shape, device)
                frame_t = frames[t]
                static_by_index = precompute_window_static(
                    model_a, frames, t, refresh_index, shape
                )

                # Score the intervention frame separately from future gains.
                current_pred_by_branch = {}
                for branch, (out_radius, _out_conf) in current_outputs.items():
                    pred = pred_numpy(out_radius, base_state["rays_map"], shape, model_a)
                    current_pred_by_branch[branch] = pred
                    metric = score_prediction(frame_t, pred)
                    current_rows.append({
                        "sequence": seq, "checkpoint": checkpoint_label,
                        "snapshot_cycle": cycle, "snapshot_age": age,
                        "snapshot_index": t, "refresh_index": refresh_index,
                        "branch": branch,
                        "output_source": "Q(F_prime)" if branch == "O_output_only" else "Q(F)",
                        **metric,
                    })
                current_af_diff = float(np.max(np.abs(
                    current_pred_by_branch["A"].astype(np.float32)
                    - current_pred_by_branch["F_only"].astype(np.float32)
                )))
                if current_af_diff > 1e-3:
                    raise RuntimeError(
                        f"F-only changed current output at {seq}/cycle{cycle}/age{age}: "
                        f"{current_af_diff} m > 0.001"
                    )

                predictions_by_branch = {}
                rows_by_branch = {}
                for branch in ("A", "O_output_only", "F_only"):
                    rr, pp = replay_future(
                        model_a, frames, static_by_index, shape, t, refresh_index,
                        future_states[branch], seq, checkpoint_label, cycle, age, branch
                    )
                    rows_by_branch[branch] = rr
                    predictions_by_branch[branch] = pp

                refresh_h = refresh_index - t
                for branch, rr in rows_by_branch.items():
                    for row in rr:
                        h = int(row["horizon"])
                        pred = predictions_by_branch[branch][h]
                        pred_a = predictions_by_branch["A"][h]
                        row["pred_max_abs_diff_vs_A_m"] = float(np.max(np.abs(
                            pred.astype(np.float32) - pred_a.astype(np.float32)
                        )))
                        if bool(row["refresh_frame"]):
                            refresh_rows.append(row)
                        else:
                            future_rows.append(row)

                # Branch invariants are checked on every future frame, including reset.
                for h in range(1, refresh_h + 1):
                    pa = predictions_by_branch["A"][h]
                    po = predictions_by_branch["O_output_only"][h]
                    pf = predictions_by_branch["F_only"][h]
                    da_o = float(np.max(np.abs(
                        pa.astype(np.float32) - po.astype(np.float32)
                    )))
                    da_f = float(np.max(np.abs(
                        pa.astype(np.float32) - pf.astype(np.float32)
                    )))
                    invariant_rows.append({
                        "sequence": seq, "checkpoint": checkpoint_label,
                        "snapshot_cycle": cycle, "snapshot_age": age,
                        "snapshot_index": t, "refresh_index": refresh_index,
                        "horizon": h, "target_index": t + h,
                        "A_vs_O_max_abs_m": da_o,
                        "A_vs_F_max_abs_m": da_f,
                        "is_scheduled_keyframe": bool(t + h == refresh_index),
                        "current_A_vs_F_max_abs_m": current_af_diff,
                    })

                del frames, static_by_index, predictions_by_branch, rows_by_branch
                del features_prime, radius_prime, confidence_prime, radius_read_a, confidence_read_a
                del base_state, future_states, current_outputs, current_pred_by_branch, frame_t
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[{seq}/{checkpoint_label}] cycle={cycle} age={age} done", flush=True)

            del model_repair, ckpt
            torch.cuda.empty_cache()
        del model_a, state_cache
        torch.cuda.empty_cache()

    future_summary, paired_effects = aggregate_future_rows(future_rows)
    snapshot_summary, age_summary = summarize_snapshot_gains(paired_effects)

    write_csv(out_dir / "y02_current_snapshot_metrics.csv", current_rows)
    write_csv(out_dir / "y02_future_frame_metrics.csv", future_rows)
    write_csv(out_dir / "y02_future_frame_summary.csv", future_summary)
    write_csv(out_dir / "y02_paired_future_effects.csv", paired_effects)
    write_csv(out_dir / "y02_snapshot_gain_summary.csv", snapshot_summary)
    write_csv(out_dir / "y02_age_gain_summary.csv", age_summary)
    write_csv(out_dir / "y02_state_diagnostics.csv", state_rows)
    write_csv(out_dir / "y02_refresh_reset_metrics.csv", refresh_rows)
    write_csv(out_dir / "y02_branch_invariants.csv", invariant_rows)

    readout_max = max((float(r["q_reconstruction_max_abs_m"]) for r in state_rows), default=0.0)
    ao_max = max((float(r["A_vs_O_max_abs_m"]) for r in invariant_rows), default=0.0)
    current_af_max = max((float(r["current_A_vs_F_max_abs_m"]) for r in invariant_rows), default=0.0)
    refresh_inv = [r for r in invariant_rows if r["is_scheduled_keyframe"]]
    refresh_max = max((max(float(r["A_vs_O_max_abs_m"]), float(r["A_vs_F_max_abs_m"]))
                       for r in refresh_inv), default=0.0)
    invariants_pass = (
        readout_max <= args.readout_tolerance
        and ao_max <= 1e-3
        and current_af_max <= 1e-3
        and refresh_max <= 1e-3
    )
    summary = {
        "status": "MECHANICS_PASS" if invariants_pass else "MECHANICS_FAIL_DO_NOT_INTERPRET",
        "experiment": "Y02 full-cycle single-intervention causal experiment",
        "training_performed": False,
        "is_smoke_or_partial": bool(args.smoke_only or args.max_cycles_per_sequence > 0),
        "runtime_sec": time.perf_counter() - start_all,
        "selected_sequences": args.sequences_list,
        "snapshot_ages": list(SNAPSHOT_AGES),
        "max_cycles_per_sequence": args.max_cycles_per_sequence,
        "source_hashes": pre["source_hashes"],
        "pretrained_weights_sha256": pre["pretrained_weights_sha256"],
        "checkpoint_metadata": pre["checkpoints"],
        "protocol": pre["protocol"],
        "counts": {
            "current_snapshot_rows": len(current_rows),
            "future_propagation_frame_rows": len(future_rows),
            "refresh_reset_rows": len(refresh_rows),
            "paired_future_effect_rows": len(paired_effects),
            "snapshot_gain_rows": len(snapshot_summary),
            "age_gain_summary_rows": len(age_summary),
            "state_diagnostic_rows": len(state_rows),
            "branch_invariant_rows": len(invariant_rows),
        },
        "mechanics": {
            "max_QF_reconstruction_abs_error_m": readout_max,
            "max_future_A_vs_O_abs_error_m": ao_max,
            "max_current_A_vs_F_abs_error_m": current_af_max,
            "max_branch_difference_at_scheduled_refresh_m": refresh_max,
            "all_invariants_pass": invariants_pass,
            "thresholds": {
                "QF_reconstruction_m": args.readout_tolerance,
                "trajectory_identity_m": 1e-3,
                "current_A_vs_F_m": 1e-3,
                "scheduled_refresh_identity_m": 1e-3,
            },
        },
        "interpretation_limits": [
            "Y02 is a single-intervention experiment on natural A-trajectory snapshots; no training is performed.",
            "Only propagation frames strictly before the next keyframe refresh contribute to future cumulative gain.",
            "The keyframe refresh is recorded separately and used only as a reset invariant.",
            "Output-only must leave the full future state untouched; its future trajectory must match A.",
            "F-only commits repaired features but retains the original radius/confidence and leaves the intervention-frame output unchanged.",
            "Snapshots at ages 8/16/24 within the same sequence overlap and are correlated; results are descriptive mechanism evidence, not independent-sample significance tests.",
            "Crowd2, Crowd3 and Balloon2 are development sequences, not an unbiased final test set.",
            "No latency or compute-saving claim is established by Y02.",
        ],
    }
    write_json(summary_path, summary)

    report = f"""# Y02：全周期单次干预实验

状态：**{summary['status']}**

- 是否训练：否
- 运行时间：{summary['runtime_sec']:.1f} 秒
- 序列：{', '.join(args.sequences_list)}
- 干预年龄：{', '.join(str(x) for x in SNAPSHOT_AGES)}
- 检查点：{', '.join(checkpoint_paths.keys())}
- 传播帧指标行：{len(future_rows)}
- 配对未来效应行：{len(paired_effects)}
- 未来 A vs output-only 最大差：{ao_max:.8g} m
- 干预帧 A vs F-only 最大差：{current_af_max:.8g} m
- 关键帧刷新时最大差：{refresh_max:.8g} m
- Q(F) 重读最大误差：{readout_max:.8g} m

## 分支语义

- A：输出 Q(F)，未来状态保持原生状态。
- O_output_only：当前输出 Q(F')，未来状态完整保留 A 状态。
- F_only：当前输出仍为 Q(F)，仅将 F' 写回未来特征状态，radius/confidence 保持原值。
- 每个检查点 C/D 分别运行；修复只在快照处应用一次，未来传播阶段完全关闭修复。
- 对每个快照记录 age 8/16/24 后的每一帧，直到下一次关键帧刷新之前；刷新帧单独记录，不计入累计收益。

## 主要判读

`y02_paired_future_effects.csv` 中 `gain_AbsRel_A_minus_branch` 为正表示干预分支比 A 更好。
`y02_snapshot_gain_summary.csv` 汇总每个快照的未来累计收益、每帧平均收益、受损帧比例及尾部伤害。
`y02_age_gain_summary.csv` 按序列、检查点、年龄汇总快照表现。

注意：同一序列内的快照窗口存在重叠，结果用于机制诊断，不应把逐帧记录当作独立样本进行显著性推断。三个序列均为开发数据，不是最终盲测集。
"""
    (out_dir / "Y02_README.md").write_text(report, encoding="utf-8")

    review_zip = out_dir / "y02_review.zip"
    include = [
        "y02_preflight.json", "y02_current_snapshot_metrics.csv",
        "y02_future_frame_metrics.csv", "y02_future_frame_summary.csv",
        "y02_paired_future_effects.csv", "y02_snapshot_gain_summary.csv",
        "y02_age_gain_summary.csv", "y02_state_diagnostics.csv",
        "y02_refresh_reset_metrics.csv", "y02_branch_invariants.csv",
        "y02_summary.json", "Y02_README.md",
    ]
    script_path = Path(__file__).resolve()
    with zipfile.ZipFile(review_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            p = out_dir / rel
            if p.is_file():
                zf.write(p, rel)
        if script_path.is_file():
            zf.write(script_path, "gt4a2_y02_full_cycle_single_intervention.py")
    print(f"[{summary['status']}] Y02 review package: {review_zip}", flush=True)
    return 0 if invariants_pass else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        raise
