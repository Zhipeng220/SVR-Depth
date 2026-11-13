#!/usr/bin/env python3
"""Train P_multi on TartanAir/KITTI Scene Flow using the v2 manifest protocol.

This is a separate branch. It never invokes or changes the Bonn compatibility
runner/checkpoints. One optimizer update consumes one 20-frame causal window;
TBPTT detaches recurrent state every four propagated frames.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "experiments" / "crosssource_v2" / "legacy_helpers"
EXPECTED_BASE_SHA256 = "615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e"
sys.path[:0] = [str(ROOT), str(HELPERS)]

from svr_depth.datasets.crosssource import (  # noqa: E402
    CrossSourcePool, DatasetBalancedWindowSampler, WindowRef,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha256(state) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def ref_to_dict(ref: WindowRef) -> dict:
    return {"dataset_id": ref.dataset_id, "scene_id": ref.scene_id,
            "sequence_id": ref.sequence_id, "manifest_path": str(ref.manifest_path),
            "start_index": ref.start_index, "length": ref.length}


def dict_to_ref(row: dict) -> WindowRef:
    return WindowRef(row["dataset_id"], row["scene_id"], row["sequence_id"],
                     Path(row["manifest_path"]), int(row["start_index"]), int(row["length"]))


def build_val_refs(pool: CrossSourcePool, window_frames: int, max_per_dataset: int, seed: int) -> list[WindowRef]:
    rng = random.Random(seed)
    by_dataset: dict[str, list[WindowRef]] = {}
    for (dataset_id, _), adapter in pool.sequences.items():
        frames = adapter.manifest.frames
        starts = []
        if len(frames) < window_frames:
            continue
        for start in range(len(frames) - window_frames + 1):
            if any(f.depth_path is not None for f in frames[start + 1:start + window_frames]):
                starts.append(start)
        if starts:
            start = rng.choice(starts)
            m = adapter.manifest
            by_dataset.setdefault(dataset_id, []).append(
                WindowRef(dataset_id, m.scene_id, m.sequence_id, m.path, start, window_frames)
            )
    refs = []
    for dataset_id, values in sorted(by_dataset.items()):
        rng.shuffle(values)
        refs.extend(values[:max_per_dataset])
    if not refs:
        raise RuntimeError("No valid 20-frame validation windows found")
    return refs


def prepare_window(model, pool, ref, device, gt2, exp1, cv2):
    adapter = pool.adapter(ref.dataset_id, ref.sequence_id)
    samples = [adapter[i] for i in range(ref.start_index, ref.start_index + ref.length)]
    first_path = adapter.manifest.frames[ref.start_index].rgb_path
    shape = exp1.shape_info_for(model, first_path, device)
    frames = []
    for offset, sample in enumerate(samples):
        if sample.rgb.shape[:2] != shape["original_shape"]:
            raise RuntimeError(f"Frame size changes within sequence {ref.sequence_id} at offset {offset}")
        img, frame_u8 = gt2.preprocess_rgb(
            sample.rgb, device, shape["paddings"], shape["network_shape"], shape["mean"], shape["std"]
        )
        use_gt = offset > 0 and sample.has_gt
        gt = torch.from_numpy(sample.depth_m.copy()).to(device) if use_gt else None
        frames.append({
            "index": offset, "img": img, "frame_u8": frame_u8,
            "gt": gt, "rgb_file": adapter.manifest.frames[ref.start_index + offset].rgb_path.name,
            "rgb_path": adapter.manifest.frames[ref.start_index + offset].rgb_path,
            "timestamp_s": sample.timestamp_s,
        })
    with torch.no_grad():
        rays, radius, confidence, features = exp1.keyframe_state(model, frames[0], shape)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    statics = []
    for i in range(1, len(frames)):
        with torch.no_grad():
            static = exp1.transition_static(model, frames[i - 1], frames[i], shape, dis)
        statics.append({k: static[k].detach() for k in ("flow_norm", "grid_sample", "image_residual", "validity_mask")})
    n_supervised = sum(frame["gt"] is not None for frame in frames[1:])
    if n_supervised < 1:
        raise RuntimeError(f"Sampled window has no supervised propagated frame: {ref}")
    return {
        "frames": frames, "statics": statics,
        "keyframe_state": {"rays_map": rays.detach(), "radius": radius.detach(),
                           "confidence": confidence.detach(), "features": [x.detach() for x in features]},
        "network_shape": shape["network_shape"], "padded_shape": shape["padded_shape"],
        "paddings": shape["paddings"], "n_supervised": n_supervised,
        "ref": ref_to_dict(ref),
    }


def move_state(state, device):
    return (state["radius"].to(device).detach(), state["confidence"].to(device).detach(),
            [x.to(device).detach() for x in state["features"]])


def detach_state(radius, confidence, features):
    return radius.detach(), confidence.detach(), [x.detach() for x in features]


def train_window(model, head, optimizer, scaler, window, device, gt2, exp1, tbptt_frames):
    frames, statics = window["frames"], window["statics"]
    n_frames = len(frames)
    if n_frames != 20 or len(statics) != n_frames - 1:
        raise RuntimeError(f"Expected 20-frame window, got {n_frames} frames / {len(statics)} statics")
    n_supervised = int(window["n_supervised"])
    radius, confidence, features = move_state(window["keyframe_state"], device)
    rays = window["keyframe_state"]["rays_map"].to(device)
    prev = frames[0]
    optimizer.zero_grad(set_to_none=True)
    losses, consumed, grad_segments = [], 0, 0
    segment_lengths = []
    for segment_begin in range(1, n_frames, tbptt_frames):
        segment_end = min(n_frames - 1, segment_begin + tbptt_frames - 1)
        loss_terms = []
        hold = [prev]
        for offset in range(segment_begin, segment_end + 1):
            curr = frames[offset]
            static = statics[offset - 1]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                radius, confidence, features = exp1.one_fast_step(
                    model, prev, curr, static, radius, confidence, features
                )
                if curr["gt"] is not None:
                    pred = gt2.make_z_depth(radius, rays, window["padded_shape"],
                                            window["paddings"], model.interpolation_mode)
                    loss, count = gt2.masked_log_depth_loss(pred, curr["gt"])
                    if not torch.isfinite(loss.detach()).all():
                        raise RuntimeError("Non-finite masked log-depth loss")
                    loss_terms.append(loss)
                    losses.append(float(loss.detach().item()))
                    consumed += 1
            hold.append((curr, static))
            prev = curr
        if loss_terms:
            scaler.scale(torch.stack(loss_terms).sum() / float(n_supervised)).backward()
            grad_segments += 1
        radius, confidence, features = detach_state(radius, confidence, features)
        prev = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in prev.items()}
        segment_lengths.append(segment_end - segment_begin + 1)
        del loss_terms, hold
    if consumed != n_supervised:
        raise RuntimeError(f"Supervision count mismatch: consumed={consumed}, expected={n_supervised}")
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in head.parameters() if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
        raise RuntimeError("P_multi has missing or non-finite gradients")
    nonzero = sum(bool(torch.count_nonzero(g).item()) for g in gradients)
    if not nonzero:
        raise RuntimeError("All P_multi gradients are zero")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0).item())
    old_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if float(scaler.get_scale()) < old_scale:
        raise RuntimeError("GradScaler skipped optimizer update due to overflow")
    return {"mean_log_depth_loss": float(np.mean(losses)), "supervised_frames": consumed,
            "gradient_segments": grad_segments, "nonzero_gradient_tensors": nonzero,
            "gradient_norm_before_clip": grad_norm, "segment_lengths": segment_lengths}


def evaluate_windows(model, pool, refs, device, gt2, exp1, cv2):
    model.eval()
    head = model.pixel_decoder.persistent_feature_repair
    head.eval()
    values: dict[str, list[float]] = {}
    with torch.no_grad():
        for ref in refs:
            window = prepare_window(model, pool, ref, device, gt2, exp1, cv2)
            frames, statics = window["frames"], window["statics"]
            radius, confidence, features = move_state(window["keyframe_state"], device)
            rays = window["keyframe_state"]["rays_map"].to(device)
            prev = frames[0]
            for offset in range(1, len(frames)):
                curr = frames[offset]
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    radius, confidence, features = exp1.one_fast_step(
                        model, prev, curr, statics[offset - 1], radius, confidence, features
                    )
                    if curr["gt"] is not None:
                        pred = gt2.make_z_depth(radius, rays, window["padded_shape"],
                                                window["paddings"], model.interpolation_mode)
                        loss, _ = gt2.masked_log_depth_loss(pred, curr["gt"])
                        values.setdefault(ref.dataset_id, []).append(float(loss.item()))
                prev = curr
    head.train()
    if not values:
        raise RuntimeError("Validation windows produced no supervised frames")
    per_dataset = {k: float(np.mean(v)) for k, v in sorted(values.items())}
    return {"per_dataset": per_dataset, "macro_loss": float(np.mean(list(per_dataset.values())))}


def write_curve(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--snapshot", required=True, help="Official VeloDepth snapshot containing config.json/model.safetensors")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--expected-base-sha256", default=EXPECTED_BASE_SHA256,
                   help="Pinned VeloDepth model.safetensors SHA256; override only with an explicitly audited snapshot")
    p.add_argument("--init-seed", type=int, default=20261103)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--tbptt-frames", type=int, default=4)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--val-windows-per-dataset", type=int, default=8)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run exactly one optimizer update; do not save a final checkpoint")
    args = p.parse_args()
    if args.steps < 1 or args.tbptt_frames < 1 or args.val_every < 1:
        p.error("steps, tbptt-frames and val-every must be positive")
    if args.smoke:
        args.steps = 1
    snapshot = Path(args.snapshot).expanduser().resolve()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete VeloDepth snapshot: {snapshot}")
    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_cfg = config.get("training", {})
    window_frames = int(training_cfg.get("window_frames", 20))
    if window_frames != 20:
        raise RuntimeError("CrossSource-v2 currently locks window_frames=20")
    if int(training_cfg.get("tbptt_frames", args.tbptt_frames)) != args.tbptt_frames:
        raise RuntimeError("CLI tbptt-frames differs from config; keep a single recorded protocol")
    weights = training_cfg.get("dataset_weights")
    if not weights:
        raise RuntimeError("Config must include training.dataset_weights")

    train_pool = CrossSourcePool.from_config(config_path, split="train", verify_files=True)
    val_pool = CrossSourcePool.from_config(config_path, split="val", verify_files=True)
    train_pool.validate_disjoint(val_pool)
    sampler = DatasetBalancedWindowSampler(
        train_pool, dataset_weights=weights, window_frames=window_frames,
        min_supervised_frames=int(training_cfg.get("min_supervised_frames", 1)), seed=args.seed,
    )
    order_path = Path(args.out_dir).expanduser().resolve() / "sample_order.json"
    out_dir = order_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if order_path.exists() and not args.resume:
        raise RuntimeError(f"Existing sample_order.json found at {order_path}; use --resume or a new output directory")
    if order_path.exists():
        order_rows = json.loads(order_path.read_text(encoding="utf-8"))
        if len(order_rows) < args.steps:
            raise RuntimeError("Existing sample_order.json is shorter than requested steps")
    else:
        order_rows = [ref_to_dict(x) for x in sampler.sample_many(args.steps)]
        order_path.write_text(json.dumps(order_rows, indent=2), encoding="utf-8")
    refs = [dict_to_ref(x) for x in order_rows[:args.steps]]
    order_sha = hashlib.sha256(json.dumps(order_rows[:args.steps], sort_keys=True).encode()).hexdigest()
    val_refs = build_val_refs(val_pool, window_frames, args.val_windows_per_dataset, args.seed + 1)

    helper_dir = str(HELPERS)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    import numpy as np
    import torch
    import cv2
    import gt2_fixed_clip_repair_train as gt2
    import gt4a2_exp1_train_eval as exp1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no training was run")
    device = torch.device("cuda")
    snapshot_sha = sha256_file(snapshot / "model.safetensors")
    if args.expected_base_sha256 and snapshot_sha != args.expected_base_sha256:
        raise RuntimeError(
            f"VeloDepth snapshot hash mismatch: expected={args.expected_base_sha256}, actual={snapshot_sha}. "
            "Use the v2.1 pinned snapshot or explicitly pass the audited SHA256."
        )
    config_sha = sha256_file(config_path)
    manifest_paths = sorted({str(adapter.manifest.path) for adapter in train_pool.sequences.values()} |
                            {str(adapter.manifest.path) for adapter in val_pool.sequences.values()})
    manifest_hashes = {x: sha256_file(Path(x)) for x in manifest_paths}
    identity = {
        "protocol": "SVR-Depth-CrossSource-v2", "method": "P_multi", "seed": args.seed,
        "initialization_seed": args.init_seed, "steps": args.steps,
        "window_frames": window_frames, "tbptt_frames": args.tbptt_frames,
        "dataset_weights": weights, "snapshot_sha256": snapshot_sha,
        "config_sha256": config_sha, "manifest_sha256": manifest_hashes,
        "source_sha256": {
            str(p.relative_to(ROOT)): sha256_file(p) for p in (
                ROOT / "svr_depth/datasets/crosssource.py",
                ROOT / "svr_depth/datasets/native_tartanair.py",
                ROOT / "svr_depth/datasets/kitti_sceneflow.py",
                ROOT / "scripts/train_crosssource_v2.py",
                HELPERS / "gt2_fixed_clip_repair_train.py",
                HELPERS / "gt4a2_exp1_train_eval.py",
            )
        },
        "sample_order_sha256": order_sha,
    }
    identity_path = out_dir / "run_identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError("run_identity.json differs; use a new output directory")
    else:
        identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")

    torch.manual_seed(args.init_seed)
    init_model = gt2.load_model(snapshot, "pre_readout", device)
    init_head_state = {k: v.detach().cpu().clone() for k, v in init_model.pixel_decoder.persistent_feature_repair.state_dict().items()}
    init_head_sha = state_sha256(init_head_state)
    del init_model
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = gt2.load_model(snapshot, "pre_readout", device)
    head = model.pixel_decoder.persistent_feature_repair
    head.load_state_dict(init_head_state, strict=True)
    model.eval(); head.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in trainable) != 170496:
        raise RuntimeError(f"Expected exactly 170496 trainable repair-head parameters, got {sum(p.numel() for p in trainable)}")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_score = float("inf")
    start_step = 0
    recovery_path = out_dir / "recovery.pt"
    if args.resume:
        if not recovery_path.is_file():
            raise FileNotFoundError(f"--resume requested but {recovery_path} is missing")
        ckpt = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if ckpt.get("identity") != identity:
            raise RuntimeError("Recovery checkpoint identity mismatch")
        head.load_state_dict(ckpt["head_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_step = int(ckpt["step"])
        best_score = float(ckpt.get("best_val_loss", float("inf")))
        torch.set_rng_state(ckpt["torch_rng_state"])
        np.random.set_state(ckpt["numpy_rng_state"])
        random.setstate(ckpt["python_rng_state"])
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])

    curve_path = out_dir / "training_curve.csv"
    curve = []
    if start_step and curve_path.is_file():
        with curve_path.open("r", newline="", encoding="utf-8-sig") as f:
            curve = [row for row in csv.DictReader(f) if int(row["step"]) <= start_step]
    start = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        window = prepare_window(model, train_pool, refs[step - 1], device, gt2, exp1, cv2)
        metrics = train_window(model, head, optimizer, scaler, window, device, gt2, exp1, args.tbptt_frames)
        row = {
            "step": step, "dataset_id": refs[step - 1].dataset_id,
            "scene_id": refs[step - 1].scene_id, "sequence_id": refs[step - 1].sequence_id,
            "start_index": refs[step - 1].start_index,
            "wall_sec": time.perf_counter() - t0,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            **metrics,
        }
        curve.append(row)
        if step % args.val_every == 0 or step == args.steps:
            val = evaluate_windows(model, val_pool, val_refs, device, gt2, exp1, cv2)
            row["val_macro_loss"] = val["macro_loss"]
            row["val_per_dataset"] = json.dumps(val["per_dataset"], sort_keys=True)
            if val["macro_loss"] < best_score:
                best_score = val["macro_loss"]
                best_path = out_dir / "best_checkpoint.pt"
                torch.save({"identity": identity, "step": step, "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                            "val": val, "best_val_loss": best_score, "init_head_sha256": init_head_sha}, best_path)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"[P_multi-CrossSource-v2] {step}/{args.steps} dataset={row['dataset_id']} "
                  f"loss={row['mean_log_depth_loss']:.6f} val={row.get('val_macro_loss', float('nan')):.6f} "
                  f"wall={row['wall_sec']:.2f}s", flush=True)
        if step % 25 == 0 or step == args.steps:
            write_curve(curve_path, curve)
        if step % 100 == 0 or step == args.steps:
            torch.save({
                "identity": identity, "step": step,
                "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
                "best_val_loss": best_score, "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            }, recovery_path)
    if not args.smoke:
        final_path = out_dir / "last_checkpoint.pt"
        torch.save({"identity": identity, "step": args.steps,
                    "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "best_val_loss": best_score, "init_head_sha256": init_head_sha}, final_path)
        print(f"[PASS] saved {final_path}; best validation loss={best_score:.6f}", flush=True)
    else:
        print("[SMOKE PASS] one optimizer update completed; no final checkpoint saved", flush=True)

if __name__ == "__main__":
    main()
