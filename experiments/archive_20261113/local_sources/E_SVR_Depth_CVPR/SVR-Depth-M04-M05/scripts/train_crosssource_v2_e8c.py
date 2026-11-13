#!/usr/bin/env python3
"""E8-C: source-balanced gradient aggregation for CrossSource-v2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_crosssource_v2 as base  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from svr_depth.datasets.crosssource import (  # noqa: E402
    CrossSourcePool,
    DatasetBalancedWindowSampler,
)

DATASETS = ("tartanair", "kitti_sceneflow")


def sample_one_from_dataset(sampler, dataset_id, rng):
    """Draw one valid window from a specific dataset using the v2 sampler pools."""
    if dataset_id not in sampler._dataset_scenes:
        raise RuntimeError(f"Dataset unavailable in sampler: {dataset_id}")
    scenes = sampler._dataset_scenes[dataset_id]
    scene_key = rng.choice(sorted(scenes))
    adapter, starts = rng.choice(scenes[scene_key])
    start = rng.choice(starts)
    m = adapter.manifest
    return base.WindowRef(
        m.dataset_id, m.scene_id, m.sequence_id, m.path, start, sampler.window_frames
    )


def collect_window_gradient(model, head, scaler, window, device, gt2, exp1, tbptt_frames):
    """Backprop one window and return its scaled, flattened head gradient."""
    frames, statics = window["frames"], window["statics"]
    n_frames = len(frames)
    if n_frames != 20 or len(statics) != n_frames - 1:
        raise RuntimeError(f"Expected 20-frame window, got {n_frames} / {len(statics)}")

    n_supervised = int(window["n_supervised"])
    radius, confidence, features = base.move_state(window["keyframe_state"], device)
    rays = window["keyframe_state"]["rays_map"].to(device)
    prev = frames[0]
    losses, consumed, grad_segments = [], 0, 0
    segment_lengths = []
    params = [p for p in head.parameters() if p.requires_grad]
    optimizer_grads_clear = True

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
                    pred = gt2.make_z_depth(
                        radius, rays, window["padded_shape"], window["paddings"],
                        model.interpolation_mode,
                    )
                    loss, _ = gt2.masked_log_depth_loss(pred, curr["gt"])
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
        radius, confidence, features = base.detach_state(radius, confidence, features)
        prev = {k: (v.detach() if isinstance(v, torch.Tensor) else v) for k, v in prev.items()}
        segment_lengths.append(segment_end - segment_begin + 1)
        del loss_terms, hold

    if consumed != n_supervised:
        raise RuntimeError(f"Supervision mismatch: consumed={consumed}, expected={n_supervised}")

    grads = []
    for p in params:
        grads.append(torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.detach().reshape(-1).clone())
    if not grads:
        raise RuntimeError("No trainable repair-head parameters")
    flat = torch.cat(grads)
    if not torch.isfinite(flat).all():
        raise RuntimeError("Non-finite source gradient")
    if torch.count_nonzero(flat).item() == 0:
        raise RuntimeError("All source gradients are zero")

    scale = float(scaler.get_scale())
    norm_scaled = float(torch.linalg.vector_norm(flat).item())
    metrics = {
        "mean_log_depth_loss": float(np.mean(losses)),
        "supervised_frames": consumed,
        "gradient_segments": grad_segments,
        "segment_lengths": segment_lengths,
        "gradient_norm": norm_scaled / scale,
    }
    # The caller clears p.grad before collecting the other source.
    return flat, metrics


def aggregate_and_step(head, optimizer, scaler, grad_t, grad_k):
    """Normalize both source gradients, aggregate, clip once, and step once."""
    nt = torch.linalg.vector_norm(grad_t)
    nk = torch.linalg.vector_norm(grad_k)
    if nt.item() <= 1e-20 or nk.item() <= 1e-20:
        raise RuntimeError(f"Cannot normalize zero gradient: T={nt.item()} K={nk.item()}")

    cosine = torch.dot(grad_t, grad_k) / (nt * nk)
    valid = (grad_t.abs() > 1e-12) & (grad_k.abs() > 1e-12)
    conflict = ((grad_t[valid] * grad_k[valid]) < 0).float().mean() if valid.any() else torch.tensor(float("nan"), device=grad_t.device)

    # Scale-preserving equal-direction aggregation. With equal source norms,
    # this reduces to the arithmetic mean (g_T + g_K) / 2.
    combined = ((nt + nk) / 4.0) * (grad_t / nt + grad_k / nk)

    optimizer.zero_grad(set_to_none=True)
    offset = 0
    for p in [p for p in head.parameters() if p.requires_grad]:
        size = p.numel()
        p.grad = combined[offset:offset + size].view_as(p).clone()
        offset += size
    if offset != combined.numel():
        raise RuntimeError("Aggregated gradient size does not match trainable parameters")

    scaler.unscale_(optimizer)
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all().item() for g in grads):
        raise RuntimeError("Aggregated gradient is missing or non-finite")
    nonzero = sum(bool(torch.count_nonzero(g).item()) for g in grads)
    if not nonzero:
        raise RuntimeError("Aggregated gradient is all zero")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0).item())
    old_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if float(scaler.get_scale()) < old_scale:
        raise RuntimeError("GradScaler skipped optimizer update due to overflow")

    scale = float(scaler.get_scale())
    # nt/nk were computed in scaled-gradient space; their ratio and cosine are invariant.
    return {
        "gradient_cosine": float(cosine.item()),
        "gradient_conflict_ratio": float(conflict.item()),
        "tartanair_over_kitti_norm_ratio": float((nt / nk).item()),
        "gradient_norm_before_clip": grad_norm,
        "nonzero_gradient_tensors": nonzero,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--init-seed", type=int, default=20261103)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--tbptt-frames", type=int, default=4)
    p.add_argument("--val-every", type=int, default=100)
    p.add_argument("--val-windows-per-dataset", type=int, default=8)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.steps = 1
    if args.steps < 1 or args.tbptt_frames < 1 or args.val_every < 1:
        p.error("steps, tbptt-frames and val-every must be positive")

    snapshot = Path(args.snapshot).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete VeloDepth snapshot: {snapshot}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    training_cfg = config.get("training", {})
    if int(training_cfg.get("window_frames", 20)) != 20:
        raise RuntimeError("E8-C locks 20-frame windows")
    if int(training_cfg.get("tbptt_frames", args.tbptt_frames)) != args.tbptt_frames:
        raise RuntimeError("CLI TBPTT differs from config")
    weights = training_cfg.get("dataset_weights")
    if not weights or set(weights) != set(DATASETS):
        raise RuntimeError(f"Config must contain exactly {DATASETS}; got {weights}")

    train_pool = CrossSourcePool.from_config(config_path, split="train", verify_files=True)
    val_pool = CrossSourcePool.from_config(config_path, split="val", verify_files=True)
    train_pool.validate_disjoint(val_pool)
    sampler = DatasetBalancedWindowSampler(
        train_pool, dataset_weights=weights, window_frames=20,
        min_supervised_frames=int(training_cfg.get("min_supervised_frames", 1)), seed=args.seed,
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    order_path = out_dir / "sample_order.json"
    recovery_path = out_dir / "recovery.pt"
    identity_path = out_dir / "run_identity.json"
    if order_path.exists() and not args.resume:
        raise RuntimeError(f"{order_path} already exists; use a new output directory")

    rng = random.Random(args.seed)
    if order_path.exists():
        order_rows = json.loads(order_path.read_text())
        if len(order_rows) < args.steps:
            raise RuntimeError("Existing sample order is shorter than requested steps")
    else:
        order_rows = []
        for step in range(1, args.steps + 1):
            rt = sample_one_from_dataset(sampler, "tartanair", rng)
            rk = sample_one_from_dataset(sampler, "kitti_sceneflow", rng)
            order_rows.append({"step": step, "tartanair": base.ref_to_dict(rt), "kitti_sceneflow": base.ref_to_dict(rk)})
        order_path.write_text(json.dumps(order_rows, indent=2))

    sample_sha = hashlib.sha256(json.dumps(order_rows[:args.steps], sort_keys=True).encode()).hexdigest()
    snapshot_sha = base.sha256_file(snapshot / "model.safetensors")
    expected_sha = base.EXPECTED_BASE_SHA256
    if snapshot_sha != expected_sha:
        raise RuntimeError(f"Snapshot hash mismatch: expected {expected_sha}, got {snapshot_sha}")
    manifest_paths = sorted(
        {str(a.manifest.path) for a in train_pool.sequences.values()} |
        {str(a.manifest.path) for a in val_pool.sequences.values()}
    )
    manifest_hashes = {x: base.sha256_file(Path(x)) for x in manifest_paths}
    helper_dir = str(base.HELPERS)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    import cv2
    import gt2_fixed_clip_repair_train as gt2
    import gt4a2_exp1_train_eval as exp1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for E8-C")
    device = torch.device("cuda")

    identity = {
        "protocol": "SVR-Depth-CrossSource-v2",
        "method": "E8-C_SourceBalancedGradientAggregation",
        "seed": args.seed,
        "initialization_seed": args.init_seed,
        "steps": args.steps,
        "window_frames": 20,
        "tbptt_frames": args.tbptt_frames,
        "dataset_pair_per_step": list(DATASETS),
        "aggregation": "equal_unit_gradient_directions_scaled_by_mean_source_norm",
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "snapshot_sha256": snapshot_sha,
        "config_sha256": base.sha256_file(config_path),
        "manifest_sha256": manifest_hashes,
        "sample_order_sha256": sample_sha,
        "source_sha256": {
            "scripts/train_crosssource_v2.py": base.sha256_file(ROOT / "scripts/train_crosssource_v2.py"),
            "scripts/train_crosssource_v2_e8c.py": base.sha256_file(Path(__file__).resolve()),
            "svr_depth/datasets/crosssource.py": base.sha256_file(ROOT / "svr_depth/datasets/crosssource.py"),
            "svr_depth/datasets/native_tartanair.py": base.sha256_file(ROOT / "svr_depth/datasets/native_tartanair.py"),
            "svr_depth/datasets/kitti_sceneflow.py": base.sha256_file(ROOT / "svr_depth/datasets/kitti_sceneflow.py"),
            "velodepth/models/decoder_velo.py": base.sha256_file(ROOT / "velodepth/models/decoder_velo.py"),
            "legacy_helpers/gt2_fixed_clip_repair_train.py": base.sha256_file(base.HELPERS / "gt2_fixed_clip_repair_train.py"),
            "legacy_helpers/gt4a2_exp1_train_eval.py": base.sha256_file(base.HELPERS / "gt4a2_exp1_train_eval.py"),
        },
    }
    if identity_path.exists():
        old = json.loads(identity_path.read_text())
        if old != identity:
            raise RuntimeError("E8-C run identity mismatch")
    else:
        identity_path.write_text(json.dumps(identity, indent=2))

    torch.manual_seed(args.init_seed)
    init_model = gt2.load_model(snapshot, "pre_readout", device)
    init_head_state = {k: v.detach().cpu().clone() for k, v in init_model.pixel_decoder.persistent_feature_repair.state_dict().items()}
    init_head_sha = base.state_sha256(init_head_state)
    del init_model
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = gt2.load_model(snapshot, "pre_readout", device)
    head = model.pixel_decoder.persistent_feature_repair
    head.load_state_dict(init_head_state, strict=True)
    model.eval()
    head.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    if n_trainable != 170496:
        raise RuntimeError(f"Expected 170496 trainable parameters, got {n_trainable}")
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_score = float("inf")
    start_step = 0

    if args.resume:
        if not recovery_path.is_file():
            raise FileNotFoundError(f"Missing recovery checkpoint: {recovery_path}")
        ck = torch.load(recovery_path, map_location="cpu", weights_only=False)
        if ck.get("identity") != identity:
            raise RuntimeError("Recovery identity mismatch")
        head.load_state_dict(ck["head_state"], strict=True)
        optimizer.load_state_dict(ck["optimizer_state"])
        scaler.load_state_dict(ck["scaler_state"])
        start_step = int(ck["step"])
        best_score = float(ck.get("best_val_loss", float("inf")))
        torch.set_rng_state(ck["torch_rng_state"])
        np.random.set_state(ck["numpy_rng_state"])
        random.setstate(ck["python_rng_state"])
        torch.cuda.set_rng_state_all(ck["cuda_rng_state"])

    val_refs = base.build_val_refs(val_pool, 20, args.val_windows_per_dataset, args.seed + 1)
    curve_path = out_dir / "training_curve.csv"
    curve = []
    if start_step and curve_path.is_file():
        with curve_path.open("r", newline="", encoding="utf-8-sig") as f:
            curve = [r for r in csv.DictReader(f) if int(r["step"]) <= start_step]

    print("========== E8-C START ==========")
    print("Device:", device, torch.cuda.get_device_name(0))
    print("Trainable parameters:", n_trainable)
    print("Optimizer updates:", args.steps)
    print("Windows per update: 2 (one TartanAir + one KITTI)")
    print("TBPTT:", args.tbptt_frames)

    for step in range(start_step + 1, args.steps + 1):
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        pair = order_rows[step - 1]
        ref_t = base.dict_to_ref(pair["tartanair"])
        ref_k = base.dict_to_ref(pair["kitti_sceneflow"])

        optimizer.zero_grad(set_to_none=True)
        win_t = base.prepare_window(model, train_pool, ref_t, device, gt2, exp1, cv2)
        grad_t, met_t = collect_window_gradient(model, head, scaler, win_t, device, gt2, exp1, args.tbptt_frames)
        optimizer.zero_grad(set_to_none=True)
        del win_t

        win_k = base.prepare_window(model, train_pool, ref_k, device, gt2, exp1, cv2)
        grad_k, met_k = collect_window_gradient(model, head, scaler, win_k, device, gt2, exp1, args.tbptt_frames)
        optimizer.zero_grad(set_to_none=True)
        del win_k

        agg = aggregate_and_step(head, optimizer, scaler, grad_t, grad_k)
        del grad_t, grad_k

        row = {
            "step": step,
            "tartanair_scene": ref_t.scene_id,
            "tartanair_sequence": ref_t.sequence_id,
            "tartanair_start": ref_t.start_index,
            "kitti_scene": ref_k.scene_id,
            "kitti_sequence": ref_k.sequence_id,
            "kitti_start": ref_k.start_index,
            "tartanair_loss": met_t["mean_log_depth_loss"],
            "kitti_loss": met_k["mean_log_depth_loss"],
            "tartanair_gradient_norm": met_t["gradient_norm"],
            "kitti_gradient_norm": met_k["gradient_norm"],
            "gradient_cosine": agg["gradient_cosine"],
            "gradient_conflict_ratio": agg["gradient_conflict_ratio"],
            "tartanair_over_kitti_norm_ratio": agg["tartanair_over_kitti_norm_ratio"],
            "gradient_norm_before_clip": agg["gradient_norm_before_clip"],
            "supervised_frames_total": met_t["supervised_frames"] + met_k["supervised_frames"],
            "wall_sec": time.perf_counter() - t0,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        }

        if not args.smoke and (step % args.val_every == 0 or step == args.steps):
            val = base.evaluate_windows(model, val_pool, val_refs, device, gt2, exp1, cv2)
            row["val_macro_loss"] = val["macro_loss"]
            row["val_per_dataset"] = json.dumps(val["per_dataset"], sort_keys=True)
            if val["macro_loss"] < best_score:
                best_score = val["macro_loss"]
                torch.save({
                    "identity": identity,
                    "step": step,
                    "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    "val": val,
                    "best_val_loss": best_score,
                    "init_head_sha256": init_head_sha,
                }, out_dir / "best_checkpoint.pt")

        curve.append(row)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(
                f"[E8-C] {step}/{args.steps} TA={row['tartanair_loss']:.6f} "
                f"KITTI={row['kitti_loss']:.6f} cos={row['gradient_cosine']:+.4f} "
                f"conflict={row['gradient_conflict_ratio']:.4f} wall={row['wall_sec']:.2f}s",
                flush=True,
            )
        if step % 10 == 0 or step == args.steps:
            base.write_curve(curve_path, curve)
        if not args.smoke and (step % 100 == 0 or step == args.steps):
            torch.save({
                "identity": identity,
                "step": step,
                "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_val_loss": best_score,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            }, recovery_path)

    if args.smoke:
        print("[E8-C SMOKE PASS] exactly one optimizer update from two source windows", flush=True)
    else:
        torch.save({
            "identity": identity,
            "step": args.steps,
            "head_state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            "best_val_loss": best_score,
            "init_head_sha256": init_head_sha,
        }, out_dir / "last_checkpoint.pt")
        print(f"[E8-C PASS] saved last_checkpoint.pt; best_val={best_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
