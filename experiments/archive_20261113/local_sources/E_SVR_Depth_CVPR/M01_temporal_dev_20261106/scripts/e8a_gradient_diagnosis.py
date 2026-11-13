#!/usr/bin/env python3
import os

import sys
import json
import csv
import random
import argparse
import gc
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "experiments/crosssource_v2/legacy_helpers"

sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(HELPERS)]

import train_crosssource_v2 as train
import gt2_fixed_clip_repair_train as gt2
import gt4a2_exp1_train_eval as exp1

from svr_depth.datasets.crosssource import (
    CrossSourcePool,
    WindowRef,
)

DATASETS = ["tartanair", "kitti_sceneflow"]


def build_refs(pool, dataset_id, count, seed, min_supervised=1):
    rng = random.Random(seed)
    candidates = []

    for (ds, _), adapter in pool.sequences.items():
        if ds != dataset_id:
            continue

        manifest = adapter.manifest
        frames = manifest.frames

        for start in range(len(frames) - 19):
            supervised = sum(
                getattr(f, "depth_path", None) is not None
                for f in frames[start + 1:start + 20]
            )

            if supervised >= min_supervised:
                candidates.append(
                    WindowRef(
                        dataset_id,
                        manifest.scene_id,
                        manifest.sequence_id,
                        manifest.path,
                        start,
                        20,
                    )
                )

    if not candidates:
        raise RuntimeError(f"No training windows for {dataset_id}")

    rng.shuffle(candidates)
    return candidates[:count]


def compute_gradient(model, head, pool, ref, device):
    head.zero_grad(set_to_none=True)

    window = train.prepare_window(
        model, pool, ref, device, gt2, exp1, cv2
    )

    frames = window["frames"]
    statics = window["statics"]

    radius, confidence, features = train.move_state(
        window["keyframe_state"], device
    )

    rays = window["keyframe_state"]["rays_map"].to(device)

    n_supervised = int(window["n_supervised"])
    prev = frames[0]

    losses = []

    for begin in range(1, len(frames), 4):
        end = min(len(frames) - 1, begin + 3)
        segment_losses = []

        for offset in range(begin, end + 1):
            curr = frames[offset]
            static = statics[offset - 1]

            # GPU forward: FP32 diagnostic, no explicit autocast
            radius, confidence, features = exp1.one_fast_step(
                model,
                prev,
                curr,
                static,
                radius,
                confidence,
                features,
            )

            if curr["gt"] is not None:
                pred = gt2.make_z_depth(
                    radius,
                    rays,
                    window["padded_shape"],
                    window["paddings"],
                    model.interpolation_mode,
                )

                loss, _ = gt2.masked_log_depth_loss(
                    pred, curr["gt"]
                )

                segment_losses.append(loss)
                losses.append(float(loss.detach()))

            prev = curr

        if segment_losses:
            (
                torch.stack(segment_losses).sum() / n_supervised
            ).backward()

        radius, confidence, features = train.detach_state(
            radius, confidence, features
        )

        prev = {
            k: v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in prev.items()
        }

    vectors = []
    missing = []

    for name, param in head.named_parameters():
        if param.grad is None:
            vectors.append(
                torch.zeros(param.numel(), dtype=torch.float64)
            )
            missing.append(name)
        else:
            vectors.append(
                param.grad.detach().cpu().double().reshape(-1)
            )

    gradient = torch.cat(vectors)

    if not torch.isfinite(gradient).all():
        raise RuntimeError("Non-finite gradient detected")

    if torch.count_nonzero(gradient) == 0:
        raise RuntimeError("All gradients are zero")

    result = {
        "gradient": gradient,
        "mean_loss": float(np.mean(losses)),
        "supervised_frames": n_supervised,
        "gradient_norm": float(torch.linalg.vector_norm(gradient)),
        "missing_gradient_parameters": missing,
        "window": train.ref_to_dict(ref),
    }

    head.zero_grad(set_to_none=True)

    del window, radius, confidence, features
    gc.collect()

    return result


def cosine(a, b):
    na = torch.linalg.vector_norm(a)
    nb = torch.linalg.vector_norm(b)

    if na < 1e-15 or nb < 1e-15:
        return None

    return float(torch.dot(a, b) / (na * nb))


def conflict_ratio(a, b, eps=1e-12):
    valid = (a.abs() > eps) & (b.abs() > eps)

    if not valid.any():
        return None

    return float(
        ((a[valid] * b[valid]) < 0).double().mean()
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/crosssource_v2/generated_20261103/C_tartanair_kitti.json",
    )

    parser.add_argument(
        "--snapshot",
        default="/root/autodl-tmp/SVR-Depth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c",
    )

    parser.add_argument(
        "--checkpoint",
        default="outputs/crosssource_v2_E4_20261104_C_s20261103_1000step/best_checkpoint.pt",
    )

    parser.add_argument(
        "--out-dir",
        default="outputs/crosssource_v2_E8A_gradient_diagnosis",
    )

    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--cpu-threads", type=int, default=8)

    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)

    if not torch.cuda.is_available():
        raise RuntimeError("E8-A GPU requested but CUDA is unavailable")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    config_path = ROOT / args.config
    snapshot = Path(args.snapshot)
    checkpoint_path = ROOT / args.checkpoint
    out = ROOT / args.out_dir

    out.mkdir(parents=True, exist_ok=True)

    if any(out.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {out}")

    print("========== E8-A GPU DIAGNOSIS ==========")
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("Threads:", torch.get_num_threads())

    # 1. Load E4-C checkpoint
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    print("E4 checkpoint step:", ckpt["step"])
    print("E4 validation:", ckpt["val"])

    # 2. Load model
    model = gt2.load_model(
        snapshot,
        "pre_readout",
        device,
    )

    head = model.pixel_decoder.persistent_feature_repair

    head.load_state_dict(
        ckpt["head_state"],
        strict=True,
    )

    for p in model.parameters():
        p.requires_grad_(False)

    for p in head.parameters():
        p.requires_grad_(True)

    model.eval()
    head.train()

    trainable = sum(
        p.numel() for p in head.parameters() if p.requires_grad
    )

    print("Trainable parameters:", trainable)

    if trainable != 170496:
        raise RuntimeError("Unexpected trainable parameter count")

    # 3. Build training pool
    pool = CrossSourcePool.from_config(
        config_path,
        split="train",
        verify_files=True,
    )

    training_cfg = json.loads(
        config_path.read_text()
    ).get("training", {})

    min_supervised = int(
        training_cfg.get("min_supervised_frames", 1)
    )

    source_results = {}

    # 4. Calculate independent source gradients
    for i, dataset in enumerate(DATASETS):
        print(f"\n========== {dataset} ==========")

        refs = build_refs(
            pool,
            dataset,
            args.windows,
            args.seed + i,
            min_supervised,
        )

        gradients = []
        records = []

        for j, ref in enumerate(refs):
            print(
                f"[{dataset}] window {j+1}/{len(refs)}",
                flush=True,
            )

            result = compute_gradient(
                model,
                head,
                pool,
                ref,
                device,
            )

            gradients.append(result["gradient"])

            records.append({
                "window": result["window"],
                "mean_loss": result["mean_loss"],
                "supervised_frames": result["supervised_frames"],
                "gradient_norm": result["gradient_norm"],
                "missing_gradient_parameters":
                    result["missing_gradient_parameters"],
            })

            print(
                f"Loss={result['mean_loss']:.6f} "
                f"GradNorm={result['gradient_norm']:.6f}",
                flush=True,
            )

        mean_gradient = torch.stack(gradients).mean(dim=0)

        source_results[dataset] = {
            "gradient": mean_gradient,
            "records": records,
            "gradient_norm": float(
                torch.linalg.vector_norm(mean_gradient)
            ),
            "mean_loss": float(
                np.mean([r["mean_loss"] for r in records])
            ),
        }

        del gradients
        gc.collect()

    # 5. Cross-source comparison
    ga = source_results["tartanair"]["gradient"]
    gb = source_results["kitti_sceneflow"]["gradient"]

    na = source_results["tartanair"]["gradient_norm"]
    nb = source_results["kitti_sceneflow"]["gradient_norm"]

    comparison = {
        "tartanair_gradient_norm": na,
        "kitti_gradient_norm": nb,
        "gradient_cosine": cosine(ga, gb),
        "conflict_ratio": conflict_ratio(ga, gb),
        "tartanair_over_kitti_norm_ratio":
            na / max(nb, 1e-15),
        "windows_per_dataset": args.windows,
    }

    print("\n========== CROSS-SOURCE RESULTS ==========")

    for key, value in comparison.items():
        print(f"{key}: {value}")

    # 6. Layer-wise comparison
    layer_parts = defaultdict(list)
    offset = 0

    for name, param in head.named_parameters():
        size = param.numel()
        layer = name.rsplit(".", 1)[0]

        layer_parts[layer].append(
            (offset, offset + size)
        )

        offset += size

    layer_results = []

    for layer, slices in layer_parts.items():
        indices_a = torch.cat([
            ga[s:e] for s, e in slices
        ])

        indices_b = torch.cat([
            gb[s:e] for s, e in slices
        ])

        layer_results.append({
            "layer": layer,
            "tartanair_norm": float(
                torch.linalg.vector_norm(indices_a)
            ),
            "kitti_norm": float(
                torch.linalg.vector_norm(indices_b)
            ),
            "cosine": cosine(indices_a, indices_b),
            "conflict_ratio": conflict_ratio(
                indices_a, indices_b
            ),
        })

    # 7. Save results
    summary = {
        "experiment": "E8-A",
        "device": "CPU",
        "checkpoint_step": ckpt["step"],
        "trainable_parameters": trainable,
        "source_results": {
            ds: {
                "gradient_norm": data["gradient_norm"],
                "mean_loss": data["mean_loss"],
                "windows": data["records"],
            }
            for ds, data in source_results.items()
        },
        "comparison": comparison,
        "layerwise": layer_results,
    }

    with open(
        out / "gradient_diagnosis.json",
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    with open(
        out / "gradient_diagnosis.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "record_type",
                "name",
                "gradient_norm",
                "cosine",
                "conflict_ratio",
            ],
        )

        writer.writeheader()

        for ds, data in source_results.items():
            writer.writerow({
                "record_type": "source",
                "name": ds,
                "gradient_norm": data["gradient_norm"],
            })

        writer.writerow({
            "record_type": "comparison",
            "name": "cross_source",
            "cosine": comparison["gradient_cosine"],
            "conflict_ratio": comparison["conflict_ratio"],
        })

        for item in layer_results:
            writer.writerow({
                "record_type": "layer",
                "name": item["layer"],
                "gradient_norm":
                    item["tartanair_norm"],
                "cosine": item["cosine"],
                "conflict_ratio": item["conflict_ratio"],
            })

    with open(
        out / "gradient_cosine_matrix.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "tartanair", "kitti_sceneflow"])
        writer.writerow([
            "tartanair", 1.0, comparison["gradient_cosine"]
        ])
        writer.writerow([
            "kitti_sceneflow", comparison["gradient_cosine"], 1.0
        ])

    print("\n[PASS] E8-A diagnosis completed")
    print("Output:", out)


if __name__ == "__main__":
    main()
