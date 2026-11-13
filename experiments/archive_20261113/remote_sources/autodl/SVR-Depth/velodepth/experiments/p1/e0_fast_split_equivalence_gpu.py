#!/usr/bin/env python3
"""GPU audit: compare the refactored fast split against the pre-split implementation.

Run from the VeloDepth repository root after applying decoder_fast_split.patch:
  python experiments/p1/e0_fast_split_equivalence_gpu.py --frames 16 --resolution_level 0

This audit does not train, alter weights, or measure end-to-end latency.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

from velodepth.models.velodepth import VeloDepth


def load_rgb(folder, count):
    paths = sorted(Path(folder).glob("*.png"))[:count]
    if len(paths) != count:
        raise RuntimeError(f"Expected {count} PNG images, found {len(paths)} in {folder}")
    frames = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Failed reading image: {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).contiguous()


def max_abs_diff(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            raise AssertionError(f"Tensor shape mismatch: {tuple(a.shape)} != {tuple(b.shape)}")
        if a.numel() == 0:
            return 0.0
        return float((a - b).abs().max().item())
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError(f"Sequence length mismatch: {len(a)} != {len(b)}")
        return max([max_abs_diff(x, y) for x, y in zip(a, b)] or [0.0])
    raise TypeError(f"Unsupported comparison types: {type(a)} vs {type(b)}")


def reference_pre_split_forward_fast(radial, lateral_features_teacher,
                                     features_residual, flow_bwd, flow_features):
    """Literal reference copy of the pre-refactor RadialModule.forward_fast."""
    features_propagated = []
    for i, (init_, residual) in enumerate(
        zip(lateral_features_teacher, features_residual)
    ):
        flow_bwd = F.interpolate(
            flow_bwd, size=init_.shape[-2:], mode="bilinear", align_corners=True
        )
        # flow2grid_sample mutates its input, so preserve the source clone exactly.
        from velodepth.models.decoder_velo import flow2grid_sample
        warp_grid = flow2grid_sample(flow_bwd.clone())
        init_warped = F.grid_sample(
            init_, warp_grid, mode="bilinear", align_corners=True
        )
        corrected_features = radial.fusion[i](
            init_warped,
            residual,
            original_features=init_,
            previous=residual if i == 0 else features_propagated[-1],
            flow_features=flow_features,
        )
        features_propagated.append(corrected_features)

    lateral_features = [features_propagated[0]] + [
        radial.process_features[i](features_propagated[i + 1])
        for i in range(len(radial.ups))
    ]
    latents = lateral_features[0]
    features_student = []
    for i, up in enumerate(radial.ups):
        latents = latents + lateral_features[i + 1]
        latents = up(latents)
        features_student.append(latents)
    logdepth_student = radial.depth_proj(features_student)
    logconf_student = logdepth_student
    return logdepth_student, logconf_student, features_propagated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", default="/root/autodl-tmp/datasets/rgbd_bonn_crowd2/rgb")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--resolution_level", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--out", default="outputs/e0_state_contract/e0_fast_split_equivalence")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this model audit; do not run in no-card mode.")

    torch.set_grad_enabled(False)
    device = torch.device("cuda")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    model.to(device)
    model.eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"

    rgbs = load_rgb(args.rgb, args.frames)
    rows = []
    original_forward_fast = model.pixel_decoder.forward_fast

    def audited_forward_fast(inputs, image_metas):
        # Run the refactored native path first; it is the output returned to VeloDepth.
        native_radius, native_confidence, native_features = original_forward_fast(
            inputs, image_metas
        )

        B, _T = image_metas[0]["B"], image_metas[0]["T"]
        teacher_features = [
            x.reshape(B, -1, *x.shape[1:])[:, :1].reshape(-1, *x.shape[1:])
            for x in inputs["features_teacher"]
        ]
        residual_features = inputs["features_residual"][::-1]
        radial = model.pixel_decoder.radial_module

        # Independently run the pre-refactor reference on the exact same tensors.
        ref_lograd, ref_logconf, ref_features = reference_pre_split_forward_fast(
            radial,
            teacher_features,
            residual_features,
            inputs["flow_bwd_refined"],
            inputs["flow_features"],
        )
        ref_radius = torch.exp(ref_lograd.clip(min=-8.0, max=8.0) + 2.0)
        ref_confidence = torch.exp(ref_logconf.clip(min=-8.0, max=10.0))

        d_radius = max_abs_diff(native_radius, ref_radius)
        d_conf = max_abs_diff(native_confidence, ref_confidence)
        d_features = max_abs_diff(native_features, ref_features)
        row = {
            "call_index": len(rows),
            "radius_max_abs": d_radius,
            "confidence_max_abs": d_conf,
            "features_max_abs": d_features,
        }
        rows.append(row)
        print(
            f"fast_call={row['call_index']} radius={d_radius:.9g} "
            f"confidence={d_conf:.9g} features={d_features:.9g}"
        )
        return native_radius, native_confidence, native_features

    model.pixel_decoder.forward_fast = audited_forward_fast
    try:
        # infer() follows the real input preprocessing and streaming schedule.
        _ = model.infer(rgbs, normalize=True)
    finally:
        model.pixel_decoder.forward_fast = original_forward_fast

    if not rows:
        raise RuntimeError("No fast-path calls were audited; verify the input sequence and model path.")

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_base.with_suffix(".csv")
    json_path = out_base.with_suffix(".json")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    maxima = {
        key: max(row[key] for row in rows)
        for key in ("radius_max_abs", "confidence_max_abs", "features_max_abs")
    }
    summary = {
        "frames_requested": args.frames,
        "resolution_level": args.resolution_level,
        "tolerance": args.tolerance,
        "fast_call_count": len(rows),
        "maxima": maxima,
        "passed": max(maxima.values()) <= args.tolerance,
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise AssertionError(
            "E0 fast split differs from the pre-split reference. Do not proceed to E2."
        )
    print("PASS: refactored fast split matches the pre-split reference.")


if __name__ == "__main__":
    main()
