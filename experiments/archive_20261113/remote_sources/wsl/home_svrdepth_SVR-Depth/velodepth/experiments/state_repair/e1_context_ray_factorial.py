#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E1: unified-ray / coordinate context factorial diagnostic for VeloDepth.

Runs a paired 2x2 spatial-feature/token-source factorial under:
  (1) legacy/local camera-ray prediction (no rays supplied to crop decoder)
  (2) unified full-frame ray field cropped to the same tile

Factors:
  A = local spatial features + local tokens
  B = full-frame spatial features cropped to tile + local tokens
  C = local spatial features + full-frame tokens cropped/preserved for tile
  D = full-frame spatial features and tokens, both cropped/preserved for tile

This is a mechanism diagnostic, not a deployment benchmark. Full-frame encoding
and full-frame ray generation are diagnostic references and must not be counted
as online method cost. Target indices come only from the frozen CSV manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "p1"))

import p2_3_v5_26g_motion_compensated_feature_reuse as base
from velodepth.models.velodepth import (
    VeloDepth, get_paddings, get_resize_factor
)

VERSION = "E1 context-ray factorial v1.1 (metric audit fix)"
GROUPS = {
    "A_local_spatial_local_token": ("local", "local"),
    "B_full_spatial_local_token": ("full_crop", "local"),
    "C_local_spatial_full_token": ("local", "full_crop"),
    "D_full_spatial_full_token": ("full_crop", "full_crop"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def read_rgb(path: Path) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Cannot read RGB image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).contiguous()


def read_gt_depth(path: Path, scale: float, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Cannot read depth image: {path}")
    if raw.ndim == 3:
        raw = raw[..., 0]
    depth = raw.astype(np.float32) / float(scale)
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    return depth, valid


def preprocess_gt_to_model_canvas(
    model, gt: np.ndarray, valid: np.ndarray, original_hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Apply the same pad/resize geometry as VeloDepth.infer; invalid padding stays masked."""
    h0, w0 = original_hw
    if gt.shape != (h0, w0):
        raise RuntimeError(f"RGB/GT size mismatch: RGB {(h0,w0)}, GT {gt.shape}")
    ratio_bounds = model.shape_constraints["ratio_bounds"]
    paddings, padded_hw = get_paddings((h0, w0), ratio_bounds)
    pl, pr, pt, pb = paddings
    padded_gt = cv2.copyMakeBorder(gt, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0)
    padded_valid = cv2.copyMakeBorder(
        valid.astype(np.uint8), pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=0
    )

    pixels_bounds = [
        model.shape_constraints["pixels_min"],
        model.shape_constraints["pixels_max"],
    ]
    if hasattr(model, "resolution_level"):
        pixels_range = pixels_bounds[1] - pixels_bounds[0]
        interval = pixels_range / 10
        pixels_bounds = (
            model.resolution_level * interval + pixels_bounds[0],
            (model.resolution_level + 1) * interval + pixels_bounds[0],
        )
    _, (new_h, new_w) = get_resize_factor(padded_hw, pixels_bounds)
    gt_rs = cv2.resize(padded_gt, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    valid_rs = cv2.resize(
        padded_valid, (new_w, new_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    meta = {
        "original_hw": [h0, w0],
        "padded_hw": list(padded_hw),
        "model_canvas_hw": [new_h, new_w],
        "paddings_lrtb": [pl, pr, pt, pb],
        "resize_geometry": "nearest GT resize after same pad/resize shape calculation",
    }
    return (
        torch.from_numpy(gt_rs.copy()).float().to("cuda"),
        torch.from_numpy(valid_rs.copy()).bool().to("cuda"),
        meta,
    )


def metric_depth(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> dict:
    valid = mask & torch.isfinite(pred) & torch.isfinite(gt) & (gt > 0)
    p = pred[valid].float()
    g = gt[valid].float()
    n = int(p.numel())
    if n == 0:
        return {"n_pixels": 0, "absrel": np.nan, "rmse": np.nan,
                "delta1": np.nan, "mae": np.nan}
    diff = p - g
    absrel = (diff.abs() / g.clamp_min(1e-6)).mean()
    rmse = torch.sqrt((diff.square()).mean())
    mae = diff.abs().mean()
    # Standard delta1 denominator is all valid GT pixels. Non-positive or
    # otherwise invalid predictions must count as failures, not be dropped.
    ratio = torch.full_like(p, float("inf"))
    positive = p > 0
    if positive.any():
        ratio[positive] = torch.maximum(
            p[positive] / g[positive].clamp_min(1e-6),
            g[positive] / p[positive].clamp_min(1e-6)
        )
    delta1 = (ratio < 1.25).float().mean()
    return {
        "n_pixels": n,
        "absrel": float(absrel.item()),
        "rmse": float(rmse.item()),
        "delta1": float(delta1.item()),
        "mae": float(mae.item()),
    }


def tensor_to_map(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    t = t.detach().float()
    if t.ndim >= 2 and tuple(t.shape[-2:]) == (h, w):
        return t.reshape(-1, h, w)[0]
    if t.ndim == 2 and t.shape[-1] == h * w:
        return t.reshape(-1, h, w)[0]
    raise ValueError(f"Cannot map tensor shape={tuple(t.shape)} to {(h,w)}")


def rays_from_output(output, h: int, w: int) -> torch.Tensor:
    rays = output[0].detach().float()
    if rays.ndim == 3 and rays.shape[1] == h * w and rays.shape[2] == 3:
        return rays.reshape(rays.shape[0], h, w, 3)[0].permute(2, 0, 1).contiguous()
    if rays.ndim == 4 and rays.shape[1] == 3 and tuple(rays.shape[-2:]) == (h, w):
        return rays[0]
    raise ValueError(f"Cannot map ray tensor shape={tuple(rays.shape)} to {(h,w)}")


def decode(model, inputs, features, tokens, device):
    outputs, _ = base.decode_with_features(model, inputs, features, tokens, device)
    h, w = inputs["image"].shape[-2:]
    radius = base.to_hw(outputs[1], h, w)
    rays = rays_from_output(outputs, h, w)
    depth = radius * rays[2]
    return outputs, radius, rays, depth


def crop_mask_regions(h: int, w: int, core_local, band: int, device):
    cy0, cy1, cx0, cx1 = core_local
    core = torch.zeros((h, w), dtype=torch.bool, device=device)
    core[cy0:cy1, cx0:cx1] = True
    tile = torch.ones((h, w), dtype=torch.bool, device=device)
    halo = tile & ~core
    expanded = torch.zeros_like(core)
    ey0, ey1 = max(0, cy0-band), min(h, cy1+band)
    ex0, ex1 = max(0, cx0-band), min(w, cx1+band)
    expanded[ey0:ey1, ex0:ex1] = True
    inner = torch.zeros_like(core)
    iy0, iy1 = min(h, cy0+band), max(0, cy1-band)
    ix0, ix1 = min(w, cx0+band), max(0, cx1-band)
    if iy1 > iy0 and ix1 > ix0:
        inner[iy0:iy1, ix0:ix1] = True
    boundary = expanded & ~inner
    return {"all_tile": tile, "core": core, "halo": halo, "boundary": boundary}


def radius_relative_l2(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> float:
    m = mask & torch.isfinite(pred) & torch.isfinite(ref)
    if not m.any():
        return float("nan")
    denom = torch.linalg.vector_norm(ref[m]).item()
    return float(torch.linalg.vector_norm((pred[m] - ref[m])).item() / max(denom, 1e-12))


def ray_angle_deg(pred: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor) -> dict:
    m = mask & torch.isfinite(pred).all(0) & torch.isfinite(ref).all(0)
    if not m.any():
        return {"ray_angle_mean_deg": np.nan, "ray_angle_p95_deg": np.nan}
    a = F.normalize(pred[:, m].T, dim=1)
    b = F.normalize(ref[:, m].T, dim=1)
    cos = (a * b).sum(1).clamp(-1, 1)
    angles = torch.rad2deg(torch.acos(cos))
    return {
        "ray_angle_mean_deg": float(angles.mean().item()),
        "ray_angle_p95_deg": float(torch.quantile(angles, 0.95).item()),
    }


def build_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "sequence", "requested_rgb_index", "actual_rgb_index",
        "rgb_path", "depth_path", "rgb_depth_dt_seconds", "status"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")
    if len(df) != 9:
        raise RuntimeError(f"Expected exactly 9 target rows; got {len(df)}")
    if not (df["status"].astype(str) == "MATCH").all():
        raise RuntimeError("Manifest contains non-MATCH rows; resolve before E1.")
    for col in ("rgb_path", "depth_path"):
        for item in df[col].astype(str):
            if not Path(item).is_file():
                raise RuntimeError(f"Manifest path missing: {item}")
    return df


def save_error_map(out_path: Path, gt, valid, preds: dict[str, torch.Tensor], title: str):
    names = list(preds)
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(4 * (len(names) + 1), 4))
    axes[0].imshow(np.where(valid.cpu().numpy(), gt.cpu().numpy(), np.nan))
    axes[0].set_title("GT depth (m)")
    for ax, name in zip(axes[1:], names):
        p = preds[name]
        err = (p - gt).abs()
        arr = err.detach().cpu().numpy()
        arr[~valid.cpu().numpy()] = np.nan
        im = ax.imshow(arr, vmin=0, vmax=np.nanpercentile(arr, 95) if np.isfinite(arr).any() else 1)
        ax.set_title(f"{name}\nabs error")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=VERSION)
    parser.add_argument("--manifest", default="outputs/state_repair/e1_target_manifest.csv")
    parser.add_argument("--resolution-level", type=int, default=0)
    parser.add_argument("--halos", default="32,64")
    parser.add_argument("--gt-scale", type=float, default=5000.0)
    parser.add_argument("--max-depth", type=float, default=10.0)
    parser.add_argument("--boundary-band", type=int, default=4)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("E1 formal run requires CUDA; current torch.cuda.is_available() is False.")
    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else ROOT
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.is_absolute():
        manifest_path = (root / args.manifest).resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        root / "outputs" / "state_repair" / time.strftime("e1_factorial_%Y%m%dT%H%M%SZ", time.gmtime())
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir()

    manifest = build_manifest(manifest_path)
    halos = sorted({int(x.strip()) for x in args.halos.split(",") if x.strip()})
    if halos != [32, 64]:
        raise RuntimeError("Frozen E1 protocol requires exactly halos 32,64.")
    if args.resolution_level < 0 or args.resolution_level >= 10:
        raise ValueError("--resolution-level must be in [0,10).")

    print(f"[1/4] Load model; manifest={manifest_path}")
    model = VeloDepth.from_pretrained("lpiccinelli/velodepth")
    device = torch.device("cuda")
    model.to(device).eval()
    model.resolution_level = args.resolution_level
    model.interpolation_mode = "bilinear"
    torch.set_grad_enabled(False)

    factorial_rows, ray_rows, target_rows = [], [], []
    source_script = root / "experiments/p1/p2_3_v5_26g_motion_compensated_feature_reuse.py"
    meta = {
        "experiment": VERSION,
        "status": "running",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(root),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_rows": len(manifest),
        "resolution_level": args.resolution_level,
        "halos": halos,
        "boundary_band_px": args.boundary_band,
        "gt_scale": args.gt_scale,
        "max_depth_m": args.max_depth,
        "metric_definitions": {
            "absrel": "mean(abs(pred-gt)/gt) over valid GT pixels and finite predictions",
            "rmse": "sqrt(mean((pred-gt)^2)) over valid GT pixels and finite predictions",
            "delta1": "fraction of all valid GT pixels with positive prediction and max(pred/gt,gt/pred)<1.25; non-positive predictions count as failures",
            "mae": "mean(abs(pred-gt)) over valid GT pixels and finite predictions",
        },
        "geometry_conditions": {
            "legacy_local_camera": "Do not provide rays to crop decoder; decoder predicts rays from crop tokens and crop shape.",
            "unified_full_frame_rays": "Supply the cropped full-frame reference ray field to the crop decoder.",
        },
        "factor_groups": GROUPS,
        "source_sha256": {
            str(Path(__file__).resolve().relative_to(root)): sha256(Path(__file__).resolve()),
            str(source_script.relative_to(root)): sha256(source_script),
            str((root / "velodepth/models/velodepth.py").relative_to(root)): sha256(root / "velodepth/models/velodepth.py"),
            str((root / "velodepth/models/decoder_velo.py").relative_to(root)): sha256(root / "velodepth/models/decoder_velo.py"),
        },
        "interpretation_limits": [
            "Full-frame reference encoding and rays are offline diagnostics, not deployable local-method cost.",
            "GT depth is nearest-resized to the model canvas using the same padding/resize geometry; metric values are evaluated on this canvas.",
            "The legacy geometry condition means local crop decoder predicts its own rays; unified condition injects the full-frame ray crop.",
            "E1 is a mechanism diagnostic on exposed Bonn development sequences, not held-out evaluation.",
        ],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for row_idx, row in manifest.iterrows():
        seq = str(row["sequence"])
        requested_idx = int(row["requested_rgb_index"])
        actual_idx = int(row["actual_rgb_index"])
        rgb_path, gt_path = Path(row["rgb_path"]), Path(row["depth_path"])
        print(f"\n[{row_idx+1}/9] {seq} requested={requested_idx} actual={actual_idx} RGB={rgb_path.name}")

        rgb_tensor = read_rgb(rgb_path)
        original_hw = tuple(rgb_tensor.shape[-2:])
        gt_raw, gt_valid_raw = read_gt_depth(gt_path, args.gt_scale, args.max_depth)
        gt_canvas, gt_valid, gt_meta = preprocess_gt_to_model_canvas(
            model, gt_raw, gt_valid_raw, original_hw
        )

        packed, metas = base.capture_window_payload(model, rgb_tensor)
        T = int(metas[0]["T"])
        if T != 1:
            raise RuntimeError(f"Expected T=1 for a target frame, got {T}")
        prepared = base.clone_payload(packed)
        model._set_image_paddings(prepared)
        full_inputs = base.frame_inputs_at(model, prepared, 0, T)
        h, w = full_inputs["image"].shape[-2:]
        if tuple(gt_canvas.shape) != (h, w):
            raise RuntimeError(f"GT canvas {(tuple(gt_canvas.shape))} != model canvas {(h,w)}")

        stride, stride_candidates = base.discover_patch_stride(model.pixel_encoder)
        if stride is None:
            raise RuntimeError("Could not infer pixel encoder patch stride.")
        geometry = {halo: base.choose_core_and_tile(h, w, stride, halo) for halo in halos}
        core = geometry[halos[0]]["core"]
        if any(g["core"] != core for g in geometry.values()):
            raise RuntimeError("Core geometry differs between Halo settings.")

        # Full-frame same-target reference, encoded once.
        full_features, full_tokens, full_encoder_ms = base.encode_grouped_features(
            model, full_inputs, device
        )
        full_out, full_radius, full_rays, full_depth = decode(
            model, full_inputs, full_features, full_tokens, device
        )
        if tuple(full_depth.shape) != (h, w):
            raise RuntimeError(f"Full depth shape {tuple(full_depth.shape)} != {(h,w)}")

        target_rows.append({
            "sequence": seq, "requested_rgb_index": requested_idx,
            "actual_rgb_index": actual_idx,
            "rgb_path": str(rgb_path), "depth_path": str(gt_path),
            "rgb_depth_dt_seconds": float(row["rgb_depth_dt_seconds"]),
            "model_canvas_h": h, "model_canvas_w": w,
            "patch_stride": stride, "patch_stride_candidates": json.dumps(stride_candidates),
            "full_encoder_ms": full_encoder_ms,
            **gt_meta,
            "n_gt_valid_pixels": int(gt_valid.sum().item()),
        })

        # Full-frame reference row for each halo/region is computed on the matching tile.
        for halo in halos:
            core_y0, core_y1, core_x0, core_x1 = core
            ty0, ty1, tx0, tx1 = geometry[halo]["tile"]
            tile_h, tile_w = ty1 - ty0, tx1 - tx0
            tile_box = geometry[halo]["tile"]
            core_local = (core_y0-ty0, core_y1-ty0, core_x0-tx0, core_x1-tx0)
            region_masks = crop_mask_regions(tile_h, tile_w, core_local, args.boundary_band, device)
            tile_gt = gt_canvas[ty0:ty1, tx0:tx1]
            tile_valid = gt_valid[ty0:ty1, tx0:tx1]
            tile_ref_radius = full_radius[ty0:ty1, tx0:tx1]
            tile_ref_rays = full_rays[:, ty0:ty1, tx0:tx1]
            tile_ref_depth = full_depth[ty0:ty1, tx0:tx1]
            for region, local_mask in region_masks.items():
                full_metric = metric_depth(
                    tile_ref_depth, tile_gt, tile_valid & local_mask
                )
                factorial_rows.append({
                    "sequence": seq, "requested_rgb_index": requested_idx,
                    "actual_rgb_index": actual_idx, "geometry_condition": "full_frame_reference",
                    "group": "FULL_FRAME_REFERENCE", "halo_px": halo,
                    "region": region, **full_metric,
                    "radius_relative_l2_to_full": 0.0,
                    "ray_angle_mean_deg": 0.0, "ray_angle_p95_deg": 0.0,
                })

            crop_inputs = base.crop_frame_inputs(full_inputs, (h, w), tile_box)
            local_features, local_tokens, local_encoder_ms = base.encode_grouped_features(
                model, crop_inputs, device
            )
            full_crop_features, audit_f = base.crop_encoded_collection(
                full_features, (h, w), tile_box, stride, "features"
            )
            full_crop_tokens, audit_t = base.crop_encoded_collection(
                full_tokens, (h, w), tile_box, stride, "tokens"
            )

            # A-D are evaluated under both geometry conditions from identical encodings.
            for geom_condition in ("legacy_local_camera", "unified_full_frame_rays"):
                geom_inputs = base.clone_payload(crop_inputs)
                if geom_condition == "unified_full_frame_rays":
                    geom_inputs["rays"] = full_rays[:, ty0:ty1, tx0:tx1].unsqueeze(0).contiguous()
                variant_preds = {}
                for group_name, (spatial_src, token_src) in GROUPS.items():
                    features = local_features if spatial_src == "local" else full_crop_features
                    tokens = local_tokens if token_src == "local" else full_crop_tokens
                    outputs, radius, rays, depth = decode(
                        model, geom_inputs, features, tokens, device
                    )
                    variant_preds[group_name] = depth.detach()
                    ray_audit = ray_angle_deg(rays, tile_ref_rays, region_masks["all_tile"])
                    for region, local_mask in region_masks.items():
                        common = tile_valid & local_mask
                        metrics = metric_depth(depth, tile_gt, common)
                        factorial_rows.append({
                            "sequence": seq, "requested_rgb_index": requested_idx,
                            "actual_rgb_index": actual_idx,
                            "geometry_condition": geom_condition,
                            "group": group_name, "halo_px": halo, "region": region,
                            **metrics,
                            "radius_relative_l2_to_full": radius_relative_l2(
                                radius, tile_ref_radius, common
                            ),
                            **ray_angle_deg(rays, tile_ref_rays, common),
                        })
                    ray_rows.append({
                        "sequence": seq, "requested_rgb_index": requested_idx,
                        "actual_rgb_index": actual_idx, "geometry_condition": geom_condition,
                        "group": group_name, "halo_px": halo,
                        "ray_angle_mean_deg_all_tile": ray_audit["ray_angle_mean_deg"],
                        "ray_angle_p95_deg_all_tile": ray_audit["ray_angle_p95_deg"],
                        "radius_relative_l2_all_tile": radius_relative_l2(
                            radius, tile_ref_radius, tile_valid
                        ),
                        "local_encoder_ms": local_encoder_ms,
                        "full_encoder_ms_reference_only": full_encoder_ms,
                        "tile_area_ratio": float(tile_h * tile_w / (h * w)),
                        "n_valid_gt_tile": int((tile_valid & region_masks["all_tile"]).sum().item()),
                    })

                # Save maps for four groups, with common GT and tile coordinate system.
                if geom_condition == "unified_full_frame_rays":
                    map_name = f"{seq}_req{requested_idx}_actual{actual_idx}_halo{halo}_unified.png"
                    save_error_map(
                        plot_dir / map_name, tile_gt, tile_valid & region_masks["all_tile"],
                        variant_preds,
                        f"{seq} RGB {actual_idx}; Halo {halo}; unified rays"
                    )

        del packed, prepared, full_inputs, full_features, full_tokens, full_out
        torch.cuda.empty_cache()

    df = pd.DataFrame(factorial_rows)
    ray_df = pd.DataFrame(ray_rows)
    df.to_csv(out_dir / "e1_context_factorial.csv", index=False, encoding="utf-8-sig")
    ray_df.to_csv(out_dir / "e1_ray_coordinate_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(target_rows).to_csv(out_dir / "e1_target_audit.csv", index=False, encoding="utf-8-sig")

    # Paired interaction table and plot: core AbsRel only, matched target / halo / geometry.
    core_df = df[df["region"] == "core"].copy()
    pivot = core_df.pivot_table(
        index=["sequence", "requested_rgb_index", "actual_rgb_index", "geometry_condition", "halo_px"],
        columns="group", values="absrel", aggfunc="first"
    ).reset_index()
    required_groups = list(GROUPS)
    for name in required_groups:
        if name not in pivot:
            pivot[name] = np.nan
    # Keep the named effect as a difference, not the raw B score.
    pivot["spatial_source_effect_local_tokens"] = (
        pivot["B_full_spatial_local_token"] - pivot["A_local_spatial_local_token"]
    )
    # Explicit paired effects with consistent signs: B-A is full-vs-local spatial source at local tokens;
    # D-C is full-vs-local spatial source at full tokens; interaction=(A-B)-(C-D).
    A = pivot["A_local_spatial_local_token"]
    B = pivot["B_full_spatial_local_token"]
    C = pivot["C_local_spatial_full_token"]
    D = pivot["D_full_spatial_full_token"]
    pivot["spatial_full_minus_local_at_local_tokens"] = B - A
    pivot["spatial_full_minus_local_at_full_tokens"] = D - C
    pivot["token_full_minus_local_at_local_spatial"] = C - A
    pivot["token_full_minus_local_at_full_spatial"] = D - B
    pivot["factorial_interaction_(A-B)-(C-D)"] = (A - B) - (C - D)
    pivot.to_csv(out_dir / "e1_paired_interactions.csv", index=False, encoding="utf-8-sig")

    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        keys = pivot.groupby(["geometry_condition", "halo_px"], dropna=False)
        for (geom, halo), sub in keys:
            vals = sub["factorial_interaction_(A-B)-(C-D)"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                label_text = geom.replace("_", "\\n") + f" H{halo}"
                ax.scatter(
                    [label_text] * len(vals), vals,
                    label=f"{geom} Halo {halo}", alpha=0.8
                )
        ax.axhline(0, linewidth=1)
        ax.set_ylabel("Factorial interaction in core AbsRel")
        ax.set_title("Paired E1 interaction by target")
        ax.tick_params(axis="x", labelrotation=0)
        fig.tight_layout()
        fig.savefig(plot_dir / "e1_paired_interaction.png", dpi=160)
        plt.close(fig)

    meta["status"] = "completed"
    meta["outputs"] = [
        "e1_context_factorial.csv", "e1_ray_coordinate_audit.csv",
        "e1_target_audit.csv", "e1_paired_interactions.csv", "plots/"
    ]
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    report = [
        "# E1 Context-Ray Factorial Report",
        "",
        f"- Status: `{meta['status']}`",
        f"- Manifest: `{manifest_path}`",
        f"- Target slots: {len(manifest)}",
        f"- Groups: A/B/C/D crossed with legacy and unified-ray conditions",
        f"- Halo: {halos}",
        "- Full-frame reference encoder/rays are diagnostic oracles, not deployment cost.",
        "- GT is evaluated on the padded/resized model canvas with nearest-neighbor depth resizing.",
        "- Inspect paired target rows, not only the aggregate means.",
        "",
        "## Files",
        "- `e1_context_factorial.csv`: per target, group, geometry, halo and region GT metrics + radius/ray audits.",
        "- `e1_ray_coordinate_audit.csv`: ray angle, radius deviation and encoder/tile metadata.",
        "- `e1_target_audit.csv`: target provenance and GT canvas details.",
        "- `e1_paired_interactions.csv`: paired factorial effects in the core region.",
        "- `plots/`: GT error maps and paired interaction plot.",
        "",
        "## Interpretation guardrails",
        "- E1 is a diagnostic on exposed Bonn development sequences; it is not held-out evidence.",
        "- If the unified-ray condition changes results, do not attribute the difference to a learned feature mechanism until the ray audit is inspected.",
        "- Positive/negative effects are descriptive with 9 target slots; report per-target variation and do not claim statistical confirmation.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nE1 output directory: {out_dir}")
    print("E1 status: completed")


if __name__ == "__main__":
    main()
