#!/usr/bin/env python3
"""
VeloDepth video demo:
- Loads a video (or folder of frames)
- Generates backward optical flow using OpenCV DIS (no RAFT at inference)
- Runs the VeloDepth temporal model on the whole sequence (T, C, H, W)
- Optionally uses a constant camera JSON for the clip
- Saves RGB / depth / rays frames and (optionally) a PLY for a chosen frame
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from velodepth.utils.camera import (Fisheye624, Spherical, OPENCV, Pinhole, MEI, Camera)
from velodepth.utils.visualization import colorize, save_file_ply


def _read_video_cv2(path: str, max_frames: int | None = None, stride: int = 1) -> np.ndarray:
    """
    returns frames as (T, H, W, 3) uint8 RGB
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1

    cap.release()
    if len(frames) == 0:
        raise ValueError("No frames read from video.")
    return np.stack(frames, axis=0).astype(np.uint8)


def _read_frames_folder(path: str, max_frames: int | None = None, stride: int = 1) -> np.ndarray:
    """
    Path should point to a folder containing images.
    returns frames as (T, H, W, 3) uint8 RGB
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(path, e)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images in folder: {path}")

    frames = []
    for i, fp in enumerate(files):
        if i % stride != 0:
            continue
        img = Image.open(fp).convert("RGB")
        frames.append(np.array(img))
        if max_frames is not None and len(frames) >= max_frames:
            break
    return np.stack(frames, axis=0).astype(np.uint8)


def _load_camera(camera_json: str, H: int, W: int, device: torch.device) -> Camera:
    """
    Load a single camera spec for the whole clip.
    JSON format expected:
    {
        "name": "Fisheye624" | "Spherical" | "OPENCV" | "Pinhole" | "MEI",
        "params": [...],  # per your camera class
    }
    """
    with open(camera_json, "r") as f:
        cam = json.load(f)
    name = cam["name"]
    params = torch.tensor(cam["params"], dtype=torch.float32, device=device)
    camera_types = {
        "Fisheye624": Fisheye624,
        "Spherical": Spherical,
        "OPENCV": OPENCV,
        "Pinhole": Pinhole,
        "MEI": MEI,
    }
    if name not in camera_types:
        raise ValueError(f"Unknown camera type: {name}")
    return camera_types[name](params=params).to(device)


def _ensure_outdir(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_frame_outputs(outdir: Path, base: str, rgb_tc_hw: torch.Tensor,
                        depth_tc_hw: torch.Tensor, rays_tc_hw: torch.Tensor,
                        t: int) -> None:
    """
    Save RGB/depth/rays for frame t.
    rgb_tc_hw: (3,H,W) uint8
    depth_tc_hw: (1,H,W) float
    rays_tc_hw: (3,H,W) float in [-1,1] approx
    """
    # denormalize RGB for saving
    rgb_u8 = rgb_tc_hw.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # depth color
    depth_np = depth_tc_hw.cpu().numpy()

    np.save(
        outdir / f"{base}_{t:06d}_depth.npy",
        depth_np.squeeze().astype(np.float32)
    )
    depth_color = colorize(depth_np.squeeze())

    # rays to 0..255 visualization
    rays_vis = ((rays_tc_hw.clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy()
    rays_u8 = rays_vis.astype(np.uint8)

    Image.fromarray(rgb_u8).save(outdir / f"{base}_{t:06d}_rgb.png")
    Image.fromarray(depth_color).save(outdir / f"{base}_{t:06d}_depth.png")
    Image.fromarray(rays_u8).save(outdir / f"{base}_{t:06d}_rays.png")

@torch.no_grad()
def run_velodepth_on_sequence(
    model: torch.nn.Module,
    frames_rgb_u8: np.ndarray,          # (T,H,W,3) uint8 RGB
    camera: Camera | None = None,
    normalize_inputs: bool = True,      # passed straight to model.infer(...)
    save_dir: str | None = None,
    save_ply_frame: int | None = None,
) -> dict:
    """
    Just call model.infer(...) on a TCHW tensor. No optical-flow here.
    Returns per-frame outputs stacked back to (T, C, H, W).
    """
    device = getattr(model, "device", next(model.parameters()).device)
    outdir = _ensure_outdir(save_dir)

    # (T,3,H, W) in [0,1]
    rgbs_raw_tc_hw = torch.from_numpy(frames_rgb_u8).to(device).permute(0, 3, 1, 2)

    # infer takes rgbs=TCHW and returns BTCHW outputs (depth/points/rays/distance)
    outputs = model.infer(
        rgbs=rgbs_raw_tc_hw,       # raw in [0,1]
        camera=camera,             # can be Camera or BatchCamera; infer will handle it
        normalize=normalize_inputs # let infer do ImageNet norm if True
    )

    # Optional saving
    if outdir is not None:
        base = "velodepth"
        T = rgbs_raw_tc_hw.shape[0]
        for t in range(T):
            _save_frame_outputs(outdir, base, rgbs_raw_tc_hw[t], outputs["depth"][t], outputs["rays"][t], t)

        if save_ply_frame is not None and 0 <= save_ply_frame < T:
            t = save_ply_frame
            pts = outputs["points"][t].permute(1, 2, 0).reshape(-1, 3).detach().cpu().numpy()
            # The input frame is already uint8 RGB; use it directly for point colors.
            rgb = frames_rgb_u8[t].reshape(-1, 3).astype(np.uint8, copy=False)
            save_file_ply(pts, rgb, str(outdir / f"{base}_{t:06d}.ply"))

    return outputs


def parse_args():
    p = argparse.ArgumentParser(description="VeloDepth Video Demo (DIS optical flow)")
    p.add_argument("--video", type=str, required=True,
                   help="Path to input video file OR a folder with frames.")
    p.add_argument("--camera_json", type=str, default=None,
                   help="Optional camera JSON applied to all frames.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Optional output directory to save RGB/depth/rays (per-frame) and PLY.")
    p.add_argument("--ply_frame", type=int, default=None,
                   help="If set, save a PLY for this frame index.")
    p.add_argument("--max_frames", type=int, default=None, help="Limit number of frames.")
    p.add_argument("--stride", type=int, default=1, help="Sample every Nth frame.")
    p.add_argument("--resolution_level", type=int, default=9, help="Model resolution bucket [0..9].")
    p.add_argument("--interpolation", type=str, default="bilinear", choices=["bilinear", "bicubic"],
                   help="Output upsampling mode inside the model.")
    p.add_argument("--frames_from_folder", action="store_true",
                   help="Interpret --video as a folder of frames instead of a video file.")
    p.add_argument("--model-id", default="lpiccinelli/velodepth",
                   help="Hugging Face model ID or local snapshot directory.")
    p.add_argument("--revision", default=None, help="Optional Hugging Face revision.")
    p.add_argument("--repair-mode", choices=["off", "pre_readout", "post_readout"],
                   default="off", help="Enable the SVR-Depth persistent feature repair head.")
    p.add_argument("--repair-checkpoint", default=None,
                   help="Trained repair-head checkpoint (.pt/.pth/.safetensors).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max_frames must be >= 1")
    if args.ply_frame is not None and args.ply_frame < 0:
        raise ValueError("--ply_frame must be >= 0")
    print("Torch:", torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from svr_depth.model import load_model
    model = load_model(
        args.model_id,
        revision=args.revision,
        repair_mode=args.repair_mode,
        repair_checkpoint=args.repair_checkpoint,
        device=device,
    )

    # Optional model runtime knobs
    model.resolution_level = args.resolution_level
    model.interpolation_mode = args.interpolation

    # Load frames
    if args.frames_from_folder:
        frames_rgb = _read_frames_folder(args.video, max_frames=args.max_frames, stride=args.stride)
    else:
        frames_rgb = _read_video_cv2(args.video, max_frames=args.max_frames, stride=args.stride)

    T, H, W, _ = frames_rgb.shape
    print(f"Loaded {T} frames at {W}x{H}")

    # Load camera (optional)
    camera = None
    if args.camera_json is not None:
        camera = _load_camera(args.camera_json, H, W, device)
        print(f"Loaded camera from {args.camera_json}")

    # Run the sequence
    outputs = run_velodepth_on_sequence(
        model,
        frames_rgb_u8=frames_rgb,
        camera=camera,
        normalize_inputs=True,
        save_dir=args.out_dir,
        save_ply_frame=args.ply_frame,
    )

    print("Output keys:", list(outputs.keys()))
    print("Done.")


if __name__ == "__main__":
    main()
