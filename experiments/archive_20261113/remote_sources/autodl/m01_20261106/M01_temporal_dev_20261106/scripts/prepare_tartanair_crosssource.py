#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svr_depth.datasets.native_tartanair import (
    build_tartanair_hdf5_manifests, build_tartanair_manifests,
)


def main():
    p = argparse.ArgumentParser(description="Build CrossSource-v2 manifests from raw TartanAir V1 or VeloDepth HDF5")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--root", help="Raw TartanAir root containing ENV/DIFFICULTY/Pxxx/{image_left,depth_left}")
    source.add_argument("--hdf5", help="VeloDepth-format TartanAir.hdf5 with embedded train.txt and sequences.json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-envs", required=True, help="Comma-separated held-out environment names")
    p.add_argument("--fps", type=float, required=True, help="Source capture FPS; explicitly confirm from selected release")
    p.add_argument("--depth-type", choices=["camera_z"], required=True,
                   help="Explicitly confirm camera-z semantics; raw radial/range depth must be converted first")
    p.add_argument("--max-depth-m", type=float, default=80.0)
    p.add_argument("--min-depth-m", type=float, default=0.05)
    p.add_argument("--intrinsics", default=None, help="Raw layout only: optional fx,fy,cx,cy")
    p.add_argument("--min-frames", type=int, default=20)
    p.add_argument("--hdf5-depth-scale", type=float, default=1000.0,
                   help="VeloDepth TartanAir loader scale (default 1000 raw units per metre)")
    p.add_argument("--hdf5-min-depth-m", type=float, default=0.01)
    p.add_argument("--hdf5-max-depth-m", type=float, default=512.0)
    p.add_argument("--max-sequences", type=int, default=None)
    a = p.parse_args()
    if a.depth_type != "camera_z":
        p.error("Only camera_z is accepted by the current P_multi training loss")
    if a.hdf5:
        result = build_tartanair_hdf5_manifests(
            a.hdf5, a.out_dir, val_envs=[x for x in a.val_envs.split(",") if x.strip()],
            fps=a.fps, depth_scale=a.hdf5_depth_scale,
            min_depth_m=a.hdf5_min_depth_m, max_depth_m=a.hdf5_max_depth_m,
            min_frames=a.min_frames, max_sequences=a.max_sequences,
        )
    else:
        k = None if not a.intrinsics else tuple(float(x) for x in a.intrinsics.split(","))
        if k is not None and len(k) != 4:
            p.error("--intrinsics must be fx,fy,cx,cy")
        result = build_tartanair_manifests(
            a.root, a.out_dir, val_envs=[x for x in a.val_envs.split(",") if x.strip()],
            fps=a.fps, depth_type=a.depth_type, max_depth_m=a.max_depth_m,
            min_depth_m=a.min_depth_m, intrinsics=k, min_frames=a.min_frames,
        )
    out = Path(a.out_dir).expanduser().resolve() / "tartanair_manifest_build.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "stats": result["stats"]}, indent=2))

if __name__ == "__main__": main()
