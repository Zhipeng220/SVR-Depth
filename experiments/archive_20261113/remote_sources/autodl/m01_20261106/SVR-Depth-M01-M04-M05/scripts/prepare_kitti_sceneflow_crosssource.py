#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svr_depth.datasets.kitti_sceneflow import build_kitti_sceneflow_manifests


def main():
    p = argparse.ArgumentParser(description="Build 20-frame KITTI Scene Flow 2015 MultiView manifests with metric camera-z GT")
    p.add_argument("--multiview-root", required=True, help="Extracted data_scene_flow_multiview root")
    p.add_argument("--main-root", required=True, help="Extracted data_scene_flow root (disp_occ_0/disp_occ_1)")
    p.add_argument("--calib-root", required=True, help="Extracted data_scene_flow_calib root")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20261103)
    a = p.parse_args()
    result = build_kitti_sceneflow_manifests(
        a.multiview_root, a.main_root, a.calib_root, a.out_dir,
        fps=a.fps, val_fraction=a.val_fraction, seed=a.seed,
    )
    out = Path(a.out_dir).expanduser().resolve() / "kitti_sceneflow_manifest_build.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out), "stats": result["stats"]}, indent=2))

if __name__ == "__main__": main()
