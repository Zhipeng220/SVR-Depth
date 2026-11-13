#!/usr/bin/env python3
"""Build A/B/C 20-frame experiment configs from generated manifests."""
from __future__ import annotations
import argparse, json
from pathlib import Path


def read_paths(report_path: Path, dataset_id: str):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report["manifests"]["train"], report["manifests"]["val"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tartanair-report", required=True)
    p.add_argument("--kitti-report", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--window-frames", type=int, default=20)
    p.add_argument("--tbptt-frames", type=int, default=4)
    a = p.parse_args()
    tartan_train, tartan_val = read_paths(Path(a.tartanair_report), "tartanair")
    kitti_train, kitti_val = read_paths(Path(a.kitti_report), "kitti_sceneflow")
    out = Path(a.out_dir).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    groups = {
        "A_tartanair_only": {"dataset_ids": ["tartanair"], "train": {"tartanair": tartan_train}, "val": {"tartanair": tartan_val}, "weights": {"tartanair": 1.0}},
        "B_kitti_only": {"dataset_ids": ["kitti_sceneflow"], "train": {"kitti_sceneflow": kitti_train}, "val": {"kitti_sceneflow": kitti_val}, "weights": {"kitti_sceneflow": 1.0}},
        "C_tartanair_kitti": {"dataset_ids": ["tartanair", "kitti_sceneflow"], "train": {"tartanair": tartan_train, "kitti_sceneflow": kitti_train}, "val": {"tartanair": tartan_val, "kitti_sceneflow": kitti_val}, "weights": {"tartanair": 0.5, "kitti_sceneflow": 0.5}},
    }
    for name, g in groups.items():
        payload = {
            "schema_version": 1,
            "protocol": "svr-depth-crosssource-v2-tartanair-kitti-sceneflow",
            "dataset_ids": g["dataset_ids"],
            "manifests": {"train": g["train"], "val": g["val"]},
            "training": {"window_frames": a.window_frames, "tbptt_frames": a.tbptt_frames,
                         "min_supervised_frames": 1, "dataset_weights": g["weights"],
                         "sampling": "dataset-weighted -> scene-uniform -> sequence-uniform -> window-uniform"},
        }
        (out / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[PASS] wrote {len(groups)} configs to {out}")

if __name__ == "__main__": main()
