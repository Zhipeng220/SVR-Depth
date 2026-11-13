import h5py
import numpy as np
import csv
import json
import io

from pathlib import Path
from PIL import Image

SRC = Path(
    "/root/autodl-tmp/datasets/VeloDepth-Evaluation/ScanNet.hdf5"
)

OUT = Path("outputs/full4_evaluation_inventory")
OUT.mkdir(parents=True, exist_ok=True)

FRAME_CSV = OUT / "scannet_frame_inventory_v2.csv"
SEQ_CSV = OUT / "scannet_sequence_inventory_v2.csv"
REPORT = OUT / "scannet_audit_v2.json"

frame_rows = []
sequence_rows = []

def get_entries(group):
    result = {}

    for name in group.keys():
        obj = group[name]

        if not isinstance(obj, h5py.Dataset):
            continue

        stem = Path(name).stem
        result[stem] = {
            "name": name,
            "shape": str(obj.shape),
            "dtype": str(obj.dtype),
            "ndim": obj.ndim,
        }

    return result

def decode_sample(group, name, kind):
    ds = group[name]

    raw = np.asarray(ds[()], dtype=np.uint8).tobytes()

    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        arr = np.asarray(im)

        result = {
            "mode": im.mode,
            "width": im.width,
            "height": im.height,
            "dtype": str(arr.dtype),
        }

        if kind == "depth":
            result.update({
                "min": int(arr.min()),
                "max": int(arr.max()),
                "zero_ratio": float(np.mean(arr == 0)),
            })

        return result

report = {
    "dataset": "ScanNet",
    "source": str(SRC),
    "sampling_for_evaluation": False,
    "scenes": 0,
    "rgb_frames": 0,
    "depth_frames": 0,
    "matched_frames": 0,
    "unmatched_rgb": 0,
    "unmatched_depth": 0,
    "decode_errors": [],
    "scenes_with_errors": [],
}

assert SRC.is_file(), f"Missing: {SRC}"

with h5py.File(SRC, "r") as f:

    scenes = sorted([
        k for k in f.keys()
        if k.startswith("scene")
        and isinstance(f[k], h5py.Group)
    ])

    report["scenes"] = len(scenes)

    print("Total scenes:", len(scenes))

    for idx, scene in enumerate(scenes, 1):

        s = f[scene]

        if "color" not in s or "depth" not in s:
            report["scenes_with_errors"].append(scene)
            continue

        cg = s["color"]
        dg = s["depth"]

        colors = get_entries(cg)
        depths = get_entries(dg)

        color_ids = set(colors)
        depth_ids = set(depths)

        matched = sorted(color_ids & depth_ids)
        missing_depth = sorted(color_ids - depth_ids)
        missing_rgb = sorted(depth_ids - color_ids)

        report["rgb_frames"] += len(colors)
        report["depth_frames"] += len(depths)
        report["matched_frames"] += len(matched)
        report["unmatched_rgb"] += len(missing_depth)
        report["unmatched_depth"] += len(missing_rgb)

        # 每一帧都写入清单，不抽样
        for fid in sorted(color_ids | depth_ids):

            c = colors.get(fid)
            d = depths.get(fid)

            frame_rows.append({
                "dataset": "ScanNet",
                "scene": scene,
                "frame_id": fid,
                "color_key": (
                    f"{scene}/color/{c['name']}" if c else ""
                ),
                "depth_key": (
                    f"{scene}/depth/{d['name']}" if d else ""
                ),
                "rgb_present": int(c is not None),
                "depth_present": int(d is not None),
                "paired": int(fid in color_ids and fid in depth_ids),
                "rgb_shape": c["shape"] if c else "",
                "rgb_dtype": c["dtype"] if c else "",
                "depth_shape": d["shape"] if d else "",
                "depth_dtype": d["dtype"] if d else "",
            })

        # 每个场景检查首、中、尾帧的实际解码
        sample_ids = sorted(color_ids & depth_ids)

        sample_ids = sorted(set(
            [sample_ids[0], sample_ids[len(sample_ids)//2],
             sample_ids[-1]]
        )) if sample_ids else []

        sample_results = []

        for fid in sample_ids:
            try:
                rgb = decode_sample(
                    cg, colors[fid]["name"], "color"
                )
                dep = decode_sample(
                    dg, depths[fid]["name"], "depth"
                )

                sample_results.append({
                    "frame_id": fid,
                    "rgb": rgb,
                    "depth": dep,
                })

            except Exception as e:
                report["decode_errors"].append({
                    "scene": scene,
                    "frame": fid,
                    "error": repr(e),
                })

        sequence_rows.append({
            "scene": scene,
            "rgb_frames": len(colors),
            "depth_frames": len(depths),
            "matched_frames": len(matched),
            "unmatched_rgb": len(missing_depth),
            "unmatched_depth": len(missing_rgb),
            "flow_frames": len(s["flow"]) if "flow" in s else 0,
            "flow_mask_frames": (
                len(s["flow_mask"]) if "flow_mask" in s else 0
            ),
            "decode_samples": len(sample_results),
            "sample_results": json.dumps(
                sample_results, ensure_ascii=False
            ),
        })

        print(
            f"[{idx:03d}/{len(scenes):03d}] {scene} "
            f"RGB={len(colors)} "
            f"Depth={len(depths)} "
            f"Matched={len(matched)}"
        )

# 输出逐帧清单
with FRAME_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=frame_rows[0].keys())
    w.writeheader()
    w.writerows(frame_rows)

# 输出逐场景清单
with SEQ_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=sequence_rows[0].keys())
    w.writeheader()
    w.writerows(sequence_rows)

report["scene_summary_rows"] = len(sequence_rows)
report["frame_inventory_rows"] = len(frame_rows)

report["all_frames_paired"] = (
    report["unmatched_rgb"] == 0
    and report["unmatched_depth"] == 0
)

report["decode_samples_pass"] = (
    len(report["decode_errors"]) == 0
)

with REPORT.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n========== SCANNET FULL AUDIT ==========")
print(json.dumps(report, indent=2, ensure_ascii=False))

print("\n[OUTPUT]", FRAME_CSV)
print("[OUTPUT]", SEQ_CSV)
print("[OUTPUT]", REPORT)
