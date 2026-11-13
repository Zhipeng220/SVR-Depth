import csv
import json
import io
import re
import hashlib
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
from PIL import Image

DATA = Path("/root/autodl-tmp/datasets/VeloDepth-Evaluation")
OUT = Path("outputs/full4_evaluation_inventory")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "Bonn": DATA / "Bonn.hdf5",
    "TUM": DATA / "TUM.hdf5",
    "ScanNet": DATA / "ScanNet.hdf5",
    "Sintel": DATA / "Sintel.hdf5",
}

FLAT_SUFFIXES = {
    "Bonn": {
        "rgb": "_c.jpg",
        "depth": "_d.png",
        "flow": "_f.png",
        "flow_mask": "_fm.png",
    },
    "TUM": {
        "rgb": "_c.jpg",
        "depth": "_d.png",
        "flow": "_f.png",
        "flow_mask": "_fm.png",
    },
    "Sintel": {
        "rgb": "_rgb.jpg",
        "depth": "_depth.png",
        "flow": "_flow_fwd.png",
        "flow_mask": "_flow_fwd_mask.png",
    },
}

SCANNET_BRANCHES = {
    "rgb": "color",
    "depth": "depth",
    "flow": "flow",
    "flow_mask": "flow_mask",
}

manifest = []
sequence_rows = []
root_audit = {}
depth_samples = {}

def decode_bytes(ds):
    value = ds[()]
    if isinstance(value, bytes):
        return value
    return np.asarray(value, dtype=np.uint8).tobytes()

def inspect_depth(f, key):
    raw = decode_bytes(f[key])
    im = Image.open(io.BytesIO(raw))
    im.load()
    arr = np.asarray(im)

    item = {
        "key": key,
        "mode": im.mode,
        "size": list(im.size),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }

    if arr.ndim == 2:
        item.update({
            "min": float(np.min(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p99": float(np.percentile(arr, 99)),
            "max": float(np.max(arr)),
            "zero_ratio": float(np.mean(arr == 0)),
        })
        item["interpretation"] = "numeric_depth_encoding_unverified"
    elif arr.ndim == 3:
        item["channel_min"] = arr.min(axis=(0, 1)).tolist()
        item["channel_max"] = arr.max(axis=(0, 1)).tolist()
        item["center_pixel"] = arr[
            arr.shape[0] // 2, arr.shape[1] // 2
        ].tolist()
        item["interpretation"] = "multichannel_encoding_unresolved"

    return item

def flat_sequences(f, dataset):
    sequences = {}

    suffixes = FLAT_SUFFIXES[dataset]

    for seq in f.keys():
        node = f[seq]
        if not isinstance(node, h5py.Group):
            continue

        roles = defaultdict(dict)

        for name in node.keys():
            for role, suffix in suffixes.items():
                if name.endswith(suffix):
                    frame_id = name[:-len(suffix)]
                    roles[role][frame_id] = f"{seq}/{name}"
                    break

        if any(roles.values()):
            sequences[seq] = roles

    return sequences

def scannet_sequences(f):
    sequences = {}

    for scene in f.keys():
        node = f[scene]

        if not isinstance(node, h5py.Group):
            continue

        if not all(
            branch in node and
            isinstance(node[branch], h5py.Group)
            for branch in SCANNET_BRANCHES.values()
        ):
            continue

        roles = defaultdict(dict)

        for role, branch in SCANNET_BRANCHES.items():
            for name in node[branch].keys():
                frame_id = Path(name).stem
                roles[role][frame_id] = (
                    f"{scene}/{branch}/{name}"
                )

        sequences[scene] = roles

    return sequences

def make_rows(dataset, seq, roles):
    ids = set()

    for mapping in roles.values():
        ids.update(mapping.keys())

    for fid in sorted(ids):
        paths = {
            role: roles.get(role, {}).get(fid, "")
            for role in [
                "rgb", "depth", "flow", "flow_mask"
            ]
        }

        manifest.append({
            "dataset": dataset,
            "sequence": seq,
            "frame_id": fid,
            "rgb_key": paths["rgb"],
            "depth_key": paths["depth"],
            "flow_key": paths["flow"],
            "flow_mask_key": paths["flow_mask"],
            "rgb_present": int(bool(paths["rgb"])),
            "depth_present": int(bool(paths["depth"])),
            "flow_present": int(bool(paths["flow"])),
            "flow_mask_present": int(bool(paths["flow_mask"])),
            "rgb_depth_paired": int(
                bool(paths["rgb"]) and bool(paths["depth"])
            ),
            "flow_without_rgb": int(
                bool(paths["flow"]) and not bool(paths["rgb"])
            ),
            "flow_mask_without_rgb": int(
                bool(paths["flow_mask"]) and not bool(paths["rgb"])
            ),
        })

    rgb = set(roles.get("rgb", {}))
    depth = set(roles.get("depth", {}))
    flow = set(roles.get("flow", {}))
    mask = set(roles.get("flow_mask", {}))

    sequence_rows.append({
        "dataset": dataset,
        "sequence": seq,
        "union_frames": len(ids),
        "rgb_frames": len(rgb),
        "depth_frames": len(depth),
        "rgb_depth_same_id": len(rgb & depth),
        "rgb_without_depth": len(rgb - depth),
        "depth_without_rgb": len(depth - rgb),
        "flow_frames": len(flow),
        "flow_without_rgb": len(flow - rgb),
        "flow_mask_frames": len(mask),
        "flow_mask_without_rgb": len(mask - rgb),
    })

def main():
    global manifest

    for dataset, path in FILES.items():
        print(f"\n========== {dataset} ==========")

        if not path.is_file():
            raise FileNotFoundError(path)

        with h5py.File(path, "r") as f:

            if dataset == "ScanNet":
                sequences = scannet_sequences(f)
            else:
                sequences = flat_sequences(f, dataset)

            sequence_names = sorted(sequences)
            seq_set = set(sequence_names)

            root_audit[dataset] = {
                "root_key_count": len(f.keys()),
                "root_keys": [
                    {
                        "name": k,
                        "type": (
                            "group" if isinstance(f[k], h5py.Group)
                            else "dataset"
                        ),
                        "is_sequence": k in seq_set,
                    }
                    for k in f.keys()
                ],
                "sequence_count": len(sequences),
                "sequence_names": sequence_names,
                "non_sequence_root_keys": [
                    k for k in f.keys() if k not in seq_set
                ],
            }

            # 保留 Sintel sequences.json 原始元数据
            if dataset == "Sintel" and "sequences.json" in f:
                try:
                    raw = decode_bytes(f["sequences.json"])
                    obj = json.loads(raw.decode("utf-8"))
                    (OUT / "sintel_sequences_metadata.json").write_text(
                        json.dumps(obj, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                    root_audit[dataset]["sequences_metadata_parsed"] = True
                except Exception as e:
                    root_audit[dataset]["sequences_metadata_error"] = repr(e)

            # 构建逐帧原始路径清单
            for seq in sequence_names:
                make_rows(dataset, seq, sequences[seq])

            # 每个数据集选3个代表序列，每序列取首/中/末深度样本
            chosen_indices = sorted(set([
                0, len(sequence_names) // 2,
                len(sequence_names) - 1
            ]))

            samples = []

            for idx in chosen_indices:
                seq = sequence_names[idx]
                depth_map = sequences[seq].get("depth", {})
                frame_ids = sorted(depth_map)

                if not frame_ids:
                    continue

                frame_indices = sorted(set([
                    0, len(frame_ids) // 2,
                    len(frame_ids) - 1
                ]))

                for fi in frame_indices:
                    fid = frame_ids[fi]
                    key = depth_map[fid]

                    try:
                        samples.append(
                            inspect_depth(f, key)
                        )
                    except Exception as e:
                        samples.append({
                            "key": key,
                            "decode_error": repr(e)
                        })

            depth_samples[dataset] = samples

        print("Root keys:", root_audit[dataset]["root_key_count"])
        print("Sequence groups:", len(sequences))
        print("Non-sequence keys:",
              root_audit[dataset]["non_sequence_root_keys"])

    # 保存全量逐帧原始清单
    manifest.sort(
        key=lambda x: (
            x["dataset"], x["sequence"], x["frame_id"]
        )
    )

    manifest_path = OUT / "full4_raw_frame_manifest_v1.csv"

    fields = list(manifest[0].keys())

    with manifest_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    # 保存序列级清单
    seq_path = OUT / "full4_sequence_inventory_v2.csv"

    with seq_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(sequence_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(sequence_rows)

    # 保存审计 JSON
    audit = {
        "datasets": root_audit,
        "total_manifest_rows": len(manifest),
        "total_rgb_depth_pairs": sum(
            x["rgb_depth_paired"] for x in manifest
        ),
        "total_rgb_frames": sum(
            x["rgb_present"] for x in manifest
        ),
        "total_depth_frames": sum(
            x["depth_present"] for x in manifest
        ),
        "depth_sample_audit": depth_samples,
        "sampling_for_manifest": False,
        "metric_depth_conversion_applied": False,
        "flow_mask_used_as_depth_mask": False,
    }

    json_path = OUT / "full4_raw_manifest_audit_v1.json"

    json_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 文件 SHA256
    h = hashlib.sha256()
    with manifest_path.open("rb") as f:
        for block in iter(
            lambda: f.read(8 * 1024 * 1024), b""
        ):
            h.update(block)

    print("\n========== FULL4 MANIFEST SUMMARY ==========")

    for dataset in FILES:
        rows = [
            x for x in sequence_rows
            if x["dataset"] == dataset
        ]

        print(
            dataset,
            "sequences =", len(rows),
            "RGB =", sum(x["rgb_frames"] for x in rows),
            "Depth =", sum(x["depth_frames"] for x in rows),
            "paired =", sum(x["rgb_depth_same_id"] for x in rows),
            "Flow =", sum(x["flow_frames"] for x in rows),
        )

    print("Manifest rows:", len(manifest))
    print("RGB-Depth pairs:", audit["total_rgb_depth_pairs"])
    print("Manifest SHA256:", h.hexdigest())
    print("OUTPUT:", manifest_path)
    print("OUTPUT:", seq_path)
    print("OUTPUT:", json_path)

if __name__ == "__main__":
    main()
