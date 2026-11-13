import csv
import json
import io
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
from PIL import Image

DATA = Path("/root/autodl-tmp/datasets/VeloDepth-Evaluation")

FILES = {
    "Bonn": DATA / "Bonn.hdf5",
    "TUM": DATA / "TUM.hdf5",
    "ScanNet": DATA / "ScanNet.hdf5",
    "Sintel": DATA / "Sintel.hdf5",
}

OUT = Path("outputs/full4_evaluation_inventory")
OUT.mkdir(parents=True, exist_ok=True)

report = []
result = {
    "datasets": {},
    "sequence_inventory": {},
    "scannet_crosscheck": {}
}

def log(s=""):
    print(s)
    report.append(str(s))

def attrs_dict(obj):
    out = {}
    for k, v in obj.attrs.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out

def inspect_node(node, path, depth=0, max_depth=3,
                 decode_state=None):
    indent = "  " * depth

    if isinstance(node, h5py.Group):
        keys = list(node.keys())

        log(f"{indent}[GROUP] {path}")
        log(f"{indent}  children={len(keys)}")
        log(f"{indent}  attrs={attrs_dict(node)}")
        log(f"{indent}  first_keys={keys[:8]}")

        if depth < max_depth:
            for key in keys[:4]:
                inspect_node(
                    node[key],
                    f"{path}/{key}",
                    depth + 1,
                    max_depth,
                    decode_state
                )

    elif isinstance(node, h5py.Dataset):
        log(f"{indent}[DATASET] {path}")
        log(f"{indent}  shape={node.shape}")
        log(f"{indent}  dtype={node.dtype}")
        log(f"{indent}  attrs={attrs_dict(node)}")

        # 最多解码三个原始图像字节流样本
        if (
            decode_state is not None
            and decode_state["count"] < 3
            and node.dtype == np.uint8
            and node.ndim == 1
            and node.size < 20_000_000
        ):
            try:
                raw = node[()].tobytes()
                im = Image.open(io.BytesIO(raw))
                im.load()

                log(
                    f"{indent}  DECODE: "
                    f"mode={im.mode}, size={im.size}"
                )
                decode_state["count"] += 1

            except Exception:
                pass

def main():
    log("========== FULL4 HDF5 SCHEMA AUDIT ==========")

    for name, path in FILES.items():
        log(f"\n========== DATASET: {name} ==========")

        if not path.exists():
            log(f"[MISSING] {path}")
            continue

        result["datasets"][name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size
        }

        log(f"File size: {path.stat().st_size:,} bytes")

        try:
            with h5py.File(path, "r") as f:
                keys = list(f.keys())

                result["datasets"][name]["root_keys"] = keys
                result["datasets"][name]["root_attrs"] = attrs_dict(f)

                log(f"Root keys count: {len(keys)}")
                log(f"Root keys: {keys[:20]}")
                log(f"Root attrs: {attrs_dict(f)}")

                state = {"count": 0}

                # 只递归检查代表性结构，不遍历解码全量图像
                for key in keys[:5]:
                    inspect_node(
                        f[key],
                        key,
                        depth=0,
                        max_depth=2,
                        decode_state=state
                    )

        except Exception as e:
            log(f"[ERROR] {name}: {repr(e)}")

    # 汇总四数据集序列清单
    inventory = OUT / "full4_sequence_inventory.csv"

    log("\n========== SEQUENCE INVENTORY CROSSCHECK ==========")

    if inventory.exists():
        totals = defaultdict(
            lambda: defaultdict(int)
        )

        with inventory.open(
            encoding="utf-8-sig",
            newline=""
        ) as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            ds = row["dataset"]

            for field in [
                "rgb_frames",
                "depth_frames",
                "rgb_depth_same_id",
                "rgb_without_depth",
                "depth_without_rgb",
                "flow_frames",
                "mask_frames"
            ]:
                totals[ds][field] += int(row[field] or 0)

        result["sequence_inventory"]["rows"] = len(rows)

        for ds, values in sorted(totals.items()):
            result["sequence_inventory"][ds] = dict(values)
            log(f"\n{ds}:")
            for k, v in values.items():
                log(f"  {k}: {v:,}")

        log(f"\nTotal sequence rows: {len(rows)}")

    else:
        log("[ERROR] Sequence inventory missing")

    # ScanNet 逐帧审计交叉核对
    frame_csv = OUT / "scannet_frame_inventory_v2.csv"
    audit_json = OUT / "scannet_audit_v2.json"

    log("\n========== SCANNET CROSSCHECK ==========")

    if frame_csv.exists():
        with frame_csv.open(
            encoding="utf-8-sig",
            newline=""
        ) as f:
            frames = list(csv.DictReader(f))

        scenes = set(x["scene"] for x in frames)
        paired = sum(x["paired"] == "1" for x in frames)

        result["scannet_crosscheck"] = {
            "frame_rows": len(frames),
            "unique_scenes": len(scenes),
            "paired_frames": paired,
            "unpaired_frames": len(frames) - paired
        }

        log(f"Frame rows: {len(frames):,}")
        log(f"Unique scenes: {len(scenes)}")
        log(f"Paired frames: {paired:,}")
        log(f"Unpaired frames: {len(frames)-paired:,}")

    if audit_json.exists():
        audit = json.loads(
            audit_json.read_text(encoding="utf-8")
        )
        log(f"Original audit passed: {audit.get('all_frames_paired')}")
        log(f"Decode samples passed: {audit.get('decode_samples_pass')}")

    # 保存报告
    (OUT / "full4_hdf5_schema_audit.txt").write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    (OUT / "full4_hdf5_schema_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    log("\n========== AUDIT COMPLETE ==========")
    log("[OUTPUT] full4_hdf5_schema_audit.txt")
    log("[OUTPUT] full4_hdf5_schema_audit.json")

if __name__ == "__main__":
    main()
