import csv
import json
import hashlib
import re
from pathlib import Path, PurePosixPath

import h5py
import numpy as np

DATA = Path("/root/autodl-tmp/datasets/VeloDepth-Evaluation")
OUT = Path("outputs/full4_evaluation_inventory")
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("Bonn", "Bonn.hdf5", "val.txt", "sequences.json"),
    ("TUM", "TUM.hdf5", "val.txt", "sequences.json"),
    ("ScanNet", "ScanNet.hdf5", "val.txt", "sequences_.json"),
    ("Sintel", "Sintel.hdf5", "training.txt", "sequences.json"),
]

def read_h5_text(h5, name):
    obj = h5[name][()]

    if isinstance(obj, (bytes, np.bytes_)):
        return bytes(obj).decode("utf-8-sig")

    if isinstance(obj, np.ndarray):
        return obj.tobytes().decode("utf-8-sig")

    return str(obj)

def parse_split(text):
    records = []

    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()

        if not line:
            continue

        parts = line.rsplit(None, 1)

        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(
                f"Invalid split line {line_no}: {line}"
            )

        records.append({
            "name": parts[0].strip(),
            "declared": int(parts[1])
        })

    return records

def resolve_sequence(name, sequences):
    if name in sequences:
        return name

    basename = PurePosixPath(name).name

    matches = [
        key for key in sequences
        if PurePosixPath(key).name == basename
    ]

    if len(matches) == 1:
        return matches[0]

    raise KeyError(
        f"Cannot uniquely resolve sequence: {name}; "
        f"matches={matches[:5]}"
    )

def frame_id(path):
    stem = PurePosixPath(path).stem

    for suffix in ("_rgb", "_depth", "_c", "_d"):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]

    return stem

def h5_exists(h5, path):
    path = str(path).replace("\\", "/").lstrip("/")
    return path in h5

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()

sequence_rows = []
frame_rows = []
dataset_reports = []

for dataset, filename, split_file, json_file in CONFIGS:

    print(f"\n========== {dataset} ==========")

    h5_path = DATA / filename

    report = {
        "dataset": dataset,
        "hdf5": str(h5_path),
        "split_file": split_file,
        "sequence_file": json_file,
        "status": "FAIL",
    }

    try:
        with h5py.File(h5_path, "r") as h5:

            split = parse_split(
                read_h5_text(h5, split_file)
            )

            sequences = json.loads(
                read_h5_text(h5, json_file)
            )

            print("Official split entries:", len(split))
            print("Sequence JSON entries:", len(sequences))

            split_names = [x["name"] for x in split]

            if len(split_names) != len(set(split_names)):
                raise RuntimeError(
                    "Duplicate sequence entries in official split"
                )

            dataset_frames = 0
            dataset_errors = []

            for rec in split:

                name = rec["name"]
                declared = rec["declared"]

                try:
                    key = resolve_sequence(name, sequences)
                    item = sequences[key]

                    images = item.get("image", [])
                    depths = item.get("depth", [])

                    if not isinstance(images, list):
                        raise TypeError("image is not a list")

                    if not isinstance(depths, list):
                        raise TypeError("depth is not a list")

                    image_ids = [frame_id(x) for x in images]
                    depth_ids = [frame_id(x) for x in depths]

                    n = max(len(images), len(depths))

                    missing_rgb = 0
                    missing_depth = 0
                    pair_mismatch = 0

                    for i in range(n):

                        rgb = images[i] if i < len(images) else ""
                        dep = depths[i] if i < len(depths) else ""

                        rgb_ok = bool(rgb) and h5_exists(h5, rgb)
                        dep_ok = bool(dep) and h5_exists(h5, dep)

                        if not rgb_ok:
                            missing_rgb += 1

                        if not dep_ok:
                            missing_depth += 1

                        matched = (
                            i < len(image_ids)
                            and i < len(depth_ids)
                            and image_ids[i] == depth_ids[i]
                        )

                        if not matched:
                            pair_mismatch += 1

                        frame_rows.append({
                            "dataset": dataset,
                            "sequence": name,
                            "sequence_key": key,
                            "index": i,
                            "frame_id": (
                                image_ids[i]
                                if i < len(image_ids)
                                else ""
                            ),
                            "rgb_path": rgb,
                            "depth_path": dep,
                            "rgb_exists": int(rgb_ok),
                            "depth_exists": int(dep_ok),
                            "pair_matched": int(matched),
                        })

                    flow = item.get("flow", [])
                    mask = item.get(
                        "flow_mask",
                        item.get("mask", [])
                    )

                    if not isinstance(flow, list):
                        flow = []

                    if not isinstance(mask, list):
                        mask = []

                    duplicate_rgb = (
                        len(image_ids) != len(set(image_ids))
                    )

                    duplicate_depth = (
                        len(depth_ids) != len(set(depth_ids))
                    )

                    errors = []

                    if len(images) != declared:
                        errors.append("RGB_COUNT_MISMATCH")

                    if len(depths) != declared:
                        errors.append("DEPTH_COUNT_MISMATCH")

                    if len(images) != len(depths):
                        errors.append("RGB_DEPTH_LENGTH_MISMATCH")

                    if missing_rgb:
                        errors.append("MISSING_RGB")

                    if missing_depth:
                        errors.append("MISSING_DEPTH")

                    if pair_mismatch:
                        errors.append("FRAME_ID_MISMATCH")

                    if duplicate_rgb or duplicate_depth:
                        errors.append("DUPLICATE_FRAME_ID")

                    dataset_frames += len(images)

                    row_status = "PASS" if not errors else "FAIL"

                    sequence_rows.append({
                        "dataset": dataset,
                        "sequence": name,
                        "sequence_key": key,
                        "declared_frames": declared,
                        "rgb_frames": len(images),
                        "depth_frames": len(depths),
                        "flow_frames": len(flow),
                        "mask_frames": len(mask),
                        "missing_rgb": missing_rgb,
                        "missing_depth": missing_depth,
                        "pair_mismatch": pair_mismatch,
                        "status": row_status,
                        "errors": ";".join(errors),
                    })

                    if errors:
                        dataset_errors.append(
                            f"{name}: {','.join(errors)}"
                        )

                except Exception as e:
                    dataset_errors.append(
                        f"{name}: {type(e).__name__}: {e}"
                    )

                    sequence_rows.append({
                        "dataset": dataset,
                        "sequence": name,
                        "sequence_key": "",
                        "declared_frames": declared,
                        "rgb_frames": 0,
                        "depth_frames": 0,
                        "flow_frames": 0,
                        "mask_frames": 0,
                        "missing_rgb": 0,
                        "missing_depth": 0,
                        "pair_mismatch": 0,
                        "status": "FAIL",
                        "errors": str(e),
                    })

            total_declared = sum(x["declared"] for x in split)

            report.update({
                "sequences": len(split),
                "declared_test_frames": total_declared,
                "actual_rgb_frames": dataset_frames,
                "sequence_json_entries": len(sequences),
                "errors": dataset_errors,
                "status": (
                    "PASS" if not dataset_errors else "FAIL"
                ),
            })

            dataset_reports.append(report)

            print("Declared test frames:", total_declared)
            print("Actual RGB frames:", dataset_frames)
            print("Errors:", len(dataset_errors))
            print("STATUS:", report["status"])

            for err in dataset_errors[:20]:
                print("[ERROR]", err)

    except Exception as e:
        report["errors"] = [
            f"{type(e).__name__}: {e}"
        ]
        dataset_reports.append(report)
        print("[FATAL]", e)

# Write sequence inventory
seq_csv = OUT / "official_test_sequence_manifest_v1.csv"

with seq_csv.open("w", newline="", encoding="utf-8") as f:
    if sequence_rows:
        writer = csv.DictWriter(
            f,
            fieldnames=list(sequence_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(sequence_rows)

# Write full frame inventory
frame_csv = OUT / "official_test_frame_manifest_v1.csv"

with frame_csv.open("w", newline="", encoding="utf-8") as f:
    fields = [
        "dataset", "sequence", "sequence_key", "index",
        "frame_id", "rgb_path", "depth_path",
        "rgb_exists", "depth_exists", "pair_matched"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(frame_rows)

overall = all(
    r["status"] == "PASS"
    for r in dataset_reports
) and len(dataset_reports) == 4

summary = {
    "protocol": "Official full test splits",
    "sampling": False,
    "subsampling": False,
    "datasets": dataset_reports,
    "total_sequences": sum(
        r.get("sequences", 0) for r in dataset_reports
    ),
    "total_declared_test_frames": sum(
        r.get("declared_test_frames", 0)
        for r in dataset_reports
    ),
    "total_rgb_frames": sum(
        r.get("actual_rgb_frames", 0)
        for r in dataset_reports
    ),
    "frame_manifest_rows": len(frame_rows),
    "all_datasets_pass": overall,
    "frame_manifest_sha256": sha256_file(frame_csv),
}

json_path = OUT / "official_test_split_audit_v1.json"

json_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n========== FINAL SUMMARY ==========")
print(json.dumps(summary, indent=2, ensure_ascii=False))

print("\nSequence manifest:", seq_csv)
print("Frame manifest:", frame_csv)
print("Audit report:", json_path)

if overall:
    print("\n[PASS] ALL FOUR OFFICIAL TEST SPLITS VERIFIED")
else:
    print("\n[FAIL] DO NOT FREEZE EVALUATION PROTOCOL")

    raise SystemExit(1)
