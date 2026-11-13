import sys
import io
import importlib.util
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

DATA = Path("/root/autodl-tmp/datasets/VeloDepth-Evaluation")

DSROOT = Path(
    "/root/autodl-tmp/SVR-Depth/velodepth/velodepth/datasets"
)

SAMPLES = {
    "Bonn": (
        "Bonn.hdf5",
        "balloon/6470-92_d.png"
    ),
    "TUM": (
        "TUM.hdf5",
        "fr1_360/1791-68_d.png"
    ),
    "ScanNet": (
        "ScanNet.hdf5",
        "scene0707_00/depth/000030.png"
    ),
    "Sintel": (
        "Sintel.hdf5",
        "alley_1/0001_depth.png"
    ),
}

print("\n========== 1. ACTIVE PACKAGE ==========")

try:
    import velodepth
    print("Package:", velodepth.__file__)
    print("Package paths:", list(velodepth.__path__))
except Exception as e:
    print("Import error:", repr(e))

for mod in [
    "velodepth.datasets.bonn",
    "velodepth.datasets.tum",
    "velodepth.datasets.scannet",
    "velodepth.datasets.sintel",
    "velodepth.datasets.utils_decode",
]:
    try:
        spec = importlib.util.find_spec(mod)
        print(mod, "=>", spec.origin if spec else None)
    except Exception as e:
        print(mod, "=> ERROR:", repr(e))

print("\n========== 2. EXACT UPSTREAM SOURCE ==========")

for name in [
    "bonn.py",
    "tum.py",
    "scannet.py",
    "sintel.py",
    "utils_decode.py",
]:
    p = DSROOT / name
    print("\n---", p, "---")

    if p.exists():
        print(p.read_text(errors="replace"))
    else:
        print("NOT FOUND")

for name in ["sequence_dataset.py", "base_dataset.py"]:
    p = DSROOT / name
    print("\n--- RELEVANT LINES:", name, "---")

    if p.exists():
        lines = p.read_text(errors="replace").splitlines()
        for i, line in enumerate(lines):
            if any(x in line for x in [
                "decode_depth",
                "depth_scale",
                "decoded_data",
            ]):
                start = max(0, i - 2)
                end = min(len(lines), i + 4)
                for j in range(start, end):
                    print(f"{j+1:04d}: {lines[j]}")
                print()

print("\n========== 3. REAL HDF5 PIXEL VALUES ==========")

def report(name, arr):
    print("Mode array shape:", arr.shape)
    print("dtype:", arr.dtype)

    if arr.ndim == 2:
        valid = arr[arr > 0]

        if not len(valid):
            print("NO VALID PIXELS")
            return

        print("raw min/max:", int(valid.min()), int(valid.max()))
        print("zero ratio:", float(np.mean(arr == 0)))

        for scale in [1000.0, 5000.0]:
            meters = valid.astype(np.float64) / scale
            print(
                "candidate scale =", scale,
                "p10/p50/p90/p99 (m) =",
                np.percentile(meters, [10, 50, 90, 99]).round(4)
            )

    elif arr.ndim == 3 and arr.shape[2] >= 3:
        a = arr.astype(np.uint32)
        print("channel min/max:")
        for i in range(3):
            print(
                i,
                int(a[..., i].min()),
                int(a[..., i].max())
            )

        candidates = {
            "R+G*256+B*65536":
                a[..., 0] + (a[..., 1] << 8) + (a[..., 2] << 16),

            "R*65536+G*256+B":
                (a[..., 0] << 16) + (a[..., 1] << 8) + a[..., 2],
        }

        for encoding, raw in candidates.items():
            valid = raw[raw > 0]
            print("\n", encoding)

            print(
                "raw p10/p50/p90/p99:",
                np.percentile(valid, [10, 50, 90, 99])
            )

            for scale in [1000.0, 5000.0]:
                print(
                    "scale", scale,
                    "meters p10/p50/p90:",
                    (np.percentile(valid, [10, 50, 90]) / scale).round(4)
                )

for name, (filename, key) in SAMPLES.items():
    path = DATA / filename

    print("\n---", name, "---")
    print("HDF5:", path)
    print("Sample:", key)

    with h5py.File(path, "r") as h:
        raw = np.asarray(h[key][()], dtype=np.uint8)
        blob = raw.tobytes()

    img = Image.open(io.BytesIO(blob))
    img.load()

    print("PNG mode:", img.mode)
    print("PNG size:", img.size)

    report(name, np.asarray(img))

print("\n========== AUDIT COMPLETE ==========")
