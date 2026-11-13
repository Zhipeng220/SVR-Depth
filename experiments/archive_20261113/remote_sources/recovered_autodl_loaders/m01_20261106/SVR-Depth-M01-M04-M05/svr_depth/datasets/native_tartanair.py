"""Native reader/manifest builder for the public TartanAir V1 directory layout.

Expected trajectory layout (from tartanair_tools):
  ENV/DIFFICULTY/Pxxx/image_left/000000_left.png
  ENV/DIFFICULTY/Pxxx/depth_left/000000_left_depth.npy

Depth is retained as float32 NPY and never quantized. Because the source
modality's axial-vs-radial convention and capture rate must match the training
contract, both are explicit CLI arguments rather than guessed here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .crosssource import CrossSourceDataError

_RGB_RE = re.compile(r"^(?P<idx>\d+)_left\.png$", re.IGNORECASE)
_DEPTH_RE = re.compile(r"^(?P<idx>\d+)_left_depth\.npy$", re.IGNORECASE)


def _contiguous_runs(items: list[tuple[int, Path, Path]]) -> list[list[tuple[int, Path, Path]]]:
    if not items:
        return []
    runs: list[list[tuple[int, Path, Path]]] = [[items[0]]]
    for item in items[1:]:
        if item[0] == runs[-1][-1][0] + 1:
            runs[-1].append(item)
        else:
            runs.append([item])
    return runs


def build_tartanair_manifests(
    root: str | Path,
    output_dir: str | Path,
    *,
    val_envs: Iterable[str],
    fps: float,
    depth_type: str,
    max_depth_m: float = 80.0,
    min_depth_m: float = 0.05,
    intrinsics: tuple[float, float, float, float] | None = None,
    min_frames: int = 20,
) -> dict:
    root = Path(root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if fps <= 0 or not np.isfinite(fps):
        raise CrossSourceDataError("TartanAir fps must be explicitly positive")
    if depth_type != "camera_z":
        raise CrossSourceDataError(
            "P_multi currently consumes camera-z depth. Verify the source modality and pass "
            "--depth-type camera_z only after confirming it; radial/range depth must be converted first."
        )
    if min_frames < 2:
        raise CrossSourceDataError("min_frames must be >= 2")

    val_envs_norm = {x.strip().lower() for x in val_envs if x.strip()}
    if not val_envs_norm:
        raise CrossSourceDataError("Specify at least one held-out environment with --val-envs")

    depth_dirs = sorted(p for p in root.rglob("depth_left") if p.is_dir())
    manifests: dict[str, list[str]] = {"train": [], "val": []}
    seen_val_envs: set[str] = set()
    stats = {"trajectories_found": 0, "sequences_written": 0, "frames_written": 0, "gt_frames": 0}

    for depth_dir in depth_dirs:
        trajectory_dir = depth_dir.parent
        image_dir = trajectory_dir / "image_left"
        if not image_dir.is_dir():
            continue
        stats["trajectories_found"] += 1
        depth_by_idx: dict[int, Path] = {}
        for path in depth_dir.glob("*_left_depth.npy"):
            match = _DEPTH_RE.match(path.name)
            if match:
                depth_by_idx[int(match.group("idx"))] = path
        paired: list[tuple[int, Path, Path]] = []
        for rgb_path in image_dir.glob("*_left.png"):
            match = _RGB_RE.match(rgb_path.name)
            if not match:
                continue
            frame_idx = int(match.group("idx"))
            depth_path = depth_by_idx.get(frame_idx)
            if depth_path is None:
                continue
            # Validate the first pair's encoding/registration without reading all files.
            if not paired:
                with Image.open(rgb_path) as im:
                    rgb_shape = (im.height, im.width)
                depth = np.load(depth_path, allow_pickle=False)
                if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
                    raise CrossSourceDataError(
                        f"Expected float depth_left NPY at {depth_path}, got {depth.dtype}/{depth.shape}"
                    )
                if depth.shape != rgb_shape:
                    raise CrossSourceDataError(
                        f"TartanAir RGB/depth shape mismatch: {rgb_path} {rgb_shape}, {depth_path} {depth.shape}"
                    )
            paired.append((frame_idx, rgb_path.resolve(), depth_path.resolve()))
        paired.sort(key=lambda x: x[0])
        if not paired:
            continue

        # ENV/DIFFICULTY/Pxxx is the documented layout. Group all trajectories
        # from an environment under one scene id to prevent environment leakage.
        try:
            env_dir = trajectory_dir.parent.parent
            env_rel = env_dir.relative_to(root).as_posix()
        except ValueError:
            env_rel = trajectory_dir.parts[-3]
        env_key = env_rel.lower()
        split = "val" if env_key.lower() in val_envs_norm or Path(env_key).name in val_envs_norm else "train"
        if split == "val":
            seen_val_envs.add(env_key)

        difficulty = trajectory_dir.parent.name
        trajectory = trajectory_dir.name
        sequence_id_base = f"{env_rel}/{difficulty}/{trajectory}"
        for run_idx, run in enumerate(_contiguous_runs(paired)):
            if len(run) < min_frames:
                continue
            # A sequence manifest is a contiguous run. Frame indices are reset
            # to 0 in the manifest; source_frame_id preserves original frame ID.
            frames = []
            for local_idx, (source_idx, rgb_path, depth_path) in enumerate(run):
                frames.append({
                    "frame_index": local_idx,
                    "timestamp_s": float(source_idx) / float(fps),
                    "source_frame_id": f"{source_idx:06d}_left",
                    "rgb": str(rgb_path),
                    "depth": str(depth_path),
                    "valid_mask": None,
                })
            payload = {
                "schema_version": 1,
                "dataset_id": "tartanair",
                "split": split,
                "scene_id": env_rel,
                "sequence_id": f"{sequence_id_base}/run{run_idx:02d}",
                "color_space": "RGB",
                "depth": {
                    "encoding": "npy_float32",
                    "scale_to_m": 1.0,
                    "type": "camera_z",
                    "registered_to_rgb": True,
                    "invalid_values": [0.0, -1.0],
                    "min_depth_m": float(min_depth_m),
                    "max_depth_m": float(max_depth_m),
                },
                "camera": None if intrinsics is None else {
                    "model": "pinhole", "intrinsics": list(intrinsics)
                },
                "frames": frames,
                "source_metadata": {
                    "layout": "tartanair_tools_v1_image_left_depth_left",
                    "fps_assumption": float(fps),
                    "depth_type_explicitly_confirmed": depth_type,
                    "environment": env_rel,
                    "difficulty": difficulty,
                    "trajectory": trajectory,
                },
            }
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", payload["sequence_id"])
            target_dir = output_dir / "manifests" / split / "tartanair"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{safe_name}.json"
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            manifests[split].append(str(target))
            stats["sequences_written"] += 1
            stats["frames_written"] += len(run)
            stats["gt_frames"] += len(run)

    if not manifests["train"]:
        raise CrossSourceDataError("No train TartanAir sequences found; verify root/layout and paired files")
    if not manifests["val"]:
        raise CrossSourceDataError(
            f"No validation TartanAir sequences found for val_envs={sorted(val_envs_norm)}. "
            "Choose an environment actually present in the root."
        )
    stats.update({"train_manifests": len(manifests["train"]), "val_manifests": len(manifests["val"]),
                  "val_envs_found": sorted(seen_val_envs)})
    return {"manifests": manifests, "stats": stats}


def _h5_blob(h5, reference: str) -> bytes:
    key = str(reference).lstrip("/")
    if key not in h5:
        raise CrossSourceDataError(f"TartanAir HDF5 reference not found: /{key}")
    value = h5[key][()]
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint8:
            return value.tobytes()
        if value.dtype.kind in ("S", "O"):
            item = value.item() if value.shape == () else value.reshape(-1)[0]
            return item if isinstance(item, bytes) else str(item).encode("latin1")
        return value.tobytes()
    if isinstance(value, np.bytes_):
        return bytes(value)
    raise CrossSourceDataError(f"Unsupported HDF5 blob type at /{key}: {type(value)}")


def _decode_h5_depth(blob: bytes, *, scale_to_m: float) -> np.ndarray:
    from io import BytesIO
    with Image.open(BytesIO(blob)) as im:
        raw = np.asarray(im)
    if raw.ndim == 2 and raw.dtype == np.uint16:
        depth_raw = raw.astype(np.float32)
    elif raw.ndim == 3 and raw.shape[2] >= 3:
        # Match VeloDepth's upstream decode_depth 24-bit channel convention.
        c0 = raw[..., 0].astype(np.uint32)
        c1 = raw[..., 1].astype(np.uint32)
        c2 = raw[..., 2].astype(np.uint32)
        depth_raw = (c0 | (c1 << 8) | (c2 << 16)).astype(np.float32)
    else:
        raise CrossSourceDataError(f"Unsupported TartanAir HDF5 depth PNG encoding: {raw.dtype}/{raw.shape}")
    return depth_raw / float(scale_to_m)


def build_tartanair_hdf5_manifests(
    hdf5_path: str | Path,
    output_dir: str | Path,
    *,
    val_envs: Iterable[str],
    fps: float,
    depth_scale: float = 1000.0,
    min_depth_m: float = 0.01,
    max_depth_m: float = 512.0,
    min_frames: int = 20,
    max_sequences: int | None = None,
) -> dict:
    """Export VeloDepth's upstream TartanAir HDF5 schema to canonical manifests.

    The schema follows `velodepth/datasets/sequence_dataset.py` and
    `velodepth/datasets/tartanair.py`: `train.txt`, `sequences.json`, embedded
    RGB/depth PNG blobs, and depth_scale=1000. Export is explicit because the
    canonical manifest loader consumes ordinary files rather than HDF5 nodes.
    """
    import h5py

    hdf5_path = Path(hdf5_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)
    if fps <= 0 or not np.isfinite(fps):
        raise CrossSourceDataError("TartanAir HDF5 fps must be explicitly positive")
    if depth_scale <= 0 or not np.isfinite(depth_scale):
        raise CrossSourceDataError("TartanAir HDF5 depth_scale must be positive")
    val_envs_norm = {x.strip().lower() for x in val_envs if x.strip()}
    if not val_envs_norm:
        raise CrossSourceDataError("Specify held-out TartanAir environments with --val-envs")

    manifests: dict[str, list[str]] = {"train": [], "val": []}
    stats = {"sequences_found": 0, "sequences_written": 0, "frames_written": 0,
             "gt_frames": 0, "train_manifests": 0, "val_manifests": 0}
    with h5py.File(hdf5_path, "r") as h5:
        for required in ("train.txt", "sequences.json"):
            if required not in h5:
                raise CrossSourceDataError(
                    f"{hdf5_path} does not match the VeloDepth TartanAir HDF5 schema: missing /{required}"
                )
        split_text = np.asarray(h5["train.txt"]).tobytes().decode("ascii").strip()
        sequences_text = np.asarray(h5["sequences.json"]).tobytes().decode("ascii")
        sequences = json.loads(sequences_text)
        rows = []
        for line in split_text.splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            rows.append((parts[0], int(parts[1])))
        if max_sequences is not None:
            rows = rows[:max_sequences]
        stats["sequences_found"] = len(rows)
        for sequence_name, declared_count in rows:
            if sequence_name not in sequences:
                raise CrossSourceDataError(f"Sequence {sequence_name!r} from train.txt is absent in sequences.json")
            sample = sequences[sequence_name]
            image_refs = sample.get("image", [])
            depth_refs = sample.get("depth", [])
            if len(image_refs) != declared_count or len(depth_refs) != declared_count:
                raise CrossSourceDataError(
                    f"{sequence_name}: train.txt count={declared_count}, image refs={len(image_refs)}, depth refs={len(depth_refs)}"
                )
            if declared_count < min_frames:
                continue
            k_values = sample.get("K")
            if k_values is not None and len(k_values) != declared_count:
                raise CrossSourceDataError(f"{sequence_name}: K count does not match frame count")
            parts = Path(sequence_name).parts
            env_name = parts[0] if len(parts) > 1 else sequence_name.split("_")[0]
            split = "val" if env_name.lower() in val_envs_norm else "train"
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sequence_name)
            export_dir = output_dir / "tartanair_hdf5_export" / split / safe_name
            rgb_dir, depth_dir = export_dir / "rgb", export_dir / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)
            frames = []
            intrinsics = None
            for idx, (image_ref, depth_ref) in enumerate(zip(image_refs, depth_refs)):
                from io import BytesIO
                rgb_blob = _h5_blob(h5, image_ref)
                with Image.open(BytesIO(rgb_blob)) as im:
                    rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
                depth_m = _decode_h5_depth(_h5_blob(h5, depth_ref), scale_to_m=depth_scale)
                if depth_m.shape != rgb.shape[:2]:
                    raise CrossSourceDataError(
                        f"{sequence_name} frame {idx}: HDF5 RGB/depth dimensions differ: {rgb.shape[:2]} vs {depth_m.shape}"
                    )
                depth_m[~np.isfinite(depth_m) | (depth_m < min_depth_m) | (depth_m > max_depth_m)] = 0.0
                if not np.any(depth_m > 0):
                    raise CrossSourceDataError(f"{sequence_name} frame {idx}: no valid depth after range filtering")
                rgb_path = rgb_dir / f"{idx:06d}.png"
                depth_path = depth_dir / f"{idx:06d}.npy"
                Image.fromarray(rgb, mode="RGB").save(rgb_path)
                np.save(depth_path, depth_m.astype(np.float32), allow_pickle=False)
                frame_intrinsics = None
                if k_values is not None:
                    K = np.asarray(k_values[idx], dtype=np.float64)
                    if K.shape == (3, 3):
                        frame_intrinsics = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
                        if intrinsics is None:
                            intrinsics = frame_intrinsics
                frames.append({
                    "frame_index": idx, "timestamp_s": idx / float(fps),
                    "source_frame_id": str(image_ref), "rgb": str(rgb_path.resolve()),
                    "depth": str(depth_path.resolve()), "valid_mask": None,
                    "intrinsics": frame_intrinsics,
                })
            payload = {
                "schema_version": 1, "dataset_id": "tartanair", "split": split,
                "scene_id": env_name, "sequence_id": sequence_name,
                "color_space": "RGB",
                "depth": {"encoding": "npy_float32", "scale_to_m": 1.0,
                          "type": "camera_z", "registered_to_rgb": True,
                          "invalid_values": [0.0], "min_depth_m": min_depth_m,
                          "max_depth_m": max_depth_m},
                "camera": None if intrinsics is None else {"model": "pinhole", "intrinsics": intrinsics},
                "frames": frames,
                "source_metadata": {"layout": "VeloDepth_SequenceDataset_HDF5",
                                    "hdf5_path": str(hdf5_path), "depth_scale": depth_scale,
                                    "depth_convention": "as decoded by VeloDepth TartanAir loader",
                                    "fps_assumption": fps},
            }
            target_dir = output_dir / "manifests" / split / "tartanair"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"{safe_name}.json"
            target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            manifests[split].append(str(target))
            stats["sequences_written"] += 1
            stats["frames_written"] += declared_count
            stats["gt_frames"] += declared_count
    if not manifests["train"] or not manifests["val"]:
        raise CrossSourceDataError(
            f"TartanAir HDF5 export requires non-empty train and val splits; found train={len(manifests['train'])}, "
            f"val={len(manifests['val'])}. Check --val-envs and sequence naming."
        )
    stats["train_manifests"] = len(manifests["train"])
    stats["val_manifests"] = len(manifests["val"])
    return {"manifests": manifests, "stats": stats}
