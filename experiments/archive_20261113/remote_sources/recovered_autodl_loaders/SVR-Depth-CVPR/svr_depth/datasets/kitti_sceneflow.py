"""KITTI Scene Flow 2015 MultiView adapter and manifest builder.

Expected extracted roots:
  multiview_root/training/image_2/<scene>_<frame>.png
  main_root/training/disp_occ_0/<scene>_10.png
  main_root/training/disp_occ_1/<scene>_10.png
  calib_root/training/calib_cam_to_cam/<scene>.txt

The 20-frame MultiView extension supplies RGB frames; the official Scene Flow
labels supervise the central t0/t1 pair (image suffixes _10/_11). Other frames
are intentionally represented as RGB-only. No GT is fabricated for them.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .crosssource import CrossSourceDataError

_FRAME_RE = re.compile(r"^(?P<scene>\d{6})_(?P<frame>\d{2})$")

def _select_protocol_frames(rows):
    """Select exactly frames 00-19 from a valid KITTI 00-20 scene.

    Already-20-frame sequences are accepted.
    A 21-frame sequence must contain exactly 00..20.
    All other layouts are rejected.
    """
    frame_ids = [int(x[0]) for x in rows]

    if frame_ids == list(range(20)):
        return rows

    if frame_ids == list(range(21)):
        return rows[:20]

    return None




def _find_dir(root: Path, name: str, split: str = "training") -> Path:
    candidates = sorted(p for p in root.rglob(name) if p.is_dir() and p.parent.name == split)
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find {split}/{name} below {root}. Extract the matching KITTI Scene Flow archive first."
        )
    if len(candidates) > 1:
        # Prefer paths whose parent tree explicitly identifies the intended package.
        scored = sorted(candidates, key=lambda p: ("multiview" not in str(p).lower() if name == "image_2" else False, str(p)))
        return scored[0]
    return candidates[0]


def _parse_projection_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        try:
            arr = np.asarray([float(x) for x in raw.split()], dtype=np.float64)
        except ValueError:
            continue
        if arr.size == 12:
            values[key.strip()] = arr.reshape(3, 4)
    p2 = values.get("P_rect_02", values.get("P2"))
    p3 = values.get("P_rect_03", values.get("P3"))
    if p2 is None or p3 is None:
        raise CrossSourceDataError(
            f"{path}: need rectified left/right projection matrices P_rect_02/P_rect_03 (or P2/P3)"
        )
    fx, fy, cx, cy = float(p2[0, 0]), float(p2[1, 1]), float(p2[0, 2]), float(p2[1, 2])
    if fx <= 0 or fy <= 0:
        raise CrossSourceDataError(f"{path}: invalid focal lengths in P_rect_02")
    tx2 = float(p2[0, 3] / p2[0, 0])
    tx3 = float(p3[0, 3] / p3[0, 0])
    baseline = abs(tx3 - tx2)
    if not np.isfinite(baseline) or baseline <= 0:
        raise CrossSourceDataError(f"{path}: non-positive stereo baseline {baseline}")
    return np.asarray([fx, fy, cx, cy], dtype=np.float64), np.asarray([fx, baseline], dtype=np.float64)


def _read_disparity(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        raw = np.asarray(im)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise CrossSourceDataError(f"KITTI disparity must be single-channel uint16 PNG: {path}, {raw.dtype}/{raw.shape}")
    return raw.astype(np.float32) / 256.0


def _gt_for_frame(scene: str, frame_idx: int, disp0_dir: Path, disp1_dir: Path) -> Path | None:
    # KITTI 2015's labelled pair is image_2/<id>_10 (t0) and _11 (t1);
    # disp_occ_0 and disp_occ_1 are the corresponding left-camera GT maps.
    if frame_idx == 10:
        candidates = [disp0_dir / f"{scene}_10.png"]
    elif frame_idx == 11:
        candidates = [disp1_dir / f"{scene}_10.png", disp1_dir / f"{scene}_11.png"]
    else:
        candidates = [disp0_dir / f"{scene}_{frame_idx:02d}.png", disp1_dir / f"{scene}_{frame_idx:02d}.png"]
    found = [p for p in candidates if p.is_file()]
    if len(found) > 1:
        raise CrossSourceDataError(f"Ambiguous KITTI GT mapping for {scene}_{frame_idx:02d}: {found}")
    return found[0] if found else None


def build_kitti_sceneflow_manifests(
    multiview_root: str | Path,
    main_root: str | Path,
    calib_root: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 10.0,
    expected_frames: int = 20,
    val_fraction: float = 0.2,
    seed: int = 20261103,
    min_gt_frames: int = 2,
) -> dict:
    multiview_root = Path(multiview_root).expanduser().resolve()
    main_root = Path(main_root).expanduser().resolve()
    calib_root = Path(calib_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    for root in (multiview_root, main_root, calib_root):
        if not root.is_dir():
            raise FileNotFoundError(root)
    if fps <= 0 or not np.isfinite(fps):
        raise CrossSourceDataError("KITTI fps must be positive")
    if expected_frames != 20:
        raise CrossSourceDataError("This CrossSource-v2 protocol currently locks KITTI MultiView to 20 frames")
    if not 0 < val_fraction < 1:
        raise CrossSourceDataError("val_fraction must be between 0 and 1")

    image_dir = _find_dir(multiview_root, "image_2")
    disp0_dir = _find_dir(main_root, "disp_occ_0")
    disp1_dir = _find_dir(main_root, "disp_occ_1")
    calib_dir = _find_dir(calib_root, "calib_cam_to_cam")

    groups: dict[str, list[tuple[int, Path]]] = {}
    for image_path in sorted(image_dir.glob("*.png")):
        match = _FRAME_RE.match(image_path.stem)
        if not match:
            continue
        scene, frame = match.group("scene"), int(match.group("frame"))
        groups.setdefault(scene, []).append((frame, image_path.resolve()))
    if not groups:
        raise CrossSourceDataError(f"No KITTI MultiView frames named <6-digit-scene>_<2-digit-frame>.png in {image_dir}")

    scenes = sorted(groups)
    rng = random.Random(seed)
    shuffled = scenes[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_scenes = set(shuffled[:n_val])
    manifests: dict[str, list[str]] = {"train": [], "val": []}
    stats = {"scenes_found": len(scenes), "sequences_written": 0, "frames_written": 0,
             "gt_frames": 0, "scenes_without_expected_20_frames": [], "scenes_without_calibration": [],
             "scenes_with_dropped_frame_20": [], "frames_dropped": 0,
             "frame_selection_policy": "retain_00_19_drop_20",
             "gt_frame_ids": {}}

    for scene in scenes:
        rows = sorted(groups[scene], key=lambda x: x[0])
        frame_ids = [x[0] for x in rows]

        selected_rows = _select_protocol_frames(rows)

        if selected_rows is None:
            stats["scenes_without_expected_20_frames"].append(
                {"scene": scene, "frame_ids": frame_ids}
            )
            continue

        if len(selected_rows) < len(rows):
            stats["scenes_with_dropped_frame_20"].append(scene)
            stats["frames_dropped"] += len(rows) - len(selected_rows)

        rows = selected_rows
        frame_ids = [x[0] for x in rows]

        calib_path = calib_dir / f"{scene}.txt"
        if not calib_path.is_file():
            stats["scenes_without_calibration"].append(scene)
            continue
        intrinsics, fx_baseline = _parse_projection_file(calib_path)
        fx, baseline = map(float, fx_baseline)
        split = "val" if scene in val_scenes else "train"
        frames = []
        gt_count = 0
        for local_idx, (source_frame, rgb_path) in enumerate(rows):
            with Image.open(rgb_path) as im:
                rgb_shape = (im.height, im.width)
            disp_path = _gt_for_frame(scene, source_frame, disp0_dir, disp1_dir)
            depth_path_value = None
            if disp_path is not None:
                disp = _read_disparity(disp_path)
                if disp.shape != rgb_shape:
                    raise CrossSourceDataError(
                        f"KITTI RGB/disp mismatch for {scene}_{source_frame:02d}: {rgb_shape} vs {disp.shape}"
                    )
                valid = np.isfinite(disp) & (disp > 0)
                depth_m = np.zeros_like(disp, dtype=np.float32)
                depth_m[valid] = (fx * baseline) / disp[valid]
                valid &= np.isfinite(depth_m) & (depth_m >= 0.1) & (depth_m <= 80.0)
                depth_m[~valid] = 0.0
                if int(valid.sum()) < 1000:
                    raise CrossSourceDataError(
                        f"KITTI GT has too few valid camera-z pixels ({int(valid.sum())}) for {disp_path}"
                    )
                depth_dir = output_dir / "depth_cache" / split / scene
                depth_dir.mkdir(parents=True, exist_ok=True)
                depth_path = depth_dir / f"{scene}_{source_frame:02d}_camera_z.npy"
                np.save(depth_path, depth_m.astype(np.float32), allow_pickle=False)
                depth_path_value = str(depth_path.resolve())
                gt_count += 1
            frames.append({
                "frame_index": local_idx,
                "timestamp_s": float(source_frame) / float(fps),
                "source_frame_id": f"{scene}_{source_frame:02d}",
                "rgb": str(rgb_path),
                "depth": depth_path_value,
                "valid_mask": None,
            })
        if gt_count < min_gt_frames:
            continue
        stats["gt_frame_ids"][scene] = [x[0] for x in rows if _gt_for_frame(scene, x[0], disp0_dir, disp1_dir) is not None]
        payload = {
            "schema_version": 1,
            "dataset_id": "kitti_sceneflow",
            "split": split,
            "scene_id": f"kitti_{scene}",
            "sequence_id": f"kitti_{scene}",
            "color_space": "RGB",
            "depth": {
                "encoding": "npy_float32", "scale_to_m": 1.0,
                "type": "camera_z", "registered_to_rgb": True,
                "invalid_values": [0.0], "min_depth_m": 0.1, "max_depth_m": 80.0,
            },
            "camera": {"model": "pinhole", "intrinsics": intrinsics.tolist()},
            "frames": frames,
            "source_metadata": {
                "benchmark": "KITTI Scene Flow 2015", "multiview_frames": 20,
                "disparity_encoding": "uint16_png_div_256", "fx_px": fx,
                "baseline_m": baseline, "label_mapping": "disp_occ_0->t0(_10), disp_occ_1->t1(_11)",
                "calibration_file": str(calib_path.resolve()),
            },
        }
        target_dir = output_dir / "manifests" / split / "kitti_sceneflow"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"kitti_{scene}.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        manifests[split].append(str(target))
        stats["sequences_written"] += 1
        stats["frames_written"] += len(frames)
        stats["gt_frames"] += gt_count

    if not manifests["train"] or not manifests["val"]:
        raise CrossSourceDataError(
            "KITTI manifest generation produced an empty split. Check MultiView 20-frame layout, "
            "calibration and disp_occ_0/disp_occ_1 naming. Stats: " + json.dumps(stats)
        )
    stats.update({"train_manifests": len(manifests["train"]), "val_manifests": len(manifests["val"]),
                  "val_scene_ids": sorted(val_scenes)})
    return {"manifests": manifests, "stats": stats}
