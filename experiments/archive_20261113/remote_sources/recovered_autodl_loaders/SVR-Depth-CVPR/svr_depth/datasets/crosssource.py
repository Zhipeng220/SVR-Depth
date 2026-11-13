"""Manifest-driven, fail-fast data contract for cross-source P_multi training.

This module intentionally does not guess native dataset directory layouts.
Each dataset must first be converted to the canonical per-sequence JSON
manifest documented in docs/crosssource_data.md.  The adapters below are
manifest adapters, not claims of native-format readers.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Iterator, Mapping

import numpy as np
from PIL import Image

SUPPORTED_DATASETS = ("tartanair", "pointodyssey", "wildrgbd", "hm3d", "waymo", "kitti_sceneflow")
SUPPORTED_SPLITS = ("train", "val", "test")
SUPPORTED_DEPTH_TYPES = ("camera_z",)
SUPPORTED_DEPTH_ENCODINGS = ("png_uint16", "tiff_uint16", "npy_float32", "npz_float32")
SCHEMA_VERSION = 1


class CrossSourceDataError(ValueError):
    """Raised when a manifest or decoded sample violates the shared contract."""


def _finite_positive(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CrossSourceDataError(f"{field} must be a finite positive number") from exc
    if not math.isfinite(number) or number <= 0:
        raise CrossSourceDataError(f"{field} must be a finite positive number")
    return number


def _resolve(base: Path, value: str | None, field: str, *, required: bool) -> Path | None:
    if value is None or str(value).strip() == "":
        if required:
            raise CrossSourceDataError(f"Missing required path field: {field}")
        return None
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _read_array(path: Path, encoding: str, *, npz_key: str | None = None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if encoding in ("png_uint16", "tiff_uint16"):
        array = np.asarray(Image.open(path))
        if array.ndim != 2 or array.dtype != np.uint16:
            raise CrossSourceDataError(
                f"{path}: expected a single-channel uint16 depth image for {encoding}; "
                f"got dtype={array.dtype}, shape={array.shape}"
            )
        return array
    if encoding == "npy_float32":
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
            raise CrossSourceDataError(
                f"{path}: expected a 2-D floating-point .npy depth map; "
                f"got dtype={array.dtype}, shape={array.shape}"
            )
        return array.astype(np.float32, copy=False)
    if encoding == "npz_float32":
        if not npz_key:
            raise CrossSourceDataError("depth.npz_key is required for npz_float32")
        with np.load(path, allow_pickle=False) as archive:
            if npz_key not in archive.files:
                raise CrossSourceDataError(f"{path}: npz key {npz_key!r} not found")
            array = archive[npz_key]
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
            raise CrossSourceDataError(
                f"{path}: expected a 2-D floating-point array at key {npz_key!r}; "
                f"got dtype={array.dtype}, shape={array.shape}"
            )
        return array.astype(np.float32, copy=False)
    raise CrossSourceDataError(f"Unsupported depth encoding: {encoding!r}")


def _read_mask(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        if mask.shape[2] not in (3, 4):
            raise CrossSourceDataError(f"Unsupported mask shape at {path}: {mask.shape}")
        mask = np.any(mask[..., :3] != 0, axis=2)
    if mask.ndim != 2:
        raise CrossSourceDataError(f"Validity mask must be 2-D: {path}, shape={mask.shape}")
    return mask.astype(bool, copy=False)


@dataclass(frozen=True)
class FrameRecord:
    frame_index: int
    timestamp_s: float
    rgb_path: Path
    depth_path: Path | None
    valid_mask_path: Path | None
    source_frame_id: str | None
    intrinsics: tuple[float, float, float, float] | None


@dataclass(frozen=True)
class FrameSample:
    """Canonical decoded sample: RGB uint8, camera-z depth in metres, valid mask."""
    dataset_id: str
    split: str
    scene_id: str
    sequence_id: str
    frame_index: int
    timestamp_s: float
    rgb: np.ndarray                 # H x W x 3, uint8, RGB order
    depth_m: np.ndarray | None      # H x W, float32 metres; invalid pixels are 0
    valid_mask: np.ndarray          # H x W, bool; all-false if no GT
    intrinsics: np.ndarray | None   # 3 x 3 pinhole K, metadata only in this phase
    source_frame_id: str | None

    @property
    def has_gt(self) -> bool:
        """Whether this frame carries usable depth supervision."""
        return self.depth_m is not None and bool(self.valid_mask.any())


@dataclass(frozen=True)
class SequenceManifest:
    path: Path
    dataset_id: str
    split: str
    scene_id: str
    sequence_id: str
    depth_encoding: str
    depth_scale_to_m: float
    depth_type: str
    registered_to_rgb: bool
    invalid_values: tuple[float, ...]
    min_depth_m: float | None
    max_depth_m: float | None
    npz_key: str | None
    camera_model: str | None
    camera_intrinsics: tuple[float, float, float, float] | None
    frames: tuple[FrameRecord, ...]

    @classmethod
    def read(cls, path: str | Path, *, verify_files: bool = True) -> "SequenceManifest":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
        except Exception as exc:
            raise CrossSourceDataError(f"Cannot parse sequence manifest {source}: {exc}") from exc
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise CrossSourceDataError(
                f"{source}: schema_version must be {SCHEMA_VERSION}, got {raw.get('schema_version')!r}"
            )
        dataset_id = str(raw.get("dataset_id", "")).strip().lower()
        split = str(raw.get("split", "")).strip().lower()
        if dataset_id not in SUPPORTED_DATASETS:
            raise CrossSourceDataError(f"Unsupported dataset_id {dataset_id!r}; expected {SUPPORTED_DATASETS}")
        if split not in SUPPORTED_SPLITS:
            raise CrossSourceDataError(f"Unsupported split {split!r}; expected {SUPPORTED_SPLITS}")
        scene_id = str(raw.get("scene_id", "")).strip()
        sequence_id = str(raw.get("sequence_id", "")).strip()
        if not scene_id or not sequence_id:
            raise CrossSourceDataError("scene_id and sequence_id must be non-empty")

        depth = raw.get("depth")
        if not isinstance(depth, Mapping):
            raise CrossSourceDataError("depth must be an object with explicit encoding and scale")
        encoding = str(depth.get("encoding", "")).strip().lower()
        if encoding not in SUPPORTED_DEPTH_ENCODINGS:
            raise CrossSourceDataError(f"Unsupported depth.encoding {encoding!r}; expected {SUPPORTED_DEPTH_ENCODINGS}")
        scale = _finite_positive(depth.get("scale_to_m"), "depth.scale_to_m")
        depth_type = str(depth.get("type", "")).strip().lower()
        if depth_type not in SUPPORTED_DEPTH_TYPES:
            raise CrossSourceDataError(
                f"Only camera_z GT is supported by the current P_multi loss; got depth.type={depth_type!r}. "
                "Convert radial/range depth to RGB-camera z-depth before manifest creation."
            )
        registered = depth.get("registered_to_rgb")
        if registered is not True:
            raise CrossSourceDataError(
                "depth.registered_to_rgb must be true. Unregistered depth/LiDAR must be calibrated, "
                "projected and z-buffered onto the RGB grid before training."
            )
        invalid_values = tuple(float(x) for x in depth.get("invalid_values", [0]))
        if any(not math.isfinite(x) for x in invalid_values):
            raise CrossSourceDataError("depth.invalid_values must contain finite raw values")
        min_depth = depth.get("min_depth_m")
        max_depth = depth.get("max_depth_m")
        min_depth = None if min_depth is None else _finite_positive(min_depth, "depth.min_depth_m")
        max_depth = None if max_depth is None else _finite_positive(max_depth, "depth.max_depth_m")
        if min_depth is not None and max_depth is not None and min_depth >= max_depth:
            raise CrossSourceDataError("depth.min_depth_m must be smaller than depth.max_depth_m")
        npz_key = depth.get("npz_key")
        if encoding == "npz_float32" and not npz_key:
            raise CrossSourceDataError("depth.npz_key is required for npz_float32")

        camera = raw.get("camera") or {}
        camera_model = camera.get("model")
        camera_intrinsics = _parse_intrinsics(camera.get("intrinsics"), "camera.intrinsics")
        if camera_model is not None and camera_model != "pinhole":
            raise CrossSourceDataError(f"Unsupported camera.model={camera_model!r}; only pinhole metadata is recognized")
        if camera_intrinsics is not None and camera_model != "pinhole":
            raise CrossSourceDataError("camera.model='pinhole' is required when camera.intrinsics is provided")

        rows = raw.get("frames")
        if not isinstance(rows, list) or not rows:
            raise CrossSourceDataError("frames must be a non-empty list")
        frames: list[FrameRecord] = []
        previous_ts = -math.inf
        seen_rgb: set[Path] = set()
        seen_depth: set[Path] = set()
        sequence_rgb_size: tuple[int, int] | None = None
        base = source.parent
        for expected_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise CrossSourceDataError(f"frames[{expected_index}] must be an object")
            index = row.get("frame_index")
            if isinstance(index, bool) or not isinstance(index, int) or index != expected_index:
                raise CrossSourceDataError(
                    f"frame indices must be contiguous from zero; frames[{expected_index}] has {index!r}"
                )
            timestamp = float(row.get("timestamp_s", float("nan")))
            if not math.isfinite(timestamp) or timestamp <= previous_ts:
                raise CrossSourceDataError(f"Non-finite or non-increasing timestamp at frame {expected_index}")
            previous_ts = timestamp
            rgb_path = _resolve(base, row.get("rgb"), f"frames[{expected_index}].rgb", required=True)
            depth_path = _resolve(base, row.get("depth"), f"frames[{expected_index}].depth", required=False)
            mask_path = _resolve(base, row.get("valid_mask"), f"frames[{expected_index}].valid_mask", required=False)
            if mask_path is not None and depth_path is None:
                raise CrossSourceDataError(f"Frame {expected_index} has valid_mask but no depth")
            intrinsics = _parse_intrinsics(row.get("intrinsics"), f"frames[{expected_index}].intrinsics")
            if rgb_path in seen_rgb:
                raise CrossSourceDataError(f"RGB file reused at frame {expected_index}: {rgb_path}")
            seen_rgb.add(rgb_path)
            if depth_path is not None:
                if depth_path in seen_depth:
                    raise CrossSourceDataError(f"GT depth file reused at frame {expected_index}: {depth_path}")
                seen_depth.add(depth_path)
            if verify_files:
                if not rgb_path.is_file():
                    raise FileNotFoundError(rgb_path)
                if depth_path is not None and not depth_path.is_file():
                    raise FileNotFoundError(depth_path)
                if mask_path is not None and not mask_path.is_file():
                    raise FileNotFoundError(mask_path)
                with Image.open(rgb_path) as rgb_image:
                    rgb_size = rgb_image.size
                if sequence_rgb_size is None:
                    sequence_rgb_size = rgb_size
                elif rgb_size != sequence_rgb_size:
                    raise CrossSourceDataError(
                        f"RGB dimensions change within {dataset_id}/{sequence_id}: "
                        f"frame {expected_index} has {rgb_size}, expected {sequence_rgb_size}"
                    )
            frames.append(FrameRecord(
                frame_index=index, timestamp_s=timestamp, rgb_path=rgb_path,
                depth_path=depth_path, valid_mask_path=mask_path,
                source_frame_id=None if row.get("source_frame_id") is None else str(row["source_frame_id"]),
                intrinsics=intrinsics,
            ))
        if not any(frame.depth_path is not None for frame in frames[1:]):
            raise CrossSourceDataError("A trainable sequence needs at least one GT frame after its first frame")
        return cls(
            path=source, dataset_id=dataset_id, split=split, scene_id=scene_id,
            sequence_id=sequence_id, depth_encoding=encoding, depth_scale_to_m=scale,
            depth_type=depth_type, registered_to_rgb=registered,
            invalid_values=invalid_values, min_depth_m=min_depth, max_depth_m=max_depth,
            npz_key=None if npz_key is None else str(npz_key),
            camera_model=camera_model,
            camera_intrinsics=camera_intrinsics, frames=tuple(frames),
        )

    def load_frame(self, index: int) -> FrameSample:
        record = self.frames[index]
        rgb = np.asarray(Image.open(record.rgb_path).convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        intrinsics_values = record.intrinsics or self.camera_intrinsics
        intrinsics = _intrinsics_matrix(intrinsics_values) if intrinsics_values is not None else None
        if record.depth_path is None:
            return FrameSample(
                self.dataset_id, self.split, self.scene_id, self.sequence_id,
                record.frame_index, record.timestamp_s, rgb, None,
                np.zeros((h, w), dtype=bool), intrinsics, record.source_frame_id,
            )

        raw_depth = _read_array(record.depth_path, self.depth_encoding, npz_key=self.npz_key)
        if raw_depth.shape != (h, w):
            raise CrossSourceDataError(
                f"RGB/depth registration shape mismatch in {self.dataset_id}/{self.sequence_id} "
                f"frame {index}: RGB={(h, w)}, depth={raw_depth.shape}. "
                "This adapter does not resize or register GT implicitly."
            )
        valid = np.isfinite(raw_depth) & (raw_depth > 0)
        for invalid in self.invalid_values:
            valid &= raw_depth != invalid
        depth_m = raw_depth.astype(np.float32, copy=False) * np.float32(self.depth_scale_to_m)
        if self.min_depth_m is not None:
            valid &= depth_m >= self.min_depth_m
        if self.max_depth_m is not None:
            valid &= depth_m <= self.max_depth_m
        if record.valid_mask_path is not None:
            supplied_mask = _read_mask(record.valid_mask_path)
            if supplied_mask.shape != (h, w):
                raise CrossSourceDataError(
                    f"Validity-mask shape mismatch at {record.valid_mask_path}: "
                    f"mask={supplied_mask.shape}, RGB={(h, w)}"
                )
            valid &= supplied_mask
        depth_m = np.where(valid, depth_m, 0.0).astype(np.float32, copy=False)
        if not valid.any():
            raise CrossSourceDataError(
                f"No valid GT pixels in {self.dataset_id}/{self.sequence_id} frame {index}"
            )
        return FrameSample(
            self.dataset_id, self.split, self.scene_id, self.sequence_id,
            record.frame_index, record.timestamp_s, rgb, depth_m, valid,
            intrinsics, record.source_frame_id,
        )


def _parse_intrinsics(value: Any, field: str) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise CrossSourceDataError(f"{field} must be [fx, fy, cx, cy]")
    values = tuple(float(x) for x in value)
    if not all(math.isfinite(x) for x in values) or values[0] <= 0 or values[1] <= 0:
        raise CrossSourceDataError(f"{field} must contain finite fx,fy>0 and finite cx,cy")
    return values


def _intrinsics_matrix(values: tuple[float, float, float, float]) -> np.ndarray:
    fx, fy, cx, cy = values
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


class ManifestSequenceAdapter:
    expected_dataset_id: str

    def __init__(self, manifest_path: str | Path, *, verify_files: bool = True):
        self.manifest = SequenceManifest.read(manifest_path, verify_files=verify_files)
        if self.manifest.dataset_id != self.expected_dataset_id:
            raise CrossSourceDataError(
                f"Adapter {self.expected_dataset_id!r} cannot read dataset_id={self.manifest.dataset_id!r}"
            )
        self._validate_dataset_contract()

    def _validate_dataset_contract(self) -> None:
        """Dataset-specific checks; override only when backed by verified metadata."""

    def __len__(self) -> int:
        return len(self.manifest.frames)

    def __getitem__(self, index: int) -> FrameSample:
        return self.manifest.load_frame(index)


class TartanAirAdapter(ManifestSequenceAdapter):
    expected_dataset_id = "tartanair"


class PointOdysseyAdapter(ManifestSequenceAdapter):
    expected_dataset_id = "pointodyssey"


class WildRGBDAdapter(ManifestSequenceAdapter):
    expected_dataset_id = "wildrgbd"


class HM3DAdapter(ManifestSequenceAdapter):
    expected_dataset_id = "hm3d"

    def _validate_dataset_contract(self) -> None:
        if self.manifest.camera_intrinsics is None and not all(f.intrinsics for f in self.manifest.frames):
            raise CrossSourceDataError(
                "HM3D rendered sequences require pinhole intrinsics at sequence or frame level; "
                "raw HM3D scene assets are not accepted as a camera trajectory."
            )


class WaymoAdapter(ManifestSequenceAdapter):
    expected_dataset_id = "waymo"

    def _validate_dataset_contract(self) -> None:
        # The current repair loss expects camera-z depth on the RGB pixel grid.
        # Raw LiDAR range images are not accepted as camera-aligned GT.
        if self.manifest.camera_intrinsics is None and not all(f.intrinsics for f in self.manifest.frames):
            raise CrossSourceDataError(
                "Waymo manifest requires pinhole intrinsics at sequence or frame level. "
                "Project LiDAR points with the matching camera calibration before training."
            )
        for frame in self.manifest.frames:
            if frame.depth_path is not None and frame.valid_mask_path is None:
                raise CrossSourceDataError(
                    f"Waymo frame {frame.frame_index} requires an explicit projected-LiDAR valid_mask"
                )


class KITTISceneFlowAdapter(ManifestSequenceAdapter):
    expected_dataset_id = "kitti_sceneflow"

    def _validate_dataset_contract(self) -> None:
        if self.manifest.camera_intrinsics is None and not all(f.intrinsics for f in self.manifest.frames):
            raise CrossSourceDataError("KITTI Scene Flow requires rectified left-camera intrinsics")
        if len(self.manifest.frames) != 20:
            raise CrossSourceDataError(
                f"KITTI Scene Flow CrossSource sequences must contain exactly 20 MultiView frames; "
                f"got {len(self.manifest.frames)}"
            )
        labelled = [f.source_frame_id for f in self.manifest.frames if f.depth_path is not None]
        if len(labelled) < 2 or not any(str(x).endswith("_10") for x in labelled) or not any(str(x).endswith("_11") for x in labelled):
            raise CrossSourceDataError(
                "KITTI Scene Flow sequence must include both labelled t0/t1 disparity frames (_10 and _11)"
            )


# Keep the original Phase 1-2 five-source registry stable for old tests/configs.
# CrossSource-v2 extends it with KITTI Scene Flow through ALL_ADAPTERS.
ADAPTERS = {
    "tartanair": TartanAirAdapter,
    "pointodyssey": PointOdysseyAdapter,
    "wildrgbd": WildRGBDAdapter,
    "hm3d": HM3DAdapter,
    "waymo": WaymoAdapter,
}
ALL_ADAPTERS = {**ADAPTERS, "kitti_sceneflow": KITTISceneFlowAdapter}


@dataclass(frozen=True)
class WindowRef:
    dataset_id: str
    scene_id: str
    sequence_id: str
    manifest_path: Path
    start_index: int
    length: int


class CrossSourcePool:
    """Collection of normalized train/val sequences with leakage checks."""
    def __init__(self, manifests: Iterable[str | Path], *, split: str, verify_files: bool = True):
        if split not in ("train", "val"):
            raise CrossSourceDataError("CrossSourcePool is for train/val only; test data must use a separate evaluator")
        self.split = split
        self.sequences: dict[tuple[str, str], ManifestSequenceAdapter] = {}
        self.by_scene: dict[tuple[str, str], list[ManifestSequenceAdapter]] = {}
        for path in manifests:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            dataset_id = str(raw.get("dataset_id", "")).strip().lower()
            adapter_cls = ALL_ADAPTERS.get(dataset_id)
            if adapter_cls is None:
                raise CrossSourceDataError(f"No adapter registered for dataset_id={dataset_id!r}")
            adapter = adapter_cls(path, verify_files=verify_files)
            if adapter.manifest.split != split:
                raise CrossSourceDataError(
                    f"Manifest split mismatch: requested {split}, got {adapter.manifest.split} in {path}"
                )
            key = (dataset_id, adapter.manifest.sequence_id)
            if key in self.sequences:
                raise CrossSourceDataError(f"Duplicate sequence_id within dataset: {key}")
            self.sequences[key] = adapter
            scene_key = (dataset_id, adapter.manifest.scene_id)
            self.by_scene.setdefault(scene_key, []).append(adapter)
        if not self.sequences:
            raise CrossSourceDataError(f"No {split} manifests supplied")

    @classmethod
    def from_config(cls, config_path: str | Path, *, split: str, verify_files: bool = True) -> "CrossSourcePool":
        source = Path(config_path).expanduser().resolve()
        config = json.loads(source.read_text(encoding="utf-8"))
        if config.get("schema_version") != SCHEMA_VERSION:
            raise CrossSourceDataError(f"Config schema_version must be {SCHEMA_VERSION}")
        split_manifests = config.get("manifests", {}).get(split)
        if not isinstance(split_manifests, Mapping):
            raise CrossSourceDataError(f"Config has no manifests.{split} mapping")
        paths: list[Path] = []
        configured_ids = config.get("dataset_ids")
        if configured_ids is None:
            configured_ids = [k for k, v in split_manifests.items() if v]
        if not isinstance(configured_ids, list) or not configured_ids:
            raise CrossSourceDataError("Config must declare a non-empty dataset_ids list or non-empty manifests")
        unknown = sorted(set(configured_ids) - set(SUPPORTED_DATASETS))
        if unknown:
            raise CrossSourceDataError(f"Unsupported dataset_ids in config: {unknown}")
        missing = [dataset_id for dataset_id in configured_ids if not split_manifests.get(dataset_id)]
        if missing:
            raise CrossSourceDataError(f"CrossSource {split} config is missing required datasets: {missing}")
        for dataset_id in configured_ids:
            values = split_manifests.get(dataset_id, [])
            if not isinstance(values, list):
                raise CrossSourceDataError(f"manifests.{split}.{dataset_id} must be a list")
            for item in values:
                p = Path(item).expanduser()
                manifest_path = (p if p.is_absolute() else source.parent / p).resolve()
                raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                actual_dataset = str(raw_manifest.get("dataset_id", "")).strip().lower()
                if actual_dataset != dataset_id:
                    raise CrossSourceDataError(
                        f"Config key {dataset_id!r} points to manifest for {actual_dataset!r}: {manifest_path}"
                    )
                paths.append(manifest_path)
        pool = cls(paths, split=split, verify_files=verify_files)
        return pool

    def validate_disjoint(self, other: "CrossSourcePool") -> None:
        if self.split == other.split:
            raise CrossSourceDataError("Disjointness check requires different splits")
        scenes_a = set(self.by_scene)
        scenes_b = set(other.by_scene)
        overlap_scenes = scenes_a & scenes_b
        if overlap_scenes:
            raise CrossSourceDataError(f"Scene leakage between {self.split} and {other.split}: {sorted(overlap_scenes)}")
        files_a, files_b = set(), set()
        for adapter in self.sequences.values():
            for frame in adapter.manifest.frames:
                files_a.add(frame.rgb_path)
                if frame.depth_path:
                    files_a.add(frame.depth_path)
                if frame.valid_mask_path:
                    files_a.add(frame.valid_mask_path)
        for adapter in other.sequences.values():
            for frame in adapter.manifest.frames:
                files_b.add(frame.rgb_path)
                if frame.depth_path:
                    files_b.add(frame.depth_path)
                if frame.valid_mask_path:
                    files_b.add(frame.valid_mask_path)
        overlap_files = files_a & files_b
        if overlap_files:
            raise CrossSourceDataError(f"Physical RGB/GT file overlap between splits: {next(iter(overlap_files))}")

    def summary(self) -> dict[str, Any]:
        datasets: dict[str, dict[str, int]] = {}
        for (dataset_id, _), adapter in self.sequences.items():
            bucket = datasets.setdefault(dataset_id, {"sequences": 0, "scenes": 0, "rgb_frames": 0, "gt_frames": 0})
            bucket["sequences"] += 1
            bucket["rgb_frames"] += len(adapter.manifest.frames)
            bucket["gt_frames"] += sum(f.depth_path is not None for f in adapter.manifest.frames)
        for dataset_id in datasets:
            datasets[dataset_id]["scenes"] = len({scene for ds, scene in self.by_scene if ds == dataset_id})
        return {"split": self.split, "total_sequences": len(self.sequences),
                "total_scenes": len(self.by_scene), "datasets": datasets}

    def adapter(self, dataset_id: str, sequence_id: str) -> ManifestSequenceAdapter:
        return self.sequences[(dataset_id, sequence_id)]


class SceneBalancedWindowSampler:
    """Uniform-over-scenes sampler for fixed-length causal training windows.

    Dataset probability is therefore proportional to its number of eligible
    scenes; sequences within a scene are sampled uniformly. This is an explicit
    SVR-Depth policy and is not claimed to reproduce unpublished VeloDepth
    per-dataset weights.
    """
    def __init__(self, pool: CrossSourcePool, *, window_frames: int = 32,
                 min_supervised_frames: int = 1, seed: int = 0):
        if pool.split != "train":
            raise CrossSourceDataError("SceneBalancedWindowSampler only accepts a train pool")
        if window_frames < 2 or min_supervised_frames < 1 or min_supervised_frames > window_frames - 1:
            raise CrossSourceDataError("Invalid window_frames/min_supervised_frames")
        self.pool = pool
        self.window_frames = window_frames
        self.min_supervised_frames = min_supervised_frames
        self.rng = random.Random(seed)
        self.eligible: dict[tuple[str, str], list[tuple[ManifestSequenceAdapter, list[int]]]] = {}
        for scene_key, adapters in pool.by_scene.items():
            eligible_sequences = []
            for adapter in adapters:
                frames = adapter.manifest.frames
                n = len(frames)
                if n < window_frames:
                    continue
                supervised = [int(f.depth_path is not None) for f in frames]
                prefix = [0]
                for value in supervised:
                    prefix.append(prefix[-1] + value)
                starts = []
                for start in range(0, n - window_frames + 1):
                    # Frame zero is the keyframe; supervise propagated frames only.
                    count = prefix[start + window_frames] - prefix[start + 1]
                    if count >= min_supervised_frames:
                        starts.append(start)
                if starts:
                    eligible_sequences.append((adapter, starts))
            if eligible_sequences:
                self.eligible[scene_key] = eligible_sequences
        if not self.eligible:
            raise CrossSourceDataError(
                f"No eligible {window_frames}-frame windows with at least "
                f"{min_supervised_frames} supervised propagated frames"
            )
        self.scene_keys = sorted(self.eligible)

    def sample(self) -> WindowRef:
        scene_key = self.rng.choice(self.scene_keys)
        adapter, starts = self.rng.choice(self.eligible[scene_key])
        start = self.rng.choice(starts)
        m = adapter.manifest
        return WindowRef(m.dataset_id, m.scene_id, m.sequence_id, m.path, start, self.window_frames)

    def sample_many(self, steps: int) -> list[WindowRef]:
        if steps < 0:
            raise CrossSourceDataError("steps must be nonnegative")
        return [self.sample() for _ in range(steps)]

    def load_window(self, ref: WindowRef) -> list[FrameSample]:
        adapter = self.pool.adapter(ref.dataset_id, ref.sequence_id)
        if adapter.manifest.path != ref.manifest_path:
            raise CrossSourceDataError("WindowRef manifest does not match the current pool")
        return [adapter[i] for i in range(ref.start_index, ref.start_index + ref.length)]


def audit_config(config_path: str | Path, *, decode_samples: bool = True) -> dict[str, Any]:
    train = CrossSourcePool.from_config(config_path, split="train", verify_files=True)
    val = CrossSourcePool.from_config(config_path, split="val", verify_files=True)
    train.validate_disjoint(val)
    result: dict[str, Any] = {"train": train.summary(), "val": val.summary(), "disjoint": True}
    if decode_samples:
        decoded = []
        for split_pool in (train, val):
            for adapter in split_pool.sequences.values():
                first = adapter[0]
                gt_frame = next((i for i, f in enumerate(adapter.manifest.frames) if f.depth_path), None)
                gt_sample = adapter[gt_frame] if gt_frame is not None else None
                decoded.append({
                    "split": split_pool.split, "dataset_id": first.dataset_id,
                    "scene_id": first.scene_id, "sequence_id": first.sequence_id,
                    "frames": len(adapter), "rgb_shape": list(first.rgb.shape),
                    "gt_shape": None if gt_sample is None or gt_sample.depth_m is None else list(gt_sample.depth_m.shape),
                    "gt_valid_pixels": 0 if gt_sample is None else int(gt_sample.valid_mask.sum()),
                    "depth_scale_to_m": adapter.manifest.depth_scale_to_m,
                    "depth_type": adapter.manifest.depth_type,
                    "registered_to_rgb": adapter.manifest.registered_to_rgb,
                    "intrinsics_available": first.intrinsics is not None,
                })
        result["decoded_sequences"] = decoded
    training_cfg = json.loads(Path(config_path).read_text(encoding="utf-8")).get("training", {})
    window_frames = int(training_cfg.get("window_frames", 32))
    weights = training_cfg.get("dataset_weights")
    if weights:
        sampler = DatasetBalancedWindowSampler(
            train, dataset_weights=weights, window_frames=window_frames,
            min_supervised_frames=int(training_cfg.get("min_supervised_frames", 1)), seed=20261103
        )
        eligible_scenes = sum(len(v) for v in sampler._dataset_scenes.values())
        policy = "explicit dataset weights -> uniform scene -> sequence -> window"
    else:
        sampler = SceneBalancedWindowSampler(
            train, window_frames=window_frames,
            min_supervised_frames=int(training_cfg.get("min_supervised_frames", 1)), seed=20261103
        )
        eligible_scenes = len(sampler.eligible)
        policy = "uniform eligible scene -> sequence -> window"
    refs = sampler.sample_many(100)
    counts: dict[str, int] = {}
    for ref in refs:
        counts[ref.dataset_id] = counts.get(ref.dataset_id, 0) + 1
    result["sampler_check"] = {
        "policy": policy, "eligible_scenes": eligible_scenes,
        "sampled_windows": len(refs), "dataset_window_counts_over_100_draws": counts,
        "dataset_weights": weights,
    }
    return result


class DatasetBalancedWindowSampler:
    """Two-level sampler: choose dataset by explicit weight, then scene/sequence/window.

    This differs from SceneBalancedWindowSampler, whose global uniform-over-scenes
    policy makes dataset probability proportional to its number of scenes. For the
    TartanAir+KITTI experiment we use an explicit 50:50 dataset prior by default.
    """
    def __init__(self, pool: CrossSourcePool, *, dataset_weights: Mapping[str, float] | None = None,
                 window_frames: int = 20, min_supervised_frames: int = 1, seed: int = 0):
        if pool.split != "train":
            raise CrossSourceDataError("DatasetBalancedWindowSampler only accepts a train pool")
        if window_frames < 2 or not 1 <= min_supervised_frames <= window_frames - 1:
            raise CrossSourceDataError("Invalid window_frames/min_supervised_frames")
        self.pool = pool
        self.window_frames = int(window_frames)
        self.min_supervised_frames = int(min_supervised_frames)
        self.rng = random.Random(seed)
        available = sorted({dataset_id for dataset_id, _ in pool.by_scene})
        if dataset_weights is None:
            dataset_weights = {x: 1.0 / len(available) for x in available}
        weights = {str(k): float(v) for k, v in dataset_weights.items()}
        if set(weights) != set(available):
            raise CrossSourceDataError(
                f"dataset_weights keys must exactly match available datasets {available}; got {sorted(weights)}"
            )
        if any(not math.isfinite(v) or v <= 0 for v in weights.values()):
            raise CrossSourceDataError("dataset weights must be finite and positive")
        total = sum(weights.values())
        self.dataset_weights = {k: v / total for k, v in weights.items()}
        self.dataset_ids = sorted(self.dataset_weights)
        self._dataset_scenes: dict[str, dict[tuple[str, str], list[tuple[ManifestSequenceAdapter, list[int]]]]] = {}
        for dataset_id in self.dataset_ids:
            by_scene: dict[tuple[str, str], list[tuple[ManifestSequenceAdapter, list[int]]]] = {}
            for scene_key, adapters in pool.by_scene.items():
                if scene_key[0] != dataset_id:
                    continue
                eligible_sequences = []
                for adapter in adapters:
                    frames = adapter.manifest.frames
                    if len(frames) < self.window_frames:
                        continue
                    supervised = [int(f.depth_path is not None) for f in frames]
                    prefix = [0]
                    for value in supervised:
                        prefix.append(prefix[-1] + value)
                    starts = []
                    for start in range(0, len(frames) - self.window_frames + 1):
                        count = prefix[start + self.window_frames] - prefix[start + 1]
                        if count >= self.min_supervised_frames:
                            starts.append(start)
                    if starts:
                        eligible_sequences.append((adapter, starts))
                if eligible_sequences:
                    by_scene[scene_key] = eligible_sequences
            if not by_scene:
                raise CrossSourceDataError(
                    f"Dataset {dataset_id} has no eligible {self.window_frames}-frame windows with "
                    f"at least {self.min_supervised_frames} supervised propagated frames"
                )
            self._dataset_scenes[dataset_id] = by_scene

    def sample(self) -> WindowRef:
        dataset_id = self.rng.choices(
            self.dataset_ids, weights=[self.dataset_weights[x] for x in self.dataset_ids], k=1
        )[0]
        scenes = self._dataset_scenes[dataset_id]
        scene_key = self.rng.choice(sorted(scenes))
        adapter, starts = self.rng.choice(scenes[scene_key])
        start = self.rng.choice(starts)
        m = adapter.manifest
        return WindowRef(m.dataset_id, m.scene_id, m.sequence_id, m.path, start, self.window_frames)

    def sample_many(self, steps: int) -> list[WindowRef]:
        if steps < 0:
            raise CrossSourceDataError("steps must be nonnegative")
        return [self.sample() for _ in range(steps)]

    def load_window(self, ref: WindowRef) -> list[FrameSample]:
        adapter = self.pool.adapter(ref.dataset_id, ref.sequence_id)
        if adapter.manifest.path != ref.manifest_path:
            raise CrossSourceDataError("WindowRef manifest does not match current pool")
        return [adapter[i] for i in range(ref.start_index, ref.start_index + ref.length)]
