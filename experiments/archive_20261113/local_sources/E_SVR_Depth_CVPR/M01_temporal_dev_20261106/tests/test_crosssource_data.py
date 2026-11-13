import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from svr_depth.datasets.crosssource import (
    ADAPTERS, CrossSourceDataError, CrossSourcePool,
    SceneBalancedWindowSampler, SequenceManifest, audit_config,
)


class CrossSourceContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_sequence(self, dataset_id, split, scene_id, sequence_id,
                        *, n=32, waymo=False, scale=0.001, depth_encoding="png_uint16",
                        registered=True, depth_type="camera_z", scene_suffix=""):
        folder = self.root / split / dataset_id / f"{sequence_id}{scene_suffix}"
        folder.mkdir(parents=True)
        frames = []
        for i in range(n):
            rgb_path = folder / f"rgb_{i:04d}.png"
            Image.fromarray(np.full((8, 10, 3), (i % 255), dtype=np.uint8), mode="RGB").save(rgb_path)
            depth_path = None
            mask_path = None
            if i > 0:
                if depth_encoding == "png_uint16":
                    depth = np.full((8, 10), 1000, dtype=np.uint16)
                    depth[0, 0] = 0
                    depth_path = folder / f"depth_{i:04d}.png"
                    Image.fromarray(depth).save(depth_path)
                else:
                    depth = np.full((8, 10), 1.0, dtype=np.float32)
                    depth_path = folder / f"depth_{i:04d}.npy"
                    np.save(depth_path, depth)
                if waymo:
                    mask = np.ones((8, 10), dtype=np.uint8)
                    mask[0, 0] = 0
                    mask_path = folder / f"mask_{i:04d}.png"
                    Image.fromarray(mask).save(mask_path)
            frames.append({
                "frame_index": i,
                "timestamp_s": i / 30.0,
                "source_frame_id": f"src-{i}",
                "rgb": rgb_path.name,
                "depth": None if depth_path is None else depth_path.name,
                "valid_mask": None if mask_path is None else mask_path.name,
            })
        manifest = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "split": split,
            "scene_id": scene_id,
            "sequence_id": sequence_id,
            "depth": {
                "encoding": depth_encoding,
                "scale_to_m": scale,
                "type": depth_type,
                "registered_to_rgb": registered,
                "invalid_values": [0],
            },
            "camera": {"model": "pinhole", "intrinsics": [100.0, 100.0, 5.0, 4.0]},
            "frames": frames,
        }
        path = folder / "sequence.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_all_five_manifest_adapters_decode_canonical_samples(self):
        for dataset_id, adapter_cls in ADAPTERS.items():
            with self.subTest(dataset=dataset_id):
                path = self._write_sequence(dataset_id, "train", f"scene-{dataset_id}", f"seq-{dataset_id}", waymo=(dataset_id == "waymo"))
                adapter = adapter_cls(path)
                sample = adapter[1]
                self.assertEqual(sample.rgb.shape, (8, 10, 3))
                self.assertEqual(sample.rgb.dtype, np.uint8)
                self.assertEqual(sample.depth_m.shape, (8, 10))
                self.assertEqual(sample.depth_m.dtype, np.float32)
                self.assertAlmostEqual(float(sample.depth_m[1, 1]), 1.0, places=6)
                self.assertFalse(bool(sample.valid_mask[0, 0]))
                self.assertEqual(sample.intrinsics.shape, (3, 3))

    def test_rejects_unregistered_depth_and_non_camera_z(self):
        path = self._write_sequence("tartanair", "train", "s1", "q1", registered=False)
        with self.assertRaisesRegex(CrossSourceDataError, "registered_to_rgb"):
            SequenceManifest.read(path)
        path = self._write_sequence("pointodyssey", "train", "s2", "q2", depth_type="radial_range")
        with self.assertRaisesRegex(CrossSourceDataError, "camera_z"):
            SequenceManifest.read(path)

    def test_hm3d_requires_render_camera_intrinsics(self):
        path = self._write_sequence("hm3d", "train", "s1", "q1")
        payload = json.loads(path.read_text())
        payload["camera"]["intrinsics"] = None
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(CrossSourceDataError, "pinhole intrinsics"):
            ADAPTERS["hm3d"](path)

    def test_waymo_requires_intrinsics_and_explicit_sparse_mask(self):
        path = self._write_sequence("waymo", "train", "s1", "q1", waymo=False)
        payload = json.loads(path.read_text())
        payload["camera"]["intrinsics"] = None
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(CrossSourceDataError, "intrinsics"):
            ADAPTERS["waymo"](path)
        payload["camera"]["intrinsics"] = [100, 100, 5, 4]
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(CrossSourceDataError, "valid_mask"):
            ADAPTERS["waymo"](path)

    def test_scene_balanced_sampler_is_deterministic_and_samples_windows(self):
        manifests = []
        for dataset_id in ADAPTERS:
            manifests.append(self._write_sequence(
                dataset_id, "train", f"scene-{dataset_id}", f"seq-{dataset_id}",
                waymo=(dataset_id == "waymo"),
            ))
        pool = CrossSourcePool(manifests, split="train")
        sampler_a = SceneBalancedWindowSampler(pool, window_frames=32, seed=42)
        sampler_b = SceneBalancedWindowSampler(pool, window_frames=32, seed=42)
        refs_a = sampler_a.sample_many(25)
        refs_b = sampler_b.sample_many(25)
        self.assertEqual(refs_a, refs_b)
        self.assertEqual({x.dataset_id for x in refs_a}, set(ADAPTERS))
        loaded = sampler_a.load_window(refs_a[0])
        self.assertEqual(len(loaded), 32)
        self.assertEqual(loaded[0].frame_index, 0)
        self.assertEqual(loaded[-1].frame_index, 31)

    def test_train_val_scene_overlap_is_rejected(self):
        train = self._write_sequence("tartanair", "train", "same-scene", "train-seq")
        val = self._write_sequence("tartanair", "val", "same-scene", "val-seq")
        train_pool = CrossSourcePool([train], split="train")
        val_pool = CrossSourcePool([val], split="val")
        with self.assertRaisesRegex(CrossSourceDataError, "Scene leakage"):
            train_pool.validate_disjoint(val_pool)

    def test_depth_shape_mismatch_fails_fast(self):
        path = self._write_sequence("wildrgbd", "train", "s1", "q1")
        payload = json.loads(path.read_text())
        depth_path = path.parent / payload["frames"][1]["depth"]
        Image.fromarray(np.ones((7, 10), dtype=np.uint16)).save(depth_path)
        adapter = ADAPTERS["wildrgbd"](path)
        with self.assertRaisesRegex(CrossSourceDataError, "shape mismatch"):
            adapter[1]

    def test_config_loads_five_sources_and_rejects_wrong_dataset_key(self):
        config_path = self.root / "crosssource.json"
        config = {"schema_version": 1, "manifests": {"train": {}, "val": {}}}
        for split in ("train", "val"):
            for dataset_id in ADAPTERS:
                manifest = self._write_sequence(
                    dataset_id, split, f"{split}-scene-{dataset_id}", f"{split}-seq-{dataset_id}",
                    waymo=(dataset_id == "waymo"),
                )
                config["manifests"][split][dataset_id] = [str(manifest.relative_to(self.root))]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        train = CrossSourcePool.from_config(config_path, split="train")
        val = CrossSourcePool.from_config(config_path, split="val")
        train.validate_disjoint(val)
        self.assertEqual(set(train.summary()["datasets"]), set(ADAPTERS))
        config["manifests"]["train"]["tartanair"][0] = config["manifests"]["train"]["pointodyssey"][0]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(CrossSourceDataError, "Config key"):
            CrossSourcePool.from_config(config_path, split="train")

    def test_audit_config_decodes_train_val_and_reports_sampler(self):
        config_path = self.root / "audit_config.json"
        config = {"schema_version": 1, "manifests": {"train": {}, "val": {}}}
        for split in ("train", "val"):
            for dataset_id in ADAPTERS:
                manifest = self._write_sequence(
                    dataset_id, split, f"audit-{split}-{dataset_id}", f"audit-{split}-{dataset_id}",
                    waymo=(dataset_id == "waymo"),
                )
                config["manifests"][split][dataset_id] = [str(manifest.relative_to(self.root))]
        config_path.write_text(json.dumps(config), encoding="utf-8")
        report = audit_config(config_path, decode_samples=True)
        self.assertTrue(report["disjoint"])
        self.assertEqual(report["sampler_check"]["eligible_scenes"], 5)
        self.assertEqual(len(report["decoded_sequences"]), 10)


if __name__ == "__main__":
    unittest.main()
