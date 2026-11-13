import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import io
import h5py

from svr_depth.datasets.crosssource import (
    CrossSourcePool, DatasetBalancedWindowSampler, KITTISceneFlowAdapter, audit_config,
)
from svr_depth.datasets.native_tartanair import build_tartanair_hdf5_manifests, build_tartanair_manifests
from svr_depth.datasets.kitti_sceneflow import build_kitti_sceneflow_manifests


class CrossSourceV2NativeAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_tartan_traj(self, env: str):
        traj = self.root / "tartan" / env / "Easy" / "P000"
        (traj / "image_left").mkdir(parents=True)
        (traj / "depth_left").mkdir(parents=True)
        for i in range(20):
            Image.fromarray(np.full((16, 20, 3), i, dtype=np.uint8)).save(traj / "image_left" / f"{i:06d}_left.png")
            np.save(traj / "depth_left" / f"{i:06d}_left_depth.npy", np.full((16, 20), 2.0, dtype=np.float32))

    def test_tartanair_native_layout_manifest_and_env_split(self):
        self._make_tartan_traj("EnvTrain")
        self._make_tartan_traj("EnvVal")
        result = build_tartanair_manifests(
            self.root / "tartan", self.root / "tartan_out", val_envs=["EnvVal"],
            fps=10.0, depth_type="camera_z", min_frames=20,
        )
        self.assertEqual(result["stats"]["sequences_written"], 2)
        train_manifest = Path(result["manifests"]["train"][0])
        val_manifest = Path(result["manifests"]["val"][0])
        self.assertIn("EnvTrain", train_manifest.read_text())
        self.assertIn("EnvVal", val_manifest.read_text())
        train = CrossSourcePool([train_manifest], split="train")
        sample = train.adapter("tartanair", "EnvTrain/Easy/P000/run00")[3]
        self.assertTrue(sample.has_gt)
        self.assertAlmostEqual(float(sample.depth_m[0, 0]), 2.0, places=6)

    def _make_kitti(self, root: Path, scene: str):
        mv = root / "mv" / "training" / "image_2"
        main0 = root / "main" / "training" / "disp_occ_0"
        main1 = root / "main" / "training" / "disp_occ_1"
        calib = root / "calib" / "training" / "calib_cam_to_cam"
        for d in (mv, main0, main1, calib): d.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            Image.fromarray(np.full((40, 40, 3), i, dtype=np.uint8)).save(mv / f"{scene}_{i:02d}.png")
        disp = np.full((40, 40), 2560, dtype=np.uint16)  # 10 px disparity
        Image.fromarray(disp).save(main0 / f"{scene}_10.png")
        Image.fromarray(disp).save(main1 / f"{scene}_10.png")
        text = (
            "P_rect_02: 100 0 20 0 0 100 20 0 0 0 1 0\n"
            "P_rect_03: 100 0 20 -10 0 100 20 0 0 0 1 0\n"
        )
        (calib / f"{scene}.txt").write_text(text)


    def test_tartanair_velodepth_hdf5_schema_export(self):
        hdf5_path = self.root / "TartanAir.hdf5"
        sequences = {}
        split_lines = []
        with h5py.File(hdf5_path, "w") as h5:
            for env in ("EnvTrain", "EnvVal"):
                sequence_name = f"{env}/Easy/P000"
                split_lines.append(f"{sequence_name} 20")
                images, depths, Ks = [], [], []
                for i in range(20):
                    rgb_io = io.BytesIO()
                    Image.fromarray(np.full((16, 20, 3), i, dtype=np.uint8)).save(rgb_io, format="PNG")
                    depth_io = io.BytesIO()
                    Image.fromarray(np.full((16, 20), 2000, dtype=np.uint16)).save(depth_io, format="PNG")
                    rgb_key, depth_key = f"rgb/{env}_{i:06d}", f"depth/{env}_{i:06d}"
                    h5.create_dataset(rgb_key, data=np.frombuffer(rgb_io.getvalue(), dtype=np.uint8))
                    h5.create_dataset(depth_key, data=np.frombuffer(depth_io.getvalue(), dtype=np.uint8))
                    images.append(rgb_key); depths.append(depth_key)
                    Ks.append([[100.0, 0.0, 10.0], [0.0, 100.0, 8.0], [0.0, 0.0, 1.0]])
                sequences[sequence_name] = {"image": images, "depth": depths, "K": Ks}
            h5.create_dataset("train.txt", data=np.frombuffer(("\n".join(split_lines) + "\n").encode("ascii"), dtype=np.uint8))
            h5.create_dataset("sequences.json", data=np.frombuffer(__import__("json").dumps(sequences).encode("ascii"), dtype=np.uint8))
        result = build_tartanair_hdf5_manifests(
            hdf5_path, self.root / "hdf5_out", val_envs=["EnvVal"], fps=10.0,
            depth_scale=1000.0, min_frames=20,
        )
        self.assertEqual(result["stats"]["sequences_written"], 2)
        train_path = Path(result["manifests"]["train"][0])
        train_pool = CrossSourcePool([train_path], split="train")
        sample = train_pool.adapter("tartanair", "EnvTrain/Easy/P000")[3]
        self.assertTrue(sample.has_gt)
        self.assertAlmostEqual(float(sample.depth_m[0, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(sample.intrinsics[0, 0]), 100.0, places=6)

    def test_kitti_scene_flow_multiview_metric_depth_and_sparse_gt(self):
        self._make_kitti(self.root, "000000")
        self._make_kitti(self.root, "000001")
        result = build_kitti_sceneflow_manifests(
            self.root / "mv", self.root / "main", self.root / "calib", self.root / "kitti_out",
            val_fraction=0.5, seed=7,
        )
        self.assertEqual(result["stats"]["sequences_written"], 2)
        all_manifests = result["manifests"]["train"] + result["manifests"]["val"]
        for path in all_manifests:
            from svr_depth.datasets.crosssource import SequenceManifest
            manifest = SequenceManifest.read(path)
            adapter = KITTISceneFlowAdapter(path)
            self.assertEqual(len(adapter), 20)
            gt_indices = [i for i, f in enumerate(manifest.frames) if f.depth_path is not None]
            self.assertEqual(gt_indices, [10, 11])
            sample = adapter[10]
            self.assertAlmostEqual(float(sample.depth_m[10, 10]), 1.0, places=5)
            self.assertTrue(sample.has_gt)
            self.assertFalse(adapter[0].has_gt)

    def test_dataset_balanced_sampler_respects_50_50_prior(self):
        self._make_tartan_traj("EnvTrain")
        self._make_tartan_traj("EnvVal")
        tartan = build_tartanair_manifests(
            self.root / "tartan", self.root / "tartan_out", val_envs=["EnvVal"],
            fps=10.0, depth_type="camera_z", min_frames=20,
        )
        self._make_kitti(self.root, "000000")
        self._make_kitti(self.root, "000001")
        kitti = build_kitti_sceneflow_manifests(
            self.root / "mv", self.root / "main", self.root / "calib", self.root / "kitti_out",
            val_fraction=0.5, seed=7,
        )
        train_paths = tartan["manifests"]["train"] + kitti["manifests"]["train"]
        pool = CrossSourcePool(train_paths, split="train")
        sampler = DatasetBalancedWindowSampler(
            pool, dataset_weights={"tartanair": 0.5, "kitti_sceneflow": 0.5},
            window_frames=20, seed=42,
        )
        refs = sampler.sample_many(1000)
        counts = {"tartanair": 0, "kitti_sceneflow": 0}
        for ref in refs:
            counts[ref.dataset_id] += 1
        self.assertGreater(counts["tartanair"], 400)
        self.assertGreater(counts["kitti_sceneflow"], 400)
        self.assertLess(abs(counts["tartanair"] - counts["kitti_sceneflow"]), 100)
        window = sampler.load_window(refs[0])
        self.assertEqual(len(window), 20)

        # Exercise the generated A/B/C-style dynamic config path and its sampler audit.
        config = {
            "schema_version": 1,
            "dataset_ids": ["tartanair", "kitti_sceneflow"],
            "manifests": {
                "train": {"tartanair": tartan["manifests"]["train"],
                          "kitti_sceneflow": kitti["manifests"]["train"]},
                "val": {"tartanair": tartan["manifests"]["val"],
                        "kitti_sceneflow": kitti["manifests"]["val"]},
            },
            "training": {"window_frames": 20, "tbptt_frames": 4, "min_supervised_frames": 1,
                         "dataset_weights": {"tartanair": 0.5, "kitti_sceneflow": 0.5}},
        }
        config_path = self.root / "crosssource_v2.json"
        config_path.write_text(__import__("json").dumps(config), encoding="utf-8")
        report = audit_config(config_path, decode_samples=True)
        self.assertTrue(report["disjoint"])
        self.assertEqual(report["sampler_check"]["dataset_weights"], config["training"]["dataset_weights"])



if __name__ == "__main__":
    unittest.main()
