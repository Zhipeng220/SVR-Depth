import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    from . import metrics
    from .official_eval_entry import checked_environment
except ImportError:
    import metrics
    from official_eval_entry import checked_environment


def inputs(prev, curr=None):
    prev = np.asarray(prev, dtype=float)
    curr = prev.copy() if curr is None else np.asarray(curr, dtype=float)
    h, w = prev.shape
    mask = np.ones((h, w), dtype=bool)
    flow = np.zeros((h, w, 2), dtype=float)
    K = np.array([[3., 0, (w-1)/2], [0, 3., (h-1)/2], [0, 0, 1.]])
    return dict(prev_pred=prev, curr_pred=curr, prev_gt=prev.copy(), curr_gt=curr.copy(),
                prev_gt_mask=mask.copy(), curr_gt_mask=mask.copy(), K_prev=K, K_curr=K,
                prev_c2w=np.eye(4), curr_c2w=np.eye(4), prev_pose_valid=True, curr_pose_valid=True,
                flow_bwd_px=flow.copy(), flow_fwd_px=flow.copy(), flow_bwd_valid=mask.copy(),
                flow_fwd_valid=mask.copy(), fb_max_error_px=0.25)


class WarpTests(unittest.TestCase):
    def test_identity_and_true_bounds(self):
        img = np.arange(12).reshape(3, 4)
        xy = metrics.pixel_grid(3, 4)
        got, valid = metrics.bilinear_sample(img, xy)
        np.testing.assert_allclose(got, img)
        self.assertTrue(valid.all())
        xy[0, 0] = [-0.01, 0]
        xy[1, 0] = [3.01, 1]
        xy[2, 0] = [0, 3.01]
        xy[0, 1] = [np.nan, 0]
        _, valid = metrics.bilinear_sample(img, xy)
        self.assertFalse(valid[0, 0] or valid[1, 0] or valid[2, 0] or valid[0, 1])

    def test_fractional_invalid_support(self):
        image = np.array([[1., np.nan, 3.]])
        xy = np.array([[[0., 0.], [0.5, 0.], [2., 0.]]])
        got, valid = metrics.bilinear_sample(image, xy)
        self.assertEqual(valid.tolist(), [[True, False, True]])
        np.testing.assert_allclose(got[0, [0, 2]], [1, 3])

    def test_known_translation_and_occlusion_direction(self):
        prev = np.tile(np.arange(1., 6.), (3, 1))
        curr = np.concatenate((prev[:, :1], prev[:, :-1]), axis=1)
        kw = inputs(prev, curr)
        kw["flow_bwd_px"][..., 0] = -1
        kw["flow_fwd_px"][..., 0] = 1
        out = metrics.evaluate_pair(**kw)
        self.assertEqual(out["tau5"]["valid_pixels"], 12)
        self.assertEqual(out["tau5"]["value"], 1.)
        # The altered forward vector is on previous-grid x=1: it invalidates
        # current-grid x=2, not x=1.
        kw["flow_fwd_px"][1, 1, 0] = 4
        xy, valid = metrics.backward_correspondence(kw["flow_bwd_px"], kw["flow_fwd_px"],
                                                     kw["flow_bwd_valid"], kw["flow_fwd_valid"], .25)
        self.assertTrue(valid[1, 1])
        self.assertFalse(valid[1, 2])
        self.assertEqual(metrics.evaluate_pair(**kw)["tau5"]["valid_pixels"], 11)

    def test_flow_masks_and_gt_masks_are_not_interchangeable(self):
        kw = inputs(np.arange(1., 13.).reshape(3, 4))
        kw["flow_bwd_valid"][0, 0] = False
        kw["flow_fwd_valid"][0, 1] = False
        kw["prev_gt_mask"][0, 2] = False
        kw["curr_gt_mask"][0, 3] = False
        self.assertEqual(metrics.evaluate_pair(**kw)["tau5"]["valid_pixels"], 8)

    def test_subpixel_correspondence_requires_all_gt_support(self):
        kw = inputs(np.ones((2, 4)))
        kw["flow_bwd_px"][..., 0] = -.5
        kw["flow_fwd_px"][..., 0] = .5
        kw["prev_gt_mask"][0, 1] = False
        out = metrics.evaluate_pair(**kw)
        # x=0 is out of bounds; source x=1 invalidates current x=1 and x=2.
        self.assertEqual(out["tau5"]["valid_pixels"], 4)

    def test_resize_crop_and_intrinsics(self):
        flow = np.empty((2, 3, 2))
        flow[:] = [1, -0.5]
        resized = metrics.resize_flow_pixels(flow, (4, 6))
        np.testing.assert_allclose(resized, np.broadcast_to([2, -1], (4, 6, 2)))
        cropped = metrics.crop_flow_pixels(resized, (1, 2), (0, 1), (2, 3))
        np.testing.assert_allclose(cropped, np.broadcast_to([3, 0], (2, 3, 2)))
        K = np.array([[2., 0, 1], [0, 3, .5], [0, 0, 1]])
        K2 = metrics.transform_intrinsics(K, (2., 2.), (1, 2))
        pixel = K @ np.array([.3, .4, 1])
        resized_crop = np.array([2*(pixel[0]+.5)-.5-2, 2*(pixel[1]+.5)-.5-1, 1])
        np.testing.assert_allclose(K2 @ np.array([.3, .4, 1]), resized_crop)

    def test_explicit_extent_units(self):
        f = np.ones((3, 5, 2)) * .25
        result = metrics.fraction_of_extent_to_pixels(f, (3, 5))
        np.testing.assert_allclose(result, np.broadcast_to([1, .5], (3, 5, 2)))
        with self.assertRaises(ValueError):
            metrics.fraction_of_extent_to_pixels(np.zeros((1, 5, 2)), (1, 5))


class MetricTests(unittest.TestCase):
    def test_tau5_strict_current_depth_denominator(self):
        kw = inputs([[19.02, 19., 18.98]], [[20., 20., 20.]])
        score = metrics.evaluate_pair(**kw)["tau5"]
        self.assertEqual((score["valid_pixels"], score["inliers"]), (3, 1))

    def test_per_frame_ssi_and_temporal_ssi(self):
        gt = np.arange(1., 13.).reshape(3, 4) / 2 + 1
        kw = inputs((gt-.3)/1.5, (gt+.4)/2)
        kw["prev_gt"] = gt.copy()
        kw["curr_gt"] = gt.copy()
        out = metrics.evaluate_pair(**kw)
        self.assertLess(out["tau5"]["value"], 1)
        self.assertEqual(out["previous"]["delta1_ssi"]["value"], 1)
        self.assertEqual(out["tau5_ssi"]["value"], 1)

    def test_ssi_strict_trim_degenerate_is_reported(self):
        kw = inputs(np.ones((2, 3)))
        out = metrics.evaluate_pair(**kw)
        self.assertEqual(out["tau5"]["value"], 1)
        self.assertEqual(out["tau5_ssi"]["status"], "ssi_unavailable")

    def test_camera_translation_compensation(self):
        kw = inputs(np.full((2, 4), 3.), np.full((2, 4), 2.))
        kw["curr_c2w"][2, 3] = 1
        self.assertEqual(metrics.evaluate_pair(**kw)["tau5"]["value"], 1)
        kw["curr_c2w"][2, 3] = 0
        self.assertEqual(metrics.evaluate_pair(**kw)["tau5"]["value"], 0)

    def test_missing_or_placeholder_pose(self):
        kw = inputs(np.ones((2, 3)))
        kw["prev_pose_valid"] = False  # identity is not proof of a measured pose
        self.assertEqual(metrics.evaluate_pair(**kw)["tau5"]["status"], "missing_or_unverified_pose")
        kw["prev_c2w"] = None
        kw["prev_pose_valid"] = True
        self.assertEqual(metrics.evaluate_pair(**kw)["tau5"]["status"], "missing_or_unverified_pose")
        kw["prev_c2w"] = np.zeros((4, 4))
        with self.assertRaises(ValueError):
            metrics.evaluate_pair(**kw)

    def test_invalid_prediction_fails_not_drops_mask(self):
        kw = inputs(np.ones((2, 3)))
        kw["prev_pred"][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "do not silently remove"):
            metrics.evaluate_pair(**kw)

    def test_no_valid_correspondence_has_no_score(self):
        kw = inputs(np.ones((2, 3)))
        kw["flow_bwd_px"][..., 0] = 100
        score = metrics.evaluate_pair(**kw)["tau5"]
        self.assertEqual(score["status"], "no_valid_pixels")
        self.assertIsNone(score["value"])

    def test_nonpositive_transformed_depth_is_a_failure(self):
        kw = inputs(np.ones((2, 3)))
        kw["curr_c2w"][2, 3] = 2
        score = metrics.evaluate_pair(**kw)["tau5"]
        self.assertEqual(score["status"], "invalid_prediction")
        self.assertEqual(score["valid_pixels"], 6)
        self.assertIsNone(score["value"])


class DataRootTests(unittest.TestCase):
    def test_dataroot_matches_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.mkdir(); b.mkdir()
            root, env = checked_environment(a, {})
            self.assertEqual(env["DATAROOT"], str(root))
            self.assertEqual(checked_environment(a, {"DATAROOT": str(a)})[0], root)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                checked_environment(a, {"DATAROOT": str(b)})
            with self.assertRaises(FileNotFoundError):
                checked_environment(Path(tmp) / "missing", {})

    def test_npz_cli_reports_missing_pose_without_creating_outputs(self):
        kw = inputs(np.arange(1., 13.).reshape(3, 4))
        for k in ("prev_pose_valid", "curr_pose_valid", "fb_max_error_px", "prev_c2w", "curr_c2w"):
            del kw[k]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pair.npz"
            np.savez(path, **kw)
            script = Path(__file__).with_name("evaluate_npz.py")
            result = subprocess.run([sys.executable, str(script), "--input", str(path), "--pair-id", "synthetic",
                                     "--flow-source", "synthetic", "--pose-source", "none",
                                     "--fb-max-error-px", ".25"], text=True, capture_output=True, check=True)
            self.assertIn('"missing_or_unverified_pose"', result.stdout)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["pair.npz"])


if __name__ == "__main__":
    unittest.main()
