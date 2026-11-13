import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

try:
    from . import hdf5_preflight as adapter
except ImportError:
    import hdf5_preflight as adapter


class ContractTests(unittest.TestCase):
    def test_upstream_evaluator_uses_last_32(self):
        self.assertEqual(adapter.upstream_last32_indices(110), list(range(78, 110)))
        self.assertEqual(adapter.upstream_last32_indices(5), list(range(5)))
        self.assertEqual(adapter.upstream_last32_indices(32), list(range(32)))
        for invalid in (0, -1, True, 3.5):
            with self.assertRaises(ValueError):
                adapter.upstream_last32_indices(invalid)

    def test_packed_12bit_channels(self):
        x = np.array([[0, 2047, 4095, 257]], dtype=np.uint16)
        y = np.array([[4095, 2047, 0, 2048]], dtype=np.uint16)
        rgb = np.stack((x & 255, ((x >> 8) & 15) | ((y & 15) << 4), y >> 4), axis=-1).astype(np.uint8)
        result = adapter.decode_packed12_normalized(rgb)
        expected = (np.stack((x, y), axis=-1).astype(np.float32) + .5) / 4095 * 2 - 1
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[0, 1], [0, 0])
        with self.assertRaises(ValueError):
            adapter.decode_packed12_normalized(rgb.astype(np.int16))
        with self.assertRaises(ValueError):
            adapter.decode_packed12_normalized(np.zeros((4, 2), dtype=np.uint8))

    def test_pose_is_checked_without_repair(self):
        pose = np.eye(4)
        self.assertTrue(adapter.pose_diagnostics(pose)['strict_valid'])
        pose[0, 0] = 1.0002
        original = pose.copy()
        result = adapter.pose_diagnostics(pose)
        self.assertEqual(result['status'], 'finite_fails_strict_check')
        self.assertFalse(result['strict_valid'])
        np.testing.assert_array_equal(pose, original)
        pose[0, 0] = np.inf
        self.assertEqual(adapter.pose_diagnostics(pose)['status'], 'nonfinite')
        self.assertEqual(adapter.pose_diagnostics(None)['status'], 'missing')
        self.assertEqual(adapter.pose_diagnostics([1, 2])['status'], 'invalid_shape')

    def test_camera_distortion_is_not_silently_ignored(self):
        params = [500., 500., 320., 240.] + [0.] * 12
        params[4] = .1
        sequence = {'camera_model': ['OPENCV'], 'camera_params': [params]}
        self.assertEqual(adapter.camera_diagnostics(sequence, 0)['status'], 'distortion_requires_handling')
        params[4] = 0.
        self.assertEqual(adapter.camera_diagnostics(sequence, 0)['status'], 'zero_distortion_metadata_only')
        self.assertFalse(adapter.camera_diagnostics(sequence, 0)['image_rectification_verified'])
        self.assertEqual(adapter.camera_diagnostics({'K': [np.eye(3).tolist()]}, 0)['status'], 'pinhole_candidate')

    def test_split_and_sequence_ambiguities_are_rejected(self):
        self.assertEqual(adapter.parse_split('  /train/a 3  \n'), [('/train/a', 3)])
        with self.assertRaises(ValueError):
            adapter.parse_split('/train/a 3\n/train/a 4\n')
        with self.assertRaises(ValueError):
            adapter.resolve_sequence({'/a/x': {}, '/b/x': {}}, 'x')


@unittest.skipUnless(importlib.util.find_spec('h5py'), 'h5py not installed')
class SyntheticHDF5Tests(unittest.TestCase):
    def test_development_split_preflight_never_scores_or_assumes_flow_index(self):
        import h5py
        pose = np.eye(4).tolist()
        sequence = {
            'image': ['rgb0', 'rgb1', 'rgb2'], 'depth': ['d0', 'd1', 'd2'],
            'K': [np.eye(3).tolist()] * 3, 'cam2w': [pose] * 3,
            'flow_fwd': ['f0', 'f1', 'f2', 'f3'], 'flow_fwd_mask': ['m0', 'm1', 'm2', 'm3'],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.hdf5'
            with h5py.File(path, 'w') as f:
                f.create_dataset('sequences.json', data=json.dumps({'/synthetic/train_a': sequence}).encode())
                f.create_dataset('train.txt', data=b'/synthetic/train_a 3\n')
            sequences, split = adapter.load_metadata(path, 'sequences.json', 'train.txt')
            self.assertEqual(split, [('/synthetic/train_a', 3)])
            result = adapter.inspect_pair(sequences['/synthetic/train_a'], 1)
            self.assertFalse(result['ready_for_temporal_scoring'])
            self.assertFalse(result['backward_flow_metadata_present'])
            self.assertEqual(result['forward_flow_candidates_unverified']['record_count'], 4)
            self.assertEqual([x['record_index'] for x in result['forward_flow_candidates_unverified']['candidate_records']], [1, 2])
            self.assertTrue(result['previous_pose']['strict_valid'])
            self.assertEqual(result['upstream_eval_last32_indices'], [0, 1, 2])
            for invalid in (-1, 2, True):
                with self.assertRaises(ValueError):
                    adapter.inspect_pair(sequence, invalid)
            script = Path(adapter.__file__)
            listed = subprocess.run([sys.executable, str(script), '--hdf5', str(path), '--list-sequences'],
                                    capture_output=True, text=True, check=True)
            parsed = json.loads(listed.stdout)
            self.assertEqual(parsed['sequences'][0]['split_name'], '/synthetic/train_a')
            pair = subprocess.run([sys.executable, str(script), '--hdf5', str(path), '--sequence',
                                   '/synthetic/train_a', '--prev-index', '0'], capture_output=True, text=True, check=True)
            self.assertFalse(json.loads(pair.stdout)['ready_for_temporal_scoring'])
            unknown = subprocess.run([sys.executable, str(script), '--hdf5', str(path), '--sequence',
                                      '/not/in/train'], capture_output=True, text=True)
            self.assertNotEqual(unknown.returncode, 0)


if __name__ == '__main__':
    unittest.main()
