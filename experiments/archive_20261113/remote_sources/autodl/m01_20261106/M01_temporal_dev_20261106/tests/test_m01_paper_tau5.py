import unittest
import importlib.util
from unittest import mock
from types import SimpleNamespace
import numpy as np
from svr_depth.evaluation.m01_v31.paper_tau5 import compute_paper_tau5
from svr_depth.evaluation.m01_v31.paper_protocol import audit_status, REQUIRED_CHECKS


class PaperTauCandidateTests(unittest.TestCase):
    def setUp(self):
        self.shape = (4, 5)
        self.depth = np.full(self.shape, 2.0)
        self.flow = np.zeros((*self.shape, 2))
        self.mask = np.ones(self.shape, dtype=bool)
        self.K = np.array([[4., 0, 2.], [0, 4., 2.], [0, 0, 1.]])
        self.pose = np.eye(4)
        self.kw = dict(flow_bwd=self.flow, flow_valid=self.mask, K_prev=self.K,
                       pose_prev=self.pose, pose_curr=self.pose, mask_prev=self.mask,
                       mask_curr=self.mask, pose_prev_valid=True, pose_curr_valid=True,
                       camera_model='pinhole_verified', pixel_center='integer',
                       flow_direction='current_to_previous')

    def test_identity_and_mask(self):
        self.kw['flow_valid'] = self.mask.copy()
        self.kw['flow_valid'][0, 0] = False
        r = compute_paper_tau5(self.depth, self.depth, **self.kw)
        self.assertEqual((r['status'], r['tau5'], r['valid_pixels']), ('ok', 1.0, 19))
        self.assertEqual(r['mask_counts']['tau5_inliers'], 19)
        self.assertFalse(r['author_equivalent'])

    def test_z_translation_is_compensated(self):
        curr_pose = self.pose.copy(); curr_pose[2, 3] = 0.5
        self.kw['pose_curr'] = curr_pose
        r = compute_paper_tau5(self.depth, np.full(self.shape, 1.5), **self.kw)
        self.assertEqual(r['tau5'], 1.0)
        self.kw['pose_curr'] = self.pose
        self.assertEqual(compute_paper_tau5(self.depth, np.full(self.shape, 1.5), **self.kw)['tau5'], 0.0)

    def test_half_pixel_convention_changes_rays(self):
        # Rotation about Y makes Z depend on X; the half-centre K must shift too.
        angle = 0.15
        pose = self.pose.copy()
        pose[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]]
        x = np.arange(self.shape[1])[None, :]
        expected = 2 * (np.cos(angle) + np.sin(angle) * (x - 2) / 4)
        expected = np.broadcast_to(expected, self.shape)
        self.kw['pose_curr'] = pose
        self.assertEqual(compute_paper_tau5(self.depth, expected, **self.kw)['tau5'], 1.0)
        self.kw['pixel_center'] = 'half'
        K_half = self.K.copy(); K_half[0, 2] += .5; K_half[1, 2] += .5
        self.kw['K_prev'] = K_half
        self.assertEqual(compute_paper_tau5(self.depth, expected, **self.kw)['tau5'], 1.0)

    def test_unavailable_and_invalid_prediction(self):
        for field, value, status in [('flow_bwd', None, 'missing_flow_or_validity_mask'),
                                     ('pose_prev_valid', False, 'missing_or_unverified_pose'),
                                     ('camera_model', 'OPENCV', 'unsupported_or_unverified_camera'),
                                     ('flow_direction', 'previous_to_current', 'unverified_flow_direction')]:
            with self.subTest(field=field):
                kw = dict(self.kw); kw[field] = value
                self.assertEqual(compute_paper_tau5(self.depth, self.depth, **kw)['status'], status)
        bad = self.depth.copy(); bad[0, 0] = np.nan
        r = compute_paper_tau5(bad, self.depth, **self.kw)
        self.assertEqual(r['status'], 'invalid_prediction')
        self.assertEqual(r['valid_pixels'], 20)

    def test_acceptance_does_not_unlock_itself(self):
        checks = {name: {'passed': True, 'evidence': 'example'} for name in REQUIRED_CHECKS}
        result = audit_status(checks)
        self.assertFalse(result['formal_aggregate_ready'])
        self.assertEqual(result['missing_checks'], [])

    def test_last32_resets_stream_index_preserves_source_index(self):
        from svr_depth.datasets import benchmark
        rows = [SimpleNamespace(index=i, rgb_path=f'{i}.png', depth_path=None, rgb_timestamp=float(i)) for i in range(40)]
        spec = {'dataset': 'Bonn', 'sequence': 'seq', 'scene': 'scene',
                'pairs_csv': 'unused', 'max_delta_ms': 20, 'frame_selection': 'release_last32'}
        with mock.patch.object(benchmark, 'read_pairs', return_value=rows), \
             mock.patch.object(benchmark, '_image', return_value=np.zeros((2, 2, 3), dtype=np.uint8)):
            frames = list(benchmark.iter_bonn(spec, load_gt=False))
        self.assertEqual([x.index for x in frames], list(range(32)))
        self.assertEqual([x.source_frame_index for x in frames], list(range(8, 40)))

    def test_candidate_integrates_offline_ssi(self):
        from svr_depth.evaluation.m01_temporal import score_temporal_pair
        def frame(index):
            return SimpleNamespace(index=index, source_frame_id=str(index), source_frame_index=index,
                gt=self.depth, gt_error='', gt_valid_mask=self.mask, flow_bwd=self.flow,
                flow_bwd_mask=self.mask, flow_valid=True, flow_bwd_source='verified-flow',
                camera_valid=True, intrinsics=self.K, pose=self.pose, pose_valid=True)
        previous = {'frame': frame(0), 'prediction': self.depth, 'ssi_aligned': self.depth}
        current = {'frame': frame(1), 'prediction': self.depth, 'ssi_aligned': self.depth}
        protocol = {'protocol': 'paper_tau5_candidate_v1', 'flow_direction': 'current_to_previous',
                    'pixel_center': 'integer'}
        result = score_temporal_pair(previous, current, protocol)
        self.assertEqual((result['tau5_candidate'], result['tau5_ssi_candidate']), (1.0, 1.0))
        self.assertEqual(result['protocol_status'], 'candidate')

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch required for OPENCV camera')
    def test_candidate_opencv_camera_matches_pinhole_when_distortion_zero(self):
        params=[4.,4.,2.,2.]+[0.]*12
        kw=dict(self.kw)
        kw.update(K_prev={'model':'OPENCV','params':params}, camera_model='opencv_verified')
        opencv=compute_paper_tau5(self.depth,self.depth,**kw)
        pinhole=compute_paper_tau5(self.depth,self.depth,**self.kw)
        self.assertEqual(opencv['status'],'ok')
        self.assertEqual(opencv['tau5'],pinhole['tau5'])


if __name__ == '__main__': unittest.main()
