import tempfile
import unittest
from pathlib import Path
from svr_depth.evaluation.m01_paper_report import build_report, verify_evidence, REQUIRED_CHECKS, sha256_file


def fixture():
    frames, pairs = [], []
    for dataset in ('Bonn','TUM','ScanNet','Sintel'):
        for method in ('A_native','A','P_multi'):
            seed = '7' if method == 'P_multi' else 'native'
            for index in range(2):
                frames.append({'dataset':dataset, 'scene':'scene', 'sequence':'seq', 'method':method, 'seed':seed,
                    'index':str(index), 'source_frame_id':str(index), 'source_frame_index':str(index),
                    'rgb_sha256':'rgb', 'gt_sha256':'gt', 'mask_sha256':'mask', 'status':'ok',
                    'upstream_depth_status':'ok', 'upstream_depth_d1':'0.9', 'upstream_depth_d1_ssi':'0.95'})
            pair = {'dataset':dataset, 'scene':'scene', 'sequence':'seq', 'method':method, 'seed':seed,
                    'prev_index':'0', 'curr_index':'1', 'status':'ok', 'ssi_status':'ok',
                    'temporal_protocol':'paper_tau5_candidate_v1', 'tau5_candidate':'0.8',
                    'tau5_ssi_candidate':'0.9', 'valid_pixels':'10', 'inliers':'8',
                    'ssi_valid_pixels':'10', 'ssi_inliers':'9'}
            for field in ('flow_bwd_sha256', 'flow_bwd_mask_sha256', 'prev_gt_mask_sha256', 'curr_gt_mask_sha256',
                          'prev_pose_sha256', 'curr_pose_sha256', 'prev_intrinsics_sha256', 'curr_intrinsics_sha256'):
                pair[field] = 'same'
            pairs.append(pair)
    return frames, pairs


class ReportTests(unittest.TestCase):
    def test_four_dataset_aggregate_with_explicit_rule(self):
        frames, pairs = fixture()
        for rule in ('equal_pair','equal_sequence','pixel_weighted'):
            result = build_report(frames, pairs, temporal_rule=rule)
            self.assertEqual(result['data_blocking_reasons'], [])
            self.assertAlmostEqual(result['tables'][0]['aggregate']['tau5'], .8)
            self.assertAlmostEqual(result['tables'][0]['aggregate']['delta1_ssi'], .95)

    def test_missing_or_mismatched_inputs_block_formal(self):
        frames, pairs = fixture()
        pairs[0]['flow_bwd_sha256'] = 'different'
        frames[0]['mask_sha256'] = 'different'
        result = build_report(frames, pairs, temporal_rule='equal_pair')
        self.assertIn('temporal_input_mismatch:Bonn', result['data_blocking_reasons'])
        self.assertIn('frame_or_mask_mismatch:Bonn', result['data_blocking_reasons'])
        del pairs[0]
        result = build_report(frames, pairs, temporal_rule='equal_pair')
        self.assertTrue(any(x.startswith('incomplete_pair_coverage:') for x in result['data_blocking_reasons']))

    def test_evidence_binds_results_and_checks_artifact_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/'review.txt'; file.write_text('reviewed')
            hashes = {'frames':'a', 'pairs':'b', 'identity':'c'}
            checks = {name:{'passed':True, 'reviewer':'researcher', 'artifact':'review.txt',
                            'sha256':sha256_file(file)} for name in REQUIRED_CHECKS}
            manifest = {'input_sha256':hashes, 'checks':checks}
            self.assertEqual(verify_evidence(manifest, temp, hashes), [])
            self.assertEqual(verify_evidence(manifest, temp, {'frames':'other'}), ['evidence_input_hash_mismatch'])
            file.write_text('changed')
            self.assertTrue(any('hash_mismatch' in x for x in verify_evidence(manifest, temp, hashes)))


if __name__ == '__main__': unittest.main()
