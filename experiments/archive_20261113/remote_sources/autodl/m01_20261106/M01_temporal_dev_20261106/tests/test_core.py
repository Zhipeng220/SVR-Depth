import csv
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

from svr_depth.datasets.bonn_raw import read_pairs
from svr_depth.evaluation.records import read_records
from svr_depth.analysis.risk import summarize_against_baseline
from svr_depth.analysis.propagation import summarize_by_age
from svr_depth.training.sampling import balanced_order, order_sha256
from svr_depth.streaming.state import StreamState
from svr_depth.streaming.runner import run_fixed_interval
from svr_depth.streaming.strategies import select_closed_loop
from svr_depth.evaluation.intervention import select_interventions

class ContractsTest(unittest.TestCase):
    def test_training_order_matches_existing_protocol(self):
        order = balanced_order(24, 1000, 20261025)
        self.assertEqual(order_sha256(order), 'aa7aff6213fdbf9ddce8703e2eb707e8cb8eab495a575f96698d9d25d9b9d934')

    def test_pair_manifest_retains_unmatched_rgb(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as d:
            root = Path(d)
            for f in ('a.png', 'b.png', 'depth.png'):
                (root / f).write_bytes(b'test')
            path = root / 'pairs.csv'
            with path.open('w', newline='', encoding='utf8') as f:
                w = csv.DictWriter(f, fieldnames=['frame_index_zero_based','rgb_timestamp','rgb_path',
                    'gt_match_status','depth_timestamp','depth_path','abs_delta_ms'])
                w.writeheader()
                w.writerow(dict(frame_index_zero_based=0, rgb_timestamp='1', rgb_path='a.png',
                    gt_match_status='MATCHED', depth_timestamp='1.005', depth_path='depth.png', abs_delta_ms='5'))
                w.writerow(dict(frame_index_zero_based=1, rgb_timestamp='2', rgb_path='b.png', gt_match_status='UNMATCHED'))
            frames = read_pairs(path)
            self.assertEqual([f.status for f in frames], ['MATCHED', 'UNMATCHED'])
            self.assertIsNone(frames[1].depth_path)

    def test_paired_risk_mask_and_missing_frame(self):
        mask = 'a' * 64
        rows = [dict(dataset='d', sequence='s', method=m, seed=seed, index=i, k=16,
                     status=status, absrel=metric, mask_sha256=mask)
                for m, seed, i, status, metric in [('A','native',0,'ok',0.1), ('P','1',0,'ok',0.12),
                                                     ('A','native',1,'unmatched',None), ('P','1',1,'ok',0.2)]]
        result = summarize_against_baseline(rows)[0]
        self.assertEqual((result['paired_valid'], result['missing_or_invalid']), (1, 1))
        self.assertEqual(result['severe_fraction'], 1)
        rows[1]['mask_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'Mismatched GT masks'):
            summarize_against_baseline(rows)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            summarize_against_baseline(rows[:-1])

    def test_age_requires_continuity_and_reset(self):
        rows = [dict(dataset='d', sequence='s', method='P', seed='1', k=16,
                     index=i, age=str(age), is_keyframe=str(age == 0), status='ok', absrel=0.1)
                for i, age in enumerate((0, 1, 2, 0, 1))]
        result = summarize_by_age(rows)
        self.assertEqual(next(x['total_frames'] for x in result if x['age'] == 0), 2)
        rows[2]['age'] = '7'
        with self.assertRaisesRegex(ValueError, 'invalid age transition'):
            summarize_by_age(rows)

    def test_state_clone_does_not_share_mutable_fields(self):
        state = StreamState(1, 2, [3], 4, {'values': [5]})
        cloned = state.clone()
        cloned.features.append(9)
        cloned.extra['values'].append(6)
        self.assertEqual(state.features, [3])
        self.assertEqual(state.extra, {'values': [5]})

    def test_closed_loop_and_intervention_state_selections(self):
        a = StreamState(1, 2, [3], 4, {'cache': [5]})
        p = StreamState(10, 20, [30], 4, {'cache': [50]})
        visible, committed = select_closed_loop('O_same_head', a, p)
        self.assertEqual((visible.radius, committed.radius, committed.features), (10, 1, [3]))
        visible, committed = select_closed_loop('F_only', a, p)
        self.assertEqual((visible.radius, committed.radius, committed.features, committed.extra), (1, 1, [30], {'cache': [5]}))
        branches = select_interventions(a, p)
        self.assertEqual(branches['F'][1], branches['T'][1])
        branches['T'][1].features.append(100)
        self.assertEqual(branches['F'][1].features, [30])

    def test_fixed_interval_processes_all_frames(self):
        output = list(run_fixed_interval(range(7), interval=3,
            keyframe=lambda frame: (frame, frame),
            propagate=lambda prev, frame, state: (frame, state + 1)))
        self.assertEqual([x.age for x in output], [0, 1, 2, 0, 1, 2, 0])
        self.assertEqual([x.index for x in output], list(range(7)))

    def test_legacy_evaluation_help_is_reachable(self):
        root = Path(__file__).resolve().parents[1]
        for protocol in ('closed_loop', 'intervention'):
            result = subprocess.run([sys.executable, str(root / 'scripts/eval.py'), protocol, '--', '--help'],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)

if __name__ == '__main__':
    unittest.main()
