import unittest
import ast
from pathlib import Path
import numpy as np
from r3_3_geometry import decompose_depth_difference, discounted_gain, diagnostic_anchors, association_rows

class GeometryTests(unittest.TestCase):

    def test_paired_future_gain_return_contract_matches_caller(self):
        source_path = Path(__file__).with_name("r3_3_geometry_gain_diagnostic.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        method = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "paired_future_gain":
                method = node
                break
        self.assertIsNotNone(method)
        tuple_return_lengths = []
        for node in ast.walk(method):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
                tuple_return_lengths.append(len(node.value.elts))
        self.assertIn(3, tuple_return_lengths)

        callers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                fn = node.value.func
                if isinstance(fn, ast.Attribute) and fn.attr == "paired_future_gain":
                    for target in node.targets:
                        if isinstance(target, ast.Tuple):
                            callers.append(len(target.elts))
        self.assertIn(3, callers, "Caller must unpack result, traces, and restored state")

    def test_identical_maps_have_zero_raw_and_local_residual(self):
        p = np.linspace(1, 10, 1200).reshape(30, 40)
        f = decompose_depth_difference(p, p)
        self.assertAlmostEqual(f['raw_rel_mean'], 0.0, places=10)
        self.assertAlmostEqual(f['scale_shift_scale'], 1.0, places=8)
        self.assertAlmostEqual(f['scale_shift_shift'], 0.0, places=8)
        self.assertLess(f['fit_residual_norm_p90'], 1e-7)

    def test_global_scale_shift_is_separated_from_local_residual(self):
        p = np.linspace(1, 10, 1200).reshape(30, 40)
        c = 1.7 * p + 0.4
        f = decompose_depth_difference(c, p)
        self.assertAlmostEqual(f['scale_shift_scale'], 1.7, places=5)
        self.assertAlmostEqual(f['scale_shift_shift'], 0.4, places=4)
        self.assertLess(f['fit_residual_norm_p90'], 1e-6)
        self.assertGreater(f['raw_rel_mean'], 0.05)

    def test_local_patch_difference_survives_global_fit(self):
        p = np.ones((40, 40), dtype=float) * 3
        c = p.copy()
        c[10:30, 10:30] += 1.2
        f = decompose_depth_difference(c, p)
        self.assertGreater(f['fit_residual_norm_p90'], 0.01)
        self.assertGreater(f['fit_residual_fraction_gt_0p05'], 0.05)

    def test_invalid_pixels_are_excluded(self):
        p = np.ones((10, 10), dtype=float) * 2
        c = p.copy(); c[0, 0] = np.nan; c[0, 1] = -1; p[0, 2] = 0
        f = decompose_depth_difference(c, p)
        self.assertEqual(f['geometry_valid_pixels'], 97)

    def test_no_valid_pixels_returns_nan_features(self):
        p = np.zeros((10, 10)); c = np.zeros((10, 10))
        f = decompose_depth_difference(c, p)
        self.assertEqual(f['geometry_valid_pixels'], 0)
        self.assertTrue(np.isnan(f['raw_rel_mean']))

    def test_discount_exponent_keeps_original_frame_index(self):
        b0 = [{'absrel': 0.2}, {'absrel': float('nan')}, {'absrel': 0.4}]
        b1 = [{'absrel': 0.1}, {'absrel': 0.0}, {'absrel': 0.1}]
        total, mean, n = discounted_gain(b0, b1, gamma=0.5)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(total, 0.5 * 0.1 + (0.5 ** 3) * 0.3)
        self.assertAlmostEqual(mean, total / 2)

    def test_anchor_schedule_matches_horizon_window(self):
        self.assertEqual(diagnostic_anchors(32, 1, 4, 16), [1, 5, 9, 13])
        self.assertEqual(diagnostic_anchors(86, 1, 4, 16), list(range(1, 70, 4)))

    def test_association_rows_are_descriptive_and_keep_sequence_scopes(self):
        rows = []
        for seq in ['a', 'b']:
            for i in range(6):
                rows.append({'sequence': seq, 'counterfactual_status': 'ok', 'x': i, 'gain_absrel_discounted_per_valid_frame': i - 2})
        out = association_rows(rows, ['x'])
        self.assertEqual(len(out), 3)
        all_row = [r for r in out if r['scope'] == 'all'][0]
        self.assertEqual(all_row['n'], 12)
        self.assertAlmostEqual(all_row['spearman_r_descriptive'], 1.0)
        self.assertIn('descriptive only', all_row['interpretation_limit'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
