import math
import unittest
from r2b_math import discounted_gain

class HorizonGainTests(unittest.TestCase):
    def test_positive_gain_means_commit_is_better(self):
        gain, n = discounted_gain({1: 0.20}, {1: 0.10}, 1, 0.9)
        self.assertAlmostEqual(gain, 0.09)
        self.assertEqual(n, 1)

    def test_negative_gain_means_discard_is_better(self):
        gain, n = discounted_gain({1: 0.10}, {1: 0.20}, 1, 0.9)
        self.assertAlmostEqual(gain, -0.09)
        self.assertEqual(n, 1)

    def test_missing_frame_keeps_original_discount_exponent(self):
        gain, n = discounted_gain({1: math.nan, 2: 0.20}, {1: math.nan, 2: 0.10}, 2, 0.9)
        self.assertAlmostEqual(gain, (0.9 ** 2) * 0.10)
        self.assertEqual(n, 1)

    def test_metric_mask_mismatch_fails_closed(self):
        with self.assertRaises(RuntimeError):
            discounted_gain({1: 0.2}, {1: math.nan}, 1, 0.9)

    def test_no_valid_future_pair_fails_closed(self):
        with self.assertRaises(RuntimeError):
            discounted_gain({1: math.nan}, {1: math.nan}, 1, 0.9)

if __name__ == "__main__":
    unittest.main()
