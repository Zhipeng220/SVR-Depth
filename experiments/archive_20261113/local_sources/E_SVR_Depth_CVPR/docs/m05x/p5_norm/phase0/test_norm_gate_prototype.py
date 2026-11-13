import math
import unittest
from norm_gate_prototype import norm_guided_commit, norm_guided_commit_batch

class NormGateTests(unittest.TestCase):
    def test_positive(self): self.assertTrue(norm_guided_commit(0.1))
    def test_negative(self): self.assertFalse(norm_guided_commit(-0.1))
    def test_zero(self): self.assertFalse(norm_guided_commit(0.0))
    def test_negative_zero(self): self.assertFalse(norm_guided_commit(-0.0))
    def test_extreme_positive(self): self.assertTrue(norm_guided_commit(1e308))
    def test_extreme_negative(self): self.assertFalse(norm_guided_commit(-1e308))
    def test_nan(self):
        with self.assertRaises(ValueError): norm_guided_commit(math.nan)
    def test_infinity(self):
        for x in (math.inf, -math.inf):
            with self.subTest(x=x), self.assertRaises(ValueError): norm_guided_commit(x)
    def test_invalid_types(self):
        for x in ('1', None, True, 1j, [1]):
            with self.subTest(x=x), self.assertRaises(TypeError): norm_guided_commit(x)
    def test_batch(self): self.assertEqual(norm_guided_commit_batch([1, 0, -1, 1e308]), [True, False, False, True])
    def test_invalid_batch_element(self):
        with self.assertRaises(ValueError): norm_guided_commit_batch([1, math.nan])
    def test_invalid_batch_container(self):
        with self.assertRaises(TypeError): norm_guided_commit_batch('123')

if __name__ == '__main__': unittest.main(verbosity=2)
