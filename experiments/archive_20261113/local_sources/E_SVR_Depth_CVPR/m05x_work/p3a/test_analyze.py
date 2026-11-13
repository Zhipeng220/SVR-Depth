import unittest
import numpy as np
from analyze import rank,rho,bootstrap,centered,relative_from_sums

class Tests(unittest.TestCase):
 def test_rank_ties_and_correlations(self):
  np.testing.assert_allclose(rank([2,1,2,4]),[2.5,1,2.5,4])
  self.assertAlmostEqual(rho([1,2,3],[3,2,1]),-1)
  self.assertIsNone(rho([1,1,1],[1,2,3]))
 def test_bootstrap_reproducible_and_degenerate(self):
  a=bootstrap([1,2,3,4],[4,3,2,1],reps=100);self.assertEqual(a,bootstrap([1,2,3,4],[4,3,2,1],reps=100))
  x=bootstrap([1,1,1],[1,2,3],reps=100);self.assertIsNone(x['low']);self.assertEqual(x['undefined_replicates'],100)
 def test_concatenated_norm(self):
  self.assertAlmostEqual(relative_from_sums(25,100),.5)
  self.assertEqual(relative_from_sums(0,0),0)
 def test_center_within_domain(self):
  groups=[[{'R':10.,'H4':100.},{'R':11.,'H4':101.}],[{'R':50.,'H4':-10.},{'R':51.,'H4':-9.}]]
  self.assertAlmostEqual(centered(groups),1.)
if __name__=='__main__':unittest.main()
