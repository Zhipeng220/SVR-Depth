import unittest
from stats_p3b import summarize,ALPHAS
class Tests(unittest.TestCase):
 def rows(self):
  return [{'dataset':d,'sequence':s,'age':a,'alpha':q,'status':'PASS','H1_AbsRel':1-q,'H4_AbsRel':1-q,'H1_delta_vs_alpha0':-q,'H4_delta_vs_alpha0':-q,'feature_delta_l2':q,'feature_relative_l2':q/10,'pilot_p2a_exposed':False} for d in ('Bonn','TUM','ScanNet') for s in ('s1','s2') for a in (1,4) for q in ALPHAS]
 def test_sequence_unit_and_endpoint(self):
  rows=self.rows();expected={(r['dataset'],r['sequence'],r['age'],r['alpha']) for r in rows};seq,domains=summarize(rows,expected,reps=50)
  self.assertEqual(len(seq),30);self.assertEqual(len(domains),15);self.assertTrue(all(r['n_sequences']==2 for r in domains));self.assertTrue(all(r['oracle_best_alpha_H4']==1. for r in seq));self.assertTrue(all(not r['H4_nonmonotonic'] for r in seq))
 def test_missing_or_duplicate_blocks(self):
  rows=self.rows();expected={(r['dataset'],r['sequence'],r['age'],r['alpha']) for r in rows}
  for data in (rows[:-1],rows+[rows[0]]):
   with self.assertRaises(ValueError):summarize(data,expected,reps=10)
if __name__=='__main__':unittest.main()
