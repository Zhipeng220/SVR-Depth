import unittest
from stats_p2b import summarize

class StatsTests(unittest.TestCase):
 def rows(self):
  return [{'dataset':'D','sequence':seq,'age':age,'status':'PASS','T_minus_O_H1':v,'T_minus_O_H4':v,'pilot_p2a_exposed':False} for seq,v in [('s1',1),('s2',3)] for age in (1,4)]
 def test_cluster_and_seed(self):
  rows=self.rows();expected={(x['dataset'],x['sequence'],x['age']) for x in rows};seq,s,c=summarize(rows,expected,reps=500)
  self.assertEqual(len(seq),2);self.assertEqual(s['D']['H4']['sequence_mean'],2);self.assertEqual(s['D']['n_sequences'],2);self.assertEqual(c,summarize(rows,expected,reps=500)[2])
 def test_missing_duplicate_or_invalid(self):
  rows=self.rows();expected={(x['dataset'],x['sequence'],x['age']) for x in rows}
  for bad in (rows[:-1],rows+[rows[0]],[{**x,'status':'INVALID'} if i==0 else x for i,x in enumerate(rows)]):
   with self.assertRaises(ValueError):summarize(bad,expected,reps=10)
if __name__=='__main__':unittest.main()
