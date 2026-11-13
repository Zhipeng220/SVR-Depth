import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import ridge
import association
def rows():
 out=[]
 for g in range(4):
  for a in range(2):
   r={'anchor_id':f'g{g}a{a}','candidate':'V5-native','physical_group_id':f'g{g}','physical_group_status':'CERTIFIED','matched_evidence':'PASS','dataset':'synthetic','U_H4':float(g+a*.1)}
   r.update({k:float((g+1)*(i+1)+a*.1) for i,k in enumerate(ridge.NORM+ridge.C3)});out.append(r)
 return out
class Tests(unittest.TestCase):
 def test_fold_group_isolation(self):
  data=rows();o=ridge.evaluate(data)
  for s in o['splits']:
   tr={data[i]['physical_group_id'] for i in s['train_indices']};te={data[i]['physical_group_id'] for i in s['test_indices']};self.assertFalse(tr&te);self.assertEqual(te,{s['test_group']})
 def test_scaling_train_only(self):
  _,m=ridge.ridge_predict([[0],[2]],[1,2],[[1000000]]);self.assertEqual(m['train_mean'][0],1);self.assertEqual(m['train_scale'][0],1)
 def test_block_unknown_and_duplicate(self):
  r=rows();r[0]['physical_group_status']='UNKNOWN'
  with self.assertRaises(ValueError):ridge.evaluate(r)
  r=rows();r[1]['anchor_id']=r[0]['anchor_id']
  with self.assertRaises(ValueError):ridge.evaluate(r)
 def test_aggregate_groups_and_bootstrap(self):
  g=ridge.evaluate(rows())['groups'];a=ridge.aggregate(g);self.assertEqual(len(g),4);self.assertIn('MAE',a['M2']);self.assertEqual(ridge.bootstrap_delta(g,resamples=50),ridge.bootstrap_delta(g,resamples=50))
 def test_rank_ties_and_zero(self):
  self.assertTrue(np.allclose(ridge.rank([2,1,1]),[2,.5,.5]));self.assertIsNone(ridge.spearman([1,1,1],[1,2,3]))
 def test_holm_and_insufficient_domain(self):
  self.assertEqual(association.holm([.01,.03,.04,.2]),[.04,.09,.09,.2]);self.assertEqual(association.analyze(rows(),permutations=10)['Bonn']['status'],'BLOCKED_INSUFFICIENT_CERTIFIED_GROUPS')
 def test_association_has_four_levels_per_domain(self):
  data=[]
  for ds in ['Bonn','TUM','ScanNet']:
   for g in range(8):
    r={'anchor_id':f'{ds}{g}','candidate':'V5-native','physical_group_id':f'{ds}{g}','physical_group_status':'CERTIFIED','matched_evidence':'PASS','dataset':ds,'U_H4':float(g)}
    r.update({k:float((g+1)**(i%3+1)) for i,k in enumerate(ridge.NORM+ridge.C3)});data.append(r)
  x=association.analyze(data,permutations=20)
  for ds in x:self.assertEqual(len(x[ds]['c3']),4);self.assertEqual(len(x[ds]['norm']),4)
if __name__=='__main__':unittest.main()

