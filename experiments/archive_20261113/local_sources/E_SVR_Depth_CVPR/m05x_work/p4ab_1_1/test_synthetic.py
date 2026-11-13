import sys, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'docs/m05x/p4ab'))
import P4B_MODEL_IMPLEMENTATION as m

def data():
 rows=[]
 for g in range(6):
  for a in range(2):
   r={'synthetic_marker':'SYNTHETIC_TEST_ONLY','anchor_id':f'g{g}a{a}','candidate':'V5-native','physical_group_id':f'g{g}','physical_group_status':'CERTIFIED','matched_evidence':'PASS','dataset':'synthetic','U_H4':g+a*.1}
   r.update({k:float((g+1)*(i+1)+a*.1) for i,k in enumerate(m.NORM+m.C3)}); rows.append(r)
 return rows
class TestSynthetic(unittest.TestCase):
 def test_group_isolation(self):
  rows=data(); result=m.evaluate(rows,synthetic_only=True)
  for fold in result['splits']:
   a={rows[i]['physical_group_id'] for i in fold['train_indices']};b={rows[i]['physical_group_id'] for i in fold['test_indices']};self.assertFalse(a&b);self.assertEqual(b,{fold['test_group']})
 def test_scaling_fitted_on_train_only(self):
  _,v=m.ridge_predict([[0],[2]],[1,2],[[1000000]],synthetic_only=True);self.assertEqual(v['train_mean'][0],1);self.assertEqual(v['train_scale'][0],1)
 def test_non_synthetic_refused(self):
  with self.assertRaises(PermissionError):m.evaluate(data())
  rows=data();rows[0].pop('synthetic_marker')
  with self.assertRaises(PermissionError):m.evaluate(rows,synthetic_only=True)
  with self.assertRaises(PermissionError):m.ridge_predict([[0],[1]],[0,1],[[2]])
 def test_unknown_and_duplicate_refused(self):
  rows=data();rows[0]['physical_group_status']='UNKNOWN'
  with self.assertRaises(ValueError):m.evaluate(rows,synthetic_only=True)
  rows=data();rows[1]['anchor_id']=rows[0]['anchor_id']
  with self.assertRaises(ValueError):m.evaluate(rows,synthetic_only=True)
 def test_metrics_and_bootstrap(self):
  groups=m.evaluate(data(),synthetic_only=True)['groups'];stats=m.aggregate(groups);self.assertEqual(len(groups),6);self.assertIn('MAE',stats['M2']);self.assertIn('RMSE',stats['M1']);self.assertEqual(m.bootstrap_delta(groups,resamples=100),m.bootstrap_delta(groups,resamples=100))
 def test_ties_and_constant(self):
  self.assertTrue(np.allclose(m.rank([2,1,1]),[2,.5,.5]));self.assertIsNone(m.spearman([1,1,1],[1,2,3]));_,v=m.ridge_predict([[1],[1]],[1,2],[[1]],synthetic_only=True);self.assertEqual(v['train_scale'][0],1)
 def test_identifiers_not_features(self):
  self.assertEqual(m.NORM,['global_l2','l0_l2','l1_l2','l2_l2','l3_l2']);self.assertFalse({'dataset','sequence_id','physical_group_id'} & set(m.NORM+m.C3))
 def test_nonfinite_rejected(self):
  rows=data();rows[0]['c3_l0']=float('nan')
  with self.assertRaises(ValueError):m.evaluate(rows,synthetic_only=True)
 def test_sequence_cannot_cross_groups(self):
  rows=data();rows[0]['sequence_id']='same';rows[2]['sequence_id']='same'
  with self.assertRaises(ValueError):m.evaluate(rows,synthetic_only=True)
if __name__=='__main__':unittest.main(verbosity=2)
