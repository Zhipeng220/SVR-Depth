import unittest
import numpy as np
from quality import metric,temporal_metric
from report import paired,boot

class CV:
    IMREAD_UNCHANGED=1
    def imread(self,path,mode):return np.array([[5000,10000],[0,1500]],dtype=np.uint16)

class MetricsTest(unittest.TestCase):
    def test_fixed_depth_mask_and_invalid_prediction(self):
        rec={'gt_match_status':'MATCHED','depth_path':'unused'}
        pred=np.array([[1,2],[99,.3]],dtype=np.float32)
        out=metric(np,pred,rec,CV(),.2,3)
        self.assertEqual(out['valid_pixels'],3);self.assertAlmostEqual(out['AbsRel'],0,places=6)
        pred[0,0]=float('nan');bad=metric(np,pred,rec,CV(),.2,3)
        self.assertEqual(bad['prediction_status'],'invalid_prediction');self.assertEqual(out['mask_sha256'],bad['mask_sha256']);self.assertEqual(bad['AbsRel'],'')
    def test_pairing_retains_denominator(self):
        a=[dict(index='0',gt_status='MATCHED',mask_sha256='h',valid_pixels='3',AbsRel='0.2'),dict(index='1',gt_status='MATCHED',mask_sha256='h',valid_pixels='3',AbsRel='')]
        b=[dict(index='0',gt_status='MATCHED',mask_sha256='h',valid_pixels='3',AbsRel='0.1'),dict(index='1',gt_status='MATCHED',mask_sha256='h',valid_pixels='3',AbsRel='0.3')]
        d,n,failed=paired(a,b);self.assertAlmostEqual(d[0],.1);self.assertEqual((n,failed),(2,1));self.assertIsNone(boot({'one':.1,'two':.2}))
        b[0]['mask_sha256']='different'
        with self.assertRaisesRegex(ValueError,'mask mismatch'):paired(a,b)
    def test_temporal_fixed_gt_intersection(self):
        rec={'gt_match_status':'MATCHED','depth_path':'unused'};pred=np.array([[1,2],[99,.3]],dtype=np.float32)
        first,state=temporal_metric(np,pred,rec,CV(),.2,3,None)
        self.assertEqual(first['temporal_status'],'no_previous_GT')
        second,state=temporal_metric(np,pred+.1,rec,CV(),.2,3,state)
        self.assertEqual(second['temporal_valid_pixels'],3);self.assertAlmostEqual(second['temporal_depth_change_mae_m'],.1,places=5)
        broken=pred.copy();broken[0,0]=np.nan
        invalid,_=temporal_metric(np,broken,rec,CV(),.2,3,state)
        self.assertEqual(invalid['temporal_status'],'invalid_prediction_or_GT')

if __name__=='__main__':unittest.main()
