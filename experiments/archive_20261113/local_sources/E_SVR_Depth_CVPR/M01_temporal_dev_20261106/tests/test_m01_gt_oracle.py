import unittest
import importlib.util
import numpy as np
from svr_depth.evaluation.m01_v31.flow_provider import forward_backward_masks
from svr_depth.evaluation.m01_v31.gt_oracle import evaluate_gt_oracle


class OracleTests(unittest.TestCase):
    def test_forward_backward_consistency_and_occlusion(self):
        f=np.zeros((3,5,2),np.float32);b=f.copy();f[...,0]=1;b[...,0]=-1
        result=forward_backward_masks(f,b,0.25)
        self.assertTrue(result['flow_bwd_valid'][:,1:].all())
        self.assertFalse(result['flow_bwd_valid'][:,0].any())
        b[1,2,0]=0
        self.assertFalse(forward_backward_masks(f,b,.25)['flow_bwd_valid'][1,2])

    def test_identity_and_translation_compensation(self):
        shape=(3,5);gt=np.full(shape,2.);mask=np.ones(shape,dtype=bool)
        flow=np.zeros(shape+(2,),dtype=np.float32);K=np.eye(3);pose=np.eye(4)
        result=evaluate_gt_oracle(gt,gt,mask,mask,flow,mask,K,pose,pose,pixel_center='integer')
        self.assertEqual((result['status'],result['tau5'],result['valid_pixels']),('ok',1.,15))
        moved=pose.copy();moved[2,3]=0.5
        result=evaluate_gt_oracle(gt,np.full(shape,1.5),mask,mask,flow,mask,K,pose,moved,pixel_center='integer')
        self.assertEqual(result['tau5'],1.)
        self.assertEqual(result['tau5_no_ego_motion'],0.)
        self.assertEqual(result['tau5_wrong_pose_direction'],0.)

    def test_invalid_pose_rejected(self):
        shape=(2,2);gt=np.ones(shape);mask=np.ones(shape,dtype=bool);bad=np.eye(4);bad[0,0]=2
        with self.assertRaises(ValueError):
            evaluate_gt_oracle(gt,gt,mask,mask,np.zeros(shape+(2,)),mask,np.eye(3),bad,np.eye(4),pixel_center='integer')

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,'PyTorch required for upstream camera parity')
    def test_upstream_opencv_camera_projection_roundtrip(self):
        import torch
        from svr_depth.evaluation.m01_v31.camera_geometry import camera_rays
        from velodepth.utils.camera import OPENCV
        params=[500.,500.,2.,2.,.01,-.001,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        y,x=np.mgrid[:4,:5];xy=np.stack((x,y),axis=-1).astype(np.float32)
        rays=camera_rays({'model':'OPENCV','params':params},xy,pixel_center='half')
        model=OPENCV(torch.tensor(params))
        with torch.no_grad():
            projected=model.project(torch.from_numpy(rays).permute(2,0,1)[None]).squeeze(0).permute(1,2,0).numpy()
        np.testing.assert_allclose(projected,xy+.5,rtol=0,atol=.02)


if __name__=='__main__':unittest.main()
