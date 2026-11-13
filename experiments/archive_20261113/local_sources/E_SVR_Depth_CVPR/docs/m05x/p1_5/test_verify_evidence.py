import json,tempfile,unittest
from pathlib import Path
import numpy as np
from verify_evidence import FIELDS,verify,sha

def fixture(root,mode='ok'):
    a={};runtime={};committed={}
    for i,f in enumerate(FIELDS):
        x=np.array([1.,2.],dtype=np.float32)
        for key in ('parent_O__','parent_T__','committed_O__','committed_T__','alias_before_T__','alias_after_T__','commit_alias_before_T__','commit_alias_after_T__'):a[key+f]=x.copy()
        for key in ('alias','commit_alias'):
            a[key+'_O_before__'+f]=np.array(1.);a[key+'_O_mutated__'+f]=np.array(2.)
        runtime[f]={'o_ptr':1+i,'t_ptr':20+i};committed[f]={'o_ptr':40+i,'t_ptr':60+i}
    a['current_D1_O']=np.ones((2,),dtype=np.float32);a['current_D1_T']=a['current_D1_O'].copy()
    for k in ('cpu','cuda_0'):a['replay_rng_O_'+k]=np.array([1]);a['replay_rng_T_'+k]=np.array([1])
    a['input_O_anchor']=np.array([1]);a['input_T_anchor']=np.array([1]);metrics={}
    for h in (1,2,3,4):
        a[f'gt_{h}']=np.array([1.,2.]);
        for b in ('O','T'):
            a[f'input_{b}_{h}']=np.array([h]);a[f'gt_valid_{b}_{h}']=np.array([True,True]);a[f'metric_valid_{b}_{h}']=np.array([True,True]);a[f'pred_{b}_{h}']=np.array([1.,2.]);metrics[f'{b}_H{h}']={'AbsRel':0.0}
            for k in ('flow_norm','grid_sample','image_residual','validity_mask'):
                a[f'static_{b}_{h}__{k}']=np.array([h]);a[f'static_after_{b}_{h}__{k}']=np.array([h])
    if mode=='parent': a['parent_T__radius'][0]=3
    if mode=='d1':a['current_D1_T'][0]=3
    if mode=='mask':a['gt_valid_T_1'][0]=False
    if mode=='alias':a['commit_alias_after_T__radius'][0]=3
    if mode=='protocol':a['static_T_1__flow_norm'][0]=3
    if mode=='missing':del a['gt_valid_T_1']
    np.savez_compressed(root/'evidence_arrays.npz',**a)
    (root/'capture_metadata.json').write_text(json.dumps({'alias_runtime':runtime,'committed_alias_runtime':committed,'repair_replay_mode':'off','horizons':[1,2,3,4],'gt_loaded_after_replays':True,'metrics':metrics,'identity':{'checkpoint':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251','backbone':'615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e','capture_script':sha(Path(__file__).with_name('capture_invariants.py').read_bytes())}}))

class Tests(unittest.TestCase):
    def test_complete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root);self.assertEqual(verify(root)[0]['status'],'PASS')
    def test_each_hard_failure(self):
        for mode in ('parent','d1','mask','alias','protocol'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as d:
                root=Path(d);fixture(root,mode);self.assertEqual(verify(root)[0]['status'],'FAIL')
    def test_missing_evidence_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root,'missing')
            self.assertNotEqual(verify(root)[0]['status'],'PASS')

if __name__=='__main__':unittest.main()
