import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import quality
from contract import validate_config
import test_m01_contract as m01_fixture

class CUDA:
    def is_available(self):return True
    def get_device_properties(self,*a):return types.SimpleNamespace(uuid='GPU-fixture')

class Torch:
    cuda=CUDA()
    def set_num_threads(self,n):pass

class CV:
    IMREAD_UNCHANGED=1
    def imread(self,path,mode):return np.full((2,2),5000,dtype=np.uint16)

class FakePredictor:
    def __init__(self,rt,plan,unit,records):self.unit=unit;self.shape={'original_shape':(2,2)};self.n=len(records)
    def step(self,i,record):
        pred=np.ones((2,2),dtype=np.float32)
        if self.unit['family']=='P' and i==1:pred[0,0]=np.nan
        return pred,dict(keyframe=i%self.unit['k']==0,age=i%self.unit['k'])
    def counts(self):
        u=self.unit;k=u['k'];n=self.n;fast=n-(n+k-1)//k
        if u['experiment']=='M04':return dict(physical_fast_steps=fast*(2 if u['arm'] in ('OUTPUT_ONLY','F_ONLY') else 1),repair_head_calls=0 if u['arm']=='OFF' else fast)
        due=sum(1 for i in range(n) if i%k and (i%k-1)%u['frequency']==0);m=u['family']
        return dict(keyframe_calls=n-fast,flow_calls=fast,native_fast_calls=fast,repair_head_calls=due if m=='P' else 0,output_adapter_calls=due if m in ('O_fast','O_multi') else 0,additional_readout_passes=due if m=='O_multi' else 0,rejected_candidates=0)
    def close(self):pass
    def finite_state(self):return True

class QualityPipelineTest(unittest.TestCase):
    def test_real_orchestration_records_invalid_predictions(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);records=[dict(gt_match_status='MATCHED' if i<2 else 'UNMATCHED',depth_path='dummy',rgb_path='dummy') for i in range(3)]
            rt=types.SimpleNamespace(torch=Torch(),np=np,cv2=CV(),exp1=types.SimpleNamespace(read_manifest=lambda *a,**k:records))
            mods={'gt4a2_y11_fallback':types.SimpleNamespace(gpu_lock_path=lambda:('fixture',root/'lock')),'gt4a2_y04a_native_pre_readout_eval':types.SimpleNamespace(load_runtime=lambda:rt),'fcntl':types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1)}
            c=m01_fixture.M01Tests().config();c['units']=validate_config(c);c.update(status='FROZEN_QUALITY_ONLY',scope='M01_ONLY',blindness='HISTORY_UNVERIFIED_NOT_BLIND_VERIFIED',input_sha256={},expected_absent_depth=[]);c['sequences'][0]['RGB']=3
            plan=root/'plan.json';plan.write_text(json.dumps(c));out=root/'result'
            with patch.object(quality,'ROOT',root),patch.object(quality,'Predictor',FakePredictor),patch.dict(sys.modules,mods):self.assertEqual(quality.run(plan,out),2)
            summary=json.loads((out/'summary.json').read_text());self.assertTrue((out/'test_access.json').is_file());self.assertEqual(summary['role'],'heldout_history_unverified');self.assertEqual(summary['status'],'QUALITY_INCOMPLETE');self.assertFalse(summary['formal_v21_complete']);self.assertEqual(len(summary['completion_hashes']),10)
            self.assertEqual(len(summary['failed_units']),3)
            p=next((out/'quality/new_scene').glob('P__s20261025__k16*/complete.json'));meta=json.loads(p.read_text());self.assertEqual(meta['failed'],1);self.assertEqual(meta['valid'],1)

if __name__=='__main__':unittest.main()
