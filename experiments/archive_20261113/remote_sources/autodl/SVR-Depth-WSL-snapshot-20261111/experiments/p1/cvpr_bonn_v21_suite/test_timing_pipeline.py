import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import quality,timing
from contract import validate_config
from test_contract import example
from test_quality_pipeline import FakePredictor,CV

class CUDA:
    def is_available(self):return True
    def get_device_properties(self,*a):return types.SimpleNamespace(uuid='GPU-fixture')
    def synchronize(self):pass
    def reset_peak_memory_stats(self):pass
    def max_memory_allocated(self):return 1024
class Torch:
    cuda=CUDA()
    def set_num_threads(self,n):pass
class Predictor(FakePredictor):
    def step(self,i,r):return np.ones((2,2),dtype=np.float32),dict(keyframe=i%self.unit['k']==0,age=i%self.unit['k'])
    def reset(self):pass

class TimingTest(unittest.TestCase):
    def test_rotation_and_quality_probes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);img=root/'rgb';img.write_bytes(b'rgb')
            records=[dict(gt_match_status='MATCHED',depth_path='dummy',rgb_path=str(img)) for i in range(2)]
            rt=types.SimpleNamespace(torch=Torch(),np=np,cv2=CV(),exp1=types.SimpleNamespace(read_manifest=lambda *a,**k:records))
            mods={'gt4a2_y11_fallback':types.SimpleNamespace(gpu_lock_path=lambda:('fixture',root/'lock')),'gt4a2_y04a_native_pre_readout_eval':types.SimpleNamespace(load_runtime=lambda:rt),'fcntl':types.SimpleNamespace(flock=lambda *a:None,LOCK_EX=1),'gt4a2_y08cal_y09pareto':types.SimpleNamespace(gpu_endpoint=lambda rt:dict(clean=True))}
            c=example();units=validate_config(c);c['units']=[units[0],next(x for x in units if x['id']=='M04__OFF__s0')]+[x for x in units if x['family']=='P' and x['k']==16 and x['seed'] in ('0','1','2') and (x['experiment']=='M01_M02' or x.get('arm')=='P_NATIVE')];c.update(status='FROZEN_QUALITY_ONLY',input_sha256={},expected_absent_depth=[]);c['sequences'][0]['RGB']=2
            p=root/'plan';p.write_text(json.dumps(c));q=root/'quality';out=root/'timing'
            with patch.object(quality,'ROOT',root),patch.object(quality,'Predictor',Predictor),patch.object(timing,'ROOT',root),patch.object(timing,'Predictor',Predictor),patch.dict(sys.modules,mods):
                self.assertEqual(quality.run(p,q),0);self.assertEqual(timing.run(p,q,out),0)
            s=json.loads((out/'summary.json').read_text());self.assertEqual(s['status'],'TIMING_COMPLETE');self.assertEqual(len(s['completed']),3*len(c['units']))
            a=c['units'][0]['id'];b=c['units'][1]['id']
            r1=json.loads((out/'new_bonn'/a/'r1.json').read_text());r2=json.loads((out/'new_bonn'/a/'r2.json').read_text())
            self.assertEqual((r1['position'],r2['position']),(0,len(c['units'])-1));self.assertGreater(r1['quality_probes'],0)
            self.assertTrue((out/'new_bonn'/b/'r3.json').is_file())
            mods['gt4a2_y08cal_y09pareto']=types.SimpleNamespace(gpu_endpoint=lambda rt:dict(clean=False))
            with patch.object(timing,'ROOT',root),patch.object(timing,'Predictor',Predictor),patch.dict(sys.modules,mods):
                self.assertEqual(timing.run(p,q,root/'rejected'),2)
            rejected=json.loads((root/'rejected/summary.json').read_text())
            self.assertEqual(rejected['status'],'TIMING_INCOMPLETE');self.assertEqual(len(rejected['failures']),3*len(c['units']))

if __name__=='__main__':unittest.main()
