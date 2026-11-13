import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from verify_p2a import verify
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'docs/m05x/p1_5'))
spec=importlib.util.spec_from_file_location('old_tests',ROOT/'docs/m05x/p1_5/test_verify_evidence.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
def fixture(root,mode='ok'):
    old.fixture(root)
    meta=json.loads((root/'capture_metadata.json').read_text());a=dict(np.load(root/'evidence_arrays.npz',allow_pickle=False))
    for f in old.FIELDS:a['committed_S__'+f]=a['committed_O__'+f].copy()
    for k in ('cpu','cuda_0'):a['replay_rng_S_'+k]=a['replay_rng_O_'+k].copy()
    a['input_S_anchor']=a['input_O_anchor'].copy()
    for h in (1,2,3,4):
        a[f'input_S_{h}']=a[f'input_O_{h}'].copy();a[f'pred_S_{h}']=a[f'pred_O_{h}'].copy()
        for k in ('flow_norm','grid_sample','image_residual','validity_mask'):
            for prefix in ('static','static_after'):a[f'{prefix}_S_{h}__{k}']=a[f'{prefix}_O_{h}__{k}'].copy()
    if mode=='sham':a['pred_S_4'][0]=3
    if mode=='feature':a['committed_T__radius'][0]=3
    if mode=='mask':a['gt_valid_T_1'][0]=False
    if mode=='rng':a['replay_rng_T_cpu'][0]=3
    np.savez_compressed(root/'evidence_arrays.npz',**a)
    meta.update(dataset='Bonn',sequence='x',age=1,K=16,sham_ptrs={f:100+i for i,f in enumerate(old.FIELDS)})
    meta['identity']['capture_script']=hashlib.sha256((ROOT/'m05x_work/p2a/capture_p2a.py').read_bytes()).hexdigest()
    meta['identity']['protocol']=hashlib.sha256((ROOT/'docs/m05x/p2a/pilot_protocol.json').read_bytes()).hexdigest()
    meta['evidence_npz_sha256']=hashlib.sha256((root/'evidence_arrays.npz').read_bytes()).hexdigest()
    (root/'capture_metadata.json').write_text(json.dumps(meta))

class Tests(unittest.TestCase):
    def test_pass_and_fail_closed(self):
        for mode in ('ok','sham','feature','mask','rng'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
                root=Path(d);fixture(root,mode);result,_=verify(root)
                self.assertEqual(result['status'],'PASS' if mode=='ok' else 'INVALID')
    def test_missing_sham_cannot_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);fixture(root);a=dict(np.load(root/'evidence_arrays.npz'));del a['pred_S_4'];np.savez_compressed(root/'evidence_arrays.npz',**a)
            with self.assertRaises(KeyError):verify(root)
if __name__=='__main__':unittest.main()
