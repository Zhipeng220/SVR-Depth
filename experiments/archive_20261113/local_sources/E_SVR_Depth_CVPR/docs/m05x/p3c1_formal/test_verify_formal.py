import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import verify_formal as v
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'m05x_work/p3c1_pilot'))
spec=importlib.util.spec_from_file_location('pilot_fixture',ROOT/'m05x_work/p3c1_pilot/test_verify_p3c1.py');pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(root,corrupt=None):
 p=pilot.fixture(root);meta=json.loads((p/'capture_metadata.json').read_text());a=dict(np.load(p/'evidence_arrays.npz',allow_pickle=False));lock=json.loads((root/'docs/m05x/p3c1_pilot/pilot_protocol.json').read_text())
 dest=root/'docs/m05x/p3c1_formal';dest.mkdir(parents=True);manifest=dest/'anchor_manifest.json';manifest.write_text('{}');lock['formal_anchor_manifest_sha256']=sha(manifest);lock['layerwise_match_relative_tolerance']=1e-5
 protocol=dest/'formal_protocol.json';protocol.write_text(json.dumps(lock));script=root/'m05x_work/p3c1_formal/capture_formal.py';script.parent.mkdir(parents=True);script.write_text('formal synthetic');meta['identity'].update(c1_lock=sha(protocol),capture_script=sha(script))
 for i in range(4):a[f'committed_V4L__feature{i}']=a[f'committed_V4N__feature{i}'].copy()
 b='V4L';a['current_D1_'+b]=a['current_D1_O'].copy();a['input_'+b+'_anchor']=a['input_O_anchor'].copy()
 for typ in ('cpu','cuda_0'):a['replay_rng_'+b+'_'+typ]=a['replay_rng_O_'+typ].copy()
 for f in ('radius','confidence','rays_map'):a['committed_'+b+'__'+f]=a['committed_O__'+f].copy()
 for h in range(1,5):
  for pre in ('input','pred','gt_valid','metric_valid'):a[f'{pre}_{b}_{h}']=a[f'{pre}_O_{h}'].copy()
  for k in ('flow_norm','grid_sample','image_residual','validity_mask'):
   for pre in ('static','static_after'):a[f'{pre}_{b}_{h}__{k}']=a[f'{pre}_O_{h}__{k}'].copy()
  meta['metrics'][f'{b}_H{h}']={'AbsRel':0.}
 meta['storage_ptrs'][b]={f:8000+i for i,f in enumerate(v.FIELDS)};meta['v4_layer_scales_gpu']=[1.]*4;meta['events']=[e for b in v.BRANCHES for e in ('replay_start_'+b,'replay_end_'+b)]+['gt_load_start']
 if corrupt=='layer_norm':a['committed_V4L__feature0'][0]+=2
 if corrupt=='layer_direction':a['committed_V4L__feature0']=a['candidate_V5__feature0'].copy()
 if corrupt=='alias':meta['storage_ptrs']['V4L']['feature0']=meta['storage_ptrs']['O']['feature0']
 if corrupt=='gt_order':meta['events'].insert(1,meta['events'].pop())
 if corrupt=='scale':meta['v4_layer_scales_gpu'][1]=2.
 np.savez_compressed(p/'evidence_arrays.npz',**a);meta['evidence_npz_sha256']=sha(p/'evidence_arrays.npz');(p/'capture_metadata.json').write_text(json.dumps(meta));return p
class Tests(unittest.TestCase):
 def test_layerwise_and_runtime_fail_closed(self):
  old=v.ROOT
  try:
   for kind in (None,'layer_norm','layer_direction','alias','gt_order','scale'):
    with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
     root=Path(d);p=fixture(root,kind);v.ROOT=root;r,_=v.verify(p);self.assertEqual(r['status'],'PASS' if kind is None else 'INVALID')
  finally:v.ROOT=old
if __name__=='__main__':unittest.main()
