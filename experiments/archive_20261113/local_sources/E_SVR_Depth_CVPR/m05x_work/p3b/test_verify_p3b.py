import hashlib,importlib.util,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import verify_p3b as v

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'m05x_work/p2a'));sys.path.insert(0,str(ROOT/'docs/m05x/p1_5'))
spec=importlib.util.spec_from_file_location('previous_fixture',ROOT/'m05x_work/p2a/test_verify_p2a.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(root,change=None):
 aid='Bonn_00_age1';p2=root/'docs/m05x/p2b/runs'/aid;p2.mkdir(parents=True);prior.fixture(p2)
 old=json.loads((p2/'capture_metadata.json').read_text());old.update(dataset='Bonn',sequence='x',age=1,inputs=[{'frame_index':0}]);old['identity'].update({k:'fake' for k in ('model_config','exp1','gt2','m05','external_eval','pair_csv')});old['evidence_npz_sha256']=sha(p2/'evidence_arrays.npz');(p2/'capture_metadata.json').write_text(json.dumps(old))
 evidence={'anchors':{aid:{'files_sha256':{'x/evidence_arrays.npz':sha(p2/'evidence_arrays.npz'),'x/capture_metadata.json':sha(p2/'capture_metadata.json')}}}};(root/'docs/m05x/p2b/evidence_manifest.json').write_text(json.dumps(evidence))
 anchors={'anchors':[{'anchor_id':aid,'dataset':'Bonn','sequence':'x','sequence_index':0,'age':1,'cycle':0}]};anchor_path=root/'docs/m05x/p2b/anchor_manifest.json';anchor_path.write_text(json.dumps(anchors))
 script=root/'m05x_work/p3b/capture_p3b.py';script.parent.mkdir(parents=True);script.write_text('synthetic')
 lock={'checkpoint_sha256':old['identity']['checkpoint'],'backbone_sha256':old['identity']['backbone'],'p2b_sequence_manifest_sha256':'fake','p2b_anchor_manifest_sha256':sha(anchor_path),'pilot_anchor_ids':[aid],'formal_anchor_ids':[aid]}
 lockpath=root/'docs/m05x/p3b/protocol_lock.json';lockpath.parent.mkdir(parents=True);lockpath.write_text(json.dumps(lock))
 a=dict(np.load(p2/'evidence_arrays.npz',allow_pickle=False));meta={**old,'anchor_id':aid,'phase':'pilot','sequence_index':0,'cycle':0,'K':16,'horizons':[1,2,3,4],'repair_replay_mode':'off','gt_loaded_after_replays':True}
 meta['identity'].update(p3_lock=sha(lockpath),protocol='fake',capture_script=sha(script))
 for j,alpha in enumerate(v.ALPHAS):
  for f in v.FIELDS:
   if f.startswith('feature'):
    a['committed_T__'+f]=a['committed_O__'+f]+np.float32(1)
    a[f'committed_A{j}__{f}']=a['committed_O__'+f]+np.float32(alpha)
   else:a[f'committed_A{j}__{f}']=a['committed_O__'+f].copy()
 for f in v.FIELDS:a['committed_S__'+f]=a['committed_O__'+f].copy()
 ptr={};meta['metrics']={}
 for k,b in enumerate(v.BRANCHES):
  ptr[b]={f:1000+100*k+i for i,f in enumerate(v.FIELDS)}
  a['current_D1_'+b]=a['current_D1_O'].copy();a['input_'+b+'_anchor']=a['input_O_anchor'].copy()
  for typ in ('cpu','cuda_0'):a[f'replay_rng_{b}_{typ}']=a[f'replay_rng_O_{typ}'].copy()
  for h in range(1,5):
   for pre in ('input','pred','gt_valid','metric_valid'):a[f'{pre}_{b}_{h}']=a[f'{pre}_O_{h}'].copy()
   for key in ('flow_norm','grid_sample','image_residual','validity_mask'):
    for pre in ('static','static_after'):a[f'{pre}_{b}_{h}__{key}']=a[f'{pre}_O_{h}__{key}'].copy()
   meta['metrics'][f'{b}_H{h}']={'AbsRel':0.}
 meta['storage_ptrs']=ptr
 if change=='endpoint':a['pred_A4_4'][0]=3
 if change=='dose':a['committed_A2__feature1'][0]=9
 if change=='alias':meta['storage_ptrs']['A1']['radius']=meta['storage_ptrs']['A0']['radius']
 if change=='mask':a['gt_valid_A3_1'][0]=False
 out=root/'run';out.mkdir();np.savez_compressed(out/'evidence_arrays.npz',**a);meta['evidence_npz_sha256']=sha(out/'evidence_arrays.npz');(out/'capture_metadata.json').write_text(json.dumps(meta))
 return out

class Tests(unittest.TestCase):
 def test_dose_and_invariants_fail_closed(self):
  oldroot=v.ROOT
  try:
   for change in (None,'endpoint','dose','alias','mask'):
    with self.subTest(change=change),tempfile.TemporaryDirectory() as d:
     root=Path(d);p=fixture(root,change);v.ROOT=root;r,_=v.verify(p);self.assertEqual(r['status'],'PASS' if change is None else 'INVALID')
  finally:v.ROOT=oldroot
if __name__=='__main__':unittest.main()
