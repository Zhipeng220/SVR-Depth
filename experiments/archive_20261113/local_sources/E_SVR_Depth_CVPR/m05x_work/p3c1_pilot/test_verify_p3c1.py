import hashlib,importlib.util,json,sys,tempfile,unittest,shutil
from pathlib import Path
import numpy as np
import verify_p3c1 as v
HERE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(HERE/'m05x_work/p3b'))
spec=importlib.util.spec_from_file_location('p3b_fixture',HERE/'m05x_work/p3b/test_verify_p3b.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(root,corrupt=None):
 previous=prior.fixture(root);aid='Bonn_00_age1';p3=root/'docs/m05x/p3b/runs'/aid;p3.parent.mkdir(parents=True);shutil.move(str(previous),str(p3));old=json.loads((p3/'capture_metadata.json').read_text());old.update(device='synthetic',torch='synthetic',cuda='synthetic');(p3/'capture_metadata.json').write_text(json.dumps(old));a=dict(np.load(p3/'evidence_arrays.npz',allow_pickle=False))
 evidence={'anchors':{aid:{'files':{'evidence_arrays.npz':sha(p3/'evidence_arrays.npz'),'capture_metadata.json':sha(p3/'capture_metadata.json')}}}};(root/'docs/m05x/p3b/evidence_manifest.json').write_text(json.dumps(evidence))
 script=root/'m05x_work/p3c1_pilot/capture_p3c1.py';script.parent.mkdir(parents=True);script.write_text('synthetic')
 lock={'anchor_ids':[aid],'checkpoint_sha256':{'V4':'v4','V5':old['identity']['checkpoint']},'backbone_sha256':old['identity']['backbone'],'decoder_sha256':'decoder','repair_source_sha256':'head','p2b_sequence_manifest_sha256':'fake','p2b_anchor_manifest_sha256':sha(root/'docs/m05x/p2b/anchor_manifest.json'),'norm_minimum':1e-12,'scale_minimum':.01,'scale_maximum':10.,'norm_match_relative_tolerance':1e-5,'noncollinearity_max_abs_cosine':.999,'formula_abs_tolerance':1e-6}
 lockpath=root/'docs/m05x/p3c1_pilot/pilot_protocol.json';lockpath.parent.mkdir(parents=True);lockpath.write_text(json.dumps(lock))
 meta={**old,'identity':{**old['identity'],'checkpoint_v4':'v4','decoder':'decoder','repair_head':'head','c1_lock':sha(lockpath),'protocol':'fake','capture_script':sha(script)},'v4_scale_gpu':1.,'inputs':[{'frame_index':0}],'head_precision':'cuda autocast float16'}
 for i in range(4):
  f=f'feature{i}';base=a['committed_O__'+f];cand5=a['committed_T__'+f];cand4=base+np.array([1.,-1.],dtype=np.float32)
  for b in ('V4','V5','after'):a[f'head_input_{b}__{f}']=base.copy()
  a['candidate_V4__'+f]=cand4.copy();a['candidate_V5__'+f]=cand5.copy();a['committed_V4M__'+f]=cand4.copy();a['committed_V4N__'+f]=cand4.copy();a['committed_V5__'+f]=cand5.copy()
 for b in v.BRANCHES:
  a['current_D1_'+b]=a['current_D1_O'].copy();a['input_'+b+'_anchor']=a['input_O_anchor'].copy()
  for typ in ('cpu','cuda_0'):a['replay_rng_'+b+'_'+typ]=a['replay_rng_O_'+typ].copy()
  for f in ('radius','confidence','rays_map'):a['committed_'+b+'__'+f]=a['committed_O__'+f].copy()
  for h in range(1,5):
   for pre in ('input','pred','gt_valid','metric_valid'):a[f'{pre}_{b}_{h}']=a[f'{pre}_O_{h}'].copy()
   for k in ('flow_norm','grid_sample','image_residual','validity_mask'):
    for pre in ('static','static_after'):a[f'{pre}_{b}_{h}__{k}']=a[f'{pre}_O_{h}__{k}'].copy()
   meta['metrics'][f'{b}_H{h}']={'AbsRel':0.}
 meta['storage_ptrs']={b:{f:1000+j*100+i for i,f in enumerate(v.FIELDS)} for j,b in enumerate(v.BRANCHES)}
 if corrupt=='norm':a['committed_V4M__feature0'][0]+=3
 if corrupt=='collinear':
  for i in range(4):a[f'candidate_V4__feature{i}']=a[f'candidate_V5__feature{i}'].copy();a[f'committed_V4N__feature{i}']=a[f'candidate_V5__feature{i}'].copy();a[f'committed_V4M__feature{i}']=a[f'candidate_V5__feature{i}'].copy()
 if corrupt=='alias':meta['storage_ptrs']['V4M']['radius']=meta['storage_ptrs']['O']['radius']
 if corrupt=='mask':a['gt_valid_V4M_1'][0]=False
 if corrupt=='sham':a['pred_S_4'][0]=3
 if corrupt=='endpoint':a['pred_V5_4'][0]=3
 out=root/'c1run';out.mkdir();np.savez_compressed(out/'evidence_arrays.npz',**a);meta['evidence_npz_sha256']=sha(out/'evidence_arrays.npz');(out/'capture_metadata.json').write_text(json.dumps(meta));return out
class Tests(unittest.TestCase):
 def test_complete_and_hard_failures(self):
  oldroot=v.ROOT
  try:
   for corrupt in (None,'norm','collinear','alias','mask','sham','endpoint'):
    with self.subTest(corrupt=corrupt),tempfile.TemporaryDirectory() as d:
     root=Path(d);p=fixture(root,corrupt);v.ROOT=root;r,_=v.verify(p);self.assertEqual(r['status'],'PASS' if corrupt is None else 'INVALID')
  finally:v.ROOT=oldroot
if __name__=='__main__':unittest.main()
