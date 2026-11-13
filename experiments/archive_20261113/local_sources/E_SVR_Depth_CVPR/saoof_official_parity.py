"""Compare official Bonn loader + forward_test with reference interleaving.

Diagnostic on 16 HDF5 container entries; raw acquisition correspondence is not assumed.
"""
import os,sys,glob,json,copy
from pathlib import Path
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0,'experiments/p1')
import gt2_fixed_clip_repair_train as gt2
from velodepth.datasets.bonn import Bonn
from velodepth.datasets import collate_fn
from saoof_v1_core import keyframe_anchor, decompose, fuse

root=Path('outputs/saoof_00_contract_v2_20261112')
cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']
old=np.array
class LegacyArray(np.ndarray):
 def tostring(self,*a,**kw): return self.tobytes(*a,**kw)
def compat(*a,**kw): return old(*a,**kw).view(LegacyArray)
try:
 np.array=compat
 ds=Bonn(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,crop=cfg['crop'],normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1,add_of=True)
finally: np.array=old
i=next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/person_tracking2_val'))
ds.get_test_idxs=lambda n,k: (list(range(78,94)),None)
item=ds.get_single_sequence(i); batch=collate_fn([item],is_batched=False)
device=torch.device('cuda'); batch['data']={k:v.to(device) for k,v in batch['data'].items()}
assert batch['data']['image'].shape[1]==16
snapshot=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
native=gt2.load_model(snapshot,'off',device); reference=gt2.load_model(snapshot,'off',device)
native.pixel_decoder.camera_gt=False # official scripts/eval.py default, without --camera-gt
original=native.pixel_decoder.forward_fast
states=[]; call=0
def wrapped(inp,meta):
 global call
 if mode=='interleaved':
  size=(224,294) if call%2==0 else (294,392)
  x=F.interpolate(inp['image'],size=size,mode='bilinear',align_corners=False)
  rin={'image':x,'image_original':x,'validity_mask':torch.ones((1,1,*size),device=device,dtype=torch.bool)}
  feats,tokens=reference.pixel_encoder(x)
  rin['features']=[reference.stacking_fn(feats[j:k]).contiguous() for j,k in reference.slices_encoder_range]
  rin['tokens']=[reference.stacking_fn(tokens[j:k]).contiguous() for j,k in reference.slices_encoder_range]
  reference.pixel_decoder.set_shapes(rin,[{'B':1,'T':1}]); reference.pixel_decoder.forward_keyframe(rin,[{'B':1,'T':1}])
 out=original(inp,meta)
 states.append([out[0].float().cpu(),out[1].float().cpu(),*[v.float().cpu() for v in out[2]]])
 call+=1
 return out
native.pixel_decoder.forward_fast=wrapped
def run(m):
 global mode,states,call
 mode=m; states=[]; call=0
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
  pred,_=native.forward_test(batch['data'],copy.deepcopy(batch['img_metas']))
 return pred['depth'].float().cpu(),states
a,sa=run('control'); b,sb=run('interleaved')
assert len(sa)==len(sb)==15, f'Expected 1 keyframe + 15 fast calls, got {len(sa)}, {len(sb)}'
assert a.shape==b.shape and a.shape[1]==16
perframe=(a-b).abs().flatten(2).max(dim=2).values.flatten().tolist()
state_max=max(float((x-y).abs().max()) for u,v in zip(sa,sb) for x,y in zip(u,v))
fallbacks=[]
for j in range(16):
 p=b[0,j,0]; ref=p.clone(); valid=torch.ones_like(p,dtype=torch.bool); empty=torch.zeros_like(valid)
 invalid=decompose(p,ref,None,valid)
 anchor=keyframe_anchor(p,ref); assert anchor is not None
 insufficient=decompose(p,ref,anchor,empty)
 def from_result(r): return fuse(p,r['global_residual'],r['local_residual'],r['valid_depth_mask'])
 fallbacks.append({'frame':j,'missing_reference':bool(torch.equal(fuse(p),p)), 'invalid_anchor':bool(torch.equal(from_result(invalid),p)),'insufficient_mask':bool(torch.equal(from_result(insufficient),p))})
result={'status':'ENGINEERING_CHECK_ONLY','sequence':ds.dataset[i]['sequence_name'],'hdf5_indices':list(range(78,94)),'shape':list(a.shape),'camera_gt':False,'schedule':'one keyframe plus 15 fast frames; adaptive thresholds disabled for K16','reference_calls':len(sb),'per_frame_depth_max_abs':perframe,'propagated_state_max_abs':state_max,'all_control_depth_finite_positive':bool((torch.isfinite(a)&(a>0)).all()),'post_forward_fallbacks':fallbacks,'limitations':'Only one HDF5 clip. Two-run same official forward_test baseline, not external official checkpoint comparison. Source RGB/GT temporal pairing unresolved. Reference rays are predicted in isolation diagnostic. Fallbacks are applied to returned depths after forward_test, not integrated into streaming execution.'}
(root/'official_parity.json').write_text(json.dumps(result,indent=2),encoding='utf8'); print(json.dumps(result,indent=2))
