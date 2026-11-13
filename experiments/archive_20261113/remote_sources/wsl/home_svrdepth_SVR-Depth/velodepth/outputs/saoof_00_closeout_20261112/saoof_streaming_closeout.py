"""Official Bonn forward_test: output-only streaming SA-OOF contract check."""
import os,sys,glob,json,copy
from pathlib import Path
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
import numpy as np,torch,cv2
import torch.nn.functional as F
sys.path.insert(0,'experiments/p1')
import gt2_fixed_clip_repair_train as gt2
from velodepth.datasets.bonn import Bonn
from velodepth.datasets import collate_fn
from saoof_v1_core import keyframe_anchor,decompose,fuse
from saoof_correspondence import correspondence_mask
from saoof_stream_output import SAOOFStreamOutput

root=Path('outputs/saoof_00_closeout_20261112'); root.mkdir(exist_ok=True,parents=True)
cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']; old=np.array
class LegacyArray(np.ndarray):
 def tostring(self,*a,**kw): return self.tobytes(*a,**kw)
def compat(*a,**kw): return old(*a,**kw).view(LegacyArray)
try:
 np.array=compat
 ds=Bonn(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,crop=cfg['crop'],normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1,add_of=True)
finally: np.array=old
ix=next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/person_tracking2_val'))
# Preserve official loader/metrics while enforcing the protocol's frozen native canvas.
ds.resizer.sample=False
ds.resizer.image_shape=[434,574]
ds.get_test_idxs=lambda n,k:(list(range(78,94)),None)
sample=ds.get_single_sequence(ix); batch=collate_fn([sample],is_batched=False); device=torch.device('cuda')
batch['data']={k:v.to(device) for k,v in batch['data'].items()}; assert batch['data']['image'].shape[1]==16 and tuple(batch['data']['image'].shape[-2:])==(434,574)
snapshot=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
native=gt2.load_model(snapshot,'off',device); ref=gt2.load_model(snapshot,'off',device)
native.pixel_decoder.camera_gt=False; ref.pixel_decoder.camera_gt=False
assert native.pixel_decoder is not ref.pixel_decoder
orig_kf=native.pixel_decoder.forward_keyframe; orig_fast=native.pixel_decoder.forward_fast
orig_encode=native.encode_decode_test
mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]; std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
def grayscale(x):
 rgb=((x.float()*std+mean).clamp(0,1)*255).byte()[0].permute(1,2,0).cpu().numpy()
 return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)

def reference(x,size,target_size):
 z=F.interpolate(x,size=size,mode='bilinear',align_corners=False)
 inp={'image':z,'image_original':z,'validity_mask':torch.ones((1,1,*size),device=device,dtype=torch.bool)}
 features,tokens=ref.pixel_encoder(z)
 inp['features']=[ref.stacking_fn(features[a:b]).contiguous() for a,b in ref.slices_encoder_range]
 inp['tokens']=[ref.stacking_fn(tokens[a:b]).contiguous() for a,b in ref.slices_encoder_range]
 ref.pixel_decoder.set_shapes(inp,[{'B':1,'T':1}]); rays,r,_,_=ref.pixel_decoder.forward_keyframe(inp,[{'B':1,'T':1}])
 rr=rays.reshape(1,*size,3).permute(0,3,1,2)
 depth=r.float()*rr[:,2:3].float()
 assert bool((torch.isfinite(depth)&(depth>0)).all())
 return F.interpolate(depth.log(),size=target_size,mode='bilinear',align_corners=False).exp()[0,0]

def keyframe(inp,meta):
 global kfrays,processor,previous
 output=orig_kf(inp,meta); rays,r,_,_=output
 h,w=inp['image'].shape[-2:]; kfrays=rays.reshape(1,h,w,3).permute(0,3,1,2).float()
 prop=(r.float()*kfrays[:,2:3])[0,0]
 previous=inp['image'].detach().clone(); processor=SAOOFStreamOutput()
 if mode not in ('control','missing'):
  r0=reference(inp['image'],(224,294),(h,w))
  assert processor.begin_cycle(prop,r0,inp['validity_mask'][0,0])
  if mode=='invalid_anchor': processor.begin_cycle(prop,None,inp['validity_mask'][0,0])
 return output

def fast(inp,meta):
 global previous,step
 native_out=orig_fast(inp,meta)
 r,c,features=native_out; h,w=inp['image'].shape[-2:]
 prop=(r.float()*kfrays[:,2:3])[0,0]; output=prop.clone(); status='control'
 if mode=='missing':
  output,status=processor.process(prop,None,None,None,inp['validity_mask'][0,0],.5,torch.full_like(prop,.5))
 elif mode!='control':
  reference_depth=reference(inp['image'],(224,294),(h,w))
  dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
  a=grayscale(previous); b=grayscale(inp['image']); ff=dis.calc(a,b,None); fb=dis.calc(b,a,None)
  if mode=='insufficient_mask': fb[:]=np.nan
  output,status=processor.process(prop,reference_depth,ff,fb,inp['validity_mask'][0,0],.5,torch.full_like(prop,.5))
 corrected.append(output.cpu()); original_depth.append(prop.cpu()); statuses.append(status)
 states.append([r.float().cpu(),c.float().cpu(),*[x.float().cpu() for x in features]])
 previous=inp['image'].detach().clone(); step+=1
 # Native state returned unmodified; corrected depth is emitted separately.
 return native_out

native.pixel_decoder.forward_keyframe=keyframe; native.pixel_decoder.forward_fast=fast
def encode_with_output(inputs,metas):
 global native_points
 packed,out=orig_encode(inputs,metas)
 native_points=out['points'].detach().float().cpu().clone()
 # Native forward state has already advanced with unmodified radius/features.
 # Materialize the earlier per-frame output selections in the official return.
 if mode=='valid':
  points=out['points'].clone()
  assert points.shape[0]==16 and len(corrected)==15
  for j,z in enumerate(corrected,1):
   target=z.to(points.device,dtype=points.dtype)
   ratio=target/points[j,2].clamp_min(torch.finfo(points.dtype).tiny)
   points[j]=points[j]*ratio.unsqueeze(0)
  out['points']=points
 return packed,out
native.encode_decode_test=encode_with_output
def run(m):
 global mode,states,corrected,original_depth,statuses,processor,previous,kfrays,step,native_points
 mode=m; states=[]; corrected=[]; original_depth=[]; statuses=[]; step=0
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
  pred,_=native.forward_test(batch['data'],copy.deepcopy(batch['img_metas']))
 assert len(states)==15 and step==15
 return pred['depth'].float().cpu(),states,corrected,original_depth,statuses,native_points

control,control_states,_,_,_,control_points=run('control'); results={}
for name in ('missing','invalid_anchor','insufficient_mask','valid'):
 pred,st,outputs,props,statuses,pre_points=run(name)
 state_max=max(float((x-y).abs().max()) for a,b in zip(control_states,st) for x,y in zip(a,b))
 changes=[float((x-y).abs().max()) for x,y in zip(outputs,props)]
 results[name]={'official_return_depth_change_max_abs':float((pred-control).abs().max()),'native_pre_output_points_max_abs':float((pre_points-control_points).abs().max()),'native_state_max_abs':state_max,'stream_output_change_max_per_frame':changes,'status_per_frame':statuses,'fallback_all_bitwise':all(torch.equal(x,y) for x,y in zip(outputs,props)),'official_return_equal_control':bool(torch.equal(pred,control))}
 assert torch.equal(pre_points,control_points) and state_max==0
 if name!='valid': assert results[name]['fallback_all_bitwise'] and torch.equal(pred,control)
 else: assert max(changes)>1e-6 and not torch.equal(pred,control) and all(s=='ok' for s in statuses)
out={'status':'ENGINEERING_CHECK_ONLY','sequence':ds.dataset[ix]['sequence_name'],'indices':list(range(78,94)),'native_input_size':[434,574],'preprocessing':'official Bonn loader, explicitly fixed resizer output at 434x574 (default config selected 476x630)','camera_gt_native':False,'camera_gt_reference':False,'reference_size':[224,294],'schedule':'1 keyframe + 15 fast; adaptive thresholds disabled','states_checked':'radius, confidence and returned features','results':results,'limitations':'Single clip; per-frame SAOOFStreamOutput adapter is called after native state generation, wrapper materializes outputs into official forward_test return after the clip. Official native/reference orchestration is still a test hook. GT pairing timestamp source unresolved; no accuracy or latency results.'}
(root/'streaming_closeout.json').write_text(json.dumps(out,indent=2),encoding='utf8'); print(json.dumps(out,indent=2))
