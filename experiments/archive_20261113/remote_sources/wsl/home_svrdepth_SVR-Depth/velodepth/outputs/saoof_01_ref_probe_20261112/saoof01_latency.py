"""V1.1 synchronized online per-frame benchmark, separate from predictions."""
import os,sys,glob,json,time,argparse,statistics,traceback
from pathlib import Path
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
import numpy as np,torch,cv2
import torch.nn.functional as F
sys.path.insert(0,'experiments/p1')
import gt2_fixed_clip_repair_train as gt2
import gt4a2_exp1_train_eval as exp1
from velodepth.datasets.bonn import Bonn
from velodepth.datasets import collate_fn
from velodepth.utils.misc import match_gt
from saoof_reference_preprocess import reference_input
from saoof_v1_core import keyframe_anchor,decompose
from saoof_correspondence import correspondence_mask

parser=argparse.ArgumentParser(); parser.add_argument('--pilot',action='store_true'); args=parser.parse_args()
root=Path('outputs/saoof_01_ref_probe_20261112')/('latency_pilot' if args.pilot else 'latency'); root.mkdir(parents=True,exist_ok=True)
status={'state':'RUNNING'}
def save(): (root/'status.json').write_text(json.dumps(status,indent=2),encoding='utf8')
save()
try:
 cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']; old=np.array
 class Legacy(np.ndarray):
  def tostring(self,*a,**kw): return self.tobytes(*a,**kw)
 def compat(*a,**kw): return old(*a,**kw).view(Legacy)
 try:
  np.array=compat
  ds=Bonn(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,crop=cfg['crop'],normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1,add_of=True)
 finally: np.array=old
 ds.resizer.sample=False; ds.resizer.image_shape=[434,574]
 ix=next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/person_tracking2_val'))
 snap=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
 device=torch.device('cuda'); native=gt2.load_model(snap,'off',device); ref=gt2.load_model(snap,'off',device)
 native.pixel_decoder.camera_gt=False; ref.pixel_decoder.camera_gt=False
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]; std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def gray(x):
  a=((x.float()*std+mean).clamp(0,1)*255).byte()[0].permute(1,2,0).cpu().numpy(); return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
 def ref_depth(frame,size):
  image=reference_input(frame,size); inp={'image':image,'image_original':image,'validity_mask':torch.ones((1,1,*size),dtype=torch.bool,device=device)}
  feats,tokens=ref.pixel_encoder(image); inp['features']=[ref.stacking_fn(feats[a:b]).contiguous() for a,b in ref.slices_encoder_range]; inp['tokens']=[ref.stacking_fn(tokens[a:b]).contiguous() for a,b in ref.slices_encoder_range]
  ref.pixel_decoder.set_shapes(inp,[{'B':1,'T':1}]); rays,r,_,_=ref.pixel_decoder.forward_keyframe(inp,[{'B':1,'T':1}]); rays=rays.reshape(1,*size,3).permute(0,3,1,2); z=r.float()*rays[:,2:3].float()
  if not bool((torch.isfinite(z)&(z>0)).all()): return None
  return F.interpolate(z.log(),size=(434,574),mode='bilinear',align_corners=False,antialias=False).exp()[0,0]
 results=[]; configs=[(s,m,a) for s in ((224,294),(294,392)) for m in (0,1,2) for a in (.2,.5)]
 if args.pilot: configs=configs[:1]
 for size,mode,alpha in configs:
  state={}; times=[]; gpu_times=[]; flow_times=[]; statuses=[]; stage='warmup'; warm=2 if args.pilot else 50; repeats=1 if args.pilot else 3; per_rep=2 if args.pilot else 16
  def tick(idx):
   # Includes HDF5 read/decode and official Loader preprocessing. Propagation
   # retrieves a pair so Loader regenerates its native backward DIS flow.
   begin=time.perf_counter(); ids=[idx] if idx%16==0 else [idx-1,idx]
   ds.get_test_idxs=lambda n,k,ids=ids:(ids,None)
   batch=collate_fn([ds.get_single_sequence(ix)],is_batched=False); d={k:v.to(device) for k,v in batch['data'].items()}
   t=len(ids)-1; x=d['image'][0,t:t+1]; valid=d['validity_mask'][0,t,0].bool()
   shape={'network_shape':(434,574),'validity_mask':valid[None,None]}
   torch.cuda.synchronize(); a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
    if idx%16==0:
     rays,r,c,features=exp1.keyframe_state(native,{'img':x},shape); state.update(rays=rays,r=r,c=c,features=features)
    else:
     prev=d['image'][0,0:1]; p={'image':x,'validity_mask':d['validity_mask'][0,t:t+1].bool()}; native._set_image_paddings(p)
     depthpad=torch.tensor(batch['img_metas'][0]['depth_paddings'][0:1],device=device)
     flow,grid=native._normalize_and_grid_from_flow(d['flow_bwd'][0,0:1],x,p['image_paddings'],depthpad)
     residual=gt2.rgb2luma(x).unsqueeze(1)-F.grid_sample(gt2.rgb2luma(prev).unsqueeze(1),grid,mode='bilinear')
     static={'flow_norm':flow,'grid_sample':grid,'image_residual':residual,'validity_mask':valid[None,None]}
     r,c,features=exp1.one_fast_step(native,{'img':prev},{'img':x},static,state['r'],state['c'],state['features']); state.update(r=r,c=c,features=features)
   b.record(); torch.cuda.synchronize(); native_gpu=a.elapsed_time(b)
   prop=(state['r'].float()*state['rays'][:,2:3].float())[0,0]
   a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True); a.record()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): reference=ref_depth(x,size)
   b.record(); torch.cuda.synchronize(); ref_gpu=a.elapsed_time(b)
   if idx%16==0:
    state['anchor']=None if reference is None else keyframe_anchor(prop,reference,valid); chosen=prop; reason='keyframe'; aux_ms=0.
   elif reference is None:
    chosen=prop; reason='missing_reference'; aux_ms=0.
   else:
    flow_start=time.perf_counter(); dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST); prevgray=gray(d['image'][0,0:1]); currgray=gray(x)
    ff=dis.calc(prevgray,currgray,None); fb=dis.calc(currgray,prevgray,None); corr,_=correspondence_mask(ff,fb); aux_ms=(time.perf_counter()-flow_start)*1000
    result=decompose(prop,reference,state['anchor'],torch.from_numpy(corr).to(device),valid); reason=result['status']; chosen=prop.clone()
    if reason=='ok':
     vm=result['valid_depth_mask']; local=result['local_residual']; m=result['global_residual']
     if mode==0: residual=torch.zeros_like(prop); residual[vm]=reference[vm].log()-prop[vm].log()
     elif mode==1: residual=m+local
     else: residual=local
     chosen[vm]=(prop[vm].log()+alpha*residual[vm]).exp()
   # Official prediction-to-GT-canvas geometry; no GT values participate.
   pad=torch.tensor(batch['img_metas'][0]['paddings'],device=device)[:,[0,2,1,3]][t:t+1]
   aligned=match_gt(chosen[None,None],d['depth'][0,t:t+1],pad,None)[0,0].clamp(min=.01)
   torch.cuda.synchronize(); end=time.perf_counter()
   assert bool(torch.isfinite(aligned).all())
   # Outside timed boundary, verify the native network-canvas prediction is
   # the same as the already completed official 16-frame prediction pass.
   cache=Path('outputs/saoof_01_ref_probe_20261112/formal/person_tracking2')/f'frame_{idx:04d}.npz'
   with np.load(cache) as z: expected=z['native']
   native_delta=float(np.max(np.abs(prop.cpu().numpy()-expected)))
   if native_delta>1e-4: raise RuntimeError(f'online/official native mismatch frame={idx}, max_abs={native_delta}')
   return (end-begin)*1000,native_gpu+ref_gpu,aux_ms,reason
  for j in range(warm): tick(j%16)
  torch.cuda.reset_peak_memory_stats(device)
  for rep in range(repeats):
   for j in range(per_rep):
    wall,gpu,flow,reason=tick(j); times.append(wall); gpu_times.append(gpu); flow_times.append(flow); statuses.append(reason)
  item={'size':list(size),'mode':mode,'alpha':alpha,'sequence':'person_tracking2_val','indices':list(range(per_rep)),'warmup_frames':warm,'repeats':repeats,'frame_count':len(times),'end_to_end_mean_ms':statistics.mean(times),'end_to_end_p50_ms':float(np.percentile(times,50)),'end_to_end_p95_ms':float(np.percentile(times,95)),'fps':1000/statistics.mean(times),'native_plus_reference_gpu_mean_ms':statistics.mean(gpu_times),'additional_fb_cpu_mean_ms':statistics.mean(flow_times),'peak_gpu_memory_mb':torch.cuda.max_memory_allocated(device)/1024**2,'fallback_count':sum(x not in ('keyframe','ok') for x in statuses),'raw_frame_ms':times,'boundary':'HDF5 read+decode+official preprocessing; keyframe reads 1 image, propagation reads previous/current pair and generates official backward DIS; native, reference, extra bidirectional DIS, correspondence/anchor/fusion, output geometry and CUDA synchronization; excludes model load and prediction cache writes'}
  results.append(item); (root/'latency.json').write_text(json.dumps({'results':results},indent=2),encoding='utf8')
  print('finished latency',size,mode,alpha,'mean_ms',item['end_to_end_mean_ms'],flush=True)
 status.update(state='PILOT_COMPLETED' if args.pilot else 'COMPLETED',configs=len(results)); save()
except Exception as ex:
 status.update(state='ENGINEERING_BLOCKED',error=repr(ex),traceback=traceback.format_exc()); save(); raise
