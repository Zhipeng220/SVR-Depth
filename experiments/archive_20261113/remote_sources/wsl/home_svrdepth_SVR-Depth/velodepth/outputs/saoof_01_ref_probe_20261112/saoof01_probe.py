"""V1.1 Bonn5 prediction probe, official loader, fixed K16, no training.

Pilot is default. --formal runs all 5x110 frames. End-to-end timing is not
inferred from offline component timing and requires a separate online pass.
"""
import os,sys,glob,json,csv,time,argparse,traceback,hashlib,subprocess
from pathlib import Path
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
import numpy as np,torch,cv2
import torch.nn.functional as F
sys.path.insert(0,'experiments/p1')
import gt2_fixed_clip_repair_train as gt2
from velodepth.datasets.bonn import Bonn
from velodepth.datasets import collate_fn
from velodepth.utils.misc import match_gt
from saoof_reference_preprocess import reference_input
from saoof_v1_core import keyframe_anchor,decompose
from saoof_correspondence import correspondence_mask

parser=argparse.ArgumentParser(); parser.add_argument('--formal',action='store_true'); args=parser.parse_args()
root=Path('outputs/saoof_01_ref_probe_20261112') / ('formal' if args.formal else 'pilot'); root.mkdir(parents=True,exist_ok=True)
status={'state':'RUNNING','protocol':'V1.1','formal':args.formal,'elapsed_start':time.time()}
def write_status(): (root/'status.json').write_text(json.dumps(status,indent=2),encoding='utf8')
write_status()
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
 names=['moving_obstructing_box','person_tracking2','removing_nonobstructing_box','moving_nonobstructing_box','placing_nonobstructing_box']
 if not args.formal: names=['person_tracking2']
 indices={n:next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/'+n+'_val')) for n in names}
 snap=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
 def sha256(path):
  h=hashlib.sha256()
  with open(path,'rb') as f:
   for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
  return h.hexdigest()
 config={'protocol':'V1.1','dataset':'Bonn official HDF5 same-index benchmark','sequences':names,'native_size':[434,574],'reference_sizes':[[224,294],[294,392]],'camera_gt':False,'K':16,'reference_resize':{'mode':'bilinear','align_corners':False,'antialias':False},'gt_depth_range':[ds.min_depth,ds.max_depth],'gt_mask':'official Bonn Loader depth_mask','fixed_alphas':[.2,.5],'aggregation':'per-frame metrics averaged per sequence; equal-sequence macro after completion','complement_population':'propagation frames; official GT mask; fallback counted as not better','flow':'DIS bidirectional, current-frame inverse bilinear sample, align_corners=False','timing_note':'offline component timing is diagnostic, not end-to-end'}
 (root/'config.yaml').write_text(json.dumps(config,indent=2),encoding='utf8')
 identity={'snapshot':str(snap),'weights_sha256':sha256(snap/'model.safetensors'),'bonn_hdf5_sha256':sha256('/home/svrdepth/datasets/velodepth_evaluation/Bonn.hdf5'),'code_sha256':sha256(__file__),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'note':'git worktree contains untracked files; implementation script has independent sha256'}
 (root/'run_identity.json').write_text(json.dumps(identity,indent=2),encoding='utf8')
 device=torch.device('cuda'); native=gt2.load_model(snap,'off',device); ref=gt2.load_model(snap,'off',device)
 native.pixel_decoder.camera_gt=False; ref.pixel_decoder.camera_gt=False
 assert native.pixel_decoder is not ref.pixel_decoder
 orig=native.encode_decode_test; captured={}
 def capture(inp,metas):
  x,y=orig(inp,metas); captured['points']=y['points'].detach(); return x,y
 native.encode_decode_test=capture
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]; std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def ref_depth(frame,size):
  image=reference_input(frame,size)
  inp={'image':image,'image_original':image,'validity_mask':torch.ones((1,1,*size),dtype=torch.bool,device=device)}
  feats,tokens=ref.pixel_encoder(image)
  inp['features']=[ref.stacking_fn(feats[a:b]).contiguous() for a,b in ref.slices_encoder_range]
  inp['tokens']=[ref.stacking_fn(tokens[a:b]).contiguous() for a,b in ref.slices_encoder_range]
  ref.pixel_decoder.set_shapes(inp,[{'B':1,'T':1}]); rays,r,_,_=ref.pixel_decoder.forward_keyframe(inp,[{'B':1,'T':1}])
  rays=rays.reshape(1,*size,3).permute(0,3,1,2); z=r.float()*rays[:,2:3].float()
  if not bool((torch.isfinite(z)&(z>0)).all()): return None
  return F.interpolate(z.log(),size=(434,574),mode='bilinear',align_corners=False,antialias=False).exp()[0,0]
 def gray(x):
  a=((x.float()*std+mean).clamp(0,1)*255).byte()[0].permute(1,2,0).cpu().numpy()
  return cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
 def score(pred,gt,mask):
  v=mask & torch.isfinite(gt)&(gt>0)&torch.isfinite(pred)&(pred>0)
  if not bool(v.any()): raise RuntimeError('empty valid evaluation mask')
  # Do not silently remove invalid predictions from official GT evaluation.
  if bool((mask & ~(torch.isfinite(pred)&(pred>0))).any()): raise RuntimeError('invalid prediction on official GT mask')
  g=gt[v]; p=pred[v]; return {'absrel':float(((p-g).abs()/g).mean()),'rmse':float(((p-g)**2).mean().sqrt()),'delta1':float((torch.maximum(p/g,g/p)<1.25).float().mean()),'valid_pixels':int(v.sum())}
 rows=[]; seqsumm=[]
 for name in names:
  t0=time.time(); seqdir=root/name; seqdir.mkdir(exist_ok=True)
  n=len(ds.sequences[indices[name]]['image']); assert n==110
  if not args.formal: n=2
  for start in range(0,n,16):
   ids=list(range(start,min(start+16,n))); ds.get_test_idxs=lambda count,k,ids=ids:(ids,None)
   item=ds.get_single_sequence(indices[name]); batch=collate_fn([item],is_batched=False)
   batch['data']={k:v.to(device) for k,v in batch['data'].items()}; d=batch['data']; T=len(ids)
   assert tuple(d['image'].shape[-2:])==(434,574)
   torch.cuda.synchronize(); native_start=time.perf_counter()
   with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): pred,_=native.forward_test(d,batch['img_metas'])
   torch.cuda.synchronize(); native_ms=(time.perf_counter()-native_start)*1000
   points=captured['points'].float(); prop=points[:,2]; original_pred=pred['depth'][0,:,0].float()
   pads=torch.tensor(batch['img_metas'][0]['paddings'],device=device)[:,[0,2,1,3]]
   masks=d['depth_mask'][0,:,0].bool(); gts=d['depth'][0,:,0].float()
   def align(z,j,clip_depth=True):
    result=match_gt(z[None,None],gts[j][None,None],pads[j:j+1],None)[0,0].float()
    return result.clamp(min=.01) if clip_depth else result
   # Hard numerical equivalence between captured native z and official returned depth.
   for j in range(T):
    if not torch.allclose(align(prop[j],j),original_pred[j],rtol=1e-5,atol=1e-4): raise RuntimeError(f'native GT alignment mismatch {name} {ids[j]}')
   anchor={}; prev=None; dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
   for j,idx in enumerate(ids):
    age=idx%16; image=d['image'][0,j:j+1]; geom=d['validity_mask'][0,j,0].bool(); gt=gts[j]; gtmask=masks[j] & (gt<=ds.max_depth)
    current=gray(image); ff=fb=None; flowmag=None; corr=torch.zeros_like(geom)
    if prev is not None:
     ff=dis.calc(prev,current,None); fb=dis.calc(current,prev,None)
     cm,_=correspondence_mask(ff,fb); corr=torch.from_numpy(cm).to(device)
     flowmag=torch.from_numpy(np.linalg.norm(fb,axis=2)).to(device)
    row={'sequence':name,'index':idx,'age':age,'valid_gt':int(gtmask.sum()),'native_chunk_ms_per_frame_diagnostic':native_ms/T}
    valid_gt=score(original_pred[j],gt,gtmask)
    for k,v in valid_gt.items(): row['native_'+k]=v
    cached={'native':prop[j].cpu().numpy(),'gt':gt.cpu().numpy(),'gt_mask':gtmask.cpu().numpy(),'geometry_mask':geom.cpu().numpy(),'corr_mask':corr.cpu().numpy()}
    for label,size in [('low',(224,294)),('mid',(294,392))]:
     torch.cuda.synchronize(); rs=time.perf_counter()
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): reference=ref_depth(image,size)
     torch.cuda.synchronize(); row[label+'_reference_gpu_call_ms_diagnostic']=(time.perf_counter()-rs)*1000
     if reference is None: raise RuntimeError(f'nonpositive reference {name} {idx} {label}')
     if age==0: anchor[label]=keyframe_anchor(prop[j],reference,geom)
     a=anchor[label]; cached[label+'_reference']=reference.cpu().numpy(); cached[label+'_b0']=np.array(float(a) if a is not None else np.nan)
     if age==0:
      # Keyframes are native output; reference-only is still measured separately.
      result=None; reasons='keyframe'; cmask=torch.zeros_like(geom); m=None; local=None; valid=geom & torch.isfinite(reference)&(reference>0)&torch.isfinite(prop[j])&(prop[j]>0)
     else:
      result=decompose(prop[j],reference,a,corr,geom); reasons=result['status']; cmask=result['correspondence_mask']; m=result['global_residual']; local=result['local_residual']; valid=result['valid_depth_mask']
     row[label+'_status']=reasons; row[label+'_corr_pixels']=int(cmask.sum()); row[label+'_anchor']=float(a) if a is not None else ''
     row[label+'_global_residual']=float(m) if m is not None else ''
     cached[label+'_local']=local.cpu().numpy() if local is not None else np.empty((0,),np.float32)
     # Reference-only is raw reference on valid pixels; invalid reference pixels fall back.
     raw=torch.where(valid,reference,prop[j]); raw_gt=align(raw,j)
     for k,v in score(raw_gt,gt,gtmask).items(): row[label+'_ref_'+k]=v
     for mode in (0,1,2):
      can=(age>0 and reasons=='ok')
      residual=torch.zeros_like(prop[j])
      if can:
       if mode==0: residual[valid]=reference[valid].log()-prop[j][valid].log()
       elif mode==1: residual[valid]=m+local[valid]
       else: residual[valid]=local[valid]
      mode_depth=prop[j].clone()
      if can: mode_depth[valid]=(prop[j][valid].log()+residual[valid]).exp()
      mode_gt=align(mode_depth,j)
      for k,v in score(mode_gt,gt,gtmask).items(): row[f'{label}_mode{mode}_'+k]=v
      # Gate-1 pixels are computed only for propagation frames, with fallback counted as not better.
      if age>0:
       better=((mode_gt-gt).abs()/gt.clamp_min(1e-12) < (original_pred[j]-gt).abs()/gt.clamp_min(1e-12)) & gtmask
       row[f'{label}_mode{mode}_better_pixels']=int(better.sum())
       cached[f'{label}_mode{mode}_better_gt']=np.packbits(better.cpu().numpy().reshape(-1))
      for alpha in (.2,.5):
       fixed=prop[j].clone()
       if can: fixed[valid]=(prop[j][valid].log()+alpha*residual[valid]).exp()
       for k,v in score(align(fixed,j),gt,gtmask).items(): row[f'{label}_mode{mode}_a{alpha}_'+k]=v
    if flowmag is not None: cached['flow_magnitude_gt']=align(flowmag,j,clip_depth=False).cpu().numpy()
    else: cached['flow_magnitude_gt']=np.empty((0,),np.float32)
    # No reference, mask or corrected prediction is fed back into native forward_test.
    np.savez_compressed(seqdir/f'frame_{idx:04d}.npz',**cached)
    rows.append(row); prev=current
    print('frame',name,idx,'of',n,flush=True)
  seqrows=[r for r in rows if r['sequence']==name]; prop_rows=[r for r in seqrows if r['age']>0]
  flow_values=[]
  for r in prop_rows:
   with np.load(seqdir/f"frame_{r['index']:04d}.npz") as z: flow_values.append(z['flow_magnitude_gt'][z['gt_mask']])
  q75=float(np.percentile(np.concatenate(flow_values),75)) if flow_values else float('nan')
  sums={}
  for r in prop_rows:
   with np.load(seqdir/f"frame_{r['index']:04d}.npz") as z:
    gm=z['gt_mask']; dynamic=gm & (z['flow_magnitude_gt']>=q75); total=int(gm.sum()); dyn=int(dynamic.sum())
    for label in ('low','mid'):
     for mode in (0,1,2):
      key=f'{label}_mode{mode}'; better=np.unpackbits(z[key+'_better_gt'],count=gm.size).reshape(gm.shape).astype(bool)
      s=sums.setdefault(key,{'better':0,'total':0,'dynamic_better':0,'dynamic_total':0}); s['better']+=int((better&gm).sum()); s['total']+=total; s['dynamic_better']+=int((better&dynamic).sum()); s['dynamic_total']+=dyn
  summary={'sequence':name,'frames':len(seqrows),'propagation_frames':len(prop_rows),'flow_q75':q75,'elapsed_seconds':time.time()-t0}
  for key in [x for x in seqrows[0] if x.endswith(('_absrel','_rmse','_delta1'))]: summary[key]=float(np.mean([r[key] for r in seqrows]))
  for key,v in sums.items(): summary[key+'_C']=v['better']/v['total']; summary[key+'_Cdynamic']=v['dynamic_better']/v['dynamic_total'] if v['dynamic_total'] else None
  seqsumm.append(summary); (seqdir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
  with (root/'metrics_per_frame.partial.csv').open('w',newline='',encoding='utf-8-sig') as f:
   keys=list(dict.fromkeys(k for row in rows for k in row)); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
  print('completed sequence',name,flush=True)
 with (root/'metrics_per_frame.csv').open('w',newline='',encoding='utf-8-sig') as f:
  keys=list(dict.fromkeys(k for row in rows for k in row)); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
 (root/'metrics_per_sequence.json').write_text(json.dumps(seqsumm,indent=2),encoding='utf8')
 status.update(state='COMPLETED_PREDICTION_ONLY' if args.formal else 'PILOT_COMPLETED',frames=len(rows),sequences=len(seqsumm),limitations='End-to-end per-configuration latency, P50/P95, peak memory and Gate-1 decision not yet measured.'); write_status()
except Exception as ex:
 status.update(state='ENGINEERING_BLOCKED',error=repr(ex),traceback=traceback.format_exc()); write_status(); raise
