"""Build isolated V1.1 Mid training/dev caches from official Bonn HDF5."""
import os,sys,glob,json,csv,time,traceback,hashlib,argparse
from pathlib import Path
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
import numpy as np,torch,cv2
import torch.nn.functional as F
sys.path.insert(0,'experiments/p1')
import gt2_fixed_clip_repair_train as gt2
from velodepth.datasets.bonn import Bonn
from velodepth.datasets import collate_fn
from saoof_reference_preprocess import reference_input
from saoof_correspondence import correspondence_mask
from saoof_v1_core import keyframe_anchor,decompose

parser=argparse.ArgumentParser(); parser.add_argument('--pilot',action='store_true'); args=parser.parse_args()
root=Path('outputs/saoof_03_gate_20261112')/('pilot' if args.pilot else 'formal'); root.mkdir(parents=True,exist_ok=True)
status={'state':'RUNNING','phase':'CACHE_BUILD'}
def save_status(): (root/'cache_status.json').write_text(json.dumps(status,indent=2),encoding='utf8')
save_status()
train_names=['crowd2','crowd3','balloon2','person_tracking','synchronous','balloon_tracking','kidnapping_box','moving_nonobstructing_box2','moving_obstructing_box2','placing_nonobstructing_box2','placing_obstructing_box','removing_nonobstructing_box2','removing_obstructing_box']
dev_names=['moving_obstructing_box','person_tracking2','removing_nonobstructing_box','moving_nonobstructing_box','placing_nonobstructing_box']
assert not set(train_names)&set(dev_names)
if args.pilot: train_names=train_names[:1]; dev_names=dev_names[:1]
try:
 cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']; old=np.array
 class Legacy(np.ndarray):
  def tostring(self,*a,**kw): return self.tobytes(*a,**kw)
 def compat(*a,**kw): return old(*a,**kw).view(Legacy)
 def dataset(split):
  try:
   np.array=compat
   d=Bonn(image_shape=cfg['image_shape'],split_file=split+'.txt',test_mode=True,crop=cfg['crop'],normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1,add_of=True)
  finally: np.array=old
  d.resizer.sample=False; d.resizer.image_shape=[434,574]; return d
 train_ds=dataset('train'); dev_ds=dataset('val')
 snap=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
 device=torch.device('cuda'); native=gt2.load_model(snap,'off',device); ref=gt2.load_model(snap,'off',device)
 native.pixel_decoder.camera_gt=False; ref.pixel_decoder.camera_gt=False
 orig=native.encode_decode_test; captured={}
 def capture(i,m):
  a,b=orig(i,m); captured['points']=b['points'].detach(); return a,b
 native.encode_decode_test=capture
 mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None]; std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
 def gray(image):
  x=((image.float()*std+mean).clamp(0,1)*255).byte()[0].permute(1,2,0).cpu().numpy(); return cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
 def ref_depth(image):
  size=(294,392); x=reference_input(image,size); inp={'image':x,'image_original':x,'validity_mask':torch.ones((1,1,*size),dtype=torch.bool,device=device)}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
   feats,toks=ref.pixel_encoder(x); inp['features']=[ref.stacking_fn(feats[a:b]).contiguous() for a,b in ref.slices_encoder_range]; inp['tokens']=[ref.stacking_fn(toks[a:b]).contiguous() for a,b in ref.slices_encoder_range]; ref.pixel_decoder.set_shapes(inp,[{'B':1,'T':1}]); rays,r,_,_=ref.pixel_decoder.forward_keyframe(inp,[{'B':1,'T':1}])
  rays=rays.reshape(1,*size,3).permute(0,3,1,2); z=r.float()*rays[:,2:3].float()
  if not bool((torch.isfinite(z)&(z>0)).all()): return None
  return F.interpolate(z.log(),size=(434,574),mode='bilinear',align_corners=False,antialias=False).exp()[0,0]
 train_stats={'flow_sum':0.,'flow_sq_sum':0.,'fb_sum':0.,'fb_sq_sum':0.,'pixels':0}; manifest=[]
 for role,names,ds,split in [('train',train_names,train_ds,'train'),('dev',dev_names,dev_ds,'val')]:
  for name in names:
   ix=next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/'+name+'_'+split)); n=len(ds.sequences[ix]['image']); assert n==(80 if role=='train' else 110)
   if args.pilot: n=16
   target=root/'cache'/role/name; target.mkdir(parents=True,exist_ok=True); t0=time.time()
   for start in range(0,n,16):
    ids=list(range(start,min(start+16,n))); ds.get_test_idxs=lambda count,k,ids=ids:(ids,None)
    batch=collate_fn([ds.get_single_sequence(ix)],is_batched=False); d={k:v.to(device) for k,v in batch['data'].items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): prediction,_=native.forward_test(d,batch['img_metas'])
    prop=captured['points'].float()[:,2]; pad=torch.tensor(batch['img_metas'][0]['paddings'])[:,[0,2,1,3]].numpy(); prev=None; anchor=None; dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    for j,idx in enumerate(ids):
     age=idx%16; image=d['image'][0,j:j+1]; geom=d['validity_mask'][0,j,0].bool(); g=gray(image)
     gx=cv2.Sobel(g.astype(np.float32)/255.,cv2.CV_32F,1,0,ksize=3,scale=.125); gy=cv2.Sobel(g.astype(np.float32)/255.,cv2.CV_32F,0,1,ksize=3,scale=.125); grad=np.sqrt(gx*gx+gy*gy)
     reference=ref_depth(image); assert reference is not None
     if age==0:
      anchor=keyframe_anchor(prop[j],reference,geom); assert anchor is not None
      corr=np.zeros(g.shape,dtype=bool); flow_corr=corr.copy(); flow=np.zeros(g.shape,np.float32); err=np.zeros(g.shape,np.float32); local=np.zeros(g.shape,np.float32); m=0.; valid=geom.cpu().numpy(); st='keyframe'
     else:
      ff=dis.calc(prev,g,None); fb=dis.calc(g,prev,None); corr,err=correspondence_mask(ff,fb); flow_corr=corr.copy(); flow=np.linalg.norm(fb,axis=2).astype(np.float32); result=decompose(prop[j],reference,anchor,torch.from_numpy(corr).to(device),geom); st=result['status']; valid=result['valid_depth_mask'].cpu().numpy()
      if st=='ok': local=result['local_residual'].cpu().numpy(); m=float(result['global_residual'])
      else: local=np.zeros(g.shape,np.float32); m=0.
      err=np.where(np.isfinite(err),err,0).astype(np.float32); corr=result['correspondence_mask'].cpu().numpy()
     gt=d['depth'][0,j,0].float().cpu().numpy(); gtmask=d['depth_mask'][0,j,0].bool().cpu().numpy()
     zprop=prop[j].cpu().numpy(); zref=reference.cpu().numpy()
     if role=='dev':
      prior=Path('outputs/saoof_01_ref_probe_20261112/formal')/name/f'frame_{idx:04d}.npz'
      with np.load(prior) as z:
       if np.max(np.abs(z['native']-zprop))>1e-4 or np.max(np.abs(z['mid_reference']-zref))>1e-4: raise RuntimeError(f'V1.1 dev prediction mismatch {name} {idx}')
       if not np.array_equal(z['gt_mask'],gtmask & (gt<=ds.max_depth)): raise RuntimeError(f'GT mask mismatch {name} {idx}')
     if role=='train' and age>0 and st=='ok':
      mask=flow_corr & geom.cpu().numpy() & np.isfinite(zprop) & (zprop>0) & np.isfinite(flow) & np.isfinite(err); f=flow[mask].astype(np.float64); q=err[mask].astype(np.float64)
      train_stats['flow_sum']+=float(f.sum()); train_stats['flow_sq_sum']+=float((f*f).sum()); train_stats['fb_sum']+=float(q.sum()); train_stats['fb_sq_sum']+=float((q*q).sum()); train_stats['pixels']+=len(f)
     np.savez_compressed(target/f'frame_{idx:04d}.npz',native=zprop,reference=zref,anchor=np.float32(float(anchor)),global_residual=np.float32(m),local=local,flow=flow,fb_error=err,rgb_grad=grad,corr_mask=corr,flow_corr_mask=flow_corr,geometry_mask=geom.cpu().numpy(),valid_depth=valid,gt=gt,gt_mask=gtmask,padding=pad[j],age=np.int16(age),status=np.array(st))
     manifest.append({'role':role,'sequence':name,'index':idx,'age':age,'status':st,'path':str((target/f'frame_{idx:04d}.npz').resolve())})
     prev=g; print(role,name,idx,st,flush=True)
   print('completed',role,name,'frames',n,'seconds',time.time()-t0,flush=True)
   with (root/'cache_manifest.partial.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
 N=train_stats['pixels']; assert N>0
 stats={'source':'Bonn13 train only; valid correspondence pixels on propagation frames','pixels':N}
 for name in ('flow','fb'):
  mu=train_stats[name+'_sum']/N; var=max(train_stats[name+'_sq_sum']/N-mu*mu,0); stats[name]={'mean':mu,'std':max(var**.5,1e-6)}
 (root/'normalization.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
 with (root/'cache_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(manifest[0])); w.writeheader(); w.writerows(manifest)
 status.update(state='COMPLETED',phase='CACHE_BUILD',train_frames=sum(x['role']=='train' for x in manifest),dev_frames=sum(x['role']=='dev' for x in manifest),fallbacks=sum(x['status'] not in ('keyframe','ok') for x in manifest)); save_status()
except Exception as ex:
 status.update(state='ENGINEERING_BLOCKED',error=repr(ex),traceback=traceback.format_exc()); save_status(); raise
