"""Frozen output-only B3 vs native on official HDF5, equal frame identities."""
import os,sys,json,glob,csv,time,traceback,argparse,hashlib
from pathlib import Path
os.environ['DATAROOT']='/home/svrdepth/datasets/velodepth_evaluation'
import numpy as np,torch,cv2
from torch import nn
sys.path[:0]=['experiments/p1','outputs/saoof_00_closeout_20261112']
import gt2_fixed_clip_repair_train as gt2
from velodepth.datasets.bonn import Bonn
from velodepth.datasets.tum import TUM
from velodepth.datasets.scannet import ScanNetVid
from velodepth.datasets import collate_fn
from velodepth.utils.misc import match_gt
from saoof_correspondence import correspondence_mask
ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--bonn',action='store_true');args=ap.parse_args()
root=Path('outputs/saoof_main_20261113')/('temporal_pilot' if args.pilot else 'temporal_bonn' if args.bonn else 'temporal_formal');root.mkdir(parents=True,exist_ok=True)
status={'state':'RUNNING','pilot':args.pilot};(root/'status.json').write_text(json.dumps(status,indent=2))
stats=json.loads(Path('outputs/saoof_03_gate_20261112/formal/normalization.json').read_text()); cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']
class Gate(nn.Module):
 def __init__(self):
  super().__init__();self.trunk=nn.Sequential(nn.Conv2d(6,16,3,padding=1),nn.GroupNorm(4,16),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.GroupNorm(4,16),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.GroupNorm(4,16),nn.ReLU());self.spatial=nn.Conv2d(16,1,1);self.global_head=nn.Linear(16,1)
 def forward(self,x):
  h=self.trunk(x);return self.global_head(h.mean((2,3)))[:,0,None,None]+self.spatial(h)[:,0]
device='cuda';seeds=[20261112,20261113,20261114];models={}
for seed in seeds:
 m=Gate().to(device).eval();ck=torch.load(f'outputs/saoof_03_gate_20261112/formal/training/fair_seed{seed}/best.pt',map_location=device,weights_only=False);m.load_state_dict(ck['state_dict']);models[seed]=m
snapshot=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
native=gt2.load_model(snapshot,'off',torch.device(device));native.pixel_decoder.camera_gt=False;orig=native.encode_decode_test;captured={}
def capture(i,m):
 a,b=orig(i,m);captured['points']=b['points'].detach();return a,b
native.encode_decode_test=capture
old=np.array
class Legacy(np.ndarray):
 def tostring(self,*a,**kw):return self.tobytes(*a,**kw)
def compat(*a,**kw):return old(*a,**kw).view(Legacy)
mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
def score(p,gt,mask):
 if not bool(mask.any()):raise RuntimeError('empty official GT mask')
 if not bool((torch.isfinite(p[mask])&(p[mask]>0)).all()):raise RuntimeError('invalid prediction on GT mask')
 a=p[mask];b=gt[mask];e=a-b
 return float((e.abs()/b).mean()),float(e.square().mean().sqrt()),float((torch.maximum(a/b,b/a)<1.25).float().mean())
def gray(x):
 rgb=((x.float()*std+mean).clamp(0,1)*255).byte()[0].permute(1,2,0).cpu().numpy();return cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
fields=['dataset','sequence','index','age','method','absrel','rmse','delta1','valid_pixels']
rows=[];summaries=[];temporal=[]
try:
 for cls in ([Bonn,TUM,ScanNetVid] if args.pilot else [Bonn] if args.bonn else [TUM,ScanNetVid]):
  try:
   np.array=compat;ds=cls(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,crop=cfg['crop'],normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1,add_of=True)
  finally:np.array=old
  ds.resizer.sample=False;ds.resizer.image_shape=[434,574]
  bonn_names=['moving_obstructing_box','person_tracking2','removing_nonobstructing_box','moving_nonobstructing_box','placing_nonobstructing_box']
  ids=([next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/moving_obstructing_box_val'))] if args.pilot and cls is Bonn else range(1)) if args.pilot else [next(i for i,x in enumerate(ds.dataset) if x['sequence_name'].endswith('/'+name+'_val')) for name in bonn_names] if args.bonn else range(len(ds.dataset))
  for ix in ids:
   name=str(ds.dataset[ix]['sequence_name']);n=len(ds.sequences[ix]['image']);n=min(n,16) if args.pilot else n; assert n>0
   seqrows=[];begin=time.time();dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
   for start in range(0,n,16):
    chosen=list(range(start,min(start+16,n)));ds.get_test_idxs=lambda count,k,chosen=chosen:(chosen,None)
    batch=collate_fn([ds.get_single_sequence(ix)],is_batched=False);d={k:v.to(device) for k,v in batch['data'].items()}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):pred,_=native.forward_test(d,batch['img_metas'])
    prop=captured['points'].float()[:,2];pad=torch.tensor(batch['img_metas'][0]['paddings'],device=device)[:,[0,2,1,3]];prev=None;prev_canvas=None;prev_geom=None
    for j,idx in enumerate(chosen):
     image=d['image'][0,j:j+1];geom=d['validity_mask'][0,j,0].bool().cpu().numpy();g=gray(image);p=prop[j].cpu().numpy();gt=d['depth'][0,j,0].float(); mask=d['depth_mask'][0,j,0].bool()&torch.isfinite(gt)&(gt>0)&(gt<=ds.max_depth)
     native_depth=pred['depth'][0,j,0].float(); check=match_gt(prop[j:j+1,None],gt[None,None],pad[j:j+1],None)[0,0].clamp_min(.01)
     if not torch.allclose(check,native_depth,atol=1e-4,rtol=1e-5):raise RuntimeError(f'native mismatch {name}:{idx}')
     outputs={'native':native_depth}; canvases={'native':p}
     if idx%16:
      ff=dis.calc(prev,g,None);fb=dis.calc(g,prev,None);raw,err=correspondence_mask(ff,fb);raw=raw&geom&np.isfinite(p)&(p>0);flow=np.linalg.norm(fb,axis=2);err=np.where(np.isfinite(err),err,0);fn=np.where(raw,(flow-stats['flow']['mean'])/stats['flow']['std'],0);en=np.where(raw,(err-stats['fb']['mean'])/stats['fb']['std'],0)
      gx=cv2.Sobel(g.astype(np.float32)/255,cv2.CV_32F,1,0,ksize=3,scale=.125);gy=cv2.Sobel(g.astype(np.float32)/255,cv2.CV_32F,0,1,ksize=3,scale=.125);base=np.log(np.maximum(p,1e-6));x=np.stack((base,fn,en,np.sqrt(gx*gx+gy*gy),np.full_like(p,(idx%16)/15),raw.astype('float32'))).astype('float32');xt=torch.from_numpy(x)[None].to(device);bt=torch.from_numpy(base).to(device)[None]
      for seed,m in models.items():
       with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):delta=m(xt)
       z=(bt+delta.float()).exp();canvases[str(seed)]=z[0].cpu().numpy();outputs[str(seed)]=match_gt(z[:,None],gt[None,None],pad[j:j+1],None)[0,0].clamp_min(.01)
     else:
      for seed in seeds:outputs[str(seed)]=native_depth;canvases[str(seed)]=p
     for method,value in outputs.items():
      a,r,d1=score(value,gt,mask)
      if args.pilot and cls is Bonn and method!='native':
       previous=Path(f'outputs/saoof_03_gate_20261112/formal/training/fair_seed{method}/metrics_per_frame.csv')
       with previous.open(encoding='utf-8-sig',newline='') as f: saved=next(z for z in csv.DictReader(f) if z['sequence']=='moving_obstructing_box' and int(z['index'])==idx)
       if abs(a-float(saved['absrel']))>1e-5:raise RuntimeError(f'Bonn B3 score mismatch {method} {idx}: {a} vs {saved["absrel"]}')
      row={'dataset':cls.__name__,'sequence':name,'index':idx,'age':idx%16,'method':method,'absrel':a,'rmse':r,'delta1':d1,'valid_pixels':int(mask.sum())};seqrows.append(row);rows.append(row)
     if idx%16:
      h,w=g.shape;xx,yy=np.meshgrid(np.arange(w,dtype=np.float32),np.arange(h,dtype=np.float32));mx=xx+fb[:,:,0];my=yy+fb[:,:,1]
      mapped_geom=cv2.remap(prev_geom.astype('float32'),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>.999
      shared=raw & mapped_geom
      for val in canvases.values(): shared &= np.isfinite(val)&(val>0)
      for prior in prev_canvas.values():
       prior_valid=(np.isfinite(prior)&(prior>0)).astype('float32')
       shared &= cv2.remap(prior_valid,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)>.999
      if not shared.any(): raise RuntimeError(f'empty temporal mask {name}:{idx}')
      for method,val in canvases.items():
       prior=prev_canvas[method]; safe=np.where(np.isfinite(prior)&(prior>0),prior,1);warped=cv2.remap(np.log(safe).astype('float32'),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
       temporal.append({'dataset':cls.__name__,'sequence':name,'index':idx,'age':idx%16,'method':method,'warped_log_depth_error':float(np.abs(np.log(val[shared])-warped[shared]).mean()),'valid_pixels':int(shared.sum()),'canvas_pixels':int(shared.size)})
     prev=g;prev_canvas={k:v.copy() for k,v in canvases.items()};prev_geom=geom.copy()
   with (root/'temporal_per_frame.partial.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(temporal[0]));w.writeheader();w.writerows(temporal)
   for method in ['native']+list(map(str,seeds)):
    rr=[x for x in seqrows if x['method']==method];summaries.append({'dataset':cls.__name__,'sequence':name,'method':method,'frames':len(rr),**{k:float(np.mean([r[k] for r in rr])) for k in ('absrel','rmse','delta1')}})
   for file,data in [('metrics_per_frame.partial.csv',rows),('metrics_per_sequence.partial.csv',summaries)]:
    with (root/file).open('w',encoding='utf-8-sig',newline='') as f:
     w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
   print(cls.__name__,ix+1,len(ds.dataset),name,n,'seconds',round(time.time()-begin,1),flush=True)
   status.update(dataset=cls.__name__,completed_sequences=ix+1,total_sequences=len(ds.dataset),frames_written=len(rows));(root/'status.json').write_text(json.dumps(status,indent=2))
 for stem in ['metrics_per_frame','metrics_per_sequence']:(root/(stem+'.partial.csv')).replace(root/(stem+'.csv'))
 (root/'temporal_per_frame.partial.csv').replace(root/'temporal_per_frame.csv')
 status['state']='COMPLETED';(root/'status.json').write_text(json.dumps(status,indent=2))
except Exception as e:
 status.update(state='ENGINEERING_BLOCKED',error=repr(e),traceback=traceback.format_exc());(root/'status.json').write_text(json.dumps(status,indent=2));raise
