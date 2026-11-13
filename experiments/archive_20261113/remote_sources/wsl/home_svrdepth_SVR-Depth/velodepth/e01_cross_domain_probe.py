import os,sys,json,glob,traceback
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
root=Path('outputs/saoof_main_20261113');root.mkdir(parents=True,exist_ok=True)
stats=json.loads(Path('outputs/saoof_03_gate_20261112/formal/normalization.json').read_text()); cfg=json.loads(Path('configs/velodepth-eval.json').read_text())['data']
class Gate(nn.Module):
 def __init__(self):
  super().__init__(); self.trunk=nn.Sequential(nn.Conv2d(6,16,3,padding=1),nn.GroupNorm(4,16),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.GroupNorm(4,16),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.GroupNorm(4,16),nn.ReLU()); self.spatial=nn.Conv2d(16,1,1);self.global_head=nn.Linear(16,1)
 def forward(self,x):
  h=self.trunk(x);return self.global_head(h.mean((2,3)))[:,0,None,None]+self.spatial(h)[:,0]
device='cuda'; model=Gate().to(device).eval(); ck=torch.load('outputs/saoof_03_gate_20261112/formal/training/fair_seed20261112/best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['state_dict'])
snapshot=Path(glob.glob('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/*/config.json')[0]).parent
native=gt2.load_model(snapshot,'off',torch.device(device));native.pixel_decoder.camera_gt=False; orig=native.encode_decode_test; captured={}
def capture(i,m):
 a,b=orig(i,m);captured['points']=b['points'].detach();return a,b
native.encode_decode_test=capture
old=np.array
class Legacy(np.ndarray):
 def tostring(self,*a,**kw):return self.tobytes(*a,**kw)
def compat(*a,**kw):return old(*a,**kw).view(Legacy)
mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
results=[]
try:
 for cls in [Bonn,TUM,ScanNetVid]:
  try:
   np.array=compat; ds=cls(image_shape=cfg['image_shape'],split_file='val.txt',test_mode=True,crop=cfg['crop'],normalize=cfg['normalization'],augmentations_db=cfg['augmentations'],shape_constraints=cfg['shape_constraints'],resize_method=cfg['resize_method'],mini=1.,num_frames=-1,add_of=True)
  finally:np.array=old
  ds.resizer.sample=False;ds.resizer.image_shape=[434,574];ix=0;n=min(16,len(ds.sequences[ix]['image'])); assert n==16
  ds.get_test_idxs=lambda count,k:(list(range(16)),None)
  batch=collate_fn([ds.get_single_sequence(ix)],is_batched=False);d={k:v.to(device) for k,v in batch['data'].items()}
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): pred,_=native.forward_test(d,batch['img_metas'])
  prop=captured['points'].float()[:,2];pad=torch.tensor(batch['img_metas'][0]['paddings'],device=device)[:,[0,2,1,3]];prev=None;dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST); diffs=[]
  for j in range(16):
   image=d['image'][0,j:j+1];geom=d['validity_mask'][0,j,0].bool().cpu().numpy(); rgb=((image.float()*std+mean).clamp(0,1)*255).byte()[0].permute(1,2,0).cpu().numpy();g=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);p=prop[j].cpu().numpy()
   if j:
    ff=dis.calc(prev,g,None);fb=dis.calc(g,prev,None);raw,err=correspondence_mask(ff,fb);raw=raw&geom&np.isfinite(p)&(p>0);flow=np.linalg.norm(fb,axis=2);err=np.where(np.isfinite(err),err,0);fn=np.where(raw,(flow-stats['flow']['mean'])/stats['flow']['std'],0);en=np.where(raw,(err-stats['fb']['mean'])/stats['fb']['std'],0)
    gx=cv2.Sobel(g.astype(np.float32)/255,cv2.CV_32F,1,0,ksize=3,scale=.125);gy=cv2.Sobel(g.astype(np.float32)/255,cv2.CV_32F,0,1,ksize=3,scale=.125);grad=np.sqrt(gx*gx+gy*gy);base=np.log(np.maximum(p,1e-6));x=np.stack((base,fn,en,grad,np.full_like(p,j/15),raw.astype('float32'))).astype('float32')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):delta=model(torch.from_numpy(x)[None].to(device))
    assert bool(torch.isfinite(delta).all());z=(torch.from_numpy(base).to(device)[None]+delta.float()).exp()
   else:z=prop[j][None]
   gt=d['depth'][0,j:j+1];aligned=match_gt(z[:,None],gt,pad[j:j+1],None).clamp_min(.01);original=pred['depth'][0,j:j+1].float(); diff=float((match_gt(prop[j:j+1,None],gt,pad[j:j+1],None).clamp_min(.01)-original).abs().max());diffs.append(diff);assert diff<1e-4
   assert bool(torch.isfinite(aligned).all());prev=g
  with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): control,_=native.forward_test(d,batch['img_metas'])
  equal=bool(torch.equal(control['depth'],pred['depth'])); assert equal
  results.append({'dataset':cls.__name__,'status':'PASS','sequence':str(ds.dataset[ix]['sequence_name']),'frames':16,'native_vs_official_max_abs':max(diffs),'rerun_native_depth_equal':equal,'note':'B3 output-only after full native clip; no in-cycle interleaving or exhaustive internal-state audit'})
  print(results[-1],flush=True)
 (root/'e01_probe.json').write_text(json.dumps(results,indent=2),encoding='utf8')
except Exception as e:
 (root/'e01_probe.json').write_text(json.dumps({'completed':results,'error':repr(e),'traceback':traceback.format_exc()},indent=2),encoding='utf8');raise
