"""Frozen SAOOF-03 training: cached Bonn13, Bonn5 early stopping, three seeds."""
import argparse,csv,json,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from velodepth.utils.misc import match_gt

parser=argparse.ArgumentParser()
parser.add_argument('--pilot',action='store_true')
args=parser.parse_args()
root=Path('outputs/saoof_03_gate_20261112')/('pilot' if args.pilot else 'formal')
out=root/'training'; out.mkdir(parents=True,exist_ok=True)
status={'state':'RUNNING','phase':'TRAINING'}
def save_status(): (out/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
save_status()
try:
 with (root/'cache_manifest.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 stats=json.loads((root/'normalization.json').read_text())
 train=[x for x in rows if x['role']=='train' and int(x['age'])>0 and x['status']=='ok']
 dev=[x for x in rows if x['role']=='dev']
 assert train and dev
 counts={s:sum(x['sequence']==s for x in train) for s in sorted({x['sequence'] for x in train})}
 class Cached(Dataset):
  def __init__(self,rs): self.rows=rs
  def __len__(self): return len(self.rows)
  def __getitem__(self,i):
   row=self.rows[i]
   with np.load(row['path']) as z:
    p=z['native'].astype('float32'); v=z['valid_depth'].astype(bool); raw=z['flow_corr_mask'].astype(bool)&z['geometry_mask'].astype(bool)&np.isfinite(p)&(p>0)
    mask=z['corr_mask'].astype(bool)&v
    flow=np.where(raw,(z['flow']-stats['flow']['mean'])/stats['flow']['std'],0).astype('float32')
    fb=np.where(raw,(z['fb_error']-stats['fb']['mean'])/stats['fb']['std'],0).astype('float32')
    age=np.full_like(p,float(z['age'])/15); m=np.full_like(p,float(z['global_residual']))
    grad=z['rgb_grad'].astype('float32'); local=z['local'].astype('float32')
    base=np.log(np.maximum(p,1e-6)).astype('float32')
    dual=np.stack((local,m,flow,fb,grad,age,mask.astype('float32')))
    fair=np.stack((base,flow,fb,grad,age,raw.astype('float32')))
    gt=z['gt'].astype('float32'); gtmask=z['gt_mask'].astype(bool)&np.isfinite(gt)&(gt>0)&(gt<=10)
    return {'dual':torch.from_numpy(dual),'fair':torch.from_numpy(fair),'base':torch.from_numpy(base),'local':torch.from_numpy(local),'m':torch.tensor(float(z['global_residual'])),'valid':torch.from_numpy(v),'gt':torch.from_numpy(gt),'gtmask':torch.from_numpy(gtmask),'padding':torch.from_numpy(z['padding'].astype('int64')),'seq':row['sequence'],'index':int(row['index']),'age':int(row['age']),'status':row['status']}
 class Gate(nn.Module):
  def __init__(self,c,dual):
   super().__init__(); self.dual=dual
   layers=[]
   for a,b in [(c,16),(16,16),(16,16)]: layers.extend([nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.ReLU()])
   self.trunk=nn.Sequential(*layers); self.spatial=nn.Conv2d(16,1,1); self.global_head=nn.Linear(16,1)
  def forward(self,b):
   h=self.trunk(b['dual'] if self.dual else b['fair']); s=self.spatial(h)[:,0]; g=self.global_head(h.mean((2,3)))[:,0]
   if self.dual:
    s=s.sigmoid(); g=g.sigmoid(); delta=g[:,None,None]*b['m'][:,None,None]+s*b['local']; delta=torch.where(b['valid'],delta,torch.zeros_like(delta))
    # Failed reference/anchor masks are exact output fallbacks.
    active=torch.tensor([x=='ok' for x in b['status']],device=delta.device)[:,None,None]; delta=torch.where(active,delta,torch.zeros_like(delta))
   else:
    delta=g[:,None,None]+s
    delta=torch.where((b['age']>0)[:,None,None],delta,torch.zeros_like(delta))
   return b['base']+delta,g,s
 def move(b): return {k:(v.to('cuda',non_blocking=True) if torch.is_tensor(v) else v) for k,v in b.items()}
 def output_gt(logp,b):
  # Same depth-space resizing, padding and clamping as the official evaluation.
  return match_gt(logp.float().exp()[:,None],b['gt'][:,None],b['padding'],None)[:,0].clamp_min(.01)
 def evaluate(model,loader,write_path=None):
  model.eval(); byseq={}; detail=[]
  with torch.no_grad():
   for b0 in loader:
    b=move(b0)
    with torch.autocast('cuda',dtype=torch.float16): lp,g,s=model(b)
    d=output_gt(lp,b); gt=b['gt']; mask=b['gtmask']
    for j,name in enumerate(b0['seq']):
     good=mask[j]; n=int(good.sum()); assert n>0
     err=(d[j][good]-gt[j][good]); rel=float((err.abs()/gt[j][good]).mean()); rmse=float(err.square().mean().sqrt()); ratio=torch.maximum(d[j][good]/gt[j][good],gt[j][good]/d[j][good]); d1=float((ratio<1.25).float().mean())
     entry={'sequence':name,'index':int(b0['index'][j]),'age':int(b0['age'][j]),'status':b0['status'][j],'absrel':rel,'rmse':rmse,'delta1':d1,'global_output':float(g[j]),'spatial_mean':float(s[j].mean())}
     byseq.setdefault(name,[]).append(entry); detail.append(entry)
  summaries=[{'sequence':n,**{k:float(np.mean([r[k] for r in rr])) for k in ('absrel','rmse','delta1')},'frames':len(rr)} for n,rr in byseq.items()]
  if write_path:
   with write_path.open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(detail[0])); w.writeheader(); w.writerows(detail)
   with write_path.with_name('metrics_per_sequence.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(summaries[0])); w.writeheader(); w.writerows(summaries)
  return float(np.mean([x['absrel'] for x in summaries])),summaries
 trainset=Cached(train); devloader=DataLoader(Cached(dev),batch_size=8,shuffle=False,num_workers=0)
 configs={'split_counts':counts,'dev_frames':len(dev),'seed_list':[20261112,20261113,20261114],'batch_size':8,'max_epochs':100,'patience':10,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'groupnorm_groups':4,'normalization':stats,'loss':'per-frame valid-GT mean absolute log depth error','sampling':'sequence-balanced replacement','grad_scaler':'CUDA AMP','ref_size':[294,392],'camera_gt':False,'keyframes':'native output for both methods','development_selection':'equal-sequence macro AbsRel'}
 (out/'config.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
 dcount=sum(p.numel() for p in Gate(7,True).parameters()); bcount=sum(p.numel() for p in Gate(6,False).parameters()); assert abs(dcount-bcount)/dcount<=.05
 print('parameters',dcount,bcount,'relative difference',abs(dcount-bcount)/dcount,flush=True)
 for seed in ([20261112] if args.pilot else configs['seed_list']):
  for method in ('dual','fair'):
   random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
   model=Gate(7 if method=='dual' else 6,method=='dual').cuda(); opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001); scaler=torch.cuda.amp.GradScaler()
   dest=out/f'{method}_seed{seed}'; dest.mkdir(parents=True,exist_ok=True); best=float('inf'); stale=0
   status.update(method=method,seed=seed,epoch=0); save_status()
   with (dest/'train_log.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=['epoch','train_loss','dev_macro_absrel','best_absrel','seconds']); writer.writeheader()
    for epoch in range(1,(2 if args.pilot else 100)+1):
     start=time.time(); gen=torch.Generator().manual_seed(seed+epoch); weights=torch.tensor([1/counts[x['sequence']] for x in train],dtype=torch.double)
     sampler=WeightedRandomSampler(weights,len(train),replacement=True,generator=gen); loader=DataLoader(trainset,batch_size=8,sampler=sampler,num_workers=0)
     model.train(); losses=[]
     for b0 in loader:
      b=move(b0); opt.zero_grad(set_to_none=True)
      with torch.autocast('cuda',dtype=torch.float16): lp,_,_=model(b)
      pred=output_gt(lp,b); gt=b['gt']; mask=b['gtmask']; assert bool(mask.flatten(1).any(1).all())
      safe_gt=torch.where(mask,gt,torch.ones_like(gt)); safe_pred=torch.where(mask,pred,torch.ones_like(pred)); err=(safe_pred.log()-safe_gt.log()).abs(); loss=(err.flatten(1).sum(1)/mask.flatten(1).sum(1)).mean()
      assert bool(torch.isfinite(loss)); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); losses.append(float(loss.detach()))
     metric,seq=evaluate(model,devloader)
     if metric<best:
      best=metric; stale=0; torch.save({'state_dict':model.state_dict(),'epoch':epoch,'best_macro_absrel':best,'seed':seed,'method':method},dest/'best.pt')
     else: stale+=1
     record={'epoch':epoch,'train_loss':float(np.mean(losses)),'dev_macro_absrel':metric,'best_absrel':best,'seconds':time.time()-start}; writer.writerow(record); f.flush(); print(method,seed,record,flush=True)
     status.update(epoch=epoch,best_absrel=best,stale=stale); save_status()
     if stale>=10: break
   ck=torch.load(dest/'best.pt',map_location='cuda',weights_only=False); model.load_state_dict(ck['state_dict']); val,seq=evaluate(model,devloader,dest/'metrics_per_frame.csv')
   (dest/'result.json').write_text(json.dumps({'method':method,'seed':seed,'best_epoch':ck['epoch'],'macro_absrel':val,'sequences':seq},indent=2),encoding='utf-8')
 status.update(state='COMPLETED',phase='TRAINING'); save_status()
except Exception as e:
 status.update(state='ENGINEERING_BLOCKED',error=repr(e),traceback=traceback.format_exc()); save_status(); raise
