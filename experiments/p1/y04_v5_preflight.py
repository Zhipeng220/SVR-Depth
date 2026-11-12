import hashlib,json,sys
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_y04_v4_bonn_train_v0_2 as v4
v4.configure_determinism()
import gt4a2_exp1_train_eval as exp1,gt4a2_c3_closed_loop as c3,gt2_fixed_clip_repair_train as gt2,gt4a2_y04_fair_multiscene as y04
v4.install_deterministic_backward();torch.set_num_threads(4)
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';init=ROOT/'outputs/gt4a2_c3/gt4a2_c3_initial_head.pt';cache=ROOT/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/cache/crowd2/cycle_cache';device=torch.device('cuda:0')
torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025)
model=gt2.load_model(snap,'pre_readout',device);head=model.pixel_decoder.persistent_feature_repair;head.load_state_dict(torch.load(init,map_location='cpu',weights_only=True),strict=True);model.eval();head.train()
reader=c3.CycleReader(cache,max_items=0);item=reader.load(0);y04.validate_cycle_geometry(item,'crowd2/0');opt=torch.optim.AdamW(head.parameters(),lr=c3.LR,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=True)
def probe():
 radius,confidence,features=c3.move_state(item['keyframe_state'],device);rays=item['keyframe_state']['rays_map'].to(device);shape={'validity_mask':torch.ones((1,1,*item['network_shape']),device=device,dtype=torch.bool)};prev=c3.to_gpu_frame(item['frames'][0],device);cur=c3.to_gpu_frame(item['frames'][1],device);static=c3.to_gpu_static(item['statics'][0],device,shape)
 assert cur['gt'] is not None
 def statehash():
  h=hashlib.sha256()
  for x in [radius,confidence,*features]:h.update(x.detach().cpu().contiguous().numpy().tobytes())
  return h.hexdigest()
 before=statehash();model.pixel_decoder.feature_repair_mode='off'
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
  r0,c0,f0=exp1.one_fast_step(model,prev,cur,static,radius.detach().clone(),confidence.detach().clone(),[x.detach().clone() for x in features]);d0=gt2.make_z_depth(r0,rays,item['padded_shape'],item['paddings'],model.interpolation_mode).float()
 assert not d0.requires_grad and statehash()==before
 model.pixel_decoder.feature_repair_mode='pre_readout'
 with torch.autocast('cuda',dtype=torch.float16):
  r1,c1,f1=exp1.one_fast_step(model,prev,cur,static,radius.detach().clone(),confidence.detach().clone(),[x.detach().clone() for x in features]);d1=gt2.make_z_depth(r1,rays,item['padded_shape'],item['paddings'],model.interpolation_mode).float();depth,_=gt2.masked_log_depth_loss(d1,cur['gt'])
 mask=torch.isfinite(d0)&torch.isfinite(d1)&(d0>0)&(d1>0);assert bool(mask.all()),'Invalid prediction in preserve mask'
 preserve=F.smooth_l1_loss(d1[mask].log(),d0[mask].detach().log(),reduction='mean',beta=1.0)
 params=[p for p in head.parameters() if p.requires_grad]
 def norm(loss,retain):
  grads=torch.autograd.grad(loss,params,retain_graph=retain,allow_unused=True);assert all(g is None or torch.isfinite(g).all() for g in grads);return float(torch.sqrt(sum(g.float().square().sum() for g in grads if g is not None)))
 nd=norm(depth,True);npres=norm(preserve,False)
 return {'depth_loss':float(depth.detach()),'preserve_loss':float(preserve.detach()),'depth_gradient_norm':nd,'preserve_gradient_norm':npres,'teacher_requires_grad':d0.requires_grad,'parent_unchanged':statehash()==before,'prediction_max_abs':float((d1-d0).abs().max()),'valid_pixels':int(mask.sum())}
initial=probe();update=c3.train_cycle(model,head,opt,scaler,item,device);after=probe()
assert initial['parent_unchanged'] and after['parent_unchanged'] and after['depth_gradient_norm']>0 and after['preserve_gradient_norm']>0
lam=0.1*after['depth_gradient_norm']/after['preserve_gradient_norm']
assert np.isfinite(lam) and lam>0
result={'status':'PARTIAL_PREFLIGHT','note':'single-frame gradient probe after one V4 depth-only update; teacher no-grad same parent; lambda is candidate only until full-cycle and recovery tests pass','initial':initial,'depth_only_update':update,'after_update':after,'candidate_lambda':lam,'calibration_rule':'weighted preserve gradient norm = 10% depth gradient norm on fixed crowd2 cycle0 age1 after one depth-only update','preflight_only':True}
(out/'preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)
