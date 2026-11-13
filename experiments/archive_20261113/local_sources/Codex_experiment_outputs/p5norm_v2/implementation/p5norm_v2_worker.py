"""One strategy, one model, fixed frozen Bonn_00_age4 engineering smoke."""
import os,sys,time,json,hashlib,resource,random,traceback
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np,torch,cv2
from p5norm_v2_policy import select
from p5norm_v2_metrics import pack,fingerprint
root=Path('/home/svrdepth/SVR-Depth/velodepth');sys.path[:0]=[str(root),str(root/'experiments/p1')]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
strategy=sys.argv[1];out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=False);result={'strategy':strategy,'status':'STARTED','anchor':'Bonn_00_age4','error':None,'events':[]}
def save(): (out/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
try:
 if strategy not in ('A','C'):raise ValueError('unsupported strategy')
 checks={'experiments/p1/cvpr_m05_formal.py':'3e872b9cdd37f6368b73308e5fd8212cdc3964221da44537cc322d24c49e9e05','experiments/p1/gt4a2_exp1_train_eval.py':'c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f','experiments/p1/gt2_fixed_clip_repair_train.py':'bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192','velodepth/models/decoder_velo.py':'35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08','velodepth/models/persistent_feature_repair.py':'6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c','outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251'}
 actual={k:sha(root/k) for k in checks};result['sha']=actual
 if actual!=checks:raise RuntimeError('frozen source mismatch')
 snap=Path('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c')
 if sha(snap/'model.safetensors')!='615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e':raise RuntimeError('backbone mismatch')
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025)
 device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats();t=time.monotonic();result['gpu_initial_free']=torch.cuda.mem_get_info()[0];save()
 model=gt2.load_model(snap,'pre_readout',device); ck=torch.load(root/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt',map_location='cpu',weights_only=False);model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 if any(m.training for m in model.modules()):raise RuntimeError('training mode')
 path=root/'outputs/y04_v4_manifest_candidate_20261108_190940/moving_nonobstructing_box_pairs_v4_candidate.csv';rows=exp1.read_manifest(path,training=False);assert Path(rows[0]['rgb_path']).name=='1548266065.39019.png'
 shape=exp1.shape_info_for(model,Path(rows[0]['rgb_path']),device);model._m05_shape=shape;frames=[exp1.load_frame(rows[i],shape,device,need_gt=False) for i in range(6)];result['inputs']=[{'index':f['index'],'path':str(f['rgb_path']),'sha256':sha(f['rgb_path'])} for f in frames];result['network_shape']=shape['network_shape'];dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
 captured={};consumed={};index=-1
 def hook(mod,ins,output):
  if index in captured:raise RuntimeError('repair called twice in same frame')
  captured[index]=([x.detach().clone() for x in ins[0]],[x.detach().clone() for x in output])
 def pre(mod,args):
  if index in consumed:raise RuntimeError('decoder called twice in same frame')
  consumed[index]=[x.detach().clone() for x in args[0]['features_teacher']]
 h1=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook);h2=model.pixel_decoder.register_forward_pre_hook(pre) if False else None
 # Decoder.forward_fast is invoked directly, so a module forward hook cannot observe it.
 def trace(frame,event,arg):
  if event=='call' and frame.f_code is gt2.fast_step.__code__:
   consumed[index]=[x.detach().clone() for x in frame.f_locals['features_state']]
 with torch.inference_mode():
  state=m05.build_parent_states(model,exp1,frames,{},0,[1],'pre_readout',m05.capture_rng(torch),torch)[1]
  for index in range(1,6):
   static=exp1.transition_static(model,frames[index-1],frames[index],shape,dis);before=state
   sys.setprofile(trace)
   try:next_state=m05.run_step(model,exp1,frames[index-1],frames[index],static,state,torch)
   finally:sys.setprofile(None)
   if index not in captured or index not in consumed:raise RuntimeError('missing single forward capture or actual read')
   s0,s1=captured.pop(index);read=consumed.pop(index)
   if not all(torch.equal(x,y) for x,y in zip(read,before['features'])):raise RuntimeError('next read mismatch')
   if not all(torch.equal(x,y) for x,y in zip(s1,next_state['features'])):raise RuntimeError('repair output differs from returned features')
   if index>=4:
    for x in s0+s1:
     if not bool(torch.isfinite(x).all()):raise RuntimeError('nonfinite feature')
    diffs=[(b.float()-a.float()).double().norm().item() for a,b in zip(s0,s1)]
    depth=m05.make_prediction(model,gt2,next_state,shape)
    if not np.isfinite(depth).all():raise RuntimeError('nonfinite depth')
    event={'frame':index,'read':pack(read),'s0':pack(s0),'s1':pack(s1),'l2':diffs,'global_l2':sum(x*x for x in diffs)**.5,'native_radius':fingerprint(next_state['radius']),'native_confidence':fingerprint(next_state['confidence']),'current_depth':{'shape':list(depth.shape),'sha256':hashlib.sha256(depth.tobytes()).hexdigest(),'mean':float(depth.mean()),'finite':True},'read_equals_previous':True}
    result['events'].append(event)
   # common prefix is native; after branch every subsequent frame follows policy
   if index>=4:next_state['features']=[x.detach().clone() for x in select(strategy,s0,s1)]
   state=next_state
  torch.cuda.synchronize();result.update(status='PASS',elapsed_s=time.monotonic()-t,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  h1.remove();save()
except Exception:
 result['status']='FAIL';result['error']=traceback.format_exc();save();print(result['error'],flush=True);sys.exit(1)
