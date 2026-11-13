#!/usr/bin/env python3
"""Single-model bounded CUDA diagnostic; execute once in frozen project."""
import os,sys,time,json,hashlib,resource,traceback,csv
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
root=Path('/home/svrdepth/SVR-Depth/velodepth'); os.chdir(root)
out=Path('/tmp/p5norm_16a_localgpu_preflight_result.json')
result={'stage':'START','error':None}
def save():out.write_text(json.dumps(result,indent=2,default=str,allow_nan=False))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
files={'experiments/p1/cvpr_m05_formal.py':'3e872b9cdd37f6368b73308e5fd8212cdc3964221da44537cc322d24c49e9e05','experiments/p1/gt4a2_exp1_train_eval.py':'c24441cb405465c1a16a86127b51aade8f9af69b5d368f43d830d0e759ee3d7f','experiments/p1/gt2_fixed_clip_repair_train.py':'bf7cd2021b1d0bbf66d75502dba6afed87f4c8ff3ef58ff7b18f56ac28044192','velodepth/models/decoder_velo.py':'35ca12629accd1a2a77595f505165fa17dffab07c763a8b639604285b94bdb08','velodepth/models/persistent_feature_repair.py':'6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c','outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251'}
def avail():return int(next(x.split()[1] for x in open('/proc/meminfo') if x.startswith('MemAvailable:')))*1024
def rss():return int(next(x.split()[1] for x in open('/proc/self/status') if x.startswith('VmRSS:')))*1024
try:
 result['sha']={k:sha(root/k) for k in files};save()
 if result['sha']!=files:raise RuntimeError('source identity mismatch')
 import torch,cv2
 sys.path.insert(0,str(root/'experiments/p1'))
 import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 device=torch.device('cuda:0');free,total=torch.cuda.mem_get_info(device)
 result['pre']={'free':free,'total':total,'host_available':avail(),'rss':rss(),'gpu':torch.cuda.get_device_name(device)};save()
 if free<6.5*1024**3 or avail()<5.5*1024**3:raise RuntimeError('preload resource gate')
 snap=Path('/home/svrdepth/hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c')
 if sha(snap/'model.safetensors')!='615335cd006663cd567e0a1c74bef7ed622ea733dbff016844ab64e3da22ce3e':raise RuntimeError('backbone mismatch')
 torch.cuda.reset_peak_memory_stats(device);t=time.monotonic();result['stage']='FACTORY_STARTED';save()
 model=gt2.load_model(snap,'pre_readout',device)
 result['factory_seconds']=time.monotonic()-t;result['stage']='FACTORY_DONE';save()
 t=time.monotonic();ck=torch.load(root/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt',map_location='cpu',weights_only=False)
 model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 torch.cuda.synchronize();result['checkpoint_seconds']=time.monotonic()-t
 result['model']={'parameter_count':sum(p.numel() for p in model.parameters()),'parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()),'buffer_bytes':sum(p.numel()*p.element_size() for p in model.buffers()),'training_modules':[n for n,m in model.named_modules() if m.training],'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak_allocated':torch.cuda.max_memory_allocated(),'free':torch.cuda.mem_get_info()[0],'host_available':avail(),'rss':rss()};result['stage']='MODEL_LOADED';save()
 if result['model']['training_modules']:raise RuntimeError('model contains training modules')
 if torch.cuda.mem_get_info()[0]<2*1024**3 or avail()<2.5*1024**3:raise RuntimeError('postload resource gate')
 manifest=root/'outputs/y04_v4_manifest_candidate_20261108_190940/moving_nonobstructing_box_pairs_v4_candidate.csv'
 rows=exp1.read_manifest(manifest,training=False);shape=exp1.shape_info_for(model,Path(rows[0]['rgb_path']),device);model._m05_shape=shape
 frames=[exp1.load_frame(rows[i],shape,device,need_gt=False) for i in range(2)]
 def info(t):return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'ptr':t.untyped_storage().data_ptr(),'storage_bytes':t.untyped_storage().nbytes(),'finite':bool(torch.isfinite(t).all().item())}
 def state_info(s):return {k:([info(x) for x in v] if isinstance(v,list) else info(v)) for k,v in s.items()}
 def attrs():
  d=model.pixel_decoder;return {'common_shape':str(getattr(d,'common_shape',None)),'radial_shapes':str(getattr(d.radial_module,'shapes',None)),'original_shapes':str(getattr(d.radial_module,'original_shapes',None)),'angular_shapes':str(getattr(d.angular_module,'shapes',None)),'feature_repair_mode':d.feature_repair_mode}
 result['input']={'paths':[str(x['rgb_path']) for x in frames],'original_shape':shape['original_shape'],'network_shape':shape['network_shape'],'tensor_shape':list(frames[0]['img'].shape),'device':str(frames[0]['img'].device)};save()
 with torch.inference_mode():
  t=time.monotonic();cache=m05.build_parent_states(model,exp1,frames,{},0,[1],'pre_readout',m05.capture_rng(torch),torch);state=cache[1];torch.cuda.synchronize();result['frame0']={'seconds':time.monotonic()-t,'state':state_info(state),'attrs':attrs(),'rss':rss(),'host_available':avail(),'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak':torch.cuda.max_memory_allocated()};result['stage']='FRAME0';save()
  if not all(v['finite'] for v in [*result['frame0']['state']['features'],result['frame0']['state']['radius'],result['frame0']['state']['confidence'],result['frame0']['state']['rays_map']]):raise RuntimeError('nonfinite frame0')
  if avail()<2.5*1024**3 or torch.cuda.mem_get_info()[0]<2*1024**3:raise RuntimeError('pre-frame1 resource gate')
  static=exp1.transition_static(model,frames[0],frames[1],shape,cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST))
  observed={}
  def trace(frame,event,arg):
   if event=='call' and frame.f_code is gt2.fast_step.__code__:observed['fast_features']=[info(x) for x in frame.f_locals['features_state']]
   if event=='call' and frame.f_code is type(model.pixel_decoder).forward_fast.__code__:observed['decoder_features']=[info(x) for x in frame.f_locals['inputs']['features_teacher']]
  t=time.monotonic();sys.setprofile(trace)
  try:state1=m05.run_step(model,exp1,frames[0],frames[1],static,state,torch)
  finally:sys.setprofile(None)
  torch.cuda.synchronize();result['frame1']={'seconds':time.monotonic()-t,'state':state_info(state1),'observed':observed,'attrs':attrs(),'rss':rss(),'host_available':avail(),'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),'peak':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};result['stage']='FRAME1';save()
  a=[x['ptr'] for x in result['frame0']['state']['features']];b=[x['ptr'] for x in observed.get('decoder_features',[])];result['readback_storage_equal']=(a==b and len(a)==4)
  if not result['readback_storage_equal']:raise RuntimeError('feature readback not observed')
  if not all(v['finite'] for v in [*result['frame1']['state']['features'],result['frame1']['state']['radius'],result['frame1']['state']['confidence'],result['frame1']['state']['rays_map']]):raise RuntimeError('nonfinite frame1')
 result['stage']='PASS'
except Exception:
 result['error']=traceback.format_exc()
finally:
 save();print('P5_RESULT',json.dumps(result,default=str),flush=True)
