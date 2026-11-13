"""Two real propagation frames from an existing frozen Hold parent; no keyframe inference."""
import os,sys,time,csv,json,random,traceback,subprocess
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path('/home/svrdepth/SVR-Depth/velodepth');OUT=ROOT/'outputs/p5norm_eng01_layerwise_smoke';sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1'),str(OUT)]
import numpy as np,torch,cv2
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
from eng01_policy import layer1_writeback
def resources():
 gpu=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line and len(line.split())>1 and line.split()[1].isdigit()};return int(gpu[0]),mem['MemAvailable']/1024
rows=[];status={'result':'NO-GO'};start=time.monotonic()
try:
 free,mem=resources();print('GPU Free MiB',free,'MemAvailable MiB',mem,flush=True)
 if free<6656 or mem<5632:raise RuntimeError('RESOURCE BLOCKED before model initialization')
 random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);dev=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats();snapshot=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c'
 model=gt2.load_model(snapshot,'pre_readout',dev);ck=torch.load(ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt',map_location='cpu',weights_only=False);model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 manifest=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940/person_tracking2_pairs_v4_candidate.csv';data=exp1.read_manifest(manifest,training=False);shape=exp1.shape_info_for(model,Path(data[0]['rgb_path']),dev);model._m05_shape=shape
 parent=ROOT/'outputs/p5norm_v2_4_independent_replication/runs/Bonn/person_tracking2/hold/anchor_1_tensors.pt';snap=torch.load(parent,map_location='cpu',weights_only=True);state={'features':[snap['s0_'+str(i)].to(dev) for i in range(1,5)],**{k:snap[k].to(dev) for k in ['radius','confidence','rays_map']}};del snap
 frames=[exp1.load_frame(data[i],shape,dev,need_gt=False) for i in [1,2,3]];dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);capture=[];reads=[];calls=[]
 def hook(mod,inp,out):
  calls.append(1);capture.append((list(inp[0]),list(out)))
 def profile(frame,event,arg):
  if event=='call' and frame.f_code is gt2.fast_step.__code__:
   f=frame.f_locals['features_state'];reads.append((list(f),[x.detach().clone() for x in f]))
 handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook)
 with torch.inference_mode():
  static=exp1.transition_static(model,frames[0],frames[1],shape,dis);native=m05.run_step(model,exp1,frames[0],frames[1],static,state,torch)
  if len(capture)!=1:raise RuntimeError('first forward must invoke one repair')
  s0,s1=capture[0]
  if len(s0)!=4 or len(s1)!=4 or not all(torch.equal(a,b) for a,b in zip(s1,native['features'])):raise RuntimeError('S1 not native next features')
  s0_before=[v.detach().clone() for v in s0];s1_before=[v.detach().clone() for v in s1];native_before={k:native[k].detach().clone() for k in ['radius','confidence','rays_map']};mixed=layer1_writeback(native,s0,s1);mixed_before=[v.clone() for v in mixed['features']];sources=s0+s1
  source_ptrs=[v.untyped_storage().data_ptr() for v in sources];written_ptrs=[v.untyped_storage().data_ptr() for v in mixed['features']]
  if len(set(written_ptrs))!=4 or set(written_ptrs)&set(source_ptrs):raise RuntimeError('writeback feature storage alias')
  for k in native_before:
   if not torch.equal(native[k],mixed[k]) or native[k].untyped_storage().data_ptr()==mixed[k].untyped_storage().data_ptr():raise RuntimeError('nonfeature mismatch or alias '+k)
  capture.clear();static=exp1.transition_static(model,frames[1],frames[2],shape,dis);sys.setprofile(profile)
  try:next_state=m05.run_step(model,exp1,frames[1],frames[2],static,mixed,torch)
  finally:sys.setprofile(None)
  if len(reads)!=1 or len(calls)!=2 or len(capture)!=1:raise RuntimeError('unexpected run_step or repair count')
  for i,(v,read) in enumerate(zip(mixed_before,reads[0][1])):
   src=s1_before[i] if i==0 else s0_before[i];delta=(read.float()-src.float()).abs();r={'layer':i+1,'expected':'S1' if i==0 else 'S0','shape':str(tuple(read.shape)),'dtype':str(read.dtype),'device':str(read.device),'max_abs_difference':float(delta.max()),'finite':bool(torch.isfinite(read).all()),'equal_expected':bool(torch.equal(read,src)),'equal_written':bool(torch.equal(read,v)),'written_storage':written_ptrs[i],'consumed_storage':reads[0][0][i].untyped_storage().data_ptr(),'source_storage':sources[i+4 if i==0 else i].untyped_storage().data_ptr(),'write_source_alias':written_ptrs[i] in source_ptrs};rows.append(r)
  if not all(x['finite'] and x['equal_expected'] and x['equal_written'] and x['written_storage']==x['consumed_storage'] and not x['write_source_alias'] for x in rows):raise RuntimeError('G1/G2 mismatch')
  if not all(torch.equal(x,y) for x,y in zip(s0,s0_before)) or not all(torch.equal(x,y) for x,y in zip(s1,s1_before)) or not all(torch.equal(x,y) for x,y in zip(mixed['features'],mixed_before)) or not all(torch.equal(native[k],v) and torch.equal(mixed[k],v) for k,v in native_before.items()):raise RuntimeError('G3 mutation')
 torch.cuda.synchronize();handle.remove();status.update(result='PASS',anchor_sequence='person_tracking2',parent_frame=data[1]['rgb_path'],intervention_frame=data[2]['rgb_path'],next_frame=data[3]['rgb_path'],parent_source=str(parent),run_step_calls=2,repair_calls=len(calls),seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576,G1='PASS',G2='PASS',G3='PASS')
except Exception:status.update(error=traceback.format_exc());print(status['error'],flush=True)
finally:
 with (OUT/'P5NORM_ENG01_STATE_TRACE.csv').open('w',newline='') as f:
  fields=['layer','expected','shape','dtype','device','max_abs_difference','finite','equal_expected','equal_written','written_storage','consumed_storage','source_storage','write_source_alias'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 (OUT/'gpu_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if status['result']!='PASS':sys.exit(1)
