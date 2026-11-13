import os,sys,json,random,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,torch,cv2
root=Path('/home/svrdepth/SVR-Depth/velodepth');out=root/'outputs/p5norm_eng09_velo_build_a';sys.path[:0]=[str(root),str(root/'experiments/p1'),str(out)]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
from eng09_velo_policy import build_policy_state
start=time.monotonic();status={'status':'FAIL'}
try:
 assert json.loads((out/'cpu_status.json').read_text())['status']=='PASS';random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats();snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';model=gt2.load_model(snap,'pre_readout',device);ck=torch.load(root/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt',map_location='cpu',weights_only=False);model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 data=exp1.read_manifest(out/'prefix_manifest.csv',training=False);shape=exp1.shape_info_for(model,Path(data[0]['rgb_path']),device);model._m05_shape=shape;assert tuple(shape['original_shape'])==(480,640) and tuple(shape['network_shape'])==(434,574);dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);capt=[];reads=[]
 def hook(mod,inp,out):capt.append(([x.detach().clone() for x in inp[0]],[x.detach().clone() for x in out]))
 def profile(frame,event,arg):
  if event=='call' and frame.f_code is gt2.fast_step.__code__:reads.append([x.detach().clone() for x in frame.f_locals['features_state']])
 handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook)
 with torch.inference_mode():
  previous=exp1.load_frame(data[0],shape,device,need_gt=False);rays,radius,confidence,features=exp1.keyframe_state(model,previous,shape);state={'features':features,'radius':radius,'confidence':confidence,'rays_map':rays};fast_count=0
  for idx in [1,2]:
   curr=exp1.load_frame(data[idx],shape,device,need_gt=False);static=exp1.transition_static(model,previous,curr,shape,dis);capt.clear();reads.clear();sys.setprofile(profile)
   try:native=m05.run_step(model,exp1,previous,curr,static,state,torch)
   finally:sys.setprofile(None)
   assert len(capt)==1 and len(reads)==1 and all(torch.equal(a,b) for a,b in zip(reads[0],state['features']));s0,s1=capt[0];assert all(torch.equal(a,b) for a,b in zip(s1,native['features']));state=build_policy_state(native,s0,s1,'0000');fast_count+=1;previous=curr
  dec=model.pixel_decoder;meta={'common_shape':list(dec.common_shape),'radial_shape':list(dec.radial_module.shapes),'radial_original_shapes':list(dec.radial_module.original_shapes),'angular_shapes':list(dec.angular_module.shapes),'network_shape':list(shape['network_shape']),'original_shape':list(shape['original_shape']),'parent_index':2,'parent_rgb':str(data[2]['rgb_path']),'parent_timestamp':str(data[2]['rgb_timestamp']),'anchor_idx':3,'h4_idx':7}
  cpu={k:([x.detach().cpu().clone() for x in v] if isinstance(v,list) else v.detach().cpu().clone()) for k,v in state.items()};stats={}
  for k,v in cpu.items():
   for j,x in enumerate(v if isinstance(v,list) else [v]):
    assert torch.isfinite(x).all();stats[k+('_'+str(j+1) if isinstance(v,list) else '')]={'shape':list(x.shape),'dtype':str(x.dtype),'min':float(x.min()),'max':float(x.max()),'mean':float(x.float().mean()),'finite':True}
  torch.save({'state':cpu,'metadata':meta},out/'P5NORM_ENG09_VELO_PARENT_STATE.pt')
 torch.cuda.synchronize();handle.remove();status.update(status='PASS',keyframe_calls=1,run_step_calls=fast_count,processed_frames=3,metadata=meta,tensor_stats=stats,seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576)
except Exception:status['error']=traceback.format_exc();print(status['error'],flush=True)
finally:(out/'parent_status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if status['status']!='PASS':sys.exit(1)
