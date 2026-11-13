import os,sys,json,random,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,torch,cv2
root=Path('/home/svrdepth/SVR-Depth/velodepth');out=root/'outputs/p5norm_eng08_temporal_position_robustness';sys.path[:0]=[str(root),str(root/'experiments/p1'),str(out)]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
from eng08_policy import build_policy_state
try:
 random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);device=torch.device('cuda:0');snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';model=gt2.load_model(snap,'pre_readout',device);ck=torch.load(root/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt',map_location='cpu',weights_only=False);model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 data=exp1.read_manifest(root/'outputs/y04_v4_manifest_candidate_20261108_190940/person_tracking2_pairs_v4_candidate.csv',training=False);shape=exp1.shape_info_for(model,Path(data[0]['rgb_path']),device);model._m05_shape=shape;assert tuple(shape['network_shape'])==(434,574) and tuple(shape['original_shape'])==(480,640)
 parent=torch.load(root/'outputs/p5norm_eng06_single_layer_h4/initial_parent.pt',map_location='cpu',weights_only=True);state={k:([x.to(device) for x in v] if isinstance(v,list) else v.to(device)) for k,v in parent.items()};previous=exp1.load_frame(data[1],shape,device,need_gt=False);model.pixel_decoder.set_shapes({'image':previous['img'],'features':[state['features'][0].permute(0,2,3,1)]},[{'T':1,'B':1}]);assert tuple(model.pixel_decoder.radial_module.original_shapes)==(434,574) and tuple(model.pixel_decoder.common_shape)==(31,41)
 dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);capt=[];observed=[]
 def hook(mod,inp,out):capt.append(([x.detach().clone() for x in inp[0]],[x.detach().clone() for x in out]))
 def profile(frame,event,arg):
  if event=='call' and frame.f_code is gt2.fast_step.__code__:observed.append([x.detach().clone() for x in frame.f_locals['features_state']])
 handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook);(out/'parents').mkdir(exist_ok=False)
 with torch.inference_mode():
  for idx,pos in [(2,'P2'),(3,'P3')]:
   curr=exp1.load_frame(data[idx],shape,device,need_gt=False);static=exp1.transition_static(model,previous,curr,shape,dis);capt.clear();observed.clear();sys.setprofile(profile)
   try:native=m05.run_step(model,exp1,previous,curr,static,state,torch)
   finally:sys.setprofile(None)
   assert len(capt)==1 and len(observed)==1 and all(torch.equal(a,b) for a,b in zip(observed[0],state['features']));s0,s1=capt[0];assert all(torch.equal(a,b) for a,b in zip(s1,native['features']));state=build_policy_state(native,s0,s1,'0000')
   cpu={k:([x.detach().cpu().clone() for x in v] if isinstance(v,list) else v.detach().cpu().clone()) for k,v in state.items()};torch.save(cpu,out/'parents'/(pos+'.pt'));(out/'parents'/(pos+'.json')).write_text(json.dumps({'position':pos,'parent_frame_index':idx,'parent_rgb':str(data[idx]['rgb_path']),'anchor_idx':idx+1,'h4_idx':idx+5},indent=2));previous=curr
 torch.cuda.synchronize();handle.remove();print('PREFIX PASS: 2 run_step; full P2 and P3 HOLD states saved',flush=True)
except Exception:print(traceback.format_exc(),flush=True);sys.exit(1)
