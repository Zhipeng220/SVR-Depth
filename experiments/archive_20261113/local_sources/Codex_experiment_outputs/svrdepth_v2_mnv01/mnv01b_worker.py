import os,sys,csv,json,time,random,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,torch,cv2
ROOT=Path('/home/svrdepth/SVR-Depth/velodepth');BASE=ROOT/'outputs/p5norm_eng09_velo_build_a';OUT=ROOT/'outputs/svrdepth_v2_mnv01/history_pilot'
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1'),str(BASE)]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
from eng09_velo_policy import build_policy_state
scene,hist,mode,anchor_s=sys.argv[1:];anchor=int(anchor_s);target=anchor+4;assert scene in ('Bonn','TUM') and hist in ('H0','H1','H2') and mode in ('prefix','HOLD','FULL') and anchor in (9,14,20)
dest=OUT/scene/('anchor_'+str(anchor))/hist;dest.mkdir(parents=True,exist_ok=True); status={'scene':scene,'history':hist,'mode':mode,'anchor':anchor,'status':'FAIL','run_step_calls':0};start=time.monotonic()
try:
 random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);dev=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats()
 snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c'
 model=gt2.load_model(snap,'pre_readout',dev); ck=torch.load(ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt',map_location='cpu',weights_only=False);model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 manifest=(ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940/person_tracking2_pairs_v4_candidate.csv') if scene=='Bonn' else (ROOT/'outputs/p5norm_eng09_velo_raw_stagea/raw_manifest.csv')
 with manifest.open(newline='',encoding='utf-8-sig') as f:data=list(csv.DictReader(f))
 assert len(data)>target and all(float(data[i]['rgb_timestamp'])>float(data[i-1]['rgb_timestamp']) for i in range(1,target+1)) and data[target]['gt_match_status']=='MATCHED'
 shape=exp1.shape_info_for(model,Path(data[0]['rgb_path']),dev);model._m05_shape=shape;assert tuple(shape['original_shape'])==(480,640) and tuple(shape['network_shape'])==(434,574)
 dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);capt=[];reads=[]
 def hook(mod,inp,out):capt.append(([x.detach().clone() for x in inp[0]],[x.detach().clone() for x in out]))
 def profile(frame,event,arg):
  if event=='call' and frame.f_code is gt2.fast_step.__code__:reads.append([x.detach().clone() for x in frame.f_locals['features_state']])
 handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook)
 def step(i,previous,state,mask):
  current=exp1.load_frame(data[i],shape,dev,need_gt=False);static=exp1.transition_static(model,previous,current,shape,dis);capt.clear();reads.clear();sys.setprofile(profile)
  try:
   status['run_step_calls']+=1;native=m05.run_step(model,exp1,previous,current,static,state,torch)
  finally:sys.setprofile(None)
  assert len(capt)==len(reads)==1 and all(torch.equal(a,b) for a,b in zip(reads[0],state['features']))
  s0,s1=capt[0];assert all(torch.equal(a,b) for a,b in zip(s1,native['features']));choice=build_policy_state(native,s0,s1,mask)
  assert all(torch.equal(choice['features'][j],(s1 if mask[j]=='1' else s0)[j]) for j in range(4))
  return current,native,choice,s0,s1
 with torch.inference_mode():
  if mode=='prefix':
   previous=exp1.load_frame(data[0],shape,dev,need_gt=False);rays,radius,confidence,features=exp1.keyframe_state(model,previous,shape);state={'features':features,'radius':radius,'confidence':confidence,'rays_map':rays};decisions=[];norm_history=[]
   for i in range(1,anchor):
    mask='1111' if (hist in ('H1','H2') and i==anchor-6) or (hist=='H2' and i==anchor-3) else '0000'
    previous,native,state,s0,s1=step(i,previous,state,mask);decisions.append({'index':i,'mask':mask});norm_history.append(float(sum(float(torch.linalg.vector_norm((b.float()-a.float()).reshape(-1)))**2 for a,b in zip(s0,s1))**.5))
   dec=model.pixel_decoder;cpu={k:([x.detach().cpu().clone() for x in v] if isinstance(v,list) else v.detach().cpu().clone()) for k,v in state.items()};meta={'parent_index':anchor-1,'parent_rgb':str(previous['rgb_path']),'parent_timestamp':str(data[anchor-1]['rgb_timestamp']),'anchor':anchor,'target':target,'common_shape':list(dec.common_shape),'radial_shape':list(dec.radial_module.shapes),'radial_original_shapes':list(dec.radial_module.original_shapes),'angular_shapes':list(dec.angular_module.shapes),'decisions':decisions,'norm_history':norm_history}
   torch.save({'state':cpu,'metadata':meta},dest/'parent.pt');status['decisions']=decisions
  else:
   payload=torch.load(dest/'parent.pt',map_location='cpu',weights_only=True);meta=payload['metadata'];frozen=payload['state'];assert meta['parent_index']==anchor-1 and meta['anchor']==anchor
   state={k:([x.to(dev) for x in v] if isinstance(v,list) else v.to(dev)) for k,v in frozen.items()};previous=exp1.load_frame(data[anchor-1],shape,dev,need_gt=False);assert str(previous['rgb_path'])==meta['parent_rgb'] and str(data[anchor-1]['rgb_timestamp'])==meta['parent_timestamp']
   model.pixel_decoder.set_shapes({'image':previous['img'],'features':[state['features'][0].permute(0,2,3,1)]},[{'T':1,'B':1}]);dec=model.pixel_decoder
   assert list(dec.common_shape)==meta['common_shape'] and list(dec.radial_module.shapes)==meta['radial_shape'] and list(dec.radial_module.original_shapes)==meta['radial_original_shapes'] and list(dec.angular_module.shapes)==meta['angular_shapes']
   for k in frozen:
    aa=frozen[k] if isinstance(frozen[k],list) else [frozen[k]];bb=state[k] if isinstance(state[k],list) else [state[k]]
    assert all(a.dtype==b.dtype and torch.equal(a,b.cpu()) for a,b in zip(aa,bb))
   context={}
   for k in ['radius','confidence','rays_map']:
    x=state[k].detach().float().reshape(-1).cpu().numpy();context[k]={'mean':float(x.mean()),'std':float(x.std()),'p10':float(np.percentile(x,10)),'p50':float(np.percentile(x,50)),'p90':float(np.percentile(x,90))}
   for i in range(anchor,target+1):
    mask=('1111' if mode=='FULL' else '0000') if i==anchor else '0000';previous,native,state,s0,s1=step(i,previous,state,mask)
    if i==anchor:
     update=[float(torch.linalg.vector_norm((b.float()-a.float()).reshape(-1))) for a,b in zip(s0,s1)];base=[float(torch.linalg.vector_norm(a.float().reshape(-1))) for a in s0]; norm=sum(x*x for x in update)**.5;den=sum(x*x for x in base)**.5
     features={'scene':scene,'history':hist,'anchor':anchor,'mode':mode,'overall_update_l2':norm,'overall_relative_norm':norm/den if den>1e-12 else None,'layer_update_l2':update,'context_parent_before_forward':context,'history_decisions':meta['decisions'],'history_norms':meta['norm_history']}
     (dest/(mode+'_features.json')).write_text(json.dumps(features,indent=2))
    if i==target:
     pred=m05.make_prediction(model,gt2,native,shape); gt=exp1.load_frame(data[i],shape,dev,need_gt=True)['gt_np'];assert np.isfinite(pred).all() and (pred>0).all();metrics=gt2.calculate_metrics(gt,pred);np.save(dest/(mode+'_h4_pred.npy'),pred);np.save(dest/(mode+'_h4_mask.npy'),(gt>0)&np.isfinite(gt)&np.isfinite(pred)&(pred>0));status['metrics']=metrics;status['h4_rgb']=data[i]['rgb_path'];status['h4_depth']=data[i]['depth_path']
 torch.cuda.synchronize();handle.remove();status.update(status='PASS',elapsed_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576)
except Exception:
 status['error']=traceback.format_exc();print(status['error'],flush=True)
finally:
 (dest/(mode+'_status.json')).write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if status['status']!='PASS':sys.exit(1)

