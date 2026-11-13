import os,sys,csv,json,time,random,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,torch,cv2
ROOT=Path('/home/svrdepth/SVR-Depth/velodepth'); BASE=ROOT/'outputs/p5norm_eng09_velo_build_a'; OUT=ROOT/'outputs/p5norm_eng10_l2_temporal_position'
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/p1'),str(BASE)]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
from eng09_velo_policy import build_policy_state
policy=sys.argv[1]; dest=OUT/policy; dest.mkdir(parents=True,exist_ok=False)
status={'policy':policy,'status':'FAIL','run_step_calls':0}; traces=[]; metrics=[]; nonfeatures=[]; updates=[]; start=time.monotonic()
def save(name,rows):
 if rows:
  with (dest/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
try:
 random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025)
 dev=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats();print('PRE_MODEL',torch.cuda.mem_get_info(),torch.cuda.memory_allocated(),torch.cuda.memory_reserved(),flush=True)
 snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c'
 model=gt2.load_model(snap,'pre_readout',dev);ck=torch.load(ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt',map_location='cpu',weights_only=False)
 model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 with (ROOT/'outputs/p5norm_eng09_velo_raw_stagea/raw_manifest.csv').open(newline='',encoding='utf8') as f:data=list(csv.DictReader(f))
 assert [int(x['frame_index_zero_based']) for x in data[2:35]]==list(range(2,35))
 assert data[19]['gt_match_status']=='UNMATCHED' and all(data[i]['gt_match_status']=='MATCHED' for i in list(range(3,19))+list(range(20,35)))
 shape=exp1.shape_info_for(model,Path(data[0]['rgb_path']),dev);model._m05_shape=shape
 assert tuple(shape['original_shape'])==(480,640) and tuple(shape['network_shape'])==(434,574)
 payload=torch.load(BASE/'P5NORM_ENG09_VELO_PARENT_STATE.pt',map_location='cpu',weights_only=True);frozen=payload['state'];meta=payload['metadata'];assert meta['parent_index']==2 and meta['anchor_idx']==3
 state={k:([x.to(dev) for x in v] if isinstance(v,list) else v.to(dev)) for k,v in frozen.items()}
 prev=exp1.load_frame(data[2],shape,dev,need_gt=False);assert str(prev['rgb_path'])==meta['parent_rgb'] and str(data[2]['rgb_timestamp'])==meta['parent_timestamp']
 model.pixel_decoder.set_shapes({'image':prev['img'],'features':[state['features'][0].permute(0,2,3,1)]},[{'T':1,'B':1}]);dec=model.pixel_decoder
 assert list(dec.common_shape)==meta['common_shape'] and list(dec.radial_module.shapes)==meta['radial_shape'] and list(dec.radial_module.original_shapes)==meta['radial_original_shapes'] and list(dec.angular_module.shapes)==meta['angular_shapes']
 for k in ['features','radius','confidence','rays_map']:
  aa=frozen[k] if isinstance(frozen[k],list) else [frozen[k]];bb=state[k] if isinstance(state[k],list) else [state[k]]
  for a,b in zip(aa,bb): assert a.dtype==b.dtype and torch.equal(a,b.cpu())
 del frozen,payload
 dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);capt=[];reads=[]
 def hook(mod,inp,out):capt.append(([x.detach().clone() for x in inp[0]],[x.detach().clone() for x in out],[x.untyped_storage().data_ptr() for x in inp[0]],[x.untyped_storage().data_ptr() for x in out]))
 def profile(frame,event,arg):
  if event=='call' and frame.f_code is gt2.fast_step.__code__:
   f=frame.f_locals['features_state'];reads.append(([x.detach().clone() for x in f],[x.untyped_storage().data_ptr() for x in f]))
 handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook); previous_selected=None;source_snapshot=None
 with torch.inference_mode():
  for i in range(3,35):
   assert float(data[i]['rgb_timestamp'])>float(data[i-1]['rgb_timestamp'])
   cur=exp1.load_frame(data[i],shape,dev,need_gt=(i!=19));static=exp1.transition_static(model,prev,cur,shape,dis);capt.clear();reads.clear()
   incoming={k:state[k].detach().clone() for k in ['radius','confidence','rays_map']};sys.setprofile(profile)
   try:
    status['run_step_calls']+=1;native=m05.run_step(model,exp1,prev,cur,static,state,torch)
   finally:sys.setprofile(None)
   if len(capt)!=1 or len(reads)!=1:raise RuntimeError('unexpected repair hook or fast_step count')
   if previous_selected is not None:
    assert all(torch.equal(a,b) for a,b in zip(state['features'],previous_selected))
    assert all(torch.equal(a,b) for a,b in zip(source_snapshot[0],source_snapshot[1]))
    for j,(a,b) in enumerate(zip(previous_selected,reads[0][0])):
     err=float((a.float()-b.float()).abs().max());assert err==0
     traces[-4+j]['next_read_equal']='PASS';traces[-4+j]['read_max_abs_difference']=err;traces[-4+j]['consumed_storage']=reads[0][1][j]
   for k in ['radius','confidence','rays_map']:
    a=incoming[k];b=native[k];assert a.shape==b.shape and torch.isfinite(a).all() and torch.isfinite(b).all();delta=b.float()-a.float()
    nonfeatures.append(dict(policy=policy,index=i,field=k,shape=str(tuple(a.shape)),dtype=str(a.dtype),device=str(a.device),input_min=float(a.min()),input_max=float(a.max()),output_min=float(b.min()),output_max=float(b.max()),max_abs_difference=float(delta.abs().max()),l2_difference=float(torch.linalg.vector_norm(delta.reshape(-1))),finite=True))
   s0,s1,p0,p1=capt[0];assert all(torch.equal(a,b) for a,b in zip(s1,native['features']))
   active_mask='0100' if (3<=i<=17 if policy=='L2_EARLY_ONLY' else 20<=i<=34) else '0000';choice=build_policy_state(native,s0,s1,active_mask);source_snapshot=(s0+s1,[x.clone() for x in s0+s1]);previous_selected=[x.clone() for x in choice['features']]
   for j in range(4):
    expected=s1[j] if active_mask[j]=='1' else s0[j];err=float((choice['features'][j].float()-expected.float()).abs().max());assert err==0
    upd=float(torch.linalg.vector_norm((s1[j].float()-s0[j].float()).reshape(-1))); updates.append(dict(policy=policy,index=i,layer=j+1,update_l2=upd))
    traces.append(dict(policy=policy,active_mask=active_mask,index=i,next_index=i+1 if i<34 else 'TERMINAL',layer=j+1,source='S1' if active_mask[j]=='1' else 'S0',shape=str(tuple(expected.shape)),dtype=str(expected.dtype),device=str(expected.device),selected_storage=choice['features'][j].untyped_storage().data_ptr(),s0_storage=p0[j],s1_storage=p1[j],consumed_storage='',selection_max_abs_difference=err,read_max_abs_difference='',next_read_equal='NOT_OBSERVED',hook_count=len(capt),finite=bool(torch.isfinite(expected).all())))
   pred=m05.make_prediction(model,gt2,native,shape); assert np.isfinite(pred).all() and (pred>0).all()
   np.save(dest/('pred_%02d.npy'%i),pred)
   if i==3:
    baseline=np.load(BASE/'first_prediction.npy');assert np.allclose(baseline,pred,rtol=1e-5,atol=1e-5)
   if i==19:
    assert cur['gt_np'] is None
    metrics.append(dict(policy=policy,index=i,timestamp=data[i]['rgb_timestamp'],unlabeled=True,gt_valid_pixels='',prediction_finite_pixels=int(np.isfinite(pred).sum()),prediction_positive_pixels=int((pred>0).sum()),original_mask_pixels='',AbsRel='',RMSE=''))
   else:
    gt=cur['gt_np'];m=gt2.calculate_metrics(gt,pred); metrics.append(dict(policy=policy,index=i,timestamp=data[i]['rgb_timestamp'],unlabeled=False,gt_valid_pixels=int((np.isfinite(gt)&(gt>0)).sum()),prediction_finite_pixels=int(np.isfinite(pred).sum()),prediction_positive_pixels=int((pred>0).sum()),original_mask_pixels=m['valid_pixels'],AbsRel=m['AbsRel'],RMSE=m['RMSE']))
   state=choice;prev=cur
   print('FRAME',i,'policy',policy,'allocated_mib',round(torch.cuda.memory_allocated()/1048576,2),'reserved_mib',round(torch.cuda.memory_reserved()/1048576,2),'host_memavailable_kib',next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),flush=True)
   if i==20:print('INDEX19_NEXT_READ_PASS',flush=True)
 torch.cuda.synchronize();handle.remove();assert sum(x['next_read_equal']=='PASS' for x in traces)==124
 status.update(status='PASS',elapsed_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/1048576,peak_reserved_mib=torch.cuda.max_memory_reserved()/1048576,next_read_pass=124)
except Exception:
 status['error']=traceback.format_exc();print(status['error'],flush=True)
finally:
 save('trace.csv',traces);save('metrics.csv',metrics);save('nonfeature.csv',nonfeatures);save('updates.csv',updates)
 (dest/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
if status['status']!='PASS':sys.exit(1)
