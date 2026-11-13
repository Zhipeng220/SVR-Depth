"""One full independent online A/C trajectory; GT is not read."""
import os,sys,json,time,resource,random,io,traceback,csv,math
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np,torch,cv2,h5py
from PIL import Image
root=Path('/home/svrdepth/SVR-Depth/velodepth');sys.path[:0]=[str(root),str(root/'experiments/p1')]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
from p5norm_v2_policy import select
domain,seq,anchor_s,out=sys.argv[1:]; anchor=int(anchor_s);strategy='single_commit';out=Path(out);out.mkdir(parents=True,exist_ok=False);preds=out/'predictions';preds.mkdir(); anchorfile=root/'outputs/p5norm_v2_ac_pilot/sequence_selection/pilot_anchor4_manifest.csv'; anchors={int(x['anchor_idx']):x for x in csv.DictReader(anchorfile.open()) if x['dataset']==domain and x['sequence_id']==seq and int(x['anchor_idx'])==anchor}; targets={int(x['target_idx']) for x in anchors.values()}; records=[]; result={'dataset':domain,'sequence':seq,'strategy':strategy,'status':'STARTED','frames':0,'updates':0,'reads':0};(out/'status.json').write_text(json.dumps(result))
try:
 random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 snapshot=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';model=gt2.load_model(snapshot,'pre_readout',device);ck=torch.load(root/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt',map_location='cpu',weights_only=False);model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 if any(x.training for x in model.modules()):raise RuntimeError('non eval module')
 h=None
 if domain=='Bonn':
  mf=root/'outputs/y04_v4_manifest_candidate_20261108_190940'/(seq+'_pairs_v4_candidate.csv');rows=exp1.read_manifest(mf,training=False);first=Path(rows[0]['rgb_path']);count=len(rows);shape=exp1.shape_info_for(model,first,device)
 else:
  h=h5py.File(Path.home()/'datasets/velodepth_evaluation'/(domain+'.hdf5'),'r');md=json.loads(bytes(h['sequences.json' if domain=='TUM' else 'sequences_.json'][:]).decode());images=md[seq]['image'];count=len(images)
  import tempfile
  with tempfile.NamedTemporaryFile(suffix='.png') as f:
   Image.open(io.BytesIO(h[images[0]][:].tobytes())).convert('RGB').save(f.name);shape=exp1.shape_info_for(model,Path(f.name),device)
 model._m05_shape=shape;dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);captured=[];observed=[]
 def hook(mod,inp,outp):captured.append(([x.detach().clone() for x in inp[0]],[x.detach().clone() for x in outp]))
 def trace(frame,event,arg):
  if event=='call' and frame.f_code is gt2.fast_step.__code__:observed.append([x.detach().clone() for x in frame.f_locals['features_state']])
 hook_handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook);previous=None;state=None
 with torch.inference_mode():
  for i in range(min(count,anchor+5)):
   if domain=='Bonn': frame=exp1.load_frame(rows[i],shape,device,need_gt=False)
   else:
    rgb=np.array(Image.open(io.BytesIO(h[images[i]][:].tobytes())).convert('RGB'))
    if tuple(rgb.shape[:2])!=tuple(shape['original_shape']):raise RuntimeError('RGB shape changed at '+str(i))
    img,u8=gt2.preprocess_rgb(rgb,device,shape['paddings'],shape['network_shape'],shape['mean'],shape['std']);frame={'index':i,'img':img,'frame_u8':u8}
   if i%16==0:
    rays,radius,confidence,features=exp1.keyframe_state(model,frame,shape);state={'radius':radius,'confidence':confidence,'features':features,'rays_map':rays}
   else:
    static=exp1.transition_static(model,previous,frame,shape,dis);captured.clear();observed.clear();sys.setprofile(trace)
    try:nxt=m05.run_step(model,exp1,previous,frame,static,state,torch)
    finally:sys.setprofile(None)
    if len(captured)!=1 or len(observed)!=1 or not all(torch.equal(a,b) for a,b in zip(observed[0],state['features'])):raise RuntimeError('feature read/repair invocation mismatch at '+str(i))
    s0,s1=captured[0]
    if i in anchors:
     ups=[];bases=[]
     for a,b in zip(s0,s1):
      x=a.float().reshape(-1);y=b.float().reshape(-1);ups.append(float(torch.linalg.vector_norm(y-x)));bases.append(float(torch.linalg.vector_norm(x)))
     overall=(sum(x*x for x in ups))**0.5;base=(sum(x*x for x in bases))**0.5
     if not all(math.isfinite(x) for x in ups+bases):raise RuntimeError('nonfinite norm')
     r={'dataset':domain,'sequence_id':seq,'strategy':strategy,'anchor_idx':i,'target_idx':int(anchors[i]['target_idx']),'frame_timestamp':anchors[i]['anchor_timestamp'],'overall_update_l2':overall,'overall_base_l2':base,'overall_relative_norm':overall/base if base>1e-12 else None,'relative_valid':all(x>1e-12 for x in bases) and base>1e-12,'capture_position':'after repair before policy selection'}
     for layer in range(4):r['update_l'+str(layer+1)]=ups[layer];r['base_l'+str(layer+1)]=bases[layer];r['relative_l'+str(layer+1)]=ups[layer]/bases[layer] if bases[layer]>1e-12 else None
     original=root/'outputs/p5norm_v2_1_norm_capture/P5NORM_V2_1_NORM_FEATURES.csv'
     prior=[z for z in csv.DictReader(original.open()) if z['dataset']==domain and z['sequence_id']==seq and z['strategy']=='A' and int(z['anchor_idx'])==anchor]
     if len(prior)!=1:raise RuntimeError('missing unique A reference')
     prior=prior[0];keys=['update_l1','update_l2','update_l3','update_l4','base_l1','base_l2','base_l3','base_l4','overall_update_l2','overall_relative_norm'];r['replay_abs_differences']={k:abs(float(r[k])-float(prior[k])) for k in keys};r['replay_match']=all(math.isclose(float(r[k]),float(prior[k]),rel_tol=1e-5,abs_tol=1e-5) for k in keys)
     if not r['replay_match']:raise RuntimeError('REPLAY_MISMATCH '+json.dumps(r['replay_abs_differences']))
     records.append(r)
    if not all(torch.equal(a,b) for a,b in zip(s1,nxt['features'])):raise RuntimeError('repair result mismatch '+str(i))
    nxt['features']=[x.detach().clone() for x in select('C' if i==anchor else 'A',s0,s1)];state=nxt;result['updates']+=1;result['reads']+=1;result.setdefault('choices',[]).append({'frame':i,'choice':'COMMIT' if i==anchor else 'HOLD'})
   pred=m05.make_prediction(model,gt2,state,shape)
   if not np.isfinite(pred).all() or not (pred>0).all():raise RuntimeError('invalid depth '+str(i))
   if i in targets:np.save(preds/('frame_%06d.npy'%i),pred)
   previous=frame;result['frames']=i+1
   if i%50==0:print('frame',i,'of',count,flush=True)
 if len(anchors)!=1 or len(records)!=1:raise RuntimeError('anchor capture incomplete')
 if sum(x['choice']=='COMMIT' for x in result['choices'])!=1:raise RuntimeError('commit count')
 if [x['choice'] for x in result['choices'] if anchor<x['frame']<=anchor+4]!=['HOLD']*4:raise RuntimeError('post-anchor hold count')
 (out/'norm_records.json').write_text(json.dumps(records,indent=2,allow_nan=False))
 torch.cuda.synchronize();hook_handle.remove();result.update(status='PASS',elapsed_s=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
except Exception:result['status']='FAIL';result['error']=traceback.format_exc();print(result['error'],flush=True)
finally:
 if 'h' in globals() and h:h.close()
 (out/'status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if result['status']!='PASS':sys.exit(1)
