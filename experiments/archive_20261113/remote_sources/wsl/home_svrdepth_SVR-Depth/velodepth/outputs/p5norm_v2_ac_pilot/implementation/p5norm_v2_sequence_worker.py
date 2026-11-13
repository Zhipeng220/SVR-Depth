"""One full independent online A/C trajectory; GT is not read."""
import os,sys,json,time,resource,random,io,traceback
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np,torch,cv2,h5py
from PIL import Image
root=Path('/home/svrdepth/SVR-Depth/velodepth');sys.path[:0]=[str(root),str(root/'experiments/p1')]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
from p5norm_v2_policy import select
domain,seq,strategy,out=sys.argv[1:];out=Path(out);out.mkdir(parents=True,exist_ok=False);preds=out/'predictions';preds.mkdir();result={'dataset':domain,'sequence':seq,'strategy':strategy,'status':'STARTED','frames':0,'updates':0,'reads':0};(out/'status.json').write_text(json.dumps(result))
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
  for i in range(count):
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
    if not all(torch.equal(a,b) for a,b in zip(s1,nxt['features'])):raise RuntimeError('repair result mismatch '+str(i))
    nxt['features']=[x.detach().clone() for x in select(strategy,s0,s1)];state=nxt;result['updates']+=1;result['reads']+=1
   pred=m05.make_prediction(model,gt2,state,shape)
   if not np.isfinite(pred).all() or not (pred>0).all():raise RuntimeError('invalid depth '+str(i))
   np.save(preds/('frame_%06d.npy'%i),pred);previous=frame;result['frames']=i+1
   if i%50==0:print('frame',i,'of',count,flush=True)
 torch.cuda.synchronize();hook_handle.remove();result.update(status='PASS',elapsed_s=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
except Exception:result['status']='FAIL';result['error']=traceback.format_exc();print(result['error'],flush=True)
finally:
 if 'h' in globals() and h:h.close()
 (out/'status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if result['status']!='PASS':sys.exit(1)
