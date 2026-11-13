import os,sys,json,time,random,io,traceback,csv,math
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np,torch,cv2,h5py
from PIL import Image
root=Path('/home/svrdepth/SVR-Depth/velodepth');sys.path[:0]=[str(root),str(root/'experiments/p1'),str(root/'outputs/p5norm_v2_ac_pilot/implementation')]
import gt2_fixed_clip_repair_train as gt2,gt4a2_exp1_train_eval as exp1,cvpr_m05_formal as m05
from p5norm_v2_policy import select
domain,seq,mode,anchor_s,out_s=sys.argv[1:];anchor=int(anchor_s);out=Path(out_s);out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
all_anchors=[x for x in csv.DictReader((root/'outputs/p5norm_v2_4_independent_replication/P5NORM_V2_4_FROZEN_ANCHORS.csv').open()) if x['dataset']==domain and x['sequence_id']==seq]; anchors={int(x['anchor_idx']):x for x in all_anchors if mode=='hold' or int(x['anchor_idx'])==anchor};targets={int(x['target_idx']) for x in anchors.values()}; records=[];result=dict(dataset=domain,sequence=seq,mode=mode,status='STARTED');h=None
try:
 random.seed(20261025);np.random.seed(20261025);torch.manual_seed(20261025);torch.cuda.manual_seed_all(20261025);device=torch.device('cuda:0');torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 snapshot=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';model=gt2.load_model(snapshot,'pre_readout',device);ck=torch.load(root/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt',map_location='cpu',weights_only=False);model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);del ck;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 if domain=='Bonn':
  rows=exp1.read_manifest(root/'outputs/y04_v4_manifest_candidate_20261108_190940'/(seq+'_pairs_v4_candidate.csv'),training=False);count=len(rows);shape=exp1.shape_info_for(model,Path(rows[0]['rgb_path']),device)
 else:
  h=h5py.File(Path.home()/'datasets/velodepth_evaluation'/(domain+'.hdf5'),'r');md=json.loads(bytes(h['sequences.json' if domain=='TUM' else 'sequences_.json'][:]).decode());images=md[seq]['image'];count=len(images)
  import tempfile
  with tempfile.NamedTemporaryFile(suffix='.png') as f:Image.open(io.BytesIO(h[images[0]][:].tobytes())).convert('RGB').save(f.name);shape=exp1.shape_info_for(model,Path(f.name),device)
 model._m05_shape=shape;dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);captured=[];observed=[]
 def hook(mod,inp,outp):captured.append(([x.detach().clone() for x in inp[0]],[x.detach().clone() for x in outp]))
 def trace(frame,event,arg):
  if event=='call' and frame.f_code is gt2.fast_step.__code__:observed.append([x.detach().clone() for x in frame.f_locals['features_state']])
 handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook);previous=None;state=None;checks=[]
 with torch.inference_mode():
  for i in range(min(count,max(targets)+1)):
   if domain=='Bonn':frame=exp1.load_frame(rows[i],shape,device,need_gt=False)
   else:
    rgb=np.array(Image.open(io.BytesIO(h[images[i]][:].tobytes())).convert('RGB'));img,u8=gt2.preprocess_rgb(rgb,device,shape['paddings'],shape['network_shape'],shape['mean'],shape['std']);frame={'index':i,'img':img,'frame_u8':u8}
   if i%16==0:
    rays,radius,confidence,features=exp1.keyframe_state(model,frame,shape);state={'radius':radius,'confidence':confidence,'features':features,'rays_map':rays}
   else:
    static=exp1.transition_static(model,previous,frame,shape,dis);captured.clear();observed.clear();sys.setprofile(trace)
    try:nxt=m05.run_step(model,exp1,previous,frame,static,state,torch)
    finally:sys.setprofile(None)
    if len(captured)!=1 or len(observed)!=1 or not all(torch.equal(a,b) for a,b in zip(observed[0],state['features'])):raise RuntimeError('feature read/repair mismatch '+str(i))
    s0,s1=captured[0]
    if not all(torch.equal(a,b) for a,b in zip(s1,nxt['features'])):raise RuntimeError('repair output mismatch')
    if i in anchors:
     u=torch.stack([torch.linalg.vector_norm((b.float()-a.float()).reshape(-1)) for a,b in zip(s0,s1)]);b=torch.stack([torch.linalg.vector_norm(a.float().reshape(-1)) for a in s0]);overall=torch.linalg.vector_norm(u);base=torch.linalg.vector_norm(b)
     record=dict(dataset=domain,sequence_id=seq,anchor_idx=i,target_idx=int(anchors[i]['target_idx']),overall_update_l2=float(overall),overall_relative_norm=float(overall/base) if base>1e-12 else None)
     for l in range(4):record['update_l'+str(l+1)]=float(u[l]);record['base_l'+str(l+1)]=float(b[l])
     records.append(record)
     tensors={'s0_'+str(k+1):v.cpu() for k,v in enumerate(s0)};tensors.update({'s1_'+str(k+1):v.cpu() for k,v in enumerate(s1)});tensors.update({k:nxt[k].cpu() for k in ['radius','confidence','rays_map']});tensors.update(update_l2=u.cpu(),base_l2=b.cpu());tensors['current_depth']=torch.from_numpy(m05.make_prediction(model,gt2,nxt,shape))
     if mode=='hold':torch.save(tensors,out/('anchor_%d_tensors.pt'%i))
     else:
      reference=torch.load(out.parent/'hold'/('anchor_%d_tensors.pt'%i),map_location='cpu',weights_only=True)
      for k,x in tensors.items():
       y=reference[k];z=(x.float()-y.float()).abs();den=y.float().abs().clamp_min(1e-8);entry=dict(dataset=domain,sequence_id=seq,anchor_idx=i,tensor=k,shape=str(tuple(x.shape)),dtype=str(x.dtype),atol=1e-5,rtol=1e-5,max_abs=float(z.max()),mean_abs=float(z.mean()),max_relative=float((z/den).max()),allclose=bool(torch.allclose(x.float(),y.float(),rtol=1e-5,atol=1e-5)));checks.append(entry)
      if not all(x['allclose'] for x in checks):result['parent_functional_mismatch']=True
    if i<max(targets):nxt['features']=[x.detach().clone() for x in select('C' if mode=='branch' and i==anchor else 'A',s0,s1)]
    state=nxt
   if i in targets:np.save(out/'predictions'/('frame_%06d.npy'%i),m05.make_prediction(model,gt2,state,shape))
   previous=frame
 if len(records)!=len(anchors):raise RuntimeError('missing anchor captures')
 (out/'norm_records.json').write_text(json.dumps(records,indent=2));(out/'alignment.json').write_text(json.dumps(checks,indent=2));torch.cuda.synchronize();handle.remove();result.update(status='PASS',elapsed_s=time.monotonic()-start,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),frames=max(targets)+1)
except Exception:result.update(status='FAIL',error=traceback.format_exc());print(result['error'],flush=True)
finally:
 if h:h.close()
 (out/'status.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if result['status']!='PASS':sys.exit(1)
