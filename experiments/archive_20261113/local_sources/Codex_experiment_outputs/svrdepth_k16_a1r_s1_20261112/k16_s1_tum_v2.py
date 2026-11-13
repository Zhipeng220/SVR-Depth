import csv,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch,h5py,io,tempfile
from PIL import Image
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1,gt2_fixed_clip_repair_train as gt2,cvpr_m04_closed_loop as m04
out=Path(sys.argv[1]);nlimit=int(sys.argv[2]);out.mkdir(parents=True,exist_ok=False); (out/'snapshots').mkdir()
snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';manifest=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940';v4=ROOT/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt';v5=ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt'
seqs=['/scratch/lpiccinelli/datasets/tum/outputs/fr1_rpy_val'];h5=h5py.File(str(Path.home()/'datasets/velodepth_evaluation/TUM.hdf5'),'r'); seqdata=json.loads(bytes(h5['sequences.json'][:]).decode('ascii'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
protocol={'purpose':'A1 Bonn fixed arm repeat; original K16 code; baseline has V5 checkpoint confirmed separately','K':16,'max_frames':nlimit}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
snaps={16:{},17:{},18:{}}; decoder_log=[]; forward_role='NONE'; counts={'fast_calls':0,'repair_calls':0,'keyframes':0,'forward_started':0,'forward_completed':0,'frames_completed':0}; current_i=-1
ledger=(out/'S0R_RUNTIME_LEDGER.jsonl').open('a',buffering=1)
def log(event,**kw):
 ledger.write(json.dumps({'event':event,'frame_index':current_i,'age':current_i%16 if current_i>=0 else None,'branch':sys.argv[3],**kw})+'\n');ledger.flush()
import traceback,atexit
def failure(t,v,tb):
 log('exception',exception_stage='runtime',traceback=''.join(traceback.format_exception(t,v,tb)),counts=counts);sys.__excepthook__(t,v,tb)
sys.excepthook=failure
@atexit.register
def final_ledger():
 log('final',counts=counts,snapshots=[x.name for x in (out/'snapshots').glob('*.npz')])
def capture_tensor_snapshot(x): return x.detach().clone().cpu().numpy()
def prof(fr,event,arg):
 global forward_role
 if event=='call' and fr.f_code in (model.pixel_decoder.forward_keyframe.__func__.__code__,model.pixel_decoder.forward_fast.__func__.__code__) and current_i in snaps:record_decoder('actual_'+fr.f_code.co_name+'_entry')
 if event=='call' and fr.f_code is m04.forward_step.__code__:forward_role=fr.f_locals['mode']
 if fr.f_code in (exp1.keyframe_state.__code__,m04.forward_step.__code__):
  if event=='call':counts['forward_started']+=1;log('forward_started',function=fr.f_code.co_name)
  elif event=='return' and arg is not None:counts['forward_completed']+=1;log('forward_completed',function=fr.f_code.co_name)
 if event=='call' and fr.f_code is gt2.fast_step.__code__:
  counts['fast_calls']+=1
  if current_i in snaps: record_decoder('fast_step_entry')
  if current_i in snaps:
   for j,x in enumerate(fr.f_locals['features_state']):snaps[current_i]['read_'+forward_role+'_feature_'+str(j+1)]=capture_tensor_snapshot(x)
   snaps[current_i]['read_'+forward_role+'_radius']=capture_tensor_snapshot(fr.f_locals['radius_state'])
 if event=='call' and fr.f_code is exp1.one_fast_step.__code__ and current_i in snaps:
  snaps[current_i]['passed_'+forward_role+'_confidence']=capture_tensor_snapshot(fr.f_locals['confidence'])
def record_decoder(stage):
 dec=model.pixel_decoder
 for name,obj in [('decoder',dec),('radial_module',dec.radial_module),('angular_module',dec.angular_module)]:
  for key in ['feature_repair_mode','common_shape','shapes','original_shapes']:
   v=getattr(obj,key,None);decoder_log.append({'frame':current_i,'arm':sys.argv[3],'forward_role':forward_role,'stage':stage,'owner':name,'attribute':key,'exists':hasattr(obj,key),'value':repr(v)})
def hook(mod,inp,out):
 counts['repair_calls']+=1
 if current_i in snaps:
  for j,x in enumerate(inp[0]):snaps[current_i]['repair_input_'+str(j+1)]=capture_tensor_snapshot(x)
  for j,x in enumerate(out):snaps[current_i]['repair_output_'+str(j+1)]=capture_tensor_snapshot(x)
orig_z=gt2.make_z_depth
def zhook(*args,**kwargs):
 if current_i in snaps:snaps[current_i]['readout_radius']=capture_tensor_snapshot(args[0]);snaps[current_i]['readout_rays']=capture_tensor_snapshot(args[1])
 return orig_z(*args,**kwargs)
gt2.make_z_depth=zhook
device=torch.device('cuda:0');fields=['sequence','arm','frame','age','status','valid_pixels','mask_sha256','AbsRel','RMSE','delta1'];rows=[]
for arm,checkpoint_path,branch in [(sys.argv[3],v5,'O_same_head' if sys.argv[3].startswith('O') else 'P_full')]:
 model=gt2.load_model(snap,'pre_readout',device);model.pixel_decoder.persistent_feature_repair.load_state_dict(torch.load(checkpoint_path,map_location='cpu',weights_only=False)['repair_head_state_dict'],strict=True);model.eval();assert callable(capture_tensor_snapshot) and isinstance(checkpoint_path,Path) and checkpoint_path.is_file();handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook)
 for seq in seqs:
  item=seqdata[seq];records=list(range(min(len(item['image']),nlimit)));first=np.array(Image.open(io.BytesIO(h5[item['image'][0]][:].tobytes())).convert('RGB'));tmp=tempfile.NamedTemporaryFile(suffix='.png');Image.fromarray(first).save(tmp.name);shape=exp1.shape_info_for(model,Path(tmp.name),device);dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);prev=None
  with torch.no_grad(), torch.autocast(device_type='cuda',dtype=torch.float16):
   for i,rec in enumerate(records):
    current_i=i;log('frame_started',frame_started=True);match=True;rgb=np.array(Image.open(io.BytesIO(h5[item['image'][i]][:].tobytes())).convert('RGB'));img,frame_u8=gt2.preprocess_rgb(rgb,device,shape['paddings'],shape['network_shape'],shape['mean'],shape['std']);raw=np.array(Image.open(io.BytesIO(h5[item['depth'][i]][:].tobytes())));frame={'index':i,'img':img,'frame_u8':frame_u8,'gt_np':raw.astype(np.float32)/1000.}
    if i%16==0:
     counts['keyframes']+=1
     if i in snaps:record_decoder('keyframe_before_off')
     model.pixel_decoder.feature_repair_mode='off';sys.setprofile(prof)
     try:
      if i in snaps:record_decoder('keyframe_off_entry')
      rays,r,c,f=exp1.keyframe_state(model,frame,shape)
      if i in snaps:record_decoder('keyframe_return')
     finally:sys.setprofile(None)
     state={'radius':r,'confidence':c,'features':f,'rays_map':rays};pred=gt2.make_z_depth(r,rays,shape['padded_shape'],shape['paddings'],model.interpolation_mode).detach().float().cpu().numpy()
    else:
     static=exp1.transition_static(model,prev,frame,shape,dis);sys.setprofile(prof)
     try:state,pred,_=m04.apply_branch(model,exp1,gt2,prev,frame,static,state,branch,shape,torch)
     finally:sys.setprofile(None)
    if i in snaps:
     record_decoder('commit_return')
     for j,x in enumerate(state['features']):snaps[i]['committed_feature_'+str(j+1)]=capture_tensor_snapshot(x)
     for k in ['radius','confidence','rays_map']:snaps[i]['committed_'+k]=capture_tensor_snapshot(state[k])
     snaps[i]['depth']=pred.copy();np.savez(out/'snapshots'/('frame%d.npz'%i),**snaps[i]);log('observation_completed',observation_completed=True,snapshot='frame%d.npz'%i)
    np.save(out/('prediction_%03d.npy'%i),pred)
    row={'sequence':seq,'arm':arm,'frame':i,'age':i%16,'status':'NO_GT'}
    if match:
     gt=frame['gt_np'];mask=(gt>0.001)&(gt<10.)&np.isfinite(gt)
     if not mask.any():row['status']='NO_VALID_GT'
     elif not np.all(np.isfinite(pred[mask])&(pred[mask]>0)):row['status']='INVALID_PRED'
     else:
      g=gt[mask].astype(np.float64);p=pred[mask].astype(np.float64);row.update(status='OK',valid_pixels=int(mask.sum()),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),AbsRel=float(np.mean(abs(g-p)/g)),RMSE=float(np.sqrt(np.mean((g-p)**2))),delta1=float(np.mean(np.maximum(g/p,p/g)<1.25)))
    rows.append(row);prev=frame;counts['frames_completed']+=1;log('frame_completed',frame_completed=True,model_forward_completed=True)
    if (i+1)%100==0 or i+1==len(records):print(arm,seq,i+1,len(records),flush=True)
 handle.remove();del model;torch.cuda.empty_cache()
with (out/'frames.csv').open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('V5C_BONN_EVALUATION_COMPLETE',len(rows),flush=True)


(out/'counts.json').write_text(json.dumps(counts,indent=2))

if decoder_log:
 with (out/'decoder.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(decoder_log[0]));w.writeheader();w.writerows(decoder_log)
