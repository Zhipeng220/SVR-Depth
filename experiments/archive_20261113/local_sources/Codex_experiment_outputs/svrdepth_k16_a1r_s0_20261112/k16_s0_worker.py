import csv,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1,gt2_fixed_clip_repair_train as gt2,cvpr_m04_closed_loop as m04
out=Path(sys.argv[1]);nlimit=int(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';manifest=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940';v4=ROOT/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt';v5=ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt'
seqs=['person_tracking2']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
protocol={'purpose':'A1 Bonn fixed arm repeat; original K16 code; baseline has V5 checkpoint confirmed separately','K':16,'max_frames':nlimit}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
snaps={16:{},17:{},18:{}}; counts={'fast_calls':0,'repair_calls':0,'keyframes':0}; current_i=-1
def cp(x): return x.detach().clone().cpu().numpy()
def prof(fr,event,arg):
 if event=='call' and fr.f_code is gt2.fast_step.__code__:
  counts['fast_calls']+=1
  if current_i in snaps:
   for j,x in enumerate(fr.f_locals['features_state']):snaps[current_i]['read_feature_'+str(j+1)]=cp(x)
   snaps[current_i]['read_radius']=cp(fr.f_locals['radius_state'])
 if event=='call' and fr.f_code is exp1.one_fast_step.__code__ and current_i in snaps:
  snaps[current_i]['passed_confidence']=cp(fr.f_locals['confidence'])
def hook(mod,inp,out):
 counts['repair_calls']+=1
 if current_i in snaps:
  for j,x in enumerate(inp[0]):snaps[current_i]['repair_input_'+str(j+1)]=cp(x)
  for j,x in enumerate(out):snaps[current_i]['repair_output_'+str(j+1)]=cp(x)
orig_z=gt2.make_z_depth
def zhook(*args,**kwargs):
 if current_i in snaps:snaps[current_i]['readout_radius']=cp(args[0]);snaps[current_i]['readout_rays']=cp(args[1])
 return orig_z(*args,**kwargs)
gt2.make_z_depth=zhook
device=torch.device('cuda:0');fields=['sequence','arm','frame','age','status','valid_pixels','mask_sha256','AbsRel','RMSE','delta1'];rows=[]
for arm,cp,branch in [('v5_P',v5,'P_full')]:
 model=gt2.load_model(snap,'pre_readout',device);model.pixel_decoder.persistent_feature_repair.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['repair_head_state_dict'],strict=True);model.eval();handle=model.pixel_decoder.persistent_feature_repair.register_forward_hook(hook)
 for seq in seqs:
  records=exp1.read_manifest(manifest/(seq+'_pairs_v4_candidate.csv'),training=False);records=records[:nlimit] if nlimit else records;shape=exp1.shape_info_for(model,Path(records[0]['rgb_path']),device);dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);prev=None
  with torch.no_grad():
   for i,rec in enumerate(records):
    current_i=i;match=str(rec.get('gt_match_status','')).upper()=='MATCHED';frame=exp1.load_frame(rec,shape,device,need_gt=match)
    if i%16==0:
     counts['keyframes']+=1
     model.pixel_decoder.feature_repair_mode='off';rays,r,c,f=exp1.keyframe_state(model,frame,shape);state={'radius':r,'confidence':c,'features':f,'rays_map':rays};pred=gt2.make_z_depth(r,rays,shape['padded_shape'],shape['paddings'],model.interpolation_mode).detach().float().cpu().numpy()
    else:
     static=exp1.transition_static(model,prev,frame,shape,dis);sys.setprofile(prof)
     try:state,pred,_=m04.apply_branch(model,exp1,gt2,prev,frame,static,state,branch,shape,torch)
     finally:sys.setprofile(None)
    if i in snaps:
     for j,x in enumerate(state['features']):snaps[i]['committed_feature_'+str(j+1)]=cp(x)
     for k in ['radius','confidence','rays_map']:snaps[i]['committed_'+k]=cp(state[k])
     snaps[i]['depth']=pred.copy()
    row={'sequence':seq,'arm':arm,'frame':i,'age':i%16,'status':'NO_GT'}
    if match:
     gt=frame['gt_np'];mask=(gt>0)&np.isfinite(gt)
     if not mask.any():row['status']='NO_VALID_GT'
     elif not np.all(np.isfinite(pred[mask])&(pred[mask]>0)):row['status']='INVALID_PRED'
     else:
      g=gt[mask].astype(np.float64);p=pred[mask].astype(np.float64);row.update(status='OK',valid_pixels=int(mask.sum()),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),AbsRel=float(np.mean(abs(g-p)/g)),RMSE=float(np.sqrt(np.mean((g-p)**2))),delta1=float(np.mean(np.maximum(g/p,p/g)<1.25)))
    rows.append(row);prev=frame
    if (i+1)%100==0 or i+1==len(records):print(arm,seq,i+1,len(records),flush=True)
 handle.remove();del model;torch.cuda.empty_cache()
with (out/'frames.csv').open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('V5C_BONN_EVALUATION_COMPLETE',len(rows),flush=True)

(out/'snapshots').mkdir()
for i,values in snaps.items():np.savez(out/'snapshots'/('frame%d.npz'%i),**values)
(out/'counts.json').write_text(json.dumps(counts,indent=2))
