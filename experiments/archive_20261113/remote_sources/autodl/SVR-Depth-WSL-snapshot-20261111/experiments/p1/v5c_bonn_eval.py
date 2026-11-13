import csv,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1,gt2_fixed_clip_repair_train as gt2,cvpr_m04_closed_loop as m04
out=Path(sys.argv[1]);nlimit=int(sys.argv[2]);out.mkdir(parents=True,exist_ok=False)
snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';manifest=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940';v4=ROOT/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt';v5=ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt'
seqs=['moving_obstructing_box','person_tracking2','removing_nonobstructing_box','moving_nonobstructing_box','placing_nonobstructing_box']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
protocol={'purpose':'V5-C Bonn5 in-domain development','sequences':seqs,'max_frames':nlimit,'K':16,'mask':'matched GT > 0','base_sha256':sha(snap/'model.safetensors'),'v4_sha256':sha(v4),'v5_sha256':sha(v5),'script_sha256':sha(Path(__file__)),'manifests':{s:sha(manifest/(s+'_pairs_v4_candidate.csv')) for s in seqs}}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
device=torch.device('cuda:0');fields=['sequence','arm','frame','age','status','valid_pixels','mask_sha256','AbsRel','RMSE','delta1'];rows=[]
for arm,cp,branch in [('v4_O',v4,'O_same_head'),('v5_O',v5,'O_same_head'),('v5_P',v5,'P_full')]:
 model=gt2.load_model(snap,'pre_readout',device);model.pixel_decoder.persistent_feature_repair.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['repair_head_state_dict'],strict=True);model.eval()
 for seq in seqs:
  records=exp1.read_manifest(manifest/(seq+'_pairs_v4_candidate.csv'),training=False);records=records[:nlimit] if nlimit else records;shape=exp1.shape_info_for(model,Path(records[0]['rgb_path']),device);dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);prev=None
  with torch.no_grad():
   for i,rec in enumerate(records):
    match=str(rec.get('gt_match_status','')).upper()=='MATCHED';frame=exp1.load_frame(rec,shape,device,need_gt=match)
    if i%16==0:
     model.pixel_decoder.feature_repair_mode='off';rays,r,c,f=exp1.keyframe_state(model,frame,shape);state={'radius':r,'confidence':c,'features':f,'rays_map':rays};pred=gt2.make_z_depth(r,rays,shape['padded_shape'],shape['paddings'],model.interpolation_mode).detach().float().cpu().numpy()
    else:
     static=exp1.transition_static(model,prev,frame,shape,dis);state,pred,_=m04.apply_branch(model,exp1,gt2,prev,frame,static,state,branch,shape,torch)
    row={'sequence':seq,'arm':arm,'frame':i,'age':i%16,'status':'NO_GT'}
    if match:
     gt=frame['gt_np'];mask=(gt>0)&np.isfinite(gt)
     if not mask.any():row['status']='NO_VALID_GT'
     elif not np.all(np.isfinite(pred[mask])&(pred[mask]>0)):row['status']='INVALID_PRED'
     else:
      g=gt[mask].astype(np.float64);p=pred[mask].astype(np.float64);row.update(status='OK',valid_pixels=int(mask.sum()),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),AbsRel=float(np.mean(abs(g-p)/g)),RMSE=float(np.sqrt(np.mean((g-p)**2))),delta1=float(np.mean(np.maximum(g/p,p/g)<1.25)))
    rows.append(row);prev=frame
    if (i+1)%100==0 or i+1==len(records):print(arm,seq,i+1,len(records),flush=True)
 del model;torch.cuda.empty_cache()
with (out/'frames.csv').open('x',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('V5C_BONN_EVALUATION_COMPLETE',len(rows),flush=True)
