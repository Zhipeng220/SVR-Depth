import csv, hashlib, json, math, os, sys
from pathlib import Path
import cv2, numpy as np, torch
ROOT=Path.cwd(); sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1
import gt2_fixed_clip_repair_train as gt2
SEQ=['moving_obstructing_box','person_tracking2','removing_nonobstructing_box','moving_nonobstructing_box','placing_nonobstructing_box']
snapshot=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c'
checkpoint=ROOT/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt'
manifest=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940'
out=Path(sys.argv[1]); limit=int(sys.argv[2]); out.mkdir(parents=True,exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
 return h.hexdigest()
identity={'purpose':'Bonn sequence-held-out in-domain development; not independent scenes','baseline':'original pretrained VeloDepth, repair off','candidate':'same pretrained VeloDepth with Bonn13 repair pre_readout gain 1','K':16,'fp16_autocast':True,'max_frames_per_sequence':limit,'snapshot_sha256':sha(snapshot/'model.safetensors'),'candidate_sha256':sha(checkpoint),'manifests':{s:sha(manifest/(s+'_pairs_v4_candidate.csv')) for s in SEQ}}
(out/'protocol.json').write_text(json.dumps(identity,indent=2),encoding='utf8')
device=torch.device('cuda:0'); ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
for arm in ('baseline','bonn13'):
 model=gt2.load_model(snapshot,'off' if arm=='baseline' else 'pre_readout',device); model.eval()
 if arm=='bonn13': model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True)
 for seq in SEQ:
  rows=exp1.read_manifest(manifest/(seq+'_pairs_v4_candidate.csv'),training=False); rows=rows[:limit] if limit else rows
  shape=exp1.shape_info_for(model,Path(rows[0]['rgb_path']),device); dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST); prev=None
  path=out/(arm+'_'+seq+'.csv')
  with path.open('x',newline='',encoding='utf8') as f:
   writer=csv.DictWriter(f,fieldnames=['sequence','arm','frame','matched','keyframe','valid_pixels','AbsRel','RMSE','delta1']);writer.writeheader()
   with torch.no_grad():
    for i,rec in enumerate(rows):
     matched=str(rec.get('gt_match_status','')).upper()=='MATCHED'; frame=exp1.load_frame(rec,shape,device,need_gt=matched)
     with torch.autocast(device_type='cuda',dtype=torch.float16):
      if i%16==0: rays,radius,confidence,features=exp1.keyframe_state(model,frame,shape)
      else:
       static=exp1.transition_static(model,prev,frame,shape,dis); radius,confidence,features=exp1.one_fast_step(model,prev,frame,static,radius,confidence,features)
      pred=gt2.make_z_depth(radius,rays,shape['padded_shape'],shape['paddings'],model.interpolation_mode).detach().float().cpu().numpy()
     data={'sequence':seq,'arm':arm,'frame':i,'matched':matched,'keyframe':i%16==0}
     if matched:
      gt=frame['gt_np']; mask=(gt>0)&np.isfinite(gt)
      if not np.all(np.isfinite(pred[mask])&(pred[mask]>0)): raise RuntimeError(f'invalid prediction {arm} {seq} {i}')
      met=gt2.calculate_metrics(gt,pred); data.update({k:met[k] for k in ('valid_pixels','AbsRel','RMSE','delta1')})
     writer.writerow(data);prev=frame
     if (i+1)%100==0 or i+1==len(rows): f.flush();print(arm,seq,i+1,'/',len(rows),flush=True)
 del model; torch.cuda.empty_cache()
for seq in SEQ:
 a=list(csv.DictReader((out/('baseline_'+seq+'.csv')).open())); b=list(csv.DictReader((out/('bonn13_'+seq+'.csv')).open())); assert len(a)==len(b)
 for x,y in zip(a,b):
  assert (x['frame'],x['matched'],x['valid_pixels'])==(y['frame'],y['matched'],y['valid_pixels'])
  if x['matched']=='True' and x['keyframe']=='True':
   assert max(abs(float(x[k])-float(y[k])) for k in ('AbsRel','RMSE','delta1'))<2e-5
print('PAIRED_MASK_AND_KEYFRAME_CHECK_PASS',flush=True)
