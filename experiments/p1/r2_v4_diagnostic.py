import csv,hashlib,io,json,sys,tempfile
from pathlib import Path
import cv2,h5py,numpy as np,torch
from PIL import Image
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1,gt2_fixed_clip_repair_train as gt2,cvpr_m04_closed_loop as m04
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
snapshot=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';ckpath=ROOT/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt';ck=torch.load(ckpath,map_location='cpu',weights_only=False)
spec={'TUM':('val.txt','sequences.json',.001,10.),'ScanNet':('val.txt','sequences_.json',.01,10.),'Sintel':('training.txt','sequences.json',.001,1000.)}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
chosen={'Bonn':['moving_obstructing_box','person_tracking2']}; data={}
for name,(split,meta,lo,hi) in spec.items():

 h=h5py.File(Path.home()/'datasets/velodepth_evaluation'/(name+'.hdf5'),'r');seqs=json.loads(bytes(h[meta][:]).decode());keys=[x.rsplit(' ',1)[0] for x in bytes(h[split][:]).decode().strip().splitlines()];chosen[name]=keys[:2];data[name]=(h,seqs,lo,hi)
identity={'purpose':'R2 exploratory continuous factorial strategies; not same-parent causal effect','sequences':chosen,'max_frames':64,'policies':{'O':'never F1; r0 c0','S00':'F1 age 1; r0 c0','D00':'F1 every propagation; r0 c0','S11':'F1 age 1; r1 c1','D11':'F1 every propagation; r1 c1'},'output':'D1 every propagation frame','K':16,'checkpoint_sha256':sha(ckpath),'base_sha256':sha(snapshot/'model.safetensors'),'script_sha256':sha(Path(__file__)),'m04_source_sha256':sha(ROOT/'experiments/p1/cvpr_m04_closed_loop.py')}
(out/'protocol.json').write_text(json.dumps(identity,indent=2),encoding='utf8')
device=torch.device('cuda');model=gt2.load_model(snapshot,'pre_readout',device);model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True);model.eval()
fields=['dataset','sequence','branch','frame','age','feature_written','rc_submitted','status','valid_pixels','mask_sha256','AbsRel','RMSE','delta1'];allrows=[];states={}
for name,seqnames in chosen.items():
 for seq in seqnames:
  if name=='Bonn':
   manifest=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940'/(seq+'_pairs_v4_candidate.csv');records=exp1.read_manifest(manifest,training=False)[:64];first=gt2.load_rgb(Path(records[0]['rgb_path']));n=len(records)
  else:
   h,seqs,lo,hi=data[name];rec=seqs[seq];n=min(64,len(rec['image']));first=np.array(Image.open(io.BytesIO(h[rec['image'][0]][:].tobytes())).convert('RGB'))
  with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
   Image.fromarray(first).save(tmp.name);shape=exp1.shape_info_for(model,Path(tmp.name),device)
  for branch in ('O','S00','D00','S11','D11'):
   dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);prev=None;state=None
   with torch.no_grad():
    for i in range(n):
     if name=='Bonn':
      match=str(records[i].get('gt_match_status','')).upper()=='MATCHED';frame=exp1.load_frame(records[i],shape,device,need_gt=match);gt=frame['gt_np'] if match else None;mask=(gt>0)&np.isfinite(gt) if match else None
     else:
      rgb=np.array(Image.open(io.BytesIO(h[rec['image'][i]][:].tobytes())).convert('RGB'));img,u8=gt2.preprocess_rgb(rgb,device,shape['paddings'],shape['network_shape'],shape['mean'],shape['std']);frame={'index':i,'img':img,'frame_u8':u8}
      raw=np.array(Image.open(io.BytesIO(h[rec['depth'][i]][:].tobytes())));raw=(raw[:,:,0].astype(np.int32)|(raw[:,:,1].astype(np.int32)<<8)|(raw[:,:,2].astype(np.int32)<<16)) if raw.ndim==3 else raw;gt=raw.astype(np.float32)/1000.;mask=np.isfinite(gt)&(gt>lo)&(gt<hi)
     if i%16==0:
      model.pixel_decoder.feature_repair_mode='off';rays,r,c,f=exp1.keyframe_state(model,frame,shape);state={'radius':r,'confidence':c,'features':f,'rays_map':rays};pred=gt2.make_z_depth(r,rays,shape['padded_shape'],shape['paddings'],model.interpolation_mode).detach().float().cpu().numpy()
     else:
      static=exp1.transition_static(model,prev,frame,shape,dis)
      rng=m04.make_rng(torch);m04.put_rng(torch,rng);s0=m04.forward_step(model,exp1,prev,frame,static,m04.copy_state(state),'off',torch);m04.put_rng(torch,rng);s1=m04.forward_step(model,exp1,prev,frame,static,m04.copy_state(state),'pre_readout',torch)
      write_f=branch in ('D00','D11') or (branch in ('S00','S11') and i%16==1)
      use_rc=branch in ('S11','D11')
      state=m04.copy_state(s1 if use_rc else s0)
      state['features']=[x.detach().clone() for x in (s1['features'] if write_f else s0['features'])]
      assert m04.maxdiff(state['radius'],(s1 if use_rc else s0)['radius'])==0 and m04.maxdiff(state['confidence'],(s1 if use_rc else s0)['confidence'])==0
      assert m04.maxdiff(state['features'],s1['features'] if write_f else s0['features'])==0
      with torch.autocast(device_type='cuda',dtype=torch.float16):pred=gt2.make_z_depth(s1['radius'],s1['rays_map'],shape['padded_shape'],shape['paddings'],model.interpolation_mode).detach().float().cpu().numpy()
     row={'dataset':name,'sequence':seq,'branch':branch,'frame':i,'age':i%16,'feature_written':bool(i%16 and (branch in ('D00','D11') or (branch in ('S00','S11') and i%16==1))),'rc_submitted':('keyframe' if i%16==0 else ('r1c1' if branch in ('S11','D11') else 'r0c0')),'status':'NO_GT'}
     if mask is not None and mask.any():
      assert pred.shape==gt.shape and np.isfinite(pred[mask]).all() and (pred[mask]>0).all()
      g=gt[mask].astype(np.float64);p=pred[mask].astype(np.float64);row.update(status='OK',valid_pixels=int(mask.sum()),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),AbsRel=float(np.mean(abs(g-p)/g)),RMSE=float(np.sqrt(np.mean((g-p)**2))),delta1=float(np.mean(np.maximum(g/p,p/g)<1.25)))
     allrows.append(row);prev=frame
   print(name,seq,branch,n,flush=True)
with (out/'frames.csv').open('x',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrows)
for h,*_ in data.values():h.close()
print('R2_STATE_INVARIANTS_PASS',flush=True)
