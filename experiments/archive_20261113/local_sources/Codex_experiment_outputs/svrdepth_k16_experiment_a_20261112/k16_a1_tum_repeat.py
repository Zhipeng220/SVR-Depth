import csv,hashlib,io,json,math,sys,tempfile,time
from pathlib import Path
import cv2,h5py,numpy as np,torch
from PIL import Image
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1
import gt2_fixed_clip_repair_train as gt2
import cvpr_m04_closed_loop as m04
spec={'TUM':('val.txt','sequences.json',0.001,10.),'ScanNet':('val.txt','sequences_.json',0.01,10.),'Sintel':('training.txt','sequences.json',0.001,1000.)}
name=sys.argv[1];out=Path(sys.argv[2]);limitseq=int(sys.argv[3]);limitframes=int(sys.argv[4]);out.mkdir(parents=True,exist_ok=False)
snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';ckpt=ROOT/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt';v5=ROOT/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt';h5path=Path.home()/'datasets/velodepth_evaluation'/(name+'.hdf5')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
split,metadata,min_depth,max_depth=spec[name]
with h5py.File(h5path,'r') as h:
 lines=bytes(h[split][:]).decode('ascii').strip().splitlines(); seqs=json.loads(bytes(h[metadata][:]).decode('ascii')); keys=[line.rsplit(' ',1)[0] for line in lines if len(line.rsplit(' ',1))==2];keys=[x for x in keys if x.endswith('/fr1_rpy_val')];assert len(keys)==1
 protocol={'dataset':name,'purpose':'A1 repeat fixed arms','K':16,'frame_limit':limitframes}
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
 device=torch.device('cuda:0')
 for arm in ['v5_O','v5_P']:
  model=gt2.load_model(snap,'pre_readout',device);model.eval()
  ck=torch.load(ckpt if arm=='v4_O' else v5,map_location='cpu',weights_only=False);model.pixel_decoder.persistent_feature_repair.load_state_dict(ck['repair_head_state_dict'],strict=True)
  path=out/(arm+'.csv')
  with path.open('x',newline='',encoding='utf8') as f:
   writer=csv.DictWriter(f,fieldnames=['sequence','index','status','keyframe','valid_pixels','mask_sha256','AbsRel','RMSE','delta1']);writer.writeheader()
   for seq in keys:
    item=seqs[seq];images=item['image'];depths=item['depth'];assert len(images)==len(depths);count=min(len(images),limitframes) if limitframes else len(images)
    first=np.asarray(Image.open(io.BytesIO(h[images[0]][:].tobytes())).convert('RGB'))
    with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
     Image.fromarray(first).save(tmp.name);shape=exp1.shape_info_for(model,Path(tmp.name),device)
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);prev=None
    with torch.no_grad():
     for i in range(count):
      rgb=np.array(Image.open(io.BytesIO(h[images[i]][:].tobytes())).convert('RGB'));assert rgb.shape[:2]==tuple(shape['original_shape'])
      img,frame_u8=gt2.preprocess_rgb(rgb,device,shape['paddings'],shape['network_shape'],shape['mean'],shape['std']); frame={'index':i,'img':img,'frame_u8':frame_u8}
      with torch.autocast(device_type='cuda',dtype=torch.float16):
       if i%16==0:
        model.pixel_decoder.feature_repair_mode='off';rays,radius,confidence,features=exp1.keyframe_state(model,frame,shape);state={'radius':radius,'confidence':confidence,'features':features,'rays_map':rays};pred=gt2.make_z_depth(radius,rays,shape['padded_shape'],shape['paddings'],model.interpolation_mode).detach().float().cpu().numpy()
       else:
        static=exp1.transition_static(model,prev,frame,shape,dis);state,pred,_=m04.apply_branch(model,exp1,gt2,prev,frame,static,state,'P_full' if arm=='v5_P' else 'O_same_head',shape,torch)
      raw=np.array(Image.open(io.BytesIO(h[depths[i]][:].tobytes())))
      if raw.ndim==3: raw=raw[:,:,0].astype(np.int32)|(raw[:,:,1].astype(np.int32)<<8)|(raw[:,:,2].astype(np.int32)<<16)
      gt=raw.astype(np.float32)/1000.;assert gt.shape==pred.shape,(name,seq,gt.shape,pred.shape)
      mask=np.isfinite(gt)&(gt>min_depth)&(gt<max_depth);row={'sequence':seq,'index':i,'keyframe':i%16==0,'valid_pixels':int(mask.sum()),'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest()}
      if not mask.any():row['status']='invalid_gt'
      elif not np.all(np.isfinite(pred[mask])&(pred[mask]>0)):row['status']='invalid_prediction'
      else:
       g=gt[mask].astype(np.float64);p=pred[mask].astype(np.float64);row.update(status='ok',AbsRel=np.mean(abs(g-p)/g),RMSE=np.sqrt(np.mean((g-p)**2)),delta1=np.mean(np.maximum(g/p,p/g)<1.25))
      writer.writerow(row);prev=frame
    f.flush();print(name,arm,seq,count,'frames',flush=True)
  del model;torch.cuda.empty_cache()
 print('V5C_EXTERNAL_COMPLETE',name,flush=True)
