import csv,hashlib,io,json,sys,tempfile
from pathlib import Path
import cv2,h5py,numpy as np,torch
from PIL import Image
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'experiments/p1'))
import gt4a2_exp1_train_eval as exp1,gt2_fixed_clip_repair_train as gt2,cvpr_m05_formal as m05
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False)
snap=Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c';cp=ROOT/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
spec={'TUM':('val.txt','sequences.json',.001,10.),'ScanNet':('val.txt','sequences_.json',.01,10.),'Sintel':('training.txt','sequences.json',.001,1000.)};sources={'Bonn':'moving_obstructing_box'};hfiles={};sequences={}
for name,(split,meta,lo,hi) in spec.items():
 h=h5py.File(Path.home()/'datasets/velodepth_evaluation'/(name+'.hdf5'),'r');s=json.loads(bytes(h[meta][:]).decode());key=bytes(h[split][:]).decode().strip().splitlines()[0].rsplit(' ',1)[0];sources[name]=key;hfiles[name]=h;sequences[name]=s[key]
protocol={'purpose':'M05 exploratory development; shared parent, replay repair off, H4','sources':sources,'cycles':[0,1],'ages':[1,4,8],'parents':['A','P'],'horizon':4,'checkpoint_sha256':sha(cp),'backbone_sha256':sha(snap/'model.safetensors'),'m05_source_sha256':sha(ROOT/'experiments/p1/cvpr_m05_formal.py'),'script_sha256':sha(Path(__file__)),'depth_contract':'Bonn /5000; TUM ScanNet Sintel /1000 with existing data-specific range; no crop','base_pretraining_overlap':'unverified'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
m05.HORIZON=4;device=torch.device('cuda:0');model=gt2.load_model(snap,'pre_readout',device);model.pixel_decoder.persistent_feature_repair.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['repair_head_state_dict'],strict=True);model.eval();counter=m05.CallCounter(model)
current=[];future=[];audit=[];details=[];timing=[]
for name,seq in sources.items():
 if name=='Bonn':
  records=exp1.read_manifest(ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940'/(seq+'_pairs_v4_candidate.csv'),training=False)[:32];first=gt2.load_rgb(Path(records[0]['rgb_path']))
 else:
  h=hfiles[name];rec=sequences[name];assert len(rec['image'])>=32;first=np.array(Image.open(io.BytesIO(h[rec['image'][0]][:].tobytes())).convert('RGB'))
 with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
  Image.fromarray(first).save(tmp.name);shape=exp1.shape_info_for(model,Path(tmp.name),device)
 model._m05_shape=shape;frames=[]
 for i in range(32):
  if name=='Bonn':
   match=str(records[i].get('gt_match_status','')).upper()=='MATCHED';frame=exp1.load_frame(records[i],shape,device,need_gt=match)
  else:
   rgb=np.array(Image.open(io.BytesIO(h[rec['image'][i]][:].tobytes())).convert('RGB'));img,u8=gt2.preprocess_rgb(rgb,device,shape['paddings'],shape['network_shape'],shape['mean'],shape['std']);raw=np.array(Image.open(io.BytesIO(h[rec['depth'][i]][:].tobytes())));raw=(raw[:,:,0].astype(np.int32)|(raw[:,:,1].astype(np.int32)<<8)|(raw[:,:,2].astype(np.int32)<<16)) if raw.ndim==3 else raw;gt=raw.astype(np.float32)/1000.;lo,hi=spec[name][2:];gt=np.where(np.isfinite(gt)&(gt>lo)&(gt<hi),gt,0).astype(np.float32);frame={'index':i,'img':img,'frame_u8':u8,'gt_np':gt}
  frames.append(frame)
 dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST);static={}
 with torch.no_grad():
  for i in range(1,32):
   if i%16:static[i]=exp1.transition_static(model,frames[i-1],frames[i],shape,dis)
  for cycle in (0,1):
   start=cycle*16
   for label,mode in [('A','off'),('P','pre_readout')]:
    rng=m05.capture_rng(torch);parents=m05.build_parent_states(model,exp1,frames,static,start,[1,4,8],mode,rng,torch)
    for age in (1,4,8):
     anchor=start+age;row={'seed':20261025,'sequence':seq,'cycle':cycle,'age':age,'anchor_index':anchor}
     a,b,c,d,e=m05.process_anchor(model,exp1,gt2,frames,static,shape,row,label,parents[age],1e-5,torch,counter)
     if not d['passed']:raise RuntimeError(f'INVARIANT FAIL {name} {cycle} {label} {age}: {d["invariants"]}')
     for coll in (a,b,c,e):
      for item in coll:item['dataset']=name
     d['dataset']=name;current+=a;future+=b;audit+=c;details.append(d);timing+=e
     print('ANCHOR_PASS',name,cycle,label,age,flush=True)
 del frames,static;torch.cuda.empty_cache()
def write(name,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (out/name).open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
write('current.csv',current);write('future.csv',future);write('states.csv',audit);write('timing.csv',timing)
(out/'invariants.json').write_text(json.dumps(details,indent=2,allow_nan=False),encoding='utf8')
for h in hfiles.values():h.close()
counter.close();print('M05_ALL_INVARIANTS_PASS',len(details),flush=True)
