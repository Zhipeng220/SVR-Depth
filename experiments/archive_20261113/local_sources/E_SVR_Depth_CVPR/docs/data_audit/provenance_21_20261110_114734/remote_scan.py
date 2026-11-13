"""Read-only CPU frame/metadata provenance inventory; no model or GPU."""
import csv,hashlib,io,json,os,sys,zlib,base64
from pathlib import Path
import cv2,h5py,numpy as np
from PIL import Image
selected=json.loads(sys.argv[1]);ROOT=Path.home()/'SVR-Depth/velodepth';BONN=Path.home()/'datasets/bonn';out={'sequences':[],'frames':[],'errors':[],'bonn_crowd3':{}}
def sha(b):return hashlib.sha256(b).hexdigest()
def finfo(p):
 p=Path(p)
 if not p.exists():return {'path':str(p),'exists':False}
 if not p.is_file():return {'path':str(p),'exists':True,'kind':'directory'}
 data=p.read_bytes();return {'path':str(p),'realpath':str(p.resolve()),'is_symlink':p.is_symlink(),'exists':True,'bytes':len(data),'sha256':sha(data)}
def frame_info(blob,mod):
 im=Image.open(io.BytesIO(blob));a=np.asarray(im)
 pix=sha((str(a.dtype)+'|'+str(a.shape)+'|').encode()+np.ascontiguousarray(a).tobytes())
 if mod=='rgb':
  view=np.asarray(im.convert('RGB'));g=cv2.cvtColor(view,cv2.COLOR_RGB2GRAY)
 else:
  if a.ndim==3 and a.shape[2]>=3:
   x=a.astype(np.uint32);v=x[:,:,0]|(x[:,:,1]<<8)|(x[:,:,2]<<16)
  else:v=a.astype(np.float32)
  g=np.log1p(v.astype(np.float32))
 small=cv2.resize(g.astype(np.float32),(32,32),interpolation=cv2.INTER_AREA);dct=cv2.dct(small)[:8,:8];med=np.median(dct.ravel()[1:]);bits=(dct.ravel()>med);ph=0
 for b in bits:ph=(ph<<1)|int(b)
 return {'sha256':sha(blob),'decoded_pixel_sha256':pix,'phash64':f'{ph:016x}','bytes':len(blob),'image_shape':list(a.shape),'dtype':str(a.dtype)}
cache={}
def pathframe(p,mod):
 key=(str(p),mod)
 if key not in cache:
  blob=Path(p).read_bytes();cache[key]=frame_info(blob,mod)
 return cache[key]
def file_list(root):
 result=[]
 if root.exists():
  for p in sorted(root.iterdir()):
   if p.is_file() and p.suffix.lower() in ('.txt','.json','.yaml','.yml','.ply','.pcd','.bag','.log','.csv'):result.append(finfo(p))
 return result
for rec in selected:
 ds=rec['dataset'];seq=rec['sequence'];sr={'dataset':ds,'sequence':seq,'source_kind':'','metadata':[]};out['sequences'].append(sr)
 try:
  if ds=='Bonn':
   folder=BONN/('rgbd_bonn_'+seq);manifest=ROOT/'outputs/y04_v4_manifest_candidate_20261108_190940'/(seq+'_pairs_v4_candidate.csv')
   with manifest.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
   sr.update(source_kind='Bonn frame files via candidate pair CSV',root=str(folder),root_realpath=str(folder.resolve()),root_is_symlink=folder.is_symlink(),pair_manifest=finfo(manifest),metadata=file_list(folder),frame_count=len(rows))
   for i,r in enumerate(rows):
    for mod,col,tc in [('rgb','rgb_path','rgb_timestamp'),('depth','depth_path','depth_timestamp')]:
     path=r.get(col,'')
     if not path or not Path(path).is_file():
      out['errors'].append({'dataset':ds,'sequence':seq,'index':i,'modality':mod,'path':path,'reason':'missing'});continue
     f=pathframe(path,mod);p=Path(path);out['frames'].append({'dataset':ds,'sequence':seq,'index':i,'modality':mod,'path':path,'realpath':str(p.resolve()),'is_symlink':p.is_symlink(),'timestamp':r.get(tc,''),**f})
   sr['directory_file_counts']={};sr['unreferenced_files']=[]
   for mod in ('rgb','depth'):
    folder_mod=folder/mod;disk=sorted(p for p in folder_mod.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.tiff','.bmp'));sr['directory_file_counts'][mod]=len(disk);referenced={f['path'] for f in out['frames'] if f['dataset']==ds and f['sequence']==seq and f['modality']==mod}
    for j,p in enumerate(q for q in disk if str(q) not in referenced):
     f=pathframe(p,mod);sr['unreferenced_files'].append(str(p));out['frames'].append({'dataset':ds,'sequence':seq,'index':len(rows)+j,'modality':mod,'path':str(p),'realpath':str(p.resolve()),'is_symlink':p.is_symlink(),'timestamp':'','unreferenced_original':True,**f})
  else:
   path=Path.home()/'datasets/velodepth_evaluation'/(ds+'.hdf5');metakey='sequences.json' if ds=='TUM' else 'sequences_.json'
   with h5py.File(path,'r') as h:
    raw=bytes(h[metakey][:]);allmeta=json.loads(raw.decode());r=allmeta[seq];sr.update(source_kind='HDF5 embedded payload',container=str(path),container_realpath=str(path.resolve()),container_is_symlink=path.is_symlink(),container_bytes=path.stat().st_size,metadata_key=metakey,metadata_sha256=sha(raw),frame_count=len(r['image']),depth_count=len(r['depth']))
    for i in range(max(len(r['image']),len(r['depth']))):
     for mod,key in [('rgb','image'),('depth','depth')]:
      if i>=len(r[key]):out['errors'].append({'dataset':ds,'sequence':seq,'index':i,'modality':mod,'reason':'metadata missing index'});continue
      k=r[key][i]
      if k not in h:out['errors'].append({'dataset':ds,'sequence':seq,'index':i,'modality':mod,'key':k,'reason':'HDF5 missing key'});continue
      blob=h[k][:].tobytes();f=frame_info(blob,mod);out['frames'].append({'dataset':ds,'sequence':seq,'index':i,'modality':mod,'path':str(path)+'::'+k,'realpath':str(path.resolve())+'::'+k,'is_symlink':path.is_symlink(),'timestamp':'','hdf5_key':k,**f})
 except Exception as e:out['errors'].append({'dataset':ds,'sequence':seq,'reason':repr(e)})
# Investigate the exact desk train/val overlap, without loading any test result.
try:
 path=Path.home()/'datasets/velodepth_evaluation/TUM.hdf5'
 with h5py.File(path,'r') as h:
  h0='fr1_desk/1457-45_d.png';h1='fr1_desk/1457-49_d.png';keys=[h0,h1,h0.replace('_d.png','_c.jpg'),h1.replace('_d.png','_c.jpg')]
  out['tum_depth_lead']=[{'key':k,**frame_info(h[k][:].tobytes(),'depth' if k.endswith('_d.png') else 'rgb')} for k in keys]
except Exception as e:out['errors'].append({'lead':'TUM depth duplicate','reason':repr(e)})
# Source and shared point-cloud inspection; only exact assets and metadata, no edits.
for seq in ['crowd3','removing_nonobstructing_box']:
 folder=BONN/('rgbd_bonn_'+seq);out['bonn_crowd3'][seq]={'root':str(folder),'metadata':file_list(folder)}
raw=json.dumps(out,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode();print('AUDIT_ZLIB_BASE64:'+base64.b64encode(zlib.compress(raw,6)).decode(),flush=True)

