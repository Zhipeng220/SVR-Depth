import hashlib,io,json,os,sys,base64,zlib
from pathlib import Path
import h5py,numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
request=json.loads(sys.argv[1]);path=Path.home()/'datasets/velodepth_evaluation/TUM.hdf5'
def sha(b):return hashlib.sha256(b).hexdigest()
def image_stats(blob,mod):
 im=Image.open(io.BytesIO(blob));a=np.asarray(im);pix=sha((str(a.dtype)+'|'+str(a.shape)+'|').encode()+np.ascontiguousarray(a).tobytes())
 if mod=='rgb':g=cv2.cvtColor(np.asarray(im.convert('RGB')),cv2.COLOR_RGB2GRAY)
 else:
  if a.ndim==3 and a.shape[2]>=3:
   x=a.astype(np.uint32);a=x[:,:,0]|(x[:,:,1]<<8)|(x[:,:,2]<<16)
  g=np.log1p(a.astype(np.float32))
 d=cv2.dct(cv2.resize(g.astype(np.float32),(32,32),interpolation=cv2.INTER_AREA))[:8,:8];bits=d.ravel()>np.median(d.ravel()[1:]);ph=0
 for b in bits:ph=(ph<<1)|int(b)
 return {'sha256':sha(blob),'decoded_pixel_sha256':pix,'phash64':f'{ph:016x}'}
result={'tum':[],'dataset_discovery':[],'metadata':{},'bonn_panels':[]}
with h5py.File(path,'r') as h:
 meta=json.loads(bytes(h['sequences.json'][:]).decode());result['metadata']['hdf5_root_keys']=[k for k in h.keys() if isinstance(h[k],h5py.Dataset) and (k.endswith('.json') or k.endswith('.txt'))]
 for split,key in [('train','fr1_desk/1457-49_d.png'),('val','fr1_desk/1457-45_d.png')]:
  matches=[(k,v) for k,v in meta.items() if k.endswith('/fr1_desk_'+split)]
  if len(matches)!=1:raise RuntimeError('sequence metadata nonunique '+split)
  sequence,r=matches[0];indices=[i for i,k in enumerate(r['depth']) if k==key]
  if len(indices)!=1:raise RuntimeError('depth key nonunique '+key)
  idx=indices[0];extras={k:('array_len_'+str(len(v)) if isinstance(v,list) else str(v)[:500]) for k,v in r.items() if k not in ('image','depth')}
  row={'split':split,'sequence':sequence,'target_index':idx,'target_depth_key':key,'sequence_record_fields':list(r),'other_metadata':extras,'neighbors':[],'hdf5_file_attrs':{str(k):str(v)[:500] for k,v in h.attrs.items()}}
  for i in range(max(0,idx-5),min(len(r['depth']),idx+6)):
   dk=r['depth'][i];rk=r['image'][i];db=h[dk][:].tobytes();rb=h[rk][:].tobytes()
   # Capture timestamp only from explicit metadata fields, never from key names.
   explicit={}
   for k,v in r.items():
    if 'timestamp' in k.lower() and isinstance(v,list) and len(v)==len(r['depth']):explicit[k]=v[i]
   row['neighbors'].append({'index':i,'relative_offset':i-idx,'rgb_key':rk,'depth_key':dk,'rgb_timestamp':'','depth_timestamp':'','explicit_time_metadata':explicit,'rgb':image_stats(rb,'rgb'),'depth':image_stats(db,'depth'),'rgb_object_attributes':{str(k):str(v)[:500] for k,v in h[rk].attrs.items()},'depth_object_attributes':{str(k):str(v)[:500] for k,v in h[dk].attrs.items()}})
  result['tum'].append(row)
# bounded source discovery: filenames only; no full dataset scanning
home=Path.home()/'datasets'
for base,dirs,files in os.walk(home):
 depth=len(Path(base).relative_to(home).parts)
 if depth>=4:dirs[:]=[]
 dirs[:]=[d for d in dirs if d.lower() not in ('rgb','depth','color','image','images')]
 for name in dirs+files:
  low=name.lower()
  if 'fr1_desk' in low or (low.endswith('.sens') and any(x in low for x in ('0707','0708','0709','0710','0711','0712','0713','0714'))):result['dataset_discovery'].append(str(Path(base)/name))
# create contact sheet from prior fixed phash-run candidates, no new frame choice based on results
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
W,H=1760,1280;sheet=Image.new('RGB',(W,H),'white');draw=ImageDraw.Draw(sheet);draw.text((10,8),'Bonn candidate pairs: RGB | Depth (each depth panel scaled independently)',font=font,fill='black')
for ri,row in enumerate(request['panels']):
 y=42+ri*304;draw.text((10,y),row['label'],font=font,fill='black')
 for ci,item in enumerate(row['frames']):
  x=10+ci*580;rgbb=Path(item['rgb_path']).read_bytes();depb=Path(item['depth_path']).read_bytes()
  if sha(rgbb)!=item['rgb_sha256'] or sha(depb)!=item['depth_sha256']:raise RuntimeError('Bonn frame hash mismatch')
  rgb=Image.open(io.BytesIO(rgbb)).convert('RGB');da=np.asarray(Image.open(io.BytesIO(depb))).astype(np.float32)
  if da.ndim==3:da=da[:,:,0]
  ok=np.isfinite(da)&(da>0)
  if ok.any():
   lo,hi=np.percentile(da[ok],[2,98]);v=np.clip((da-lo)/max(hi-lo,1),0,1);v[~ok]=0
  else:v=np.zeros_like(da)
  color=cv2.applyColorMap((v*255).astype(np.uint8),cv2.COLORMAP_VIRIDIS);color=Image.fromarray(cv2.cvtColor(color,cv2.COLOR_BGR2RGB))
  rgb.thumbnail((278,215));color.thumbnail((278,215));sheet.paste(rgb,(x,y+27));sheet.paste(color,(x+285,y+27))
  draw.text((x,y+246),f"index {item['index']}  RGB ts {item['rgb_timestamp']}",font=small,fill='black')
  result['bonn_panels'].append({'label':row['label'],'index':item['index'],'rgb_path':item['rgb_path'],'depth_path':item['depth_path'],'rgb_timestamp':item['rgb_timestamp'],'depth_timestamp':item['depth_timestamp'],'rgb_sha256':sha(rgbb),'depth_sha256':sha(depb)})
buf=io.BytesIO();sheet.save(buf,format='PNG',optimize=True);result['contact_sheet_png_base64']=base64.b64encode(buf.getvalue()).decode()
raw=json.dumps(result,ensure_ascii=False,separators=(',',':')).encode();print('AUDIT_ZLIB_BASE64:'+base64.b64encode(zlib.compress(raw,6)).decode(),flush=True)

