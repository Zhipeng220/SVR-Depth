import getpass,paramiko,shlex
from pathlib import Path
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
probe=r'''import csv,hashlib,io,json,os,shutil
from pathlib import Path
import cv2,h5py,numpy as np,torch
from PIL import Image
r=Path.cwd(); home=Path.home();output={'disk':{},'cuda':{},'datasets':{}}
for path in (str(r),'/tmp'):
 t=shutil.disk_usage(path);output['disk'][path]={'free':t.free,'total':t.total}
output['cuda']={'available':torch.cuda.is_available(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch':torch.__version__,'cuda':torch.version.cuda}
bonn=['moving_obstructing_box','person_tracking2','removing_nonobstructing_box','moving_nonobstructing_box','placing_nonobstructing_box']
items=[]
for seq in sorted(bonn):
 p=r/'outputs/y04_v4_manifest_candidate_20261108_190940'/f'{seq}_pairs_v4_candidate.csv'; row={'sequence':seq,'path':str(p),'eligible':False,'reason':''}
 try:
  with p.open(encoding='utf-8-sig',newline='') as f: rows=sorted(csv.DictReader(f),key=lambda x:int(x['frame_index_zero_based']))
  if len(rows)<16:raise ValueError('fewer than 16 frames')
  if [int(x['frame_index_zero_based']) for x in rows[:16]]!=list(range(16)):raise ValueError('nonconsecutive indices')
  for i in range(9):
   if not Path(rows[i]['rgb_path']).is_file():raise ValueError(f'missing RGB {i}')
  for i in range(2,9):
   if rows[i].get('gt_match_status','').upper()!='MATCHED' or not Path(rows[i]['depth_path']).is_file():raise ValueError(f'GT unmatched/missing {i}')
   gt=cv2.imread(rows[i]['depth_path'],cv2.IMREAD_UNCHANGED)
   if gt is None or gt.dtype!=np.uint16 or gt.ndim!=2 or not (gt>0).any():raise ValueError(f'GT invalid {i}')
  row.update(eligible=True,frames=len(rows),pair_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),max_delta_ms=max(float(rows[i].get('abs_delta_ms') or 0) for i in range(2,9)))
 except Exception as e:row['reason']=str(e)
 items.append(row)
output['datasets']['Bonn']=items
for name,meta,min_depth in [('TUM','sequences.json',.001),('ScanNet','sequences_.json',.01)]:
 p=home/'datasets/velodepth_evaluation'/(name+'.hdf5');items=[]
 try:
  with h5py.File(p,'r') as h:
   seq=json.loads(bytes(h[meta][:]).decode()); lines=bytes(h['val.txt'][:]).decode().strip().splitlines(); keys=sorted({line.rsplit(' ',1)[0] for line in lines if len(line.rsplit(' ',1))==2})
   output['datasets'][name+'_candidate_count']=len(keys)
   for key in keys:
    if sum(x['eligible'] for x in items)>=8:break
    item={'sequence':key,'hdf5':str(p),'eligible':False,'reason':''}
    try:
     rec=seq[key]; imgs=rec['image']; gts=rec['depth']
     if len(imgs)!=len(gts) or len(imgs)<16:raise ValueError('length/alignment invalid')
     for i in range(9):
      if imgs[i] not in h or gts[i] not in h:raise ValueError(f'missing key {i}')
      if Path(imgs[i]).stem.removesuffix('_c')!=Path(gts[i]).stem.removesuffix('_d'):raise ValueError(f'key alignment mismatch {i}')
     shapes=[]
     for i in range(9):
      with Image.open(io.BytesIO(h[imgs[i]][:].tobytes())) as im:shapes.append(im.size)
     if len(set(shapes))!=1:raise ValueError('RGB shapes differ')
     for i in range(2,9):
      raw=np.asarray(Image.open(io.BytesIO(h[gts[i]][:].tobytes())))
      if raw.ndim==3:raw=raw[:,:,0].astype(np.int32)|(raw[:,:,1].astype(np.int32)<<8)|(raw[:,:,2].astype(np.int32)<<16)
      if raw.shape[::-1]!=shapes[i]:raise ValueError(f'GT shape mismatch {i}')
      gt=raw.astype(np.float32)/1000
      if not (np.isfinite(gt)&(gt>min_depth)&(gt<10)).any():raise ValueError(f'GT empty {i}')
     item.update(eligible=True,frames=len(imgs),metadata_sha256=hashlib.sha256(bytes(h[meta][:])).hexdigest(),split='val.txt')
    except Exception as e:item['reason']=str(e)
    items.append(item)
 except Exception as e:output['datasets'][name+'_error']=repr(e)
 output['datasets'][name]=items
print(json.dumps(output,ensure_ascii=False))
'''
cmd="cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python - <<'PY'\n"+probe+"\nPY\n"
_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=600);data=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();Path('docs/m05x/p2b/scan_raw.json').write_text(data,encoding='utf-8');Path('docs/m05x/p2b/scan_stderr.log').write_text(err,encoding='utf-8');print('EXIT',rc,'bytes',len(data),'stderr',err,flush=True);c.close();raise SystemExit(rc)
