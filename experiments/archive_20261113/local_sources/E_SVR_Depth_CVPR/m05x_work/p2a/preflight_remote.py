import getpass,hashlib,json,paramiko,shlex
from pathlib import Path
p=getpass.getpass('SSH password: '); c=paramiko.SSHClient(); c.load_system_host_keys(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
probe=r'''import hashlib,json,os,sys
from pathlib import Path
import h5py,torch
r=Path.cwd();home=Path.home()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
paths=['outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt','experiments/p1/cvpr_m05_formal.py','experiments/p1/gt4a2_exp1_train_eval.py','experiments/p1/gt2_fixed_clip_repair_train.py','outputs/y04_v4_manifest_candidate_20261108_190940/moving_obstructing_box_pairs_v4_candidate.csv','experiments/p1/v5c_external_eval.py']
for item in paths:
 p=r/item;print('FILE',json.dumps({'path':str(p),'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None}))
p=home/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c'
for name in ('model.safetensors','config.json'):
 q=p/name;print('FILE',json.dumps({'path':str(q),'exists':q.is_file(),'sha256':sha(q) if q.is_file() else None}))
print('TORCH',torch.__version__,'CUDA',torch.cuda.is_available(),torch.version.cuda,'DEVICE',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
for name,key,meta in [('TUM','/scratch/lpiccinelli/datasets/tum/outputs/fr2_360_hemisphere_val','sequences.json'),('ScanNet','scene0707_00_val','sequences_.json')]:
 p=home/'datasets/velodepth_evaluation'/(name+'.hdf5')
 try:
  with h5py.File(p,'r') as h:
   seq=json.loads(bytes(h[meta][:]).decode());record=seq.get(key)
   if record is None:print('DATA',json.dumps({'dataset':name,'path':str(p),'sequence':key,'available':False,'similar':[x for x in seq if key.split('/')[-1] in x][:5]}));continue
   images=record.get('image',[]);depth=record.get('depth',[])
   print('DATA',json.dumps({'dataset':name,'path':str(p),'sequence':key,'available':True,'images':len(images),'depths':len(depth),'sample_images':images[:2],'sample_depths':depth[:2],'first9_images_exist':all(x in h for x in images[:9]),'first9_depths_exist':all(x in h for x in depth[:9]),'meta_sha256':hashlib.sha256(bytes(h[meta][:])).hexdigest()}))
 except Exception as e:print('DATA',json.dumps({'dataset':name,'path':str(p),'sequence':key,'available':False,'error':repr(e)}))
'''
cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python - <<\'PY\'\n'+probe+'\nPY\n'
_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=180);data=o.read().decode(errors='replace');err=e.read().decode(errors='replace');Path('docs/m05x/p2a/preflight.log').write_text(data+'\nSTDERR\n'+err,encoding='utf-8');print(data,err,flush=True);c.close()
