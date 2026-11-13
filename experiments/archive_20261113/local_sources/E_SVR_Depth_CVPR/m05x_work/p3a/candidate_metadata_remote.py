import getpass,paramiko,shlex
from pathlib import Path
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
probe=r'''import h5py,json
from pathlib import Path
o={}
for name,meta in [('TUM','sequences.json'),('ScanNet','sequences_.json'),('Sintel','sequences.json')]:
 p=Path.home()/'datasets/velodepth_evaluation'/(name+'.hdf5')
 try:
  with h5py.File(p,'r') as h:
   seq=json.loads(bytes(h[meta][:]).decode());splits={}
   for split in ['train.txt','training.txt','val.txt','test.txt']:
    if split in h:
     lines=bytes(h[split][:]).decode().splitlines();splits[split]=[line.rsplit(' ',1)[0] for line in lines if len(line.rsplit(' ',1))==2]
   items=[]
   for key,rec in seq.items():
    im=rec.get('image',[]);gt=rec.get('depth',[]);same=len(im)==len(gt);probe_n=min(16,len(im),len(gt));exists=bool(probe_n>=16 and all(i in h and d in h for i,d in zip(im[:probe_n],gt[:probe_n])))
    items.append({'sequence':key,'image_count':len(im),'depth_count':len(gt),'metadata_lengths_equal':same,'first16_keys_exist':exists})
   o[name]={'hdf5':str(p),'splits':splits,'sequences':items,'note':'metadata and HDF5 key existence only; no RGB/depth pixels or metrics read'}
 except Exception as e:o[name]={'error':repr(e)}
print(json.dumps(o,ensure_ascii=False))
'''
cmd="/home/svrdepth/miniconda3/envs/svrdepth/bin/python - <<'PY'\n"+probe+"\nPY\n"
_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=300);data=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();Path('docs/m05x/p3a/candidate_metadata_raw.json').write_text(data,encoding='utf-8');Path('docs/m05x/p3a/candidate_metadata_error.log').write_text(err,encoding='utf-8');print('EXIT',rc,'bytes',len(data),err,flush=True);c.close();raise SystemExit(rc)
