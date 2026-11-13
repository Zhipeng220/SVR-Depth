"""Read-only metadata, frame-key and representative encoded-file hash audit; no metrics or GPU."""
import getpass,paramiko,shlex
from pathlib import Path
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
probe=r'''import h5py,hashlib,json
from pathlib import Path
out={}
for domain,metadata in [('TUM','sequences.json'),('ScanNet','sequences_.json')]:
 p=Path.home()/'datasets/velodepth_evaluation'/(domain+'.hdf5');results=[]
 try:
  with h5py.File(p,'r') as h:
   seq=json.loads(bytes(h[metadata][:]).decode());val=bytes(h['val.txt'][:]).decode().splitlines();keys=[x.rsplit(' ',1)[0] for x in val if len(x.rsplit(' ',1))==2]
   for vk in keys:
    tk=vk[:-4]+'_train' if vk.endswith('_val') else '';v=seq[vk];t=seq.get(tk);r={'dataset':domain,'val_sequence':vk,'train_sequence':tk,'pair_found':t is not None,'val_frame_count':len(v.get('image',[]))}
    if t is not None:
     vi=v.get('image',[]);ti=t.get('image',[]);common=sorted(set(vi)&set(ti));r.update(train_frame_count=len(ti),common_rgb_key_count=len(common),common_key_examples=common[:3],val_first_keys=vi[:2],train_first_keys=ti[:2]);samples=[]
     for key in common[:2]:samples.append({'key':key,'sha256':hashlib.sha256(h[key][:].tobytes()).hexdigest()})
     r['common_rgb_hash_samples']=samples
     for kind in ('depth',):
      dv=v.get(kind,[]);dt=t.get(kind,[]);shared=sorted(set(dv)&set(dt));r['common_depth_key_count']=len(shared);r['common_depth_hash_samples']=[{'key':key,'sha256':hashlib.sha256(h[key][:].tobytes()).hexdigest()} for key in shared[:2]]
    results.append(r)
  out[domain]={'path':str(p),'results':results}
 except Exception as e:out[domain]={'error':repr(e),'results':results}
print(json.dumps(out,ensure_ascii=False))
'''
cmd="/home/svrdepth/miniconda3/envs/svrdepth/bin/python - <<'PY'\n"+probe+"\nPY\n";_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=300);data=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();Path('docs/m05x/p3c0/video_metadata_remote.json').write_text(data,encoding='utf8');Path('docs/m05x/p3c0/video_metadata_remote.log').write_text(err,encoding='utf8');print('EXIT',rc,'bytes',len(data),'err',err,flush=True);c.close();raise SystemExit(rc)
