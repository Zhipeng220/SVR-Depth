"""CPU-only read of frozen head tensors; no model forward, no training."""
import getpass,paramiko,shlex
from pathlib import Path
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
probe=r'''import hashlib,json,torch
from pathlib import Path
r=Path.cwd();paths={'V4':r/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt','V5':r/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt'};expect={'V4':'56d7b8133b67d642f608ab95bf02c68d25dc0e3a45cf18c350a8efc6d96d0365','V5':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
o={'cuda_visible_devices':'','head_forward_executed':False,'models':{}};heads={}
for name,p in paths.items():
 digest=sha(p)
 if digest!=expect[name]:raise RuntimeError('frozen checkpoint hash mismatch '+name)
 ck=torch.load(p,map_location='cpu',weights_only=True);heads[name]=ck['repair_head_state_dict'];o['models'][name]={'path':str(p),'sha256':digest,'head_parameter_count':sum(v.numel() for v in heads[name].values()),'head_key_shapes':{k:list(v.shape) for k,v in heads[name].items()},'head_dtypes':{k:str(v.dtype) for k,v in heads[name].items()}}
samekeys=set(heads['V4'])==set(heads['V5']);sameshapes=samekeys and all(heads['V4'][k].shape==heads['V5'][k].shape for k in heads['V4']);o.update(same_keys=samekeys,same_shapes=sameshapes)
if sameshapes:
 sum_delta=sum(float(((heads['V4'][k].double()-heads['V5'][k].double())**2).sum()) for k in heads['V4']);dot=sum(float((heads['V4'][k].double()*heads['V5'][k].double()).sum()) for k in heads['V4']);aa=sum(float((heads['V4'][k].double()**2).sum()) for k in heads['V4']);bb=sum(float((heads['V5'][k].double()**2).sum()) for k in heads['V5']);o.update(parameter_delta_l2=sum_delta**.5,parameter_cosine=dot/(aa*bb)**.5 if aa and bb else None,exact_equal=sum_delta==0.)
print(json.dumps(o,ensure_ascii=False))
'''
cmd="cd ~/SVR-Depth/velodepth && CUDA_VISIBLE_DEVICES='' /home/svrdepth/miniconda3/envs/svrdepth/bin/python - <<'PY'\n"+probe+"\nPY\n";_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=300);data=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();Path('docs/m05x/p3c0/head_parameter_audit.json').write_text(data,encoding='utf8');Path('docs/m05x/p3c0/head_parameter_audit.log').write_text(err,encoding='utf8');print('EXIT',rc,'bytes',len(data),err,flush=True);c.close();raise SystemExit(rc)
