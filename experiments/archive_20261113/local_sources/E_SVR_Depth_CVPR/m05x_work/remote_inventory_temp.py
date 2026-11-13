import getpass,paramiko,shlex
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=12);p=None
cmd=r'''cd ~/SVR-Depth/velodepth || exit 1
pwd
git rev-parse HEAD 2>&1
git status --short 2>&1
python3 - <<'PY'
import pathlib,hashlib,json
r=pathlib.Path.cwd()
paths=['experiments/p1/gt4a2_y04_v4_bonn_train_v0_2.py']
paths += [str(x.relative_to(r)) for x in (r/'outputs/y04_v4_manifest_candidate_20261108_190940').glob('*_pairs_v4_candidate.csv')]
for d in ['outputs/y04_v4_bonn5_pretrained_comparison_20261109_082114','outputs/y04_v4_external_compare_20261109_084229','outputs/y04_v5c_bonn_20261109_155801','outputs/y04_v5c_external_20261109_171333']:
 paths += [str(x.relative_to(r)) for x in (r/d).rglob('protocol.json')]
for d in ['outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748','outputs/y04_v5_preserve_seed20261025_20261109_112607']:
 p=r/d
 if p.exists(): paths += [str(x.relative_to(r)) for x in p.iterdir() if x.is_file() and (x.suffix in ('.pt','.json','.csv','.py') or 'sample' in x.name)]
paths += [str(pathlib.Path.home()/'hf_cache/hub/models--lpiccinelli--velodepth/snapshots/2b2ee9fe44d0b261e736653042cecd7693f99c8c/model.safetensors')]
for s in paths:
 p=pathlib.Path(s);p=p if p.is_absolute() else r/p
 if p.is_file():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
  print(json.dumps({'path':str(p),'exists':True,'bytes':p.stat().st_size,'sha256':h.hexdigest()}),flush=True)
 else:print(json.dumps({'path':str(p),'exists':False}),flush=True)
PY'''
_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=180);data=o.read().decode(errors='replace');err=e.read().decode(errors='replace');print(data);print(err);open('m05x_work/remote_inventory_raw4.txt','w',encoding='utf-8').write(data+'\nSTDERR:\n'+err);c.close()
