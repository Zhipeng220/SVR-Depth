"""CPU-only same-input head-output geometry; no state commit, replay, GT, or GPU."""
import getpass,paramiko,shlex
from pathlib import Path
p=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=p,look_for_keys=False,allow_agent=False,timeout=20);p=None
probe=r'''import hashlib,importlib.util,json,torch,numpy as np
from pathlib import Path
torch.set_num_threads(2);root=Path.cwd();base=root/'outputs/m05x_p3b_20261109_223610';src=root/'velodepth/models/persistent_feature_repair.py'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
if sha(src)!='6dfc8b0c5565ea9139b8c266c5ef337b3309ce0fde2f365428fc44390b84117c':raise RuntimeError('head source mismatch')
paths={'V4':root/'outputs/y04_v4_bonn13_indomain_seed20261025_20261108_235748/final.pt','V5':root/'outputs/y04_v5_preserve_seed20261025_20261109_112607/final.pt'};expected={'V4':'56d7b8133b67d642f608ab95bf02c68d25dc0e3a45cf18c350a8efc6d96d0365','V5':'2fd09d01c8a88c31fab164b0bfc0e87d889d3a9356aa77d3d3a3b60fd1039251'}
spec=importlib.util.spec_from_file_location('frozen_repair_head',src);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);models={}
for name,p in paths.items():
 if sha(p)!=expected[name]:raise RuntimeError('checkpoint mismatch '+name)
 weights=torch.load(p,map_location='cpu',weights_only=True)['repair_head_state_dict'];channels=int(weights['blocks.0.0.weight'].shape[1]);bottleneck=int(weights['blocks.0.0.weight'].shape[0]);model=module.PersistentFeatureRepairHead(channels=channels,bottleneck=bottleneck,levels=4).cpu().eval();model.load_state_dict(weights,strict=True);models[name]=model
out={'device':'cpu','cuda_visible_devices':'','precision':'float32 no autocast','state_commit_performed':False,'future_replay_performed':False,'gt_read':False,'head_source_sha256':sha(src),'checkpoint_sha256':expected,'selected_anchors':['Bonn_00_age1','TUM_00_age1','ScanNet_00_age1'],'results':[]}
for aid in out['selected_anchors']:
 d=base/aid;meta=json.loads((d/'capture_metadata.json').read_text());f=d/'evidence_arrays.npz'
 if sha(f)!=meta['evidence_npz_sha256']:raise RuntimeError('NPZ SHA mismatch '+aid)
 aliases=meta['array_aliases'];old=[];saved=[]
 with np.load(f,allow_pickle=False) as z:
  for i in range(4):
   k='committed_O__feature'+str(i);t='committed_T__feature'+str(i);old.append(torch.from_numpy(z[aliases.get(k,k)].copy()).float());saved.append(torch.from_numpy(z[aliases.get(t,t)].copy()).float())
 with torch.inference_mode():o4=models['V4'](old);o5=models['V5'](old)
 d4=[(y-x).double() for x,y in zip(old,o4)];d5=[(y-x).double() for x,y in zip(old,o5)];dg=[(y-x).double() for x,y in zip(old,saved)]
 def norm(xs):return sum(float((x*x).sum()) for x in xs)**.5
 def dot(xs,ys):return sum(float((x*y).sum()) for x,y in zip(xs,ys))
 n4,n5,ng=norm(d4),norm(d5),norm(dg);difference=norm([a-b for a,b in zip(d5,dg)]);dot45=dot(d4,d5);dot4g=dot(d4,dg);scale=ng/n4 if n4 else None
 if scale is None:raise ValueError('zero V4 update norm')
 scaled=[((x+torch.tensor(scale,dtype=torch.float32)*(y-x))-x).double() for x,y in zip(old,o4)];scaled_norm=norm(scaled)
 out['results'].append({'anchor_id':aid,'dataset':meta['dataset'],'sequence':meta['sequence'],'age':meta['age'],'source_npz_sha256':sha(f),'v4_delta_l2_cpu':n4,'v5_delta_l2_cpu':n5,'v5_delta_l2_saved':ng,'cos_v4_vs_v5_cpu':dot45/(n4*n5) if n4*n5 else None,'cos_v4_cpu_vs_v5_saved':dot4g/(n4*ng) if n4*ng else None,'scale_v4_to_saved_v5_norm':scale,'matched_v4_l2_after_float32_reconstruction':scaled_norm,'matched_norm_relative_error':abs(scaled_norm-ng)/ng if ng else None,'v5_cpu_vs_saved_delta_l2':difference,'v5_cpu_vs_saved_relative_l2':difference/ng if ng else None,'v4_v5_delta_equal_exact':all(torch.equal(x,y) for x,y in zip(d4,d5))})
print(json.dumps(out,ensure_ascii=False,allow_nan=False))
'''
cmd="cd ~/SVR-Depth/velodepth && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/svrdepth/miniconda3/envs/svrdepth/bin/python - <<'PY'\n"+probe+"\nPY\n";_,o,e=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=600);data=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();Path('docs/m05x/p3c0/candidate_cpu_geometry.json').write_text(data,encoding='utf8');Path('docs/m05x/p3c0/candidate_cpu_geometry.log').write_text(err,encoding='utf8');print('EXIT',rc,'bytes',len(data),'err',err,flush=True);c.close();raise SystemExit(rc)
