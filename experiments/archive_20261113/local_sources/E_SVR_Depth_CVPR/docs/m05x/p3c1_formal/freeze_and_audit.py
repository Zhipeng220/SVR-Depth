import getpass,hashlib,json,shlex
from pathlib import Path
import paramiko

root=Path.cwd();pilot=root/'docs/m05x/p3c1_pilot';out=root/'docs/m05x/p3c1_formal';out.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
manifest=json.loads((pilot/'evidence_manifest.json').read_text(encoding='utf8'))
for name,entry in manifest['files'].items():
 p=pilot/name
 if not p.is_file() or sha(p)!=entry['sha256']:raise RuntimeError('pilot local integrity mismatch: '+name)
for name in ['P3C1_PILOT_REPORT.md','candidate_geometry.csv','paired_future_results.csv','invariant_validation.json','evidence_manifest.json']:(pilot/name).read_bytes()
(root/'docs/m05x/p3c0/P3C0_REPORT.md').read_text(encoding='utf8')
anchors=json.loads((root/'docs/m05x/p2b/anchor_manifest.json').read_text(encoding='utf8'))['anchors'];pids=['Bonn_00_age1','TUM_00_age1','ScanNet_00_age1']
if len(anchors)!=42 or sum(a['anchor_id'] in pids for a in anchors)!=3:raise RuntimeError('unexpected anchor set')
protocol=json.loads((pilot/'pilot_protocol.json').read_text(encoding='utf8'))
frozen={'created_before_formal_results':True,'pilot_anchor_ids':pids,'formal_anchors':[a for a in anchors if a['anchor_id'] not in pids],'expected_formal_count':39,'p2b_anchor_manifest_sha256':sha(root/'docs/m05x/p2b/anchor_manifest.json'),'p3c1_pilot_manifest_sha256':sha(pilot/'evidence_manifest.json'),'pilot_protocol_sha256':sha(pilot/'pilot_protocol.json'),'norm_relative_tolerance':protocol['norm_match_relative_tolerance'],'layer_norm_minimum':protocol['norm_minimum'],'noncollinearity_max_abs_cosine':protocol['noncollinearity_max_abs_cosine'],'formula_abs_tolerance':protocol['formula_abs_tolerance'],'p3b_endpoints':'bitwise exact','sham':'bitwise exact','bootstrap':{'seed':20261110,'resamples':10000,'method':'sequence-level percentile'},'primary_endpoint':'H4 V4-layerwise-matched minus V5-native','selection':'all 39 nonpilot P2-B anchors, no replacement','frozen_checkpoint_sha256':protocol['checkpoint_sha256']}
p=out/'anchor_manifest.json'
if p.exists():
 if json.loads(p.read_text(encoding='utf8'))!=frozen:raise RuntimeError('existing frozen selection differs')
else:p.write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
progress=json.loads((pilot/'execution_progress.json').read_text(encoding='utf8'));remote=progress['remote_base']
pw=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
paths=[remote+'/capture_p3c1.py']+[remote+'/'+aid+'/'+name for aid in pids for name in ['evidence_arrays.npz','capture_metadata.json']]
cmd='sha256sum '+' '.join(shlex.quote(x) for x in paths)+' && df -Pk /home/svrdepth/SVR-Depth/velodepth/outputs && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -c '+shlex.quote('import torch; print("CUDA",torch.cuda.is_available(),torch.cuda.get_device_name(0),torch.cuda.mem_get_info())')
_,stdout,stderr=c.exec_command(cmd,timeout=180);raw=stdout.read().decode();err=stderr.read().decode();rc=stdout.channel.recv_exit_status();c.close()
if rc:raise RuntimeError('remote audit command failed: '+err)
remote_hash={line.split()[1]:line.split()[0] for line in raw.splitlines() if len(line.split())>=2 and len(line.split()[0])==64}
expected={paths[0]:sha(root/'m05x_work/p3c1_pilot/capture_p3c1.py')}
for aid in pids:
 for name in ['evidence_arrays.npz','capture_metadata.json']:expected[remote+'/'+aid+'/'+name]=sha(pilot/'runs'/aid/name)
if remote_hash!=expected:raise RuntimeError('remote pilot evidence or source hash mismatch')
for aid in pids:
 capture=(pilot/'runs'/aid/'capture.log').read_text(encoding='utf8');verify=(pilot/'runs'/aid/'verify.log').read_text(encoding='utf8')
 if 'CAPTURE_COMPLETE' not in capture or '"status": "PASS"' not in verify:raise RuntimeError('pilot execution log incomplete: '+aid)
(out/'pilot_reaudit.json').write_text(json.dumps({'status':'PASS','remote_base':remote,'remote_sha256':remote_hash,'remote_command':cmd,'remote_stdout':raw,'pilot_local_manifest_verified':True,'logs_reviewed':pids,'formal_anchor_manifest_sha256':sha(p)},indent=2,ensure_ascii=False)+'\n',encoding='utf8')
print('PILOT_REAUDIT_PASS',len(expected),'remote files; FROZEN_FORMAL_ANCHORS',len(frozen['formal_anchors']))
