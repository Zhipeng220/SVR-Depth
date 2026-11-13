import base64,getpass,hashlib,io,json,paramiko,re,shlex,zlib
from pathlib import Path
import numpy as np
ROOT=Path.cwd();out=ROOT/'outputs/P4A_CrossDomain_Replication_20261110_110108';p=out/'preflight';formal=ROOT/'docs/m05x/p3c1_formal';errors=[]
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as q:
  for b in iter(lambda:q.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
try:
 source=(p/'original_smoke_capture.py').read_text(encoding='utf8');report=ROOT/'docs/p4a/P4A_C3_CAPTURE_SMOKE_REPORT.md';blocks=re.findall(r'```json\n(.*?)\n```',report.read_text(encoding='utf8'),re.S);smoke,grids=map(json.loads,blocks)
 assert smoke['anchor']=='Bonn_00_age4' and smoke['pass'] and smoke['hooks_restored'] and smoke['historical_d1_equal']
 for d in grids.values():
  for item in d.values():
   b=zlib.decompress(base64.b64decode(item['data'],validate=True));assert hashlib.sha256(b).hexdigest()==item['sha256_npy'];np.load(io.BytesIO(b),allow_pickle=False)
 for text in ["def wrapped_flow(flow,mask=True):","def wrapped_sample(inp,grid,*args,**kwargs):","x=delta[selected].double();y=w.float()[selected].double()","torch.dot(x,y).item()/(nx.item()*ny.item())","finally:"]:
  if text not in source:raise RuntimeError('original smoke implementation missing '+text)
 prior=json.loads((formal/'evidence_manifest.json').read_text(encoding='utf8'))['files'];local={}
 for ds,aid in [('TUM','TUM_00_age4'),('ScanNet','ScanNet_00_age4')]:
  base=formal/'runs'/aid;files={}
  for n in ['capture_metadata.json','evidence_arrays.npz','verification/invariant_validation.json']:
   rel='runs/'+aid+'/'+n;f=base/n;h=sha(f)
   if h!=prior[rel]['sha256']:raise RuntimeError('local evidence hash mismatch '+rel)
   files[n]=h
  validation=json.loads((base/'verification/invariant_validation.json').read_text());assert validation['status']=='PASS';meta=json.loads((base/'capture_metadata.json').read_text());assert meta['anchor_id']==aid
  local[ds]={'anchor':aid,'metadata':files['capture_metadata.json'],'npz':files['evidence_arrays.npz'],'validation':files['verification/invariant_validation.json']}
 credentials={ds:{'metadata':v['metadata'],'npz':v['npz']} for ds,v in local.items()};pw=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u - '+shlex.quote(json.dumps(credentials));stdin,stdout,stderr=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=600);stdin.write((p/'remote_preflight.py').read_text(encoding='utf8'));stdin.channel.shutdown_write();output=stdout.read().decode();err=stderr.read().decode();rc=stdout.channel.recv_exit_status();c.close()
 if rc:raise RuntimeError('remote preflight exit '+str(rc)+' '+err[-5000:])
 line=next((s[len('PREFLIGHT_JSON:'):] for s in output.splitlines() if s.startswith('PREFLIGHT_JSON:')),None)
 if not line:raise RuntimeError('missing remote preflight JSON')
 remote=json.loads(line);errors.extend(remote['errors'])
 result={'status':'PASS' if not errors else 'BLOCKED','smoke_original_implementation':{'path':str(p/'original_smoke_capture.py'),'sha256':sha(p/'original_smoke_capture.py'),'provenance':'recovered verbatim from this chat file-change delete record; original temporary file was removed after previous deliverable','formula':'unchanged smoke selected masked values float64 torch.dot / product of L2 norms'},'smoke_report':{'path':str(report),'sha256':sha(report),'embedded_grid_arrays_verified':8},'local_history':local,'remote':remote,'errors':errors,'selection':'TUM_00_age4 and ScanNet_00_age4 fixed before cross-domain results','gpu_in_preflight':False}
except Exception as e:
 result={'status':'BLOCKED','errors':errors+[repr(e)],'gpu_in_preflight':False}
(p/'P4A_CrossDomain_Preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# P4-A 跨域预检','',f"**状态：{result['status']}。** 预检只读，没有运行 GPU、读取未来帧或 GT。",'', '固定样本：TUM_00_age4、ScanNet_00_age4；只有两域同时通过且冻结模型身份均一致才可执行。','']
if result.get('remote'):
 for ds,r in result['remote']['domains'].items():lines.append(f"- {ds}：{r['status']}；录像 `{r.get('sequence','UNKNOWN')}`；失败检查 {[k for k,v in r.get('checks',{}).items() if not v]}；原因 {r.get('error') or '无'}。")
 lines+=['',f"原始 Bonn Smoke 实现从本对话的文件删除变更记录逐字恢复，SHA256 `{result['smoke_original_implementation']['sha256']}`；其 C3 计算代码、hook 包装、finally 与 8 个报告内 NPY 网格均已检查。恢复文件是供核验的新副本，不改历史输出。",'','服务器 4 项模型身份、原 parent NPZ、metadata、历史 D1、当前及上一 RGB、序列元数据身份及数组完整性检查详见 JSON。']
lines+=['',f"错误：{result.get('errors')}。",'','严格复用原 C3 计算，GT 不参与输入、mask 或计算；若 BLOCKED，不运行 GPU。']
(p/'P4A_CrossDomain_Preflight.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('PREFLIGHT',result['status'],result.get('errors'));raise SystemExit(0 if result['status']=='PASS' else 1)
