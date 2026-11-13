import getpass,json,paramiko,shlex,subprocess,sys,time
from pathlib import Path
root=Path.cwd();out=root/'outputs/P4A_CrossDomain_Replication_20261110_110108';p=out/'preflight';pre=json.loads((p/'P4A_CrossDomain_Preflight.json').read_text(encoding='utf8'))
if pre['status']!='PASS' or any(x['status']!='PASS' for x in pre['remote']['domains'].values()):raise RuntimeError('preflight gate blocked')
pw=getpass.getpass('SSH password: ');c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('192.168.0.236',port=2222,username='svrdepth',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
s=c.open_sftp();remote='/home/svrdepth/SVR-Depth/velodepth/outputs/P4A_CrossDomain_Replication_20261110_110108';s.mkdir(remote);s.put(str(p/'cross_domain_capture.py'),remote+'/cross_domain_capture.py');rows=[];error=''
try:
 for domain,aid in [('TUM','TUM_00_age4'),('ScanNet','ScanNet_00_age4')]:
  target=out/domain
  if target.exists():raise FileExistsError(target)
  for name in ['capture','metrics','equivalence','logs']: (target/name).mkdir(parents=True,exist_ok=True)
  local=pre['local_history'][domain];args=[remote+'/cross_domain_capture.py',domain,aid,local['metadata'],local['npz'],remote+'/'+domain];cmd='cd ~/SVR-Depth/velodepth && /home/svrdepth/miniconda3/envs/svrdepth/bin/python -u '+' '.join(map(shlex.quote,args));print('START',domain,aid,flush=True)
  _,stdout,stderr=c.exec_command('bash -lc '+shlex.quote(cmd),timeout=1800);ch=stdout.channel;chunks=[]
  while not ch.exit_status_ready() or ch.recv_ready() or ch.recv_stderr_ready():
   if ch.recv_ready():chunks.append(ch.recv(65536))
   if ch.recv_stderr_ready():chunks.append(ch.recv_stderr(65536))
   time.sleep(.1)
  rc=ch.recv_exit_status();(target/'logs/remote_capture.log').write_bytes(b''.join(chunks));row={'domain':domain,'anchor':aid,'remote_command':cmd,'capture_exit':rc};rows.append(row)
  if rc:row['status']='FAIL';error='capture failed '+domain;print('FAIL',error,flush=True);break
  for src,dst in [('capture.npz','capture/capture.npz'),('result.json','metrics/result.json')]:
   print('DOWNLOAD',domain,src,flush=True);s.get(remote+'/'+domain+'/'+src,str(target/dst),prefetch=True,max_concurrent_prefetch_requests=64)
  result=json.loads((target/'metrics/result.json').read_text(encoding='utf8'));(target/'equivalence/baseline_diagnostic.json').write_text(json.dumps(result['baseline_vs_diagnostic'],indent=2)+'\n',encoding='utf8')
  verify=subprocess.run([sys.executable,str(p/'verify_capture.py'),str(target),aid],capture_output=True,text=True,encoding='utf8',errors='replace');(target/'logs/independent_verify.log').write_text(verify.stdout+'\n'+verify.stderr,encoding='utf8');row['verify_exit']=verify.returncode;row['status']='PASS' if verify.returncode==0 and result['pass'] else 'FAIL';print('VERIFIED',domain,row['status'],flush=True)
  (out/'execution_status.json').write_text(json.dumps({'runs':rows,'error':''},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
  if row['status']!='PASS':error='independent verification failed '+domain;break
except Exception as e:error=repr(e);raise
finally:
 (out/'execution_status.json').write_text(json.dumps({'runs':rows,'error':error},ensure_ascii=False,indent=2)+'\n',encoding='utf8');s.close();c.close()
print('FINISHED',len(rows),error,flush=True);raise SystemExit(1 if error else 0)
